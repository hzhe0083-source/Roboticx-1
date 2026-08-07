"""Evaluate an end-to-end fine-tuned checkpoint on raw video data.

Loads the e2e checkpoint (V-JEPA weights + Qwen LoRA/unfrozen weights + VA
policy) and runs the open-loop evaluation on the video dataset, matching
evaluate.py's metrics (chunk_mae vs persistence baseline, first-step MAE).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from prepare_pnpw_features import QwenTextBackbone
from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop eval of an e2e checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="video .pt (pnpw_video format)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--flow-steps", type=int, default=8,
        help="Euler integration steps (paper protocol = 32, §9)",
    )
    parser.add_argument(
        "--perturb",
        choices=("none", "blank", "swap"),
        default="none",
        help="language perturbation: blank clears instructions, swap shifts task ids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    policy = VACompoundPolicy(config).eval().to(device)
    policy.load_state_dict(ckpt["model"])

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", max_tokens=64, local_files_only=True
    )
    vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
    vision_backbone.freeze_all()

    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    if ckpt.get("qwen_state_dict"):
        qwen_state = {
            k.removeprefix("text_model."): v for k, v in ckpt["qwen_state_dict"].items()
        }
        missing, unexpected = text_backbone.text_model.load_state_dict(
            qwen_state, strict=False
        )
        print(f"qwen loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if ckpt.get("lora"):
        from va_compound.backbones import apply_lora

        rank = int(ckpt.get("training_contract", {}).get("lora_rank", 32))
        apply_lora(text_backbone.text_model, rank=rank)
        own = dict(text_backbone.text_model.named_parameters())
        for name, value in ckpt["lora"].items():
            clean = name.removeprefix("text_model.")
            if clean in own:
                own[clean].data.copy_(value)
    text_backbone.text_model.eval()

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    n = payload["actions"].shape[0] if args.max_samples <= 0 else min(args.max_samples, payload["actions"].shape[0])
    # Per-task language caches (batch=1).
    tasks = payload["metadata"]["tasks"]
    if args.perturb == "blank":
        # 语言流切除：零 hidden/mask 直接构建零 cache。不能用空文本走 Qwen 编码
        # （Qwen3.5 对空序列 mask 崩溃：cache_position[0] 越界）。
        hidden = torch.zeros(
            len(tasks), 1, config.language_dim, dtype=torch.float16
        )
        mask = torch.zeros(len(tasks), 1, dtype=torch.bool)
        print("perturb: blank instructions (zero language cache)")
    else:
        if args.perturb == "swap":
            tasks = tasks[1:] + tasks[:1]
            print("perturb: swapped task ids")
        hidden, mask = text_backbone.encode(tasks)
    task_caches = [
        policy.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(tasks))
    ]

    first_maes, chunk_maes, successes = [], [], []
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            frames = payload["video_frames"][start:stop].to(device).float().div_(255.0)
            frames = (frames - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)
            b, t, w, c, h, ww = frames.shape
            instruction_ids = payload["instruction_id"][start:stop]
            # Sample-wise rollout (batch=1 recommended).
            for sample in range(stop - start):
                sample_frames = frames[sample : sample + 1]
                sample_tokens = vision_backbone(
                    sample_frames.reshape(1 * t, w, c, h, ww),
                    pooling="flat",
                ).reshape(1, t, -1, 768)
                cache = task_caches[int(instruction_ids[sample])]
                sample_memory = None
                for time_index in range(t):
                    cond, sample_memory = policy.encode_condition(
                        sample_tokens[:, time_index],
                        payload["proprio"][start + sample, time_index][None].to(device),
                        payload["previous_action"][start + sample, time_index][None].to(device),
                        language_cache=cache,
                        visual_memory=sample_memory,
                        return_visual_memory=True,
                    )
                    chunk = policy.sample_actions(cond, steps=args.flow_steps)
                    target = payload["actions"][start + sample, time_index].to(device)
                    first_maes.append(
                        float((chunk[0, 0] - target[0]).abs().mean().cpu())
                    )
                    chunk_maes.append(float((chunk[0] - target).abs().mean().cpu()))
                    successes.append(float((first_maes[-1] < 0.05)))

    import numpy as np

    first = np.array(first_maes)
    chunk = np.array(chunk_maes)
    succ = np.array(successes)
    print(f"=== e2e open-loop eval: {n} samples x {t} decision points ===")
    print(f"first_mae_norm: {first.mean():.5f}  chunk_mae_norm: {chunk.mean():.5f}")
    print(f"success(首步<0.05): {succ.mean():.1%}")

    # Persistence baseline on the same samples.
    prev = payload["previous_action"][:n]
    acts = payload["actions"][:n]
    base_first = (prev[:, :, None, :] - acts[:, :, :1, :]).abs().mean((-1, -2)).numpy()
    base_chunk = (prev[:, :, None, :] - acts).abs().mean((-1, -2)).numpy()
    print(f"persistence baseline: first={base_first.mean():.5f} chunk={base_chunk.mean():.5f}")
    print(f"vs baseline chunk: {chunk.mean() - base_chunk.mean():+.5f}")


if __name__ == "__main__":
    main()
