#!/usr/bin/env python
"""Held-out simulator evaluation of a policy checkpoint's coarse DINO MT-VJ head.

This is a mechanism test, not closed-loop task evidence. It restores only the
jointly trained LanguageMetricField from a task35 policy checkpoint, encodes
independent true 480px simulator renders through frozen DINO at 224px, and
reports localization/visibility metrics for the corrected role order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from eval_metaworld import _load_dino_metric_from_policy
from prepare_metaworld_metric import make_metric_batch
from scripts.task35_proc import trainer_processes
from scripts.train_metric_roi_dino import CANONICAL, load_language
from va_compound.backbones import TimmActionVisionBackbone
from va_compound.longtraj_frames import LongTrajFramesDataset
from va_compound.model import VACompoundConfig, dense_coords


def encode_dino_frames_one_at_a_time(
    backbone: TimmActionVisionBackbone, images: torch.Tensor
) -> dict[int, torch.Tensor]:
    """Encode ``[N, 3, H, W]`` DINO frames sequentially.

    A batched 16-frame ViT-L/14 encode can spike past a 16 GiB laptop GPU
    when holdout follows a large policy load. Per-frame encode is identical
    in token order after concatenation.
    """
    if images.ndim != 4:
        raise ValueError(f"DINO holdout images must be [N,3,H,W], got {tuple(images.shape)}")
    parts = [
        backbone.forward_hierarchical_dense(images[index : index + 1])
        for index in range(int(images.shape[0]))
    ]
    return {
        layer: torch.cat([part[layer] for part in parts], dim=0)
        for layer in parts[0]
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def want_holdout_cuda(
    *,
    force: bool,
    cuda_visible: str | None,
    cuda_available: bool,
    trainer_alive: bool,
) -> bool:
    """Refuse CUDA holdout while the FM trainer owns the GPU."""
    want_cuda = bool(cuda_available) and cuda_visible != ""
    if want_cuda and trainer_alive and not force:
        raise SystemExit(
            "FM trainer still running; refusing to take the GPU. "
            "Pass --force or set CUDA_VISIBLE_DEVICES= to run on CPU."
        )
    return want_cuda


def summarize(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().cpu().numpy().astype(np.float64)
    return {
        "n": int(array.size),
        "rmse_px": float(np.sqrt(np.mean(array**2))),
        "mean_px": float(array.mean()),
        "median_px": float(np.median(array)),
        "p90_px": float(np.quantile(array, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--language-data", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2777)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow CUDA holdout while the FM trainer is still alive.",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.batch < 1:
        raise ValueError("samples and batch must be positive")
    want_cuda = want_holdout_cuda(
        force=args.force,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_available=torch.cuda.is_available(),
        trainer_alive=bool(trainer_processes()),
    )
    device = torch.device("cuda" if want_cuda else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**checkpoint["config"])
    if not config.dino_dense_metric or config.main_vision_grid != 16:
        raise ValueError("checkpoint must use task35 DINO dense metric grid16")
    metric_head, _ = _load_dino_metric_from_policy(checkpoint, config, device)
    backbone = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype="float16",
        model_id=config.main_vision_model_id,
        image_size=CANONICAL,
        feature_dim=1024,
        output_layers=(11, 23),
        checkpoint_path=args.dino_checkpoint,
        local_files_only=True,
    )
    backbone.freeze_all()
    dataset = LongTrajFramesDataset(
        args.language_data, min_sequence_length=4, include_frames=False
    )
    lang_hid, lang_mask = load_language(dataset.payload, "peg-insert-side-v3", device)
    coords = dense_coords(512, device=device)
    rng = np.random.default_rng(args.seed)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    role_errors = [[] for _ in range(4)]
    role_vis_logits = [[] for _ in range(4)]
    role_vis_targets = [[] for _ in range(4)]
    done = 0
    while done < args.samples:
        count = min(args.batch, args.samples - done)
        sim = make_metric_batch(
            "peg-insert-side-v3", rng, count, include_raw_frames=True
        )
        raw = np.ascontiguousarray(sim["raw_frames"][:, (2, 3)])
        if raw.shape != (count, 2, 480, 480, 3):
            raise ValueError(f"unexpected raw frame shape {raw.shape}")
        images = (
            torch.from_numpy(raw)
            .permute(0, 1, 4, 2, 3)
            .reshape(count * 2, 3, 480, 480)
            .float()
            .div_(255.0)
            .to(device)
        )
        images = F.interpolate(
            images,
            size=(CANONICAL, CANONICAL),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        with torch.inference_mode():
            hierarchical = encode_dino_frames_one_at_a_time(
                backbone, ((images - mean) / std).half()
            )
            out = metric_head(
                hierarchical[5].reshape(count, -1, 1024).float(),
                hierarchical[11].reshape(count, -1, 1024).float(),
                lang_hid[None].expand(count, -1, -1),
                lang_mask[None].expand(count, -1),
                coords,
            )
        targets = torch.from_numpy(sim["keypoints"]).to(device)
        visibility = torch.from_numpy(sim["visibility"]).to(device)
        errors = (out.p - targets).norm(dim=-1) * 480.0
        for role in range(4):
            valid = visibility[:, role] > 0.5
            if bool(valid.any()):
                role_errors[role].append(errors[valid, role])
            role_vis_logits[role].append(out.visibility_logits[:, role])
            role_vis_targets[role].append(visibility[:, role])
        done += count
        print(f"evaluated {done}/{args.samples}", flush=True)
    names = ["tool", "pegGrasp", "hole", "pegHead"]
    roles = {}
    all_error = []
    all_logits = []
    all_targets = []
    for index, name in enumerate(names):
        errors = torch.cat(role_errors[index])
        logits = torch.cat(role_vis_logits[index])
        targets = torch.cat(role_vis_targets[index])
        all_error.append(errors)
        all_logits.append(logits)
        all_targets.append(targets)
        roles[name] = {
            **summarize(errors),
            "visibility_bce": float(
                F.binary_cross_entropy_with_logits(logits, targets).cpu()
            ),
            "visibility_accuracy": float(
                ((torch.sigmoid(logits) >= 0.5) == (targets >= 0.5)).float().mean().cpu()
            ),
        }
    result = {
        "contract": "dino_metric_policy_task35_holdout_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "samples": args.samples,
        "seed": args.seed,
        "source_pixels": 480,
        "encoder_pixels": 224,
        "role_order": names,
        "aggregate": {
            **summarize(torch.cat(all_error)),
            "visibility_bce": float(
                F.binary_cross_entropy_with_logits(
                    torch.cat(all_logits), torch.cat(all_targets)
                ).cpu()
            ),
        },
        "roles": roles,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
