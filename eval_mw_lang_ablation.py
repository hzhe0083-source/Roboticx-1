"""MetaWorld 语言流消融（报告 §8.3 重建）。

背景：原 §8.3 实验脚本未留存，且发现 data/metaworld_features.pt 的
instruction_id 曾因分批合并 bug 重叠（每批 0-9，全文件仅 10 个唯一值）。
本脚本在修复版（全量 49 任务、instruction_id 0-48 全局唯一）数据上重建实验。

口径：
- 错误指令 = 将样本语言替换为 instruction_id 轮转 +1（mod 49）任务的指令
  嵌入；其余输入（视觉/状态/上一动作）完全相同。
- 输出决策点 0 的 chunk_mae（原 §8.3 口径）与全序列 chunk_mae（§8.2 口径），
  均用归一化动作；附持久性基线对照。

用法（GPU 空闲时）：
    python eval_mw_lang_ablation.py --checkpoint checkpoints/metaworld_va8_40k.pt \
        --data data/metaworld_features.pt --device cuda
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from evaluate import persistence_baseline
from train import FeatureDataset, ensure_sequence, move_batch
from va_compound import VACompoundConfig, VACompoundPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaWorld language-stream ablation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="limit sample count (smoke-test use)",
    )
    parser.add_argument(
        "--taskid-lang", action="store_true",
        help="add task-id condition: language replaced by Qwen-encoded 'task {i}' texts",
    )
    parser.add_argument(
        "--model-dtype", default="bfloat16",
        help="dtype for the Qwen encoder used in --taskid-lang",
    )
    return parser.parse_args()


def build_wrong_language(payload: dict, device: torch.device) -> torch.Tensor:
    """把每个样本的语言替换为 instruction_id 轮转 +1 任务的指令嵌入。

    同任务样本的语言嵌入相同（同一任务文本的编码），故按 instruction_id
    取该任务第一个样本的嵌入即可构造替换表。
    """
    inst = payload["instruction_id"].tolist()
    unique = sorted(set(inst))
    per_inst = {value: payload["language_hidden"][inst.index(value)] for value in unique}
    n_tasks = len(unique)
    wrong = torch.stack([per_inst[(value + 1) % n_tasks] for value in inst]).to(device)
    return wrong


def build_taskid_language(
    payload: dict,
    device: torch.device,
    *,
    dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把语言替换为 Qwen 编码的 'task {i}' 文本（仅携带任务标识、无任务语义）。

    与数据中 language_hidden 同构：fp16、按 instruction_id 对齐。
    返回 (hidden_override, mask_override)，形状均与 payload["language_hidden"]/[mask] 一致。
    """
    from va_compound.backbones import QwenTextBackbone

    inst = payload["instruction_id"].tolist()
    n_tasks = len(set(inst))
    texts = [f"task {i}" for i in range(n_tasks)]
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype=dtype, local_files_only=True
    )
    hidden, mask = text_backbone.encode(texts)
    hidden = hidden.to(device="cpu", dtype=torch.float16)
    mask = mask.cpu()
    del text_backbone
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inst_tensor = torch.tensor(inst, dtype=torch.long)
    return hidden[inst_tensor].to(device), mask[inst_tensor].to(device)


@torch.no_grad()
def eval_ablation(
    model: VACompoundPolicy,
    payload: dict,
    indices: list[int],
    device: torch.device,
    *,
    flow_steps: int,
    batch_size: int,
    language_override: tuple[torch.Tensor, torch.Tensor | None] | None,
) -> dict[str, float]:
    """开环逐决策点评估；统计决策点 0 与全序列的 chunk_mae（归一化）。"""
    model.eval()
    per_episode: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    keys = ("vision_tokens", "language_hidden", "proprio", "previous_action", "actions")
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(indices[start : start + batch_size])
        batch = {key: payload[key][batch_indices] for key in keys}
        if "language_mask" in payload:
            batch["language_mask"] = payload["language_mask"][batch_indices]
        if language_override is not None:
            hidden_override, mask_override = language_override
            batch["language_hidden"] = hidden_override[batch_indices]
            if mask_override is not None:
                batch["language_mask"] = mask_override[batch_indices]
        batch["episode_id"] = payload["episode_id"][batch_indices]
        batch = ensure_sequence(move_batch(batch, device), 1)
        language_cache = model.build_language_cache(
            batch["language_hidden"],
            batch.get("language_mask"),
        )
        visual_memory = None
        conditions = []
        for time_index in range(batch["vision_tokens"].shape[1]):
            condition, visual_memory = model.encode_condition(
                batch["vision_tokens"][:, time_index],
                batch["proprio"][:, time_index],
                batch["previous_action"][:, time_index],
                language_cache=language_cache,
                visual_memory=visual_memory,
                return_visual_memory=True,
            )
            conditions.append(condition)
        conditions = torch.stack(conditions, dim=1)  # [B, T, H, D]
        if getattr(model.config, "direct_head", False):
            predictions = model.decode_actions(
                conditions.reshape(-1, conditions.shape[-2], conditions.shape[-1])
            )
        else:
            predictions = model.sample_actions(
                conditions.reshape(-1, conditions.shape[-2], conditions.shape[-1]),
                steps=flow_steps,
            )
        predicted = predictions.reshape(
            conditions.shape[0], conditions.shape[1], predictions.shape[-2], -1
        )
        target = batch["actions"].to(device, dtype=predicted.dtype)
        chunk0 = (predicted[:, :1] - target[:, :1]).abs().mean(dim=(-1, -2))  # [B,1]
        chunk_all = (predicted - target).abs().mean(dim=(-1, -2))  # [B,T]
        for row, (c0, chunk_row) in enumerate(zip(chunk0.tolist(), chunk_all.tolist(), strict=True)):
            eid = batch["episode_id"][row]
            episode = int(eid[0] if eid.ndim else eid)
            per_episode[episode]["chunk0"].append(c0[0])
            per_episode[episode]["chunk_all"].extend(chunk_row)

    def aggregate(key: str) -> float:
        values = torch.tensor(
            [v for ep in per_episode.values() for v in ep[key]]
        )
        return float(values.mean())

    return {
        "chunk0": aggregate("chunk0"),
        "chunk_all": aggregate("chunk_all"),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = checkpoint.get("training_contract", {})
    config = VACompoundConfig(**checkpoint["config"])
    flow_steps = args.flow_steps or int(contract.get("flow_steps", 8))
    model = VACompoundPolicy(config).to(device)
    model.load_state_dict(checkpoint["model"])

    dataset = FeatureDataset(args.data, require_pairs=False)
    payload = dataset.payload
    indices = list(range(dataset.length))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    wrong_language = build_wrong_language(payload, device)

    clean = eval_ablation(
        model, payload, indices, device,
        flow_steps=flow_steps, batch_size=args.batch_size, language_override=None,
    )
    wrong = eval_ablation(
        model, payload, indices, device,
        flow_steps=flow_steps, batch_size=args.batch_size,
        language_override=(wrong_language, None),
    )
    baseline = persistence_baseline(payload, indices, success_threshold=0.05)

    n_tasks = len(set(payload["instruction_id"].tolist()))
    print(f"samples={len(indices)} tasks={n_tasks} flow_steps={flow_steps}")
    print(f"clean: chunk0={clean['chunk0']:.5f} chunk_all={clean['chunk_all']:.5f}")
    print(f"wrong: chunk0={wrong['chunk0']:.5f} chunk_all={wrong['chunk_all']:.5f}")
    delta0 = wrong["chunk0"] / clean["chunk0"] - 1.0
    delta_all = wrong["chunk_all"] / clean["chunk_all"] - 1.0
    print(
        f"delta: chunk0 {delta0:+.1%} chunk_all {delta_all:+.1%} "
        f"(positive = wrong instruction hurts, language stream matters)"
    )

    if args.taskid_lang:
        taskid_language = build_taskid_language(
            payload, device, dtype=args.model_dtype
        )
        taskid = eval_ablation(
            model, payload, indices, device,
            flow_steps=flow_steps, batch_size=args.batch_size,
            language_override=taskid_language,
        )
        print(f"taskid: chunk0={taskid['chunk0']:.5f} chunk_all={taskid['chunk_all']:.5f}")
        delta_tid0 = taskid["chunk0"] / clean["chunk0"] - 1.0
        delta_tid_all = taskid["chunk_all"] / clean["chunk_all"] - 1.0
        print(
            f"delta: chunk0 {delta_tid0:+.1%} chunk_all {delta_tid_all:+.1%} "
            f"(task-id vs clean: >0 = full text carries info beyond task identity)"
        )

    print(
        f"baseline(persistence): chunk_mae_norm={baseline['chunk_mae_norm']:.5f} "
        f"success={baseline['success_rate']:.1%}"
    )


if __name__ == "__main__":
    main()
