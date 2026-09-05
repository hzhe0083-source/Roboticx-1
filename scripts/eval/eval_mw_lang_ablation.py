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
    python scripts/eval/eval_mw_lang_ablation.py --checkpoint checkpoints/metaworld_va8_40k.pt \
        --data data/metaworld_features.pt --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from collections import defaultdict

import torch

from evaluate import persistence_baseline
from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.data.feature_dataset import FeatureDataset
from va_compound.training.batch import ensure_sequence, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaWorld language-stream ablation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--vision-pooling",
        choices=("flat", "spatial", "spatiotemporal"),
        default=None,
        help="defaults to the pooling recorded in the checkpoint contract",
    )
    parser.add_argument(
        "--local-slots-data",
        type=Path,
        default=None,
        help="Stage A/B：ST288 索引 .pt（含 vision_tokens_st_npy 与 coords），"
        "local_slots checkpoint 的消融评估用",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="limit sample count (smoke-test use)",
    )
    parser.add_argument(
        "--taskid-lang", action="store_true",
        help="add task-id condition: language replaced by Qwen-encoded 'task {i}' texts",
    )
    parser.add_argument(
        "--perturb",
        choices=("wrong", "blank", "swap", "none"),
        default="wrong",
        help="language-stream perturbation (vs clean): wrong = 同分布错误指令；"
        "blank = 语言流置零（zero hidden + zero mask）；swap = 换到另一任务指令；"
        "none = 只评估 clean",
    )
    parser.add_argument(
        "--state-take",
        type=int,
        default=4,
        choices=(0, 4),
        help="proprio 截取协议（开环 chunk-MAE 消融专用）：本脚本的 proprio 来自"
        "预计算 features（数据侧 4 维，eval_metaworld.py 的 --state-take 8/39"
        "需要环境 obs + 零初始化投影扩展，此处不适用——仅 0/4 合法）；"
        "0 = proprio 恒零（RGB-only 开环消融）",
    )
    parser.add_argument(
        "--servo-ablation",
        choices=("none", "zero-gain", "gain-shuffle", "wrong-role", "open-loop"),
        default="none",
        help="不支持项标注：servo 四消融（--servo-ablation/--fovea）只在"
        "eval_metaworld.py 的闭环 C² 部署中有意义；本脚本是开环 chunk-MAE 消融，"
        "传非 none 会明确报错而不是静默忽略",
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
    vision_key: str = "vision_tokens",
    local_tokens: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
    state_take: int = 4,
) -> dict[str, float]:
    """开环逐决策点评估；统计决策点 0 与全序列的 chunk_mae（归一化）。

    ``state_take=0``（--state-take 0）：proprio 恒零（RGB-only 开环消融）。
    """
    model.eval()
    per_episode: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    keys = (vision_key, "language_hidden", "proprio", "previous_action", "actions")
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
        if state_take == 0:
            batch["proprio"] = torch.zeros_like(batch["proprio"])
        language_cache = model.build_language_cache(
            batch["language_hidden"],
            batch.get("language_mask"),
        )
        visual_memory = None
        conditions = []
        semantic_contexts = []
        # 训练侧 rollout_policy 在 flow_semantic 时把槽输出（vision_in）作为
        # flow head 逐层 cross-attn 的语义上下文；评估必须走同一路径，否则
        # 槽语义通道被静默丢弃（cond_kv 回退为 action_condition）。
        use_flow_semantic = (
            getattr(model.config, "flow_semantic", False)
            and not getattr(model.config, "direct_head", False)
        )
        for time_index in range(batch[vision_key].shape[1]):
            if local_tokens is not None:
                if coords is None:
                    raise ValueError("local_tokens requires coords")
                vision_in = model.build_local_vision(
                    local_tokens[batch_indices][:, time_index].to(device),
                    coords.to(device),
                    language_cache.role_queries,
                )
            else:
                vision_in = batch[vision_key][:, time_index]
            condition, visual_memory = model.encode_condition(
                vision_in,
                batch["proprio"][:, time_index],
                batch["previous_action"][:, time_index],
                language_cache=language_cache,
                visual_memory=visual_memory,
                return_visual_memory=True,
            )
            conditions.append(condition)
            if use_flow_semantic:
                semantic_contexts.append(vision_in)
        conditions = torch.stack(conditions, dim=1)  # [B, T, H, D]
        flat = conditions.reshape(-1, conditions.shape[-2], conditions.shape[-1])
        semantic_flat = (
            torch.stack(semantic_contexts, dim=1).reshape(
                -1, *semantic_contexts[0].shape[1:]
            )
            if semantic_contexts
            else None
        )
        if getattr(model.config, "c2_controller", False):
            # C² 部署契约：P 投影当前视觉 → 收缩解码。
            c_current = model.control_projector(
                batch[vision_key].reshape(-1, *batch[vision_key].shape[2:])
            )
            predictions = model.decode_actions(flat, c_current=c_current)
        elif getattr(model.config, "direct_head", False):
            predictions = model.decode_actions(flat)
        else:
            predictions = model.sample_actions(
                flat, steps=flow_steps, semantic_context=semantic_flat
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
    contract_pooling = contract.get("vision_pooling")
    if (
        args.vision_pooling
        and contract_pooling
        and args.vision_pooling != contract_pooling
    ):
        raise ValueError(
            f"--vision-pooling {args.vision_pooling} conflicts with the checkpoint "
            f"contract ({contract_pooling}); omit the flag to use the recorded pooling"
        )
    pooling = args.vision_pooling or contract_pooling or "flat"
    if args.servo_ablation != "none":
        # 不支持项标注：开环 chunk-MAE 消融没有闭环伺服环节（设计 §七 Step 2
        # 四消融只在 eval_metaworld.py 的 C² plan/feedback 部署中有意义）。
        raise ValueError(
            "--servo-ablation 仅支持 none：本脚本是开环 chunk-MAE 消融，"
            "servo 四消融（zero-gain/gain-shuffle/wrong-role/open-loop）与 "
            "--fovea 只在 eval_metaworld.py 闭环 C² 部署中有意义"
        )
    vision_key = "vision_tokens_spatial" if pooling == "spatial" else "vision_tokens"
    flow_steps = args.flow_steps or int(contract.get("flow_steps", 8))
    model = VACompoundPolicy(config).to(device)
    model.load_state_dict(checkpoint["model"])

    # Stage A/B：local_slots 288-token 消融支持（与 mw_single_step_acc 同模式）。
    local_payload = None
    if args.local_slots_data:
        import numpy as np

        local_payload = torch.load(
            args.local_slots_data,
            map_location="cpu",
            weights_only=False,  # meta 含 numpy coords（本地可信 scratch 文件）
        )
        if not config.local_slots:
            raise ValueError("--local-slots-data 需要 config.local_slots 的 checkpoint")
    local_tokens = None
    coords = None
    if local_payload is not None:
        from va_compound.live_vjepa import load_st288_memmap

        npy_path = local_payload["vision_tokens_st_npy"]
        local_tokens = load_st288_memmap(npy_path, local_payload["metadata"])
        coords = torch.from_numpy(local_payload["coords"])
        if getattr(config, "dense_readout", False):
            # Step 0 dense readout checkpoint：数据必须已是 1152-token 密集特征。
            if local_tokens.shape[-2] != 1152:
                raise ValueError(
                    f"dense_readout checkpoint requires 1152-token dense features, "
                    f"got {local_tokens.shape[-2]}"
                )
        print(f"local-slots: ST288 loaded ({tuple(local_tokens.shape)})")
    dataset = FeatureDataset(args.data, require_pairs=False, vision_key=vision_key)
    payload = dataset.payload
    indices = list(range(dataset.length))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    clean = eval_ablation(
        model, payload, indices, device,
        flow_steps=flow_steps, batch_size=args.batch_size, language_override=None,
        vision_key=vision_key, local_tokens=local_tokens, coords=coords,
        state_take=args.state_take,
    )
    n_tasks = len(set(payload["instruction_id"].tolist()))
    print(f"samples={len(indices)} tasks={n_tasks} flow_steps={flow_steps}")
    print(f"clean: chunk0={clean['chunk0']:.5f} chunk_all={clean['chunk_all']:.5f}")

    if args.perturb == "wrong":
        override: tuple[torch.Tensor, torch.Tensor | None] = (
            build_wrong_language(payload, device), None,
        )
    elif args.perturb == "blank":
        override = (
            torch.zeros_like(payload["language_hidden"]).to(device),
            torch.zeros_like(payload["language_mask"]).to(device)
            if "language_mask" in payload else None,
        )
    elif args.perturb == "swap":
        inst = payload["instruction_id"].tolist()
        unique = sorted(set(inst))
        per_inst = {
            value: payload["language_hidden"][inst.index(value)] for value in unique
        }
        override = (
            torch.stack(
                [per_inst[(value + 1) % len(unique)] for value in inst]
            ).to(device),
            None,
        )
    else:
        override = None

    perturbed = None
    if override is not None:
        perturbed = eval_ablation(
            model, payload, indices, device,
            flow_steps=flow_steps, batch_size=args.batch_size,
            language_override=override,
            vision_key=vision_key, local_tokens=local_tokens, coords=coords,
            state_take=args.state_take,
        )
        print(
            f"{args.perturb}: chunk0={perturbed['chunk0']:.5f} "
            f"chunk_all={perturbed['chunk_all']:.5f}"
        )
        delta0 = perturbed["chunk0"] / clean["chunk0"] - 1.0
        delta_all = perturbed["chunk_all"] / clean["chunk_all"] - 1.0
        print(
            f"delta: chunk0 {delta0:+.1%} chunk_all {delta_all:+.1%} "
            f"(positive = perturbation hurts, language stream matters)"
        )
    baseline = persistence_baseline(payload, indices, success_threshold=0.05)

    if args.taskid_lang:
        taskid_language = build_taskid_language(
            payload, device, dtype=args.model_dtype
        )
        taskid = eval_ablation(
            model, payload, indices, device,
            flow_steps=flow_steps, batch_size=args.batch_size,
            language_override=taskid_language,
            vision_key=vision_key, local_tokens=local_tokens, coords=coords,
            state_take=args.state_take,
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
