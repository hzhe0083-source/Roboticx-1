from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from va_compound.statistics import fmt_ci, macro_bootstrap_ci
from train import FeatureDataset, ensure_sequence, move_batch
from va_compound import VACompoundConfig, VACompoundPolicy


@torch.no_grad()
def evaluate_policy(
    model: VACompoundPolicy,
    payload: dict,
    indices: list[int],
    device: torch.device,
    *,
    flow_steps: int,
    success_threshold: float,
    batch_size: int,
    action_scale: Tensor,
) -> dict[int, dict[str, float]]:
    """Open-loop rollout: per decision point, sample a chunk per time step and
    compare against the expert chunk, aggregated per episode.

    The PNPW dataset has no success labels, so success is proxied by the
    first-step normalized action error falling below ``success_threshold``.
    ``action_scale = (q99 - q01) / 2`` converts the per-dimension normalized
    MAE back to the raw action scale without any q01 offset.
    """
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
            )  # [B*T, H, A]
        predicted = predictions.reshape(
            conditions.shape[0], conditions.shape[1], predictions.shape[-2], -1
        )
        target = batch["actions"].to(device, dtype=predicted.dtype)

        first_step_mae = (predicted[:, :, 0] - target[:, :, 0]).abs().mean(dim=-1)
        first_step_mae_raw = (
            (predicted[:, :, 0] - target[:, :, 0])
            .abs()
            .mul(action_scale.to(device=device, dtype=predicted.dtype))
            .mean(dim=-1)
        )
        chunk_mae = predicted.sub(target).abs().mean(dim=(-1, -2))
        chunk_mse = predicted.sub(target).pow(2).mean(dim=(-1, -2))
        for index, episode_id in enumerate(batch["episode_id"].tolist()):
            per_episode[episode_id]["first_step_mae"].extend(
                first_step_mae[index].tolist()
            )
            per_episode[episode_id]["first_step_mae_raw"].extend(
                first_step_mae_raw[index].tolist()
            )
            per_episode[episode_id]["chunk_mae"].extend(chunk_mae[index].tolist())
            per_episode[episode_id]["chunk_mse"].extend(chunk_mse[index].tolist())
            per_episode[episode_id]["success"].extend(
                (first_step_mae[index] < success_threshold).tolist()
            )
    return {
        episode_id: {
            "samples": len(stats["success"]),
            "first_step_mae": float(torch.tensor(stats["first_step_mae"]).mean()),
            "first_step_mae_raw": float(torch.tensor(stats["first_step_mae_raw"]).mean()),
            "chunk_mae": float(torch.tensor(stats["chunk_mae"]).mean()),
            "chunk_mse": float(torch.tensor(stats["chunk_mse"]).mean()),
            "success_rate": sum(stats["success"]) / len(stats["success"]),
        }
        for episode_id, stats in sorted(per_episode.items())
    }


def persistence_baseline(
    payload: dict,
    indices: list[int],
    *,
    success_threshold: float,
) -> dict[str, float]:
    """Copy-previous-action baseline: predict a_t by a_{t-1}.

    PNPW-style 30 FPS data is smooth, so this trivial policy already passes
    the first-step threshold; every model metric must be read against it
    (see Past-Token Prediction / copycat analysis, arXiv:2505.09561).
    """
    selected = torch.tensor(indices)
    previous = payload["previous_action"][selected]
    actions = payload["actions"][selected]
    first_mae = (previous[:, :, None, :] - actions[:, :, :1, :]).abs().mean(dim=(-1, -2))
    chunk_mae = (previous[:, :, None, :] - actions).abs().mean(dim=(-1, -2))
    return {
        "first_mae_norm": float(first_mae.mean()),
        "chunk_mae_norm": float(chunk_mae.mean()),
        "success_rate": float((first_mae < success_threshold).float().mean()),
    }


def summarize(stats: dict[int, dict[str, float]]) -> dict[str, float]:
    values = list(stats.values())
    flat_first = torch.tensor([entry["first_step_mae"] for entry in values])
    flat_chunk = torch.tensor([entry["chunk_mae"] for entry in values])
    flat_mse = torch.tensor([entry["chunk_mse"] for entry in values])
    flat_success = torch.tensor([entry["success_rate"] for entry in values])
    flat_first_raw = torch.tensor([entry["first_step_mae_raw"] for entry in values])
    return {
        "episodes": len(values),
        "samples": sum(entry["samples"] for entry in values),
        "first_step_mae_mean": float(flat_first.mean()),
        "first_step_mae_std": float(flat_first.std()),
        "first_step_mae_p95": float(flat_first.quantile(0.95)),
        "first_step_mae_raw_mean": float(flat_first_raw.mean()),
        "chunk_mae_mean": float(flat_chunk.mean()),
        "chunk_mae_std": float(flat_chunk.std()),
        "chunk_mse_mean": float(flat_mse.mean()),
        "success_rate_mean": float(flat_success.mean()),
        "success_rate_std": float(flat_success.std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-episode open-loop rollout evaluation of a VA compound checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--vision-pooling",
        choices=("flat", "spatial"),
        default=None,
        help="defaults to the pooling recorded in the checkpoint contract",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=10,
        help="evaluate the last N episodes (sorted by episode_id); use -1 for all",
    )
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument(
        "--perturb",
        choices=("none", "blank", "swap"),
        default="none",
        help="language perturbation: blank zeros the language stream; "
        "swap replaces it with the next task's instruction (instruction_id +1)",
    )
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)  # 固定 flow 采样噪声（口径要求：重跑可复现，2026-08-05 审查补充）
    if args.val_episodes == 0:
        raise ValueError("--val-episodes must be positive or -1")
    if args.success_threshold <= 0.0:
        raise ValueError("--success-threshold must be positive")

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
    vision_key = "vision_tokens_spatial" if pooling == "spatial" else "vision_tokens"
    flow_steps = args.flow_steps or int(contract.get("flow_steps", 8))

    dataset = FeatureDataset(
        args.data,
        require_pairs=False,
        vision_key=vision_key,
    )
    payload = dataset.payload
    if args.perturb == "blank":
        payload = dict(payload)
        payload["language_hidden"] = torch.zeros_like(payload["language_hidden"])
        if "language_mask" in payload:
            payload["language_mask"] = torch.zeros_like(payload["language_mask"])
    elif args.perturb == "swap":
        inst = payload["instruction_id"].tolist()
        unique = sorted(set(inst))
        per_inst = {
            value: payload["language_hidden"][inst.index(value)] for value in unique
        }
        payload = dict(payload)
        payload["language_hidden"] = torch.stack(
            [per_inst[(value + 1) % len(unique)] for value in inst]
        )
    episode_ids = sorted(payload["episode_id"].unique().tolist())
    if args.val_episodes == -1:
        selected = episode_ids
    else:
        if args.val_episodes > len(episode_ids):
            raise ValueError(
                f"--val-episodes {args.val_episodes} exceeds available {len(episode_ids)}"
            )
        selected = episode_ids[-args.val_episodes :]
    indices = [
        index
        for index in range(dataset.length)
        if int(payload["episode_id"][index]) in selected
    ]
    if not indices:
        raise ValueError("no samples in the selected evaluation episodes")

    model = VACompoundPolicy(config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)

    q01 = payload["normalization"]["action_q01"]
    q99 = payload["normalization"]["action_q99"]
    action_scale = (q99 - q01) / 2
    per_episode = evaluate_policy(
        model,
        payload,
        indices,
        torch.device(args.device),
        flow_steps=flow_steps,
        success_threshold=args.success_threshold,
        batch_size=args.batch_size,
        action_scale=action_scale,
    )
    baseline = persistence_baseline(
        payload,
        indices,
        success_threshold=args.success_threshold,
    )
    summary = summarize(per_episode)

    # 宏平均 + bootstrap 95% CI（固定种子）：任务（instruction_id，缺失时退化为
    # episode_id）为宏平均与重采样单元，口径见 va_compound/statistics.py。
    if "instruction_id" in payload:
        group_field = payload["instruction_id"]
    else:
        group_field = payload["episode_id"]
    episode_group = {}
    for index in indices:
        episode = int(payload["episode_id"][index])
        if episode not in episode_group:
            episode_group[episode] = int(group_field[index])
    groups = np.asarray([episode_group[ep] for ep in per_episode])
    episode_chunk = np.asarray([entry["chunk_mae"] for entry in per_episode.values()])
    episode_success = np.asarray(
        [entry["success_rate"] for entry in per_episode.values()]
    )
    chunk_macro, chunk_lo, chunk_hi = macro_bootstrap_ci(
        episode_chunk, groups, n_boot=2000, seed=0
    )
    success_macro, success_lo, success_hi = macro_bootstrap_ci(
        episode_success, groups, n_boot=2000, seed=0
    )
    print(
        f"summary_macro: tasks={len(np.unique(groups))} "
        f"chunk_mae_norm={fmt_ci(chunk_macro, chunk_lo, chunk_hi)} "
        f"success={success_macro:.1%} [{success_lo:.1%}, {success_hi:.1%}]"
    )

    print(
        f"checkpoint={args.checkpoint} pooling={pooling} flow_steps={flow_steps} "
        f"episodes={summary['episodes']} samples={summary['samples']}"
    )
    print(
        "success_threshold(norm)="
        f"{args.success_threshold} -> denormalized MAE {float((args.success_threshold * (q99 - q01) / 2).mean()):.4f}"
    )
    print(
        f"{'episode':>8} {'samples':>7} {'first_mae_norm':>14} {'chunk_mae_norm':>14} "
        f"{'first_mae_raw':>13} {'success':>8}"
    )
    for episode_id, entry in per_episode.items():
        print(
            f"{episode_id:>8} {entry['samples']:>7} {entry['first_step_mae']:>14.5f} "
            f"{entry['chunk_mae']:>14.5f} {entry['first_step_mae_raw']:>13.4f} {entry['success_rate']:>7.1%}"
        )
    print(
        f"summary: first_mae_norm={summary['first_step_mae_mean']:.5f}±"
        f"{summary['first_step_mae_std']:.5f} p95={summary['first_step_mae_p95']:.5f} "
        f"first_mae_raw={summary['first_step_mae_raw_mean']:.4f} "
        f"chunk_mae_norm={summary['chunk_mae_mean']:.5f}±{summary['chunk_mae_std']:.5f} "
        f"chunk_mse_norm={summary['chunk_mse_mean']:.5f} "
        f"success={summary['success_rate_mean']:.1%}±{summary['success_rate_std']:.1%}"
    )
    print(
        f"baseline(persistence): first_mae_norm={baseline['first_mae_norm']:.5f} "
        f"chunk_mae_norm={baseline['chunk_mae_norm']:.5f} "
        f"success={baseline['success_rate']:.1%}\n"
        f"vs baseline: success {summary['success_rate_mean'] - baseline['success_rate']:+.1%} "
        f"chunk_mae {summary['chunk_mae_mean'] - baseline['chunk_mae_norm']:+.5f} "
        f"(negative is better for MAE)"
    )

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "samples",
                    "first_step_mae_norm",
                    "chunk_mae_norm",
                    "success_rate",
                    "pooling",
                ],
            )
            writer.writeheader()
            for episode_id, entry in per_episode.items():
                writer.writerow(
                    {
                        "episode": episode_id,
                        "samples": entry["samples"],
                        "first_step_mae_norm": f"{entry['first_step_mae']:.6f}",
                        "chunk_mae_norm": f"{entry['chunk_mae']:.6f}",
                        "success_rate": f"{entry['success_rate']:.6f}",
                        "pooling": pooling,
                    }
                )
        print(f"saved={args.csv.resolve()}")


if __name__ == "__main__":
    main()
