#!/usr/bin/env python
"""C²-VA Stage A: v5 executed-action labels（denorm → clip(raw,-1,1) → 重新归一化）。

读 data/metaworld_features_v4.pt，写 data/metaworld_features_v5.pt 与
"Press a button" 单任务子集 data/mw_buttonpress_v5.pt。

动机（已实测）：v4 标签 = robust_normalize(raw) 后 clip 到 [-1,1]，但 MetaWorld
环境执行动作时 clip(raw, -1, 1)。21.4% 的 raw 动作超出 [-1,1]（max 10.0 /
min -12.7），导致同一执行动作对应不同标签（raw=1.0 → 标签 -0.94，raw=7.3 →
标签 1.0，执行完全相同）——标签污染。

v5 标签管线（纯函数在顶部，tests/test_direct_head.py 直接 import）：
  1. denorm 回 raw（与 eval_metaworld.py 相同的线性映射）；
  2. clip(raw, -1, 1) = 环境实际执行的动作；
  3. 用 executed 分布按维重算 1%/99% 分位数（gripper 维天然在 [-1,1]）；
  4. 重新归一化。executed∈[-1,1]，但新分位数下低于 q01 / 高于 q99 的尾值映射后
     仍会越过 ±1，故保留 clip —— 保证标签域与 eval 解码路径（clip→denorm）完全一致。
previous_action 走同一管线（与 v4 相同：actions 与 prev 共享同一 normalization
字典），t=0 保持全零（无先前动作；归一化零经新分位数映射后偏离零，须显式保留）。
v4 契约 prev[t] == actions[t-1][5]（DECISION_STRIDE=6，前一个 chunk 的第 6 步），
管线为确定性函数，v5 中该关系自动保持（脚本输出交叉验证误差）。

用法：python scripts/make_v5_executed_actions.py [--quantile 0.01]
（2026-08-07 从 /tmp 迁入仓库，Codex 修正 9：避免测试依赖临时文件）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent  # 仓库根（迁移自 /tmp，2026-08-07 Codex 修正 9）
V4 = ROOT / "data" / "metaworld_features_v4.pt"
V5 = ROOT / "data" / "metaworld_features_v5.pt"
BUTTONPRESS = ROOT / "data" / "mw_buttonpress_v5.pt"
BUTTON_TASK = "Press a button"
ACTION_CONTRACT = "executed-clip-v5"

N_DIM_KEYS = (
    "vision_tokens",
    "language_hidden",
    "language_mask",
    "proprio",
    "previous_action",
    "actions",
    "pair_id",
    "instruction_id",
    "episode_id",
)


def denorm(norm_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """归一化动作 → 原始动作（与 eval_metaworld.py 逐字一致：norm*(q99-q01)/2+(q99+q01)/2）。"""
    return norm_actions * (q99 - q01) / 2 + (q99 + q01) / 2


def executed_from_actions(
    norm_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> torch.Tensor:
    """v4 归一化标签 → 环境实际执行的原始动作（MetaWorld 执行时 clip(raw,-1,1)）。"""
    return torch.clamp(denorm(norm_actions, q01, q99), -1.0, 1.0)


def quantiles_from_executed(
    executed: torch.Tensor, quantile: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """executed 分布按维的 1%/99% 分位数（gripper 维天然在 [-1,1]，同样处理）。"""
    flat = executed.reshape(-1, executed.shape[-1])
    q_low, q_high = torch.quantile(
        flat, torch.tensor([quantile, 1.0 - quantile], dtype=flat.dtype), dim=0
    )
    return q_low, q_high


def renorm_executed(
    executed: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> torch.Tensor:
    """executed 原始动作 → 新归一化标签（robust_normalize，尾值保留 clip）。"""
    mid = (q01 + q99) / 2
    half_range = (q99 - q01) / 2
    scale = torch.where(
        torch.abs(half_range) < 1e-8, torch.ones_like(half_range), half_range
    )
    return torch.clamp((executed - mid) / scale, -1.0, 1.0)


def build_executed_labels(
    norm_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """完整 v5 标签管线：denorm → clip(raw,-1,1) → 重算分位数 → 重新归一化。

    Returns (labels, new_q01, new_q99)。
    """
    executed = executed_from_actions(norm_actions, q01, q99)
    new_q01, new_q99 = quantiles_from_executed(executed)
    labels = renorm_executed(executed, new_q01, new_q99)
    return labels, new_q01, new_q99


def process_previous_action(
    previous_action: torch.Tensor,
    old_q01: torch.Tensor,
    old_q99: torch.Tensor,
    new_q01: torch.Tensor,
    new_q99: torch.Tensor,
) -> torch.Tensor:
    """previous_action 走同一 executed 管线，且与 actions 共享同一新归一化空间。

    v4 中 previous_action 与 actions 共用同一个 normalization 字典，v5 保持该
    约定（eval 的 prev 反馈 last_norm 也在此空间）：用 actions 派生的新分位数
    重新归一化，而不是给 prev 单独算一套分位数（否则两套空间互不可比）。
    t=0 保持全零（无先前动作的契约标记，与 eval 首决策 prev=0 一致）。
    """
    executed = executed_from_actions(previous_action, old_q01, old_q99)
    labels = renorm_executed(executed, new_q01, new_q99)
    labels[:, 0] = 0.0
    return labels


def check_bucket_consistency(
    executed: torch.Tensor, labels: torch.Tensor, decimals: int = 6
) -> tuple[int, int]:
    """同一 executed raw 值（四舍五入到 decimals 位）必须对应唯一标签。

    Returns (bucket_count, violation_count)。确定性管线下应为 0 违规。
    """
    bucket_count = 0
    violations = 0
    for dim in range(executed.shape[-1]):
        key = torch.round(executed[..., dim], decimals=decimals).reshape(-1)
        value = torch.round(labels[..., dim], decimals=decimals).reshape(-1)
        order = torch.argsort(key)
        key_sorted, value_sorted = key[order], value[order]
        same_key = key_sorted[1:] == key_sorted[:-1]
        diff_value = value_sorted[1:] != value_sorted[:-1]
        violations += int((same_key & diff_value).sum().item())
        bucket_count += int((key_sorted[1:] != key_sorted[:-1]).sum().item()) + 1
    return bucket_count, violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantile", type=float, default=0.01)
    args = parser.parse_args()
    if not (0.0 < args.quantile < 0.5):
        raise SystemExit("--quantile must be in (0, 0.5)")

    if not V4.exists():
        raise SystemExit(f"missing {V4}")
    payload = torch.load(V4, map_location="cpu", weights_only=True)
    old_q01 = payload["normalization"]["action_q01"]
    old_q99 = payload["normalization"]["action_q99"]

    labels, new_q01, new_q99 = build_executed_labels(payload["actions"], old_q01, old_q99)
    prev = process_previous_action(
        payload["previous_action"], old_q01, old_q99, new_q01, new_q99
    )
    executed = executed_from_actions(payload["actions"], old_q01, old_q99)

    # ---- 验证 1：新旧标签不一致比例（预期约 20% 行有变化） ----
    old_labels = payload["actions"]
    changed = old_labels != labels
    rows_changed = changed.any(dim=(-1, -2, -3))
    print(f"[1] label mismatch: element={changed.float().mean().item():.4%} "
          f"row={rows_changed.float().mean().item():.4%} "
          f"(rows={int(rows_changed.sum().item())}/{old_labels.shape[0]})")

    # ---- 验证 2：去饱和效果（|label|==1 占比下降） ----
    for name, tensor in (("v4", old_labels), ("v5", labels)):
        print(f"[2] {name} |label|==1 fraction: "
              f"{(tensor.abs() >= 1.0).float().mean().item():.4%}")

    # ---- 验证 3：同一 executed raw 值 → 唯一标签 ----
    bucket_count, violations = check_bucket_consistency(executed, labels)
    print(f"[3] executed-value buckets={bucket_count} violations={violations} "
          f"(expect 0)")

    # ---- 验证 4：denorm(新标签) ≈ executed（最大误差） ----
    roundtrip = denorm(labels, new_q01, new_q99)
    max_err = float((roundtrip - executed).abs().max().item())
    print(f"[4] max |denorm(new_label) - executed| = {max_err:.6g}")

    # ---- prev 一致性：v4 契约 prev[t]==actions[t-1][5] 在 v5 自动保持 ----
    prev_err = float((prev[:, 1:] - labels[:, :-1, 5]).abs().max().item())
    print(f"[prev] max |prev_v5[t] - actions_v5[t-1][5]| = {prev_err:.6g} "
          f"(v4 contract, DECISION_STRIDE=6)")
    print(f"[prev] t=0 kept zero: {bool((prev[:, 0] == 0).all().item())}")
    print(f"[q] old q01={old_q01.numpy()} q99={old_q99.numpy()}")
    print(f"[q] new q01={new_q01.numpy()} q99={new_q99.numpy()}")

    # ---- 写 v5：其余键逐字节复制 v4（值不变），normalization.state 不变 ----
    v5 = dict(payload)
    v5["actions"] = labels
    v5["previous_action"] = prev
    v5["normalization"] = dict(payload["normalization"])
    v5["normalization"]["action_q01"] = new_q01
    v5["normalization"]["action_q99"] = new_q99
    v5["metadata"] = dict(payload["metadata"])
    v5["metadata"]["action_contract"] = ACTION_CONTRACT
    v5["metadata"]["source"] = V4.name
    torch.save(v5, V5)
    print(f"[out] wrote {V5} (action_contract={ACTION_CONTRACT}, source={V4.name})")

    # ---- "Press a button" 单任务子集（索引运行时从数据读出） ----
    tasks = payload["metadata"]["tasks"]
    if BUTTON_TASK not in tasks:
        raise SystemExit(f"task {BUTTON_TASK!r} not found in v4 metadata.tasks")
    button_index = tasks.index(BUTTON_TASK)
    keep = payload["instruction_id"] == button_index
    subset = {
        key: (v5[key][keep] if key in N_DIM_KEYS else v5[key]) for key in v5
    }
    subset["instruction_id"] = torch.zeros(
        int(keep.sum().item()), dtype=torch.long
    )
    torch.save(subset, BUTTONPRESS)
    print(f"[out] wrote {BUTTONPRESS} (task={BUTTON_TASK!r} index={button_index} "
          f"samples={int(keep.sum().item())}, instruction_id->0)")


if __name__ == "__main__":
    main()
