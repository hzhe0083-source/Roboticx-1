#!/usr/bin/env python
"""fullframe Stage1/2 executed-action 标签重建（denorm → clip(raw,-1,1) → 重新归一化）。

读 data/metaworld_fullframe_skeleton.pt，写 data/metaworld_fullframe_executed.pt。

动机（已实测，2026-08-09）：fullframe skeleton 训练标签用裁剪前 raw action 的
q01/q99 归一化（action_q01=[-4.08,-3.84,-12.84,-1]、action_q99=[9.37,8.18,10,1]），
但 MetaWorld 环境执行动作时 clip(raw, -1, 1)：
  - 64.1% 样本至少一个 xyz 分量超出 [-1,1]，z 维 38.35% 超界（raw 最大 10、最小 -12.84）
  - raw z=1/5/10 在环境里执行成同一个动作，训练标签却不同（同输入多目标噪声）
  - z∈[-1,1] 物理范围在归一化空间仅占 [0.037,0.212]，等权 flow-MSE 下精细幅度
    监督权重被压缩 ~5 倍 → 精细任务（抓取/插入）标签既被污染又被弱化
本脚本完全复用 scripts/make_v5_executed_actions.py 的已验证管线（纯函数同构）。

prev 契约（Grok 审查纠正）：prepare_metaworld.py 构建时已正确置零 episode
首决策、其余为前一帧真实动作——重建脚本不再改动 t=0。

用法：python scripts/make_fullframe_executed.py [--quantile 0.01]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "metaworld_fullframe_skeleton.pt"
OUT = ROOT / "data" / "metaworld_fullframe_executed.pt"
ACTION_CONTRACT = "executed-clip-fullframe"


def denorm(norm_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """归一化动作 → 原始动作（与 eval_metaworld.py 逐字一致：norm*(q99-q01)/2+(q99+q01)/2）。"""
    return norm_actions * (q99 - q01) / 2 + (q99 + q01) / 2


def executed_from_actions(
    norm_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> torch.Tensor:
    """归一化标签 → 环境实际执行的原始动作（MetaWorld 执行时 clip(raw,-1,1)）。"""
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
    """完整 executed 标签管线：denorm → clip(raw,-1,1) → 重算分位数 → 重新归一化。

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
    """previous_action 走同一 executed 管线，与 actions 共享新归一化空间。

    t=0 契约（Grok 审查纠正，2026-08-09）：prepare_metaworld.py 构建时已正确处理
    ——episode 绝对首决策（start+offset*cs==0）置零、其余为前一帧真实动作
    （正确的 teacher-forcing prev）。本脚本**不得**再置零 t=0，否则把 89%
    正确的中段 prev 清成 0，训练"首决策 prev=0"与闭环中段自激 prev≠0 错配。
    q01/q99=±1 恒等映射下原 0 值保持 0，非零值经 clip+renorm 映射。
    """
    executed = executed_from_actions(previous_action, old_q01, old_q99)
    labels = renorm_executed(executed, new_q01, new_q99)
    # episode 首决策（原 prev t=0 全零的行）显式保持真零：归一化零点经新分位数
    # 映射后偏离零（[-0.39,-0.36,+0.12,0]），必须保留，否则与闭环首决策 prev=0 错配
    # （v5 管线同款约定；Grok 审查纠正：非首决策的 t=0 保留真实前一帧动作）。
    first_decision = (previous_action[:, 0].abs().sum(-1) == 0)
    if int(first_decision.sum().item()) > 0:
        labels[:, 0][first_decision] = 0.0
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

    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    payload = torch.load(SRC, map_location="cpu", weights_only=True)
    old_q01 = payload["normalization"]["action_q01"]
    old_q99 = payload["normalization"]["action_q99"]

    labels, new_q01, new_q99 = build_executed_labels(payload["actions"], old_q01, old_q99)
    prev = process_previous_action(
        payload["previous_action"], old_q01, old_q99, new_q01, new_q99
    )
    executed = executed_from_actions(payload["actions"], old_q01, old_q99)

    # ---- 验证 1：新旧标签不一致比例（预期 element ~19.7%、row ~64%） ----
    old_labels = payload["actions"]
    changed = old_labels != labels
    rows_changed = changed.any(dim=(-1, -2, -3))
    print(f"[1] label mismatch: element={changed.float().mean().item():.4%} "
          f"row={rows_changed.float().mean().item():.4%} "
          f"(rows={int(rows_changed.sum().item())}/{old_labels.shape[0]})")

    # ---- 验证 2：去饱和效果（|label|==1 占比变化） ----
    for name, tensor in (("skeleton", old_labels), ("executed", labels)):
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

    # ---- prev 契约：episode 首决策 0 保留 + t>0 保持 prev[t]==actions[t-1][5] ----
    prev_err = float((prev[:, 1:] - labels[:, :-1, 5]).abs().max().item())
    nonzero = int((prev[:, 0].abs().sum(-1) > 0).sum())
    print(f"[prev] t=0 nonzero rows kept: {nonzero}/{prev.shape[0]} "
          f"(中段 teacher-forcing prev，仅 episode 首决策为 0)")
    print(f"[prev] max |prev_v[t] - actions_v[t-1][5]| = {prev_err:.6g} "
          f"(DECISION_STRIDE=6 contract)")
    print(f"[q] old q01={old_q01.numpy()} q99={old_q99.numpy()}")
    print(f"[q] new q01={new_q01.numpy()} q99={new_q99.numpy()}")

    # ---- 写新文件：其余键逐字节复制，normalization.state 不变 ----
    out = dict(payload)
    out["actions"] = labels
    out["previous_action"] = prev
    out["normalization"] = dict(payload["normalization"])
    out["normalization"]["action_q01"] = new_q01
    out["normalization"]["action_q99"] = new_q99
    out["metadata"] = dict(payload["metadata"])
    out["metadata"]["action_contract"] = ACTION_CONTRACT
    out["metadata"]["source"] = SRC.name
    torch.save(out, OUT)
    print(f"[out] wrote {OUT} (action_contract={ACTION_CONTRACT}, source={SRC.name})")


if __name__ == "__main__":
    main()
