#!/usr/bin/env python
"""M_fork：command-fork routing margin（Codex Q3 S1 主指标）。

在 held-out 严格 fork 对上评估：同一观测（同帧同 proprio/prev），只换语言指令，
模型预测动作应分别接近各自指令的正确专家分支。

  M_fork = [d(â_A, a*_B) + d(â_B, a*_A) − d(â_A, a*_A) − d(â_B, a*_B)]
           / [2 (d(a*_A, a*_B) + ε)]

M>0 = 输出不仅随语言变化，且朝各自正确分支路由；M≤0 = 变化方向错误或
语言不改变输出。d 用归一化空间 MAE（首动作维度）。

用法：
  python scripts/eval_fork_margin.py \
      --checkpoint checkpoints/pair_D_40k.pt \
      --fork data/mw_fork_drawer.pt \
      [--held-out-pairs 12] [--flow-steps 32] [--device cuda]
输出：逐对 M_fork + 配对 bootstrap 95% CI + 每分支 MAE。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 根模块（train 等）导入兼容

from train import FeatureDataset
from va_compound.model import VACompoundConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fork", type=Path, required=True, help="fork 数据集 .pt（held-out 对）")
    p.add_argument("--held-out-pairs", type=int, default=12,
                   help="从末尾取 N 对作为 held-out（与训练切分同侧：训练用前 60 对）")
    p.add_argument("--flow-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    model = VACompoundPolicy = __import__(
        "va_compound.model", fromlist=["VACompoundPolicy"]
    ).VACompoundPolicy(config)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    dataset = FeatureDataset(args.fork, require_pairs=False)
    payload = dataset.payload
    pairs = sorted(set(payload["pair_id"].tolist()))
    held = pairs[-args.held_out_pairs :]
    print(f"fork pairs: {len(pairs)}，held-out: {len(held)}")

    flow_steps = args.flow_steps
    margins, d_aa, d_ba = [], [], []  # d(â_A,a*_A), d(â_B,a*_A) 等
    for pid in held:
        rows = torch.nonzero(payload["pair_id"] == pid).flatten().tolist()
        assert len(rows) == 2
        i, j = rows
        # 分支 A/B 由 instruction_id 区分（0=A,1=B）
        if payload["instruction_id"][i] > payload["instruction_id"][j]:
            i, j = j, i
        # 同一观测（fork 契约：帧/proprio/prev 相同）
        obs = {k: payload[k][[i, j]].to(device) for k in
               ("vision_tokens", "proprio", "previous_action")}
        lang = payload["language_hidden"][[i, j]].to(device)
        mask = payload["language_mask"][[i, j]].to(device)
        actions = payload["actions"][[i, j], 0, 0]  # 专家首动作（归一化）
        a_star_a, a_star_b = actions[0], actions[1]

        with torch.inference_mode():
            cache = model.build_language_cache(lang, mask)
            preds = []
            for row in range(2):
                cond, _ = model.encode_condition(
                    obs["vision_tokens"][row : row + 1, 0],
                    obs["proprio"][row : row + 1, 0],
                    obs["previous_action"][row : row + 1, 0],
                    language_cache=cache,
                )
                # 确定性 Euler 32 步（noise=None 走 sample_actions 的确定性路径）
                acts = model.sample_actions(cond, steps=flow_steps)
                preds.append(acts[0, 0].cpu())
        a_hat_a, a_hat_b = preds[0], preds[1]

        def dist(x, y):
            return float((x - y).abs().mean())

        d_aa.append(dist(a_hat_a, a_star_a))
        d_bb = dist(a_hat_b, a_star_b)
        d_ab = dist(a_hat_a, a_star_b)
        d_ba_ = dist(a_hat_b, a_star_a)
        denom = 2.0 * (dist(a_star_a, a_star_b) + 1e-6)
        margins.append((d_ab + d_ba_ - d_aa[-1] - d_bb) / denom)
        print(
            f"pair {pid}: M_fork={margins[-1]:+.3f} "
            f"d(âA,a*A)={d_aa[-1]:.4f} d(âA,a*B)={d_ab:.4f} "
            f"d(âB,a*B)={d_bb:.4f} d(âB,a*A)={d_ba_:.4f}"
        )

    m = np.array(margins)
    rng = np.random.default_rng(0)
    boot = np.array(
        [m[rng.choice(len(m), len(m), replace=True)].mean() for _ in range(2000)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nM_fork mean={m.mean():+.3f} 95%CI=[{lo:+.3f},{hi:+.3f}] n={len(m)}")
    print(
        f"interpretation: M>0 = 语言把动作路由到正确分支（正=语言 grounding 直接证据）；"
        f"M<=0 = 方向错误或语言不改变输出"
    )


if __name__ == "__main__":
    main()
