"""量化 belief 循环里 stage embedding 的无衰减累积。

``_forward_from_snapshot`` 每次调用都执行 ``belief = belief + stage_embed[stage]``，
而这个 belief 会被写进 ``WAMState.belief`` 持久化、成为下一次 propose 的输入。
因此 stage embedding 会按 propose 次数线性累积进持久记忆——累积量与模型学到
什么无关，纯粹是调用次数的函数。

闭环 horizon=500、planning_stride=2 => 250 个决策点 × 8 个 stage = 2000 次 propose。
本脚本从训练过的 checkpoint 读实际参数，算出这个纯几何累积项的幅度，并与
belief 的初始幅度对比。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

CKPT = sys.argv[1]
N_DECISIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 250

payload = torch.load(CKPT, map_location="cpu", weights_only=True)
state = payload["model"] if "model" in payload else payload

stage_key = next(k for k in state if k.endswith("wmrm.stage_embed.weight"))
belief_key = next(k for k in state if k.endswith("wmrm.belief_tokens"))
stage_embed = state[stage_key].float()
belief_tokens = state[belief_key].float()

n_stages = stage_embed.shape[0]
n_calls = N_DECISIONS * n_stages
per_cycle = stage_embed.sum(dim=0)

print(f"checkpoint: {Path(CKPT).name}")
print(f"stage_embed: {tuple(stage_embed.shape)}  belief_tokens: {tuple(belief_tokens.shape)}")
print()
print(f"{'stage':>6} {'|embed|':>12}")
for i in range(n_stages):
    print(f"{i:>6} {stage_embed[i].norm():>12.4f}")
print()
print(f"|initial belief_tokens| (per token)   = {belief_tokens.norm(dim=-1).mean():.4f}")
print(f"|sum of one full 8-stage cycle|       = {per_cycle.norm():.4f}")
print()
print(f"闭环 horizon=500, planning_stride=2 => {N_DECISIONS} 决策点 x {n_stages} stage "
      f"= {n_calls} 次 propose")
print(f"纯 stage-embed 累积项 |{N_DECISIONS} x sum| = "
      f"{(per_cycle * N_DECISIONS).norm():.1f}")
ratio = (per_cycle * N_DECISIONS).norm() / belief_tokens.norm(dim=-1).mean()
print(f"相对初始 belief 幅度                  = {ratio:.0f}x")
print()
print("说明：该项不含任何 belief_write / belief_from_world 的贡献，是纯粹由 propose")
print("调用次数决定的几何累积。非线性 reader（evidence_from_belief / belief_write）")
print("读到被抬高的 belief 后输出同步放大，实际增长快于此线性下界。")
