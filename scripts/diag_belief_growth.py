"""量出闭环里 belief / world_map 的实际增长曲线，并验证 map 重锚是否生效。

静态估计（读训练过的 checkpoint 参数）给的是：一个完整 8-stage 周期的 stage
embedding 和范数 0.7734，初始 belief 每 token 0.3836，一集 2000 次 propose 的纯
累积项约 193.3 = 初始幅度的 504 倍。那是个下界，不含 belief_write 贡献。

本脚本把它变成实测：逐 propose 记录传入的 belief / world_map 范数，按决策点打表，
并统计 --world-map-reset-every 的实际触发次数。据此区分两件事：

  belief 范数随决策点单调爬升、map 范数每个决策点归位  => map 重锚生效，
      发散由 belief 独自驱动（即 stage_embed 累积是主因）
  map 范数同样爬升                                    => 重锚没生效，先查改动

用法（服务器 ORA0 根目录）：
    python scripts/diag_belief_growth.py <ckpt> <features> [task_id] [horizon]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = sys.argv[1]
FEATURES = sys.argv[2]
TASK_ID = sys.argv[3] if len(sys.argv) > 3 else "16"
HORIZON = sys.argv[4] if len(sys.argv) > 4 else "500"

import eval_metaworld  # noqa: E402
from va_compound.wmrm import WAM4VA  # noqa: E402

_orig_propose = WAM4VA.propose
_orig_reset_map = eval_metaworld._reset_world_map

_state = {"calls": 0, "decisions": 0, "map_resets": 0, "first_bad": None}


def _norm(tensor) -> float:
    if tensor is None:
        return float("nan")
    value = tensor.detach().float()
    if not torch.isfinite(value).all():
        return float("inf")
    return float(value.norm())


def patched_reset_map(memory):
    result = _orig_reset_map(memory)
    if result is not memory:
        _state["map_resets"] += 1
    return result


def patched_propose(self, *args, **kwargs):
    stage = int(kwargs.get("stage_index", 0))
    snapshot = kwargs.get("state")
    belief_norm = _norm(getattr(snapshot, "belief", None) if snapshot else None)
    map_norm = _norm(getattr(snapshot, "world_map", None) if snapshot else None)
    _state["calls"] += 1
    if stage == 0:
        _state["decisions"] += 1
        print(
            f"decision {_state['decisions']:>4}  propose {_state['calls']:>5}  "
            f"map_resets {_state['map_resets']:>4}  "
            f"|belief|={belief_norm:>12.3f}  |world_map|={map_norm:>12.3f}",
            flush=True,
        )
    proposal = _orig_propose(self, *args, **kwargs)
    message = proposal.world_message
    if _state["first_bad"] is None and not torch.isfinite(message).all():
        _state["first_bad"] = (_state["decisions"], _state["calls"], stage)
        print(
            f"!! first non-finite world_message at decision "
            f"{_state['decisions']}, propose {_state['calls']}, stage {stage}",
            flush=True,
        )
    return proposal


WAM4VA.propose = patched_propose
eval_metaworld._reset_world_map = patched_reset_map

sys.argv = [
    "eval_metaworld.py",
    "--checkpoint", CKPT,
    "--features", FEATURES,
    "--main-vision-checkpoint",
    "/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m"
    "/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors",
    "--task-ids", TASK_ID,
    "--trials-per-task", "1",
    "--execution-horizon", "2",
    "--horizon", HORIZON,
    "--direct-head", "auto",
    "--flow-samples", "1",
    "--device", "cuda",
    "--world-reset-every", "0",
    "--world-map-reset-every", "1",
]

try:
    eval_metaworld.main()
finally:
    print(
        f"\n=== summary: propose calls={_state['calls']} "
        f"decisions={_state['decisions']} map_resets={_state['map_resets']} "
        f"first_non_finite={_state['first_bad']} ===",
        flush=True,
    )
