"""判定实验：World→VA 的消息通道在部署时到底贡献了多少成功率。

peer_sync_h6 的核心主张是 World 每个 stage 发布 world_message，作为下一层 VA
注意力的 K/V，给策略一份可空间寻址的预测记忆。本脚本保留 World 的全部状态递推
（belief/world_map 照常更新、照常算），只把交给 VA 的那份消息置零，其余配置与
基线评测逐项相同。

  成功率≈基线 => VA 实际上没在用这份消息，双模型交互在部署时是空转
  成功率显著下降 => 消息确实在贡献，交互通道有效

用法（在服务器 ORA0 根目录）：
    python scripts/diag_world_message_ablation.py <ckpt> <features> [zero|frozen]

``frozen`` 模式改为把 stage 0 的消息复用到全部 8 个 stage，用于区分「消息内容
无用」与「逐 stage 精化无用」。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = sys.argv[1]
FEATURES = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "zero"
if MODE not in {"zero", "frozen"}:
    raise SystemExit(f"mode must be 'zero' or 'frozen', got {MODE}")

from va_compound.policy.model import VACouplingLayer  # noqa: E402

_orig_forward = VACouplingLayer.forward
_orig_forward_sequential = VACouplingLayer.forward_sequential
_stats = {"calls": 0, "message_norm": 0.0}
_frozen: list[torch.Tensor] = []


def _ablate(kwargs):
    message = kwargs.get("state")
    if message is None:
        return kwargs
    _stats["calls"] += 1
    _stats["message_norm"] += float(message.detach().float().norm())
    if MODE == "zero":
        ablated = torch.zeros_like(message)
    else:
        if not _frozen:
            _frozen.append(message.detach().clone())
        ablated = _frozen[0] if _frozen[0].shape == message.shape else message
    return {**kwargs, "state": ablated}


def patched_forward(self, *args, **kwargs):
    return _orig_forward(self, *args, **_ablate(kwargs))


def patched_forward_sequential(self, *args, **kwargs):
    return _orig_forward_sequential(self, *args, **_ablate(kwargs))


VACouplingLayer.forward = patched_forward
VACouplingLayer.forward_sequential = patched_forward_sequential
print(f"[ABLATE] world_message mode={MODE}", flush=True)

import eval_metaworld as E  # noqa: E402

DINO = os.environ.get(
    "DINO",
    "/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors",
)
OUT = ROOT / "logs" / f"world_message_ablation_{MODE}.json"

sys.argv = [
    "eval_metaworld.py",
    "--checkpoint", CKPT,
    "--features", FEATURES,
    "--main-vision-checkpoint", DINO,
    "--task-ids", "0,16",
    "--trials-per-task", os.environ.get("TRIALS_PER_TASK", "10"),
    "--execution-horizon", "2",
    "--horizon", "500",
    "--direct-head", "auto",
    "--flow-samples", "1",
    "--device", "cuda",
    "--output-json", str(OUT),
]

try:
    E.main()
finally:
    calls = _stats["calls"]
    mean_norm = _stats["message_norm"] / max(calls, 1)
    print(
        f"[ABLATE] reader calls={calls} mean |world_message|={mean_norm:.4f} "
        f"(pre-ablation magnitude, confirms the channel carried signal)",
        flush=True,
    )
