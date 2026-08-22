"""量化 peer joint 训练里 VA 损失与 World 损失各自贡献的梯度。

`backward_peer_joint_losses` 先 `va_loss.backward()` 再 `world_loss.backward()`，
两次反传累积到同一个 optimizer step。训练契约是
World 目标会写进 VA 层；Flow 只训练预测图的 policy projection/reader，不再写入
st_predictor。日志里 world_objective≈12 而 flow≈0.11，但损失量级不等于
梯度量级，本脚本按参数分组精确测量两者的实际梯度贡献比。

做法：在 VA backward 之后逐参数快照 `.grad`，World backward 之后再取差值，
得到 World 的精确贡献张量，然后按参数名首段分组统计 L2 范数。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train as T  # noqa: E402

STEPS_TO_REPORT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
# 可选第二个参数：exact-resume checkpoint。在训练过的权重上复测，用于区分
# "world→action 敏感度本来就是 0" 与 "只是初始化时小、训练后会长大"。
RESUME = sys.argv[2] if len(sys.argv) > 2 else None
TAG = "resumed" if RESUME else "scratch"
OUT = ROOT / "logs" / f"diag_peer_gradient_balance.{TAG}.json"

_orig_backward = T.backward_peer_joint_losses
_records: list[dict] = []


def _find_model():
    """训练循环的局部变量里取 model（诊断专用，避免改动 train.py 结构）。"""
    frame = sys._getframe(2)
    while frame is not None:
        candidate = frame.f_locals.get("model")
        if candidate is not None and hasattr(candidate, "named_parameters"):
            return candidate
        frame = frame.f_back
    raise RuntimeError("could not locate model in caller frames")


def _group(name: str) -> str:
    head = name.split(".", 1)[0]
    if head in {"wmrm"}:
        return "world(wmrm)"
    if head in {"flow", "flow_head", "velocity"}:
        return f"flow({head})"
    return f"va({head})"


def patched_backward(va_loss, world_loss_or_forward):
    model = _find_model()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    va_loss.backward()
    # VA backward 之后的快照：这是 VA 侧的纯贡献。
    after_va = {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in params}

    result = (
        world_loss_or_forward()
        if callable(world_loss_or_forward)
        else world_loss_or_forward
    )
    world_loss = result[0] if isinstance(result, tuple) else result
    world_loss.backward()

    va_sq: dict[str, float] = defaultdict(float)
    world_sq: dict[str, float] = defaultdict(float)
    for name, param in params:
        group = _group(name)
        base = after_va[name]
        if base is not None:
            va_sq[group] += float(base.double().pow(2).sum())
        if param.grad is None:
            continue
        # World 的精确贡献 = 累积后的 grad 减去 VA backward 留下的那部分。
        delta = param.grad.detach().double()
        if base is not None:
            delta = delta - base.double()
        world_sq[group] += float(delta.pow(2).sum())

    record = {
        "va_loss": float(va_loss.detach()),
        "world_loss": float(world_loss.detach()),
        "va_grad_norm": {k: va_sq[k] ** 0.5 for k in sorted(va_sq)},
        "world_grad_norm": {k: world_sq[k] ** 0.5 for k in sorted(world_sq)},
    }
    _records.append(record)
    tag = f"[GRADBAL {len(_records):03d}]"
    print(
        f"{tag} va_loss={record['va_loss']:.4f} world_loss={record['world_loss']:.4f}",
        flush=True,
    )
    for group in sorted(set(va_sq) | set(world_sq)):
        v = va_sq[group] ** 0.5
        w = world_sq[group] ** 0.5
        ratio = (w / v) if v > 0 else float("inf")
        print(
            f"{tag}   {group:22s} va={v:.4e} world={w:.4e} world/va={ratio:.2f}x",
            flush=True,
        )
    if len(_records) >= STEPS_TO_REPORT:
        OUT.write_text(json.dumps(_records, indent=2), encoding="utf-8")
        print(f"[GRADBAL] wrote {OUT}", flush=True)
        raise SystemExit(0)
    return result


T.backward_peer_joint_losses = patched_backward

DINO = (
    "/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2."
    "lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors"
)

sys.argv = [
    "train.py",
    "--va-data", "data/hard2_peer_h6_p2_va_train_v1.pt",
    "--world-data", "data/hard2_peer_h6_p2_world_train_v1.pt",
    "--visual-world-supervision",
    "--world-split-manifest", "data/hard2_peer_h6_p2_world_split_v1.json",
    "--va-world-mode", "peer_sync_h6",
    "--planning-stride", "2", "--control-stride", "2",
    "--wam4va", "--wmrm-inject", "all", "--wmrm-target", "dino",
    "--wmrm-adep-weight", "0", "--wmrm-cycle-steps", "2",
    "--wmrm-world-weight", "1.0",
    "--dino-main-vision", "--dino-dense-metric",
    "--main-vision-checkpoint", DINO,
    "--main-vision-grid", "16", "--main-vision-frames", "4",
    "--main-vision-temporal", "--main-vision-temporal-scale", "1.0",
    "--main-vision-encode-batch", "8",
    "--metric-geometry-inject", "--wmrm-map-size", "16",
    "--wmrm-map-channels", "1024", "--wmrm-world-grid", "16",
    "--wmrm-predictor", "st_blocks", "--wmrm-predictor-depth", "6",
    "--wmrm-predictor-width", "384", "--wmrm-predictor-heads", "12",
    "--single-task", "--task-sampling", "balanced",
    "--task-locality-block-batches", "64", "--batch-size", "18",
    "--sequence-length", "4", "--min-sequence-length", "4",
    "--num-workers", "0", "--lr", "0.0001", "--seed", "0", "--device", "cuda",
    "--feature-autocast-bf16", "--va-layers", "8",
    "--va-attention-backend", "auto", "--flow-cond", "adaln",
    "--flow-layers", "6", "--flow-steps", "8",
    "--flow-prefix-steps", "2", "--flow-prefix-weight", "1.0",
    "--flow-tail-weight", "0.036",
    "--mtvj-train-metric-head", "--lr-mtvj-metric-head", "0.0003",
    "--mtvj-train-relation", "--lr-mtvj-relation", "0.00002",
    "--mtvj-visual-aux-every", "10", "--mtvj-visual-aux-batch", "8",
    "--save-every", "0", "--save", "/tmp/gradbal_probe.pt",
]

sys.argv += ["--steps", str(STEPS_TO_REPORT + 2)]
if RESUME:
    # 只需要训练过的权重来测 ∂(flow loss)/∂(world_action_readout)；--resume 跳过
    # exact 契约校验（该契约嵌了绝对本地路径，无法跨机器复用）。
    sys.argv += ["--resume-weights", RESUME]
    print(f"[GRADBAL] loading trained weights from {RESUME}", flush=True)

T.main()
