"""VA 推理频率实测（规格阶段 3 轻量对比的 84Hz 数据点）。

口径：一个决策点 = 4 帧窗口的 VA 前向一次 + Flow Head 32 步 Euler 积分
（部署配置：4 层 VA，--flow-steps 32，§3.5 部署口径）。
频率 = 1 / 单决策点墙钟时间。不包含 V-JEPA/Qwen 编码（预计算特征，见报告 §9）。

用法（GPU 空闲时）：
    python bench_inference.py --checkpoint checkpoints/libero_e2e_B40k.pt
    python bench_inference.py --layers 4 --flow-steps 32 --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from va_compound import VACompoundConfig, VACompoundPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VA decision-point inference benchmark")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = VACompoundConfig(**ckpt["config"])
    else:
        config = VACompoundConfig(num_layers=args.layers)
    model = VACompoundPolicy(config).to(device).eval()
    if args.checkpoint is not None:
        model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config: {config.num_layers} layers, flow_steps={args.flow_steps}, "
          f"params={n_params / 1e6:.2f}M")

    # 单决策点输入：4 帧窗口（与训练窗口一致），随机但固定形状
    B, T, P, D = 1, 4, 64, 768  # vision_tokens [B, T, tokens, dim]
    vision = torch.randn(B, T, P, D, device=device)
    proprio = torch.randn(B, T, config.proprio_dim, device=device)
    prev_action = torch.randn(B, T, config.action_dim, device=device)
    language = torch.randn(B, 16, config.language_dim, device=device)
    language_mask = torch.ones(B, 16, dtype=torch.bool, device=device)

    language_cache = model.build_language_cache(language, language_mask)
    conditions = []
    visual_memory = None
    for t in range(T):
        condition, visual_memory = model.encode_condition(
            vision[:, t], proprio[:, t], prev_action[:, t],
            language_cache=language_cache, visual_memory=visual_memory,
            return_visual_memory=True,
        )
        conditions.append(condition)
    conditions = torch.stack(conditions, dim=1).reshape(B * T, *conditions[0].shape[1:])

    def step() -> None:
        model.sample_actions(conditions, steps=args.flow_steps)

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / args.iters
    print(f"per-decision latency: {elapsed * 1e3:.2f} ms -> {1.0 / elapsed:.1f} Hz "
          f"({args.flow_steps} Euler steps, {config.num_layers}-layer VA)")


if __name__ == "__main__":
    main()
