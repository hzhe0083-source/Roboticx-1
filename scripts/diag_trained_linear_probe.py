#!/usr/bin/env python
"""Gate 1.5：训练式线性探针（隔离 head 参数化问题，2026-08-10）。

问题：模板探针（闭式，原始 h11，8-13px）通过；v2 head（共享 768→192 投影 +
双线性 + L2，tiny 集 58px）失败。本实验用「4 个 768 维原始空间查询 + L2 +
softmax 只过 t=1 片 576 patch + CE(σ=8px=0.5patch)」训练同一 64 样本集——
如果它能过拟合（RMSE<15px），说明 head 的投影/读出形式是瓶颈；
如果它也失败，说明是 SGD/CE/σ 层面的问题。

用法：python scripts/diag_trained_linear_probe.py [--steps 2000]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from train_metric_visual import IMAGE_SIZE, preprocess_frames  # noqa: E402
from va_compound.backbones import VJEPA21Backbone  # noqa: E402

GRID = 24
PATCH_PX = IMAGE_SIZE / GRID


def gaussian_targets(keypoints, sigma_px, grid=GRID, image_size=IMAGE_SIZE):
    """同 train_metric_visual（只对 24×24 单片）。"""
    sigma = sigma_px / (image_size / grid)
    yc = keypoints[..., 0:1] * grid - 0.5
    xc = keypoints[..., 1:2] * grid - 0.5
    yy = torch.arange(grid, device=keypoints.device, dtype=keypoints.dtype)
    xx = torch.arange(grid, device=keypoints.device, dtype=keypoints.dtype)
    dist2 = (yy.view(1, 1, 1, grid) - yc.unsqueeze(-1)) ** 2 + (
        xx.view(1, 1, grid, 1) - xc.unsqueeze(-1)
    ) ** 2
    target = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/metric_tiny64.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sigma-px", type=float, default=8.0)
    ap.add_argument("--temp-init", type=float, default=10.0)
    ap.add_argument("--init-template", action="store_true",
                    help="查询以闭式模板（GT 特征均值）初始化，区分'优化找不到'vs'目标不匹配'")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed = torch.load(args.data, map_location="cpu", weights_only=False)
    n = len(fixed["frames"])
    kp = torch.from_numpy(np.asarray(fixed["keypoints"])).float().to(device)  # [n,4,2]
    print(f"[probe] {n} 样本, device={device}, σ={args.sigma_px}px ({args.sigma_px/16:.2f} patch)")

    # 特征：h11（t=1 片 576 patch）
    backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    backbone.freeze_all()
    video = preprocess_frames(np.asarray(fixed["frames"]), device)
    with torch.no_grad():
        outs = backbone.encode_multi(video, out_layers=(11,))
    h11 = outs[0].float()[:, GRID * GRID:, :]  # [n, 576, 768]（t=1 片）
    del video, outs
    print(f"[probe] h11(t=1): {tuple(h11.shape)}")

    # 模型：4 个 768 维查询 + 可学习温度；L2 归一化；softmax over 576
    if args.init_template:
        # 闭式模板初始化：每角色 = 训练样本 GT 位置特征均值（归一化）
        kp_all = kp.cpu().numpy()
        with torch.no_grad():
            fn_all = h11 / h11.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        template = torch.zeros(4, 768, device=device)
        for r in range(4):
            locs = (
                np.clip(np.floor(kp_all[:, r, 0] * GRID).astype(int), 0, GRID - 1) * GRID
                + np.clip(np.floor(kp_all[:, r, 1] * GRID).astype(int), 0, GRID - 1)
            )
            template[r] = fn_all[torch.arange(n), locs].mean(dim=0)
        query = nn.Parameter(template)
        print("[probe] 查询以闭式模板初始化（GT 特征均值）")
    else:
        query = nn.Parameter(
            torch.empty(4, 768, device=device).uniform_(-0.05, 0.05)
        )
    temp = nn.Parameter(torch.tensor(args.temp_init, device=device))
    opt = torch.optim.Adam([query, temp], lr=args.lr)

    coords = torch.cartesian_prod(
        torch.arange(GRID, device=device), torch.arange(GRID, device=device)
    ).float()  # [576, 2] (y,x) 0-23
    patch_center = (coords + 0.5) / GRID  # 0-1 归一化

    n_train = n
    rmse_sum, rmse_cnt = 0.0, 0
    for step in range(args.steps):
        idx = torch.randperm(n, device=device)[: args.batch_size]
        fn = h11[idx] / h11[idx].norm(dim=-1, keepdim=True).clamp_min(1e-9)  # [B,576,768]
        qn = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # [4,768]
        scores = (fn @ qn.T) * temp  # [B,576,4] → [B,4,576]
        scores = scores.permute(0, 2, 1)
        logp = torch.log_softmax(scores, dim=-1)
        targets = gaussian_targets(kp[idx], args.sigma_px)  # [B,4,24,24]
        vis = torch.from_numpy(np.asarray(fixed["visibility"])).float().to(device)[idx]
        ce = -(targets.view(args.batch_size, 4, -1) * logp).sum(-1) * vis
        loss_ce = ce.sum() / vis.sum().clamp_min(1.0)
        # 位置（soft-argmax）
        probs = scores.softmax(dim=-1)
        p_hat = (probs.unsqueeze(-1) * patch_center.view(1, 1, 576, 2)).sum(2)  # [B,4,2]
        pos = (F.smooth_l1_loss(p_hat, kp[idx], reduction="none").sum(-1) * vis).sum() / vis.sum().clamp_min(1.0)
        loss = loss_ce + pos
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        err = (p_hat.detach() - kp[idx]) * IMAGE_SIZE
        rmse_sum += float((err.norm(dim=-1) ** 2 * vis).sum())
        rmse_cnt += float(vis.sum())
        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            rmse = math.sqrt(rmse_sum / max(rmse_cnt, 1))
            per_role = []
            for r in range(4):
                m = vis[:, r] > 0.5
                if m.sum():
                    per_role.append(
                        f"{r}={float((err[m, r].norm(dim=-1)).mean()):.1f}px"
                    )
            print(f"step {step+1}/{args.steps} loss {loss.item():.4f} "
                  f"ce {loss_ce.item():.4f} pos {pos.item():.4f} "
                  f"temp {float(temp.detach()):.2f} train RMSE {rmse:.2f}px "
                  f"[{' '.join(per_role)}]", flush=True)


if __name__ == "__main__":
    main()
