#!/usr/bin/env python
"""MT-VJ 阶段 V 诊断：per-role RMSE / 查询退化 / heatmap 熵 / GT 标签统计 / 叠图。

跑法：
    python scripts/diag_metric_field.py --tasks peg-insert-side-v3,assembly-v3,hand-insert-v3

输出：
    - 每任务每角色 RMSE（可见角色，px）+ 两个基线（图像中心 / 本批 GT 均值）
    - 角色查询两两 cosine（每个任务文本一组）——角色退化检测
    - heatmap 熵（均匀 ln576≈6.36 参考）
    - spatial_bias 每角色 argmax 固定点——"空间先验收敛"检测
    - GT 标签统计（每角色位置均值/散布/可见率）
    - 叠图 /tmp/metric_diag.png（GT 圆 + 预测叉，当前帧）
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from prepare_metaworld_metric import ROLE_NAMES, make_metric_batch  # noqa: E402
from scripts.build_longtraj_features import ENV_TO_TASK  # noqa: E402
from train_metric_visual import (  # noqa: E402
    IMAGE_SIZE,
    build_language_cache,
    gather_language,
    preprocess_frames,
)
from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone  # noqa: E402
from va_compound.live_vjepa import _dense_coords  # noqa: E402
from va_compound.metric_visual_head import LanguageMetricField  # noqa: E402

COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255)]


def softmax_entropy(log_heatmap: torch.Tensor) -> torch.Tensor:
    """log_heatmap [B, R, 24, 24]（log P，t 已求和）→ 每样本每角色熵 [B, R]。"""
    p = log_heatmap.exp()
    return -(p * (p + 1e-12).log()).sum(dim=(-2, -1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/metric_field.pt")
    ap.add_argument("--tasks", default="peg-insert-side-v3,assembly-v3,hand-insert-v3")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-png", default="/tmp/metric_diag.png")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(f"[diag] ckpt: steps_done={ck['config'].get('steps_done')} "
          f"tasks={ck['config'].get('tasks')} contract={ck.get('contract')}")
    metric_head = LanguageMetricField().to(device)
    metric_head.load_state_dict(ck["metric_head"])
    metric_head.eval()

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    vision_backbone.freeze_all()
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    language_cache, lang_ok = build_language_cache(
        text_backbone, [ENV_TO_TASK[t] for t in tasks]
    )
    print(f"[diag] Qwen 语言缓存: {'OK' if lang_ok else 'DEGRADED'}")
    coords = torch.from_numpy(_dense_coords()).to(device)

    # spatial_bias 固定点（每角色 argmax patch → y,x 图像坐标）
    bias = metric_head.spatial_bias.detach().reshape(4, -1).cpu()  # [4, 1152]
    argmax_idx = bias.argmax(dim=1)
    print("\n[spatial_bias] 每角色 argmax patch（0-1 y,x）→ 固定位置检测:")
    for r in range(4):
        idx = int(argmax_idx[r])
        y = (idx % 576) // 24
        x = idx % 24
        t_slice = idx // 576
        print(f"  {ROLE_NAMES[r]}: bias_max={bias[r].max():.3f} "
              f"argmax=(t={t_slice}, y={y / 24:.3f}, x={x / 24:.3f})")

    tiles: list = []
    for task in tasks:
        rng = np.random.default_rng(args.seed)
        batch = make_metric_batch(task, rng, args.n)
        kp = torch.from_numpy(batch["keypoints"]).to(device)
        vis = torch.from_numpy(batch["visibility"]).to(device)
        video = preprocess_frames(batch["frames"], device)
        with torch.no_grad():
            h5, h11 = vision_backbone.encode_multi(video, out_layers=(5, 11))
            lang_hidden, lang_mask = gather_language(
                language_cache, batch["language_text"], device
            )
            out = metric_head(h5, h11, lang_hidden, lang_mask, coords)

        err_px = (out.p - kp) * IMAGE_SIZE  # [B, 4, 2]
        print(f"\n=== task {task} (n={args.n}, 语言文本 {batch['language_text'][0]!r}) ===")
        print("  [GT 标签统计] 每角色 均值位置/散布(px)/可见率:")
        for r in range(4):
            kp_px = kp[:, r].cpu().numpy() * IMAGE_SIZE
            visr = vis[:, r].cpu().numpy()
            frac = float(visr.mean())
            if frac > 0:
                mean_pos = kp_px[visr > 0.5].mean(axis=0)
                spread = kp_px[visr > 0.5].std(axis=0)
                print(f"    {ROLE_NAMES[r]:9s} mean=({mean_pos[0]:6.1f},{mean_pos[1]:6.1f}) "
                      f"std=({spread[0]:5.1f},{spread[1]:5.1f}) px vis={frac:.2f}")
            else:
                print(f"    {ROLE_NAMES[r]:9s} —— 全部不可见 vis={frac:.2f}")

        print("  [RMSE] 可见角色像素误差:")
        for r in range(4):
            m = vis[:, r] > 0.5
            if m.sum() == 0:
                print(f"    {ROLE_NAMES[r]:9s} n/a（无可见样本）")
                continue
            e = torch.sqrt((err_px[m, r] ** 2).sum(-1))
            pred_px = out.p[:, r][m].cpu().numpy() * IMAGE_SIZE
            kp_px = kp[:, r][m].cpu().numpy() * IMAGE_SIZE
            center_rmse = float(
                np.sqrt(((pred_px - 192.0) ** 2).sum(-1).mean())
            )
            gt_mean_rmse = float(
                np.sqrt(((pred_px - kp_px.mean(axis=0)) ** 2).sum(-1).mean())
            )
            print(f"    {ROLE_NAMES[r]:9s} RMSE={float(e.mean()):6.2f}px  "
                  f"(基线: 图中心 {center_rmse:6.2f}px, 本批GT均值 {gt_mean_rmse:6.2f}px)")

        # 预测跨样本散布（坍缩检测：pred 方差 ≈ 0 → 固定点先验）
        pred_px_all = (out.p.detach().cpu().numpy() * IMAGE_SIZE)
        vis_np = vis.cpu().numpy()
        print("  [预测散布] 每角色跨样本 std（px）:")
        for r in range(4):
            m = vis_np[:, r] > 0.5
            if m.sum() > 1:
                std = pred_px_all[m, r].std(axis=0)
                print(f"    {ROLE_NAMES[r]:9s} std=({std[0]:5.1f},{std[1]:5.1f}) px")

        # heatmap 熵（可见角色）
        ent = softmax_entropy(out.log_heatmap.detach())
        mask = (vis > 0.5).float()
        ent_mean = (ent * mask).sum() / mask.sum().clamp_min(1.0)
        print(f"  [heatmap 熵] 可见角色均值 {float(ent_mean):.2f} "
              f"(均匀 576 格参考 ln576={math.log(576):.2f})")

        # 叠图：前 2 个样本（GT 圆 + 预测叉）
        from PIL import Image, ImageDraw

        for i in range(min(2, args.n)):
            frame = batch["frames"][i, -1]
            img = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(img)
            for r in range(4):
                y, x = kp[i, r].cpu().numpy()
                gy, gx = y * IMAGE_SIZE, x * IMAGE_SIZE
                py_, px_ = out.p[i, r].cpu().numpy()
                py_, px_ = py_ * IMAGE_SIZE, px_ * IMAGE_SIZE
                if vis[i, r] > 0.5:
                    draw.ellipse([gx - 7, gy - 7, gx + 7, gy + 7],
                                 outline=COLORS[r], width=3)
                else:
                    draw.ellipse([gx - 7, gy - 7, gx + 7, gy + 7],
                                 outline=(128, 128, 128), width=1)
                draw.line([px_ - 6, py_ - 6, px_ + 6, py_ + 6], fill=COLORS[r], width=3)
                draw.line([px_ - 6, py_ + 6, px_ + 6, py_ - 6], fill=COLORS[r], width=3)
                draw.text((gx + 9, gy - 5), f"{ROLE_NAMES[r][0]}(GT)", fill=COLORS[r])
                draw.text((px_ + 9, py_ - 5), f"{ROLE_NAMES[r][0]}(P)", fill=COLORS[r])
            tiles.append(img)

    # 角色查询两两 cosine（每任务文本）
    print("\n[角色查询] 每任务文本 4 角色查询两两 cosine（退化→接近 1）:")
    unique_texts = sorted({ENV_TO_TASK[t] for t in tasks})
    for text in unique_texts:
        hidden, mask = language_cache[text]
        hidden, mask = hidden.to(device), mask.to(device)
        with torch.no_grad():
            lp = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / (
                mask.sum().clamp_min(1.0)
            )
            q = metric_head.lang_pool(lp.float()).view(1, 4, -1)
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cos = (qn @ qn.transpose(1, 2))[0]
        row = ", ".join(f"r{r}:{cos[r].tolist()}" for r in range(4))
        print(f"  {text!r}: {row}")

    if tiles:
        out = Image.new("RGB", (len(tiles) * IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        for j, img in enumerate(tiles):
            out.paste(img, (j * IMAGE_SIZE, 0))
        out.save(args.out_png)
        print(f"\n[diag] 叠图已保存 {args.out_png}（{len(tiles)} 个样本并排；"
              f"圆=GT 叉=预测 同色同角色）")


if __name__ == "__main__":
    main()
