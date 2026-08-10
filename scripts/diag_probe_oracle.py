#!/usr/bin/env python
"""MT-VJ 决定性诊断 v2（修复 oracle 逐角色 logits + RMSE 平方 + 探针分片）：

A. Oracle 读出测试：每角色独立 GT 峰 → 读出（δ=0）→ 每角色 RMSE。
   正确网格：RMSE ≈ patch 量化下限（16/√12 ≈ 4.6px）；错位 → 数十 px。
   t=0 / t=1 片单独 vs 两片各半（两片坐标相同，差异来自标签放置方式）。
B. 最近质心线性探针（分片报告）：d11 → GT patch 线性天花板。
   分开"信息不存在"（质心探针 ≈ GT 均值基线）与"SGD 没找到"
   （质心探针明显更好）。

用法：python scripts/diag_probe_oracle.py --task peg-insert-side-v3 --n-probe 32
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
from train_metric_visual import IMAGE_SIZE, preprocess_frames  # noqa: E402
from va_compound.backbones import VJEPA21Backbone  # noqa: E402
from va_compound.live_vjepa import _dense_coords  # noqa: E402

GRID = 24
SLICES = 2
PATCH_PX = IMAGE_SIZE / GRID  # 16 px


def _coords01() -> np.ndarray:
    coords = np.asarray(_dense_coords(), dtype=np.float64)  # [1152, 3] (t,y,x) [-1,1]
    return (coords[:, 1:] + 1.0) / 2.0  # [1152, 2] (y,x) 0-1


def oracle_readout(batch: dict) -> None:
    """A. 每角色独立 GT 峰 → soft-argmax 读出（δ=0）→ 每角色 RMSE。"""
    kp = batch["keypoints"]  # [B, 4, 2] y,x 0-1
    coords01 = _coords01()
    n = kp.shape[0]
    print("\n=== A. Oracle 读出（每角色独立 GT 峰 → soft-argmax，δ=0）===")
    print(f"    量化下限参考：patch 中心误差 std ≈ {PATCH_PX / math.sqrt(12):.2f}px，"
          f"max ≈ {PATCH_PX / 2:.1f}px")
    variants = {"t=0 片单独": [0], "t=1 片单独": [1], "两片各半": [0, 1]}
    for name, slices in variants.items():
        print(f"    [{name}]")
        for r in range(4):
            errs = []
            for i in range(n):
                logits = np.full((SLICES, GRID, GRID), -1e9)
                yi = min(GRID - 1, int(np.floor(kp[i, r, 0] * GRID)))
                xi = min(GRID - 1, int(np.floor(kp[i, r, 1] * GRID)))
                for s in slices:
                    logits[s, yi, xi] = 0.0
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                flat = probs.reshape(-1)
                pred = (flat[:, None] * coords01).sum(axis=0)  # [2] y,x
                errs.append(float(np.linalg.norm((pred - kp[i, r]) * IMAGE_SIZE)))
            rmse = math.sqrt(float(np.mean(np.square(errs))))
            print(f"      {ROLE_NAMES[r]:9s} RMSE={rmse:6.2f}px")
    print("    GT 每角色跨样本 std（px）:")
    for r in range(4):
        std = kp[:, r].std(axis=0) * IMAGE_SIZE
        print(f"      {ROLE_NAMES[r]:9s} std=({std[0]:.1f},{std[1]:.1f})")


def nearest_centroid_probe(d11: np.ndarray, batch: dict) -> None:
    """B. 模板匹配线性探针：w_r = train 样本 GT 位置的特征均值（归一化），
    测试样本逐 patch 打分 s_p = f_p·w_r → argmax → 位置。这正是 head 视觉通路
    （常量查询 = 学习线性模板）的线性天花板测试。"""
    kp = batch["keypoints"]
    n = d11.shape[0]
    n_train = max(1, n * 3 // 4)
    fn = d11 / (np.linalg.norm(d11, axis=-1, keepdims=True) + 1e-9)  # [n, 1152, 768]
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n)
    gt_loc = (
        np.clip(np.floor(kp[:, :, 0] * GRID).astype(int), 0, GRID - 1) * GRID
        + np.clip(np.floor(kp[:, :, 1] * GRID).astype(int), 0, GRID - 1)
    )  # [n, 4] 每角色 GT 位置（t=1 片内索引 0..575）
    print("\n=== B. 模板匹配线性探针（d11 常量查询 → GT patch 线性天花板）===")
    print(f"    样本 {n}（train {n_train} / test {n - n_train}）")
    for slice_mode in ("t=1 片", "t=0 片", "两片混合"):
        slices = [0, 1] if slice_mode == "两片混合" else [0 if slice_mode == "t=0 片" else 1]
        print(f"    [{slice_mode}] 模板用片 {slices}，候选 patch 取 t=1 片:")
        for r in range(4):
            w = np.zeros(d11.shape[-1])
            for i in train_idx:
                for s in slices:
                    w += fn[i, s * GRID * GRID + gt_loc[i, r]]
            w /= (len(slices) * len(train_idx))
            wn = w / (np.linalg.norm(w) + 1e-9)
            errs, hits = [], 0
            for i in test_idx:
                s = fn[i, GRID * GRID:, :] @ wn  # [576]
                best = int(np.argmax(s))
                pred_yx = np.array([(best // GRID + 0.5) / GRID,
                                    (best % GRID + 0.5) / GRID])
                err = float(np.linalg.norm((pred_yx - kp[i, r]) * IMAGE_SIZE))
                errs.append(err)
                if err <= PATCH_PX:
                    hits += 1
            rmse = math.sqrt(float(np.mean(np.square(errs))))
            # 探针自检：train 样本上的得分（应远好于 test，否则模板无信息）
            tr_errs = []
            for i in train_idx:
                s = fn[i, GRID * GRID:, :] @ wn
                best = int(np.argmax(s))
                pred_yx = np.array([(best // GRID + 0.5) / GRID,
                                    (best % GRID + 0.5) / GRID])
                tr_errs.append(float(np.linalg.norm((pred_yx - kp[i, r]) * IMAGE_SIZE)))
            tr_rmse = math.sqrt(float(np.mean(np.square(tr_errs))))
            mean_kp = kp[train_idx, r].mean(axis=0)
            base = math.sqrt(float(np.mean(np.square(
                np.linalg.norm((mean_kp - kp[test_idx, r]) * IMAGE_SIZE, axis=1)
            ))))
            print(f"      {ROLE_NAMES[r]:9s} test RMSE={rmse:6.2f}px "
                  f"hit={hits}/{len(errs)}  (train 自检 {tr_rmse:6.2f}px, "
                  f"GT均值基线 {base:6.2f}px)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="peg-insert-side-v3")
    ap.add_argument("--n-oracle", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    batch_o = make_metric_batch(args.task, rng, args.n_oracle)
    oracle_readout(batch_o)

    batch_p = make_metric_batch(args.task, rng, args.n_probe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        backbone = VJEPA21Backbone.from_pretrained(
            device=device, dtype="float16", local_files_only=True
        )
        backbone.freeze_all()
        video = preprocess_frames(batch_p["frames"], device)
        with torch.no_grad():
            outs = backbone.encode_multi(video, out_layers=(11,))
        d11 = outs[0].float().cpu().numpy()
        del video, outs
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("    GPU OOM → 回退 CPU")
        device = torch.device("cpu")
        backbone = VJEPA21Backbone.from_pretrained(
            device=device, dtype="float16", local_files_only=True
        )
        backbone.freeze_all()
        video = preprocess_frames(batch_p["frames"], device)
        with torch.no_grad():
            outs = backbone.encode_multi(video, out_layers=(11,))
        d11 = outs[0].float().cpu().numpy()
    print(f"    d11: {d11.shape}")
    nearest_centroid_probe(d11, batch_p)


if __name__ == "__main__":
    main()
