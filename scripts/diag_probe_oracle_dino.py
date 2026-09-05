#!/usr/bin/env python
"""DINO 几何探针（P0，2026-08-16）：冻结 DINOv2 在 MetaWorld 域的定位线性可读性。

与 scripts/diag_probe_oracle.py（V-JEPA 版，8-13px 门）同协议，只换编码器：
- 帧源：make_metric_batch 仿真真值（keypoints/visibility 0-1）；
- 编码：TimmActionVisionBackbone（冻结 fp16，224px，block11/block23）；
- 特征：帧 [d-2,d] 两片全 16×16 patch（512 token，1024 维），不池化；
- 测试：
  A. Oracle 读出量化下限（16 网格 @224px：patch 14px，std≈4.0px）；
  B. 模板匹配线性探针（= 语言度量头视觉通路的线性天花板）：
     block11 / block23 分别测，test RMSE vs GT 均值基线。
  判据：test RMSE < ~15px 且显著低于 GT 均值基线 → 几何在特征里；
        否则 DINO 冻结特征在该域无 mm 级几何 → DINO 线出局。

用法：python scripts/diag_probe_oracle_dino.py --task peg-insert-side-v3 --n-probe 32
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from prepare_metaworld_metric import ROLE_NAMES, make_metric_batch  # noqa: E402
from va_compound.model import VACompoundConfig  # noqa: E402
from va_compound.vision.encoding import _build_dino_main_backbone  # noqa: E402

GRID = 16
SLICES = 2
IMAGE_SIZE = 224
PATCH_PX = IMAGE_SIZE / GRID  # 14 px
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _coords01() -> np.ndarray:
    rows = []
    for t in range(2):
        for y in range(GRID):
            for x in range(GRID):
                rows.append(((y + 0.5) / GRID, (x + 0.5) / GRID))
    return np.asarray(rows, dtype=np.float64)  # [512, 2] (y,x) 0-1


def preprocess(batch_frames: np.ndarray, device) -> torch.Tensor:
    """make_metric_batch 帧 [B,4,384,384,3] → 帧 [d-2,d] 两片 → [B,2,3,224,224] 归一化。"""
    sel = np.ascontiguousarray(batch_frames[:, (2, 3)])  # [B,2,H,W,3]
    b, s = sel.shape[:2]
    imgs = torch.from_numpy(sel.reshape(b * s, *sel.shape[2:])).permute(0, 3, 1, 2)
    imgs = imgs.float().div_(255.0).to(device)
    if tuple(imgs.shape[-2:]) != (224, 224):
        imgs = F.interpolate(imgs, size=(224, 224), mode="bicubic",
                             align_corners=False, antialias=True)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (imgs - mean) / std  # [B*2, 3, 224, 224]


def oracle_readout(batch: dict) -> None:
    kp = batch["keypoints"]  # [B, 4, 2] y,x 0-1
    coords01 = _coords01()
    n = kp.shape[0]
    print("\n=== A. Oracle 读出量化下限（16 网格 @224px）===")
    print(f"    patch 中心误差 std ≈ {PATCH_PX / math.sqrt(12):.2f}px, "
          f"max ≈ {PATCH_PX / 2:.1f}px")
    for name, slices in (("t=0 片", [0]), ("t=1 片", [1]), ("两片各半", [0, 1])):
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
                pred = (flat[:, None] * coords01).sum(axis=0)
                errs.append(float(np.linalg.norm((pred - kp[i, r]) * IMAGE_SIZE)))
            print(f"      {ROLE_NAMES[r]:9s} RMSE={math.sqrt(np.mean(np.square(errs))):6.2f}px")
    print("    GT 每角色跨样本 std（px）:")
    for r in range(4):
        std = kp[:, r].std(axis=0) * IMAGE_SIZE
        print(f"      {ROLE_NAMES[r]:9s} std=({std[0]:.1f},{std[1]:.1f})")


def nearest_centroid_probe(feats: np.ndarray, batch: dict, name: str) -> None:
    """模板匹配线性探针（同 V-JEPA 版协议，网格 16）。"""
    kp = batch["keypoints"]
    n = feats.shape[0]
    n_train = max(1, n * 3 // 4)
    fn = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-9)  # [n,512,D]
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n)
    gt_loc = (
        np.clip(np.floor(kp[:, :, 0] * GRID).astype(int), 0, GRID - 1) * GRID
        + np.clip(np.floor(kp[:, :, 1] * GRID).astype(int), 0, GRID - 1)
    )
    print(f"\n=== B. 模板匹配线性探针（{name}，样本 {n}：train {n_train}/test {n - n_train}）===")
    for slice_mode in ("t=1 片", "t=0 片", "两片混合"):
        slices = [0, 1] if slice_mode == "两片混合" else [0 if slice_mode == "t=0 片" else 1]
        print(f"    [{slice_mode}] 模板用片 {slices}，候选 patch 取 t=1 片:")
        for r in range(4):
            w = np.zeros(feats.shape[-1])
            for i in train_idx:
                for s in slices:
                    w += fn[i, s * GRID * GRID + gt_loc[i, r]]
            w /= len(slices) * len(train_idx)
            wn = w / (np.linalg.norm(w) + 1e-9)
            errs, hits = [], 0
            for i in test_idx:
                s = fn[i, GRID * GRID:, :] @ wn  # [256]
                best = int(np.argmax(s))
                pred_yx = np.array([(best // GRID + 0.5) / GRID,
                                    (best % GRID + 0.5) / GRID])
                err = float(np.linalg.norm((pred_yx - kp[i, r]) * IMAGE_SIZE))
                errs.append(err)
                if err <= PATCH_PX:
                    hits += 1
            rmse = math.sqrt(float(np.mean(np.square(errs))))
            mean_kp = kp[train_idx, r].mean(axis=0)
            base = math.sqrt(float(np.mean(np.square(
                np.linalg.norm((mean_kp - kp[test_idx, r]) * IMAGE_SIZE, axis=1)
            ))))
            print(f"      {ROLE_NAMES[r]:9s} test RMSE={rmse:6.2f}px "
                  f"hit={hits}/{len(errs)}  (GT 均值基线 {base:6.2f}px)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="peg-insert-side-v3")
    ap.add_argument("--n-oracle", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--main-vision-checkpoint", type=Path,
                    default=Path("/home/ryan/.cache/huggingface/hub/models--timm--"
                                 "vit_large_patch14_reg4_dinov2.lvd142m/snapshots/"
                                 "f3c408e77602bb412aa65fb03dfa0d5f95cb3832/"
                                 "model.safetensors"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    batch_o = make_metric_batch(args.task, rng, args.n_oracle)
    oracle_readout(batch_o)

    batch_p = make_metric_batch(args.task, rng, args.n_probe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = VACompoundConfig(
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        main_vision_image_size=224,
        main_vision_dim=1024,
        main_vision_grid=8,
        main_vision_frames=4,
        main_vision_tokens=256,
    )
    import argparse as _ap

    backbone = _build_dino_main_backbone(
        _ap.Namespace(main_vision_checkpoint=args.main_vision_checkpoint),
        config, device,
    )
    video = preprocess(batch_p["frames"], device)  # [B*2, 3, 224, 224]
    with torch.no_grad():
        hierarchical = backbone.forward_hierarchical_dense(video.half())
    for key, name in ((5, "block11"), (11, "block23")):
        feats = hierarchical[key].float().cpu().numpy()  # [B*2, 256, 1024]
        feats = feats.reshape(-1, 2, 256, 1024).reshape(-1, 512, 1024)
        nearest_centroid_probe(feats, batch_p, name)


if __name__ == "__main__":
    main()
