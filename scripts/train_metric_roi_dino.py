#!/usr/bin/env python
"""DINO ROI 精修头训练（2026-08-16，用户决策：ROI 特写路径）。

物理动机：冻结 DINO@224px 的 patch=14px，全帧定位量化下限 ~4px；把 480px
原始渲染里粗定位附近的 96-192px 区域裁剪放大到 224px 后，14px patch 覆盖原图 ~6-12px，
并配合连续 offset 回归获得相对全帧更细的局部定位。

协议（镜像 train_metric_roi.py，编码器换 DINO）：
- 数据：make_metric_batch 仿真真值（keypoints/visibility + true 480px raw_frames）；
- 粗定位：GT + 480 原图坐标的均匀抖动（±10px；plan_metric_roi 再叠加
  中心/尺寸抖动）→ 强制 pegHead-hole 的 96-192px 原图 ROI；
- 编码：crop_metric_roi_video 原图裁剪 → 224px 双时间片（帧 [d-2,d]）→
  冻结 DINO forward_hierarchical_dense → h5/h11 [B,512,1024]；
- 精修头：LanguageMetricField(h_dim=1024, grid=16, mode_readout/l2_norm/
  learnable_temp) 从零训练（Adam 3e-4）；
- loss：hinge+pos+offset（crop 坐标，224px 尺度）+ 所选角色对 BCE 可见度。

用法：
  python scripts/train_metric_roi_dino.py \
    --task peg-insert-side-v3 --steps 1000 --batch 16 --lr 3e-4 \
    --save checkpoints/dino_metric_roi_head_p35_1k.pt
"""
from __future__ import annotations

import argparse
import json
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

from prepare_metaworld_metric import make_metric_batch  # noqa: E402
from scripts.build_longtraj_features import ENV_TO_TASK  # noqa: E402
from train_metric_visual import compute_losses  # noqa: E402
from va_compound.metric_roi import (  # noqa: E402
    DINO_METRIC_ROI_CONTRACT,
    TASK35_METRIC_ROLE_CONTRACT,
    crop_metric_roi_video,
    crop_to_full,
    full_to_crop,
    gt_crop_visibility,
    plan_metric_roi,
)

CANONICAL = 224  # DINO 输入分辨率；ROI 裁剪后放大到该尺寸
ROI_GEOMETRY_SIZE = 480  # task35 crop 规划/修正均使用原始渲染像素


def load_language(dataset_payload, task: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """单任务预计算语言 hidden（与 _dino_visual_aux_loss 同源）。"""
    text = ENV_TO_TASK.get(task, task)
    id_all = dataset_payload["instruction_id"]
    tid = None
    tasks = dataset_payload.get("metadata", {}).get("tasks", [])
    for i, t in enumerate(tasks):
        if t == text:
            tid = i
            break
    if tid is None:
        raise KeyError(f"任务 {text!r} 不在 metadata.tasks")
    hits = (id_all == tid).nonzero()
    if hits.numel() == 0:
        raise KeyError(f"任务 {text!r} 无样本")
    row = int(hits[0, 0])
    hid = dataset_payload["language_hidden"][row].float().to(device)
    mask = dataset_payload["language_mask"][row].to(device)
    return hid, mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="peg-insert-side-v3")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--save", type=Path,
                    default=ROOT / "checkpoints/dino_metric_roi_head_p35_1k.pt")
    ap.add_argument("--dino-checkpoint", type=Path,
                    default=Path("/home/ryan/.cache/huggingface/hub/models--timm--"
                                 "vit_large_patch14_reg4_dinov2.lvd142m/snapshots/"
                                 "f3c408e77602bb412aa65fb03dfa0d5f95cb3832/"
                                 "model.safetensors"))
    ap.add_argument("--jitter-px", type=float, default=10.0,
                    help="GT 粗定位模拟误差（px，224 尺度）")
    ap.add_argument(
        "--language-data",
        type=Path,
        default=ROOT / "data/metaworld_longtraj_windows_h6_dino35_clean_v2_seed350.pt",
        help="仅用于读取 task35 预计算语言 hidden/mask 的 windows payload。",
    )
    args = ap.parse_args()
    if args.task != "peg-insert-side-v3":
        raise ValueError(
            "this v2 ROI trainer is bound to the task35 aligned role contract"
        )

    from va_compound.backbones import TimmActionVisionBackbone
    from va_compound.metric_visual_head import LanguageMetricField
    from va_compound.model import dense_coords
    from va_compound.longtraj_frames import LongTrajFramesDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = TimmActionVisionBackbone.from_pretrained(
        device=device, dtype="float16",
        model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        image_size=CANONICAL, feature_dim=1024, output_layers=(11, 23),
        checkpoint_path=args.dino_checkpoint, local_files_only=True,
    )
    backbone.freeze_all()
    roi_head = LanguageMetricField(
        lang_dim=2048, h_dim=1024, d_proj=192, n_roles=4,
        l2_norm=True, learnable_temp=True, mode_readout=True, grid=16,
    ).to(device)
    optimizer = torch.optim.Adam(roi_head.parameters(), lr=args.lr)
    dataset = LongTrajFramesDataset(
        args.language_data,
        min_sequence_length=4,
        feature_cache=None,
        include_frames=False,
    )
    lang_hid, lang_mask = load_language(dataset.payload, args.task, device)
    coords = dense_coords(512, device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    print(f"ROI head params: {sum(p.numel() for p in roi_head.parameters()):,}；"
          f"{args.steps} 步 × batch {args.batch}（sim 在线生成）", flush=True)

    for step in range(1, args.steps + 1):
        sim = make_metric_batch(
            args.task, rng, args.batch, include_raw_frames=True
        )
        frames_np = np.ascontiguousarray(
            sim["raw_frames"]
        )  # [B,4,480,480,3] true render pixels
        kp = torch.from_numpy(sim["keypoints"]).to(device)  # [B,4,2] 0-1
        vis = torch.from_numpy(sim["visibility"]).to(device)  # [B,4]
        # Coarse error and ROI geometry are expressed in native 480px render
        # pixels; only the extracted crop is resized to DINO's 224px input.
        jitter = (torch.rand_like(kp) * 2.0 - 1.0) * (
            args.jitter_px / ROI_GEOMETRY_SIZE
        )
        coarse = (kp + jitter).clamp(0.0, 1.0)
        selection = plan_metric_roi(
            coarse,
            vis,
            ROI_GEOMETRY_SIZE,
            center_jitter_px=args.jitter_px * 0.8,
            size_jitter=0.1,
            training=True,
            forced_pair_index=1,  # task35: (pegHead, hole)
        )
        raw_video = (
            torch.from_numpy(frames_np[:, (2, 3)]).float().div_(255.0)
            .permute(0, 1, 4, 2, 3).to(device)
        )  # [B,2,3,H,W]（crop_metric_roi_video 契约：NCHW）
        cropped = crop_metric_roi_video(
            raw_video,
            selection.roi,
            canonical_image_size=CANONICAL,
            roi_geometry_size=ROI_GEOMETRY_SIZE,
        )  # [B,2,3,224,224]
        b, w = cropped.shape[:2]
        inputs = ((cropped.reshape(b * w, 3, CANONICAL, CANONICAL) - mean) / std)
        with torch.no_grad():
            hierarchical = backbone.forward_hierarchical_dense(inputs.half())
        h5 = hierarchical[5].reshape(b, -1, 1024).float()
        h11 = hierarchical[11].reshape(b, -1, 1024).float()
        out = roi_head(
            h5, h11,
            lang_hid[None].expand(b, -1, -1),
            lang_mask[None].expand(b, -1),
            coords,
        )
        kp_crop = full_to_crop(kp, selection.roi, ROI_GEOMETRY_SIZE)
        crop_vis = gt_crop_visibility(
            kp, vis, selection.roi, ROI_GEOMETRY_SIZE
        )
        pair_mask = selection.role_mask.to(vis.dtype)
        loc_mask = crop_vis * pair_mask
        loc_loss, parts = compute_losses(
            out, kp_crop, loc_mask, torch.zeros(b, 6, device=device),
            loc_only=True, offset_supervision=True, hinge=True,
            hinge_margin=0.1, alias_consistency_weight=0.0,
            geometry_consistency_weight=0.0, image_size=CANONICAL,
        )
        vis_per = F.binary_cross_entropy_with_logits(
            out.visibility_logits, crop_vis, reduction="none"
        )
        vis_loss = (vis_per * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
        loss = loc_loss + vis_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # 指标：所选角色对在整帧坐标（224 尺度）的 RMSE
        batch_i = torch.arange(b, device=device)[:, None]
        refined_crop = out.p[batch_i, selection.pair_roles]
        refined_full = crop_to_full(
            refined_crop, selection.roi, ROI_GEOMETRY_SIZE
        )
        gt_full = kp[batch_i, selection.pair_roles]
        err_px = (
            (refined_full - gt_full).norm(dim=-1) * ROI_GEOMETRY_SIZE
        )
        rmse = torch.sqrt((err_px.square() * crop_vis[batch_i, selection.pair_roles])
                          .sum() / crop_vis[batch_i, selection.pair_roles].sum().clamp_min(1.0))
        if step % 50 == 0 or step == 1:
            print(
                f"step={step} loss={float(loss.detach()):.4f} hinge={parts['hinge']:.4f} "
                f"vis={float(vis_loss.detach()):.4f} rmse_full={float(rmse.detach()):.2f}px "
                f"(粗定位误差参考 {args.jitter_px * 0.58:.1f}px)",
                flush=True,
            )

    payload = {
        "roi_metric_head": {k: v.detach().cpu() for k, v in roi_head.state_dict().items()},
        "ctor_config": {
            "lang_dim": 2048, "h_dim": 1024, "d_proj": 192, "n_roles": 4,
            "l2_norm": True, "learnable_temp": True, "temp_init": 10.0,
            "freeze_bias": False, "mode_readout": True, "grid": 16,
        },
        "canonical_image_size": CANONICAL,
        "roi_geometry_size": ROI_GEOMETRY_SIZE,
        "min_roi_size": 96, "max_roi_size": 192, "distance_scale": 2.0,
        "max_delta_px": 32.0,
        "contract": DINO_METRIC_ROI_CONTRACT,
        "metric_role_contract": TASK35_METRIC_ROLE_CONTRACT,
        "role_order": ["tool", "pegGrasp", "hole", "pegHead"],
        "role_pairs": [[0, 1], [3, 2]],
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "jitter_px": args.jitter_px, "task": args.task,
        "raw_frame_contract": "true_simulator_render_480px_v1",
        "language_data": str(args.language_data.resolve()),
    }
    tmp = args.save.with_suffix(args.save.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
