#!/usr/bin/env python
"""held-out 定位评测（阶段 A 收尾，2026-08-12）：加载 metric_field_v5 视觉头，
用**新随机种子**在线仿真生成评测样本（模型训练时未见过的具体仿真状态），
按难度分组统计定位 RMSE（px，仅可见角色）与可见度准确率。

用法：
  python -u scripts/eval_metric_v5_heldout.py \
    --checkpoint checkpoints/metric_field_v5_all49.pt \
    --samples-per-task 5 --seed 12345 --device cuda

与训练同一套组件：VJEPA21Backbone（冻结 fp16）、QwenTextBackbone 语言缓存
（与 train_metric_visual 完全一致）、make_metric_batch 真值、LanguageMetricField
（checkpoint config strict 恢复）。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from prepare_metaworld_metric import make_metric_batch
from scripts.build_longtraj_features import ENV_TO_TASK
from scripts.mt50_difficulty import task_weights_for
from train_metric_visual import (
    build_language_cache,
    gather_language,
    preprocess_frames,
)
from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone
from va_compound.live_vjepa import _dense_coords
from va_compound.metric_visual_head import LanguageMetricField

SUPPORTED = sorted(ENV_TO_TASK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    known = {
        "lang_dim",
        "h_dim",
        "d_proj",
        "n_roles",
        "l2_norm",
        "learnable_temp",
        "temp_init",
        "freeze_bias",
        "mode_readout",
    }
    kwargs = {k: v for k, v in cfg.items() if k in known}
    metric_head = LanguageMetricField(**kwargs).to(device)
    metric_head.load_state_dict(ckpt["metric_head"], strict=True)
    metric_head.eval()
    print(f"loaded {args.checkpoint}: config={ {k: cfg.get(k) for k in known} }")

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    vision_backbone.freeze_all()
    vision_backbone.eval()
    lang_dtype, lang_device = "float16", device
    try:
        text_backbone = QwenTextBackbone.from_pretrained(
            device=lang_device, dtype=lang_dtype, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen 加载失败：{exc!r}") from exc
    language_cache, _ = build_language_cache(
        text_backbone, [ENV_TO_TASK[t] for t in SUPPORTED]
    )
    coords = torch.from_numpy(_dense_coords()).to(device)

    # 难度分组（easy/med/hard/vh），与训练加权同一来源
    weights = task_weights_for(SUPPORTED)
    w2name = {0.5: "easy", 1.0: "medium", 2.0: "hard", 3.0: "very_hard"}
    groups: dict[str, list[str]] = {}
    for task, w in zip(SUPPORTED, weights):
        groups.setdefault(w2name[float(w)], []).append(task)

    rng = np.random.default_rng(args.seed)
    totals: dict[str, dict] = {}
    for task in SUPPORTED:
        sim = make_metric_batch(task, rng, args.samples_per_task)
        frames = np.asarray(sim["frames"])
        video = preprocess_frames(frames, device)  # [B,4,3,384,384]
        with torch.no_grad():
            h5, h11 = vision_backbone.encode_multi(video, out_layers=(5, 11))
            h5, h11 = h5.float(), h11.float()
            lang_hidden, lang_mask = gather_language(
                language_cache, [ENV_TO_TASK[task]] * args.samples_per_task, device
            )
            out = metric_head(h5, h11, lang_hidden, lang_mask, coords)
        kp = torch.from_numpy(sim["keypoints"]).to(device)
        vis = torch.from_numpy(sim["visibility"]).to(device)
        err_px = (out.p - kp).norm(dim=-1) * 384.0  # [B, 4]
        rmse = torch.sqrt(
            ((err_px.square() * vis).sum() / vis.sum().clamp_min(1.0))
        ).item()
        vis_acc = (
            ((out.visibility_logits > 0.0).float() == vis).float().mean().item()
        )
        totals[task] = {"rmse": rmse, "vis_acc": vis_acc, "n_vis": int(vis.sum())}

    print("\n=== held-out 定位评测汇总（新仿真种子，未见样本）===")
    for gname, tasks_g in groups.items():
        rmses = [totals[t]["rmse"] for t in tasks_g]
        accs = [totals[t]["vis_acc"] for t in tasks_g]
        print(
            f"{gname:10s} ({len(tasks_g):2d} tasks): "
            f"RMSE {np.mean(rmses):6.2f}±{np.std(rmses):4.2f}px  "
            f"vis_acc {np.mean(accs):.3f}"
        )
    all_r = [totals[t]["rmse"] for t in SUPPORTED]
    all_a = [totals[t]["vis_acc"] for t in SUPPORTED]
    print(
        f"{'ALL':10s} ({len(SUPPORTED):2d} tasks): "
        f"RMSE {np.mean(all_r):6.2f}±{np.std(all_r):4.2f}px  "
        f"vis_acc {np.mean(all_a):.3f}"
    )
    worst = sorted(totals, key=lambda t: totals[t]["rmse"], reverse=True)[:5]
    print("最差 5 任务:", [(t, f"{totals[t]['rmse']:.1f}px") for t in worst])


if __name__ == "__main__":
    main()
