#!/usr/bin/env python
"""Paired held-out evaluation for the task35 DINO ROI v2 artifact.

The coarse input is a reproducible synthetic localization perturbation, because
the policy's coarse DINO metric head is trained jointly later.  This isolates
the standalone ROI mechanism: on identical held-out simulator renders, does the
frozen ROI head improve the forced pegHead-hole pair over the jittered coarse
coordinates that determine its crop?
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

from prepare_metaworld_metric import make_metric_batch
from scripts.train_metric_roi_dino import CANONICAL, ROI_GEOMETRY_SIZE, load_language
from va_compound.backbones import TimmActionVisionBackbone
from va_compound.longtraj_frames import LongTrajFramesDataset
from va_compound.metric_roi import (
    crop_metric_roi_video,
    gt_crop_visibility,
    load_dino_metric_roi_checkpoint,
    merge_roi_refinement,
    plan_metric_roi,
)
from va_compound.model import dense_coords


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarized_errors(error_px: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    values = error_px[valid].detach().cpu().numpy().astype(np.float64)
    if values.size == 0:
        raise ValueError("held-out evaluation has no visible selected roles")
    return {
        "n": int(values.size),
        "rmse_px": float(np.sqrt(np.mean(values**2))),
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--language-data", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1777)
    parser.add_argument("--jitter-px", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.batch < 1:
        raise ValueError("samples and batch must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_head = load_dino_metric_roi_checkpoint(args.checkpoint, device)
    backbone = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype="float16",
        model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        image_size=CANONICAL,
        feature_dim=1024,
        output_layers=(11, 23),
        checkpoint_path=args.dino_checkpoint,
        local_files_only=True,
    )
    backbone.freeze_all()
    dataset = LongTrajFramesDataset(
        args.language_data,
        min_sequence_length=4,
        feature_cache=None,
        include_frames=False,
    )
    lang_hid, lang_mask = load_language(dataset.payload, "peg-insert-side-v3", device)
    coords = dense_coords(512, device=device)
    rng = np.random.default_rng(args.seed)
    jitter_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    coarse_errors: list[torch.Tensor] = []
    roi_errors: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    pair_contracts: list[torch.Tensor] = []

    done = 0
    while done < args.samples:
        count = min(args.batch, args.samples - done)
        sim = make_metric_batch(
            "peg-insert-side-v3", rng, count, include_raw_frames=True
        )
        kp = torch.from_numpy(sim["keypoints"]).to(device=device, dtype=torch.float32)
        vis = torch.from_numpy(sim["visibility"]).to(device=device, dtype=torch.float32)
        jitter = (
            torch.rand(kp.shape, device=device, generator=jitter_gen) * 2.0 - 1.0
        ) * (args.jitter_px / ROI_GEOMETRY_SIZE)
        coarse = (kp + jitter).clamp(0.0, 1.0)
        selection = plan_metric_roi(
            coarse, vis, ROI_GEOMETRY_SIZE, forced_pair_index=1
        )
        raw_video = (
            torch.from_numpy(np.ascontiguousarray(sim["raw_frames"][:, (2, 3)]))
            .float()
            .div_(255.0)
            .permute(0, 1, 4, 2, 3)
            .to(device)
        )
        cropped = crop_metric_roi_video(
            raw_video,
            selection.roi,
            canonical_image_size=CANONICAL,
            roi_geometry_size=ROI_GEOMETRY_SIZE,
        )
        b, w = cropped.shape[:2]
        inputs = (cropped.reshape(b * w, 3, CANONICAL, CANONICAL) - mean) / std
        with torch.inference_mode():
            hierarchical = backbone.forward_hierarchical_dense(inputs.half())
            out = roi_head(
                hierarchical[5].reshape(b, -1, 1024).float(),
                hierarchical[11].reshape(b, -1, 1024).float(),
                lang_hid[None].expand(b, -1, -1),
                lang_mask[None].expand(b, -1),
                coords,
            )
            batch_index = torch.arange(b, device=device)[:, None]
            refined_crop = out.p[batch_index, selection.pair_roles]
            merged, _ = merge_roi_refinement(
                coarse,
                vis,
                refined_crop,
                selection,
                ROI_GEOMETRY_SIZE,
                alpha=1.0,
                max_delta_px=float(roi_head._mtvj_roi_config["max_delta_px"]),
            )
            refined = merged[batch_index, selection.pair_roles]
            target = kp[batch_index, selection.pair_roles]
            coarse_pair = coarse[batch_index, selection.pair_roles]
            selected_visible = gt_crop_visibility(
                kp, vis, selection.roi, ROI_GEOMETRY_SIZE
            )[batch_index, selection.pair_roles].bool()
            coarse_errors.append(
                (coarse_pair - target).norm(dim=-1) * ROI_GEOMETRY_SIZE
            )
            roi_errors.append(
                (refined - target).norm(dim=-1) * ROI_GEOMETRY_SIZE
            )
            valid_masks.append(selected_visible)
            pair_contracts.append(selection.pair_roles.detach().cpu())
        done += count
        print(f"evaluated {done}/{args.samples}", flush=True)

    coarse_error = torch.cat(coarse_errors)
    roi_error = torch.cat(roi_errors)
    valid = torch.cat(valid_masks)
    roles = torch.cat(pair_contracts)
    if not torch.equal(roles, torch.tensor([3, 2]).expand_as(roles)):
        raise ValueError("task35 ROI did not force pair [pegHead=3, hole=2]")
    coarse_summary = summarized_errors(coarse_error, valid)
    roi_summary = summarized_errors(roi_error, valid)
    paired = (roi_error - coarse_error)[valid]
    improvement_fraction = float((paired < 0).float().mean().detach().cpu())
    result = {
        "contract": "dino_metric_roi_task35_holdout_v1",
        "task": "peg-insert-side-v3",
        "role_order": ["tool", "pegGrasp", "hole", "pegHead"],
        "forced_pair": ["pegHead", "hole"],
        "samples": args.samples,
        "seed": args.seed,
        "jitter_px": args.jitter_px,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "coarse": coarse_summary,
        "roi": roi_summary,
        "rmse_delta_px": roi_summary["rmse_px"] - coarse_summary["rmse_px"],
        "paired_improvement_fraction": improvement_fraction,
        "gate": {
            "rmse_improved": roi_summary["rmse_px"] < coarse_summary["rmse_px"],
            "median_improved": roi_summary["median_px"] < coarse_summary["median_px"],
            "majority_improved": improvement_fraction > 0.5,
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
