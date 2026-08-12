#!/usr/bin/env python
"""Train an MT-VJ high-resolution ROI refiner on fresh simulator samples.

This is deliberately a standalone stage.  It never changes the policy path:
the fixed full-frame metric head chooses one actionable role pair and one crop,
then a second ``LanguageMetricField`` predicts that pair in crop coordinates.
V-JEPA, Qwen and the full-frame head remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from prepare_metaworld_metric import SUPPORTED_TASKS, make_metric_batch
from scripts.build_longtraj_features import ENV_TO_TASK
from train_metric_visual import (
    CONTRACT as COARSE_CONTRACT,
    IMAGE_MEAN,
    IMAGE_STD,
    build_language_cache,
    compute_losses,
    gather_language,
    task_for_step,
)
from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone
from va_compound.live_vjepa import _dense_coords
from va_compound.metric_roi import (
    crop_metric_roi_video,
    crop_to_full,
    full_to_crop,
    gt_crop_visibility,
    metric_head_state_sha256,
    plan_metric_roi,
    prepare_metric_roi_video,
)
from va_compound.metric_visual_head import LanguageMetricField, MetricFieldOutput


CONTRACT = "mt_vj_metric_roi_v1"
TRAINING_STATE_VERSION = 1
IMAGE_SIZE = 384
ROLE_PAIRS = ((0, 1), (3, 2))
COARSE_SOURCE_STAGE_V = "stage_v_checkpoint"
COARSE_SOURCE_POLICY = "policy_embedded_metric_head"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_checkpoint_identity(
    path: str | Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "contract": checkpoint.get("contract"),
    }


def _identity_mismatches(saved: Mapping[str, Any], current: Mapping[str, Any]) -> dict:
    return {
        key: (saved.get(key), current.get(key))
        for key in ("sha256", "size_bytes", "contract")
        if saved.get(key) != current.get(key)
    }


def metric_head_ctor(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(inspect.signature(LanguageMetricField.__init__).parameters) - {"self"}
    return {key: config[key] for key in keys if key in config}


def load_coarse_source(
    *,
    coarse_checkpoint_path: str | Path | None,
    coarse_policy_checkpoint_path: str | Path | None = None,
    runtime_metric_checkpoint_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Resolve either the original Stage-V head or a policy-embedded Stage-C head.

    The returned path/identity always name the immutable external Stage-V metric
    checkpoint required by runtime.  In policy mode only the actual embedded
    ``mtvj_metric_head`` weights are used for coarse predictions.
    """

    has_stage_v = coarse_checkpoint_path is not None
    has_policy = coarse_policy_checkpoint_path is not None
    if has_stage_v == has_policy:
        raise ValueError(
            "provide exactly one of --coarse-checkpoint or "
            "--coarse-policy-checkpoint"
        )
    if has_stage_v and runtime_metric_checkpoint_path is not None:
        raise ValueError(
            "--runtime-metric-checkpoint is only valid with "
            "--coarse-policy-checkpoint"
        )
    if has_policy and runtime_metric_checkpoint_path is None:
        raise ValueError(
            "--coarse-policy-checkpoint requires --runtime-metric-checkpoint"
        )

    runtime_path = Path(
        coarse_checkpoint_path if has_stage_v else runtime_metric_checkpoint_path
    ).expanduser().resolve(strict=True)
    runtime_checkpoint = torch.load(
        runtime_path, map_location="cpu", weights_only=True
    )
    runtime_config = dict(validate_coarse_checkpoint(runtime_checkpoint))
    runtime_identity = metric_checkpoint_identity(runtime_path, runtime_checkpoint)
    if has_stage_v:
        normalized = dict(runtime_checkpoint)
        source = COARSE_SOURCE_STAGE_V
    else:
        policy_path = Path(coarse_policy_checkpoint_path).expanduser().resolve(strict=True)
        policy = torch.load(policy_path, map_location="cpu", weights_only=True)
        training_contract = policy.get("training_contract")
        if not isinstance(training_contract, Mapping) or (
            training_contract.get("metric_head_checkpointed") is not True
        ):
            raise ValueError(
                "coarse policy must declare training_contract."
                "metric_head_checkpointed=True"
            )
        policy_state = policy.get("mtvj_metric_head")
        policy_config = policy.get("mtvj_metric_head_config")
        policy_identity = policy.get("mtvj_metric_checkpoint_identity")
        missing = [
            key
            for key, value in (
                ("mtvj_metric_head", policy_state),
                ("mtvj_metric_head_config", policy_config),
                ("mtvj_metric_checkpoint_identity", policy_identity),
            )
            if not isinstance(value, Mapping)
        ]
        if missing:
            raise ValueError(f"coarse policy is missing valid fields: {missing}")
        identity_mismatch = _identity_mismatches(policy_identity, runtime_identity)
        if identity_mismatch:
            raise ValueError(
                "runtime metric checkpoint differs from the policy's saved external "
                f"identity: {identity_mismatch}"
            )
        constructor_keys = set(
            inspect.signature(LanguageMetricField.__init__).parameters
        ) - {"self"}
        missing_policy_ctor = sorted(constructor_keys - set(policy_config))
        missing_runtime_ctor = sorted(constructor_keys - set(runtime_config))
        if missing_policy_ctor or missing_runtime_ctor:
            raise ValueError(
                "coarse policy/runtime metric constructor contract is incomplete: "
                f"policy_missing={missing_policy_ctor}, "
                f"runtime_missing={missing_runtime_ctor}"
            )
        policy_ctor = {key: policy_config[key] for key in constructor_keys}
        runtime_ctor = {key: runtime_config[key] for key in constructor_keys}
        if policy_ctor != runtime_ctor:
            raise ValueError(
                "coarse policy metric-head constructor differs from runtime metric "
                f"checkpoint: policy={policy_ctor}, runtime={runtime_ctor}"
            )
        normalized = {
            "contract": runtime_checkpoint["contract"],
            "config": runtime_config,
            "metric_head": policy_state,
        }
        source = COARSE_SOURCE_POLICY

    source_record = {
        "kind": source,
        "runtime_metric_checkpoint": dict(runtime_identity),
        "actual_coarse_head_state_sha256": metric_head_state_sha256(
            normalized["metric_head"]
        ),
    }
    return runtime_path, normalized, source_record


def validate_coarse_checkpoint(
    checkpoint: Mapping[str, Any], *, require_all_tasks: bool = True
) -> Mapping[str, Any]:
    if checkpoint.get("contract") != COARSE_CONTRACT:
        raise ValueError(
            f"coarse contract={checkpoint.get('contract')!r}; expected {COARSE_CONTRACT!r}"
        )
    if not isinstance(checkpoint.get("metric_head"), Mapping):
        raise ValueError("coarse checkpoint is missing metric_head state")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("coarse checkpoint is missing config")
    if config.get("language_cache_available") is not True:
        raise ValueError("coarse checkpoint was not trained with verified Qwen features")
    if config.get("loc_only") is not False:
        raise ValueError("ROI requires a coarse head trained with visibility supervision")
    tasks = config.get("tasks")
    if not isinstance(tasks, (list, tuple)):
        raise ValueError("coarse config.tasks must be a sequence")
    if require_all_tasks and set(tasks) != set(SUPPORTED_TASKS):
        missing = sorted(set(SUPPORTED_TASKS) - set(tasks))
        extra = sorted(set(tasks) - set(SUPPORTED_TASKS))
        raise ValueError(f"coarse checkpoint is not the all-49 head; missing={missing}, extra={extra}")
    steps_done = int(config.get("steps_done", 0))
    if steps_done < 10000 or steps_done != int(config.get("steps", -1)):
        raise ValueError(
            "ROI requires the completed all-49 coarse stage (steps_done == steps >= 10000)"
        )
    return config


def build_frozen_coarse_and_roi(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[LanguageMetricField, LanguageMetricField, dict[str, Any]]:
    config = dict(validate_coarse_checkpoint(checkpoint))
    ctor = metric_head_ctor(config)
    coarse = LanguageMetricField(**ctor).to(device)
    coarse.load_state_dict(checkpoint["metric_head"], strict=True)
    coarse.requires_grad_(False).eval()
    roi_head = LanguageMetricField(**ctor).to(device)
    roi_head.load_state_dict(checkpoint["metric_head"], strict=True)
    roi_head.train()
    return coarse, roi_head, ctor


def preprocess_raw_roi_frames(
    raw_frames: np.ndarray, roi: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Crop true raw-render pixels and return normalized canonical V-JEPA input."""
    frames = np.asarray(raw_frames)
    if frames.ndim != 5 or frames.shape[1] != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"raw_frames must be [B,4,S,S,3], got {frames.shape}"
        )
    if frames.shape[2] != frames.shape[3] or frames.shape[2] <= IMAGE_SIZE:
        raise ValueError("ROI training requires square raw renders larger than 384")
    raw_video = prepare_metric_roi_video(
        frames[:, None], device, image_size=None
    )
    cropped = crop_metric_roi_video(
        raw_video, roi, canonical_image_size=IMAGE_SIZE
    )
    mean = IMAGE_MEAN.to(device=device, dtype=cropped.dtype).unsqueeze(0)
    std = IMAGE_STD.to(device=device, dtype=cropped.dtype).unsqueeze(0)
    return (cropped - mean) / std


def preprocess_raw_full_frames(
    raw_frames: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Deployment-identical raw render -> canonical full-frame preprocessing."""
    raw_video = prepare_metric_roi_video(
        np.asarray(raw_frames)[:, None], device, image_size=IMAGE_SIZE
    )
    mean = IMAGE_MEAN.to(device=device, dtype=raw_video.dtype).unsqueeze(0)
    std = IMAGE_STD.to(device=device, dtype=raw_video.dtype).unsqueeze(0)
    return (raw_video - mean) / std


def compute_roi_losses(
    output: MetricFieldOutput,
    keypoints_full: torch.Tensor,
    visibility_full: torch.Tensor,
    selection,
    *,
    visibility_weight: float = 1.0,
    hinge_margin: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge + position + offset on crop-visible selected roles; BCE on the pair."""
    keypoints_crop = full_to_crop(keypoints_full, selection.roi, IMAGE_SIZE)
    crop_visibility = gt_crop_visibility(
        keypoints_full, visibility_full, selection.roi, IMAGE_SIZE
    )
    pair_mask = selection.role_mask.to(visibility_full.dtype)
    localization_mask = crop_visibility * pair_mask
    usable_pair = localization_mask.sum(dim=1) > 0
    dummy_relation = keypoints_full.new_zeros((keypoints_full.shape[0], 6))
    loc_loss, parts = compute_losses(
        output,
        keypoints_crop,
        localization_mask,
        dummy_relation,
        loc_only=True,
        offset_supervision=True,
        hinge=True,
        hinge_margin=hinge_margin,
        alias_consistency_weight=0.0,
        geometry_consistency_weight=0.0,
    )
    vis_per = F.binary_cross_entropy_with_logits(
        output.visibility_logits, crop_visibility, reduction="none"
    )
    vis_loss = (vis_per * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    total = loc_loss + float(visibility_weight) * vis_loss
    return total, {
        **parts,
        "vis": float(vis_loss.detach()),
        "selected_visible": float(localization_mask.sum().detach()),
        "usable_pair_fraction": float(usable_pair.float().mean().detach()),
    }


def selected_pair_full_prediction(output: MetricFieldOutput, selection) -> torch.Tensor:
    """Return the ROI head's selected two predictions in full-frame coordinates."""
    batch = torch.arange(output.p.shape[0], device=output.p.device)[:, None]
    pair_crop = output.p[batch, selection.pair_roles]
    return crop_to_full(pair_crop, selection.roi, IMAGE_SIZE)


def merge_selected_visibility(
    coarse_visibility: torch.Tensor,
    roi_visibility: torch.Tensor,
    selection,
    *,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Blend ROI visibility only for the selected pair; alpha=0 is an exact no-op."""
    if coarse_visibility.shape != roi_visibility.shape or coarse_visibility.ndim != 2:
        raise ValueError("coarse/ROI visibility must share shape [B,R]")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("visibility alpha must be in [0,1]")
    mask = selection.role_mask.to(device=coarse_visibility.device)
    blended = coarse_visibility + float(alpha) * (
        roi_visibility - coarse_visibility
    ) * mask.to(coarse_visibility.dtype)
    return torch.where(mask if float(alpha) != 0.0 else torch.zeros_like(mask), blended, coarse_visibility)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone MT-VJ ROI refiner")
    coarse = parser.add_mutually_exclusive_group(required=True)
    coarse.add_argument(
        "--coarse-checkpoint",
        help="completed Stage-V metric checkpoint (legacy/default coarse source)",
    )
    coarse.add_argument(
        "--coarse-policy-checkpoint",
        help="final policy whose embedded mtvj_metric_head supplies coarse weights",
    )
    parser.add_argument(
        "--runtime-metric-checkpoint",
        help="immutable external Stage-V metric file required by the final policy runtime",
    )
    parser.add_argument("--resume")
    parser.add_argument("--save", default="checkpoints/metric_roi_all49.pt")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--center-jitter-min-px", type=float, default=None)
    parser.add_argument("--center-jitter-max-px", type=float, default=None)
    parser.add_argument("--size-jitter", type=float, default=None)
    parser.add_argument("--min-roi-size", type=float, default=None)
    parser.add_argument("--max-roi-size", type=float, default=None)
    parser.add_argument("--distance-scale", type=float, default=None)
    parser.add_argument("--max-delta-px", type=float, default=None)
    parser.add_argument("--visibility-weight", type=float, default=None)
    parser.add_argument("--hinge-margin", type=float, default=None)
    return parser.parse_args(argv)


_DEFAULTS = {
    "steps": 5000,
    "batch_size": 8,
    "grad_accum": 4,
    "lr": 1e-4,
    "seed": 0,
    "center_jitter_min_px": 16.0,
    "center_jitter_max_px": 24.0,
    "size_jitter": 0.10,
    "min_roi_size": 96.0,
    "max_roi_size": 192.0,
    "distance_scale": 2.0,
    "max_delta_px": 32.0,
    "visibility_weight": 1.0,
    "hinge_margin": 0.1,
}
_IMMUTABLE = set(_DEFAULTS) - {"steps"}


def resolve_args(
    args: argparse.Namespace, resume_checkpoint: Mapping[str, Any] | None
) -> argparse.Namespace:
    config = dict(resume_checkpoint.get("config", {})) if resume_checkpoint else {}
    if resume_checkpoint is not None:
        if int(config.get("training_state_version", 0)) != TRAINING_STATE_VERSION:
            raise ValueError("unsupported ROI training_state_version")
        if config.get("task_sampling") != "weighted":
            raise ValueError("ROI exact resume requires weighted task sampling")
        if list(config.get("tasks", [])) != list(SUPPORTED_TASKS):
            raise ValueError("ROI exact resume task order differs from canonical all-49 order")
        if int(config.get("canonical_image_size", 0)) != IMAGE_SIZE:
            raise ValueError("ROI exact resume requires canonical_image_size=384")
    for key, default in _DEFAULTS.items():
        requested = getattr(args, key)
        saved = config.get(key)
        if resume_checkpoint is not None and key in _IMMUTABLE:
            if saved is None:
                raise ValueError(f"ROI resume checkpoint lacks semantic field {key!r}")
            if requested is not None and requested != saved:
                raise ValueError(f"resume mismatch for {key}: CLI={requested}, checkpoint={saved}")
            value = saved
        elif key == "steps" and resume_checkpoint is not None:
            value = saved if requested is None else requested
        else:
            value = default if requested is None else requested
        setattr(args, key, value)
    numeric = (
        args.lr,
        args.center_jitter_min_px,
        args.center_jitter_max_px,
        args.size_jitter,
        args.min_roi_size,
        args.max_roi_size,
        args.distance_scale,
        args.max_delta_px,
        args.visibility_weight,
        args.hinge_margin,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("ROI numeric arguments must be finite")
    if args.steps < 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.lr <= 0:
        raise ValueError("steps must be >=0 and batch-size/grad-accum/lr must be positive")
    if args.steps % args.grad_accum:
        raise ValueError("--steps must end on a complete gradient-accumulation boundary")
    if not 0 <= args.center_jitter_min_px <= args.center_jitter_max_px:
        raise ValueError("need 0 <= center-jitter-min-px <= center-jitter-max-px")
    if not 0 <= args.size_jitter < 1:
        raise ValueError("--size-jitter must be in [0,1)")
    if not 0 < args.min_roi_size <= args.max_roi_size <= IMAGE_SIZE:
        raise ValueError("need 0 < min-roi-size <= max-roi-size <= 384")
    if args.distance_scale <= 0:
        raise ValueError("--distance-scale must be positive")
    if args.max_delta_px < 0 or args.visibility_weight < 0 or args.hinge_margin < 0:
        raise ValueError("max-delta-px, visibility-weight and hinge-margin must be non-negative")
    if args.log_every <= 0 or args.save_every <= 0:
        raise ValueError("log-every and save-every must be positive")
    return args


def _rng_state(
    optimizer: torch.optim.Optimizer, rng: np.random.Generator, completed_steps: int
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "optimizer_steps_done": completed_steps,
    }


def build_checkpoint_config(
    args: argparse.Namespace,
    tasks: list[str],
    completed_steps: int,
    *,
    head_ctor: Mapping[str, Any],
    coarse_sha256: str,
    coarse_head_state_sha256: str,
    coarse_source: str = COARSE_SOURCE_STAGE_V,
) -> dict[str, Any]:
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "steps": int(args.steps),
        "steps_done": int(completed_steps),
        "optimizer_steps_done": int(completed_steps // args.grad_accum),
        "tasks": list(tasks),
        "task_sampling": "weighted",
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "image_size": IMAGE_SIZE,
        "canonical_image_size": IMAGE_SIZE,
        "raw_frame_source": "make_metric_batch.raw_frames",
        "raw_frame_size": 480,
        "head_ctor": dict(head_ctor),
        "role_names": ["tool", "object", "target", "interface"],
        "role_pairs": [list(pair) for pair in ROLE_PAIRS],
        "alpha_default": 0.0,
        "eval_alpha": 1.0,
        "min_roi_size": float(args.min_roi_size),
        "max_roi_size": float(args.max_roi_size),
        "distance_scale": float(args.distance_scale),
        "center_jitter_min_px": float(args.center_jitter_min_px),
        "center_jitter_max_px": float(args.center_jitter_max_px),
        "size_jitter": float(args.size_jitter),
        "max_delta_px": float(args.max_delta_px),
        "visibility_weight": float(args.visibility_weight),
        "hinge_margin": float(args.hinge_margin),
        "loss": "selected_pair_crop_visible_(hinge+position+offset)+selected_pair_visibility_BCE",
        "coarse_sha256": coarse_sha256,
        "coarse_head_state_sha256": coarse_head_state_sha256,
        "coarse_source": coarse_source,
    }


def build_checkpoint_payload(
    args: argparse.Namespace,
    tasks: list[str],
    completed_steps: int,
    *,
    head_ctor: Mapping[str, Any],
    coarse_sha256: str,
    coarse_head_state_sha256: str,
    coarse_checkpoint: Mapping[str, Any],
    roi_head: LanguageMetricField,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    coarse_source: str = COARSE_SOURCE_STAGE_V,
    runtime_metric_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = build_checkpoint_config(
        args,
        tasks,
        completed_steps,
        head_ctor=head_ctor,
        coarse_sha256=coarse_sha256,
        coarse_head_state_sha256=coarse_head_state_sha256,
        coarse_source=coarse_source,
    )
    runtime_identity = dict(runtime_metric_identity or {})
    runtime_identity.setdefault("sha256", coarse_sha256)
    runtime_identity.setdefault("contract", coarse_checkpoint["contract"])
    payload = {
        "contract": CONTRACT,
        "config": config,
        "coarse": {
            "sha256": coarse_sha256,
            "coarse_head_state_sha256": coarse_head_state_sha256,
            "contract": coarse_checkpoint["contract"],
            "config": dict(coarse_checkpoint["config"]),
            "source": coarse_source,
            "runtime_metric_checkpoint": runtime_identity,
        },
        "roi_metric_head": roi_head.state_dict(),
    }
    payload.update(_rng_state(optimizer, rng, completed_steps))
    payload["optimizer_steps_done"] = completed_steps // args.grad_accum
    return payload


def restore_training_state(
    checkpoint: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> None:
    required = {
        "optimizer",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "optimizer_steps_done",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"ROI checkpoint cannot resume exactly; missing={missing}")
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if checkpoint["cuda_rng_state_all"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def _atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    if resume is not None and resume.get("contract") != CONTRACT:
        raise ValueError(f"resume contract={resume.get('contract')!r}; expected {CONTRACT!r}")
    args = resolve_args(args, resume)
    device = torch.device(args.device)
    coarse_path, coarse_checkpoint, coarse_source_record = load_coarse_source(
        coarse_checkpoint_path=args.coarse_checkpoint,
        coarse_policy_checkpoint_path=args.coarse_policy_checkpoint,
        runtime_metric_checkpoint_path=args.runtime_metric_checkpoint,
    )
    coarse_sha = coarse_source_record["runtime_metric_checkpoint"]["sha256"]
    coarse_source = coarse_source_record["kind"]
    coarse_config = dict(validate_coarse_checkpoint(coarse_checkpoint))
    if resume is not None:
        if resume.get("coarse", {}).get("sha256") != coarse_sha:
            raise ValueError("coarse checkpoint SHA-256 changed; exact ROI resume refused")
        if resume.get("coarse", {}).get("config") != coarse_config:
            raise ValueError("coarse checkpoint config changed; exact ROI resume refused")
        if resume.get("coarse", {}).get(
            "source", COARSE_SOURCE_STAGE_V
        ) != coarse_source:
            raise ValueError("coarse source changed; exact ROI resume refused")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    coarse_head, roi_head, head_ctor = build_frozen_coarse_and_roi(
        coarse_checkpoint, device
    )
    coarse_head_sha = metric_head_state_sha256(coarse_head)
    if coarse_head_sha != coarse_source_record["actual_coarse_head_state_sha256"]:
        raise RuntimeError("loaded coarse metric-head state SHA changed unexpectedly")
    if resume is not None and (
        resume.get("coarse", {}).get("coarse_head_state_sha256")
        != coarse_head_sha
    ):
        raise ValueError("coarse metric-head state changed; exact ROI resume refused")
    if resume is not None:
        roi_head.load_state_dict(resume["roi_metric_head"], strict=True)
    optimizer = torch.optim.Adam(roi_head.parameters(), lr=args.lr)

    vision_dtype = "float16" if device.type == "cuda" else "float32"
    vision = VJEPA21Backbone.from_pretrained(
        device=device, dtype=vision_dtype, local_files_only=True
    )
    vision.freeze_all()
    vision.eval()
    text = QwenTextBackbone.from_pretrained(
        device=device, dtype=vision_dtype, local_files_only=True
    )
    text.requires_grad_(False).eval()
    language_cache, language_ok = build_language_cache(
        text, [ENV_TO_TASK[task] for task in SUPPORTED_TASKS]
    )
    if not language_ok:
        raise RuntimeError("Qwen language cache degraded")
    del text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    coords = torch.from_numpy(_dense_coords()).to(device)

    start_step = int(resume.get("config", {}).get("steps_done", 0)) if resume else 0
    if start_step > args.steps:
        raise ValueError(f"resume step {start_step} exceeds target {args.steps}")
    if start_step % args.grad_accum:
        raise ValueError("resume step is inside a gradient-accumulation window")
    if resume is not None:
        if int(resume.get("optimizer_steps_done", -1)) != start_step // args.grad_accum:
            raise ValueError("optimizer_steps_done disagrees with steps_done/grad_accum")
        restore_training_state(resume, optimizer, rng)
    optimizer.zero_grad(set_to_none=True)

    tasks = list(SUPPORTED_TASKS)

    def checkpoint_payload(completed_steps: int) -> dict[str, Any]:
        return build_checkpoint_payload(
            args,
            tasks,
            completed_steps,
            head_ctor=head_ctor,
            coarse_sha256=coarse_sha,
            coarse_head_state_sha256=coarse_head_sha,
            coarse_checkpoint=coarse_checkpoint,
            roi_head=roi_head,
            optimizer=optimizer,
            rng=rng,
            coarse_source=coarse_source,
            runtime_metric_identity=coarse_source_record[
                "runtime_metric_checkpoint"
            ],
        )

    print(
        f"ROI train: {start_step}->{args.steps}, batch={args.batch_size}, "
        f"grad_accum={args.grad_accum}, lr={args.lr:g}, "
        f"coarse_source={coarse_source}, runtime_sha={coarse_sha[:12]}, "
        f"head_sha={coarse_head_sha[:12]}",
        flush=True,
    )
    last_saved = start_step
    running_sq = 0.0
    running_n = 0
    for step in range(start_step, args.steps):
        task = task_for_step(tasks, step, args.seed, "weighted")
        batch = make_metric_batch(
            task, rng, args.batch_size, include_raw_frames=True
        )
        frames = np.asarray(batch["frames"])
        language_hidden, language_mask = gather_language(
            language_cache, list(batch["language_text"]), device
        )
        with torch.no_grad():
            full_video = preprocess_raw_full_frames(
                np.asarray(batch["raw_frames"]), device
            )
            h5, h11 = vision.encode_multi(full_video, out_layers=(5, 11))
            coarse_output = coarse_head(h5, h11, language_hidden, language_mask, coords)
            jitter = float(
                rng.uniform(args.center_jitter_min_px, args.center_jitter_max_px)
            )
            selection = plan_metric_roi(
                coarse_output.p.clamp(0.0, 1.0),
                coarse_output.visibility,
                IMAGE_SIZE,
                min_size=args.min_roi_size,
                max_size=args.max_roi_size,
                distance_scale=args.distance_scale,
                training=True,
                center_jitter_px=jitter,
                size_jitter=args.size_jitter,
            )
            crop_video = preprocess_raw_roi_frames(
                np.asarray(batch["raw_frames"]), selection.roi, device
            )
            crop_h5, crop_h11 = vision.encode_multi(crop_video, out_layers=(5, 11))
        output = roi_head(
            crop_h5, crop_h11, language_hidden, language_mask, coords
        )
        keypoints = torch.from_numpy(np.asarray(batch["keypoints"])).to(device)
        visibility = torch.from_numpy(np.asarray(batch["visibility"])).to(device)
        loss, parts = compute_roi_losses(
            output,
            keypoints,
            visibility,
            selection,
            visibility_weight=args.visibility_weight,
            hinge_margin=args.hinge_margin,
        )
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"non-finite ROI loss at step {step + 1}")
        (loss / args.grad_accum).backward()
        boundary = (step + 1) % args.grad_accum == 0
        if boundary:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pair_full = selected_pair_full_prediction(output, selection)
            batch_index = torch.arange(args.batch_size, device=device)[:, None]
            true_pair = keypoints[batch_index, selection.pair_roles]
            pair_vis = (
                gt_crop_visibility(keypoints, visibility, selection.roi, IMAGE_SIZE)
                * selection.role_mask
            )[batch_index, selection.pair_roles]
            running_sq += float((((pair_full - true_pair) * IMAGE_SIZE) ** 2).sum(-1).mul(pair_vis).sum())
            running_n += int(pair_vis.sum())
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            rmse = math.sqrt(running_sq / max(running_n, 1))
            detail = " ".join(f"{key}={value:.4f}" for key, value in parts.items())
            print(f"step {step + 1}/{args.steps} loss={float(loss):.4f} {detail} raw_roi_visible_RMSE={rmse:.2f}px", flush=True)
        if boundary and step + 1 - last_saved >= args.save_every:
            _atomic_save(checkpoint_payload(step + 1), args.save)
            last_saved = step + 1
            print(f"  ROI checkpoint @ {step + 1} -> {args.save}", flush=True)

    _atomic_save(checkpoint_payload(args.steps), args.save)
    loaded = torch.load(args.save, map_location="cpu", weights_only=True)
    if loaded.get("contract") != CONTRACT:
        raise RuntimeError("saved ROI checkpoint failed contract verification")
    print(f"ROI checkpoint saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
