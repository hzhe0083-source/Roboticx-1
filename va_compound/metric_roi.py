"""Geometry core for lossless MT-VJ full-frame ROI refinement.

The module deliberately contains no image encoder and is not wired into the
policy runtime.  It only plans one square crop, converts coordinates between
the full frame and that crop, masks simulator labels that fall outside the
crop, and merges a bounded two-role correction.  Coordinates are normalized
``(y, x)`` in ``[0, 1]``; ROI geometry is pixel-space ``(cy, cx, size)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROLE_PAIRS = ((0, 1), (3, 2))  # (tool, object), (interface, target)
ROI_MIN_SIZE = 96.0
ROI_MAX_SIZE = 192.0
METRIC_ROI_CONTRACT = "mt_vj_metric_roi_v1"
METRIC_ROI_CONTRACT_VERSION = 1
DINO_METRIC_ROI_CONTRACT = "dino_metric_roi_task35_v2"
TASK35_METRIC_ROLE_CONTRACT = "slots_tool_pegGrasp_hole_pegHead_v1"
METRIC_ROI_HEAD_CONFIG_KEYS = (
    "lang_dim",
    "h_dim",
    "d_proj",
    "n_roles",
    "l2_norm",
    "learnable_temp",
    "temp_init",
    "freeze_bias",
    "mode_readout",
)


@dataclass(frozen=True)
class MetricROI:
    """One selected interaction crop per batch item."""

    roi: Tensor  # [B, 3] pixel-space (cy, cx, size)
    pair_index: Tensor  # [B], 0=(tool, object), 1=(interface, target)
    pair_roles: Tensor  # [B, 2], role ids in the selected pair
    role_mask: Tensor  # [B, 4] bool, true only for the selected pair
    confidence: Tensor  # [B], product of the selected pair's visibility


def _image_hw(image_size: int | Sequence[int]) -> tuple[float, float]:
    if isinstance(image_size, int):
        height = width = image_size
    elif len(image_size) == 2:
        height, width = image_size
    else:
        raise ValueError(f"image_size must be int or (height, width), got {image_size!r}")
    height, width = float(height), float(width)
    if height <= 0.0 or width <= 0.0:
        raise ValueError(f"image dimensions must be positive, got {height}x{width}")
    return height, width


def _check_positions(p: Tensor, name: str, roles: int | None = None) -> None:
    if p.ndim != 3 or p.shape[-1] != 2 or (roles is not None and p.shape[1] != roles):
        expected = f"[B, {roles}, 2]" if roles is not None else "[B, N, 2]"
        raise ValueError(f"{name} must be {expected}, got {tuple(p.shape)}")
    if not p.is_floating_point() or not torch.isfinite(p).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def plan_metric_roi(
    coarse_p: Tensor,
    coarse_visibility: Tensor,
    image_size: int | Sequence[int],
    *,
    min_size: float = ROI_MIN_SIZE,
    max_size: float = ROI_MAX_SIZE,
    distance_scale: float = 2.0,
    training: bool = False,
    center_jitter_px: float = 0.0,
    size_jitter: float = 0.0,
    generator: torch.Generator | None = None,
    forced_pair_index: int | None = None,
) -> MetricROI:
    """Select a role pair and plan one dynamic square crop.

    By default the pair with the larger visibility product is selected.  A
    forced pair is available for task-specific contracts; task35 deployment
    fixes pair 1 = ``(pegHead, hole)`` so ROI capacity cannot be diverted to the
    already-easy gripper/pegGrasp pair.

    Crop size is ``2 * max(|dy|, |dx|)`` in pixels, clipped to 96--192 by
    default.  Optional training jitter moves the center uniformly in pixels
    and scales the side length uniformly by ``1 +/- size_jitter``.  The final
    crop is kept inside the source image, so the same geometry can later be
    used by any raw-pixel crop implementation without padding ambiguity.
    """

    _check_positions(coarse_p, "coarse_p", roles=4)
    if coarse_visibility.shape != coarse_p.shape[:2]:
        raise ValueError(
            "coarse_visibility must be [B, 4], got "
            f"{tuple(coarse_visibility.shape)}"
        )
    if not coarse_visibility.is_floating_point() or not torch.isfinite(
        coarse_visibility
    ).all():
        raise ValueError("coarse_visibility must be a finite floating-point tensor")
    if (coarse_p < 0.0).any() or (coarse_p > 1.0).any():
        raise ValueError("coarse_p must be normalized to [0, 1]")
    if (coarse_visibility < 0.0).any() or (coarse_visibility > 1.0).any():
        raise ValueError("coarse_visibility must be in [0, 1]")

    height, width = _image_hw(image_size)
    if min_size <= 0.0 or max_size < min_size:
        raise ValueError(f"need 0 < min_size <= max_size, got {min_size}/{max_size}")
    if max_size > min(height, width):
        raise ValueError(
            f"max_size {max_size} must fit inside image {height:g}x{width:g}"
        )
    if distance_scale < 0.0:
        raise ValueError("distance_scale must be non-negative")
    if center_jitter_px < 0.0 or not 0.0 <= size_jitter < 1.0:
        raise ValueError("need center_jitter_px >= 0 and 0 <= size_jitter < 1")

    device = coarse_p.device
    dtype = coarse_p.dtype
    pairs = torch.tensor(ROLE_PAIRS, device=device, dtype=torch.long)  # [2, 2]
    pair_scores = torch.stack(
        (
            coarse_visibility[:, 0] * coarse_visibility[:, 1],
            coarse_visibility[:, 3] * coarse_visibility[:, 2],
        ),
        dim=-1,
    )
    if forced_pair_index is None:
        pair_index = pair_scores.argmax(dim=-1)  # deterministic: first pair wins ties
    else:
        if forced_pair_index not in (0, 1):
            raise ValueError("forced_pair_index must be 0, 1, or None")
        pair_index = torch.full(
            (coarse_p.shape[0],),
            int(forced_pair_index),
            device=device,
            dtype=torch.long,
        )
    pair_roles = pairs[pair_index]  # [B, 2]
    batch_index = torch.arange(coarse_p.shape[0], device=device)[:, None]
    pair_p = coarse_p[batch_index, pair_roles]  # [B, 2, 2], y/x

    scale_yx = coarse_p.new_tensor((height, width))
    pair_px = pair_p * scale_yx
    center = pair_px.mean(dim=1)
    span = (pair_px[:, 0] - pair_px[:, 1]).abs().amax(dim=-1)
    size = (distance_scale * span).clamp(min=min_size, max=max_size)

    if training and (center_jitter_px > 0.0 or size_jitter > 0.0):
        if center_jitter_px > 0.0:
            noise = torch.rand(
                center.shape, device=device, dtype=dtype, generator=generator
            )
            center = center + (noise * 2.0 - 1.0) * center_jitter_px
        if size_jitter > 0.0:
            noise = torch.rand(
                size.shape, device=device, dtype=dtype, generator=generator
            )
            size = size * (1.0 + (noise * 2.0 - 1.0) * size_jitter)
            size = size.clamp(min=min_size, max=max_size)

    half = size / 2.0
    center_y = center[:, 0].clamp(min=half, max=height - half)
    center_x = center[:, 1].clamp(min=half, max=width - half)
    roi = torch.stack((center_y, center_x, size), dim=-1)

    role_mask = torch.zeros(
        (coarse_p.shape[0], 4), device=device, dtype=torch.bool
    )
    role_mask.scatter_(1, pair_roles, True)
    confidence = pair_scores.gather(1, pair_index[:, None]).squeeze(1)
    return MetricROI(roi, pair_index, pair_roles, role_mask, confidence)


def full_to_crop(
    p_full: Tensor, roi: Tensor, image_size: int | Sequence[int]
) -> Tensor:
    """Map normalized full-frame ``(y, x)`` coordinates into a square crop."""

    _check_positions(p_full, "p_full")
    roi = torch.as_tensor(roi, device=p_full.device, dtype=p_full.dtype)
    if roi.shape != (p_full.shape[0], 3) or not torch.isfinite(roi).all():
        raise ValueError(f"roi must be finite [B, 3], got {tuple(roi.shape)}")
    height, width = _image_hw(image_size)
    shape = (p_full.shape[0],) + (1,) * (p_full.ndim - 2)
    cy, cx, size = (roi[:, i].reshape(shape) for i in range(3))
    if (size <= 0.0).any():
        raise ValueError("roi size must be positive")
    out = p_full.clone()
    out[..., 0] = (p_full[..., 0] * height - (cy - size / 2.0)) / size
    out[..., 1] = (p_full[..., 1] * width - (cx - size / 2.0)) / size
    return out


def crop_to_full(
    p_crop: Tensor, roi: Tensor, image_size: int | Sequence[int]
) -> Tensor:
    """Inverse of :func:`full_to_crop` for normalized crop coordinates."""

    _check_positions(p_crop, "p_crop")
    roi = torch.as_tensor(roi, device=p_crop.device, dtype=p_crop.dtype)
    if roi.shape != (p_crop.shape[0], 3) or not torch.isfinite(roi).all():
        raise ValueError(f"roi must be finite [B, 3], got {tuple(roi.shape)}")
    height, width = _image_hw(image_size)
    shape = (p_crop.shape[0],) + (1,) * (p_crop.ndim - 2)
    cy, cx, size = (roi[:, i].reshape(shape) for i in range(3))
    if (size <= 0.0).any():
        raise ValueError("roi size must be positive")
    out = p_crop.clone()
    out[..., 0] = (cy - size / 2.0 + p_crop[..., 0] * size) / height
    out[..., 1] = (cx - size / 2.0 + p_crop[..., 1] * size) / width
    return out


def gt_crop_visibility(
    gt_p: Tensor,
    gt_visibility: Tensor,
    roi: Tensor,
    image_size: int | Sequence[int],
) -> Tensor:
    """Keep simulator visibility only for labels that lie inside the crop."""

    _check_positions(gt_p, "gt_p", roles=4)
    if gt_visibility.shape != gt_p.shape[:2]:
        raise ValueError(f"gt_visibility must be [B, 4], got {tuple(gt_visibility.shape)}")
    local = full_to_crop(gt_p, roi, image_size)
    inside = ((local >= 0.0) & (local <= 1.0)).all(dim=-1)
    return gt_visibility * inside.to(gt_visibility.dtype)


def merge_roi_refinement(
    coarse_p: Tensor,
    coarse_visibility: Tensor,
    refined_pair_p_crop: Tensor,
    selection: MetricROI,
    image_size: int | Sequence[int],
    *,
    alpha: float | Tensor = 0.0,
    max_delta_px: float = 32.0,
) -> tuple[Tensor, Tensor]:
    """Merge a bounded selected-pair correction while preserving visibility.

    ``refined_pair_p_crop`` is ``[B, 2, 2]`` in the selected crop.  Corrections
    are clipped component-wise to ``max_delta_px`` and scattered only to the
    two selected roles.  A zero alpha uses ``where`` to guarantee exact
    element-wise equality with ``coarse_p`` (including signed zero).
    """

    _check_positions(coarse_p, "coarse_p", roles=4)
    _check_positions(refined_pair_p_crop, "refined_pair_p_crop", roles=2)
    if refined_pair_p_crop.shape[0] != coarse_p.shape[0]:
        raise ValueError("coarse/refined batch sizes must match")
    if coarse_visibility.shape != coarse_p.shape[:2]:
        raise ValueError(f"coarse_visibility must be [B, 4], got {tuple(coarse_visibility.shape)}")
    if selection.roi.shape != (coarse_p.shape[0], 3) or selection.pair_roles.shape != (
        coarse_p.shape[0],
        2,
    ):
        raise ValueError("selection batch does not match coarse_p")
    if max_delta_px < 0.0:
        raise ValueError("max_delta_px must be non-negative")

    refined_pair_full = crop_to_full(refined_pair_p_crop, selection.roi, image_size)
    batch_index = torch.arange(coarse_p.shape[0], device=coarse_p.device)[:, None]
    coarse_pair = coarse_p[batch_index, selection.pair_roles]
    height, width = _image_hw(image_size)
    limits = coarse_p.new_tensor((max_delta_px / height, max_delta_px / width))
    pair_delta = (refined_pair_full - coarse_pair).clamp(min=-limits, max=limits)

    delta = torch.zeros_like(coarse_p)
    delta[batch_index, selection.pair_roles] = pair_delta
    mask = selection.role_mask.to(device=coarse_p.device).unsqueeze(-1)
    alpha_t = torch.as_tensor(alpha, device=coarse_p.device, dtype=coarse_p.dtype)
    if alpha_t.ndim == 1:
        if alpha_t.shape[0] != coarse_p.shape[1]:
            raise ValueError("1-D alpha must have one value per role")
        alpha_t = alpha_t.view(1, -1, 1)
    updated = (coarse_p + alpha_t * mask.to(coarse_p.dtype) * delta).clamp(0.0, 1.0)
    zero_gate = (alpha_t * mask.to(coarse_p.dtype)) == 0.0
    p_final = torch.where(zero_gate, coarse_p, updated)
    return p_final, coarse_visibility


def metric_roi_checkpoint_identity(path: Path, checkpoint: dict) -> dict:
    """Content identity for the immutable external ROI checkpoint."""

    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(resolved.stat().st_size),
        "contract": checkpoint.get("contract"),
    }


def metric_head_state_sha256(module_or_state: nn.Module | Mapping[str, Tensor]) -> str:
    """Deterministically hash an exact metric-head state_dict.

    Every sorted key is framed together with its dtype, shape and contiguous
    CPU bytes.  This identifies the weights actually used by the coarse head,
    independently of the external checkpoint file that originally supplied it.
    """

    state = (
        module_or_state.state_dict()
        if isinstance(module_or_state, nn.Module)
        else module_or_state
    )
    if not isinstance(state, Mapping):
        raise TypeError("metric head state must be an nn.Module or tensor mapping")
    digest = hashlib.sha256()

    def update_field(value: bytes) -> None:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)

    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise TypeError("metric head state_dict must map string keys to tensors")
        value = tensor.detach().cpu().contiguous()
        update_field(key.encode("utf-8"))
        update_field(str(value.dtype).encode("ascii"))
        digest.update(struct.pack("<Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<q", int(dimension)))
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        update_field(raw)
    return digest.hexdigest()


def _identity_mismatches(saved: dict, current: dict) -> dict:
    return {
        key: (saved.get(key), current.get(key))
        for key in ("sha256", "size_bytes", "contract")
        if saved.get(key) != current.get(key)
    }


def _validate_metric_roi_config(
    config: dict,
    coarse_identity: dict,
    coarse_head_state_sha256: str,
) -> dict:
    if not isinstance(config, dict):
        raise ValueError("MT-VJ ROI checkpoint config must be a dict")
    required = {
        "training_state_version",
        "steps_done",
        "canonical_image_size",
        "head_ctor",
        "role_pairs",
        "alpha_default",
        "eval_alpha",
        "min_roi_size",
        "max_roi_size",
        "distance_scale",
        "max_delta_px",
        "coarse_sha256",
        "coarse_head_state_sha256",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"MT-VJ ROI checkpoint config is incomplete: missing={missing}")
    head_ctor = config.get("head_ctor")
    if not isinstance(head_ctor, dict):
        raise ValueError("MT-VJ ROI checkpoint config.head_ctor is required")
    missing_ctor = sorted(set(METRIC_ROI_HEAD_CONFIG_KEYS) - set(head_ctor))
    if missing_ctor:
        raise ValueError(
            "MT-VJ ROI checkpoint has incomplete head_ctor: "
            f"missing={missing_ctor}"
        )
    if int(head_ctor.get("n_roles", -1)) != 4:
        raise ValueError("MT-VJ ROI head must keep the four-role metric contract")
    if config.get("role_pairs") != [[0, 1], [3, 2]]:
        raise ValueError(
            "MT-VJ ROI role_pairs must be [[0, 1], [3, 2]] "
            "(tool/object, interface/target)"
        )
    image_size = int(config.get("canonical_image_size", 0))
    if image_size != 384:
        raise ValueError(
            "MT-VJ ROI config.canonical_image_size must be exactly 384"
        )
    if int(config.get("training_state_version", 0)) != 1:
        raise ValueError("MT-VJ ROI checkpoint lacks a supported training-state contract")
    if int(config.get("steps_done", 0)) <= 0:
        raise ValueError("MT-VJ ROI checkpoint must contain trained weights (steps_done > 0)")
    min_size = float(config["min_roi_size"])
    max_size = float(config["max_roi_size"])
    if not 0.0 < min_size <= max_size <= image_size:
        raise ValueError("MT-VJ ROI requires 0 < min_roi_size <= max_roi_size <= 384")
    if float(config["distance_scale"]) <= 0.0:
        raise ValueError("MT-VJ ROI distance_scale must be positive")
    if float(config["max_delta_px"]) < 0.0:
        raise ValueError("MT-VJ ROI max_delta_px must be non-negative")
    if float(config["alpha_default"]) != 0.0:
        raise ValueError("MT-VJ ROI alpha_default must be the lossless value 0.0")
    declared_coarse = config.get("coarse_sha256")
    actual_coarse = coarse_identity.get("sha256")
    if not declared_coarse or declared_coarse != actual_coarse:
        raise ValueError(
            "MT-VJ ROI coarse SHA does not match --metric-visual-checkpoint: "
            f"roi={declared_coarse!r}, metric={actual_coarse!r}"
        )
    declared_head = config.get("coarse_head_state_sha256")
    if not declared_head or declared_head != coarse_head_state_sha256:
        raise ValueError(
            "MT-VJ ROI coarse head state SHA does not match the actually loaded "
            f"metric head: roi={declared_head!r}, loaded={coarse_head_state_sha256!r}"
        )
    return dict(config)


def load_metric_roi_checkpoint(
    path: Path,
    device: torch.device,
    *,
    coarse_identity: dict,
    coarse_head_state_sha256: str,
    policy_state: dict[str, Tensor] | None = None,
    policy_config: dict | None = None,
    policy_identity: dict | None = None,
    policy_training_contract: dict | None = None,
) -> nn.Module:
    """Load a frozen ROI head, preferring strictly checkpointed policy weights.

    The external ROI file remains mandatory: it proves both the ROI artifact's
    content identity and the exact coarse metric checkpoint it was trained on.
    A policy that already checkpointed ROI weights must reproduce that external
    artifact exactly; it never silently falls back to newly supplied weights.
    """

    from va_compound.metric_visual_head import LanguageMetricField

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("contract") != METRIC_ROI_CONTRACT:
        raise ValueError(
            f"MT-VJ ROI checkpoint contract={checkpoint.get('contract')!r} != "
            f"{METRIC_ROI_CONTRACT!r}"
        )
    if "roi_metric_head" not in checkpoint:
        raise ValueError("MT-VJ ROI checkpoint is missing 'roi_metric_head'")
    external_config = _validate_metric_roi_config(
        checkpoint.get("config"), coarse_identity, coarse_head_state_sha256
    )
    coarse_record = checkpoint.get("coarse")
    if not isinstance(coarse_record, dict):
        raise ValueError("MT-VJ ROI checkpoint is missing the coarse identity record")
    if coarse_record.get("sha256") != external_config["coarse_sha256"]:
        raise ValueError(
            "MT-VJ ROI checkpoint coarse.sha256 disagrees with config.coarse_sha256"
        )
    if coarse_record.get("coarse_head_state_sha256") != external_config[
        "coarse_head_state_sha256"
    ]:
        raise ValueError(
            "MT-VJ ROI checkpoint coarse/config coarse-head state SHA fields disagree"
        )
    if coarse_record.get("contract") != coarse_identity.get("contract"):
        raise ValueError(
            "MT-VJ ROI coarse contract does not match --metric-visual-checkpoint: "
            f"roi={coarse_record.get('contract')!r}, "
            f"metric={coarse_identity.get('contract')!r}"
        )
    current_identity = metric_roi_checkpoint_identity(path, checkpoint)
    policy_contract = policy_training_contract or {}
    policy_enabled = policy_contract.get("mtvj_roi_enabled") is True
    supplied_policy_fields = any(
        value is not None for value in (policy_state, policy_config, policy_identity)
    )
    if policy_enabled:
        missing = [
            key
            for key, value in (
                ("mtvj_roi_head", policy_state),
                ("mtvj_roi_config", policy_config),
                ("mtvj_roi_checkpoint_identity", policy_identity),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "policy checkpoint declares mtvj_roi_enabled=True but is missing "
                f"{missing}"
            )
    elif supplied_policy_fields:
        raise ValueError(
            "policy checkpoint contains MT-VJ ROI payload without "
            "training_contract.mtvj_roi_enabled=True"
        )

    if policy_enabled:
        identity_mismatch = _identity_mismatches(policy_identity or {}, current_identity)
        if identity_mismatch:
            raise ValueError(
                "external MT-VJ ROI checkpoint differs from the policy artifact: "
                f"{identity_mismatch}"
            )
        saved_config = _validate_metric_roi_config(
            policy_config or {}, coarse_identity, coarse_head_state_sha256
        )
        if saved_config != external_config:
            raise ValueError(
                "policy MT-VJ ROI config differs from the external ROI checkpoint"
            )
        constructor_config = saved_config["head_ctor"]
        state = policy_state
        source_identity = dict(policy_identity or {})
    else:
        constructor_config = external_config["head_ctor"]
        state = checkpoint["roi_metric_head"]
        saved_config = external_config
        source_identity = current_identity

    ctor = {key: constructor_config[key] for key in METRIC_ROI_HEAD_CONFIG_KEYS}
    head = LanguageMetricField(**ctor).to(device)
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"MT-VJ ROI head/config shape mismatch: {exc}") from exc
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head._mtvj_roi_config = dict(saved_config)
    head._mtvj_roi_checkpoint_identity = dict(source_identity)
    head._mtvj_roi_coarse_identity = dict(coarse_identity)
    return head


def prepare_metric_roi_video(
    frames: np.ndarray | Tensor,
    device: torch.device,
    *,
    image_size: int | None,
) -> Tensor:
    """Raw RGB frames to unnormalised ``[B*T, W, 3, H, W]`` float video.

    ``image_size=None`` preserves render resolution for a true raw-pixel ROI.
    A numeric size is used only by the full-frame compatibility path.
    """

    if isinstance(frames, np.ndarray):
        value = torch.from_numpy(np.ascontiguousarray(frames))
    elif isinstance(frames, Tensor):
        value = frames.detach()
    else:
        raise TypeError("frames must be a numpy array or torch Tensor")
    if value.ndim != 6 or value.shape[-1] != 3:
        raise ValueError(
            "frames must be [B, T, W, H, W, 3], got "
            f"{tuple(value.shape)}"
        )
    b, t, window, height, width, _ = value.shape
    value = value.reshape(b * t * window, height, width, 3)
    value = value.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    source_is_uint8 = (
        frames.dtype == np.uint8
        if isinstance(frames, np.ndarray)
        else frames.dtype == torch.uint8
    )
    if source_is_uint8:
        value = value.div_(255.0)
    elif value.max().item() > 1.0:
        value = value.div_(255.0)
    output_height, output_width = height, width
    if image_size is not None and (height, width) != (image_size, image_size):
        value = F.interpolate(
            value,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        output_height = output_width = image_size
    return value.reshape(b * t, window, 3, output_height, output_width)


def crop_metric_roi_video(
    raw_video: Tensor,
    roi: Tensor,
    *,
    canonical_image_size: int,
    roi_geometry_size: int | None = None,
) -> Tensor:
    """Crop raw render pixels and resample into the encoder resolution.

    ``roi`` is expressed in ``roi_geometry_size`` pixel coordinates. Legacy
    V-JEPA artifacts omit it, so geometry and encoder output both use the
    canonical 384px space. Task35 DINO v2 sets geometry=480 and output=224:
    a 96px ROI then means exactly 96 source-render pixels, not 96*480/224.
    """

    if raw_video.ndim != 5 or raw_video.shape[2] != 3:
        raise ValueError(
            "raw_video must be [N, W, 3, S, S], got "
            f"{tuple(raw_video.shape)}"
        )
    samples, window, _, height, width = raw_video.shape
    if height != width:
        raise ValueError("MT-VJ ROI runtime requires square raw frames")
    if canonical_image_size <= 0:
        raise ValueError("canonical_image_size must be positive")
    geometry_size = int(
        canonical_image_size if roi_geometry_size is None else roi_geometry_size
    )
    if geometry_size <= 0:
        raise ValueError("roi_geometry_size must be positive")
    roi = torch.as_tensor(roi, device=raw_video.device, dtype=raw_video.dtype)
    if roi.shape != (samples, 3) or not torch.isfinite(roi).all():
        raise ValueError(f"roi must be finite [{samples}, 3]")
    raw_scale = height / float(geometry_size)
    roi_raw = roi * raw_scale
    size = roi_raw[:, 2]
    if (size <= 0.0).any() or (size > height).any():
        raise ValueError(f"scaled raw roi size must be in (0, {height}]")

    output_size = canonical_image_size
    raw_output_size = height
    indices = torch.arange(
        raw_output_size, device=raw_video.device, dtype=raw_video.dtype
    )
    half = raw_output_size / 2.0
    scale = raw_output_size / size
    grid_y = (indices[:, None] - half) / scale[None, :] + roi_raw[:, 0][None, :]
    grid_x = (indices[:, None] - half) / scale[None, :] + roi_raw[:, 1][None, :]
    grid_y = (grid_y + 0.5) * (2.0 / height) - 1.0
    grid_x = (grid_x + 0.5) * (2.0 / width) - 1.0
    grid = torch.stack(
        (
            grid_x.T[:, None, :].expand(samples, raw_output_size, raw_output_size),
            grid_y.T[:, :, None].expand(samples, raw_output_size, raw_output_size),
        ),
        dim=-1,
    )
    grid = grid[:, None].expand(
        samples, window, raw_output_size, raw_output_size, 2
    ).reshape(
        samples * window, raw_output_size, raw_output_size, 2
    )
    # First perform the actual high-resolution crop in raw render space.  Only
    # afterwards resize that crop to V-JEPA's canonical input; this must not be
    # replaced by full-frame downsampling followed by a canonical crop.
    raw_crop = F.grid_sample(
        raw_video.reshape(samples * window, 3, height, width),
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    cropped = F.interpolate(
        raw_crop,
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return cropped.reshape(samples, window, 3, output_size, output_size)


def refine_metric_roi_positions(
    coarse_p: Tensor,
    coarse_visibility: Tensor,
    raw_video: Tensor,
    backbone: nn.Module,
    roi_head: nn.Module,
    language_hidden: Tensor,
    language_mask: Tensor,
    coords: Tensor,
    *,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Run frozen V-JEPA + ROI head and merge a bounded selected-pair residual."""

    if alpha == 0.0:
        return coarse_p, coarse_visibility
    config = getattr(roi_head, "_mtvj_roi_config", None)
    if not isinstance(config, dict):
        raise ValueError("ROI head lacks its validated runtime config")
    image_size = int(config["canonical_image_size"])
    if raw_video.shape[0] != coarse_p.shape[0]:
        raise ValueError("raw ROI video and coarse metric batch sizes differ")
    selection = plan_metric_roi(
        coarse_p.detach().clamp(0.0, 1.0),
        coarse_visibility.detach().clamp(0.0, 1.0),
        image_size,
        min_size=float(config["min_roi_size"]),
        max_size=float(config["max_roi_size"]),
        distance_scale=float(config["distance_scale"]),
    )
    cropped = crop_metric_roi_video(
        raw_video,
        selection.roi,
        canonical_image_size=image_size,
    )
    mean = cropped.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = cropped.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    inputs = (cropped - mean) / std
    with torch.no_grad():
        hierarchical = backbone.forward_hierarchical_dense(
            inputs, out_layers=(5, 11)
        )
        head_dtype = next(roi_head.parameters()).dtype
        roi_out = roi_head(
            hierarchical[5].to(dtype=head_dtype),
            hierarchical[11].to(dtype=head_dtype),
            language_hidden.to(device=coarse_p.device, dtype=head_dtype),
            language_mask.to(device=coarse_p.device),
            coords.to(device=coarse_p.device, dtype=head_dtype),
        )
        batch_index = torch.arange(coarse_p.shape[0], device=coarse_p.device)[:, None]
        refined_pair = roi_out.p[batch_index, selection.pair_roles]
    return merge_roi_refinement(
        coarse_p,
        coarse_visibility,
        refined_pair,
        selection,
        image_size,
        alpha=alpha,
        max_delta_px=float(config["max_delta_px"]),
    )


def load_dino_metric_roi_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> nn.Module:
    """Load the task35 DINO ROI artifact with strict role/geometry identity.

    Unlike the legacy v1 artifact, v2 records the corrected role semantics and
    is therefore safe to attach to a policy whose metric slots are
    ``[tool, pegGrasp, hole, pegHead]``.  A legacy checkpoint is rejected rather
    than silently reinterpreted under different labels.
    """
    from va_compound.metric_visual_head import LanguageMetricField

    resolved = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    if checkpoint.get("contract") != DINO_METRIC_ROI_CONTRACT:
        raise ValueError(
            f"DINO ROI contract={checkpoint.get('contract')!r} != "
            f"{DINO_METRIC_ROI_CONTRACT!r}"
        )
    if checkpoint.get("metric_role_contract") != TASK35_METRIC_ROLE_CONTRACT:
        raise ValueError(
            "DINO ROI metric role contract mismatch: "
            f"{checkpoint.get('metric_role_contract')!r} != "
            f"{TASK35_METRIC_ROLE_CONTRACT!r}"
        )
    if checkpoint.get("role_order") != ["tool", "pegGrasp", "hole", "pegHead"]:
        raise ValueError("DINO ROI role_order must be task35 aligned")
    if checkpoint.get("role_pairs") != [[0, 1], [3, 2]]:
        raise ValueError("DINO ROI role_pairs must encode pegHead-hole as pair 1")
    if checkpoint.get("raw_frame_contract") != "true_simulator_render_480px_v1":
        raise ValueError("DINO ROI must be trained from true 480px simulator renders")
    if int(checkpoint.get("roi_geometry_size", -1)) != 480:
        raise ValueError("DINO ROI geometry must be planned in 480px render space")
    state = checkpoint.get("roi_metric_head")
    ctor = checkpoint.get("ctor_config")
    if not isinstance(state, Mapping) or not isinstance(ctor, dict):
        raise ValueError("DINO ROI checkpoint lacks roi_metric_head/ctor_config")
    if int(ctor.get("grid", -1)) != 16 or int(ctor.get("h_dim", -1)) != 1024:
        raise ValueError("DINO ROI head must use grid=16 and h_dim=1024")
    head = LanguageMetricField(
        lang_dim=int(ctor["lang_dim"]),
        h_dim=int(ctor["h_dim"]),
        d_proj=int(ctor["d_proj"]),
        n_roles=int(ctor["n_roles"]),
        l2_norm=bool(ctor["l2_norm"]),
        learnable_temp=bool(ctor["learnable_temp"]),
        temp_init=float(ctor.get("temp_init", 10.0)),
        freeze_bias=bool(ctor.get("freeze_bias", False)),
        mode_readout=bool(ctor["mode_readout"]),
        grid=int(ctor["grid"]),
    ).to(device)
    head.load_state_dict(state, strict=True)
    head._mtvj_roi_config = {
        "canonical_image_size": int(checkpoint.get("canonical_image_size", 224)),
        "roi_geometry_size": int(checkpoint["roi_geometry_size"]),
        "min_roi_size": float(checkpoint.get("min_roi_size", ROI_MIN_SIZE)),
        "max_roi_size": float(checkpoint.get("max_roi_size", ROI_MAX_SIZE)),
        "distance_scale": float(checkpoint.get("distance_scale", 2.0)),
        "max_delta_px": float(checkpoint.get("max_delta_px", 32.0)),
    }
    head._dino_roi_identity = metric_roi_checkpoint_identity(resolved, checkpoint)
    head._dino_metric_role_contract = checkpoint["metric_role_contract"]
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def refine_metric_roi_positions_dino(
    coarse_p: Tensor,
    coarse_visibility: Tensor,
    raw_video: Tensor,
    backbone: nn.Module,
    roi_head: nn.Module,
    language_hidden: Tensor,
    language_mask: Tensor,
    coords: Tensor,
    *,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """DINO 版 ROI 精修（2026-08-16）：与 refine_metric_roi_positions 同协议，
    编码器为冻结 DINO（TimmActionVisionBackbone，224px，输入需预归一化——
    调用方负责 (x-mean)/std 与 .half()；这里只做裁剪与合并）。

    coarse_p/coarse_visibility：粗 metric 头输出（[B,4,2] 0-1 / [B,4]）；
    raw_video：[B, 2, 3, 480, 480] 0-1 双时间片原图；roi_head 消费裁剪后的
    block11/block23 证据 [B,512,1024] 与 dense_coords(512)。
    """
    if alpha == 0.0:
        return coarse_p, coarse_visibility
    config = getattr(roi_head, "_mtvj_roi_config", None)
    image_size = (
        int(config["canonical_image_size"]) if isinstance(config, dict) else 224
    )
    geometry_size = (
        int(config.get("roi_geometry_size", image_size))
        if isinstance(config, dict)
        else image_size
    )
    min_size = float(config["min_roi_size"]) if isinstance(config, dict) else ROI_MIN_SIZE
    max_size = float(config["max_roi_size"]) if isinstance(config, dict) else ROI_MAX_SIZE
    distance_scale = (
        float(config["distance_scale"]) if isinstance(config, dict) else 2.0
    )
    max_delta_px = float(config["max_delta_px"]) if isinstance(config, dict) else 32.0
    if raw_video.shape[0] != coarse_p.shape[0]:
        raise ValueError("raw ROI video and coarse metric batch sizes differ")
    selection = plan_metric_roi(
        coarse_p.detach().clamp(0.0, 1.0),
        coarse_visibility.detach().clamp(0.0, 1.0),
        geometry_size,
        min_size=min_size,
        max_size=max_size,
        distance_scale=distance_scale,
        forced_pair_index=(
            1
            if getattr(roi_head, "_dino_metric_role_contract", None)
            == TASK35_METRIC_ROLE_CONTRACT
            else None
        ),
    )
    cropped = crop_metric_roi_video(
        raw_video,
        selection.roi,
        canonical_image_size=image_size,
        roi_geometry_size=geometry_size,
    )
    mean = cropped.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = cropped.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    inputs = (cropped - mean) / std
    with torch.no_grad():
        hierarchical = backbone.forward_hierarchical_dense(
            inputs.reshape(-1, 3, image_size, image_size).half()
        )
        head_dtype = next(roi_head.parameters()).dtype
        b = coarse_p.shape[0]
        roi_out = roi_head(
            hierarchical[5].reshape(b, -1, hierarchical[5].shape[-1]).to(dtype=head_dtype),
            hierarchical[11].reshape(b, -1, hierarchical[11].shape[-1]).to(dtype=head_dtype),
            language_hidden.to(device=coarse_p.device, dtype=head_dtype),
            language_mask.to(device=coarse_p.device),
            coords.to(device=coarse_p.device, dtype=head_dtype),
        )
        batch_index = torch.arange(b, device=coarse_p.device)[:, None]
        refined_pair = roi_out.p[batch_index, selection.pair_roles]
    return merge_roi_refinement(
        coarse_p,
        coarse_visibility,
        refined_pair,
        selection,
        geometry_size,
        alpha=alpha,
        max_delta_px=max_delta_px,
    )
