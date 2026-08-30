"""Loss-side contracts for action-conditioned visual world supervision.

This module deliberately has no model imports.  Targets and oracle motion
weights are detached before they are used, so this code cannot provide a
future-visual shortcut to the predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class VisualWorldLoss:
    """Per-sample visual losses and masks on one feature-error scale."""

    loss_per_sample: Tensor
    all_per_sample: Tensor
    motion_per_sample: Tensor
    topk_per_sample: Tensor
    top10_per_sample: Tensor
    static_per_sample: Tensor
    copy_all_per_sample: Tensor
    copy_motion_per_sample: Tensor
    copy_topk_per_sample: Tensor
    copy_top10_per_sample: Tensor
    copy_static_per_sample: Tensor
    motion_energy_per_sample: Tensor
    motion_weights: Tensor
    topk_mask: Tensor
    top10_mask: Tensor
    static_mask: Tensor


@dataclass(frozen=True)
class VisualNoRegressionLoss:
    """Per-sample penalties for the all-map and static copy guards."""

    loss_per_sample: Tensor
    all_hinge_per_sample: Tensor
    static_hinge_per_sample: Tensor


@dataclass(frozen=True)
class GatedStaticCopyAnchorLoss:
    """Per-sample static-map trust region, activated only after copy regression."""

    loss_per_sample: Tensor
    anchor_per_sample: Tensor
    active_per_sample: Tensor


@dataclass(frozen=True)
class VisualStaticRelativePenaltyLoss:
    """Per-sample penalty for exceeding the static copy-error gate."""

    loss_per_sample: Tensor
    hinge_per_sample: Tensor
    normalized_quadratic_per_sample: Tensor


@dataclass(frozen=True)
class VisualStaticStageChainLoss:
    """Per-stage static hinge against copy or the detached previous stage."""

    loss_per_sample: Tensor
    boundary_per_sample: Tensor


@dataclass(frozen=True)
class ActionTop10RankingLoss:
    """Per-sample relative action-ranking penalties on top-motion patches."""

    loss_per_sample: Tensor
    shuffle_hinge_per_sample: Tensor
    zero_hinge_per_sample: Tensor
    either_strong_hinge_per_sample: Tensor


@dataclass(frozen=True)
class ActionTop10GapLoss:
    """Top-motion gap penalty between real and shuffled actions."""

    loss_per_sample: Tensor
    error_gap_per_sample: Tensor


def transition_mask(
    action_valid_mask: Tensor,
    *,
    cycle_steps: int = 6,
    time_index: int | None = None,
) -> Tensor:
    """Return the only transitions eligible for World supervision.

    A transition is valid iff all six executable actions at the current
    decision are valid and the first action at the next decision is valid.
    The mask is intentionally strict: callers must provide the recorded mask,
    rather than silently treating missing data as valid.
    """

    if action_valid_mask.ndim != 3 or action_valid_mask.dtype != torch.bool:
        raise ValueError(
            "action_valid_mask must be bool [B,T,H], got "
            f"{tuple(action_valid_mask.shape)}/{action_valid_mask.dtype}"
        )
    if cycle_steps < 1 or cycle_steps > action_valid_mask.shape[-1]:
        raise ValueError(
            f"cycle_steps={cycle_steps} is outside horizon "
            f"{action_valid_mask.shape[-1]}"
        )
    if action_valid_mask.shape[1] < 2:
        full = action_valid_mask.new_empty(
            action_valid_mask.shape[0], 0, dtype=torch.bool
        )
    else:
        full = action_valid_mask[:, :-1, :cycle_steps].all(dim=-1)
        full = full & action_valid_mask[:, 1:, 0]
    if time_index is None:
        return full
    if time_index < 0 or time_index >= full.shape[1]:
        raise IndexError(f"time_index={time_index} outside {full.shape[1]} transitions")
    return full[:, time_index]


def _masked_mean(values: Tensor, mask: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Mean over a mask with a graph-connected zero for an empty mask."""

    if values.shape != mask.shape:
        raise ValueError(
            f"values/mask shape mismatch: {tuple(values.shape)} vs {tuple(mask.shape)}"
        )
    weights = mask.to(dtype=values.dtype)
    numerator = (values * weights).sum(dim=-1)
    denominator = weights.sum(dim=-1)
    # numerator is connected to values even when the mask is empty.
    return numerator / denominator.clamp_min(eps)


def masked_numerator_denominator(
    values: list[Tensor],
    masks: list[Tensor],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Accumulate a common masked numerator/denominator across stages."""

    if len(values) != len(masks) or not values:
        raise ValueError("values and masks must be non-empty and have equal length")
    if weights is None:
        weights = [1.0] * len(values)
    if len(weights) != len(values):
        raise ValueError("stage weight count must match value count")
    reference = values[0]
    numerator = reference.sum() * 0.0
    denominator = reference.new_zeros(())
    for value, mask, weight in zip(values, masks, weights, strict=True):
        if value.ndim != 1 or mask.ndim != 1 or value.shape != mask.shape:
            raise ValueError("stage values and masks must be matching [B] tensors")
        if not math.isfinite(float(weight)) or weight < 0.0:
            raise ValueError("stage weights must be finite and non-negative")
        wmask = mask.to(dtype=value.dtype)
        numerator = numerator + float(weight) * (value * wmask).sum()
        denominator = denominator + float(weight) * wmask.sum()
    return numerator, denominator


def masked_reduction(
    values: list[Tensor],
    masks: list[Tensor],
    weights: list[float] | tuple[float, ...] | None = None,
) -> Tensor:
    """Common masked reduction; empty effective batches return connected zero."""

    numerator, denominator = masked_numerator_denominator(values, masks, weights)
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def canonical_stage_weight_overrides(
    overrides: dict[int, float] | None,
) -> dict[int, float]:
    """Normalize override keys to ints so contracts compare cleanly."""
    if not overrides:
        return {}
    return {int(index): float(value) for index, value in overrides.items()}


def apply_stage_weight_overrides(
    weights: tuple[float, ...],
    overrides: dict[int, float] | None,
) -> tuple[float, ...]:
    """Replace selected stage weights after the decay/floor schedule."""
    resolved_overrides = canonical_stage_weight_overrides(overrides)
    if not resolved_overrides:
        return weights
    resolved = list(weights)
    for index, value in resolved_overrides.items():
        if index < 0 or index >= len(resolved):
            raise ValueError(
                f"stage weight override index {index} is outside "
                f"0..{max(len(resolved) - 1, 0)}"
            )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("stage weight overrides must be finite and non-negative")
        resolved[index] = value
    return tuple(resolved)


def stage_supervision_weights(
    num_stages: int,
    *,
    auxiliary_decay: float = 0.25,
    floor: float = 0.1,
    overrides: dict[int, float] | None = None,
) -> tuple[float, ...]:
    """Final refinement has weight 1; earlier auxiliary stages decay to ``floor``.

    Unfloored ``0.25**(n-1-i)`` spans 16384x across 8 stages, so stage 0-3
    were effectively unsupervised.  Measured log-weight vs log-error Pearson
    r = -0.868: unweighted stages diverged to 3000x copy error and those
    maps were still published into VA layers 0-4.

    ``overrides`` replaces individual slots after that schedule. The hard2
    S5/S6 experiment uses ``{5: 0.5, 6: 1.0}`` so the eight-stage vector
    becomes ``(0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 1.0, 1.0)``.
    """

    if num_stages < 1:
        return ()
    if not 0.0 < auxiliary_decay < 1.0:
        raise ValueError("auxiliary_decay must be in (0,1)")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0,1]")
    return apply_stage_weight_overrides(
        tuple(
            max(floor, auxiliary_decay ** (num_stages - 1 - index))
            for index in range(num_stages)
        ),
        overrides,
    )


def late_stage_anchor_loss(
    stage_objectives: list[Tensor] | tuple[Tensor, ...],
    *,
    weight: float,
    stage_weights: dict[int, float] | None = None,
) -> Tensor:
    """Add an extra S5/S6 term that is not folded into the stage-mean denominator.

    ``stage_objectives[i]`` must already be a reduced scalar for stage ``i``.
    A zero weight returns a zero tensor on the same device/dtype as stage 0.
    """

    if not stage_objectives:
        raise ValueError("late_stage_anchor_loss requires at least one stage")
    reference = stage_objectives[0]
    if weight == 0.0:
        return reference * 0.0
    if not 0.0 < float(weight):
        raise ValueError("late-stage anchor weight must be non-negative")
    if stage_weights is None:
        from va_compound.world_contract import WORLD_LATE_STAGE_ANCHOR_STAGE_WEIGHTS

        resolved = dict(WORLD_LATE_STAGE_ANCHOR_STAGE_WEIGHTS)
    else:
        resolved = dict(stage_weights)
    missing = [index for index in resolved if int(index) >= len(stage_objectives)]
    if missing:
        raise ValueError(
            "late-stage anchor needs stages "
            f"{sorted(int(index) for index in resolved)}; "
            f"got {len(stage_objectives)} stages"
        )
    total = reference * 0.0
    for index, stage_weight in resolved.items():
        total = total + float(stage_weight) * stage_objectives[int(index)]
    return float(weight) * total


def _top_mask(flat_energy: Tensor, fraction: float) -> Tensor:
    patches = flat_energy.shape[1]
    count = max(1, int(math.ceil(float(fraction) * patches)))
    indices = flat_energy.topk(count, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat_energy, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def _validate_per_sample_values(**values: Tensor) -> None:
    reference: Tensor | None = None
    for name, value in values.items():
        if value.ndim != 1:
            raise ValueError(f"{name} must be a per-sample [B] tensor")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if reference is not None and value.shape != reference.shape:
            raise ValueError(
                f"per-sample shape mismatch: {name}={tuple(value.shape)} "
                f"vs {tuple(reference.shape)}"
            )
        reference = value


def visual_no_regression_loss(
    visual: VisualWorldLoss,
    *,
    all_copy_ratio: float = 1.0,
    static_copy_ratio: float = 1.05,
) -> VisualNoRegressionLoss:
    """Penalize all-map or static MSE that regresses beyond copy-last-frame.

    The two hinges are averaged so the combined value remains on the MSE
    scale.  Copy errors are explicitly detached even though
    :func:`visual_world_loss` already computes them from detached oracles.
    """

    ratios = (all_copy_ratio, static_copy_ratio)
    if any(not math.isfinite(float(value)) or value < 0.0 for value in ratios):
        raise ValueError("copy ratios must be finite and non-negative")
    _validate_per_sample_values(
        all_per_sample=visual.all_per_sample,
        copy_all_per_sample=visual.copy_all_per_sample,
        static_per_sample=visual.static_per_sample,
        copy_static_per_sample=visual.copy_static_per_sample,
    )
    all_hinge = torch.relu(
        visual.all_per_sample
        - float(all_copy_ratio) * visual.copy_all_per_sample.detach()
    )
    static_hinge = torch.relu(
        visual.static_per_sample
        - float(static_copy_ratio) * visual.copy_static_per_sample.detach()
    )
    return VisualNoRegressionLoss(
        loss_per_sample=0.5 * (all_hinge + static_hinge),
        all_hinge_per_sample=all_hinge,
        static_hinge_per_sample=static_hinge,
    )


def gated_static_copy_anchor_loss(
    prediction: Tensor,
    current: Tensor,
    visual: VisualWorldLoss,
    *,
    static_copy_ratio: float = 1.05,
    eps: float = 1e-8,
) -> GatedStaticCopyAnchorLoss:
    """Anchor regressing static patches to the detached current visual map.

    The oracle static mask and copy-regression gate are detached reduction
    inputs.  Consequently, only ``prediction`` receives gradients and patches
    selected by top-k motion receive exactly zero gradient from this loss.
    """

    if prediction.ndim != 4 or current.shape != prediction.shape:
        raise ValueError(
            "prediction/current must be matching [B,C,H,W] tensors, got "
            f"{tuple(prediction.shape)} and {tuple(current.shape)}"
        )
    if not math.isfinite(float(static_copy_ratio)) or static_copy_ratio < 0.0:
        raise ValueError("static_copy_ratio must be finite and non-negative")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    expected_mask_shape = (
        prediction.shape[0],
        prediction.shape[2],
        prediction.shape[3],
    )
    if (
        visual.static_mask.dtype != torch.bool
        or visual.static_mask.shape != expected_mask_shape
    ):
        raise ValueError(
            "visual.static_mask must be bool [B,H,W], got "
            f"{tuple(visual.static_mask.shape)}/{visual.static_mask.dtype}"
        )
    _validate_per_sample_values(
        static_per_sample=visual.static_per_sample,
        copy_static_per_sample=visual.copy_static_per_sample,
    )
    if visual.static_per_sample.shape[0] != prediction.shape[0]:
        raise ValueError("visual static metrics must match prediction batch size")

    current_detached = current.detach().float()
    static_mask = visual.static_mask.detach().flatten(1)
    anchor_error = (
        prediction.float() - current_detached
    ).square().mean(dim=1).flatten(1)
    anchor_per_sample = _masked_mean(anchor_error, static_mask, eps=eps)
    with torch.no_grad():
        active_per_sample = visual.static_per_sample.detach() > (
            float(static_copy_ratio) * visual.copy_static_per_sample.detach()
        )
    loss_per_sample = anchor_per_sample * active_per_sample.to(
        dtype=anchor_per_sample.dtype
    )
    return GatedStaticCopyAnchorLoss(
        loss_per_sample=loss_per_sample,
        anchor_per_sample=anchor_per_sample,
        active_per_sample=active_per_sample,
    )


def static_copy_anchor_loss(
    prediction: Tensor,
    current: Tensor,
    static_mask: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Anchor static patches to copy-last-frame without an activation gate."""

    if prediction.ndim != 4 or current.shape != prediction.shape:
        raise ValueError(
            "prediction/current must be matching [B,C,H,W] tensors, got "
            f"{tuple(prediction.shape)} and {tuple(current.shape)}"
        )
    expected_mask_shape = (
        prediction.shape[0],
        prediction.shape[2],
        prediction.shape[3],
    )
    if static_mask.dtype != torch.bool or static_mask.shape != expected_mask_shape:
        raise ValueError(
            "static_mask must be bool [B,H,W], got "
            f"{tuple(static_mask.shape)}/{static_mask.dtype}"
        )
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    error = (
        prediction.float() - current.detach().float()
    ).square().mean(dim=1).flatten(1)
    return _masked_mean(error, static_mask.detach().flatten(1), eps=eps)


def visual_static_relative_penalty(
    visual: VisualWorldLoss,
    *,
    static_copy_ratio: float = 1.05,
    eps: float = 1e-6,
) -> VisualStaticRelativePenaltyLoss:
    """Penalize static MSE beyond the copy baseline in the gate's units.

    The linear hinge supplies a direct boundary gradient.  The normalized
    quadratic term increases pressure when copy error is small without
    changing the DINO-feature MSE scale.  The copy baseline is detached.
    """

    if not math.isfinite(float(static_copy_ratio)) or static_copy_ratio < 0.0:
        raise ValueError("static_copy_ratio must be finite and non-negative")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    _validate_per_sample_values(
        static_per_sample=visual.static_per_sample,
        copy_static_per_sample=visual.copy_static_per_sample,
    )
    copy_static = visual.copy_static_per_sample.detach()
    hinge = torch.relu(
        visual.static_per_sample - float(static_copy_ratio) * copy_static
    )
    normalized_quadratic = 0.5 * hinge.square() / (copy_static + float(eps))
    return VisualStaticRelativePenaltyLoss(
        loss_per_sample=hinge + normalized_quadratic,
        hinge_per_sample=hinge,
        normalized_quadratic_per_sample=normalized_quadratic,
    )


def visual_static_stage_chain_loss(
    visual: VisualWorldLoss,
    previous_static_per_sample: Tensor | None = None,
    *,
    static_copy_ratio: float = 1.05,
) -> VisualStaticStageChainLoss:
    """Stop each refinement stage from accumulating static-region error."""

    if not math.isfinite(float(static_copy_ratio)) or static_copy_ratio < 0.0:
        raise ValueError("static_copy_ratio must be finite and non-negative")
    _validate_per_sample_values(
        static_per_sample=visual.static_per_sample,
        copy_static_per_sample=visual.copy_static_per_sample,
    )
    copy_budget = (
        float(static_copy_ratio) * visual.copy_static_per_sample.detach()
    )
    if previous_static_per_sample is None:
        boundary = copy_budget
    else:
        _validate_per_sample_values(
            previous_static_per_sample=previous_static_per_sample
        )
        if previous_static_per_sample.shape != visual.static_per_sample.shape:
            raise ValueError("previous static loss must match the current batch")
        boundary = torch.minimum(previous_static_per_sample.detach(), copy_budget)
    return VisualStaticStageChainLoss(
        loss_per_sample=torch.relu(visual.static_per_sample - boundary),
        boundary_per_sample=boundary,
    )


def action_top10_pairwise_loss(
    real: VisualWorldLoss,
    counterfactual: VisualWorldLoss,
    *,
    minimum_relative_degradation: float = 0.10,
) -> Tensor:
    """Require one counterfactual action to be relatively worse on top10."""

    margin = float(minimum_relative_degradation)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "minimum_relative_degradation must be finite and non-negative"
        )
    real_top10 = real.top10_per_sample
    counterfactual_top10 = counterfactual.top10_per_sample
    _validate_per_sample_values(
        real_top10_per_sample=real_top10,
        counterfactual_top10_per_sample=counterfactual_top10,
    )
    return torch.relu((1.0 + margin) * real_top10 - counterfactual_top10)


def action_top10_gap_loss(
    real: VisualWorldLoss,
    shuffled: VisualWorldLoss,
    *,
    minimum_relative_degradation: float = 0.05,
) -> ActionTop10GapLoss:
    """Require shuffled actions to be worse using a control-variate gradient.

    The forward value equals ``relu((1 + margin) * E_real - E_shuffle)``.
    Writing it as a detached margin minus the error gap changes the active
    gradient to ``grad(E_real) - grad(E_shuffle)``.  Predictor effects shared by
    both matched-context branches therefore cancel to first order.
    """

    margin = float(minimum_relative_degradation)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "minimum_relative_degradation must be finite and non-negative"
        )
    real_top10 = real.top10_per_sample
    shuffled_top10 = shuffled.top10_per_sample
    _validate_per_sample_values(
        real_top10_per_sample=real_top10,
        shuffled_top10_per_sample=shuffled_top10,
    )
    error_gap = shuffled_top10 - real_top10
    loss = torch.relu(margin * real_top10.detach() - error_gap)
    return ActionTop10GapLoss(
        loss_per_sample=loss,
        error_gap_per_sample=error_gap,
    )


def action_top10_oracle_straight_through_gap_loss(
    real: VisualWorldLoss,
    shuffled: VisualWorldLoss,
    real_prediction: Tensor,
    shuffled_prediction: Tensor,
    target: Tensor,
    current: Tensor,
    *,
    minimum_relative_degradation: float = 0.12,
) -> ActionTop10GapLoss:
    """Use the exact action-error gap with a persistent motion-direction gradient.

    The straight-through value is exactly ``E_shuffle - E_real``. Its active
    gradient instead increases the real-vs-shuffled prediction difference in
    the detached oracle motion direction, so action learning does not vanish
    merely because the real prediction residual becomes small.
    """

    margin = float(minimum_relative_degradation)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "minimum_relative_degradation must be finite and non-negative"
        )
    if (
        real_prediction.ndim != 4
        or shuffled_prediction.shape != real_prediction.shape
        or target.shape != real_prediction.shape
        or current.shape != real_prediction.shape
    ):
        raise ValueError(
            "predictions/target/current must be matching [B,C,H,W] tensors"
        )
    _validate_per_sample_values(
        real_top10_per_sample=real.top10_per_sample,
        shuffled_top10_per_sample=shuffled.top10_per_sample,
    )
    expected_mask_shape = (
        real_prediction.shape[0],
        real_prediction.shape[2],
        real_prediction.shape[3],
    )
    if real.top10_mask.dtype != torch.bool or real.top10_mask.shape != expected_mask_shape:
        raise ValueError("real.top10_mask must be bool [B,H,W]")
    if not torch.equal(real.top10_mask, shuffled.top10_mask):
        raise ValueError("real and shuffled top10 masks must match")

    exact_gap = shuffled.top10_per_sample - real.top10_per_sample
    motion = target.detach().float() - current.detach().float()
    prediction_gap = real_prediction.float() - shuffled_prediction.float()
    oracle_gap_map = 2.0 * (prediction_gap * motion).mean(dim=1).flatten(1)
    oracle_gap = _masked_mean(
        oracle_gap_map,
        real.top10_mask.detach().flatten(1),
    )
    straight_through_gap = oracle_gap + (exact_gap - oracle_gap).detach()
    loss = torch.relu(
        margin * real.top10_per_sample.detach() - straight_through_gap
    )
    return ActionTop10GapLoss(
        loss_per_sample=loss,
        error_gap_per_sample=exact_gap,
    )


def action_top10_ranking_loss(
    real: VisualWorldLoss,
    shuffled: VisualWorldLoss,
    zero: VisualWorldLoss,
    *,
    minimum_relative_degradation: float = 0.05,
    strong_relative_degradation: float = 0.10,
    detach_real_margin: bool = False,
) -> ActionTop10RankingLoss:
    """Require both action ablations to be 5% worse and one to be 10% worse.

    Relative margins multiply the real-action top10 MSE instead of dividing
    by it, keeping every hinge in the original DINO-feature MSE scale.  The
    three logical constraints are averaged for the combined penalty.
    """

    ratios = (minimum_relative_degradation, strong_relative_degradation)
    if any(not math.isfinite(float(value)) or value < 0.0 for value in ratios):
        raise ValueError("relative degradation margins must be finite and non-negative")
    if strong_relative_degradation < minimum_relative_degradation:
        raise ValueError(
            "strong_relative_degradation must be at least "
            "minimum_relative_degradation"
        )
    real_top10 = real.top10_per_sample
    shuffled_top10 = shuffled.top10_per_sample
    zero_top10 = zero.top10_per_sample
    _validate_per_sample_values(
        real_top10_per_sample=real_top10,
        shuffled_top10_per_sample=shuffled_top10,
        zero_top10_per_sample=zero_top10,
    )
    margin_source = real_top10.detach() if detach_real_margin else real_top10
    minimum = (1.0 + float(minimum_relative_degradation)) * margin_source
    strong = (1.0 + float(strong_relative_degradation)) * margin_source
    shuffle_hinge = torch.relu(minimum - shuffled_top10)
    zero_hinge = torch.relu(minimum - zero_top10)
    either_strong_hinge = torch.relu(
        strong - torch.maximum(shuffled_top10, zero_top10)
    )
    return ActionTop10RankingLoss(
        loss_per_sample=(shuffle_hinge + zero_hinge + either_strong_hinge) / 3.0,
        shuffle_hinge_per_sample=shuffle_hinge,
        zero_hinge_per_sample=zero_hinge,
        either_strong_hinge_per_sample=either_strong_hinge,
    )


def visual_world_loss(
    prediction: Tensor,
    target: Tensor,
    current: Tensor,
    *,
    topk_fraction: float = 0.20,
    diagnostic_top10_fraction: float = 0.10,
    all_weight: float = 0.25,
    motion_weight: float = 0.25,
    topk_weight: float = 0.50,
    motion_clip: float = 4.0,
    eps: float = 1e-8,
    feature_metric: str = "mse",
) -> VisualWorldLoss:
    """Compute action-relevant DINO-map supervision and matched diagnostics.

    ``target`` and ``current`` are oracle inputs used only for reduction.  The
    predictor receives neither tensor through this function.
    """

    if prediction.ndim != 4:
        raise ValueError("visual maps must be [B,C,H,W]")
    if prediction.shape != target.shape or prediction.shape != current.shape:
        raise ValueError(
            "prediction/target/current shape mismatch: "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, {tuple(current.shape)}"
        )
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0,1]")
    if not 0.0 < diagnostic_top10_fraction <= 1.0:
        raise ValueError("diagnostic_top10_fraction must be in (0,1]")
    component_weights = (all_weight, motion_weight, topk_weight)
    if any(float(value) < 0.0 for value in component_weights):
        raise ValueError("visual loss weights must be non-negative")
    if abs(sum(component_weights) - 1.0) > 1e-6:
        raise ValueError("visual loss weights must sum to 1")
    if motion_clip <= 0.0 or eps <= 0.0:
        raise ValueError("motion_clip and eps must be positive")
    if feature_metric not in {"mse", "cosine"}:
        raise ValueError("feature_metric must be mse|cosine")

    # Keep feature reductions in float32. Capacity runs use channel-normalized
    # cosine error plus a small bounded norm term so direction dominates without
    # allowing the predicted feature magnitude to drift freely.
    prediction_float = prediction.float()
    target_detached = target.detach().float()
    current_detached = current.detach().float()

    def feature_error(left: Tensor, right: Tensor) -> Tensor:
        if feature_metric == "mse":
            return (left - right).square().mean(dim=1)
        left_norm = torch.linalg.vector_norm(left, dim=1, keepdim=True)
        right_norm = torch.linalg.vector_norm(right, dim=1, keepdim=True)
        direction = 1.0 - (
            F.normalize(left, dim=1, eps=eps)
            * F.normalize(right, dim=1, eps=eps)
        ).sum(dim=1).clamp(-1.0, 1.0)
        relative_norm = (
            (left_norm - right_norm)
            / (left_norm + right_norm).clamp_min(eps)
        ).square().squeeze(1)
        return 0.95 * direction + 0.05 * relative_norm

    error = feature_error(prediction_float, target_detached).flatten(1)
    copy_error = feature_error(current_detached, target_detached).flatten(1)

    with torch.no_grad():
        energy = feature_error(target_detached, current_detached)
        flat_energy = energy.flatten(1)
        mean_energy = flat_energy.mean(dim=1, keepdim=True)
        motion_weights = flat_energy / mean_energy.clamp_min(eps)
        motion_weights = motion_weights.clamp(max=float(motion_clip))
        # All-static samples have zero mean energy; use uniform motion weights.
        static_sample = mean_energy <= eps
        motion_weights = torch.where(
            static_sample,
            torch.ones_like(motion_weights),
            motion_weights,
        )
        # The pre-clip normalization makes one unit equal to the sample's mean
        # motion energy.  Do not renormalize after clipping: doing so would let
        # a sparse outlier exceed ``motion_clip`` again.  L_all supplies the
        # non-zero static-region contribution in the composite objective.
        topk_mask = _top_mask(flat_energy, topk_fraction)
        top10_mask = _top_mask(flat_energy, diagnostic_top10_fraction)
        static_mask = ~topk_mask

    all_per_sample = error.mean(dim=1)
    motion_per_sample = (
        error * motion_weights
    ).sum(dim=1) / motion_weights.sum(dim=1).clamp_min(eps)
    topk_per_sample = _masked_mean(error, topk_mask, eps=eps)
    top10_per_sample = _masked_mean(error, top10_mask, eps=eps)
    static_per_sample = _masked_mean(error, static_mask, eps=eps)

    copy_all_per_sample = copy_error.mean(dim=1)
    copy_motion_per_sample = (
        copy_error * motion_weights
    ).sum(dim=1) / motion_weights.sum(dim=1).clamp_min(eps)
    copy_topk_per_sample = _masked_mean(copy_error, topk_mask, eps=eps)
    copy_top10_per_sample = _masked_mean(copy_error, top10_mask, eps=eps)
    copy_static_per_sample = _masked_mean(copy_error, static_mask, eps=eps)
    loss_per_sample = (
        float(all_weight) * all_per_sample
        + float(motion_weight) * motion_per_sample
        + float(topk_weight) * topk_per_sample
    )
    return VisualWorldLoss(
        loss_per_sample=loss_per_sample,
        all_per_sample=all_per_sample,
        motion_per_sample=motion_per_sample,
        topk_per_sample=topk_per_sample,
        top10_per_sample=top10_per_sample,
        static_per_sample=static_per_sample,
        copy_all_per_sample=copy_all_per_sample,
        copy_motion_per_sample=copy_motion_per_sample,
        copy_topk_per_sample=copy_topk_per_sample,
        copy_top10_per_sample=copy_top10_per_sample,
        copy_static_per_sample=copy_static_per_sample,
        motion_energy_per_sample=mean_energy.squeeze(1),
        motion_weights=motion_weights,
        topk_mask=topk_mask.view(prediction.shape[0], prediction.shape[2], prediction.shape[3]),
        top10_mask=top10_mask.view(prediction.shape[0], prediction.shape[2], prediction.shape[3]),
        static_mask=static_mask.view(prediction.shape[0], prediction.shape[2], prediction.shape[3]),
    )


def _world_task_ids(
    batch: dict[str, Tensor], time_index: int, device: torch.device
) -> Tensor:
    task_id = batch.get("task_id")
    if task_id is None:
        task_id = batch.get("task_ids")
    if task_id is None:
        task_id = batch.get("instruction_id")
    if task_id is None:
        raise ValueError("visual World supervision requires per-sample task ids")
    task_id = torch.as_tensor(task_id, device=device)
    if task_id.ndim > 1:
        task_id = task_id[:, time_index]
    if task_id.ndim != 1 or task_id.shape[0] != batch["actions"].shape[0]:
        raise ValueError("World task ids must have shape [B] or [B,T]")
    return task_id.to(dtype=torch.long)


def _nearest_cross_episode_donors(
    proprio: Tensor,
    task_ids: Tensor,
    episode_ids: Tensor,
    eligible: Tensor,
) -> Tensor:
    """Return deterministic proprio-nearest cross-episode donor indices."""

    batch = proprio.shape[0]
    if proprio.ndim != 2:
        raise ValueError("proprio must be [B,P]")
    if task_ids.ndim != 1 or task_ids.shape[0] != batch:
        raise ValueError("task_ids must be [B]")
    if episode_ids.ndim != 1 or episode_ids.shape[0] != batch:
        raise ValueError("episode_ids must be [B]")
    if eligible.ndim != 1 or eligible.shape[0] != batch or eligible.dtype != torch.bool:
        raise ValueError("eligible must be bool [B]")

    state = proprio.detach().to(device="cpu", dtype=torch.float64)
    task = task_ids.detach().to(device="cpu", dtype=torch.int64)
    episode = episode_ids.detach().to(device="cpu", dtype=torch.int64)
    eligible = eligible.detach().to(device="cpu", dtype=torch.bool)
    donors = torch.full((batch,), -1, dtype=torch.int64)
    task_rows: dict[int, list[int]] = {}
    for row in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
        task_rows.setdefault(int(task[row]), []).append(row)

    # Rows are appended in global index order, so a distance tie still selects
    # the same lowest global donor as the former batch-wide argmin.  Limiting
    # each scan to one task changes the work from O(N^2) to sum_t O(N_t^2).
    for rows in task_rows.values():
        indices = torch.tensor(rows, dtype=torch.int64)
        task_state = state.index_select(0, indices)
        task_episode = episode.index_select(0, indices)
        for local_row, row in enumerate(rows):
            candidate = task_episode.ne(task_episode[local_row])
            if not bool(candidate.any()):
                continue
            distance = (task_state - task_state[local_row]).square().sum(dim=-1)
            distance.masked_fill_(~candidate, float("inf"))
            donors[row] = indices[distance.argmin()]
    return donors


def prepare_visual_world_action_ranking(
    payload: dict,
    *,
    planning_stride: int = 6,
) -> dict[str, object]:
    """Attach the fixed train-split shuffled-action table to a dataset payload."""

    from va_compound.world_contract import (
        PEER_PLANNING_STRIDES,
        WORLD_ACTION_DONOR_CONTRACT,
    )

    if planning_stride not in PEER_PLANNING_STRIDES:
        raise ValueError(
            f"planning_stride must be one of {sorted(PEER_PLANNING_STRIDES)}"
        )

    actions = torch.as_tensor(payload["actions"])
    proprio = torch.as_tensor(payload["proprio"])
    task_ids = torch.as_tensor(payload["instruction_id"], dtype=torch.int64)
    episode_ids = torch.as_tensor(payload["episode_id"], dtype=torch.int64)
    metadata = payload.get("metadata") or {}
    world_horizon = int(metadata.get("world_target_horizon", planning_stride))
    if not 1 <= world_horizon <= actions.shape[2]:
        raise ValueError("World target horizon is outside the logged action chunk")
    explicit_valid = payload.get("world_target_valid_mask")
    valid = (
        torch.as_tensor(explicit_valid, dtype=torch.bool)
        if explicit_valid is not None
        else transition_mask(
            torch.as_tensor(payload["action_valid_mask"]),
            cycle_steps=planning_stride,
        )
    )
    if explicit_valid is not None and tuple(valid.shape) != tuple(actions.shape[:2]):
        raise ValueError("world_target_valid_mask must have shape [N,T]")
    rows, times = torch.nonzero(valid, as_tuple=True)
    flat_actions = actions[rows, times, :world_horizon]
    flat_proprio = proprio[rows, times]
    flat_tasks = task_ids[rows]
    flat_episodes = episode_ids[rows]
    donors = _nearest_cross_episode_donors(
        flat_proprio,
        flat_tasks,
        flat_episodes,
        torch.ones(rows.numel(), dtype=torch.bool),
    )
    if bool((donors < 0).any()):
        raise ValueError("visual World action ranking found a transition without a donor")
    shuffled = flat_actions.index_select(0, donors)
    distinct = shuffled.ne(flat_actions).any(dim=(1, 2))
    table = actions.new_zeros(
        (
            actions.shape[0],
            valid.shape[1],
            world_horizon,
            actions.shape[-1],
        )
    )
    table[rows, times] = shuffled
    rank_mask = torch.zeros_like(valid)
    rank_mask[rows, times] = distinct
    payload["world_rank_shuffle_action"] = table
    payload["world_rank_shuffle_mask"] = rank_mask

    identity = torch.stack((rows, times, donors, distinct.to(torch.int64)), dim=1)
    digest = hashlib.sha256(identity.contiguous().numpy().tobytes()).hexdigest()
    return {
        "world_action_donor_contract": WORLD_ACTION_DONOR_CONTRACT,
        "world_action_donor_sha256": digest,
        "world_action_donor_transitions": int(rows.numel()),
        "world_action_rank_transitions": int(distinct.sum()),
    }


def _summarize_visual_world_metrics(
    final_records: list[dict[str, Tensor]],
    stage_records: list[list[tuple[Tensor, Tensor, Tensor]]],
) -> dict[int, dict[str, object]]:
    """Reduce final-stage World/copy metrics independently for each task."""

    if not final_records:
        return {}
    task_values = sorted(
        {
            int(value)
            for record in final_records
            for value in torch.unique(record["task_ids"]).detach().cpu()
        }
    )
    output: dict[int, dict[str, object]] = {}
    metric_names = (
        "world_all",
        "copy_all",
        "world_motion",
        "copy_motion",
        "world_top10",
        "copy_top10",
        "world_static",
        "copy_static",
        "motion_energy",
    )
    for task_id in task_values:
        masks = [
            record["valid"] & record["task_ids"].eq(task_id)
            for record in final_records
        ]
        reduced = {
            name: float(
                masked_reduction(
                    [record[name] for record in final_records], masks
                ).detach()
            )
            for name in metric_names
        }
        reduced["gain_all"] = reduced["copy_all"] - reduced["world_all"]
        reduced["gain_motion"] = (
            reduced["copy_motion"] - reduced["world_motion"]
        )
        copy_top10 = float(reduced["copy_top10"])
        reduced["relative_gain_top10"] = (
            (copy_top10 - float(reduced["world_top10"])) / copy_top10
            if copy_top10 > 0.0
            else 0.0
        )
        reduced["transitions"] = sum(int(mask.sum().item()) for mask in masks)
        per_stage = []
        for records in stage_records:
            stage_masks = [
                valid & task_ids.eq(task_id)
                for task_ids, valid, _ in records
            ]
            per_stage.append(
                float(
                    masked_reduction(
                        [value for _, _, value in records], stage_masks
                    ).detach()
                )
            )
        reduced["stage_losses"] = per_stage
        output[task_id] = reduced
    return output


__all__ = [
    "ActionTop10GapLoss",
    "ActionTop10RankingLoss",
    "GatedStaticCopyAnchorLoss",
    "VisualNoRegressionLoss",
    "VisualStaticRelativePenaltyLoss",
    "VisualStaticStageChainLoss",
    "VisualWorldLoss",
    "action_top10_gap_loss",
    "action_top10_oracle_straight_through_gap_loss",
    "action_top10_pairwise_loss",
    "action_top10_ranking_loss",
    "apply_stage_weight_overrides",
    "canonical_stage_weight_overrides",
    "gated_static_copy_anchor_loss",
    "late_stage_anchor_loss",
    "masked_numerator_denominator",
    "masked_reduction",
    "stage_supervision_weights",
    "static_copy_anchor_loss",
    "transition_mask",
    "visual_no_regression_loss",
    "visual_static_relative_penalty",
    "visual_static_stage_chain_loss",
    "visual_world_loss",
]
