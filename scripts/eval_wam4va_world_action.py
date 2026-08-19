#!/usr/bin/env python
"""Held-out action-dependence evaluation for WAM4VA visual World prediction.

The evaluator is intentionally read-only with respect to its checkpoint and
dataset.  It accepts only a fixed episode-level eval split and, by default,
only checkpoints marked with a supported visual-motion supervision contract.
The completed ``visual_motion_v1`` through gap-v6 lineages remain read-only
compatible; new runs use ``visual_motion_oracle_stgap_v7``. For every valid
World transition it compares the final
and intermediate predictors under:

* the logged executable action prefix ``[6, 4]``;
* a same-task, different-episode proprio-nearest action prefix;
* a zero action prefix;
* copy-last-frame (no model forward).

All conditions reuse the exact same visual/state/language inputs and the same
target-derived oracle reductions.  The target never enters the predictor.
Confidence intervals use paired, within-task episode bootstrap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from va_compound.model import (  # noqa: E402
    VACompoundConfig,
    VACompoundPolicy,
    VisualMemory,
)
from va_compound.world_supervision import (  # noqa: E402
    transition_mask,
    visual_world_loss,
)


# v1-v4 are retained for read-only evaluation of completed lineages. New runs
# use v5; every version shares the same held-out prediction protocol.
LEGACY_WORLD_SUPERVISION_CONTRACT = "visual_motion_v1"
PREVIOUS_WORLD_SUPERVISION_CONTRACT = "visual_motion_constrained_v2"
PRIOR_WORLD_SUPERVISION_CONTRACT = "visual_motion_constrained_v3"
LAST_WORLD_SUPERVISION_CONTRACT = "visual_motion_constrained_v4"
CONSTRAINED_WORLD_SUPERVISION_CONTRACT = "visual_motion_constrained_v5"
GAP_WORLD_SUPERVISION_CONTRACT = "visual_motion_gap_v6"
WORLD_SUPERVISION_CONTRACT = "visual_motion_oracle_stgap_v7"
SUPPORTED_WORLD_SUPERVISION_CONTRACTS = frozenset(
    {
        LEGACY_WORLD_SUPERVISION_CONTRACT,
        PREVIOUS_WORLD_SUPERVISION_CONTRACT,
        PRIOR_WORLD_SUPERVISION_CONTRACT,
        LAST_WORLD_SUPERVISION_CONTRACT,
        CONSTRAINED_WORLD_SUPERVISION_CONTRACT,
        GAP_WORLD_SUPERVISION_CONTRACT,
        WORLD_SUPERVISION_CONTRACT,
    }
)
WORLD_LOGGED_BRANCH_CONTRACT = "matched_context_full_forward_v1"
CONSTRAINED_WORLD_LOSS_WEIGHTS = {
    "all": 0.25,
    "motion": 0.25,
    "top20": 0.50,
}
CONSTRAINED_WORLD_TRANSITION = "current_first6_and_next_first_v1"
CONSTRAINED_WORLD_STAGE_AUXILIARY_DECAY = 0.25
LAST_CONSTRAINED_WORLD_NO_REGRESSION = {
    "all_ratio": 1.0,
    "static_ratio": 1.05,
    "weight": 1.0,
}
CONSTRAINED_WORLD_NO_REGRESSION = {
    "all_ratio": 1.0,
    "weight": 1.0,
    "components": ["all"],
}
PREVIOUS_CONSTRAINED_WORLD_ACTION_RANKING = {
    "stage": "final_logged",
    "top10_min_relative_margin": 0.05,
    "top10_strong_relative_margin": 0.10,
    "weight": 0.25,
    "negatives": ["shuffle", "zero"],
}
PRIOR_CONSTRAINED_WORLD_STATIC_COPY_ANCHOR = {
    "static_ratio": 1.05,
    "weight": 1.0,
    "region": "outside_top20",
    "gate": "per_sample",
}
PRIOR_CONSTRAINED_WORLD_ACTION_RANKING = {
    "stage": "full_8stage_counterfactual_final",
    "top10_relative_margin": 0.10,
    "weight": 1.0,
    "negatives": ["shuffle", "zero"],
    "schedule": "alternating_global_step_plus_time",
    "rng": "logged_branch_replay",
    "gradient": "final_stage_recompute",
}
LAST_CONSTRAINED_WORLD_STATIC_COPY_CONSTRAINT = {
    "static_ratio": 1.05,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "relative_hinge_plus_half_normalized_square_v1",
    "eps": 1e-6,
}
LAST_CONSTRAINED_WORLD_ACTION_RANKING = {
    "stage": "full_8stage_counterfactual_final",
    "top10_min_relative_margin": 0.05,
    "top10_strong_relative_margin": 0.10,
    "weight": 4.0,
    "negatives": ["shuffle", "zero"],
    "schedule": "both_each_valid_transition",
    "mask": "per_negative_and_both_for_strong",
    "rng": "logged_branch_replay",
    "gradient": "final_stage_recompute",
}
CONSTRAINED_WORLD_STATIC_COPY_CONSTRAINT = {
    "static_ratio": 1.05,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "stage_chain_exact_hinge_v1",
    "reduction": "sum_stages_then_masked_transition_mean",
    "boundary": "copy_then_detached_min_previous_copy",
}
CONSTRAINED_WORLD_ACTION_RANKING = {
    "stage": "full_8stage_counterfactual_final",
    "top10_min_relative_margin": 0.05,
    "top10_strong_relative_margin": 0.10,
    "weight": 1.0,
    "negatives": ["shuffle", "zero"],
    "schedule": "both_each_valid_transition",
    "mask": "per_negative_and_both_for_strong",
    "rng": "logged_branch_replay",
    "gradient": "wrong_actions_only_detached_real_margin_v1",
}
GAP_WORLD_STATIC_COPY_CONSTRAINT = {
    "static_ratio": 1.05,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "copy_budget_hinge_v1",
    "reduction": "stage_aux_weighted_masked_mean",
    "boundary": "1.05_detached_copy_each_stage",
}
_GAP_WORLD_ACTION_RANKING_COMMON = {
    "top10_min_relative_margin": 0.05,
    "weight": 1.0,
    "negatives": ["shuffle"],
    "diagnostic_negatives": ["zero"],
    "context": "logged_stage_detached_pair",
    "gradient": "control_variate_real_minus_shuffle_v1",
}
GAP_WORLD_ACTION_RANKINGS = (
    {
        "stage": "final_direct_matched_context",
        **_GAP_WORLD_ACTION_RANKING_COMMON,
        "schedule": "final_each_valid_transition",
    },
    {
        "stage": "rotating_8stage_direct_matched_context",
        **_GAP_WORLD_ACTION_RANKING_COMMON,
        "schedule": "(global_step+time_index)%num_stages",
    },
)
ORACLE_STGAP_WORLD_STATIC_COPY_CONSTRAINT = {
    "static_ratio": 1.0,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "copy_budget_hinge_plus_always_copy_anchor_v1",
    "reduction": "stage_aux_weighted_masked_mean",
    "boundary": "1.00_detached_copy_each_stage",
}
ORACLE_STGAP_WORLD_STATIC_COPY_CONSTRAINTS = (
    ORACLE_STGAP_WORLD_STATIC_COPY_CONSTRAINT,
    {**ORACLE_STGAP_WORLD_STATIC_COPY_CONSTRAINT, "weight": 2.0},
)
_ORACLE_STGAP_WORLD_ACTION_RANKING_COMMON = {
    "top10_min_relative_margin": 0.12,
    "weight": 1.0,
    "negatives": ["shuffle"],
    "diagnostic_negatives": ["zero"],
    "context": "logged_stage_detached_pair",
    "gradient": "oracle_motion_straight_through_exact_gap_v1",
}
ORACLE_STGAP_WORLD_ACTION_RANKINGS = tuple(
    ranking
    for base_ranking in (
        {
            "stage": "final_direct_matched_context",
            **_ORACLE_STGAP_WORLD_ACTION_RANKING_COMMON,
            "schedule": "final_each_valid_transition",
        },
        {
            "stage": "rotating_8stage_direct_matched_context",
            **_ORACLE_STGAP_WORLD_ACTION_RANKING_COMMON,
            "schedule": "(global_step+time_index)%num_stages",
        },
    )
    for ranking in (base_ranking, {**base_ranking, "per_sample_cap": 0.2})
)
DEFAULT_CYCLE_STEPS = 6
PROPOSAL_FLOW_STEPS = 8
DEFAULT_TASK_IDS = (0, 16)
DEFAULT_TASK_NAMES = {0: "assembly-v3", 16: "door-unlock-v3"}
CI_LEVEL = 0.95


def _as_1d_tensor(values: Tensor | Sequence, *, name: str) -> Tensor:
    tensor = torch.as_tensor(values)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(tensor.shape)}")
    return tensor


def nearest_episode_shuffle(
    actions: Tensor,
    proprio: Tensor,
    task_ids: Tensor | Sequence[int],
    episode_ids: Tensor | Sequence[int],
    *,
    chunk_size: int = 2048,
) -> tuple[Tensor, Tensor]:
    """Return proprio-nearest donor actions and donor indices.

    Donors are constrained to the same task and a different episode.  Inputs
    are transition-level rows, and ``actions`` must contain the executable
    ``[6, 4]`` prefix.  Ties are resolved by the lowest input row, making the
    mapping deterministic and independent of batching.
    """

    if actions.ndim != 3 or tuple(actions.shape[1:]) != (6, 4):
        raise ValueError(
            f"actions must be [N,6,4], got {tuple(actions.shape)}"
        )
    if proprio.ndim != 2 or proprio.shape[0] != actions.shape[0]:
        raise ValueError("proprio must be [N,P] and align with actions")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    task = _as_1d_tensor(task_ids, name="task_ids").to(torch.int64)
    episode = _as_1d_tensor(episode_ids, name="episode_ids").to(torch.int64)
    n = actions.shape[0]
    if task.shape[0] != n or episode.shape[0] != n:
        raise ValueError("task_ids/episode_ids must align with actions")
    if n == 0:
        raise ValueError("nearest-episode shuffle needs at least one transition")
    if not torch.isfinite(proprio).all():
        raise ValueError("proprio must contain only finite values")

    # Selection happens on CPU float64 so a GPU batch size or mixed precision
    # cannot change the fixed donor mapping.
    state_cpu = proprio.detach().to(device="cpu", dtype=torch.float64)
    task_cpu = task.detach().cpu()
    episode_cpu = episode.detach().cpu()
    donors = torch.full((n,), -1, dtype=torch.int64)

    for task_value in torch.unique(task_cpu, sorted=True).tolist():
        task_rows = torch.nonzero(task_cpu == task_value, as_tuple=False).flatten()
        task_episodes = torch.unique(episode_cpu[task_rows])
        if task_episodes.numel() < 2:
            raise ValueError(
                f"task {task_value} has only {task_episodes.numel()} episode(s); "
                "different-episode shuffle is impossible"
            )
        candidate_state = state_cpu[task_rows]
        candidate_episode = episode_cpu[task_rows]
        for start in range(0, task_rows.numel(), chunk_size):
            query_rows = task_rows[start : start + chunk_size]
            query_state = state_cpu[query_rows]
            # Squared Euclidean distance preserves nearest-neighbor ordering.
            distance = (query_state[:, None] - candidate_state[None]).square().sum(-1)
            same_episode = (
                episode_cpu[query_rows, None] == candidate_episode[None, :]
            )
            distance.masked_fill_(same_episode, math.inf)
            nearest_local = distance.argmin(dim=1)
            nearest_distance = distance.gather(1, nearest_local[:, None]).squeeze(1)
            if not torch.isfinite(nearest_distance).all():
                raise ValueError(
                    f"task {task_value} has a transition without a cross-episode donor"
                )
            donors[query_rows] = task_rows[nearest_local]

    if bool((donors < 0).any()):
        raise RuntimeError("internal error: incomplete shuffled-action donor mapping")
    action_donors = donors.to(device=actions.device)
    shuffled = actions.index_select(0, action_donors).clone()
    return shuffled, action_donors


def _episode_means(
    values: Tensor | Sequence[float],
    episode_ids: Tensor | Sequence[int],
    *,
    task_ids: Tensor | Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    value = _as_1d_tensor(values, name="values").detach().cpu().to(torch.float64)
    episode = _as_1d_tensor(episode_ids, name="episode_ids").detach().cpu()
    if value.shape != episode.shape or value.numel() == 0:
        raise ValueError("values/episode_ids must be aligned non-empty vectors")
    if not torch.isfinite(value).all():
        raise ValueError("bootstrap values must be finite")
    if task_ids is not None:
        task = _as_1d_tensor(task_ids, name="task_ids").detach().cpu()
        if task.shape != value.shape:
            raise ValueError("task_ids must align with values")
        unique_tasks = torch.unique(task)
        if unique_tasks.numel() != 1:
            raise ValueError(
                "episode bootstrap must be called within exactly one task"
            )

    unique_episodes = torch.unique(episode, sorted=True)
    means = torch.stack([value[episode == ep].mean() for ep in unique_episodes])
    return means.numpy(), unique_episodes.numpy()


def _ci_summary(point: float, draws: np.ndarray, confidence: float) -> dict[str, float]:
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(point),
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(confidence),
    }


def _undefined_ci(confidence: float) -> dict[str, float | None]:
    return {
        "estimate": None,
        "low": None,
        "high": None,
        "confidence": float(confidence),
    }


def paired_episode_bootstrap(
    candidate: Tensor | Sequence[float],
    reference: Tensor | Sequence[float],
    episode_ids: Tensor | Sequence[int],
    *,
    task_ids: Tensor | Sequence[int] | None = None,
    n_resamples: int = 4000,
    seed: int = 0,
    confidence: float = CI_LEVEL,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Paired within-task bootstrap with episodes as the sampling unit.

    Transition values are first averaged inside each episode.  Episodes are
    then sampled with replacement, using the same sampled episodes for the
    candidate and reference.  This prevents overlapping windows from acting
    like independent evidence and prevents a long episode from dominating.
    """

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    cand = _as_1d_tensor(candidate, name="candidate")
    ref = _as_1d_tensor(reference, name="reference")
    if cand.shape != ref.shape:
        raise ValueError("candidate/reference must be paired and shape-identical")
    cand_ep, ref_ep, cand_draw, ref_draw = _paired_episode_draws(
        cand,
        ref,
        episode_ids,
        task_ids=task_ids,
        n_resamples=n_resamples,
        seed=seed,
    )
    n_episodes = cand_ep.shape[0]
    difference_draw = cand_draw - ref_draw
    cand_point = float(cand_ep.mean())
    ref_point = float(ref_ep.mean())
    difference_point = cand_point - ref_point

    relative_point: float | None
    valid_relative_draw = np.abs(ref_draw) > eps
    if abs(ref_point) <= eps or not bool(valid_relative_draw.all()):
        relative = _undefined_ci(confidence)
        relative_point = None
    else:
        relative_point = difference_point / ref_point
        relative = _ci_summary(
            relative_point,
            difference_draw / ref_draw,
            confidence,
        )

    return {
        "candidate": _ci_summary(cand_point, cand_draw, confidence),
        "reference": _ci_summary(ref_point, ref_draw, confidence),
        "difference": _ci_summary(difference_point, difference_draw, confidence),
        "relative_difference": relative,
        "n_episodes": int(n_episodes),
        "n_observations": int(cand.numel()),
        "sampling_unit": "episode",
        "episode_weighting": "equal",
        "paired": True,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _paired_episode_draws(
    candidate: Tensor,
    reference: Tensor,
    episode_ids: Tensor | Sequence[int],
    *,
    task_ids: Tensor | Sequence[int] | None,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cand_ep, episode_values = _episode_means(
        candidate, episode_ids, task_ids=task_ids
    )
    ref_ep, reference_episodes = _episode_means(
        reference, episode_ids, task_ids=task_ids
    )
    if not np.array_equal(episode_values, reference_episodes):
        raise RuntimeError("candidate/reference episode alignment changed")
    rng = np.random.default_rng(seed)
    n_episodes = cand_ep.shape[0]
    sampled = rng.integers(0, n_episodes, size=(n_resamples, n_episodes))
    return (
        cand_ep,
        ref_ep,
        cand_ep[sampled].mean(axis=1),
        ref_ep[sampled].mean(axis=1),
    )


def task_macro_paired_episode_bootstrap(
    candidate: Tensor | Sequence[float],
    reference: Tensor | Sequence[float],
    episode_ids: Tensor | Sequence[int],
    task_ids: Tensor | Sequence[int],
    *,
    n_resamples: int = 4000,
    seed: int = 0,
    confidence: float = CI_LEVEL,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Bootstrap an equal-task macro from task-local episode resamples."""

    if n_resamples < 1 or not 0.0 < confidence < 1.0 or eps <= 0.0:
        raise ValueError("invalid macro bootstrap arguments")
    cand = _as_1d_tensor(candidate, name="candidate")
    ref = _as_1d_tensor(reference, name="reference")
    episode = _as_1d_tensor(episode_ids, name="episode_ids")
    task = _as_1d_tensor(task_ids, name="task_ids")
    if not (cand.shape == ref.shape == episode.shape == task.shape) or cand.numel() == 0:
        raise ValueError("macro bootstrap inputs must be aligned non-empty vectors")
    task_values = torch.unique(task, sorted=True)
    if task_values.numel() < 1:
        raise ValueError("macro bootstrap needs at least one task")

    candidate_points = []
    reference_points = []
    candidate_draws = []
    reference_draws = []
    episode_counts = []
    for task_value in task_values.tolist():
        selected = task == task_value
        cand_ep, ref_ep, cand_draw, ref_draw = _paired_episode_draws(
            cand[selected],
            ref[selected],
            episode[selected],
            task_ids=task[selected],
            n_resamples=n_resamples,
            seed=_stable_seed(seed, f"macro-task-{task_value}"),
        )
        candidate_points.append(float(cand_ep.mean()))
        reference_points.append(float(ref_ep.mean()))
        candidate_draws.append(cand_draw)
        reference_draws.append(ref_draw)
        episode_counts.append(int(cand_ep.shape[0]))

    candidate_draw = np.stack(candidate_draws, axis=1)
    reference_draw = np.stack(reference_draws, axis=1)
    difference_draw = candidate_draw - reference_draw
    candidate_macro_draw = candidate_draw.mean(axis=1)
    reference_macro_draw = reference_draw.mean(axis=1)
    difference_macro_draw = difference_draw.mean(axis=1)
    candidate_point_array = np.asarray(candidate_points)
    reference_point_array = np.asarray(reference_points)
    difference_point_array = candidate_point_array - reference_point_array
    candidate_point = float(candidate_point_array.mean())
    reference_point = float(reference_point_array.mean())
    difference_point = float(difference_point_array.mean())

    relative_denominator_valid = np.abs(reference_draw) > eps
    point_denominator_valid = np.abs(reference_point_array) > eps
    if not bool(relative_denominator_valid.all()) or not bool(point_denominator_valid.all()):
        relative = _undefined_ci(confidence)
    else:
        relative_draw = (difference_draw / reference_draw).mean(axis=1)
        relative_point = float(
            (difference_point_array / reference_point_array).mean()
        )
        relative = _ci_summary(relative_point, relative_draw, confidence)
    relative["aggregation"] = "equal_task_macro_of_task_relative_differences"

    return {
        "candidate": {
            **_ci_summary(candidate_point, candidate_macro_draw, confidence),
            "aggregation": "equal_task_macro",
        },
        "reference": {
            **_ci_summary(reference_point, reference_macro_draw, confidence),
            "aggregation": "equal_task_macro",
        },
        "difference": {
            **_ci_summary(difference_point, difference_macro_draw, confidence),
            "aggregation": "equal_task_macro",
        },
        "relative_difference": relative,
        "n_tasks": int(task_values.numel()),
        "task_ids": [int(value) for value in task_values.tolist()],
        "episodes_per_task": episode_counts,
        "sampling_unit": "episode_within_task",
        "task_weighting": "equal",
        "paired": True,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _negate_ci(interval: Mapping[str, Any]) -> dict[str, Any]:
    estimate = interval.get("estimate")
    low = interval.get("low")
    high = interval.get("high")
    return {
        **dict(interval),
        "estimate": None if estimate is None else -float(estimate),
        "low": None if high is None else -float(high),
        "high": None if low is None else -float(low),
        "confidence": float(interval.get("confidence", CI_LEVEL)),
    }


def check_target_permutation_invariance(
    forward_with_target: Callable[[Tensor], Tensor | tuple[Tensor, Tensor]],
    target: Tensor,
    *,
    permutation: Tensor | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run a target permutation check and require bitwise-equal predictions.

    ``forward_with_target`` deliberately receives a target argument so tests
    can catch accidental target use.  A correct evaluator closure ignores it
    until after the predictor has returned, then may use it only for loss-side
    diagnostics.
    """

    if target.ndim < 1 or target.shape[0] < 2:
        raise ValueError("target permutation needs at least two samples")
    batch = target.shape[0]
    if permutation is None:
        perm = torch.roll(torch.arange(batch, device=target.device), shifts=1)
    else:
        perm = torch.as_tensor(permutation, device=target.device, dtype=torch.int64)
    if perm.shape != (batch,) or not torch.equal(
        torch.sort(perm).values, torch.arange(batch, device=target.device)
    ):
        raise ValueError("permutation must contain every batch index exactly once")
    if torch.equal(perm, torch.arange(batch, device=target.device)):
        raise ValueError("target permutation must be non-identity")

    permuted_target = target.index_select(0, perm)
    target_changed = not torch.equal(target, permuted_target)
    target_finite = bool(
        torch.isfinite(target).all() and torch.isfinite(permuted_target).all()
    )

    def unpack(output: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor | None]:
        if isinstance(output, Tensor):
            return output, None
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], Tensor)
            or not isinstance(output[1], Tensor)
        ):
            raise TypeError(
                "forward_with_target must return prediction or (prediction, loss)"
            )
        return output

    with torch.inference_mode():
        prediction, loss = unpack(forward_with_target(target))
        permuted_prediction, permuted_loss = unpack(
            forward_with_target(permuted_target)
        )
    if prediction.shape != permuted_prediction.shape:
        raise ValueError("prediction shape changed under target permutation")
    bitwise_equal = torch.equal(prediction, permuted_prediction)
    prediction_finite = bool(
        torch.isfinite(prediction).all()
        and torch.isfinite(permuted_prediction).all()
    )
    if prediction.numel():
        raw_max_abs_difference = float(
            (prediction.float() - permuted_prediction.float()).abs().max().item()
        )
        max_abs_difference = (
            raw_max_abs_difference
            if math.isfinite(raw_max_abs_difference)
            else None
        )
    else:
        max_abs_difference = 0.0
    loss_changed = None
    loss_finite = None
    if (loss is None) != (permuted_loss is None):
        raise ValueError("loss return changed under target permutation")
    if loss is not None and permuted_loss is not None:
        if loss.shape != permuted_loss.shape:
            raise ValueError("loss shape changed under target permutation")
        loss_changed = not torch.equal(loss, permuted_loss)
        loss_finite = bool(
            torch.isfinite(loss).all() and torch.isfinite(permuted_loss).all()
        )
    loss_path_valid = loss_changed is True and loss_finite is True
    return {
        "passed": bool(
            target_changed
            and target_finite
            and bitwise_equal
            and prediction_finite
            and loss_path_valid
        ),
        "target_changed": bool(target_changed),
        "target_finite": target_finite,
        "prediction_bitwise_equal": bool(bitwise_equal),
        "prediction_finite": prediction_finite,
        "max_abs_prediction_difference": max_abs_difference,
        "loss_changed": loss_changed,
        "loss_finite": loss_finite,
        "loss_path_valid": loss_path_valid,
        "permutation": perm.detach().cpu().tolist(),
    }


def _estimate(interval: Mapping[str, Any]) -> float | None:
    value = interval.get("estimate")
    return None if value is None else float(value)


def _low(interval: Mapping[str, Any]) -> float | None:
    value = interval.get("low")
    return None if value is None else float(value)


def _confidence(interval: Mapping[str, Any]) -> float | None:
    value = interval.get("confidence")
    return None if value is None else float(value)


def _target_permutation_exact(report: Mapping[str, Any]) -> bool:
    max_difference = report.get("max_abs_prediction_difference")
    return bool(
        report.get("passed") is True
        and report.get("target_changed") is True
        and report.get("target_finite") is True
        and report.get("prediction_bitwise_equal") is True
        and report.get("prediction_finite") is True
        and isinstance(max_difference, (int, float))
        and not isinstance(max_difference, bool)
        and math.isfinite(float(max_difference))
        and float(max_difference) == 0.0
        and report.get("loss_changed") is True
        and report.get("loss_finite") is True
        and report.get("loss_path_valid") is True
    )


def _checkpoint_world_contract(
    checkpoint: Mapping[str, Any], *, allow_unmarked: bool
) -> tuple[str | None, str | None, bool]:
    contract = checkpoint.get("training_contract") or {}
    supervision = contract.get("world_supervision")
    logged_branch = contract.get("world_logged_branch")
    base_valid = bool(
        supervision in SUPPORTED_WORLD_SUPERVISION_CONTRACTS
        and logged_branch == WORLD_LOGGED_BRANCH_CONTRACT
    )
    if supervision in {
        PREVIOUS_WORLD_SUPERVISION_CONTRACT,
        PRIOR_WORLD_SUPERVISION_CONTRACT,
        LAST_WORLD_SUPERVISION_CONTRACT,
        CONSTRAINED_WORLD_SUPERVISION_CONTRACT,
        GAP_WORLD_SUPERVISION_CONTRACT,
        WORLD_SUPERVISION_CONTRACT,
    }:
        constrained_valid = bool(
            base_valid
            and contract.get("world_loss_weights")
            == CONSTRAINED_WORLD_LOSS_WEIGHTS
            and contract.get("world_transition") == CONSTRAINED_WORLD_TRANSITION
            and contract.get("world_stage_auxiliary_decay")
            == CONSTRAINED_WORLD_STAGE_AUXILIARY_DECAY
        )
        if supervision == PREVIOUS_WORLD_SUPERVISION_CONTRACT:
            valid = bool(
                constrained_valid
                and contract.get("world_no_regression")
                == LAST_CONSTRAINED_WORLD_NO_REGRESSION
                and contract.get("world_action_ranking")
                == PREVIOUS_CONSTRAINED_WORLD_ACTION_RANKING
            )
        elif supervision == PRIOR_WORLD_SUPERVISION_CONTRACT:
            valid = bool(
                constrained_valid
                and contract.get("world_no_regression")
                == LAST_CONSTRAINED_WORLD_NO_REGRESSION
                and contract.get("world_static_copy_anchor")
                == PRIOR_CONSTRAINED_WORLD_STATIC_COPY_ANCHOR
                and contract.get("world_action_ranking")
                == PRIOR_CONSTRAINED_WORLD_ACTION_RANKING
            )
        elif supervision == LAST_WORLD_SUPERVISION_CONTRACT:
            valid = bool(
                constrained_valid
                and contract.get("world_no_regression")
                == LAST_CONSTRAINED_WORLD_NO_REGRESSION
                and contract.get("world_static_copy_constraint")
                == LAST_CONSTRAINED_WORLD_STATIC_COPY_CONSTRAINT
                and contract.get("world_action_ranking")
                == LAST_CONSTRAINED_WORLD_ACTION_RANKING
            )
        elif supervision == CONSTRAINED_WORLD_SUPERVISION_CONTRACT:
            valid = bool(
                constrained_valid
                and contract.get("world_no_regression")
                == CONSTRAINED_WORLD_NO_REGRESSION
                and contract.get("world_static_copy_constraint")
                == CONSTRAINED_WORLD_STATIC_COPY_CONSTRAINT
                and contract.get("world_action_ranking")
                == CONSTRAINED_WORLD_ACTION_RANKING
            )
        elif supervision == GAP_WORLD_SUPERVISION_CONTRACT:
            valid = bool(
                constrained_valid
                and contract.get("world_no_regression")
                == CONSTRAINED_WORLD_NO_REGRESSION
                and contract.get("world_static_copy_constraint")
                == GAP_WORLD_STATIC_COPY_CONSTRAINT
                and contract.get("world_action_ranking")
                in GAP_WORLD_ACTION_RANKINGS
            )
        else:
            valid = bool(
                constrained_valid
                and contract.get("world_no_regression")
                == CONSTRAINED_WORLD_NO_REGRESSION
                and contract.get("world_static_copy_constraint")
                in ORACLE_STGAP_WORLD_STATIC_COPY_CONSTRAINTS
                and contract.get("world_action_ranking")
                in ORACLE_STGAP_WORLD_ACTION_RANKINGS
            )
    else:
        # v1 checkpoints predate the constrained no-regression/action-ranking
        # fields and remain valid for read-only diagnostics.
        valid = base_valid
    if not valid and not allow_unmarked:
        raise ValueError(
            "checkpoint does not match the visual World training graph: "
            f"world_supervision={supervision!r}, world_logged_branch={logged_branch!r}, "
            f"world_loss_weights={contract.get('world_loss_weights')!r}, "
            f"world_transition={contract.get('world_transition')!r}, "
            "world_stage_auxiliary_decay="
            f"{contract.get('world_stage_auxiliary_decay')!r}, "
            f"world_no_regression={contract.get('world_no_regression')!r}, "
            "world_static_copy_anchor="
            f"{contract.get('world_static_copy_anchor')!r}, "
            "world_static_copy_constraint="
            f"{contract.get('world_static_copy_constraint')!r}, "
            f"world_action_ranking={contract.get('world_action_ranking')!r}"
        )
    return supervision, logged_branch, valid


def _task_bootstrap_contract(report: Mapping[str, Any]) -> bool:
    bootstrap = report.get("bootstrap")
    return bool(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("sampling_unit") == "episode"
        and bootstrap.get("episode_weighting") == "equal"
        and bootstrap.get("paired") is True
        and isinstance(bootstrap.get("n_resamples"), int)
        and not isinstance(bootstrap.get("n_resamples"), bool)
        and bootstrap["n_resamples"] >= 2
    )


def _macro_bootstrap_contract(report: Mapping[str, Any]) -> bool:
    bootstrap = report.get("bootstrap")
    return bool(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("sampling_unit") == "episode_within_task"
        and bootstrap.get("task_weighting") == "equal"
        and bootstrap.get("paired") is True
        and isinstance(bootstrap.get("n_resamples"), int)
        and not isinstance(bootstrap.get("n_resamples"), bool)
        and bootstrap["n_resamples"] >= 2
    )


def evaluate_go_no_go(
    per_task: Mapping[str, Mapping[str, Any]],
    task_macro: Mapping[str, Any],
    *,
    full_heldout_evaluation: bool = True,
    checkpoint_world_supervision_valid: bool = True,
    checkpoint_world_logged_branch_valid: bool = True,
    min_top10_gain: float = 0.10,
    max_static_ratio: float = 1.05,
    min_action_degradation: float = 0.05,
    strong_action_degradation: float = 0.10,
) -> dict[str, Any]:
    """Evaluate the visual/action Go/No-Go gate without pooled masking.

    Every task must pass every task-local condition.  The task-macro top-10
    condition is checked in addition, not used to rescue a failed task.
    """

    if not isinstance(full_heldout_evaluation, bool):
        raise TypeError("full_heldout_evaluation must be bool")
    if not isinstance(checkpoint_world_supervision_valid, bool):
        raise TypeError("checkpoint_world_supervision_valid must be bool")
    if not isinstance(checkpoint_world_logged_branch_valid, bool):
        raise TypeError("checkpoint_world_logged_branch_valid must be bool")
    if not per_task:
        raise ValueError("gate needs at least one task")
    required_task_ids = set(DEFAULT_TASK_IDS)
    task_ids = [report.get("task_id") for report in per_task.values()]
    task_identity_matches = bool(
        {
            str(task_name): report.get("task_id")
            for task_name, report in per_task.items()
        }
        == {name: task_id for task_id, name in DEFAULT_TASK_NAMES.items()}
    )
    macro_task_count = task_macro.get("n_tasks")
    macro_task_ids = task_macro.get("task_ids")
    task_count_matches = bool(
        len(per_task) == len(DEFAULT_TASK_IDS)
        and all(
            isinstance(task_id, int) and not isinstance(task_id, bool)
            for task_id in task_ids
        )
        and set(task_ids) == required_task_ids
        and isinstance(macro_task_count, int)
        and not isinstance(macro_task_count, bool)
        and macro_task_count == len(DEFAULT_TASK_IDS)
        and isinstance(macro_task_ids, Sequence)
        and not isinstance(macro_task_ids, (str, bytes))
        and set(macro_task_ids) == required_task_ids
    )
    task_checks: dict[str, dict[str, bool]] = {}
    for task_name, report in per_task.items():
        metrics = report.get("metrics", {})
        action = report.get("action_dependency", {})
        world_all = _estimate(metrics.get("world_all", {}))
        copy_all = _estimate(metrics.get("copy_all", {}))
        rel_top10 = _estimate(metrics.get("relative_gain_top10", {}))
        world_static = _estimate(metrics.get("world_static", {}))
        copy_static = _estimate(metrics.get("copy_static", {}))
        shuffle_rel = _estimate(
            action.get("shuffle_relative_degradation_top10", {})
        )
        zero_rel = _estimate(action.get("zero_relative_degradation_top10", {}))
        shuffle_low = _low(action.get("shuffle_minus_real_top10", {}))
        zero_low = _low(action.get("zero_minus_real_top10", {}))
        shuffle_confidence = _confidence(
            action.get("shuffle_minus_real_top10", {})
        )
        zero_confidence = _confidence(action.get("zero_minus_real_top10", {}))
        n_episodes = report.get("n_episodes")
        episode_bootstrap_valid = bool(
            isinstance(n_episodes, int)
            and not isinstance(n_episodes, bool)
            and n_episodes >= 2
        )
        bootstrap_contract_valid = _task_bootstrap_contract(report)
        target_ok = _target_permutation_exact(
            report.get("target_permutation", {})
        )

        finite = all(
            value is not None and math.isfinite(value)
            for value in (
                world_all,
                copy_all,
                rel_top10,
                world_static,
                copy_static,
                shuffle_rel,
                zero_rel,
                shuffle_low,
                zero_low,
                shuffle_confidence,
                zero_confidence,
            )
        )
        action_ci_is_95pct = bool(
            finite
            and math.isclose(  # type: ignore[arg-type]
                shuffle_confidence, CI_LEVEL, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(  # type: ignore[arg-type]
                zero_confidence, CI_LEVEL, rel_tol=0.0, abs_tol=1e-12
            )
        )
        checks = {
            "finite_metrics": finite,
            "episode_bootstrap_has_multiple_episodes": episode_bootstrap_valid,
            "episode_bootstrap_contract_valid": bootstrap_contract_valid,
            "pred_all_le_copy_all": bool(
                finite and world_all <= copy_all  # type: ignore[operator]
            ),
            "relative_gain_top10_ge_10pct": bool(
                finite and rel_top10 >= min_top10_gain  # type: ignore[operator]
            ),
            "static_le_105pct_copy": bool(
                finite
                and world_static <= max_static_ratio * copy_static  # type: ignore[operator]
            ),
            "shuffle_top10_ge_5pct_worse": bool(
                finite and shuffle_rel >= min_action_degradation  # type: ignore[operator]
            ),
            "zero_top10_ge_5pct_worse": bool(
                finite and zero_rel >= min_action_degradation  # type: ignore[operator]
            ),
            "action_difference_ci_is_95pct": action_ci_is_95pct,
            "shuffle_ci_direction_positive": bool(
                action_ci_is_95pct and shuffle_low > 0.0  # type: ignore[operator]
            ),
            "zero_ci_direction_positive": bool(
                action_ci_is_95pct and zero_low > 0.0  # type: ignore[operator]
            ),
            "one_action_ablation_ge_10pct_worse": bool(
                finite
                and max(shuffle_rel, zero_rel) >= strong_action_degradation  # type: ignore[arg-type]
            ),
            "target_permutation_bitwise_invariant": target_ok,
        }
        checks["passed"] = all(checks.values())
        task_checks[task_name] = checks

    macro_gain_interval = task_macro.get("relative_gain_top10", {})
    macro_gain = _estimate(macro_gain_interval)
    macro_definition_valid = bool(
        macro_gain_interval.get("aggregation")
        == "equal_task_macro_of_task_relative_differences"
    )
    macro_bootstrap_valid = _macro_bootstrap_contract(task_macro)
    macro_check = bool(
        macro_definition_valid
        and macro_bootstrap_valid
        and macro_gain is not None
        and math.isfinite(macro_gain)
        and macro_gain >= min_top10_gain
    )
    passed = bool(
        full_heldout_evaluation
        and checkpoint_world_supervision_valid
        and checkpoint_world_logged_branch_valid
        and task_identity_matches
        and task_count_matches
        and macro_check
        and all(row["passed"] for row in task_checks.values())
    )
    return {
        "decision": "GO" if passed else "NO-GO",
        "passed": passed,
        "required_task_count": len(DEFAULT_TASK_IDS),
        "required_task_ids": list(DEFAULT_TASK_IDS),
        "full_heldout_evaluation": full_heldout_evaluation,
        "checkpoint_world_supervision_valid": checkpoint_world_supervision_valid,
        "checkpoint_world_logged_branch_valid": checkpoint_world_logged_branch_valid,
        "task_identity_matches": task_identity_matches,
        "task_count_matches": task_count_matches,
        "macro_relative_gain_is_task_macro": macro_definition_valid,
        "macro_bootstrap_contract_valid": macro_bootstrap_valid,
        "task_macro_relative_gain_top10_ge_10pct": macro_check,
        "per_task": task_checks,
        "thresholds": {
            "min_relative_gain_top10": float(min_top10_gain),
            "max_static_copy_ratio": float(max_static_ratio),
            "min_action_relative_degradation": float(min_action_degradation),
            "one_action_min_relative_degradation": float(strong_action_degradation),
            "action_difference_ci_low_must_exceed": 0.0,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**32)


def _validate_fixed_eval_payload(
    payload: Mapping[str, Any], expected_task_ids: Sequence[int]
) -> dict[str, Any]:
    from scripts.split_wam4va_episode_holdout import (
        MANIFEST_CONTRACT,
        canonical_manifest_sha256,
        mask_stats,
    )

    required = (
        "actions",
        "previous_action",
        "proprio",
        "language_hidden",
        "instruction_id",
        "episode_id",
        "action_valid_mask",
        "recovery_mask",
        "frame_refs",
        "metadata",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"fixed eval dataset missing keys: {missing}")
    actions = torch.as_tensor(payload["actions"])
    if actions.ndim != 4 or tuple(actions.shape[2:]) != (48, 4):
        raise ValueError(f"eval actions must be [N,T,48,4], got {tuple(actions.shape)}")
    mask = torch.as_tensor(payload["action_valid_mask"])
    if mask.dtype != torch.bool or mask.shape != actions.shape[:-1]:
        raise ValueError("action_valid_mask must be bool [N,T,48]")
    episodes = _as_1d_tensor(payload["episode_id"], name="episode_id").to(torch.int64)
    task_ids = _as_1d_tensor(payload["instruction_id"], name="instruction_id").to(
        torch.int64
    )
    if episodes.shape[0] != actions.shape[0] or task_ids.shape[0] != actions.shape[0]:
        raise ValueError("episode/task ids must align with eval windows")
    actual_tasks = sorted(int(value) for value in torch.unique(task_ids).tolist())
    expected = sorted(int(value) for value in expected_task_ids)
    if actual_tasks != expected:
        raise ValueError(f"eval task ids {actual_tasks} != expected {expected}")

    metadata = dict(payload.get("metadata") or {})
    contract = dict(metadata.get("split_contract") or {})
    if metadata.get("split_name") != "eval":
        raise ValueError("dataset metadata must declare split_name='eval'")
    actual_episodes = sorted(int(value) for value in torch.unique(episodes).tolist())
    declared_episodes = metadata.get("split_episode_ids")
    if declared_episodes is None or sorted(int(value) for value in declared_episodes) != actual_episodes:
        raise ValueError("payload episodes do not match metadata.split_episode_ids")
    for field in ("n_subset_windows", "split_windows"):
        if int(metadata.get(field, -1)) != actions.shape[0]:
            raise ValueError(f"metadata.{field} does not match payload")
    task_counts = {
        int(task_id): int((task_ids == task_id).sum().item())
        for task_id in torch.unique(task_ids).tolist()
    }
    declared_counts = {
        int(task_id): int(count)
        for task_id, count in dict(metadata.get("split_task_counts") or {}).items()
    }
    if task_counts != declared_counts:
        raise ValueError("payload tasks do not match metadata.split_task_counts")

    manifest_id = contract.get("manifest_id")
    manifest_sha = contract.get("manifest_sha256")
    if contract.get("contract") != MANIFEST_CONTRACT:
        raise ValueError("fixed eval split has the wrong manifest contract")
    manifest_tasks = {
        int(item["task_id"]): str(item.get("task_name"))
        for item in contract.get("tasks", [])
    }
    if manifest_tasks != DEFAULT_TASK_NAMES:
        raise ValueError(
            "fixed eval split must contain assembly-v3 + door-unlock-v3"
        )
    if not manifest_id or not manifest_sha:
        raise ValueError("fixed eval split is missing manifest identity")
    if metadata.get("split_manifest_id") != manifest_id:
        raise ValueError("metadata/embedded manifest_id mismatch")
    if metadata.get("split_manifest_sha256") != manifest_sha:
        raise ValueError("metadata/embedded manifest_sha256 mismatch")
    if metadata.get("split_manifest_path") != contract.get("manifest_path"):
        raise ValueError("metadata/embedded manifest_path mismatch")
    actual_manifest_sha = canonical_manifest_sha256(contract)
    if actual_manifest_sha != manifest_sha:
        raise ValueError(
            "embedded split manifest canonical SHA mismatch: "
            f"declared={manifest_sha}, actual={actual_manifest_sha}"
        )

    splits = contract.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("fixed eval split requires embedded train/eval manifest")
    eval_contract = dict(splits.get("eval") or {})
    contract_eval_episodes = eval_contract.get("episode_ids")
    if contract_eval_episodes is None or sorted(
        int(value) for value in contract_eval_episodes
    ) != actual_episodes:
        raise ValueError("payload episodes do not match split_contract.splits.eval")
    if int(eval_contract.get("windows", -1)) != actions.shape[0]:
        raise ValueError("payload windows do not match split_contract.splits.eval")
    actual_mask_stats = mask_stats(
        dict(payload), torch.arange(actions.shape[0], dtype=torch.int64)
    )
    if actual_mask_stats != eval_contract.get("mask_stats"):
        raise ValueError("payload masks do not match split_contract.splits.eval")
    manifest_task_counts = {
        int(item["task_id"]): int(item["windows"])
        for item in eval_contract.get("tasks", [])
    }
    if manifest_task_counts != task_counts:
        raise ValueError("payload tasks do not match split_contract.splits.eval.tasks")
    for item in eval_contract.get("tasks", []):
        task_id = int(item["task_id"])
        task_episode_ids = sorted(
            int(value)
            for value in torch.unique(episodes[task_ids == task_id]).tolist()
        )
        if task_episode_ids != sorted(int(value) for value in item["episode_ids"]):
            raise ValueError(
                f"payload task {task_id} episodes do not match split manifest"
            )
    train_contract = dict(splits.get("train") or {})
    train_episodes = {int(value) for value in train_contract.get("episode_ids", [])}
    overlap = train_episodes.intersection(actual_episodes)
    if overlap:
        raise ValueError(f"train/eval episode overlap in split contract: {sorted(overlap)}")
    source = contract.get("source")
    if not isinstance(source, Mapping) or not source.get("sha256"):
        raise ValueError("split contract must contain source.sha256")
    if int(metadata.get("source_n_windows", -1)) != int(source.get("n_windows", -2)):
        raise ValueError("metadata/source manifest window count mismatch")
    transition = dict(contract.get("transition_rule") or {})
    if (
        transition.get("current_action_prefix_steps") != DEFAULT_CYCLE_STEPS
        or transition.get("next_action_index") != 0
    ):
        raise ValueError("split manifest transition rule does not match cycle-6 protocol")
    validation = dict(contract.get("validation") or {})
    for check in ("episode_single_task", "episode_disjoint", "rows_disjoint", "rows_exhaustive"):
        if validation.get(check) is not True:
            raise ValueError(f"split manifest validation.{check} is not true")

    return {
        "n_windows": int(actions.shape[0]),
        "episode_ids": actual_episodes,
        "task_ids": actual_tasks,
        "split_name": "eval",
        "manifest_id": manifest_id,
        "manifest_sha256": manifest_sha,
        "manifest_path": metadata.get("split_manifest_path"),
        "source_path": source.get("path"),
        "source_sha256": source.get("sha256"),
    }


def _task_names(payload: Mapping[str, Any]) -> dict[int, str]:
    task_ids = torch.as_tensor(payload["instruction_id"], dtype=torch.int64)
    refs = payload["frame_refs"]
    names: dict[int, str] = {}
    for index, task_id in enumerate(task_ids.tolist()):
        name = str(refs[index][0])
        existing = names.setdefault(int(task_id), name)
        if existing != name:
            raise ValueError(f"task id {task_id} maps to multiple frame-ref names")
    if names != DEFAULT_TASK_NAMES:
        raise ValueError(
            f"eval task mapping {names} != required {DEFAULT_TASK_NAMES}"
        )
    return names


def _extract_transition_records(payload: Mapping[str, Any], cycle_steps: int) -> dict[str, Tensor]:
    mask = transition_mask(
        torch.as_tensor(payload["action_valid_mask"]), cycle_steps=cycle_steps
    )
    rows, times = torch.nonzero(mask, as_tuple=True)
    if rows.numel() == 0:
        raise ValueError("fixed eval dataset has no valid World transitions")
    actions = torch.as_tensor(payload["actions"])[rows, times, :cycle_steps]
    if tuple(actions.shape[1:]) != (6, 4):
        raise ValueError("World action prefix must be exactly [6,4]")
    return {
        "row": rows.to(torch.int64),
        "time": times.to(torch.int64),
        "action": actions,
        "proprio": torch.as_tensor(payload["proprio"])[rows, times],
        "task_id": torch.as_tensor(payload["instruction_id"])[rows].to(torch.int64),
        "episode_id": torch.as_tensor(payload["episode_id"])[rows].to(torch.int64),
    }


def _load_model_and_metric(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[VACompoundPolicy, Any, Any, VACompoundConfig]:
    config = VACompoundConfig(**checkpoint["config"])
    if not config.wmrm or config.wmrm_target != "dino":
        raise ValueError("checkpoint must contain a DINO-target WAM4VA model")
    if config.wmrm_cycle_steps != DEFAULT_CYCLE_STEPS:
        raise ValueError("held-out protocol requires wmrm_cycle_steps=6")
    if (
        config.main_vision_grid != 16
        or config.main_vision_frames != 4
        or config.main_vision_dim != 1024
    ):
        raise ValueError("held-out protocol requires native 4x16x16x1024 DINO input")
    formal_gate_config = {
        "num_layers": 8,
        "action_horizon": 48,
        "wmrm_inject": "all",
        "wmrm_handshake": True,
        "wmrm_predictor": "st_blocks",
        "wmrm_map_size": 16,
        "wmrm_map_channels": 1024,
        "wmrm_world_grid": 16,
    }
    mismatches = {
        name: (getattr(config, name), expected)
        for name, expected in formal_gate_config.items()
        if getattr(config, name) != expected
    }
    if mismatches:
        raise ValueError(
            f"checkpoint does not match the formal held-out gate architecture: {mismatches}"
        )
    model = VACompoundPolicy(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    metric_head = relation_encoder = None
    if config.dino_dense_metric:
        from train import _build_dino_metric_stack

        metric_head, relation_encoder = _build_dino_metric_stack(
            device,
            config,
            train_metric_head=False,
            train_relation=False,
            saved_ctor_config=checkpoint.get("mtvj_metric_head_config"),
        )
        if "mtvj_metric_head" not in checkpoint or "mtvj_relation_encoder" not in checkpoint:
            raise ValueError("DINO-metric checkpoint is missing metric stack weights")
        metric_head.load_state_dict(checkpoint["mtvj_metric_head"], strict=True)
        relation_encoder.load_state_dict(checkpoint["mtvj_relation_encoder"], strict=True)
        metric_head.eval()
        relation_encoder.eval()
    return model, metric_head, relation_encoder, config


def _world_forward(
    model: VACompoundPolicy,
    vision: Tensor,
    proprio: Tensor,
    previous_action: Tensor,
    language_cache: Any,
    dense_evidence: dict[int, Tensor] | None,
    metric_tokens: Tensor | None,
    metric_g: Tensor | None,
    env_action: Tensor,
    *,
    visual_memory: VisualMemory | None = None,
) -> Tensor:
    model.encode_condition(
        vision,
        proprio,
        previous_action,
        language_cache=language_cache,
        dense_evidence=dense_evidence,
        metric_tokens=metric_tokens,
        metric_g=metric_g,
        visual_memory=visual_memory,
        env_action=env_action,
        detach_wmrm_stage_state=True,
    )
    auxes = list(getattr(model, "last_wmrm_auxes", ()) or ())
    if not auxes:
        raise RuntimeError("WAM4VA forward emitted no World predictions")
    maps = []
    for stage, aux in enumerate(auxes):
        prediction = aux.z_tokens
        if prediction is None or prediction.ndim != 4:
            raise RuntimeError(f"World stage {stage} did not emit a spatial DINO map")
        maps.append(prediction)
    return torch.stack(maps, dim=0)


def _index_visual_memory(
    memory: VisualMemory | None,
    indices: Tensor,
) -> VisualMemory | None:
    if memory is None:
        return None
    if indices.ndim != 1:
        raise ValueError("visual-memory indices must be 1-D")

    def select(value: Tensor | None) -> Tensor | None:
        if value is None:
            return None
        return value.index_select(0, indices.to(device=value.device))

    return VisualMemory(
        layers=tuple(select(layer) for layer in memory.layers),
        evidence=select(memory.evidence),
        task=select(memory.task),
        task_spec=select(memory.task_spec),
        pending_future=select(memory.pending_future),
        gate=memory.gate,
    )


def _proposal_noise(
    model: VACompoundPolicy,
    rows: Tensor,
    history_time: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    row_values = rows.detach().cpu().to(torch.int64).tolist()
    samples = []
    for row in row_values:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(seed, f"proposal-row-{row}-time-{history_time}")
        )
        samples.append(
            torch.randn(
                model.config.action_horizon,
                model.config.action_dim,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(samples).to(device=device, dtype=dtype)


def _proposal_pre_step_memory(
    model: VACompoundPolicy,
    vision_tokens: Tensor,
    proprio: Tensor,
    previous_action: Tensor,
    language_cache: Any,
    dense_evidence: dict[int, Tensor] | None,
    metric_tokens: Tensor | None,
    metric_g: Tensor | None,
    rows: Tensor,
    time_index: int,
    *,
    seed: int,
    flow_steps: int = PROPOSAL_FLOW_STEPS,
) -> VisualMemory | None:
    """Replay proposal handshake history and return current pre-step memory."""

    if time_index < 0 or time_index >= vision_tokens.shape[1]:
        raise ValueError("time_index is outside the encoded sequence")
    if rows.ndim != 1 or rows.shape[0] != vision_tokens.shape[0]:
        raise ValueError("rows must align with the sequence batch")
    if flow_steps < 1:
        raise ValueError("flow_steps must be positive")

    visual_memory = None
    for history_time in range(time_index):
        history_dense = (
            None
            if dense_evidence is None
            else {key: value[:, history_time] for key, value in dense_evidence.items()}
        )
        history_metric = (
            None if metric_tokens is None else metric_tokens[:, history_time]
        )
        history_metric_g = None if metric_g is None else metric_g[:, history_time]
        proposal_condition, _ = model.encode_condition(
            vision_tokens[:, history_time],
            proprio[:, history_time],
            previous_action[:, history_time],
            language_cache=language_cache,
            visual_memory=visual_memory,
            return_visual_memory=True,
            skip_wmrm=True,
            dense_evidence=history_dense,
            metric_tokens=history_metric,
            metric_g=history_metric_g,
        )
        noise = _proposal_noise(
            model,
            rows,
            history_time,
            seed,
            device=proposal_condition.device,
            dtype=proposal_condition.dtype,
        )
        proposal_action = model.decode_actions(
            proposal_condition,
            steps=flow_steps,
            noise=noise,
        )
        if proposal_action.shape[1] < DEFAULT_CYCLE_STEPS:
            raise RuntimeError(
                "proposal action horizon is shorter than the cycle-6 handshake"
            )
        proposal_action = proposal_action[:, :DEFAULT_CYCLE_STEPS].clamp(-1.0, 1.0)
        _, visual_memory = model.encode_condition(
            vision_tokens[:, history_time],
            proprio[:, history_time],
            previous_action[:, history_time],
            language_cache=language_cache,
            visual_memory=visual_memory,
            return_visual_memory=True,
            dense_evidence=history_dense,
            metric_tokens=history_metric,
            metric_g=history_metric_g,
            env_action=proposal_action,
        )

    return None if visual_memory is None else visual_memory.detach()


def _time_task_batches(records: Mapping[str, Tensor], batch_size: int) -> list[Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches = []
    task_ids = records["task_id"]
    times = records["time"]
    for time_index in torch.unique(times, sorted=True).tolist():
        time_mask = times == int(time_index)
        for task_id in torch.unique(task_ids[time_mask], sorted=True).tolist():
            indices = torch.nonzero(
                time_mask & (task_ids == int(task_id)), as_tuple=False
            ).flatten()
            for start in range(0, indices.numel(), batch_size):
                batches.append(indices[start : start + batch_size])
    return batches


def _append_loss_metrics(
    collector: dict[str, list[Tensor]],
    prefix: str,
    stages: Tensor,
    target: Tensor,
    current: Tensor,
) -> Any:
    final = None
    for stage_index, prediction in enumerate(stages):
        result = visual_world_loss(prediction, target, current)
        stage_prefix = f"stage{stage_index}_{prefix}"
        for name in ("loss", "all", "motion", "top10", "static"):
            source = "loss_per_sample" if name == "loss" else f"{name}_per_sample"
            collector[f"{stage_prefix}_{name}"].append(
                getattr(result, source).detach().cpu()
            )
        final = result
    assert final is not None
    for name in ("loss", "all", "motion", "top10", "static"):
        source = "loss_per_sample" if name == "loss" else f"{name}_per_sample"
        collector[f"{prefix}_{name}"].append(getattr(final, source).detach().cpu())
    return final


def _episode_point(values: Tensor, episodes: Tensor) -> float:
    means, _ = _episode_means(values, episodes)
    return float(means.mean())


def _summarize_task(
    values: Mapping[str, Tensor],
    episodes: Tensor,
    task_ids: Tensor,
    *,
    n_resamples: int,
    seed: int,
    stage_count: int,
) -> dict[str, Any]:
    def pair(candidate: str, reference: str, label: str) -> dict[str, Any]:
        return paired_episode_bootstrap(
            values[candidate],
            values[reference],
            episodes,
            task_ids=task_ids,
            n_resamples=n_resamples,
            seed=_stable_seed(seed, label),
        )

    all_pair = pair("real_all", "copy_all", "all-copy")
    motion_pair = pair("real_motion", "copy_motion", "motion-copy")
    top10_pair = pair("real_top10", "copy_top10", "top10-copy")
    static_pair = pair("real_static", "copy_static", "static-copy")
    shuffle_all = pair("shuffle_all", "real_all", "shuffle-all")
    zero_all = pair("zero_all", "real_all", "zero-all")
    shuffle_top10 = pair("shuffle_top10", "real_top10", "shuffle-top10")
    zero_top10 = pair("zero_top10", "real_top10", "zero-top10")

    stages = []
    for stage in range(stage_count):
        row: dict[str, Any] = {"stage": stage}
        for mode in ("real", "shuffle", "zero"):
            for metric in ("loss", "all", "motion", "top10", "static"):
                key = f"stage{stage}_{mode}_{metric}"
                row[f"{mode}_{metric}"] = _episode_point(values[key], episodes)
        stages.append(row)

    return {
        "valid_transitions": int(episodes.numel()),
        "episodes": sorted(int(value) for value in torch.unique(episodes).tolist()),
        "n_episodes": int(torch.unique(episodes).numel()),
        "motion_energy": _episode_point(values["motion_energy"], episodes),
        "metrics": {
            "world_all": all_pair["candidate"],
            "copy_all": all_pair["reference"],
            "gain_all": _negate_ci(all_pair["difference"]),
            "world_motion": motion_pair["candidate"],
            "copy_motion": motion_pair["reference"],
            "gain_motion": _negate_ci(motion_pair["difference"]),
            "world_top10": top10_pair["candidate"],
            "copy_top10": top10_pair["reference"],
            "gain_top10": _negate_ci(top10_pair["difference"]),
            "relative_gain_top10": _negate_ci(top10_pair["relative_difference"]),
            "world_static": static_pair["candidate"],
            "copy_static": static_pair["reference"],
        },
        "action_dependency": {
            "real_all": all_pair["candidate"],
            "shuffle_all": shuffle_all["candidate"],
            "zero_all": zero_all["candidate"],
            "shuffle_minus_real_all": shuffle_all["difference"],
            "zero_minus_real_all": zero_all["difference"],
            "real_top10": top10_pair["candidate"],
            "shuffle_top10": shuffle_top10["candidate"],
            "zero_top10": zero_top10["candidate"],
            "shuffle_minus_real_top10": shuffle_top10["difference"],
            "zero_minus_real_top10": zero_top10["difference"],
            "shuffle_relative_degradation_top10": shuffle_top10[
                "relative_difference"
            ],
            "zero_relative_degradation_top10": zero_top10[
                "relative_difference"
            ],
        },
        "stages": stages,
        "bootstrap": {
            "sampling_unit": "episode",
            "episode_weighting": "equal",
            "paired": True,
            "n_resamples": int(n_resamples),
        },
    }


def _build_task_macro(
    values: Mapping[str, Tensor],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    episodes = values["episode_id"]
    tasks = values["task_id"]

    def pair(candidate: str, reference: str, label: str) -> dict[str, Any]:
        return task_macro_paired_episode_bootstrap(
            values[candidate],
            values[reference],
            episodes,
            tasks,
            n_resamples=n_resamples,
            seed=_stable_seed(seed, f"macro-{label}"),
        )

    all_pair = pair("real_all", "copy_all", "all-copy")
    motion_pair = pair("real_motion", "copy_motion", "motion-copy")
    top10_pair = pair("real_top10", "copy_top10", "top10-copy")
    static_pair = pair("real_static", "copy_static", "static-copy")
    shuffle_all = pair("shuffle_all", "real_all", "shuffle-all")
    zero_all = pair("zero_all", "real_all", "zero-all")
    shuffle_top10 = pair("shuffle_top10", "real_top10", "shuffle-top10")
    zero_top10 = pair("zero_top10", "real_top10", "zero-top10")
    return {
        "n_tasks": int(torch.unique(tasks).numel()),
        "task_ids": sorted(int(value) for value in torch.unique(tasks).tolist()),
        "aggregation": "equal_task_macro_of_task_local_episode_bootstrap",
        "metrics": {
            "world_all": all_pair["candidate"],
            "copy_all": all_pair["reference"],
            "gain_all": _negate_ci(all_pair["difference"]),
            "world_motion": motion_pair["candidate"],
            "copy_motion": motion_pair["reference"],
            "gain_motion": _negate_ci(motion_pair["difference"]),
            "world_top10": top10_pair["candidate"],
            "copy_top10": top10_pair["reference"],
            "gain_top10": _negate_ci(top10_pair["difference"]),
            "relative_gain_top10": _negate_ci(top10_pair["relative_difference"]),
            "world_static": static_pair["candidate"],
            "copy_static": static_pair["reference"],
        },
        "action_dependency": {
            "real_all": all_pair["candidate"],
            "shuffle_all": shuffle_all["candidate"],
            "zero_all": zero_all["candidate"],
            "shuffle_minus_real_all": shuffle_all["difference"],
            "zero_minus_real_all": zero_all["difference"],
            "real_top10": top10_pair["candidate"],
            "shuffle_top10": shuffle_top10["candidate"],
            "zero_top10": zero_top10["candidate"],
            "shuffle_minus_real_top10": shuffle_top10["difference"],
            "zero_minus_real_top10": zero_top10["difference"],
            "shuffle_relative_degradation_top10": shuffle_top10[
                "relative_difference"
            ],
            "zero_relative_degradation_top10": zero_top10[
                "relative_difference"
            ],
        },
        "bootstrap": {
            "sampling_unit": "episode_within_task",
            "task_weighting": "equal",
            "paired": True,
            "n_resamples": int(n_resamples),
        },
    }


def _parse_task_ids(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("task ids must be a non-empty unique CSV")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--main-vision-checkpoint", type=Path)
    parser.add_argument("--dino-feature-cache", type=Path)
    parser.add_argument("--longtraj-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--task-ids", type=_parse_task_ids, default=DEFAULT_TASK_IDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encode-batch", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-transitions",
        type=int,
        help="optional deterministic cap per task (diagnostic smoke only)",
    )
    parser.add_argument(
        "--allow-unmarked-checkpoint",
        action="store_true",
        help="diagnostic migration only; default rejects old weighted-MSE checkpoints",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2 for target permutation checks")
    if args.encode_batch < 1 or args.bootstrap_resamples < 1:
        raise ValueError("encode/bootstrap batch counts must be positive")
    if args.max_transitions is not None and args.max_transitions < 2:
        raise ValueError("max-transitions must be >=2")
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    eval_path = args.eval_data.expanduser().resolve(strict=True)
    output_path = args.output_json.expanduser().absolute()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    marker, logged_branch, checkpoint_world_contract_valid = (
        _checkpoint_world_contract(
            checkpoint, allow_unmarked=args.allow_unmarked_checkpoint
        )
    )
    payload = torch.load(eval_path, map_location="cpu", weights_only=True)
    split_summary = _validate_fixed_eval_payload(payload, args.task_ids)
    manifest_path = Path(str(split_summary["manifest_path"])).expanduser().resolve(
        strict=True
    )
    external_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if external_manifest != payload["metadata"]["split_contract"]:
        raise ValueError("external split manifest differs from embedded split_contract")
    source_path = Path(str(split_summary["source_path"])).expanduser().resolve(strict=True)
    actual_source_sha = _sha256_file(source_path)
    if actual_source_sha != split_summary["source_sha256"]:
        raise ValueError(
            "fixed split source SHA mismatch: "
            f"declared={split_summary['source_sha256']}, actual={actual_source_sha}"
        )
    checkpoint_split_sha = (checkpoint.get("training_contract") or {}).get(
        "split_manifest_sha256"
    )
    if not checkpoint_split_sha:
        raise ValueError(
            "checkpoint training_contract is missing split_manifest_sha256"
        )
    if checkpoint_split_sha != split_summary["manifest_sha256"]:
        raise ValueError(
            "checkpoint/eval split manifest mismatch: "
            f"checkpoint={checkpoint_split_sha}, eval={split_summary['manifest_sha256']}"
        )
    names = _task_names(payload)
    records = _extract_transition_records(payload, DEFAULT_CYCLE_STEPS)

    # Deterministic donor selection uses all valid eval transitions before any
    # optional diagnostic truncation, so --max-transitions cannot alter donors.
    shuffled_actions, donor_indices = nearest_episode_shuffle(
        records["action"],
        records["proprio"],
        records["task_id"],
        records["episode_id"],
    )
    records["shuffle_action"] = shuffled_actions
    records["donor_index"] = donor_indices.cpu()
    full_order = sorted(
        range(records["row"].numel()),
        key=lambda index: (
            int(records["task_id"][index]),
            int(records["episode_id"][index]),
            int(records["row"][index]),
            int(records["time"][index]),
        ),
    )
    if args.max_transitions is not None:
        order = []
        for task_id in args.task_ids:
            task_order = [
                index
                for index in full_order
                if int(records["task_id"][index]) == int(task_id)
            ]
            order.extend(task_order[: args.max_transitions])
    else:
        order = full_order
    record_order = torch.as_tensor(order, dtype=torch.int64)
    records = {key: value.index_select(0, record_order) for key, value in records.items()}

    device = torch.device(args.device)
    model, metric_head, relation_encoder, config = _load_model_and_metric(
        checkpoint, device
    )
    from train import (
        DinoFeatureCache,
        _build_dino_main_backbone,
        _dino_main_encode_from_cache,
        _dino_main_online_encode,
        _dino_metric_tokens,
    )
    from va_compound.longtraj_frames import LongTrajFramesDataset, mtvj_collate

    dino_cache = None
    main_backbone = None
    if args.dino_feature_cache is not None:
        dino_cache = DinoFeatureCache(args.dino_feature_cache.expanduser().resolve(strict=True))
    else:
        if args.main_vision_checkpoint is None:
            raise ValueError(
                "online held-out DINO encoding requires --main-vision-checkpoint"
            )
        # Preserve the .safetensors symlink suffix.  Resolving the Hugging Face
        # cache link exposes an extensionless blob that timm 0.9 misclassifies
        # as a torch pickle checkpoint.
        main_path = args.main_vision_checkpoint.expanduser().absolute()
        expected_sha = (checkpoint.get("training_contract") or {}).get(
            "main_vision_checkpoint_sha256"
        )
        actual_sha = _sha256_file(main_path)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                "main vision checkpoint SHA mismatch: "
                f"checkpoint={expected_sha}, runtime={actual_sha}"
            )
        main_backbone = _build_dino_main_backbone(
            SimpleNamespace(main_vision_checkpoint=main_path), config, device
        )

    dataset = LongTrajFramesDataset(
        eval_path,
        longtraj_dir=args.longtraj_dir,
        min_sequence_length=4,
        decode_cache_tasks=1,
        feature_cache=args.dino_feature_cache,
        include_frames=dino_cache is None,
    )
    collector: dict[str, list[Tensor]] = defaultdict(list)
    target_checks: dict[int, dict[str, Any]] = {}
    stage_count: int | None = None
    processed = 0
    evaluation_batches = _time_task_batches(records, args.batch_size)

    with torch.inference_mode():
        for record_indices_cpu in evaluation_batches:
            rows = records["row"].index_select(0, record_indices_cpu)
            times = records["time"].index_select(0, record_indices_cpu)
            time_index = int(times[0])
            if not bool((times == time_index).all()):
                raise RuntimeError("proposal-history batches must share current time")
            items = [dataset[int(row)] for row in rows.tolist()]
            batch = mtvj_collate(items)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and key not in {"frame_cache_rows"}:
                    batch[key] = value.to(device)

            if dino_cache is not None:
                vision_tokens, dense_all = _dino_main_encode_from_cache(
                    batch["frame_cache_rows"],
                    dino_cache,
                    device,
                    grid=config.main_vision_grid,
                    window=config.main_vision_frames,
                    return_dense=True,
                )
            else:
                vision_tokens, dense_all = _dino_main_online_encode(
                    batch["frames"],
                    main_backbone,
                    device,
                    encode_batch=args.encode_batch,
                    grid=config.main_vision_grid,
                    window=config.main_vision_frames,
                    return_dense=True,
                )
            metric_all, metric_g_all = _dino_metric_tokens(
                metric_head,
                relation_encoder,
                dense_all,
                batch,
                device,
                train_metric_head=False,
            )
            model_dtype = model.vision_projection.weight.dtype
            vision_tokens = vision_tokens.to(dtype=model_dtype)
            dense_all = {
                key: value.to(dtype=model_dtype) for key, value in dense_all.items()
            }
            if metric_all is not None:
                metric_all = metric_all.to(dtype=model_dtype)
            if metric_g_all is not None:
                metric_g_all = metric_g_all.to(dtype=model_dtype)

            current_tokens = vision_tokens[:, time_index]
            target_tokens = vision_tokens[:, time_index + 1]
            current = model.wmrm.encode_dino_map(current_tokens)
            target = model.wmrm.encode_dino_map(target_tokens)
            if current is None or target is None:
                raise RuntimeError("failed to recover native DINO current/target maps")
            dense = {key: value[:, time_index] for key, value in dense_all.items()}
            metric = None if metric_all is None else metric_all[:, time_index]
            metric_g = None if metric_g_all is None else metric_g_all[:, time_index]
            language_cache = model.build_language_cache(
                batch["language_hidden"], batch.get("language_mask")
            )
            pre_step_memory = _proposal_pre_step_memory(
                model,
                vision_tokens,
                batch["proprio"],
                batch["previous_action"],
                language_cache,
                dense_all,
                metric_all,
                metric_g_all,
                rows,
                time_index,
                seed=args.seed,
            )
            real_action = records["action"].index_select(
                0, record_indices_cpu
            ).to(device)
            shuffle_action = records["shuffle_action"].index_select(
                0, record_indices_cpu
            ).to(device)
            zero_action = torch.zeros_like(real_action)
            current_proprio = batch["proprio"][:, time_index]
            current_previous_action = batch["previous_action"][:, time_index]

            forward_args = (
                model,
                current_tokens,
                current_proprio,
                current_previous_action,
                language_cache,
                dense,
                metric,
                metric_g,
            )
            real_stages = _world_forward(
                *forward_args, real_action, visual_memory=pre_step_memory
            )
            shuffle_stages = _world_forward(
                *forward_args, shuffle_action, visual_memory=pre_step_memory
            )
            zero_stages = _world_forward(
                *forward_args, zero_action, visual_memory=pre_step_memory
            )
            if stage_count is None:
                stage_count = int(real_stages.shape[0])
            if not (
                real_stages.shape == shuffle_stages.shape == zero_stages.shape
                and real_stages.shape[0] == stage_count
            ):
                raise RuntimeError("World stage shape/count changed across action conditions")

            real_result = _append_loss_metrics(
                collector, "real", real_stages, target, current
            )
            shuffle_result = _append_loss_metrics(
                collector, "shuffle", shuffle_stages, target, current
            )
            zero_result = _append_loss_metrics(
                collector, "zero", zero_stages, target, current
            )
            for result in (shuffle_result, zero_result):
                if not (
                    torch.equal(result.motion_weights, real_result.motion_weights)
                    and torch.equal(result.top10_mask, real_result.top10_mask)
                    and torch.equal(result.static_mask, real_result.static_mask)
                ):
                    raise RuntimeError("oracle reductions changed across action conditions")
            for name in ("all", "motion", "top10", "static"):
                collector[f"copy_{name}"].append(
                    getattr(real_result, f"copy_{name}_per_sample").detach().cpu()
                )
            collector["motion_energy"].append(
                real_result.motion_energy_per_sample.detach().cpu()
            )
            collector["task_id"].append(
                records["task_id"].index_select(0, record_indices_cpu).cpu()
            )
            collector["episode_id"].append(
                records["episode_id"].index_select(0, record_indices_cpu).cpu()
            )

            # Run one empirical target-permutation check per task.  The closure
            # invokes World first, then uses the supplied target only in loss.
            batch_tasks = records["task_id"].index_select(0, record_indices_cpu)
            for task_id in torch.unique(batch_tasks, sorted=True).tolist():
                if task_id in target_checks:
                    continue
                indices_cpu = torch.nonzero(
                    batch_tasks == task_id, as_tuple=False
                ).flatten()
                if indices_cpu.numel() < 2:
                    continue
                indices = indices_cpu.to(device)
                local_target = target.index_select(0, indices)
                local_current = current.index_select(0, indices)
                local_memory = _index_visual_memory(pre_step_memory, indices)
                local_language_cache = model.build_language_cache(
                    batch["language_hidden"].index_select(0, indices),
                    (
                        None
                        if batch.get("language_mask") is None
                        else batch["language_mask"].index_select(0, indices)
                    ),
                )

                def forward_with_target(
                    loss_target: Tensor,
                ) -> tuple[Tensor, Tensor]:
                    maps = _world_forward(
                        model,
                        current_tokens.index_select(0, indices),
                        current_proprio.index_select(0, indices),
                        current_previous_action.index_select(0, indices),
                        local_language_cache,
                        {key: value.index_select(0, indices) for key, value in dense.items()},
                        None if metric is None else metric.index_select(0, indices),
                        None if metric_g is None else metric_g.index_select(0, indices),
                        real_action.index_select(0, indices),
                        visual_memory=local_memory,
                    )
                    loss = visual_world_loss(
                        maps[-1], loss_target, local_current
                    ).loss_per_sample
                    return maps, loss

                check = check_target_permutation_invariance(
                    forward_with_target, local_target
                )
                if check["target_changed"]:
                    target_checks[int(task_id)] = check
            processed += int(record_indices_cpu.numel())
            print(
                f"world held-out: {processed}/{records['row'].numel()} transitions",
                flush=True,
            )

    if stage_count is None:
        raise RuntimeError("evaluation emitted no World stages")
    missing_target_checks = sorted(set(args.task_ids) - set(target_checks))
    for task_id in missing_target_checks:
        target_checks[int(task_id)] = {
            "passed": False,
            "target_changed": False,
            "prediction_bitwise_equal": False,
            "max_abs_prediction_difference": None,
            "reason": "no batch contained two targets for this task",
        }

    joined = {key: torch.cat(parts, dim=0) for key, parts in collector.items()}
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in args.task_ids:
        task_mask = joined["task_id"] == int(task_id)
        if not bool(task_mask.any()):
            raise ValueError(f"no evaluated transition for task {task_id}")
        task_values = {
            key: value[task_mask]
            for key, value in joined.items()
            if key not in {"task_id", "episode_id"}
        }
        task_report = _summarize_task(
            task_values,
            joined["episode_id"][task_mask],
            joined["task_id"][task_mask],
            n_resamples=args.bootstrap_resamples,
            seed=_stable_seed(args.seed, f"task-{task_id}"),
            stage_count=stage_count,
        )
        task_report["task_id"] = int(task_id)
        task_report["task_name"] = names[int(task_id)]
        task_report["target_permutation"] = target_checks[int(task_id)]
        per_task[names[int(task_id)]] = task_report

    task_macro_full = _build_task_macro(
        joined,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    gate_macro = {
        "relative_gain_top10": task_macro_full["metrics"]["relative_gain_top10"],
        "n_tasks": task_macro_full["n_tasks"],
        "task_ids": task_macro_full["task_ids"],
        "bootstrap": task_macro_full["bootstrap"],
    }
    full_heldout_evaluation = args.max_transitions is None
    gate = evaluate_go_no_go(
        per_task,
        gate_macro,
        full_heldout_evaluation=full_heldout_evaluation,
        checkpoint_world_supervision_valid=checkpoint_world_contract_valid,
        checkpoint_world_logged_branch_valid=(
            logged_branch == WORLD_LOGGED_BRANCH_CONTRACT
        ),
    )
    report = {
        "contract": "wam4va_world_action_heldout_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "global_step": checkpoint.get("global_step"),
            "world_supervision": marker,
            "world_logged_branch": logged_branch,
            "world_contract_valid": checkpoint_world_contract_valid,
        },
        "eval_dataset": {
            "path": str(eval_path),
            "sha256": _sha256_file(eval_path),
            **split_summary,
        },
        "protocol": {
            "cycle_steps": DEFAULT_CYCLE_STEPS,
            "action_shape": [6, 4],
            "world_logged_branch": WORLD_LOGGED_BRANCH_CONTRACT,
            "proposal_flow_steps": PROPOSAL_FLOW_STEPS,
            "proposal_history": "replay_decisions_0_through_t_minus_1",
            "proposal_noise": "cpu_standard_normal_seeded_by_eval_seed_row_history_time",
            "condition_memory": "shared_detached_proposal_pre_step_visual_memory",
            "transition_mask": "current first 6 all valid and next first action valid",
            "shuffle": "same_task_different_episode_proprio_nearest",
            "bootstrap": "paired_within_task_equal_episode_weight",
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "world_stage_count": stage_count,
            "target_enters_predictor": False,
            "full_heldout_evaluation": full_heldout_evaluation,
            "max_transitions_per_task": args.max_transitions,
        },
        "per_task": per_task,
        "task_macro": task_macro_full,
        "gate": gate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"{gate['decision']}: report written to {output_path}", flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
