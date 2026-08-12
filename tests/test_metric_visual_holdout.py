from __future__ import annotations

import numpy as np
import pytest

from scripts.eval_metric_visual_holdout import (
    CONTRACT,
    compute_gated_state_metrics,
    compute_localization_metrics,
    compute_pair_visibility_coverage,
    compute_visibility_metrics,
    validate_checkpoint_contract,
)


def _valid_checkpoint() -> dict:
    return {
        "contract": CONTRACT,
        "config": {
            "h_dim": 768,
            "d_proj": 192,
            "n_roles": 4,
            "lang_dim": 2048,
            "image_size": 384,
            "tasks": ["door-lock-v3"],
            "language_cache_available": True,
        },
        "metric_head": {"weight": np.zeros(1)},
        "relation_encoder": {},
    }


def test_checkpoint_contract_accepts_matching_door_lock_head() -> None:
    checkpoint = _valid_checkpoint()
    assert validate_checkpoint_contract(checkpoint, "door-lock-v3") is checkpoint["config"]


def test_checkpoint_contract_rejects_unseen_task() -> None:
    checkpoint = _valid_checkpoint()
    checkpoint["config"]["tasks"] = ["assembly-v3"]
    with pytest.raises(ValueError, match="absent from checkpoint training tasks"):
        validate_checkpoint_contract(checkpoint, "door-lock-v3")


def test_visible_metric_aggregation_is_per_point_not_per_role() -> None:
    target = np.zeros((3, 4, 2), dtype=np.float32)
    prediction = target.copy()
    # Pixel errors: tool=[0, 3, 4], object=[5]. Other roles are invisible.
    prediction[1, 0, 1] = 3.0 / 384.0
    prediction[2, 0, 1] = 4.0 / 384.0
    prediction[0, 1, 1] = 5.0 / 384.0
    visibility = np.zeros((3, 4), dtype=np.float32)
    visibility[:, 0] = 1.0
    visibility[0, 1] = 1.0

    metrics = compute_localization_metrics(prediction, target, visibility)

    assert metrics["roles"]["tool"]["visible_count"] == 3
    assert metrics["roles"]["tool"]["rmse_px"] == pytest.approx(np.sqrt(25.0 / 3.0))
    assert metrics["roles"]["object"]["rmse_px"] == pytest.approx(5.0)
    assert metrics["roles"]["target"]["rmse_px"] is None
    aggregate = metrics["aggregate"]
    assert aggregate["visible_count"] == 4
    assert aggregate["rmse_px"] == pytest.approx(np.sqrt(50.0 / 4.0))
    assert aggregate["median_px"] == pytest.approx(3.5)


def test_pck_thresholds_are_inclusive() -> None:
    target = np.zeros((3, 4, 2), dtype=np.float64)
    prediction = target.copy()
    prediction[:, 0, 1] = np.asarray([4.0, 5.0, 10.0]) / 384.0
    visibility = np.zeros((3, 4), dtype=bool)
    visibility[:, 0] = True

    tool = compute_localization_metrics(prediction, target, visibility)["roles"]["tool"]

    assert tool["pck@5px"] == pytest.approx(2.0 / 3.0)
    assert tool["pck@10px"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("prediction_shape", "target_shape", "visibility_shape", "message"),
    [
        ((2, 4), (2, 4), (2, 4), "predictions must have shape"),
        ((2, 4, 2), (2, 3, 2), (2, 4), "targets shape"),
        ((2, 4, 2), (2, 4, 2), (2, 3), "visibility must have shape"),
    ],
)
def test_metric_shape_contract(
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    visibility_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_localization_metrics(
            np.zeros(prediction_shape),
            np.zeros(target_shape),
            np.zeros(visibility_shape),
        )


def test_nonfinite_predictions_fail_fast() -> None:
    prediction = np.zeros((1, 4, 2), dtype=np.float32)
    prediction[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        compute_localization_metrics(
            prediction,
            np.zeros_like(prediction),
            np.ones((1, 4), dtype=bool),
        )


def test_visibility_balanced_accuracy_exposes_hidden_class_failure() -> None:
    targets = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.float32)
    probabilities = np.full_like(targets, 0.9)

    aggregate = compute_visibility_metrics(probabilities, targets)["aggregate"]

    assert aggregate["accuracy"] == pytest.approx(0.5)
    assert aggregate["visible_recall"] == pytest.approx(1.0)
    assert aggregate["hidden_recall"] == pytest.approx(0.0)
    assert aggregate["balanced_accuracy"] == pytest.approx(0.5)


def test_gated_state_metric_matches_policy_input_contract() -> None:
    target = np.zeros((1, 4, 2), dtype=np.float32)
    target[0, 0] = [0.5, 0.5]
    prediction = target.copy()
    true_visibility = np.zeros((1, 4), dtype=np.float32)
    true_visibility[0, 0] = 1.0
    predicted_visibility = true_visibility.copy()

    perfect = compute_gated_state_metrics(
        prediction, predicted_visibility, target, true_visibility
    )["aggregate"]
    assert perfect["rmse_px"] == pytest.approx(0.0)

    predicted_visibility[0, 0] = 0.0
    missed = compute_gated_state_metrics(
        prediction, predicted_visibility, target, true_visibility
    )["roles"]["tool"]
    assert missed["rmse_px"] == pytest.approx(np.sqrt(0.5) * 384.0)


def test_actionable_pair_visibility_coverage() -> None:
    # rows: tool-object only, interface-target only, neither, both
    visibility = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    coverage = compute_pair_visibility_coverage(visibility)

    assert coverage["tool_object"] == pytest.approx(0.5)
    assert coverage["interface_target"] == pytest.approx(0.5)
    assert coverage["any_actionable_pair"] == pytest.approx(0.75)
