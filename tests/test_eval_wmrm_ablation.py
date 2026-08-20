from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from eval_metaworld import (
    Plan,
    SynchronousPlanQueue,
    resolve_execution_horizon,
    wmrm_ablation_provenance,
    wmrm_ablation_writes,
)


_ANALYZER_PATH = Path(__file__).parents[1] / "scripts" / "analyze_wmrm_ablation.py"
_SPEC = importlib.util.spec_from_file_location("wmrm_ablation_analyzer", _ANALYZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ANALYZER)


@pytest.mark.parametrize(
    ("mode", "writes"),
    [
        ("normal", (True, True)),
        ("action-write-off", (False, True)),
        ("vision-write-off", (True, False)),
        ("both-write-off", (False, False)),
        ("proposal-only", (False, False)),
    ],
)
def test_wmrm_ablation_modes(mode: str, writes: tuple[bool, bool]) -> None:
    assert wmrm_ablation_writes(mode) == writes
    provenance = wmrm_ablation_provenance(mode)
    assert provenance["wmrm_ablation_mode"] == mode
    assert provenance["wmrm_action_write_enabled"] is writes[0]
    assert provenance["wmrm_vision_write_enabled"] is writes[1]
    assert provenance["wmrm_proposal_only"] is (mode == "proposal-only")


def test_wmrm_ablation_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown WMRM"):
        wmrm_ablation_writes("invalid")


def test_synchronous_plan_queue_uses_absolute_time_prefixes() -> None:
    actions = np.arange(24, dtype=np.float32).reshape(6, 4)
    for horizon in (1, 2, 3, 6):
        queue = SynchronousPlanQueue(horizon)
        plan = queue.replace(10, actions)
        assert plan.start_step == 10
        assert plan.stop_step == 10 + horizon
        for step in range(10, 10 + horizon):
            np.testing.assert_array_equal(queue.action_at(step), actions[step - 10])
        assert not queue.needs_plan(10 + horizon - 1)
        assert queue.needs_plan(10 + horizon)


def test_synchronous_plan_queue_replaces_expired_plan_without_modulo_replay() -> None:
    queue = SynchronousPlanQueue(2)
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.full((3, 4), 99, dtype=np.float32)
    queue.replace(0, first)
    queue.replace(2, second)
    np.testing.assert_array_equal(queue.action_at(2), second[0])
    with pytest.raises(IndexError):
        queue.action_at(1)


def test_execute_six_queue_matches_legacy_action_trace() -> None:
    queue = SynchronousPlanQueue(6)
    chunks = {
        step: np.arange(48 * 4, dtype=np.float32).reshape(48, 4) + step * 1000
        for step in (0, 6, 12)
    }
    active = None
    chunk_start = 0
    for step in range(13):
        if active is None or step - chunk_start >= 6:
            active = chunks[step]
            chunk_start = step
            queue.replace(step, active)
        legacy = active[(step - chunk_start) % active.shape[0]]
        np.testing.assert_array_equal(queue.action_at(step), legacy)


def test_plan_queue_preserves_full_decoded_chunk_for_telemetry() -> None:
    decoded = np.arange(48 * 4, dtype=np.float32).reshape(48, 4)
    queue = SynchronousPlanQueue(6)
    plan = queue.replace(0, decoded)
    assert plan.actions.shape == (6, 4)
    recorded = decoded.astype(float).tolist()
    assert len(recorded) == 48


@pytest.mark.parametrize("value", (1, 2, 3, 6))
def test_resolve_execution_horizon(value: int) -> None:
    assert resolve_execution_horizon(Namespace(execution_horizon=value, execute_steps=None)) == value


def test_resolve_execution_horizon_defaults_and_rejects_mismatch() -> None:
    assert resolve_execution_horizon(Namespace(execution_horizon=None, execute_steps=None)) == 6
    with pytest.raises(ValueError, match="disagree"):
        resolve_execution_horizon(Namespace(execution_horizon=2, execute_steps=3))


def _payload(mode: str, *, trials: list[dict] | None = None, **overrides) -> dict:
    action_write, vision_write = {
        "normal": (True, True),
        "action-off": (False, True),
        "vision-off": (True, False),
        "both-off": (False, False),
        "proposal-only": (False, False),
    }[mode]
    payload = {
        "contract": "metaworld_closed_loop_trials_v1",
        "checkpoint_sha256": "checkpoint",
        "task_ids": [0],
        "trials_per_task": 1,
        "execute_steps": 6,
        "prediction_horizon": 48,
        "execution_horizon": 6,
        "world_horizon": 6,
        "flow_solver_steps": 8,
        "control_hz": 80,
        "training_control_stride": 6,
        "control_stride": 6,
        "observation_stride": 2,
        "memory_reset_every": 0,
        "wmrm_mode": {
            "normal": "normal",
            "action-off": "action-write-off",
            "vision-off": "vision-write-off",
            "both-off": "both-write-off",
            "proposal-only": "proposal-only",
        }[mode],
        "horizon": 60,
        "wmrm_ablation_mode": {
            "normal": "normal",
            "action-off": "action-write-off",
            "vision-off": "vision-write-off",
            "both-off": "both-write-off",
            "proposal-only": "proposal-only",
        }[mode],
        "wmrm_action_write_enabled": action_write,
        "wmrm_vision_write_enabled": vision_write,
        "wmrm_proposal_only": mode == "proposal-only",
        "trials": trials or [
            {
                "task_id": 0,
                "seed": 0,
                "success": False,
                "action_chunks": [[[0.0, 0.0], [0.0, 0.0]]],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _write_payload(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_analyzer_keeps_training_and_execution_horizons_independent(
    tmp_path: Path,
) -> None:
    baseline = _write_payload(
        tmp_path,
        "normal",
        _payload("normal", execute_steps=2, execution_horizon=2),
    )
    candidate = _write_payload(
        tmp_path,
        "action-off",
        _payload("action-off", execute_steps=2, execution_horizon=2),
    )
    result = _ANALYZER.analyze([baseline, candidate])
    assert result["execution_horizon"] == 2
    assert result["world_horizon"] == 6
    assert result["training_control_stride"] == 6
    assert result["prediction_horizon"] == 48


def test_analyzer_rejects_execution_alias_mismatch(tmp_path: Path) -> None:
    path = _write_payload(
        tmp_path,
        "normal",
        _payload("normal", execute_steps=6, execution_horizon=2),
    )
    with pytest.raises(ValueError, match="execution horizon alias mismatch"):
        _ANALYZER.analyze([path])


def test_analyzer_rejects_training_stride_alias_mismatch(tmp_path: Path) -> None:
    path = _write_payload(
        tmp_path,
        "normal",
        _payload("normal", training_control_stride=6, control_stride=2),
    )
    with pytest.raises(ValueError, match="training control stride alias mismatch"):
        _ANALYZER.analyze([path])


def test_analyzer_rejects_wmrm_mode_mismatch(tmp_path: Path) -> None:
    baseline = _write_payload(tmp_path, "normal", _payload("normal"))
    candidate = _write_payload(
        tmp_path,
        "action-off",
        _payload("action-off", wmrm_mode="normal"),
    )
    with pytest.raises(ValueError, match="mode provenance mismatch"):
        _ANALYZER.analyze([baseline, candidate])


def test_analyzer_rejects_duplicate_candidate_trial_keys(tmp_path: Path) -> None:
    baseline = _write_payload(tmp_path, "normal", _payload("normal"))
    candidate = _payload(
        "action-off",
        trials=[
            {"task_id": 0, "seed": 0, "success": False, "action_chunks": [[[0.0, 0.0], [0.0, 0.0]]]},
            {"task_id": 0, "seed": 0, "success": True, "action_chunks": [[[1.0, 1.0], [1.0, 1.0]]]},
        ],
    )
    with pytest.raises(ValueError, match="duplicate task/seed"):
        _ANALYZER.analyze([baseline, _write_payload(tmp_path, "action-off", candidate)])


def test_analyzer_truncates_action_chunk_length_mismatch(tmp_path: Path) -> None:
    baseline = _write_payload(tmp_path, "normal", _payload("normal"))
    candidate = _payload(
        "action-off",
        trials=[
            {"task_id": 0, "seed": 0, "success": False, "action_chunks": [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]},
        ],
    )
    result = _ANALYZER.analyze(
        [baseline, _write_payload(tmp_path, "action-off", candidate)]
    )
    divergence = result["modes"][1]["action_divergence"]
    assert divergence["paired_trials"] == 1
    assert divergence["truncated_trials"] == 1
    assert divergence["chunk_l1_mean"] == 0.0


def test_analyzer_rejects_action_chunk_width_mismatch(tmp_path: Path) -> None:
    baseline = _write_payload(tmp_path, "normal", _payload("normal"))
    candidate = _payload(
        "action-off",
        trials=[
            {"task_id": 0, "seed": 0, "success": False, "action_chunks": [[[0.0], [0.0]]]},
        ],
    )
    with pytest.raises(ValueError, match="per-decision shape mismatch"):
        _ANALYZER.analyze([baseline, _write_payload(tmp_path, "action-off", candidate)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("wmrm_action_write_enabled", 1),
        ("wmrm_vision_write_enabled", False),
        ("wmrm_proposal_only", True),
    ],
)
def test_analyzer_rejects_mismatched_provenance(
    tmp_path: Path, field: str, value: object
) -> None:
    baseline = _write_payload(tmp_path, "normal", _payload("normal"))
    candidate = _payload("action-off", **{field: value})
    with pytest.raises(ValueError, match="provenance mismatch"):
        _ANALYZER.analyze([baseline, _write_payload(tmp_path, "action-off", candidate)])
