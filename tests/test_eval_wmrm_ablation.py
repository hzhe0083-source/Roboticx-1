from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eval_metaworld import wmrm_ablation_provenance, wmrm_ablation_writes


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


def test_analyzer_rejects_action_chunk_shape_mismatch(tmp_path: Path) -> None:
    baseline = _write_payload(tmp_path, "normal", _payload("normal"))
    candidate = _payload(
        "action-off",
        trials=[
            {"task_id": 0, "seed": 0, "success": False, "action_chunks": [[[0.0, 0.0]]]},
        ],
    )
    with pytest.raises(ValueError, match="action chunk shape mismatch"):
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
