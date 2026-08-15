from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.migrate_task35_longtraj_schema import migrate


def _payload(*, perturbed: bool) -> dict:
    event = {"start": 3, "end": 5} if perturbed else None
    return {
        "task": "peg-insert-side-v3",
        "n_episodes": 1,
        "episodes": [
            {
                "frames": [b"jpeg"],
                "actions": np.zeros((8, 4), dtype=np.float32),
                "states": np.zeros((8, 4), dtype=np.float32),
                "perturbed": perturbed,
                "perturb_event": event,
                "perturb_start": 3 if perturbed else None,
                "recovery_mask": np.asarray(
                    [False, False, False, True, True, False, False, False]
                    if perturbed
                    else [False] * 8
                ),
            }
        ],
        "metadata": {"contract": "long_trajectory_scripted_v2"},
    }


def test_clean_schema_migration_is_metadata_only(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clean.pt"
    source.write_bytes(b"source")
    before = _payload(perturbed=False)
    frames = before["episodes"][0]["frames"]
    actions = before["episodes"][0]["actions"]
    migrated = migrate(before, layer="clean", source=source)
    assert migrated["episodes"][0]["n_perturb_events"] == 0
    assert migrated["metadata"]["perturbation_data_present"] is False
    assert migrated["episodes"][0]["frames"] is frames
    assert migrated["episodes"][0]["actions"] is actions


def test_recovery_schema_migration_requires_actual_event(tmp_path: Path) -> None:
    source = tmp_path / "recovery.pt"
    source.write_bytes(b"source")
    migrated = migrate(_payload(perturbed=True), layer="recovery", source=source)
    assert migrated["episodes"][0]["n_perturb_events"] == 1
    assert migrated["metadata"]["perturbation_data_present"] is True
    with pytest.raises(ValueError, match="violates recovery purity"):
        migrate(copy.deepcopy(_payload(perturbed=False)), layer="recovery", source=source)


def test_clean_schema_rejects_recovery_episode(tmp_path: Path) -> None:
    source = tmp_path / "clean.pt"
    source.write_bytes(b"source")
    with pytest.raises(ValueError, match="violates clean purity"):
        migrate(_payload(perturbed=True), layer="clean", source=source)
