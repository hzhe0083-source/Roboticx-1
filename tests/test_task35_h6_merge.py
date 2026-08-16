from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.merge_task35_h6_windows import merge


def _layer(n: int, recovery: bool) -> dict:
    valid = torch.ones(n, 4, 6, dtype=torch.bool)
    recovery_mask = torch.zeros_like(valid)
    decision = torch.zeros(n, 4, dtype=torch.bool)
    if recovery:
        recovery_mask[:, :, 2:] = True
        decision[:, 1:] = True
    instruction = torch.full((n,), 35, dtype=torch.long)
    return {
        "actions": torch.randn(n, 4, 6, 4),
        "previous_action": torch.randn(n, 4, 4),
        "proprio": torch.randn(n, 4, 4),
        "instruction_id": instruction,
        "episode_id": torch.arange(n),
        "pair_id": torch.arange(n),
        "action_valid_mask": valid,
        "recovery_mask": recovery_mask,
        "decision_recovery": decision,
        "language_hidden": torch.randn(n, 3, 8),
        "language_mask": torch.ones(n, 3, dtype=torch.bool),
        "frame_refs": [("source", i, [[0, 0, 0, 0]] * 4) for i in range(n)],
        "normalization": {"action_q01": torch.zeros(4)},
        "metadata": {"action_horizon": 6, "tasks": ["Insert a peg sideways"]},
    }


def test_merge_preserves_h6_and_offsets_ids(tmp_path: Path) -> None:
    clean = _layer(2, False)
    recovery = _layer(3, True)
    clean_path = tmp_path / "clean.pt"
    recovery_path = tmp_path / "recovery.pt"
    clean_path.write_bytes(b"clean")
    recovery_path.write_bytes(b"recovery")
    output = merge(clean, recovery, clean_path, recovery_path)
    assert output["actions"].shape == (5, 4, 6, 4)
    assert output["data_layer"].tolist() == [0, 0, 1, 1, 1]
    assert len(torch.unique(output["episode_id"])) == 5
    assert len(torch.unique(output["pair_id"])) == 5
    assert output["metadata"]["roi_relation_pair"] == ["pegHead", "hole"]


def test_merge_rejects_non_h6_or_impure_clean(tmp_path: Path) -> None:
    clean = _layer(1, False)
    recovery = _layer(1, True)
    clean_path = tmp_path / "clean.pt"
    recovery_path = tmp_path / "recovery.pt"
    clean_path.write_bytes(b"clean")
    recovery_path.write_bytes(b"recovery")
    clean["actions"] = torch.randn(1, 4, 48, 4)
    with pytest.raises(ValueError, match="exact H6"):
        merge(clean, recovery, clean_path, recovery_path)
