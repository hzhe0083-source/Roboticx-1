from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import build_longtraj_features as build


def _episode(*, n: int = 90, success_frame: int = 45,
             perturbed: bool = False,
             zero_blocks: tuple[tuple[int, int], ...] = ()) -> dict:
    actions = np.full((n, 4), 0.25, dtype=np.float32)
    for start, end in zero_blocks:
        actions[start:end] = 0.0
    return {
        "frames": [b"jpeg-not-decoded-in-phase1"] * n,
        "actions": actions,
        "states": np.zeros((n, 4), dtype=np.float32),
        "success_frame": success_frame,
        "perturbed": perturbed,
    }


def test_infer_perturbed_v1_repairs_timeline_and_masks() -> None:
    ep = _episode(perturbed=True, zero_blocks=((20, 32),))
    semantics = build.resolve_episode_semantics(ep, "legacy:episode[0]", "infer")

    assert semantics["first_success"] == 57  # legacy 45 + stored 12-step block
    assert semantics["perturb_start"] == 20
    assert semantics["perturb_end"] == 32
    assert semantics["legacy_inferred"] is True
    np.testing.assert_array_equal(
        np.flatnonzero(~semantics["valid"][:58]), np.arange(20, 32)
    )
    assert not semantics["valid"][58:].any()
    assert semantics["recovery"][20:58].all()
    assert not semantics["recovery"][:20].any()
    assert not semantics["recovery"][58:].any()


def test_infer_nonperturbed_v1_keeps_success_timeline() -> None:
    ep = _episode(perturbed=False)
    semantics = build.resolve_episode_semantics(ep, "legacy:episode[1]", "infer")

    assert semantics["first_success"] == 45
    assert semantics["perturb_start"] is None
    assert semantics["valid"][:46].all()
    assert not semantics["valid"][46:].any()
    assert not semantics["recovery"].any()


def test_infer_rejects_partially_annotated_legacy_episode() -> None:
    ep = _episode(perturbed=True, zero_blocks=((20, 32),))
    ep["settle_mask"] = np.zeros(len(ep["actions"]), dtype=bool)
    with pytest.raises(ValueError, match="partial timeline annotations"):
        build.resolve_episode_semantics(ep, "partial", "infer")


@pytest.mark.parametrize(
    "blocks,match",
    [
        ((), "exactly one"),
        (((8, 20), (28, 40)), "exactly one"),
        (((8, 21),), "exactly one"),  # 13 zeros is not the v1 event contract
    ],
)
def test_infer_perturbed_v1_fails_on_ambiguous_or_missing_block(
    blocks: tuple[tuple[int, int], ...], match: str,
) -> None:
    ep = _episode(perturbed=True, zero_blocks=blocks)
    with pytest.raises(ValueError, match=match):
        build.resolve_episode_semantics(ep, "ambiguous", "infer")


def test_infer_v1_masks_unseen_recovery_targets_in_h48_phase1() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        task = "reach-v3"  # inference is task-agnostic
        task_text = build.ENV_TO_TASK[task]
        ref = root / "ref.pt"
        source = root / f"metaworld_longtraj_{task}.pt"
        output = root / "all_task_repaired_h48.pt"
        norm = {
            "action_q01": torch.full((4,), -1.0),
            "action_q99": torch.full((4,), 1.0),
            "state_q01": torch.full((4,), -1.0),
            "state_q99": torch.full((4,), 1.0),
        }
        torch.save({
            "normalization": norm,
            "metadata": {"tasks": [task_text]},
            "instruction_id": torch.tensor([0]),
            "language_hidden": torch.zeros(1, 2, 3),
            "language_mask": torch.ones(1, 2, dtype=torch.bool),
        }, ref)
        torch.save({
            "task": task,
            "episodes": [_episode(
                n=100, success_frame=55, perturbed=True,
                zero_blocks=((30, 42),),
            )],
        }, source)

        build.phase1(
            48,
            input_paths=[source],
            output_path=output,
            ref_path=ref,
            legacy_policy="infer",
        )
        payload = torch.load(output, map_location="cpu", weights_only=True)
        mask = payload["action_valid_mask"]
        # Window 0 decision 18 cannot know the random event at action 30.
        assert not bool(mask[0, 3, 12])
        # Window 5 starts at the now-visible perturb; recovery action 43 is valid.
        assert bool(mask[5, 0, 13])
        assert payload["first_success"].unique().item() == 67
        assert payload["metadata"]["legacy_episodes_inferred"] == 1
        assert payload["metadata"]["legacy_perturb_events_inferred"] == 1
