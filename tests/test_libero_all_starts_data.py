"""Tests for compact all-starts H15 data preparation and validation."""
from __future__ import annotations

import copy
import pytest
import torch

import prepare_libero as prepare
from tests.test_libero_episode_data import synthetic_hdf5_data
from va_compound.data.libero import (
    ALL_STARTS_DATA_CONTRACT,
    EXECUTION_HORIZON,
    _validate_data,
)


def test_prepare_and_validate_all_starts_data(synthetic_hdf5_data, tmp_path, monkeypatch):
    data_dir, frames_dir = synthetic_hdf5_data
    monkeypatch.setattr(
        prepare,
        "_official_task_specs",
        lambda *args: [
            {"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "task 3"}
        ],
    )
    monkeypatch.setattr(prepare, "_ensure_longtraj", lambda *args, **kwargs: None)

    out_path = tmp_path / "all_starts_payload.pt"
    args = prepare._parser().parse_args([
        "--data", str(out_path),
        "--hdf5-dir", str(data_dir.parent),
        "--longtraj", str(frames_dir),
        "--suites", "libero_10",
        "--local-task-ids", "3",
        "--dense-windows",
        "--architecture-version", "dual_tower_h15_v1",
        "--window-sampling", "all_starts_random_tbptt8_v1",
    ])
    prepare.prepare_data(args)

    payload = _validate_data(out_path, architecture_version="dual_tower_h15_v1")
    assert payload["metadata"]["contract"] == ALL_STARTS_DATA_CONTRACT
    assert payload["metadata"]["sampling_contract"] == "all_starts_random_tbptt8_v1"
    assert payload["metadata"]["storage_sequence_length"] == 1
    assert payload["metadata"]["sequence_length"] == 8
    assert payload["metadata"]["memory_contract"] == "offset_replay_tbptt8_v1"
    assert len(payload["metadata"]["episode_lengths"]) == 50

    N = len(payload["actions"])
    expected_total = sum(L - EXECUTION_HORIZON for L in payload["metadata"]["episode_lengths"])
    assert N == expected_total

    assert payload["actions"].shape == (N, 1, 15, 7)
    assert payload["proprio"].shape == (N, 1, 9)
    assert payload["previous_action"].shape == (N, 1, 7)
    assert payload["world_state_delta"].shape == (N, 1, 9)
    assert payload["action_valid_mask"].shape == (N, 1, 15)
    assert payload["world_target_valid_mask"].shape == (N, 1)
    assert payload["action_valid_mask"].all()
    assert payload["world_target_valid_mask"].all()
    assert payload["world_rank_shuffle_action"].shape == (N, 1, 15, 7)
    assert payload["world_rank_shuffle_mask"].shape == (N, 1)

    from va_compound.vision.longtraj_frames import LongTrajFramesDataset
    from va_compound.data.all_starts import AllStartsStreamDataset, AllStartsWindowBatchSampler
    import numpy as np
    base = LongTrajFramesDataset(out_path, longtraj_dir=frames_dir, include_world_target_frames=True)
    frame_count = 2 * max(payload["metadata"]["episode_lengths"])
    fake_frames = np.arange(frame_count, dtype=np.uint16)[:, None, None, None]
    monkeypatch.setattr(base, "_decode_task", lambda _: {i: fake_frames for i in range(50)})
    dataset = AllStartsStreamDataset(base)
    sampler = AllStartsWindowBatchSampler(payload, 4, 42, 1, 0, 2)
    index = next(iter(sampler))[0]
    item = dataset[index]
    rows = list(index[0])
    assert item["frames"].shape[:2] == (8, 5)
    assert item["world_target_frames"].shape[:2] == (8, 1)
    for position, row in enumerate(rows):
        expected_current = payload["frame_refs"][row][2][0]
        assert item["frames"][position, :, 0, 0, 0].tolist() == expected_current
        assert item["world_target_frames"][position, 0, 0, 0, 0] == payload["world_target_frame_refs"][row][2][0][0]

    # Adjacent row continuity: previous_action[d+1, 0] == actions[d, 0, 0]
    ep0_rows = torch.where(payload["episode_id"] == 0)[0]
    assert torch.equal(payload["previous_action"][ep0_rows[1:], 0], payload["actions"][ep0_rows[:-1], 0, 0])

    # Validation errors on corruptions
    with pytest.raises(ValueError, match="unexpected data contract"):
        _validate_data(out_path, architecture_version="dual_tower_expert_v1")

    # Strict window_sampling kw match
    _validate_data(
        out_path,
        architecture_version="dual_tower_h15_v1",
        window_sampling="all_starts_random_tbptt8_v1",
    )
    with pytest.raises(ValueError, match="window_sampling mismatch"):
        _validate_data(
            out_path,
            architecture_version="dual_tower_h15_v1",
            window_sampling="episode_contiguous_p15_v1",
        )

    # Test _validate_run_schedule with all-starts contract
    from va_compound.data.libero import _validate_run_schedule
    import argparse
    train_args = argparse.Namespace(
        architecture_version="dual_tower_h15_v1",
        window_sampling="all_starts_random_tbptt8_v1",
        batch_size=8,
        mixed_tasks=1,
        stage1_steps=100,
        epochs=3,
        gpus=1,
        anchor_fraction=0.0,
        max_steps=None,
        seed=42,
    )
    first_len, total_len, grouping = _validate_run_schedule(payload, train_args)
    assert first_len > 0
    assert total_len >= first_len
    assert grouping == "single_task_t8_local1_deferred_v1"

    broken = copy.deepcopy(payload)
    broken["crop_start"][1] += 1
    torch.save(broken, tmp_path / "broken_crop.pt")
    with pytest.raises(ValueError, match="crop_starts must be contiguous"):
        _validate_data(tmp_path / "broken_crop.pt", architecture_version="dual_tower_h15_v1")

    broken_prev = copy.deepcopy(payload)
    broken_prev["previous_action"][1, 0, 0] += 0.5
    torch.save(broken_prev, tmp_path / "broken_prev.pt")
    with pytest.raises(ValueError, match="previous_action continuity"):
        _validate_data(tmp_path / "broken_prev.pt", architecture_version="dual_tower_h15_v1")

    # Inner action target overlap broken (actions[d, 0, 1:] != actions[d+1, 0, :-1])
    broken_act = copy.deepcopy(payload)
    broken_act["actions"][0, 0, 5, 0] += 0.1
    torch.save(broken_act, tmp_path / "broken_act.pt")
    with pytest.raises(ValueError, match="inner action target continuity"):
        _validate_data(tmp_path / "broken_act.pt", architecture_version="dual_tower_h15_v1")

    # Wrong wrist base even if future target matches same shifted base
    broken_base = copy.deepcopy(payload)
    for row in range(payload["metadata"]["episode_lengths"][0] - EXECUTION_HORIZON):
        broken_base["frame_refs"][row][2][0][4] += 10
        broken_base["world_target_frame_refs"][row][2][0][0] += 10
    torch.save(broken_base, tmp_path / "broken_base.pt")
    with pytest.raises(ValueError, match="wrist base does not match demo length"):
        _validate_data(tmp_path / "broken_base.pt", architecture_version="dual_tower_h15_v1")

    # Corrupt state normalization
    broken_norm = copy.deepcopy(payload)
    broken_norm["normalization"]["state_q01"] = torch.tensor([float("nan")] * 9)
    torch.save(broken_norm, tmp_path / "broken_norm.pt")
    with pytest.raises(ValueError, match="finite state_q01"):
        _validate_data(tmp_path / "broken_norm.pt", architecture_version="dual_tower_h15_v1")
