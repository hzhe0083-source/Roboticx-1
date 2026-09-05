"""Tests for LIBERO joint episode data preparation, validation, and dataset loading."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import prepare_libero as prepare
from va_compound.data.libero import (
    EXECUTION_HORIZON,
    JOINT_DATA_CONTRACT,
    SEQUENCE,
    _validate_data,
)
from va_compound.vision.longtraj_frames import LongTrajFramesDataset


@pytest.fixture
def synthetic_hdf5_data(tmp_path):
    """Create synthetic 50 demo HDF5 fixture for libero_10 task 3."""
    data_dir = tmp_path / "datasets" / "libero_10"
    data_dir.mkdir(parents=True)
    lengths = [160 + (ep % 3) * 15 for ep in range(50)]
    for task in range(10):
        with h5py.File(data_dir / f"task{task}.hdf5", "w") as f:
            data = f.create_group("data")
            data.attrs["problem_info"] = json.dumps(
                {"language_instruction": f"task {task}"}
            )
            if task != 3:
                continue
            for ep, length in enumerate(lengths):
                demo = data.create_group(f"demo_{ep}")
                # Create distinctive actions and states
                time_step = np.arange(length, dtype=np.float32)
                # raw actions bounded in [-1, 1]
                actions = np.sin(time_step[:, None] * 0.1) * 0.9
                demo["actions"] = np.repeat(actions, 7, axis=1)
                obs = demo.create_group("obs")
                # joint_states: 7 dims, gripper_states: 2 dims
                obs["joint_states"] = np.repeat((time_step[:, None] / 200.0) * 2.0 - 1.0, 7, axis=1)
                obs["gripper_states"] = np.repeat((time_step[:, None] / 200.0), 2, axis=1)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    return data_dir, frames_dir


def test_prepare_and_validate_joint_episode_data(synthetic_hdf5_data, tmp_path, monkeypatch):
    data_dir, frames_dir = synthetic_hdf5_data
    monkeypatch.setattr(
        prepare,
        "_official_task_specs",
        lambda *args: [
            {"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "task 3"}
        ],
    )
    monkeypatch.setattr(prepare, "_ensure_longtraj", lambda *args, **kwargs: None)

    out_path = tmp_path / "joint_payload.pt"
    args = prepare._parser().parse_args([
        "--data", str(out_path),
        "--hdf5-dir", str(data_dir.parent),
        "--longtraj", str(frames_dir),
        "--suites", "libero_10",
        "--local-task-ids", "3",
        "--dense-windows",
        "--architecture-version", "dual_tower_expert_v1",
    ])
    prepare.prepare_data(args)

    payload = _validate_data(out_path, architecture_version="dual_tower_expert_v1")

    expected_delta = torch.tensor([.3] * 7 + [.15] * 2) / payload["normalization"]["state_delta_scale"]
    torch.testing.assert_close(payload["world_state_delta"][0, 0], expected_delta)
    for mutation in ("crop", "scale", "frame", "count_type", "cross_window_action"):
        broken = copy.deepcopy(payload)
        if mutation == "crop":
            broken["crop_start"][1] += 15
        elif mutation == "scale":
            broken["normalization"]["state_delta_scale"] *= 2
        elif mutation == "frame":
            broken["world_target_frame_refs"][0][2][0][0] += 1
        elif mutation == "count_type":
            broken["decision_count"] = broken["decision_count"].float()
        else:
            broken["previous_action"][1, 0, 0] += 1
        invalid_path = tmp_path / f"invalid_{mutation}.pt"
        torch.save(broken, invalid_path)
        with pytest.raises(ValueError):
            _validate_data(invalid_path, architecture_version="dual_tower_expert_v1")

    # Contract check
    assert payload["metadata"]["contract"] == JOINT_DATA_CONTRACT
    assert payload["metadata"]["state_delta_contract"] == "joint7_gripper2_unclipped_q01q99_delta_h15_v1"
    assert payload["metadata"]["memory_contract"] == "episode_tbptt8_v1"
    assert payload["metadata"]["window_sampling"] == "episode_contiguous_p15_v1"
    assert payload["metadata"]["window_bound"] == "complete_p15_masked_h50_v1"

    # Shapes and dtypes
    N = len(payload["actions"])
    assert payload["actions"].shape == (N, SEQUENCE, 50, 7)
    assert payload["actions"].dtype == torch.float32
    assert torch.isfinite(payload["actions"]).all()

    assert payload["proprio"].shape == (N, SEQUENCE, 9)
    assert payload["proprio"].dtype == torch.float32
    assert torch.isfinite(payload["proprio"]).all()

    assert payload["previous_action"].shape == (N, SEQUENCE, 7)
    assert payload["previous_action"].dtype == torch.float32
    assert torch.isfinite(payload["previous_action"]).all()

    assert payload["decision_valid_mask"].shape == (N, SEQUENCE)
    assert payload["decision_valid_mask"].dtype == torch.bool

    assert payload["decision_count"].shape == (N,)
    assert payload["decision_count"].dtype == torch.long
    assert torch.equal(payload["decision_count"], payload["decision_valid_mask"].sum(dim=-1))

    assert payload["episode_start"].shape == (N,)
    assert payload["episode_start"].dtype == torch.bool
    assert payload["episode_end"].shape == (N,)
    assert payload["episode_end"].dtype == torch.bool

    assert payload["crop_start"].shape == (N,)
    assert payload["crop_start"].dtype == torch.long

    assert payload["world_state_delta"].shape == (N, SEQUENCE, 9)
    assert payload["world_state_delta"].dtype == torch.float32
    assert torch.isfinite(payload["world_state_delta"]).all()

    # Padded decisions zero delta and all false masks
    padded_mask = ~payload["decision_valid_mask"]
    assert torch.all(payload["world_state_delta"][padded_mask] == 0.0)
    assert not torch.any(payload["action_valid_mask"][padded_mask])
    assert torch.equal(payload["world_target_valid_mask"], payload["decision_valid_mask"])

    # Normalization scale check
    scale = payload["normalization"]["state_delta_scale"]
    assert scale.shape == (9,)
    assert torch.all(scale > 0)
    assert torch.isfinite(scale).all()

    # Episode continuity check
    episodes = torch.unique(payload["episode_id"])
    assert len(episodes) == 50
    for ep in episodes:
        rows = torch.where(payload["episode_id"] == ep)[0]
        assert payload["episode_start"][rows[0]]
        assert not payload["episode_start"][rows[1:]].any()
        assert payload["episode_end"][rows[-1]]
        assert not payload["episode_end"][rows[:-1]].any()
        assert payload["crop_start"][rows[0]] == 0

        crops = payload["crop_start"][rows]
        counts = payload["decision_count"][rows]
        assert torch.equal(crops[1:], crops[:-1] + 15 * counts[:-1])

    # World action donors check
    assert "world_rank_shuffle_action" in payload
    assert "world_rank_shuffle_mask" in payload
    assert payload["world_rank_shuffle_action"].shape == (N, SEQUENCE, EXECUTION_HORIZON, 7)
    assert payload["world_rank_shuffle_mask"].shape == (N, SEQUENCE)
    assert not torch.any(payload["world_rank_shuffle_mask"][padded_mask])


def test_longtraj_frames_dataset_propagates_joint_row_fields(synthetic_hdf5_data, tmp_path, monkeypatch):
    data_dir, frames_dir = synthetic_hdf5_data
    monkeypatch.setattr(
        prepare,
        "_official_task_specs",
        lambda *args: [
            {"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "task 3"}
        ],
    )
    monkeypatch.setattr(prepare, "_ensure_longtraj", lambda *args, **kwargs: None)

    out_path = tmp_path / "joint_payload.pt"
    args = prepare._parser().parse_args([
        "--data", str(out_path),
        "--hdf5-dir", str(data_dir.parent),
        "--longtraj", str(frames_dir),
        "--suites", "libero_10",
        "--local-task-ids", "3",
        "--dense-windows",
        "--architecture-version", "dual_tower_expert_v1",
    ])
    prepare.prepare_data(args)

    monkeypatch.setattr(
        LongTrajFramesDataset,
        "_decode_task",
        lambda self, task_file: [[np.zeros((480, 480, 3), dtype=np.uint8)] * 300 for _ in range(50)],
    )

    ds = LongTrajFramesDataset(out_path)
    assert len(ds) == len(ds.payload["actions"])

    sample = ds[0]
    expected_keys = {
        "actions", "previous_action", "proprio", "language_hidden", "instruction_id", "pair_id",
        "episode_id", "crop_start", "decision_valid_mask", "decision_count", "episode_start",
        "episode_end", "world_state_delta", "world_rank_shuffle_action", "world_rank_shuffle_mask",
        "world_target_valid_mask", "action_valid_mask", "frames",
    }
    for key in expected_keys:
        assert key in sample, f"Missing key {key} in dataset sample"
        if isinstance(sample[key], torch.Tensor):
            assert torch.equal(sample[key], ds.payload[key][0])
