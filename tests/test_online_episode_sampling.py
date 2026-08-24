from __future__ import annotations

import io
import json
import threading
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from train import TaskLocalityWeightedSampler
from va_compound.vision import longtraj_frames as longtraj_impl
from va_compound.longtraj_frames import (
    ONLINE_EPISODE_CONTRACT,
    OnlineLongTrajEpisodeDataset,
)


def _jpeg(value: int) -> bytes:
    image = Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def _episode(length: int, offset: float) -> dict:
    actions = np.linspace(-0.8 + offset, 0.8 + offset, length * 4).reshape(length, 4)
    states = np.linspace(-0.5 + offset, 0.5 + offset, length * 4).reshape(length, 4)
    return {
        "frames": [_jpeg(index) for index in range(length)],
        "actions": actions.astype(np.float32),
        "states": states.astype(np.float32),
        "first_success": length - 1,
        "frame_valid": np.ones(length, dtype=bool),
        "action_executed": np.ones(length, dtype=bool),
        "action_supervision_valid": np.ones(length, dtype=bool),
        "recovery_mask": np.zeros(length, dtype=bool),
    }


def _dataset(tmp_path, *, seed: int = 7) -> OnlineLongTrajEpisodeDataset:
    task = "synthetic-online-v3"
    raw_path = tmp_path / f"metaworld_longtraj_{task}.pt"
    torch.save(
        {"task": task, "episodes": [_episode(80, 0.0), _episode(80, 0.05)]},
        raw_path,
    )
    reference_path = tmp_path / "language.pt"
    torch.save(
        {
            "normalization": {
                "action_q01": torch.full((4,), -1.0),
                "action_q99": torch.full((4,), 1.0),
                "state_q01": torch.full((4,), -1.0),
                "state_q99": torch.full((4,), 1.0),
            },
            "instruction_id": torch.tensor([0]),
            "language_hidden": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            "language_mask": torch.ones(1, 3, dtype=torch.bool),
            "metadata": {"tasks": ["Do the synthetic task"]},
        },
        reference_path,
    )
    index_path = tmp_path / "episodes.json"
    index_path.write_text(
        json.dumps(
            {
                "contract": ONLINE_EPISODE_CONTRACT,
                "sampling_protocol": {
                    "sequence_length": 4,
                    "action_horizon": 15,
                    "decision_stride": 15,
                    "crop_start_stride": 1,
                    "world_target_horizon": 15,
                },
                "language_reference": {"path": str(reference_path)},
                "tasks": [
                    {
                        "task_id": 0,
                        "task": task,
                        "description": "Do the synthetic task",
                        "source_path": str(raw_path),
                        "sha256": "0" * 64,
                        "size_bytes": raw_path.stat().st_size,
                    }
                ],
                "episodes": [
                    {
                        "task_id": 0,
                        "task": task,
                        "episode_index": episode,
                        "episode_id": episode,
                        "length": 80,
                        "valid_start_count": 20,
                        "split": "train",
                    }
                    for episode in range(2)
                ],
            }
        )
    )
    return OnlineLongTrajEpisodeDataset(
        index_path,
        longtraj_dir=tmp_path,
        samples_per_episode=3,
        sampling_seed=seed,
        include_world_target_frames=True,
    )


def test_online_dataset_contains_no_offline_windows_and_preserves_continuity(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    assert len(dataset) == 6
    assert "actions" not in dataset.index
    assert dataset.payload["actions"].numel() == 0
    assert "frame_refs" not in dataset.payload
    assert dataset.model_schema == {
        "language_dim": 4,
        "action_horizon": 15,
        "action_dim": 4,
        "proprio_dim": 4,
    }

    item = dataset[0]
    assert item["actions"].shape == (4, 15, 4)
    assert item["frames"].shape == (4, 4, 4, 4, 3)
    assert item["world_target_frames"].shape == (4, 1, 4, 4, 3)
    np.testing.assert_array_equal(
        item["previous_action"][1:], item["actions"][:-1, 14]
    )
    assert item["world_rank_shuffle_action"].shape == (4, 15, 4)
    assert bool(item["world_rank_shuffle_mask"].any())


def test_online_dataset_decodes_only_referenced_frames_and_reuses_cache(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    original = longtraj_impl._decode_jpeg_bytes
    with patch.object(longtraj_impl, "_decode_jpeg_bytes", wraps=original) as decode:
        first = dataset[0]
        first_decode_count = decode.call_count
        # One item references 4x4 current frames plus four World endpoints.
        # The old eager path decoded all 160 frames from both episodes.
        assert 0 < first_decode_count <= 20
        np.testing.assert_array_equal(dataset[0]["frames"], first["frames"])
        assert decode.call_count == first_decode_count


def test_online_dataset_prefetches_raw_task_without_blocking_or_reloading(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    source = dataset._task_source("synthetic-online-v3")
    key = longtraj_impl._raw_task_key(source)
    with longtraj_impl._PROCESS_RAW_TASK_LOCK:
        longtraj_impl._PROCESS_RAW_TASKS.pop(key, None)
        longtraj_impl._PROCESS_RAW_TASK_FUTURES.pop(key, None)

    started = threading.Event()
    release = threading.Event()
    original_load = torch.load

    def slow_load(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_load(*args, **kwargs)

    with patch.object(longtraj_impl.torch, "load", side_effect=slow_load) as load:
        begin = time.monotonic()
        dataset.prefetch_task_ids([0])
        assert time.monotonic() - begin < 0.5
        assert started.wait(timeout=2)
        dataset.prefetch_task_ids([0])
        assert load.call_count == 1
        release.set()
        data = longtraj_impl._load_process_raw_task(source)

    assert data["task"] == "synthetic-online-v3"
    assert load.call_count == 1


def test_online_starts_change_by_epoch_and_are_not_p15_aligned(tmp_path) -> None:
    dataset = _dataset(tmp_path, seed=19)
    starts = []
    for epoch in range(12):
        dataset.set_epoch(epoch)
        epoch_starts = [int(dataset[index]["crop_start"]) for index in range(3)]
        assert len(set(epoch_starts)) == 3
        starts.extend(epoch_starts)
    assert len(set(starts)) > 3
    assert any(start % 15 != 0 for start in starts)

    clone = _dataset(tmp_path, seed=19)
    clone.set_epoch(11)
    dataset.set_epoch(11)
    assert int(clone[1]["crop_start"]) == int(dataset[1]["crop_start"])


def test_locality_sampler_sets_online_dataset_epoch(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    sampler = TaskLocalityWeightedSampler(
        dataset.payload["instruction_id"],
        dataset.payload["episode_id"],
        torch.ones(1),
        batch_size=2,
        sampling_mode="full",
        epoch_dataset=dataset,
    )
    list(sampler)
    assert dataset.epoch == 0
    sampler.advance(len(sampler))
    list(sampler)
    assert dataset.epoch == 1


def test_locality_sampler_starts_next_task_load_after_first_current_batch() -> None:
    class RecordingDataset:
        def __init__(self) -> None:
            self.epoch = -1
            self.prefetched: list[list[int]] = []

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def prefetch_task_ids(self, task_ids: list[int]) -> None:
            self.prefetched.append(list(task_ids))

    dataset = RecordingDataset()
    instruction_id = torch.tensor([0] * 8 + [1] * 8)
    sampler = TaskLocalityWeightedSampler(
        instruction_id,
        torch.arange(16),
        torch.ones(2),
        batch_size=2,
        seed=3,
        sampling_mode="full",
        epoch_dataset=dataset,
    )
    iterator = iter(sampler)
    first = next(iterator)
    first_task = int(instruction_id[first[0]])
    assert dataset.prefetched == []

    second = next(iterator)
    assert int(instruction_id[second[0]]) == first_task
    assert dataset.prefetched == [[1 - first_task]]
