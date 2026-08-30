from __future__ import annotations

import io
import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from train import TaskLocalityWeightedSampler
from scripts.build_dagger_online_index import build_augmented_index
from va_compound.flow import masked_flow_matching_loss
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


def _episode(
    length: int, offset: float, *, perturb_start: int | None = None
) -> dict:
    actions = np.linspace(-0.8 + offset, 0.8 + offset, length * 4).reshape(length, 4)
    states = np.linspace(-0.5 + offset, 0.5 + offset, length * 4).reshape(length, 4)
    valid = np.ones(length, dtype=bool)
    recovery = np.zeros(length, dtype=bool)
    settle = np.zeros(length, dtype=bool)
    episode = {
        "frames": [_jpeg(index) for index in range(length)],
        "actions": actions.astype(np.float32),
        "states": states.astype(np.float32),
        "first_success": length - 1,
        "frame_valid": np.ones(length, dtype=bool),
        "action_executed": np.ones(length, dtype=bool),
        "action_supervision_valid": valid,
        "recovery_mask": recovery,
        "settle_mask": settle,
    }
    if perturb_start is not None:
        perturb_end = perturb_start + 5
        valid[perturb_start:perturb_end] = False
        settle[perturb_start:perturb_end] = True
        recovery[perturb_start:] = True
        episode.update(
            perturbed=True,
            n_perturb_events=1,
            perturb_start=perturb_start,
            perturb_end=perturb_end,
        )
    return episode


def _dataset(
    tmp_path,
    *,
    seed: int = 7,
    recovery_samples: int = 0,
    recovery_episode: bool = False,
    action_horizon: int = 15,
    length: int | None = None,
    short_episode_padding: bool = False,
) -> OnlineLongTrajEpisodeDataset:
    task = "synthetic-online-v3"
    episode_length = (
        int(length) if length is not None else 120 if recovery_episode else 80
    )
    raw_path = tmp_path / f"metaworld_longtraj_{task}.pt"
    torch.save(
        {
            "task": task,
            "episodes": [
                _episode(
                    episode_length,
                    0.0,
                    perturb_start=70 if recovery_episode else None,
                ),
                _episode(episode_length, 0.05),
            ],
        },
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
                        "length": episode_length,
                        "valid_start_count": (
                            episode_length
                            if short_episode_padding
                            else episode_length - 60
                        ),
                        "split": "train",
                    }
                    for episode in range(2)
                ],
            }
            | (
                {
                    "sampling_protocol": {
                        "sequence_length": 4,
                        "action_horizon": 15,
                        "decision_stride": 15,
                        "crop_start_stride": 1,
                        "world_target_horizon": 15,
                        "short_episode_padding": "repeat_last_mask_actions_v1",
                    }
                }
                if short_episode_padding
                else {}
            )
        )
    )
    return OnlineLongTrajEpisodeDataset(
        index_path,
        longtraj_dir=tmp_path,
        samples_per_episode=6 if recovery_episode else 3,
        recovery_samples_per_episode=recovery_samples,
        sampling_seed=seed,
        include_world_target_frames=True,
        action_horizon=action_horizon,
    )


def test_online_recovery_slots_select_visible_expert_corrections(tmp_path) -> None:
    dataset = _dataset(
        tmp_path,
        recovery_samples=3,
        recovery_episode=True,
    )
    assert dataset.payload["metadata"]["recovery_samples_per_episode"] == 3
    for index in range(6):
        item = dataset[index]
        recovery_valid = item["recovery_mask"] & item["action_valid_mask"]
        assert bool(recovery_valid.any()) is (index < 3)
    assert not any(
        bool((dataset[index]["recovery_mask"] & dataset[index]["action_valid_mask"]).any())
        for index in range(6, 12)
    )


def test_online_dataset_contains_no_offline_windows_and_preserves_continuity(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    assert len(dataset) == 6
    assert not dataset.short_episode_padding
    assert "short_episode_padding" not in dataset.payload["metadata"]
    assert dataset._valid_starts(dataset._episodes[0]) == list(range(20))
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


def test_h50_keeps_h15_crop_distribution_and_masks_only_the_extension(tmp_path) -> None:
    h15 = _dataset(tmp_path, seed=31)
    h50 = _dataset(tmp_path, seed=31, action_horizon=50)
    assert h15._valid_starts(h15._episodes[0]) == h50._valid_starts(h50._episodes[0])

    for epoch in range(3):
        h15.set_epoch(epoch)
        h50.set_epoch(epoch)
        item15 = h15[0]
        item50 = h50[0]
        assert int(item15["crop_start"]) == int(item50["crop_start"])
        assert item50["actions"].shape == (4, 50, 4)
        np.testing.assert_array_equal(item50["actions"][:, :15], item15["actions"])
        np.testing.assert_array_equal(
            item50["action_valid_mask"][:, :15], item15["action_valid_mask"]
        )
        np.testing.assert_array_equal(
            item50["world_target_valid_mask"], item15["world_target_valid_mask"]
        )
        assert item50["world_rank_shuffle_action"].shape == (4, 15, 4)
        assert not bool(item50["action_valid_mask"][:, 15:].all())


def test_opt_in_short_episode_padding_masks_every_nonexistent_action(tmp_path) -> None:
    dataset = _dataset(
        tmp_path,
        seed=37,
        action_horizon=50,
        length=14,
        short_episode_padding=True,
    )
    assert dataset.short_episode_padding
    assert dataset._valid_starts(dataset._episodes[0]) == list(range(14))

    item = dataset[0]
    assert item["actions"].shape == (4, 50, 4)
    assert item["frames"].shape == (4, 4, 4, 4, 3)
    assert item["world_target_frames"].shape == (4, 1, 4, 4, 3)
    assert not bool(item["world_target_valid_mask"].any())

    crop = dataset._crop(dataset._episodes[0], 0)
    valid = crop["action_valid_mask"]
    assert int(valid.sum()) == 14
    assert bool(valid[0, :14].all())
    assert not bool(valid[0, 14:].any())
    assert not bool(valid[1:].any())
    np.testing.assert_array_equal(
        crop["actions"][0, 14:],
        np.broadcast_to(crop["actions"][0, 13], (36, 4)),
    )

    predicted = torch.zeros(1, 4, 50, 4)
    target = torch.full_like(predicted, 100.0)
    target[0][torch.from_numpy(valid)] = 1.0
    loss, prefix, tail = masked_flow_matching_loss(
        predicted,
        target,
        {"action_valid_mask": torch.from_numpy(valid).unsqueeze(0)},
        prefix_steps=15,
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(prefix, torch.tensor(1.0))
    torch.testing.assert_close(tail, torch.tensor(0.0))


def test_online_dataset_can_read_an_episode_overlay_without_copying_base_data(
    tmp_path,
) -> None:
    dataset = _dataset(tmp_path)
    dagger_dir = tmp_path / "dagger"
    dagger_dir.mkdir()
    overlay_path = dagger_dir / (
        "metaworld_longtraj_synthetic-online-v3_dagger_seed14042_t0-1.pt"
    )
    overlay_episode = _episode(120, 0.15, perturb_start=15)
    overlay_episode["actions"][:] = 0.95
    overlay_episode["episode_seed"] = 14042
    reference = torch.load(
        Path(json.loads(dataset.path.read_text())["language_reference"]["path"]),
        map_location="cpu",
        weights_only=True,
    )
    torch.save(
        {
            "task": "synthetic-online-v3",
            "episodes": [overlay_episode],
            "normalization": reference["normalization"],
            "metadata": {
                "contract": "current_policy_dagger_v1",
                "checkpoint": "/tmp/policy.pt",
            },
        },
        overlay_path,
    )
    augmented_path = tmp_path / "augmented.json"
    augmented = build_augmented_index(
        dataset.path,
        dagger_dir,
        augmented_path,
        repeat=2,
    )
    assert augmented["dagger_augmentation"]["unique_episodes"] == 1
    assert augmented["dagger_augmentation"]["logical_train_episodes"] == 2

    overlaid = OnlineLongTrajEpisodeDataset(
        augmented_path,
        longtraj_dir=tmp_path,
        samples_per_episode=3,
        sampling_seed=7,
    )
    merged_path = dagger_dir / "merged" / (
        "metaworld_longtraj_synthetic-online-v3_dagger_merged_v1.pt"
    )
    assert overlaid._entry_source(overlaid._episodes[-1]) == merged_path
    assert np.allclose(overlaid[6]["actions"], 0.95)

    second_dir = tmp_path / "dagger-round2"
    second_dir.mkdir()
    second_overlay = second_dir / (
        "metaworld_longtraj_synthetic-online-v3_dagger_seed24042_t0-1.pt"
    )
    second_episode = _episode(120, 0.25, perturb_start=15)
    second_episode["episode_seed"] = 24042
    torch.save(
        {
            "task": "synthetic-online-v3",
            "episodes": [second_episode],
            "normalization": reference["normalization"],
            "metadata": {
                "contract": "current_policy_dagger_v1",
                "checkpoint": "/tmp/policy-round2.pt",
            },
        },
        second_overlay,
    )
    cumulative_path = tmp_path / "augmented-round2.json"
    cumulative = build_augmented_index(
        augmented_path,
        second_dir,
        cumulative_path,
        repeat=1,
    )
    assert cumulative["dagger_augmentation"]["round_count"] == 2
    assert cumulative["dagger_augmentation"]["cumulative_unique_episodes"] == 2
    assert len(cumulative["additional_sources"]) == 2


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


def test_negative_index_is_fixed_epoch_zero_anchor_replay(tmp_path) -> None:
    dataset = _dataset(tmp_path, seed=23)
    dataset.set_epoch(4)
    first = dataset[-1]
    dataset.set_epoch(9)
    second = dataset[-1]
    assert bool(first["anchor_replay"])
    assert bool(second["anchor_replay"])
    assert int(first["pair_id"]) == int(second["pair_id"]) == 0
    assert int(first["crop_start"]) == int(second["crop_start"])
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert not bool(dataset[0]["anchor_replay"])


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
