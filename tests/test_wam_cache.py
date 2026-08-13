"""CPU tests for the WAM cache contract (Task 5): split masks, anchor index,
last-slice pooling, and the builder/dataset roundtrip.

All interfaces live in `va_compound/wam_cache.py` (sibling agent); tests
skip with "dependency not yet implemented" until it lands.  The builder
roundtrip tries the documented synthetic fake path (`build_wam_cache(None,
...)`) first and a pre-encoded record-list file second, so it survives
either fake-input design chosen by the implementing agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    from va_compound.wam_cache import (
        WAMCacheDataset,
        build_wam_cache,
        wam_anchor_index,
        wam_last_slice_pool,
        wam_split_from_episode,
    )
except ImportError:
    WAMCacheDataset = None  # type: ignore[assignment]
    build_wam_cache = None  # type: ignore[assignment]
    wam_anchor_index = None  # type: ignore[assignment]
    wam_last_slice_pool = None  # type: ignore[assignment]
    wam_split_from_episode = None  # type: ignore[assignment]


def _need(name, obj):
    if obj is None:
        pytest.skip(f"dependency not yet implemented: va_compound.wam_cache.{name}")
    return obj


def _episode_task_ids(n_episodes=30, n_tasks=3):
    episode_ids = [episode for episode in range(n_episodes)]
    task_ids = [episode % n_tasks for episode in range(n_episodes)]
    return (
        torch.tensor(episode_ids),
        torch.tensor(task_ids),
    )


def test_split_no_leakage() -> None:
    split = _need("wam_split_from_episode", wam_split_from_episode)
    episode_ids, task_ids = _episode_task_ids()
    # Repeat rows per episode (multiple decision points per episode).
    episode_ids = episode_ids.repeat_interleave(3)
    task_ids = task_ids.repeat_interleave(3)

    masks = split(episode_ids, task_ids)

    assert len(masks) == 3
    for mask in masks:
        assert mask.dtype == torch.bool
        assert mask.shape == episode_ids.shape
    union = sum(mask.int() for mask in masks)
    assert torch.equal(union, torch.ones_like(union))
    # Every episode must live entirely in exactly one split.
    for episode in torch.unique(episode_ids).tolist():
        rows = episode_ids == episode
        assert sum(int(mask[rows].all()) for mask in masks) == 1


def test_split_ratio_8_1_1() -> None:
    split = _need("wam_split_from_episode", wam_split_from_episode)
    episode_ids, task_ids = _episode_task_ids(n_episodes=30)

    train, val, test = split(episode_ids, task_ids)

    counts = [int(train.sum()), int(val.sum()), int(test.sum())]
    assert counts == [24, 3, 3]  # 30 episodes -> 24/3/3 = 8:1:1


def test_anchor_is_last_of_window() -> None:
    anchor = _need("wam_anchor_index", wam_anchor_index)
    assert anchor(seq_len=4) == 3
    assert anchor() == 3


def test_last_slice_pool_shapes() -> None:
    pool = _need("wam_last_slice_pool", wam_last_slice_pool)
    torch.manual_seed(3)
    h11 = torch.randn(2, 1152, 768)

    pooled = pool(h11)

    assert pooled.shape == (2, 16, 768)
    expected = (
        h11[:, 576:]                     # last time slice (t=1)
        .reshape(2, 4, 6, 4, 6, 768)     # 24x24 grid -> 4x4 blocks of 6x6
        .mean(dim=(2, 4))
        .reshape(2, 16, 768)
    )
    torch.testing.assert_close(pooled, expected, rtol=1e-4, atol=1e-5)


def _fake_record(episode: int, task: int) -> dict:
    return {
        "episode_id": episode,
        "task_id": task,
        "action_condition": torch.randn(48, 512),
        "va_layers": [torch.randn(16, 512) for _ in range(8)],
        "spatial16": torch.randn(16, 768),
        "geo8": torch.randn(8),
        "actions": torch.randn(48, 4),
        "target_latent": torch.randn(3, 16, 768),
        "target_geo": torch.randn(3, 2, 8),
    }


def test_dataset_roundtrip(tmp_path: Path) -> None:
    builder = _need("build_wam_cache", build_wam_cache)
    dataset_cls = _need("WAMCacheDataset", WAMCacheDataset)
    torch.manual_seed(0)
    out_dir = tmp_path / "cache"
    base_ckpt = tmp_path / "base.pt"
    torch.save({"dummy": torch.zeros(1)}, base_ckpt)

    manifest = None
    errors = []
    # 1) Documented synthetic fake path: windows_pt=None builds fake records.
    try:
        manifest = builder(None, str(out_dir), base_ckpt=str(base_ckpt))
    except (TypeError, ValueError, FileNotFoundError) as error:
        errors.append(repr(error))
    # 2) Fallback: a pre-encoded record list as the windows file.
    if manifest is None:
        windows = tmp_path / "synthetic_windows.pt"
        torch.save(
            [_fake_record(episode, episode % 3) for episode in range(6)], windows
        )
        try:
            manifest = builder(str(windows), str(out_dir), base_ckpt=str(base_ckpt))
        except (TypeError, ValueError, FileNotFoundError) as error:
            errors.append(repr(error))
    assert manifest is not None, f"build_wam_cache fake path rejected: {errors}"
    assert manifest.contract == "e7_wam_cache_v1"
    assert manifest.per_task_files, "builder wrote no per-task shards"

    dataset = dataset_cls(str(out_dir), manifest, split="train")
    assert len(dataset) >= 1

    shard = torch.load(out_dir / manifest.per_task_files[0])
    records = shard if isinstance(shard, list) else shard.get("records")
    assert records, f"unexpected shard layout in {manifest.per_task_files[0]}"

    item = dataset[0]
    matched = 0
    for key, value in records[0].items():
        if key in item and torch.is_tensor(item[key]):
            torch.testing.assert_close(item[key], value, rtol=0.0, atol=0.0)
            matched += 1
    assert matched > 0, "dataset item shares no tensor keys with the shard record"

    # Read-back must be deterministic.
    again = dataset[0]
    for key, value in item.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(again[key], value, rtol=0.0, atol=0.0)
        else:
            assert again[key] == value
