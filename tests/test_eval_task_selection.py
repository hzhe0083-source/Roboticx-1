from __future__ import annotations

import numpy as np
import pytest
import torch

from eval_metaworld import (
    TASK35_EVAL50_SEEDS,
    evaluation_episode_seed,
    select_eval_tasks,
    task35_ablation_dense,
    task35_ablation_frames,
    task35_ablation_geometry,
)


def test_subset_preserves_global_task_ids_and_seeds() -> None:
    tasks = [f"task-{index}" for index in range(20)]

    selected = select_eval_tasks(tasks, "14,3", max_tasks=49)

    assert selected == [(14, "task-14"), (3, "task-3")]
    assert evaluation_episode_seed(selected[0][0], 2) == 14002
    assert evaluation_episode_seed(selected[1][0], 2) == 3002


def test_task35_paired_protocol_uses_seeds_35000_through_35049() -> None:
    assert [evaluation_episode_seed(35, trial) for trial in range(50)] == list(
        TASK35_EVAL50_SEEDS
    )


def test_task35_metadata_index_maps_to_peg_insert_side() -> None:
    tasks = [f"task-{index}" for index in range(49)]
    tasks[35] = "Insert a peg sideways"
    assert select_eval_tasks(tasks, "35", max_tasks=49) == [
        (35, "Insert a peg sideways")
    ]


def test_require_task35_peg_insert_side_is_fail_closed() -> None:
    from eval_metaworld import require_task35_peg_insert_side

    mapping = {"Insert a peg sideways": "peg-insert-side-v3"}
    assert (
        require_task35_peg_insert_side([(35, "Insert a peg sideways")], mapping)
        == "peg-insert-side-v3"
    )
    with pytest.raises(ValueError, match="exactly --task-ids 35"):
        require_task35_peg_insert_side([(0, "Insert a peg sideways")], mapping)
    with pytest.raises(ValueError, match="expected peg-insert-side-v3"):
        require_task35_peg_insert_side(
            [(35, "Insert a peg sideways")],
            {"Insert a peg sideways": "peg-unplug-side-v3"},
        )


def test_default_selection_retains_metadata_indices() -> None:
    tasks = ["a", "b", "c"]
    assert select_eval_tasks(tasks, None, max_tasks=2) == [(0, "a"), (1, "b")]


def test_task_selection_rejects_out_of_range_global_id() -> None:
    with pytest.raises(ValueError, match="out of range"):
        select_eval_tasks(["a", "b"], "2", max_tasks=49)


def test_task35_temporal_ablation_reverses_only_frame_order() -> None:
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(4)]
    reversed_frames = task35_ablation_frames(frames, "temporal-reverse")
    assert [int(frame[0, 0, 0]) for frame in reversed_frames] == [3, 2, 1, 0]
    assert task35_ablation_frames(frames, "none") is frames


def test_task35_dense_zero_preserves_shape_dtype_and_source() -> None:
    dense = {
        5: torch.randn(2, 8, 4, dtype=torch.float16),
        11: torch.randn(2, 8, 4, dtype=torch.float16),
    }
    zero = task35_ablation_dense(dense, "dense-zero")
    for layer in (5, 11):
        assert zero[layer].shape == dense[layer].shape
        assert zero[layer].dtype == dense[layer].dtype
        assert torch.count_nonzero(zero[layer]) == 0
        assert torch.count_nonzero(dense[layer]) > 0
    assert task35_ablation_dense(dense, "none") is dense


def test_task35_geometry_ablations_are_deterministic_single_route_changes() -> None:
    metric = torch.arange(8, dtype=torch.float32)[None]
    assert torch.equal(
        task35_ablation_geometry(metric, "geometry-zero"), torch.zeros_like(metric)
    )
    assert task35_ablation_geometry(metric, "geometry-shuffle").tolist() == [
        [2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0]
    ]
    assert task35_ablation_geometry(metric, "none") is metric
