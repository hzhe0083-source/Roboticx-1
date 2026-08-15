from __future__ import annotations

import pytest

from eval_metaworld import evaluation_episode_seed, select_eval_tasks


def test_subset_preserves_global_task_ids_and_seeds() -> None:
    tasks = [f"task-{index}" for index in range(20)]

    selected = select_eval_tasks(tasks, "14,3", max_tasks=49)

    assert selected == [(14, "task-14"), (3, "task-3")]
    assert evaluation_episode_seed(selected[0][0], 2) == 14002
    assert evaluation_episode_seed(selected[1][0], 2) == 3002


def test_task35_paired_protocol_uses_seeds_35000_through_35049() -> None:
    assert [evaluation_episode_seed(35, trial) for trial in range(50)] == list(
        range(35000, 35050)
    )


def test_default_selection_retains_metadata_indices() -> None:
    tasks = ["a", "b", "c"]
    assert select_eval_tasks(tasks, None, max_tasks=2) == [(0, "a"), (1, "b")]


def test_task_selection_rejects_out_of_range_global_id() -> None:
    with pytest.raises(ValueError, match="out of range"):
        select_eval_tasks(["a", "b"], "2", max_tasks=49)
