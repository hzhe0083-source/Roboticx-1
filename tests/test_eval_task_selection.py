from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eval_metaworld import (
    TASK35_EVAL50_SEEDS,
    build_dagger_episode,
    cached_task35_language,
    dagger_takeover_step,
    evaluation_episode_seed,
    load_metaworld_description_to_env,
    select_eval_tasks,
    task35_ablation_dense,
    task35_ablation_frames,
    task35_ablation_geometry,
    validate_language_features,
)
from scripts.build_longtraj_features import resolve_episode_semantics
from scripts.mt50_difficulty import (
    MT50_BENCHMARK_ENV_ALIASES,
    MT50_BENCHMARK_GROUPS,
    summarize_mt50_benchmark_trials,
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


def test_formal_mt50_protocol_shares_reset_seeds_across_tasks() -> None:
    assert [
        evaluation_episode_seed(0, trial, base_seed=4042) for trial in range(10)
    ] == list(range(4042, 4052))
    assert evaluation_episode_seed(49, 9, base_seed=4042) == 4051


def test_dagger_takeover_keeps_only_expert_recovery_supervision() -> None:
    takeover = dagger_takeover_step(7, 14042, 45, 120)
    assert takeover % 15 == 0
    assert 45 <= takeover <= 120
    n = takeover + 81
    success_step = takeover + 20
    success = [index == success_step for index in range(n)]
    episode = build_dagger_episode(
        episode_seed=14042,
        takeover_step=takeover,
        prefix_keep=45,
        frames=[np.full((2, 2, 3), index, dtype=np.uint8) for index in range(n)],
        actions=[np.full(4, index, dtype=np.float32) for index in range(n)],
        states=[np.full(4, index, dtype=np.float32) for index in range(n)],
        action_success=success,
        action_source=[
            "current_policy" if index < takeover else "expert_takeover"
            for index in range(n)
        ],
    )
    assert episode is not None
    assert episode["perturb_start"] == 45
    assert episode["first_success"] == 65
    assert not episode["action_supervision_valid"][:45].any()
    assert episode["action_supervision_valid"][45:66].all()
    assert not episode["action_supervision_valid"][66:].any()
    semantics = resolve_episode_semantics(episode, "dagger-test", legacy_policy="error")
    assert np.array_equal(semantics["valid"], episode["action_supervision_valid"])
    assert np.array_equal(semantics["recovery"], episode["recovery_mask"])
    assert build_dagger_episode(
        episode_seed=14043,
        takeover_step=45,
        prefix_keep=45,
        frames=[np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(10)],
        actions=[np.zeros(4, dtype=np.float32) for _ in range(10)],
        states=[np.zeros(4, dtype=np.float32) for _ in range(10)],
        action_success=[False] * 9 + [True],
        action_source=["current_policy"] * 10,
    ) is None


def test_official_mt50_four_tier_average_is_equal_weighted() -> None:
    assert [len(tasks) for tasks in MT50_BENCHMARK_GROUPS.values()] == [28, 11, 6, 5]
    trials = [
        {
            "env_name": task,
            "success": group in {"easy", "hard"},
        }
        for group, tasks in MT50_BENCHMARK_GROUPS.items()
        for task in tasks
    ]
    summary = summarize_mt50_benchmark_trials(trials)
    assert summary["complete_mt50"] is True
    assert summary["bucket_average"] == pytest.approx(0.5)
    assert summary["raw_episode_success"] == pytest.approx(34 / 50)


def test_native_metaworld_aliases_keep_official_bucket_membership() -> None:
    inverse_alias = {canonical: native for native, canonical in MT50_BENCHMARK_ENV_ALIASES.items()}
    trials = [
        {"env_name": inverse_alias.get(task, task), "success": True}
        for tasks in MT50_BENCHMARK_GROUPS.values()
        for task in tasks
    ]
    summary = summarize_mt50_benchmark_trials(trials)
    assert summary["complete_mt50"] is True
    assert summary["bucket_average"] == 1.0


def test_push_and_push_back_descriptions_are_disambiguated(tmp_path) -> None:
    config = tmp_path / "metaworld_config.json"
    config.write_text(
        json.dumps(
            {
                "TASK_DESCRIPTIONS": {
                    "push-back-v3": "Push the puck to a goal",
                    "push-v3": "Push the puck to a goal",
                }
            }
        )
    )
    mapping = load_metaworld_description_to_env(config)
    assert mapping["Push the puck to a goal"] == "push-v3"
    assert mapping["Pull a puck to a goal"] == "push-back-v3"


def test_compact_language_features_must_share_normalization_and_cover_mt50() -> None:
    normalization = {
        "state_q01": torch.zeros(4),
        "state_q99": torch.ones(4),
    }
    base = {"normalization": normalization}
    language = {
        "normalization": {key: value.clone() for key, value in normalization.items()},
        "metadata": {"tasks": [f"task-{index}" for index in range(50)]},
        "instruction_id": torch.arange(50),
    }
    validate_language_features(base, language)
    language["normalization"]["state_q99"][0] = 2
    with pytest.raises(ValueError, match="normalization differs"):
        validate_language_features(base, language)


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


def test_cached_task35_language_uses_identical_instruction_rows() -> None:
    hidden = torch.zeros(2, 13, 2048)
    hidden[:, 0, 0] = 3.5
    features = {
        "instruction_id": torch.tensor([35, 35]),
        "language_hidden": hidden,
        "language_mask": torch.ones(2, 13, dtype=torch.bool),
    }
    cached, mask = cached_task35_language(features, torch.device("cpu"))
    assert tuple(cached.shape) == (1, 13, 2048)
    assert float(cached[0, 0, 0]) == 3.5
    assert bool(mask.all())


def test_cached_task35_language_rejects_missing_task() -> None:
    with pytest.raises(ValueError, match="no instruction_id=35"):
        cached_task35_language(
            {
                "instruction_id": torch.tensor([0]),
                "language_hidden": torch.zeros(1, 2, 4),
                "language_mask": torch.ones(1, 2, dtype=torch.bool),
            },
            torch.device("cpu"),
        )


def test_cached_task35_language_rejects_mixed_rows() -> None:
    hidden = torch.zeros(2, 2, 4)
    hidden[1, 0, 0] = 1.0
    with pytest.raises(ValueError, match="not identical"):
        cached_task35_language(
            {
                "instruction_id": torch.tensor([35, 35]),
                "language_hidden": hidden,
                "language_mask": torch.ones(2, 2, dtype=torch.bool),
            },
            torch.device("cpu"),
        )


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
