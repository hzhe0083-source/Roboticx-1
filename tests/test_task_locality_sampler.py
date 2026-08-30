from __future__ import annotations

from collections import Counter

import pytest
import torch

from train import (
    TaskLocalityWeightedSampler,
    backward_pcgrad,
    merge_separate_pcgrad_gradients,
    partition_separate_pcgrad_parameters,
    pop_update_gradients,
)


def _sampler(seed: int = 7, block_batches: int = 2) -> TaskLocalityWeightedSampler:
    # 两任务，每任务3条episode、每episode10个窗口。
    instruction = torch.tensor([0] * 30 + [1] * 30)
    episode = torch.tensor(
        [ep for ep in range(3) for _ in range(10)]
        + [10 + ep for ep in range(3) for _ in range(10)]
    )
    return TaskLocalityWeightedSampler(
        instruction,
        episode,
        torch.tensor([1.0, 2.0]),
        batch_size=4,
        seed=seed,
        block_batches=block_batches,
    )


def test_sampler_is_finite_and_each_batch_is_task_local() -> None:
    sampler = _sampler()
    schedule = list(sampler)
    assert len(schedule) == len(sampler) == 15
    tasks = []
    for batch in schedule:
        assert len(batch) == 4
        batch_tasks = {sampler.task_ids[index] for index in batch}
        assert len(batch_tasks) == 1
        tasks.append(next(iter(batch_tasks)))

    # A same-task draw may follow another independently balanced block; the
    # schedule is still chunked at block_batches boundaries.


def test_each_task_block_is_episode_balanced() -> None:
    sampler = _sampler(block_batches=3)
    schedule = list(sampler)
    for start in range(0, len(schedule), sampler.block_batches):
        block = schedule[start : start + sampler.block_batches]
        task = sampler.task_ids[block[0][0]]
        assert all(sampler.task_ids[index] == task for batch in block for index in batch)
        counts = Counter(sampler.episode_ids[index] for batch in block for index in batch)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_sampler_resume_reproduces_exact_remaining_indices() -> None:
    baseline = _sampler(seed=11)
    full = list(baseline)

    running = _sampler(seed=11)
    first = full[:5]
    assert list(running)[:5] == first
    running.advance(5)
    state = running.state_dict()

    resumed = _sampler(seed=11)
    resumed.load_state_dict(state)
    assert list(resumed) == full[5:]


def test_sampler_state_rejects_dataset_or_config_mismatch() -> None:
    state = _sampler().state_dict()
    different = _sampler(block_batches=3)
    with pytest.raises(ValueError, match="block_batches"):
        different.load_state_dict(state)

    changed = _sampler()
    changed.dataset_fingerprint = "different"
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        changed.load_state_dict(state)

    changed_weight = TaskLocalityWeightedSampler(
        instruction_id=torch.tensor([0] * 30 + [1] * 30),
        episode_id=torch.tensor(
            [ep for ep in range(3) for _ in range(10)]
            + [10 + ep for ep in range(3) for _ in range(10)]
        ),
        task_weights=torch.tensor([1.0, 3.0]),
        batch_size=4,
        seed=7,
        block_batches=2,
    )
    with pytest.raises(ValueError, match="task_weights"):
        changed_weight.load_state_dict(state)


def test_two_task_exposure_preserves_requested_weights() -> None:
    counts = Counter()
    sampler = _sampler(seed=23, block_batches=1)
    for epoch in range(200):
        sampler.epoch = epoch
        for batch in sampler:
            counts[sampler.task_ids[batch[0]]] += 1
    ratio = counts[1] / counts[0]
    assert ratio == pytest.approx(2.0, rel=0.12)


def test_balanced_sampler_has_exact_per_epoch_task_quotas() -> None:
    instruction = torch.repeat_interleave(torch.arange(5), 10)
    episode = torch.arange(50) // 2
    sampler = TaskLocalityWeightedSampler(
        instruction_id=instruction,
        episode_id=episode,
        task_weights=torch.ones(5),
        batch_size=4,
        seed=29,
        block_batches=2,
        sampling_mode="balanced",
    )

    counts = Counter()
    for batch in sampler:
        batch_tasks = {sampler.task_ids[index] for index in batch}
        assert len(batch_tasks) == 1
        counts[next(iter(batch_tasks))] += 1

    # 50 // 4 = 12 batches, split exactly as 3/3/2/2/2.
    assert sum(counts.values()) == len(sampler) == 12
    assert set(counts) == set(range(5))
    assert max(counts.values()) - min(counts.values()) == 1


def test_balanced_sampler_state_rejects_sampling_mode_change() -> None:
    instruction = torch.tensor([0] * 8 + [1] * 8)
    episode = torch.arange(16) // 2
    balanced = TaskLocalityWeightedSampler(
        instruction,
        episode,
        torch.ones(2),
        batch_size=2,
        sampling_mode="balanced",
    )
    weighted = TaskLocalityWeightedSampler(
        instruction,
        episode,
        torch.ones(2),
        batch_size=2,
        sampling_mode="weighted",
    )

    with pytest.raises(ValueError, match="sampling_mode"):
        weighted.load_state_dict(balanced.state_dict())


def test_data_parallel_shards_keep_the_same_task_and_are_disjoint() -> None:
    instruction = torch.tensor([0] * 30 + [1] * 30)
    episode = torch.tensor(
        [ep for ep in range(3) for _ in range(10)]
        + [10 + ep for ep in range(3) for _ in range(10)]
    )
    kwargs = dict(
        instruction_id=instruction,
        episode_id=episode,
        task_weights=torch.ones(2),
        batch_size=8,
        seed=11,
        block_batches=2,
        sampling_mode="balanced",
        world_size=2,
    )
    rank0 = TaskLocalityWeightedSampler(**kwargs, rank=0)
    rank1 = TaskLocalityWeightedSampler(**kwargs, rank=1)
    global_sampler = TaskLocalityWeightedSampler(
        instruction, episode, torch.ones(2), 8, seed=11,
        block_batches=2, sampling_mode="balanced",
    )
    local0, local1, full = list(rank0), list(rank1), list(global_sampler)
    assert len(local0) == len(local1) == len(full) == len(global_sampler)
    for left, right, parent in zip(local0, local1, full, strict=True):
        assert len(left) == len(right) == 4
        assert set(left).isdisjoint(right)
        assert sorted(left + right) == sorted(parent)
        tasks = {rank0.task_ids[index] for index in left + right}
        assert tasks == {rank0.task_ids[parent[0]]}


def test_data_parallel_rejects_a_batch_that_does_not_divide() -> None:
    instruction = torch.arange(8) * 0
    episode = torch.arange(8)
    with pytest.raises(ValueError, match="must divide"):
        TaskLocalityWeightedSampler(
            instruction, episode, torch.ones(1), batch_size=3, world_size=2
        )


def test_sampler_state_rejects_world_size_change() -> None:
    instruction = torch.tensor([0] * 8 + [1] * 8)
    episode = torch.arange(16) // 2
    single = TaskLocalityWeightedSampler(
        instruction, episode, torch.ones(2), batch_size=4
    )
    dual = TaskLocalityWeightedSampler(
        instruction, episode, torch.ones(2), batch_size=4, world_size=2
    )
    with pytest.raises(ValueError, match="world_size"):
        dual.load_state_dict(single.state_dict())


def test_full_sampler_covers_all_10722_rows_once_with_even_ddp_tail() -> None:
    n_rows = 10_722
    instruction = torch.arange(n_rows) % 49
    episode = torch.arange(n_rows) // 7
    kwargs = dict(
        instruction_id=instruction,
        episode_id=episode,
        task_weights=torch.ones(49),
        batch_size=48,
        seed=31,
        block_batches=4,
        sampling_mode="full",
    )
    global_sampler = TaskLocalityWeightedSampler(**kwargs)
    rank0 = TaskLocalityWeightedSampler(**kwargs, rank=0, world_size=2)
    rank1 = TaskLocalityWeightedSampler(**kwargs, rank=1, world_size=2)

    full = list(global_sampler)
    local0 = list(rank0)
    local1 = list(rank1)
    assert len(full) == len(local0) == len(local1) == 224
    assert [len(batch) for batch in full[:-1]] == [48] * 223
    assert len(full[-1]) == 18
    assert len(local0[-1]) == len(local1[-1]) == 9
    assert sum(map(len, local0)) == sum(map(len, local1)) == n_rows // 2
    flattened = [index for batch in full for index in batch]
    assert len(flattened) == n_rows
    assert sorted(flattened) == list(range(n_rows))
    flattened_tasks = [global_sampler.task_ids[index] for index in flattened]
    task_runs = [
        task
        for offset, task in enumerate(flattened_tasks)
        if offset == 0 or task != flattened_tasks[offset - 1]
    ]
    assert len(task_runs) == 49
    assert sorted(task_runs) == list(range(49))
    assert all(
        len({global_sampler.task_ids[index] for index in batch}) <= 2
        for batch in full
    )
    for left, right, parent in zip(local0, local1, full, strict=True):
        assert len(left) == len(right)
        assert set(left).isdisjoint(right)
        assert sorted(left + right) == sorted(parent)


def test_full_sampler_23_epoch_cursor_and_state_roundtrip() -> None:
    n_rows = 10_722
    sampler = TaskLocalityWeightedSampler(
        instruction_id=torch.arange(n_rows) % 49,
        episode_id=torch.arange(n_rows) // 7,
        task_weights=torch.ones(49),
        batch_size=48,
        seed=37,
        sampling_mode="full",
        rank=1,
        world_size=2,
    )
    steps_per_epoch = len(sampler)
    assert steps_per_epoch == 224
    sampler.advance(23 * steps_per_epoch - 1)
    assert sampler.epoch == 22
    assert sampler.batch_cursor == 223
    assert [len(batch) for batch in sampler] == [9]

    resumed = TaskLocalityWeightedSampler(
        instruction_id=torch.arange(n_rows) % 49,
        episode_id=torch.arange(n_rows) // 7,
        task_weights=torch.ones(49),
        batch_size=48,
        seed=37,
        sampling_mode="full",
        rank=1,
        world_size=2,
    )
    resumed.load_state_dict(sampler.state_dict())
    assert resumed.state_dict()["sampler_contract_version"] == 3
    assert resumed.state_dict()["full_schedule_contract"] == "task_contiguous_stream_v1"
    assert list(resumed) == list(sampler)
    resumed.advance()
    assert resumed.epoch == 23
    assert resumed.batch_cursor == 0


def test_full_sampler_can_share_task_order_but_shuffle_rows_independently() -> None:
    instruction = torch.repeat_interleave(torch.arange(4), 13)
    episode = torch.arange(len(instruction)) // 3
    kwargs = dict(
        instruction_id=instruction,
        episode_id=episode,
        task_weights=torch.ones(4),
        batch_size=8,
        sampling_mode="full",
        task_order_seed=17,
    )
    va = TaskLocalityWeightedSampler(**kwargs, seed=0)
    world = TaskLocalityWeightedSampler(**kwargs, seed=1)

    va_batches = list(va)
    world_batches = list(world)
    va_task_rows = [
        [va.task_ids[index] for index in batch] for batch in va_batches
    ]
    world_task_rows = [
        [world.task_ids[index] for index in batch] for batch in world_batches
    ]
    assert va_task_rows == world_task_rows
    assert va_batches != world_batches

    incompatible = TaskLocalityWeightedSampler(
        **{**kwargs, "task_order_seed": 18}, seed=0
    )
    with pytest.raises(ValueError, match="task_order_seed"):
        incompatible.load_state_dict(va.state_dict())


def test_full_sampler_rejects_uneven_ddp_tail() -> None:
    with pytest.raises(ValueError, match="divide evenly across world_size"):
        TaskLocalityWeightedSampler(
            instruction_id=torch.zeros(11, dtype=torch.long),
            episode_id=torch.arange(11),
            task_weights=torch.ones(1),
            batch_size=8,
            sampling_mode="full",
            world_size=2,
        )


def test_full_sampler_even_tail_preserves_manual_ddp_mean_gradient() -> None:
    n_rows = 10_722
    kwargs = dict(
        instruction_id=torch.arange(n_rows) % 49,
        episode_id=torch.arange(n_rows) // 7,
        task_weights=torch.ones(49),
        batch_size=48,
        seed=41,
        sampling_mode="full",
    )
    global_tail = list(TaskLocalityWeightedSampler(**kwargs))[-1]
    rank0_tail = list(
        TaskLocalityWeightedSampler(**kwargs, rank=0, world_size=2)
    )[-1]
    rank1_tail = list(
        TaskLocalityWeightedSampler(**kwargs, rank=1, world_size=2)
    )[-1]
    values = torch.linspace(-1.0, 1.0, n_rows)
    global_parameter = torch.nn.Parameter(torch.tensor(0.75))
    rank0_parameter = torch.nn.Parameter(global_parameter.detach().clone())
    rank1_parameter = torch.nn.Parameter(global_parameter.detach().clone())

    (global_parameter * values[global_tail]).square().mean().backward()
    (rank0_parameter * values[rank0_tail]).square().mean().backward()
    (rank1_parameter * values[rank1_tail]).square().mean().backward()

    # manual_post_backward_grad_allreduce_v1 averages the two local gradients.
    averaged = (rank0_parameter.grad + rank1_parameter.grad) / 2
    torch.testing.assert_close(averaged, global_parameter.grad)


def test_mixed_sampler_has_four_tasks_and_fixed_25_percent_anchors() -> None:
    instruction = torch.repeat_interleave(torch.arange(8), 20)
    episode = torch.arange(160) // 2
    kwargs = dict(
        instruction_id=instruction,
        episode_id=episode,
        task_weights=torch.ones(8),
        batch_size=16,
        seed=43,
        sampling_mode="mixed",
        mixed_tasks_per_batch=4,
        anchor_replay_fraction=0.25,
        task_order_seed=5,
        block_batches=2,
    )
    sampler = TaskLocalityWeightedSampler(**kwargs)
    schedule = list(sampler)
    assert len(schedule) == len(sampler) == 10
    for batch in schedule:
        real = [-index - 1 if index < 0 else index for index in batch]
        counts = Counter(sampler.task_ids[index] for index in real)
        assert len(counts) == 4
        assert set(counts.values()) == {4}
        assert sum(index < 0 for index in batch) == 4

    task_sets = [
        frozenset(sampler.task_ids[index] for index in real)
        for batch in schedule
        for real in [[-index - 1 if index < 0 else index for index in batch]]
    ]
    assert all(task_sets[start] == task_sets[start + 1] for start in range(0, 10, 2))
    assert task_sets[0].isdisjoint(task_sets[2])

    anchors0 = {
        -index - 1 for batch in schedule for index in batch if index < 0
    }
    sampler.advance(len(sampler))
    anchors1 = {
        -index - 1 for batch in sampler for index in batch if index < 0
    }
    assert anchors1 == anchors0


def test_mixed_sampler_keeps_dagger_rows_out_of_fixed_anchors() -> None:
    instruction = torch.repeat_interleave(torch.arange(8), 20)
    eligible = torch.ones(160, dtype=torch.bool)
    for task in range(8):
        eligible[task * 20 + 10 : task * 20 + 20] = False
    sampler = TaskLocalityWeightedSampler(
        instruction_id=instruction,
        episode_id=torch.arange(160) // 2,
        task_weights=torch.ones(8),
        batch_size=16,
        seed=43,
        sampling_mode="mixed",
        mixed_tasks_per_batch=4,
        anchor_replay_fraction=0.25,
        anchor_eligible=eligible,
    )
    schedule = list(sampler)
    anchors = [-index - 1 for batch in schedule for index in batch if index < 0]
    fresh = [index for batch in schedule for index in batch if index >= 0]
    assert all(bool(eligible[index]) for index in anchors)
    assert any(not bool(eligible[index]) for index in fresh)


def test_mixed_sampler_keeps_all_tasks_on_both_data_parallel_ranks() -> None:
    instruction = torch.repeat_interleave(torch.arange(8), 20)
    episode = torch.arange(160) // 2
    kwargs = dict(
        instruction_id=instruction,
        episode_id=episode,
        task_weights=torch.ones(8),
        batch_size=16,
        seed=47,
        sampling_mode="mixed",
        mixed_tasks_per_batch=4,
        anchor_replay_fraction=0.25,
        world_size=2,
    )
    rank0 = TaskLocalityWeightedSampler(**kwargs, rank=0)
    rank1 = TaskLocalityWeightedSampler(**kwargs, rank=1)
    for left, right in zip(rank0, rank1, strict=True):
        for batch in (left, right):
            real = [-index - 1 if index < 0 else index for index in batch]
            counts = Counter(rank0.task_ids[index] for index in real)
            assert len(counts) == 4
            assert set(counts.values()) == {2}


def test_pcgrad_removes_opposite_gradients_and_keeps_aligned_gradients() -> None:
    conflicting = torch.nn.Parameter(torch.tensor(1.0))
    stats = backward_pcgrad(
        [conflicting, -conflicting], [("conflicting", conflicting)], seed=3
    )
    torch.testing.assert_close(conflicting.grad, torch.tensor(0.0))
    assert stats == {"conflicts": 2, "comparisons": 2}

    aligned = torch.nn.Parameter(torch.tensor(1.0))
    stats = backward_pcgrad(
        [aligned, 2.0 * aligned], [("aligned", aligned)], seed=3
    )
    torch.testing.assert_close(aligned.grad, torch.tensor(1.5))
    assert stats == {"conflicts": 0, "comparisons": 2}


def test_world_gradient_cannot_oppose_pcgrad_action_direction() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    action_only = torch.nn.Parameter(torch.tensor(1.0))
    world_only = torch.nn.Parameter(torch.tensor(1.0))
    stats, world_loss = backward_pcgrad(
        [shared[0] + action_only, shared[1] + action_only],
        [
            ("shared", shared),
            ("action_only", action_only),
            ("world_only", world_only),
        ],
        seed=3,
        auxiliary_loss_or_forward=lambda: (
            -2.0 * shared[0] + 3.0 * shared[1] + 4.0 * world_only
        ),
    )
    torch.testing.assert_close(shared.grad, torch.tensor([0.5, 3.5]))
    torch.testing.assert_close(action_only.grad, torch.tensor(1.0))
    torch.testing.assert_close(world_only.grad, torch.tensor(4.0))
    torch.testing.assert_close(world_loss, torch.tensor(5.0))
    assert stats["world_conflicts"] == 1
    assert stats["world_comparisons"] == 2


def test_separate_action_world_pcgrad_only_guards_shared_dino() -> None:
    action = torch.nn.Parameter(torch.tensor(1.0))
    world = torch.nn.Parameter(torch.tensor(1.0))
    dino = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    named = [
        ("model.action", action),
        ("model.wmrm.weight", world),
        ("main_vision_backbone.model.weight", dino),
    ]
    action_private, world_private, shared = (
        partition_separate_pcgrad_parameters(named)
    )
    backward_pcgrad(
        [action + dino[0], 2.0 * action + dino[0]],
        [*action_private, *shared],
        seed=3,
    )
    action_gradients = pop_update_gradients([*action_private, *shared])
    backward_pcgrad(
        [world - dino[0] + dino[1], 2.0 * world - dino[0] + dino[1]],
        [*world_private, *shared],
        seed=3,
    )
    stats = merge_separate_pcgrad_gradients(
        action_private, shared, action_gradients
    )
    torch.testing.assert_close(action.grad, torch.tensor(1.5))
    torch.testing.assert_close(world.grad, torch.tensor(1.5))
    torch.testing.assert_close(dino.grad, torch.tensor([1.0, 1.0]))
    assert stats["dino_projected"] == 1
    assert stats["dino_cosine"] < 0.0
    assert stats["dino_post_cosine"] == pytest.approx(0.0, abs=1e-6)
