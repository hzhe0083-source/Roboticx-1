from __future__ import annotations

from collections import Counter

import pytest
import torch

from train import TaskLocalityWeightedSampler


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
