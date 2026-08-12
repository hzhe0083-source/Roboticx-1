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
