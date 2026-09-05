from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from va_compound.data.samplers import TaskLocalityWeightedSampler, TaskWeightedSampler

def iter_forever(loader):
    """无限循环迭代 DataLoader（epoch 结束自动重启）。"""
    while True:
        yield from iter(loader)


def next_peer_joint_batches(
    va_iterator,
    va_loader,
    world_iterator,
    world_loader,
):
    """Fetch one independent VA batch and one independent World batch."""

    def next_or_restart(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    va_batch, va_iterator = next_or_restart(va_iterator, va_loader)
    world_batch, world_iterator = next_or_restart(world_iterator, world_loader)
    if va_batch is world_batch:
        raise RuntimeError("peer joint streams returned the same batch object")
    return va_batch, world_batch, va_iterator, world_iterator


class PeerJointBatchPrefetcher:
    """Keep a bounded FIFO of VA+World batches on one background thread."""

    def __init__(
        self,
        va_iterator,
        va_loader,
        world_iterator,
        world_loader,
        *,
        depth: int = 1,
    ) -> None:
        if depth < 1:
            raise ValueError("peer batch prefetch depth must be positive")
        self.va_iterator = va_iterator
        self.va_loader = va_loader
        self.world_iterator = world_iterator
        self.world_loader = world_loader
        self.depth = int(depth)
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="peer-batch-prefetch"
        )
        self._futures = deque()

    @property
    def queued_batches(self) -> int:
        return len(self._futures)

    def _fetch_next(self):
        result = next_peer_joint_batches(
            self.va_iterator,
            self.va_loader,
            self.world_iterator,
            self.world_loader,
        )
        self.va_iterator = result[2]
        self.world_iterator = result[3]
        return result

    def submit(self) -> None:
        if len(self._futures) >= self.depth:
            raise RuntimeError("peer batch prefetch queue is full")
        self._futures.append(self._pool.submit(self._fetch_next))

    def fill(self, max_batches: int) -> int:
        """Fill up to both ``depth`` and the caller's safe batch limit."""
        if max_batches < 0:
            raise ValueError("peer batch prefetch max_batches must be non-negative")
        target = min(self.depth, int(max_batches))
        added = 0
        while len(self._futures) < target:
            self.submit()
            added += 1
        return added

    def result(self):
        if not self._futures:
            raise RuntimeError("peer batch prefetch has no in-flight batch")
        return self._futures.popleft().result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def peer_prefetch_must_wait_for_commit(*samplers) -> bool:
    """Do not rebuild an exhausted iterator until its sampler enters next epoch."""
    return any(
        isinstance(sampler, (TaskLocalityWeightedSampler, TaskWeightedSampler))
        and sampler.batch_cursor + 1 >= len(sampler)
        for sampler in samplers
        if sampler is not None
    )


def peer_prefetch_fill_limit(
    remaining_steps: int,
    *samplers,
    current_batch_consumed: bool = False,
) -> int:
    """Cap queued fetches to this run and the current committed sampler epoch."""
    if remaining_steps < 0:
        raise ValueError("remaining prefetch steps must be non-negative")
    locality_samplers = [
        sampler
        for sampler in samplers
        if isinstance(sampler, (TaskLocalityWeightedSampler, TaskWeightedSampler))
    ]
    if not locality_samplers:
        return int(remaining_steps)
    consumed = int(current_batch_consumed)
    epoch_remaining = min(
        len(sampler) - sampler.batch_cursor - consumed
        for sampler in locality_samplers
    )
    return min(int(remaining_steps), max(0, epoch_remaining))
