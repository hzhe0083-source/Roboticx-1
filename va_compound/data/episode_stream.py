"""Deterministic, rank-aligned episode windows for truncated recurrent training."""
from __future__ import annotations

import random

import torch


class EpisodeStreamDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row, stream, active = index
        item = dict(self.dataset[row])
        item["stream_id"] = torch.tensor(stream, dtype=torch.long)
        item["stream_active"] = torch.tensor(active, dtype=torch.bool)
        if not active:
            for key in ("decision_valid_mask", "action_valid_mask", "world_target_valid_mask", "world_rank_shuffle_mask"):
                item[key] = torch.zeros_like(item[key])
            item["decision_count"] = torch.tensor(0, dtype=torch.long)
        return item


class EpisodeWindowBatchSampler:
    def __init__(self, payload, batch_size, seed=0, mixed_tasks_per_batch=2, rank=0, world_size=1):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mixed_tasks_per_batch = int(mixed_tasks_per_batch)
        self.rank, self.world_size = int(rank), int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("invalid episode sampler rank")
        divisor = self.mixed_tasks_per_batch * self.world_size
        if divisor < 1 or self.batch_size < divisor or self.batch_size % divisor:
            raise ValueError("episode batch must divide into tasks and ranks")
        self.rows = {}
        for row, (task, episode, start) in enumerate(zip(payload["instruction_id"].tolist(), payload["episode_id"].tolist(), payload["crop_start"].tolist(), strict=True)):
            self.rows.setdefault(task, {}).setdefault(episode, []).append((start, row))
        for episodes in self.rows.values():
            for episode, rows in episodes.items():
                episodes[episode] = [row for _, row in sorted(rows)]
        if len(self.rows) % self.mixed_tasks_per_batch:
            raise ValueError("task count must divide into complete episode task groups")
        self.epoch = self.batch_cursor = 0
        self.dataset_content_identity = None
        self._cached_epoch = None
        self._schedule = []

    def bind_dataset_content_identity(self, identity):
        self.dataset_content_identity = dict(identity)

    def _build(self):
        if self._cached_epoch == self.epoch:
            return
        rng = random.Random(self.seed + self.epoch)
        tasks = sorted(self.rows)
        slots = self.batch_size // self.mixed_tasks_per_batch
        schedule = []
        for offset in range(0, len(tasks), self.mixed_tasks_per_batch):
            group = tasks[offset:offset + self.mixed_tasks_per_batch]
            queues = {}
            for task in group:
                buckets = {}
                for episode, rows in self.rows[task].items():
                    buckets.setdefault(len(rows), []).append(episode)
                episodes = []
                for length in sorted(buckets):
                    bucket = sorted(buckets[length])
                    rng.shuffle(bucket)
                    episodes.extend(bucket)
                queues[task] = episodes
            for cohort in range(0, max(map(len, queues.values())), slots):
                streams = []
                for task_index, task in enumerate(group):
                    for slot in range(slots):
                        position = cohort + slot
                        episode = queues[task][position] if position < len(queues[task]) else None
                        rows = self.rows[task][episode] if episode is not None else []
                        fallback = next(iter(self.rows[task].values()))[0]
                        streams.append((task_index * slots + slot, rows, fallback))
                for window in range(max(len(rows) for _, rows, _ in streams)):
                    batch = []
                    for stream, rows, fallback in streams:
                        if stream % self.world_size != self.rank:
                            continue
                        active = window < len(rows)
                        row = rows[window] if active else (rows[-1] if rows else fallback)
                        batch.append((row, stream, active))
                    schedule.append(batch)
        self._schedule = schedule
        self._cached_epoch = self.epoch

    def __len__(self):
        self._build()
        return len(self._schedule)

    def __iter__(self):
        self._build()
        yield from self._schedule[self.batch_cursor:]

    def advance(self, batches=1):
        for _ in range(batches):
            self.batch_cursor += 1
            if self.batch_cursor == len(self):
                self.epoch += 1
                self.batch_cursor = 0
                self._cached_epoch = None
            elif self.batch_cursor > len(self):
                raise ValueError("episode sampler cursor overflow")

    def state_dict(self):
        return {"contract": "episode_tbptt8_v1", "epoch": self.epoch,
                "batch_cursor": self.batch_cursor, "seed": self.seed,
                "batch_size": self.batch_size, "mixed_tasks": self.mixed_tasks_per_batch,
                "world_size": self.world_size, "dataset_content_identity": self.dataset_content_identity}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {"epoch", "batch_cursor"}:
            if state.get(key) != expected[key]:
                raise ValueError(f"episode sampler {key} mismatch")
        self.epoch = int(state["epoch"])
        self.batch_cursor = int(state["batch_cursor"])
        self._cached_epoch = None
        if self.epoch < 0 or not 0 <= self.batch_cursor < len(self):
            raise ValueError("invalid episode sampler position")
