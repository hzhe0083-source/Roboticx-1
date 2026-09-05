"""Full legal-start coverage with randomized, chronological TBPTT windows."""
from __future__ import annotations

import random

import numpy as np
import torch

from va_compound.data.episode_stream import EpisodeWindowBatchSampler

SAMPLING_CONTRACT = "all_starts_random_tbptt8_v1"
MEMORY_CONTRACT = "offset_replay_tbptt8_v1"


class AllStartsWindowBatchSampler(EpisodeWindowBatchSampler):
    def __init__(self, payload, batch_size, seed=0, mixed_tasks_per_batch=2, rank=0, world_size=1):
        starts = payload["crop_start"]
        replay_payload = dict(payload, episode_id=payload["episode_id"] * 15 + starts.remainder(15))
        super().__init__(replay_payload, batch_size, seed, mixed_tasks_per_batch, rank, world_size)
        for episodes in self.rows.values():
            for replay, rows in episodes.items():
                actual = starts[rows].tolist()
                expected = list(range(replay % 15, actual[-1] + 1, 15))
                if actual != expected:
                    raise ValueError("all-start replay is missing or duplicates a decision")

    def _build(self):
        if self._cached_epoch == self.epoch:
            return
        rng = random.Random(self.seed + self.epoch)
        windows = {}
        for task, episodes in sorted(self.rows.items()):
            windows[task] = {}
            for replay, rows in sorted(episodes.items()):
                first = rng.randint(1, min(8, len(rows)))
                chunks = [tuple(rows[:first])]
                chunks.extend(tuple(rows[i:i + 8]) for i in range(first, len(rows), 8))
                windows[task][replay] = chunks
        tasks = sorted(windows)
        slots = self.batch_size // self.mixed_tasks_per_batch
        schedule = []
        for offset in range(0, len(tasks), self.mixed_tasks_per_batch):
            group = tasks[offset:offset + self.mixed_tasks_per_batch]
            queues = {}
            for task in group:
                buckets = {}
                for replay, chunks in windows[task].items():
                    buckets.setdefault(len(chunks), []).append(replay)
                queues[task] = []
                lengths = sorted(buckets)
                rng.shuffle(lengths)
                for length in lengths:
                    rng.shuffle(buckets[length])
                    queues[task].extend(buckets[length])
            for cohort in range(0, max(map(len, queues.values())), slots):
                streams = []
                for task_index, task in enumerate(group):
                    fallback = next(iter(windows[task].values()))[0]
                    for slot in range(slots):
                        pos = cohort + slot
                        chunks = windows[task][queues[task][pos]] if pos < len(queues[task]) else []
                        streams.append((task_index * slots + slot, chunks, fallback))
                for window in range(max(len(chunks) for _, chunks, _ in streams)):
                    batch = []
                    for stream, chunks, fallback in streams:
                        if stream % self.world_size == self.rank:
                            active = window < len(chunks)
                            batch.append((chunks[window] if active else fallback, stream, active))
                    schedule.append(batch)
        self._schedule = schedule
        self._cached_epoch = self.epoch

    def epoch_lengths(self, epochs):
        if epochs < 1:
            raise ValueError("epoch count must be positive")
        old = self.epoch, self._cached_epoch, self._schedule
        try:
            lengths = []
            for epoch in range(epochs):
                self.epoch, self._cached_epoch = epoch, None
                lengths.append(len(self))
            return lengths
        finally:
            self.epoch, self._cached_epoch, self._schedule = old

    def state_dict(self):
        return dict(super().state_dict(), contract=SAMPLING_CONTRACT)


class AllStartsStreamDataset:
    _TEMPORAL = {
        "actions", "proprio", "previous_action", "vision_tokens", "dino_last4",
        "action_valid_mask", "recovery_mask", "world_target_valid_mask",
        "world_rank_shuffle_action", "world_rank_shuffle_mask", "world_target_map",
        "world_state_delta", "frames", "world_target_frames", "frame_cache_rows",
    }
    _ZERO_PAD = {"action_valid_mask", "recovery_mask", "world_target_valid_mask",
                 "world_rank_shuffle_mask", "world_state_delta"}

    def __init__(self, dataset):
        self.dataset = dataset
        payload = dataset.payload
        self.starts = payload["crop_start"]
        self.episodes = payload["episode_id"]
        self.last = {}
        for episode, start in zip(self.episodes.tolist(), self.starts.tolist(), strict=True):
            key = episode * 15 + start % 15
            self.last[key] = max(start, self.last.get(key, start))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rows, stream, active = index
        if not 1 <= len(rows) <= 8:
            raise ValueError("all-start window must contain 1..8 decisions")
        episode = int(self.episodes[rows[0]])
        start = int(self.starts[rows[0]])
        replay = episode * 15 + start % 15
        for i, row in enumerate(rows):
            if int(self.episodes[row]) != episode or int(self.starts[row]) != start + i * 15:
                raise ValueError("all-start window crosses a replay boundary")
        items = [self.dataset[row] for row in rows]
        item = dict(items[0])
        for key, first in items[0].items():
            temporal = key in self._TEMPORAL or (
                key == "language_hidden" and first.ndim == 3
            ) or (key == "language_mask" and first.ndim == 2)
            if not temporal:
                continue
            if isinstance(first, torch.Tensor):
                value = torch.cat([x[key] for x in items], dim=0)
                pad = value[-1:].expand(8 - len(rows), *value.shape[1:])
                if key in self._ZERO_PAD:
                    pad = torch.zeros_like(pad)
                value = torch.cat((value, pad), dim=0)
                if not active and key in self._ZERO_PAD:
                    value = torch.zeros_like(value)
            elif isinstance(first, np.ndarray):
                value = np.concatenate([x[key] for x in items], axis=0)
                value = np.concatenate((value, np.repeat(value[-1:], 8 - len(rows), axis=0)))
            else:
                raise TypeError(f"unsupported temporal field {key}")
            item[key] = value
        count = len(rows) if active else 0
        item.update(
            stream_id=torch.tensor(stream), stream_active=torch.tensor(active),
            episode_id=torch.tensor(episode), replay_id=torch.tensor(replay),
            replay_offset=torch.tensor(start % 15), crop_start=torch.tensor(start),
            decision_count=torch.tensor(count), decision_valid_mask=torch.arange(8) < count,
            episode_start=torch.tensor(start < 15),
            episode_end=torch.tensor(start + (len(rows) - 1) * 15 == self.last[replay]),
        )
        return item
