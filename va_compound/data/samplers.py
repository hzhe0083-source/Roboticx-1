from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import math
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from va_compound.exact_resume import _normalize_contract_value
from va_compound.data.feature_dataset import build_pair_groups



class TaskWeightedSampler(Sampler[list[int]]):
    """难度分层采样（E7，2026-08-09，sota_plan_v2.md 第 11 项）：

    per-sample 权重（instruction_id → MT50 难度：easy 0.5 / med 1.0 /
    hard 2.0 / vh 3.0，除以任务窗口数消除长度偏置，Codex P1-2）多项式抽样；
    每 epoch 有放回（replacement=True，实现困难任务过采样）抽取 n 个样本、
    分批 yield，最后不足一批丢弃（等效 drop_last）。

    ``__iter__`` 不自行推进 cursor；只有优化器更新成功后由主循环调用
    :meth:`advance`，使 DINO-main weighted 路径也能 exact-resume。
    2026-08-16 的 6k 档案 ``sampler_state=None``：恢复时从 epoch=0 重开，
    不根据 global_step 反推（6k→20k 续训本身就是从 epoch 0 重开的）。
    """

    def __init__(self, per_sample_weights: Tensor, batch_size: int, seed: int = 0) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if per_sample_weights.ndim != 1:
            raise ValueError("per_sample_weights must be 1-D")
        self.weights = per_sample_weights.to(torch.float64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_cursor = 0
        self.dataset_content_identity: dict | None = None

    def __len__(self) -> int:
        return max(1, len(self.weights) // self.batch_size)

    def _weights_fingerprint(self) -> str:
        return hashlib.sha256(
            self.weights.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

    def _build_epoch(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        n = len(self.weights)
        indices = torch.multinomial(
            self.weights, n, replacement=True, generator=generator
        ).tolist()
        return [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices) - self.batch_size + 1, self.batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        schedule = self._build_epoch()
        yield from schedule[self.batch_cursor :]

    def advance(self, batches: int = 1) -> None:
        if batches < 0:
            raise ValueError("batches must be non-negative")
        total = self.batch_cursor + int(batches)
        self.epoch += total // len(self)
        self.batch_cursor = total % len(self)

    def state_dict(self) -> dict:
        return {
            "sampler_kind": "task_weighted",
            "sampler_contract_version": 1,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "n_weights": int(self.weights.numel()),
            "weights_sha256": self._weights_fingerprint(),
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "sampler_kind": "task_weighted",
            "sampler_contract_version": 1,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "n_weights": int(self.weights.numel()),
            "weights_sha256": self._weights_fingerprint(),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"sampler state mismatch on {key}: {state.get(key)!r} != {value!r}"
                )
        epoch = int(state.get("epoch", -1))
        cursor = int(state.get("batch_cursor", -1))
        if epoch < 0 or not 0 <= cursor < len(self):
            raise ValueError(f"invalid sampler epoch/cursor: {epoch}/{cursor}")
        self.epoch = epoch
        self.batch_cursor = cursor


class TaskLocalityWeightedSampler(Sampler[list[int]]):
    """有限、可恢复的任务局部性采样器，且在任务块内按 episode 均衡。

    ``weighted`` / ``balanced`` 每个 epoch 严格产生 ``N // batch_size`` 个
    batch；``full`` 则产生 ``ceil(N / batch_size)`` 个 batch，并让每一行
    恰好出现一次。weighted/balanced 抽样块含至多 ``block_batches`` 个同任务
    batch，在 JPEG 解码局部性与跨任务曝光之间取折中。``full`` 不拆任务块：
    随机任务顺序后把一个任务的全部行连续铺平，再按 global batch 切分，
    因此任务边界批可能混合相邻任务。``mixed`` 每批均匀选择多个不同任务，
    每个任务固定保留一部分 epoch-0 行作为 replay anchor。
    块内轮询 episode，不再让长轨迹因滑窗更多而被额外过采样。
    ``__iter__`` 不自行推进 cursor；只有优化器更新成功后由主循环调用
    :meth:`advance`，使 checkpoint 能精确指向“已完成更新”的下一批。
    """

    def __init__(
        self,
        instruction_id: Tensor,
        episode_id: Tensor,
        task_weights: Tensor,
        batch_size: int,
        seed: int = 0,
        block_batches: int = 16,
        sampling_mode: str = "weighted",
        *,
        task_order_seed: int | None = None,
        mixed_tasks_per_batch: int = 4,
        anchor_replay_fraction: float = 0.25,
        rank: int = 0,
        world_size: int = 1,
        epoch_dataset: Dataset | None = None,
        anchor_eligible: Tensor | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if block_batches < 1:
            raise ValueError("block_batches must be positive")
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside world_size {world_size}")
        # ``batch_size`` stays the *global* batch so ``__len__`` keeps counting
        # optimizer steps per epoch and the resume cursor keeps meaning the same
        # thing on any number of ranks.  Each rank yields a disjoint stride of
        # every global batch, which keeps both ranks inside the same task block:
        # a rank drifting onto its own task would double the resident decoded
        # frames, and two tasks per rank does not fit the host memory budget.
        if batch_size % world_size:
            raise ValueError(
                f"global batch {batch_size} must divide across {world_size} ranks"
            )
        if instruction_id.ndim != 1 or episode_id.ndim != 1:
            raise ValueError("instruction_id/episode_id must be 1-D")
        if instruction_id.shape != episode_id.shape or instruction_id.numel() == 0:
            raise ValueError("instruction_id/episode_id must have the same non-zero length")
        if task_weights.ndim != 1:
            raise ValueError("task_weights must be 1-D")
        if sampling_mode not in {"weighted", "balanced", "full", "mixed"}:
            raise ValueError(
                "sampling_mode must be 'weighted', 'balanced', 'full', or 'mixed'"
            )
        self.task_ids = [int(value) for value in instruction_id.tolist()]
        self.episode_ids = [int(value) for value in episode_id.tolist()]
        self.task_w = task_weights.to(torch.float64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.block_batches = int(block_batches)
        self.sampling_mode = sampling_mode
        self.task_order_seed = (
            self.seed if task_order_seed is None else int(task_order_seed)
        )
        self.mixed_tasks_per_batch = int(mixed_tasks_per_batch)
        self.anchor_replay_fraction = float(anchor_replay_fraction)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch_dataset = epoch_dataset
        if anchor_eligible is None:
            anchor_eligible = torch.ones_like(instruction_id, dtype=torch.bool)
        if anchor_eligible.shape != instruction_id.shape or anchor_eligible.dtype != torch.bool:
            raise ValueError("anchor_eligible must be a bool tensor aligned with rows")
        self.anchor_eligible = [bool(value) for value in anchor_eligible.tolist()]
        self.epoch = 0
        self.batch_cursor = 0
        self.by_task_episode: dict[int, dict[int, list[int]]] = {}
        for index, (task, episode) in enumerate(zip(self.task_ids, self.episode_ids, strict=True)):
            self.by_task_episode.setdefault(task, {}).setdefault(episode, []).append(index)
        self.tasks = sorted(self.by_task_episode)
        if self.tasks[-1] >= len(self.task_w) or bool((self.task_w[self.tasks] <= 0).any()):
            raise ValueError("task_weights must contain a positive entry for every task id")
        if self.sampling_mode in {"balanced", "mixed"}:
            active_weights = self.task_w[self.tasks]
            if not bool(torch.all(active_weights == active_weights[0])):
                raise ValueError(
                    f"{self.sampling_mode} sampling requires equal active task weights"
                )
        if self.sampling_mode == "mixed":
            if not 2 <= self.mixed_tasks_per_batch <= len(self.tasks):
                raise ValueError(
                    "mixed_tasks_per_batch must be in [2, number of active tasks]"
                )
            if self.batch_size % self.mixed_tasks_per_batch:
                raise ValueError(
                    "mixed batch size must divide across mixed_tasks_per_batch"
                )
            per_task = self.batch_size // self.mixed_tasks_per_batch
            if per_task % self.world_size:
                raise ValueError(
                    "mixed per-task rows must divide evenly across world_size"
                )
            anchors = round(per_task * self.anchor_replay_fraction)
            if not 0 <= self.anchor_replay_fraction < 1 or not 0 <= anchors < per_task:
                raise ValueError(
                    "anchor_replay_fraction must leave at least one fresh row per task"
                )
            self.mixed_rows_per_task = per_task
            self.anchor_rows_per_task = anchors
            anchor_rng = random.Random(self.task_order_seed + 0xA11CE)
            self.anchor_rows: dict[int, list[int]] = {}
            for task in self.tasks:
                episodes = list(self.by_task_episode[task].values())
                anchor_rng.shuffle(episodes)
                candidates = [
                    next(index for index in rows if self.anchor_eligible[index])
                    for rows in episodes
                    if any(self.anchor_eligible[index] for index in rows)
                ]
                if len(candidates) < anchors:
                    candidates = [
                        index
                        for rows in self.by_task_episode[task].values()
                        for index in rows
                        if self.anchor_eligible[index]
                    ]
                    anchor_rng.shuffle(candidates)
                if len(candidates) < anchors:
                    raise ValueError(f"task {task} has too few anchor-eligible rows")
                self.anchor_rows[task] = candidates[:anchors]
        self.task_probs = torch.stack([self.task_w[t] for t in self.tasks])
        self.task_probs = self.task_probs / self.task_probs.sum().clamp_min(1e-12)
        self._n = len(self.task_ids)
        if self.sampling_mode == "full" and self._n % self.world_size:
            raise ValueError(
                "full sampling requires dataset size to divide evenly across "
                f"world_size: {self._n} rows / {self.world_size} ranks"
            )
        digest_input = torch.stack(
            (instruction_id.to(torch.int64), episode_id.to(torch.int64)), dim=1
        ).cpu().contiguous().numpy().tobytes()
        self.dataset_fingerprint = hashlib.sha256(digest_input).hexdigest()
        self.dataset_content_identity: dict | None = None

    def bind_dataset_content_identity(self, identity: dict) -> None:
        """Cache the expensive payload identity and bind sampler state to it."""
        normalized = _normalize_contract_value(identity)
        encoded = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        self.dataset_content_identity = normalized
        self.dataset_fingerprint = hashlib.sha256(encoded).hexdigest()

    def __len__(self) -> int:
        if self.sampling_mode in {"full", "mixed"}:
            return math.ceil(self._n / self.batch_size)
        return max(1, self._n // self.batch_size)

    def _choose_task(self, rng: random.Random, previous: int | None) -> int:
        if len(self.tasks) == 1:
            return self.tasks[0]
        # Do not forbid the previous task outright: with only two tasks that
        # collapses every requested weighting to forced 1:1 alternation. The
        # run-length cap is already enforced by one fixed-size block per draw;
        # adjacent same-task draws simply remain two independently balanced
        # blocks and preserve the requested long-run task probability.
        del previous
        return int(
            rng.choices(
                self.tasks,
                weights=[float(self.task_w[task]) for task in self.tasks],
                k=1,
            )[0]
        )

    def _build_epoch(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self.sampling_mode == "full":
            # Keep every task contiguous so each decoded task is visited once
            # per epoch.  A separate task-order RNG lets shared VA/World streams
            # align task switches while their row RNGs remain independent.
            task_order = list(self.tasks)
            task_rng = random.Random(self.task_order_seed + self.epoch)
            task_rng.shuffle(task_order)
            epoch_rows: list[int] = []
            for task in task_order:
                episode_queues = [
                    list(rows) for rows in self.by_task_episode[task].values()
                ]
                for queue in episode_queues:
                    rng.shuffle(queue)
                rng.shuffle(episode_queues)
                task_rows: list[int] = []
                active = episode_queues
                while active:
                    rng.shuffle(active)
                    next_active: list[list[int]] = []
                    for queue in active:
                        task_rows.append(queue.pop())
                        if queue:
                            next_active.append(queue)
                    active = next_active
                epoch_rows.extend(task_rows)

            batches = [
                epoch_rows[start : start + self.batch_size]
                for start in range(0, len(epoch_rows), self.batch_size)
            ]
            if len(batches) != len(self) or sum(map(len, batches)) != self._n:
                raise RuntimeError("full sampler built an invalid epoch schedule")
            return batches

        if self.sampling_mode == "mixed":
            queues: dict[tuple[int, int], list[int]] = {}
            offsets: dict[tuple[int, int], int] = {}
            for task, episodes in self.by_task_episode.items():
                for episode, rows in episodes.items():
                    anchors = set(self.anchor_rows[task])
                    queue = [row for row in rows if row not in anchors]
                    if not queue:
                        continue
                    rng.shuffle(queue)
                    queues[(task, episode)] = queue
                    offsets[(task, episode)] = 0

            def take_rows(task: int, count: int) -> list[int]:
                episodes = [
                    episode
                    for episode in self.by_task_episode[task]
                    if (task, episode) in queues
                ]
                selected: list[int] = []
                while len(selected) < count:
                    rng.shuffle(episodes)
                    for episode in episodes:
                        key = (task, episode)
                        queue = queues[key]
                        offset = offsets[key]
                        if offset >= len(queue):
                            rng.shuffle(queue)
                            offset = 0
                        selected.append(queue[offset])
                        offsets[key] = offset + 1
                        if len(selected) == count:
                            break
                return selected

            task_rng = random.Random(self.task_order_seed + self.epoch)
            task_pool: list[int] = []

            def choose_tasks() -> list[int]:
                chosen: list[int] = []
                while len(chosen) < self.mixed_tasks_per_batch:
                    if not task_pool:
                        task_pool.extend(self.tasks)
                        task_rng.shuffle(task_pool)
                    task = task_pool.pop()
                    if task in chosen:
                        task_pool.insert(0, task)
                        continue
                    chosen.append(task)
                return chosen

            fresh = self.mixed_rows_per_task - self.anchor_rows_per_task
            batches: list[list[int]] = []
            while len(batches) < len(self):
                block_tasks = choose_tasks()
                for _ in range(min(self.block_batches, len(self) - len(batches))):
                    batch: list[int] = []
                    for task in block_tasks:
                        batch.extend(take_rows(task, fresh))
                        # Negative indices address fixed rows at dataset epoch 0.
                        batch.extend(-(index + 1) for index in self.anchor_rows[task])
                    if len(batch) != self.batch_size:
                        raise RuntimeError("mixed sampler built an invalid batch")
                    batches.append(batch)
            return batches

        # 每个 (task, episode) 维护独立无放回队列；耗尽才重洗。
        queues: dict[tuple[int, int], list[int]] = {}
        offsets: dict[tuple[int, int], int] = {}
        for task, episodes in self.by_task_episode.items():
            for episode, rows in episodes.items():
                queue = list(rows)
                rng.shuffle(queue)
                queues[(task, episode)] = queue
                offsets[(task, episode)] = 0

        def take_rows(task: int, count: int) -> list[int]:
            episodes = list(self.by_task_episode[task])
            selected: list[int] = []
            while len(selected) < count:
                rng.shuffle(episodes)
                for episode in episodes:
                    key = (task, episode)
                    queue = queues[key]
                    offset = offsets[key]
                    if offset >= len(queue):
                        rng.shuffle(queue)
                        offset = 0
                    selected.append(queue[offset])
                    offsets[key] = offset + 1
                    if len(selected) == count:
                        break
            return selected

        batches: list[list[int]] = []
        if self.sampling_mode == "balanced":
            # Exact per-task exposure in every epoch.  For 59,557 rows,
            # batch=16 and 49 tasks this gives 47 tasks x 76 batches and
            # 2 tasks x 75 batches; which tasks receive the shorter quota is
            # deterministically reshuffled from seed + epoch.
            task_order = list(self.tasks)
            rng.shuffle(task_order)
            base, remainder = divmod(len(self), len(task_order))
            quotas = {
                task: base + int(rank < remainder)
                for rank, task in enumerate(task_order)
            }
            blocks: list[tuple[int, int]] = []
            for task in task_order:
                remaining = quotas[task]
                while remaining:
                    size = min(self.block_batches, remaining)
                    blocks.append((task, size))
                    remaining -= size
            rng.shuffle(blocks)
            for task, n_batches in blocks:
                rows = take_rows(task, n_batches * self.batch_size)
                batches.extend(
                    rows[start : start + self.batch_size]
                    for start in range(0, len(rows), self.batch_size)
                )
        else:
            previous_task: int | None = None
            while len(batches) < len(self):
                task = self._choose_task(rng, previous_task)
                n_batches = min(self.block_batches, len(self) - len(batches))
                rows = take_rows(task, n_batches * self.batch_size)
                batches.extend(
                    rows[start : start + self.batch_size]
                    for start in range(0, len(rows), self.batch_size)
                )
                previous_task = task
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.epoch_dataset is not None:
            set_epoch = getattr(self.epoch_dataset, "set_epoch", None)
            if not callable(set_epoch):
                raise TypeError("epoch_dataset must provide set_epoch(epoch)")
            set_epoch(self.epoch)
        schedule = self._build_epoch()
        prefetch_tasks = (
            getattr(self.epoch_dataset, "prefetch_task_ids", None)
            if self.epoch_dataset is not None
            else None
        )
        run_task: int | None = None
        run_batches = 0
        for position, batch in enumerate(
            schedule[self.batch_cursor :], start=self.batch_cursor
        ):
            batch_tasks = {
                self.task_ids[-index - 1 if index < 0 else index]
                for index in batch
            }
            if self.sampling_mode == "mixed" and callable(prefetch_tasks):
                prefetch_tasks(sorted(batch_tasks))
            if len(batch_tasks) == 1:
                task = next(iter(batch_tasks))
                if task == run_task:
                    run_batches += 1
                else:
                    run_task = task
                    run_batches = 1
                # The first batch has now loaded the current task.  On the
                # next batch, start loading the following task on an independent
                # worker, leaving the remaining task-local batches to hide it.
                if run_batches == 2 and callable(prefetch_tasks):
                    following = next(
                        (
                            self.task_ids[-index - 1 if index < 0 else index]
                            for future_batch in schedule[position + 1 :]
                            for index in future_batch
                            if self.task_ids[-index - 1 if index < 0 else index]
                            != task
                        ),
                        None,
                    )
                    if following is None and self.sampling_mode == "full":
                        next_order = list(self.tasks)
                        random.Random(
                            self.task_order_seed + self.epoch + 1
                        ).shuffle(next_order)
                        following = next_order[0]
                    if following is not None:
                        prefetch_tasks([following])
            else:
                run_task = None
                run_batches = 0
            yield (
                batch
                if self.world_size == 1
                else batch[self.rank :: self.world_size]
            )

    def advance(self, batches: int = 1) -> None:
        if batches < 0:
            raise ValueError("batches must be non-negative")
        total = self.batch_cursor + int(batches)
        self.epoch += total // len(self)
        self.batch_cursor = total % len(self)

    def state_dict(self) -> dict:
        state = {
            "sampler_contract_version": 3,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "block_batches": self.block_batches,
            "sampling_mode": self.sampling_mode,
            "dataset_fingerprint": self.dataset_fingerprint,
            "active_tasks": self.tasks,
            "task_weights": [float(self.task_w[task]) for task in self.tasks],
            "world_size": self.world_size,
        }
        if self.sampling_mode == "full":
            state.update(
                {
                    "full_schedule_contract": "task_contiguous_stream_v1",
                    "task_order_seed": self.task_order_seed,
                }
            )
        if self.sampling_mode == "mixed":
            state.update(
                {
                    "mixed_schedule_contract": "balanced_multitask_anchor_v1",
                    "task_order_seed": self.task_order_seed,
                    "mixed_tasks_per_batch": self.mixed_tasks_per_batch,
                    "anchor_replay_fraction": self.anchor_replay_fraction,
                    "anchor_epoch": 0,
                }
            )
        return state

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "sampler_contract_version": 3,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "block_batches": self.block_batches,
            "sampling_mode": self.sampling_mode,
            "dataset_fingerprint": self.dataset_fingerprint,
            "active_tasks": self.tasks,
            "task_weights": [float(self.task_w[task]) for task in self.tasks],
        }
        if self.sampling_mode == "full":
            expected.update(
                {
                    "full_schedule_contract": "task_contiguous_stream_v1",
                    "task_order_seed": self.task_order_seed,
                }
            )
        if self.sampling_mode == "mixed":
            expected.update(
                {
                    "mixed_schedule_contract": "balanced_multitask_anchor_v1",
                    "task_order_seed": self.task_order_seed,
                    "mixed_tasks_per_batch": self.mixed_tasks_per_batch,
                    "anchor_replay_fraction": self.anchor_replay_fraction,
                    "anchor_epoch": 0,
                }
            )
        # Absent ``world_size`` is a single-process checkpoint (the field was
        # added when data-parallel sharding landed).  A changed GPU count must
        # not silently keep the same cursor: the yielded rows would be a
        # different disjoint slice of each global batch.
        if int(state.get("world_size", 1)) != self.world_size:
            raise ValueError(
                f"sampler state mismatch on world_size: "
                f"{state.get('world_size', 1)!r} != {self.world_size!r}"
            )
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"sampler state mismatch on {key}: {state.get(key)!r} != {value!r}"
                )
        epoch = int(state.get("epoch", -1))
        cursor = int(state.get("batch_cursor", -1))
        if epoch < 0 or not 0 <= cursor < len(self):
            raise ValueError(f"invalid sampler epoch/cursor: {epoch}/{cursor}")
        self.epoch = epoch
        self.batch_cursor = cursor
