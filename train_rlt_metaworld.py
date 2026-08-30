"""RL Token (RLT) for the frozen ORA0 VLA on MetaWorld.

Stage 1 trains an encoder/decoder RL-token bottleneck on demonstrations. Stage 2
freezes the VLA and RL token, then trains a Gaussian chunk actor and twin critic
off-policy with VLA reference-action conditioning and regularization.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import queue
import random
import shutil
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eval_metaworld import (
    OBSERVATION_STRIDE,
    VISION_WINDOW,
    _apply_local_vision,
    _main_vision_encode_window,
    _reset_world_state,
    cached_task_language,
    evaluation_episode_seed,
    load_metaworld_description_to_env,
    select_eval_tasks,
    state_take_normalize,
    validate_language_features,
)
from va_compound.backbones import TimmActionVisionBackbone
from va_compound.model import VACompoundConfig, VACompoundPolicy
from va_compound.rlt import RLTokenConfig, RLTokenModule

CONTRACT = "ora0_rl_token_rlt_v2"
TOKEN_CONTRACT = "ora0_rl_token_reconstruction_v1"
PILOT_ACTION_HORIZON = 15


@dataclass(frozen=True)
class RefineConfig:
    gamma: float = 0.99
    beta: float = 5.0
    fixed_std: float = 0.05
    reference_dropout: float = 0.5
    hidden_dim: int = 256
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    utd: int = 5
    policy_delay: int = 10
    chunk_length: int = 6
    world_reset_every: int = 4
    reward_mode: str = "success"
    reward_scale: float = 1.0


def _collector_device(base_device: str, devices: str, worker_id: int) -> torch.device:
    base = torch.device(base_device)
    if not devices:
        return base
    if base.type != "cuda":
        raise ValueError("--collector-devices requires a CUDA learner device")
    try:
        indices = [
            int(token.strip()) for token in devices.split(",") if token.strip()
        ]
    except ValueError as error:
        raise ValueError(
            "--collector-devices must be comma-separated CUDA indices"
        ) from error
    if not indices or min(indices) < 0:
        raise ValueError("--collector-devices must contain non-negative CUDA indices")
    return torch.device("cuda", indices[worker_id % len(indices)])


def dino_main_frame_indices(
    window: int = VISION_WINDOW, stride: int = OBSERVATION_STRIDE
) -> list[int]:
    """Match eval_metaworld: [d-6, d-4, d-2, d] for the default 4-frame window."""
    if window < 1 or stride < 1:
        raise ValueError("vision window and stride must be positive")
    return list(range(-stride * window + 1, 0, stride))


def vision_buffer_length(
    window: int = VISION_WINDOW, stride: int = OBSERVATION_STRIDE
) -> int:
    if window < 1 or stride < 1:
        raise ValueError("vision window and stride must be positive")
    return (window - 1) * stride + 1


def squeeze_action_chunk(
    action_chunk: torch.Tensor, horizon: int, action_dim: int
) -> np.ndarray:
    chunk = np.asarray(action_chunk.detach().float().cpu().numpy())
    if chunk.ndim == 3:
        if chunk.shape[0] != 1:
            raise ValueError(
                f"batched action chunk must have batch size 1, got {chunk.shape}"
            )
        chunk = chunk[0]
    if chunk.shape != (horizon, action_dim):
        raise ValueError(
            f"action chunk must have shape ({horizon}, {action_dim}), got {chunk.shape}"
        )
    return np.clip(chunk, -1.0, 1.0).astype(np.float32)


class ChunkActor(nn.Module):
    """Gaussian chunk actor conditioned on state and the frozen VLA reference."""

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int,
        hidden_dim: int,
        fixed_std: float,
        *,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if fixed_std < 0:
            raise ValueError("fixed_std must be non-negative")
        self.horizon = horizon
        self.action_dim = action_dim
        self.fixed_std = fixed_std
        self.residual = residual
        chunk_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + chunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_dim),
        )
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        drop_reference: bool = False,
        reference_dropout: float = 0.0,
    ) -> torch.Tensor:
        if reference.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("reference action chunk has the wrong shape")
        base = reference if self.residual else None
        if drop_reference and reference_dropout:
            keep = (
                torch.rand(state.shape[0], 1, 1, device=state.device)
                >= reference_dropout
            )
            reference = reference * keep
        output = torch.tanh(
            self.net(torch.cat((state, reference.flatten(1)), dim=-1))
        ).view(-1, self.horizon, self.action_dim)
        return (base + output).clamp(-1.0, 1.0) if base is not None else output

    def sample(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        deterministic: bool = False,
        *,
        drop_reference: bool = False,
        reference_dropout: float = 0.0,
    ) -> torch.Tensor:
        mean = self(
            state,
            reference,
            drop_reference=drop_reference,
            reference_dropout=reference_dropout,
        )
        if not deterministic and self.fixed_std:
            mean = mean + self.fixed_std * torch.randn_like(mean)
        return mean.clamp(-1.0, 1.0)


class TwinCritic(nn.Module):
    def __init__(
        self, state_dim: int, horizon: int, action_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        input_dim = state_dim + horizon * action_dim

        def mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = mlp()
        self.q2 = mlp()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        actual_steps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actual_steps is not None:
            indices = torch.arange(action.shape[1], device=action.device)
            action = action * (indices[None, :, None] < actual_steps[:, None, None])
        value_input = torch.cat((state, action.flatten(1)), dim=-1)
        return self.q1(value_input).squeeze(-1), self.q2(value_input).squeeze(-1)


def chunk_td_target(
    rewards: torch.Tensor,
    done: torch.Tensor,
    actual_steps: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discount rewards inside each executed chunk, then bootstrap once."""
    indices = torch.arange(rewards.shape[1], device=rewards.device)
    mask = indices[None] < actual_steps[:, None]
    discounts = rewards.new_tensor(gamma) ** indices
    chunk_return = (rewards * mask * discounts).sum(dim=1)
    bootstrap = (1.0 - done.float()) * (rewards.new_tensor(gamma) ** actual_steps) * next_q
    return chunk_return + bootstrap


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        task_ids: list[int] | tuple[int, ...] | None = None,
        *,
        anchor_fraction: float = 0.25,
        success_fraction: float = 0.5,
    ) -> None:
        if (
            capacity < 1
            or not 0.0 <= anchor_fraction <= 1.0
            or not 0.0 <= success_fraction <= 1.0
        ):
            raise ValueError("replay capacity/fraction are invalid")
        task_ids = tuple(dict.fromkeys(task_ids or (0,)))
        if not task_ids:
            raise ValueError("replay requires at least one task")
        self.task_ids = task_ids
        self.anchor_fraction = anchor_fraction
        self.success_fraction = success_fraction
        self.per_task_capacity = max(1, capacity // len(task_ids))
        self.anchors: dict[int, list[tuple[torch.Tensor, ...]]] = {
            task_id: [] for task_id in task_ids
        }
        self.online_success: dict[int, list[tuple[torch.Tensor, ...]]] = {
            task_id: [] for task_id in task_ids
        }
        self.online_failure: dict[int, list[tuple[torch.Tensor, ...]]] = {
            task_id: [] for task_id in task_ids
        }
        self.online_success_positions = {task_id: 0 for task_id in task_ids}
        self.online_failure_positions = {task_id: 0 for task_id in task_ids}

    def __len__(self) -> int:
        return sum(
            len(self.anchors[task_id])
            + len(self.online_success[task_id])
            + len(self.online_failure[task_id])
            for task_id in self.task_ids
        )

    def add(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        next_state: torch.Tensor,
        next_reference: torch.Tensor,
        done: bool,
        actual_steps: int,
        task_id: int = 0,
        *,
        anchor: bool = False,
        successful: bool | None = None,
    ) -> None:
        task_id = int(task_id)
        if task_id not in self.online_success:
            raise ValueError(f"replay received unknown task {task_id}")
        row = (
            state.detach().float().cpu(),
            reference.detach().float().cpu(),
            action.detach().float().cpu(),
            rewards.detach().float().cpu(),
            next_state.detach().float().cpu(),
            next_reference.detach().float().cpu(),
            torch.tensor(float(done)),
            torch.tensor(int(actual_steps), dtype=torch.long),
            torch.tensor(task_id, dtype=torch.long),
        )
        if anchor:
            self.anchors[task_id].append(row)
            return
        if successful is None:
            successful = bool(rewards[:actual_steps].gt(0).any())
        rows_by_task = self.online_success if successful else self.online_failure
        positions = (
            self.online_success_positions
            if successful
            else self.online_failure_positions
        )
        capacity = max(1, self.per_task_capacity // 2)
        rows = rows_by_task[task_id]
        if len(rows) < capacity:
            rows.append(row)
            return
        position = positions[task_id]
        rows[position] = row
        positions[task_id] = (position + 1) % capacity

    def available_task_ids(self) -> list[int]:
        return [
            task_id
            for task_id in self.task_ids
            if self.anchors[task_id]
            or self.online_success[task_id]
            or self.online_failure[task_id]
        ]

    def _sample_rows(self, task_id: int, size: int) -> list[tuple[torch.Tensor, ...]]:
        anchors = self.anchors[task_id]
        successful = self.online_success[task_id]
        failed = self.online_failure[task_id]
        if not anchors and not successful and not failed:
            return []
        anchor_count = round(size * self.anchor_fraction) if anchors else 0
        online_count = size - anchor_count if successful or failed else 0
        if not successful and not failed:
            anchor_count = size
        elif not anchors:
            online_count = size
        success_count = round(online_count * self.success_fraction) if successful else 0
        failure_count = online_count - success_count if failed else 0
        if not failed:
            success_count = online_count
        elif not successful:
            failure_count = online_count
        sampled_pools = [
            [random.choices(anchors, k=anchor_count), anchors],
            [random.choices(successful, k=success_count), successful],
            [random.choices(failed, k=failure_count), failed],
        ]
        minimum_positive = max(1, round(size / 16))
        positive_count = sum(
            bool(row[3][: int(row[7])].gt(0).any())
            for rows, _ in sampled_pools
            for row in rows
        )
        for rows, pool in sampled_pools:
            if positive_count >= minimum_positive:
                break
            positive = [
                row for row in pool if bool(row[3][: int(row[7])].gt(0).any())
            ]
            if not positive:
                continue
            replace = [
                index
                for index, row in enumerate(rows)
                if not bool(row[3][: int(row[7])].gt(0).any())
            ]
            random.shuffle(replace)
            for index in replace[: minimum_positive - positive_count]:
                rows[index] = random.choice(positive)
                positive_count += 1
                if positive_count >= minimum_positive:
                    break
        rows = [row for sampled, _ in sampled_pools for row in sampled]
        random.shuffle(rows)
        return rows

    def sample(self, size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        tasks = self.available_task_ids()
        count = size
        allocations = {task_id: 0 for task_id in tasks}
        order = tasks.copy()
        random.shuffle(order)
        for index in range(count):
            allocations[order[index % len(order)]] += 1
        rows = [
            row
            for task_id, task_count in allocations.items()
            for row in self._sample_rows(task_id, task_count)
        ]
        random.shuffle(rows)
        return tuple(torch.stack(items).to(device) for items in zip(*rows, strict=True))

    def sample_task(
        self, task_id: int, size: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        rows = self._sample_rows(int(task_id), size)
        if not rows:
            raise RuntimeError(f"task {task_id} has no replay transitions")
        return tuple(torch.stack(items).to(device) for items in zip(*rows, strict=True))

    def extend_stacked(
        self,
        batch: tuple[torch.Tensor, ...],
        *,
        anchor: bool = False,
        successful: bool | None = None,
    ) -> None:
        if not batch:
            return
        if len(batch) != 9:
            raise ValueError(f"stacked replay must contain 9 fields, got {len(batch)}")
        length = len(batch[0])
        if any(len(tensor) != length for tensor in batch):
            raise ValueError("stacked replay fields have inconsistent lengths")
        for index in range(length):
            self.add(
                *(tensor[index] for tensor in batch[:8]),
                task_id=int(batch[8][index]),
                anchor=anchor,
                successful=successful,
            )

    def stacked(self) -> tuple[torch.Tensor, ...]:
        rows = [
            row
            for task_id in self.task_ids
            for row in (
                *self.anchors[task_id],
                *self.online_success[task_id],
                *self.online_failure[task_id],
            )
        ]
        if not rows:
            return ()
        return tuple(torch.stack(items) for items in zip(*rows, strict=True))


def overlapping_transition_rows(
    states: dict[int, object],
    planned_actions: torch.Tensor,
    executed_rewards: torch.Tensor,
    *,
    next_start: int,
    executed_steps: int,
    done: bool,
    chunk_length: int,
    stride: int,
) -> tuple[list[tuple], int]:
    """Emit ready stride-overlapped C-step rows with endpoints at ``t + C``."""
    if chunk_length < 1 or stride < 1 or chunk_length % stride:
        raise ValueError("transition stride must be positive and divide chunk length")
    if planned_actions.ndim != 2 or executed_rewards.ndim != 1:
        raise ValueError("transition tapes must be [steps, action_dim] and [steps]")
    if executed_steps != len(executed_rewards) or executed_steps > len(planned_actions):
        raise ValueError("executed transition tape lengths are inconsistent")

    rows = []
    start = int(next_start)
    while start < executed_steps and start + chunk_length <= len(planned_actions):
        actual_steps = min(chunk_length, executed_steps - start)
        terminal = bool(done and start + actual_steps == executed_steps)
        if actual_steps < chunk_length and not terminal:
            break
        if start not in states:
            raise ValueError(f"missing transition state at step {start}")
        endpoint = start + chunk_length
        if not terminal and endpoint not in states:
            break
        following = states[start] if terminal else states[endpoint]
        rewards = executed_rewards.new_zeros(chunk_length)
        rewards[:actual_steps] = executed_rewards[start : start + actual_steps]
        rows.append(
            (
                start,
                states[start],
                planned_actions[start:endpoint],
                rewards,
                following,
                terminal,
                actual_steps,
            )
        )
        start += stride
    return rows, start


def recovery_transition_rows(
    state_vectors: dict[int, torch.Tensor],
    expert_actions: torch.Tensor,
    valid_recovery: torch.Tensor,
    *,
    first_success: int,
    chunk_length: int,
    stride: int,
) -> list[tuple]:
    """Build DAgger rows with intervention actions as action and reference."""
    if expert_actions.ndim != 2 or valid_recovery.shape != (len(expert_actions),):
        raise ValueError("recovery actions/mask have inconsistent shapes")
    if not 0 <= first_success < len(expert_actions):
        raise ValueError("first_success is outside the recovery episode")
    executed_steps = first_success + 1
    required = executed_steps + chunk_length - 1
    planned = expert_actions
    if len(planned) < required:
        planned = torch.cat(
            (planned, planned[-1:].expand(required - len(planned), -1)), dim=0
        )
    # Evo-RLT replaces the reference with the intervening expert action so the
    # actor regularizer preserves, rather than erases, recovery supervision.
    references = {
        step: (state, planned[step : step + chunk_length])
        for step, state in state_vectors.items()
    }
    rewards = expert_actions.new_zeros(executed_steps)
    rewards[first_success] = 1.0
    rows, _ = overlapping_transition_rows(
        references,
        planned,
        rewards,
        next_start=0,
        executed_steps=executed_steps,
        done=True,
        chunk_length=chunk_length,
        stride=stride,
    )
    return [
        row
        for row in rows
        if bool(valid_recovery[row[0] : row[0] + row[-1]].all())
    ]


@dataclass
class PolicyState:
    condition: torch.Tensor
    proprio: torch.Tensor
    reference: torch.Tensor


def refine_state(
    policy_state: PolicyState, rl_token: RLTokenModule
) -> torch.Tensor:
    return torch.cat(
        (rl_token.encode(policy_state.condition.float()), policy_state.proprio.float()),
        dim=-1,
    )


def _validate_rlt_inputs(
    config: VACompoundConfig, checkpoint: dict, features: dict
) -> tuple[int, int, int]:
    """Accept only the measured H15/P15 peer configuration used by this pilot."""
    expected = {
        "va_world_mode": "peer_sync_h6",
        "wmrm": True,
        "slot_free_policy": True,
        "main_vision_backbone": "dinov2_vitl14_reg4",
        "main_vision_frames": 4,
        "main_vision_temporal": True,
        "local_slots": False,
        "dense_readout": False,
        "dense_readout_mtvj": False,
        "dino_dense_metric": False,
        "action_vision_backbone": "none",
        "plan_resampler": False,
        "scene_teacher": False,
        "direct_head": False,
        "c2_controller": False,
        "flow_semantic": False,
        "proprio_dim": 4,
        "action_dim": 4,
    }
    mismatch = {
        key: (getattr(config, key, None), value)
        for key, value in expected.items()
        if getattr(config, key, None) != value
    }
    horizon = int(config.action_horizon)
    stride = int(config.planning_stride)
    execution = int(config.deployment_execution_horizon or stride)
    world_horizon = int(config.wmrm_cycle_steps)
    if not (
        horizon
        == stride
        == execution
        == world_horizon
        == PILOT_ACTION_HORIZON
    ):
        mismatch["H/P/execution/world"] = (
            horizon,
            stride,
            execution,
            world_horizon,
            PILOT_ACTION_HORIZON,
        )
    metadata = features.get("metadata") or {}
    for key, value in {
        "action_horizon": horizon,
        "planning_stride": stride,
        "control_stride": stride,
        "sequence_length": int(config.main_vision_frames),
    }.items():
        if metadata.get(key) != value:
            mismatch[f"features.{key}"] = (metadata.get(key), value)
    if mismatch:
        raise ValueError(f"RLT pilot input contract mismatch: {mismatch}")
    for key in ("language_hidden", "language_mask", "instruction_id", "normalization"):
        if key not in features:
            raise ValueError(f"RLT features missing {key}")
    flow_steps = int(
        (checkpoint.get("training_contract") or {}).get("flow_steps")
        or checkpoint.get("flow_steps")
        or 8
    )
    if flow_steps < 1:
        raise ValueError("checkpoint flow_steps must be positive")
    return horizon, stride, flow_steps


class MetaWorldRunner:
    def __init__(
        self,
        *,
        model: VACompoundPolicy,
        vision: TimmActionVisionBackbone,
        language_cache,
        env,
        device: torch.device,
        state_q01: np.ndarray,
        state_scale: np.ndarray,
        action_q01: np.ndarray,
        action_q99: np.ndarray,
        chunk_length: int,
        flow_steps: int,
        episode_horizon: int,
        world_reset_every: int,
        reward_mode: str,
        reward_scale: float,
    ) -> None:
        if reward_mode not in {"success", "dense"} or reward_scale <= 0:
            raise ValueError("invalid RLT reward mode/scale")
        self.model = model
        self.vision = vision
        self.language_cache = language_cache
        self.env = env
        self.device = device
        self.state_q01 = state_q01
        self.state_scale = state_scale
        self.action_q01 = action_q01
        self.action_q99 = action_q99
        self.chunk_length = chunk_length
        self.flow_steps = flow_steps
        self.episode_horizon = episode_horizon
        self.world_reset_every = world_reset_every
        self.reward_mode = reward_mode
        self.reward_scale = float(reward_scale)
        self.window = int(model.config.main_vision_frames)
        self.action_dim = int(model.config.action_dim)
        self.frames: deque[np.ndarray] = deque(
            maxlen=vision_buffer_length(self.window)
        )
        self.obs = np.empty(0)
        self.last_norm = np.zeros(self.action_dim, dtype=np.float32)
        self.memory = None
        self.primitive_steps = 0
        self.decision_count = 0

    def reset(self, seed: int) -> PolicyState:
        np.random.seed(seed)
        torch.manual_seed(1_000_000 + seed)
        self.obs, _ = self.env.reset(seed=seed)
        frame = self.env.render()
        self.frames.clear()
        self.frames.extend([frame] * self.frames.maxlen)
        self.last_norm.fill(0)
        self.memory = None
        self.primitive_steps = 0
        self.decision_count = 0
        return self.policy_state()

    @torch.inference_mode()
    def policy_state(self, *, commit_memory: bool = True) -> PolicyState:
        if (
            commit_memory
            and self.world_reset_every > 0
            and self.decision_count > 0
            and self.decision_count % self.world_reset_every == 0
        ):
            self.memory = _reset_world_state(self.memory)
        cuda_devices = []
        if not commit_memory and self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        # Intermediate stride-2 states are replay observations, not controller
        # decisions. Keep both recurrent memory and flow RNG unchanged.
        with torch.random.fork_rng(devices=cuda_devices, enabled=not commit_memory):
            frames = [
                self.frames[index]
                for index in dino_main_frame_indices(self.window)
            ]
            tokens = _main_vision_encode_window(
                frames,
                self.vision,
                self.device,
                grid=int(self.model.config.main_vision_grid),
                window=self.window,
            )
            vision_input = _apply_local_vision(self.model, tokens, self.language_cache)
            state = state_take_normalize(self.obs, 4, self.state_q01, self.state_scale)
            proprio = torch.as_tensor(state, device=self.device)[None]
            previous = torch.as_tensor(
                self.last_norm, device=self.device, dtype=torch.float32
            )[None]
            condition, following_memory = self.model.encode_condition(
                vision_input,
                proprio,
                previous,
                language_cache=self.language_cache,
                visual_memory=self.memory,
                return_visual_memory=True,
            )
            reference = self.model.decode_actions(
                condition, steps=self.flow_steps
            ).clamp(-1.0, 1.0)[:, : self.chunk_length]
        if commit_memory:
            self.memory = following_memory
            self.decision_count += 1
        return PolicyState(condition, proprio, reference)

    def execute(
        self, action_chunk: torch.Tensor, *, capture_stride: int | None = None
    ) -> tuple[
        torch.Tensor,
        bool,
        bool,
        int,
        torch.Tensor,
        list[tuple[int, PolicyState]],
    ]:
        chunk = squeeze_action_chunk(action_chunk, self.chunk_length, self.action_dim)
        rewards = np.zeros(self.chunk_length, dtype=np.float32)
        # Keep the commanded suffix. Zero-filling unexecuted steps would make
        # success BC learn "set the unused tail to 0" instead of matching VLA.
        stored = chunk.copy()
        done = success = False
        actual_steps = 0
        intermediate_states = []
        for index, norm_action in enumerate(chunk):
            if self.primitive_steps >= self.episode_horizon:
                done = True
                break
            action = (
                np.clip(norm_action, -1.0, 1.0)
                * (self.action_q99 - self.action_q01)
                / 2
                + (self.action_q99 + self.action_q01) / 2
            )
            self.obs, env_reward, terminated, truncated, info = self.env.step(action)
            if not np.isfinite(env_reward):
                raise RuntimeError(f"MetaWorld returned non-finite reward {env_reward}")
            self.frames.append(self.env.render())
            self.last_norm = np.clip(norm_action, -1.0, 1.0).astype(np.float32)
            stored[index] = self.last_norm
            self.primitive_steps += 1
            actual_steps += 1
            success = bool(info.get("success"))
            reward = env_reward if self.reward_mode == "dense" else float(success)
            rewards[index] = float(reward) * self.reward_scale
            done = (
                success
                or bool(terminated)
                or bool(truncated)
                or self.primitive_steps >= self.episode_horizon
            )
            if done:
                break
            if (
                capture_stride is not None
                and actual_steps % capture_stride == 0
                and actual_steps < self.chunk_length
            ):
                intermediate_states.append(
                    (actual_steps, self.policy_state(commit_memory=False))
                )
        return (
            torch.from_numpy(rewards),
            done,
            success,
            actual_steps,
            torch.from_numpy(stored),
            intermediate_states,
        )


def run_episode(
    runner: MetaWorldRunner,
    seed: int,
    action_fn: Callable[[PolicyState], torch.Tensor],
    on_transition: Callable[
        [PolicyState, torch.Tensor, torch.Tensor, PolicyState, bool, int], None
    ]
    | None = None,
    on_chunk: Callable[[], None] | None = None,
    transition_stride: int | None = None,
) -> tuple[bool, int]:
    transition_stride = runner.chunk_length if transition_stride is None else transition_stride
    if runner.chunk_length % transition_stride:
        raise ValueError("transition stride must divide runner chunk length")
    current = runner.reset(seed)
    states: dict[int, PolicyState] = {0: current}
    planned = torch.empty((0, runner.action_dim), dtype=torch.float32)
    reward_tape = torch.empty(0, dtype=torch.float32)
    next_start = 0
    success = False
    while True:
        chunk_start = runner.primitive_steps
        action = action_fn(current)
        rewards, done, success, actual_steps, stored, intermediate = runner.execute(
            action,
            capture_stride=(
                transition_stride
                if on_transition is not None and transition_stride < runner.chunk_length
                else None
            ),
        )
        planned = torch.cat((planned, stored), dim=0)
        reward_tape = torch.cat((reward_tape, rewards[:actual_steps]), dim=0)
        states.update(
            {chunk_start + offset: state for offset, state in intermediate}
        )
        next_state = current if done else runner.policy_state()
        if not done:
            states[runner.primitive_steps] = next_state
        if on_transition is not None:
            rows, next_start = overlapping_transition_rows(
                states,
                planned,
                reward_tape,
                next_start=next_start,
                executed_steps=runner.primitive_steps,
                done=done,
                chunk_length=runner.chunk_length,
                stride=transition_stride,
            )
            for _, start_state, row_action, row_rewards, following, terminal, steps in rows:
                on_transition(
                    start_state,
                    row_action,
                    row_rewards,
                    following,
                    terminal,
                    steps,
                )
        if on_chunk is not None:
            on_chunk()
        if done:
            return success, runner.primitive_steps
        current = next_state


def store_transition(
    replay: ReplayBuffer,
    rl_token: RLTokenModule,
    current: PolicyState,
    action: torch.Tensor,
    rewards: torch.Tensor,
    following: PolicyState,
    done: bool,
    actual_steps: int,
    task_id: int,
) -> None:
    replay.add(
        refine_state(current, rl_token)[0],
        current.reference[0],
        action,
        rewards,
        refine_state(following, rl_token)[0],
        following.reference[0],
        done,
        actual_steps,
        task_id,
    )


def update_bc(
    replay: ReplayBuffer,
    actor: ChunkActor,
    actor_optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    steps: int,
) -> float:
    if not replay or steps < 1:
        return float("nan")
    actor.train()
    loss_value = float("nan")
    for step in range(steps):
        state, reference, action, _, _, _, _, actual_steps, _ = replay.sample(
            batch_size, device
        )
        error = (actor(state, reference) - action).square()
        executed = (
            torch.arange(action.shape[1], device=device)[None]
            < actual_steps[:, None]
        )
        loss = error[executed].mean() if executed.any() else error.new_zeros(())
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        loss_value = float(loss.detach())
        if (step + 1) % max(1, steps // 5) == 0:
            print(
                f"rlt bc step={step + 1}/{steps} loss={loss_value:.6f}",
                flush=True,
            )
    actor.eval()
    if not np.isfinite(loss_value):
        raise RuntimeError("success BC became non-finite")
    return loss_value


def update_rlt(
    replay: ReplayBuffer,
    actor: ChunkActor,
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    config: RefineConfig,
    device: torch.device,
    update_step: int,
    updates: int,
) -> tuple[int, dict[str, float]]:
    if not replay:
        return update_step, {}
    metrics: dict[str, float] = {}
    for _ in range(updates):
        (
            state,
            reference,
            action,
            rewards,
            next_state,
            next_reference,
            done,
            actual_steps,
            _,
        ) = replay.sample(config.batch_size, device)
        with torch.no_grad():
            next_action = actor(next_state, next_reference)
            next_q1, next_q2 = target_critic(next_state, next_action)
            target = chunk_td_target(
                rewards,
                done,
                actual_steps,
                torch.minimum(next_q1, next_q2),
                config.gamma,
            )
        q1, q2 = critic(state, action, actual_steps)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        metrics["critic_loss"] = float(critic_loss.detach())
        mask = torch.arange(rewards.shape[1], device=device)[None] < actual_steps[:, None]
        metrics["reward_mean"] = float(rewards[mask].mean())
        metrics["td_target_mean"] = float(target.mean())
        with torch.no_grad():
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.lerp_(param, config.tau)
        update_step += 1
        if update_step % config.policy_delay:
            continue
        actor_action = actor(
            state,
            reference,
            drop_reference=True,
            reference_dropout=config.reference_dropout,
        )
        actor_q1, actor_q2 = critic(state, actor_action)
        bc = (actor_action - reference).square().sum(dim=(1, 2)).mean()
        actor_loss = -torch.minimum(actor_q1, actor_q2).mean() + config.beta * bc
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        metrics.update(actor_loss=float(actor_loss.detach()), bc_loss=float(bc.detach()))
    return update_step, metrics


def update_multitask_rlt(
    replay: ReplayBuffer,
    actor: ChunkActor,
    critics: nn.ModuleDict,
    target_critics: nn.ModuleDict,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: dict[str, torch.optim.Optimizer],
    config: RefineConfig,
    device: torch.device,
    update_step: int,
    updates: int,
    *,
    update_actor: bool = True,
) -> tuple[int, dict[str, float]]:
    """Update balanced per-task critics and PCGrad the shared actor."""
    from train import backward_pcgrad

    task_ids = replay.available_task_ids()
    if not task_ids:
        return update_step, {}
    task_ids.sort()
    metrics: dict[str, float] = {}
    tasks_per_update = min(4, len(task_ids))
    for _ in range(updates):
        selected = random.Random(update_step).sample(task_ids, tasks_per_update)
        per_task_batch = max(1, config.batch_size // len(selected))
        batches = {
            task_id: replay.sample_task(task_id, per_task_batch, device)
            for task_id in selected
        }
        critic_losses = []
        reward_means = []
        target_means = []
        for task_id, batch in batches.items():
            (
                state,
                reference,
                action,
                rewards,
                next_state,
                next_reference,
                done,
                actual_steps,
                _,
            ) = batch
            critic = critics[str(task_id)]
            target_critic = target_critics[str(task_id)]
            with torch.no_grad():
                next_action = actor(next_state, next_reference)
                next_q1, next_q2 = target_critic(next_state, next_action)
                target = chunk_td_target(
                    rewards,
                    done,
                    actual_steps,
                    torch.minimum(next_q1, next_q2),
                    config.gamma,
                )
            q1, q2 = critic(state, action, actual_steps)
            loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            optimizer = critic_optimizers[str(task_id)]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            critic_losses.append(loss.detach())
            mask = (
                torch.arange(rewards.shape[1], device=device)[None]
                < actual_steps[:, None]
            )
            reward_means.append(rewards[mask].mean().detach())
            target_means.append(target.mean().detach())
            with torch.no_grad():
                for target_param, param in zip(
                    target_critic.parameters(), critic.parameters()
                ):
                    target_param.lerp_(param, config.tau)

        update_step += 1
        metrics["critic_loss"] = float(torch.stack(critic_losses).mean())
        metrics["reward_mean"] = float(torch.stack(reward_means).mean())
        metrics["td_target_mean"] = float(torch.stack(target_means).mean())
        metrics["tasks_per_update"] = float(len(selected))
        if not update_actor or update_step % config.policy_delay:
            continue

        task_actor_losses = []
        task_bc_losses = []
        for task_id, batch in batches.items():
            state, reference = batch[:2]
            actor_action = actor(
                state,
                reference,
                drop_reference=True,
                reference_dropout=config.reference_dropout,
            )
            actor_q1, actor_q2 = critics[str(task_id)](state, actor_action)
            bc = (actor_action - reference).square().sum(dim=(1, 2)).mean()
            task_bc_losses.append(bc.detach())
            task_actor_losses.append(
                -torch.minimum(actor_q1, actor_q2).mean() + config.beta * bc
            )
        actor_optimizer.zero_grad(set_to_none=True)
        if len(task_actor_losses) > 1:
            stats = backward_pcgrad(
                task_actor_losses,
                list(actor.named_parameters()),
                seed=update_step,
            )
        else:
            task_actor_losses[0].backward()
            stats = {"conflicts": 0, "comparisons": 0}
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        metrics.update(
            actor_loss=float(torch.stack([loss.detach() for loss in task_actor_losses]).mean()),
            bc_loss=float(torch.stack(task_bc_losses).mean()),
            pcgrad_conflicts=float(stats["conflicts"]),
            pcgrad_comparisons=float(stats["comparisons"]),
        )
    return update_step, metrics


def _select_rlt_tasks(all_tasks: list[str], task_ids: str) -> list[tuple[int, str]]:
    if task_ids.strip().lower() == "all":
        return select_eval_tasks(all_tasks, None, len(all_tasks))
    selected = select_eval_tasks(all_tasks, task_ids, 0)
    if len({task_id for task_id, _ in selected}) != len(selected):
        raise ValueError("--task-ids contains duplicates")
    return selected


@torch.no_grad()
def _extract_vla_token_batch(
    batch: dict,
    model: VACompoundPolicy,
    vision: TimmActionVisionBackbone,
    device: torch.device,
    *,
    encode_batch: int,
    memory_commit_every: int = 1,
) -> torch.Tensor:
    """Return final VLA tokens [B*T,H,D], after peer-World layer injection."""
    from train import _dino_main_online_encode

    if memory_commit_every < 1:
        raise ValueError("memory commit cadence must be positive")

    tokens = _dino_main_online_encode(
        batch["frames"],
        vision,
        device,
        encode_batch=encode_batch,
        grid=int(model.config.main_vision_grid),
        window=int(model.config.main_vision_frames),
    )
    dtype = model.action_queries.dtype
    language = batch["language_hidden"].to(device=device, dtype=dtype)
    language_mask = batch["language_mask"].to(device=device)
    language_cache = model.build_language_cache(language, language_mask, detach=True)
    proprio = batch["proprio"].to(device=device, dtype=dtype)
    previous = batch["previous_action"].to(device=device, dtype=dtype)
    memory = None
    conditions = []
    for decision in range(proprio.shape[1]):
        vision_input = _apply_local_vision(
            model, tokens[:, decision], language_cache
        )
        condition, following_memory = model.encode_condition(
            vision_input,
            proprio[:, decision],
            previous[:, decision],
            language_cache=language_cache,
            visual_memory=memory,
            return_visual_memory=True,
        )
        if decision % memory_commit_every == 0:
            memory = following_memory
        conditions.append(condition.float())
    return torch.cat(conditions, dim=0).detach()


def _token_validation(
    loader: DataLoader,
    rl_token: RLTokenModule,
    model: VACompoundPolicy,
    vision: TimmActionVisionBackbone,
    device: torch.device,
    *,
    encode_batch: int,
    batches: int,
) -> dict[str, float | bool]:
    rl_token.eval()
    losses: list[float] = []
    no_token_losses: list[float] = []
    shuffled_token_losses: list[float] = []
    zero_losses: list[float] = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        target = _extract_vla_token_batch(
            batch, model, vision, device, encode_batch=encode_batch
        )
        with torch.no_grad():
            encoded = rl_token.encode_multi(target)
            losses.append(float(F.mse_loss(rl_token.decode(encoded, target), target)))
            no_token_losses.append(
                float(
                    F.mse_loss(
                        rl_token.decode(torch.zeros_like(encoded), target), target
                    )
                )
            )
            shuffled_token_losses.append(
                float(
                    F.mse_loss(
                        rl_token.decode(encoded.roll(1, dims=0), target), target
                    )
                )
            )
            zero_losses.append(float(target.square().mean()))
    if not losses:
        raise RuntimeError("RL-token validation produced no batches")
    reconstruction = float(np.mean(losses))
    no_token = float(np.mean(no_token_losses))
    shuffled_token = float(np.mean(shuffled_token_losses))
    zero = float(np.mean(zero_losses))
    return {
        "reconstruction_mse": reconstruction,
        "no_token_mse": no_token,
        "shuffled_token_mse": shuffled_token,
        "zero_mse": zero,
        "reconstruction_over_no_token": reconstruction / max(no_token, 1e-12),
        "gate_pass": bool(
            np.isfinite(reconstruction)
            and reconstruction < no_token
            and reconstruction < shuffled_token
        ),
    }


def _token_task_local_batches(
    row_tasks: list[int],
    indices: list[int],
    batch_size: int,
    seed: int,
    block_batches: int = 16,
) -> list[list[int]]:
    """Shuffle short single-task blocks to balance JPEG locality and task mixing."""
    if batch_size < 1 or block_batches < 1:
        raise ValueError("token batch and task-block sizes must be positive")
    rng = random.Random(seed)
    by_task: dict[int, list[int]] = {}
    for index in indices:
        by_task.setdefault(int(row_tasks[index]), []).append(index)
    tasks = list(by_task)
    rng.shuffle(tasks)
    blocks: list[list[list[int]]] = []
    for task_id in tasks:
        rows = by_task[task_id]
        rng.shuffle(rows)
        task_batches = [
            rows[start : start + batch_size]
            for start in range(0, len(rows), batch_size)
        ]
        blocks.extend(
            task_batches[start : start + block_batches]
            for start in range(0, len(task_batches), block_batches)
        )
    rng.shuffle(blocks)
    return [batch for block in blocks for batch in block]


def _train_rl_token(
    args: argparse.Namespace,
    model: VACompoundPolicy,
    vision: TimmActionVisionBackbone,
    selected_tasks: list[tuple[int, str]],
    device: torch.device,
) -> None:
    from va_compound.longtraj_frames import (
        OnlineLongTrajEpisodeDataset,
        mtvj_collate,
    )

    dataset = OnlineLongTrajEpisodeDataset(
        args.demo_index,
        longtraj_dir=args.longtraj_dir,
        samples_per_episode=args.token_samples_per_episode,
        recovery_samples_per_episode=args.token_recovery_samples_per_episode,
        sampling_seed=args.seed,
        decode_cache_tasks=max(1, args.token_decode_cache_tasks),
        include_world_target_frames=False,
    )
    demo_tasks = [str(item["description"]) for item in dataset.index["tasks"]]
    mismatched = [
        task_id
        for task_id, description in selected_tasks
        if task_id >= len(demo_tasks) or demo_tasks[task_id] != description
    ]
    if mismatched:
        raise ValueError(
            "--demo-index task ordering differs from --features at task ids "
            f"{mismatched[:8]}"
        )
    task_set = {task_id for task_id, _ in selected_tasks}
    row_tasks = dataset.payload["instruction_id"].tolist()
    indices = [index for index, task_id in enumerate(row_tasks) if task_id in task_set]
    if args.token_dagger_only:
        indices = [
            index
            for index in indices
            if not bool(dataset.payload["anchor_eligible"][index])
        ]
    if not indices:
        raise ValueError("--demo-index contains none of the selected tasks")
    covered_tasks = {row_tasks[index] for index in indices}
    if covered_tasks != task_set:
        raise ValueError(
            "RL-token rows lack selected task ids "
            f"{sorted(task_set - covered_tasks)}"
        )
    print(
        f"rl-token data rows={len(indices)} tasks={len(covered_tasks)} "
        f"dagger_only={int(args.token_dagger_only)}",
        flush=True,
    )
    def epoch_loader(epoch: int, workers: int) -> DataLoader:
        dataset.set_epoch(epoch)
        batches = _token_task_local_batches(
            row_tasks,
            indices,
            args.token_batch_size,
            args.seed + epoch,
            args.token_task_block_batches,
        )

        def scheduled_batches():
            for position, batch in enumerate(batches):
                if workers == 0:
                    upcoming = batches[position : position + 2]
                    dataset.prefetch_indices(
                        [index for rows in upcoming for index in rows]
                    )
                yield batch

        return DataLoader(
            dataset,
            batch_sampler=scheduled_batches(),
            num_workers=workers,
            collate_fn=mtvj_collate,
        )

    token_config = RLTokenConfig(
        token_dim=int(model.config.hidden_dim),
        num_tokens=args.token_count,
        heads=args.token_heads,
        encoder_layers=args.token_layers,
        decoder_layers=args.token_layers,
        ff_dim=(args.token_ff_dim or 4 * int(model.config.hidden_dim)),
    )
    rl_token = RLTokenModule(token_config).to(device)
    optimizer = torch.optim.AdamW(
        rl_token.parameters(), lr=args.token_lr, weight_decay=1e-4
    )
    cached_targets = []
    cache_batches = len(
        _token_task_local_batches(
            row_tasks,
            indices,
            args.token_batch_size,
            args.seed,
            args.token_task_block_batches,
        )
    )
    for batch_index, batch in enumerate(
        epoch_loader(0, args.token_workers), start=1
    ):
        cached_targets.append(
            _extract_vla_token_batch(
                batch, model, vision, device, encode_batch=args.token_encode_batch
            ).cpu()
        )
        if batch_index == 1 or batch_index % args.token_log_every == 0:
            print(
                f"rl-token cache batch={batch_index}/{cache_batches}",
                flush=True,
            )
    target_cache = torch.cat(cached_targets)
    if device.type == "cuda":
        target_cache = target_cache.pin_memory()
    train_batch = args.token_batch_size * int(
        dataset.payload["metadata"]["sequence_length"]
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(target_cache), generator=generator)
    cursor = 0
    epoch = 0
    for step in range(1, args.token_steps + 1):
        if cursor + train_batch > len(order):
            epoch += 1
            order = torch.randperm(len(target_cache), generator=generator)
            cursor = 0
        target = target_cache[order[cursor : cursor + train_batch]].to(
            device, non_blocking=True
        )
        cursor += train_batch
        rl_token.train()
        loss = rl_token.reconstruction_loss(target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rl_token.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"RL-token loss became non-finite at step {step}")
        if step == 1 or step % args.token_log_every == 0:
            print(
                f"rl-token step={step}/{args.token_steps} epoch={epoch} "
                f"loss={float(loss.detach()):.6f}",
                flush=True,
            )

    validation_loader = epoch_loader(1_000_000 + args.seed, 0)
    metrics = _token_validation(
        validation_loader,
        rl_token,
        model,
        vision,
        device,
        encode_batch=args.token_encode_batch,
        batches=args.token_val_batches,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": TOKEN_CONTRACT,
            "base_checkpoint": str(args.checkpoint.resolve()),
            "token_source": "VACompoundPolicy.encode_condition",
            "token_scope": "shared_multitask",
            "token_sequence_length": int(model.config.action_horizon),
            "task_ids": [task_id for task_id, _ in selected_tasks],
            "task_descriptions": {
                str(task_id): description
                for task_id, description in selected_tasks
            },
            "token_config": token_config.to_dict(),
            "train_steps": args.token_steps,
            "validation": metrics,
            "rl_token": rl_token.state_dict(),
            "encoder": rl_token.encoder_state_dict(),
        },
        args.output,
    )
    print(f"rl-token validation={metrics}", flush=True)
    print(f"rl-token saved: {args.output}", flush=True)
    if not metrics["gate_pass"]:
        raise RuntimeError(
            "RL-token gate failed: reconstruction does not beat the same decoder "
            "with its RL-token input removed and shuffled across states"
        )


def _load_rl_token(
    payload: dict,
    *,
    base_checkpoint: Path,
    model_config: VACompoundConfig,
    device: torch.device,
) -> RLTokenModule:
    if payload.get("contract") != TOKEN_CONTRACT:
        raise ValueError("unsupported RL-token checkpoint contract")
    if payload.get("token_scope") != "shared_multitask":
        raise ValueError("RL-token checkpoint is not the shared multitask contract")
    if payload.get("token_source") != "VACompoundPolicy.encode_condition":
        raise ValueError("RL-token checkpoint uses a different VLA token source")
    saved_base = payload.get("base_checkpoint")
    if saved_base is None or Path(saved_base).resolve() != base_checkpoint.resolve():
        raise ValueError("RL-token checkpoint was trained from a different frozen VLA")
    if not bool((payload.get("validation") or {}).get("gate_pass")):
        raise ValueError("RL-token checkpoint did not pass its reconstruction gate")
    token_config = RLTokenConfig(**payload["token_config"])
    if token_config.token_dim != int(model_config.hidden_dim):
        raise ValueError("RL-token dimension does not match the frozen VLA")
    if int(payload.get("token_sequence_length", -1)) != int(
        model_config.action_horizon
    ):
        raise ValueError("RL-token sequence length does not match the frozen VLA")
    rl_token = RLTokenModule(token_config, with_decoder=False).to(device)
    rl_token.load_encoder_state_dict(payload["encoder"])
    return rl_token.eval().requires_grad_(False)


@torch.no_grad()
def _prefill_recovery_replays(
    args: argparse.Namespace,
    replay: ReplayBuffer,
    rl_token: RLTokenModule,
    model: VACompoundPolicy,
    vision: TimmActionVisionBackbone,
    selected_tasks: list[tuple[int, str]],
    device: torch.device,
) -> dict[int, int]:
    """Encode fixed DAgger recovery episodes into frozen-VLA-referenced replay."""
    from scripts.build_longtraj_features import (
        clip_frame_indices,
        resolve_episode_semantics,
    )
    from va_compound.longtraj_frames import OnlineLongTrajEpisodeDataset

    dataset = OnlineLongTrajEpisodeDataset(
        args.demo_index,
        longtraj_dir=args.longtraj_dir,
        samples_per_episode=1,
        sampling_seed=args.seed,
        decode_cache_tasks=max(1, args.token_decode_cache_tasks),
    )
    demo_tasks = [str(item["description"]) for item in dataset.index["tasks"]]
    mismatched = [
        task_id
        for task_id, description in selected_tasks
        if task_id >= len(demo_tasks) or demo_tasks[task_id] != description
    ]
    if mismatched:
        raise ValueError(
            "--demo-index task ordering differs from --features at task ids "
            f"{mismatched[:8]}"
        )

    counts: dict[int, int] = {}
    for task_id, task_text in selected_tasks:
        seen = set()
        entries = []
        for entry in dataset._episodes_by_task.get(task_id, []):
            if "source_path" not in entry:
                continue
            key = dataset._entry_key(entry)
            if key not in seen:
                seen.add(key)
                entries.append(entry)
        entries = entries[-args.replay_prefill_episodes_per_task :]
        added = positive = 0
        for entry in entries:
            source = dataset._load_entry(entry)
            contract = (source.get("metadata") or {}).get("contract")
            if contract not in {"current_policy_dagger_v1", "current_policy_dagger_merged_v1"}:
                continue
            episode_index = int(entry["episode_index"])
            episode = source["episodes"][episode_index]
            semantics = resolve_episode_semantics(
                episode,
                f"{entry['source_path']}:episode[{episode_index}]",
                legacy_policy="infer",
            )
            first_success = semantics["first_success"]
            if first_success is None:
                raise ValueError("DAgger recovery episode has no first success")
            first_success = int(first_success)
            action_success = np.asarray(episode.get("action_success"), dtype=bool)
            if action_success.shape != (len(episode["actions"]),):
                raise ValueError("DAgger action-success timeline has the wrong shape")
            succeeded = np.flatnonzero(action_success)
            if not len(succeeded) or int(succeeded[0]) != first_success:
                raise ValueError("DAgger first-success reward is not aligned to its action")

            actions = dataset._normalize(
                np.asarray(episode["actions"], dtype=np.float32),
                dataset._aq01,
                dataset._aq99,
            ).astype(np.float32)
            proprio = dataset._normalize(
                np.asarray(episode["states"], dtype=np.float32),
                dataset._sq01,
                dataset._sq99,
            ).astype(np.float32)
            anchors = list(range(0, first_success + 1, args.replay_stride))
            frame_rows = [
                [int(index) for index in clip_frame_indices(anchor)]
                for anchor in anchors
            ]
            flat = [index for row in frame_rows for index in row]
            decoded = dataset._decode_episode_frames(entry, flat)
            frames = np.stack(decoded).reshape(
                1,
                len(anchors),
                len(frame_rows[0]),
                *decoded[0].shape,
            )
            previous = np.stack(
                [
                    np.zeros(actions.shape[1], dtype=np.float32)
                    if anchor == 0
                    else actions[anchor - 1]
                    for anchor in anchors
                ]
            )
            anchor_proprio = torch.from_numpy(proprio[anchors])[None]
            conditions = _extract_vla_token_batch(
                {
                    "frames": frames,
                    "language_hidden": dataset._reference_hidden[task_id][None],
                    "language_mask": dataset._reference_mask[task_id][None],
                    "proprio": anchor_proprio,
                    "previous_action": torch.from_numpy(previous)[None],
                },
                model,
                vision,
                device,
                encode_batch=args.token_encode_batch,
                memory_commit_every=args.chunk_length // args.replay_stride,
            )
            encoded = torch.cat(
                (
                    rl_token.encode(conditions.float()),
                    anchor_proprio[0].to(device),
                ),
                dim=-1,
            ).cpu()
            state_vectors = dict(zip(anchors, encoded, strict=True))
            valid_recovery = torch.from_numpy(
                np.asarray(semantics["valid"], dtype=bool)
                & np.asarray(semantics["recovery"], dtype=bool)
            )
            rows = recovery_transition_rows(
                state_vectors,
                torch.from_numpy(actions),
                valid_recovery,
                first_success=first_success,
                chunk_length=args.chunk_length,
                stride=args.replay_stride,
            )
            for _, current, action, rewards, following, done, actual_steps in rows:
                state, reference = current
                next_state, next_reference = following
                replay.add(
                    state,
                    reference,
                    action,
                    rewards,
                    next_state,
                    next_reference,
                    done,
                    actual_steps,
                    task_id,
                    anchor=True,
                )
                added += 1
                positive += int(bool(rewards[:actual_steps].sum()))
        if not added:
            raise RuntimeError(f"DAgger prefill produced no recovery rows for task {task_id}")
        counts[task_id] = added
        print(
            f"rlt recovery prefill task={task_id}:{task_text} episodes={len(entries)} "
            f"transitions={added} terminal_positive={positive} "
            f"stride={args.replay_stride} C={args.chunk_length}",
            flush=True,
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "token-train", "train", "eval"),
        default="train",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--language-features", type=Path)
    parser.add_argument("--main-vision-checkpoint", type=Path, required=True)
    parser.add_argument("--token-checkpoint", type=Path)
    parser.add_argument("--rlt-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demo-index", type=Path)
    parser.add_argument("--longtraj-dir", type=Path)
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-steps", type=int, default=2000)
    parser.add_argument("--token-batch-size", type=int, default=2)
    parser.add_argument("--token-lr", type=float, default=1e-4)
    parser.add_argument("--token-count", type=int, default=1)
    parser.add_argument("--token-heads", type=int, default=8)
    parser.add_argument("--token-layers", type=int, default=3)
    parser.add_argument("--token-ff-dim", type=int, default=0)
    parser.add_argument("--token-encode-batch", type=int, default=16)
    parser.add_argument("--token-samples-per-episode", type=int, default=2)
    parser.add_argument("--token-recovery-samples-per-episode", type=int, default=1)
    parser.add_argument("--token-decode-cache-tasks", type=int, default=2)
    parser.add_argument("--token-workers", type=int, default=0)
    parser.add_argument("--token-val-batches", type=int, default=10)
    parser.add_argument("--token-log-every", type=int, default=20)
    parser.add_argument("--token-task-block-batches", type=int, default=16)
    parser.add_argument("--token-dagger-only", action="store_true")
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--bootstrap-updates", type=int, default=1000)
    parser.add_argument("--online-episodes", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-episode-seed-base", type=int)
    parser.add_argument("--episode-horizon", type=int, default=500)
    parser.add_argument("--world-reset-every", type=int, default=4)
    parser.add_argument("--reward-mode", choices=("success", "dense"), default="success")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--replay-stride", type=int, default=2)
    parser.add_argument("--replay-prefill-episodes-per-task", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-length", type=int, default=6)
    parser.add_argument("--utd", type=int, default=5)
    parser.add_argument("--policy-delay", type=int, default=10)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--fixed-std", type=float, default=0.05)
    parser.add_argument("--reference-dropout", type=float, default=0.5)
    parser.add_argument("--save-every-episodes", type=int, default=50)
    parser.add_argument("--collectors", type=int, default=1)
    parser.add_argument("--collector-devices", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, action_horizon: int | None = None) -> None:
    for path in (args.checkpoint, args.features, args.main_vision_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.language_features is not None and not args.language_features.is_file():
        raise FileNotFoundError(args.language_features)
    if args.mode in {"token-train", "train"} and args.output is None:
        raise ValueError("--output is required in token-train/train mode")
    if args.mode == "token-train" and (
        args.demo_index is None or not args.demo_index.is_file()
    ):
        raise ValueError("--demo-index is required in token-train mode")
    if args.mode == "train" and (
        args.token_checkpoint is None or not args.token_checkpoint.is_file()
    ):
        raise ValueError("--token-checkpoint is required in train mode")
    if args.mode == "eval" and (
        args.rlt_checkpoint is None or not args.rlt_checkpoint.is_file()
    ):
        raise ValueError("--rlt-checkpoint is required in eval mode")
    positive = {
        "episode_horizon": args.episode_horizon,
        "replay_capacity": args.replay_capacity,
        "batch_size": args.batch_size,
        "utd": args.utd,
        "policy_delay": args.policy_delay,
        "chunk_length": args.chunk_length,
        "replay_stride": args.replay_stride,
        "collectors": args.collectors,
    }
    if any(value < 1 for value in positive.values()):
        raise ValueError(f"positive arguments required: {positive}")
    _validate_chunk_length(args.chunk_length, action_horizon)
    if args.chunk_length % args.replay_stride:
        raise ValueError("--replay-stride must divide --chunk-length")
    if args.replay_stride not in {OBSERVATION_STRIDE, args.chunk_length}:
        raise ValueError(
            f"--replay-stride must be {OBSERVATION_STRIDE} or one full chunk"
        )
    if args.replay_prefill_episodes_per_task < 0:
        raise ValueError("--replay-prefill-episodes-per-task must be non-negative")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.reward_mode == "dense" and args.replay_prefill_episodes_per_task:
        raise ValueError("dense reward cannot mix with sparse recovery replay prefill")
    if args.mode == "train" and args.replay_prefill_episodes_per_task and (
        args.demo_index is None or not args.demo_index.is_file()
    ):
        raise ValueError("--demo-index is required when recovery replay prefill is enabled")
    if args.mode == "train" and (
        args.warmup_episodes < 1
        or args.bootstrap_updates < 0
        or args.online_episodes < 0
    ):
        raise ValueError("train counts require warmup >= 1 and update/online >= 0")
    token_positive = {
        "token_steps": args.token_steps,
        "token_batch_size": args.token_batch_size,
        "token_count": args.token_count,
        "token_heads": args.token_heads,
        "token_layers": args.token_layers,
        "token_encode_batch": args.token_encode_batch,
        "token_samples_per_episode": args.token_samples_per_episode,
        "token_decode_cache_tasks": args.token_decode_cache_tasks,
        "token_val_batches": args.token_val_batches,
        "token_log_every": args.token_log_every,
        "token_task_block_batches": args.token_task_block_batches,
    }
    if args.mode == "token-train" and any(
        value < 1 for value in token_positive.values()
    ):
        raise ValueError(f"positive RL-token arguments required: {token_positive}")
    if args.mode == "token-train" and args.token_batch_size < 2:
        raise ValueError("--token-batch-size must be at least 2 for the shuffled-token gate")
    if args.token_workers < 0 or args.token_ff_dim < 0 or args.token_lr <= 0:
        raise ValueError("invalid RL-token worker/ff-dim/learning-rate argument")
    if not 0 <= args.token_recovery_samples_per_episode <= args.token_samples_per_episode:
        raise ValueError("token recovery samples must be within samples per episode")
    if (
        args.eval_episodes < 0
        or args.beta < 0
        or args.fixed_std < 0
        or args.world_reset_every < 0
    ):
        raise ValueError("eval episodes, beta, and fixed std must be non-negative")
    if args.save_every_episodes < 0:
        raise ValueError("--save-every-episodes must be non-negative")
    if not 0.0 <= args.reference_dropout <= 1.0:
        raise ValueError("--reference-dropout must be in [0, 1]")


def _validate_chunk_length(
    chunk_length: int, action_horizon: int | None = None
) -> None:
    if chunk_length < 1:
        raise ValueError("--chunk-length must be positive")
    if action_horizon is not None and chunk_length > action_horizon:
        raise ValueError(
            "--chunk-length cannot exceed the VLA action horizon"
        )


def _make_runners(
    selected_tasks,
    features,
    language_features,
    model,
    vision,
    device,
    chunk_length,
    flow_steps,
    episode_horizon,
    world_reset_every,
    reward_mode,
    reward_scale,
) -> dict[int, MetaWorldRunner]:
    import metaworld

    descriptions_to_env = load_metaworld_description_to_env()
    normalization = features["normalization"]
    sq01 = normalization["state_q01"].numpy()
    sq99 = normalization["state_q99"].numpy()
    state_scale = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    aq01 = normalization["action_q01"].numpy()
    aq99 = normalization["action_q99"].numpy()
    runners = {}
    for task_id, task_text in selected_tasks:
        env_name = descriptions_to_env.get(task_text)
        if env_name is None:
            raise ValueError(f"no MetaWorld environment for task {task_text!r}")
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](
            render_mode="rgb_array", camera_name="corner2"
        )
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env._freeze_rand_vec = False
        hidden, mask = cached_task_language(
            language_features, device, task_id=task_id
        )
        runners[task_id] = MetaWorldRunner(
            model=model,
            vision=vision,
            language_cache=model.build_language_cache(hidden, mask),
            env=env,
            device=device,
            state_q01=sq01,
            state_scale=state_scale,
            action_q01=aq01,
            action_q99=aq99,
            chunk_length=chunk_length,
            flow_steps=flow_steps,
            episode_horizon=episode_horizon,
            world_reset_every=world_reset_every,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
        )
    return runners


def _collector_worker(
    worker_id: int,
    args: argparse.Namespace,
    selected_tasks: list[tuple[int, str]],
    refine_config_payload: dict,
    command_queue,
    result_queue,
) -> None:
    """Persistent rollout worker with its own frozen VLA and environments."""

    runners = {}
    try:
        device = _collector_device(args.device, args.collector_devices, worker_id)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        random.seed(args.seed + 10_000 + worker_id)
        np.random.seed(args.seed + 10_000 + worker_id)
        torch.manual_seed(args.seed + 10_000 + worker_id)
        torch.set_grad_enabled(False)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        features = torch.load(args.features, map_location="cpu", weights_only=True)
        language_features = features
        if args.language_features is not None:
            language_features = torch.load(
                args.language_features, map_location="cpu", weights_only=True
            )
            validate_language_features(features, language_features)
        model_config = VACompoundConfig(**checkpoint["config"])
        _, _, flow_steps = _validate_rlt_inputs(model_config, checkpoint, features)
        model = VACompoundPolicy(model_config).eval().to(device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "collector base mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        model.runtime_execution_horizon = args.chunk_length
        model.requires_grad_(False)
        vision = TimmActionVisionBackbone.from_pretrained(
            device=device,
            dtype="float16",
            model_id=model_config.main_vision_model_id,
            image_size=model_config.main_vision_image_size,
            feature_dim=model_config.main_vision_dim,
            output_layers=(11, 23),
            checkpoint_path=args.main_vision_checkpoint,
            local_files_only=True,
        )
        vision.freeze_all()
        token_payload = torch.load(
            args.token_checkpoint, map_location="cpu", weights_only=True
        )
        rl_token = _load_rl_token(
            token_payload,
            base_checkpoint=args.checkpoint,
            model_config=model_config,
            device=device,
        )
        runners = _make_runners(
            selected_tasks,
            features,
            language_features,
            model,
            vision,
            device,
            args.chunk_length,
            flow_steps,
            args.episode_horizon,
            args.world_reset_every,
            args.reward_mode,
            args.reward_scale,
        )
        refine_config = RefineConfig(**refine_config_payload)
        actor = ChunkActor(
            model_config.hidden_dim + model_config.proprio_dim,
            refine_config.chunk_length,
            model_config.action_dim,
            refine_config.hidden_dim,
            refine_config.fixed_std,
            residual=True,
        ).eval().to(device)
        del checkpoint, features, language_features, token_payload
        gc.collect()
        result_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "tasks": [task_id for task_id, _ in selected_tasks],
                "device": str(device),
                "gpu_bytes": torch.cuda.memory_allocated(device),
            }
        )

        while True:
            command = command_queue.get()
            kind = command["kind"]
            if kind == "stop":
                return
            if kind == "prefill":
                local_replay = ReplayBuffer(
                    args.replay_capacity,
                    [task_id for task_id, _ in selected_tasks],
                )
                counts = _prefill_recovery_replays(
                    args,
                    local_replay,
                    rl_token,
                    model,
                    vision,
                    selected_tasks,
                    device,
                )
                torch.save(
                    {"transitions": local_replay.stacked(), "counts": counts},
                    command["path"],
                )
                result_queue.put(
                    {
                        "kind": "prefill",
                        "worker_id": worker_id,
                        "path": command["path"],
                        "transitions": len(local_replay),
                    }
                )
                continue
            if kind not in {"warmup", "online", "eval"}:
                raise ValueError(f"unknown collector command {kind!r}")

            if kind in {"online", "eval"}:
                actor.load_state_dict(command["actor"])
            local_replay = ReplayBuffer(
                args.replay_capacity,
                [task_id for task_id, _ in selected_tasks],
            )
            chunks = 0

            def collect(current, action, rewards, following, done, actual_steps):
                store_transition(
                    local_replay,
                    rl_token,
                    current,
                    action,
                    rewards,
                    following,
                    done,
                    actual_steps,
                    int(command["task_id"]),
                )

            def count_chunk() -> None:
                nonlocal chunks
                chunks += 1

            if kind == "warmup":
                action_fn = lambda state: state.reference
            else:
                deterministic = kind == "eval"

                def action_fn(state):
                    return actor.sample(
                        refine_state(state, rl_token),
                        state.reference,
                        deterministic=deterministic,
                    )

            collect_fn = None if kind == "eval" else collect
            success, steps = run_episode(
                runners[int(command["task_id"])],
                int(command["seed"]),
                action_fn,
                collect_fn,
                count_chunk if collect_fn is not None else None,
                transition_stride=args.replay_stride,
            )
            result_queue.put(
                {
                    "kind": "episode",
                    "phase": kind,
                    "worker_id": worker_id,
                    "job_id": int(command["job_id"]),
                    "task_id": int(command["task_id"]),
                    "success": bool(success),
                    "steps": int(steps),
                    "chunks": chunks,
                    "transitions": local_replay.stacked(),
                }
            )
    except BaseException:
        result_queue.put(
            {
                "kind": "error",
                "worker_id": worker_id,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        for runner in runners.values():
            try:
                runner.env.close()
            except Exception:
                pass


def _collector_result(result_queue, processes, timeout: float = 900.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for asynchronous collector")
        try:
            message = result_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            failed = [
                process.pid for process in processes if process.exitcode is not None
            ]
            if failed:
                raise RuntimeError(f"collector processes exited early: {failed}")
            continue
        if message.get("kind") == "error":
            raise RuntimeError(
                f"collector {message['worker_id']} failed:\n{message['traceback']}"
            )
        return message


def _cpu_actor_state(actor: ChunkActor) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}


def _assert_identity_residual(actor: ChunkActor) -> None:
    if actor.residual and any(
        torch.count_nonzero(parameter).item()
        for parameter in (actor.net[-1].weight, actor.net[-1].bias)
    ):
        raise RuntimeError("residual actor changed before online training")


def _stop_collectors(processes, command_queues) -> None:
    for process, commands in zip(processes, command_queues, strict=True):
        if process.is_alive():
            commands.put({"kind": "stop"})
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _save(
    path: Path,
    *,
    base_checkpoint: Path,
    task_ids: list[int],
    task_descriptions: dict[int, str],
    rl_token: RLTokenModule,
    token_payload: dict,
    actors: nn.ModuleDict,
    critics: nn.ModuleDict,
    target_critics: nn.ModuleDict,
    config: RefineConfig,
    horizon: int,
    action_dim: int,
    online_episodes: int,
    update_steps: dict[str, int],
    vla_warmup_successes: int,
    vla_warmup_episodes: int,
    replay_stride: int,
    recovery_prefill_transitions: dict[int, int],
    collectors: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # ponytail: save deployable weights periodically; exact replay/optimizer
    # resume can be added if process interruption becomes a measured cost.
    torch.save(
        {
            "contract": CONTRACT,
            "base_checkpoint": str(base_checkpoint.resolve()),
            "task_ids": task_ids,
            "task_descriptions": {
                str(task_id): description
                for task_id, description in task_descriptions.items()
            },
            "head_scope": "shared",
            "critic_scope": "per_task",
            "replay_sampling": "task_balanced_anchor25_success50_positive6p25",
            "actor_gradient": "task_pcgrad_random4",
            "actor_parameterization": "reference_residual_skip_dropout_v2",
            "rl_token": {
                "contract": TOKEN_CONTRACT,
                "base_checkpoint": str(base_checkpoint.resolve()),
                "token_source": token_payload.get("token_source"),
                "token_scope": token_payload.get("token_scope"),
                "token_sequence_length": token_payload["token_sequence_length"],
                "task_ids": token_payload.get("task_ids"),
                "task_descriptions": token_payload.get("task_descriptions"),
                "token_config": token_payload["token_config"],
                "validation": token_payload["validation"],
                "encoder": rl_token.encoder_state_dict(),
            },
            "rlt_config": asdict(config),
            "horizon": horizon,
            "chunk_length": config.chunk_length,
            "action_dim": action_dim,
            "online_episodes": online_episodes,
            "update_steps": update_steps,
            "vla_warmup_successes": vla_warmup_successes,
            "vla_warmup_episodes": vla_warmup_episodes,
            "replay_stride": replay_stride,
            "collectors": collectors,
            "recovery_prefill_transitions": {
                str(task_id): count
                for task_id, count in recovery_prefill_transitions.items()
            },
            "actors": actors.state_dict(),
            "critics": critics.state_dict(),
            "target_critics": target_critics.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _train_rlt_async(
    args: argparse.Namespace,
    selected_tasks: list[tuple[int, str]],
    actors: nn.ModuleDict,
    critics: nn.ModuleDict,
    target_critics: nn.ModuleDict,
    rl_token: RLTokenModule,
    token_payload: dict,
    refine_config: RefineConfig,
    model_config: VACompoundConfig,
    horizon: int,
    device: torch.device,
) -> None:
    if args.collectors > len(selected_tasks):
        raise ValueError("--collectors cannot exceed the selected task count")
    collector_devices = [
        _collector_device(args.device, args.collector_devices, worker_id)
        for worker_id in range(args.collectors)
    ]
    unavailable = [
        str(device)
        for device in collector_devices
        if device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ]
    if unavailable:
        raise ValueError(
            f"collector CUDA devices are not visible: {unavailable}; "
            f"visible count={torch.cuda.device_count()}"
        )
    actor = actors["shared"]
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=refine_config.actor_lr
    )
    critic_optimizers = {
        key: torch.optim.Adam(critic.parameters(), lr=refine_config.critic_lr)
        for key, critic in critics.items()
    }
    replay = ReplayBuffer(
        args.replay_capacity,
        [task_id for task_id, _ in selected_tasks],
    )
    task_shards = [selected_tasks[index :: args.collectors] for index in range(args.collectors)]
    task_to_worker = {
        task_id: worker_id
        for worker_id, shard in enumerate(task_shards)
        for task_id, _ in shard
    }
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    command_queues = [context.Queue() for _ in task_shards]
    processes = [
        context.Process(
            target=_collector_worker,
            args=(
                worker_id,
                args,
                shard,
                asdict(refine_config),
                command_queues[worker_id],
                result_queue,
            ),
            name=f"rlt-collector-{worker_id}",
        )
        for worker_id, shard in enumerate(task_shards)
    ]
    prefill_paths = [
        args.output.with_name(
            f".{args.output.stem}.prefill-{os.getpid()}-{worker_id}.pt"
        )
        for worker_id in range(args.collectors)
    ]

    def build_jobs(count: int, phase: str, trial: dict[int, int]) -> list[list[dict]]:
        jobs = [[] for _ in processes]
        for job_id in range(count):
            task_id, _ = selected_tasks[job_id % len(selected_tasks)]
            seed = evaluation_episode_seed(task_id, trial[task_id])
            trial[task_id] += 1
            jobs[task_to_worker[task_id]].append(
                {
                    "kind": phase,
                    "job_id": job_id,
                    "task_id": task_id,
                    "seed": seed,
                }
            )
        return jobs

    def run_jobs(job_lists, phase: str, handle) -> None:
        positions = [0] * len(processes)
        active: set[int] = set()
        fixed_actor = _cpu_actor_state(actor) if phase == "eval" else None

        def dispatch(worker_id: int) -> None:
            position = positions[worker_id]
            if position >= len(job_lists[worker_id]):
                return
            job = dict(job_lists[worker_id][position])
            positions[worker_id] += 1
            if phase in {"online", "eval"}:
                job["actor"] = fixed_actor or _cpu_actor_state(actor)
            command_queues[worker_id].put(job)
            active.add(worker_id)

        for worker_id in range(len(processes)):
            dispatch(worker_id)
        while active:
            message = _collector_result(result_queue, processes)
            if message.get("kind") != "episode" or message.get("phase") != phase:
                raise RuntimeError(f"unexpected collector result: {message}")
            worker_id = int(message["worker_id"])
            if worker_id not in active:
                raise RuntimeError(f"collector {worker_id} returned without an active job")
            active.remove(worker_id)
            # Dispatch before learning so rollout and learner GPU work overlap.
            dispatch(worker_id)
            handle(message)

    for process in processes:
        process.start()
    try:
        ready = {}
        while len(ready) < len(processes):
            message = _collector_result(result_queue, processes)
            if message.get("kind") != "ready":
                raise RuntimeError(f"collector did not initialize cleanly: {message}")
            ready[int(message["worker_id"])] = message
        print(
            "rlt async collectors ready "
            f"count={len(ready)} devices="
            f"{[ready[index]['device'] for index in sorted(ready)]} gpu_gib="
            f"{sum(item['gpu_bytes'] for item in ready.values()) / 2**30:.2f}",
            flush=True,
        )

        prefill_counts = {task_id: 0 for task_id, _ in selected_tasks}
        if args.replay_prefill_episodes_per_task:
            for commands, path in zip(command_queues, prefill_paths, strict=True):
                commands.put({"kind": "prefill", "path": path})
            for _ in processes:
                message = _collector_result(result_queue, processes, timeout=3600)
                if message.get("kind") != "prefill":
                    raise RuntimeError(f"unexpected prefill result: {message}")
                path = Path(message["path"])
                payload = torch.load(path, map_location="cpu", weights_only=True)
                replay.extend_stacked(tuple(payload["transitions"]), anchor=True)
                prefill_counts.update(
                    {int(task_id): int(count) for task_id, count in payload["counts"].items()}
                )
                path.unlink(missing_ok=True)
                print(
                    f"rlt async prefill worker={message['worker_id']} "
                    f"transitions={message['transitions']} replay={len(replay)}",
                    flush=True,
                )

        trial = {task_id: 100 for task_id, _ in selected_tasks}
        warmup_successes = 0
        warmup_completed = 0

        def accept_warmup(message: dict) -> None:
            nonlocal warmup_successes, warmup_completed
            replay.extend_stacked(
                tuple(message["transitions"]),
                successful=bool(message["success"]),
            )
            warmup_successes += int(message["success"])
            warmup_completed += 1
            task_id = int(message["task_id"])
            print(
                f"rlt warmup episode={warmup_completed}/{args.warmup_episodes} "
                f"task={task_id}:{dict(selected_tasks)[task_id]} "
                f"success={int(message['success'])} steps={message['steps']} "
                f"worker={message['worker_id']}",
                flush=True,
            )

        run_jobs(build_jobs(args.warmup_episodes, "warmup", trial), "warmup", accept_warmup)
        if not replay:
            raise RuntimeError("asynchronous warmup produced no transitions")
        print(
            f"rlt vla baseline warmup_success={warmup_successes}/{args.warmup_episodes} "
            f"replay={len(replay)}",
            flush=True,
        )
        bc_loss = 0.0
        update_steps = {"shared": 0}
        update_steps["shared"], metrics = update_multitask_rlt(
            replay,
            actor,
            critics,
            target_critics,
            actor_optimizer,
            critic_optimizers,
            refine_config,
            device,
            0,
            args.bootstrap_updates,
            update_actor=False,
        )
        _assert_identity_residual(actor)
        print(
            f"rlt bootstrap shared tasks={len(selected_tasks)} "
            f"updates={args.bootstrap_updates} replay={len(replay)} "
            f"bc_loss={bc_loss:.6f} metrics={metrics}",
            flush=True,
        )

        def save_training(completed_online_episodes: int) -> None:
            _save(
                args.output,
                base_checkpoint=args.checkpoint,
                task_ids=[task_id for task_id, _ in selected_tasks],
                task_descriptions=dict(selected_tasks),
                rl_token=rl_token,
                token_payload=token_payload,
                actors=actors,
                critics=critics,
                target_critics=target_critics,
                config=refine_config,
                horizon=horizon,
                action_dim=model_config.action_dim,
                online_episodes=completed_online_episodes,
                update_steps=update_steps,
                vla_warmup_successes=warmup_successes,
                vla_warmup_episodes=args.warmup_episodes,
                replay_stride=args.replay_stride,
                recovery_prefill_transitions=prefill_counts,
                collectors=args.collectors,
            )
            snapshot = args.output.with_name(
                f"{args.output.stem}_e{completed_online_episodes}{args.output.suffix}"
            )
            shutil.copy2(args.output, snapshot)

        save_training(0)
        print(f"rlt bootstrap snapshot saved: {args.output}", flush=True)
        online_completed = 0

        def accept_online(message: dict) -> None:
            nonlocal online_completed, metrics
            replay.extend_stacked(
                tuple(message["transitions"]),
                successful=bool(message["success"]),
            )
            updates = int(message["chunks"]) * refine_config.utd
            update_steps["shared"], metrics = update_multitask_rlt(
                replay,
                actor,
                critics,
                target_critics,
                actor_optimizer,
                critic_optimizers,
                refine_config,
                device,
                update_steps["shared"],
                updates,
            )
            online_completed += 1
            task_id = int(message["task_id"])
            print(
                f"rlt online episode={online_completed}/{args.online_episodes} "
                f"task={task_id}:{dict(selected_tasks)[task_id]} "
                f"success={int(message['success'])} steps={message['steps']} "
                f"chunks={message['chunks']} updates={updates} "
                f"worker={message['worker_id']} replay={len(replay)} metrics={metrics}",
                flush=True,
            )
            if args.save_every_episodes and online_completed % args.save_every_episodes == 0:
                save_training(online_completed)
                print(f"rlt snapshot saved: {args.output}", flush=True)

        run_jobs(build_jobs(args.online_episodes, "online", trial), "online", accept_online)
        save_training(args.online_episodes)
        print(f"rlt saved: {args.output}", flush=True)

        if args.eval_episodes:
            eval_jobs = [[] for _ in processes]
            job_id = 0
            for task_id, _ in selected_tasks:
                for trial_index in range(args.eval_episodes):
                    eval_jobs[task_to_worker[task_id]].append(
                        {
                            "kind": "eval",
                            "job_id": job_id,
                            "task_id": task_id,
                            "seed": evaluation_episode_seed(
                                task_id,
                                trial_index,
                                base_seed=args.eval_episode_seed_base,
                            ),
                        }
                    )
                    job_id += 1
            eval_rows = {task_id: [] for task_id, _ in selected_tasks}

            def accept_eval(message: dict) -> None:
                eval_rows[int(message["task_id"])].append(message)

            run_jobs(eval_jobs, "eval", accept_eval)
            total_wins = 0
            for task_id, task_text in selected_tasks:
                rows = eval_rows[task_id]
                wins = sum(int(row["success"]) for row in rows)
                total_wins += wins
                successful_steps = [row["steps"] for row in rows if row["success"]]
                print(
                    f"rlt eval task={task_id}:{task_text} "
                    f"success={wins}/{args.eval_episodes} "
                    f"mean_steps={np.mean([row['steps'] for row in rows]):.1f} "
                    f"success_mean_steps="
                    f"{np.mean(successful_steps) if successful_steps else float('nan'):.1f}",
                    flush=True,
                )
            total_trials = len(selected_tasks) * args.eval_episodes
            print(
                f"rlt eval overall success={total_wins}/{total_trials} "
                f"accuracy={total_wins / max(total_trials, 1):.4f}",
                flush=True,
            )
    finally:
        _stop_collectors(processes, command_queues)
        for path in prefill_paths:
            path.unlink(missing_ok=True)


def _eval_actor(
    selected_tasks,
    runners: dict[int, MetaWorldRunner],
    actors: nn.ModuleDict,
    rl_token: RLTokenModule,
    eval_episodes: int,
    episode_seed_base: int | None,
) -> None:
    total_wins = 0
    actor = actors["shared"]
    for task_id, task_text in selected_tasks:
        wins = 0
        episode_steps: list[int] = []
        successful_steps: list[int] = []
        for trial_index in range(eval_episodes):
            seed = evaluation_episode_seed(
                task_id, trial_index, base_seed=episode_seed_base
            )
            success, steps = run_episode(
                runners[task_id],
                seed,
                lambda state, bound=actor: bound.sample(
                    refine_state(state, rl_token),
                    state.reference,
                    deterministic=True,
                ),
            )
            wins += int(success)
            episode_steps.append(steps)
            if success:
                successful_steps.append(steps)
        total_wins += wins
        print(
            f"rlt eval task={task_id}:{task_text} success={wins}/{eval_episodes} "
            f"mean_steps={np.mean(episode_steps):.1f} "
            f"success_mean_steps="
            f"{np.mean(successful_steps) if successful_steps else float('nan'):.1f}",
            flush=True,
        )
    total_trials = len(selected_tasks) * eval_episodes
    print(
        f"rlt eval overall success={total_wins}/{total_trials} "
        f"accuracy={total_wins / max(total_trials, 1):.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    language_features = features
    if args.language_features is not None:
        language_features = torch.load(
            args.language_features, map_location="cpu", weights_only=True
        )
        validate_language_features(features, language_features)
    config = VACompoundConfig(**checkpoint["config"])
    horizon, stride, flow_steps = _validate_rlt_inputs(config, checkpoint, features)
    _validate_args(args, action_horizon=horizon)
    selected_tasks = _select_rlt_tasks(
        list(language_features["metadata"]["tasks"]), args.task_ids
    )
    print(
        f"rlt preflight: PASS tasks={[task_id for task_id, _ in selected_tasks]} "
        f"H={horizon} P={stride} C={args.chunk_length} flow_steps={flow_steps} "
        f"reward={args.reward_mode}x{args.reward_scale:g}",
        flush=True,
    )
    if args.mode == "preflight":
        return

    model = VACompoundPolicy(config).eval().to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"base checkpoint/model mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    # Receding C < H15 lets encode_condition rebuild a stale World map.
    model.runtime_execution_horizon = args.chunk_length
    model.requires_grad_(False)
    vision = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype="float16",
        model_id=config.main_vision_model_id,
        image_size=config.main_vision_image_size,
        feature_dim=config.main_vision_dim,
        output_layers=(11, 23),
        checkpoint_path=args.main_vision_checkpoint,
        local_files_only=True,
    )
    vision.freeze_all()
    if args.mode == "token-train":
        _train_rl_token(args, model, vision, selected_tasks, device)
        return

    if args.mode == "eval":
        payload = torch.load(args.rlt_checkpoint, map_location="cpu", weights_only=True)
        if payload.get("contract") != CONTRACT:
            raise ValueError("unsupported RLT checkpoint contract")
        if payload.get("head_scope") != "shared":
            raise ValueError("RLT checkpoint does not contain the shared multitask head")
        refine_config = RefineConfig(**payload["rlt_config"])
        if refine_config.chunk_length != args.chunk_length:
            raise ValueError(
                "eval --chunk-length must match the checkpoint "
                f"({args.chunk_length} != {refine_config.chunk_length})"
            )
        if refine_config.world_reset_every != args.world_reset_every:
            raise ValueError(
                "eval --world-reset-every must match the checkpoint "
                f"({args.world_reset_every} != {refine_config.world_reset_every})"
            )
        if int(payload.get("horizon", -1)) != horizon:
            raise ValueError("refine checkpoint horizon does not match the frozen VLA")
        saved_base = payload.get("base_checkpoint")
        if saved_base is None:
            raise ValueError("refine checkpoint missing base_checkpoint")
        if Path(saved_base).resolve() != args.checkpoint.resolve():
            raise ValueError(
                "eval --checkpoint must be the frozen VA used at train time: "
                f"{args.checkpoint.resolve()} != {Path(saved_base).resolve()}"
            )
        trained_tasks = set(map(int, payload.get("task_ids") or []))
        requested_tasks = {task_id for task_id, _ in selected_tasks}
        if not requested_tasks <= trained_tasks:
            raise ValueError(
                "RLT checkpoint was not trained on task ids "
                f"{sorted(requested_tasks - trained_tasks)}"
            )
        trained_descriptions = payload.get("task_descriptions") or {}
        description_mismatch = [
            task_id
            for task_id, description in selected_tasks
            if trained_descriptions.get(str(task_id)) != description
        ]
        if description_mismatch:
            raise ValueError(
                "RLT checkpoint task descriptions differ at ids "
                f"{description_mismatch[:8]}"
            )
        token_payload = payload.get("rl_token")
        if not isinstance(token_payload, dict):
            raise ValueError("RLT checkpoint is missing its embedded RL-token encoder")
    else:
        token_payload = torch.load(
            args.token_checkpoint, map_location="cpu", weights_only=True
        )
        refine_config = RefineConfig(
            batch_size=args.batch_size,
            utd=args.utd,
            policy_delay=args.policy_delay,
            beta=args.beta,
            fixed_std=args.fixed_std,
            reference_dropout=args.reference_dropout,
            chunk_length=args.chunk_length,
            world_reset_every=args.world_reset_every,
            reward_mode=args.reward_mode,
            reward_scale=args.reward_scale,
        )
    rl_token = _load_rl_token(
        token_payload,
        base_checkpoint=args.checkpoint,
        model_config=config,
        device=device,
    )
    token_tasks = set(map(int, token_payload.get("task_ids") or []))
    requested_tasks = {task_id for task_id, _ in selected_tasks}
    if token_tasks and not requested_tasks <= token_tasks:
        raise ValueError(
            "RL-token checkpoint lacks task ids "
            f"{sorted(requested_tasks - token_tasks)}"
        )
    token_descriptions = token_payload.get("task_descriptions") or {}
    description_mismatch = [
        task_id
        for task_id, description in selected_tasks
        if token_descriptions.get(str(task_id)) != description
    ]
    if description_mismatch:
        raise ValueError(
            "RL-token checkpoint task descriptions differ at ids "
            f"{description_mismatch[:8]}"
        )
    async_train = args.mode == "train" and args.collectors > 1
    runners = None
    if not async_train:
        runners = _make_runners(
            selected_tasks,
            features,
            language_features,
            model,
            vision,
            device,
            args.chunk_length,
            flow_steps,
            args.episode_horizon,
            args.world_reset_every,
            refine_config.reward_mode,
            refine_config.reward_scale,
        )
    state_dim = config.hidden_dim + config.proprio_dim
    actor_residual = (
        args.mode != "eval"
        or payload.get("actor_parameterization")
        in {"reference_residual_v1", "reference_residual_skip_dropout_v2"}
    )
    actors = nn.ModuleDict(
        {
            "shared": ChunkActor(
                state_dim,
                refine_config.chunk_length,
                config.action_dim,
                refine_config.hidden_dim,
                refine_config.fixed_std,
                residual=actor_residual,
            )
        }
    ).to(device)
    if args.mode == "eval":
        actors.load_state_dict(payload["actors"])
        actors.eval().requires_grad_(False)
    else:
        critics = nn.ModuleDict(
            {
                str(task_id): TwinCritic(
                    state_dim,
                    refine_config.chunk_length,
                    config.action_dim,
                    refine_config.hidden_dim,
                )
                for task_id, _ in selected_tasks
            }
        ).to(device)
        target_critics = nn.ModuleDict(
            {
                key: copy.deepcopy(critic).requires_grad_(False)
                for key, critic in critics.items()
            }
        )
        if args.warmup_episodes < len(selected_tasks):
            raise ValueError(
                "multitask RLT requires at least one warmup episode per selected task"
            )
        if async_train:
            rl_token.cpu()
            del model, vision, checkpoint, features, language_features
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _train_rlt_async(
                args,
                selected_tasks,
                actors,
                critics,
                target_critics,
                rl_token,
                token_payload,
                refine_config,
                config,
                horizon,
                device,
            )
            return
        replay = ReplayBuffer(
            args.replay_capacity,
            [task_id for task_id, _ in selected_tasks],
        )
        prefill_counts = {task_id: 0 for task_id, _ in selected_tasks}
        if args.replay_prefill_episodes_per_task:
            prefill_counts = _prefill_recovery_replays(
                args,
                replay,
                rl_token,
                model,
                vision,
                selected_tasks,
                device,
            )
        trial = {task_id: 100 for task_id, _ in selected_tasks}
        warmup_successes = 0
        for episode in range(args.warmup_episodes):
            task_id, task_text = selected_tasks[episode % len(selected_tasks)]
            episode_replay = ReplayBuffer(args.replay_capacity, [task_id])

            def collect(
                current, action, rewards, following, done, actual_steps
            ) -> None:
                store_transition(
                    episode_replay,
                    rl_token,
                    current,
                    action,
                    rewards,
                    following,
                    done,
                    actual_steps,
                    task_id,
                )

            seed = evaluation_episode_seed(task_id, trial[task_id])
            trial[task_id] += 1
            success, steps = run_episode(
                runners[task_id],
                seed,
                lambda state: state.reference,
                collect,
                transition_stride=args.replay_stride,
            )
            replay.extend_stacked(
                episode_replay.stacked(), successful=bool(success)
            )
            warmup_successes += int(success)
            print(
                f"rlt warmup episode={episode + 1}/{args.warmup_episodes} "
                f"task={task_id}:{task_text} success={int(success)} steps={steps}",
                flush=True,
            )
        if not replay:
            raise RuntimeError("warmup produced no transitions")
        print(
            f"rlt vla baseline warmup_success={warmup_successes}/{args.warmup_episodes}",
            flush=True,
        )
        actor = actors["shared"]
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=refine_config.actor_lr
        )
        critic_optimizers = {
            key: torch.optim.Adam(critic.parameters(), lr=refine_config.critic_lr)
            for key, critic in critics.items()
        }
        update_steps = {"shared": 0}
        metrics: dict[str, float] = {}
        bc_loss = 0.0
        update_steps["shared"], metrics = update_multitask_rlt(
            replay,
            actor,
            critics,
            target_critics,
            actor_optimizer,
            critic_optimizers,
            refine_config,
            device,
            0,
            args.bootstrap_updates,
            update_actor=False,
        )
        _assert_identity_residual(actor)
        print(
            f"rlt bootstrap shared tasks={len(selected_tasks)} "
            f"updates={args.bootstrap_updates} replay={len(replay)} "
            f"bc_loss={bc_loss:.6f} metrics={metrics}",
            flush=True,
        )

        def save_training(completed_online_episodes: int) -> None:
            _save(
                args.output,
                base_checkpoint=args.checkpoint,
                task_ids=[task_id for task_id, _ in selected_tasks],
                task_descriptions=dict(selected_tasks),
                rl_token=rl_token,
                token_payload=token_payload,
                actors=actors,
                critics=critics,
                target_critics=target_critics,
                config=refine_config,
                horizon=horizon,
                action_dim=config.action_dim,
                online_episodes=completed_online_episodes,
                update_steps=update_steps,
                vla_warmup_successes=warmup_successes,
                vla_warmup_episodes=args.warmup_episodes,
                replay_stride=args.replay_stride,
                recovery_prefill_transitions=prefill_counts,
            )
            snapshot = args.output.with_name(
                f"{args.output.stem}_e{completed_online_episodes}{args.output.suffix}"
            )
            shutil.copy2(args.output, snapshot)

        save_training(0)
        print(f"rlt bootstrap snapshot saved: {args.output}", flush=True)
        for episode in range(args.online_episodes):
            task_id, task_text = selected_tasks[episode % len(selected_tasks)]
            episode_replay = ReplayBuffer(args.replay_capacity, [task_id])
            chunks = 0

            def act(state: PolicyState) -> torch.Tensor:
                with torch.no_grad():
                    return actor.sample(
                        refine_state(state, rl_token),
                        state.reference,
                        deterministic=False,
                    )

            def learn(
                current, action, rewards, following, done, actual_steps
            ) -> None:
                store_transition(
                    episode_replay,
                    rl_token,
                    current,
                    action,
                    rewards,
                    following,
                    done,
                    actual_steps,
                    task_id,
                )

            def count_chunk() -> None:
                nonlocal chunks
                chunks += 1

            seed = evaluation_episode_seed(task_id, trial[task_id])
            trial[task_id] += 1
            success, steps = run_episode(
                runners[task_id],
                seed,
                act,
                learn,
                count_chunk,
                transition_stride=args.replay_stride,
            )
            replay.extend_stacked(
                episode_replay.stacked(), successful=bool(success)
            )
            update_steps["shared"], metrics = update_multitask_rlt(
                replay,
                actor,
                critics,
                target_critics,
                actor_optimizer,
                critic_optimizers,
                refine_config,
                device,
                update_steps["shared"],
                chunks * refine_config.utd,
            )
            print(
                f"rlt online episode={episode + 1}/{args.online_episodes} "
                f"task={task_id}:{task_text} success={int(success)} steps={steps} "
                f"replay={len(replay)} metrics={metrics}",
                flush=True,
            )
            if args.save_every_episodes and (
                episode + 1
            ) % args.save_every_episodes == 0:
                save_training(episode + 1)
                print(f"rlt snapshot saved: {args.output}", flush=True)
        save_training(args.online_episodes)
        print(f"rlt saved: {args.output}", flush=True)

    if args.eval_episodes:
        _eval_actor(
            selected_tasks,
            runners,
            actors,
            rl_token,
            args.eval_episodes,
            args.eval_episode_seed_base,
        )


if __name__ == "__main__":
    main()
