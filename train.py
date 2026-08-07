from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
import math
from pathlib import Path
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from va_compound import VACompoundConfig, VACompoundPolicy


def build_pair_groups(
    pair_id: Tensor,
    instruction_id: Tensor,
) -> dict[int, dict[int, list[int]]]:
    """Map every pair_id to its per-instruction sample indices.

    Shared pair-contract builder reused by FeatureDataset and E2EDataset
    (plus PairedBatchSampler's payload fallback): each pair must contain at
    least two different instruction_id values, otherwise paired training is
    impossible.  Validation matches the original FeatureDataset contract.
    """
    groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (pair, instruction) in enumerate(
        zip(pair_id.tolist(), instruction_id.tolist(), strict=True)
    ):
        groups[int(pair)][int(instruction)].append(index)
    if not groups:
        raise ValueError("paired training requires at least one pair_id")
    for pair, by_instruction in groups.items():
        if len(by_instruction) < 2:
            raise ValueError(
                f"pair_id={pair} needs at least two different instruction_id values"
            )
    return {pair: dict(values) for pair, values in groups.items()}


class FeatureDataset(Dataset):
    """Paired multi-goal feature sequences for the P0 training contract.

    Samples with the same ``pair_id`` must share the exact first observation,
    robot state, and previous action, while using different instructions and
    expert action chunks.  At that decision point language is therefore the
    only input that can explain the required action difference.
    """

    REQUIRED = (
        "vision_tokens",
        "language_hidden",
        "proprio",
        "previous_action",
        "actions",
        "pair_id",
        "instruction_id",
    )
    SEQUENCE_KEYS = ("vision_tokens", "proprio", "previous_action", "actions")

    def __init__(
        self,
        path: Path,
        *,
        require_pairs: bool = True,
        min_sequence_length: int = 4,
        pair_start_atol: float = 0.0,
        min_pair_action_delta: float = 1e-3,
        pair_start_cosine: float = 0.0,
        vision_key: str = "vision_tokens",
        step_targets: Tensor | None = None,
        step_mask: Tensor | None = None,
    ) -> None:
        if min_sequence_length < 2:
            raise ValueError("min_sequence_length must be at least 2")
        if pair_start_atol < 0.0 or min_pair_action_delta <= 0.0:
            raise ValueError("pair tolerances must be non-negative with action delta positive")

        payload = torch.load(path, map_location="cpu", weights_only=True)
        missing = [key for key in self.REQUIRED if key not in payload]
        if missing:
            raise ValueError(f"missing tensors in dataset: {missing}")
        if vision_key not in payload:
            raise ValueError(
                f"dataset has no vision variant '{vision_key}'; "
                f"available: {sorted(key for key in payload if key.startswith('vision_tokens'))}"
            )
        self.payload = payload
        self.vision_key = vision_key
        self.length = int(payload["actions"].shape[0])
        if self.length == 0:
            raise ValueError("training dataset is empty")
        if any(payload[key].shape[0] != self.length for key in self.REQUIRED):
            raise ValueError("dataset tensors have different sample counts")
        if step_targets is not None:
            # C²-VA Stage B：v6a per-chunk-step 期望视觉目标 [N, T, 6, C]
            # （P 投影后的控制状态，C = c2_control_dim；模型侧校验维度）。
            expected_front = (self.length, payload["actions"].shape[1], 6)
            if step_targets.ndim != 4 or tuple(step_targets.shape[:3]) != expected_front:
                raise ValueError(
                    f"step_targets must have shape [{expected_front[0]}, "
                    f"{expected_front[1]}, 6, control_dim], got {tuple(step_targets.shape)}"
                )
        self.step_targets = step_targets
        if step_mask is not None:
            if tuple(step_mask.shape) != (self.length,):
                raise ValueError(
                    f"step_mask must have shape [{self.length}], got {tuple(step_mask.shape)}"
                )
        self.step_mask = step_mask

        self._validate_shapes(min_sequence_length)
        self.pair_groups = self._build_pair_groups() if require_pairs else {}
        if require_pairs:
            self._validate_pair_contract(
                pair_start_atol, min_pair_action_delta, pair_start_cosine
            )

    def _validate_shapes(self, min_sequence_length: int) -> None:
        vision = self.payload[self.vision_key]
        language = self.payload["language_hidden"]
        proprio = self.payload["proprio"]
        previous = self.payload["previous_action"]
        actions = self.payload["actions"]
        if vision.ndim != 4:
            raise ValueError(f"{self.vision_key} must have shape [N,T,Nv,Dv]")
        if language.ndim != 3:
            raise ValueError("language_hidden must have shape [N,Nl,Dl]")
        if proprio.ndim != 3 or previous.ndim != 3:
            raise ValueError("proprio and previous_action must have shape [N,T,D]")
        if actions.ndim != 4:
            raise ValueError("actions must have shape [N,T,H,Da]")
        if previous.shape[-1] != actions.shape[-1]:
            raise ValueError("previous_action and actions must use the same action dimension")
        sequence_keys = (self.vision_key, "proprio", "previous_action", "actions")
        sequence_lengths = {int(self.payload[key].shape[1]) for key in sequence_keys}
        if len(sequence_lengths) != 1:
            raise ValueError("all sequence tensors must use the same T")
        sequence_length = sequence_lengths.pop()
        if sequence_length < min_sequence_length:
            raise ValueError(
                f"paired VA training requires T>={min_sequence_length}, got T={sequence_length}"
            )
        for key in ("pair_id", "instruction_id"):
            value = self.payload[key]
            if value.ndim != 1 or value.shape[0] != self.length:
                raise ValueError(f"{key} must have shape [N]")
            if value.dtype == torch.bool or value.is_floating_point():
                raise ValueError(f"{key} must contain integer identifiers")
        if "language_mask" in self.payload:
            mask = self.payload["language_mask"]
            if mask.shape != language.shape[:2]:
                raise ValueError("language_mask must have shape [N,Nl]")

    def _build_pair_groups(self) -> dict[int, dict[int, list[int]]]:
        return build_pair_groups(self.payload["pair_id"], self.payload["instruction_id"])

    def _validate_pair_contract(
        self,
        pair_start_atol: float,
        min_pair_action_delta: float,
        pair_start_cosine: float = 0.0,
    ) -> None:
        for pair_id, by_instruction in self.pair_groups.items():
            all_indices = [index for indices in by_instruction.values() for index in indices]
            reference = all_indices[0]
            for index in all_indices[1:]:
                for key in (self.vision_key, "proprio", "previous_action"):
                    reference_start = self.payload[key][reference, 0]
                    candidate_start = self.payload[key][index, 0]
                    if key == self.vision_key and pair_start_cosine > 0.0:
                        # 余弦门控契约（2026-08-07）：LIBERO 同场景跨指令的首
                        # 观测在特征空间余弦 >= pair_start_cosine（实测 0.993-
                        # 0.997），残差为文档化视觉差异；proprio/prev 仍走严格
                        # 容差。严格逐 token 相等在该数据上不存在（目标对象
                        # 配置本身是任务定义的一部分）。
                        a = reference_start.flatten().float()
                        b = candidate_start.flatten().float()
                        cosine = float(
                            (F.normalize(a, dim=0) * F.normalize(b, dim=0)).sum().item()
                        )
                        if cosine < pair_start_cosine:
                            raise ValueError(
                                f"pair_id={pair_id} first {key} cosine {cosine:.4f} "
                                f"< {pair_start_cosine:.4f}"
                            )
                    elif not torch.allclose(
                        reference_start,
                        candidate_start,
                        rtol=0.0,
                        atol=pair_start_atol,
                    ):
                        difference = float((reference_start - candidate_start).abs().max())
                        raise ValueError(
                            f"pair_id={pair_id} does not share its first {key}; "
                            f"max_abs_delta={difference:.6g}"
                        )
            instructions = list(by_instruction)
            for left_position, left_instruction in enumerate(instructions):
                for right_instruction in instructions[left_position + 1 :]:
                    for left in by_instruction[left_instruction]:
                        for right in by_instruction[right_instruction]:
                            action_delta = (
                                self.payload["actions"][left, 0]
                                - self.payload["actions"][right, 0]
                            )
                            target_delta = float(action_delta.abs().mean())
                            if target_delta < min_pair_action_delta:
                                raise ValueError(
                                    f"pair_id={pair_id} has no identifiable action difference; "
                                    f"mean_abs_delta={target_delta:.6g}"
                                )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {key: self.payload[key][index] for key in self.REQUIRED}
        item["vision_tokens"] = self.payload[self.vision_key][index]
        if "language_mask" in self.payload:
            item["language_mask"] = self.payload["language_mask"][index]
        if self.step_targets is not None:
            item["step_targets"] = self.step_targets[index]
        if self.step_mask is not None:
            item["step_mask"] = self.step_mask[index]
        return item


class RecoveryDataset(Dataset):
    """C²-VA Stage B 恢复数据（v6b，prepare_mw_recovery.py 输出）。

    每样本一个恢复 transition：
    - vision_tokens_t [1, 64, 768]：扰动分支当前步的 4 帧窗口 V-JEPA 特征；
    - proprio / prev_action：归一化状态与前一执行动作（v5 空间）；
    - expert_action：专家恢复动作（executed 空间，clip 后）——a^{E,δ}；
    - step_index：0..5（branch 内恢复步，对应 Action Token 索引 i）；
    - c_perturbed / c_nominal：P 投影的扰动/名义分支状态（e_i = c^δ − c^0）；
    - branch_id：branch 标识（收缩指标按 branch 聚合）。
    语言条件从 payload 的按钮任务切片广播（C² 为单任务训练）。
    """

    REQUIRED = (
        "vision_tokens_t",
        "proprio",
        "prev_action",
        "expert_action",
        "step_index",
        "branch_id",
        "c_perturbed",
        "c_nominal",
    )

    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        missing = [key for key in self.REQUIRED if key not in payload]
        if missing:
            raise ValueError(f"missing tensors in recovery dataset: {missing}")
        self.payload = payload
        self.length = int(payload["expert_action"].shape[0])
        if self.length == 0:
            raise ValueError("recovery dataset is empty")
        for key in self.REQUIRED:
            if payload[key].ndim == 0 or payload[key].shape[0] != self.length:
                raise ValueError(f"recovery tensor {key} sample count mismatch")
        if payload["vision_tokens_t"].ndim != 4:
            raise ValueError("vision_tokens_t must have shape [N, 1, tokens, vision_dim]")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        payload = self.payload
        return {
            "vision_tokens": payload["vision_tokens_t"][index],  # [1, 64, 768]
            "language_hidden": payload["language_hidden"][0],
            "language_mask": payload["language_mask"][0],
            "proprio": payload["proprio"][index],
            "prev_action": payload["prev_action"][index],
            "expert_action": payload["expert_action"][index],
            "step_index": payload["step_index"][index],
            "branch_id": payload["branch_id"][index],
            "c_perturbed": payload["c_perturbed"][index],
            "c_nominal": payload["c_nominal"][index],
        }


class E2EDataset(Dataset):
    """Raw video frames + instruction text for end-to-end fine-tuning.

    Each sample carries the uint8 vision windows [T, W, 3, H, W] of its
    decision points plus the instruction string; state/action tensors match
    the feature dataset contract.
    """

    def __init__(self, path: Path, min_sequence_length: int = 4) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        missing = [
            key
            for key in ("video_frames", "instructions", "proprio", "previous_action", "actions")
            if key not in payload
        ]
        if missing:
            raise ValueError(f"missing tensors in e2e dataset: {missing}")
        self.payload = payload
        self.length = int(payload["actions"].shape[0])
        if self.length == 0:
            raise ValueError("e2e dataset is empty")
        frames = payload["video_frames"]
        if frames.ndim != 6:
            raise ValueError("video_frames must have shape [N,T,W,3,H,W]")
        if len(payload["instructions"]) != self.length:
            raise ValueError("instructions must have one entry per sample")
        sequence_length = frames.shape[1]
        if sequence_length < min_sequence_length:
            raise ValueError(
                f"e2e training requires T>={min_sequence_length}, got T={sequence_length}"
            )
        if "pair_id" in payload and "instruction_id" in payload:
            # 配对 E2E 数据（2026-08-07）：pair 结构与 feature 数据一致（每 pair 两行）。
            self.pair_groups = build_pair_groups(
                payload["pair_id"], payload["instruction_id"]
            )
        else:
            # 旧数据无 pair 字段：无配对，走 --single-task 兼容路径。
            self.pair_groups = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        payload = self.payload
        has_pairs = "pair_id" in payload and "instruction_id" in payload
        return {
            "video_frames": payload["video_frames"][index],
            "instruction": payload["instructions"][index],
            "proprio": payload["proprio"][index],
            "previous_action": payload["previous_action"][index],
            "actions": payload["actions"][index],
            # 旧数据缺失时回退：pair_id=index（每样本唯一）、instruction_id=0。
            "pair_id": int(payload["pair_id"][index]) if has_pairs else index,
            "instruction_id": int(payload["instruction_id"][index]) if has_pairs else 0,
        }


class PairedBatchSampler(Sampler[list[int]]):
    """Yield two different instructions from every selected decision pair."""

    def __init__(self, dataset: Dataset, batch_size: int, seed: int = 0) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("paired batch_size must be a positive even number")
        groups = getattr(dataset, "pair_groups", None)
        if groups is None:
            # 泛化（2026-08-07）：任意带 payload pair_id/instruction_id 的
            # 数据集（如无 pair_groups 属性的旧式数据集）也能配对采样。
            payload = dataset.payload
            if "pair_id" not in payload or "instruction_id" not in payload:
                raise ValueError(
                    "paired sampling requires pair_id/instruction_id in the dataset"
                )
            groups = build_pair_groups(payload["pair_id"], payload["instruction_id"])
        if not groups:
            raise ValueError("paired training requires at least one pair_id")
        self.groups = groups
        self.pairs_per_batch = batch_size // 2
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.groups) / self.pairs_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pair_ids = list(self.groups)
        random_generator.shuffle(pair_ids)
        batch: list[int] = []
        for pair_id in pair_ids:
            by_instruction = self.groups[pair_id]
            instructions = random_generator.sample(list(by_instruction), 2)
            batch.extend(
                random_generator.choice(by_instruction[instruction])
                for instruction in instructions
            )
            if len(batch) == self.pairs_per_batch * 2:
                yield batch
                batch = []
        if batch:
            yield batch


def synthetic_sequence(
    config: VACompoundConfig,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Build paired smoke data; each pair differs only in language at t=0."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("synthetic paired batch_size must be even")
    if sequence_length < 2:
        raise ValueError("synthetic sequence must contain at least two steps")
    pair_count = batch_size // 2

    def duplicate_pairs(value: Tensor) -> Tensor:
        return value.repeat_interleave(2, dim=0)

    vision = duplicate_pairs(
        torch.randn(
            pair_count,
            sequence_length,
            16,
            config.vision_dim,
            device=device,
        )
    )
    proprio = duplicate_pairs(
        torch.randn(pair_count, sequence_length, config.proprio_dim, device=device)
    )
    previous_action = duplicate_pairs(
        torch.randn(pair_count, sequence_length, config.action_dim, device=device)
    )
    instruction_id = torch.arange(2, device=device).repeat(pair_count)
    pair_id = torch.arange(pair_count, device=device).repeat_interleave(2)
    language_by_instruction = torch.randn(2, 8, config.language_dim, device=device)
    language = language_by_instruction[instruction_id]

    visual_signal = vision[..., : config.action_dim].mean(dim=2)
    previous_visual = torch.cat(
        (torch.zeros_like(visual_signal[:, :1]), visual_signal[:, :-1]),
        dim=1,
    )
    language_signal = language[:, :, : config.action_dim].mean(dim=1)[:, None]
    base = torch.tanh(visual_signal + 0.5 * previous_visual + language_signal)
    horizon = torch.linspace(
        0.0,
        0.1,
        config.action_horizon,
        device=device,
    )[None, None, :, None]
    actions = base[:, :, None, :].expand(-1, -1, config.action_horizon, -1) + horizon
    return {
        "vision_tokens": vision,
        "language_hidden": language,
        "language_mask": torch.ones(batch_size, 8, dtype=torch.bool, device=device),
        "proprio": proprio,
        "previous_action": previous_action,
        "actions": actions,
        "pair_id": pair_id,
        "instruction_id": instruction_id,
    }


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def ensure_sequence(
    batch: dict[str, Tensor],
    min_sequence_length: int,
) -> dict[str, Tensor]:
    if batch["vision_tokens"].ndim != 4 or batch["actions"].ndim != 4:
        raise ValueError("vision/actions must be paired short sequences")
    sequence_length = batch["vision_tokens"].shape[1]
    if sequence_length < min_sequence_length:
        raise ValueError(
            f"paired VA training requires T>={min_sequence_length}, got T={sequence_length}"
        )
    return batch


def sample_flow_matching_inputs(
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a straight probability path from Gaussian noise to expert actions.
    """
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, sequence, horizon, action_dim]")
    noise = torch.randn_like(actions)
    flow_time = torch.rand(
        actions.shape[:2],
        device=actions.device,
        dtype=actions.dtype,
    )
    tau = flow_time[:, :, None, None]
    noisy_actions = (1.0 - tau) * noise + tau * actions
    target_velocity = actions - noise
    return noisy_actions, flow_time, target_velocity


def sample_pair_intervention(
    actions: Tensor,
    partner: Tensor,
    probe_tau_max: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared-source counterfactual flow intervention (2026-08-07 upgrade).

    For each genuine same-state pair (i, j), all non-language flow inputs are
    identical: the probe x = (1-tau)*eps + tau*mid_{ij} with mid_{ij} =
    (a_i + a_j)/2 and a per-pair tau ~ U[0, probe_tau_max].  Only the language
    condition differs, so any velocity difference at the probe is attributable
    to language.  tau=0 is included (x = eps, the canonical shared source);
    tau in (0, 0.5] adds the shared midpoint probe that constrains the
    mid-flow field.  Targets are the linear-FM vector field values
    u_i = (a_i - x) / (1 - tau).
    """
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, sequence, horizon, action_dim]")
    if partner.shape != (actions.shape[0],):
        raise ValueError("partner must have shape [batch]")

    batch = actions.shape[0]
    device = actions.device
    dtype = actions.dtype
    shared_noise = torch.randn_like(actions[:, 0])
    # Per-pair shared tau (identical for both partners).
    shared_tau = torch.rand(batch, device=device, dtype=dtype) * probe_tau_max
    mid = torch.zeros_like(actions[:, 0])
    for left in range(batch):
        right = int(partner[left])
        if left < right:
            shared_noise[right] = shared_noise[left]
            shared_tau[right] = shared_tau[left]
            mid[left] = 0.5 * (actions[left, 0] + actions[right, 0])
            mid[right] = mid[left]
    # Pairs without a genuine partner get tau=0 with x=eps (well-defined).
    tau_b = shared_tau[:, None, None]
    probe = (1.0 - tau_b) * shared_noise + tau_b * mid
    denom = (1.0 - tau_b).clamp_min(1e-6)
    target_velocity = (actions[:, 0] - probe) / denom
    return probe, shared_tau, target_velocity


def rollout_policy(
    model: VACompoundPolicy,
    batch: dict[str, Tensor],
    noisy_actions: Tensor,
    flow_time: Tensor,
    *,
    text_backbone=None,
    scene_teacher=None,
    tasks=None,
) -> tuple[Tensor, Tensor]:
    if model.config.plan_resampler:
        # Plan-Cache 方案 B：首决策场景摘要（vision 全局均值）→ plan tokens，
        # cat 进语言序列后统一 build 缓存（与闭环"episode 首帧建缓存"语义一致）。
        scene_summary = batch["vision_tokens"][:, 0].mean(dim=1)  # [B, vision_dim]
        language_cache = model.build_plan_cache(
            scene_summary, batch["language_hidden"], batch.get("language_mask")
        )
    elif model.config.scene_teacher:
        # Plan-Cache 方案 A：Qwen 看场景 teacher 在线计算 readout plan hidden。
        if text_backbone is None or scene_teacher is None or tasks is None:
            raise ValueError("scene_teacher config requires text_backbone/scene_teacher/tasks")
        instructions = [tasks[int(index)] for index in batch["instruction_id"]]
        scene_summary = batch["vision_tokens"][:, 0].mean(dim=1)
        plan, _ = text_backbone.encode_with_scene(
            instructions, scene_summary, scene_teacher.scene_projector, scene_teacher.readout_tokens
        )
        plan = plan.to(dtype=batch["language_hidden"].dtype)
        language_hidden = torch.cat((batch["language_hidden"], plan), dim=1)
        language_mask = batch.get("language_mask")
        if language_mask is None:
            language_mask = torch.ones(
                language_hidden.shape[:2], dtype=torch.bool, device=language_hidden.device
            )
        else:
            language_mask = torch.cat(
                (
                    language_mask,
                    torch.ones(plan.shape[:2], dtype=torch.bool, device=plan.device),
                ),
                dim=1,
            )
        language_cache = model.build_language_cache(language_hidden, language_mask)
    else:
        language_cache = model.build_language_cache(
            batch["language_hidden"],
            batch.get("language_mask"),
        )
    visual_memory = None
    predicted_velocities = []
    action_conditions = []
    memories: list[VisualMemory] | None = [] if model.config.future_predict else None
    direct_predictions = [] if model.config.direct_head else None
    c2_references = [] if model.config.c2_controller else None
    for time_index in range(batch["vision_tokens"].shape[1]):
        condition, visual_memory = model.encode_condition(
            batch["vision_tokens"][:, time_index],
            batch["proprio"][:, time_index],
            batch["previous_action"][:, time_index],
            language_cache=language_cache,
            visual_memory=visual_memory,
            return_visual_memory=True,
        )
        if model.config.c2_controller:
            # C²-VA Stage B：c_current = P(当前决策视觉均值)；解码
            # a = clip(ū − K·(c_current − c̄))；reference 序列供 L_f 监督。
            c_current = model.control_projector(batch["vision_tokens"][:, time_index])
            params = model.controller_params(condition, c_current)
            direct_predictions.append(params.apply_controller(c_current))
            c2_references.append(params.reference)
        elif direct_predictions is not None:
            # C²-VA Stage A：Direct Head 一次前向解码完整 chunk（无采样噪声）。
            direct_predictions.append(model.decode_actions(condition))
        else:
            predicted_velocities.append(
                model.flow_velocity(
                    condition,
                    noisy_actions[:, time_index],
                    flow_time[:, time_index],
                )
            )
        action_conditions.append(condition)
        if memories is not None:
            memories.append(visual_memory)
    out = (
        torch.stack(
            direct_predictions if direct_predictions is not None else predicted_velocities,
            dim=1,
        ),
        torch.stack(action_conditions, dim=1),
    )
    if c2_references is not None:
        return out + (torch.stack(c2_references, dim=1),)
    if memories is not None:
        return out + (memories,)
    return out


def paired_partner_indices(pair_id: Tensor, instruction_id: Tensor) -> Tensor:
    """Return one different-instruction partner for every batch row."""
    if pair_id.ndim != 1 or instruction_id.shape != pair_id.shape:
        raise ValueError("pair_id and instruction_id must have matching shape [B]")
    partner = torch.empty_like(pair_id, dtype=torch.long)
    for value in torch.unique(pair_id).tolist():
        indices = torch.nonzero(pair_id == value, as_tuple=False).flatten()
        if indices.numel() != 2:
            raise ValueError("every training batch must contain exactly two rows per pair_id")
        first, second = indices[0], indices[1]
        if instruction_id[first] == instruction_id[second]:
            raise ValueError("paired batch rows must use different instruction_id values")
        partner[first] = second
        partner[second] = first
    return partner


def semantic_pair_loss(
    predicted: Tensor,
    target: Tensor,
    partner: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared-source counterfactual pair loss (2026-08-07 upgrade).

    Genuine pairs only (partner[i] > i, each pair counted once):
      abs_loss  = L(v(C_i, x, tau), u_i) + L(v(C_j, x, tau), u_j)
      delta     = L(v(C_i, x, tau) - v(C_j, x, tau), u_i - u_j)
    with x/tau shared within the pair (see sample_pair_intervention), so
    absolute terms pin both endpoints of the language condition and the
    delta term pins the language-caused flow bifurcation.  Samples without a
    genuine partner contribute zero (pair_id=arange MW data keeps pair=0).
    """
    if predicted.ndim != 3 or target.shape != predicted.shape:
        raise ValueError("paired velocities must have shape [batch, horizon, action_dim]")
    left = torch.nonzero(partner > torch.arange(partner.shape[0], device=partner.device), as_tuple=False).flatten()
    if left.numel() == 0:
        return (
            predicted.new_zeros(()),
            predicted.new_zeros(()),
            predicted.new_zeros(()),
        )
    right = partner[left]
    abs_loss = F.smooth_l1_loss(predicted[left], target[left]) + F.smooth_l1_loss(
        predicted[right], target[right]
    )
    delta_loss = F.smooth_l1_loss(
        predicted[left] - predicted[right], target[left] - target[right]
    )
    loss = abs_loss + delta_loss
    predicted_magnitude = (predicted[left] - predicted[right]).detach().abs().mean()
    target_magnitude = (target[left] - target[right]).detach().abs().mean()
    return loss, predicted_magnitude, target_magnitude


def semantic_pair_loss_legacy(
    predicted: Tensor,
    target: Tensor,
    partner: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Old L_pair (ablation): delta-only smooth-L1 on partner differences.

    Kept bit-compatible with the pre-2026-08-07 behavior: full-batch indexing
    through ``partner`` (self-pairs contribute zero deltas).
    """
    if predicted.ndim != 3 or target.shape != predicted.shape:
        raise ValueError("paired velocities must have shape [batch, horizon, action_dim]")
    predicted_delta = predicted - predicted[partner]
    target_delta = target - target[partner]
    loss = F.smooth_l1_loss(predicted_delta, target_delta)
    predicted_magnitude = predicted_delta.detach().abs().mean()
    target_magnitude = target_delta.detach().abs().mean()
    return loss, predicted_magnitude, target_magnitude


def iter_forever(loader):
    """无限循环迭代 DataLoader（epoch 结束自动重启）。"""
    while True:
        yield from iter(loader)


def compute_c2_loss(
    model: VACompoundPolicy,
    clean_batch: dict[str, Tensor],
    rec_batch: dict[str, Tensor] | None,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, float]]:
    """C²-VA Stage B 损失（Codex 评审 2026-08-07 版）。

    L = L_action + λf·L_future + λr·L_recovery（λc ≡ 0，收缩仅指标）：
    - L_action：clean 解码动作 smooth_l1（全 4 维）+ recovery 样本 token i
      解码动作 vs 专家恢复动作（防止 ū 漂移）；
    - L_future：c̄_i vs 期望视觉投影目标（clean 用 v6a per-step 目标，只算
      token 0..5；recovery 用名义分支投影 c^0_i），λf=0.1；
    - L_recovery：Huber(K_i·e_i, sg(ū_i − a^{E,δ}_i))，e_i = sg(c^δ_i − c^0_i)
      （数据给；‖e‖≈0 的样本无信息，mask 掉），λr=1.0。
    """
    logs: dict[str, float] = {}
    predictions, _, references = rollout_policy(model, clean_batch, None, None)
    act = F.smooth_l1_loss(predictions, clean_batch["actions"])
    logs["act"] = float(act.item())
    logs["arm"] = float(F.smooth_l1_loss(predictions[..., :3], clean_batch["actions"][..., :3]).item())
    logs["grip"] = float(F.smooth_l1_loss(predictions[..., 3:], clean_batch["actions"][..., 3:]).item())
    # L_f（clean）：c̄ vs v6a per-chunk-step 目标（token 0..min(5, H-1)）。
    n_future = min(6, references.shape[2])
    mask = clean_batch.get("step_mask")
    if mask is not None:
        m = mask.reshape(-1, 1, 1, 1)  # [B,1,1,1]
        denom = max(float(m.sum().item()) * n_future, 1.0)
        future = (
            F.smooth_l1_loss(
                references[:, :, :n_future] * m,
                clean_batch["step_targets"][:, :, :n_future] * m,
                reduction="sum",
            )
            / denom
        )
    else:
        future = F.smooth_l1_loss(references[:, :, :n_future], clean_batch["step_targets"][:, :, :n_future])
    logs["future"] = float(future.item())
    rec_act = rec_future = rec_residual = act.new_zeros(())
    rec_used = 0
    if rec_batch is not None and len(rec_batch["expert_action"]) > 0:
        rb = rec_batch
        condition = model.encode_condition(
            rb["vision_tokens"][:, 0],
            rb["proprio"],
            rb["prev_action"],
            language_cache=model.build_language_cache(
                rb["language_hidden"], rb["language_mask"]
            ),
        )
        params = model.controller_params(condition, rb["c_perturbed"])
        decoded = params.apply_controller(rb["c_perturbed"])
        rows = torch.arange(condition.shape[0], device=condition.device)
        step_index = rb["step_index"]
        u_i = params.nominal[rows, step_index]
        gain_i = params.gain[rows, step_index]
        error_i = (rb["c_perturbed"] - rb["c_nominal"]).detach()  # sg(c^δ − c^0)
        correction = torch.einsum("bac,bc->ba", gain_i, error_i)  # K_i·e_i [B, A]
        residual = (u_i - rb["expert_action"]).detach()  # sg(ū − a^{E,δ})
        mask = error_i.norm(dim=-1) > 1e-4  # e≈0 的样本对 K 无信息（如动作注入第 0 步）
        if bool(mask.any().item()):
            rec_residual = F.smooth_l1_loss(correction[mask], residual[mask])
            rec_used = int(mask.sum().item())
        rec_future = F.smooth_l1_loss(params.reference[rows, step_index], rb["c_nominal"])
        rec_act = F.smooth_l1_loss(decoded[rows, step_index], rb["expert_action"])
    logs["rec"] = float(rec_residual.item())
    logs["rec_future"] = float(rec_future.item())
    logs["rec_act"] = float(rec_act.item())
    logs["rec_used"] = float(rec_used)
    loss = (
        act
        + rec_act
        + args.c2_lambda_f * future
        + args.c2_lambda_f * rec_future
        + args.c2_lambda_r * rec_residual
    )
    return loss, logs


def compute_contract_metrics(
    payload: dict,
    *,
    rho6: float = 0.8,
    rho1: float | None = None,
    mask: Tensor | None = None,
) -> dict[str, float]:
    """held-out recovery 数据的收缩指标（Codex 修正 1：不参与训练）。

    d_i = ‖c_i^δ − c_i^0‖₂/√C（每恢复步的扰动偏差）；
    M_c1 = mean_i max(0, d_{i+1} − ρ₁·d_i)（ρ₁ = ρ₆^(1/6) ≈ 0.9635，每原始步）；
    M_c6 = max(0, d_5 − ρ₆·d_0)（ρ₆ = 0.8，6 步周期，按 branch 首末）。
    若 K 学会了恢复，d_i 应随 i 收缩 → M_c 下降。
    """
    c_perturbed = payload["c_perturbed"]
    c_nominal = payload["c_nominal"]
    if mask is not None:
        c_perturbed = c_perturbed[mask]
        c_nominal = c_nominal[mask]
    control_dim = c_perturbed.shape[-1]
    d = (c_perturbed - c_nominal).norm(dim=-1) / math.sqrt(control_dim)
    if rho1 is None:
        rho1 = rho6 ** (1.0 / 6.0)
    by_branch: dict[int, dict[int, float]] = defaultdict(dict)
    if mask is not None:
        branch_ids = payload["branch_id"][mask].tolist()
        step_indices = payload["step_index"][mask].tolist()
    else:
        branch_ids = payload["branch_id"].tolist()
        step_indices = payload["step_index"].tolist()
    for branch_id, step, value in zip(
        branch_ids, step_indices, d.tolist(), strict=True
    ):
        by_branch[int(branch_id)][int(step)] = value
    m1_terms: list[float] = []
    mc6_terms: list[float] = []
    d0_terms: list[float] = []
    d5_terms: list[float] = []
    for by_step in by_branch.values():
        steps = sorted(by_step)
        for left, right in zip(steps, steps[1:]):
            if right == left + 1:
                m1_terms.append(max(0.0, by_step[right] - rho1 * by_step[left]))
        if steps[0] == 0:
            d0_terms.append(by_step[0])
        if steps[-1] == 5:
            d5_terms.append(by_step[5])
        if len(steps) >= 2 and steps[0] == 0 and steps[-1] == 5:
            mc6_terms.append(max(0.0, by_step[5] - rho6 * by_step[0]))
    return {
        "d0": sum(d0_terms) / len(d0_terms) if d0_terms else 0.0,
        "d5": sum(d5_terms) / len(d5_terms) if d5_terms else 0.0,
        "M_c1": sum(m1_terms) / len(m1_terms) if m1_terms else 0.0,
        "M_c6": sum(mc6_terms) / len(mc6_terms) if mc6_terms else 0.0,
        "rho1": rho1,
        "rho6": rho6,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired multi-goal VA compound flow-matching trainer"
    )
    parser.add_argument("--data", type=Path, help="optional paired precomputed .pt dataset")
    parser.add_argument(
        "--e2e-data",
        type=Path,
        help="raw video/text dataset for end-to-end fine-tuning (V-JEPA+Qwen in graph)",
    )
    parser.add_argument("--lora-rank", type=int, default=0, help="Qwen LoRA rank (0 keeps Qwen fully frozen)")
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument(
        "--vision-unfreeze-all",
        action="store_true",
        help="truly full V-JEPA unfreezing (stem + all blocks + norms) for e2e "
        "training (2026-08-06 user instruction + Codex design)",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=None,
        help="V-JEPA blocks to unfreeze (default: all)",
    )
    parser.add_argument(
        "--qwen-unfreeze-blocks",
        type=int,
        default=0,
        help="Qwen decoder layers to unfreeze (0 = LoRA instead)",
    )
    parser.add_argument(
        "--semantic-adapter",
        action="store_true",
        help="第三种方案（2026-08-07）：冻结 Qwen 先验 + 仅顶部层 LoRA + "
        "zero-init 门控融合（与 --lora-rank>0 / --qwen-unfreeze-blocks 互斥，"
        "仅 --e2e-data 路径）",
    )
    parser.add_argument("--semantic-lora-rank", type=int, default=8)
    parser.add_argument("--semantic-top-layers", type=int, default=4)
    parser.add_argument(
        "--semantic-anchor-weight",
        type=float,
        default=0.0,
        help="anchor_loss 权重（锚定层 hidden 不偏离冻结先验；0 关闭）",
    )
    parser.add_argument(
        "--semantic-geometry-weight",
        type=float,
        default=0.0,
        help="geometry_loss 权重（G 矩阵几何约束；0 关闭）",
    )
    parser.add_argument(
        "--semantic-anchor-layers",
        type=str,
        default="",
        help="逗号分隔的 anchor 层索引（默认：被 LoRA 适配的顶部层）",
    )
    parser.add_argument(
        "--compile-task",
        action="store_true",
        help="compile-task（Stage A，2026-08-07）：SemanticCompiler 把视觉 token/"
        "语义历史/场景变化编译为语义 tokens 追加到语言序列，每 --compile-every "
        "步重编译（仅 --e2e-data 路径；与 --scene-teacher/--plan-resampler 互斥）",
    )
    parser.add_argument(
        "--compile-every",
        type=int,
        default=4,
        help="compile-task 重编译间隔（每 compile_every 步 + t=0）",
    )
    parser.add_argument(
        "--compile-n-scene",
        type=int,
        default=16,
        help="compile-task 从 visual tokens 抽取的场景 token 数（adaptive pooling）",
    )
    parser.add_argument(
        "--compile-n-readout",
        type=int,
        default=16,
        help="SemanticCompiler readout token 数（第二轮架构重构 token 放开）",
    )
    parser.add_argument(
        "--semantic-act-grad-scale",
        type=float,
        default=0.1,
        help="η_act（第二轮架构重构）:SemanticAdapter LoRA 参数梯度缩放系数"
        "（语义适配器动作侧梯度，默认 0.1；1.0 = 不缩放）",
    )
    parser.add_argument(
        "--semantic-lora-suffixes",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="SemanticAdapter LoRA 投影后缀（逗号分隔；默认全 7 种 "
        "q/k/v/o/gate/up/down，第二轮架构重构可配置子集）",
    )
    parser.add_argument(
        "--language-max-length",
        type=int,
        default=64,
        help="Qwen 语言 tokenizer max_length（e2e 路径透传）",
    )
    parser.add_argument(
        "--role-query",
        action="store_true",
        help="第二轮架构重构：learned role queries 替代语言 mask-weighted mean "
        "摘要（TaskResampler 与 action_query_cond 共享同一 RoleQueryResampler）",
    )
    parser.add_argument(
        "--role-query-tokens",
        type=int,
        default=16,
        help="RoleQueryResampler 的 role query 数（token 放开）",
    )
    parser.add_argument(
        "--dual-attention",
        action="store_true",
        help="第二轮架构重构：非 sequential VA 层动作 query 拆 physical/semantic "
        "双注意力（sequential 层保持旧共享路径；与 --sequential-coupling 同开时"
        "仅警告，sequential 层不拆）",
    )
    parser.add_argument(
        "--flow-semantic",
        action="store_true",
        help="第二轮架构重构：flow head 逐层读语义上下文（compile readout "
        "tokens，需 --compile-task 才有上下文；flow_cond=adaln 时经 cross-attn 注入）",
    )
    parser.add_argument(
        "--training-stage",
        choices=("a", "b", "c"),
        default=None,
        help="训练阶段契约（None 不校验）：a = --compile-task（禁 --semantic-adapter/"
        "anchor/geometry 损失）；b = --semantic-adapter；c = b 的全部要求 + "
        "推荐 --vision-unfreeze-all（仅提示不强制）",
    )
    parser.add_argument("--qwen-lr", type=float, default=1e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument(
        "--language-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--vision-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--e2e-pooling", choices=("flat", "spatial"), default="flat")
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="train Flow Matching without the unavailable multi-instruction pair loss",
    )
    parser.add_argument("--mode", choices=("bidir_va", "uni_a"), default="bidir_va")
    parser.add_argument(
        "--attention-variant",
        choices=("flat", "smc"),
        default="flat",
        help="shared-softmax attention variant: 'flat' (baseline) or 'smc' "
        "source-measure correction (log N_s subtracted per source before softmax)",
    )
    parser.add_argument(
        "--action-query-cond",
        action="store_true",
        help="Qwen-conditioned action queries (2026-08-06 GPT 方案 A): language "
        "summary -> MLP -> per-horizon query offsets, zero-init so training starts "
        "identical to the static-query baseline",
    )
    parser.add_argument(
        "--memory-split",
        action="store_true",
        help="causal-decomposed memory (2026-08-07): protected evidence memory "
        "(K/V-only, gated by vision+state) + dynamic task workspace (gated)",
    )
    parser.add_argument("--evidence-tokens", type=int, default=16)
    parser.add_argument("--task-tokens", type=int, default=8)
    parser.add_argument(
        "--future-predict-weight",
        type=float,
        default=0.0,
        help="weight of the future-latent prediction loss (2026-08-07 "
        "审阅落地③; 0 disables it; requires precomputed vision features)",
    )
    parser.add_argument(
        "--future-predict",
        action="store_true",
        help="enable the future-latent predictor module (审阅落地③)",
    )
    parser.add_argument(
        "--sequential-coupling",
        type=int,
        default=0,
        help="every N-th VA layer uses sequential A->V/T->A coupling "
        "(0 = all-joint, legacy behavior; 2026-08-07 审阅落地④)",
    )
    parser.add_argument(
        "--flow-cond",
        choices=("entry", "adaln"),
        default="entry",
        help="flow head conditioning: entry (legacy, add at input) or adaln "
        "(per-layer AdaLN-Zero + cross-attention; 2026-08-07)",
    )
    parser.add_argument(
        "--evsm",
        action="store_true",
        help="evidence-verified speculative memory (2026-08-07 Codex 主推): "
        "task proposals go to scratch and are committed only when the "
        "future-latent prediction matches observed vision (requires "
        "--memory-split --future-predict)",
    )
    parser.add_argument("--evsm-kappa", type=float, default=0.02)
    parser.add_argument("--evsm-temp", type=float, default=0.005)
    parser.add_argument(
        "--plan-resampler",
        action="store_true",
        help="Plan-Cache 方案 B: scene-conditioned plan tokens from the "
        "PlanResampler (scene summary + language -> 8 plan tokens appended to "
        "the language cache; VA attention untouched)",
    )
    parser.add_argument(
        "--scene-teacher",
        action="store_true",
        help="Plan-Cache 方案 A: Qwen 看场景 teacher -- online scene-conditioned "
        "readout plan hidden via QwenTextBackbone.encode_with_scene (frozen "
        "Qwen with gradients; projector/readout are trained; requires --data "
        "with metadata.tasks)",
    )
    parser.add_argument(
        "--direct-head",
        action="store_true",
        help="C²-VA Stage A: replace flow matching with a deterministic direct "
        "action head (2-layer MLP -> tanh) regressing normalized executed-action "
        "labels (v5 data: denorm -> clip(raw,-1,1) -> renorm). Default off = "
        "the existing flow path, unchanged.",
    )
    parser.add_argument(
        "--c2-controller",
        action="store_true",
        help="C²-VA Stage B（Codex 评审版）：收缩控制型 VA——每 Action Token = "
        "{ū, c̄, K}，执行 a_i = clip(ū_i − K_i·(c_current − c̄_i), −1, 1)。"
        "K 由恢复残差损失（v6b）监督，L_contract 仅作 held-out 指标（λc 必须 0）。"
        "需要 --direct-head --single-task --data；与 --scene-teacher/"
        "--plan-resampler/--future-predict/--evsm 互斥。",
    )
    parser.add_argument(
        "--c2-v6a",
        type=Path,
        default=Path("data/mw_buttonpress_v6a.pt"),
        help="v6a per-chunk-step 期望视觉目标（P 投影后 [N, T, 6, control_dim]）",
    )
    parser.add_argument(
        "--c2-v6b",
        type=Path,
        default=Path("data/mw_buttonpress_v6b.pt"),
        help="v6b 恢复数据（prepare_mw_recovery.py 输出；含冻结 PCA 投影权重）",
    )
    parser.add_argument("--c2-lambda-f", type=float, default=0.1, help="λf：c̄ 未来目标权重")
    parser.add_argument("--c2-lambda-r", type=float, default=1.0, help="λr：恢复残差损失权重")
    parser.add_argument(
        "--c2-lambda-c",
        type=float,
        default=0.0,
        help="收缩损失权重。离线 successor 对 K 无梯度（Codex 判决），必须为 0；"
        "收缩只作 held-out 指标记录。",
    )
    parser.add_argument(
        "--c2-recovery-ratio",
        type=float,
        default=0.25,
        help="每 batch 中恢复样本占比（clean:recovery = (1−r):r，默认 3:1）",
    )
    parser.add_argument(
        "--c2-unfreeze-stage-a",
        action="store_true",
        help="C² 首轮默认冻结 Stage A 参数（VA 复合体 + Direct Head），只训练 "
        "reference_head/gain_head（P 恒冻结）；此 flag 解锁全量微调",
    )
    parser.add_argument(
        "--c2-contract-every",
        type=int,
        default=500,
        help="在 held-out 恢复数据上打印收缩指标（d_i / M_c）的间隔步数",
    )
    parser.add_argument(
        "--c2-contract-rho6",
        type=float,
        default=0.8,
        help="6 步周期的收缩因子 ρ₆（每原始步 ρ₁ = ρ₆^(1/6)）",
    )
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        help="per-head RMSNorm on Q/K in the VA coupling layers (Su Shen QK-Norm)",
    )
    parser.add_argument(
        "--vision-pooling",
        choices=("flat", "spatial"),
        default="flat",
        help="vision feature variant: 'flat' (A) or 'spatial' (B) pooled tokens",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="must be even")
    parser.add_argument("--sequence-length", type=int, default=4, help="synthetic BPTT length")
    parser.add_argument("--min-sequence-length", type=int, default=4)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-start-atol", type=float, default=0.0)
    parser.add_argument(
        "--pair-start-cosine",
        type=float,
        default=0.0,
        help="vision-side pair contract: require first-state feature cosine "
        ">= this (LIBERO same-scene forks: 0.99; 0.0 = legacy strict atol)",
    )
    parser.add_argument("--min-pair-action-delta", type=float, default=1e-3)
    parser.add_argument(
        "--pair-probe-tau-max",
        type=float,
        default=0.5,
        help="shared-source CF probe: tau ~ U[0, max] per pair (0.5 = "
        "source + midpoint probes; 0.0 = source-only, the legacy tau=0 point)",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("shared_cf", "legacy"),
        default="shared_cf",
        help="shared_cf = shared-source counterfactual (abs + delta, "
        "probe-tau>=0); legacy = old tau=0 delta-only loss (ablation)",
    )
    parser.add_argument("--flow-steps", type=int, default=8, help="deployment Euler steps")
    parser.add_argument(
        "--va-layers",
        type=int,
        default=4,
        help="VACouplingLayer count in the decision stack (depth probe)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--prev-dropout",
        type=float,
        default=0.0,
        help="probability of zeroing previous_action per training sample (0 = off). "
        "P0-1 closed-loop prev self-excitation contract fix (2026-08-06 Codex): "
        "training uses teacher-forced prev, deployment uses the model's own output; "
        "dropout aligns the first-decision prev=0 condition. Features path only.",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.0,
        help="SAM sharpness-aware minimization radius (0 = off). Flatness-preserving "
        "finetuning keeps instruction following (arXiv:2606.23641): perturb weights by "
        "rho*grad/||grad|| then take the real step; costs one extra forward/backward.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume training from a feature-pipeline checkpoint (--data mode)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="periodically overwrite --save every N steps (atomic tmp+rename); "
        "0 disables periodic saves (crash loses the whole run)",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Arg 级契约校验（main 入口调用；纯参数检查，不加载数据/模型）。"""
    if args.steps <= 0 or args.flow_steps <= 0 or args.lr <= 0.0:
        raise ValueError("training steps, flow steps, and learning rate must be positive")
    if args.pair_loss_weight < 0.0:
        raise ValueError("pair loss weight must be non-negative")
    if not args.single_task and (args.batch_size < 2 or args.batch_size % 2):
        raise ValueError("paired batch size must be even")
    if args.single_task and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.evsm and not (args.memory_split and args.future_predict):
        raise ValueError("--evsm requires --memory-split and --future-predict")
    if args.evsm and args.future_predict_weight <= 0.0:
        raise ValueError("--evsm requires a positive --future-predict-weight")
    if args.evsm and args.evsm_temp <= 0.0:
        raise ValueError("--evsm-temp must be positive")
    if args.compile_task and not args.e2e_data:
        raise ValueError(
            "--compile-task requires --e2e-data (online SemanticCompiler path only)"
        )
    if args.compile_every < 1:
        raise ValueError("--compile-every must be >= 1")
    if args.compile_task and (args.scene_teacher or args.plan_resampler):
        raise ValueError(
            "--compile-task is mutually exclusive with "
            "--scene-teacher/--plan-resampler"
        )
    if args.training_stage == "a":
        if not args.compile_task:
            raise ValueError("--training-stage a requires --compile-task")
        if args.semantic_adapter:
            raise ValueError("--training-stage a forbids --semantic-adapter")
        if args.semantic_anchor_weight != 0.0 or args.semantic_geometry_weight != 0.0:
            raise ValueError(
                "--training-stage a requires --semantic-anchor-weight=0 and "
                "--semantic-geometry-weight=0"
            )
    elif args.training_stage in ("b", "c"):
        if not args.semantic_adapter:
            raise ValueError(
                f"--training-stage {args.training_stage} requires --semantic-adapter"
            )
    if args.plan_resampler and args.scene_teacher:
        raise ValueError("--plan-resampler and --scene-teacher are mutually exclusive")
    if args.plan_resampler and args.e2e_data:
        raise ValueError("--plan-resampler is not supported with --e2e-data")
    if args.scene_teacher and (args.e2e_data or not args.data):
        raise ValueError("--scene-teacher requires --data with precomputed features (needs metadata.tasks)")
    if args.direct_head and args.e2e_data:
        raise ValueError("--direct-head is not supported with --e2e-data (e2e rollout is flow-specific)")
    if args.c2_controller and not args.direct_head:
        raise ValueError("--c2-controller requires --direct-head")
    if args.c2_controller and not args.single_task:
        raise ValueError("--c2-controller requires --single-task (v6a/v6b 为按钮任务单任务数据)")
    if args.c2_controller and not args.data:
        raise ValueError("--c2-controller requires --data (precomputed features)")
    if args.c2_controller and (
        args.future_predict or args.evsm or args.plan_resampler or args.scene_teacher
    ):
        raise ValueError(
            "--c2-controller 与 --future-predict/--evsm/--plan-resampler/--scene-teacher "
            "互斥（Codex 修正 10：Stage B 不接其他结构；768D future head 与 L_f 重复）"
        )
    if args.c2_controller and args.c2_lambda_c > 0.0:
        raise ValueError(
            "--c2-lambda-c 必须为 0：固定离线 successor 下收缩项对 K 无梯度"
            "（Codex 判决），收缩只作为 held-out 指标记录"
        )
    if args.c2_controller and not (0.0 <= args.c2_recovery_ratio < 1.0):
        raise ValueError("--c2-recovery-ratio must be in [0, 1)")
    if args.semantic_adapter and not args.e2e_data:
        raise ValueError(
            "--semantic-adapter requires --e2e-data (online Qwen path only; "
            "the precomputed-feature path has no online Qwen)"
        )
    if args.semantic_adapter and args.lora_rank != 0:
        raise ValueError(
            "--semantic-adapter is mutually exclusive with --lora-rank > 0 "
            "(top-layer LoRA only)"
        )
    if args.semantic_adapter and args.qwen_unfreeze_blocks > 0:
        raise ValueError(
            "--semantic-adapter is mutually exclusive with --qwen-unfreeze-blocks"
        )
    if args.semantic_adapter and args.semantic_top_layers <= 0:
        raise ValueError("--semantic-top-layers must be positive with --semantic-adapter")
    if args.semantic_anchor_weight < 0.0 or args.semantic_geometry_weight < 0.0:
        raise ValueError("semantic anchor/geometry weights must be non-negative")
    # 第二轮架构重构（2026-08-08）参数校验；role-query/dual-attention/
    # flow-semantic 与 --compile-task 无强制依赖（可独立开）。
    if args.role_query_tokens < 1:
        raise ValueError("--role-query-tokens must be >= 1")
    if args.compile_n_readout < 1:
        raise ValueError("--compile-n-readout must be >= 1")
    if args.language_max_length < 1:
        raise ValueError("--language-max-length must be >= 1")
    if args.semantic_act_grad_scale < 0.0:
        raise ValueError("--semantic-act-grad-scale must be non-negative")
    if not [s.strip() for s in args.semantic_lora_suffixes.split(",") if s.strip()]:
        raise ValueError(
            "--semantic-lora-suffixes must be a non-empty comma-separated list"
        )
    if args.dual_attention and args.sequential_coupling > 0:
        print(
            "warning: --dual-attention only splits the non-sequential VA layers; "
            "sequential layers keep the legacy shared path"
        )


def scale_semantic_lora_grads(text_backbone: nn.Module, scale: float) -> None:
    """η_act 梯度缩放（第二轮架构重构 2026-08-08）。

    SemanticAdapter 的 LoRA 参数承担语言→动作的语义适配，η_act < 1 抑制其对
    指令嵌入几何的过快扰动；非 LoRA 参数（门控/编译器/策略）不受影响。
    ``scale == 1.0`` 时为空操作。SAM 路径的两次 backward 后都调用（第一次
    缩放只改变扰动 e_w 的范数而非方向——ρ·g/‖g‖ 与缩放无关；实际步长由
    第二次缩放后的梯度决定，因此两次缩放才使 η_act 对 SAM 生效）。
    """
    if scale == 1.0:
        return
    for name, parameter in text_backbone.named_parameters():
        if (
            parameter.requires_grad
            and parameter.grad is not None
            and ("lora_a" in name or "lora_b" in name)
        ):
            parameter.grad.mul_(scale)


def save_checkpoint(args, config, model, e2e_model, scene_teacher=None) -> None:
    """原子保存 checkpoint（tmp 文件 + rename），供周期/最终保存复用。"""
    if not args.save:
        return
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if e2e_model is not None:
        payload = {
            "config": config.__dict__,
            "model": e2e_model.policy.state_dict(),
            "lora": {
                # semantic 模式下参数名带 "text_backbone." 前缀（QwenSemanticBackbone
                # 包装 QwenTextBackbone），一并剥离使 resume 能按 text_model 相对名匹配。
                name.removeprefix("text_backbone.").removeprefix("text_model."): parameter.detach().cpu()
                for name, parameter in e2e_model.text_backbone.named_parameters()
                if "lora_a" in name or "lora_b" in name
            },
            "qwen_state_dict": {
                name.removeprefix("text_model."): parameter.detach().cpu()
                for name, parameter in e2e_model.text_backbone.named_parameters()
                if parameter.requires_grad
                and "lora_a" not in name
                and "lora_b" not in name
                and not name.startswith("gate.")
            },
            "vjepa_state_dict": e2e_model.vision_backbone.model.state_dict(),
            "training_contract": {
                "paired_multi_goal": False,
                "action_decoder": "conditional_flow_matching",
                "vision_pooling": args.e2e_pooling,
                "flow_steps": args.flow_steps,
                "min_sequence_length": args.min_sequence_length,
                "e2e": True,
                "lora_rank": args.lora_rank,
                "unfreeze_blocks": (
                    args.unfreeze_blocks
                    if args.unfreeze_blocks is not None
                    else len(e2e_model.vision_backbone.model.blocks)
                ),
            },
        }
        gate = getattr(e2e_model.text_backbone, "gate", None)
        if gate is not None:
            payload["semantic_gate"] = gate.state_dict()
        compiler = getattr(e2e_model, "compiler", None)
        payload["semantic_compiler"] = (
            {
                key: value.detach().cpu()
                for key, value in compiler.state_dict().items()
            }
            if compiler is not None
            else None
        )
    else:
        payload = {
            "config": config.__dict__,
            "model": model.state_dict(),
            "training_contract": {
                "paired_multi_goal": not args.single_task,
                "action_decoder": (
                    "direct_head" if args.direct_head else "conditional_flow_matching"
                ),
                "c2_controller": args.c2_controller,
                "vision_pooling": args.vision_pooling,
                "flow_steps": args.flow_steps,
                "min_sequence_length": args.min_sequence_length,
                "pair_loss_weight": args.pair_loss_weight,
                "pair_start_atol": args.pair_start_atol,
                "min_pair_action_delta": args.min_pair_action_delta,
            },
        }
        if scene_teacher is not None:
            payload["scene_teacher"] = scene_teacher.state_dict()
    tmp_path = args.save.with_suffix(args.save.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(args.save)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021) 简洁实现。

    平坦化微调保留指令遵循（arXiv:2606.23641）：权重先沿梯度方向扰动
    rho * grad/||grad||，在 worst-case 邻域重算 loss，再走真实 Adam 步。
    训练循环：loss.backward() → first_step(zero_grad=True) →
    loss2.backward() → second_step(zero_grad=True)。
    """

    def __init__(self, params, base_optimizer, rho: float, **kwargs) -> None:
        assert rho >= 0.0, "SAM rho must be non-negative"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
                self.state[p].pop("e_w", None)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None) -> None:  # noqa: D102
        self.base_optimizer.step()

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norm = torch.zeros((), device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norm += p.grad.norm() ** 2
        return norm.sqrt()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.training_stage == "c" and not args.vision_unfreeze_all:
        print(
            "hint: --training-stage c recommends --vision-unfreeze-all "
            "(not enforced)"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    loader = None
    iterator = None
    model = None
    e2e_model = None
    if args.e2e_data:
        dataset = E2EDataset(args.e2e_data, min_sequence_length=args.min_sequence_length)
        payload = dataset.payload
        config = VACompoundConfig(
            language_dim=2048,
            vision_dim=768,
            action_horizon=int(payload["actions"].shape[-2]),
            action_dim=int(payload["actions"].shape[-1]),
            proprio_dim=int(payload["proprio"].shape[-1]),
            mode=args.mode,
            num_layers=args.va_layers,
            qk_norm=args.qk_norm,
            attention_variant=args.attention_variant,
            action_query_cond=args.action_query_cond,
            memory_split=args.memory_split,
            evidence_tokens=args.evidence_tokens,
            task_tokens=args.task_tokens,
            future_predict=args.future_predict,
            sequential_coupling=args.sequential_coupling,
            flow_cond=args.flow_cond,
            evsm=args.evsm,
            evsm_kappa=args.evsm_kappa,
            evsm_temp=args.evsm_temp,
            plan_resampler=args.plan_resampler,
            scene_teacher=args.scene_teacher,
            direct_head=args.direct_head,
            role_query=args.role_query,
            role_query_tokens=args.role_query_tokens,
            dual_attention=args.dual_attention,
            flow_semantic=args.flow_semantic,
        )
        if args.single_task:
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        else:
            loader = DataLoader(
                dataset,
                batch_sampler=PairedBatchSampler(dataset, args.batch_size, args.seed),
            )
        iterator = iter(loader)
        smoke_batch = None
    elif args.data:
        vision_key = (
            "vision_tokens_spatial" if args.vision_pooling == "spatial" else "vision_tokens"
        )
        c2_step_targets = None
        c2_step_mask = None
        c2_clean_n = None
        recovery_loader = None
        if args.c2_controller:
            # v6a per-chunk-step 期望视觉目标（与 v5 样本逐行对齐；C² L_f 用）。
            v6a_payload = torch.load(args.c2_v6a, map_location="cpu", weights_only=True)
            c2_step_targets = v6a_payload["step_targets"]
            c2_step_mask = v6a_payload.get("step_mask")
            # 每 batch 的 clean 样本数（clean:recovery = (1−r):r，r 默认 0.25）。
            c2_clean_n = min(
                max(int(round(args.batch_size * (1.0 - args.c2_recovery_ratio))), 1),
                args.batch_size - 1,
            )
        dataset = FeatureDataset(
            args.data,
            require_pairs=not args.single_task,
            min_sequence_length=args.min_sequence_length,
            pair_start_atol=args.pair_start_atol,
            min_pair_action_delta=args.min_pair_action_delta,
            pair_start_cosine=args.pair_start_cosine,
            vision_key=vision_key,
            step_targets=c2_step_targets,
            step_mask=c2_step_mask,
        )
        config = VACompoundConfig(
            language_dim=int(dataset.payload["language_hidden"].shape[-1]),
            vision_dim=int(dataset.payload[vision_key].shape[-1]),
            action_horizon=int(dataset.payload["actions"].shape[-2]),
            action_dim=int(dataset.payload["actions"].shape[-1]),
            proprio_dim=int(dataset.payload["proprio"].shape[-1]),
            mode=args.mode,
            num_layers=args.va_layers,
            qk_norm=args.qk_norm,
            attention_variant=args.attention_variant,
            action_query_cond=args.action_query_cond,
            memory_split=args.memory_split,
            evidence_tokens=args.evidence_tokens,
            task_tokens=args.task_tokens,
            future_predict=args.future_predict,
            sequential_coupling=args.sequential_coupling,
            flow_cond=args.flow_cond,
            evsm=args.evsm,
            evsm_kappa=args.evsm_kappa,
            evsm_temp=args.evsm_temp,
            plan_resampler=args.plan_resampler,
            scene_teacher=args.scene_teacher,
            direct_head=args.direct_head,
            c2_controller=args.c2_controller,
            role_query=args.role_query,
            role_query_tokens=args.role_query_tokens,
            dual_attention=args.dual_attention,
            flow_semantic=args.flow_semantic,
        )
        if args.single_task:
            loader = DataLoader(
                dataset,
                batch_size=c2_clean_n if args.c2_controller else args.batch_size,
                shuffle=True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_sampler=PairedBatchSampler(dataset, args.batch_size, args.seed),
            )
        iterator = iter(loader)
        smoke_batch = None
        if args.c2_controller:
            recovery_dataset = RecoveryDataset(args.c2_v6b)
            recovery_loader = DataLoader(
                recovery_dataset,
                batch_size=args.batch_size - c2_clean_n,
                shuffle=True,
            )
    else:
        config = VACompoundConfig(
            mode=args.mode, num_layers=args.va_layers, qk_norm=args.qk_norm,
            attention_variant=args.attention_variant,
            action_query_cond=args.action_query_cond,
            memory_split=args.memory_split,
            evidence_tokens=args.evidence_tokens,
            task_tokens=args.task_tokens,
            future_predict=args.future_predict,
            sequential_coupling=args.sequential_coupling,
            flow_cond=args.flow_cond,
            evsm=args.evsm,
            evsm_kappa=args.evsm_kappa,
            evsm_temp=args.evsm_temp,
            plan_resampler=args.plan_resampler,
            scene_teacher=args.scene_teacher,
            direct_head=args.direct_head,
            role_query=args.role_query,
            role_query_tokens=args.role_query_tokens,
            dual_attention=args.dual_attention,
            flow_semantic=args.flow_semantic,
        )
        smoke_batch = synthetic_sequence(
            config,
            args.batch_size,
            args.sequence_length,
            device,
        )

    if args.e2e_data:
        from va_compound.end_to_end import build_e2e_policy, parameter_groups

        semantic_anchor_layers = None
        if args.semantic_anchor_layers:
            semantic_anchor_layers = tuple(
                int(value) for value in args.semantic_anchor_layers.split(",")
            )
        e2e_model, counts = build_e2e_policy(
            config=config,
            device=device,
            language_dtype=args.language_dtype,
            vision_dtype=args.vision_dtype,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            unfreeze_blocks=args.unfreeze_blocks,
            qwen_unfreeze_blocks=args.qwen_unfreeze_blocks,
            pooling=args.e2e_pooling,
            vision_unfreeze_all=args.vision_unfreeze_all,
            semantic_adapter=args.semantic_adapter,
            semantic_lora_rank=args.semantic_lora_rank,
            semantic_top_layers=args.semantic_top_layers,
            semantic_anchor_layers=semantic_anchor_layers,
            semantic_lora_suffixes=tuple(
                s.strip() for s in args.semantic_lora_suffixes.split(",") if s.strip()
            ),
            language_max_length=args.language_max_length,
            compile_task=args.compile_task,
            compile_every=args.compile_every,
            n_scene_tokens=args.compile_n_scene,
            compile_n_readout=args.compile_n_readout,
        )
        e2e_model.train()
        # 冻结参数保持 eval()（2026-08-05 Codex v2）：nn.Module.train() 会把冻结的
        # V-JEPA 前段/patch embed 切回 train mode；dropout/drop-path 近零影响小，
        # 但冻结 BN/其他状态层必须保持 eval，协议才干净
        for module in e2e_model.vision_backbone.model.modules():
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.eval()
        optimizer = torch.optim.AdamW(
            parameter_groups(
                e2e_model,
                lora_lr=args.lora_lr,
                vision_lr=args.vision_lr,
                policy_lr=args.lr,
                qwen_lr=args.qwen_lr,
            ),
            weight_decay=1e-4,
        )
        semantic_log = ""
        if args.semantic_adapter:
            semantic_log = (
                f"semantic=top{counts['semantic_top_layers']}"
                f"(lora={counts['semantic_lora_layers']}) "
            )
        compile_log = ""
        if args.compile_task:
            compile_log = f"compile={counts['compile_every']} "
            if args.training_stage:
                compile_log += f"stage={args.training_stage} "
        print(
            f"e2e: {semantic_log}{compile_log}"
            f"lora_layers={counts['lora_layers']} "
            f"unfrozen_vjepa_blocks={counts['unfrozen_vjepa_blocks']} "
            f"unfrozen_qwen_blocks={counts['unfrozen_qwen_blocks']} "
            f"pooling={args.e2e_pooling} "
            f"contract={'single_task' if args.single_task else 'paired'}"
        )
    elif args.single_task:
        model = VACompoundPolicy(config).to(device)
        if args.c2_controller:
            # C² Stage B：冻结 PCA 控制投影 P 的权重来自 v6b 恢复数据的
            # recovery 差空间 top-16 PCA + whitening（Codex 修正 2，P 不端到端学）。
            pca = recovery_dataset.payload["pca"]
            model.control_projector.set_pca(pca["weight"], pca["bias"])
            if not args.c2_unfreeze_stage_a:
                # 首轮冻结 Stage A（VA 复合体 + Direct Head），只训 reference/gain head。
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.c2_head.requires_grad_(True)
                print(
                    "c2: Stage A 参数冻结（VA + Direct Head），仅训练 "
                    "reference_head/gain_head（P 为冻结 PCA）"
                )
            else:
                print("c2: --c2-unfreeze-stage-a 全量微调（P 仍冻结）")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=1e-4,
        )
    else:
        model = VACompoundPolicy(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Plan-Cache 方案 A：加载冻结 Qwen（local files only，fp16/bf16）+ 可训练
    # SceneTeacher（projector + readout tokens 进 optimizer）。指令文本从
    # dataset payload 的 metadata.tasks 按 instruction_id 重建（与数据 prep
    # 阶段 compute language_hidden 所用的字符串完全一致）。
    scene_teacher = None
    text_backbone = None
    tasks = None
    if args.scene_teacher:
        from va_compound.backbones import QwenTextBackbone, SceneTeacher

        tasks = dataset.payload["metadata"]["tasks"]
        if not tasks:
            raise ValueError("--scene-teacher requires metadata.tasks in the dataset")
        text_backbone = QwenTextBackbone.from_pretrained(
            device=device, dtype=args.language_dtype, local_files_only=True
        )
        scene_teacher = SceneTeacher(
            language_dim=config.language_dim, vision_dim=config.vision_dim
        ).to(device)
        optimizer = torch.optim.AdamW(
            [*model.parameters(), *scene_teacher.parameters()],
            lr=args.lr,
            weight_decay=1e-4,
        )
        print(
            f"scene_teacher: qwen={args.language_dtype} "
            f"projector+readout params={sum(p.numel() for p in scene_teacher.parameters()):,}"
        )

    if args.sam_rho > 0:
        optimizer = SAM(
            optimizer.param_groups,
            torch.optim.AdamW,
            rho=args.sam_rho,
            lr=args.lr,
            weight_decay=1e-4,
        )
        print(f"SAM enabled: rho={args.sam_rho} (2x forward per step)")

    if e2e_model is not None:
        e2e_model.train()
    else:
        model.train()
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        resume_config = resume_ckpt["config"]
        for key in ("num_layers", "hidden_dim", "action_dim", "proprio_dim", "mode"):
            if resume_config.get(key) != getattr(config, key):
                raise ValueError(
                    f"resume config mismatch on {key}: "
                    f"{resume_config.get(key)} vs {getattr(config, key)}"
                )
        if e2e_model is not None:
            e2e_model.policy.load_state_dict(resume_ckpt["model"])
            e2e_model.vision_backbone.model.load_state_dict(resume_ckpt["vjepa_state_dict"])
            if resume_ckpt.get("qwen_state_dict"):
                qwen_state = {
                    k.removeprefix("text_model."): v
                    for k, v in resume_ckpt["qwen_state_dict"].items()
                }
                e2e_model.text_backbone.text_model.load_state_dict(qwen_state, strict=False)
            if resume_ckpt.get("lora"):
                own = dict(e2e_model.text_backbone.text_model.named_parameters())
                for name, value in resume_ckpt["lora"].items():
                    clean = name.removeprefix("text_model.")
                    if clean in own:
                        own[clean].data.copy_(value)
            if resume_ckpt.get("semantic_gate"):
                gate = getattr(e2e_model.text_backbone, "gate", None)
                if gate is not None:
                    gate.load_state_dict(resume_ckpt["semantic_gate"])
            if resume_ckpt.get("semantic_compiler"):
                own_compiler = getattr(e2e_model, "compiler", None)
                if own_compiler is not None:
                    own_compiler.load_state_dict(
                        resume_ckpt["semantic_compiler"], strict=False
                    )
            print(f"e2e resumed from {args.resume}")
        else:
            if args.c2_controller:
                # Stage A → C² 迁移：仅允许新 C² keys（c2_head/control_projector）缺失；
                # P 的 PCA 权重随后由 v6b 注入覆盖（保持与当前数据一致）。
                has_c2_keys = any(
                    key.startswith(("c2_head.", "control_projector."))
                    for key in resume_ckpt["model"]
                )
                if not has_c2_keys:
                    missing, unexpected = model.load_state_dict(
                        resume_ckpt["model"], strict=False
                    )
                    allowed_missing = {
                        key for key in model.state_dict()
                        if key.startswith(("c2_head.", "control_projector."))
                    }
                    forbidden = set(missing) - allowed_missing
                    if forbidden:
                        raise ValueError(
                            "c2 resume migration: unexpected missing keys: "
                            f"{sorted(forbidden)[:8]}"
                        )
                    print(
                        f"resumed Stage A checkpoint (c2 migration): "
                        f"missing c2 keys = {len(missing)}, unexpected = {len(unexpected)}"
                    )
                else:
                    model.load_state_dict(resume_ckpt["model"])
            else:
                model.load_state_dict(resume_ckpt["model"])
            if scene_teacher is not None:
                if resume_ckpt.get("scene_teacher") is None:
                    raise ValueError(
                        "resume checkpoint has no scene_teacher weights (--scene-teacher run required)"
                    )
                scene_teacher.load_state_dict(resume_ckpt["scene_teacher"])
            print(f"resumed from {args.resume}")
    if args.c2_controller and recovery_loader is not None:
        # resume 之后再次注入 PCA：P 的权重恒取自当前 v6b 文件（冻结），
        # 与 ckpt 是否携带旧 P 权重无关。
        model.control_projector.set_pca(
            recovery_dataset.payload["pca"]["weight"],
            recovery_dataset.payload["pca"]["bias"],
        )
    clean_iter = None
    rec_iter = None
    if args.c2_controller:
        clean_iter = iter_forever(loader)
        rec_iter = iter_forever(recovery_loader)
    for step in range(1, args.steps + 1):
        rec_batch = None
        if args.c2_controller:
            # C² 3:1 混合：clean 部分（v5 + v6a 目标）+ recovery 部分（v6b）。
            batch = move_batch(next(clean_iter), device)
            rec_batch = move_batch(next(rec_iter), device)
        elif iterator is None:
            batch = smoke_batch
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
        if e2e_model is None:
            batch = ensure_sequence(batch, args.min_sequence_length)
        if args.prev_dropout > 0.0:
            # P0-1 previous_action 闭环自激的契约修复（2026-08-06 Codex 判决顺序）：
            # 训练时以 prev_dropout 概率把 previous_action 置零（与闭环首决策
            # prev=0 对齐），迫使策略不依赖"prev 永远为真值"的 teacher-forcing 优势。
            prev_mask = (
                torch.rand(batch["previous_action"].shape[0], device=device)
                < args.prev_dropout
            )
            batch["previous_action"] = batch["previous_action"] * (
                ~prev_mask
            ).view(-1, 1, 1).float()
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(args, config, model, e2e_model, scene_teacher)
            print(f"step={step} periodic checkpoint saved to {args.save}")

        if model.config.c2_controller:
            # C²-VA Stage B 训练路径（SAM 不适用：冻结参数下无意义，跳过）。
            loss, c2_logs = compute_c2_loss(model, batch, rec_batch, args)
            contract_log = ""
            if args.c2_contract_every > 0 and step % args.c2_contract_every == 0:
                heldout = ~recovery_dataset.payload["split"]
                metrics = compute_contract_metrics(
                    recovery_dataset.payload,
                    rho6=args.c2_contract_rho6,
                    mask=heldout,
                )
                contract_log = (
                    f" contract[d0={metrics['d0']:.4f} d5={metrics['d5']:.4f} "
                    f"M_c1={metrics['M_c1']:.4f} M_c6={metrics['M_c6']:.4f} "
                    f"(ρ1={metrics['rho1']:.4f}/ρ6={metrics['rho6']:.2f})]"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(
                f"step={step} mode={args.mode} contract=single_c2 "
                f"sequence={batch['actions'].shape[1]} "
                f"loss={loss.item():.6f} act={c2_logs['act']:.6f} "
                f"arm={c2_logs['arm']:.6f} grip={c2_logs['grip']:.6f} "
                f"future={c2_logs['future']:.6f} rec={c2_logs['rec']:.6f} "
                f"rec_future={c2_logs['rec_future']:.6f} rec_act={c2_logs['rec_act']:.6f} "
                f"rec_used={int(c2_logs['rec_used'])} "
                f"grad={float(gradient_norm):.6f}{contract_log}"
            )
            continue

        noisy_actions, flow_time, target_velocity = sample_flow_matching_inputs(
            batch["actions"]
        )

        def compute_loss(batch, noisy_actions, flow_time):
            """Forward + flow/pair loss（SAM 二次前向复用同一噪声/配对）。"""
            evsm_gates = []
            if e2e_model is not None:
                mb = move_batch(batch, device)
                predicted_velocity, action_conditions, _ = e2e_model.rollout(
                    mb["video_frames"],
                    mb["instruction"],
                    mb["proprio"],
                    mb["previous_action"],
                    noisy_actions,
                    flow_time,
                    compile_every=args.compile_every,
                )
                flow_loss = e2e_model.policy.flow_matching_loss(predicted_velocity, target_velocity)
                if args.single_task:
                    pair_loss = flow_loss.new_zeros(())
                    pred_delta = flow_loss.new_zeros(())
                    tgt_delta = flow_loss.new_zeros(())
                else:
                    # 配对 E2E（2026-08-07）：与 feature 路径同构的共享源
                    # 反事实干预——语言是 pair 内唯一输入差，速度场差即语言贡献。
                    partner = paired_partner_indices(batch["pair_id"], batch["instruction_id"])
                    pair_noise, pair_time, pair_target_velocity = sample_pair_intervention(
                        batch["actions"], partner, probe_tau_max=args.pair_probe_tau_max
                    )
                    pair_predicted_velocity = e2e_model.policy.flow_velocity(
                        action_conditions[:, 0], pair_noise, pair_time
                    )
                    pair_loss, pred_delta, tgt_delta = semantic_pair_loss(
                        pair_predicted_velocity, pair_target_velocity, partner
                    )
                future_loss = flow_loss.new_zeros(())
            else:
                rollout = rollout_policy(
                    model,
                    batch,
                    noisy_actions,
                    flow_time,
                    text_backbone=text_backbone,
                    scene_teacher=scene_teacher,
                    tasks=tasks,
                )
                if model.config.future_predict:
                    predicted_velocity, action_conditions, memories = rollout
                else:
                    predicted_velocity, action_conditions = rollout
                if model.config.direct_head:
                    # C²-VA Stage A：Direct Head 回归归一化 executed 动作标签
                    # （v5 数据：一个执行动作 ↔ 一个标签）。pair 是 flow 专属
                    # 损失（共享噪声/中点探针），direct 模式跳过（打印 pair=0）。
                    flow_loss = F.smooth_l1_loss(predicted_velocity, batch["actions"])
                    pair_loss = flow_loss.new_zeros(())
                    pred_delta = flow_loss.new_zeros(())
                    tgt_delta = flow_loss.new_zeros(())
                else:
                    flow_loss = model.flow_matching_loss(predicted_velocity, target_velocity)
                    if args.single_task:
                        pair_loss = flow_loss.new_zeros(())
                        pred_delta = flow_loss.new_zeros(())
                        tgt_delta = flow_loss.new_zeros(())
                    else:
                        partner = paired_partner_indices(batch["pair_id"], batch["instruction_id"])
                        if args.pair_mode == "legacy":
                            # 旧 L_pair：tau=0 共享噪声 delta-only（消融对照）。
                            pair_noise, pair_time, pair_target_velocity = sample_pair_intervention(
                                batch["actions"], partner, probe_tau_max=0.0
                            )
                            pair_predicted_velocity = model.flow_velocity(
                                action_conditions[:, 0], pair_noise, pair_time
                            )
                            pair_loss, pred_delta, tgt_delta = semantic_pair_loss_legacy(
                                pair_predicted_velocity, pair_target_velocity, partner
                            )
                        else:
                            pair_noise, pair_time, pair_target_velocity = sample_pair_intervention(
                                batch["actions"],
                                partner,
                                probe_tau_max=args.pair_probe_tau_max,
                            )
                            pair_predicted_velocity = model.flow_velocity(
                                action_conditions[:, 0], pair_noise, pair_time
                            )
                            pair_loss, pred_delta, tgt_delta = semantic_pair_loss(
                                pair_predicted_velocity, pair_target_velocity, partner
                            )
                if model.config.future_predict:
                    # 未来 latent 预测（审阅落地③）：从 (E_t, T_t, C_t) 预测
                    # t+1 决策点的冻结 V-JEPA 特征均值；target 来自预计算特征
                    # bank（冻结特征即 stop-grad，此处再显式 detach 保险）。
                    # EVSM 下 T_t 取暂存提议（spec）——未来预测对应"执行当前
                    # 提议后的状态"；提交验证在 encode_condition 内完成。
                    future_terms = []
                    for t in range(batch["vision_tokens"].shape[1] - 1):
                        _ev = memories[t].evidence
                        _tk = (
                            memories[t].task_spec
                            if model.config.evsm and memories[t].task_spec is not None
                            else memories[t].task
                        )
                        # EVSM 验证门在第 t+1 周期对第 t 周期的 spec 生效；
                        # 与 future term (t -> t+1) 对齐的是 memories[t+1].gate。
                        if model.config.evsm and memories[t + 1].gate is not None:
                            evsm_gates.append(memories[t + 1].gate)
                        try:
                            pred_future = model.future_predictor(
                                action_conditions[:, t], _ev, _tk
                            )
                        except RuntimeError:
                            print(
                                f"[DEBUG] future mismatch at step {step} t={t}: "
                                f"cond={tuple(action_conditions[:, t].shape)} "
                                f"ev={tuple(_ev.shape) if _ev is not None else None} "
                                f"task={tuple(_tk.shape) if _tk is not None else None} "
                                f"batch_vt={tuple(batch['vision_tokens'].shape)}",
                                flush=True,
                            )
                            raise
                        target_future = batch["vision_tokens"][:, t + 1].mean(dim=1)
                        future_terms.append(
                            model.future_predictor.future_loss(pred_future, target_future)
                        )
                    future_loss = (
                        torch.stack(future_terms).mean()
                        if future_terms
                        else flow_loss.new_zeros(())
                    )
                else:
                    future_loss = flow_loss.new_zeros(())
            semantic_anchor_loss = flow_loss.new_zeros(())
            semantic_geom_loss = flow_loss.new_zeros(())
            if args.semantic_adapter:
                # 第三种方案（2026-08-07）：anchor/geometry 约束。prior 侧永远
                # no_grad（encode_prior_states 带 @torch.no_grad）；adapted 单次
                # 前向取 output_hidden_states 复用给 anchor + geometry。
                semantic = e2e_model.text_backbone  # QwenSemanticBackbone
                unique = list(dict.fromkeys(mb["instruction"]))
                need_anchor = args.semantic_anchor_weight > 0.0 and bool(
                    semantic.anchor_layers
                )
                need_geom = args.semantic_geometry_weight > 0.0
                if need_anchor or need_geom:
                    layers_needed = list(semantic.anchor_layers)
                    if need_geom and (semantic.num_layers - 1) not in layers_needed:
                        layers_needed.append(semantic.num_layers - 1)
                    prior_layers, prior_mask = semantic.encode_prior_states(unique, layers_needed)
                    adapted_layers, _ = semantic.encode_adapted_states(unique, layers_needed)
                    if need_anchor:
                        semantic_anchor_loss = semantic.anchor_loss(
                            prior_layers, adapted_layers
                        )
                    if need_geom:
                        semantic_geom_loss = semantic.geometry_loss(
                            prior_layers[semantic.num_layers - 1],
                            adapted_layers[semantic.num_layers - 1],
                            prior_mask,
                        )
            evsm_gate_mean = (
                sum(evsm_gates) / len(evsm_gates) if evsm_gates else None
            )
            total = (
                flow_loss
                + args.pair_loss_weight * pair_loss
                + args.future_predict_weight * future_loss
            )
            if args.semantic_adapter:
                total = (
                    total
                    + args.semantic_anchor_weight * semantic_anchor_loss
                    + args.semantic_geometry_weight * semantic_geom_loss
                )
            return (
                total,
                flow_loss,
                pair_loss,
                pred_delta,
                tgt_delta,
                future_loss,
                evsm_gate_mean,
                semantic_anchor_loss,
                semantic_geom_loss,
            )

        (
            loss,
            flow_loss,
            pair_loss,
            predicted_delta,
            target_delta,
            future_loss,
            evsm_gate_mean,
            semantic_anchor_loss,
            semantic_geom_loss,
        ) = compute_loss(batch, noisy_actions, flow_time)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if e2e_model is not None and args.semantic_adapter:
            scale_semantic_lora_grads(
                e2e_model.text_backbone, args.semantic_act_grad_scale
            )
        clip_params = (
            e2e_model.parameters()
            if e2e_model is not None
            else (
                [*model.parameters(), *scene_teacher.parameters()]
                if scene_teacher is not None
                else model.parameters()
            )
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        if args.sam_rho > 0:
            # SAM：先沿梯度方向扰动权重（worst-case 邻域），重算 loss 后走真实步。
            # η_act 对两次 backward 的梯度都缩放（见 scale_semantic_lora_grads
            # 文档：first_step 的扰动方向与缩放无关，实际步长由第二次缩放后的
            # 梯度决定，因此两次缩放才使 η_act 对 SAM 生效）。
            optimizer.first_step(zero_grad=True)
            loss2, _, _, _, _, _, _, _, _ = compute_loss(batch, noisy_actions, flow_time)
            loss2.backward()
            if e2e_model is not None and args.semantic_adapter:
                scale_semantic_lora_grads(
                    e2e_model.text_backbone, args.semantic_act_grad_scale
                )
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
        gate_log = (
            f" evsm_gate={evsm_gate_mean:.3f}" if evsm_gate_mean is not None else ""
        )
        semantic_log = ""
        if args.semantic_adapter:
            semantic_log = (
                f" anchor={semantic_anchor_loss.item():.6f} "
                f"geom={semantic_geom_loss.item():.6f}"
            )
        compile_step_log = ""
        if args.compile_task:
            compile_step_log = f" compile={args.compile_every}"
            if args.training_stage:
                compile_step_log += f" stage={args.training_stage}"
        print(
            f"step={step} mode={args.mode} contract="
            f"{'e2e_single' if e2e_model is not None else ('single' if args.single_task else 'paired')} "
            f"sequence={noisy_actions.shape[1]} "
            f"loss={loss.item():.6f} flow={flow_loss.item():.6f} "
            f"pair={pair_loss.item():.6f} future={future_loss.item():.6f} "
            f"goal_delta={predicted_delta.item():.6f}/"
            f"{target_delta.item():.6f} grad={float(gradient_norm):.6f}"
            f"{gate_log}{semantic_log}{compile_step_log}"
        )

    if args.save:
        save_checkpoint(args, config, model, e2e_model, scene_teacher)


if __name__ == "__main__":
    main()
