from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from va_compound.flow import ACTION_MASK_KEYS

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
        local_tokens: Tensor | None = None,
        coords: Tensor | None = None,
    ) -> None:
        if min_sequence_length < 2:
            raise ValueError("min_sequence_length must be at least 2")
        if pair_start_atol < 0.0 or min_pair_action_delta <= 0.0:
            raise ValueError("pair tolerances must be non-negative with action delta positive")

        payload = torch.load(path, map_location="cpu", weights_only=True)
        missing = [key for key in self.REQUIRED if key not in payload]
        if local_tokens is not None:
            # ST288 特征路径（--local-slots-data）：vision_tokens 由 local_tokens
            # 派生（__getitem__ 同源给出 vision_tokens/vision_tokens_st），
            # payload 无需存储 18.6GB 占位（2026-08-09 E7 长轨迹数据）。
            missing = [key for key in missing if key != "vision_tokens"]
        if missing:
            raise ValueError(f"missing tensors in dataset: {missing}")
        if vision_key not in payload and local_tokens is None:
            raise ValueError(
                f"dataset has no vision variant '{vision_key}'; "
                f"available: {sorted(key for key in payload if key.startswith('vision_tokens'))}"
            )
        self.payload = payload
        self.vision_key = vision_key
        self.length = int(payload["actions"].shape[0])
        if self.length == 0:
            raise ValueError("training dataset is empty")
        if any(
            payload[key].shape[0] != self.length
            for key in self.REQUIRED
            if key in payload  # vision_tokens 缺失时由 local_tokens 派生（ST288 路径）
        ):
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
        if local_tokens is not None:
            if tuple(local_tokens.shape[:2]) != (self.length, 4):
                raise ValueError(
                    f"local_tokens must be [N, 4, tokens, dim], got {tuple(local_tokens.shape)}"
                )
        if coords is not None and coords.ndim != 2:
            raise ValueError(f"coords must be [N, 3], got {tuple(coords.shape)}")
        self.local_tokens = local_tokens
        self.coords = coords

        self._validate_shapes(min_sequence_length)
        self.pair_groups = self._build_pair_groups() if require_pairs else {}
        if require_pairs:
            self._validate_pair_contract(
                pair_start_atol, min_pair_action_delta, pair_start_cosine
            )

    def _validate_shapes(self, min_sequence_length: int) -> None:
        vision = (
            self.local_tokens
            if self.local_tokens is not None
            else self.payload[self.vision_key]
        )
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
        vision_seq = (
            self.local_tokens
            if self.local_tokens is not None
            else self.payload[self.vision_key]
        )
        sequence_keys = (
            vision_seq,
            self.payload["proprio"],
            self.payload["previous_action"],
            self.payload["actions"],
        )
        sequence_lengths = {int(key.shape[1]) for key in sequence_keys}
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
        item = {key: self.payload[key][index] for key in self.REQUIRED if key in self.payload}
        for key in (
            "episode_id",
            "world_rank_shuffle_action",
            "world_rank_shuffle_mask",
        ):
            if key in self.payload:
                item[key] = self.payload[key][index]
        for key in ACTION_MASK_KEYS:
            if key in self.payload:
                item[key] = self.payload[key][index]
        if self.local_tokens is not None:
            vision = self.local_tokens[index]  # [4, 288, 768] ST288（与 vision_tokens_st 同源）
            item["vision_tokens"] = vision
            item["vision_tokens_st"] = vision
            item["coords"] = self.coords
        else:
            item["vision_tokens"] = self.payload[self.vision_key][index]
        if "language_mask" in self.payload:
            item["language_mask"] = self.payload["language_mask"][index]
        if self.step_targets is not None:
            item["step_targets"] = self.step_targets[index]
        if self.step_mask is not None:
            item["step_mask"] = self.step_mask[index]
        return item


class IndexedDataset(Dataset):
    """带原始行索引的 Dataset 包装（--perturb-data 混批需索引 payload 视觉/帧）。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, dict]:
        return index, self.dataset[index]
