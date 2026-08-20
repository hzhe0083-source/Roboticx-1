from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.wmrm import matched_no_fixed_point_perm, wmrm_world_loss
from va_compound.world_supervision import (
    action_top10_oracle_straight_through_gap_loss,
    masked_reduction as masked_world_reduction,
    stage_supervision_weights,
    static_copy_anchor_loss,
    transition_mask as world_transition_mask,
    visual_no_regression_loss,
    visual_world_loss,
)


WORLD_SUPERVISION_CONTRACT = "visual_motion_oracle_stgap_v7"
WORLD_TRANSITION_CONTRACT = "current_first6_and_next_first_v1"
WORLD_LOSS_COMPONENT_WEIGHTS = {
    "all": 0.25,
    "motion": 0.25,
    "top20": 0.50,
}
WORLD_STAGE_AUXILIARY_DECAY = 0.25
WORLD_LOGGED_BRANCH_CONTRACT = "matched_context_full_forward_v1"
WORLD_ACTION_DONOR_CONTRACT = "train_split_task_cross_episode_proprio_nearest_v1"
PEER_WORLD_TOPOLOGY_CONTRACT = "pre_stage_snapshot_parallel_va_world_v1"
PEER_WORLD_ACTION_SOURCE_CONTRACT = "deterministic_readout_main_explicit_env_override_supervision_v1"
PEER_WORLD_READOUT_CONTRACT = {
    "loss": "smooth_l1_logged_h6_v1",
    "validity": "world_transition_mask_all_h6_current_and_next_first_v1",
    "stage_supervision": "final_peer_stage_only_v1",
    "reduction": "masked_mean_over_valid_transitions_v1",
    "diagnostic": "rmse_same_mask_and_stage_v1",
}
FEATURE_AUTOCAST_CONTRACT = "bf16_nograd_decode_cache_isolated_v1"
WORLD_NO_REGRESSION = {
    "all_ratio": 1.0,
    "weight": 1.0,
    "components": ["all"],
}
WORLD_STATIC_COPY_CONSTRAINT = {
    "static_ratio": 1.0,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "copy_budget_hinge_plus_always_copy_anchor_v1",
    "reduction": "stage_aux_weighted_masked_mean",
    "boundary": "1.00_detached_copy_each_stage",
}
_WORLD_ACTION_RANKING_COMMON = {
    "top10_min_relative_margin": 0.12,
    "weight": 1.0,
    "negatives": ["shuffle"],
    "diagnostic_negatives": ["zero"],
    "context": "logged_stage_detached_pair",
    "gradient": "oracle_motion_straight_through_exact_gap_v1",
}


def world_action_ranking_contract(
    stage_mode: str,
    per_sample_cap: float | None = None,
) -> dict[str, object]:
    if stage_mode == "final":
        stage = "final_direct_matched_context"
        schedule = "final_each_valid_transition"
    elif stage_mode == "cycle":
        stage = "rotating_8stage_direct_matched_context"
        schedule = "(global_step+time_index)%num_stages"
    else:
        raise ValueError(f"unsupported World action-ranking stage mode: {stage_mode}")
    contract = {
        "stage": stage,
        **_WORLD_ACTION_RANKING_COMMON,
        "schedule": schedule,
    }
    if per_sample_cap is not None:
        contract["per_sample_cap"] = per_sample_cap
    return contract


# Programmatic callers retain a deterministic default; experiment runners pass
# the final/cycle choice explicitly and persist the resolved dictionary.
WORLD_ACTION_RANKING = world_action_ranking_contract("cycle")


def wmrm_next_feature_target(
    model: VACompoundPolicy,
    batch: dict[str, Tensor],
    time_index: int,
    *,
    dense_evidence: dict[int, Tensor] | None = None,
    metric_g: Tensor | None = None,
) -> Tensor:
    """Next VA-cycle visual feature. Target encoder is stop-grad (JEPA-style)."""
    nxt = time_index + 1
    kind = getattr(model.config, "wmrm_target", "dino")
    if kind == "metric":
        if metric_g is None:
            raise ValueError("wmrm_target=metric requires metric_g")
        return metric_g[:, nxt]
    if kind == "vjepa":
        if dense_evidence is None or 11 not in dense_evidence:
            raise ValueError("wmrm_target=vjepa requires dense_evidence[11] (H11)")
        return dense_evidence[11][:, nxt].mean(dim=1).detach()
    vision_next = batch.get("vision_tokens")
    if vision_next is None or nxt >= vision_next.shape[1]:
        raise ValueError("wmrm_target=dino requires vision_tokens at t+1 (VA cycle)")
    raw = vision_next[:, nxt]
    wmrm = getattr(model, "wmrm", None)
    if wmrm is not None and hasattr(wmrm, "encode_dino_map"):
        mapped = wmrm.encode_dino_map(raw)
        if mapped is not None:
            return mapped.detach()
    return raw.detach()
from va_compound.backbones import pool_flat_tokens, pool_mtvj_coarse_tokens
from va_compound.metric_roi import (
    DINO_METRIC_ROI_CONTRACT,
    METRIC_ROI_CONTRACT_VERSION,
    TASK35_METRIC_ROLE_CONTRACT,
    load_dino_metric_roi_checkpoint,
    load_metric_roi_checkpoint,
    metric_head_state_sha256,
    prepare_metric_roi_video,
    refine_metric_roi_positions,
    refine_metric_roi_positions_dino,
)
from va_compound.servo import InteractionServo
from scripts.mt50_difficulty import task_weights_for


ACTION_MASK_KEYS = ("action_valid_mask", "horizon_mask")

MTVJ_METRIC_HEAD_CONFIG_KEYS = (
    "lang_dim",
    "h_dim",
    "d_proj",
    "n_roles",
    "l2_norm",
    "learnable_temp",
    "temp_init",
    "freeze_bias",
    "mode_readout",
    "grid",
)
# 缺省即向后兼容的构造键（旧 checkpoint 无此键 → 填默认值，不视为不完整）。
_MTVJ_METRIC_HEAD_OPTIONAL_KEYS = frozenset({"grid"})
_MTVJ_METRIC_HEAD_CONFIG_DEFAULTS = {
    "lang_dim": 2048,
    "h_dim": 768,
    "d_proj": 192,
    "n_roles": 4,
    "l2_norm": False,
    "learnable_temp": False,
    "temp_init": 10.0,
    "freeze_bias": False,
    "mode_readout": False,
    "grid": 24,
}

# The action-policy relation encoder keeps its historical 8-D input shape, but
# all new all-task migrations use visibility-gated coordinates.  Keeping these
# names/version explicit prevents an old p-only checkpoint from being evaluated
# under the new semantics by accident.
MTVJ_METRIC_STATE_SOURCE = "p_times_visibility_flat"
MTVJ_METRIC_CONTRACT_VERSION = 3
MTVJ_LEGACY_METRIC_STATE_SOURCE = "p_flat"
MTVJ_LEGACY_METRIC_CONTRACT_VERSION = 2


# Action-only visual tower presets.  The policy config persists every resolved
# field, so eval does not have to guess a model's width/resolution later.
ACTION_VISION_SPECS = {
    "dinov2_vitl14_reg4": {
        "model_id": "vit_large_patch14_reg4_dinov2.lvd142m",
        "image_size": 224,
        "feature_dim": 1024,
        "output_layers": (11, 23),
    },
    "dinov3_vitb16": {
        "model_id": "vit_base_patch16_dinov3.lvd1689m",
        "image_size": 256,
        "feature_dim": 768,
        "output_layers": (5, 11),
    },
    "dinov3_vitl16": {
        "model_id": "vit_large_patch16_dinov3.lvd1689m",
        "image_size": 256,
        "feature_dim": 1024,
        "output_layers": (11, 23),
    },
}


def _mtvj_metric_positions(out, source: str = MTVJ_METRIC_STATE_SOURCE) -> Tensor:
    """Return the declared 8-D state; v2 stays reproducible, v3 gates visibility."""
    p = out.p
    if p.ndim != 3 or p.shape[-2:] != (4, 2):
        raise ValueError(f"MT-VJ out.p must be [N, 4, 2], got {tuple(p.shape)}")
    if source == MTVJ_LEGACY_METRIC_STATE_SOURCE:
        return p.reshape(p.shape[0], 8)
    if source != MTVJ_METRIC_STATE_SOURCE:
        raise ValueError(f"unknown MT-VJ metric state source: {source!r}")
    visibility = out.visibility
    if visibility.shape != p.shape[:-1]:
        raise ValueError(
            "MT-VJ out.visibility must match out.p roles: "
            f"{tuple(visibility.shape)} != {tuple(p.shape[:-1])}"
        )
    return (p * visibility.unsqueeze(-1)).reshape(p.shape[0], 8)


_mtvj_visibility_gated_positions = _mtvj_metric_positions


def _canonical_mtvj_metric_head_config(
    config: dict | None,
    *,
    require_complete: bool = False,
) -> dict:
    """Return the complete, weights-only-safe LanguageMetricField ctor contract.

    ``grid``（2026-08-15 新增）是向后兼容的缺省键：旧 checkpoint 无此键时
    填默认 24（V-JEPA），require_complete 不要求其存在。
    """
    raw = dict(config or {})
    if require_complete:
        required = set(MTVJ_METRIC_HEAD_CONFIG_KEYS) - _MTVJ_METRIC_HEAD_OPTIONAL_KEYS
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "主 checkpoint 缺少完整 mtvj_metric_head_config："
                f"missing={missing}"
            )
    values = {
        key: raw.get(key, default)
        for key, default in _MTVJ_METRIC_HEAD_CONFIG_DEFAULTS.items()
    }
    for key in ("lang_dim", "h_dim", "d_proj", "n_roles", "grid"):
        values[key] = int(values[key])
    for key in (
        "l2_norm",
        "learnable_temp",
        "freeze_bias",
        "mode_readout",
    ):
        values[key] = bool(values[key])
    values["temp_init"] = float(values["temp_init"])
    return values


def _mtvj_metric_head_constructor_config(metric_head: nn.Module) -> dict:
    """Read every constructor semantic from a live LanguageMetricField."""
    missing = [
        key for key in MTVJ_METRIC_HEAD_CONFIG_KEYS if not hasattr(metric_head, key)
    ]
    if missing:
        raise ValueError(
            "metric head 无法保存完整构造契约："
            f"missing attributes={missing}"
        )
    return _canonical_mtvj_metric_head_config(
        {key: getattr(metric_head, key) for key in MTVJ_METRIC_HEAD_CONFIG_KEYS},
        require_complete=True,
    )


def _mtvj_metric_checkpoint_identity(path: Path, checkpoint: dict) -> dict:
    """Fingerprint the immutable external metric checkpoint used for migration."""
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(resolved.stat().st_size),
        "contract": checkpoint.get("contract"),
    }


def _mtvj_metric_identity_mismatches(saved: dict, current: dict) -> dict:
    """Compare semantic identity fields; a copied identical file remains valid."""
    return {
        key: (saved.get(key), current.get(key))
        for key in ("sha256", "size_bytes", "contract")
        if saved.get(key) != current.get(key)
    }


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
            # 仅当存在真 pair（某组 id>=0 且 >1 行）时才构建 pair 组并严格校验；
            # 旧版单任务 e2e payload（libero_video/v2 的 pair_id 每组仅 1 行）
            # 视为无配对，走 --single-task 兼容路径（与 FeatureDataset 的
            # require_pairs=False 语义一致）。
            vals, counts = payload["pair_id"].unique(return_counts=True)
            has_real_pairs = bool(int(((vals > -1) & (counts > 1)).sum()))
            if has_real_pairs:
                self.pair_groups = build_pair_groups(
                    payload["pair_id"], payload["instruction_id"]
                )
            else:
                self.pair_groups = {}
        else:
            # 旧数据无 pair 字段：无配对，走 --single-task 兼容路径。
            self.pair_groups = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        payload = self.payload
        has_pairs = "pair_id" in payload and "instruction_id" in payload
        item = {
            "video_frames": payload["video_frames"][index],
            "instruction": payload["instructions"][index],
            "proprio": payload["proprio"][index],
            "previous_action": payload["previous_action"][index],
            "actions": payload["actions"][index],
            # 旧数据缺失时回退：pair_id=index（每样本唯一）、instruction_id=0。
            "pair_id": int(payload["pair_id"][index]) if has_pairs else index,
            "instruction_id": int(payload["instruction_id"][index]) if has_pairs else 0,
        }
        for key in ACTION_MASK_KEYS:
            if key in payload:
                item[key] = payload[key][index]
        return item


class IndexedDataset(Dataset):
    """带原始行索引的 Dataset 包装（--perturb-data 混批需索引 payload 视觉/帧）。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, dict]:
        return index, self.dataset[index]


class PairedBatchSampler(Sampler[list[int]]):
    """Yield two different instructions from every selected decision pair."""

    def __init__(self, dataset: Dataset, batch_size: int, seed: int = 0) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("paired batch_size must be a positive even number")
        groups = getattr(dataset, "pair_groups", None)
        if not groups:
            # 泛化（2026-08-07）：任意带 payload pair_id/instruction_id 的
            # 数据集（如无 pair_groups 属性的旧式数据集，或 require_pairs=False
            # 的空 groups——E 组打乱配对形态）也能配对采样。
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

    每个 epoch 严格产生 ``N // batch_size`` 个 batch；每个抽样块含至多
    ``block_batches`` 个同任务 batch（为保持真实权重，相邻同任务块可连续），
    在 JPEG 解码局部性与跨任务曝光之间取折中。
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
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if block_batches < 1:
            raise ValueError("block_batches must be positive")
        if instruction_id.ndim != 1 or episode_id.ndim != 1:
            raise ValueError("instruction_id/episode_id must be 1-D")
        if instruction_id.shape != episode_id.shape or instruction_id.numel() == 0:
            raise ValueError("instruction_id/episode_id must have the same non-zero length")
        if task_weights.ndim != 1:
            raise ValueError("task_weights must be 1-D")
        if sampling_mode not in {"weighted", "balanced"}:
            raise ValueError("sampling_mode must be 'weighted' or 'balanced'")
        self.task_ids = [int(value) for value in instruction_id.tolist()]
        self.episode_ids = [int(value) for value in episode_id.tolist()]
        self.task_w = task_weights.to(torch.float64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.block_batches = int(block_batches)
        self.sampling_mode = sampling_mode
        self.epoch = 0
        self.batch_cursor = 0
        self.by_task_episode: dict[int, dict[int, list[int]]] = {}
        for index, (task, episode) in enumerate(zip(self.task_ids, self.episode_ids, strict=True)):
            self.by_task_episode.setdefault(task, {}).setdefault(episode, []).append(index)
        self.tasks = sorted(self.by_task_episode)
        if self.tasks[-1] >= len(self.task_w) or bool((self.task_w[self.tasks] <= 0).any()):
            raise ValueError("task_weights must contain a positive entry for every task id")
        if self.sampling_mode == "balanced":
            active_weights = self.task_w[self.tasks]
            if not bool(torch.all(active_weights == active_weights[0])):
                raise ValueError("balanced sampling requires equal active task weights")
        self.task_probs = torch.stack([self.task_w[t] for t in self.tasks])
        self.task_probs = self.task_probs / self.task_probs.sum().clamp_min(1e-12)
        self._n = len(self.task_ids)
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
        }

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


EXACT_RESUME_VERSION = 2
EXACT_RUN_CONTRACT_VERSION = 1
WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION = (
    "wmrm_detach_proposal_stage_state_v1"
)
WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION = "wmrm_world_weight_1_to_0_5_v1"
WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION = (
    "wmrm_static_constraint_weight_4_to_2_v1"
)
WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION = "wmrm_action_rank_cap_none_to_0_2_v1"

# These only control how long/where the already-defined run is executed. They
# cannot change the next stochastic optimizer update and therefore are allowed
# to differ when an exact run is continued.
_EXACT_RUN_OPERATIONAL_ARGS = {
    "steps",
    "save",
    "save_every",
    "save_step_copies",
    "resume",
    "resume_exact",
    "resume_weights",
    # This selects a narrowly controlled compatibility policy for validation; it
    # does not become part of the semantic run contract saved after migration.
    "resume_exact_contract_migration",
    # Content/config identity is recorded separately, so an identical external
    # metric checkpoint copied to another filename is still the same input.
    "metric_visual_checkpoint",
    "mtvj_roi_checkpoint",
    # One-shot initialization migration.  The resulting head constructor,
    # weights and content identity are checkpointed, so replaying this flag is
    # neither necessary nor allowed during an exact continuation.
    "replace_mtvj_metric_head_from_external",
    "data",
}

# E7 WAM args/config enter the exact-run contract only through the conditional
# "wam" section (appended when --wam-joint is on). Excluding them
# unconditionally from argument_semantics/model_config keeps the WAM-off
# contract key-for-key identical to pre-WAM (main-era) checkpoints.
_WAM_CONTRACT_ARG_KEYS = {"wam_joint", "wam_alpha", "wam_ckpt"}


def _normalize_contract_value(value):
    """Convert runtime objects into deterministic weights-only-safe values."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, dict):
        return {
            str(key): _normalize_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported exact-run contract value {type(value).__name__}: {value!r}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_visual_world_training_split(
    payload: dict,
    data_path: Path,
    manifest_path: Path,
    *,
    va_world_mode: str = "legacy",
) -> dict[str, str]:
    """Validate the immutable episode-level train split before model startup."""

    from scripts.split_wam4va_episode_holdout import (
        MANIFEST_CONTRACT,
        PEER_SYNC_H6_CONTRACT,
        canonical_manifest_sha256,
        transition_mask as split_transition_mask,
    )

    resolved_data = data_path.expanduser().resolve(strict=True)
    resolved_manifest = manifest_path.expanduser().resolve(strict=True)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if manifest.get("contract") != MANIFEST_CONTRACT:
        raise ValueError(
            f"unexpected World split contract: {manifest.get('contract')!r}"
        )
    manifest_sha = canonical_manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != manifest_sha:
        raise ValueError("World split manifest canonical SHA mismatch")
    if Path(str(manifest.get("manifest_path", ""))).expanduser().resolve() != resolved_manifest:
        raise ValueError("World split manifest_path does not match the supplied file")

    metadata = payload.get("metadata") or {}
    if metadata.get("split_name") != "train":
        raise ValueError("visual World training requires metadata.split_name='train'")
    if metadata.get("split_contract") != manifest:
        raise ValueError("embedded split_contract differs from the external manifest")
    if metadata.get("split_manifest_sha256") != manifest_sha:
        raise ValueError("training payload split_manifest_sha256 mismatch")
    if Path(str(metadata.get("split_manifest_path", ""))).expanduser().resolve() != resolved_manifest:
        raise ValueError("training payload split_manifest_path mismatch")

    actions = payload.get("actions")
    action_valid = payload.get("action_valid_mask")
    recovery = payload.get("recovery_mask")
    task_ids = payload.get("instruction_id")
    episode_ids = payload.get("episode_id")
    if not isinstance(actions, Tensor) or actions.ndim != 4:
        raise ValueError("visual World actions must be [N,T,H,A]")
    if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
        raise ValueError("visual World actions must be finite floating-point values")
    peer_mode = va_world_mode == "peer_sync_h6"
    expected_shape = (4, 6, 4) if peer_mode else (4, 48, 4)
    expected_protocol = PEER_SYNC_H6_CONTRACT if peer_mode else MANIFEST_CONTRACT
    metadata_contract = metadata.get("contract")
    manifest_protocol = (manifest.get("data_protocol") or {}).get("contract")
    if tuple(actions.shape[1:]) != expected_shape:
        label = "peer_sync_h6" if peer_mode else "legacy visual World"
        raise ValueError(
            f"{label} training requires T={expected_shape[0]}/H={expected_shape[1]}/"
            f"A={expected_shape[2]}, got {tuple(actions.shape[1:])}"
        )
    if peer_mode:
        if metadata_contract != PEER_SYNC_H6_CONTRACT:
            raise ValueError(
                f"peer_sync_h6 requires metadata.contract={PEER_SYNC_H6_CONTRACT!r}"
            )
        if metadata.get("logged_action_chunk") != "full_h6":
            raise ValueError("peer_sync_h6 requires the full logged H6 action chunk")
        for key in ("parent_identity", "source_identities", "output_identity"):
            if not metadata.get(key):
                raise ValueError(f"peer_sync_h6 requires metadata.{key}")
    elif metadata_contract == PEER_SYNC_H6_CONTRACT:
        raise ValueError("legacy visual World rejects peer_sync_h6 data")
    if manifest_protocol != expected_protocol:
        raise ValueError(
            f"World split data protocol mismatch: expected {expected_protocol!r}, "
            f"got {manifest_protocol!r}"
        )
    expected_mask_shape = actions.shape[:-1]
    for name, value in (
        ("action_valid_mask", action_valid),
        ("recovery_mask", recovery),
    ):
        if (
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or value.shape != expected_mask_shape
        ):
            raise ValueError(
                f"{name} must be bool {tuple(expected_mask_shape)} for visual World"
            )
    for name, value in (("instruction_id", task_ids), ("episode_id", episode_ids)):
        if (
            not isinstance(value, Tensor)
            or value.ndim != 1
            or value.shape[0] != actions.shape[0]
            or value.dtype == torch.bool
            or value.is_floating_point()
        ):
            raise ValueError(f"{name} must be an integer [N] tensor")
    actual_tasks = sorted(int(value) for value in torch.unique(task_ids).tolist())
    if actual_tasks != [0, 16]:
        raise ValueError(
            "visual World joint training requires task ids [0,16], got "
            f"{actual_tasks}"
        )

    splits = manifest.get("splits") or {}
    train_contract = splits.get("train") or {}
    eval_contract = splits.get("eval") or {}
    if Path(str(train_contract.get("output_path", ""))).expanduser().resolve() != resolved_data:
        raise ValueError("World split train output_path does not match --data")
    if int(train_contract.get("windows", -1)) != int(actions.shape[0]):
        raise ValueError("World split train window count mismatch")
    if metadata.get("output_identity") != train_contract.get("output_identity"):
        raise ValueError("World split training output identity mismatch")
    source_contract = manifest.get("source") or {}
    if metadata.get("parent_identity") != source_contract:
        raise ValueError("World split training parent identity mismatch")
    if metadata.get("source_identities") != (
        source_contract.get("payload_source_identities") or []
    ):
        raise ValueError("World split training source identities mismatch")
    actual_episodes = sorted(int(value) for value in torch.unique(episode_ids).tolist())
    declared_episodes = sorted(int(value) for value in train_contract.get("episode_ids", []))
    if actual_episodes != declared_episodes:
        raise ValueError("World split train episode list mismatch")
    eval_episodes = {int(value) for value in eval_contract.get("episode_ids", [])}
    if set(actual_episodes) & eval_episodes:
        raise ValueError("World split train/eval episode leakage detected")

    expected_names = {0: "assembly-v3", 16: "door-unlock-v3"}
    manifest_tasks = {
        int(item["task_id"]): str(item.get("task_name"))
        for item in manifest.get("tasks", [])
    }
    if manifest_tasks != expected_names:
        raise ValueError(
            "World split task contract must be assembly-v3 + door-unlock-v3, got "
            f"{manifest_tasks}"
        )

    transition = split_transition_mask(action_valid)
    transition_stats = (train_contract.get("mask_stats") or {}).get("transition") or {}
    if (
        int(transition.sum()) != int(transition_stats.get("true", -1))
        or transition.numel() != int(transition_stats.get("total", -1))
        or not bool(transition.any())
    ):
        raise ValueError("World split transition-mask statistics mismatch")
    task_contracts = {
        int(item["task_id"]): item for item in train_contract.get("tasks", [])
    }
    if sorted(task_contracts) != [0, 16]:
        raise ValueError("World split train task list mismatch")
    for task_id, item in task_contracts.items():
        selected = task_ids == task_id
        task_episodes = sorted(
            int(value) for value in torch.unique(episode_ids[selected]).tolist()
        )
        if int(selected.sum()) != int(item.get("windows", -1)):
            raise ValueError(f"World split task {task_id} window count mismatch")
        if task_episodes != sorted(int(value) for value in item.get("episode_ids", [])):
            raise ValueError(f"World split task {task_id} episode list mismatch")
        task_transition = transition[selected]
        task_stats = (item.get("mask_stats") or {}).get("transition") or {}
        if (
            int(task_transition.sum()) != int(task_stats.get("true", -1))
            or task_transition.numel() != int(task_stats.get("total", -1))
        ):
            raise ValueError(f"World split task {task_id} transition stats mismatch")

    source = manifest.get("source") or {}
    source_path = Path(str(source.get("path", ""))).expanduser().resolve(strict=True)
    source_sha = str(source.get("sha256", ""))
    if not source_sha or _sha256_file(source_path) != source_sha:
        raise ValueError("World split source SHA mismatch")
    if int(metadata.get("source_n_windows", -1)) != int(source.get("n_windows", -2)):
        raise ValueError("World split source window count mismatch")

    return {
        "manifest_id": str(manifest.get("manifest_id")),
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": manifest_sha,
        "source_path": str(source_path),
        "source_sha256": source_sha,
    }


def validate_visual_world_resume_contract(
    checkpoint: dict,
    split_identity: dict[str, object],
    action_ranking: dict[str, object] | None = None,
    static_constraint_weight: float = 4.0,
    migration_id: str | None = None,
    va_world_mode: str = "legacy",
) -> None:
    """Reject exact continuation from an old or differently split loss graph."""

    contract = checkpoint.get("training_contract") or {}
    expected = {
        "world_supervision": WORLD_SUPERVISION_CONTRACT,
        "world_transition": WORLD_TRANSITION_CONTRACT,
        "world_loss_weights": WORLD_LOSS_COMPONENT_WEIGHTS,
        "world_stage_auxiliary_decay": WORLD_STAGE_AUXILIARY_DECAY,
        "world_no_regression": WORLD_NO_REGRESSION,
        "world_static_copy_constraint": {
            **WORLD_STATIC_COPY_CONSTRAINT,
            "weight": float(static_constraint_weight),
        },
        "world_action_ranking": (
            WORLD_ACTION_RANKING if action_ranking is None else action_ranking
        ),
        "world_action_donor_contract": WORLD_ACTION_DONOR_CONTRACT,
        "world_action_donor_sha256": split_identity[
            "world_action_donor_sha256"
        ],
        "world_action_donor_transitions": split_identity[
            "world_action_donor_transitions"
        ],
        "world_action_rank_transitions": split_identity[
            "world_action_rank_transitions"
        ],
        "world_logged_branch": WORLD_LOGGED_BRANCH_CONTRACT,
        "split_manifest_sha256": split_identity["manifest_sha256"],
        "split_source_sha256": split_identity["source_sha256"],
    }
    if va_world_mode == "peer_sync_h6":
        expected.update(
            {
                "va_world_mode": va_world_mode,
                "peer_world_topology": PEER_WORLD_TOPOLOGY_CONTRACT,
                "peer_world_action_source": PEER_WORLD_ACTION_SOURCE_CONTRACT,
                "peer_world_readout": PEER_WORLD_READOUT_CONTRACT,
            }
        )
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if migration_id == WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION:
        source_ranking = world_action_ranking_contract(
            "final" if action_ranking and action_ranking.get("stage") == "final_direct_matched_context" else "cycle"
        )
        mismatches = {
            key: values
            for key, values in mismatches.items()
            if not (
                key == "world_action_ranking"
                and values[0] == source_ranking
                and values[1] == action_ranking
            )
        }
    if mismatches:
        raise ValueError(
            "--resume-exact requires the same visual-motion World contract: "
            f"{mismatches}"
        )


def runtime_resource_stats(device: torch.device) -> dict[str, float | int]:
    """Cheap Linux process/GPU counters for long-run stability logs."""

    rss_mib = 0.0
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss_mib = resident_pages * os.sysconf("SC_PAGE_SIZE") / float(1 << 20)
    except (OSError, IndexError, ValueError):
        pass
    try:
        fd_count = sum(1 for _ in Path("/proc/self/fd").iterdir())
    except OSError:
        fd_count = -1
    allocated = reserved = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / float(1 << 20)
        reserved = torch.cuda.memory_reserved(device) / float(1 << 20)
    return {
        "rss_mib": rss_mib,
        "fd_count": fd_count,
        "cuda_allocated_mib": allocated,
        "cuda_reserved_mib": reserved,
    }


def _sampled_file_identity(path: Path) -> dict:
    """Cheap identity for referenced JPEG containers, honestly marked sampled.

    The windows payload itself gets a full SHA-256.  Its referenced longtraj
    containers can total tens of GiB, so we combine exact stat metadata with
    deterministic beginning/middle/end samples instead of pretending to have
    hashed every JPEG byte.
    """
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    block_bytes = 256 * 1024
    if stat.st_size <= 3 * block_bytes:
        offsets = [0]
    else:
        offsets = [0, max(0, (stat.st_size - block_bytes) // 2), stat.st_size - block_bytes]
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            block = stream.read(block_bytes)
            digest.update(int(offset).to_bytes(8, "little", signed=False))
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sampled_sha256": digest.hexdigest(),
        "sample_offsets": offsets,
        "sample_bytes": block_bytes,
    }


def _payload_schema(payload: dict) -> dict:
    """Record the explicit tensor/content schema protected by the file hash."""
    tensors = {}
    non_tensors = {}
    for key, value in sorted(payload.items()):
        if isinstance(value, Tensor):
            tensors[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
        elif key == "metadata" and isinstance(value, dict):
            # Metadata is small and changes data semantics (task names, strides,
            # horizon); store it explicitly as well as protecting it by SHA-256.
            non_tensors[key] = _normalize_contract_value(value)
        elif isinstance(value, (list, tuple)):
            non_tensors[key] = {
                "container": type(value).__name__,
                "length": len(value),
            }
        else:
            non_tensors[key] = {"type": type(value).__name__}
    return {"tensors": tensors, "non_tensors": non_tensors}


def build_dataset_content_identity(
    path: str | Path,
    payload: dict,
    *,
    longtraj_dir: str | Path | None = None,
) -> dict:
    """Strong identity for the MT-VJ windows payload, computed once per run.

    ``full_file_sha256`` covers actions, masks, frame references, language and
    metadata in the actual serialized payload—not only sampler task/episode IDs.
    Referenced longtraj JPEG containers use a clearly labelled sampled identity
    plus exact path/stat metadata to keep startup bounded.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    before_key = (before.st_size, before.st_mtime_ns)
    after_key = (after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise RuntimeError(f"dataset changed while fingerprinting: {resolved}")

    sources = []
    refs = payload.get("frame_refs")
    if isinstance(refs, (list, tuple)) and refs:
        root = (
            Path(longtraj_dir).expanduser().resolve(strict=False)
            if longtraj_dir is not None
            else resolved.parent
        )
        task_files = sorted({str(ref[0]) for ref in refs})
        for task_file in task_files:
            raw = Path(task_file)
            if raw.is_absolute():
                source = raw
            elif raw.suffix == ".pt" and (root / raw).exists():
                source = root / raw
            else:
                source = root / f"metaworld_longtraj_{task_file}.pt"
            if source.exists():
                sources.append(_sampled_file_identity(source))
            else:
                sources.append(
                    {
                        "path": str(source.resolve(strict=False)),
                        "missing": True,
                    }
                )

    return {
        "identity_algorithm": "full_payload_sha256+referenced_source_sample_v1",
        "resolved_path": str(resolved),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "full_file_sha256": digest,
        "payload_schema": _payload_schema(payload),
        "referenced_sources": sources,
    }


def _optimizer_contract(optimizer: torch.optim.Optimizer) -> dict:
    kind = "sam_adamw" if isinstance(optimizer, SAM) else "adamw"
    groups = []
    for group in optimizer.param_groups:
        parameters = list(group["params"])
        groups.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": list(group.get("betas", ())),
                "eps": float(group.get("eps", 0.0)),
                "amsgrad": bool(group.get("amsgrad", False)),
                "rho": float(group.get("rho", 0.0)),
                "parameter_count": len(parameters),
                "parameter_numel": int(sum(parameter.numel() for parameter in parameters)),
                "parameter_signature": [
                    {
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "requires_grad": bool(parameter.requires_grad),
                    }
                    for parameter in parameters
                ],
            }
        )
    return {"kind": kind, "groups": groups}


def build_exact_run_contract(
    args: argparse.Namespace,
    config,
    optimizer: torch.optim.Optimizer,
    sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None,
    metric_head: nn.Module | None = None,
    roi_head: nn.Module | None = None,
) -> dict:
    """Freeze every current MT-VJ CLI/data/objective semantic for exact resume."""
    argument_semantics = {
        key: _normalize_contract_value(value)
        for key, value in sorted(vars(args).items())
        if key not in _EXACT_RUN_OPERATIONAL_ARGS
        and key not in _WAM_CONTRACT_ARG_KEYS
    }
    metric_config = (
        _mtvj_metric_head_constructor_config(metric_head)
        if metric_head is not None
        else None
    )
    metric_identity = (
        getattr(metric_head, "_mtvj_external_checkpoint_identity", None)
        if metric_head is not None
        else None
    )
    if isinstance(metric_identity, dict):
        # Identity is content-based; path spelling is not a model semantic.
        metric_identity = {
            key: metric_identity.get(key)
            for key in ("sha256", "size_bytes", "contract")
        }
    roi_identity = None
    if roi_head is not None:
        roi_identity = getattr(roi_head, "_dino_roi_identity", None)
        if roi_identity is None:
            roi_identity = getattr(
                roi_head, "_mtvj_roi_checkpoint_identity", None
            )
    if isinstance(roi_identity, dict):
        roi_identity = {
            key: roi_identity.get(key)
            for key in ("sha256", "size_bytes", "contract")
        }
    # wam_joint is recorded in the conditional "wam" section below; stripping it
    # here keeps the WAM-off model_config byte-compatible with pre-WAM configs.
    model_config = dict(getattr(config, "__dict__", {}))
    model_config.pop("wam_joint", None)
    if not getattr(config, "wmrm", False):
        model_config.pop("wmrm", None)
        model_config.pop("wmrm_rank", None)
        model_config.pop("wmrm_world_dim", None)
        model_config.pop("wmrm_inject", None)
        model_config.pop("wmrm_mixer_dropout", None)
        model_config.pop("wmrm_target", None)
        model_config.pop("wmrm_detach_proposal_stage_state", None)
        model_config.pop("wmrm_predictor", None)
        model_config.pop("wmrm_predictor_depth", None)
        model_config.pop("wmrm_predictor_width", None)
        model_config.pop("wmrm_predictor_heads", None)
    contract = {
        "contract_version": EXACT_RUN_CONTRACT_VERSION,
        "data_identity": getattr(sampler, "dataset_content_identity", None),
        "arguments": argument_semantics,
        "model_config": model_config,
        "optimizer": _optimizer_contract(optimizer),
        "mtvj": {
            "metric_head_config": metric_config,
            "metric_checkpoint_identity": metric_identity,
            "metric_head_joint_trained": bool(
                getattr(args, "mtvj_train_metric_head", False)
            ),
            "relation_joint_trained": bool(
                getattr(args, "mtvj_train_relation", False)
            ),
            "metric_state_source": getattr(
                metric_head, "_mtvj_metric_state_source", None
            ),
            "metric_contract_version": getattr(
                metric_head, "_mtvj_metric_contract_version", None
            ),
            "roi_config": getattr(roi_head, "_mtvj_roi_config", None),
            "roi_checkpoint_identity": roi_identity,
        },
    }
    if getattr(args, "va_world_mode", "legacy") == "peer_sync_h6":
        contract["peer_world"] = {
            "topology": PEER_WORLD_TOPOLOGY_CONTRACT,
            "action_source": PEER_WORLD_ACTION_SOURCE_CONTRACT,
            "readout": PEER_WORLD_READOUT_CONTRACT,
        }
    if getattr(args, "feature_autocast_bf16", False):
        contract["feature_autocast"] = {
            "contract": FEATURE_AUTOCAST_CONTRACT,
            "dtype": "bfloat16",
            "training_cache_enabled": True,
            "no_grad_decode_cache_enabled": False,
        }
    if getattr(args, "wam_joint", False):
        # WAM-on contract: freeze alpha and the content identity of the base
        # WAM checkpoint (path spelling is not a model semantic).
        wam_ckpt_sha256 = "none"
        if getattr(args, "wam_ckpt", None):
            wam_ckpt_sha256 = _sha256_file(Path(args.wam_ckpt))
        contract["wam"] = {
            "wam_alpha": float(getattr(args, "wam_alpha", 1.0)),
            "wam_ckpt_sha256": wam_ckpt_sha256,
        }
    return _normalize_contract_value(contract)


def exact_run_contract_mismatches(saved: dict, current: dict) -> list[tuple[str, object, object]]:
    """Return deterministic leaf-level differences for a clear failure message."""
    mismatches: list[tuple[str, object, object]] = []
    missing = "<missing>"

    def compare(left, right, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else str(key)
                if key not in left:
                    mismatches.append((child, missing, right[key]))
                elif key not in right:
                    mismatches.append((child, left[key], missing))
                else:
                    compare(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append((f"{path}.length", len(left), len(right)))
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if type(left) is not type(right) or left != right:
            mismatches.append((path, left, right))

    compare(_normalize_contract_value(saved), _normalize_contract_value(current), "")
    return mismatches


def _normalize_legacy_exact_run_contract(contract: dict) -> dict:
    """Fill only defaults that are known for contracts predating these fields."""
    normalized = _normalize_contract_value(contract)
    arguments = normalized.get("arguments")
    if isinstance(arguments, dict):
        arguments.setdefault("wmrm_detach_proposal_stage_state", False)
        arguments.setdefault("wmrm_static_constraint_weight", 4.0)
        arguments.setdefault("wmrm_action_rank_per_sample_cap", None)
        arguments.setdefault("max_gradient_norm", None)
    model_config = normalized.get("model_config")
    if isinstance(model_config, dict):
        model_config.setdefault("wmrm_detach_proposal_stage_state", False)
        # Checkpoints predating the peer core are exactly the legacy topology.
        # Normalizing this absent field is compatibility, not a migration to peer.
        model_config.setdefault("va_world_mode", "legacy")
    if isinstance(arguments, dict):
        arguments.setdefault("va_world_mode", "legacy")
    return normalized


def validate_exact_run_contract(
    saved: dict | None,
    current: dict,
    *,
    migration_id: str | None = None,
) -> None:
    if saved is None:
        raise ValueError(
            "--resume-exact checkpoint is missing exact_run_contract; "
            "use --resume for legacy weights-only loading"
        )
    normalized_saved = _normalize_legacy_exact_run_contract(saved)
    normalized_current = _normalize_legacy_exact_run_contract(current)
    mismatches = exact_run_contract_mismatches(normalized_saved, normalized_current)
    if migration_id == WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION:
        saved_target, current_target = 1.0, 0.5
        allowed_paths = {"arguments.wmrm_world_weight"}
        saved_arguments = normalized_saved.get("arguments")
        current_arguments = normalized_current.get("arguments")
        saved_weight = (
            saved_arguments.get("wmrm_world_weight")
            if isinstance(saved_arguments, dict)
            else None
        )
        current_weight = (
            current_arguments.get("wmrm_world_weight")
            if isinstance(current_arguments, dict)
            else None
        )
        coherent = (
            isinstance(saved_arguments, dict)
            and isinstance(current_arguments, dict)
            and type(saved_weight) is float
            and type(current_weight) is float
            and saved_weight == saved_target
            and current_weight == current_target
            and {path for path, _, _ in mismatches} == allowed_paths
            and all(
                type(left) is float
                and type(right) is float
                and left == saved_target
                and right == current_target
                for path, left, right in mismatches
            )
        )
        if coherent:
            return
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        ) or f"no coherent old-{saved_target} to new-{current_target} transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if migration_id == WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION:
        allowed_paths = {"arguments.wmrm_action_rank_per_sample_cap"}
        saved_arguments = normalized_saved.get("arguments")
        current_arguments = normalized_current.get("arguments")
        saved_cap = (
            saved_arguments.get("wmrm_action_rank_per_sample_cap")
            if isinstance(saved_arguments, dict)
            else "<missing>"
        )
        current_cap = (
            current_arguments.get("wmrm_action_rank_per_sample_cap")
            if isinstance(current_arguments, dict)
            else "<missing>"
        )
        coherent = (
            isinstance(saved_arguments, dict)
            and isinstance(current_arguments, dict)
            and saved_cap is None
            and type(current_cap) is float
            and current_cap == 0.2
            and type(saved_arguments.get("wmrm_static_constraint_weight")) is float
            and saved_arguments.get("wmrm_static_constraint_weight") == 2.0
            and current_arguments.get("wmrm_static_constraint_weight") == 2.0
            and type(saved_arguments.get("wmrm_world_weight")) is float
            and saved_arguments.get("wmrm_world_weight") == 1.0
            and current_arguments.get("wmrm_world_weight") == 1.0
            and saved_arguments.get("wmrm_detach_proposal_stage_state") is True
            and current_arguments.get("wmrm_detach_proposal_stage_state") is True
            and {path for path, _, _ in mismatches} == allowed_paths
            and all(
                left is None and type(right) is float and right == 0.2
                for path, left, right in mismatches
            )
        )
        if coherent:
            return
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        ) or "no coherent static2/world1/detached action-rank cap None to 0.2 transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if migration_id == WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION:
        allowed_paths = {"arguments.wmrm_static_constraint_weight"}
        saved_arguments = normalized_saved.get("arguments")
        current_arguments = normalized_current.get("arguments")
        saved_weight = (
            saved_arguments.get("wmrm_static_constraint_weight")
            if isinstance(saved_arguments, dict)
            else None
        )
        current_weight = (
            current_arguments.get("wmrm_static_constraint_weight")
            if isinstance(current_arguments, dict)
            else None
        )
        coherent = (
            isinstance(saved_arguments, dict)
            and isinstance(current_arguments, dict)
            and type(saved_weight) is float
            and type(current_weight) is float
            and saved_weight == 4.0
            and current_weight == 2.0
            and saved_arguments.get("wmrm_world_weight") == 1.0
            and current_arguments.get("wmrm_world_weight") == 1.0
            and saved_arguments.get("wmrm_detach_proposal_stage_state") is True
            and current_arguments.get("wmrm_detach_proposal_stage_state") is True
            and {path for path, _, _ in mismatches} == allowed_paths
            and all(
                type(left) is float
                and type(right) is float
                and left == 4.0
                and right == 2.0
                for path, left, right in mismatches
            )
        )
        if coherent:
            return
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        ) or "no coherent static-constraint 4.0 to 2.0 transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if migration_id is not None:
        if migration_id != WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION:
            raise ValueError(
                "unsupported --resume-exact-contract-migration: "
                f"{migration_id!r}"
            )
        allowed_paths = {
            "arguments.wmrm_detach_proposal_stage_state",
            "model_config.wmrm_detach_proposal_stage_state",
        }
        # The migration changes one semantic flag, but the exact contract stores
        # it in two representations.  Normalize legacy omissions to False, then
        # require both sides to be coherent before allowing the transition.  In
        # particular, never accept a partial/contradictory transition where only
        # one representation changes (or where either contract is internally
        # inconsistent).
        saved_arguments = normalized_saved.get("arguments")
        saved_model_config = normalized_saved.get("model_config")
        current_arguments = normalized_current.get("arguments")
        current_model_config = normalized_current.get("model_config")
        detach_paths = (
            (saved_arguments, "arguments"),
            (saved_model_config, "model_config"),
            (current_arguments, "arguments"),
            (current_model_config, "model_config"),
        )
        if not all(isinstance(value, dict) for value, _ in detach_paths):
            details = "arguments and model_config must both be mappings"
        else:
            saved_values = (
                saved_arguments["wmrm_detach_proposal_stage_state"],
                saved_model_config["wmrm_detach_proposal_stage_state"],
            )
            current_values = (
                current_arguments["wmrm_detach_proposal_stage_state"],
                current_model_config["wmrm_detach_proposal_stage_state"],
            )
            coherent = (
                saved_values == (False, False)
                and current_values == (True, True)
                and {path for path, _, _ in mismatches} == allowed_paths
                and all(left is False and right is True for path, left, right in mismatches)
            )
            if coherent:
                return
            details = "; ".join(
                f"{path}: checkpoint={left!r}, runtime={right!r}"
                for path, left, right in mismatches[:12]
            ) or "no coherent old-false-both to new-true-both transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if mismatches:
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        )
        if len(mismatches) > 12:
            details += f"; ... and {len(mismatches) - 12} more"
        raise ValueError(f"--resume-exact exact_run_contract mismatch: {details}")


def capture_rng_state(*, include_cuda: bool = True) -> dict:
    """Capture global RNGs using only weights-only-safe primitives/tensors."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            # torch 2.2 cannot construct tensors from NumPy uint32 arrays.
            # MT19937 words fit losslessly in int64 and are cast back on restore.
            "state": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if include_cuda and torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNGs captured by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"exact resume RNG state missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("exact resume checkpoint contains CUDA RNG but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "exact resume CUDA device count mismatch: "
                f"checkpoint={len(cuda_states)} runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def exact_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    """Serialize AdamW, including SAM's real base optimizer state."""
    if isinstance(optimizer, SAM):
        if not isinstance(optimizer.base_optimizer, torch.optim.AdamW):
            raise TypeError("--resume-exact only supports SAM with an AdamW base optimizer")
        if any("pre_perturbation" in value for value in optimizer.state.values()):
            raise RuntimeError("cannot checkpoint exact state between SAM first_step/second_step")
        return {
            "kind": "sam_adamw",
            "state_dict": optimizer.base_optimizer.state_dict(),
            "rho": [float(group["rho"]) for group in optimizer.param_groups],
        }
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("--resume-exact currently supports AdamW (or SAM over AdamW) only")
    return {"kind": "adamw", "state_dict": optimizer.state_dict()}


def restore_exact_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved: dict,
) -> None:
    """Strictly restore an optimizer produced by :func:`exact_optimizer_state_dict`."""
    expected_kind = "sam_adamw" if isinstance(optimizer, SAM) else "adamw"
    if saved.get("kind") != expected_kind:
        raise ValueError(
            "exact resume optimizer mismatch: "
            f"checkpoint={saved.get('kind')!r} runtime={expected_kind!r}"
        )
    if isinstance(optimizer, SAM):
        saved_rho = [float(value) for value in saved.get("rho", [])]
        runtime_rho = [float(group["rho"]) for group in optimizer.param_groups]
        if saved_rho != runtime_rho:
            raise ValueError(
                f"exact resume SAM rho mismatch: {saved_rho!r} != {runtime_rho!r}"
            )
        optimizer.base_optimizer.load_state_dict(saved["state_dict"])
        optimizer.param_groups = optimizer.base_optimizer.param_groups
    else:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("--resume-exact currently supports AdamW only")
        optimizer.load_state_dict(saved["state_dict"])
    validate_optimizer_update_state(optimizer)


def build_exact_resume_state(
    optimizer: torch.optim.Optimizer,
    global_step: int,
    sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None,
    exact_run_contract: dict,
) -> dict:
    """Build the training-state portion of an exact-resumable checkpoint."""
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    uses_cuda = any(
        parameter.is_cuda
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    return {
        "exact_resume_version": EXACT_RESUME_VERSION,
        "global_step": int(global_step),
        "optimizer_state": exact_optimizer_state_dict(optimizer),
        "sampler_state": sampler.state_dict() if sampler is not None else None,
        "rng_state": capture_rng_state(include_cuda=uses_cuda),
        "exact_run_contract": _normalize_contract_value(exact_run_contract),
    }


def restore_exact_resume_state(
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None,
    *,
    runtime_exact_run_contract: dict | None = None,
    migration_id: str | None = None,
    restore_rng: bool = True,
) -> int:
    """Restore optimizer/sampler and optionally RNG; return completed updates."""
    required = {
        "exact_resume_version",
        "global_step",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "exact_run_contract",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "--resume-exact requires a new exact checkpoint; missing keys: "
            f"{sorted(missing)}. Use --resume for legacy weights-only loading."
        )
    if checkpoint["exact_resume_version"] != EXACT_RESUME_VERSION:
        raise ValueError(
            "unsupported exact resume version: "
            f"{checkpoint['exact_resume_version']} != {EXACT_RESUME_VERSION}"
        )
    if runtime_exact_run_contract is not None:
        # Contract comparison intentionally precedes every mutable restore below.
        validate_exact_run_contract(
            checkpoint["exact_run_contract"],
            runtime_exact_run_contract,
            migration_id=migration_id,
        )
    saved_sampler = checkpoint["sampler_state"]
    if saved_sampler is None:
        if isinstance(sampler, TaskLocalityWeightedSampler):
            raise ValueError(
                "--resume-exact checkpoint has sampler_state=None but the "
                "runtime built a sampler; refuse to invent locality state"
            )
        # TaskWeightedSampler or no sampler: 6k / this 6k→20k run stored None.
        # Keep epoch=0 rather than inferring a cursor from global_step.
    elif sampler is None:
        raise ValueError("--resume-exact requires a checkpointed sampler state")
    restore_exact_optimizer_state(optimizer, checkpoint["optimizer_state"])
    if saved_sampler is not None:
        sampler.load_state_dict(saved_sampler)
    global_step = int(checkpoint["global_step"])
    if global_step < 0:
        raise ValueError(f"invalid exact checkpoint global_step: {global_step}")
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return global_step


def synthetic_sequence(
    config: VACompoundConfig,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    *,
    with_frames: bool = False,
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
    batch = {
        "vision_tokens": vision,
        "language_hidden": language,
        "language_mask": torch.ones(batch_size, 8, dtype=torch.bool, device=device),
        "proprio": proprio,
        "previous_action": previous_action,
        "actions": actions,
        "pair_id": pair_id,
        "instruction_id": instruction_id,
    }
    if with_frames:
        # MT-VJ 冒烟：合成随机帧（与 LiveVJEPADataset 同款 [B, T, W, 384, 384, 3]
        # uint8 契约；W=4 帧窗，384×384 为 live 解码器产物尺寸）。
        batch["frames"] = torch.randint(
            0,
            256,
            (batch_size, sequence_length, 4, 384, 384, 3),
            dtype=torch.uint8,
            device=device,
        )
    return batch


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def feature_policy_autocast(device: torch.device, enabled: bool):
    """BF16 feature forward with the normal training weight cache."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
        cache_enabled=True,
    )


def feature_no_grad_decode_autocast(device: torch.device, enabled: bool):
    """Keep no-grad proposal casts out of the enclosing training cache."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
        cache_enabled=False,
    )


def _enable_optional_action_masks(dataset: Dataset) -> None:
    """Expose optional action masks through all dataset adapters.

    ``LongTrajFramesDataset`` and ``LiveVJEPADataset`` build their item dicts from
    a ``REQUIRED`` tuple.  Keep those classes backward compatible and extend the
    tuple only for payloads that actually contain a mask.
    """
    payload = getattr(dataset, "payload", None)
    if not isinstance(payload, dict):
        return
    present = tuple(key for key in ACTION_MASK_KEYS if key in payload)
    if not present:
        return
    length = len(dataset)
    for key in present:
        value = payload[key]
        if not isinstance(value, Tensor) or value.ndim == 0 or value.shape[0] != length:
            raise ValueError(
                f"{key} must be a tensor with first dimension equal to dataset length {length}"
            )
    source = getattr(dataset, "_inner", dataset)
    required = getattr(source, "REQUIRED", None)
    if required is not None:
        source.REQUIRED = tuple(dict.fromkeys((*required, *present)))
    print(f"action masks enabled: {', '.join(present)}", flush=True)


def _expand_action_mask(mask: Tensor, reference: Tensor, name: str) -> Tensor:
    """Broadcast a common action/horizon mask shape to ``[B,T,H,A]``."""
    if reference.ndim != 4:
        raise ValueError("flow tensors must have shape [batch, sequence, horizon, action_dim]")
    batch, sequence, horizon, action_dim = reference.shape
    shape = tuple(mask.shape)
    if shape == tuple(reference.shape):
        expanded = mask
    elif shape == (batch, sequence, horizon):
        expanded = mask.unsqueeze(-1)
    elif shape == (batch, sequence):
        expanded = mask[:, :, None, None]
    elif shape == (batch, horizon):
        expanded = mask[:, None, :, None]
    elif shape == (sequence, horizon):
        expanded = mask[None, :, :, None]
    elif shape == (horizon,):
        expanded = mask[None, None, :, None]
    else:
        raise ValueError(
            f"{name} shape {shape} cannot broadcast to flow tensor "
            f"{(batch, sequence, horizon, action_dim)}"
        )
    expanded = expanded.to(device=reference.device, dtype=reference.dtype).expand_as(reference)
    if not bool(torch.isfinite(expanded).all()) or bool((expanded < 0).any()):
        raise ValueError(f"{name} must contain finite non-negative weights")
    return expanded


def masked_flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    batch: dict | None = None,
    *,
    prefix_steps: int = 6,
    prefix_weight: float = 1.0,
    tail_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flow MSE with optional validity masks and prefix/tail weighting.

    Returns ``(training_loss, raw_prefix_mse, raw_tail_mse)``.  The diagnostic
    terms use validity masks but intentionally exclude prefix/tail scalar
    weights, so changing ``tail_weight`` does not hide tail quality.
    """
    if predicted_velocity.shape != target_velocity.shape or predicted_velocity.ndim != 4:
        raise ValueError(
            "predicted_velocity and target_velocity must share shape "
            "[batch, sequence, horizon, action_dim]"
        )
    if prefix_steps <= 0:
        raise ValueError("prefix_steps must be positive")
    if prefix_weight < 0.0 or tail_weight < 0.0:
        raise ValueError("prefix/tail flow weights must be non-negative")

    squared_error = (predicted_velocity - target_velocity).square()
    validity = torch.ones_like(squared_error)
    if batch is not None:
        for key in ACTION_MASK_KEYS:
            mask = batch.get(key)
            if mask is not None:
                validity = validity * _expand_action_mask(mask, squared_error, key)

    split = min(prefix_steps, squared_error.shape[-2])

    def region_mean(error: Tensor, weight: Tensor) -> Tensor:
        denominator = weight.sum()
        if not bool(denominator > 0):
            return error.new_zeros(())
        return (error * weight).sum() / denominator

    prefix_loss = region_mean(
        squared_error[..., :split, :], validity[..., :split, :]
    )
    tail_loss = region_mean(
        squared_error[..., split:, :], validity[..., split:, :]
    )
    horizon_weights = squared_error.new_full((squared_error.shape[-2],), tail_weight)
    horizon_weights[:split] = prefix_weight
    element_weights = validity * horizon_weights.view(1, 1, -1, 1)
    denominator = element_weights.sum()
    if not bool(denominator > 0):
        raise ValueError("flow loss has zero valid weighted action elements")

    # Preserve the exact legacy reduction when no masking/reweighting is active.
    has_mask = batch is not None and any(batch.get(key) is not None for key in ACTION_MASK_KEYS)
    if not has_mask and prefix_weight == 1.0 and tail_weight == 1.0:
        loss = F.mse_loss(predicted_velocity, target_velocity)
    else:
        loss = (squared_error * element_weights).sum() / denominator
    return loss, prefix_loss, tail_loss


def effective_action_valid_fraction(batch: dict | None, reference: Tensor) -> Tensor:
    """Return the supervised fraction after combining all optional masks.

    Reporting this next to loss prevents a lower number caused by masking bad
    H48 targets from being mistaken for genuine policy improvement.
    """
    validity = torch.ones_like(reference)
    if batch is not None:
        for key in ACTION_MASK_KEYS:
            mask = batch.get(key)
            if mask is not None:
                validity = validity * _expand_action_mask(mask, reference, key)
    return validity.mean()


def ensure_sequence(
    batch: dict[str, Tensor],
    min_sequence_length: int,
) -> dict[str, Tensor]:
    # MT-VJ（dense_readout_mtvj）：batch 无 vision_tokens（在线 dense 用 frames），
    # 序列长度以 actions 为准（2026-08-10）。
    if "vision_tokens" in batch:
        if batch["vision_tokens"].ndim != 4 or batch["actions"].ndim != 4:
            raise ValueError("vision/actions must be paired short sequences")
        sequence_length = batch["vision_tokens"].shape[1]
    elif "frames" in batch:
        if batch["frames"].ndim != 6 or batch["actions"].ndim != 4:
            raise ValueError("frames/actions must be paired short sequences")
        sequence_length = batch["frames"].shape[1]
    else:
        raise ValueError("batch 缺 vision_tokens/frames（无视觉输入的序列校验）")
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


def sample_flow_matching_inputs_paired(
    actions: Tensor,
    is_perturbed: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """clean/perturbed 配对行共享同一 (τ, ε)（设计 §六.2：两条监督间的动作差异
    不被随机 flow noise 淹没）。

    配对布局契约（mix_perturb_batch）：perturbed 行 k 的配对 clean 行 = 前一行
    （[c0,p0,c1,p1,...]），故 ``is_perturbed`` 中第 k 个 True 行直接复制前一行的
    (τ, ε)。其余行为独立采样（与 ``sample_flow_matching_inputs`` 语义一致）。
    """
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, sequence, horizon, action_dim]")
    if is_perturbed.ndim != 1 or is_perturbed.shape[0] != actions.shape[0]:
        raise ValueError("is_perturbed must be [batch] bool 且与 actions 同批")
    noise = torch.randn_like(actions)
    flow_time = torch.rand(
        actions.shape[:2],
        device=actions.device,
        dtype=actions.dtype,
    )
    p_idx = torch.nonzero(is_perturbed, as_tuple=False).flatten()
    if p_idx.numel():
        c_idx = p_idx - 1
        if bool((c_idx < 0).any()) or bool(is_perturbed[c_idx].any()):
            raise ValueError(
                "配对布局契约破坏：perturbed 行的前一行必须是 clean 行"
            )
        # 共享 (τ, ε)：复制配对 clean 行（randn 后原地覆盖，无梯度历史）。
        noise[p_idx] = noise[c_idx]
        flow_time[p_idx] = flow_time[c_idx]
    tau = flow_time[:, :, None, None]
    noisy_actions = (1.0 - tau) * noise + tau * actions
    target_velocity = actions - noise
    return noisy_actions, flow_time, target_velocity


def mix_perturb_batch(
    clean: dict[str, Tensor],
    perturbed: dict[str, Tensor],
    p_vision: Tensor,
    m: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """clean [n_clean] 与 perturbed [m] 交错配对 → [B=n_clean+m]（--perturb-data
    混合加载；clean:perturbed 比例由 ``m = round(B·ratio)`` 决定）。

    布局 ``[c0,p0,c1,p1,..., tail]``：paired 段 2m 行（clean 前 m 行与全部
    perturbed 行交错），tail = clean[m:]（不丢行）；``p_vision`` [m, T, N, D]
    为 perturbed 行的视觉 token（与训练路径同构，dtype 以 p_vision 为准）。
    返回 (mixed, is_perturbed [B] bool)——is_perturbed 供
    ``sample_flow_matching_inputs_paired`` 共享 (τ, ε)。
    """
    n_clean = int(clean["actions"].shape[0])
    if not (1 <= m <= n_clean):
        raise ValueError(f"m={m} 需满足 1 ≤ m ≤ clean 行数 {n_clean}")
    if int(perturbed["actions"].shape[0]) != m:
        raise ValueError(
            f"perturbed 行数 {perturbed['actions'].shape[0]} != m={m}"
        )
    mixed: dict[str, Tensor] = {}
    for key, value in clean.items():
        if not isinstance(value, Tensor):
            mixed[key] = value
            continue
        if key in ("vision_tokens", "vision_tokens_st"):
            continue  # 视觉键单独拼接（paired 段用 p_vision）
        if key in perturbed:
            head = torch.stack([value[:m], perturbed[key]], dim=1).flatten(0, 1)
            mixed[key] = torch.cat([head, value[m:]], dim=0)
        elif key == "coords":
            # 全局坐标常量（行无关）：扩展行数保持一致。
            mixed[key] = value[0:1].expand(n_clean + m, -1, -1).contiguous()
        else:
            raise ValueError(
                f"混批契约破坏：perturbed 批缺少键 {key!r}"
            )
    for key in ("vision_tokens", "vision_tokens_st"):
        if key in clean:
            head = torch.stack(
                [clean[key][:m].to(dtype=p_vision.dtype), p_vision], dim=1
            ).flatten(0, 1)
            mixed[key] = torch.cat(
                [head, clean[key][m:].to(dtype=p_vision.dtype)], dim=0
            )
    is_perturbed = torch.zeros(n_clean + m, dtype=torch.bool)
    is_perturbed[1 : 2 * m : 2] = True
    return mixed, is_perturbed


def _encode_perturb_frames(
    frames: Tensor,
    backbone,
    device: torch.device,
    *,
    dense: bool,
) -> Tensor:
    """perturb 存储帧 [m, T, W, 96, 96, 3] uint8 → [m, T, N, D] fp32。

    冻结 V-JEPA 在线编码（``preprocess_batch`` 内部 bicubic 放大到 384）。
    注意：扰动数据当前为 96×96 低分辨率小样本 + 原始预训练骨干，与 clean 侧
    微调特征存在域差——精确配对路径须数据侧重提取（--local-slots 变体）。
    """
    if frames.ndim != 6 or frames.shape[-1] != 3:
        raise ValueError(
            f"perturb frames 必须为 [m, T, W, H, W, 3] uint8，got {tuple(frames.shape)}"
        )
    from va_compound.live_vjepa import encode_live_frames

    with torch.no_grad():
        encoded = encode_live_frames(
            frames.numpy(), backbone, device, dense=dense
        )
    return encoded.float()


def _maybe_build_perturb_backbone(
    args: argparse.Namespace,
    device: torch.device,
    vision_backbone,
    use_payload_vision: bool,
) -> None:
    """--perturb-data 帧在线编码骨干：live 路径复用主骨干（no_grad）；feature
    路径构建冻结 V-JEPA（~238 MiB，设计 §九 显存纪律：只读、冻结）。"""
    if (
        args.perturb_data is None
        or args.live_vjepa
        or use_payload_vision
        or args.e2e_data
    ):
        return None
    from va_compound.backbones import VJEPA21Backbone

    backbone = VJEPA21Backbone.from_pretrained(
        device=device,
        dtype="float32",
        max_tokens=144,
        local_files_only=True,
    )
    backbone.freeze_all()
    print(
        "perturb-data: perturbed 帧在线编码用冻结 V-JEPA（~238 MiB，只读）；"
        "与 clean 特征可能存在骨干域差（数据侧重提取为精确路径）",
        flush=True,
    )
    return backbone


def _mtvj_config_kwargs(args: argparse.Namespace) -> dict:
    """MT-VJ（契约 §5/§6）：--dense-readout-mtvj 时打开 model 的 dense 层。

    flag 未给时返回空 dict——config 构造签名与旧版逐字一致（不要求 model.py
    的 ``dense_readout_mtvj`` 字段存在，保证既有路径行为不变）。
    """
    if not getattr(args, "dense_readout_mtvj", False):
        return {}
    kwargs = {"dense_readout_mtvj": True}
    action_vision = getattr(args, "action_vision_backbone", "none")
    if action_vision != "none":
        spec = ACTION_VISION_SPECS[action_vision]
        kwargs.update(
            action_vision_backbone=action_vision,
            action_vision_model_id=spec["model_id"],
            action_vision_dim=spec["feature_dim"],
            action_vision_image_size=spec["image_size"],
            action_vision_layers=spec["output_layers"],
        )
    return kwargs


def _main_vision_config_kwargs(args: argparse.Namespace) -> dict:
    """DINO-main replacement config（V-JEPA 路径 flag 关闭即禁用，不删除）。"""
    if not getattr(args, "dino_main_vision", False):
        return {}
    spec = ACTION_VISION_SPECS["dinov2_vitl14_reg4"]
    grid = int(args.main_vision_grid)
    frames = int(args.main_vision_frames)
    kwargs = {
        "main_vision_backbone": "dinov2_vitl14_reg4",
        "main_vision_model_id": spec["model_id"],
        "main_vision_image_size": spec["image_size"],
        "main_vision_dim": spec["feature_dim"],
        "main_vision_grid": grid,
        "main_vision_frames": frames,
        "main_vision_tokens": grid * grid * frames,
        "main_vision_temporal": bool(
            getattr(args, "main_vision_temporal", False)
        ),
        "main_vision_temporal_scale": float(
            getattr(args, "main_vision_temporal_scale", 1.0)
        ),
    }
    if getattr(args, "dino_dense_metric", False):
        # DINO-metric：复用逐层 dense K/V 机制（DenseEvidenceProjector 以
        # main_vision_dim=1024 构造），metric head/relation encoder 从零构建。
        kwargs["dense_readout_mtvj"] = True
        kwargs["dino_dense_metric"] = True
        kwargs["metric_geometry_inject"] = bool(
            getattr(args, "metric_geometry_inject", False)
        )
        kwargs["metric_geometry_dim"] = 8
    return kwargs


def _maybe_build_mtvj_backbone(device: torch.device):
    """MT-VJ（契约 §6）：冻结 fp16 V-JEPA（本地缓存），与 live 可训练骨干独立。

    dense evidence 只读（forward_hierarchical_dense {5,11}，绝不反向）；fp16
    参数量减半（~119 MiB）。``prepare_pnpw_features.VJEPA21Backbone`` 与
    ``va_compound.backbones.VJEPA21Backbone`` 是同一类（prepare_pnpw_features
    顶部转导入）。
    """
    from prepare_pnpw_features import VJEPA21Backbone

    backbone = VJEPA21Backbone.from_pretrained(
        device=device,
        dtype="float16",
        max_tokens=144,
        local_files_only=True,
    )
    backbone.freeze_all()
    print(
        "mtvj: 冻结 V-JEPA（fp16）dense evidence 骨干就绪，"
        f"params={sum(p.numel() for p in backbone.parameters()):,}",
        flush=True,
    )
    return backbone


def _maybe_build_action_vision_backbone(
    args: argparse.Namespace,
    config: VACompoundConfig,
    device: torch.device,
):
    """Load the frozen action-only tower from an explicit local checkpoint."""
    if config.action_vision_backbone == "none":
        return None
    checkpoint = args.action_vision_checkpoint
    if checkpoint is None:
        raise ValueError(
            f"--action-vision-backbone {config.action_vision_backbone} requires "
            "--action-vision-checkpoint"
        )
    # Preserve the ``.safetensors`` symlink suffix: resolving the HF cache
    # symlink to its extensionless blob makes timm incorrectly call torch.load.
    checkpoint = checkpoint.expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"action vision checkpoint is missing: {checkpoint}")
    from va_compound.backbones import TimmActionVisionBackbone

    backbone = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype="float16",
        model_id=config.action_vision_model_id,
        image_size=config.action_vision_image_size,
        feature_dim=config.action_vision_dim,
        output_layers=tuple(config.action_vision_layers),
        checkpoint_path=checkpoint,
        local_files_only=True,
    )
    backbone.freeze_all()
    args.action_vision_checkpoint_sha256 = _sha256_file(checkpoint)
    print(
        f"action-vision: frozen {config.action_vision_backbone} "
        f"({config.action_vision_model_id}, {config.action_vision_image_size}px, "
        f"dim={config.action_vision_dim}, "
        f"params={sum(p.numel() for p in backbone.parameters()):,})",
        flush=True,
    )
    return backbone


def _action_vision_online_encode(
    frames,
    backbone,
    device: torch.device,
    *,
    encode_batch: int,
) -> dict[int, Tensor]:
    """Encode the two newest frames into two temporal patch grids per decision.

    The raw window is ``[d-6,d-4,d-2,d]``; the action tower uses ``[d-4,d]``
    (the newest frame from each of V-JEPA's two temporal bins).
    V-JEPA continues to consume the complete window for the base/metric/WAM
    paths.  Chunking bounds the frozen ViT-L workspace on a 16-GiB GPU.
    """
    if encode_batch < 1:
        raise ValueError("action vision encode_batch must be positive")
    if frames.ndim != 6 or frames.shape[-1] != 3:
        raise ValueError(
            "action vision frames must be [B,T,W,H,W,3], got "
            f"{tuple(frames.shape)}"
        )
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    batch_size, sequence_length, window, height, width, _ = frames_np.shape
    if window != 4:
        raise ValueError(f"action vision requires the four-frame window, got {window}")
    selected = np.ascontiguousarray(frames_np[:, :, (1, 3), :, :, :].reshape(
        batch_size * sequence_length * 2, height, width, 3
    ))
    images = torch.from_numpy(selected).permute(0, 3, 1, 2).float().div_(255.0)
    outputs: dict[int, list[Tensor]] = {5: [], 11: []}
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    for start in range(0, images.shape[0], encode_batch):
        chunk = images[start : start + encode_batch].to(device)
        if tuple(chunk.shape[-2:]) != (backbone.image_size, backbone.image_size):
            chunk = F.interpolate(
                chunk,
                size=(backbone.image_size, backbone.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        chunk = (chunk - mean) / std
        hierarchical = backbone.forward_hierarchical_dense(chunk)
        for layer in (5, 11):
            outputs[layer].append(hierarchical[layer])
        del chunk, hierarchical
    dense = {}
    for layer, chunks in outputs.items():
        tokens = torch.cat(chunks, dim=0)
        dense[layer] = tokens.reshape(
            batch_size, sequence_length, -1, tokens.shape[-1]
        )
    return dense


def _build_dino_main_backbone(
    args: argparse.Namespace,
    config: VACompoundConfig,
    device: torch.device,
):
    """Frozen DINOv2 tower as the REPLACEMENT main vision backbone.

    V-JEPA stays available in the repository for the legacy path; this tower
    is only built under ``--dino-main-vision``.
    """
    checkpoint = args.main_vision_checkpoint.expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"main vision checkpoint is missing: {checkpoint}")
    from va_compound.backbones import TimmActionVisionBackbone

    backbone = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype="float16",
        model_id=config.main_vision_model_id,
        image_size=config.main_vision_image_size,
        feature_dim=config.main_vision_dim,
        output_layers=(11, 23),  # reuse the canonical mid/final-key contract
        checkpoint_path=checkpoint,
        local_files_only=True,
    )
    backbone.freeze_all()
    args.main_vision_checkpoint_sha256 = _sha256_file(checkpoint)
    print(
        f"dino-main: frozen {config.main_vision_backbone} REPLACES V-JEPA as the "
        f"VA main vision ({config.main_vision_image_size}px, "
        f"dim={config.main_vision_dim}, {config.main_vision_tokens} tokens/decision, "
        f"params={sum(p.numel() for p in backbone.parameters()):,})",
        flush=True,
    )
    return backbone


def _dino_main_online_encode(
    frames,
    backbone,
    device: torch.device,
    *,
    encode_batch: int,
    grid: int,
    window: int,
    return_dense: bool = False,
) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
    """DINO-main vision tokens: [B, T, window*grid*grid, dim] fp32 per decision.

    Every decision consumes the complete ``window``-frame history window
    ``[d-6,d-4,d-2,d]``; each frame is encoded by the frozen tower and its
    16x16 patch grid is average-pooled to ``grid x grid``. Timm NLC patch
    order is row-major; the pooling grid order is verified by
    tests/test_dino_main_vision.py.

    ``return_dense=True``（DINO-metric，2026-08-15）：额外返回 dense
    evidence ``{5: [B, T, 512, D], 11: [B, T, 512, D]}``——canonical key 5 =
    block11（g，帧 [d-2,d] 各 256 patch），key 11 = block23（d），两帧沿
    token 维拼接（前 256 = d-2，后 256 = d，t→y→x 序），供
    DenseEvidenceProjector/语言度量场消费（与 V-JEPA {5,11} 语义对齐）。
    """
    if encode_batch < 1:
        raise ValueError("main vision encode_batch must be positive")
    if not (1 <= grid <= 16) or window < 1:
        raise ValueError("dino-main grid/window out of range")
    if frames.ndim != 6 or frames.shape[-1] != 3:
        raise ValueError(
            "dino-main frames must be [B,T,W,H,W,3], got " f"{tuple(frames.shape)}"
        )
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    batch_size, sequence_length, win, height, width, _ = frames_np.shape
    if win != window:
        raise ValueError(f"dino-main requires the {window}-frame window, got {win}")
    selected = np.ascontiguousarray(
        frames_np.reshape(batch_size * sequence_length * window, height, width, 3)
    )
    images = torch.from_numpy(selected).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    chunks: list[Tensor] = []
    dense5: list[Tensor] = []
    dense11: list[Tensor] = []
    for start in range(0, images.shape[0], encode_batch):
        chunk = images[start : start + encode_batch].to(device)
        if tuple(chunk.shape[-2:]) != (backbone.image_size, backbone.image_size):
            chunk = F.interpolate(
                chunk,
                size=(backbone.image_size, backbone.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        chunk = (chunk - mean) / std
        hierarchical = backbone.forward_hierarchical_dense(chunk)
        tokens = hierarchical[11]
        if tokens.shape[-2] != 256 or tokens.shape[-1] != backbone.feature_dim:
            raise RuntimeError(
                "dino-main expects 256 patch tokens per frame, got "
                f"{tuple(tokens.shape)}"
            )
        chunks.append(tokens.float())
        if return_dense:
            # 只保留帧 [d-2, d]（窗口内 w ∈ {2, 3}）的两层 patch 证据。
            flat_indices = [
                start + j
                for j in range(tokens.shape[0])
                if (start + j) % window in (2, 3)
            ]
            if flat_indices:
                local = [idx - start for idx in flat_indices]
                for source, target in (
                    (hierarchical[5], dense5),
                    (hierarchical[11], dense11),
                ):
                    picked = source[local]
                    if picked.shape[-2] != 256 or picked.shape[-1] != backbone.feature_dim:
                        raise RuntimeError(
                            "dino-metric expects 256 patch tokens per frame at "
                            f"block {5 if source is hierarchical[5] else 11}, got "
                            f"{tuple(picked.shape)}"
                        )
                    target.append(picked.float())
        del chunk, hierarchical, tokens
    tokens = torch.cat(chunks, dim=0)  # [B*T*W, 256, D]
    dim = tokens.shape[-1]
    tokens = tokens.reshape(
        batch_size * sequence_length * window, 16, 16, dim
    ).permute(0, 3, 1, 2)
    tokens = F.adaptive_avg_pool2d(tokens, (grid, grid))
    tokens = tokens.permute(0, 2, 3, 1).reshape(
        batch_size, sequence_length, window * grid * grid, dim
    )
    if not return_dense:
        return tokens
    dense_evidence = {
        layer: torch.cat(parts, dim=0).reshape(
            batch_size, sequence_length, -1, parts[0].shape[-1]
        )
        for layer, parts in ((5, dense5), (11, dense11))
    }
    for layer, evidence in dense_evidence.items():
        if evidence.shape[-2] != 512:
            raise RuntimeError(
                f"dino-metric dense evidence {layer} must be 512 tokens "
                f"(2 frames x 256 patches), got {tuple(evidence.shape)}"
            )
    return tokens, dense_evidence


class DinoFeatureCache:
    """预计算 DINO block11/block23 特征缓存（2026-08-15，步时优化）。

    在线 ViT-L 编码占训练步时 84%（profile：2.97s/3.51s）；冻结塔确定性，
    全部唯一帧特征离线预计算为 fp16 memmap（scripts/build_dino_feature_cache.py），
    训练循环从缓存读。位级一致性由预计算脚本内置验证（torch.equal）保证；
    eval 仍在线编码真实新帧。
    """

    def __init__(self, path: Path) -> None:
        import json
        import pickle

        self.path = Path(path).expanduser()
        if not self.path.is_dir():
            raise ValueError(f"DINO feature cache directory missing: {self.path}")
        with (self.path / "meta.json").open() as fh:
            self.meta = json.load(fh)
        with (self.path / "index.pkl").open("rb") as fh:
            self.index: dict = pickle.load(fh)
        expected_features = self.meta.get("feature_sha256")
        if (
            self.meta.get("feature_identity_contract") != "sha256_full_npy_v1"
            or not isinstance(expected_features, dict)
        ):
            raise ValueError("DINO feature cache lacks full feature SHA-256 metadata")
        for name in ("block11.npy", "block23.npy"):
            expected = expected_features.get(name)
            actual = _sha256_file(self.path / name)
            if not expected or actual != expected:
                raise ValueError(
                    f"DINO feature cache {name} SHA-256 mismatch: "
                    f"expected={expected!r}, actual={actual}"
                )
        self.block23 = np.load(
            self.path / "block23.npy", mmap_mode="r"
        )  # [N, 256, 1024] fp16
        self.block11 = np.load(
            self.path / "block11.npy", mmap_mode="r"
        )  # [N, 256, 1024] fp16
        if self.block23.shape != self.block11.shape:
            raise ValueError("feature cache block23/block11 shape mismatch")
        if self.block23.shape[0] != len(self.index):
            raise ValueError("feature cache rows != index length")
        print(
            f"dino feature cache: {len(self.index)} frames, "
            f"{self.meta.get('model_id')} @{self.meta.get('image_size')}px, "
            f"chunk={self.meta.get('chunk')}, "
            f"dataset_sha256={self.meta.get('dataset_sha256', '?')[:12]}…",
            flush=True,
        )

    def frames(self, rows: np.ndarray) -> dict[int, torch.Tensor]:
        """rows [B, T, W] int64 → {5: [B,T,W,256,D], 11: [...]} GPU fp16。

        键语义与 online forward_hierarchical_dense 相同（5=block11，11=block23）。
        """
        b, t, w = rows.shape
        flat = rows.reshape(-1)
        out = {}
        for key, mem in ((5, self.block11), (11, self.block23)):
            picked = np.asarray(mem[flat])  # [B*T*W, 256, 1024] fp16
            out[key] = torch.from_numpy(picked).reshape(b, t, w, 256, -1)
        return out


def _dino_main_encode_from_cache(
    rows: torch.Tensor,
    cache: DinoFeatureCache,
    device: torch.device,
    *,
    grid: int,
    window: int,
    return_dense: bool = False,
) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
    """缓存读 + 与在线路径同构的池化/证据组装（位级一致，见 precompute 验证）。

    与 _dino_main_online_encode 的差异仅在于 block 特征来自 memmap 而非塔前向：
    同一 16×16→grid×grid adaptive_avg_pool、同一 [d-2,d] 两帧 evidence 序。
    """
    if rows.ndim != 3 or rows.shape[-1] != window:
        raise ValueError(
            f"frame_cache_rows 必须 [B, T, {window}]，got {tuple(rows.shape)}"
        )
    rows_np = rows.detach().cpu().numpy().astype(np.int64)
    evidence = cache.frames(rows_np)  # {5, 11}: [B,T,W,256,D] fp16 CPU
    b, t, w, n_patch, dim = evidence[11].shape
    if n_patch != 256 or dim != 1024:
        raise RuntimeError(
            f"feature cache 期望 256×1024 每帧，got {n_patch}×{dim}"
        )
    tokens = evidence[11].to(device).float()  # [B,T,W,256,D]
    tokens = tokens.reshape(b * t * w, 16, 16, dim).permute(0, 3, 1, 2)
    tokens = F.adaptive_avg_pool2d(tokens, (grid, grid))
    tokens = tokens.permute(0, 2, 3, 1).reshape(b, t, w * grid * grid, dim)
    if not return_dense:
        return tokens
    dense = {}
    for key in (5, 11):
        ev = evidence[key].to(device).float()  # [B,T,W,256,D]
        dense[key] = torch.cat((ev[:, :, 2], ev[:, :, 3]), dim=2).reshape(
            b, t, 512, dim
        )
    return tokens, dense


def _build_dino_metric_stack(
    device: torch.device,
    config: VACompoundConfig,
    *,
    train_metric_head: bool,
    train_relation: bool,
    saved_ctor_config: dict | None = None,
) -> tuple[nn.Module, nn.Module]:
    """DINO-metric（2026-08-15）：从零构建 LanguageMetricField + RelationStateEncoder。

    V-JEPA 的 metric head（768 维 / 1152 token / 24×24 网格）与 DINO 不兼容，
    不复用其权重；此处按 DINO 特征从零构建：h_dim=1024、grid=16（原生
    224px/14 patch，不插值）、两帧 512 token。v3 模式读出（V-JEPA 探针实证的
    更优读出）。训练语义与 V-JEPA 联合微调同构：rel_mlp 仅辅助不动、
    temperature（非 l2_norm 时）冻结、recon 冻结。
    """
    from va_compound.metric_visual_head import (
        LanguageMetricField,
        RelationStateEncoder,
    )

    ctor_config = _canonical_mtvj_metric_head_config(saved_ctor_config or {})
    if saved_ctor_config is not None:
        # 续训：严格使用主 checkpoint 的构造配置（否则权重形状不匹配）。
        if int(ctor_config["h_dim"]) != int(config.main_vision_dim) or int(
            ctor_config["grid"]
        ) != 16:
            raise ValueError(
                "DINO-metric resume 的 mtvj_metric_head_config 与 DINO 特征不兼容："
                f"h_dim={ctor_config['h_dim']} (期望 {config.main_vision_dim}), "
                f"grid={ctor_config['grid']} (期望 16)"
            )
    else:
        # 与生产 V-JEPA metric 栈（metric_field_v6，mode_readout/l2_norm/
        # learnable_temp）同一读出语义，只是视觉输入换 DINO（1024 维、16 网格）。
        ctor_config.update(
            h_dim=int(config.main_vision_dim),
            grid=16,
            l2_norm=True,
            learnable_temp=True,
            mode_readout=True,
        )
    metric_head = LanguageMetricField(
        lang_dim=int(ctor_config["lang_dim"]),
        h_dim=int(ctor_config["h_dim"]),
        d_proj=int(ctor_config["d_proj"]),
        n_roles=int(ctor_config["n_roles"]),
        l2_norm=bool(ctor_config["l2_norm"]),
        learnable_temp=bool(ctor_config["learnable_temp"]),
        temp_init=float(ctor_config["temp_init"]),
        freeze_bias=bool(ctor_config["freeze_bias"]),
        mode_readout=bool(ctor_config["mode_readout"]),
        grid=int(ctor_config["grid"]),
    ).to(device)
    relation_encoder = RelationStateEncoder(
        state_dim=8, d_model=int(config.hidden_dim)
    ).to(device)
    metric_head._mtvj_metric_state_source = MTVJ_METRIC_STATE_SOURCE
    metric_head._mtvj_metric_contract_version = MTVJ_METRIC_CONTRACT_VERSION
    metric_head._mtvj_metric_head_source = "dino-metric-from-scratch"
    # 无外部 metric checkpoint：来源指纹为 DINO 从零训练的合成标记（save
    # checkpoint 的指纹门控对 dino 来源豁免）。
    metric_head._mtvj_external_checkpoint_identity = {
        "source": "dino-metric-from-scratch",
        "sha256": "none-dino-metric-from-scratch",
    }
    metric_head.train(train_metric_head)
    for name, parameter in metric_head.named_parameters():
        action_connected = not name.startswith("rel_mlp.")
        if name == "temperature" and not metric_head.l2_norm:
            action_connected = False
        if name == "spatial_bias" and metric_head.freeze_bias:
            action_connected = False
        parameter.requires_grad_(train_metric_head and action_connected)
    relation_encoder.train(train_relation)
    for name, parameter in relation_encoder.named_parameters():
        if name.startswith("recon."):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(train_relation)
    print(
        "dino-metric: LanguageMetricField 从零构建 "
        f"(h_dim={metric_head.h_dim}, grid={metric_head.grid}, "
        f"dense_tokens={metric_head.dense_tokens}, "
        f"mode_readout={metric_head.mode_readout}) + "
        f"RelationStateEncoder(state_dim=8, d_model={config.hidden_dim})；"
        f"train_metric_head={train_metric_head}, train_relation={train_relation}",
        flush=True,
    )
    return metric_head, relation_encoder


def _dino_metric_tokens(
    metric_head: nn.Module,
    relation_encoder: nn.Module,
    dense_evidence: dict[int, Tensor],
    batch: dict[str, Tensor],
    device: torch.device,
    *,
    train_metric_head: bool = False,
    roi_head: nn.Module | None = None,
    roi_backbone: nn.Module | None = None,
    roi_frames=None,
    roi_alpha: float = 0.0,
) -> tuple[Tensor | None, Tensor | None]:
    """DINO-metric dense evidence → ``(metric_tokens, metric_g)``。

    与 ``_mtvj_online_encode`` 同构（仅 token 数/网格不同）：language hidden
    沿 T 复制 → coarse LanguageMetricField → 可选、训练/评测同构的 task35 ROI
    refinement → ``g_t = p * visibility [B,T,8]`` → RelationStateEncoder
    ``[B,T,2,d_model]``。冻结时 detach；联合训练时 coarse localization 保留梯度。
    """
    if metric_head is None or relation_encoder is None:
        return None, None
    from va_compound.model import dense_coords

    batch_size, sequence_length, _, _ = dense_evidence[11].shape
    head_dtype = next(metric_head.parameters()).dtype
    language_hidden = batch["language_hidden"].to(device=device, dtype=head_dtype)
    language_mask = batch.get("language_mask")
    if language_mask is None:
        language_mask = torch.ones(
            language_hidden.shape[:2], dtype=torch.bool, device=device
        )
    else:
        language_mask = language_mask.to(device=device)
    coords = dense_coords(512, device=device, dtype=head_dtype)

    language_flat = language_hidden.repeat_interleave(sequence_length, dim=0)
    mask_flat = language_mask.repeat_interleave(sequence_length, dim=0)

    def run_metric_head():
        flat = {
            layer: evidence.reshape(
                batch_size * sequence_length, -1, evidence.shape[-1]
            ).to(dtype=head_dtype)
            for layer, evidence in dense_evidence.items()
        }
        return metric_head(
            flat[5],
            flat[11],
            language_flat,
            mask_flat,
            coords,
        )

    def apply_roi(out):
        if roi_head is None or roi_alpha == 0.0:
            return out
        if roi_backbone is None or roi_frames is None:
            raise ValueError(
                "DINO ROI policy training requires the frozen backbone and raw frames"
            )
        frames_np = (
            roi_frames.detach().cpu().numpy()
            if isinstance(roi_frames, Tensor)
            else np.asarray(roi_frames)
        )
        if frames_np.ndim != 6 or frames_np.shape[:3] != (
            batch_size,
            sequence_length,
            4,
        ):
            raise ValueError(
                "DINO ROI frames must be [B,T,4,H,W,3], got "
                f"{tuple(frames_np.shape)}"
            )
        raw = (
            torch.from_numpy(
                np.ascontiguousarray(frames_np[:, :, (2, 3)]).reshape(
                    batch_size * sequence_length,
                    2,
                    *frames_np.shape[3:],
                )
            )
            .float()
            .div_(255.0)
            .permute(0, 1, 4, 2, 3)
            .to(device)
        )
        out.p, out.visibility = refine_metric_roi_positions_dino(
            out.p,
            out.visibility,
            raw,
            roi_backbone,
            roi_head,
            language_flat,
            mask_flat,
            coords,
            alpha=roi_alpha,
        )
        return out

    if train_metric_head:
        trainable = [p for p in metric_head.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "--mtvj-train-metric-head 已开启但 DINO metric head 没有可训练参数"
            )
        out = apply_roi(run_metric_head())
        g = _mtvj_metric_positions(
            out,
            getattr(
                metric_head,
                "_mtvj_metric_state_source",
                MTVJ_LEGACY_METRIC_STATE_SOURCE,
            ),
        ).reshape(batch_size, sequence_length, -1)
    else:
        with torch.no_grad():
            out = apply_roi(run_metric_head())
            g = (
                _mtvj_metric_positions(
                    out,
                    getattr(
                        metric_head,
                        "_mtvj_metric_state_source",
                        MTVJ_LEGACY_METRIC_STATE_SOURCE,
                    ),
                )
                .reshape(batch_size, sequence_length, -1)
                .detach()
            )
    return _mtvj_relation_tokens(g, relation_encoder), g


def _load_mtvj_metric_checkpoint(
    path: Path,
    device: torch.device,
    config: VACompoundConfig,
    *,
    train_relation: bool = False,
    train_metric_head: bool = False,
    policy_relation_state: dict[str, Tensor] | None = None,
    policy_metric_state: dict[str, Tensor] | None = None,
    policy_metric_config: dict | None = None,
    policy_metric_identity: dict | None = None,
    policy_metric_migration: dict | None = None,
    policy_training_contract: dict | None = None,
    exact_resume: bool = False,
    replace_metric_head_from_external: bool = False,
) -> tuple[nn.Module, nn.Module]:
    """MT-VJ（契约 §2/§6）：加载 metric head 与 relation encoder。

    checkpoint 契约：``{"config": {...}, "metric_head": state_dict,
    "relation_encoder": state_dict, "contract": "mt_vj_metric_field_v1"}``。
    ctor 参数从 checkpoint config 按签名过滤注入（缺省用契约默认值）；
    V-JEPA 始终冻结；metric localization path 与 relation encoder 可分别由
    动作 loss 以独立小学习率联合微调。
    """
    from va_compound.metric_visual_head import LanguageMetricField, RelationStateEncoder

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy_contract = policy_training_contract or {}
    if exact_resume and replace_metric_head_from_external:
        raise ValueError(
            "--replace-mtvj-metric-head-from-external 禁止与 --resume-exact 同时使用"
        )
    if replace_metric_head_from_external and policy_relation_state is None:
        raise ValueError(
            "--replace-mtvj-metric-head-from-external requires the --resume "
            "checkpoint to contain mtvj_relation_encoder；迁移只替换 metric head，"
            "不会随机重建主策略的 8D relation encoder"
        )
    if replace_metric_head_from_external and not train_relation:
        raise ValueError(
            "visibility-gated metric-state migration preserves old relation weights "
            "only as a warm start；requires train_relation=True"
        )
    if (
        policy_contract.get("metric_head_checkpointed") is True
        and not replace_metric_head_from_external
    ):
        missing_main = [
            key
            for key, value in (
                ("mtvj_metric_head", policy_metric_state),
                ("mtvj_metric_head_config", policy_metric_config),
                ("mtvj_metric_checkpoint_identity", policy_metric_identity),
            )
            if value is None
        ]
        if missing_main:
            raise ValueError(
                "resume checkpoint 声明 metric_head_checkpointed=True，"
                f"但缺少 {missing_main}"
            )
    if not replace_metric_head_from_external and policy_metric_state is not None and (
        policy_metric_config is None or policy_metric_identity is None
    ):
        raise ValueError(
            "主 checkpoint 含 mtvj_metric_head，但缺少完整构造配置或外部来源指纹；"
            "拒绝用当前外部 checkpoint 猜测网络语义"
        )
    if (
        not replace_metric_head_from_external
        and policy_metric_config is not None
        and policy_metric_state is None
    ):
        raise ValueError(
            "主 checkpoint 含 mtvj_metric_head_config 但缺少 mtvj_metric_head"
        )

    required_external_states = []
    if policy_metric_state is None or replace_metric_head_from_external:
        required_external_states.append("metric_head")
    if policy_relation_state is None:
        required_external_states.append("relation_encoder")
    for key in required_external_states:
        if key not in ckpt:
            raise ValueError(
                f"--metric-visual-checkpoint {path} 缺少键 {key!r}（契约 §2）"
            )
    contract = ckpt.get("contract")
    if (
        replace_metric_head_from_external
        and contract != "mt_vj_metric_field_v1"
    ):
        raise ValueError(
            "显式 MT-VJ metric-head 迁移要求 external checkpoint contract="
            f"'mt_vj_metric_field_v1'，实际为 {contract!r}"
        )
    if contract is not None and contract != "mt_vj_metric_field_v1":
        raise ValueError(
            f"--metric-visual-checkpoint contract={contract!r} != "
            f"'mt_vj_metric_field_v1'（阶段 V checkpoint 不匹配）"
        )
    external_config = ckpt.get("config")
    external_ctor_config = _canonical_mtvj_metric_head_config(
        external_config,
        require_complete=replace_metric_head_from_external,
    )
    external_visibility_proven = (
        isinstance(external_config, dict)
        and external_config.get("loc_only") is False
        and external_config.get("relation_encoder_trained") is True
        and int(external_config.get("training_state_version", 0)) >= 2
        and int(external_config.get("steps_done", 0)) > 0
    )
    if policy_metric_state is None and not external_visibility_proven:
        raise ValueError(
            "MT-VJ visibility-gated runtime requires an external checkpoint that "
            "proves visibility training (loc_only=False, "
            "relation_encoder_trained=True, training_state_version>=2, "
            "steps_done>0)"
        )
    if replace_metric_head_from_external or policy_relation_state is None:
        runtime_metric_source = MTVJ_METRIC_STATE_SOURCE
        runtime_metric_version = MTVJ_METRIC_CONTRACT_VERSION
    else:
        runtime_metric_source = policy_contract.get(
            "metric_state_source", MTVJ_LEGACY_METRIC_STATE_SOURCE
        )
        runtime_metric_version = int(
            policy_contract.get(
                "metric_contract_version", MTVJ_LEGACY_METRIC_CONTRACT_VERSION
            )
        )
        allowed_runtime = {
            (MTVJ_LEGACY_METRIC_STATE_SOURCE, MTVJ_LEGACY_METRIC_CONTRACT_VERSION),
            (MTVJ_METRIC_STATE_SOURCE, MTVJ_METRIC_CONTRACT_VERSION),
        }
        if (runtime_metric_source, runtime_metric_version) not in allowed_runtime:
            raise ValueError(
                "主 checkpoint 的 MT-VJ metric-state 契约未知："
                f"source={runtime_metric_source!r}, version={runtime_metric_version}"
            )
    required_metric_tasks: tuple[str, ...] = ()
    if replace_metric_head_from_external:
        from scripts.build_longtraj_features import ENV_TO_TASK

        required_metric_tasks = tuple(sorted(ENV_TO_TASK))
        external_tasks = (
            external_config.get("tasks")
            if isinstance(external_config, dict)
            else None
        )
        external_task_set = (
            {str(task) for task in external_tasks}
            if isinstance(external_tasks, (list, tuple))
            else set()
        )
        missing_tasks = sorted(set(required_metric_tasks) - external_task_set)
        if missing_tasks:
            raise ValueError(
                "显式 MT-VJ metric-head 迁移要求 external checkpoint "
                "config.tasks 覆盖全部 49 个 MetaWorld 任务；"
                f"缺少 {missing_tasks}（实际为 {external_tasks!r}）"
            )
        if not external_visibility_proven:
            raise ValueError(
                "显式 MT-VJ metric-head 迁移要求 checkpoint 明确证明 visibility "
                "head 已训练：loc_only=False, relation_encoder_trained=True, "
                "training_state_version>=2, steps_done>0；拒绝把 loc-only/random "
                f"vis_mlp 接入动作门控（config={external_config!r}）"
            )
    current_identity = _mtvj_metric_checkpoint_identity(path, ckpt)
    migration_record = None
    if replace_metric_head_from_external:
        ctor_config = external_ctor_config
        metric_state = ckpt["metric_head"]
        source_identity = current_identity
        metric_source = "external metric checkpoint (explicit all-task migration)"
        migration_record = {
            "contract_version": 3,
            "kind": "replace_mtvj_metric_head_from_external",
            "metric_state_transition": {
                "from": policy_contract.get("metric_state_source"),
                "to": MTVJ_METRIC_STATE_SOURCE,
            },
            "relation_encoder_initialization": "policy_warm_start_requires_finetune",
            "required_tasks": list(required_metric_tasks),
            "replaced_policy_metric_head": policy_metric_state is not None,
            "source_checkpoint_identity": dict(current_identity),
            "previous_policy_checkpoint_identity": (
                dict(policy_metric_identity)
                if isinstance(policy_metric_identity, dict)
                else None
            ),
        }
    elif policy_metric_state is not None:
        ctor_config = _canonical_mtvj_metric_head_config(
            policy_metric_config,
            require_complete=True,
        )
        config_mismatch = (
            {"checkpoint": ctor_config, "external": external_ctor_config}
            if ctor_config != external_ctor_config
            else None
        )
        identity_mismatch = _mtvj_metric_identity_mismatches(
            policy_metric_identity or {}, current_identity
        )
        if config_mismatch or identity_mismatch:
            detail = (
                f"constructor={config_mismatch}, fingerprint={identity_mismatch}"
            )
            if exact_resume:
                raise ValueError(
                    "--resume-exact 的 MT-VJ 外部 checkpoint 与保存时不一致："
                    f"{detail}"
                )
            print(
                "WARNING: --resume 检测到外部 MT-VJ checkpoint 已变化；"
                "本次严格使用主 checkpoint 的构造配置和权重，不使用变化后的外部语义。"
                f" {detail}",
                flush=True,
            )
        metric_state = policy_metric_state
        source_identity = dict(policy_metric_identity or {})
        metric_source = "main policy checkpoint"
        if isinstance(policy_metric_migration, dict):
            migration_record = dict(policy_metric_migration)
    else:
        ctor_config = external_ctor_config
        metric_state = ckpt["metric_head"]
        source_identity = current_identity
        metric_source = "external metric checkpoint (legacy migration)"

    metric_head = LanguageMetricField(**ctor_config).to(device)
    try:
        metric_head.load_state_dict(metric_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "MT-VJ metric head 与保存的构造配置不兼容；"
            f"source={metric_source}: {exc}"
        ) from exc
    metric_head._mtvj_constructor_config = dict(ctor_config)
    metric_head._mtvj_external_checkpoint_identity = dict(source_identity)
    metric_head._mtvj_current_external_checkpoint_identity = dict(current_identity)
    metric_head._mtvj_metric_head_source = metric_source
    metric_head._mtvj_metric_head_migration = migration_record
    metric_head._mtvj_metric_state_source = runtime_metric_source
    metric_head._mtvj_metric_contract_version = runtime_metric_version
    # metric tokens 保持 8 维，但用可见度门控定位坐标，避免把未监督/不可见角色
    # 的任意坐标注入动作策略。
    # 下未训练的 out.relation；RelationStateEncoder 以 state_dim=8 重建——旧权重是
    # loc-only 随机初始化且从未训练（Codex v4 审查 P0-1），维度不匹配时直接丢弃。
    relation_d_model = (
        int(config.hidden_dim)
        if policy_relation_state is not None
        else int((ckpt.get("config") or {}).get("d_model", 512))
    )

    def new_relation_encoder() -> nn.Module:
        return RelationStateEncoder(state_dim=8, d_model=relation_d_model).to(device)

    relation_encoder = new_relation_encoder()
    old_rel_sd = (
        policy_relation_state
        if policy_relation_state is not None
        else ckpt["relation_encoder"]
    )
    try:
        relation_encoder.load_state_dict(old_rel_sd, strict=True)
    except RuntimeError as e:
        if policy_relation_state is not None:
            raise ValueError(
                "主 checkpoint 的 mtvj_relation_encoder 与 8D p_flat/"
                f"hidden_dim={config.hidden_dim} 契约不兼容：{e}"
            ) from e
        # load_state_dict 可能在报 shape mismatch 前已拷贝兼容的 bias/norm；
        # 必须重新构建，不能留下“半旧半新”的隐式状态。
        relation_encoder = new_relation_encoder()
        print(
            f"[mtvj] relation_encoder state_dim 不兼容，丢弃旧随机权重重建 "
            f"（loc-only 下未训练，Codex P0-1）：{e}"
        )
    metric_head.train(train_metric_head)
    for name, parameter in metric_head.named_parameters():
        # The gated action state consumes out.p and out.visibility.  rel_mlp is
        # still auxiliary-only and must not enter the action optimizer.
        action_connected = not name.startswith("rel_mlp.")
        if name == "temperature" and not metric_head.l2_norm:
            action_connected = False
        if name == "spatial_bias" and metric_head.freeze_bias:
            action_connected = False
        parameter.requires_grad_(train_metric_head and action_connected)
    relation_encoder.train(train_relation)
    for name, parameter in relation_encoder.named_parameters():
        # recon 只有阶段 V 的重建辅助 loss 才会用到；动作前向不调用，
        # 因此联合微调时仍冻结，避免“进 optimizer 但永远无梯度”。
        parameter.requires_grad_(train_relation and not name.startswith("recon."))
    # metric_tokens [B, 2, d_model] 加入每层 action cross-attention（契约 §5）：
    # d_model 必须等于 VACompoundConfig.hidden_dim，启动即校验（fail-fast）。
    # 注意：metric tokens 输入已切换为 v4 定位 out.p（8 维），probe 用 state_dim=8。
    state_dim = 8
    with torch.no_grad():
        try:
            probe_g, _ = relation_encoder(
                torch.zeros(1, state_dim, device=device),
                torch.zeros(1, state_dim, device=device),
            )
        except RuntimeError as exc:
            raise ValueError(
                f"relation encoder 探针失败（state_dim={state_dim}？）：{exc}"
            ) from exc
    if probe_g.shape[-1] != config.hidden_dim:
        raise ValueError(
            f"relation encoder d_model={probe_g.shape[-1]} != "
            f"VACompoundConfig.hidden_dim={config.hidden_dim}"
            "（metric_tokens 加入每层 action cross-attention 需同维）"
        )
    print(
        f"mtvj: {'可训练定位路径' if train_metric_head else '冻结'} metric head "
        f"（params={sum(p.numel() for p in metric_head.parameters()):,}，"
        f"trainable={sum(p.numel() for p in metric_head.parameters() if p.requires_grad):,}）"
        f" + {'可训练' if train_relation else '冻结'} relation encoder "
        f"（params={sum(p.numel() for p in relation_encoder.parameters()):,}）"
        f"，trainable={sum(p.numel() for p in relation_encoder.parameters() if p.requires_grad):,}"
        f" from {metric_source}；external={path}",
        flush=True,
    )
    return metric_head, relation_encoder


def _mtvj_metric_deltas(g: Tensor) -> Tensor:
    """Return causal metric deltas, with no invented predecessor at t=0."""
    if g.ndim != 3:
        raise ValueError(f"MT-VJ metric state must be [B, T, D], got {tuple(g.shape)}")
    nu = torch.zeros_like(g)
    if g.shape[1] > 1:
        nu[:, 1:] = g[:, 1:] - g[:, :-1]
    return nu


def _mtvj_relation_tokens(g: Tensor, relation_encoder: nn.Module) -> Tensor:
    """Encode metric positions while preserving every enabled upstream gradient."""
    if g.ndim != 3:
        raise ValueError(f"MT-VJ metric state must be [B, T, D], got {tuple(g.shape)}")
    batch_size, sequence_length, _ = g.shape
    nu = _mtvj_metric_deltas(g)
    z_g, z_nu = relation_encoder(
        g.reshape(batch_size * sequence_length, -1),
        nu.reshape(batch_size * sequence_length, -1),
    )
    return torch.stack((z_g, z_nu), dim=1).reshape(
        batch_size, sequence_length, 2, -1
    )


def _mtvj_online_encode(
    frames,
    backbone,
    metric_head,
    relation_encoder,
    batch: dict[str, Tensor],
    device: torch.device,
    *,
    train_metric_head: bool = False,
    roi_head: nn.Module | None = None,
    roi_alpha: float = 0.0,
) -> tuple[dict[int, Tensor], Tensor | None, Tensor | None]:
    """MT-VJ（契约 §6）：frames [B, T, W, 384, 384, 3] uint8 → (dense, tokens, g)。

    - dense_evidence：``{5: [B, T, 1152, D], 11: [B, T, 1152, D]}``——与 live 路径
      同款 ``preprocess_batch`` 预处理（ImageNet 归一化 [B*T, W, 3, 384, 384]）→
      冻结 ``forward_hierarchical_dense``（fp16）→ 未池化全 patch（t→y→x 序），
      注入模型前以 fp32 交付（与既有 ``vision_tokens = encoded.float()`` 约定一致）；
    - metric_tokens：有 metric_head 时 ``[B, T, 2, d_model] = stack(z_g, z_nu)``，
      g_t = ``out.p * out.visibility`` 展平后的8维可见度门控坐标，
      ν_t = g_t − g_{t−1}（首决策 ν≡0，与 servo g_prev 语义一致）；
    - metric_g：同一 ``g_t``，``[B, T, 8]``，供 WAM ``geo8`` 与动作残差条件
      使用；无 metric_head 时为 None（WAM 不得假装有几何）。
    V-JEPA 始终 no_grad；metric localization path 与 relation encoder 可选联合微调。
    """
    if frames.ndim != 6 or frames.shape[-1] != 3:
        raise ValueError(
            f"MT-VJ frames 必须为 [B, T, W, H, W, 3] uint8，got {tuple(frames.shape)}"
        )
    from va_compound.live_vjepa import IMAGE_SIZE, VISION_WINDOW, _dense_coords

    batch_size, sequence_length, window, height, width, _ = frames.shape
    if window != VISION_WINDOW:
        raise ValueError(
            f"MT-VJ 需要 W={VISION_WINDOW} 帧窗（契约 §1 与 live 路径一致），"
            f"got W={window}"
        )
    # GPU 预处理（2026-08-10，Codex P1-13 优化 + phase2 卡死修复同款）：
    # preprocess_batch 的 CPU bicubic+antialias + list() 逐元素转换极慢（曾卡死），
    # 且 480 原尺寸帧由 GPU bicubic 缩到 384。与 phase2 编码路径同一实现。
    # Preserve the original full-frame path byte-for-byte when ROI is disabled.
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    b, t, w, hh, ww, _ = frames_np.shape
    flat_frames = np.ascontiguousarray(frames_np.reshape(b * t * w, hh, ww, 3))
    video = torch.from_numpy(flat_frames).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    if video.shape[-1] != IMAGE_SIZE or video.shape[-2] != IMAGE_SIZE:
        video = F.interpolate(
            video, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    mean = torch.tensor((0.485, 0.456, 0.406), device=video.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=video.device).view(1, 3, 1, 1)
    inputs = ((video - mean) / std).reshape(b * t, w, 3, IMAGE_SIZE, IMAGE_SIZE)
    raw_video = (
        prepare_metric_roi_video(frames, device, image_size=None)
        if roi_head is not None and roi_alpha != 0.0
        else None
    )
    with torch.no_grad():
        hierarchical = backbone.forward_hierarchical_dense(inputs, out_layers=(5, 11))
    dense_evidence = {
        layer: tokens.reshape(batch_size, sequence_length, -1, tokens.shape[-1]).float()
        for layer, tokens in hierarchical.items()
    }
    if sorted(dense_evidence) != [5, 11]:
        raise ValueError(
            f"forward_hierarchical_dense 应返回 {5, 11} 两层，"
            f"got {sorted(dense_evidence)}"
        )
    metric_tokens = None
    metric_g = None
    if metric_head is not None:
        if relation_encoder is None:
            raise ValueError("metric_head 存在但 relation_encoder 缺失（内部契约破坏）")
        head_dtype = next(metric_head.parameters()).dtype
        language_hidden = batch["language_hidden"].to(device=device, dtype=head_dtype)
        language_mask = batch.get("language_mask")
        if language_mask is None:
            language_mask = torch.ones(
                language_hidden.shape[:2], dtype=torch.bool, device=device
            )
        else:
            language_mask = language_mask.to(device=device)
        coords = torch.from_numpy(_dense_coords()).to(device=device, dtype=head_dtype)
        flat = {
            layer: ev.reshape(batch_size * sequence_length, -1, ev.shape[-1]).to(
                dtype=head_dtype
            )
            for layer, ev in dense_evidence.items()
        }
        def run_metric_head():
            return metric_head(
                flat[5],
                flat[11],
                language_hidden.repeat_interleave(sequence_length, dim=0),
                language_mask.repeat_interleave(sequence_length, dim=0),
                coords,
            )

        if train_metric_head:
            trainable = [p for p in metric_head.parameters() if p.requires_grad]
            if not trainable:
                raise ValueError(
                    "--mtvj-train-metric-head 已开启但 metric head 没有可训练参数"
                )
            out = run_metric_head()
            if roi_head is not None and roi_alpha != 0.0:
                out.p, out.visibility = refine_metric_roi_positions(
                    out.p,
                    out.visibility,
                    raw_video,
                    backbone,
                    roi_head,
                    language_hidden.repeat_interleave(sequence_length, dim=0),
                    language_mask.repeat_interleave(sequence_length, dim=0),
                    coords,
                    alpha=roi_alpha,
                )
            g = _mtvj_metric_positions(
                out,
                getattr(metric_head, "_mtvj_metric_state_source", MTVJ_LEGACY_METRIC_STATE_SOURCE),
            ).reshape(
                batch_size, sequence_length, -1
            )
        else:
            with torch.no_grad():
                out = run_metric_head()
                if roi_head is not None and roi_alpha != 0.0:
                    out.p, out.visibility = refine_metric_roi_positions(
                        out.p,
                        out.visibility,
                        raw_video,
                        backbone,
                        roi_head,
                        language_hidden.repeat_interleave(sequence_length, dim=0),
                        language_mask.repeat_interleave(sequence_length, dim=0),
                        coords,
                        alpha=roi_alpha,
                    )
                g = _mtvj_metric_positions(
                    out,
                    getattr(metric_head, "_mtvj_metric_state_source", MTVJ_LEGACY_METRIC_STATE_SOURCE),
                ).reshape(
                    batch_size, sequence_length, -1
                ).detach()
        # metric tokens 输入 = 可见度门控定位坐标（[B*T, R=4, 2] → [B,T,8]），
        # 替代 loc-only 下未训练的 out.relation（Codex v4 审查 P0-1：随机 relation
        # 注入 = 随机语义；13.45px 反事实验证过的定位才是有效度量证据）。
        # 不能放在上面的 no_grad 里：联合微调时动作 loss 需反传到
        # relation encoder；冻结模式下参数和 g 都不求导，仍不会建图。
        metric_g = g
        metric_tokens = _mtvj_relation_tokens(g, relation_encoder)
    return dense_evidence, metric_tokens, metric_g


def _mtvj_encode_frames_dense(
    frames_np: np.ndarray, backbone: nn.Module, device: torch.device,
) -> dict[int, Tensor]:
    """MT-VJ 帧 → 冻结 V-JEPA dense evidence（与 _mtvj_online_encode 预处理同款：
    GPU bicubic 384 + ImageNet 归一化 + forward_hierarchical_dense fp16 → {5,11}
    fp32。输入 [B, T, W, 384, 384, 3] uint8；双数据流辅助批次复用。"""
    from va_compound.live_vjepa import IMAGE_SIZE

    b, t, w, hh, ww, _ = frames_np.shape
    flat = np.ascontiguousarray(frames_np.reshape(b * t * w, hh, ww, 3))
    video = torch.from_numpy(flat).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    if video.shape[-1] != IMAGE_SIZE or video.shape[-2] != IMAGE_SIZE:
        video = F.interpolate(
            video, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    mean = torch.tensor((0.485, 0.456, 0.406), device=video.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=video.device).view(1, 3, 1, 1)
    inputs = ((video - mean) / std).reshape(b * t, w, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        hierarchical = backbone.forward_hierarchical_dense(inputs, out_layers=(5, 11))
    dense_evidence = {
        layer: tokens.reshape(b, t, -1, tokens.shape[-1]).float()
        for layer, tokens in hierarchical.items()
    }
    if sorted(dense_evidence) != [5, 11]:
        raise ValueError(
            f"forward_hierarchical_dense 应返回 {5, 11} 两层，"
            f"got {sorted(dense_evidence)}"
        )
    return dense_evidence


def _mtvj_visual_aux_loss(
    backbone: nn.Module,
    metric_head: nn.Module,
    task: str,
    rng: np.random.Generator,
    aux_batch: int,
    lang_aux_cache: dict,
    device: torch.device,
    loc_lambda: float,
    vis_lambda: float,
    sigma_px: float = 4.0,
    hinge_margin: float = 0.1,
    sim_batch: dict | None = None,
) -> tuple[Tensor, dict]:
    """双数据流视觉辅助批次（阶段 C，2026-08-12）：在线仿真真值（
    make_metric_batch）→ 冻结 V-JEPA dense 编码 → metric head 前向 →
    L_aux = λ_loc·(hinge + pos + offset) + λ_vis·BCE(visibility)。

    - 只反传 metric head：V-JEPA no_grad、语言缓存为预计算张量（冻结）、
      rel_mlp 无梯度（compute_losses loc_only=True 不产生 rel 项）。
    - 语言用数据集预计算 hidden（metadata.tasks[tid] ↔ instruction_id=tid，
      与 build_longtraj_features 的 task_language_t 一致），不加载 Qwen。
    - 动作批次与辅助批次不需要同一帧：标准多数据集联合目标。
    """
    from prepare_metaworld_metric import make_metric_batch
    from scripts.build_longtraj_features import ENV_TO_TASK
    from train_metric_visual import compute_losses
    from va_compound.live_vjepa import _dense_coords

    sim = (
        make_metric_batch(task, rng, aux_batch)
        if sim_batch is None
        else sim_batch
    )
    text = ENV_TO_TASK.get(task, task)
    if text not in lang_aux_cache:
        raise KeyError(
            f"visual aux: 语言缓存缺少 {text!r}（instruction_id 映射不一致）；"
            f"任务 {task!r} 无法取预计算 hidden"
        )
    head_dtype = next(metric_head.parameters()).dtype
    hid, mask = lang_aux_cache[text]
    lang_hidden = hid.repeat(aux_batch, 1, 1).to(device=device, dtype=head_dtype)
    lang_mask = mask.repeat(aux_batch, 1).to(device=device)
    frames_np = np.asarray(sim["frames"])  # [B, 4, 384, 384, 3] uint8
    if frames_np.ndim != 5 or frames_np.shape[1] != 4:
        raise ValueError(
            f"make_metric_batch frames 必须 [B,4,384,384,3] uint8，"
            f"got {tuple(frames_np.shape)}"
        )
    dense_evidence = _mtvj_encode_frames_dense(
        frames_np[:, None, :, :, :, :], backbone, device  # [B, T=1, W=4, ...]
    )
    flat = {
        layer: ev.reshape(aux_batch, -1, ev.shape[-1]).to(dtype=head_dtype)
        for layer, ev in dense_evidence.items()
    }
    coords = torch.from_numpy(_dense_coords()).to(device=device, dtype=head_dtype)
    out = metric_head(flat[5], flat[11], lang_hidden, lang_mask, coords)
    keypoints = torch.from_numpy(sim["keypoints"]).to(device=device, dtype=head_dtype)
    visibility = torch.from_numpy(sim["visibility"]).to(device=device, dtype=head_dtype)
    loc_total, parts = compute_losses(
        out, keypoints, visibility, torch.zeros_like(out.p),
        sigma_px=sigma_px, loc_only=True, offset_supervision=True,
        hinge=True, hinge_margin=hinge_margin,
    )
    vis_loss = F.binary_cross_entropy_with_logits(
        out.visibility_logits, visibility, reduction="mean"
    )
    vis = visibility
    radial_error_px = (out.p - keypoints).norm(dim=-1) * 384.0
    rmse_px = torch.sqrt(
        (radial_error_px.square() * vis).sum() / vis.sum().clamp_min(1.0)
    )
    loss_aux = loc_lambda * loc_total + vis_lambda * vis_loss
    parts.update(
        {
            "loc": loc_total.item(),
            "vis": vis_loss.item(),
            "rmse_px": rmse_px.item(),
            "total": loss_aux.item(),
        }
    )
    return loss_aux, parts


def _dino_visual_aux_loss(
    main_backbone,
    metric_head,
    task: str,
    rng: np.random.Generator,
    aux_batch: int,
    lang_aux_cache: dict,
    device: torch.device,
    loc_lambda: float,
    vis_lambda: float,
    sigma_px: float = 4.0,
    hinge_margin: float = 0.1,
    sim_batch: dict | None = None,
) -> tuple[Tensor, dict]:
    """DINO 版视觉辅助批次（2026-08-16）：MT-VJ 高清头的真正训练信号。

    与 V-JEPA ``_mtvj_visual_aux_loss`` 同协议，仅编码器/网格不同：仿真真值
    （make_metric_batch 定位/可见度 + true 480px raw_frames）→ 冻结 DINO block11/block23 两帧
    [d-2,d] 全 patch evidence（512 token）→ metric head（grid=16）→
    L_aux = λloc·(hinge + pos + offset，image_size=224) + λvis·BCE(visibility)。
    只反传 metric head（head_dtype fp32；DINO 冻结只读）。
    """
    from prepare_metaworld_metric import make_metric_batch
    from scripts.build_longtraj_features import ENV_TO_TASK
    from train_metric_visual import compute_losses
    from va_compound.model import dense_coords

    sim = (
        make_metric_batch(task, rng, aux_batch, include_raw_frames=True)
        if sim_batch is None
        else sim_batch
    )
    text = ENV_TO_TASK.get(task, task)
    if text not in lang_aux_cache:
        raise KeyError(
            f"visual aux: 语言缓存缺少 {text!r}（instruction_id 映射不一致）；"
            f"任务 {task!r} 无法取预计算 hidden"
        )
    head_dtype = next(metric_head.parameters()).dtype
    hid, mask = lang_aux_cache[text]
    lang_hidden = hid.repeat(aux_batch, 1, 1).to(device=device, dtype=head_dtype)
    lang_mask = mask.repeat(aux_batch, 1).to(device=device)
    frames_np = np.asarray(
        sim["raw_frames"]
    )  # [B, 4, 480, 480, 3] true simulator renders
    if frames_np.shape != (aux_batch, 4, 480, 480, 3) or frames_np.dtype != np.uint8:
        raise ValueError(
            "DINO visual aux raw_frames must be uint8 [B,4,480,480,3], "
            f"got {tuple(frames_np.shape)}/{frames_np.dtype}"
        )
    # 帧 [d-2,d]（窗口索引 2,3）→ 与训练 DINO dense evidence 同一预处理。
    sel = np.ascontiguousarray(frames_np[:, (2, 3)])  # [B,2,H,W,3]
    b, s = sel.shape[:2]
    images = (
        torch.from_numpy(sel.reshape(b * s, *sel.shape[2:]))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .to(device)
    )
    if tuple(images.shape[-2:]) != (main_backbone.image_size, main_backbone.image_size):
        images = F.interpolate(
            images,
            size=(main_backbone.image_size, main_backbone.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        hierarchical = main_backbone.forward_hierarchical_dense(
            ((images - mean) / std).half()
        )
    flat = {
        layer: tokens.reshape(b, -1, tokens.shape[-1]).float().to(dtype=head_dtype)
        for layer, tokens in hierarchical.items()
    }  # {5, 11}: [B, 512, 1024]
    coords = dense_coords(512, device=device, dtype=head_dtype)
    out = metric_head(flat[5], flat[11], lang_hidden, lang_mask, coords)
    keypoints = torch.from_numpy(sim["keypoints"]).to(device=device, dtype=head_dtype)
    visibility = torch.from_numpy(sim["visibility"]).to(device=device, dtype=head_dtype)
    loc_total, parts = compute_losses(
        out, keypoints, visibility, torch.zeros_like(out.p),
        sigma_px=sigma_px, loc_only=True, offset_supervision=True,
        hinge=True, hinge_margin=hinge_margin, image_size=main_backbone.image_size,
    )
    vis_loss = F.binary_cross_entropy_with_logits(
        out.visibility_logits, visibility, reduction="mean"
    )
    radial_error_px = (out.p - keypoints).norm(dim=-1) * float(main_backbone.image_size)
    rmse_px = torch.sqrt(
        (radial_error_px.square() * visibility).sum() / visibility.sum().clamp_min(1.0)
    )
    loss_aux = loc_lambda * loc_total + vis_lambda * vis_loss
    parts.update(
        {
            "loc": loc_total.item(),
            "vis": vis_loss.item(),
            "rmse_px": rmse_px.item(),
            "total": loss_aux.item(),
        }
    )
    return loss_aux, parts


def _mtvj_visual_aux_sample(
    task_descriptions: list[str],
    task_weights: Tensor,
    env_by_description: dict[str, str],
    *,
    seed: int,
    global_step: int,
) -> tuple[str, np.random.Generator]:
    """Choose one reproducible auxiliary environment for a committed step.

    Dataset metadata stores human-readable instructions while MetaWorld's data
    generator accepts environment names.  Keeping that conversion here makes
    the contract testable and prevents exact resumes from silently changing
    the auxiliary sequence.
    """
    if not task_descriptions:
        raise ValueError("visual aux: metadata.tasks is empty")
    probabilities = torch.as_tensor(task_weights, dtype=torch.float64).cpu().numpy()
    if probabilities.shape != (len(task_descriptions),):
        raise ValueError(
            "visual aux: task weight count does not match metadata.tasks: "
            f"{probabilities.shape} vs {len(task_descriptions)}"
        )
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(
        np.random.SeedSequence((int(seed), int(global_step), 0xA17))
    )
    description = task_descriptions[int(rng.choice(len(task_descriptions), p=probabilities))]
    env_name = env_by_description.get(description)
    if env_name is None:
        raise KeyError(
            "visual aux: metadata.tasks 描述无法映射到 MetaWorld 环境名："
            f"{description!r}"
        )
    return env_name, rng


def _prepare_mtvj_visual_aux_step(
    task_descriptions: list[str],
    task_weights: Tensor,
    env_by_description: dict[str, str],
    *,
    seed: int,
    global_step: int,
    every: int,
    aux_batch: int,
    include_raw_frames: bool,
) -> tuple[str, np.random.Generator, dict] | None:
    """Generate the CPU simulator batch before the training graph is built."""

    if every <= 0 or global_step % every != 0:
        return None
    task, rng = _mtvj_visual_aux_sample(
        task_descriptions,
        task_weights,
        env_by_description,
        seed=seed,
        global_step=global_step,
    )
    from prepare_metaworld_metric import make_metric_batch

    kwargs = {"include_raw_frames": True} if include_raw_frames else {}
    sim_batch = make_metric_batch(task, rng, aux_batch, **kwargs)
    return task, rng, sim_batch


def servo_correction_t0(
    model: VACompoundPolicy,
    servo: InteractionServo,
    batch: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    """t=0 决策点的伺服修正 [B, A]（pair 分支复用；确定性重算）。

    与 rollout_policy 主路径 t=0 完全相同的输入（g_prev=None → ν≡0），因此
    结果逐位一致——pair 探针与主分支保持同一策略输出（角色查询经
    build_language_cache 重建，确定性）。
    """
    target_dtype = model.vision_projection.weight.dtype
    language_cache = model.build_language_cache(
        batch["language_hidden"], batch.get("language_mask")
    )
    dense = batch["vision_tokens_st"][:, 0].to(dtype=target_dtype)
    coords = batch["coords"][0].to(device=device)
    readout = model.slot_reader(
        dense,
        language_cache.role_queries.to(dtype=target_dtype),
        coords,
    )
    out = servo(
        readout,
        batch["proprio"][:, 0],
        language_cache.role_queries.mean(dim=1).to(dtype=target_dtype),
        a_prev=None,  # t=0 无 g_prev → ν≡0（与 rollout_policy 主路径一致）
        g_prev=None,
    )
    return out.correction.to(dtype=target_dtype)


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


def _world_task_ids(
    batch: dict[str, Tensor], time_index: int, device: torch.device
) -> Tensor:
    task_id = batch.get("task_id")
    if task_id is None:
        task_id = batch.get("task_ids")
    if task_id is None:
        task_id = batch.get("instruction_id")
    if task_id is None:
        raise ValueError("visual World supervision requires per-sample task ids")
    task_id = torch.as_tensor(task_id, device=device)
    if task_id.ndim > 1:
        task_id = task_id[:, time_index]
    if task_id.ndim != 1 or task_id.shape[0] != batch["actions"].shape[0]:
        raise ValueError("World task ids must have shape [B] or [B,T]")
    return task_id.to(dtype=torch.long)


def _nearest_cross_episode_donors(
    proprio: Tensor,
    task_ids: Tensor,
    episode_ids: Tensor,
    eligible: Tensor,
) -> Tensor:
    """Return deterministic proprio-nearest cross-episode donor indices."""

    batch = proprio.shape[0]
    if proprio.ndim != 2:
        raise ValueError("proprio must be [B,P]")
    if task_ids.ndim != 1 or task_ids.shape[0] != batch:
        raise ValueError("task_ids must be [B]")
    if episode_ids.ndim != 1 or episode_ids.shape[0] != batch:
        raise ValueError("episode_ids must be [B]")
    if eligible.ndim != 1 or eligible.shape[0] != batch or eligible.dtype != torch.bool:
        raise ValueError("eligible must be bool [B]")

    state = proprio.detach().to(device="cpu", dtype=torch.float64)
    task = task_ids.detach().to(device="cpu", dtype=torch.int64)
    episode = episode_ids.detach().to(device="cpu", dtype=torch.int64)
    eligible = eligible.detach().to(device="cpu", dtype=torch.bool)
    donors = torch.full((batch,), -1, dtype=torch.int64)
    for row in range(batch):
        if not bool(eligible[row]):
            continue
        candidate = eligible & task.eq(task[row]) & ~episode.eq(episode[row])
        if not bool(candidate.any()):
            continue
        distance = (state - state[row]).square().sum(dim=-1)
        distance.masked_fill_(~candidate, float("inf"))
        donors[row] = int(distance.argmin())
    return donors


def prepare_visual_world_action_ranking(payload: dict) -> dict[str, object]:
    """Attach the fixed train-split shuffled-action table to a dataset payload."""

    actions = torch.as_tensor(payload["actions"])
    proprio = torch.as_tensor(payload["proprio"])
    task_ids = torch.as_tensor(payload["instruction_id"], dtype=torch.int64)
    episode_ids = torch.as_tensor(payload["episode_id"], dtype=torch.int64)
    valid = world_transition_mask(
        torch.as_tensor(payload["action_valid_mask"]), cycle_steps=6
    )
    rows, times = torch.nonzero(valid, as_tuple=True)
    flat_actions = actions[rows, times, :6]
    flat_proprio = proprio[rows, times]
    flat_tasks = task_ids[rows]
    flat_episodes = episode_ids[rows]
    donors = _nearest_cross_episode_donors(
        flat_proprio,
        flat_tasks,
        flat_episodes,
        torch.ones(rows.numel(), dtype=torch.bool),
    )
    if bool((donors < 0).any()):
        raise ValueError("visual World action ranking found a transition without a donor")
    shuffled = flat_actions.index_select(0, donors)
    distinct = shuffled.ne(flat_actions).any(dim=(1, 2))
    table = actions.new_zeros((actions.shape[0], actions.shape[1] - 1, 6, 4))
    table[rows, times] = shuffled
    rank_mask = torch.zeros_like(valid)
    rank_mask[rows, times] = distinct
    payload["world_rank_shuffle_action"] = table
    payload["world_rank_shuffle_mask"] = rank_mask

    identity = torch.stack((rows, times, donors, distinct.to(torch.int64)), dim=1)
    digest = hashlib.sha256(identity.contiguous().numpy().tobytes()).hexdigest()
    return {
        "world_action_donor_contract": WORLD_ACTION_DONOR_CONTRACT,
        "world_action_donor_sha256": digest,
        "world_action_donor_transitions": int(rows.numel()),
        "world_action_rank_transitions": int(distinct.sum()),
    }


def _summarize_visual_world_metrics(
    final_records: list[dict[str, Tensor]],
    stage_records: list[list[tuple[Tensor, Tensor, Tensor]]],
) -> dict[int, dict[str, object]]:
    """Reduce final-stage World/copy metrics independently for each task."""

    if not final_records:
        return {}
    task_values = sorted(
        {
            int(value)
            for record in final_records
            for value in torch.unique(record["task_ids"]).detach().cpu()
        }
    )
    output: dict[int, dict[str, object]] = {}
    metric_names = (
        "world_all",
        "copy_all",
        "world_motion",
        "copy_motion",
        "world_top10",
        "copy_top10",
        "world_static",
        "copy_static",
        "motion_energy",
    )
    for task_id in task_values:
        masks = [
            record["valid"] & record["task_ids"].eq(task_id)
            for record in final_records
        ]
        reduced = {
            name: float(
                masked_world_reduction(
                    [record[name] for record in final_records], masks
                ).detach()
            )
            for name in metric_names
        }
        reduced["gain_all"] = reduced["copy_all"] - reduced["world_all"]
        reduced["gain_motion"] = (
            reduced["copy_motion"] - reduced["world_motion"]
        )
        copy_top10 = float(reduced["copy_top10"])
        reduced["relative_gain_top10"] = (
            (copy_top10 - float(reduced["world_top10"])) / copy_top10
            if copy_top10 > 0.0
            else 0.0
        )
        reduced["transitions"] = sum(int(mask.sum().item()) for mask in masks)
        per_stage = []
        for records in stage_records:
            stage_masks = [
                valid & task_ids.eq(task_id)
                for task_ids, valid, _ in records
            ]
            per_stage.append(
                float(
                    masked_world_reduction(
                        [value for _, _, value in records], stage_masks
                    ).detach()
                )
            )
        reduced["stage_losses"] = per_stage
        output[task_id] = reduced
    return output


def rollout_policy(
    model: VACompoundPolicy,
    batch: dict[str, Tensor],
    noisy_actions: Tensor,
    flow_time: Tensor,
    *,
    text_backbone=None,
    scene_teacher=None,
    tasks=None,
    servo: InteractionServo | None = None,
    servo_stats: dict | None = None,
    dense_evidence: dict[int, Tensor] | None = None,
    metric_tokens: Tensor | None = None,
    action_dense_evidence: dict[int, Tensor] | None = None,
    metric_g: Tensor | None = None,
    wmrm_adep_margin: float = 0.05,
    visual_world_supervision: bool = False,
    wmrm_adep_enabled: bool = False,
    flow_steps: int = 8,
    world_action_rank_step: int = 0,
    world_action_rank_stage: str = "cycle",
    wmrm_action_rank_per_sample_cap: float | None = None,
    wmrm_static_constraint_weight: float = 4.0,
    feature_autocast_bf16: bool = False,
) -> tuple[Tensor, Tensor]:
    if world_action_rank_step < 0:
        raise ValueError("world_action_rank_step must be non-negative")
    if world_action_rank_stage not in {"final", "cycle"}:
        raise ValueError("world_action_rank_stage must be 'final' or 'cycle'")
    peer_world_mode = getattr(model.config, "va_world_mode", "legacy") == "peer_sync_h6"
    if peer_world_mode and wmrm_adep_enabled:
        raise ValueError(
            "peer_sync_h6 rejects nonzero wmrm_adep until action-dependence "
            "counterfactuals use the exact same immutable stage snapshot"
        )
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
    wmrm_world_terms: list[Tensor] = []
    visual_world_stage_records: list[list[tuple[Tensor, Tensor, Tensor]]] = []
    visual_world_objective_stage_records: list[
        list[tuple[Tensor, Tensor, Tensor]]
    ] = []
    visual_world_guard_stage_records: list[list[tuple[Tensor, Tensor, Tensor]]] = []
    visual_world_static_constraint_stage_records: list[
        list[tuple[Tensor, Tensor, Tensor]]
    ] = []
    visual_world_action_shuffle_records: list[tuple[Tensor, Tensor, Tensor]] = []
    peer_readout_loss_records: list[tuple[Tensor, Tensor]] = []
    peer_readout_squared_error_records: list[tuple[Tensor, Tensor]] = []
    visual_world_final_records: list[dict[str, Tensor]] = []
    wmrm_pi_kl_terms: list[Tensor] = []
    wmrm_adep_terms: list[Tensor] = []
    wmrm_med_terms: list[Tensor] = []
    memories: list[VisualMemory] | None = [] if model.config.future_predict else None
    direct_predictions = [] if model.config.direct_head else None
    c2_references = [] if model.config.c2_controller else None
    # Step 2 双新息伺服：语言条件（role queries 均值）与跨决策关系状态。
    target_dtype = model.vision_projection.weight.dtype
    lang_cond = None
    g_prev = None
    transition_validity = None
    rank_shuffle_actions = None
    rank_shuffle_validity = None
    if visual_world_supervision:
        if model.wmrm is None or getattr(model.config, "wmrm_target", None) != "dino":
            raise ValueError("visual World supervision requires WAM4VA with DINO targets")
        action_valid_mask = batch.get("action_valid_mask")
        if action_valid_mask is None:
            raise ValueError(
                "visual World supervision requires the recorded action_valid_mask"
            )
        transition_validity = world_transition_mask(
            action_valid_mask,
            cycle_steps=model.wmrm.cycle_steps,
        )
        rank_shuffle_actions = batch.get("world_rank_shuffle_action")
        rank_shuffle_validity = batch.get("world_rank_shuffle_mask")
        expected_actions = (
            batch["actions"].shape[0],
            batch["actions"].shape[1] - 1,
            model.wmrm.cycle_steps,
            batch["actions"].shape[-1],
        )
        expected_mask = expected_actions[:2]
        if (
            rank_shuffle_actions is None
            or tuple(rank_shuffle_actions.shape) != expected_actions
        ):
            raise ValueError(
                "visual World action ranking requires fixed train-split shuffled "
                f"actions with shape {expected_actions}"
            )
        if (
            rank_shuffle_validity is None
            or rank_shuffle_validity.dtype != torch.bool
            or tuple(rank_shuffle_validity.shape) != expected_mask
        ):
            raise ValueError(
                "visual World action ranking requires bool donor mask with shape "
                f"{expected_mask}"
            )
    if servo is not None:
        if model.slot_reader is None or language_cache.role_queries is None:
            raise ValueError(
                "servo 需要 local_slots + multi_mode 角色读出路径（--servo 校验）"
            )
        lang_cond = language_cache.role_queries.mean(dim=1).to(dtype=target_dtype)
        if servo_stats is not None:
            for key in ("stage", "innovation_flag", "beta", "hyp_entropy", "correction"):
                servo_stats[key] = []
    for time_index in range(batch["actions"].shape[1]):
        pre_step_visual_memory = visual_memory
        semantic_context = None
        mtvj_kwargs = {}
        if dense_evidence is not None:
            # MT-VJ（契约 §5）：dense evidence 只做 K/V（不进 VA 自注意力），
            # metric_tokens（RelationStateEncoder 输出，可选联合微调）加入 action
            # cross-attention；DINO-main 的 full grid tokens 保持为 base vision。
            mtvj_kwargs["dense_evidence"] = {
                layer: evidence[:, time_index] for layer, evidence in dense_evidence.items()
            }
            if metric_tokens is not None:
                mtvj_kwargs["metric_tokens"] = metric_tokens[:, time_index]
        if metric_g is not None:
            mtvj_kwargs["metric_g"] = metric_g[:, time_index]
        if action_dense_evidence is not None:
            mtvj_kwargs["action_dense_evidence"] = {
                layer: evidence[:, time_index]
                for layer, evidence in action_dense_evidence.items()
            }
        if model.config.local_slots:
            vision_in = model.build_local_vision(
                batch["vision_tokens_st"][:, time_index],
                batch["coords"][0].to(device=batch["vision_tokens"].device),
                language_cache.role_queries,
            )
            if model.config.flow_semantic and not model.config.direct_head:
                # π0 式逐层 cross-attn：槽/关系 token（语言实例化语义上下文）
                # 作为 flow head 的 cross-attn K/V，逐层注入。
                semantic_context = vision_in  # [B, 25, vision_dim]
        elif dense_evidence is not None and not model.config.dino_dense_metric:
            # 传统 V-JEPA MT-VJ：VA base vision = H11 Pool16。DINO-main 必须保留
            # batch['vision_tokens'] 的四帧 full-grid token；旧代码在训练时错误地
            # 也走 Pool16，而 eval 保留 grid16，造成 train/eval P0 mismatch。
            h11 = dense_evidence[11][:, time_index]  # [B, 1152, 768]
            vision_in = pool_mtvj_coarse_tokens(h11)  # [B, 16, 768]
        else:
            vision_in = batch["vision_tokens"][:, time_index]
        world_action = None
        if model.wmrm is not None:
            cycle = model.wmrm.cycle_steps
            if batch["actions"].shape[2] < cycle:
                raise ValueError(
                    f"WAM4VA needs {cycle} executable actions, "
                    f"but the training chunk has {batch['actions'].shape[2]}"
                )
            if peer_world_mode:
                # Peer mode owns its deterministic executable-action readout inside
                # the single main encode.  A preliminary Flow decode would create a
                # second, stale snapshot and violate the peer topology contract.
                if cycle != 6 or model.config.action_horizon != 6:
                    raise ValueError("peer_sync_h6 rollout requires exact H6")
                if getattr(model, "world_action_readout", None) is None:
                    raise ValueError("peer_sync_h6 requires world_action_readout")
            elif getattr(model.config, "wmrm_handshake", True):
                proposal_cond, _ = model.encode_condition(
                    vision_in,
                    batch["proprio"][:, time_index],
                    batch["previous_action"][:, time_index],
                    language_cache=language_cache,
                    visual_memory=pre_step_visual_memory,
                    return_visual_memory=True,
                    skip_wmrm=True,
                    **mtvj_kwargs,
                )
                with feature_no_grad_decode_autocast(
                    proposal_cond.device, feature_autocast_bf16
                ):
                    with torch.no_grad():
                        decoded = model.decode_actions(
                            proposal_cond, steps=flow_steps
                        )
                if decoded.shape[1] < cycle:
                    raise ValueError(
                        f"decoded action horizon {decoded.shape[1]} "
                        f"is shorter than WAM4VA cycle {cycle}"
                    )
                world_action = decoded[:, :cycle].clamp(-1.0, 1.0)
            else:
                world_action = batch["actions"][:, time_index, :cycle].clamp(-1.0, 1.0)
        peer_stage_snapshots: list[tuple[tuple, dict]] = []
        original_peer_propose = None
        if peer_world_mode:
            original_peer_propose = model.wmrm.propose

            def record_peer_snapshot(*proposal_args, **proposal_kwargs):
                peer_stage_snapshots.append((proposal_args, dict(proposal_kwargs)))
                return original_peer_propose(*proposal_args, **proposal_kwargs)

            model.wmrm.propose = record_peer_snapshot
        try:
            condition, visual_memory = model.encode_condition(
                vision_in,
                batch["proprio"][:, time_index],
                batch["previous_action"][:, time_index],
                language_cache=language_cache,
                visual_memory=pre_step_visual_memory,
                return_visual_memory=True,
                env_action=world_action,
                detach_wmrm_stage_state=bool(
                    getattr(model.config, "wmrm_detach_proposal_stage_state", False)
                ),
                **mtvj_kwargs,
            )
        finally:
            if original_peer_propose is not None:
                model.wmrm.propose = original_peer_propose
        proposal_auxes = list(getattr(model, "last_wmrm_auxes", None) or ())
        if not proposal_auxes and getattr(model, "last_wmrm", None) is not None:
            proposal_auxes = [model.last_wmrm]
        proposal_pres = list(getattr(model, "last_wmrm_pre_actions", None) or ())
        proposal_meds = list(getattr(model, "last_wmrm_meds", None) or ())
        proposal_kls = list(getattr(model, "last_wmrm_pi_kls", None) or ())
        proposal_last = getattr(model, "last_wmrm", None)
        proposal_last_kl = getattr(model, "last_wmrm_pi_kl", None)
        if model.wmrm is not None and time_index + 1 < batch["actions"].shape[1]:
            target = wmrm_next_feature_target(
                model,
                batch,
                time_index,
                dense_evidence=dense_evidence,
                metric_g=metric_g,
            )
            if not proposal_auxes:
                raise ValueError("WAM4VA produced no world predictions at a supervised step")
            if visual_world_supervision:
                if transition_validity is None:
                    raise RuntimeError("visual World stage context is incomplete")
                valid = transition_validity[:, time_index]
                task_ids = _world_task_ids(batch, time_index, target.device)
                current = model.wmrm.encode_dino_map(
                    batch["vision_tokens"][:, time_index]
                )
                if current is None or current.shape != target.shape:
                    raise ValueError(
                        "visual World current/target maps must match, got "
                        f"{None if current is None else tuple(current.shape)} and "
                        f"{tuple(target.shape)}"
                    )
                logged_action = batch["actions"][
                    :, time_index, : model.wmrm.cycle_steps
                ]
                if peer_world_mode:
                    logged_auxes = list(proposal_auxes)
                    # The main peer encode already consumed the one authoritative
                    # pre-stage snapshot per layer.  Supervision changes only the
                    # executable action supplied to the predictor; it must not
                    # rerun VA, mutate VisualMemory, or advance WAMState again.
                    logged_pres = proposal_pres
                    # Readout supervision follows the exact World transition-valid
                    # mask and has one explicit aggregation contract: supervise only
                    # the final peer stage, then average all H6 action coordinates.
                    final_readout = proposal_auxes[-1].env_action
                    if final_readout is None:
                        raise RuntimeError(
                            "final peer World stage did not expose deterministic readout"
                        )
                    logged_readout = logged_action.to(dtype=final_readout.dtype)
                    readout_error = F.smooth_l1_loss(
                        final_readout,
                        logged_readout,
                        reduction="none",
                    ).mean(dim=(-1, -2))
                    readout_squared = (
                        final_readout - logged_readout
                    ).square().mean(dim=(-1, -2))
                    peer_readout_loss_records.append((valid, readout_error))
                    peer_readout_squared_error_records.append(
                        (valid, readout_squared)
                    )
                else:
                    logged_visual_memory = (
                        None
                        if pre_step_visual_memory is None
                        else pre_step_visual_memory.detach()
                    )
                    try:
                        logged_condition = model.encode_condition(
                            vision_in,
                            batch["proprio"][:, time_index],
                            batch["previous_action"][:, time_index],
                            language_cache=language_cache,
                            visual_memory=logged_visual_memory,
                            env_action=logged_action,
                            detach_wmrm_stage_state=True,
                            **mtvj_kwargs,
                        )
                        logged_auxes = list(
                            getattr(model, "last_wmrm_auxes", None) or ()
                        )
                        logged_pres = list(
                            getattr(model, "last_wmrm_pre_actions", None) or ()
                        )
                        del logged_condition
                    finally:
                        # The independent logged-action branch must not replace the
                        # proposal branch consumed by Flow/handshake regularizers.
                        model.last_wmrm = proposal_last
                        model.last_wmrm_auxes = proposal_auxes
                        model.last_wmrm_pre_actions = proposal_pres
                        model.last_wmrm_meds = proposal_meds
                        model.last_wmrm_pi_kls = proposal_kls
                        model.last_wmrm_pi_kl = proposal_last_kl
                    if len(logged_auxes) != len(proposal_auxes):
                        raise RuntimeError(
                            "logged/proposal World stage counts differ: "
                            f"{len(logged_auxes)} vs {len(proposal_auxes)}"
                        )
                    if len(logged_pres) != len(logged_auxes):
                        raise RuntimeError(
                            "logged World pre-action/stage counts differ: "
                            f"{len(logged_pres)} vs {len(logged_auxes)}"
                        )
                final_visual = None
                logged_visuals = []
                for inject_i, aux in enumerate(logged_auxes):
                    if peer_world_mode:
                        if len(peer_stage_snapshots) != len(proposal_auxes):
                            raise RuntimeError(
                                "peer snapshot/stage counts differ: "
                                f"{len(peer_stage_snapshots)} vs {len(proposal_auxes)}"
                            )
                        snapshot_args, snapshot_kwargs = peer_stage_snapshots[inject_i]
                        snapshot_kwargs["env_action"] = logged_action
                        logged_proposal = model.wmrm.propose(
                            *snapshot_args, **snapshot_kwargs
                        )
                        aux = logged_proposal.aux
                        logged_auxes[inject_i] = aux
                    logged_map = aux.z_tokens
                    if logged_map is None or logged_map.shape != target.shape:
                        raise ValueError(
                            "logged-action World prediction must be the full DINO map: "
                            f"{None if logged_map is None else tuple(logged_map.shape)} "
                            f"vs {tuple(target.shape)}"
                        )
                    visual = visual_world_loss(logged_map, target, current)
                    logged_visuals.append(visual)
                    guard = visual_no_regression_loss(
                        visual,
                        all_copy_ratio=float(WORLD_NO_REGRESSION["all_ratio"]),
                        static_copy_ratio=float(
                            WORLD_STATIC_COPY_CONSTRAINT["static_ratio"]
                        ),
                    )
                    while len(visual_world_stage_records) <= inject_i:
                        visual_world_stage_records.append([])
                        visual_world_objective_stage_records.append([])
                        visual_world_guard_stage_records.append([])
                        visual_world_static_constraint_stage_records.append([])
                    visual_world_stage_records[inject_i].append(
                        (task_ids, valid, visual.loss_per_sample)
                    )
                    visual_world_objective_stage_records[inject_i].append(
                        (
                            task_ids,
                            valid,
                            visual.loss_per_sample
                            + float(WORLD_NO_REGRESSION["weight"])
                            * guard.all_hinge_per_sample,
                        )
                    )
                    visual_world_guard_stage_records[inject_i].append(
                        (task_ids, valid, guard.all_hinge_per_sample)
                    )
                    visual_world_static_constraint_stage_records[inject_i].append(
                        (
                            task_ids,
                            valid,
                            guard.static_hinge_per_sample
                            + static_copy_anchor_loss(
                                logged_map,
                                current,
                                visual.static_mask,
                            ),
                        )
                    )
                    if inject_i == len(logged_auxes) - 1:
                        final_visual = visual
                        visual_world_final_records.append(
                            {
                                "task_ids": task_ids.detach(),
                                "valid": valid.detach(),
                                "world_all": visual.all_per_sample.detach(),
                                "copy_all": visual.copy_all_per_sample.detach(),
                                "world_motion": visual.motion_per_sample.detach(),
                                "copy_motion": visual.copy_motion_per_sample.detach(),
                                "world_top10": visual.top10_per_sample.detach(),
                                "copy_top10": visual.copy_top10_per_sample.detach(),
                                "world_static": visual.static_per_sample.detach(),
                                "copy_static": visual.copy_static_per_sample.detach(),
                                "motion_energy": visual.motion_energy_per_sample.detach(),
                            }
                        )
                if final_visual is None:
                    raise RuntimeError("logged World branch produced no final visual loss")
                if rank_shuffle_actions is None or rank_shuffle_validity is None:
                    raise RuntimeError("visual World action-ranking batch is incomplete")
                shuffle_valid = valid & rank_shuffle_validity[:, time_index]
                if bool(valid.any()):
                    rank_stage = (
                        len(logged_auxes) - 1
                        if world_action_rank_stage == "final"
                        else (world_action_rank_step + time_index)
                        % len(logged_auxes)
                    )
                    rank_aux = logged_auxes[rank_stage]
                    if rank_aux.predict_belief is None:
                        raise RuntimeError(
                            "ranked logged World stage lacks predictor belief"
                        )
                    previous_map = (
                        logged_auxes[rank_stage - 1].z_tokens.detach()
                        if rank_stage > 0
                        and logged_auxes[rank_stage - 1].z_tokens is not None
                        else None
                    )
                    if peer_world_mode:
                        snapshot_args, snapshot_kwargs = peer_stage_snapshots[rank_stage]
                        snapshot_kwargs["env_action"] = rank_shuffle_actions[:, time_index]
                        shuffled_map = model.wmrm.propose(
                            *snapshot_args, **snapshot_kwargs
                        ).aux.z_tokens
                        # Zero is diagnostic-only in the v7 ranking objective, but
                        # peer mode still evaluates it as an explicit env-action
                        # override on the exact same immutable stage snapshot.
                        snapshot_kwargs["env_action"] = torch.zeros_like(logged_action)
                        zero_map = model.wmrm.propose(
                            *snapshot_args, **snapshot_kwargs
                        ).aux.z_tokens
                        if zero_map is None or zero_map.shape != target.shape:
                            raise ValueError(
                                "zero-action World prediction must be the full DINO map"
                            )
                    else:
                        _, _, _, shuffled_map = model.wmrm.predict_world(
                            logged_pres[rank_stage],
                            rank_aux.proprio,
                            rank_aux.predict_belief,
                            rank_aux.task_summary,
                            dino_tokens=rank_aux.dino_tokens,
                            env_action=rank_shuffle_actions[:, time_index],
                            previous_map=previous_map,
                        )
                    if shuffled_map is None or shuffled_map.shape != target.shape:
                        raise ValueError(
                            "action-gap World prediction must be the full DINO map"
                        )
                    real_visual = logged_visuals[rank_stage]
                    shuffled_visual = visual_world_loss(
                        shuffled_map, target, current
                    )
                    for name in (
                        "motion_weights",
                        "topk_mask",
                        "top10_mask",
                        "static_mask",
                    ):
                        if not torch.equal(
                            getattr(shuffled_visual, name), getattr(real_visual, name)
                        ):
                            raise RuntimeError(
                                "action-gap oracle reduction changed under shuffled "
                                f"action: {name}"
                            )
                    ranking = action_top10_oracle_straight_through_gap_loss(
                        real_visual,
                        shuffled_visual,
                        logged_auxes[rank_stage].z_tokens,
                        shuffled_map,
                        target,
                        current,
                        minimum_relative_degradation=float(
                            WORLD_ACTION_RANKING["top10_min_relative_margin"]
                        ),
                    )
                    ranking_loss_per_sample = ranking.loss_per_sample
                    if wmrm_action_rank_per_sample_cap is not None:
                        cap = ranking_loss_per_sample.new_tensor(
                            float(wmrm_action_rank_per_sample_cap)
                        )
                        forward = ranking_loss_per_sample.clamp_max(cap)
                        scale = torch.where(
                            ranking_loss_per_sample.detach() <= cap,
                            torch.ones_like(ranking_loss_per_sample),
                            torch.maximum(
                                ranking_loss_per_sample.new_tensor(0.1),
                                cap
                                / ranking_loss_per_sample.detach().clamp_min(
                                    torch.finfo(ranking_loss_per_sample.dtype).eps
                                ),
                            ),
                        )
                        scaled = ranking_loss_per_sample * scale
                        ranking_loss_per_sample = scaled + (forward - scaled).detach()
                    visual_world_action_shuffle_records.append(
                        (task_ids, shuffle_valid, ranking_loss_per_sample)
                    )
            else:
                # Legacy World supervision remains available for old experiments;
                # visual-motion runs use the separate logged-action branch above.
                for aux in proposal_auxes:
                    pred = aux.z_tokens if aux.z_tokens is not None else aux.z_hat
                    if pred.shape != target.shape:
                        raise ValueError(
                            "world target shape must match prediction: "
                            f"{tuple(target.shape)} vs {tuple(pred.shape)}"
                        )
                    wmrm_world_terms.append(wmrm_world_loss(pred, target))
            if wmrm_adep_enabled:
                for inject_i, aux in enumerate(proposal_auxes):
                    if inject_i >= len(proposal_pres):
                        continue
                    eye_mean = aux.evidence.mean(dim=1)
                    task_id = _world_task_ids(batch, time_index, aux.z_hat.device)
                    donor = proposal_pres[inject_i].clone()
                    cycle = min(model.wmrm.cycle_steps, donor.shape[1])
                    perm = matched_no_fixed_point_perm(task_id, eye_mean, aux.proprio)
                    donor[:, :cycle] = proposal_pres[inject_i][perm][:, :cycle]
                    env = aux.env_action
                    if env is not None:
                        env = env[perm]
                    previous_map = (
                        proposal_auxes[inject_i - 1].z_tokens
                        if inject_i > 0
                        else None
                    )
                    if previous_map is not None:
                        previous_map = previous_map.detach()
                    z_alt, _, _, tok_shuf = model.wmrm.predict_world(
                        donor,
                        aux.proprio,
                        aux.belief,
                        aux.task_summary,
                        dino_tokens=aux.dino_tokens,
                        env_action=env,
                        previous_map=previous_map,
                    )
                    if tok_shuf is None:
                        pred_real, z_shuf = aux.z_hat, z_alt
                    else:
                        pred_real = (
                            aux.z_tokens if aux.z_tokens is not None else aux.z_hat
                        )
                        z_shuf = tok_shuf
                    wmrm_adep_terms.append(
                        model.wmrm.action_dep_hinge(
                            pred_real,
                            z_shuf,
                            target,
                            margin=wmrm_adep_margin,
                        )
                    )
        if proposal_meds:
            wmrm_med_terms.extend(proposal_meds)
        if proposal_kls:
            wmrm_pi_kl_terms.extend(proposal_kls)
        elif proposal_last_kl is not None:
            wmrm_pi_kl_terms.append(proposal_last_kl)
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
            velocity = model.flow_velocity(
                condition,
                noisy_actions[:, time_index],
                flow_time[:, time_index],
                semantic_context=semantic_context,
            )
            if (
                getattr(model, "wam", None) is not None
                and getattr(model, "wam_alpha", 0.0) != 0
            ):
                # α=0 时整段 WAM 代码不执行（不是乘零），避免无意义 forward
                # 与额外噪声来源。
                from va_compound.wam_cache import wam_last_slice_pool
                B = condition.shape[0]
                if dense_evidence is not None and 11 in dense_evidence:
                    h11_t = dense_evidence[11][:, time_index]          # [B,1152,768]
                    spatial16 = wam_last_slice_pool(h11_t)             # [B,16,768]
                else:
                    spatial16 = condition.new_zeros((B, 16, model.config.vision_dim))
                if metric_g is not None:
                    geo8 = metric_g[:, time_index].to(
                        device=condition.device, dtype=condition.dtype
                    )
                    if geo8.shape != (B, 8):
                        raise ValueError(
                            "WAM geo8 必须为当前决策的 p*vis 8 维状态 "
                            f"[B, 8]，got {tuple(geo8.shape)}"
                        )
                else:
                    geo8 = condition.new_zeros((B, 8))
                wam_layers = visual_memory.layers if visual_memory is not None else ()
                if len(wam_layers) == 0:
                    raise ValueError(
                        "WAM 残差通路需要非空 VA 记忆快照；"
                        "encode_condition(..., return_visual_memory=True) 未返回 layers"
                    )
                wam_dv, _ = model.wam(
                    action_condition=condition, va_layers=wam_layers,
                    spatial_tokens=spatial16, geo_tokens=geo8,
                    noisy_actions=noisy_actions[:, time_index],
                    noisy_scene_latents=condition.new_zeros(
                        (B, 3, 16, model.wam.config.vision_dim)
                    ),
                    noisy_scene_geo=condition.new_zeros((B, 3, 2, 8)),
                    flow_time=flow_time[:, time_index],
                )
                velocity = velocity + model.wam_alpha * wam_dv
            if servo is not None:
                # Step 2：双新息伺服（设计 §七 Step 2 / 最小完整算法）——
                # MultiModeReadout 由角色读出路径提供（与 build_local_vision
                # 内部重复一次 reader 前向；按 Agent E 文件契约不改 model.py，
                # Q=6×N keys 开销可忽略）。修正加到 flow 速度输出（直路径 FM
                # 的 v 目标 = a−noise → v 修正等价最终动作空间修正
                # a = clip(a_base + βΔa, −1, 1)）。g_prev 跨决策 detach 维护
                # （ν 的增益缩放仍可微，跨步时序不建图）。
                dense = batch["vision_tokens_st"][:, time_index].to(dtype=target_dtype)
                coords = batch["coords"][0].to(device=batch["vision_tokens"].device)
                readout = model.slot_reader(
                    dense,
                    language_cache.role_queries.to(dtype=target_dtype),
                    coords,
                )
                servo_out = servo(
                    readout,
                    batch["proprio"][:, time_index],
                    lang_cond,
                    a_prev=(
                        batch["previous_action"][:, time_index]
                        if g_prev is not None
                        else None  # 首决策 ν≡0（无 g_prev，无新息信息）
                    ),
                    g_prev=g_prev,
                )
                correction = servo_out.correction.to(dtype=target_dtype)
                g_prev = servo_out.g.detach()
                velocity = velocity + correction[:, None, :]  # [B, H, A]
                if servo_stats is not None:
                    for key, value in (
                        ("stage", servo_out.stage),
                        ("innovation_flag", servo_out.innovation_flag),
                        ("beta", servo_out.beta),
                        ("hyp_entropy", servo_out.hyp_entropy),
                        ("correction", correction),
                    ):
                        servo_stats[key].append(value.detach().cpu())
            predicted_velocities.append(velocity)
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
    if servo_stats is not None and servo_stats.get("stage") is not None:
        for key in ("stage", "innovation_flag", "beta", "hyp_entropy", "correction"):
            servo_stats[key] = torch.stack(servo_stats[key], dim=1)  # [B, T, ...]
    if c2_references is not None:
        return out + (torch.stack(c2_references, dim=1),)
    if visual_world_stage_records:
        stage_weights = stage_supervision_weights(
            len(visual_world_stage_records),
            auxiliary_decay=WORLD_STAGE_AUXILIARY_DECAY,
        )

        def reduce_stage_records(
            records_by_stage: list[list[tuple[Tensor, Tensor, Tensor]]],
        ) -> Tensor:
            values: list[Tensor] = []
            masks: list[Tensor] = []
            weights: list[float] = []
            for stage_weight, records in zip(
                stage_weights, records_by_stage, strict=True
            ):
                for _, valid, value in records:
                    values.append(value)
                    masks.append(valid)
                    weights.append(stage_weight)
            return masked_world_reduction(values, masks, weights)

        base_world_loss = reduce_stage_records(visual_world_stage_records)
        no_regression_loss = reduce_stage_records(
            visual_world_guard_stage_records
        )
        objective_world_loss = reduce_stage_records(
            visual_world_objective_stage_records
        )

        static_constraint_loss = reduce_stage_records(
            visual_world_static_constraint_stage_records
        )

        def reduce_action_records(
            records: list[tuple[Tensor, Tensor, Tensor]],
        ) -> Tensor:
            if not records:
                return objective_world_loss * 0.0
            return masked_world_reduction(
                [value for _, _, value in records],
                [valid for _, valid, _ in records],
            )

        action_shuffle_loss = reduce_action_records(
            visual_world_action_shuffle_records
        )
        action_zero_loss = objective_world_loss * 0.0
        action_strong_loss = objective_world_loss * 0.0
        action_rank_loss = action_shuffle_loss
        if peer_readout_loss_records:
            readout_loss = masked_world_reduction(
                [value for _, value in peer_readout_loss_records],
                [mask for mask, _ in peer_readout_loss_records],
            )
            readout_mse = masked_world_reduction(
                [value for _, value in peer_readout_squared_error_records],
                [mask for mask, _ in peer_readout_squared_error_records],
            )
            readout_rmse = readout_mse.clamp_min(0.0).sqrt()
        else:
            readout_loss = objective_world_loss * 0.0
            readout_rmse = objective_world_loss.detach() * 0.0
        model.last_wmrm_base_loss = base_world_loss
        model.last_world_no_regression_loss = no_regression_loss
        model.last_world_static_constraint_loss = static_constraint_loss
        model.last_world_action_rank_loss = action_rank_loss
        model.last_world_action_shuffle_loss = action_shuffle_loss
        model.last_world_action_zero_loss = action_zero_loss
        model.last_world_action_strong_loss = action_strong_loss
        model.last_world_action_readout_loss = readout_loss
        model.last_world_action_readout_rmse = readout_rmse
        model.last_wmrm_loss = (
            objective_world_loss
            + float(wmrm_static_constraint_weight)
            * static_constraint_loss
            + float(WORLD_ACTION_RANKING["weight"]) * action_rank_loss
            + readout_loss
        )
        model.last_visual_world_metrics = _summarize_visual_world_metrics(
            visual_world_final_records,
            visual_world_stage_records,
        )
    elif wmrm_world_terms:
        model.last_wmrm_loss = torch.stack(wmrm_world_terms).mean()
        model.last_wmrm_base_loss = model.last_wmrm_loss
        model.last_world_no_regression_loss = model.last_wmrm_loss * 0.0
        model.last_world_static_constraint_loss = model.last_wmrm_loss * 0.0
        model.last_world_action_rank_loss = model.last_wmrm_loss * 0.0
        model.last_world_action_shuffle_loss = model.last_wmrm_loss * 0.0
        model.last_world_action_zero_loss = model.last_wmrm_loss * 0.0
        model.last_world_action_strong_loss = model.last_wmrm_loss * 0.0
        model.last_world_action_readout_loss = model.last_wmrm_loss * 0.0
        model.last_world_action_readout_rmse = model.last_wmrm_loss.detach() * 0.0
        model.last_visual_world_metrics = {}
    else:
        model.last_wmrm_loss = None
        model.last_wmrm_base_loss = None
        model.last_world_no_regression_loss = None
        model.last_world_static_constraint_loss = None
        model.last_world_action_rank_loss = None
        model.last_world_action_shuffle_loss = None
        model.last_world_action_zero_loss = None
        model.last_world_action_strong_loss = None
        model.last_world_action_readout_loss = None
        model.last_world_action_readout_rmse = None
        model.last_visual_world_metrics = {}
    if wmrm_pi_kl_terms:
        model.last_wmrm_pi_kl_loss = torch.stack(wmrm_pi_kl_terms).mean()
    else:
        model.last_wmrm_pi_kl_loss = None
    if wmrm_adep_terms:
        model.last_wmrm_adep_loss = torch.stack(wmrm_adep_terms).mean()
    else:
        model.last_wmrm_adep_loss = None
    if wmrm_med_terms:
        model.last_wmrm_med_loss = torch.stack(wmrm_med_terms).mean()
    else:
        model.last_wmrm_med_loss = None
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
        # 逐元素掩码平均（2026-08-08 修复：旧版分母只除 B_valid×n_future，
        # 分子覆盖 B×T×H×C，缺失 T×C=64 倍归一化——潜伏 bug，本次数据无
        # step_mask 未触发，但需保证掩码路径语义正确）。
        m = mask.reshape(-1, 1, 1, 1)  # [B,1,1,1] 按样本有效
        num = F.smooth_l1_loss(
            references[:, :, :n_future] * m,
            clean_batch["step_targets"][:, :, :n_future] * m,
            reduction="sum",
        )
        denom = m.expand(-1, references.shape[1], n_future, references.shape[3]).sum()
        future = num / denom.clamp_min(1.0)
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
        "--live-vjepa",
        action="store_true",
        help="Stage B：--data 路径在线 V-JEPA 编码（帧来自 raw parquet，可对 "
        "V-JEPA 反向传播；输出 288 token，等价于 ST288；需 --single-task "
        "--direct-head）",
    )
    parser.add_argument(
        "--live-root",
        type=Path,
        default=Path(
            "/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
        ),
        help="MetaWorld LeRobot parquet 根目录（--live-vjepa 用）",
    )
    parser.add_argument(
        "--control-stride",
        type=int,
        default=6,
        help="决策点间隔（80 FPS 帧，--live-vjepa 用）：6=13.3Hz（v5 默认），"
        "2=40Hz，1=80Hz。须与数据提取时一致（与 payload 行数对齐）。",
    )
    parser.add_argument(
        "--sequences-per-episode",
        type=int,
        default=4,
        help="每 episode 采样窗口数（--live-vjepa 用）：4=v5 行对齐，"
        "8-16=全轨迹覆盖。须与数据提取时一致（由 payload metadata 校验）。",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=0,
        help="相位完整采样窗口数（--live-vjepa 用）：0=关闭（用 "
        "--sequences-per-episode），6-8=进度分箱 + 强制覆盖末段。"
        "须与数据提取时一致。",
    )
    parser.add_argument(
        "--phase-seed",
        type=int,
        default=0,
        help="相位采样起点扰动种子（--live-vjepa 用）",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="只保留成功 episode（--live-vjepa 用；按 raw next.success 列过滤，"
        "须与数据提取时一致，由 payload metadata 校验）",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="全帧监督（--live-vjepa 用）：窗口起点每 control-stride 帧滑动，"
        "每个决策点都被训练覆盖。须与数据提取时一致，由 payload metadata 校验。",
    )
    parser.add_argument(
        "--frame-aug",
        action="store_true",
        help="训练时帧增强（--live-vjepa 用，π0.5 配方）：RandomCrop 0.95 + "
        "Rotate ±5° + ColorJitter，V-JEPA 编码前逐帧应用，每 epoch 重新随机。"
        "注意：几何增强会扰动 local_slots 的坐标网格对应关系。",
    )
    parser.add_argument(
        "--no-frame-aug-geometric",
        dest="frame_aug_geometric",
        action="store_false",
        default=True,
        help="--frame-aug 开启时仅保留光度增强（ColorJitter），关闭几何增广"
        "（crop/rotate）：几何扰动 ≈ ±1cm 定位噪声且 slot 坐标未同步变换，"
        "精细任务（抓取/插入）下按 E1 修复（2026-08-09 审计 R4）。",
    )
    parser.add_argument(
        "--lr-vision",
        type=float,
        default=3e-6,
        help="V-JEPA 解冻参数的学习率（--live-vjepa 用；默认 3e-6 低 LR 防坍塌）",
    )
    parser.add_argument(
        "--vision-unfreeze-last",
        type=int,
        default=0,
        help="解冻 V-JEPA 最后 N 个 block（0 = 保持冻结；与 --vision-unfreeze-all 互斥）",
    )
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
        "双注意力（sequential 层保持旧共享路径；与 --sequential-coupling=1 同开"
        "时报错——每层都是 sequential 双注意力永不生效；>1 时仅警告）",
    )
    parser.add_argument(
        "--flow-semantic",
        action="store_true",
        help="第二轮架构重构：flow head 逐层读语义上下文（compile readout "
        "tokens，需 --compile-task 才有上下文；flow_cond=adaln 时经 cross-attn 注入）",
    )
    parser.add_argument(
        "--wam-joint",
        action="store_true",
        help="E7 WAM v1：联合世界动作流残差通路（独立 WAM 模块只读推理，"
        "与 --future-predict/--evsm 互斥）",
    )
    parser.add_argument(
        "--wam-alpha",
        type=float,
        default=1.0,
        help="E7 WAM v1：动作残差速度缩放系数（默认 1.0；0 = 关闭 WAM 贡献）",
    )
    parser.add_argument(
        "--wam-ckpt",
        type=str,
        default=None,
        help="E7 WAM v1：WAM 权重来源——独立训练器 checkpoint（含 wam_model 键）"
        "或裸 state_dict",
    )
    parser.add_argument(
        "--wmrm",
        action="store_true",
        help="WAM4VA（原 --wmrm）：世界预测调制 VA 动作流（与 --wam-joint 互斥）",
    )
    parser.add_argument(
        "--wam4va",
        action="store_true",
        dest="wmrm",
        help="WAM4VA：同 --wmrm。WAM 向 VA 注入未来信息，动作仍由 VA 发出",
    )
    parser.add_argument(
        "--va-world-mode",
        choices=("legacy", "peer_sync_h6"),
        default="legacy",
        help=(
            "VA/World topology: legacy preserves sequential writeback; "
            "peer_sync_h6 uses one shared pre-stage snapshot and a deterministic H6 "
            "executable-action readout"
        ),
    )
    parser.add_argument(
        "--wmrm-target",
        choices=("dino", "vjepa", "metric"),
        default="dino",
        help="WAM 下一步监督：dino=下一决策 DINO 投影均值（与 VA 同周期）；"
        "vjepa=下一决策 H11 均值（冻塔白得空间）；metric=旧几何",
    )
    parser.add_argument(
        "--wmrm-world-weight",
        type=float,
        default=1.0,
        help="WMRM 世界预测 MSE 权重（仅 --wmrm；target 为下一决策 metric_g，stop-grad）",
    )
    parser.add_argument(
        "--wmrm-static-constraint-weight",
        type=float,
        default=4.0,
        help="visual World static-copy constraint loss weight (default: 4.0)",
    )
    parser.add_argument(
        "--wmrm-action-rank-per-sample-cap",
        type=float,
        default=None,
        help=(
            "cap each action-ranking sample before masked transition reduction "
            "(default: uncapped)"
        ),
    )
    parser.add_argument(
        "--visual-world-supervision",
        action="store_true",
        help="use visual-motion-aware full-map World supervision on logged actions",
    )
    parser.add_argument(
        "--world-split-manifest",
        type=Path,
        help="immutable episode-level train/eval split manifest for visual World runs",
    )
    parser.add_argument(
        "--world-action-rank-stage",
        choices=("final", "cycle"),
        default="cycle",
        help="v6 shuffled-action gap supervision at the final or rotating WAM stage",
    )
    parser.add_argument(
        "--wmrm-inject",
        choices=("last", "all", "even"),
        default="all",
        help="WAM↔VA 握手：all 每层对传；last 只末端；even 奇数层+末层",
    )
    parser.add_argument(
        "--wmrm-pi-kl-weight",
        type=float,
        default=0.0,
        help="可选：π 对 z_hat 不敏感惩罚（默认 0，联合训练只用 world+flow）",
    )
    parser.add_argument(
        "--wmrm-pi-kl-margin",
        type=float,
        default=0.1,
        help="π shuffle-KL 下限（nat）",
    )
    parser.add_argument(
        "--wmrm-lang-align-weight",
        type=float,
        default=0.0,
        help="可选：belief 与语言摘要对齐（默认 0）",
    )
    parser.add_argument(
        "--wmrm-adep-weight",
        type=float,
        default=0.0,
        help="可选：动作打乱 hinge（默认 0；JEPA 世界已吃日志动作）",
    )
    parser.add_argument(
        "--wmrm-adep-margin",
        type=float,
        default=0.05,
        help="动作打乱后 world MSE 至少应增加的幅度",
    )
    parser.add_argument(
        "--wmrm-med-weight",
        type=float,
        default=0.0,
        help="可选：强迫 ẑ 进入 FM 条件的 hinge（默认 0；握手已写 A'）",
    )
    parser.add_argument(
        "--wmrm-med-margin",
        type=float,
        default=0.05,
        help="shuffle ẑ 后 FM condition 的最小 signed shift",
    )
    parser.add_argument(
        "--wmrm-cycle-steps",
        type=int,
        default=6,
        dest="wmrm_cycle_steps",
        help="WAM 执行前缀步数（须与闭环 --execute-steps 一致）",
    )
    parser.add_argument(
        "--wmrm-detach-proposal-stage-state",
        action="store_true",
        help=(
            "训练 proposal 分支在相邻 WMRM stage map 间 stop-grad；"
            "前向值不变，默认关闭以保留旧反向语义"
        ),
    )
    parser.add_argument(
        "--wmrm-map-size",
        type=int,
        default=16,
        dest="wmrm_map_size",
        help="无参平均池化把 DINO patch 压成 map_size×map_size 再预测（默认 16，对齐 DINO 网格）",
    )
    parser.add_argument(
        "--wmrm-map-channels",
        type=int,
        default=32,
        dest="wmrm_map_channels",
        help="DINO 空间图通道数",
    )
    parser.add_argument(
        "--wmrm-world-grid",
        type=int,
        default=16,
        dest="wmrm_world_grid",
        help="握手空间格子，默认 16=满 DINO 网格，不再 16→4 池化",
    )
    parser.add_argument(
        "--wmrm-predictor",
        choices=("legacy", "st_blocks"),
        default="legacy",
        dest="wmrm_predictor",
        help="世界图预测器：legacy=浅卷积；st_blocks=V-JEPA2 风格时空 Transformer",
    )
    parser.add_argument(
        "--wmrm-predictor-depth",
        type=int,
        default=6,
        dest="wmrm_predictor_depth",
    )
    parser.add_argument(
        "--wmrm-predictor-width",
        type=int,
        default=384,
        dest="wmrm_predictor_width",
    )
    parser.add_argument(
        "--wmrm-predictor-heads",
        type=int,
        default=12,
        dest="wmrm_predictor_heads",
    )
    parser.add_argument(
        "--wmrm-only",
        action="store_true",
        help="第一阶段 JEPA 式：断开握手、世界头不读 VA 的 A，只训下一 latent。"
        "第二阶段去掉本开关 --resume 后再接上握手并训 VA+FM。",
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
    parser.add_argument(
        "--feature-autocast-bf16",
        action="store_true",
        help="Run the feature-policy forward in CUDA BF16 autocast while keeping "
        "model/optimizer parameters and loss reductions in FP32.",
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
        "--va-attention-backend",
        choices=("manual", "auto"),
        default="manual",
        help="VA shared attention implementation: manual materializes FP32 QK scores; "
        "auto uses fused SDPA on the compatible flat/non-dual path and falls back "
        "otherwise.",
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
        "--flow-cond",        choices=("entry", "adaln"),
        default="entry",
        help="flow head conditioning: entry (legacy, add at input) or adaln "
        "(per-layer AdaLN-Zero + cross-attention; 2026-08-07)",
    )
    parser.add_argument(
        "--flow-layers",
        type=int,
        default=2,
        help="flow head transformer layers (π0-style expert 加厚用；"
        "resume 时新层随机初始化，已有层继承)",
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
        "--local-slots-data",
        type=Path,
        default=None,
        help="PULSE-VA Stage A：dense spatiotemporal tokens [N,4,288,768] + coords "
        "[288,3]（scripts/extract_st288_finetuned.py 产出，Stage B 用微调 backbone "
        "重提取；Stage A 为 prepare_mw_local_features.py 同款契约）。开启后视觉流变为 "
        "16 coarse + 6 语言角色槽 + 3 关系 token = 25 tokens；仅 action loss。",
    )
    parser.add_argument(
        "--dense-readout",
        action="store_true",
        help="Step 0（C²-IRF v2 设计 §七）：角色查询直接读出 1152 个 dense patch "
        "token（2×24×24，V-JEPA 不池化）；VA 视觉流仍为 16 coarse（从 1152 "
        "avg-pool）+ 6 角色槽 + 3 关系 token = 25 tokens。需要 --live-vjepa 或 "
        "--local-slots-data（1152-token 密集特征）；与 --local-slots-direct288 "
        "互斥（§九：1152 不进 VA 自注意力）。",
    )
    parser.add_argument(
        "--dense-readout-mtvj",
        action="store_true",
        help="MT-VJ（artifacts/mt_vj_contract.md §5/§6）：持久 dense action "
        "readout——每决策把 4 帧窗在线编码为 {5,11} dense evidence（冻结 "
        "V-JEPA forward_hierarchical_dense，fp16 本地加载）注入 VA 层 K/V "
        "（config.dense_readout_mtvj=True；1152 只做 K/V，query 仅 action "
        "tokens）。需要 --live-vjepa（LiveVJEPADataset 提供原始帧）或无 --data "
        "的合成冒烟（synthetic frames）；与 --dense-readout/--perturb-data/"
        "--e2e-data 互斥。",
    )
    parser.add_argument(
        "--action-vision-backbone",
        choices=("none", *ACTION_VISION_SPECS),
        default="none",
        help="Add a frozen DINO action-only tower while retaining the existing "
        "V-JEPA base/metric/WAM paths. The new per-layer residual is zero-init, "
        "so ordinary migration from an E7 checkpoint starts exactly unchanged.",
    )
    parser.add_argument(
        "--action-vision-checkpoint",
        type=Path,
        default=None,
        help="Local timm-compatible safetensors/pth for the selected action tower. "
        "Required when --action-vision-backbone is not none; never downloaded "
        "implicitly during training.",
    )
    parser.add_argument(
        "--action-vision-encode-batch",
        type=int,
        default=4,
        help="Frozen action-tower image microbatch (4 is the safe ViT-L default "
        "for the 16-GiB training GPU).",
    )
    parser.add_argument(
        "--dino-main-vision",
        action="store_true",
        help="DINO-main replacement（2026-08-14 用户决策）：冻结 DINOv2 替换 "
        "V-JEPA 作为 VA 主视觉骨干，VA/FM/投影从零可训练。V-JEPA/dense/metric "
        "代码保留在仓库中但被禁用（不删除）；要求 LongTrajFramesDataset 原始帧数据。",
    )
    parser.add_argument(
        "--main-vision-checkpoint",
        type=Path,
        default=None,
        help="DINO-main 视觉塔的本地 timm 权重（--dino-main-vision 必填）。",
    )
    parser.add_argument(
        "--main-vision-encode-batch",
        type=int,
        default=16,
        help="DINO-main 冻结塔图像 microbatch（ViT-L 16-GiB GPU 安全默认 16）。",
    )
    parser.add_argument(
        "--main-vision-grid",
        type=int,
        default=8,
        help="DINO-main 每帧 16x16 patch 网格池化到 grid x grid（默认 8 → 64 帧内 token）。",
    )
    parser.add_argument(
        "--main-vision-frames",
        type=int,
        default=4,
        help="DINO-main 每决策消费的窗口帧数（默认 4 = [d-6,d-4,d-2,d]）。",
    )
    parser.add_argument(
        "--main-vision-temporal",
        action="store_true",
        help="为 frame-major DINO patch tokens 加 learned 四帧 slot embedding；"
        "打破旧路径对 [d-6,d-4,d-2,d] 顺序的集合置换不变性。",
    )
    parser.add_argument(
        "--main-vision-temporal-scale",
        type=float,
        default=1.0,
        help="learned frame embedding 的乘法 gate（训练默认 1；0 仅用于因果消融）。",
    )
    parser.add_argument(
        "--dino-feature-cache",
        type=Path,
        default=None,
        help="DINO-main/DINO-metric 预计算特征缓存目录（scripts/"
        "build_dino_feature_cache.py 生成；block11/block23 fp16 memmap；"
        "task35 ROI 精插还要求 exact raw_frames.npy）。"
        "冻结塔在线编码占步时 84%，缓存读把 13000 步从 ~9.4h 降到 ~2.5h；"
        "位级一致性由预计算脚本内置 torch.equal 验证，eval 仍在线编码。",
    )
    parser.add_argument(
        "--dino-dense-metric",
        action="store_true",
        help="DINO-metric（2026-08-15 用户决策）：DINO-main 下接回 MT-VJ dense + "
        "metric 全栈。dense evidence = DINO block11(g)/block23(d) 两帧 [d-2,d] "
        "patch（512 token，1024 维）+ Δt；LanguageMetricField 以 h_dim=1024、"
        "grid=16 从零训练（不复用 V-JEPA metric 权重）。metric head/relation "
        "encoder 可用 --mtvj-train-metric-head/--mtvj-train-relation 以动作 loss "
        "联合微调。",
    )
    parser.add_argument(
        "--metric-geometry-inject",
        action="store_true",
        help="把 metric/ROI 的 8-D p×visibility 经 zero-init Linear 直接加到 "
        "state/action-query 条件；保留旧 2-token route 供消融，但不再只依赖它。",
    )
    parser.add_argument(
        "--dino-roi-checkpoint",
        type=Path,
        default=None,
        help="task35 DINO ROI v2 artifact；策略训练时也从原始帧运行同一 crop "
        "refinement，禁止 eval-only ROI 分布移位。",
    )
    parser.add_argument(
        "--dino-roi-alpha",
        type=float,
        default=None,
        help="task35 DINO ROI 有界残差融合系数 [0,1]。",
    )
    parser.add_argument(
        "--task35-precision-contract",
        action="store_true",
        help="最终 task35 精插实验 fail-fast：要求 matched clean/recovery H6、"
        "grid16 四帧 temporal、DINO MT-VJ、直接 8D geometry、ROI v2、WAM off。",
    )
    parser.add_argument(
        "--action-vision-only",
        action="store_true",
        help="Freeze the existing E7 policy and train only the new action-vision "
        "projector/cross-attention branch. Recommended for the first causal pilot.",
    )
    parser.add_argument(
        "--lr-action-vision",
        type=float,
        default=2e-5,
        help="Learning rate for the additive action-vision branch.",
    )
    parser.add_argument(
        "--metric-visual-checkpoint",
        type=Path,
        default=None,
        help="MT-VJ 阶段 V checkpoint（契约 §2：{metric_head, relation_encoder, "
        "contract='mt_vj_metric_field_v1'}）：加载 LanguageMetricField + "
        "RelationStateEncoder（默认均冻结），每决策产出 metric_tokens "
        "[B, 2, d_model]（g_t = out.p 四角色坐标展平，ν_t = g_t − g_{t−1}）加入 "
        "action cross-attention。需要 --dense-readout-mtvj。",
    )
    parser.add_argument(
        "--replace-mtvj-metric-head-from-external",
        action="store_true",
        help="一次性 clean-FT 迁移：普通 --resume 时用当前 "
        "--metric-visual-checkpoint 的 all-task constructor/weights/"
        "identity 显式替换主 policy 中的旧 metric head，同时严格保留主 policy "
        "的 8D relation encoder。禁止 --resume-exact；默认关闭时仍以主 policy "
        "head 为准。",
    )
    parser.add_argument(
        "--mtvj-roi-checkpoint",
        type=Path,
        default=None,
        help="可选 MT-VJ 原图 ROI 精修 checkpoint（contract="
        "'mt_vj_metric_roi_v1'）。默认关闭；提供时仍冻结 ROI head，并复用同一"
        "冻结 V-JEPA 对每个四帧窗口做第二次原图 crop 编码。",
    )
    parser.add_argument(
        "--mtvj-roi-alpha",
        type=float,
        default=None,
        help="ROI 有界残差融合系数 [0,1]；启用 --mtvj-roi-checkpoint 时必须显式给出。",
    )
    parser.add_argument(
        "--mtvj-train-relation",
        action="store_true",
        help="仅解冻 MT-VJ RelationStateEncoder，让 8 维定位坐标→VA tokens "
        "由动作 loss 联合微调；V-JEPA 与 metric head 仍冻结。",
    )
    parser.add_argument(
        "--lr-mtvj-relation",
        type=float,
        default=2e-5,
        help="MT-VJ relation encoder 的独立学习率（默认 2e-5）。",
    )
    parser.add_argument(
        "--mtvj-train-metric-head",
        action="store_true",
        help="解冻 MT-VJ LanguageMetricField 中连接 out.p/out.visibility→action loss "
        "的定位与可见度路径；rel_mlp 辅助分支与 V-JEPA 仍冻结。",
    )
    parser.add_argument(
        "--lr-mtvj-metric-head",
        type=float,
        default=1e-6,
        help="MT-VJ metric localization path 的独立极小学习率（默认 1e-6）。",
    )
    parser.add_argument(
        "--mtvj-visual-aux-every",
        type=int,
        default=0,
        help="双数据流联合训练（阶段 C，2026-08-12）：每 N 个动作 step 插入一个"
        "在线仿真视觉辅助批次（make_metric_batch 提供精确定位/可见度真值），"
        "辅助 loss = λ_loc·(hinge+pos+offset) + λ_vis·BCE(vis)，只反传 metric "
        "head（V-JEPA no_grad、语言缓存冻结、rel_mlp 无梯度）。0 = 关闭。",
    )
    parser.add_argument(
        "--mtvj-visual-aux-loc-lambda",
        type=float,
        default=1.0,
        help="视觉辅助 loss 定位项（hinge+pos+offset）权重（默认 1.0）。",
    )
    parser.add_argument(
        "--mtvj-visual-aux-vis-lambda",
        type=float,
        default=0.5,
        help="视觉辅助 loss 可见度 BCE 权重（默认 0.5）。",
    )
    parser.add_argument(
        "--mtvj-visual-aux-batch",
        type=int,
        default=8,
        help="视觉辅助批次大小（在线仿真生成；默认 8，与 train_metric_visual 一致）。",
    )
    parser.add_argument(
        "--multi-mode",
        action="store_true",
        help="Step 1（C²-IRF v2 设计 §七 Step 1）：多模式读出——每角色 heatmap "
        "（2 时间片 × grid²）局部 NMS 取 top-2 峰 + 5×5 局部 soft-argmax "
        "（跨 patch 亚像素 μ/Σ，修复全局加权平均的假中点）+ learned NULL 键值"
        "（遮挡时查询选 NULL，vis=1−P(∅)）+ 寻址偏置 b_coord/b_track（γ=0.01）。"
        "视觉流变为 16 coarse + 12 modes + 3 relations = 31 tokens。与 "
        "--dense-readout 兼容（1152 网格；288 网格亦可）；需要 --live-vjepa 或 "
        "--local-slots-data；与 --local-slots-direct288 互斥。",
    )
    parser.add_argument(
        "--servo",
        action="store_true",
        help="Step 2（C²-IRF v2 设计 §七 Step 2）：双新息中央凹交互伺服——显式"
        "关系状态 g（RelationStateProjector，G=16）→ 任务误差 r=g*−g（g* 零初始"
        "化，初始 ≡0 对齐）+ 模型新息 ν（缩放 β、触发重读 flag）+ 低秩有界增益"
        "（κ=κ_max·tanh(ρ)，ρ 零初始化 → 训练起点修正≈0；只称 learned gain）+ "
        "4 假设配对混合（H(w)>τ_H 降 β）。修正加到 flow 速度输出（等价最终动作"
        "修正 a=clip(a_base+βΔa)），损失仍 L_FM。需要 --multi-mode；与 "
        "--direct-head/--c2-controller/--head-only/--scene-teacher 互斥。",
    )
    parser.add_argument(
        "--servo-only",
        action="store_true",
        help="Step 2 第一阶段（设计 §六.3）：冻结 base policy（VA/flow/入口投影），"
        "只训 reader（role_compiler/slot_reader/vis_conditioner）+ relation 投影 + "
        "servo——否则 base path 吸收恢复数据、servo 分支保持关闭。隐含启用 --servo。",
    )
    parser.add_argument(
        "--servo-dls",
        action="store_true",
        help="Step 2 阻尼最小二乘开关（设计 §一.2）：Δa=(KWKᵀ+λI)⁻¹KWr（W=diag(w_g)"
        "由视觉协方差与可见度决定；仅解 4×4 线性系统），替代低秩直乘 K·r。",
    )
    parser.add_argument(
        "--servo-lambda",
        type=float,
        default=1e-2,
        help="--servo-dls 阻尼系数 λ（(KWKᵀ+λI)⁻¹ 中的 λ）",
    )
    parser.add_argument(
        "--servo-rank",
        type=int,
        default=2,
        help="低秩增益 K=κ·U·Vᵀ 的秩 r（设计 §五：rank 2 或 4）",
    )
    parser.add_argument(
        "--lr-servo",
        type=float,
        default=None,
        help="Step 2 伺服模块 LR（默认 = --lr-slot → --lr）",
    )
    parser.add_argument(
        "--perturb-data",
        type=Path,
        default=None,
        help="微扰恢复数据（data/metaworld_perturbations.pt，v5 同构 + "
        "perturb_type/perturb_magnitude 标注；设计 §六.1）：与 --data clean 行按 "
        "--servo-perturb-ratio 混合训练，clean/perturbed 配对行共享同一 (τ,ε)（"
        "动作差异不被 flow noise 淹没）。需要 --single-task；与 --fork-data 互斥。"
        "视觉与路径同构时直接用 payload 特征；否则（--servo 路径）用存储帧冻结 "
        "V-JEPA 在线编码（96×96 放大 384，近似）。",
    )
    parser.add_argument(
        "--servo-perturb-ratio",
        type=float,
        default=0.5,
        help="--perturb-data 每批 perturbed 行占比（(0, 0.5]；配对行共享 (τ,ε)）",
    )
    parser.add_argument(
        "--role-seeds",
        type=Path,
        default=None,
        help="PULSE-VA 角色种子：冻结 Qwen 编码 6 条角色描述的原始嵌入 [6, language_dim]"
        "（make_role_seeds.py 输出）；经 lang_proj 投影后初始化 role_seeds（对称性破缺）。",
    )
    parser.add_argument(
        "--local-slots-direct288",
        action="store_true",
        help="消融格：288 时空 token 直送 VA（无槽/关系），隔离分辨率增益",
    )
    parser.add_argument(
        "--lang-fixed-vector",
        action="store_true",
        help="grounding 对照（Codex 2026-08-08）：语言通道替换为数据集全局均值常量向量，"
        "重训同容量模型——完整模型 vs 固定语言基线的差距即语言条件的因果贡献。"
        "仅 feature 路径（非 live）可用。",
    )
    parser.add_argument(
        "--local-slots-fixed-query",
        action="store_true",
        help="消融格：槽只用固定角色种子（无语言交叉注意），对照语言实例化增益",
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
        choices=("flat", "spatial", "spatiotemporal"),
        default="flat",
        help="vision feature variant: 'flat' (A), 'spatial' (B), or "
        "'spatiotemporal' (ST288/live 288-token；仅影响 training_contract 记录，"
        "供闭环评估对齐在线池化)",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="must be even")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker 数（--live-vjepa 帧解码是 CPU 瓶颈：单线程冷解码 "
        "64 帧 ≈ 1.7s/step 而 GPU 仅忙 ~0.5s；num_workers>=4 与 GPU 重叠解码，"
        "step 时间降至 GPU 上限。0 = 现状主进程串行）",
    )
    parser.add_argument(
        "--task-sampling",
        choices=("uniform", "balanced", "weighted"),
        default="uniform",
        help="难度分层采样（E7 用 weighted，2026-08-09）：按 instruction_id → "
        "MT50 难度权重（easy 0.5/med 1.0/hard 2.0/vh 3.0，scripts/mt50_difficulty.py）"
        "多项式抽样，困难任务过采样、简单任务降采样；balanced = 每个 epoch "
        "严格均衡所有活跃任务；uniform = 按数据行均匀采样（会继承窗口数偏置）",
    )
    parser.add_argument(
        "--task-locality-block-batches",
        type=int,
        default=16,
        help="MT-VJ weighted/balanced sampler 每个同任务块的 batch 数（默认 16；"
        "解码切换成为瓶颈时可调到 32）。",
    )
    parser.add_argument(
        "--fork-data",
        type=Path,
        default=None,
        help="pair 生死门（Codex Q5b）：严格 fork 数据集 .pt（含 pair_id/"
        "instruction_id，2 行/对）。C/D/E 三组都走双 loader（v5 FM 批 + fork 批，"
        "同 --fork-k 交替），仅 --pair-loss-weight 不同：C=0、D=1、"
        "E=1 + 打乱配对数据。fork 批 pair loss 只对真配对生效（flow head 专属，"
        "故 --fork-data 禁止 --direct-head）",
    )
    parser.add_argument(
        "--fork-k",
        type=int,
        default=83,
        help="v5:fork 批交替比（每 k 个 v5 批插 1 个 fork 批）。按自然暴露对齐："
        "k = (9927/B_v)/(N_f/4)；留 12 对后 N_f=120 → k≈83。C/D/E 三组必须同 k",
    )
    parser.add_argument(
        "--fork-skip-contract",
        action="store_true",
        help="E 组（打乱配对）跳过 fork 契约断言——E 的配对不满足同帧约束，"
        "解释限于错误配对压力测试（Q5b③）",
    )
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
        "--flow-prefix-steps",
        type=int,
        default=6,
        help="flow horizon 前缀长度；默认 6，与闭环每次实际执行步数一致。",
    )
    parser.add_argument(
        "--flow-prefix-weight",
        type=float,
        default=1.0,
        help="flow 前缀逐元素 MSE 权重（默认 1.0，保持旧行为）。",
    )
    parser.add_argument(
        "--flow-tail-weight",
        type=float,
        default=1.0,
        help="flow 尾部逐元素 MSE 权重（默认 1.0；H48 的6步前缀若希望约80/20 "
        "总权重，尾部应约0.036，而非0.1）。",
    )
    parser.add_argument(
        "--va-layers",
        type=int,
        default=4,
        help="VACouplingLayer count in the decision stack (depth probe)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-slot",
        type=float,
        default=None,
        help="PULSE-VA：新槽模块 LR（默认 = --lr）；Codex Stage A 建议 1e-4",
    )
    parser.add_argument(
        "--lr-va",
        type=float,
        default=None,
        help="PULSE-VA：共享 VA/头 LR（默认 = --lr）；Codex Stage A 建议 3e-5",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Stage 1 对齐模式：只训练 flow head（VA/槽/V-JEPA 全部冻结）"
        "——随机初始化的动作头噪声梯度不污染已训练的视觉/集成参数；"
        "Stage 2 再去掉本开关全量微调。",
    )
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
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=None,
        help="abort an update when the aggregate gradient norm exceeds this threshold "
        "(default: disabled). Individual finite gradient elements may exceed it. "
        "This argument is bound into the exact-resume contract.",
    )
    parser.add_argument("--seed", type=int, default=0)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        type=Path,
        help="load model weights from a checkpoint; optimizer/sampler/RNG restart",
    )
    resume_group.add_argument(
        "--resume-exact",
        type=Path,
        help="strictly continue model, AdamW, TaskLocality sampler, step, and RNG state",
    )
    resume_group.add_argument(
        "--resume-weights",
        type=Path,
        help=(
            "load model/MT-VJ weights only; optimizer, sampler, RNG and step restart. "
            "Allowed with --task35-precision-contract so a new data/cache SHA can be stamped"
        ),
    )
    parser.add_argument(
        "--resume-exact-contract-migration",
        choices=[
            WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION,
            WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION,
            WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION,
            WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION,
        ],
        help=(
            "allow one named, controlled exact-contract compatibility transition; "
            "the selector is operational and is not saved as a run semantic"
        ),
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
    parser.add_argument(
        "--save-step-copies",
        action="store_true",
        help="also write --save stem_s{step}.pt on each periodic save",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Arg 级契约校验（main 入口调用；纯参数检查，不加载数据/模型）。"""
    finite_positive = {
        "--lr": args.lr,
    }
    for flag, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{flag} must be a positive finite value")
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.steps <= 0 or args.flow_steps <= 0:
        raise ValueError("training steps and flow steps must be positive")
    if not math.isfinite(args.pair_loss_weight) or args.pair_loss_weight < 0.0:
        raise ValueError("pair loss weight must be a non-negative finite value")
    if args.max_gradient_norm is not None and (
        not math.isfinite(args.max_gradient_norm) or args.max_gradient_norm <= 0.0
    ):
        raise ValueError("--max-gradient-norm must be a positive finite value")
    cap = args.wmrm_action_rank_per_sample_cap
    if cap is not None and (
        not math.isfinite(cap) or cap <= 0.0
    ):
        raise ValueError(
            "--wmrm-action-rank-per-sample-cap must be a positive finite value"
        )
    if cap is not None and not getattr(args, "visual_world_supervision", False):
        raise ValueError(
            "--wmrm-action-rank-per-sample-cap only applies with "
            "--visual-world-supervision"
        )
    if args.resume_exact_contract_migration is not None and args.resume_exact is None:
        raise ValueError(
            "--resume-exact-contract-migration requires --resume-exact"
        )
    peer_world = getattr(args, "va_world_mode", "legacy") == "peer_sync_h6"
    if peer_world:
        required = {
            "--wam4va": bool(getattr(args, "wmrm", False)),
            "--wmrm-cycle-steps 6": int(getattr(args, "wmrm_cycle_steps", 0)) == 6,
            "--flow-prefix-steps 6": int(getattr(args, "flow_prefix_steps", 0)) == 6,
            "--wmrm-inject all": getattr(args, "wmrm_inject", None) == "all",
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "--va-world-mode peer_sync_h6 missing required settings: "
                + ", ".join(missing)
            )
        if args.resume is not None or getattr(args, "resume_weights", None) is not None:
            raise ValueError(
                "peer_sync_h6 training may only start scratch or use --resume-exact"
            )
        if getattr(args, "wmrm_only", False):
            raise ValueError(
                "--wmrm-only with peer_sync_h6 has unsupported readout ownership; "
                "use legacy or train the full peer policy"
            )
        if float(getattr(args, "wmrm_adep_weight", 0.0)) != 0.0:
            raise ValueError(
                "peer_sync_h6 requires --wmrm-adep-weight 0 until "
                "same-snapshot action-dependence counterfactuals are implemented"
            )
        if getattr(args, "resume_exact", None) is not None:
            peer_checkpoint = torch.load(
                args.resume_exact, map_location="cpu", weights_only=True
            )
            saved_mode = (peer_checkpoint.get("config") or {}).get(
                "va_world_mode", "legacy"
            )
            if saved_mode != "peer_sync_h6":
                raise ValueError(
                    "--resume-exact peer_sync_h6 requires a peer_sync_h6 checkpoint; "
                    f"checkpoint mode is {saved_mode!r}"
                )
    visual_world = bool(getattr(args, "visual_world_supervision", False))
    split_manifest = getattr(args, "world_split_manifest", None)
    if visual_world:
        required = {
            "--data": args.data is not None,
            "--world-split-manifest": split_manifest is not None,
            "--wam4va": bool(getattr(args, "wmrm", False)),
            "--wmrm-target dino": getattr(args, "wmrm_target", None) == "dino",
            "--wmrm-cycle-steps 6": int(getattr(args, "wmrm_cycle_steps", 0)) == 6,
            "--wmrm-inject all": getattr(args, "wmrm_inject", None) == "all",
            "WAM4VA handshake enabled": bool(
                getattr(args, "wmrm_handshake", True)
            ),
            "--va-layers 8": int(getattr(args, "va_layers", 0)) == 8,
            "--wmrm-predictor st_blocks": getattr(args, "wmrm_predictor", None)
            == "st_blocks",
            "--wmrm-predictor-depth 6": int(
                getattr(args, "wmrm_predictor_depth", 0)
            )
            == 6,
            "--wmrm-predictor-width 384": int(
                getattr(args, "wmrm_predictor_width", 0)
            )
            == 384,
            "--wmrm-predictor-heads 12": int(
                getattr(args, "wmrm_predictor_heads", 0)
            )
            == 12,
            "--wmrm-map-size 16": int(getattr(args, "wmrm_map_size", 0)) == 16,
            "--wmrm-map-channels 1024": int(
                getattr(args, "wmrm_map_channels", 0)
            )
            == 1024,
            "--wmrm-world-grid 16": int(getattr(args, "wmrm_world_grid", 0))
            == 16,
            "--dino-main-vision": bool(getattr(args, "dino_main_vision", False)),
            "--main-vision-grid 16": int(getattr(args, "main_vision_grid", 0)) == 16,
            "--main-vision-frames 4": int(getattr(args, "main_vision_frames", 0))
            == 4,
            "--sequence-length 4": int(getattr(args, "sequence_length", 0)) == 4,
            "--min-sequence-length 4": int(
                getattr(args, "min_sequence_length", 0)
            )
            == 4,
            "--single-task joint sampler": bool(args.single_task),
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "--visual-world-supervision missing required settings: "
                + ", ".join(missing)
            )
        if args.resume is not None or getattr(args, "resume_weights", None) is not None:
            raise ValueError(
                "visual World training may only start scratch or use --resume-exact"
            )
    elif split_manifest is not None:
        raise ValueError(
            "--world-split-manifest requires --visual-world-supervision"
        )
    flow_prefix_steps = getattr(args, "flow_prefix_steps", 6)
    flow_prefix_weight = getattr(args, "flow_prefix_weight", 1.0)
    flow_tail_weight = getattr(args, "flow_tail_weight", 1.0)
    if flow_prefix_steps <= 0:
        raise ValueError("--flow-prefix-steps must be positive")
    if flow_prefix_weight < 0.0 or flow_tail_weight < 0.0:
        raise ValueError("--flow-prefix-weight/--flow-tail-weight must be non-negative")
    if flow_prefix_weight == 0.0 and flow_tail_weight == 0.0:
        raise ValueError("flow prefix and tail weights cannot both be zero")
    replace_metric_head = getattr(
        args, "replace_mtvj_metric_head_from_external", False
    )
    action_vision = getattr(args, "action_vision_backbone", "none")
    action_vision_only = getattr(args, "action_vision_only", False)
    action_vision_checkpoint = getattr(args, "action_vision_checkpoint", None)
    if action_vision != "none":
        if action_vision not in ACTION_VISION_SPECS:
            raise ValueError(f"unsupported action vision backbone: {action_vision!r}")
        if not args.dense_readout_mtvj:
            raise ValueError(
                "--action-vision-backbone requires --dense-readout-mtvj "
                "(V-JEPA remains the metric/WAM anchor)"
            )
        if action_vision_checkpoint is None:
            raise ValueError(
                "--action-vision-backbone requires --action-vision-checkpoint"
            )
        if not action_vision_checkpoint.expanduser().is_file():
            raise ValueError(
                f"action vision checkpoint does not exist: {action_vision_checkpoint}"
            )
        if getattr(args, "action_vision_encode_batch", 0) < 1:
            raise ValueError("--action-vision-encode-batch must be positive")
        if getattr(args, "lr_action_vision", 0.0) <= 0.0:
            raise ValueError("--lr-action-vision must be positive")
        if getattr(args, "resume_exact", None) is not None:
            # Exact continuation is valid only after a DINO checkpoint exists;
            # an old E7 checkpoint needs the ordinary additive migration path.
            resume_payload = torch.load(
                args.resume_exact, map_location="cpu", weights_only=True
            )
            saved_backbone = (resume_payload.get("config") or {}).get(
                "action_vision_backbone", "none"
            )
            if saved_backbone != action_vision:
                raise ValueError(
                    "first action-vision migration requires ordinary --resume; "
                    f"checkpoint has {saved_backbone!r}, requested {action_vision!r}"
                )
    elif action_vision_checkpoint is not None or action_vision_only:
        raise ValueError(
            "--action-vision-checkpoint/--action-vision-only requires a non-none "
            "--action-vision-backbone"
        )
    if action_vision_only and (
        getattr(args, "head_only", False)
        or getattr(args, "servo_only", False)
        or getattr(args, "c2_controller", False)
        or getattr(args, "mtvj_train_relation", False)
        or getattr(args, "mtvj_train_metric_head", False)
    ):
        raise ValueError(
            "--action-vision-only freezes the base policy/metric path and is "
            "incompatible with head/servo/C2 or MT-VJ joint-training flags"
        )
    # DINO-main replacement（2026-08-14 用户决策）：V-JEPA/dense/metric 全部
    # 保留在代码中（flag 关闭即禁用，不删除），此处只校验组合合法性。
    dino_main_vision = bool(getattr(args, "dino_main_vision", False))
    if dino_main_vision:
        if not args.data:
            raise ValueError("--dino-main-vision requires --data (LongTrajFramesDataset)")
        if args.dense_readout_mtvj:
            raise ValueError(
                "--dino-main-vision replaces V-JEPA: --dense-readout-mtvj "
                "(V-JEPA dense/metric path) must stay off"
            )
        if action_vision != "none":
            raise ValueError(
                "--dino-main-vision replaces the auxiliary --action-vision-backbone "
                "branch; set --action-vision-backbone none"
            )
        if action_vision_only:
            raise ValueError("--dino-main-vision is incompatible with --action-vision-only")
        if (
            args.live_vjepa
            or args.local_slots_data
            or args.scene_teacher
            or args.plan_resampler
            or args.servo
        ):
            raise ValueError(
                "--dino-main-vision is a minimal VA+flow experiment; "
                "V-JEPA/MT-VJ/slot/servo/plan paths must stay off"
            )
        if args.metric_visual_checkpoint is not None:
            raise ValueError(
                "--dino-main-vision: --metric-visual-checkpoint (MT-VJ metric) "
                "must stay off"
            )
        if args.main_vision_checkpoint is None:
            raise ValueError("--dino-main-vision requires --main-vision-checkpoint")
        if not args.main_vision_checkpoint.expanduser().is_file():
            raise FileNotFoundError(
                f"main vision checkpoint does not exist: {args.main_vision_checkpoint}"
            )
        if args.main_vision_encode_batch < 1:
            raise ValueError("--main-vision-encode-batch must be positive")
        # DINO 视觉辅助 loss（2026-08-16 已移植）：--mtvj-visual-aux-every 在
        # --dino-dense-metric 下走 _dino_visual_aux_loss（仿真真值 → DINO dense
        # evidence → grid-16 metric head），不再报错。
        if getattr(args, "dino_dense_metric", False) and getattr(
            args, "mtvj_visual_aux_every", 0
        ) > 0:
            if not getattr(args, "mtvj_train_metric_head", False):
                raise ValueError(
                    "--dino-dense-metric + --mtvj-visual-aux-every 要求 "
                    "--mtvj-train-metric-head（辅助 loss 反传目标是 metric head）"
                )
        if args.dino_feature_cache is not None:
            cache_dir = args.dino_feature_cache.expanduser()
            if not cache_dir.is_dir():
                raise ValueError(
                    f"--dino-feature-cache directory missing: {cache_dir}"
                )
            for name in ("meta.json", "index.pkl", "block23.npy", "block11.npy"):
                if not (cache_dir / name).exists():
                    raise ValueError(
                        f"--dino-feature-cache 缺少 {name}: {cache_dir}"
                    )
    elif getattr(args, "main_vision_checkpoint", None) is not None:
        raise ValueError("--main-vision-checkpoint requires --dino-main-vision")
    if getattr(args, "dino_dense_metric", False) and not dino_main_vision:
        raise ValueError("--dino-dense-metric requires --dino-main-vision")
    if getattr(args, "dino_feature_cache", None) is not None and not dino_main_vision:
        raise ValueError("--dino-feature-cache requires --dino-main-vision")
    if getattr(args, "main_vision_temporal", False) and not dino_main_vision:
        raise ValueError("--main-vision-temporal requires --dino-main-vision")
    if not math.isfinite(float(getattr(args, "main_vision_temporal_scale", 1.0))):
        raise ValueError("--main-vision-temporal-scale must be finite")
    if getattr(args, "metric_geometry_inject", False) and not getattr(
        args, "dino_dense_metric", False
    ):
        raise ValueError("--metric-geometry-inject requires --dino-dense-metric")
    dino_roi_checkpoint = getattr(args, "dino_roi_checkpoint", None)
    dino_roi_alpha = getattr(args, "dino_roi_alpha", None)
    if dino_roi_checkpoint is None:
        if dino_roi_alpha is not None:
            raise ValueError("--dino-roi-alpha requires --dino-roi-checkpoint")
    else:
        if not getattr(args, "dino_dense_metric", False):
            raise ValueError("--dino-roi-checkpoint requires --dino-dense-metric")
        if not dino_roi_checkpoint.expanduser().is_file():
            raise FileNotFoundError(
                f"DINO ROI checkpoint does not exist: {dino_roi_checkpoint}"
            )
        if dino_roi_alpha is None or not math.isfinite(dino_roi_alpha) or not 0.0 <= dino_roi_alpha <= 1.0:
            raise ValueError(
                "--dino-roi-checkpoint requires finite --dino-roi-alpha in [0,1]"
            )
    if getattr(args, "task35_precision_contract", False):
        required = {
            "--data": args.data is not None,
            "--single-task": args.single_task,
            "--dino-main-vision": dino_main_vision,
            "--dino-dense-metric": getattr(args, "dino_dense_metric", False),
            "--main-vision-grid 16": int(getattr(args, "main_vision_grid", 0)) == 16,
            "--main-vision-frames 4": int(getattr(args, "main_vision_frames", 0)) == 4,
            "--main-vision-temporal": getattr(args, "main_vision_temporal", False),
            "--metric-geometry-inject": getattr(args, "metric_geometry_inject", False),
            "--dino-feature-cache": getattr(args, "dino_feature_cache", None)
            is not None,
            "--dino-roi-checkpoint": dino_roi_checkpoint is not None,
            "--dino-roi-alpha 1": dino_roi_alpha is not None
            and float(dino_roi_alpha) == 1.0,
            "--mtvj-train-metric-head": getattr(args, "mtvj_train_metric_head", False),
            "--mtvj-train-relation": getattr(args, "mtvj_train_relation", False),
            "--mtvj-visual-aux-every > 0": args.mtvj_visual_aux_every > 0,
            "--mtvj-visual-aux-batch > 0": args.mtvj_visual_aux_batch > 0,
            "--va-attention-backend auto": args.va_attention_backend == "auto",
            "--task-sampling weighted": args.task_sampling == "weighted",
            "--num-workers 0": args.num_workers == 0,
            "no ordinary resume": args.resume is None,
            "WAM off": not getattr(args, "wam_joint", False),
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                "--task35-precision-contract missing required settings: "
                + ", ".join(missing)
            )
    roi_checkpoint = getattr(args, "mtvj_roi_checkpoint", None)
    roi_alpha = getattr(args, "mtvj_roi_alpha", None)
    if roi_checkpoint is None and roi_alpha is not None:
        raise ValueError("--mtvj-roi-alpha requires --mtvj-roi-checkpoint")
    if roi_checkpoint is not None:
        if not args.dense_readout_mtvj or args.metric_visual_checkpoint is None:
            raise ValueError(
                "--mtvj-roi-checkpoint requires --dense-readout-mtvj and "
                "--metric-visual-checkpoint"
            )
        if getattr(args, "mtvj_train_metric_head", False):
            raise ValueError(
                "--mtvj-roi-checkpoint forbids --mtvj-train-metric-head: ROI is "
                "bound to a fixed coarse checkpoint and may only run while the "
                "coarse metric head is frozen"
            )
        if roi_alpha is None or not math.isfinite(roi_alpha) or not 0.0 <= roi_alpha <= 1.0:
            raise ValueError(
                "--mtvj-roi-checkpoint requires finite --mtvj-roi-alpha in [0,1]"
            )
    if replace_metric_head:
        if getattr(args, "resume_exact", None) is not None:
            raise ValueError(
                "--replace-mtvj-metric-head-from-external 禁止与 "
                "--resume-exact 同时使用；这是一次性普通 --resume 迁移"
            )
        if args.resume is None:
            raise ValueError(
                "--replace-mtvj-metric-head-from-external requires ordinary --resume"
            )
        if args.metric_visual_checkpoint is None or not args.dense_readout_mtvj:
            raise ValueError(
                "--replace-mtvj-metric-head-from-external requires "
                "--dense-readout-mtvj and --metric-visual-checkpoint"
            )
        if getattr(args, "mtvj_train_metric_head", False):
            raise ValueError(
                "clean-FT migration keeps the replacement metric head frozen；"
                "不要同时使用 --mtvj-train-metric-head"
            )
        if not getattr(args, "mtvj_train_relation", False):
            raise ValueError(
                "visibility-gated metric-state migration changes the 8D relation "
                "semantics；必须同时启用 --mtvj-train-relation 让旧 projection "
                "作为 warm start 适配"
            )
    if not args.single_task and (args.batch_size < 2 or args.batch_size % 2):
        raise ValueError("paired batch size must be even")
    if args.single_task and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if getattr(args, "task_locality_block_batches", 16) <= 0:
        raise ValueError("--task-locality-block-batches must be positive")
    if getattr(args, "resume_exact", None) is not None:
        if args.num_workers != 0:
            raise ValueError("--resume-exact requires --num-workers 0 (worker RNG is not checkpointed)")
        if not (
            args.data is not None
            and args.single_task
            and args.task_sampling in {"weighted", "balanced"}
            and (
                args.dense_readout_mtvj
                or getattr(args, "dino_main_vision", False)
            )
        ):
            raise ValueError(
                "--resume-exact currently requires the single-task weighted/balanced "
                "MT-VJ or DINO-main data path (TaskLocalityWeightedSampler or "
                "TaskWeightedSampler)"
            )
        if args.fork_data is not None or args.perturb_data is not None or args.c2_controller:
            raise ValueError(
                "--resume-exact does not support fork/perturb/C2 auxiliary loaders"
            )
    if args.evsm and not (args.memory_split and args.future_predict):
        raise ValueError("--evsm requires --memory-split and --future-predict")
    if args.evsm and args.future_predict_weight <= 0.0:
        raise ValueError("--evsm requires a positive --future-predict-weight")
    if args.evsm and args.evsm_temp <= 0.0:
        raise ValueError("--evsm-temp must be positive")
    if getattr(args, "wam_alpha", 1.0) == 0.0:
        # α=0 语义 = WAM 完全关闭：不构造、不恢复、不进契约，保证旧路径
        # 全局 bit-identical（含 RNG 消费）。必须赶在所有 wam 校验之前。
        args.wam_joint = False
    if getattr(args, "wmrm", False) and getattr(args, "wam_joint", False):
        raise ValueError("--wmrm is mutually exclusive with --wam-joint")
    if getattr(args, "wmrm", False) and args.memory_split:
        raise ValueError("--wmrm is mutually exclusive with --memory-split")
    if getattr(args, "wmrm", False) and (args.direct_head or args.c2_controller):
        raise ValueError("--wmrm/--wam4va is mutually exclusive with --direct-head/--c2-controller")
    if getattr(args, "wmrm_only", False) and not getattr(args, "wmrm", False):
        raise ValueError("--wmrm-only requires --wmrm/--wam4va")
    if getattr(args, "wmrm_only", False) and (
        args.head_only
        or args.servo_only
        or getattr(args, "action_vision_only", False)
    ):
        raise ValueError("--wmrm-only is mutually exclusive with --head-only/--servo-only/--action-vision-only")
    if getattr(args, "wmrm_only", False):
        # Stage-1 JEPA-style: no handshake, no VA action leak, only L_world.
        args.wmrm_handshake = False
        args.wmrm_med_weight = 0.0
        args.wmrm_adep_weight = 0.0
        args.wmrm_pi_kl_weight = 0.0
        args.mtvj_train_metric_head = False
        args.mtvj_train_relation = False
    if getattr(args, "wmrm", False) and float(getattr(args, "wmrm_world_weight", 1.0)) <= 0.0:
        raise ValueError("--wmrm requires positive --wmrm-world-weight")
    if getattr(args, "wmrm_world_weight", 1.0) < 0.0:
        raise ValueError("--wmrm-world-weight must be non-negative")
    static_constraint_weight = float(
        getattr(args, "wmrm_static_constraint_weight", 4.0)
    )
    if not math.isfinite(static_constraint_weight) or static_constraint_weight < 0.0:
        raise ValueError("--wmrm-static-constraint-weight must be finite and non-negative")
    if static_constraint_weight != 4.0 and not getattr(
        args, "visual_world_supervision", False
    ):
        raise ValueError(
            "--wmrm-static-constraint-weight only applies with --visual-world-supervision"
        )
    if int(getattr(args, "wmrm_cycle_steps", 6)) < 1:
        raise ValueError("--wmrm-cycle-steps must be >= 1")
    if getattr(args, "wmrm", False) and getattr(args, "wmrm_target", "dino") == "vjepa":
        dino_main = bool(getattr(args, "dino_main_vision", False))
        backbone = getattr(args, "main_vision_backbone", "vjepa")
        if dino_main or (backbone is not None and backbone != "vjepa"):
            raise ValueError(
                "wmrm_target=vjepa is incompatible with DINO main vision "
                "(dino_main_vision / main_vision_backbone != vjepa); "
                "do not infer V-JEPA from dense key 11"
            )
    if getattr(args, "wam_joint", False) and (args.future_predict or args.evsm):
        raise ValueError("--wam-joint is mutually exclusive with --future-predict/--evsm")
    if getattr(args, "wam_joint", False) and args.memory_split:
        raise ValueError(
            "--wam-joint is mutually exclusive with --memory-split "
            "（WAM CA 读 VA 层快照 layers，memory_split 会把 layers 置空）"
        )
    if getattr(args, "wam_joint", False) and (args.direct_head or args.c2_controller):
        raise ValueError(
            "--wam-joint 与 --direct-head/--c2-controller 互斥"
            "（WAM 残差只作用于 flow Euler 速度路径）"
        )
    if getattr(args, "wam_joint", False) and (
        not args.dense_readout_mtvj or args.metric_visual_checkpoint is None
    ):
        raise ValueError(
            "--wam-joint requires --dense-readout-mtvj and "
            "--metric-visual-checkpoint（WAM spatial16/geo8 的来源）"
        )
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
    if args.fork_data:
        # pair 生死门约束（Codex Q5b）：flow head 专属（pair loss 在 direct 模式跳过）、
        # 预计算特征路径（非 live）、单任务 v5 主 loader + fork 双 loader。
        if args.direct_head:
            raise ValueError("--fork-data requires the flow head (--direct-head 下 pair loss 被跳过)")
        if args.live_vjepa:
            raise ValueError("--fork-data is not supported with --live-vjepa (预计算特征路径)")
        if not args.single_task:
            raise ValueError("--fork-data requires --single-task (v5 主 loader FM-only)")
        if not args.data:
            raise ValueError("--fork-data requires --data (v5 特征数据集)")
        if args.fork_k < 1:
            raise ValueError("--fork-k must be >= 1")
        if args.c2_controller:
            raise ValueError("--fork-data is mutually exclusive with --c2-controller")
    if args.c2_controller and not args.direct_head:
        raise ValueError("--c2-controller requires --direct-head")
    if args.c2_controller and not args.single_task:
        raise ValueError("--c2-controller requires --single-task (v6a/v6b 为按钮任务单任务数据)")
    if args.c2_controller and not args.data:
        raise ValueError("--c2-controller requires --data (precomputed features)")
    if args.live_vjepa:
        if not args.data:
            raise ValueError("--live-vjepa requires --data (v5 paired .pt 与 parquet 行对齐)")
        if not args.single_task:
            # Stage B 限定 single-task：配对帧级契约留待数据侧（flow 在
            # single-task 下退化为 FM-only，可正常训练）。
            raise ValueError("--live-vjepa requires --single-task (Stage B 限定)")
        # flow 模式合法（π0 式 flow matching action expert；pair 在 single-task
        # 下退化为 FM-only，direct_head 不再强制）。
        if args.local_slots_data:
            raise ValueError("--live-vjepa is mutually exclusive with --local-slots-data")
        if args.c2_controller:
            raise ValueError("--live-vjepa is mutually exclusive with --c2-controller")
        if args.scene_teacher:
            raise ValueError("--live-vjepa is mutually exclusive with --scene-teacher")
        if args.sam_rho > 0:
            raise ValueError(
                "--live-vjepa forbids --sam-rho（SAM 二次前向复用已释放的视觉计算图，"
                "且扰动后骨干未重新编码，Codex P1-2）"
            )
        if args.vision_unfreeze_all and args.vision_unfreeze_last:
            raise ValueError(
                "--vision-unfreeze-all and --vision-unfreeze-last are mutually exclusive"
            )
        if args.batch_size * 4 * 4 > 128:
            print(
                "warn: --live-vjepa batch*T*W > 128 帧（V-JEPA 解冻反向显存大，"
                "16GB 卡建议 batch-size <= 8）",
                flush=True,
            )
    if args.dense_readout:
        # Step 0 dense readout：只允许 local_slots 读出路径（live 在线或
        # 预计算 1152-token 特征），角色查询 cross-attn 压缩到 25 tokens 后
        # 才进 VA——1152 直送 VA 会在 VA 内做 1152×1152 自注意力（§九 禁止）。
        if not (args.live_vjepa or args.local_slots_data):
            raise ValueError(
                "--dense-readout requires --live-vjepa or --local-slots-data "
                "（1152 patch 只在 local_slots 读出路径可用）"
            )
        if args.local_slots_direct288:
            raise ValueError(
                "--dense-readout 与 --local-slots-direct288 互斥：1152 token 直送 "
                "VA 会在 VA 内做 1152×1152 自注意力（设计文档 §九 明确禁止）"
            )
    if args.multi_mode:
        # Step 1 多模式读出：只允许 local_slots 读出路径（与 dense_readout 同约束）；
        # 与 --dense-readout 兼容（1152 网格），288 网格亦可（12×12 消融）。
        if not (args.live_vjepa or args.local_slots_data):
            raise ValueError(
                "--multi-mode requires --live-vjepa or --local-slots-data "
                "（角色查询读出路径）"
            )
        if args.local_slots_direct288:
            raise ValueError(
                "--multi-mode 与 --local-slots-direct288 互斥：direct288 无角色读出路径"
            )
    if (
        getattr(args, "dense_readout_mtvj", False)
        or getattr(args, "metric_visual_checkpoint", None) is not None
        or getattr(args, "mtvj_train_relation", False)
        or getattr(args, "mtvj_train_metric_head", False)
        or replace_metric_head
    ):
        # MT-VJ（契约 §6）：在线 dense 编码需要原始帧（live 数据集或合成冒烟）。
        if args.e2e_data:
            raise ValueError(
                "--dense-readout-mtvj/--metric-visual-checkpoint 不支持 --e2e-data"
                "（MT-VJ 在线 dense 编码仅 live/合成冒烟路径）"
            )
        if args.data is not None and not args.live_vjepa:
            # 2026-08-10：--dense-readout-mtvj + --data 走 LongTrajFramesDataset
            # （windows_h48.pt + longtraj JPEG 帧在线解码，MT-VJ 主数据路径）。
            print(
                "--dense-readout-mtvj: --data 为 longtraj windows 文件，"
                "帧由 LongTrajFramesDataset 从 JPEG 在线解码",
                flush=True,
            )
        if args.metric_visual_checkpoint is not None and not args.dense_readout_mtvj:
            raise ValueError(
                "--metric-visual-checkpoint requires --dense-readout-mtvj"
                "（metric_tokens 由 dense action readout 消费）"
            )
        train_relation = getattr(args, "mtvj_train_relation", False)
        train_metric_head = getattr(args, "mtvj_train_metric_head", False)
        train_mtvj_action_path = train_relation or train_metric_head
        dino_joint = dino_main_vision and getattr(args, "dino_dense_metric", False)
        if train_mtvj_action_path:
            if not (
                (args.dense_readout_mtvj and args.metric_visual_checkpoint is not None)
                or dino_joint
            ):
                raise ValueError(
                    "MT-VJ joint training requires --dense-readout-mtvj and "
                    "--metric-visual-checkpoint（或 --dino-main-vision "
                    "--dino-dense-metric 从零 metric 栈）"
                )
            if args.resume is None and getattr(args, "resume_exact", None) is None and not dino_joint:
                # DINO-metric 例外：metric 栈从零构建是设计本身（_build_dino_
                # metric_stack），无需 V-JEPA 的 warm-start 语义。
                raise ValueError(
                    "MT-VJ joint training requires --resume or --resume-exact"
                    "（--dino-dense-metric 从零 metric 栈除外）"
                )
            if args.sam_rho > 0.0:
                raise ValueError(
                    "MT-VJ joint training forbids --sam-rho：SAM 二次反向需要"
                    "重新计算 metric/relation tokens"
                )
            if args.head_only or args.servo_only or args.c2_controller:
                raise ValueError(
                    "MT-VJ joint training 与 --head-only/--servo-only/"
                    "--c2-controller 的冻结语义冲突"
                )
        if train_relation:
            if getattr(args, "lr_mtvj_relation", 2e-5) <= 0.0:
                raise ValueError("--lr-mtvj-relation must be positive")
        if train_metric_head and getattr(args, "lr_mtvj_metric_head", 1e-6) <= 0.0:
            raise ValueError("--lr-mtvj-metric-head must be positive")
        if getattr(args, "mtvj_visual_aux_every", 0) > 0:
            if not (args.dense_readout_mtvj or dino_joint):
                raise ValueError(
                    "--mtvj-visual-aux-every requires --dense-readout-mtvj"
                    "（或 --dino-dense-metric 的 DINO 版辅助 loss）"
                )
            if args.metric_visual_checkpoint is None and not dino_joint:
                raise ValueError(
                    "--mtvj-visual-aux-every requires --metric-visual-checkpoint"
                    "（--dino-dense-metric 从零 metric 栈除外）"
                )
            if not train_metric_head:
                raise ValueError(
                    "--mtvj-visual-aux-every 要求解冻视觉头（--mtvj-train-metric-head）："
                    "辅助 loss 的反传目标是 metric head，冻结时无梯度可学"
                )
            if not args.single_task:
                raise ValueError(
                    "--mtvj-visual-aux-every requires --single-task "
                    "（辅助语言缓存与任务局部性采样只在该路径定义）"
                )
            if not train_relation:
                raise ValueError(
                    "--mtvj-visual-aux-every requires --mtvj-train-relation："
                    "视觉坐标漂移时 relation bridge 必须同步适配"
                )
            if args.task_sampling not in {"weighted", "balanced"}:
                raise ValueError(
                    "--mtvj-visual-aux-every requires --task-sampling "
                    "weighted|balanced（辅助任务需要显式任务权重）"
                )
            if args.sam_rho > 0.0:
                raise ValueError(
                    "--mtvj-visual-aux-every forbids --sam-rho（辅助 loss 的"
                    "二次前向语义未定义）"
                )
        if args.dense_readout:
            raise ValueError(
                "--dense-readout-mtvj 与 --dense-readout 互斥（Step 0 1152 槽读出 vs "
                "MT-VJ dense evidence 注入，两套 dense 机制语义重叠）"
            )
        if args.perturb_data is not None:
            raise ValueError(
                "--dense-readout-mtvj 与 --perturb-data 互斥（混批行不对齐 "
                "dense evidence/metric_tokens）"
            )
    if getattr(args, "servo_only", False):
        args.servo = True  # --servo-only 隐含启用 --servo（第一阶段冻结模式）
    if getattr(args, "servo", False):
        # Step 2 双新息伺服：消费 MultiModeReadout（--multi-mode 读出路径）；
        # 修正作用于 flow 速度输出 → 与 direct/c2 互斥；head-only 下 servo 无梯度。
        if not args.multi_mode:
            raise ValueError("--servo requires --multi-mode（伺服消费 MultiModeReadout）")
        if args.direct_head or args.c2_controller:
            raise ValueError(
                "--servo 与 --direct-head/--c2-controller 互斥（修正作用于 flow 输出）"
            )
        if args.head_only:
            raise ValueError("--servo 与 --head-only 互斥（servo 需要可训练参数路径）")
        if args.scene_teacher:
            raise ValueError(
                "--servo 与 --scene-teacher 互斥（scene_teacher 重建 optimizer，"
                "servo 参数会掉出参数组）"
            )
        if getattr(args, "servo_rank", 2) < 1:
            raise ValueError("--servo-rank 必须为正")
    if getattr(args, "servo_dls", False) and not getattr(args, "servo", False):
        raise ValueError("--servo-dls requires --servo")
    if getattr(args, "servo_lambda", 1e-2) <= 0.0:
        raise ValueError("--servo-lambda 必须为正")
    perturb_data = getattr(args, "perturb_data", None)
    if perturb_data is not None:
        # Step 2 微扰混合（设计 §六.1/§六.2）：paired 批与 perturbed 行 pair_id
        # 冲突；fork 契约与混批互斥；配对布局需要 m ∈ [1, B//2]。
        if args.e2e_data:
            raise ValueError("--perturb-data is not supported with --e2e-data")
        if not args.single_task:
            raise ValueError(
                "--perturb-data requires --single-task（配对批与 perturbed 行 pair_id 冲突）"
            )
        if args.fork_data is not None:
            raise ValueError("--perturb-data 与 --fork-data 互斥（混批破坏 fork 配对契约）")
        if not (0.0 < getattr(args, "servo_perturb_ratio", 0.5) <= 0.5):
            raise ValueError("--servo-perturb-ratio 须在 (0, 0.5]（每批 perturbed 占比）")
        if args.batch_size < 2:
            raise ValueError("--perturb-data requires --batch-size >= 2（配对混批）")
        if not args.data:
            raise ValueError("--perturb-data requires --data（clean 行来源）")
        if args.c2_controller:
            raise ValueError("--perturb-data 与 --c2-controller 互斥（混批破坏 C² 干净/恢复契约）")
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
    # 第二轮架构重构（2026-08-08）参数校验。P0-高优 fail-fast：语义上下文/
    # role query/双注意力有结构性前置条件时直接报错，而不是静默失效。
    if args.flow_semantic and not args.compile_task and not args.live_vjepa:
        raise ValueError(
            "--flow-semantic requires --compile-task (e2e) or --live-vjepa "
            "(semantic_context = langslot 槽输出 [B, 25, vision_dim])"
        )
    if args.flow_semantic and args.flow_cond != "adaln":
        raise ValueError(
            "--flow-semantic requires --flow-cond=adaln (entry mode has no "
            "per-layer cross-attention and ignores semantic_context)"
        )
    if args.role_query and not (args.memory_split or args.action_query_cond):
        raise ValueError(
            "--role-query requires --memory-split or --action-query-cond "
            "(role queries summarize the language key for exactly these two paths)"
        )
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
    if args.dual_attention and args.sequential_coupling == 1:
        raise ValueError(
            "--dual-attention is incompatible with --sequential-coupling=1 "
            "(every VA layer is sequential; dual attention would never apply)"
        )
    if args.dual_attention and args.sequential_coupling > 1:
        print(
            "warning: --dual-attention only splits the non-sequential VA layers; "
            "sequential layers keep the legacy shared path"
        )


def validate_finite_update_scalars(
    named_losses: list[tuple[str, object]],
) -> None:
    """Reject non-finite scalar losses before autograd can touch parameters."""
    for name, value in named_losses:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.numel() != 1:
            raise RuntimeError(
                f"non-scalar update loss {name}: shape={tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value.detach()).item()):
            raise FloatingPointError(
                f"non-finite update loss {name}: value={value.detach().item()!r}"
            )


def validate_update_gradients(
    named_parameters,
    *,
    max_norm: float | None = None,
) -> float:
    """Validate current parameters/gradients and return their aggregate grad norm.

    The parameter check shares the already-required gradient traversal, so every
    update is guarded without copying parameters or optimizer state. ``None``
    gradients are allowed because conditional branches can leave modules unused.
    ``max_norm`` applies only to the aggregate norm, never to individual elements.
    """
    norm_terms: list[float] = []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameter_value = parameter.detach()
        if not bool(torch.isfinite(parameter_value).all().item()):
            bad = (~torch.isfinite(parameter_value)).flatten().nonzero(as_tuple=False)[0].item()
            value = parameter_value.flatten()[bad].item()
            raise FloatingPointError(f"non-finite parameter {name}: value={value!r}")
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient.detach())
        if not bool(finite.all().item()):
            bad = (~finite).flatten().nonzero(as_tuple=False)[0].item()
            value = gradient.detach().flatten()[bad].item()
            raise FloatingPointError(
                f"non-finite gradient {name}: value={value!r}"
            )
        norm_terms.append(float(gradient.detach().double().norm().item()))
    norm = math.sqrt(math.fsum(term * term for term in norm_terms))
    if not math.isfinite(norm):
        raise FloatingPointError(f"non-finite aggregate gradient norm: value={norm!r}")
    if max_norm is not None and norm > max_norm:
        raise FloatingPointError(
            f"gradient threshold exceeded aggregate_norm: value={norm!r} "
            f"> threshold={max_norm!r}"
        )
    return norm


def validate_optimizer_update_state(
    optimizer: torch.optim.Optimizer,
    *,
    validate_values: bool = True,
) -> None:
    """Validate optimizer hyperparameters, parameters, and existing tensor state.

    This full state scan runs once after startup/resume. Per update, callers repeat
    the cheap param-group validation and use :func:`validate_update_gradients` to
    scan live parameters; AdamW state is not transactionally copied or rescanned
    because finite source state plus guarded parameters, gradients, and arithmetic
    inputs make the next ordinary update finite.
    """
    base = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
    for group_index, group in enumerate(base.param_groups):
        def finite_number(key: str, *, minimum: float, strict: bool = False) -> float:
            try:
                value = float(group[key])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] {key}: {group.get(key)!r}"
                ) from exc
            valid = math.isfinite(value) and (value > minimum if strict else value >= minimum)
            if not valid:
                comparator = ">" if strict else ">="
                raise ValueError(
                    f"invalid optimizer group[{group_index}] {key}: {value!r}; "
                    f"must be finite and {comparator} {minimum}"
                )
            return value

        finite_number("lr", minimum=0.0)
        if "initial_lr" in group:
            finite_number("initial_lr", minimum=0.0)
        finite_number("weight_decay", minimum=0.0)
        finite_number("eps", minimum=0.0, strict=True)
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"invalid optimizer group[{group_index}] betas: {betas!r}")
        for beta_index, beta in enumerate(betas):
            try:
                beta_value = float(beta)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] beta[{beta_index}]: {beta!r}"
                ) from exc
            if not math.isfinite(beta_value) or not 0.0 <= beta_value < 1.0:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] beta[{beta_index}]: "
                    f"{beta_value!r}; must be finite and in [0, 1)"
                )
        if isinstance(optimizer, SAM):
            sam_group = optimizer.param_groups[group_index]
            try:
                rho = float(sam_group["rho"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] rho: {sam_group.get('rho')!r}"
                ) from exc
            if not math.isfinite(rho) or rho < 0.0:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] rho: {rho!r}; "
                    "must be finite and >= 0"
                )
        if validate_values:
            for parameter_index, parameter in enumerate(group["params"]):
                if not bool(torch.isfinite(parameter.detach()).all().item()):
                    raise FloatingPointError(
                        f"non-finite optimizer parameter group[{group_index}]"
                        f"[{parameter_index}]"
                    )

    def validate_state_value(path: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise FloatingPointError(f"non-finite optimizer state {path}")
        elif isinstance(value, dict):
            for key, child in value.items():
                validate_state_value(f"{path}.{key}", child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                validate_state_value(f"{path}[{index}]", child)
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite optimizer state {path}: {value!r}")

    for state_index, state in enumerate(base.state.values()):
        if validate_values:
            validate_state_value(f"state[{state_index}]", state)


def named_trainable_parameters(*modules):
    """Yield unique, stably named trainable parameters from update modules."""
    seen: set[int] = set()
    for prefix, module in modules:
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield f"{prefix}.{name}", parameter


def named_optimizer_parameters(optimizer, *modules):
    """Name every unique trainable optimizer parameter, including external heads."""
    known = {
        id(parameter): name
        for name, parameter in named_trainable_parameters(*modules)
    }
    seen: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield (
                known.get(
                    id(parameter),
                    f"optimizer.group[{group_index}].parameter[{parameter_index}]",
                ),
                parameter,
            )


def clip_update_gradients(named_parameters, *, max_norm: float) -> float:
    """Clip already-validated gradients, with PyTorch's finite-error guard."""
    unique_parameters = []
    seen: set[int] = set()
    for _, parameter in named_parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            unique_parameters.append(parameter)
    return float(
        torch.nn.utils.clip_grad_norm_(
            unique_parameters,
            max_norm,
            error_if_nonfinite=True,
        ).item()
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


def _maybe_build_live_vision(args, device):
    """Stage B：在线 V-JEPA 构建（Codex P0-2：single-task 分支也必须创建；
    P0-3：参数保持 FP32，前向 autocast BF16——FP16 参数直接 AdamW 更新会
    整步归零，FP16 ULP 6.1e-5 > lr 3e-6）。"""
    if not args.live_vjepa:
        return None
    from va_compound.backbones import VJEPA21Backbone

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device,
        dtype="float32",  # FP32 master 参数（P0-3）
        max_tokens=144,  # 12x12 grid（spatiotemporal 双片 → 288 token）
        local_files_only=True,
    )
    if args.vision_unfreeze_all:
        vision_backbone.unfreeze_all()
        print("live-vjepa: V-JEPA 全量解冻（FP32 参数，BF16 autocast 前向）")
    elif args.vision_unfreeze_last > 0:
        vision_backbone.unfreeze_last(args.vision_unfreeze_last)
        print(f"live-vjepa: V-JEPA 解冻最后 {args.vision_unfreeze_last} 个 block（FP32）")
    return vision_backbone


def _feature_optimizer_groups(args, model, vision_backbone):
    """Stage A/B 参数分组：槽模块高 LR / VA 低 LR / V-JEPA 最低 LR。

    --head-only（Stage 1 对齐）时只训 flow head：VA（含内部视觉流
    q_v/k_v/u_v）、槽、V-JEPA 全部冻结——避免随机初始化的动作头噪声
    梯度污染已训练好的视觉/集成参数（用户 2026-08-08 架构裁决）。
    """
    def is_action_vision_parameter(name: str) -> bool:
        return name.startswith("action_dense_evidence_proj.") or ".action_dense_" in name

    if getattr(args, "action_vision_only", False):
        action_params = []
        for name, param in model.named_parameters():
            selected = is_action_vision_parameter(name)
            param.requires_grad_(selected)
            if selected:
                action_params.append(param)
        if not action_params:
            raise ValueError(
                "--action-vision-only requested but the policy has no action vision branch"
            )
        print(
            "action-vision-only: frozen base E7; trainable additive branch "
            f"params={sum(p.numel() for p in action_params):,} "
            f"@ lr={args.lr_action_vision}",
            flush=True,
        )
        return [{"params": action_params, "lr": args.lr_action_vision}]
    if getattr(args, "wmrm_only", False):
        if getattr(model.config, "va_world_mode", "legacy") != "legacy":
            raise ValueError(
                "--wmrm-only does not support peer readout ownership; use legacy"
            )
        if getattr(model, "wmrm", None) is None:
            raise ValueError("--wmrm-only requires a constructed WAM4VA module")
        wmrm_params, frozen_names = [], []
        for name, param in model.named_parameters():
            if name.startswith("wmrm."):
                wmrm_params.append(param)
            else:
                frozen_names.append(name)
                param.requires_grad_(False)
        if not wmrm_params:
            raise ValueError("--wmrm-only found no wmrm.* parameters")
        print(
            f"wmrm-only: freeze VA/FM ({len(frozen_names)} tensors); "
            f"train latent predictor only "
            f"({sum(p.numel() for p in wmrm_params):,} params)",
            flush=True,
        )
        return [{"params": wmrm_params, "lr": args.lr}]
    if args.head_only:
        head_params, rest_names = [], []
        for name, param in model.named_parameters():
            if name.startswith("flow_head."):
                head_params.append(param)
            else:
                rest_names.append(name)
                param.requires_grad_(False)
        if not head_params:
            raise ValueError("--head-only requires flow head（--direct-head 不支持）")
        print(
            f"head-only: flow head 可训练参数 "
            f"{sum(p.numel() for p in head_params):,}；VA/槽/V-JEPA 冻结 "
            f"（{len(rest_names)} 组参数 requires_grad=False）"
        )
        return [{"params": head_params, "lr": args.lr}]
    if args.servo_only:
        # Step 2 第一阶段（设计 §六.3）：冻结 base policy（VA/flow/入口投影），
        # 只训 reader（角色编译/多模式读出/vis 条件）+ relation 投影——否则
        # 89% 拟合能力的 base path 会吸收恢复数据，servo 分支保持关闭。
        trainable_prefixes = (
            "role_compiler.",
            "slot_reader.",
            "relation_tokens.",
            "vis_conditioner.",
        )
        trainable_params, frozen_names = [], []
        for name, param in model.named_parameters():
            if name.startswith(trainable_prefixes):
                trainable_params.append(param)
            else:
                frozen_names.append(name)
                param.requires_grad_(False)
        if not trainable_params:
            raise ValueError("--servo-only 需要局部槽模块（--local-slots-data/--live-vjepa）")
        print(
            f"servo-only: 冻结 VA/flow 等 {len(frozen_names)} 组参数，只训 "
            f"reader/relation/servo（可训练 {sum(p.numel() for p in trainable_params):,}）",
            flush=True,
        )
        return [
            {
                "params": trainable_params,
                "lr": args.lr_slot if args.lr_slot is not None else args.lr,
            }
        ]
    groups = None
    if model.config.local_slots:
        slot_names = ("role_compiler", "slot_reader", "relation_tokens")
        slot_params, rest_params = [], []
        for name, param in model.named_parameters():
            (slot_params if any(name.startswith(s) for s in slot_names) else rest_params).append(
                param
            )
        groups = [
            {"params": rest_params, "lr": args.lr_va if args.lr_va is not None else args.lr},
            {"params": slot_params, "lr": args.lr_slot if args.lr_slot is not None else args.lr},
        ]
    else:
        if model.config.action_vision_backbone != "none":
            action_params, base_params = [], []
            for name, param in model.named_parameters():
                (action_params if is_action_vision_parameter(name) else base_params).append(param)
            groups = [
                {"params": base_params, "lr": args.lr},
                {"params": action_params, "lr": args.lr_action_vision},
            ]
        else:
            groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if vision_backbone is not None:
        vision_params = [p for p in vision_backbone.parameters() if p.requires_grad]
        if vision_params:
            groups.append({"params": vision_params, "lr": args.lr_vision})
            print(
                f"live-vjepa: V-JEPA 可训练参数 {sum(p.numel() for p in vision_params):,} "
                f"@ lr={args.lr_vision}"
            )
    return groups


def _mtvj_relation_optimizer_group(
    args: argparse.Namespace,
    relation_encoder: nn.Module | None,
) -> dict | None:
    """Return the isolated low-LR group for action-connected relation weights."""
    if not args.mtvj_train_relation:
        return None
    if relation_encoder is None:
        raise ValueError("--mtvj-train-relation 已开启但 relation encoder 未构建")
    parameters = [p for p in relation_encoder.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("--mtvj-train-relation 已开启但没有可训练参数")
    return {"params": parameters, "lr": args.lr_mtvj_relation}


def _mtvj_metric_head_optimizer_group(
    args: argparse.Namespace,
    metric_head: nn.Module | None,
) -> dict | None:
    """Return the isolated tiny-LR group for action-connected localization weights."""
    if not args.mtvj_train_metric_head:
        return None
    if metric_head is None:
        raise ValueError("--mtvj-train-metric-head 已开启但 metric head 未构建")
    parameters = [p for p in metric_head.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("--mtvj-train-metric-head 已开启但没有可训练参数")
    return {"params": parameters, "lr": args.lr_mtvj_metric_head}


def _module_action_gradient_norm(module: nn.Module, flag: str, device: torch.device) -> Tensor:
    """Fail fast when a requested joint-training branch is disconnected."""
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(
            f"{flag} 已开启，但动作 loss 未连接这些可训练参数：{missing[:8]}"
        )
    gradients = [
        parameter.grad.detach().norm(2)
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(
            f"{flag} 已开启，但动作 loss 没有传到任何可训练参数；拒绝继续伪解冻训练"
        )
    return torch.stack(gradients).norm(2).to(device=device)


def _restore_mtvj_policy_modules(
    resume_ckpt: dict,
    *,
    relation_encoder: nn.Module | None,
    metric_head: nn.Module | None,
    train_relation: bool,
    replace_metric_head_from_external: bool = False,
    allow_scratch_relation: bool = False,
) -> None:
    """Strictly restore MT-VJ runtime modules from the main policy checkpoint.

    Legacy policy checkpoints did not contain the metric head.  For that one-way
    migration the already-strictly-loaded external metric checkpoint remains the
    source; every checkpoint saved by this code records and subsequently requires
    ``mtvj_metric_head``. ``allow_scratch_relation``（DINO-metric 从零）放行
    随机初始化 relation encoder 的联合训练——V-JEPA 路径仍拒绝（其 relation
    bridge 语义依赖已监督的 metric head）。
    """
    contract = resume_ckpt.get("training_contract") or {}
    if relation_encoder is not None:
        saved_relation = resume_ckpt.get("mtvj_relation_encoder")
        if saved_relation is None:
            if train_relation and not allow_scratch_relation:
                raise ValueError(
                    "--mtvj-train-relation requires --resume checkpoint "
                    "containing mtvj_relation_encoder；拒绝从随机映射起步"
                )
            print(
                "resume: 旧 checkpoint 未保存 MT-VJ relation encoder；"
                "本次使用外部 metric checkpoint 的兼容映射，并会从下一次保存起写入。"
                if not allow_scratch_relation
                else "resume: 旧 checkpoint 无 relation encoder；DINO-metric "
                "从零联合训练（随机初始化，动作 loss 反传）",
                flush=True,
            )
        else:
            relation_encoder.load_state_dict(saved_relation, strict=True)
            print("mtvj: relation encoder 从主 checkpoint 严格恢复", flush=True)

    if metric_head is not None:
        saved_head = resume_ckpt.get("mtvj_metric_head")
        if saved_head is not None:
            if replace_metric_head_from_external:
                print(
                    "mtvj migration: 已严格恢复主 policy 的 8D relation encoder；"
                    "显式跳过主 policy 的旧 metric head，保留 external "
                    "all-task head",
                    flush=True,
                )
                return
            saved_config = resume_ckpt.get("mtvj_metric_head_config")
            saved_identity = resume_ckpt.get("mtvj_metric_checkpoint_identity")
            if saved_config is None or saved_identity is None:
                raise ValueError(
                    "主 checkpoint 含 mtvj_metric_head，但缺少完整构造配置或"
                    "外部来源指纹"
                )
            expected_config = _canonical_mtvj_metric_head_config(
                saved_config,
                require_complete=True,
            )
            runtime_config = _mtvj_metric_head_constructor_config(metric_head)
            if runtime_config != expected_config:
                raise ValueError(
                    "runtime metric head 构造配置与主 checkpoint 不一致："
                    f"checkpoint={expected_config}, runtime={runtime_config}"
                )
            metric_head.load_state_dict(saved_head, strict=True)
            print("mtvj: metric head 从主 checkpoint 严格恢复", flush=True)
        elif contract.get("metric_head_checkpointed"):
            raise ValueError(
                "resume checkpoint 声明 metric_head_checkpointed=True，"
                "但缺少 mtvj_metric_head"
            )
        else:
            print(
                "mtvj: resume checkpoint 无 metric head 权重；本模块保持当前"
                "（外部 checkpoint 严格加载或 DINO-metric 从零）初始化，"
                "下一次保存将写入主 checkpoint",
                flush=True,
            )


def final_checkpoint_save_due(
    save_path: Path | None,
    global_step: int,
    last_saved_global_step: int | None,
) -> bool:
    """Return whether the completed run still needs its final checkpoint save."""
    return save_path is not None and last_saved_global_step != global_step


def save_checkpoint(
    args,
    config,
    model,
    e2e_model,
    scene_teacher=None,
    vision_backbone=None,
    servo=None,
    relation_encoder=None,
    metric_head=None,
    roi_head=None,
    optimizer=None,
    global_step: int = 0,
    sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None = None,
    exact_run_contract: dict | None = None,
) -> None:
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
                name.removeprefix("text_backbone.").removeprefix("text_model."): parameter.detach().cpu()
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
                # P0-3：semantic adapter / compile-task 恢复所需字段。旧实现
                # 不存这些 → eval_metaworld 用 lora_rank=32 建 LoRA、不构造
                # QwenSemanticBackbone/门控/compiler，新 checkpoint 无法恢复。
                "semantic_adapter": args.semantic_adapter,
                "semantic_lora_rank": args.semantic_lora_rank,
                "semantic_lora_alpha": args.lora_alpha,
                "semantic_top_layers": args.semantic_top_layers,
                "semantic_anchor_layers": args.semantic_anchor_layers,
                "semantic_lora_suffixes": args.semantic_lora_suffixes,
                "compile_task": args.compile_task,
                "compile_every": args.compile_every,
                "n_scene_tokens": args.compile_n_scene,
                "compile_n_readout": args.compile_n_readout,
                "language_max_length": args.language_max_length,
                "flow_semantic": args.flow_semantic,
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
                "vision_pooling": (
                    "dense" if args.dense_readout else args.vision_pooling
                ),
                "flow_steps": args.flow_steps,
                "min_sequence_length": args.min_sequence_length,
                "pair_loss_weight": args.pair_loss_weight,
                "pair_mode": args.pair_mode,
                "pair_probe_tau_max": args.pair_probe_tau_max,
                "pair_start_atol": args.pair_start_atol,
                "min_pair_action_delta": args.min_pair_action_delta,
                "task_sampling": args.task_sampling,
                "task_locality_block_batches": args.task_locality_block_batches,
                # Step 2（C²-IRF v2）：双新息伺服契约（评估侧据此重建 InteractionServo）。
                "servo": args.servo,
                "servo_only": args.servo_only,
                "servo_dls": args.servo_dls,
                "servo_rank": args.servo_rank,
                "servo_lambda": args.servo_lambda,
                # MT-VJ runtime contract: keep 8-D shape but null coordinates for
                # roles the visual head predicts as invisible.
                "metric_tokens_enabled": relation_encoder is not None,
                "metric_state_source": (
                    getattr(
                        metric_head,
                        "_mtvj_metric_state_source",
                        MTVJ_LEGACY_METRIC_STATE_SOURCE,
                    )
                    if relation_encoder is not None else None
                ),
                "metric_state_dim": 8 if relation_encoder is not None else None,
                "metric_d_model": config.hidden_dim if relation_encoder is not None else None,
                "metric_contract_version": (
                    getattr(
                        metric_head,
                        "_mtvj_metric_contract_version",
                        MTVJ_LEGACY_METRIC_CONTRACT_VERSION,
                    )
                    if relation_encoder is not None else None
                ),
                "metric_relation_joint_trained": (
                    bool(args.mtvj_train_relation)
                    if relation_encoder is not None
                    else False
                ),
                "metric_relation_lr": (
                    args.lr_mtvj_relation
                    if relation_encoder is not None and args.mtvj_train_relation
                    else None
                ),
                "metric_head_checkpointed": metric_head is not None,
                "metric_head_constructor_contract_version": (
                    1 if metric_head is not None else None
                ),
                "metric_head_joint_trained": (
                    bool(args.mtvj_train_metric_head)
                    if metric_head is not None
                    else False
                ),
                "metric_head_lr": (
                    args.lr_mtvj_metric_head
                    if metric_head is not None and args.mtvj_train_metric_head
                    else None
                ),
                "mtvj_visual_aux_every": int(args.mtvj_visual_aux_every),
                "mtvj_visual_aux_batch": int(args.mtvj_visual_aux_batch),
                "mtvj_visual_aux_loc_lambda": float(
                    args.mtvj_visual_aux_loc_lambda
                ),
                "mtvj_visual_aux_vis_lambda": float(
                    args.mtvj_visual_aux_vis_lambda
                ),
                "mtvj_visual_aux_pixel_contract": (
                    "true_simulator_render_480_to_dino224_v1"
                    if getattr(config, "dino_dense_metric", False)
                    and args.mtvj_visual_aux_every > 0
                    else None
                ),
                "action_vision_enabled": (
                    getattr(config, "action_vision_backbone", "none") != "none"
                ),
                "action_vision_backbone": getattr(
                    config, "action_vision_backbone", "none"
                ),
                "action_vision_model_id": getattr(
                    config, "action_vision_model_id", None
                ),
                "action_vision_image_size": getattr(
                    config, "action_vision_image_size", None
                ),
                "action_vision_feature_dim": getattr(
                    config, "action_vision_dim", None
                ),
                "action_vision_output_layers": list(
                    getattr(config, "action_vision_layers", ())
                ),
                "action_vision_frame_indices": [1, 3],
                "action_vision_checkpoint_sha256": getattr(
                    args, "action_vision_checkpoint_sha256", None
                ),
                # DINO-main replacement contract（评估侧严格校验）。
                "main_vision_backbone": getattr(
                    config, "main_vision_backbone", "vjepa"
                ),
                "main_vision_model_id": getattr(
                    config, "main_vision_model_id", None
                ),
                "main_vision_image_size": getattr(
                    config, "main_vision_image_size", None
                ),
                "main_vision_feature_dim": getattr(
                    config, "main_vision_dim", None
                ),
                "main_vision_grid": getattr(config, "main_vision_grid", None),
                "main_vision_frames": getattr(
                    config, "main_vision_frames", None
                ),
                "main_vision_tokens": getattr(
                    config, "main_vision_tokens", None
                ),
                "main_vision_temporal": bool(
                    getattr(config, "main_vision_temporal", False)
                ),
                "main_vision_temporal_scale": float(
                    getattr(config, "main_vision_temporal_scale", 1.0)
                ),
                "dino_base_vision_contract": (
                    "full_frame_major_grid_tokens_with_dense_kv_additive"
                    if getattr(config, "dino_dense_metric", False)
                    else "main_vision_tokens"
                ),
                "metric_geometry_inject": bool(
                    getattr(config, "metric_geometry_inject", False)
                ),
                "metric_geometry_dim": int(
                    getattr(config, "metric_geometry_dim", 8)
                ),
                "task35_metric_role_contract": (
                    TASK35_METRIC_ROLE_CONTRACT
                    if getattr(config, "dino_dense_metric", False)
                    else None
                ),
                "dino_roi_enabled": roi_head is not None
                and getattr(args, "dino_roi_checkpoint", None) is not None,
                "dino_roi_alpha": (
                    float(args.dino_roi_alpha)
                    if getattr(args, "dino_roi_checkpoint", None) is not None
                    else None
                ),
                "dino_roi_contract": (
                    DINO_METRIC_ROI_CONTRACT
                    if getattr(args, "dino_roi_checkpoint", None) is not None
                    else None
                ),
                "task35_precision_contract": bool(
                    getattr(args, "task35_precision_contract", False)
                ),
                "task35_data_sha256": getattr(
                    args, "task35_data_sha256", None
                ),
                "task35_raw_frames_sha256": getattr(
                    args, "task35_raw_frames_sha256", None
                ),
                "task35_dino_feature_sha256": getattr(
                    args, "task35_dino_feature_sha256", None
                ),
                "wam_enabled": bool(getattr(args, "wam_joint", False)),
                "main_vision_checkpoint_sha256": getattr(
                    args, "main_vision_checkpoint_sha256", None
                ),
                "flow_prefix_steps": args.flow_prefix_steps,
                "flow_prefix_weight": args.flow_prefix_weight,
                "flow_tail_weight": args.flow_tail_weight,
            },
        }
        if servo is not None:
            payload["servo"] = servo.state_dict()
        if scene_teacher is not None:
            payload["scene_teacher"] = scene_teacher.state_dict()
        if args.live_vjepa and vision_backbone is not None:
            # Stage B：解冻后的 V-JEPA 权重必须随 checkpoint 保存（评估侧
            # eval_metaworld.py 已支持 vjepa_state_dict 恢复）。
            payload["vjepa_state_dict"] = vision_backbone.model.state_dict()
        if getattr(model, "wam", None) is not None:
            # 追加式：WAM 启用时写入完整恢复契约（wam_model + wam_config +
            # 来源指纹），不改动任何既有键语义（eval_metaworld --wam auto 据此识别）。
            import dataclasses

            payload["wam_model"] = {
                key: value.detach().cpu()
                for key, value in model.wam.state_dict().items()
            }
            payload["wam_config"] = dataclasses.asdict(model.wam.config)
            payload["wam_base_ckpt_sha256"] = (
                _sha256_file(Path(args.wam_ckpt))
                if getattr(args, "wam_ckpt", None)
                else "builtin"
            )
            payload["training_contract"]["wam_joint"] = True
            payload["training_contract"]["wam_contract_version"] = 1
    if getattr(args, "visual_world_supervision", False):
        split_identity = getattr(args, "visual_world_split_identity", None)
        if not isinstance(split_identity, dict):
            raise ValueError(
                "visual World checkpoint requires a validated split identity"
            )
        payload["training_contract"].update(
            {
                "world_supervision": WORLD_SUPERVISION_CONTRACT,
                "world_transition": WORLD_TRANSITION_CONTRACT,
                "world_loss_weights": dict(WORLD_LOSS_COMPONENT_WEIGHTS),
                "world_stage_auxiliary_decay": WORLD_STAGE_AUXILIARY_DECAY,
                "world_no_regression": dict(WORLD_NO_REGRESSION),
                "world_static_copy_constraint": {
                    **WORLD_STATIC_COPY_CONSTRAINT,
                    "weight": float(
                        getattr(args, "wmrm_static_constraint_weight", 4.0)
                    ),
                },
                "world_action_ranking": world_action_ranking_contract(
                    getattr(args, "world_action_rank_stage", "cycle"),
                    getattr(args, "wmrm_action_rank_per_sample_cap", None),
                ),
                "world_action_donor_contract": WORLD_ACTION_DONOR_CONTRACT,
                "world_action_donor_sha256": split_identity[
                    "world_action_donor_sha256"
                ],
                "world_action_donor_transitions": split_identity[
                    "world_action_donor_transitions"
                ],
                "world_action_rank_transitions": split_identity[
                    "world_action_rank_transitions"
                ],
                "world_action_source": "logged_cycle6",
                "world_target_stop_gradient": True,
                "world_logged_branch": WORLD_LOGGED_BRANCH_CONTRACT,
                "va_world_mode": getattr(args, "va_world_mode", "legacy"),
                "peer_world_topology": (
                    PEER_WORLD_TOPOLOGY_CONTRACT
                    if getattr(args, "va_world_mode", "legacy") == "peer_sync_h6"
                    else None
                ),
                "peer_world_action_source": (
                    PEER_WORLD_ACTION_SOURCE_CONTRACT
                    if getattr(args, "va_world_mode", "legacy") == "peer_sync_h6"
                    else None
                ),
                "peer_world_readout": (
                    PEER_WORLD_READOUT_CONTRACT
                    if getattr(args, "va_world_mode", "legacy") == "peer_sync_h6"
                    else None
                ),
                "split_manifest_id": split_identity["manifest_id"],
                "split_manifest_path": split_identity["manifest_path"],
                "split_manifest_sha256": split_identity["manifest_sha256"],
                "split_source_sha256": split_identity["source_sha256"],
            }
        )
    if (
        roi_head is not None
        and getattr(args, "dino_roi_checkpoint", None) is not None
    ):
        payload["dino_roi_checkpoint_identity"] = dict(
            getattr(roi_head, "_dino_roi_identity", {})
        )
        if not payload["dino_roi_checkpoint_identity"].get("sha256"):
            raise ValueError("saving DINO ROI policy requires a verified ROI identity")
    if (
        roi_head is not None
        and getattr(args, "dino_roi_checkpoint", None) is None
    ):
        # Legacy MT-VJ ROI checkpoints embed their head/config/identity directly.
        # Keep this behavior for programmatic callers that pass a validated
        # ``roi_head`` without reconstructing the CLI namespace.
        roi_config = getattr(roi_head, "_mtvj_roi_config", None)
        roi_identity = getattr(roi_head, "_mtvj_roi_checkpoint_identity", None)
        roi_coarse_identity = getattr(roi_head, "_mtvj_roi_coarse_identity", None)
        if not isinstance(roi_config, dict) or not isinstance(roi_identity, dict):
            raise ValueError("保存 MT-VJ ROI head 需要已校验的 config/identity")
        if not roi_identity.get("sha256") or not isinstance(roi_coarse_identity, dict):
            raise ValueError("保存 MT-VJ ROI head 需要完整 ROI/coarse SHA identity")
        payload["training_contract"].update(
            {
                "mtvj_roi_enabled": True,
                "mtvj_roi_alpha": float(args.mtvj_roi_alpha),
                "mtvj_roi_contract_version": METRIC_ROI_CONTRACT_VERSION,
                "mtvj_roi_coarse_sha256": roi_coarse_identity.get("sha256"),
                "mtvj_roi_coarse_head_state_sha256": roi_config.get(
                    "coarse_head_state_sha256"
                ),
                "mtvj_roi_head_checkpointed": True,
            }
        )
        payload["mtvj_roi_head"] = {
            key: value.detach().cpu()
            for key, value in roi_head.state_dict().items()
        }
        payload["mtvj_roi_config"] = dict(roi_config)
        payload["mtvj_roi_checkpoint_identity"] = dict(roi_identity)
    if relation_encoder is not None:
        payload["mtvj_relation_encoder"] = {
            key: value.detach().cpu()
            for key, value in relation_encoder.state_dict().items()
        }
    if metric_head is not None:
        metric_config = _mtvj_metric_head_constructor_config(metric_head)
        metric_identity = getattr(
            metric_head, "_mtvj_external_checkpoint_identity", None
        )
        dino_from_scratch = (
            isinstance(metric_identity, dict)
            and metric_identity.get("source") == "dino-metric-from-scratch"
        )
        if not isinstance(metric_identity, dict) or not (
            metric_identity.get("sha256") or dino_from_scratch
        ):
            raise ValueError(
                "保存 MT-VJ metric head 需要已校验的外部 checkpoint 来源指纹；"
                "请通过 _load_mtvj_metric_checkpoint 构造该模块"
            )
        payload["mtvj_metric_head"] = {
            key: value.detach().cpu()
            for key, value in metric_head.state_dict().items()
        }
        payload["mtvj_metric_head_config"] = metric_config
        payload["mtvj_metric_checkpoint_identity"] = dict(metric_identity)
        metric_source = getattr(
            metric_head, "_mtvj_metric_head_source", "unknown"
        )
        payload["training_contract"]["metric_head_source"] = metric_source
        migration_record = getattr(
            metric_head, "_mtvj_metric_head_migration", None
        )
        if isinstance(migration_record, dict):
            payload["mtvj_metric_head_migration"] = dict(migration_record)
    if optimizer is not None:
        if exact_run_contract is None:
            exact_run_contract = build_exact_run_contract(
                args, config, optimizer, sampler, metric_head, roi_head
            )
        payload.update(
            build_exact_resume_state(
                optimizer, global_step, sampler, exact_run_contract
            )
        )
    tmp_path = args.save.with_suffix(args.save.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(args.save)
    if getattr(args, "save_step_copies", False) and global_step > 0:
        step_path = args.save.with_name(
            f"{args.save.stem}_s{int(global_step)}{args.save.suffix}"
        )
        step_tmp = step_path.with_suffix(step_path.suffix + ".tmp")
        if step_path.exists():
            if _sha256_file(step_path) != _sha256_file(args.save):
                raise FileExistsError(
                    f"refusing to overwrite checkpoint step copy: {step_path}"
                )
        else:
            if step_tmp.exists():
                raise FileExistsError(
                    f"stale checkpoint step-copy temporary exists: {step_tmp}"
                )
            shutil.copy2(args.save, step_tmp)
            step_tmp.replace(step_path)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021) 简洁实现。

    平坦化微调保留指令遵循（arXiv:2606.23641）：权重先沿梯度方向扰动
    rho * grad/||grad||，在 worst-case 邻域重算 loss，再走真实 Adam 步。
    训练循环：loss.backward() → first_step(zero_grad=True) →
    loss2.backward() → second_step(zero_grad=True)。
    """

    def __init__(self, params, base_optimizer, rho: float, **kwargs) -> None:
        if not math.isfinite(float(rho)) or rho < 0.0:
            raise ValueError("SAM rho must be finite and non-negative")
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
                original = p.detach().clone(memory_format=torch.preserve_format)
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)
                self.state[p]["pre_perturbation"] = original
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore_step(self, zero_grad: bool = False) -> None:
        """Undo a first-step perturbation without applying the base optimizer."""
        for group in self.param_groups:
            for p in group["params"]:
                original = self.state[p].pop("pre_perturbation", None)
                if original is not None:
                    p.copy_(original)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        self.restore_step(zero_grad=False)
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


def _validate_dino_roi_resume_contract(
    resume_checkpoint: dict | None,
    *,
    runtime_checkpoint: Path | None,
    runtime_alpha: float | None,
    runtime_identity: dict | None = None,
) -> bool:
    """Fail fast when a resumed DINO policy changes its ROI distribution.

    Returns whether the saved policy used DINO ROI. Identity is checked after
    the runtime artifact has been loaded, while path/alpha checks can run first.
    """
    if resume_checkpoint is None:
        return False
    training_contract = resume_checkpoint.get("training_contract") or {}
    resume_dino_roi = training_contract.get("dino_roi_enabled") is True
    if not resume_dino_roi:
        return False
    if runtime_checkpoint is None:
        raise ValueError(
            "resume checkpoint requires --dino-roi-checkpoint; refusing to "
            "silently continue on a different geometry distribution"
        )
    saved_alpha = training_contract.get("dino_roi_alpha")
    if saved_alpha is None or runtime_alpha is None or float(saved_alpha) != float(
        runtime_alpha
    ):
        raise ValueError(
            "--dino-roi-alpha must exactly match resume training: "
            f"policy={saved_alpha!r}, runtime={runtime_alpha!r}"
        )
    if runtime_identity is None:
        return True
    expected_identity = resume_checkpoint.get("dino_roi_checkpoint_identity")
    if not isinstance(expected_identity, dict):
        raise ValueError("resume checkpoint declares DINO ROI but lacks its identity")
    mismatches = {
        key: (expected_identity.get(key), runtime_identity.get(key))
        for key in ("sha256", "size_bytes", "contract")
        if expected_identity.get(key) != runtime_identity.get(key)
    }
    if mismatches:
        raise ValueError(f"DINO ROI checkpoint identity mismatch on resume: {mismatches}")
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.training_stage == "c" and not args.vision_unfreeze_all:
        print(
            "hint: --training-stage c recommends --vision-unfreeze-all "
            "(not enforced)"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    loader = None
    iterator = None
    sampler = None
    model = None
    e2e_model = None
    vision_backbone = None
    perturb_payload = None
    perturb_iter = None
    use_payload_vision = True
    m = 0
    task_log_names: dict[int, str] = {}
    env_by_description: dict[str, str] = {}
    lang_aux_cache: dict[str, tuple[Tensor, Tensor]] = {}
    aux_tasks: list[str] = []
    aux_task_w = torch.empty(0, dtype=torch.float64)
    fork_iter = None  # 修复 HEAD 隐患：无 --data 的合成冒烟路径此前 UnboundLocalError
    if args.e2e_data:
        dataset = E2EDataset(args.e2e_data, min_sequence_length=args.min_sequence_length)
        _enable_optional_action_masks(dataset)
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
            va_attention_backend=args.va_attention_backend,
            action_query_cond=args.action_query_cond,
            memory_split=args.memory_split,
            evidence_tokens=args.evidence_tokens,
            task_tokens=args.task_tokens,
            future_predict=args.future_predict,
            sequential_coupling=args.sequential_coupling,
            flow_cond=args.flow_cond,
            flow_layers=args.flow_layers,
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
            wam_joint=args.wam_joint,
            wmrm=args.wmrm,
            va_world_mode=args.va_world_mode,
            wmrm_inject=args.wmrm_inject,
            wmrm_target=getattr(args, "wmrm_target", "dino"),
            wmrm_world_dim=(
                768
                if getattr(args, "wmrm_target", "dino") == "vjepa"
                else (
                    8
                    if getattr(args, "wmrm_target", "dino") == "metric"
                    else getattr(args, "hidden_dim", 512)
                )
            ),
            wmrm_cycle_steps=getattr(args, "wmrm_cycle_steps", 6),
            wmrm_med_margin=getattr(args, "wmrm_med_margin", 0.05),
            wmrm_handshake=getattr(args, "wmrm_handshake", True),
            wmrm_detach_proposal_stage_state=getattr(
                args, "wmrm_detach_proposal_stage_state", False
            ),
            wmrm_map_size=getattr(args, "wmrm_map_size", 16),
            wmrm_map_channels=getattr(args, "wmrm_map_channels", 32),
            wmrm_world_grid=getattr(args, "wmrm_world_grid", 16),
            wmrm_predictor=getattr(args, "wmrm_predictor", "legacy"),
            wmrm_predictor_depth=getattr(args, "wmrm_predictor_depth", 6),
            wmrm_predictor_width=getattr(args, "wmrm_predictor_width", 384),
            wmrm_predictor_heads=getattr(args, "wmrm_predictor_heads", 12),
            **_mtvj_config_kwargs(args),
        )
        if args.single_task:
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
            )
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
        local_payload = (
            torch.load(args.local_slots_data, map_location="cpu", weights_only=True)
            if args.local_slots_data
            else None
        )
        local_tokens = None
        if local_payload is not None:
            # ST288 大数组在 .npy memmap（16.36 GiB FP16，mmap 零 RAM 峰值）。
            from va_compound.live_vjepa import load_st288_memmap

            npy_path = local_payload["vision_tokens_st_npy"]
            local_tokens = load_st288_memmap(npy_path, local_payload["metadata"])
            if args.dense_readout:
                # Step 0 预计算路径：数据必须已是 1152-token 密集特征
                # （coords [1152,3] 随数据；FeatureDataset 不限制 token 数）。
                if local_tokens.shape[-2] != 1152:
                    raise ValueError(
                        f"--dense-readout with --local-slots-data requires "
                        f"1152-token dense features, got {local_tokens.shape[-2]}"
                    )
        if (
            (args.dense_readout_mtvj or getattr(args, "dino_main_vision", False))
            and not args.live_vjepa
        ):
            # MT-VJ / DINO-main 主数据路径：longtraj windows + JPEG 帧在线解码
            # （2026-08-10）。DINO-main 复用同一帧数据源（V-JEPA 路径保留）。
            from va_compound.longtraj_frames import LongTrajFramesDataset

            dataset = LongTrajFramesDataset(
                args.data,
                min_sequence_length=args.min_sequence_length,
                decode_cache_tasks=2 if getattr(args, "dino_main_vision", False) else 1,
                feature_cache=(
                    args.dino_feature_cache
                    if getattr(args, "dino_main_vision", False)
                    else None
                ),
                include_frames=(
                    args.dino_feature_cache is None
                    or getattr(args, "dino_roi_checkpoint", None) is not None
                ),
            )
            print(
                f"{'DINO-main' if args.dino_main_vision else 'MT-VJ'} data: "
                f"LongTrajFramesDataset（{len(dataset)} 样本，"
                f"帧从 longtraj JPEG 在线解码，"
                f"decode_cache_tasks={dataset.decode_cache_tasks}）",
                flush=True,
            )
        elif args.live_vjepa:
            # Stage B：在线 V-JEPA 编码（帧变体；vision_tokens 键被移除）。
            from va_compound.live_vjepa import LiveVJEPADataset

            dataset = LiveVJEPADataset(
                args.data,
                args.live_root,
                min_sequence_length=args.min_sequence_length,
                vision_pooling=args.vision_pooling,
                control_stride=args.control_stride,
                spe=args.sequences_per_episode,
                phase_bins=args.phase_bins,
                phase_seed=args.phase_seed,
                success_only=args.success_only,
                sliding=args.sliding_window,
                frame_aug=args.frame_aug,
                frame_aug_geometric=args.frame_aug_geometric,
                dense_readout=args.dense_readout,
            )
        else:
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
                local_tokens=local_tokens,
                coords=(local_payload["coords"] if local_payload is not None else None),
            )
        _enable_optional_action_masks(dataset)
        if getattr(args, "visual_world_supervision", False):
            args.visual_world_split_identity = validate_visual_world_training_split(
                dataset.payload,
                args.data,
                args.world_split_manifest,
                va_world_mode=getattr(args, "va_world_mode", "legacy"),
            )
            donor_identity = prepare_visual_world_action_ranking(dataset.payload)
            args.visual_world_split_identity.update(donor_identity)
            print(
                "visual World split: PASS "
                f"manifest={args.visual_world_split_identity['manifest_id']} "
                f"sha256={args.visual_world_split_identity['manifest_sha256']} "
                "action_donors="
                f"{donor_identity['world_action_rank_transitions']}/"
                f"{donor_identity['world_action_donor_transitions']}",
                flush=True,
            )
        descriptions = list(
            dataset.payload.get("metadata", {}).get("tasks", [])
        )
        if descriptions:
            from scripts.build_longtraj_features import ENV_TO_TASK

            env_by_description = {
                description: env_name for env_name, description in ENV_TO_TASK.items()
            }
            task_log_names = {
                index: env_by_description.get(description, f"task-{index}")
                for index, description in enumerate(descriptions)
            }
        if getattr(args, "task35_precision_contract", False):
            if int(dataset.payload["actions"].shape[-2]) != 6:
                raise ValueError(
                    "task35 DINO dense metric requires exact H6 action chunks"
                )
            action_mask = dataset.payload.get("action_valid_mask")
            expected_mask_shape = dataset.payload["actions"].shape[:-1]
            if (
                not isinstance(action_mask, torch.Tensor)
                or action_mask.dtype != torch.bool
                or action_mask.shape != expected_mask_shape
                or not bool(action_mask.any())
            ):
                raise ValueError(
                    "task35 precision payload requires nonempty boolean "
                    f"action_valid_mask with shape {tuple(expected_mask_shape)}"
                )
            metadata = dataset.payload.get("metadata") or {}
            if (
                metadata.get("contract")
                != "task35_clean_recovery_h6_matched_va_v1"
                or metadata.get("task") != "peg-insert-side-v3"
                or metadata.get("task_role_order")
                != ["tool", "pegGrasp", "hole", "pegHead"]
                or metadata.get("roi_relation_pair") != ["pegHead", "hole"]
                or int(metadata.get("n_clean_windows", 0)) <= 0
                or int(metadata.get("n_recovery_windows", 0)) <= 0
            ):
                raise ValueError(
                    "task35 precision training requires the strict matched "
                    "clean/recovery H6 payload and role contract"
                )
            instruction_ids = dataset.payload["instruction_id"].unique().tolist()
            if instruction_ids != [35]:
                raise ValueError(
                    "task35 precision payload must contain only instruction_id 35, "
                    f"got {instruction_ids}"
                )
            if int(args.main_vision_frames) != 4:
                raise ValueError(
                    "task35 DINO dense metric requires exactly four frames "
                    "[d-6,d-4,d-2,d]"
                )
            if int(args.main_vision_grid) != 16:
                raise ValueError(
                    "task35 DINO dense metric requires --main-vision-grid 16 "
                    "(4*256=1024 base-vision tokens)"
                )
            if not getattr(args, "main_vision_temporal", False):
                raise ValueError(
                    "task35 DINO dense metric requires --main-vision-temporal"
                )
            if not getattr(args, "metric_geometry_inject", False):
                raise ValueError(
                    "task35 DINO dense metric requires --metric-geometry-inject"
                )
            if getattr(args, "dino_roi_checkpoint", None) is None:
                raise ValueError(
                    "task35 DINO dense metric requires a v2 --dino-roi-checkpoint"
                )
            if dataset.cached_raw_frames is None:
                raise ValueError(
                    "task35 precision training requires DINO cache raw_frames.npy "
                    "to avoid mixed-container JPEG decode and lock ROI pixels"
                )
            cache_meta = json.loads(
                (args.dino_feature_cache / "meta.json").read_text()
            )
            data_digest = hashlib.sha256()
            with args.data.expanduser().open("rb") as stream:
                for block in iter(lambda: stream.read(16 << 20), b""):
                    data_digest.update(block)
            args.task35_data_sha256 = data_digest.hexdigest()
            if cache_meta.get("dataset_sha256") != args.task35_data_sha256:
                raise ValueError(
                    "task35 precision DINO cache was not built from the exact "
                    "matched H6 payload"
                )
            raw_path = args.dino_feature_cache / "raw_frames.npy"
            expected_raw_sha = cache_meta.get("raw_frames_sha256")
            args.task35_raw_frames_sha256 = expected_raw_sha
            if (
                cache_meta.get("raw_frame_contract")
                != "exact_decoded_longtraj_jpeg_480_v1"
                or cache_meta.get("raw_frame_shape")
                != [int(dataset.cached_raw_frames.shape[0]), 480, 480, 3]
                or not expected_raw_sha
                or _sha256_file(raw_path) != expected_raw_sha
            ):
                raise ValueError(
                    "task35 precision DINO cache lacks the exact 480px raw-frame "
                    "identity contract"
                )
        dino_main_kwargs = _main_vision_config_kwargs(args)
        config = VACompoundConfig(
            language_dim=int(dataset.payload["language_hidden"].shape[-1]),
            vision_dim=(
                int(dino_main_kwargs["main_vision_dim"])
                if dino_main_kwargs  # DINO-main：主视觉维 = DINO 特征维
                else 768 if args.live_vjepa
                else int(local_tokens.shape[-1])
                if local_tokens is not None  # ST288 路径无 vision_tokens 键（Codex P0-4）
                else 768 if args.dense_readout_mtvj  # MT-VJ：在线 dense，H11 特征维 768
                else int(dataset.payload[vision_key].shape[-1])
            ),
            action_horizon=int(dataset.payload["actions"].shape[-2]),
            action_dim=int(dataset.payload["actions"].shape[-1]),
            proprio_dim=int(dataset.payload["proprio"].shape[-1]),
            mode=args.mode,
            num_layers=args.va_layers,
            qk_norm=args.qk_norm,
            attention_variant=args.attention_variant,
            va_attention_backend=args.va_attention_backend,
            action_query_cond=args.action_query_cond,
            memory_split=args.memory_split,
            evidence_tokens=args.evidence_tokens,
            task_tokens=args.task_tokens,
            future_predict=args.future_predict,
            sequential_coupling=args.sequential_coupling,
            flow_cond=args.flow_cond,
            flow_layers=args.flow_layers,
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
            wam_joint=args.wam_joint,
            wmrm=args.wmrm,
            va_world_mode=args.va_world_mode,
            wmrm_inject=args.wmrm_inject,
            wmrm_target=getattr(args, "wmrm_target", "dino"),
            wmrm_world_dim=(
                768
                if getattr(args, "wmrm_target", "dino") == "vjepa"
                else (
                    8
                    if getattr(args, "wmrm_target", "dino") == "metric"
                    else getattr(args, "hidden_dim", 512)
                )
            ),
            wmrm_cycle_steps=getattr(args, "wmrm_cycle_steps", 6),
            wmrm_med_margin=getattr(args, "wmrm_med_margin", 0.05),
            wmrm_handshake=getattr(args, "wmrm_handshake", True),
            wmrm_detach_proposal_stage_state=getattr(
                args, "wmrm_detach_proposal_stage_state", False
            ),
            wmrm_map_size=getattr(args, "wmrm_map_size", 16),
            wmrm_map_channels=getattr(args, "wmrm_map_channels", 32),
            wmrm_world_grid=getattr(args, "wmrm_world_grid", 16),
            wmrm_predictor=getattr(args, "wmrm_predictor", "legacy"),
            wmrm_predictor_depth=getattr(args, "wmrm_predictor_depth", 6),
            wmrm_predictor_width=getattr(args, "wmrm_predictor_width", 384),
            wmrm_predictor_heads=getattr(args, "wmrm_predictor_heads", 12),
            local_slots=(args.local_slots_data is not None) or args.live_vjepa,
            local_slots_direct288=args.local_slots_direct288,
            local_slots_fixed_query=args.local_slots_fixed_query,
            dense_readout=args.dense_readout,
            multi_mode=args.multi_mode,
            local_slot_tokens=1152 if args.dense_readout else 288,
            **dino_main_kwargs,
            **_mtvj_config_kwargs(args),
        )
        if args.perturb_data is not None:
            # Step 2 微扰混合（设计 §六.1）：clean [B−m] + perturbed [m] → [B]。
            # 视觉与路径同构（token 数一致）时直接用 payload 特征；否则（--servo
            # 的 local_slots 路径，payload 为 flat-64）用存储帧在线编码（近似）。
            perturb_payload = torch.load(
                args.perturb_data, map_location="cpu", weights_only=True
            )
            expected_n = (
                config.local_slot_tokens
                if config.local_slots
                else int(dataset.payload[vision_key].shape[-2])
            )
            use_payload_vision = (
                int(perturb_payload["vision_tokens"].shape[-2]) == expected_n
            )
            if not use_payload_vision:
                frames = perturb_payload.get("frames")
                if frames is None or frames.ndim != 6 or frames.shape[0] == 0:
                    raise ValueError(
                        "--servo/local_slots 路径下 --perturb-data 需要 dense/ST 特征或存储帧："
                        f"payload vision_tokens={tuple(perturb_payload['vision_tokens'].shape)} "
                        f"（期望 {expected_n}），且无帧（--no-store-frames 生成的数据不可用）"
                    )
            m = max(
                1,
                min(
                    int(round(args.batch_size * args.servo_perturb_ratio)),
                    args.batch_size // 2,
                ),
            )
            perturb_dataset = FeatureDataset(
                args.perturb_data,
                require_pairs=False,
                min_sequence_length=args.min_sequence_length,
                vision_key="vision_tokens",
            )
            perturb_loader = DataLoader(
                IndexedDataset(perturb_dataset),
                batch_size=m,
                shuffle=True,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                drop_last=True,  # 100 行微扰数据 → 最后不足 m 的批会触发 mix 报错（2026-08-09 修复）
            )
            perturb_iter = iter_forever(perturb_loader)
            print(
                f"perturb-data: {args.perturb_data}（{len(perturb_dataset)} 行，"
                f"每批 {m} 行，paired 共享 (τ,ε)；"
                f"vision={'payload' if use_payload_vision else 'frames-online'}）",
                flush=True,
            )
        if args.single_task:
            effective_batch = (
                args.batch_size - m
                if args.perturb_data is not None
                else (c2_clean_n if args.c2_controller else args.batch_size)
            )
            if args.task_sampling in {"weighted", "balanced"}:
                tasks = list(dataset.payload.get("metadata", {}).get("tasks", []))
                if not tasks:
                    raise ValueError(
                        f"--task-sampling {args.task_sampling} 需要数据集 metadata.tasks "
                        "（instruction_id → 难度权重映射）"
                    )
                raw_task_w = (
                    torch.ones(len(tasks), dtype=torch.float64)
                    if args.task_sampling == "balanced"
                    else torch.tensor(task_weights_for(tasks), dtype=torch.float64)
                )
                # Codex P1-2（2026-08-09）：曝光 = 窗口数 × 难度权重会引入轨迹
                # 长度偏置（各任务窗口 360-2186）。除以任务窗口数 → 每任务总曝光
                # ∝ 难度权重，消除窗口数偏置（任务级分层）。
                task_rows = torch.bincount(
                    dataset.payload["instruction_id"], minlength=len(tasks)
                ).to(torch.float64)
                task_w = raw_task_w / task_rows.clamp_min(1.0)
                per_sample = task_w[dataset.payload["instruction_id"]]
                print(
                    f"--task-sampling {args.task_sampling}: 任务权重 "
                    f"{sorted(set(raw_task_w.tolist()))}（active_tasks="
                    f"{int((task_rows > 0).sum())}, samples={len(per_sample)}；"
                    + (
                        "每 epoch 严格均衡任务 batch）"
                        if args.task_sampling == "balanced"
                        else "按难度分层）"
                    ),
                    flush=True,
                )
                # 双数据流辅助批次（阶段 C）：任务文本 → 数据集预计算
                # language_hidden/mask 缓存（metadata.tasks[tid] ↔ instruction_id=tid，
                # 与 build_longtraj_features.task_language_t 同源）；辅助任务按
                # 难度权重 raw_task_w 多项式采样。单任务子集（2026-08-16）只缓存
                # 数据里实际出现的任务。
                lang_aux_cache = {}
                aux_tasks = []
                aux_weights: list[float] = []
                if args.mtvj_visual_aux_every > 0:
                    hid_all = dataset.payload["language_hidden"]
                    mask_all = dataset.payload["language_mask"]
                    id_all = dataset.payload["instruction_id"]
                    for tid, text in enumerate(tasks):
                        hits = (id_all == tid).nonzero()
                        if hits.numel() == 0:
                            continue
                        row = int(hits[0, 0])
                        lang_aux_cache[text] = (
                            hid_all[row].float(),
                            mask_all[row],
                        )
                        aux_tasks.append(text)
                        aux_weights.append(float(raw_task_w[tid]))
                    if not aux_tasks:
                        raise ValueError("visual aux: 数据集没有任何活跃任务文本")
                    aux_task_w = torch.tensor(aux_weights, dtype=torch.float64)
                    aux_task_w = aux_task_w / aux_task_w.sum()
                    print(
                        "mtvj visual aux: 每 "
                        f"{args.mtvj_visual_aux_every} 步一个在线仿真视觉批次"
                        f"（λ_loc={args.mtvj_visual_aux_loc_lambda}, "
                        f"λ_vis={args.mtvj_visual_aux_vis_lambda}, "
                        f"batch={args.mtvj_visual_aux_batch}）",
                        flush=True,
                    )
                if args.dense_readout_mtvj or getattr(args, "dino_main_vision", False):
                    # LongTraj JPEG frames: single-task batches + decode cache.
                    # DINO-main must use this path; mixing tasks with cache=1
                    # stalls the GPU on 3s full-task decodes.
                    from va_compound.longtraj_frames import mtvj_collate
                    sampler = TaskLocalityWeightedSampler(
                        dataset.payload["instruction_id"],
                        dataset.payload["episode_id"],
                        raw_task_w,
                        effective_batch,
                        args.seed,
                        args.task_locality_block_batches,
                        args.task_sampling,
                    )
                    if args.save is not None or args.resume_exact is not None:
                        # Full payload hashing happens once here; periodic/final
                        # checkpoints reuse the sampler-cached identity.
                        sampler.bind_dataset_content_identity(
                            build_dataset_content_identity(
                                args.data,
                                dataset.payload,
                                longtraj_dir=getattr(dataset, "longtraj_dir", None),
                            )
                        )
                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        collate_fn=mtvj_collate,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0,
                        # Keep iterator base-seed generation off the global torch
                        # RNG restored by --resume-exact. With num_workers=0 this
                        # generator has no data/augmentation semantics.
                        generator=torch.Generator().manual_seed(args.seed),
                    )
                elif args.task_sampling == "balanced":
                    sampler = TaskLocalityWeightedSampler(
                        dataset.payload["instruction_id"],
                        dataset.payload["episode_id"],
                        raw_task_w,
                        effective_batch,
                        args.seed,
                        args.task_locality_block_batches,
                        args.task_sampling,
                    )
                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0,
                        generator=torch.Generator().manual_seed(args.seed),
                    )
                else:
                    sampler = TaskWeightedSampler(
                        per_sample, effective_batch, args.seed
                    )
                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0,
                        generator=torch.Generator().manual_seed(args.seed),
                    )
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=effective_batch,
                    shuffle=True,
                    num_workers=args.num_workers,
                    persistent_workers=args.num_workers > 0,
                )
        else:
            loader = DataLoader(
                dataset,
                batch_sampler=PairedBatchSampler(dataset, args.batch_size, args.seed),
            )
        fork_iter = None
        if args.fork_data is not None:
            # pair 生死门（Q5b）：fork 数据集独立 loader（PairedBatchSampler，
            # 每批 2 对=4 行全真配对），与 v5 批按 --fork-k 交替。
            from va_compound.live_vjepa import _slot_coords  # noqa: F401

            fork_dataset = FeatureDataset(
                args.fork_data,
                # E 组（打乱配对）不满足同帧契约 → 跳过校验（采样器走
                # build_pair_groups 泛化回退）；D 组严格校验。
                require_pairs=not args.fork_skip_contract,
                min_sequence_length=args.min_sequence_length,
                pair_start_atol=args.pair_start_atol,
                min_pair_action_delta=args.min_pair_action_delta,
                pair_start_cosine=args.pair_start_cosine,
                vision_key=(
                    "vision_tokens_spatial"
                    if args.vision_pooling == "spatial"
                    else "vision_tokens"
                ),
            )
            fork_loader = DataLoader(
                fork_dataset,
                batch_sampler=PairedBatchSampler(fork_dataset, args.batch_size, args.seed),
            )
            fork_iter = iter_forever(fork_loader)
            print(
                f"fork-data: {args.fork_data}（{len(fork_dataset)} 行，"
                f"交替比 1:{args.fork_k}，pair_loss_weight={args.pair_loss_weight}）",
                flush=True,
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
            va_attention_backend=args.va_attention_backend,
            action_query_cond=args.action_query_cond,
            memory_split=args.memory_split,
            evidence_tokens=args.evidence_tokens,
            task_tokens=args.task_tokens,
            future_predict=args.future_predict,
            sequential_coupling=args.sequential_coupling,
            flow_cond=args.flow_cond,
            flow_layers=args.flow_layers,
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
            wam_joint=args.wam_joint,
            wmrm=args.wmrm,
            va_world_mode=args.va_world_mode,
            wmrm_inject=args.wmrm_inject,
            wmrm_target=getattr(args, "wmrm_target", "dino"),
            wmrm_world_dim=(
                768
                if getattr(args, "wmrm_target", "dino") == "vjepa"
                else (
                    8
                    if getattr(args, "wmrm_target", "dino") == "metric"
                    else getattr(args, "hidden_dim", 512)
                )
            ),
            wmrm_cycle_steps=getattr(args, "wmrm_cycle_steps", 6),
            wmrm_med_margin=getattr(args, "wmrm_med_margin", 0.05),
            wmrm_handshake=getattr(args, "wmrm_handshake", True),
            wmrm_detach_proposal_stage_state=getattr(
                args, "wmrm_detach_proposal_stage_state", False
            ),
            wmrm_map_size=getattr(args, "wmrm_map_size", 16),
            wmrm_map_channels=getattr(args, "wmrm_map_channels", 32),
            wmrm_world_grid=getattr(args, "wmrm_world_grid", 16),
            wmrm_predictor=getattr(args, "wmrm_predictor", "legacy"),
            wmrm_predictor_depth=getattr(args, "wmrm_predictor_depth", 6),
            wmrm_predictor_width=getattr(args, "wmrm_predictor_width", 384),
            wmrm_predictor_heads=getattr(args, "wmrm_predictor_heads", 12),
            **_mtvj_config_kwargs(args),
        )
        smoke_batch = synthetic_sequence(
            config,
            args.batch_size,
            args.sequence_length,
            device,
            with_frames=args.dense_readout_mtvj,
        )

    # MT-VJ（契约 §6）：fp16 V-JEPA 始终冻结只读；metric localization path
    # 与 relation encoder 默认冻结，可分别显式联合微调。
    # MT-VJ flags 都未给时以下对象全为 None，旧路径不变。
    mtvj_backbone = None
    action_vision_backbone = None
    metric_head = None
    relation_encoder = None
    roi_head = None
    resume_path = (
        args.resume_exact
        if args.resume_exact is not None
        else args.resume
        if args.resume is not None
        else getattr(args, "resume_weights", None)
    )
    preloaded_resume_ckpt = None
    if args.metric_visual_checkpoint is not None and resume_path is not None:
        # Metric head 的网络形状必须先由主 checkpoint 决定，再建 optimizer；
        # 因此在通用 resume 块之前只读一次，后面复用同一个 payload。
        preloaded_resume_ckpt = torch.load(
            resume_path, map_location="cpu", weights_only=True
        )
    if getattr(args, "dino_dense_metric", False) and resume_path is not None:
        # DINO-metric 续训：构造配置严格取自主 checkpoint（同 §2 契约）。
        if preloaded_resume_ckpt is None:
            preloaded_resume_ckpt = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
    if args.dense_readout_mtvj:
        mtvj_backbone = _maybe_build_mtvj_backbone(device)
    action_vision_backbone = _maybe_build_action_vision_backbone(
        args, config, device
    )
    main_vision_backbone = None
    dino_cache = None
    if getattr(args, "dino_main_vision", False):
        # DINO-main：冻结 DINOv2 替换 V-JEPA 主视觉（V-JEPA 路径保留未删除）。
        main_vision_backbone = _build_dino_main_backbone(args, config, device)
        if args.dino_feature_cache is not None:
            # 特征缓存模式：训练循环从 memmap 读预计算特征（塔仅用于校验/
            # 不在循环内前向）。位级一致由 build_dino_feature_cache.py 验证。
            dino_cache = DinoFeatureCache(args.dino_feature_cache)
            if getattr(args, "task35_precision_contract", False):
                args.task35_dino_feature_sha256 = dict(
                    dino_cache.meta["feature_sha256"]
                )
            if (
                dino_cache.meta.get("model_id") != config.main_vision_model_id
                or int(dino_cache.meta.get("image_size", 0))
                != config.main_vision_image_size
                or int(dino_cache.meta.get("window", 0))
                != config.main_vision_frames
            ):
                raise ValueError(
                    "DINO feature cache 元信息与配置不一致："
                    f"{dino_cache.meta} vs "
                    f"model={config.main_vision_model_id}, "
                    f"size={config.main_vision_image_size}, "
                    f"window={config.main_vision_frames}"
                    "（缓存存全 16×16 patch，grid 在读取时池化，不参与比对）"
                )
    if getattr(args, "dino_dense_metric", False):
        # DINO-metric：metric 栈从零构建（resume 时用主 checkpoint 的构造
        # 配置），不复用 V-JEPA --metric-visual-checkpoint 权重。
        metric_head, relation_encoder = _build_dino_metric_stack(
            device,
            config,
            train_metric_head=args.mtvj_train_metric_head,
            train_relation=args.mtvj_train_relation,
            saved_ctor_config=(
                preloaded_resume_ckpt.get("mtvj_metric_head_config")
                if preloaded_resume_ckpt is not None
                else None
            ),
        )
        resume_dino_roi = _validate_dino_roi_resume_contract(
            preloaded_resume_ckpt,
            runtime_checkpoint=args.dino_roi_checkpoint,
            runtime_alpha=args.dino_roi_alpha,
        )
        if args.dino_roi_checkpoint is not None:
            roi_head = load_dino_metric_roi_checkpoint(
                args.dino_roi_checkpoint, device
            )
            if resume_dino_roi:
                _validate_dino_roi_resume_contract(
                    preloaded_resume_ckpt,
                    runtime_checkpoint=args.dino_roi_checkpoint,
                    runtime_alpha=args.dino_roi_alpha,
                    runtime_identity=getattr(roi_head, "_dino_roi_identity", {}),
                )
            print(
                "dino-metric: frozen task35 ROI refinement active during policy "
                f"training (alpha={args.dino_roi_alpha}, "
                f"contract={DINO_METRIC_ROI_CONTRACT}, "
                f"roles={TASK35_METRIC_ROLE_CONTRACT})",
                flush=True,
            )
    if args.metric_visual_checkpoint is not None:
        policy_contract = (
            (preloaded_resume_ckpt.get("training_contract") or {})
            if preloaded_resume_ckpt is not None
            else {}
        )
        metric_head, relation_encoder = _load_mtvj_metric_checkpoint(
            args.metric_visual_checkpoint,
            device,
            config,
            train_relation=args.mtvj_train_relation,
            train_metric_head=args.mtvj_train_metric_head,
            policy_relation_state=(
                preloaded_resume_ckpt.get("mtvj_relation_encoder")
                if preloaded_resume_ckpt is not None
                else None
            ),
            policy_metric_state=(
                preloaded_resume_ckpt.get("mtvj_metric_head")
                if preloaded_resume_ckpt is not None
                else None
            ),
            policy_metric_config=(
                preloaded_resume_ckpt.get("mtvj_metric_head_config")
                if preloaded_resume_ckpt is not None
                else None
            ),
            policy_metric_identity=(
                preloaded_resume_ckpt.get("mtvj_metric_checkpoint_identity")
                if preloaded_resume_ckpt is not None
                else None
            ),
            policy_metric_migration=(
                preloaded_resume_ckpt.get("mtvj_metric_head_migration")
                if preloaded_resume_ckpt is not None
                else None
            ),
            policy_training_contract=policy_contract,
            exact_resume=args.resume_exact is not None,
            replace_metric_head_from_external=(
                args.replace_mtvj_metric_head_from_external
            ),
        )
        policy_roi_enabled = policy_contract.get("mtvj_roi_enabled") is True
        if policy_roi_enabled and args.mtvj_roi_checkpoint is None:
            raise ValueError(
                "resume checkpoint requires --mtvj-roi-checkpoint; refusing to "
                "silently disable its trained ROI runtime"
            )
        if args.mtvj_roi_checkpoint is not None:
            if policy_roi_enabled:
                saved_alpha = policy_contract.get("mtvj_roi_alpha")
                if saved_alpha is None or float(saved_alpha) != float(args.mtvj_roi_alpha):
                    raise ValueError(
                        "--mtvj-roi-alpha must exactly match the resume checkpoint: "
                        f"policy={saved_alpha!r}, runtime={args.mtvj_roi_alpha!r}"
                    )
            coarse_identity = getattr(
                metric_head, "_mtvj_current_external_checkpoint_identity", None
            )
            if not isinstance(coarse_identity, dict):
                raise ValueError("MT-VJ metric head lacks its external coarse identity")
            roi_head = load_metric_roi_checkpoint(
                args.mtvj_roi_checkpoint,
                device,
                coarse_identity=coarse_identity,
                coarse_head_state_sha256=metric_head_state_sha256(metric_head),
                policy_state=(
                    preloaded_resume_ckpt.get("mtvj_roi_head")
                    if preloaded_resume_ckpt is not None
                    else None
                ),
                policy_config=(
                    preloaded_resume_ckpt.get("mtvj_roi_config")
                    if preloaded_resume_ckpt is not None
                    else None
                ),
                policy_identity=(
                    preloaded_resume_ckpt.get("mtvj_roi_checkpoint_identity")
                    if preloaded_resume_ckpt is not None
                    else None
                ),
                policy_training_contract=policy_contract,
            )
            print(
                "mtvj: frozen ROI head loaded "
                f"(alpha={args.mtvj_roi_alpha}, "
                f"params={sum(p.numel() for p in roi_head.parameters()):,})",
                flush=True,
            )

    # Step 2：双新息中央凹交互伺服（C²-IRF v2 §七 Step 2；--servo-only 隐含
    # --servo 已在 validate_args 生效）。独立模块（契约文件 va_compound/servo.py），
    # 不进 VACompoundPolicy；checkpoint 单独存 "servo" 键 + training_contract 字段。
    servo = None
    servo_stats = None
    if args.servo:
        servo = InteractionServo(
            vision_dim=config.vision_dim,
            lang_dim=config.hidden_dim,
            action_dim=config.action_dim,
            rank=args.servo_rank,
            dls=args.servo_dls,
            dls_lambda=args.servo_lambda,
        ).to(device)
        servo_stats = {}
        servo_lr = (
            args.lr_servo
            if args.lr_servo is not None
            else (args.lr_slot if args.lr_slot is not None else args.lr)
        )
        print(
            f"servo: params={sum(p.numel() for p in servo.parameters()):,} "
            f"dls={args.servo_dls} rank={args.servo_rank} lr={servo_lr} "
            f"only={args.servo_only}",
            flush=True,
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
        vision_backbone = _maybe_build_live_vision(args, device)
        if args.role_seeds and config.local_slots:
            seeds = torch.load(args.role_seeds, map_location="cpu", weights_only=True)[
                "role_seeds"
            ]
            model.role_compiler.set_role_description_embeddings(seeds)
            print(f"PULSE-VA: role seeds initialized from {args.role_seeds}")
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
        groups = _feature_optimizer_groups(args, model, vision_backbone)
        if servo is not None:
            groups.append({"params": list(servo.parameters()), "lr": servo_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    else:
        model = VACompoundPolicy(config).to(device)
        vision_backbone = _maybe_build_live_vision(args, device)
        if args.role_seeds and config.local_slots:
            seeds = torch.load(args.role_seeds, map_location="cpu", weights_only=True)[
                "role_seeds"
            ]
            model.role_compiler.set_role_description_embeddings(seeds)
            print(f"PULSE-VA: role seeds initialized from {args.role_seeds}")
        groups = _feature_optimizer_groups(args, model, vision_backbone)
        if servo is not None:
            groups.append({"params": list(servo.parameters()), "lr": servo_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    # Plan-Cache 方案 A：加载冻结 Qwen（local files only，fp16/bf16）+ 可训练
    # SceneTeacher（projector + readout tokens 进 optimizer）。指令文本从
    # dataset payload 的 metadata.tasks 按 instruction_id 重建（与数据 prep
    # 阶段 compute language_hidden 所用的字符串完全一致）。
    scene_teacher = None
    text_backbone = None
    # Keep the dataset task list alive for MT-VJ auxiliary sampling even when
    # SceneTeacher is disabled.  The previous ``tasks = None`` assignment made
    # Stage-C fail on its first auxiliary update.
    tasks = list(dataset.payload.get("metadata", {}).get("tasks", []))
    if args.scene_teacher:
        from va_compound.backbones import QwenTextBackbone, SceneTeacher

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

    relation_group = _mtvj_relation_optimizer_group(args, relation_encoder)
    if relation_group is not None:
        optimizer.add_param_group(relation_group)
        print(
            "mtvj: relation action path 加入 optimizer "
            f"（params={sum(p.numel() for p in relation_group['params']):,}, "
            f"lr={args.lr_mtvj_relation}）",
            flush=True,
        )
    metric_head_group = _mtvj_metric_head_optimizer_group(args, metric_head)
    if metric_head_group is not None:
        optimizer.add_param_group(metric_head_group)
        print(
            "mtvj: metric localization action path 加入 optimizer "
            f"（params={sum(p.numel() for p in metric_head_group['params']):,}, "
            f"lr={args.lr_mtvj_metric_head}）；V-JEPA 保持冻结",
            flush=True,
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
    validate_optimizer_update_state(optimizer)

    # E7 WAM v1（M0）：--wam-joint 时 attach 独立 WAM 模块（只读残差通路，
    # 由 scripts/train_wam_e7.py 独立训练）。object.__setattr__ 绕过
    # nn.Module 注册——WAM 参数不进 model.state_dict()/optimizer，
    # 保证 checkpoint 与优化器契约完全追加式。
    if args.wam_joint and model is not None:
        from va_compound.wam import WAMConfig, JointWorldActionFlow

        wam = JointWorldActionFlow(
            WAMConfig(
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
            )
        ).to(device)
        if args.wam_ckpt:
            wam_state = torch.load(args.wam_ckpt, map_location="cpu", weights_only=True)
            if isinstance(wam_state, dict) and "wam_model" in wam_state:
                wam_state = wam_state["wam_model"]
            wam.load_state_dict(wam_state)
            print(f"wam: 权重加载自 {args.wam_ckpt}")
        object.__setattr__(model, "wam", wam)
        object.__setattr__(model, "wam_alpha", args.wam_alpha)
        print(
            f"wam: JointWorldActionFlow attached "
            f"({wam.num_params():,} params, alpha={args.wam_alpha})"
        )

    # Cheap after startup: the expensive data digest is already cached on the
    # locality sampler. This immutable value is reused by every checkpoint.
    runtime_exact_run_contract = build_exact_run_contract(
        args, config, optimizer, sampler, metric_head, roi_head
    )

    if e2e_model is not None:
        e2e_model.train()
    else:
        model.train()
    global_step = 0
    resume_rng_state = None
    exact_resume = args.resume_exact is not None
    if resume_path is not None:
        resume_ckpt = (
            preloaded_resume_ckpt
            if preloaded_resume_ckpt is not None
            else torch.load(resume_path, map_location="cpu", weights_only=True)
        )
        if exact_resume:
            if getattr(args, "visual_world_supervision", False):
                validate_visual_world_resume_contract(
                    resume_ckpt,
                    args.visual_world_split_identity,
                    world_action_ranking_contract(
                        getattr(args, "world_action_rank_stage", "cycle"),
                        getattr(args, "wmrm_action_rank_per_sample_cap", None),
                    ),
                    float(getattr(args, "wmrm_static_constraint_weight", 4.0)),
                    args.resume_exact_contract_migration,
                    getattr(args, "va_world_mode", "legacy"),
                )
            # Fail before restoring model/optimizer/sampler/RNG if any data,
            # objective, sampler, architecture or optimizer semantic changed.
            validate_exact_run_contract(
                resume_ckpt.get("exact_run_contract"),
                runtime_exact_run_contract,
                migration_id=args.resume_exact_contract_migration,
            )
        resume_config = resume_ckpt["config"]
        for key in (
            "num_layers",
            "hidden_dim",
            "action_dim",
            "proprio_dim",
            "mode",
            "va_world_mode",
            "wmrm_predictor",
            "wmrm_predictor_depth",
            "wmrm_predictor_width",
            "wmrm_predictor_heads",
        ):
            if key.startswith("wmrm_") and not getattr(config, "wmrm", False):
                continue
            left = resume_config.get(key)
            if key == "va_world_mode" and left is None:
                left = "legacy"
            right = getattr(config, key, None)
            if key.startswith("wmrm_") and left is None:
                left = {
                    "wmrm_predictor": "legacy",
                    "wmrm_predictor_depth": 6,
                    "wmrm_predictor_width": 384,
                    "wmrm_predictor_heads": 12,
                }[key]
            if left != right:
                raise ValueError(
                    f"resume config mismatch on {key}: {left} vs {right}"
                )
        if e2e_model is not None:
            e2e_model.policy.load_state_dict(resume_ckpt["model"])
            e2e_model.vision_backbone.model.load_state_dict(resume_ckpt["vjepa_state_dict"])
            if resume_ckpt.get("qwen_state_dict"):
                qwen_state = {
                    k.removeprefix("text_backbone.").removeprefix("text_model."): v
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
            print(f"e2e resumed from {resume_path}")
        else:
            if exact_resume:
                model.load_state_dict(resume_ckpt["model"], strict=True)
            elif args.c2_controller:
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
                    missing, unexpected = model.load_state_dict(
                        resume_ckpt["model"], strict=False
                    )
                    # entry→adaln flow 迁移：新 flow 参数（ada_mlps/ca_*）允许
                    # missing（随机初始化），V-JEPA/VA/共享 head 权重仍继承。
                    if missing or unexpected:
                        print(
                            f"resume (non-strict): missing={len(missing)} "
                            f"unexpected={len(unexpected)}"
                        )
            else:
                # 架构迁移（Codex P0-5，2026-08-09）：H8→H48 时 action_queries
                # [8,512]→[48,512] 等 shape 不匹配键在 strict=False 下仍会崩。
                # 显式过滤并重新初始化（其余权重正常继承）。
                state = dict(resume_ckpt["model"])
                own_shapes = {
                    key: tuple(value.shape)
                    for key, value in model.state_dict().items()
                }
                mismatched = [
                    key for key in state
                    if key in own_shapes and tuple(state[key].shape) != own_shapes[key]
                ]
                if mismatched:
                    print(
                        f"resume migration: {len(mismatched)} 个键 shape 不匹配，"
                        f"重新初始化（新架构）：{sorted(mismatched)[:8]}"
                    )
                    state = {key: value for key, value in state.items()
                             if key not in mismatched}
                missing, unexpected = model.load_state_dict(state, strict=False)
                allowed_new_missing: set[str] = set()
                if (
                    config.action_vision_backbone != "none"
                    and resume_config.get("action_vision_backbone", "none") == "none"
                ):
                    allowed_new_missing.update(
                        key
                        for key in model.state_dict()
                        if key.startswith("action_dense_evidence_proj.")
                        or ".action_dense_" in key
                    )
                if config.metric_geometry_inject and not resume_config.get(
                    "metric_geometry_inject", False
                ):
                    allowed_new_missing.update(
                        key
                        for key in model.state_dict()
                        if key.startswith("geometry_projection.")
                    )
                if config.main_vision_temporal and not resume_config.get(
                    "main_vision_temporal", False
                ):
                    allowed_new_missing.update(
                        key
                        for key in model.state_dict()
                        if key.startswith("main_vision_frame_embedding.")
                    )
                if allowed_new_missing:
                    forbidden_missing = set(missing) - allowed_new_missing
                    if forbidden_missing:
                        raise ValueError(
                            "architecture migration has unrelated missing keys: "
                            f"{sorted(forbidden_missing)[:8]}"
                        )
                    if unexpected:
                        raise ValueError(
                            "architecture migration has unexpected checkpoint keys: "
                            f"{sorted(unexpected)[:8]}"
                        )
                if missing or unexpected:
                    print(
                        f"resume (non-strict): missing={len(missing)} "
                        f"unexpected={len(unexpected)}"
                    )
            if vision_backbone is not None and resume_ckpt.get("vjepa_state_dict") is not None:
                vision_backbone.model.load_state_dict(resume_ckpt["vjepa_state_dict"])
                print("live-vjepa: V-JEPA 权重从 checkpoint 恢复")
            if scene_teacher is not None:
                if resume_ckpt.get("scene_teacher") is None:
                    raise ValueError(
                        "resume checkpoint has no scene_teacher weights (--scene-teacher run required)"
                    )
                scene_teacher.load_state_dict(resume_ckpt["scene_teacher"])
            if servo is not None:
                if resume_ckpt.get("servo") is not None:
                    servo.load_state_dict(resume_ckpt["servo"])
                    print("servo: 权重从 checkpoint 恢复")
                else:
                    print("resume: checkpoint 无 servo 权重（servo 随机初始化）")
            _restore_mtvj_policy_modules(
                resume_ckpt,
                relation_encoder=relation_encoder,
                metric_head=metric_head,
                train_relation=args.mtvj_train_relation,
                replace_metric_head_from_external=(
                    args.replace_mtvj_metric_head_from_external
                ),
                allow_scratch_relation=bool(
                    getattr(config, "dino_dense_metric", False)
                ),
            )
            if getattr(model, "wam", None) is not None:
                # 确定性续训：WAM 权重与构造配置必须成对出现，不允许随机
                # 初始化 WAM 继续训练（--wam-ckpt 外部文件不受 resume 兜底）。
                if resume_ckpt.get("wam_model") is None:
                    raise ValueError(
                        "--wam-joint resume 需要含 wam_model 的 checkpoint"
                    )
                saved_wam_config = resume_ckpt.get("wam_config")
                if saved_wam_config is None:
                    raise ValueError(
                        "resume checkpoint 含 wam_model 但缺少 wam_config"
                        "（不允许随机初始化 WAM 续训）"
                    )
                import dataclasses

                runtime_wam_config = dataclasses.asdict(model.wam.config)
                if saved_wam_config != runtime_wam_config:
                    raise ValueError(
                        "resume checkpoint wam_config 与运行时 WAMConfig 不一致："
                        f"checkpoint={saved_wam_config}, runtime={runtime_wam_config}"
                    )
                model.wam.load_state_dict(resume_ckpt["wam_model"])
                print("wam: WAM 权重从 resume checkpoint 恢复")
            print(f"resumed from {resume_path}")
        if exact_resume:
            global_step = restore_exact_resume_state(
                resume_ckpt,
                optimizer,
                sampler,
                runtime_exact_run_contract=runtime_exact_run_contract,
                migration_id=args.resume_exact_contract_migration,
                restore_rng=False,
            )
            resume_rng_state = resume_ckpt["rng_state"]
            print(f"exact training state restored at global_step={global_step}", flush=True)
        else:
            # Metadata only: --resume does not restore optimizer/sampler/RNG.
            # --resume-weights is a new run on possibly new data, so the update
            # counter restarts. Ordinary --resume keeps the known update count.
            if getattr(args, "resume_weights", None) is not None:
                global_step = 0
                print(
                    "resume-weights: loaded model; optimizer/sampler/RNG/step restart",
                    flush=True,
                )
            else:
                global_step = int(resume_ckpt.get("global_step", 0))
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
    mtvj_dense_evidence = None
    mtvj_metric_tokens = None
    mtvj_metric_g = None
    action_dense_evidence = None
    # --perturb-data 帧在线编码骨干（live 路径复用主骨干；feature 路径冻结 V-JEPA）。
    perturb_backbone = _maybe_build_perturb_backbone(
        args, device, vision_backbone, use_payload_vision
    )
    if args.lang_fixed_vector:
        # grounding 对照（Codex 2026-08-08）：语言通道 = 数据集全局均值常量向量，
        # 循环外预计算一次；完整模型 vs 固定语言基线的差距即语言条件的因果贡献。
        if args.live_vjepa:
            raise ValueError("--lang-fixed-vector 仅支持 feature 路径（非 live）")
        lang_fixed_vec = dataset.payload["language_hidden"].mean(dim=(0, 1), keepdim=True)
        print(f"lang-fixed-vector: 语言通道替换为全局均值（shape={tuple(lang_fixed_vec.shape)}）")

    if resume_rng_state is not None:
        # DataLoader iterator construction consumes a torch base-seed. Rebuild it
        # from the restored sampler first, then restore global RNG immediately
        # before fetching the next batch/noise.
        iterator = iter(loader)
        restore_rng_state(resume_rng_state)

    last_saved_global_step: int | None = None

    def commit_successful_update(local_step: int, consumed_locality_batch: bool) -> None:
        """Advance all resumable state only after the optimizer update succeeds."""
        nonlocal global_step, last_saved_global_step
        if consumed_locality_batch:
            sampler.advance()
        global_step += 1
        if (
            args.save is not None
            and args.save_every > 0
            and global_step % args.save_every == 0
        ):
            save_checkpoint(
                args,
                config,
                model,
                e2e_model,
                scene_teacher,
                vision_backbone,
                servo=servo,
                relation_encoder=relation_encoder,
                metric_head=metric_head,
                roi_head=roi_head,
                optimizer=optimizer,
                global_step=global_step,
                sampler=sampler,
                exact_run_contract=runtime_exact_run_contract,
            )
            last_saved_global_step = global_step
            print(
                f"step={local_step} global_step={global_step} "
                f"periodic checkpoint saved to {args.save}",
                flush=True,
            )

    for step in range(1, args.steps + 1):
        rec_batch = None
        consumed_locality_batch = False
        is_fork_batch = fork_iter is not None and step % (args.fork_k + 1) == 0
        if args.c2_controller:
            # C² 3:1 混合：clean 部分（v5 + v6a 目标）+ recovery 部分（v6b）。
            batch = move_batch(next(clean_iter), device)
            rec_batch = move_batch(next(rec_iter), device)
            consumed_locality_batch = isinstance(
                sampler, (TaskLocalityWeightedSampler, TaskWeightedSampler)
            )
        elif iterator is None:
            batch = smoke_batch
        else:
            is_fork_batch = fork_iter is not None and step % (args.fork_k + 1) == 0
            if is_fork_batch:
                batch = move_batch(next(fork_iter), device)
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                consumed_locality_batch = isinstance(
                    sampler, (TaskLocalityWeightedSampler, TaskWeightedSampler)
                )
        if mtvj_backbone is not None:
            # MT-VJ（契约 §6）：在线 dense 编码——frames（live 数据集或合成冒烟）
            # → 冻结 V-JEPA forward_hierarchical_dense（fp16）→ {5,11} →
            # dense_evidence；给了 checkpoint 时 metric head + relation encoder
            # → metric_tokens。全程 no_grad（冻结只读）。本块不 pop frames
            # （live 块随后 pop，避免 KeyError）。
            frames_mtvj = batch.get("frames")
            if frames_mtvj is None:
                raise ValueError(
                    "--dense-readout-mtvj 需要原始帧：batch 无 'frames' 键"
                    "（--live-vjepa 数据集或合成冒烟提供）"
                )
            if isinstance(frames_mtvj, torch.Tensor):
                frames_mtvj = frames_mtvj.cpu().numpy()
            mtvj_dense_evidence, mtvj_metric_tokens, mtvj_metric_g = _mtvj_online_encode(
                frames_mtvj,
                mtvj_backbone,
                metric_head,
                relation_encoder,
                batch,
                device,
                train_metric_head=args.mtvj_train_metric_head,
                roi_head=roi_head,
                roi_alpha=(
                    float(args.mtvj_roi_alpha) if roi_head is not None else 0.0
                ),
            )
            if action_vision_backbone is not None:
                action_dense_evidence = _action_vision_online_encode(
                    frames_mtvj,
                    action_vision_backbone,
                    device,
                    encode_batch=args.action_vision_encode_batch,
                )
        if main_vision_backbone is not None:
            # DINO-main replacement（2026-08-14 用户决策）：冻结 DINOv2 特征
            # 替换 V-JEPA 作为 VA 主视觉。V-JEPA/dense/metric 代码保留在仓库
            # 中（--dense-readout-mtvj 等 flag 关闭即禁用），此处仅旁路。
            # DINO-metric（2026-08-15）：同一次窗口编码附带 block11/block23
            # 两帧 [d-2,d] patch evidence + metric tokens（--dino-dense-metric）。
            # 特征缓存模式（--dino-feature-cache）：从 memmap 读预计算特征，
            # 跳过在线 ViT-L 前向（占步时 84%）。
            if dino_cache is not None:
                rows = batch.get("frame_cache_rows")
                if rows is None:
                    raise ValueError(
                        "--dino-feature-cache 需要 batch 'frame_cache_rows' 键"
                    )
                if config.dino_dense_metric:
                    vision_tokens, dense_evidence = _dino_main_encode_from_cache(
                        rows,
                        dino_cache,
                        device,
                        grid=config.main_vision_grid,
                        window=config.main_vision_frames,
                        return_dense=True,
                    )
                    batch["vision_tokens"] = vision_tokens
                    mtvj_dense_evidence = dense_evidence
                    mtvj_metric_tokens, mtvj_metric_g = _dino_metric_tokens(
                        metric_head,
                        relation_encoder,
                        dense_evidence,
                        batch,
                        device,
                        train_metric_head=args.mtvj_train_metric_head,
                        roi_head=roi_head,
                        roi_backbone=main_vision_backbone,
                        roi_frames=batch.get("frames"),
                        roi_alpha=(
                            float(args.dino_roi_alpha) if roi_head is not None else 0.0
                        ),
                    )
                else:
                    batch["vision_tokens"] = _dino_main_encode_from_cache(
                        rows,
                        dino_cache,
                        device,
                        grid=config.main_vision_grid,
                        window=config.main_vision_frames,
                    )
            else:
                frames_main = batch.get("frames")
                if frames_main is None:
                    raise ValueError(
                        "--dino-main-vision 需要原始帧：batch 无 'frames' 键"
                    )
                if isinstance(frames_main, torch.Tensor):
                    frames_main = frames_main.cpu().numpy()
                if config.dino_dense_metric:
                    vision_tokens, dense_evidence = _dino_main_online_encode(
                        frames_main,
                        main_vision_backbone,
                        device,
                        encode_batch=args.main_vision_encode_batch,
                        grid=config.main_vision_grid,
                        window=config.main_vision_frames,
                        return_dense=True,
                    )
                    batch["vision_tokens"] = vision_tokens
                    mtvj_dense_evidence = dense_evidence
                    mtvj_metric_tokens, mtvj_metric_g = _dino_metric_tokens(
                        metric_head,
                        relation_encoder,
                        dense_evidence,
                        batch,
                        device,
                        train_metric_head=args.mtvj_train_metric_head,
                        roi_head=roi_head,
                        roi_backbone=main_vision_backbone,
                        roi_frames=batch.get("frames"),
                        roi_alpha=(
                            float(args.dino_roi_alpha) if roi_head is not None else 0.0
                        ),
                    )
                else:
                    batch["vision_tokens"] = _dino_main_online_encode(
                        frames_main,
                        main_vision_backbone,
                        device,
                        encode_batch=args.main_vision_encode_batch,
                        grid=config.main_vision_grid,
                        window=config.main_vision_frames,
                    )
                batch.pop("frames", None)
            # Cache+ROI keeps raw frames only until the crop refinement above.
            # Remove the large CPU array before generic device collation.
            batch.pop("frames", None)
        if args.live_vjepa:
            # Stage B：在线 V-JEPA 编码（frames 仍为 CPU numpy；编码在 GPU 上，
            # 输出 [B, T, 288, D] 与 ST288 同构，替换进 batch）。
            # Codex P0-3：FP32 参数 + BF16 autocast 前向（FP16 参数 AdamW 更新归零）。
            from va_compound.live_vjepa import encode_live_frames

            frames = batch.pop("frames")
            if isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=any(
                    p.requires_grad for p in vision_backbone.parameters()
                ),
            ):
                encoded = encode_live_frames(
                    frames, vision_backbone, device, dense=args.dense_readout
                )
            batch["vision_tokens"] = encoded.float()  # VA 期待 fp32（与 v5 缓存一致）
            batch["vision_tokens_st"] = batch["vision_tokens"]
        if perturb_iter is not None and not is_fork_batch:
            # Step 2 微扰混合（设计 §六）：clean [B−m] + perturbed [m] → [B]；
            # 配对布局 [c0,p0,c1,p1,...] 供 sample_flow_matching_inputs_paired
            # 共享同一 (τ,ε)（动作差异不被随机 flow noise 淹没）。
            batch = move_batch(batch, device)
            p_indices, p_batch = next(perturb_iter)
            p_batch = move_batch(p_batch, device)
            if use_payload_vision:
                p_vision = perturb_payload["vision_tokens"][p_indices].to(device)
            else:
                p_vision = _encode_perturb_frames(
                    perturb_payload["frames"][p_indices],
                    perturb_backbone
                    if perturb_backbone is not None
                    else vision_backbone,
                    device,
                    dense=args.dense_readout,
                )
            batch, is_perturbed = mix_perturb_batch(batch, p_batch, p_vision, m)
            batch["is_perturbed"] = is_perturbed.to(device)
        else:
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
            validate_finite_update_scalars([("c2.total", loss)])
            loss.backward()
            c2_named_parameters = list(
                named_trainable_parameters(("model", model))
            )
            validate_optimizer_update_state(optimizer, validate_values=False)
            validate_update_gradients(
                c2_named_parameters,
                max_norm=args.max_gradient_norm,
            )
            gradient_norm = clip_update_gradients(
                c2_named_parameters, max_norm=1.0
            )
            optimizer.step()
            validate_optimizer_update_state(optimizer)
            commit_successful_update(step, consumed_locality_batch)
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

        if args.lang_fixed_vector:
            batch["language_hidden"] = lang_fixed_vec.expand(
                batch["language_hidden"].shape
            ).to(device)
            if "language_mask" in batch:
                batch["language_mask"] = torch.ones_like(batch["language_mask"])

        noisy_actions, flow_time, target_velocity = (
            sample_flow_matching_inputs_paired(batch["actions"], batch["is_perturbed"])
            if batch.get("is_perturbed") is not None
            and bool(batch["is_perturbed"].any())
            else sample_flow_matching_inputs(batch["actions"])
        )

        next_global_step = global_step + 1
        prepared_visual_aux = _prepare_mtvj_visual_aux_step(
            aux_tasks,
            aux_task_w,
            env_by_description,
            seed=args.seed,
            global_step=next_global_step,
            every=args.mtvj_visual_aux_every,
            aux_batch=args.mtvj_visual_aux_batch,
            include_raw_frames=bool(getattr(args, "dino_main_vision", False)),
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
                flow_loss, flow_prefix_loss, flow_tail_loss = masked_flow_matching_loss(
                    predicted_velocity,
                    target_velocity,
                    batch,
                    prefix_steps=args.flow_prefix_steps,
                    prefix_weight=args.flow_prefix_weight,
                    tail_weight=args.flow_tail_weight,
                )
                if args.single_task:
                    pair_loss = flow_loss.new_zeros(())
                    pred_delta = flow_loss.new_zeros(())
                    tgt_delta = flow_loss.new_zeros(())
                else:
                    # 配对 E2E（2026-08-07）：与 feature 路径同构的共享源
                    # 反事实干预——语言是 pair 内唯一输入差，速度场差即语言贡献。
                    partner = paired_partner_indices(batch["pair_id"], batch["instruction_id"])
                    # P0-高优：pair 契约——vision/proprio/prev 严格相同，
                    # 反事实干预的输入差只能是语言（否则语言贡献被污染）。
                    if not (
                        torch.equal(mb["video_frames"], mb["video_frames"][partner])
                        and torch.equal(mb["proprio"], mb["proprio"][partner])
                        and torch.equal(
                            mb["previous_action"], mb["previous_action"][partner]
                        )
                    ):
                        raise ValueError(
                            "e2e pair contract violated: video_frames/proprio/"
                            "previous_action must be identical within each pair"
                        )
                    pair_noise, pair_time, pair_target_velocity = sample_pair_intervention(
                        batch["actions"], partner, probe_tau_max=args.pair_probe_tau_max
                    )
                    pair_predicted_velocity = e2e_model.policy.flow_velocity(
                        action_conditions[:, 0],
                        pair_noise,
                        pair_time,
                        # P0-高优：flow_semantic 下 pair 分支复用 t=0 编译的
                        # 语义上下文（与 action_conditions[:, 0] 同一决策点）。
                        semantic_context=getattr(
                            e2e_model, "first_semantic_context", None
                        ),
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
                    servo=servo,
                    servo_stats=servo_stats,
                    dense_evidence=mtvj_dense_evidence,
                    metric_tokens=mtvj_metric_tokens,
                    action_dense_evidence=action_dense_evidence,
                    metric_g=mtvj_metric_g,
                    wmrm_adep_margin=float(getattr(args, "wmrm_adep_margin", 0.05)),
                    visual_world_supervision=bool(
                        getattr(args, "visual_world_supervision", False)
                    ),
                    wmrm_adep_enabled=float(
                        getattr(args, "wmrm_adep_weight", 0.0)
                    )
                    > 0.0,
                    flow_steps=int(args.flow_steps),
                    world_action_rank_step=next_global_step,
                    world_action_rank_stage=getattr(
                        args, "world_action_rank_stage", "cycle"
                    ),
                    wmrm_action_rank_per_sample_cap=getattr(
                        args, "wmrm_action_rank_per_sample_cap", None
                    ),
                    wmrm_static_constraint_weight=float(
                        getattr(args, "wmrm_static_constraint_weight", 4.0)
                    ),
                    feature_autocast_bf16=bool(args.feature_autocast_bf16),
                )
                if model.config.future_predict:
                    predicted_velocity, action_conditions, memories = rollout
                else:
                    predicted_velocity, action_conditions = rollout
                if model.config.direct_head:
                    # Deterministic H6 baseline uses the exact same validity masks as FM.
                    # Reusing the masked reducer prevents settle/post-success targets from
                    # entering the direct loss merely because its decoder is deterministic.
                    absolute_error = F.smooth_l1_loss(
                        predicted_velocity,
                        batch["actions"],
                        reduction="none",
                    )
                    validity = torch.ones_like(absolute_error)
                    for key in ACTION_MASK_KEYS:
                        mask_value = batch.get(key)
                        if mask_value is not None:
                            validity = validity * _expand_action_mask(
                                mask_value, absolute_error, key
                            )
                    denominator = validity.sum()
                    if not bool(denominator > 0):
                        raise ValueError("direct loss has zero valid action elements")
                    flow_loss = (absolute_error * validity).sum() / denominator
                    _, flow_prefix_loss, flow_tail_loss = masked_flow_matching_loss(
                        predicted_velocity,
                        batch["actions"],
                        batch,
                        prefix_steps=args.flow_prefix_steps,
                    )
                    pair_loss = flow_loss.new_zeros(())
                    pred_delta = flow_loss.new_zeros(())
                    tgt_delta = flow_loss.new_zeros(())
                else:
                    flow_loss, flow_prefix_loss, flow_tail_loss = masked_flow_matching_loss(
                        predicted_velocity,
                        target_velocity,
                        batch,
                        prefix_steps=args.flow_prefix_steps,
                        prefix_weight=args.flow_prefix_weight,
                        tail_weight=args.flow_tail_weight,
                    )
                    if args.single_task and not is_fork_batch:
                        pair_loss = flow_loss.new_zeros(())
                        pred_delta = flow_loss.new_zeros(())
                        tgt_delta = flow_loss.new_zeros(())
                    else:
                        partner = paired_partner_indices(batch["pair_id"], batch["instruction_id"])
                        if args.fork_data is not None and not args.fork_skip_contract:
                            # 生死门契约断言（Q5b③）：fork 对必须同帧同 proprio/prev
                            # （语言是 pair 内唯一输入差）。E 组打乱数据跳过。
                            if not (
                                torch.equal(
                                    batch["proprio"], batch["proprio"][partner]
                                )
                                and torch.equal(
                                    batch["previous_action"],
                                    batch["previous_action"][partner],
                                )
                                and torch.allclose(
                                    batch["vision_tokens"],
                                    batch["vision_tokens"][partner],
                                    atol=1e-4,
                                    rtol=1e-4,
                                )
                            ):
                                raise ValueError(
                                    "fork pair contract violated: proprio/previous_action/"
                                    "vision_tokens must be identical within each pair"
                                )
                        if args.pair_mode == "legacy":
                            # 旧 L_pair：tau=0 共享噪声 delta-only（消融对照）。
                            pair_noise, pair_time, pair_target_velocity = sample_pair_intervention(
                                batch["actions"], partner, probe_tau_max=0.0
                            )
                            pair_predicted_velocity = model.flow_velocity(
                                action_conditions[:, 0], pair_noise, pair_time
                            )
                            if servo is not None:
                                # Step 2：pair 探针与主分支同一策略输出——t=0
                                # 伺服修正（确定性重算，与 rollout_policy 同值）。
                                pair_predicted_velocity = (
                                    pair_predicted_velocity
                                    + servo_correction_t0(model, servo, batch, device)[
                                        :, None, :
                                    ]
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
                            if servo is not None:
                                # Step 2：pair 探针与主分支同一策略输出——t=0
                                # 伺服修正（确定性重算，与 rollout_policy 同值）。
                                pair_predicted_velocity = (
                                    pair_predicted_velocity
                                    + servo_correction_t0(model, servo, batch, device)[
                                        :, None, :
                                    ]
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
                # no_grad（encode_prior_states 带 @torch.no_grad + LoRA 关闭，
                # P0-1：旧实现 prior 实际走 LoRA，anchor 恒为零）；adapted 单次
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
                    # P0-4：scene 条件路径同样参与 anchor/geometry——用 rollout
                    # t=0 编译的真实场景输入（与 compiler 前向逐位一致）。
                    compiler = getattr(e2e_model, "compiler", None)
                    scene_inputs = getattr(e2e_model, "_compile_scene_inputs", None)
                    if (
                        args.compile_task
                        and compiler is not None
                        and scene_inputs is not None
                    ):
                        # scene 输入从 rollout 图 detach：场景 anchor 只反传到
                        # compiler/Qwen（视觉侧梯度由动作损失的 rollout 图提供），
                        # 且避免与 action_total 共享图边导致拆分 backward 时
                        # "backward through the graph a second time"（P0-5）。
                        scene_tokens0, history0, delta0 = (
                            tensor.detach() for tensor in scene_inputs
                        )
                        scene_tokens = pool_flat_tokens(
                            scene_tokens0, e2e_model.n_scene_tokens
                        )
                        s_prior, s_adapted, s_mask = compiler.scene_states(
                            semantic,
                            mb["instruction"],
                            scene_tokens,
                            history0,
                            delta0,
                            layers_needed,
                        )
                        if need_anchor:
                            semantic_anchor_loss = semantic_anchor_loss + semantic.anchor_loss(
                                s_prior, s_adapted
                            )
                        if need_geom:
                            semantic_geom_loss = semantic_geom_loss + semantic.geometry_loss(
                                s_prior[semantic.num_layers - 1],
                                s_adapted[semantic.num_layers - 1],
                                s_mask,
                            )
            evsm_gate_mean = (
                sum(evsm_gates) / len(evsm_gates) if evsm_gates else None
            )
            # P0-5：动作损失与语义损失分开返回——backward 时 LoRA 参数只缩放
            # 动作侧梯度（η_act），anchor/geometry 梯度完整。
            wmrm_loss = getattr(model, "last_wmrm_loss", None)
            if wmrm_loss is None:
                wmrm_loss = flow_loss.new_zeros(())
            pi_kl = getattr(model, "last_wmrm_pi_kl_loss", None)
            if pi_kl is None:
                pi_kl_hinge = flow_loss.new_zeros(())
            else:
                margin = float(getattr(args, "wmrm_pi_kl_margin", 0.1))
                pi_kl_hinge = torch.relu(margin - pi_kl)
            lang_align = flow_loss.new_zeros(())
            last_aux = getattr(model, "last_wmrm", None)
            if last_aux is not None and getattr(last_aux, "task_summary", None) is not None:
                lang_align = 1.0 - F.cosine_similarity(
                    last_aux.belief.mean(dim=1),
                    last_aux.task_summary.detach(),
                    dim=-1,
                ).mean()
            adep = getattr(model, "last_wmrm_adep_loss", None)
            if adep is None:
                adep = flow_loss.new_zeros(())
            med = getattr(model, "last_wmrm_med_loss", None)
            if med is None:
                med = flow_loss.new_zeros(())
            if getattr(args, "wmrm_only", False):
                action_total = (
                    float(getattr(args, "wmrm_world_weight", 1.0)) * wmrm_loss
                    + float(getattr(args, "wmrm_pi_kl_weight", 0.0)) * pi_kl_hinge
                    + float(getattr(args, "wmrm_lang_align_weight", 0.0)) * lang_align
                    + float(getattr(args, "wmrm_adep_weight", 0.0)) * adep
                )
            else:
                action_total = (
                    flow_loss
                    + args.pair_loss_weight * pair_loss
                    + args.future_predict_weight * future_loss
                    + float(getattr(args, "wmrm_world_weight", 1.0)) * wmrm_loss
                    + float(getattr(args, "wmrm_pi_kl_weight", 0.0)) * pi_kl_hinge
                    + float(getattr(args, "wmrm_lang_align_weight", 0.0)) * lang_align
                    + float(getattr(args, "wmrm_adep_weight", 0.0)) * adep
                    + float(getattr(args, "wmrm_med_weight", 0.0)) * med
                )
            semantic_total = (
                args.semantic_anchor_weight * semantic_anchor_loss
                + args.semantic_geometry_weight * semantic_geom_loss
            )
            return (
                action_total,
                action_total + semantic_total,
                flow_loss,
                flow_prefix_loss,
                flow_tail_loss,
                pair_loss,
                pred_delta,
                tgt_delta,
                future_loss,
                evsm_gate_mean,
                semantic_anchor_loss,
                semantic_geom_loss,
            )

        # No accumulation is used on this path.  Release the previous update's
        # gradients before constructing the next 8-stage World graph; keeping
        # them through the forward needlessly raises the transient CUDA peak.
        optimizer.zero_grad(set_to_none=True)
        with feature_policy_autocast(
            device, bool(args.feature_autocast_bf16)
        ):
            (
                action_total,
                total_loss,
                flow_loss,
                flow_prefix_loss,
                flow_tail_loss,
                pair_loss,
                predicted_delta,
                target_delta,
                future_loss,
                evsm_gate_mean,
                semantic_anchor_loss,
                semantic_geom_loss,
            ) = compute_loss(batch, noisy_actions, flow_time)

        validate_finite_update_scalars(
            [
                ("total", total_loss),
                ("action", action_total),
                ("flow", flow_loss),
                ("flow_prefix", flow_prefix_loss),
                ("flow_tail", flow_tail_loss),
                ("pair", pair_loss),
                ("future", future_loss),
                ("semantic_anchor", semantic_anchor_loss),
                ("semantic_geometry", semantic_geom_loss),
            ]
        )
        # P0-5：动作损失与语义损失分开 backward——LoRA 参数只缩放动作侧梯度
        # （η_act），anchor/geometry 梯度完整（旧实现统一缩放两者）。
        action_total.backward()
        # 双数据流视觉辅助批次（阶段 C）：每 N 步一个在线仿真批次，辅助 loss
        # 累积到同一优化器 step；辅助分支只反传 metric head，且视觉头与
        # VA/relation 分别 clip，避免大辅助梯度压小动作学习信号。
        aux_parts: dict[str, float] = {}
        if prepared_visual_aux is not None:
            aux_task, aux_rng, aux_sim_batch = prepared_visual_aux
            if getattr(args, "dino_main_vision", False):
                # DINO 版视觉辅助（2026-08-16 移植）：仿真真值 → DINO dense
                # evidence → metric head（grid=16）。V-JEPA 路径保持原样。
                aux_loss, aux_parts = _dino_visual_aux_loss(
                    main_vision_backbone,
                    metric_head,
                    aux_task,
                    aux_rng,
                    args.mtvj_visual_aux_batch,
                    lang_aux_cache,
                    device,
                    loc_lambda=args.mtvj_visual_aux_loc_lambda,
                    vis_lambda=args.mtvj_visual_aux_vis_lambda,
                    sim_batch=aux_sim_batch,
                )
            else:
                aux_loss, aux_parts = _mtvj_visual_aux_loss(
                    mtvj_backbone,
                    metric_head,
                    aux_task,
                    aux_rng,
                    args.mtvj_visual_aux_batch,
                    lang_aux_cache,
                    device,
                    loc_lambda=args.mtvj_visual_aux_loc_lambda,
                    vis_lambda=args.mtvj_visual_aux_vis_lambda,
                    sim_batch=aux_sim_batch,
                )
            validate_finite_update_scalars([("visual_aux", aux_loss)])
            aux_loss.backward()
        if e2e_model is not None and args.semantic_adapter:
            scale_semantic_lora_grads(
                e2e_model.text_backbone, args.semantic_act_grad_scale
            )
        if args.semantic_adapter and (
            args.semantic_anchor_weight > 0.0 or args.semantic_geometry_weight > 0.0
        ):
            semantic_total.backward()
        clip_params = (
            e2e_model.parameters()
            if e2e_model is not None
            else (
                [*model.parameters(), *scene_teacher.parameters()]
                if scene_teacher is not None
                else (
                    [*model.parameters(), *vision_backbone.parameters()]
                    if vision_backbone is not None
                    else model.parameters()
                )
            )
        )
        if servo is not None:
            clip_params = [*clip_params, *servo.parameters()]
        relation_gradient_norm = None
        if relation_encoder is not None and args.mtvj_train_relation:
            relation_gradient_norm = _module_action_gradient_norm(
                relation_encoder, "--mtvj-train-relation", device
            )
            clip_params = [
                *clip_params,
                *(p for p in relation_encoder.parameters() if p.requires_grad),
            ]
        metric_head_gradient_norm = None
        metric_clip_params: list[Tensor] = []
        if metric_head is not None and args.mtvj_train_metric_head:
            metric_head_gradient_norm = _module_action_gradient_norm(
                metric_head, "--mtvj-train-metric-head", device
            )
            # Visual auxiliary updates can be much larger than the flow loss.
            # Clip the metric head independently so an auxiliary step cannot
            # shrink the VA/relation gradients through one shared global norm.
            metric_clip_params = [
                p for p in metric_head.parameters() if p.requires_grad
            ]
        update_named_parameters = list(
            named_optimizer_parameters(
                optimizer,
                ("e2e_model", e2e_model),
                ("model", model),
                ("scene_teacher", scene_teacher),
                ("vision_backbone", vision_backbone),
                ("servo", servo),
                ("relation_encoder", relation_encoder),
                ("metric_head", metric_head),
            )
        )
        validate_optimizer_update_state(optimizer, validate_values=False)
        validate_update_gradients(
            update_named_parameters, max_norm=args.max_gradient_norm
        )
        main_parameter_ids = {id(parameter) for parameter in clip_params}
        main_named_parameters = [
            (name, parameter)
            for name, parameter in update_named_parameters
            if id(parameter) in main_parameter_ids
        ]
        gradient_norm = clip_update_gradients(main_named_parameters, max_norm=1.0)
        metric_named_parameters = [
            (name, parameter)
            for name, parameter in update_named_parameters
            if id(parameter) in {id(p) for p in metric_clip_params}
        ]
        metric_clip_norm = (
            clip_update_gradients(metric_named_parameters, max_norm=1.0)
            if metric_named_parameters
            else None
        )
        if args.sam_rho > 0:
            # SAM：先沿梯度方向扰动权重（worst-case 邻域），重算 loss 后走真实步。
            # η_act 对两次 backward 都按动作/语义拆分缩放（见 scale_semantic_lora_grads
            # 文档：first_step 的扰动方向与缩放无关——ρ·g/‖g‖ 与缩放无关；实际步长
            # 由第二次缩放后的梯度决定，因此两次缩放才使 η_act 对 SAM 生效）。
            optimizer.first_step(zero_grad=True)
            try:
                with feature_policy_autocast(
                    device, bool(args.feature_autocast_bf16)
                ):
                    second_losses = compute_loss(batch, noisy_actions, flow_time)
                action_total2, semantic_total2 = second_losses[:2]
                validate_finite_update_scalars(
                    [("sam.total", second_losses[1]), ("sam.action", action_total2)]
                )
                action_total2.backward()
                if e2e_model is not None and args.semantic_adapter:
                    scale_semantic_lora_grads(
                        e2e_model.text_backbone, args.semantic_act_grad_scale
                    )
                if args.semantic_adapter and (
                    args.semantic_anchor_weight > 0.0
                    or args.semantic_geometry_weight > 0.0
                ):
                    semantic_total2.backward()
                validate_optimizer_update_state(optimizer, validate_values=False)
                validate_update_gradients(
                    update_named_parameters,
                    max_norm=args.max_gradient_norm,
                )
                clip_update_gradients(main_named_parameters, max_norm=1.0)
                if metric_named_parameters:
                    clip_update_gradients(metric_named_parameters, max_norm=1.0)
            except Exception:
                optimizer.restore_step(zero_grad=True)
                raise
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
        validate_optimizer_update_state(optimizer)
        commit_successful_update(step, consumed_locality_batch)
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
        relation_log = (
            f" rel_grad={float(relation_gradient_norm):.6f}"
            if relation_gradient_norm is not None
            else ""
        )
        metric_head_log = (
            f" metric_grad={float(metric_head_gradient_norm):.6f}"
            if metric_head_gradient_norm is not None
            else ""
        )
        aux_log = ""
        if aux_parts:
            aux_log = (
                f" aux_total={aux_parts.get('total', 0.0):.4f}"
                f" aux_hinge={aux_parts.get('hinge', 0.0):.4f}"
                f" aux_pos={aux_parts.get('pos', 0.0):.4f}"
                f" aux_offset={aux_parts.get('offset', 0.0):.4f}"
                f" aux_vis={aux_parts.get('vis', 0.0):.4f}"
                f" aux_rmse={aux_parts.get('rmse_px', 0.0):.1f}px"
            )
        if metric_clip_norm is not None:
            metric_head_log += f" metric_clip={float(metric_clip_norm):.6f}"
        servo_log = ""
        if servo_stats is not None:
            # Step 2 伺服运行日志：信任缩放 β / 重读触发率 / 假设熵 / 修正幅度 / 阶段分布。
            stage = servo_stats["stage"]  # [B, T] long
            counts = {
                name: int((stage == idx).sum())
                for idx, name in (
                    (0, "coarse"),
                    (1, "appr"),
                    (2, "pre"),
                    (3, "contact"),
                    (4, "unc"),
                )
            }
            servo_log = (
                f" servo[β={float(servo_stats['beta'].mean()):.3f} "
                f"flag={float(servo_stats['innovation_flag'].mean()):.2f} "
                f"H={float(servo_stats['hyp_entropy'].mean()):.2f} "
                f"|Δa|={float(servo_stats['correction'].norm(dim=-1).mean()):.4f} "
                f"stage={'/'.join(f'{k}:{v}' for k, v in counts.items())}]"
            )
        task_ids = sorted(
            int(value)
            for value in torch.unique(batch["instruction_id"]).detach().cpu()
        )
        task_log = "/".join(
            task_log_names.get(value, str(value)) for value in task_ids
        )
        valid_fraction = effective_action_valid_fraction(
            batch, batch["actions"]
        ).detach()
        world_task_log = ""
        visual_metrics = getattr(model, "last_visual_world_metrics", {}) or {}
        if visual_metrics:
            task_parts = []
            for task_id, metrics in sorted(visual_metrics.items()):
                stage_text = ",".join(
                    f"{float(value):.5f}" for value in metrics["stage_losses"]
                )
                task_parts.append(
                    f"{task_log_names.get(task_id, str(task_id))}:"
                    f"all={metrics['world_all']:.5f}/{metrics['copy_all']:.5f} "
                    f"gain={metrics['gain_all']:.5f} "
                    f"motion={metrics['world_motion']:.5f}/{metrics['copy_motion']:.5f} "
                    f"mgain={metrics['gain_motion']:.5f} "
                    f"top10={metrics['world_top10']:.5f}/{metrics['copy_top10']:.5f} "
                    f"rel={metrics['relative_gain_top10']:.3f} "
                    f"static={metrics['world_static']:.5f}/{metrics['copy_static']:.5f} "
                    f"n={metrics['transitions']} energy={metrics['motion_energy']:.5f} "
                    f"stages={stage_text}"
                )
            world_task_log = " world_task[" + " | ".join(task_parts) + "]"
        world_constraint_log = ""
        if getattr(model, "last_world_no_regression_loss", None) is not None:
            world_constraint_log = (
                f" world_base={float(model.last_wmrm_base_loss):.6f}"
                f" world_guard={float(model.last_world_no_regression_loss):.6f}"
                " world_static_constraint="
                f"{float(model.last_world_static_constraint_loss):.6f}"
                f" world_action_rank={float(model.last_world_action_rank_loss):.6f}"
            )
        resources = runtime_resource_stats(device)
        resource_log = (
            f" resources[rss={resources['rss_mib']:.1f}MiB "
            f"fd={resources['fd_count']} "
            f"cuda={resources['cuda_allocated_mib']:.1f}/"
            f"{resources['cuda_reserved_mib']:.1f}MiB]"
        )
        print(
            f"step={global_step} mode={args.mode} contract="
            f"{'e2e_single' if e2e_model is not None else ('single' if args.single_task else 'paired')} "
            f"task={task_log} action_valid={float(valid_fraction):.4f} "
            f"sequence={noisy_actions.shape[1]} "
            f"loss={total_loss.item():.6f} flow={flow_loss.item():.6f} "
            f"flow_first{min(args.flow_prefix_steps, noisy_actions.shape[-2])}="
            f"{flow_prefix_loss.item():.6f} "
            f"flow_tail{max(noisy_actions.shape[-2] - args.flow_prefix_steps, 0)}="
            f"{flow_tail_loss.item():.6f} "
            f"pair={pair_loss.item():.6f} future={future_loss.item():.6f} "
            f"world={float((getattr(model, 'last_wmrm_loss', None) if model is not None else None) or 0.0):.6f} "
            f"goal_delta={predicted_delta.item():.6f}/"
            f"{target_delta.item():.6f} grad={float(gradient_norm):.6f}"
            f"{relation_log}{metric_head_log}{aux_log}{gate_log}{semantic_log}"
            f"{compile_step_log}{servo_log}{world_task_log}{resource_log}"
            f"{world_constraint_log}"
        )

    if final_checkpoint_save_due(
        args.save, global_step, last_saved_global_step
    ):
        save_checkpoint(
            args,
            config,
            model,
            e2e_model,
            scene_teacher,
            vision_backbone,
            servo=servo,
            relation_encoder=relation_encoder,
            metric_head=metric_head,
            roi_head=roi_head,
            optimizer=optimizer,
            global_step=global_step,
            sampler=sampler,
            exact_run_contract=runtime_exact_run_contract,
        )


if __name__ == "__main__":
    main()
