#!/usr/bin/env python
"""Dedicated LIBERO H50/P15 VA+WM+PCGrad trainer.

Fresh runs initialize the LIBERO policy, VA, WM, Flow, and fusion modules from
scratch.  Only the Qwen and DINO backbone weights are pretrained.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from va_compound.training.batch import ensure_sequence, feature_policy_autocast, move_batch
from va_compound.training.gradients import (
    backward_pcgrad,
    merge_separate_pcgrad_gradients,
    pop_update_gradients,
)
from va_compound.training.prefetch import (
    PeerJointBatchPrefetcher,
    peer_prefetch_fill_limit,
    peer_prefetch_must_wait_for_commit,
)
from va_compound.data.samplers import TaskLocalityWeightedSampler
from va_compound.data.episode_stream import EpisodeStreamDataset, EpisodeWindowBatchSampler
from va_compound.data.all_starts import (
    AllStartsStreamDataset,
    AllStartsWindowBatchSampler,
    MEMORY_CONTRACT as ALL_STARTS_MEMORY_CONTRACT,
    SAMPLING_CONTRACT as ALL_STARTS_SAMPLING_CONTRACT,
)
from va_compound.training.episode_memory import EpisodeMemoryBank
from va_compound.training.gradient_microbatch import task_microbatch_forward
from va_compound.training.libero_transfer import is_all_starts_transfer, validate_transfer_source, TRANSFER_INITIALIZATION
from va_compound.vision.dual_tower_batch import encode_dual_tower_batch
from va_compound.vision.encoding import _dino_main_online_encode
from va_compound.utils.exact_resume import _sha256_file
from va_compound.training.rollout import rollout_policy

from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.backbones import QwenTextBackbone, TimmActionVisionBackbone
from va_compound.data_parallel import (
    barrier,
    broadcast_parameters,
    initialize,
    reduce_scalar_mean,
    resolve_world_topology,
    shutdown,
)
from va_compound.longtraj_frames import LongTrajFramesDataset, mtvj_collate
from va_compound.utils.flow import masked_flow_matching_loss, sample_flow_matching_inputs

from va_compound.data.libero import (
    ACTION_HORIZON,
    CROSS_MODAL_VA_LAYERS,
    DATA_CONTRACT,
    DECISION_OFFSETS,
    EXECUTION_HORIZON,
    FRESH_INITIALIZATION,
    FUSION_LAYERS,
    H15_DATA_CONTRACT,
    ALL_STARTS_DATA_CONTRACT,
    JOINT_DATA_CONTRACT,
    LIBERO_SUITES,
    SEQUENCE,
    SOURCE_DATA_CONTRACT,
    SOURCE_INITIALIZATION,
    VISION_OFFSETS,
    WORLD_HORIZON,
    _StaticAnchorDataset,
    _attach_dense_world_action_donors,
    _local_task_ids,
    _normalize,
    _official_task_specs,
    _suite_names,
    _validate_data,
    _validate_dense_source,
    _validate_run_schedule,
)


def validate_cross_modal_language_contract(metadata: dict) -> None:
    if (
        metadata.get("qwen_keep_layers") != 24
        or metadata.get("qwen_fusion_layers") != FUSION_LAYERS
        or metadata.get("qwen_base_readout") != "layer23_final_norm"
        or metadata.get("qwen_fusion_reduce") != "none"
        or metadata.get("language_dim") != 1024
        or metadata.get("language_source")
        != "online_qwen35_0_8b_last6_full_v1"
    ):
        raise ValueError("DINO/Qwen bridge requires full Qwen layers 18-23")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/root/libero_spatial_ora0_v1/"
            "libero_4suite_h50p15_t4_dualview5_maskedtail_v2.pt"
        ),
    )
    parser.add_argument(
        "--longtraj",
        type=Path,
        default=Path("/root/libero_spatial_ora0_v1/longtraj"),
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=Path("/root/libero_spatial_ora0_v1/datasets"),
    )
    parser.add_argument("--suites", default=",".join(LIBERO_SUITES))
    parser.add_argument(
        "--local-task-ids",
        help="Comma-separated local task ids; requires exactly one --suites entry.",
    )
    parser.add_argument(
        "--language-reference",
        type=Path,
        default=Path(
            "/root/libero_spatial_ora0_v1/"
            "libero_4suite_qwen35_2b_l0_14_mean10_14_v1.pt"
        ),
    )
    parser.add_argument(
        "--dino",
        type=Path,
        default=Path("/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors"),
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path("/root/models/Qwen3.5-0.8B"),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(
            "/root/ora0_ckpts/"
            "libero_4suite_scratch_h50p15_twostage_e50_b32_v1.pt"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-weights", type=Path)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--windows-per-demo", type=int, default=16)
    parser.add_argument("--dense-windows", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mixed-tasks", type=int, default=4)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-new", type=float, default=3e-5)
    parser.add_argument("--lr-qwen", type=float, default=1e-6)
    parser.add_argument("--lr-dino", type=float, default=1e-6)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prev-dropout", type=float, default=1.0)
    parser.add_argument("--encode-batch", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--va-last3-cross-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dino-qwen-cross-modal-bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--architecture-version",
        choices=("legacy", "dual_tower_expert_v1", "dual_tower_h15_v1"),
        default="legacy",
        help="Architecture version for LIBERO policy training.",
    )
    parser.add_argument(
        "--world-state-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for World state delta transition prediction.",
    )
    parser.add_argument(
        "--window-sampling",
        choices=("episode_contiguous_p15_v1", "all_starts_random_tbptt8_v1"),
        default="episode_contiguous_p15_v1",
        help="Window sampling strategy for LIBERO joint training.",
    )
    parser.add_argument(
        "--episode-microbatch",
        type=int,
        default=1,
        help="Microbatch size for joint episode window rollout execution.",
    )
    parser.add_argument(
        "--joint-observation-chunk",
        type=int,
        default=0,
        help="Chunk size for joint frontend dual-tower observation encoding (0 disables chunking).",
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Epoch offset for all-starts stream sampling.",
    )
    parser.add_argument("--gradient-microbatch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=2)
    return parser


def _fresh_config(
    *,
    va_last3_cross_attn: bool = False,
    dino_qwen_cross_modal_bridge: bool = False,
    architecture_version: str = "legacy",
    world_state_loss_weight: float = 1.0,
) -> VACompoundConfig:
    is_h15 = architecture_version == "dual_tower_h15_v1"
    is_joint = architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
    return VACompoundConfig(
        architecture_version=architecture_version,
        fusion_pair_count=len(FUSION_LAYERS),
        language_dim=1024,
        vision_dim=1024,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        action_horizon=15 if is_h15 else ACTION_HORIZON,
        action_dim=7,
        proprio_dim=9,
        world_state_supervision=True if is_joint else False,
        world_state_loss_weight=world_state_loss_weight,
        flow_layers=3 if is_joint else 6,
        flow_hidden_dim=512,
        dropout=0.0,
        mode="bidir_va",
        qk_norm=False,
        attention_variant="flat",
        va_attention_backend="auto",
        planning_stride=EXECUTION_HORIZON,
        deployment_execution_horizon=EXECUTION_HORIZON,
        wmrm=True,
        wmrm_full_language_tokens=True,
        wmrm_reads_fused_va_tokens=True,
        wmrm_world_dim=1024,
        wmrm_inject="all",
        wmrm_target="dino",
        wmrm_cycle_steps=WORLD_HORIZON,
        wmrm_map_size=16,
        wmrm_map_channels=1024,
        wmrm_world_grid=16,
        wmrm_predictor="st_blocks",
        wmrm_predictor_depth=6,
        wmrm_predictor_width=1024,
        wmrm_predictor_heads=32,
        wmrm_predictor_copies=1,
        runtime_integrity_checks=False,
        va_world_mode="peer_sync_h6",
        sequential_coupling=0,
        flow_cond="adaln",
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        main_vision_image_size=224,
        main_vision_dim=1024,
        main_vision_grid=16,
        main_vision_tokens=1280,
        main_vision_frames=5,
        main_vision_temporal=True,
        main_vision_temporal_scale=1.0,
        slot_free_policy=True,
        va_last3_cross_attn=va_last3_cross_attn if architecture_version == "legacy" else False,
        dino_qwen_cross_modal_bridge=dino_qwen_cross_modal_bridge if architecture_version == "legacy" else False,
        dino_qwen_cross_modal_layers=len(FUSION_LAYERS),
    )


def _initialize_scratch_fusion(model: VACompoundPolicy) -> None:
    """Identity at step zero, but with a full-matrix learning path."""
    if getattr(getattr(model, "config", None), "architecture_version", "legacy") in ("dual_tower_expert_v1", "dual_tower_h15_v1"):
        return
    if model.dino_qwen_bridge is None or model.va_last3_readout is None:
        raise ValueError("scratch LIBERO requires both fusion readers")
    readers = [
        *model.dino_qwen_bridge.vision_readers,
        *model.dino_qwen_bridge.language_readers,
        *model.va_last3_readout.readers,
    ]
    with torch.no_grad():
        for reader in readers:
            reader.attn.out_proj.weight.zero_()
            reader.attn.out_proj.bias.zero_()
            reader.ff[-1].weight.zero_()
            reader.ff[-1].bias.zero_()
        model.dino_qwen_bridge.gates.fill_(1.0)
        model.va_last3_readout.gates.fill_(1.0)


def _validate_model_contract(config: VACompoundConfig) -> None:
    expected = {
        "action_horizon": 15 if config.architecture_version == "dual_tower_h15_v1" else 50,
        "action_dim": 7,
        "proprio_dim": 9,
        "language_dim": 1024,
        "hidden_dim": 1024,
        "num_heads": 16,
        "flow_hidden_dim": 512,
        "planning_stride": 15,
        "deployment_execution_horizon": 15,
        "wmrm_cycle_steps": WORLD_HORIZON,
        "num_layers": 8,
        "wmrm": True,
        "wmrm_full_language_tokens": True,
        "wmrm_reads_fused_va_tokens": True,
        "va_world_mode": "peer_sync_h6",
        "wmrm_predictor": "st_blocks",
        "wmrm_world_dim": 1024,
        "wmrm_predictor_width": 1024,
        "wmrm_predictor_heads": 32,
        "flow_cond": "adaln",
        "va_attention_backend": "auto",
        "main_vision_backbone": "dinov2_vitl14_reg4",
        "main_vision_dim": 1024,
        "main_vision_grid": 16,
        "main_vision_frames": 5,
        "main_vision_tokens": 1280,
        "main_vision_temporal": True,
        "va_last3_cross_attn": True,
        "dino_qwen_cross_modal_bridge": True,
        "dino_qwen_cross_modal_layers": len(FUSION_LAYERS),
    }
    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1"):
        expected.update(
            va_last3_cross_attn=False,
            dino_qwen_cross_modal_bridge=False,
            flow_layers=3,
            fusion_pair_count=6,
            tail_flow_condition_grad=False,
            world_state_supervision=True,
        )
    mismatch = {
        key: (getattr(config, key), value)
        for key, value in expected.items()
        if getattr(config, key) != value
    }
    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1"):
        weight = getattr(config, "world_state_loss_weight", None)
        if weight is None or not math.isfinite(weight) or weight < 0:
            mismatch["world_state_loss_weight"] = (weight, "finite >= 0")
    if mismatch:
        raise ValueError(f"LIBERO H{config.action_horizon}/P15 model contract mismatch: {mismatch}")


def preflight(args: argparse.Namespace) -> None:
    for path in (args.data, args.longtraj, args.dino, args.qwen):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.resume is not None and args.resume_weights is not None:
        raise ValueError("choose --resume or --resume-weights, not both")
    if args.resume_weights is not None and getattr(args, "architecture_version", "legacy") != "legacy" and not is_all_starts_transfer(args):
        arch = getattr(args, "architecture_version", "legacy")
        raise ValueError(f"{arch} requires fresh initialization or same-version --resume; --resume-weights is legacy-only")
    if args.resume is None and args.resume_weights is None and getattr(args, "architecture_version", "legacy") == "legacy":
        raise ValueError("dense continuation requires --resume-weights")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.resume_weights is not None:
        if not args.resume_weights.is_file():
            raise FileNotFoundError(args.resume_weights)
        (validate_transfer_source if is_all_starts_transfer(args) else _validate_dense_source)(
            torch.load(
                args.resume_weights,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        )
    is_all_starts_preflight = getattr(args, "window_sampling", None) == ALL_STARTS_SAMPLING_CONTRACT
    payload = _validate_data(
        args.data,
        architecture_version=getattr(args, "architecture_version", "legacy"),
        **({"window_sampling": ALL_STARTS_SAMPLING_CONTRACT} if is_all_starts_preflight else {}),
    )
    if not is_all_starts_preflight and payload.get("metadata", {}).get("contract") == ALL_STARTS_DATA_CONTRACT:
        raise ValueError("all-starts data contract requires --window-sampling all_starts_random_tbptt8_v1")
    if is_all_starts_preflight and getattr(args, "architecture_version", "legacy") != "dual_tower_h15_v1":
        raise ValueError("all_starts_random_tbptt8_v1 is only supported with dual_tower_h15_v1")
    if args.prev_dropout != 1.0:
        raise ValueError("the H50/P15 run requires --prev-dropout 1")
    if args.lr != 1e-5 or args.lr_new != 3e-5:
        raise ValueError("the all-fixes run requires --lr 1e-5 --lr-new 3e-5")
    if args.lr_qwen != 1e-6 or args.lr_dino != 1e-6:
        raise ValueError("the two-stage run requires Qwen/DINO lr1e-6")
    qwen_config = json.loads((args.qwen / "config.json").read_text())
    text_config = qwen_config.get("text_config") or {}
    if (
        text_config.get("num_hidden_layers") != 24
        or text_config.get("hidden_size") != 1024
    ):
        raise ValueError("--qwen must be the full 24-layer Qwen3.5-0.8B")
    _qwen_weight_file(args.qwen)
    if not math.isfinite(args.world_state_loss_weight) or args.world_state_loss_weight < 0:
        raise ValueError("--world-state-loss-weight must be finite and >= 0")
    if args.gradient_microbatch < 0 or (args.gradient_microbatch and (args.mixed_tasks != 1 or args.architecture_version != "dual_tower_h15_v1")):
        raise ValueError("gradient microbatch requires single-task H15 and a nonnegative size")
    if args.episode_microbatch < 1:
        raise ValueError("--episode-microbatch must be >= 1")
    if args.joint_observation_chunk < 0:
        raise ValueError("--joint-observation-chunk must be >= 0")
    config = _fresh_config(
        va_last3_cross_attn=args.va_last3_cross_attn,
        dino_qwen_cross_modal_bridge=args.dino_qwen_cross_modal_bridge,
        architecture_version=getattr(args, "architecture_version", "legacy"),
        world_state_loss_weight=args.world_state_loss_weight,
    )
    _validate_model_contract(config)
    if config.dino_qwen_cross_modal_bridge:
        validate_cross_modal_language_contract(payload.get("metadata") or {})
    VACompoundPolicy(config)
    if args.batch_size % args.gpus or args.batch_size % args.mixed_tasks:
        raise ValueError("global batch must divide across GPUs and mixed tasks")
    if is_all_starts_transfer(args):
        if args.stage1_steps not in (None, 0):
            raise ValueError("H15 transfer requires stage1_steps=0; trained DINO remains unfrozen")
        args.stage1_steps = 0
    if args.stage1_steps is None:
        if is_all_starts_preflight:
            temp_sampler = AllStartsWindowBatchSampler(
                payload,
                args.batch_size,
                seed=args.seed,
                mixed_tasks_per_batch=args.mixed_tasks,
                rank=0,
                world_size=args.gpus,
                epoch_offset=args.epoch_offset,
            )
            lengths = temp_sampler.epoch_lengths(max(2, args.epochs))
            args.stage1_steps = sum(lengths[:2])
        else:
            args.stage1_steps = 8000
    steps_per_epoch, total_steps, _ = _validate_run_schedule(payload, args)
    if is_all_starts_preflight:
        print(
            f"PASS T8 compact all-starts LIBERO H15 VA8+WM+PCGrad; rows={len(payload['actions'])} "
            f"tasks={payload['metadata']['n_tasks']} epoch1_steps={steps_per_epoch} "
            f"total_steps={total_steps}"
        )
    else:
        print(
            f"PASS T8 dense LIBERO H50/P15 VA8+WM+PCGrad; rows={len(payload['actions'])} "
            f"tasks={payload['metadata']['n_tasks']} steps/epoch={steps_per_epoch} "
            f"total_steps={total_steps}"
        )


def _prepare_batch(
    raw: dict,
    *,
    vision: TimmActionVisionBackbone,
    device: torch.device,
    encode_batch: int,
    world: bool,
    prev_dropout: float,
    layerwise_cross_modal: bool,
    cached_vision: Tensor | None = None,
    joint_text=None,
    joint_tasks=None,
    joint_fusion=None,
    joint_observation_chunk: int = 0,
) -> dict:
    raw = dict(raw)
    frames = raw.pop("frames")
    if isinstance(frames, Tensor):
        frames = frames.cpu().numpy()
    if joint_fusion is not None:
        if cached_vision is not None or layerwise_cross_modal:
            raise ValueError("joint frontend cannot use independent feature caches or legacy bridge")
        instructions = [joint_tasks[int(task_id)] for task_id in raw["instruction_id"]]
        encode_kwargs = {"grid": 16}
        if joint_observation_chunk:
            encode_kwargs["observation_chunk_size"] = joint_observation_chunk
        raw["vision_tokens"], raw["language_hidden"], raw["language_mask"] = encode_dual_tower_batch(
            frames, instructions, vision, joint_text, joint_fusion, device, **encode_kwargs,
        )
    else:
        encoded = cached_vision
        if encoded is None:
            encoded = _dino_main_online_encode(
                frames,
                vision,
                device,
                encode_batch=encode_batch,
                grid=16,
                window=5,
                return_last_layers=len(FUSION_LAYERS) if layerwise_cross_modal else 0,
            )
        if layerwise_cross_modal:
            raw["vision_tokens"], raw["dino_last4"] = encoded
        else:
            raw["vision_tokens"] = encoded
    target = raw.pop("world_target_frames", None)
    if world:
        if target is None:
            raise ValueError("World batch is missing endpoint frames")
        if isinstance(target, Tensor):
            target = target.cpu().numpy()
        with torch.no_grad():
            tokens = _dino_main_online_encode(
                target,
                vision,
                device,
                encode_batch=encode_batch,
                grid=16,
                window=1,
            )
        batch, sequence, patches, dim = tokens.shape
        raw["world_target_map"] = tokens.reshape(batch, sequence, 16, 16, dim).permute(0, 1, 4, 2, 3).detach()
    row_fields = {}
    for key in (
        "decision_valid_mask",
        "stream_id",
        "stream_active",
        "episode_id",
        "replay_id",
        "replay_offset",
        "crop_start",
        "decision_count",
        "episode_start",
        "episode_end",
        "world_rank_shuffle_action",
        "world_rank_shuffle_mask",
        "world_state_delta",
    ):
        if key in raw:
            row_fields[key] = raw.pop(key)
    prepared = ensure_sequence(move_batch(raw, device), SEQUENCE)
    for key, value in row_fields.items():
        prepared[key] = value.to(device=device) if isinstance(value, Tensor) else value
    if prev_dropout >= 1.0:
        prepared["previous_action"].zero_()
    elif prev_dropout > 0:
        drop = torch.rand(prepared["previous_action"].shape[0], device=device) < prev_dropout
        prepared["previous_action"] *= (~drop).view(-1, 1, 1)
    return prepared


def _rescale_task_loss_by_effective_count(
    loss: Tensor,
    local_count: Tensor | float,
    topology,
    device: torch.device,
) -> Tensor:
    if topology.world_size <= 1 or not topology.is_distributed:
        return loss
    import torch.distributed as dist

    local_tensor = (
        local_count.detach().clone().float().to(device)
        if isinstance(local_count, Tensor)
        else torch.tensor([float(local_count)], device=device, dtype=torch.float32)
    )
    if local_tensor.ndim == 0:
        local_tensor = local_tensor.unsqueeze(0)
    global_count = local_tensor.clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    scale = (local_tensor * float(topology.world_size) / global_count.clamp_min(1.0))[0]
    return loss * scale


def _split_raw_tasks(raw: dict) -> list[dict]:
    task_ids = raw["instruction_id"]
    if not isinstance(task_ids, Tensor) or task_ids.ndim != 1:
        raise ValueError("raw instruction_id must be a batch vector")
    result = []
    for task_id in torch.unique(task_ids, sorted=True):
        indices = torch.nonzero(task_ids == task_id).flatten()
        numpy_indices = indices.cpu().numpy()
        result.append(
            {
                key: (
                    value.index_select(0, indices)
                    if isinstance(value, Tensor)
                    and value.ndim
                    and value.shape[0] == len(task_ids)
                    else value[numpy_indices]
                    if isinstance(value, np.ndarray)
                    and value.ndim
                    and value.shape[0] == len(task_ids)
                    else value
                )
                for key, value in raw.items()
            }
        )
    return result


def _merge_raw_tasks(tasks: list[dict]) -> dict:
    if not tasks:
        raise ValueError("cannot merge an empty task group")
    return {
        key: (
            torch.cat([task[key] for task in tasks], dim=0)
            if isinstance(tasks[0][key], Tensor)
            else np.concatenate([task[key] for task in tasks], axis=0)
        )
        for key in tasks[0]
    }


def _encode_language(
    batch: dict,
    text_backbone: QwenTextBackbone,
    tasks: list[str],
    *,
    layerwise_cross_modal: bool,
) -> None:
    if batch["language_hidden"].ndim == 4:
        return
    task_ids, inverse = torch.unique(
        batch["instruction_id"], sorted=True, return_inverse=True
    )
    instructions = [tasks[int(task_id)] for task_id in task_ids]
    if layerwise_cross_modal:
        layers = FUSION_LAYERS
        hierarchy, mask = text_backbone.encode_trainable(
            instructions, output_layers=layers
        )
        norm = getattr(text_backbone.text_model, "norm", None)
        hidden = norm(hierarchy[23]) if norm is not None else hierarchy[23]
        batch["qwen_last4"] = torch.stack(
            [hierarchy[layer] for layer in layers], dim=1
        )[inverse]
    else:
        hidden, mask = text_backbone.encode_trainable(instructions)
    batch["language_hidden"] = hidden[inverse]
    batch["language_mask"] = mask[inverse]


def _selected_state(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _load_selected_state(module: torch.nn.Module, state: dict[str, Tensor]) -> None:
    parameters = dict(module.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(
            f"trainable state mismatch: missing={sorted(expected - set(state))[:5]} "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    with torch.no_grad():
        for name in expected:
            parameters[name].copy_(state[name])


def _qwen_weight_file(root: Path) -> Path:
    candidates = sorted(root.glob("model*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected one Qwen safetensors shard in {root}, got {len(candidates)}")
    return candidates[0]


def _atomic_checkpoint(
    path: Path,
    *,
    config: VACompoundConfig,
    model: VACompoundPolicy,
    text_backbone: QwenTextBackbone,
    vision: TimmActionVisionBackbone,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    stage1_steps: int,
    encode_batch: int,
    qwen_sha256: str,
    dino_sha256: str,
    data_sha256: str,
    run_metadata: dict,
    pcgrad_forward_grouping: str,
    source_checkpoint: str,
    source_global_step: int,
    sampler,
    world_sampler,
    episode_runtime_states: list[dict] | None = None,
    window_sampling: str = "episode_contiguous_p15_v1",
    epoch_lengths: list[int] | None = None,
    execution_config: dict | None = None,
) -> None:
    if window_sampling == ALL_STARTS_SAMPLING_CONTRACT:
        if config.architecture_version != "dual_tower_h15_v1" or not epoch_lengths or any(n < 1 for n in epoch_lengths):
            raise ValueError("all-start checkpoint requires H15 and positive epoch lengths")
        if sum(epoch_lengths) != total_steps:
            raise ValueError("all-start checkpoint total does not match epoch lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "config": config.__dict__,
            "model": model.state_dict(),
            "qwen_trainable_state_dict": _selected_state(text_backbone),
            "main_vision_trainable_state_dict": (
                _selected_state(vision) if step > stage1_steps else None
            ),
            "optimizer": optimizer.state_dict(),
            "global_step": step,
            "sampler_state": sampler.state_dict(),
            "world_sampler_state": world_sampler.state_dict(),
            "source_checkpoint": source_checkpoint,
            "source_global_step": source_global_step,
            **({"episode_runtime_states": episode_runtime_states} if episode_runtime_states is not None else {}),
            "training_contract": {
                "initialization": (
                    TRANSFER_INITIALIZATION
                    if window_sampling == ALL_STARTS_SAMPLING_CONTRACT and source_global_step >= 0
                    else "fresh_dual_tower_h15_v1"
                    if config.architecture_version == "dual_tower_h15_v1"
                    else "fresh_dual_tower_expert_v1"
                    if config.architecture_version == "dual_tower_expert_v1"
                    else FRESH_INITIALIZATION
                ),
                "source_checkpoint": source_checkpoint,
                "source_global_step": source_global_step,
                "optimizer_initialization": (
                    "continued_adamw_reset_streams_v1"
                    if window_sampling == ALL_STARTS_SAMPLING_CONTRACT and source_global_step >= 0
                    else "fresh_adamw_v1"
                    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
                    else "continued_from_source_v1"
                ),
                "suites": list(run_metadata["suites"]),
                "n_tasks": int(run_metadata["n_tasks"]),
                "task_specs": list(run_metadata["task_specs"]),
                "data_contract": (
                    ALL_STARTS_DATA_CONTRACT
                    if window_sampling == ALL_STARTS_SAMPLING_CONTRACT
                    else H15_DATA_CONTRACT
                    if config.architecture_version == "dual_tower_h15_v1"
                    else JOINT_DATA_CONTRACT
                    if config.architecture_version == "dual_tower_expert_v1"
                    else DATA_CONTRACT
                ),
                **(
                    {
                        "window_sampling": ALL_STARTS_SAMPLING_CONTRACT,
                        "sampling_contract": ALL_STARTS_SAMPLING_CONTRACT,
                        "epoch_lengths": list(epoch_lengths),
                    }
                    if window_sampling == ALL_STARTS_SAMPLING_CONTRACT and epoch_lengths is not None
                    else {}
                ),
                "data_sha256": data_sha256,
                "action_decoder": "conditional_flow_matching",
                "flow_steps": 8,
                "action_horizon": config.action_horizon,
                "sequence_length": SEQUENCE,
                "memory_reset_every": 0 if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1") else SEQUENCE,
                "planning_stride": EXECUTION_HORIZON,
                "deployment_execution_horizon": EXECUTION_HORIZON,
                "wmrm_cycle_steps": WORLD_HORIZON,
                "pcgrad": True,
                "pcgrad_scope": "per_task_action_and_world_separate_dino_guard_v1",
                "pcgrad_forward_grouping": pcgrad_forward_grouping,
                "peer_training_mode": "joint_dual_stream",
                "phase": (
                    "stage2_qwen_va_dino_joint"
                    if step > stage1_steps
                    else "stage1_qwen_va"
                ),
                "stage1_steps": stage1_steps,
                "total_steps": total_steps,
                "step_rng": "seed_plus_source_and_phase_step_times_world_plus_rank_v1",
                "qwen_joint_trained": True,
                "qwen_training": "last6_full_layers18_23_v1",
                "qwen_trainable_layers": FUSION_LAYERS,
                "qwen_final_norm_frozen": True,
                "qwen_keep_layers": 24,
                "qwen_hidden_dim": 1024,
                "qwen_fusion_layers": FUSION_LAYERS,
                "cross_modal_va_layers": CROSS_MODAL_VA_LAYERS,
                "qwen_base_readout": "layer23_final_norm",
                "qwen_fusion_reduce": "none",
                "qwen_gradient_checkpointing": False,
                "qwen_world_cache": (
                    "per_observation_joint_live_v1"
                    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
                    else "same_step_action_detached_v1"
                ),
                "qwen_base_sha256": qwen_sha256,
                "main_vision_joint_trained": step > stage1_steps,
                "main_vision_trainable_layers": FUSION_LAYERS,
                "main_vision_base_sha256": dino_sha256,
                "main_vision_grid": 16,
                "main_vision_frames": 5,
                "vision_input": "agentview_history4_plus_current_wrist_v2",
                "world_target_view": "eye_in_hand_rgb",
                "fusion_initialization": (
                    "dual_tower_zero_output_v1"
                    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
                    else "zero_output_unit_gate_v1"
                ),
                "architecture_version": config.architecture_version,
                **(
                    {
                        "execution_gradient_contract": (
                            "h15_unified_live_va_v1"
                            if config.architecture_version == "dual_tower_h15_v1"
                            else "p15_live_h50_tail_detached_v1"
                        ),
                        "memory_contract": (
                            ALL_STARTS_MEMORY_CONTRACT
                            if window_sampling == ALL_STARTS_SAMPLING_CONTRACT
                            else "episode_tbptt8_v1"
                        ),
                        "state_delta_contract": "joint7_gripper2_unclipped_q01q99_delta_h15_v1",
                        "world_state_loss_weight": config.world_state_loss_weight,
                    }
                    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
                    else {}
                ),
                "main_vision_encode_batch": encode_batch,
                "stage1_world_current_vision_cache": (
                    "disabled_joint_frontend_v1"
                    if config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
                    else "same_step_action_detached_v1"
                ),
                "flow_slot_identity": "per_slot_action_condition_v1",
                "flow_prefix_steps": EXECUTION_HORIZON,
                "flow_prefix_weight": 1.0 if config.architecture_version == "dual_tower_h15_v1" else 3.0,
                "flow_tail_weight": 0.0 if config.architecture_version == "dual_tower_h15_v1" else 1.0,
                "previous_action_input": "zero_v1",
                "dino_fusion_layers": FUSION_LAYERS,
                "dino_base_readout": "block23_norm",
                "dino_fusion_reduce": "none",
                "wmrm_feature_metric": "cosine",
                "wmrm_world_dim": 1024,
                "wmrm_predictor_width": 1024,
                "flow_hidden_dim": 512,
                "wmrm_target_teacher": "shared_online_dino_block23_stopgrad_v1",
                "wmrm_evidence": "post_va_vl_fused_tokens_v1",
                "lr_base": 1e-5,
                "lr_new": 3e-5,
                "lr_qwen": optimizer.param_groups[2]["lr"],
                "lr_dino": optimizer.param_groups[3]["lr"],
                "va_last3_cross_attn": config.va_last3_cross_attn,
                "dino_qwen_cross_modal_bridge": (
                    config.dino_qwen_cross_modal_bridge
                ),
                **({"execution_config": dict(execution_config)} if execution_config is not None else {}),
            },
        },
        temporary,
    )
    temporary.replace(path)


def _stage2_enabled(step: int, stage1_steps: int) -> bool:
    return step > stage1_steps


def _unfreeze_vision_tail(vision, *, joint_frontend: bool) -> None:
    vision.unfreeze_last(len(FUSION_LAYERS))
    if joint_frontend:
        vision.model.set_grad_checkpointing(False)


def train(args: argparse.Namespace) -> None:
    joint_frontend = getattr(args, "architecture_version", "legacy") in ("dual_tower_expert_v1", "dual_tower_h15_v1")
    if args.resume is not None and args.resume_weights is not None:
        raise ValueError("choose --resume or --resume-weights, not both")
    if args.resume_weights is not None and getattr(args, "architecture_version", "legacy") != "legacy" and not is_all_starts_transfer(args):
        arch = getattr(args, "architecture_version", "legacy")
        raise ValueError(f"{arch} requires fresh initialization or same-version --resume; --resume-weights is legacy-only")
    if args.resume is None and args.save.exists():
        raise FileExistsError(f"refusing to overwrite {args.save}")
    if args.reset_optimizer and args.resume is None:
        raise ValueError("--reset-optimizer requires --resume")
    if args.resume is None and args.resume_weights is None and getattr(args, "architecture_version", "legacy") == "legacy":
        raise ValueError("dense continuation requires --resume-weights")
    if args.prev_dropout != 1.0:
        raise ValueError("the H50/P15 run requires --prev-dropout 1")
    if args.lr != 1e-5 or args.lr_new != 3e-5:
        raise ValueError("the all-fixes run requires --lr 1e-5 --lr-new 3e-5")
    if args.gradient_microbatch < 0 or (args.gradient_microbatch and (args.mixed_tasks != 1 or args.architecture_version != "dual_tower_h15_v1")):
        raise ValueError("gradient microbatch requires single-task H15 and a nonnegative size")
    if args.episode_microbatch < 1:
        raise ValueError("--episode-microbatch must be >= 1")
    if args.joint_observation_chunk < 0:
        raise ValueError("--joint-observation-chunk must be >= 0")

    is_all_starts = getattr(args, "window_sampling", None) == ALL_STARTS_SAMPLING_CONTRACT
    if is_all_starts and getattr(args, "architecture_version", "legacy") != "dual_tower_h15_v1":
        raise ValueError("all_starts_random_tbptt8_v1 is only supported with dual_tower_h15_v1")

    payload = _validate_data(
        args.data,
        architecture_version=getattr(args, "architecture_version", "legacy"),
        **({"window_sampling": ALL_STARTS_SAMPLING_CONTRACT} if is_all_starts else {}),
    )
    if not is_all_starts and payload.get("metadata", {}).get("contract") == ALL_STARTS_DATA_CONTRACT:
        raise ValueError("all-starts data contract requires --window-sampling all_starts_random_tbptt8_v1")

    if is_all_starts_transfer(args):
        if args.stage1_steps not in (None, 0):
            raise ValueError("H15 transfer requires stage1_steps=0; trained DINO remains unfrozen")
        args.stage1_steps = 0
    if args.stage1_steps is None:
        if is_all_starts:
            temp_sampler = AllStartsWindowBatchSampler(
                payload,
                args.batch_size,
                seed=args.seed,
                mixed_tasks_per_batch=args.mixed_tasks,
                rank=0,
                world_size=args.gpus,
                epoch_offset=args.epoch_offset,
            )
            lengths = temp_sampler.epoch_lengths(max(2, args.epochs))
            args.stage1_steps = sum(lengths[:2])
        else:
            args.stage1_steps = 8000
    if args.stage1_steps < 0:
        raise ValueError("stage1 steps must be non-negative")
    metadata = payload["metadata"]
    tasks = list(metadata["tasks"])
    _, expected_total_steps, pcgrad_forward_grouping = _validate_run_schedule(
        payload, args
    )
    topology = resolve_world_topology()
    device = torch.device(f"cuda:{topology.local_rank}" if torch.cuda.is_available() else "cpu")
    if topology.is_distributed:
        torch.cuda.set_device(device)
    prefetcher = None
    initialize(topology, device)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        dataset = LongTrajFramesDataset(
            args.data, args.longtraj, min_sequence_length=SEQUENCE, decode_cache_tasks=4
        )
        world_dataset = LongTrajFramesDataset(
            args.data,
            args.longtraj,
            min_sequence_length=SEQUENCE,
            decode_cache_tasks=4,
            include_world_target_frames=True,
        )
        task_weights = torch.ones(len(tasks), dtype=torch.float64)
        data_sha256 = _sha256_file(args.data)
        if is_all_starts:
            sampler = AllStartsWindowBatchSampler(
                dataset.payload,
                batch_size=args.batch_size,
                seed=args.seed,
                mixed_tasks_per_batch=args.mixed_tasks,
                rank=topology.rank,
                world_size=topology.world_size,
                epoch_offset=args.epoch_offset,
            )
            world_sampler = AllStartsWindowBatchSampler(
                world_dataset.payload,
                batch_size=args.batch_size,
                seed=args.seed,
                mixed_tasks_per_batch=args.mixed_tasks,
                rank=topology.rank,
                world_size=topology.world_size,
                epoch_offset=args.epoch_offset,
            )
        elif joint_frontend:
            sampler = EpisodeWindowBatchSampler(
                dataset.payload,
                batch_size=args.batch_size,
                seed=args.seed,
                mixed_tasks_per_batch=args.mixed_tasks,
                rank=topology.rank,
                world_size=topology.world_size,
            )
            world_sampler = EpisodeWindowBatchSampler(
                world_dataset.payload,
                batch_size=args.batch_size,
                seed=args.seed,
                mixed_tasks_per_batch=args.mixed_tasks,
                rank=topology.rank,
                world_size=topology.world_size,
            )
        else:
            sampler = TaskLocalityWeightedSampler(
                dataset.payload["instruction_id"],
                dataset.payload["episode_id"],
                task_weights,
                args.batch_size,
                args.seed,
                16,
                "mixed",
                mixed_tasks_per_batch=args.mixed_tasks,
                anchor_replay_fraction=args.anchor_fraction,
                rank=topology.rank,
                world_size=topology.world_size,
                anchor_eligible=dataset.payload["anchor_eligible"],
            )
            world_sampler = TaskLocalityWeightedSampler(
                world_dataset.payload["instruction_id"],
                world_dataset.payload["episode_id"],
                task_weights,
                args.batch_size,
                args.seed,
                16,
                "mixed",
                mixed_tasks_per_batch=args.mixed_tasks,
                anchor_replay_fraction=args.anchor_fraction,
                rank=topology.rank,
                world_size=topology.world_size,
                anchor_eligible=world_dataset.payload["anchor_eligible"],
            )
        data_identity = {"contract": metadata["contract"], "sha256": data_sha256}
        sampler.bind_dataset_content_identity(data_identity)
        world_sampler.bind_dataset_content_identity(data_identity)
        if is_all_starts:
            action_train_dataset = AllStartsStreamDataset(dataset)
            world_train_dataset = AllStartsStreamDataset(world_dataset)
        elif joint_frontend:
            action_train_dataset = EpisodeStreamDataset(dataset)
            world_train_dataset = EpisodeStreamDataset(world_dataset)
        else:
            action_train_dataset = _StaticAnchorDataset(dataset)
            world_train_dataset = _StaticAnchorDataset(world_dataset)
        loader = DataLoader(
            action_train_dataset,
            batch_sampler=sampler,
            collate_fn=mtvj_collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed + topology.rank),
        )
        world_loader = DataLoader(
            world_train_dataset,
            batch_sampler=world_sampler,
            collate_fn=mtvj_collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(
                args.seed + topology.rank + 10_000
            ),
        )
        action_memory = (
            EpisodeMemoryBank(replay_offsets=True)
            if is_all_starts
            else EpisodeMemoryBank()
            if joint_frontend
            else None
        )
        world_memory = (
            EpisodeMemoryBank(replay_offsets=True)
            if is_all_starts
            else EpisodeMemoryBank()
            if joint_frontend
            else None
        )

        source_path = args.resume if args.resume is not None else args.resume_weights
        source = (
            torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if source_path is not None
            else None
        )
        config = (
            VACompoundConfig(**source["config"])
            if source is not None
            else _fresh_config(
                va_last3_cross_attn=args.va_last3_cross_attn,
                dino_qwen_cross_modal_bridge=args.dino_qwen_cross_modal_bridge,
                architecture_version=getattr(args, "architecture_version", "legacy"),
                world_state_loss_weight=args.world_state_loss_weight,
            )
        )
        joint_frontend = config.architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
        if config.architecture_version != getattr(args, "architecture_version", "legacy"):
            raise ValueError("checkpoint architecture mismatch; start a fresh run for the new architecture")
        if source is not None and joint_frontend:
            config_weight = getattr(config, "world_state_loss_weight", None)
            if config_weight != args.world_state_loss_weight:
                raise ValueError(
                    f"resume world_state_loss_weight {config_weight} != args {args.world_state_loss_weight}"
                )
        _validate_model_contract(config)
        if config.dino_qwen_cross_modal_bridge:
            validate_cross_modal_language_contract(dataset.payload.get("metadata") or {})
            validate_cross_modal_language_contract(
                world_dataset.payload.get("metadata") or {}
            )
        model = VACompoundPolicy(config)
        model.episode_microbatch = args.episode_microbatch
        if source is not None:
            model.load_state_dict(source["model"], strict=True)
        else:
            _initialize_scratch_fusion(model)
        model.to(device).train()

        text_backbone = QwenTextBackbone.from_pretrained(
            str(args.qwen),
            device=device,
            dtype="bfloat16",
            local_files_only=True,
        )
        if (
            len(text_backbone.text_model.layers) != 24
            or int(text_backbone.text_model.config.hidden_size) != 1024
        ):
            raise ValueError("Qwen3.5-0.8B must retain all 24 decoder layers")
        text_backbone.unfreeze_last(len(FUSION_LAYERS), freeze_final_norm=True)
        text_backbone.train()

        vision = TimmActionVisionBackbone.from_pretrained(
            device=device,
            dtype="float32",
            model_id=config.main_vision_model_id,
            image_size=config.main_vision_image_size,
            feature_dim=config.main_vision_dim,
            output_layers=(11, 23),
            checkpoint_path=args.dino,
            local_files_only=True,
        )
        vision.freeze_all()
        epoch_lengths = sampler.epoch_lengths(args.epochs) if is_all_starts else [len(sampler)] * args.epochs
        planned_total_steps = sum(epoch_lengths)
        if planned_total_steps != expected_total_steps:
            raise ValueError(
                f"sampler produced {planned_total_steps} steps, expected {expected_total_steps}"
            )
        start_step = int(source.get("global_step", 0)) if args.resume else 0
        stage2 = bool(args.resume_weights) or _stage2_enabled(
            max(1, start_step) if is_all_starts and args.stage1_steps == 0 else start_step, args.stage1_steps
        )
        model.runtime_dino_qwen_bridge_enabled = stage2
        if model.dino_qwen_bridge is not None:
            model.dino_qwen_bridge.requires_grad_(stage2)
        if stage2:
            _unfreeze_vision_tail(vision, joint_frontend=joint_frontend)
        if args.resume:
            contract = source.get("training_contract") or {}
            if not joint_frontend and (not source.get("source_checkpoint") or int(
                source.get("source_global_step", -1)
            ) != 1_000):
                raise ValueError("dense resume checkpoint lacks T4 s1000 lineage")
            required = {
                "initialization": (
                    TRANSFER_INITIALIZATION
                    if is_all_starts and int(source.get("source_global_step", -1)) >= 0
                    else "fresh_dual_tower_h15_v1"
                    if config.architecture_version == "dual_tower_h15_v1"
                    else "fresh_dual_tower_expert_v1"
                    if config.architecture_version == "dual_tower_expert_v1"
                    else FRESH_INITIALIZATION
                ),
                "source_checkpoint": source["source_checkpoint"],
                "source_global_step": (int(source.get("source_global_step", -1)) if joint_frontend else 1_000),
                "optimizer_initialization": (
                    "continued_adamw_reset_streams_v1"
                    if is_all_starts and int(source.get("source_global_step", -1)) >= 0
                    else "fresh_adamw_v1"
                    if joint_frontend
                    else "continued_from_source_v1"
                ),
                "data_contract": (
                    ALL_STARTS_DATA_CONTRACT
                    if is_all_starts
                    else H15_DATA_CONTRACT
                    if config.architecture_version == "dual_tower_h15_v1"
                    else JOINT_DATA_CONTRACT
                    if config.architecture_version == "dual_tower_expert_v1"
                    else DATA_CONTRACT
                ),
                "data_sha256": data_sha256,
                "suites": list(metadata["suites"]),
                "n_tasks": int(metadata["n_tasks"]),
                "task_specs": list(metadata["task_specs"]),
                "sequence_length": SEQUENCE,
                **(
                    {
                        "window_sampling": ALL_STARTS_SAMPLING_CONTRACT,
                        "sampling_contract": ALL_STARTS_SAMPLING_CONTRACT,
                        "epoch_lengths": list(epoch_lengths),
                    }
                    if is_all_starts
                    else {}
                ),
                **({
                    "action_horizon": 15,
                    "flow_prefix_steps": 15,
                    "flow_prefix_weight": 1.0,
                    "flow_tail_weight": 0.0,
                } if config.architecture_version == "dual_tower_h15_v1" else {}),
                "memory_reset_every": 0 if joint_frontend else SEQUENCE,
                "stage1_steps": args.stage1_steps,
                "total_steps": planned_total_steps,
                "step_rng": "seed_plus_source_and_phase_step_times_world_plus_rank_v1",
                "qwen_keep_layers": 24,
                "qwen_training": "last6_full_layers18_23_v1",
                "qwen_trainable_layers": FUSION_LAYERS,
                "qwen_final_norm_frozen": True,
                "qwen_hidden_dim": 1024,
                "qwen_fusion_layers": FUSION_LAYERS,
                "cross_modal_va_layers": CROSS_MODAL_VA_LAYERS,
                "qwen_base_readout": "layer23_final_norm",
                "qwen_fusion_reduce": "none",
                "qwen_gradient_checkpointing": False,
                "qwen_world_cache": (
                    "per_observation_joint_live_v1"
                    if joint_frontend
                    else "same_step_action_detached_v1"
                ),
                "wmrm_evidence": "post_va_vl_fused_tokens_v1",
                "pcgrad_scope": "per_task_action_and_world_separate_dino_guard_v1",
                "pcgrad_forward_grouping": pcgrad_forward_grouping,
                "main_vision_trainable_layers": FUSION_LAYERS,
                "main_vision_frames": 5,
                "dino_fusion_layers": FUSION_LAYERS,
                "dino_base_readout": "block23_norm",
                "dino_fusion_reduce": "none",
                "wmrm_world_dim": 1024,
                "wmrm_predictor_width": 1024,
                "flow_hidden_dim": 512,
                "vision_input": "agentview_history4_plus_current_wrist_v2",
                "world_target_view": "eye_in_hand_rgb",
                "wmrm_target_teacher": "shared_online_dino_block23_stopgrad_v1",
                "fusion_initialization": (
                    "dual_tower_zero_output_v1"
                    if joint_frontend
                    else "zero_output_unit_gate_v1"
                ),
                "architecture_version": config.architecture_version,
                **(
                    {
                        "execution_gradient_contract": (
                            "h15_unified_live_va_v1"
                            if config.architecture_version == "dual_tower_h15_v1"
                            else "p15_live_h50_tail_detached_v1"
                        ),
                        "memory_contract": (
                            ALL_STARTS_MEMORY_CONTRACT
                            if is_all_starts
                            else "episode_tbptt8_v1"
                        ),
                        "state_delta_contract": "joint7_gripper2_unclipped_q01q99_delta_h15_v1",
                        "world_state_loss_weight": config.world_state_loss_weight,
                    }
                    if joint_frontend
                    else {}
                ),
                "main_vision_encode_batch": args.encode_batch,
                "stage1_world_current_vision_cache": (
                    "disabled_joint_frontend_v1"
                    if joint_frontend
                    else "same_step_action_detached_v1"
                ),
                "phase": (
                    "stage2_qwen_va_dino_joint" if stage2 else "stage1_qwen_va"
                ),
            }
            if not joint_frontend:
                required.pop("architecture_version")
            if "execution_config" in contract:
                expected_exec = {
                    "episode_microbatch": args.episode_microbatch,
                    "joint_observation_chunk": args.joint_observation_chunk,
                    "gradient_microbatch": args.gradient_microbatch,
                }
                saved_exec = dict(contract["execution_config"])
                saved_exec.setdefault("gradient_microbatch", 0)
                if saved_exec != expected_exec:
                    raise ValueError(
                        f"execution_config mismatch: checkpoint {contract['execution_config']} != args {expected_exec}"
                    )
            else:
                if args.episode_microbatch != 1 or args.joint_observation_chunk != 0 or args.gradient_microbatch:
                    raise ValueError(
                        "checkpoints without execution_config require default episode_microbatch=1 and joint_observation_chunk=0"
                    )
            mismatch = {
                key: (contract.get(key), value)
                for key, value in required.items()
                if contract.get(key) != value
            }
            if mismatch:
                raise ValueError(f"LIBERO two-stage resume contract mismatch: {mismatch}")
            _load_selected_state(text_backbone, source["qwen_trainable_state_dict"])
            if stage2:
                state = source.get("main_vision_trainable_state_dict")
                if not isinstance(state, dict):
                    raise ValueError("stage-2 resume lacks trained DINO state")
                _load_selected_state(vision, state)
        elif args.resume_weights:
            (validate_transfer_source if is_all_starts_transfer(args) else _validate_dense_source)(source)
            _load_selected_state(
                text_backbone, source["qwen_trainable_state_dict"]
            )
            _load_selected_state(
                vision, source["main_vision_trainable_state_dict"]
            )

        fast_prefixes = (
            "action_queries",
            "state_projection.",
            "state_type_embed.",
            "dual_tower_fusion.",
            "flow_condition_projection.",
            "action_expert.",
            "tail_action_expert.",
            "extension_action_expert.",
            "flow_head.",
            "tail_flow_head.",
            "extension_flow_head.",
            "va_last3_readout.",
            "dino_qwen_bridge.",
            "wmrm.env_",
            "wmrm.world_from_state.",
            "wmrm.st_predictor.",
            "world_action_readout.",
        )
        fast = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(fast_prefixes)
        ]
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(fast_prefixes)
        ]
        qwen_parameters = [
            parameter for parameter in text_backbone.parameters() if parameter.requires_grad
        ]
        dino_parameters = [
            parameter
            for name, parameter in vision.named_parameters()
            if name.startswith(tuple(f"model.blocks.{index}." for index in FUSION_LAYERS))
            or name.startswith("model.norm.")
        ]
        if not all((fast, base_parameters, qwen_parameters, dino_parameters)):
            raise RuntimeError("LIBERO optimizer parameter partition is incomplete")
        optimizer = torch.optim.AdamW(
            [
                {"params": base_parameters, "lr": args.lr},
                {"params": fast, "lr": args.lr_new},
                {"params": qwen_parameters, "lr": args.lr_qwen},
                {"params": dino_parameters, "lr": args.lr_dino},
            ],
            weight_decay=args.weight_decay,
        )
        if args.resume:
            if not args.reset_optimizer:
                optimizer.load_state_dict(source["optimizer"])
            sampler.load_state_dict(source["sampler_state"])
            world_sampler.load_state_dict(source["world_sampler_state"])
            for label, active_sampler in (
                ("VA", sampler),
                ("World", world_sampler),
            ):
                if is_all_starts:
                    completed = sum(epoch_lengths[:active_sampler.epoch]) + active_sampler.batch_cursor
                else:
                    completed = active_sampler.epoch * len(active_sampler) + active_sampler.batch_cursor
                if completed != start_step:
                    raise ValueError(
                        f"{label} sampler position {completed} != checkpoint step {start_step}"
                    )
            if joint_frontend:
                if "episode_runtime_states" not in source:
                    raise ValueError("joint resume checkpoint missing episode_runtime_states")
                runtime_states = source["episode_runtime_states"]
                if not isinstance(runtime_states, list) or len(runtime_states) != topology.world_size:
                    raise ValueError(
                        f"episode_runtime_states must be a list of length {topology.world_size}, "
                        f"got {type(runtime_states)} of length {len(runtime_states) if isinstance(runtime_states, list) else 'N/A'}"
                    )
                rank_state = runtime_states[topology.rank]
                if not isinstance(rank_state, dict) or "action" not in rank_state or "world" not in rank_state:
                    raise ValueError(f"rank {topology.rank} episode_runtime_state invalid format")
                action_memory.load_state_dict(rank_state["action"])
                world_memory.load_state_dict(rank_state["world"])
        elif args.resume_weights:
            optimizer.load_state_dict(source["optimizer"])

        broadcast_parameters(list(model.parameters()), topology)
        broadcast_parameters(qwen_parameters, topology)
        broadcast_parameters(dino_parameters, topology)
        named_parameters = [
            *((f"model.{name}", parameter) for name, parameter in model.named_parameters()),
            *((f"qwen.{name}", parameter) for name, parameter in text_backbone.named_parameters()),
            *((f"main_vision.{name}", parameter) for name, parameter in vision.named_parameters()),
        ]
        qwen_sha256 = _sha256_file(_qwen_weight_file(args.qwen))
        dino_sha256 = _sha256_file(args.dino)
        if source is not None:
            contract = source["training_contract"]
            if contract.get("qwen_base_sha256") != qwen_sha256:
                raise ValueError("Qwen base SHA differs from the source checkpoint")
            if contract.get("main_vision_base_sha256") != dino_sha256:
                raise ValueError("DINO base SHA differs from the source checkpoint")
        lineage_source = (
            str(args.resume_weights)
            if args.resume_weights is not None
            else (source or {}).get(
                "source_checkpoint",
                "fresh_dual_tower_h15_v1"
                if config.architecture_version == "dual_tower_h15_v1"
                else "fresh_dual_tower_expert_v1",
            )
        )
        lineage_step = (
            int(source["global_step"])
            if args.resume_weights is not None
            else int((source or {}).get("source_global_step", -1))
        )
        del source
        total_steps = planned_total_steps
        if args.max_steps is not None:
            if args.max_steps < 1:
                raise ValueError("--max-steps must be positive")
            total_steps = min(total_steps, args.max_steps)
        epoch_boundary_steps = set(np.cumsum(epoch_lengths).tolist()) if is_all_starts else set()
        iterator, world_iterator = iter(loader), iter(world_loader)
        prefetcher = PeerJointBatchPrefetcher(
            iterator, loader, world_iterator, world_loader, depth=1
        )
        prefetcher.fill(
            peer_prefetch_fill_limit(
                total_steps - start_step, sampler, world_sampler
            )
        )
        active_stage2 = stage2
        for step in range(start_step + 1, total_steps + 1):
            step_seed = (
                args.seed
                + (lineage_step + step) * topology.world_size
                + topology.rank
            )
            torch.manual_seed(step_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(step_seed)
            next_stage2 = _stage2_enabled(step, args.stage1_steps)
            if next_stage2 != active_stage2:
                model.runtime_dino_qwen_bridge_enabled = next_stage2
                if model.dino_qwen_bridge is not None:
                    model.dino_qwen_bridge.requires_grad_(next_stage2)
                if next_stage2:
                    _unfreeze_vision_tail(vision, joint_frontend=joint_frontend)
                else:
                    vision.freeze_all()
                active_stage2 = next_stage2
                if topology.is_primary:
                    print(f"phase=stage{2 if active_stage2 else 1} step={step}", flush=True)
            raw, raw_world, iterator, world_iterator = prefetcher.result()
            prefetch_after_commit = False
            if step < total_steps:
                fill_limit = peer_prefetch_fill_limit(
                    total_steps - step,
                    sampler,
                    world_sampler,
                    current_batch_consumed=True,
                )
                prefetcher.fill(fill_limit)
                prefetch_after_commit = (
                    fill_limit == 0
                    and peer_prefetch_must_wait_for_commit(
                        sampler, world_sampler
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            for key in ("instruction_id", "episode_id", "pair_id"):
                if not torch.equal(raw[key], raw_world[key]):
                    raise RuntimeError(f"Action/World {key} batches are not aligned")
            action_tasks = _split_raw_tasks(raw)
            world_tasks = _split_raw_tasks(raw_world)
            action_task_ids = [int(task["instruction_id"][0]) for task in action_tasks]
            world_task_ids = [int(task["instruction_id"][0]) for task in world_tasks]
            if action_task_ids != world_task_ids or len(action_task_ids) != args.mixed_tasks:
                raise RuntimeError("PCGrad batch does not contain the configured tasks")
            group_size = (
                1
                if args.mixed_tasks in (1, 2)
                else 2
                if active_stage2
                else args.mixed_tasks
            )
            action_language_cache = {}
            action_vision_cache = {}
            action_values = []
            action_forwards = []
            world_components = {
                "visual": [],
                "guard": [],
                "static": [],
                "rank": [],
                "readout": [],
            }
            if joint_frontend:
                world_components["state"] = []
            for offset in range(0, len(action_tasks), group_size):
                group_ids = action_task_ids[offset : offset + group_size]
                task_raw = _merge_raw_tasks(
                    action_tasks[offset : offset + group_size]
                )

                def action_forward(task_raw=task_raw, group_ids=group_ids):
                    batch = _prepare_batch(
                        task_raw,
                        vision=vision,
                        device=device,
                        encode_batch=args.encode_batch,
                        joint_text=text_backbone, joint_tasks=tasks,
                        joint_fusion=model.dual_tower_fusion,
                        joint_observation_chunk=args.joint_observation_chunk,
                        world=False,
                        prev_dropout=args.prev_dropout,
                        layerwise_cross_modal=active_stage2 and not joint_frontend,
                    )
                    noisy, flow_time, target_velocity = sample_flow_matching_inputs(
                        batch["actions"]
                    )
                    with feature_policy_autocast(device, True):
                        _encode_language(
                            batch,
                            text_backbone,
                            tasks,
                            layerwise_cross_modal=active_stage2 and not joint_frontend,
                        )
                        for task_id in group_ids:
                            mask = batch["instruction_id"] == task_id
                            action_language_cache[task_id] = {
                                key: batch[key][mask].detach()
                                for key in (
                                    "language_hidden",
                                    "language_mask",
                                    "qwen_last4",
                                )
                                if key in batch
                            }
                            if not active_stage2 and not joint_frontend:
                                action_vision_cache[task_id] = batch[
                                    "vision_tokens"
                                ][mask].detach()
                        predicted, _ = rollout_policy(
                            model,
                            batch,
                            noisy,
                            flow_time,
                            train_world_model=False,
                            feature_autocast_bf16=True,
                            episode_memory=action_memory if joint_frontend else None,
                        )
                        losses = []
                        for task_id in group_ids:
                            mask = batch["instruction_id"] == task_id
                            task_batch = {
                                key: value[mask]
                                if isinstance(value, Tensor)
                                and value.ndim
                                and value.shape[0] == len(mask)
                                else value
                                for key, value in batch.items()
                            }
                            is_h15 = config.architecture_version == "dual_tower_h15_v1"
                            prefix_weight = 1.0 if is_h15 else 3.0
                            tail_weight = 0.0 if is_h15 else 1.0
                            if joint_frontend and not bool(task_batch["action_valid_mask"].any()):
                                loss = predicted[mask].sum() * 0
                                local_count = 0.0
                            else:
                                loss = masked_flow_matching_loss(
                                    predicted[mask],
                                    target_velocity[mask],
                                    task_batch,
                                    prefix_steps=EXECUTION_HORIZON,
                                    prefix_weight=prefix_weight,
                                    tail_weight=tail_weight,
                                )[0]
                                if joint_frontend:
                                    act_mask = task_batch["action_valid_mask"]
                                    split = min(EXECUTION_HORIZON, act_mask.shape[-1])
                                    prefix_count = act_mask[..., :split].sum().float()
                                    tail_count = act_mask[..., split:].sum().float()
                                    local_count = prefix_weight * prefix_count + tail_weight * tail_count
                            if joint_frontend and not args.gradient_microbatch:
                                loss = _rescale_task_loss_by_effective_count(
                                    loss, local_count, topology, device
                                )
                            losses.append(loss)
                            action_values.append(loss.detach())
                    return losses

                action_forwards.append(task_microbatch_forward(action_forward, task_raw, group_ids, args.gradient_microbatch, topology, device) if args.gradient_microbatch else action_forward)
            world_values = []
            world_forwards = []
            for offset in range(0, len(world_tasks), group_size):
                group_ids = world_task_ids[offset : offset + group_size]
                task_raw = _merge_raw_tasks(
                    world_tasks[offset : offset + group_size]
                )

                def world_forward(task_raw=task_raw, group_ids=group_ids):
                    cached_vision = (
                        None
                        if active_stage2 or joint_frontend
                        else torch.cat(
                            [action_vision_cache[task_id] for task_id in group_ids],
                            dim=0,
                        )
                    )
                    world_batch = _prepare_batch(
                        task_raw,
                        vision=vision,
                        device=device,
                        encode_batch=args.encode_batch,
                        joint_text=text_backbone, joint_tasks=tasks,
                        joint_fusion=model.dual_tower_fusion,
                        joint_observation_chunk=args.joint_observation_chunk,
                        world=True,
                        prev_dropout=args.prev_dropout,
                        layerwise_cross_modal=active_stage2 and not joint_frontend,
                        cached_vision=cached_vision,
                    )
                    world_noisy, world_time, _ = sample_flow_matching_inputs(
                        world_batch["actions"]
                    )
                    with feature_policy_autocast(device, True):
                        if not joint_frontend:
                            first = action_language_cache[group_ids[0]]
                            world_batch.update(
                                {
                                    key: torch.cat(
                                        [
                                            action_language_cache[task_id][key]
                                            for task_id in group_ids
                                        ],
                                        dim=0,
                                    )
                                    for key in first
                                }
                            )
                        rollout_policy(
                            model,
                            world_batch,
                            world_noisy,
                            world_time,
                            visual_world_supervision=True,
                            compute_action_output=False,
                            world_action_rank_stage="final",
                            wmrm_feature_metric="cosine",
                            summarize_visual_world_metrics=False,
                            feature_autocast_bf16=True,
                            episode_memory=world_memory if joint_frontend else None,
                        )
                    metric_attrs = [
                        ("visual", "last_wmrm_base_loss"),
                        ("guard", "last_world_no_regression_loss"),
                        ("static", "last_world_static_constraint_loss"),
                        ("rank", "last_world_action_rank_loss"),
                        ("readout", "last_world_action_readout_loss"),
                    ]
                    if joint_frontend:
                        metric_attrs.append(("state", "last_world_state_delta_loss"))
                    for name, attribute in metric_attrs:
                        value = getattr(model, attribute, None)
                        if value is None:
                            raise RuntimeError(f"World did not expose {attribute}")
                        world_components[name].append(value.detach().clone())
                    losses = dict(model.last_wmrm_task_losses)
                    if set(losses) != set(group_ids):
                        raise RuntimeError(
                            "grouped World forward produced the wrong task losses"
                        )
                    ordered = []
                    for task_id in group_ids:
                        loss = losses[task_id]
                        if joint_frontend and not args.gradient_microbatch:
                            task_mask = world_batch["instruction_id"] == task_id
                            local_count = (
                                world_batch["decision_count"][task_mask].sum().float()
                                if "decision_count" in world_batch
                                else world_batch["decision_valid_mask"][task_mask].sum().float()
                            )
                            loss = _rescale_task_loss_by_effective_count(
                                loss, local_count, topology, device
                            )
                        ordered.append(loss)
                    world_values.extend(loss.detach() for loss in ordered)
                    return ordered

                world_forwards.append(task_microbatch_forward(world_forward, task_raw, group_ids, args.gradient_microbatch, topology, device) if args.gradient_microbatch else world_forward)

            action_private = []
            world_private = []
            shared_dino = []
            for name, parameter in named_parameters:
                if name.startswith("main_vision."):
                    shared_dino.append((name, parameter))
                elif name.startswith(
                    ("model.wmrm.", "model.world_action_readout.")
                ) or name in {
                    "model.wmrm_stage_scale",
                    "model.wmrm_belief_message_scale",
                }:
                    world_private.append((name, parameter))
                else:
                    action_private.append((name, parameter))
            action_stats = backward_pcgrad(
                action_forwards,
                [*action_private, *shared_dino],
                seed=args.seed + step,
                topology=topology,
                compact_prefixes=("qwen.", "main_vision."),
                allow_inactive_ranks=joint_frontend,
                allow_single_task=joint_frontend and args.mixed_tasks == 1,
            )
            action_gradients = pop_update_gradients(
                [*action_private, *shared_dino]
            )
            world_stats = backward_pcgrad(
                world_forwards,
                [*world_private, *shared_dino],
                seed=args.seed + step,
                topology=topology,
                compact_prefixes=("main_vision.",),
                allow_inactive_ranks=joint_frontend,
                allow_single_task=joint_frontend and args.mixed_tasks == 1,
            )
            dino_stats = merge_separate_pcgrad_gradients(
                action_private,
                shared_dino,
                action_gradients,
            )
            action_gradients.clear()
            del action_gradients
            action_language_cache.clear()
            action_vision_cache.clear()
            stats = {
                "conflicts": action_stats["conflicts"],
                "comparisons": action_stats["comparisons"],
                "world_conflicts": world_stats["conflicts"],
                "world_comparisons": world_stats["comparisons"],
                **dino_stats,
            }
            clip_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            grad = torch.nn.utils.clip_grad_norm_(clip_parameters, 1.0)
            optimizer.step()
            if joint_frontend:
                action_memory.commit()
                world_memory.commit()
            sampler.advance()
            world_sampler.advance()
            if prefetch_after_commit:
                prefetcher.fill(
                    peer_prefetch_fill_limit(
                        total_steps - step, sampler, world_sampler
                    )
                )
            if any(
                (
                    sum(epoch_lengths[:active_sampler.epoch]) + active_sampler.batch_cursor
                    if is_all_starts
                    else active_sampler.epoch * len(active_sampler) + active_sampler.batch_cursor
                )
                != step
                for active_sampler in (sampler, world_sampler)
            ):
                raise RuntimeError("VA/World sampler positions diverged from global_step")

            action_value = reduce_scalar_mean(
                float(torch.stack(action_values).mean()), topology, device
            )
            world_logged = reduce_scalar_mean(
                float(torch.stack(world_values).mean()), topology, device
            )
            log_step = step == 1 or step % 10 == 0
            if log_step:
                component_log = {
                    name: reduce_scalar_mean(
                        float(torch.stack(values).mean()), topology, device
                    )
                    for name, values in world_components.items()
                }
            if topology.is_primary and log_step:
                msg = (
                    f"step={step}/{total_steps} action={action_value:.6f} "
                    f"world={world_logged:.6f} grad={float(grad):.4f} "
                    f"world_visual={component_log['visual']:.6f} "
                    f"world_guard_loss={component_log['guard']:.6f} "
                    f"world_static={component_log['static']:.6f} "
                    f"world_rank={component_log['rank']:.6f} "
                    f"world_readout={component_log['readout']:.6f} "
                )
                if joint_frontend:
                    state_raw = component_log["state"]
                    state_weighted = state_raw * config.world_state_loss_weight
                    msg += f"world_state_raw={state_raw:.6f} world_state_weighted={state_weighted:.6f} "
                msg += (
                    f"pcgrad={stats['conflicts']}/{stats['comparisons']} "
                    f"world_guard={stats['world_conflicts']}/{stats['world_comparisons']}"
                )
                print(msg, flush=True)
            save_checkpoint = (
                (step in epoch_boundary_steps or step == total_steps)
                if is_all_starts
                else (step % args.save_every == 0 or step == total_steps)
            )
            if save_checkpoint:
                episode_runtime_states = None
                if joint_frontend:
                    local_runtime_state = {
                        "action": action_memory.state_dict(),
                        "world": world_memory.state_dict(),
                    }
                    if topology.is_distributed:
                        import torch.distributed as dist

                        episode_runtime_states = [None] * topology.world_size
                        dist.all_gather_object(episode_runtime_states, local_runtime_state)
                    else:
                        episode_runtime_states = [local_runtime_state]
                barrier(topology)
                if topology.is_primary:
                    _atomic_checkpoint(
                        args.save,
                        config=config,
                        model=model,
                        text_backbone=text_backbone,
                        vision=vision,
                        optimizer=optimizer,
                        step=step,
                        total_steps=planned_total_steps,
                        stage1_steps=args.stage1_steps,
                        encode_batch=args.encode_batch,
                        qwen_sha256=qwen_sha256,
                        dino_sha256=dino_sha256,
                        data_sha256=data_sha256,
                        run_metadata=metadata,
                        pcgrad_forward_grouping=pcgrad_forward_grouping,
                        source_checkpoint=lineage_source,
                        source_global_step=lineage_step,
                        sampler=sampler,
                        world_sampler=world_sampler,
                        episode_runtime_states=episode_runtime_states,
                        window_sampling=getattr(args, "window_sampling", "episode_contiguous_p15_v1"),
                        epoch_lengths=epoch_lengths if is_all_starts else None,
                        execution_config={
                            "episode_microbatch": args.episode_microbatch,
                            "joint_observation_chunk": args.joint_observation_chunk,
                    "gradient_microbatch": args.gradient_microbatch,
                        },
                    )
                    print(f"saved {args.save} at step {step}", flush=True)
                barrier(topology)
    finally:
        if prefetcher is not None:
            prefetcher.close()
        shutdown(topology)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        from prepare_libero import prepare_data

        prepare_data(args)
    elif args.command == "preflight":
        preflight(args)
    else:
        if "RANK" not in os.environ and args.gpus > 1:
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={args.gpus}",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
            )
        train(args)


if __name__ == "__main__":
    main()
