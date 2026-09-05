from __future__ import annotations
import argparse
from torch import Tensor
from va_compound import VACompoundPolicy
from va_compound.world_contract import PEER_H15_TO_H50_ACTION_MIGRATION
from va_compound.training.gradients import is_wmrm_predictor_parameter_name

def _main_vision_config_kwargs(args: argparse.Namespace) -> dict:
    """DINO-main replacement config（V-JEPA 路径 flag 关闭即禁用，不删除）。"""
    if not getattr(args, "dino_main_vision", False):
        return {}
    spec = {"model_id": "vit_large_patch14_reg4_dinov2.lvd142m", "image_size": 224, "feature_dim": 1024}
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
    return kwargs

def _split_wmrm_predictor_optimizer_groups(groups, model, args):
    """Peel ``wmrm.st_predictor`` out of existing groups when it has its own LR."""
    lr_predictor = getattr(args, "lr_wmrm_predictor", None)
    if lr_predictor is None:
        return groups
    predictor_params = []
    predictor_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if is_wmrm_predictor_parameter_name(name) and parameter.requires_grad:
            predictor_params.append(parameter)
            predictor_ids.add(id(parameter))
    if not predictor_params:
        raise ValueError(
            "--lr-wmrm-predictor requires trainable wmrm.st_predictor parameters"
        )
    rewritten = []
    for group in groups:
        remaining = [
            parameter
            for parameter in group["params"]
            if id(parameter) not in predictor_ids
        ]
        if remaining:
            rewritten.append({**group, "params": remaining})
    rewritten.append({"params": predictor_params, "lr": float(lr_predictor)})
    grouped_ids = [id(parameter) for group in rewritten for parameter in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("wmrm.st_predictor optimizer groups overlap")
    print(
        f"wmrm-predictor: {sum(parameter.numel() for parameter in predictor_params):,} "
        f"params @ lr={float(lr_predictor)}",
        flush=True,
    )
    return rewritten

def migrate_peer_h15_to_h50_state(
    model: VACompoundPolicy,
    checkpoint: dict,
) -> dict[str, Tensor]:
    """Expand only action-token state while loading every shared tensor strictly."""

    saved_config = checkpoint.get("config") or {}
    if (
        saved_config.get("action_horizon") != 15
        or saved_config.get("planning_stride") != 15
        or saved_config.get("deployment_execution_horizon") != 15
        or saved_config.get("wmrm_cycle_steps") != 15
        or saved_config.get("action_query_cond", False)
        or model.config.action_horizon != 50
        or model.config.planning_stride != 15
        or model.config.deployment_execution_horizon != 15
        or model.config.wmrm_cycle_steps != 15
        or model.config.action_query_cond
        or model.tail_flow_head is None
        or model.extension_flow_head is None
    ):
        raise ValueError(
            f"{PEER_H15_TO_H50_ACTION_MIGRATION} requires the H15/P15 s3224 "
            "topology and an H50/P15 target without action_query_cond"
        )
    saved = checkpoint["model"]
    own = model.state_dict()
    extension_keys = {
        key for key in own if key.startswith("extension_flow_head.")
    }
    expected_source_keys = set(own) - extension_keys
    if set(saved) != expected_source_keys:
        raise ValueError(
            "H15->H50 migration key mismatch: "
            f"missing={sorted(expected_source_keys - set(saved))[:8]}, "
            f"unexpected={sorted(set(saved) - expected_source_keys)[:8]}"
        )
    mismatched = {
        key: (tuple(saved[key].shape), tuple(own[key].shape))
        for key in expected_source_keys
        if tuple(saved[key].shape) != tuple(own[key].shape)
        and key != "action_queries"
    }
    if mismatched:
        raise ValueError(f"H15->H50 changed shared tensor shapes: {mismatched}")
    if tuple(saved["action_queries"].shape) != (15, model.config.hidden_dim):
        raise ValueError("H15 source action_queries must have shape [15, hidden_dim]")

    state = dict(saved)
    queries = own["action_queries"].clone()
    queries[:15].copy_(saved["action_queries"])
    state["action_queries"] = queries
    for key in extension_keys:
        source = "tail_flow_head." + key.removeprefix("extension_flow_head.")
        if source not in saved:
            raise ValueError(f"H15 source lacks {source}")
        state[key] = saved[source].clone()
    return state

def validate_cross_modal_language_contract(metadata: dict) -> None:
    if metadata.get("qwen_fusion_layers") != list(range(10, 15)) or metadata.get(
        "qwen_layer_reduce"
    ) != "mean_then_final_norm":
        raise ValueError(
            "DINO/Qwen bridge requires a Qwen 10-14 mean language cache"
        )

def _feature_optimizer_groups(args, model, vision_backbone):
    """Build stable policy parameter groups for action-only or joint training."""

    if getattr(args, "wmrm_only", False):
        if getattr(model, "wmrm", None) is None:
            raise ValueError("--wmrm-only requires a constructed WAM4VA module")
        wmrm_params, frozen_names = [], []
        for name, param in model.named_parameters():
            if name.startswith(("wmrm.", "world_action_readout.")):
                wmrm_params.append(param)
            else:
                frozen_names.append(name)
                param.requires_grad_(False)
        if not wmrm_params:
            raise ValueError("--wmrm-only found no World parameters")
        print(
            f"wmrm-only: freeze VA/FM ({len(frozen_names)} tensors); "
            f"train World state/predictor only "
            f"({sum(p.numel() for p in wmrm_params):,} params)",
            flush=True,
        )
        return _split_wmrm_predictor_optimizer_groups(
            [{"params": wmrm_params, "lr": args.lr}], model, args
        )
    if getattr(args, "va_only", False):
        va_params, frozen_names = [], []
        for name, param in model.named_parameters():
            if name.startswith(("wmrm.", "world_action_readout.")):
                frozen_names.append(name)
                param.requires_grad_(False)
            else:
                va_params.append(param)
        if not va_params:
            raise ValueError("--va-only found no VA/Flow parameters")
        groups = [{"params": va_params, "lr": args.lr}]
        print(
            f"va-only: freeze World ({len(frozen_names)} tensors); "
            f"train VA/Flow only ({sum(p.numel() for p in va_params):,} params)",
            flush=True,
        )
        return groups
    if args.head_only:
        head_params, rest_names = [], []
        for name, param in model.named_parameters():
            if name.startswith("flow_head.") or (
                model.config.va_last3_cross_attn
                and name.startswith("va_last3_readout.")
            ) or (
                model.config.dino_qwen_cross_modal_bridge
                and name.startswith("dino_qwen_bridge.")
            ):
                head_params.append(param)
            else:
                rest_names.append(name)
                param.requires_grad_(False)
        if not head_params:
            raise ValueError("--head-only requires flow head（--direct-head 不支持）")
        print(
            f"head-only: flow/readout 可训练参数 "
            f"{sum(p.numel() for p in head_params):,}；VA/槽/V-JEPA 冻结 "
            f"（{len(rest_names)} 组参数 requires_grad=False）"
        )
        return [{"params": head_params, "lr": args.lr}]
    if model.config.local_slots:
        raise ValueError("local-slot generic training is retired")
    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    return _split_wmrm_predictor_optimizer_groups(groups, model, args)
