from __future__ import annotations
import argparse
import shutil
import torch
from torch import nn
from va_compound.exact_resume import EXACT_RUN_CONTRACT_VERSION, _EXACT_RUN_OPERATIONAL_ARGS, _normalize_contract_value, _optimizer_contract, build_exact_resume_state
from va_compound.world_contract import FEATURE_AUTOCAST_CONTRACT, PEER_DATA_ISOLATION_CONTRACT, PEER_ACTION_ONLY_DATA_CONTRACT, PEER_DUAL_STREAM_OPTIMIZER_CONTRACT, PEER_GRADIENT_BOUNDARY_CONTRACT, PEER_H15_PREFIX_TAIL_FLOW_CONTRACT, PEER_H50_NESTED_FLOW_CONTRACT, PEER_HIGH_FREQUENCY_CONTRACT, PEER_SHARED_FULL_DATA_CONTRACT, PEER_WORLD_ACTION_SOURCE_CONTRACT, PEER_WORLD_READOUT_CONTRACT, PEER_WORLD_TOPOLOGY_CONTRACT, WORLD_ACTION_DONOR_CONTRACT, WORLD_LOGGED_BRANCH_CONTRACT, WORLD_LOSS_COMPONENT_WEIGHTS, WORLD_NO_REGRESSION, WORLD_STAGE_AUXILIARY_DECAY, WORLD_STAGE_AUXILIARY_FLOOR, WORLD_STATIC_COPY_CONSTRAINT, WORLD_SUPERVISION_CONTRACT, WORLD_TRANSITION_CONTRACT, world_action_ranking_contract, world_late_stage_anchor_contract
from va_compound.metric_roi import ASSEMBLY_METRIC_ROLE_CONTRACT, DINO_METRIC_ROI_CONTRACT, TASK35_METRIC_ROLE_CONTRACT
from va_compound.data.samplers import TaskLocalityWeightedSampler, TaskWeightedSampler
from va_compound.utils.exact_resume import _sha256_file
from va_compound.training.gradients import separate_pcgrad_scope
from va_compound.training.config import visual_world_stage_weight_overrides

MTVJ_LEGACY_METRIC_STATE_SOURCE = "p_flat"
MTVJ_LEGACY_METRIC_CONTRACT_VERSION = 2

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
    world_sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None = None,
    exact_run_contract: dict | None = None,
    main_vision_backbone=None,
) -> None:
    """原子保存 checkpoint（tmp 文件 + rename），供周期/最终保存复用。"""
    if not args.save:
        return
    args.save.parent.mkdir(parents=True, exist_ok=True)
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
            "mixed_tasks_per_batch": args.mixed_tasks_per_batch,
            "anchor_replay_fraction": args.anchor_replay_fraction,
            "pcgrad": bool(args.pcgrad),
            "pcgrad_scope": (
                (
                    separate_pcgrad_scope(args)
                    if getattr(args, "pcgrad_separate_world", False)
                    else "per_task_va_action_v1"
                    if getattr(args, "va_only", False)
                    else "per_task_va_action_then_project_world_v2"
                )
                if args.pcgrad
                else None
            ),
            "peer_training_mode": (
                "va_only"
                if getattr(args, "va_only", False)
                else "joint_dual_stream"
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else (
                    "world_only"
                    if getattr(args, "wmrm_only", False)
                    else "va_only"
                    if getattr(args, "va_only", False)
                    else None
                )
            ),
            "va_world_mode": getattr(config, "va_world_mode", "legacy"),
            "peer_world_topology": (
                PEER_WORLD_TOPOLOGY_CONTRACT
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_world_action_source": (
                PEER_WORLD_ACTION_SOURCE_CONTRACT
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_world_readout": (
                PEER_WORLD_READOUT_CONTRACT
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_gradient_boundary": (
                PEER_GRADIENT_BOUNDARY_CONTRACT
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_flow_topology": (
                PEER_H50_NESTED_FLOW_CONTRACT
                if int(getattr(config, "action_horizon", 0)) == 50
                else PEER_H15_PREFIX_TAIL_FLOW_CONTRACT
                if getattr(model, "tail_flow_head", None) is not None
                else None
            ),
            "deployment_execution_horizon": int(
                getattr(config, "deployment_execution_horizon", 0)
                or getattr(config, "planning_stride", 6)
            ),
            "peer_data_isolation": (
                (
                    PEER_ACTION_ONLY_DATA_CONTRACT
                    if getattr(args, "va_only", False)
                    else PEER_SHARED_FULL_DATA_CONTRACT
                    if getattr(args, "peer_shared_full_data", False)
                    else PEER_DATA_ISOLATION_CONTRACT
                )
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_dual_stream_optimizer": (
                PEER_DUAL_STREAM_OPTIMIZER_CONTRACT
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                and not getattr(args, "va_only", False)
                else None
            ),
            "optimizer_state_sharding": (
                "torch_zero_redundancy_v1"
                if getattr(args, "zero_redundancy_optimizer", False)
                else None
            ),
            "planning_stride": int(getattr(args, "planning_stride", 6)),
            "planning_hz": 80.0
            / int(getattr(args, "planning_stride", 6)),
            "peer_high_frequency_contract": (
                PEER_HIGH_FREQUENCY_CONTRACT
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_va_data_identity": (
                getattr(sampler, "dataset_content_identity", None)
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_world_data_identity": (
                getattr(world_sampler, "dataset_content_identity", None)
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
            "peer_data_isolation_summary": (
                getattr(args, "peer_data_isolation", None)
                if getattr(config, "va_world_mode", "legacy")
                == "peer_sync_h6"
                else None
            ),
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
            "assembly_metric_role_contract": (
                ASSEMBLY_METRIC_ROLE_CONTRACT
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
            "main_vision_checkpoint_sha256": getattr(
                args, "main_vision_checkpoint_sha256", None
            ),
            "main_vision_joint_trained": bool(
                main_vision_backbone is not None
                and any(
                    parameter.requires_grad
                    for parameter in main_vision_backbone.parameters()
                )
            ),
            "main_vision_lr": (
                float(args.lr_vision)
                if main_vision_backbone is not None
                and any(
                    parameter.requires_grad
                    for parameter in main_vision_backbone.parameters()
                )
                else None
            ),
            "flow_prefix_steps": args.flow_prefix_steps,
            "flow_prefix_weight": args.flow_prefix_weight,
            "flow_tail_weight": args.flow_tail_weight,
        },
    }
    if (
        main_vision_backbone is not None
        and payload["training_contract"]["main_vision_joint_trained"]
    ):
        payload["main_vision_state_dict"] = (
            main_vision_backbone.model.state_dict()
        )
    peer_migration_record = getattr(
        args, "_peer_resume_weights_contract_migration", None
    )
    if isinstance(peer_migration_record, dict):
        payload["peer_resume_weights_contract_migration"] = dict(
            peer_migration_record
        )
    if getattr(args, "visual_world_supervision", False):
        split_identity = getattr(args, "visual_world_split_identity", None)
        if not isinstance(split_identity, dict):
            raise ValueError(
                "visual World checkpoint requires a validated split identity"
            )
        payload["training_contract"].update(
            {
                "world_supervision": WORLD_SUPERVISION_CONTRACT,
                "world_transition": (
                    f"explicit_endpoint_h{int(config.wmrm_cycle_steps)}_v1"
                    if int(config.wmrm_cycle_steps)
                    > int(getattr(args, "planning_stride", 6))
                    else WORLD_TRANSITION_CONTRACT
                    if int(getattr(args, "planning_stride", 6)) == 6
                    else "current_first"
                    f"{int(getattr(args, 'planning_stride', 6))}"
                    "_and_next_first_v1"
                ),
                "world_loss_weights": dict(WORLD_LOSS_COMPONENT_WEIGHTS),
                "world_feature_metric": getattr(args, "wmrm_feature_metric", "mse"),
                "world_stage_auxiliary_decay": WORLD_STAGE_AUXILIARY_DECAY,
                "world_stage_auxiliary_floor": WORLD_STAGE_AUXILIARY_FLOOR,
                "world_stage_weight_overrides": visual_world_stage_weight_overrides(
                    args
                ),
                "world_late_stage_anchor": world_late_stage_anchor_contract(
                    float(getattr(args, "wmrm_late_stage_anchor_weight", 0.0))
                ),
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
                "world_action_donor_contract": split_identity.get(
                    "world_action_donor_contract", WORLD_ACTION_DONOR_CONTRACT
                ),
                "world_action_donor_sha256": split_identity[
                    "world_action_donor_sha256"
                ],
                "world_action_donor_transitions": split_identity[
                    "world_action_donor_transitions"
                ],
                "world_action_rank_transitions": split_identity[
                    "world_action_rank_transitions"
                ],
                "world_action_source": (
                    f"logged_h{int(config.action_horizon)}_world_horizon_"
                    f"{int(config.wmrm_cycle_steps)}"
                ),
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
                "planning_stride": int(getattr(args, "planning_stride", 6)),
                "planning_hz": 80.0
                / int(getattr(args, "planning_stride", 6)),
                "peer_high_frequency_contract": (
                    PEER_HIGH_FREQUENCY_CONTRACT
                    if getattr(args, "va_world_mode", "legacy") == "peer_sync_h6"
                    else None
                ),
                "split_manifest_id": split_identity["manifest_id"],
                "split_manifest_path": split_identity["manifest_path"],
                "split_manifest_sha256": split_identity["manifest_sha256"],
                "split_source_sha256": split_identity["source_sha256"],
            }
        )
    if optimizer is not None:
        if exact_run_contract is None:
            exact_run_contract = build_exact_run_contract(
                args,
                config,
                optimizer,
                sampler,
                metric_head,
                roi_head,
                world_sampler,
            )
        payload.update(
            build_exact_resume_state(
                optimizer, global_step, sampler, exact_run_contract
            )
        )
        if world_sampler is not None:
            payload["world_sampler_state"] = world_sampler.state_dict()
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


def build_exact_run_contract(
    args: argparse.Namespace,
    config,
    optimizer: torch.optim.Optimizer,
    sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None,
    metric_head: nn.Module | None = None,
    roi_head: nn.Module | None = None,
    world_sampler: TaskLocalityWeightedSampler | TaskWeightedSampler | None = None,
) -> dict:
    """Freeze every current MT-VJ CLI/data/objective semantic for exact resume."""
    argument_semantics = {
        key: _normalize_contract_value(value)
        for key, value in sorted(vars(args).items())
        if key not in _EXACT_RUN_OPERATIONAL_ARGS
    }
    if metric_head is not None or roi_head is not None:
        raise ValueError("metric/ROI trainer checkpoints are retired")
    metric_config = None
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
    model_config = dict(getattr(config, "__dict__", {}))
    if not getattr(config, "wmrm", False):
        model_config.pop("wmrm", None)
        model_config.pop("wmrm_world_dim", None)
        model_config.pop("wmrm_inject", None)
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
        planning_stride = int(getattr(args, "planning_stride", 6))
        contract["peer_world"] = {
            "topology": PEER_WORLD_TOPOLOGY_CONTRACT,
            "action_source": PEER_WORLD_ACTION_SOURCE_CONTRACT,
            "readout": PEER_WORLD_READOUT_CONTRACT,
            "gradient_boundary": PEER_GRADIENT_BOUNDARY_CONTRACT,
            "flow_topology": (
                PEER_H50_NESTED_FLOW_CONTRACT
                if int(getattr(config, "action_horizon", 0)) == 50
                else PEER_H15_PREFIX_TAIL_FLOW_CONTRACT
                if getattr(config, "deployment_execution_horizon", 0) == 15
                else None
            ),
            "deployment_execution_horizon": int(
                getattr(config, "deployment_execution_horizon", 0)
                or planning_stride
            ),
            "data_isolation": (
                PEER_ACTION_ONLY_DATA_CONTRACT
                if getattr(args, "va_only", False)
                else PEER_SHARED_FULL_DATA_CONTRACT
                if getattr(args, "peer_shared_full_data", False)
                else PEER_DATA_ISOLATION_CONTRACT
            ),
            "optimizer": (
                None
                if getattr(args, "va_only", False)
                else PEER_DUAL_STREAM_OPTIMIZER_CONTRACT
            ),
            "planning_stride": planning_stride,
            "planning_hz": 80.0 / planning_stride,
            "high_frequency": PEER_HIGH_FREQUENCY_CONTRACT,
            "va_data_identity": getattr(
                sampler, "dataset_content_identity", None
            ),
            "world_data_identity": getattr(
                world_sampler, "dataset_content_identity", None
            ),
        }
    if getattr(args, "feature_autocast_bf16", False):
        contract["feature_autocast"] = {
            "contract": FEATURE_AUTOCAST_CONTRACT,
            "dtype": "bfloat16",
            "training_cache_enabled": True,
            "no_grad_decode_cache_enabled": False,
        }
    return _normalize_contract_value(contract)