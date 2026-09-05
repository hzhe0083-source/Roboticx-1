from __future__ import annotations
import os
from pathlib import Path
import random
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.data_parallel import DATA_PARALLEL_CONTRACT, barrier as data_parallel_barrier, broadcast_parameters, initialize as initialize_data_parallel, reduce_update_gradients, resolve_world_topology, shutdown as shutdown_data_parallel
from va_compound.exact_resume import build_dataset_content_identity, final_checkpoint_save_due, restore_exact_resume_state, restore_rng_state, validate_exact_run_contract
from va_compound.flow import ACTION_MASK_KEYS, effective_action_valid_fraction, masked_flow_matching_loss, sample_flow_matching_inputs
from va_compound.world_contract import PEER_DATA_ISOLATION_CONTRACT, PEER_H15_P2_TO_P15_TEMPORAL_MIGRATION, PEER_H15_PREFIX_TAIL_FLOW_CONTRACT, PEER_H15_PREFIX_TAIL_FLOW_MIGRATION, PEER_H15_TO_H50_ACTION_MIGRATION, PEER_H50_ACTION_ONLY_TO_JOINT_MIGRATION, PEER_H50_NESTED_FLOW_CONTRACT, PEER_SHARED_FULL_DATA_CONTRACT, PEER_VA8_TO_VA16_CAPACITY_MIGRATION, PEER_WORLD8_TO_WORLD7_REPAIR_MIGRATION, validate_peer_data_isolation, validate_peer_resume_weights_contract, validate_visual_world_resume_contract, validate_visual_world_training_split, world_action_ranking_contract
from va_compound.world_supervision import prepare_visual_world_action_ranking
from va_compound.metric_roi import ASSEMBLY_METRIC_ROLE_CONTRACT
from scripts.mt50_difficulty import task_weights_for
from va_compound.vision.encoding import DinoFeatureCache, _build_dino_main_backbone, _dino_main_encode_from_cache, _dino_main_online_encode
from va_compound.training.prefetch import PeerJointBatchPrefetcher, next_peer_joint_batches, peer_prefetch_fill_limit, peer_prefetch_must_wait_for_commit
from va_compound.data.samplers import TaskLocalityWeightedSampler, TaskWeightedSampler
from va_compound.training.model_setup import _feature_optimizer_groups, _main_vision_config_kwargs, migrate_peer_h15_to_h50_state, validate_cross_modal_language_contract
from va_compound.training.gradients import backward_pcgrad, backward_peer_joint_losses, clip_main_and_optional_predictor_gradients, clip_update_gradients, consolidate_zero_optimizer_state, merge_separate_pcgrad_gradients, named_optimizer_parameters, partition_separate_pcgrad_parameters, pop_update_gradients, separate_pcgrad_scope, validate_finite_update_scalars, validate_optimizer_update_state, validate_preclip_gradient_norms
from va_compound.training.checkpoint import build_exact_run_contract, save_checkpoint
from va_compound.training.batch import ensure_sequence, feature_policy_autocast, move_batch
from va_compound.training.rollout import rollout_policy
from va_compound.training.config import validate_args, visual_world_stage_weight_overrides

def run_metaworld(args) -> None:
    topology = resolve_world_topology()
    validate_args(args)
    if topology.is_distributed:
        if not str(args.device).startswith("cuda"):
            raise ValueError("multi-GPU data parallelism requires --device cuda")
        if int(args.batch_size) % topology.world_size:
            raise ValueError(
                f"--batch-size {args.batch_size} must divide across "
                f"{topology.world_size} ranks"
            )
        torch.cuda.set_device(topology.local_rank)
        args.data_parallel_world_size = topology.world_size
        args.data_parallel_contract = DATA_PARALLEL_CONTRACT
    elif getattr(args, "zero_redundancy_optimizer", False):
        raise ValueError("--zero-redundancy-optimizer requires multiple GPU ranks")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if topology.is_distributed:
        device = torch.device(f"cuda:{topology.local_rank}")
    else:
        device = torch.device(args.device)
    initialize_data_parallel(topology, device)

    dual_peer_data = bool(
        getattr(args, "va_data", None) is not None
        and getattr(args, "world_data", None) is not None
    )
    primary_data = args.va_data if dual_peer_data else args.data
    loader = None
    iterator = None
    sampler = None
    world_loader = None
    world_iterator = None
    world_sampler = None
    world_dataset = None
    model = None
    e2e_model = None
    vision_backbone = None
    task_log_names: dict[int, str] = {}
    env_by_description: dict[str, str] = {}
    lang_aux_cache: dict[str, tuple[Tensor, Tensor]] = {}
    aux_tasks: list[str] = []
    fork_iter = None  # 修复 HEAD 隐患：无 --data 的合成冒烟路径此前 UnboundLocalError
    from va_compound.longtraj_frames import (
        LongTrajFramesDataset,
        OnlineLongTrajEpisodeDataset,
    )
    if getattr(args, "online_episode_sampling", False):
        dataset = OnlineLongTrajEpisodeDataset(
            primary_data,
            longtraj_dir=getattr(args, "longtraj_dir", None),
            samples_per_episode=args.online_episode_samples,
            recovery_samples_per_episode=(
                args.online_recovery_samples_per_episode
            ),
            sampling_seed=args.seed,
            decode_cache_tasks=max(
                int(getattr(args, "longtraj_decode_cache_tasks", None) or 1), 2
            ),
            include_world_target_frames=False,
            action_horizon=args.online_action_horizon,
        )
    else:
        dataset = LongTrajFramesDataset(
            primary_data,
            longtraj_dir=getattr(args, "longtraj_dir", None),
            min_sequence_length=args.min_sequence_length,
            decode_cache_tasks=(
                max(int(getattr(args, "longtraj_decode_cache_tasks", None) or 1), 2)
                if dual_peer_data
                else (
                    getattr(args, "longtraj_decode_cache_tasks", None)
                    or (
                        2
                        if getattr(args, "dino_main_vision", False)
                        else 1
                    )
                )
            ),
            feature_cache=(
                args.dino_feature_cache
                if getattr(args, "dino_main_vision", False)
                else None
            ),
            include_frames=(
                args.dino_feature_cache is None
                or getattr(args, "dino_roi_checkpoint", None) is not None
            ),
            include_world_target_frames=False,
        )
    print(
        f"{'DINO-main' if args.dino_main_vision else 'MT-VJ'} data: "
        f"{type(dataset).__name__}（{len(dataset)} 样本，"
        f"帧从 longtraj JPEG 在线解码，"
        f"decode_cache_tasks={dataset.decode_cache_tasks}）",
        flush=True,
    )
    _enable_optional_action_masks(dataset)
    if getattr(args, "visual_world_supervision", False) and not dual_peer_data:
        args.visual_world_split_identity = validate_visual_world_training_split(
            dataset.payload,
            primary_data,
            args.world_split_manifest,
            va_world_mode=getattr(args, "va_world_mode", "legacy"),
            planning_stride=int(getattr(args, "planning_stride", 6)),
        )
        donor_identity = prepare_visual_world_action_ranking(
            dataset.payload,
            planning_stride=int(getattr(args, "planning_stride", 6)),
        )
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
    if args.dino_qwen_cross_modal_bridge:
        validate_cross_modal_language_contract(
            dataset.payload.get("metadata") or {}
        )
    dino_main_kwargs = _main_vision_config_kwargs(args)
    online_model_schema = getattr(dataset, "model_schema", None)
    if online_model_schema is None:
        dataset_language_hidden = dataset.payload["language_hidden"]
        dataset_action_horizon = int(dataset.payload["actions"].shape[-2])
        dataset_action_dim = int(dataset.payload["actions"].shape[-1])
        dataset_proprio_dim = int(dataset.payload["proprio"].shape[-1])
    else:
        dataset_language_hidden = dataset.task_language_hidden
        dataset_action_horizon = int(online_model_schema["action_horizon"])
        dataset_action_dim = int(online_model_schema["action_dim"])
        dataset_proprio_dim = int(online_model_schema["proprio_dim"])
    config = VACompoundConfig(
        language_dim=int(dataset_language_hidden.shape[-1]),
        vision_dim=int(dino_main_kwargs["main_vision_dim"]),
        action_horizon=dataset_action_horizon,
        planning_stride=args.planning_stride,
        deployment_execution_horizon=(
            args.deployment_execution_horizon or args.planning_stride
        ),
        action_dim=dataset_action_dim,
        proprio_dim=dataset_proprio_dim,
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
        va_last3_cross_attn=args.va_last3_cross_attn,
        dino_qwen_cross_modal_bridge=args.dino_qwen_cross_modal_bridge,
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
        wmrm=args.wmrm,
        wmrm_full_language_tokens=args.wmrm_full_language_tokens,
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
        wmrm_predictor_copies=getattr(args, "wmrm_predictor_copies", 1),
        wmrm_stage_gate_start=getattr(args, "wmrm_stage_gate_start", None),
        tail_flow_condition_grad=(
            getattr(args, "capacity_new_only", False)
            or getattr(args, "capacity_phase2_gates", False)
            or int(getattr(args, "online_action_horizon", 15)) == 50
        ),
        capacity_stage_gate_policy_grad=getattr(
            args, "capacity_phase2_gates", False
        ),
        runtime_integrity_checks=args.runtime_integrity_checks,
        slot_free_policy=args.slot_free_policy,
        local_slots=(args.local_slots_data is not None) or args.live_vjepa,
        local_slots_direct288=args.local_slots_direct288,
        local_slots_fixed_query=args.local_slots_fixed_query,
        dense_readout=args.dense_readout,
        multi_mode=args.multi_mode,
        local_slot_tokens=1152 if args.dense_readout else 288,
        **dino_main_kwargs,
    )
    effective_batch = args.batch_size
    if args.task_sampling in {"weighted", "balanced", "full", "mixed"}:
        tasks = list(dataset.payload.get("metadata", {}).get("tasks", []))
        if not tasks:
            raise ValueError(
                f"--task-sampling {args.task_sampling} 需要数据集 metadata.tasks "
                "（instruction_id → 难度权重映射）"
            )
        raw_task_w = (
            torch.ones(len(tasks), dtype=torch.float64)
            if args.task_sampling in {"balanced", "full", "mixed"}
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
                "每 epoch 每行无放回恰好一次）"
                if args.task_sampling == "full"
                else (
                    "每 batch 多任务均衡 + 固定 anchor replay）"
                    if args.task_sampling == "mixed"
                    else (
                    "每 epoch 严格均衡任务 batch）"
                    if args.task_sampling == "balanced"
                    else "按难度分层）"
                    )
                )
            ),
            flush=True,
        )
        # 双数据流辅助批次（阶段 C）：任务文本 → 数据集预计算
        # language_hidden/mask 缓存（metadata.tasks[tid] ↔ instruction_id=tid，
        # 与 build_longtraj_features.task_language_t 同源）；辅助任务按
        # 难度权重 raw_task_w 多项式采样。单任务子集（2026-08-16）只缓存
        # 数据里实际出现的任务。
        aux_weights: list[float] = []
        from va_compound.longtraj_frames import mtvj_collate
        sampler = TaskLocalityWeightedSampler(
            dataset.payload["instruction_id"],
            dataset.payload["episode_id"],
            raw_task_w,
            effective_batch,
            args.seed,
            args.task_locality_block_batches,
            args.task_sampling,
            task_order_seed=(
                args.seed
                if args.task_sampling in {"full", "mixed"}
                and getattr(args, "peer_shared_full_data", False)
                else None
            ),
            mixed_tasks_per_batch=args.mixed_tasks_per_batch,
            anchor_replay_fraction=args.anchor_replay_fraction,
            rank=topology.rank,
            world_size=topology.world_size,
            epoch_dataset=(
                dataset
                if getattr(args, "online_episode_sampling", False)
                else None
            ),
            anchor_eligible=dataset.payload.get("anchor_eligible"),
        )
        if args.save is not None or args.resume_exact is not None:
            # Full payload hashing happens once here; periodic/final
            # checkpoints reuse the sampler-cached identity.
            sampler.bind_dataset_content_identity(
                build_dataset_content_identity(
                    primary_data,
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
    else:
        loader = DataLoader(
            dataset,
            batch_size=effective_batch,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    fork_iter = None
    if dual_peer_data:
        from va_compound.longtraj_frames import (
            LongTrajFramesDataset,
            OnlineLongTrajEpisodeDataset,
            mtvj_collate,
        )

        if getattr(args, "online_episode_sampling", False):
            world_dataset = OnlineLongTrajEpisodeDataset(
                args.world_data,
                longtraj_dir=getattr(args, "longtraj_dir", None),
                samples_per_episode=args.online_episode_samples,
                recovery_samples_per_episode=(
                    args.online_recovery_samples_per_episode
                ),
                sampling_seed=args.seed + 1,
                decode_cache_tasks=max(
                    int(getattr(args, "longtraj_decode_cache_tasks", None) or 1), 2
                ),
                include_world_target_frames=config.action_horizon == 15,
                action_horizon=args.online_action_horizon,
            )
        else:
            world_dataset = LongTrajFramesDataset(
                args.world_data,
                longtraj_dir=getattr(args, "longtraj_dir", None),
                min_sequence_length=args.min_sequence_length,
                decode_cache_tasks=(
                    max(int(getattr(args, "longtraj_decode_cache_tasks", None) or 1), 2)
                ),
                feature_cache=None,
                include_frames=True,
                include_world_target_frames=config.action_horizon == 15,
            )
        _enable_optional_action_masks(world_dataset)
        if config.dino_qwen_cross_modal_bridge:
            validate_cross_modal_language_contract(
                world_dataset.payload.get("metadata") or {}
            )
        args.visual_world_split_identity = validate_visual_world_training_split(
            world_dataset.payload,
            args.world_data,
            args.world_split_manifest,
            va_world_mode=getattr(args, "va_world_mode", "legacy"),
            planning_stride=int(getattr(args, "planning_stride", 6)),
        )
        args.peer_data_isolation = validate_peer_data_isolation(
            dataset.payload,
            world_dataset.payload,
            planning_stride=int(getattr(args, "planning_stride", 6)),
            shared_full_data=bool(
                getattr(args, "peer_shared_full_data", False)
            ),
        )
        if getattr(args, "online_episode_sampling", False):
            donor_identity = {
                key: args.visual_world_split_identity[key]
                for key in (
                    "world_action_donor_contract",
                    "world_action_donor_sha256",
                    "world_action_donor_transitions",
                    "world_action_rank_transitions",
                )
            }
        else:
            donor_identity = prepare_visual_world_action_ranking(
                world_dataset.payload,
                planning_stride=int(getattr(args, "planning_stride", 6)),
            )
            args.visual_world_split_identity.update(donor_identity)
        world_tasks = list(
            world_dataset.payload.get("metadata", {}).get("tasks", [])
        )
        if not world_tasks:
            raise ValueError("--world-data requires metadata.tasks")
        world_raw_task_w = (
            torch.ones(len(world_tasks), dtype=torch.float64)
            if args.task_sampling in {"balanced", "full", "mixed"}
            else torch.tensor(
                task_weights_for(world_tasks), dtype=torch.float64
            )
        )
        if args.task_sampling in {"weighted", "balanced", "full", "mixed"}:
            world_sampler = TaskLocalityWeightedSampler(
                world_dataset.payload["instruction_id"],
                world_dataset.payload["episode_id"],
                world_raw_task_w,
                args.batch_size,
                args.seed + 1,
                args.task_locality_block_batches,
                args.task_sampling,
                task_order_seed=(
                    args.seed
                    if args.task_sampling in {"full", "mixed"}
                    and getattr(args, "peer_shared_full_data", False)
                    else None
                ),
                mixed_tasks_per_batch=args.mixed_tasks_per_batch,
                anchor_replay_fraction=args.anchor_replay_fraction,
                rank=topology.rank,
                world_size=topology.world_size,
                epoch_dataset=(
                    world_dataset
                    if getattr(args, "online_episode_sampling", False)
                    else None
                ),
                anchor_eligible=world_dataset.payload.get("anchor_eligible"),
            )
            if args.save is not None or args.resume_exact is not None:
                world_sampler.bind_dataset_content_identity(
                    build_dataset_content_identity(
                        args.world_data,
                        world_dataset.payload,
                        longtraj_dir=getattr(
                            world_dataset, "longtraj_dir", None
                        ),
                    )
                )
            world_loader = DataLoader(
                world_dataset,
                batch_sampler=world_sampler,
                collate_fn=mtvj_collate,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                generator=torch.Generator().manual_seed(args.seed + 1),
            )
        else:
            world_loader = DataLoader(
                world_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=mtvj_collate,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                generator=torch.Generator().manual_seed(args.seed + 1),
            )
        world_iterator = iter(world_loader)
        print(
            "peer joint data: PASS "
            f"VA={args.va_data} ({len(dataset)} rows), "
            f"World={args.world_data} ({len(world_dataset)} rows), "
            f"episodes={args.peer_data_isolation['va_episode_count']}+"
            f"{args.peer_data_isolation['world_episode_count']}",
            flush=True,
        )
    iterator = iter(loader)
    smoke_batch = None

    # MT-VJ（契约 §6）：fp16 V-JEPA 始终冻结只读；metric localization path
    # 与 relation encoder 默认冻结，可分别显式联合微调。
    # MT-VJ flags 都未给时以下对象全为 None，旧路径不变。
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
    main_vision_backbone = None
    dino_cache = None
    main_vision_backbone = _build_dino_main_backbone(args, config, device)
    if args.dino_feature_cache is not None:
        # 特征缓存模式：训练循环从 memmap 读预计算特征（塔仅用于校验/
        # 不在循环内前向）。位级一致由 build_dino_feature_cache.py 验证。
        dino_cache = DinoFeatureCache(args.dino_feature_cache)
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

    # Step 2：双新息中央凹交互伺服（C²-IRF v2 §七 Step 2；--servo-only 隐含
    # --servo 已在 validate_args 生效）。独立模块（契约文件 va_compound/servo.py），
    # 不进 VACompoundPolicy；checkpoint 单独存 "servo" 键 + training_contract 字段。
    servo = None
    servo_stats = None

    model = VACompoundPolicy(config).to(device)
    vision_backbone = None
    groups = _feature_optimizer_groups(args, model, vision_backbone)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    if main_vision_backbone is not None:
        main_vision_params = [
            parameter
            for parameter in main_vision_backbone.parameters()
            if parameter.requires_grad
        ]
        if main_vision_params:
            optimizer.add_param_group(
                {"params": main_vision_params, "lr": args.lr_vision}
            )
            print(
                "dino-main: optimizer added "
                f"{sum(parameter.numel() for parameter in main_vision_params):,} "
                f"params @ lr={args.lr_vision}",
                flush=True,
            )

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


    if getattr(args, "zero_redundancy_optimizer", False):
        from torch.distributed.optim import ZeroRedundancyOptimizer

        parameter_groups = [
            {**group, "params": list(group["params"])}
            for group in optimizer.param_groups
        ]
        del optimizer
        optimizer = ZeroRedundancyOptimizer(
            parameter_groups,
            optimizer_class=torch.optim.AdamW,
            parameters_as_bucket_view=False,
            lr=args.lr,
            weight_decay=1e-4,
        )
        print(
            "optimizer: AdamW state sharded across "
            f"{topology.world_size} ranks with ZeroRedundancyOptimizer",
            flush=True,
        )

    validate_optimizer_update_state(optimizer)

    # Cheap after startup: the expensive data digest is already cached on the
    # locality sampler. This immutable value is reused by every checkpoint.
    runtime_exact_run_contract = build_exact_run_contract(
        args,
        config,
        optimizer,
        sampler,
        metric_head,
        roi_head,
        world_sampler,
    )
    broadcast_parameters(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        topology,
    )
    if topology.is_distributed and topology.is_primary:
        print(
            f"data_parallel contract={DATA_PARALLEL_CONTRACT} "
            f"world_size={topology.world_size} global_batch={args.batch_size} "
            f"local_batch={int(args.batch_size) // topology.world_size}",
            flush=True,
        )

    model.train()
    if main_vision_backbone is not None:
        main_vision_backbone.train()
    global_step = 0
    resume_rng_state = None
    exact_resume = args.resume_exact is not None
    if resume_path is not None:
        resume_ckpt = (
            preloaded_resume_ckpt
            if preloaded_resume_ckpt is not None
            else torch.load(resume_path, map_location="cpu", weights_only=True)
        )
        prior_peer_migration = resume_ckpt.get(
            "peer_resume_weights_contract_migration"
        )
        if isinstance(prior_peer_migration, dict):
            args._peer_resume_weights_contract_migration = dict(
                prior_peer_migration
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
                    int(getattr(args, "planning_stride", 6)),
                    float(getattr(args, "wmrm_late_stage_anchor_weight", 0.0)),
                    visual_world_stage_weight_overrides(args),
                    world_horizon=int(config.wmrm_cycle_steps),
                    assembly_metric_role_contract=(
                        ASSEMBLY_METRIC_ROLE_CONTRACT
                        if getattr(config, "dino_dense_metric", False)
                        else None
                    ),
                    peer_data_isolation_contract=(
                        PEER_SHARED_FULL_DATA_CONTRACT
                        if getattr(args, "peer_shared_full_data", False)
                        else PEER_DATA_ISOLATION_CONTRACT
                    ),
                )
            # Fail before restoring model/optimizer/sampler/RNG if any data,
            # objective, sampler, architecture or optimizer semantic changed.
            validate_exact_run_contract(
                resume_ckpt.get("exact_run_contract"),
                runtime_exact_run_contract,
                migration_id=args.resume_exact_contract_migration,
            )
            if world_sampler is not None:
                saved_world_sampler = resume_ckpt.get("world_sampler_state")
                if saved_world_sampler is None:
                    raise ValueError(
                        "peer joint --resume-exact requires world_sampler_state"
                    )
                # Validate the second stream before restoring model/optimizer or
                # primary sampler state. The sampler checks immutable fields
                # before assigning its epoch/cursor.
                world_sampler.load_state_dict(saved_world_sampler)
        elif (
            getattr(args, "resume_weights", None) is not None
            and getattr(config, "va_world_mode", "legacy") == "peer_sync_h6"
        ):
            contract = resume_ckpt.get("training_contract") or {}
            migrating_peer_world = (
                getattr(args, "resume_weights_migration", None)
                == PEER_WORLD8_TO_WORLD7_REPAIR_MIGRATION
            )
            migrating_prefix_tail_flow = (
                getattr(args, "resume_weights_migration", None)
                == PEER_H15_PREFIX_TAIL_FLOW_MIGRATION
            )
            migrating_p2_to_p15 = (
                getattr(args, "resume_weights_migration", None)
                == PEER_H15_P2_TO_P15_TEMPORAL_MIGRATION
            )
            migrating_h15_to_h50 = (
                getattr(args, "resume_weights_migration", None)
                == PEER_H15_TO_H50_ACTION_MIGRATION
            )
            migrating_action_only_to_joint = (
                getattr(args, "resume_weights_migration", None)
                == PEER_H50_ACTION_ONLY_TO_JOINT_MIGRATION
            )
            migrating_va_depth = (
                getattr(args, "resume_weights_migration", None)
                == PEER_VA8_TO_VA16_CAPACITY_MIGRATION
            )
            migration_record = validate_peer_resume_weights_contract(
                contract,
                planning_stride=int(getattr(args, "planning_stride", 6)),
                migrating_peer_world=migrating_peer_world,
                migrating_prefix_tail_flow=migrating_prefix_tail_flow,
                migrating_p2_to_p15=migrating_p2_to_p15,
                migrating_h15_to_h50=migrating_h15_to_h50,
                migrating_action_only_to_joint=(
                    migrating_action_only_to_joint
                ),
                migrating_va_depth=migrating_va_depth,
                action_horizon=int(config.action_horizon),
                world_horizon=int(config.wmrm_cycle_steps),
                deployment_execution_horizon=int(
                    getattr(config, "deployment_execution_horizon", 0)
                    or getattr(config, "planning_stride", 6)
                ),
                peer_flow_topology=(
                    PEER_H50_NESTED_FLOW_CONTRACT
                    if int(config.action_horizon) == 50
                    else PEER_H15_PREFIX_TAIL_FLOW_CONTRACT
                    if getattr(model, "tail_flow_head", None) is not None
                    else None
                ),
                assembly_metric_role_contract=(
                    ASSEMBLY_METRIC_ROLE_CONTRACT
                    if getattr(config, "dino_dense_metric", False)
                    else None
                ),
                peer_data_isolation_contract=(
                    PEER_SHARED_FULL_DATA_CONTRACT
                    if getattr(args, "peer_shared_full_data", False)
                    else PEER_DATA_ISOLATION_CONTRACT
                ),
                target_pcgrad_scope=separate_pcgrad_scope(args),
            )
            if migration_record is not None:
                migration_record = {
                    **migration_record,
                    "source_global_step": int(resume_ckpt.get("global_step", 0)),
                }
                args._peer_resume_weights_contract_migration = migration_record
                migration_names = ",".join(
                    str(item["kind"])
                    for item in migration_record["migrations"]
                )
                print(
                    "resume-weights semantic migration: " + migration_names,
                    flush=True,
                )
        resume_config = resume_ckpt["config"]
        for key in (
            "num_layers",
            "hidden_dim",
            "action_horizon",
            "action_dim",
            "proprio_dim",
            "mode",
            "va_world_mode",
            "planning_stride",
            "deployment_execution_horizon",
            "wmrm_cycle_steps",
            "wmrm_predictor",
            "wmrm_predictor_depth",
            "wmrm_predictor_width",
            "wmrm_predictor_heads",
            "wmrm_predictor_copies",
        ):
            if (
                getattr(args, "resume_weights_migration", None)
                == PEER_H15_TO_H50_ACTION_MIGRATION
                and key == "action_horizon"
            ):
                continue
            if key.startswith("wmrm_") and not getattr(config, "wmrm", False):
                continue
            left = resume_config.get(key)
            if key == "va_world_mode" and left is None:
                left = "legacy"
            if key == "planning_stride" and left is None:
                left = 6
            if key == "deployment_execution_horizon" and left is None:
                left = resume_config.get("planning_stride") or 6
            right = getattr(config, key, None)
            if key.startswith("wmrm_") and left is None:
                left = {
                    "wmrm_cycle_steps": 6,
                    "wmrm_predictor": "legacy",
                    "wmrm_predictor_depth": 6,
                    "wmrm_predictor_width": 384,
                    "wmrm_predictor_heads": 12,
                    "wmrm_predictor_copies": 1,
                }[key]
            if left != right:
                raise ValueError(
                    f"resume config mismatch on {key}: {left} vs {right}"
                )
        if exact_resume:
            model.load_state_dict(resume_ckpt["model"], strict=True)
        elif (
            getattr(args, "resume_weights", None) is not None
            and getattr(config, "va_world_mode", "legacy") == "peer_sync_h6"
        ):
            if (
                getattr(args, "resume_weights_migration", None)
                == PEER_H15_TO_H50_ACTION_MIGRATION
            ):
                state = migrate_peer_h15_to_h50_state(model, resume_ckpt)
                model.load_state_dict(state, strict=True)
                print(
                    "resume-weights migration: preserved the complete H15 policy; "
                    "initialized H35 action queries and extension Flow explicitly",
                    flush=True,
                )
            elif (
                config.va_last3_cross_attn
                and not resume_config.get("va_last3_cross_attn", False)
            ) or (
                config.dino_qwen_cross_modal_bridge
                and not resume_config.get(
                    "dino_qwen_cross_modal_bridge", False
                )
            ):
                missing, unexpected = model.load_state_dict(
                    resume_ckpt["model"], strict=False
                )
                allowed_missing = {
                    key
                    for key in model.state_dict()
                    if (
                        config.va_last3_cross_attn
                        and not resume_config.get(
                            "va_last3_cross_attn", False
                        )
                        and key.startswith("va_last3_readout.")
                    )
                    or (
                        config.dino_qwen_cross_modal_bridge
                        and not resume_config.get(
                            "dino_qwen_cross_modal_bridge", False
                        )
                        and key.startswith("dino_qwen_bridge.")
                    )
                }
                if set(missing) != allowed_missing or unexpected:
                    raise ValueError(
                    "VA fusion migration changed unrelated tensors: "
                        f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                    )
                print(
                    "resume-weights migration: preserved VA/World/Flow and "
                    "initialized the zero-gated fusion/readout modules",
                    flush=True,
                )
            else:
                model.load_state_dict(resume_ckpt["model"], strict=True)
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
        print(f"resumed from {resume_path}")
        saved_main_vision = resume_ckpt.get("main_vision_state_dict")
        saved_main_trained = bool(
            (resume_ckpt.get("training_contract") or {}).get(
                "main_vision_joint_trained", False
            )
        )
        if main_vision_backbone is not None and saved_main_vision is not None:
            main_vision_backbone.model.load_state_dict(
                saved_main_vision, strict=True
            )
            print("dino-main: trained weights restored from checkpoint", flush=True)
        elif saved_main_trained:
            raise ValueError(
                "checkpoint declares trained DINO-main but lacks "
                "main_vision_state_dict"
            )
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
    mtvj_dense_evidence = None
    mtvj_metric_tokens = None
    mtvj_metric_g = None
    action_dense_evidence = None
    # --perturb-data 帧在线编码骨干（live 路径复用主骨干；feature 路径冻结 V-JEPA）。
    if args.lang_fixed_vector:
        # grounding 对照（Codex 2026-08-08）：语言通道 = 数据集全局均值常量向量，
        # 循环外预计算一次；完整模型 vs 固定语言基线的差距即语言条件的因果贡献。
        lang_fixed_vec = dataset_language_hidden.mean(dim=(0, 1), keepdim=True)
        print(f"lang-fixed-vector: 语言通道替换为全局均值（shape={tuple(lang_fixed_vec.shape)}）")

    def prepare_peer_world_batch(raw_batch):
        """Encode one physically separate World batch without VA-label reuse."""
        if not dual_peer_data:
            raise RuntimeError("peer World batch preparation requires dual streams")
        if main_vision_backbone is None:
            raise ValueError("peer joint World stream requires DINO-main encoding")
        frames = raw_batch.get("frames")
        if frames is None:
            raise ValueError("--world-data batch has no raw frames")
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        dense_evidence = None
        vision_tokens = _dino_main_online_encode(
            frames,
            main_vision_backbone,
            device,
            encode_batch=args.main_vision_encode_batch,
            grid=config.main_vision_grid,
            window=config.main_vision_frames,
            return_dense=False,
            last_four_mean=config.dino_qwen_cross_modal_bridge,
        )
        raw_batch["vision_tokens"] = vision_tokens
        target_frames = raw_batch.pop("world_target_frames", None)
        if target_frames is not None:
            if isinstance(target_frames, torch.Tensor):
                target_frames = target_frames.cpu().numpy()
            with torch.no_grad():
                target_tokens = _dino_main_online_encode(
                    target_frames,
                    main_vision_backbone,
                    device,
                    encode_batch=args.main_vision_encode_batch,
                    grid=config.main_vision_grid,
                    window=1,
                    return_dense=False,
                    last_four_mean=config.dino_qwen_cross_modal_bridge,
                )
            batch_size, sequence, patches, dim = target_tokens.shape
            grid = config.main_vision_grid
            if patches != grid * grid:
                raise RuntimeError(
                    "explicit World endpoint must encode one complete patch grid"
                )
            raw_batch["world_target_map"] = target_tokens.reshape(
                batch_size, sequence, grid, grid, dim
            ).permute(0, 1, 4, 2, 3).detach()
        metric_tokens = None
        metric_g = None
        raw_batch.pop("frames", None)
        prepared = move_batch(raw_batch, device)
        prepared = ensure_sequence(prepared, args.min_sequence_length)
        if args.prev_dropout > 0.0:
            prev_mask = (
                torch.rand(
                    prepared["previous_action"].shape[0], device=device
                )
                < args.prev_dropout
            )
            prepared["previous_action"] = prepared["previous_action"] * (
                ~prev_mask
            ).view(-1, 1, 1).float()
        if args.lang_fixed_vector:
            prepared["language_hidden"] = lang_fixed_vec.expand(
                prepared["language_hidden"].shape
            ).to(device)
            if "language_mask" in prepared:
                prepared["language_mask"] = torch.ones_like(
                    prepared["language_mask"]
                )
        return prepared, dense_evidence, metric_tokens, metric_g

    if resume_rng_state is not None:
        # DataLoader iterator construction consumes a torch base-seed. Rebuild it
        # from the restored sampler first, then restore global RNG immediately
        # before fetching the next batch/noise.
        iterator = iter(loader)
        if world_loader is not None:
            world_iterator = iter(world_loader)
        restore_rng_state(resume_rng_state)

    last_saved_global_step: int | None = None
    peer_batch_prefetcher = None
    if getattr(args, "peer_batch_prefetch", False):
        peer_batch_prefetcher = PeerJointBatchPrefetcher(
            iterator,
            loader,
            world_iterator,
            world_loader,
            depth=int(args.peer_batch_prefetch_depth),
        )
        peer_batch_prefetcher.fill(
            peer_prefetch_fill_limit(
                args.steps, sampler, world_sampler
            )
        )

    def commit_successful_update(local_step: int, consumed_locality_batch: bool) -> None:
        """Advance all resumable state only after the optimizer update succeeds."""
        nonlocal global_step, last_saved_global_step
        if consumed_locality_batch:
            sampler.advance()
        if world_sampler is not None:
            world_sampler.advance()
        global_step += 1
        if (
            args.save is not None
            and args.save_every > 0
            and global_step % args.save_every == 0
        ):
            consolidate_zero_optimizer_state(optimizer, to=0)
            if topology.is_primary:
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
                    world_sampler=world_sampler,
                    exact_run_contract=runtime_exact_run_contract,
                    main_vision_backbone=main_vision_backbone,
                )
            last_saved_global_step = global_step
            if topology.is_primary:
                print(
                    f"step={local_step} global_step={global_step} "
                    f"periodic checkpoint saved to {args.save}",
                    flush=True,
                )

    for step in range(1, args.steps + 1):
        world_raw_batch = None
        consumed_locality_batch = False
        prefetch_after_commit = False
        if dual_peer_data:
            if peer_batch_prefetcher is None:
                (
                    batch,
                    world_raw_batch,
                    iterator,
                    world_iterator,
                ) = next_peer_joint_batches(
                    iterator,
                    loader,
                    world_iterator,
                    world_loader,
                )
            else:
                (
                    batch,
                    world_raw_batch,
                    iterator,
                    world_iterator,
                ) = peer_batch_prefetcher.result()
                if step < args.steps:
                    fill_limit = peer_prefetch_fill_limit(
                        args.steps - step,
                        sampler,
                        world_sampler,
                        current_batch_consumed=True,
                    )
                    peer_batch_prefetcher.fill(fill_limit)
                    prefetch_after_commit = (
                        fill_limit == 0
                        and peer_prefetch_must_wait_for_commit(
                            sampler, world_sampler
                        )
                    )
            consumed_locality_batch = isinstance(
                sampler, (TaskLocalityWeightedSampler, TaskWeightedSampler)
            )
        elif iterator is None:
            batch = smoke_batch
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            consumed_locality_batch = isinstance(
                sampler, (TaskLocalityWeightedSampler, TaskWeightedSampler)
            )
        next_global_step = global_step + 1
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
                batch["vision_tokens"] = _dino_main_online_encode(
                    frames_main,
                    main_vision_backbone,
                    device,
                    encode_batch=args.main_vision_encode_batch,
                    grid=config.main_vision_grid,
                    window=config.main_vision_frames,
                    last_four_mean=config.dino_qwen_cross_modal_bridge,
                )
                batch.pop("frames", None)
            # Cache+ROI keeps raw frames only until the crop refinement above.
            # Remove the large CPU array before generic device collation.
            batch.pop("frames", None)
        batch = move_batch(batch, device)
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

        if args.lang_fixed_vector:
            batch["language_hidden"] = lang_fixed_vec.expand(
                batch["language_hidden"].shape
            ).to(device)
            if "language_mask" in batch:
                batch["language_mask"] = torch.ones_like(batch["language_mask"])

        noisy_actions, flow_time, target_velocity = sample_flow_matching_inputs(
            batch["actions"]
        )

        def compute_loss(
            batch,
            noisy_actions,
            flow_time,
            target_velocity,
            *,
            objective: str = "joint",
            dense_evidence=None,
            metric_tokens=None,
            action_evidence=None,
            metric_g=None,
        ):
            """Compute the selected action or logged-transition World objective."""
            if objective not in {"joint", "va", "world"}:
                raise ValueError(f"unknown training objective: {objective!r}")
            evsm_gates = []
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
                dense_evidence=dense_evidence,
                metric_tokens=metric_tokens,
                action_dense_evidence=action_evidence,
                metric_g=metric_g,
                wmrm_adep_margin=float(getattr(args, "wmrm_adep_margin", 0.05)),
                visual_world_supervision=bool(
                    getattr(args, "visual_world_supervision", False)
                )
                and objective != "va",
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
                wmrm_late_stage_anchor_weight=float(
                    getattr(args, "wmrm_late_stage_anchor_weight", 0.0)
                ),
                wmrm_stage_weight_overrides=visual_world_stage_weight_overrides(
                    args
                ),
                wmrm_feature_metric=getattr(args, "wmrm_feature_metric", "mse"),
                wmrm_progress_ordinal_weight=(
                    float(getattr(args, "wmrm_lang_align_weight", 0.0))
                    if objective != "va"
                    else 0.0
                ),
                summarize_visual_world_metrics=next_global_step % 10 == 0,
                feature_autocast_bf16=bool(args.feature_autocast_bf16),
                train_world_model=(
                    objective != "va"
                    and not bool(getattr(args, "va_only", False))
                ),
                compute_action_output=objective != "world",
            )
            predicted_velocity, action_conditions = rollout
            if objective == "world":
                flow_loss = predicted_velocity.new_zeros(())
                flow_prefix_loss = predicted_velocity.new_zeros(())
                flow_tail_loss = predicted_velocity.new_zeros(())
                pair_loss = predicted_velocity.new_zeros(())
                pred_delta = predicted_velocity.new_zeros(())
                tgt_delta = predicted_velocity.new_zeros(())
            else:
                flow_loss, flow_prefix_loss, flow_tail_loss = masked_flow_matching_loss(
                    predicted_velocity,
                    target_velocity,
                    batch,
                    prefix_steps=args.flow_prefix_steps,
                    prefix_weight=args.flow_prefix_weight,
                    tail_weight=args.flow_tail_weight,
                )
            pair_loss = flow_loss.new_zeros(())
            pred_delta = flow_loss.new_zeros(())
            tgt_delta = flow_loss.new_zeros(())
            if objective == "world":
                future_loss = flow_loss.new_zeros(())
            future_loss = flow_loss.new_zeros(())
            semantic_anchor_loss = flow_loss.new_zeros(())
            semantic_geom_loss = flow_loss.new_zeros(())
            evsm_gate_mean = (
                sum(evsm_gates) / len(evsm_gates) if evsm_gates else None
            )
            # P0-5：动作损失与语义损失分开返回——backward 时 LoRA 参数只缩放
            # 动作侧梯度（η_act），anchor/geometry 梯度完整。
            wmrm_loss = getattr(model, "last_wmrm_loss", None)
            if wmrm_loss is None:
                wmrm_loss = flow_loss.new_zeros(())
            task_action_losses: list[Tensor] = []
            if args.pcgrad and objective != "world":
                instruction_ids = batch["instruction_id"]
                batch_size = int(instruction_ids.shape[0])
                for task_id in torch.unique(instruction_ids, sorted=True):
                    task_mask = instruction_ids == task_id
                    task_batch = {
                        key: (
                            value[task_mask]
                            if isinstance(value, Tensor)
                            and value.ndim > 0
                            and int(value.shape[0]) == batch_size
                            else value
                        )
                        for key, value in batch.items()
                    }
                    task_flow, _, _ = masked_flow_matching_loss(
                        predicted_velocity[task_mask],
                        target_velocity[task_mask],
                        task_batch,
                        prefix_steps=args.flow_prefix_steps,
                        prefix_weight=args.flow_prefix_weight,
                        tail_weight=args.flow_tail_weight,
                    )
                    task_action_losses.append(task_flow)
                if len(task_action_losses) != int(args.mixed_tasks_per_batch):
                    raise RuntimeError(
                        "PCGrad batch task count mismatch: "
                        f"{len(task_action_losses)} != {args.mixed_tasks_per_batch}"
                    )
            adep = getattr(model, "last_wmrm_adep_loss", None)
            if adep is None:
                adep = flow_loss.new_zeros(())
            progress_loss = getattr(model, "last_wmrm_progress_loss", None)
            if progress_loss is None:
                progress_loss = flow_loss.new_zeros(())
            progress_weight = float(getattr(args, "wmrm_lang_align_weight", 0.0))
            if objective == "world" or getattr(args, "wmrm_only", False):
                action_total = (
                    float(getattr(args, "wmrm_world_weight", 1.0)) * wmrm_loss
                    + float(getattr(args, "wmrm_adep_weight", 0.0)) * adep
                    + progress_weight * progress_loss
                )
            elif objective == "va" or getattr(args, "va_only", False):
                action_total = (
                    flow_loss
                    + args.pair_loss_weight * pair_loss
                    + args.future_predict_weight * future_loss
                )
            else:
                action_total = (
                    flow_loss
                    + args.pair_loss_weight * pair_loss
                    + args.future_predict_weight * future_loss
                    + float(getattr(args, "wmrm_world_weight", 1.0)) * wmrm_loss
                    + float(getattr(args, "wmrm_adep_weight", 0.0)) * adep
                    + progress_weight * progress_loss
                )
            semantic_total = (
                flow_loss.new_zeros(())
                if objective == "world" or getattr(args, "wmrm_only", False)
                else args.semantic_anchor_weight * semantic_anchor_loss
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
                task_action_losses,
            )

        # Clear the previous update once. Peer joint mode then accumulates one
        # VA backward and one physically separate World backward before stepping.
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
                task_action_losses,
            ) = compute_loss(
                batch,
                noisy_actions,
                flow_time,
                target_velocity,
                objective="va" if dual_peer_data else "joint",
                dense_evidence=mtvj_dense_evidence,
                metric_tokens=mtvj_metric_tokens,
                action_evidence=action_dense_evidence,
                metric_g=mtvj_metric_g,
            )

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
                *(
                    (f"pcgrad.task_{index}", task_loss)
                    for index, task_loss in enumerate(task_action_losses)
                ),
            ]
        )
        world_action_total = action_total.new_zeros(())
        pcgrad_stats: dict[str, float | int] | None = None
        pcgrad_named_parameters = (
            list(
                named_optimizer_parameters(
                    optimizer,
                    ("e2e_model", e2e_model),
                    ("model", model),
                    ("scene_teacher", scene_teacher),
                    ("vision_backbone", vision_backbone),
                    ("main_vision_backbone", main_vision_backbone),
                    ("servo", servo),
                    ("relation_encoder", relation_encoder),
                    ("metric_head", metric_head),
                )
            )
            if args.pcgrad
            else []
        )
        if dual_peer_data:
            if world_raw_batch is None:
                raise RuntimeError("peer joint step did not fetch a World batch")

            def world_forward():
                nonlocal world_raw_batch
                (
                    world_batch,
                    world_dense_evidence,
                    world_metric_tokens,
                    world_metric_g,
                ) = prepare_peer_world_batch(world_raw_batch)
                world_noisy_actions, world_flow_time, world_target_velocity = (
                    sample_flow_matching_inputs(world_batch["actions"])
                )
                with feature_policy_autocast(
                    device, bool(args.feature_autocast_bf16)
                ):
                    losses = compute_loss(
                        world_batch,
                        world_noisy_actions,
                        world_flow_time,
                        world_target_velocity,
                        objective="world",
                        dense_evidence=world_dense_evidence,
                        metric_tokens=world_metric_tokens,
                        action_evidence=None,
                        metric_g=world_metric_g,
                    )
                validate_finite_update_scalars(
                    [
                        ("world.total", losses[1]),
                        ("world.objective", losses[0]),
                        ("world.flow_excluded", losses[2]),
                    ]
                )
                if getattr(args, "pcgrad_separate_world", False):
                    task_losses = getattr(model, "last_wmrm_task_losses", {})
                    if len(task_losses) != int(args.mixed_tasks_per_batch):
                        raise RuntimeError(
                            "World PCGrad task count mismatch: "
                            f"{len(task_losses)} != {args.mixed_tasks_per_batch}"
                        )
                    weighted = {
                        task_id: float(args.wmrm_world_weight) * loss
                        for task_id, loss in task_losses.items()
                    }
                    validate_finite_update_scalars(
                        [
                            (f"world.pcgrad.task_{task_id}", loss)
                            for task_id, loss in weighted.items()
                        ]
                    )
                    return losses, weighted
                return losses

            if args.pcgrad and getattr(args, "pcgrad_separate_world", False):
                action_private, world_private, shared_dino = (
                    partition_separate_pcgrad_parameters(
                        pcgrad_named_parameters
                    )
                )
                action_stats = backward_pcgrad(
                    task_action_losses,
                    [*action_private, *shared_dino],
                    seed=args.seed + next_global_step,
                    topology=topology,
                    compact_prefixes=("main_vision_backbone.",),
                )
                action_gradients = pop_update_gradients(
                    [*action_private, *shared_dino]
                )
                world_losses, world_task_losses = world_forward()
                world_stats = backward_pcgrad(
                    list(world_task_losses.values()),
                    [*world_private, *shared_dino],
                    seed=args.seed + next_global_step,
                    topology=topology,
                    compact_prefixes=("main_vision_backbone.",),
                )
                dino_stats = merge_separate_pcgrad_gradients(
                    action_private,
                    shared_dino,
                    action_gradients,
                )
                pcgrad_stats = {
                    "conflicts": action_stats["conflicts"],
                    "comparisons": action_stats["comparisons"],
                    "world_conflicts": world_stats["conflicts"],
                    "world_comparisons": world_stats["comparisons"],
                    **dino_stats,
                }
            elif args.pcgrad:
                pcgrad_stats, world_losses = backward_pcgrad(
                    task_action_losses,
                    pcgrad_named_parameters,
                    seed=args.seed + next_global_step,
                    topology=topology,
                    auxiliary_loss_or_forward=world_forward,
                )
            else:
                world_losses = backward_peer_joint_losses(
                    action_total, world_forward
                )
            world_action_total = world_losses[0]
            total_loss = total_loss.detach() + world_losses[1].detach()
        else:
            # P0-5：动作损失与语义损失分开 backward——LoRA 参数只缩放
            # 动作侧梯度，anchor/geometry 梯度完整。
            if args.pcgrad:
                pcgrad_stats = backward_pcgrad(
                    task_action_losses,
                    pcgrad_named_parameters,
                    seed=args.seed + next_global_step,
                    topology=topology,
                )
            else:
                action_total.backward()
        aux_parts: dict[str, float] = {}
        clip_params = model.parameters()
        if main_vision_backbone is not None:
            clip_params = [
                *clip_params,
                *(
                    parameter
                    for parameter in main_vision_backbone.parameters()
                    if parameter.requires_grad
                ),
            ]
        relation_gradient_norm = None
        metric_head_gradient_norm = None
        metric_clip_params: list[Tensor] = []
        update_named_parameters = list(
            named_optimizer_parameters(
                optimizer,
                ("e2e_model", e2e_model),
                ("model", model),
                ("scene_teacher", scene_teacher),
                ("vision_backbone", vision_backbone),
                ("main_vision_backbone", main_vision_backbone),
                ("servo", servo),
                ("relation_encoder", relation_encoder),
                ("metric_head", metric_head),
            )
        )
        validate_optimizer_update_state(optimizer, validate_values=False)
        if not args.pcgrad:
            reduce_update_gradients(update_named_parameters, topology)
        main_parameter_ids = {id(parameter) for parameter in clip_params}
        main_named_parameters = [
            (name, parameter)
            for name, parameter in update_named_parameters
            if id(parameter) in main_parameter_ids
        ]
        gradient_norm, predictor_clip_norm = clip_main_and_optional_predictor_gradients(
            main_named_parameters,
            predictor_max_norm=getattr(args, "wmrm_predictor_grad_clip", None),
            main_max_norm=1.0,
        )
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
        raw_gradient_norm = validate_preclip_gradient_norms(
            gradient_norm,
            predictor_clip_norm,
            metric_clip_norm,
            max_norm=args.max_gradient_norm,
        )
        if raw_gradient_norm > 100.0:
            # The ordinary path has one device synchronization per clip group.
            # Per-parameter norms are reserved for the exceptional spike path.
            largest_gradients = sorted(
                (
                    (float(parameter.grad.detach().double().norm().item()), name)
                    for name, parameter in update_named_parameters
                    if parameter.grad is not None
                ),
                reverse=True,
            )[:12]
            print(
                "gradient_spike "
                f"global_step={next_global_step} raw_norm={raw_gradient_norm:.6g} "
                + " ".join(
                    f"{name}={norm:.6g}" for norm, name in largest_gradients
                ),
                flush=True,
            )
        optimizer.step()
        validate_optimizer_update_state(optimizer, validate_values=False)
        commit_successful_update(step, consumed_locality_batch)
        if prefetch_after_commit:
            peer_batch_prefetcher.fill(
                peer_prefetch_fill_limit(
                    args.steps - step, sampler, world_sampler
                )
            )
        gate_log = (
            f" evsm_gate={evsm_gate_mean:.3f}" if evsm_gate_mean is not None else ""
        )
        if global_step % 10 == 0:
            bridge = getattr(model, "dino_qwen_bridge", None)
            if bridge is not None:
                bridge_gates = bridge.gates.detach().abs()
                gate_log += (
                    f" bridge_alpha_abs_mean={float(bridge_gates[:, 0].mean()):.6f}"
                    f" bridge_beta_abs_mean={float(bridge_gates[:, 1].mean()):.6f}"
                )
            readout = getattr(model, "va_last3_readout", None)
            if readout is not None:
                readout_gates = readout.gates.detach().abs()
                gate_log += "".join(
                    f" va_last3_gate{index}_abs={float(gate):.6f}"
                    for index, gate in enumerate(readout_gates)
                )
            flow_head = getattr(model, "flow_head", None)
        semantic_log = ""
        compile_step_log = ""
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
        pcgrad_log = ""
        if pcgrad_stats is not None:
            pcgrad_log = (
                f" pcgrad={pcgrad_stats['conflicts']}/"
                f"{pcgrad_stats['comparisons']} "
                f"anchor={args.anchor_replay_fraction:.3f}"
            )
            if "world_conflicts" in pcgrad_stats:
                label = (
                    "wm_pcgrad"
                    if getattr(args, "pcgrad_separate_world", False)
                    else "wm_guard"
                )
                pcgrad_log += (
                    f" {label}={pcgrad_stats['world_conflicts']}/"
                    f"{pcgrad_stats['world_comparisons']}"
                )
            if "dino_cosine" in pcgrad_stats:
                pcgrad_log += (
                    f" dino_cos={pcgrad_stats['dino_cosine']:.5f}->"
                    f"{pcgrad_stats['dino_post_cosine']:.5f}"
                    f" dino_projected={pcgrad_stats['dino_projected']}"
                )
        progress_log = ""
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
        if predictor_clip_norm is not None:
            metric_head_log += f" predictor_clip={float(predictor_clip_norm):.6f}"
        servo_log = ""
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
        recovery_mask = batch.get("recovery_mask")
        action_valid_mask = batch.get("action_valid_mask")
        recovery_rows = 0.0
        if isinstance(recovery_mask, torch.Tensor):
            recovery_valid = recovery_mask.bool()
            if isinstance(action_valid_mask, torch.Tensor):
                recovery_valid &= action_valid_mask.bool()
            recovery_rows = float(
                recovery_valid.flatten(1).any(dim=1).float().mean()
            )
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
        if (
            next_global_step % 10 == 0
            and getattr(model, "last_world_no_regression_loss", None) is not None
        ):
            late_anchor = getattr(model, "last_world_late_stage_anchor_loss", None)
            world_constraint_log = (
                f" world_base={float(model.last_wmrm_base_loss):.6f}"
                f" world_guard={float(model.last_world_no_regression_loss):.6f}"
                " world_static_constraint="
                f"{float(model.last_world_static_constraint_loss):.6f}"
                f" world_action_rank={float(model.last_world_action_rank_loss):.6f}"
                " world_late_anchor="
                f"{0.0 if late_anchor is None else float(late_anchor):.6f}"
            )
        resources = runtime_resource_stats(device)
        resource_log = (
            f" resources[rss={resources['rss_mib']:.1f}MiB "
            f"fd={resources['fd_count']} "
            f"cuda={resources['cuda_allocated_mib']:.1f}/"
            f"{resources['cuda_reserved_mib']:.1f}MiB]"
        )
        if topology.is_primary:
            print(
                f"step={global_step} mode={args.mode} contract="
                f"{'e2e_single' if e2e_model is not None else ('single' if args.single_task else 'paired')} "
                f"task={task_log} action_valid={float(valid_fraction):.4f} "
                f"recovery_rows={recovery_rows:.4f} "
                f"sequence={noisy_actions.shape[1]} "
                f"loss={total_loss.item():.6f} flow={flow_loss.item():.6f} "
                f"flow_first{min(args.flow_prefix_steps, noisy_actions.shape[-2])}="
                f"{flow_prefix_loss.item():.6f} "
                f"flow_tail{max(noisy_actions.shape[-2] - args.flow_prefix_steps, 0)}="
                f"{flow_tail_loss.item():.6f} "
                f"pair={pair_loss.item():.6f} future={future_loss.item():.6f} "
                f"world_objective={world_action_total.item():.6f} "
                f"world={float((getattr(model, 'last_wmrm_loss', None) if model is not None else None) or 0.0):.6f} "
                f"goal_delta={predicted_delta.item():.6f}/"
                f"{target_delta.item():.6f} grad={float(gradient_norm):.6f}"
                f"{relation_log}{metric_head_log}{pcgrad_log}{progress_log}"
                f"{aux_log}{gate_log}{semantic_log}"
                f"{compile_step_log}{servo_log}{world_task_log}{resource_log}"
                f"{world_constraint_log}"
            )

    if peer_batch_prefetcher is not None:
        peer_batch_prefetcher.close()
    if topology.is_distributed:
        data_parallel_barrier(topology)
    final_save_due = final_checkpoint_save_due(
        args.save, global_step, last_saved_global_step
    )
    if final_save_due:
        consolidate_zero_optimizer_state(optimizer, to=0)
        if topology.is_primary:
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
                world_sampler=world_sampler,
                exact_run_contract=runtime_exact_run_contract,
                main_vision_backbone=main_vision_backbone,
            )
    shutdown_data_parallel(topology)


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