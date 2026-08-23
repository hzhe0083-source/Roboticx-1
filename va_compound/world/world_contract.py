"""Visual-World supervision and peer VA/World contract definitions + validation.

The contract constants pin the World loss graph, peer topology, data isolation
and action-ranking semantics. The validators prove that a training split and a
checkpoint agree with those contracts before model startup or exact resume.

``validate_visual_world_training_split`` performs file/stat hashing via
``va_compound.exact_resume._sha256_file``; ``validate_visual_world_resume_contract``
reuses the ``WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION`` migration id from the
same module. Both are imported at module scope — exact_resume is a leaf module
with no back-reference to this one, so there is no cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from va_compound.exact_resume import (
    WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION,
    _sha256_file,
)


WORLD_SUPERVISION_CONTRACT = "visual_motion_oracle_stgap_v7"
WORLD_TRANSITION_CONTRACT = "current_first6_and_next_first_v1"
WORLD_LOSS_COMPONENT_WEIGHTS = {
    "all": 0.25,
    "motion": 0.25,
    "top20": 0.50,
}
WORLD_STAGE_AUXILIARY_DECAY = 0.25
WORLD_STAGE_AUXILIARY_FLOOR = 0.1
WORLD_LATE_STAGE_ANCHOR_SOURCE = "visual_plus_no_regression_objective"
WORLD_LATE_STAGE_ANCHOR_STAGE_WEIGHTS = {
    5: 0.5,
    6: 1.0,
}


def world_late_stage_anchor_contract(weight: float = 0.0) -> dict[str, object]:
    """Extra S5/S6 objective that is added after the normalized stage mean.

    The default weight is 0 so existing exact-resume checkpoints keep the old
    loss graph. A positive weight does not renormalize S7's share of
    ``objective_world_loss``.
    """

    return {
        "weight": float(weight),
        "stage_weights": dict(WORLD_LATE_STAGE_ANCHOR_STAGE_WEIGHTS),
        "source": WORLD_LATE_STAGE_ANCHOR_SOURCE,
    }


WORLD_LATE_STAGE_ANCHOR = world_late_stage_anchor_contract(0.0)
WORLD_LOGGED_BRANCH_CONTRACT = "matched_context_full_forward_v1"
WORLD_ACTION_DONOR_CONTRACT = "train_split_task_cross_episode_proprio_nearest_v1"
# VA has N layers; World proposes on layers 0..N-2. Each proposal predicts the
# same next-decision endpoint from the current DINO anchor. The previous stage
# map is detached refinement context, never an additive physical-state base.
PEER_WORLD_TOPOLOGY_CONTRACT = (
    "world_minus_one_same_endpoint_fixed_current_anchor_v2"
)
PEER_WORLD_ACTION_SOURCE_CONTRACT = "deterministic_readout_main_explicit_env_override_supervision_v1"
# The policy consumes the predicted map value, while its action loss trains the
# map-to-policy projection/readers rather than changing the physical predictor.
# World reconstruction and action-counterfactual objectives own the dynamics.
PEER_GRADIENT_BOUNDARY_CONTRACT = (
    "world_map_stopgrad_policy_projection_trainable_v1"
)
PEER_LEGACY_TOPOLOGY_CONTRACT = "one_stage_delayed_bidirectional_state_kv_v1"
PEER_LEGACY_GRADIENT_BOUNDARY_CONTRACT = (
    "fully_differentiable_bidirectional_messages_v1"
)
PEER_WORLD8_TO_WORLD7_REPAIR_MIGRATION = (
    "peer_world8_h6_to_world7_h15_fixed_anchor_fresh_world_v3"
)
PEER_H15_PREFIX_TAIL_FLOW_CONTRACT = (
    "h6_prefix_h9_tail_one_way_detached_flow_v1"
)
PEER_H15_PREFIX_TAIL_FLOW_MIGRATION = (
    "peer_h15_uniform_to_prefix_tail_flow_from_s1752_v1"
)
PEER_DATA_ISOLATION_CONTRACT = "separate_va_world_episode_datasets_per_step_v1"
PEER_DUAL_STREAM_OPTIMIZER_CONTRACT = (
    "va_backward_then_world_backward_one_optimizer_step_v1"
)
PEER_PLANNING_STRIDES = frozenset({1, 2, 3, 6})
PEER_HIGH_FREQUENCY_CONTRACT = {
    "action_prediction": "full_action_chunk_each_decision_v2",
    "world_transition": "logged_world_horizon_action_chunk_v2",
    "world_target": "explicit_endpoint_at_world_horizon_v2",
    "readout_auxiliary": "all_world_stages_full_logged_action_chunk_mean_v3",
}
# The formal H15/P15 checkpoint predates all-stage readout supervision.  Keep
# its exact source semantics named so weights-only initialization can accept
# only this one known transition without weakening any action/World contract.
PEER_READOUT_V2_HIGH_FREQUENCY_CONTRACT = {
    **PEER_HIGH_FREQUENCY_CONTRACT,
    "readout_auxiliary": "full_logged_action_chunk_v2",
}
PEER_LEGACY_HIGH_FREQUENCY_CONTRACT = {
    "action_prediction": "full_h6_each_decision_v1",
    "world_transition": "logged_h6_prefix_planning_stride_v1",
    "world_target": "adjacent_decision_at_data_stride_v1",
    "readout_auxiliary": "full_logged_h6_v1",
}
PEER_WORLD_READOUT_CONTRACT = {
    "va_stream": "causal_deterministic_action_chunk_readout_v2",
    "world_stream": "explicit_logged_chunk_at_world_horizon_v2",
    "loss": "all_stage_mean_logged_chunk_auxiliary_without_forward_label_injection_v3",
}
PEER_WORLD_READOUT_V2_CONTRACT = {
    **PEER_WORLD_READOUT_CONTRACT,
    "loss": "logged_chunk_auxiliary_without_forward_label_injection_v2",
}
PEER_READOUT_V2_TO_V3_WEIGHTS_MIGRATION = (
    "peer_readout_final_to_all_stage_mean_weights_only_v1"
)
ASSEMBLY_METRIC_ROLE_WEIGHTS_MIGRATION = (
    "assembly_metric_duplicate_handle_to_reward_center_weights_only_v1"
)
PEER_WEIGHTS_SEMANTIC_MIGRATION_CONTRACT = (
    "peer_resume_weights_semantic_migrations_v1"
)
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


def validate_visual_world_training_split(
    payload: dict,
    data_path: Path,
    manifest_path: Path,
    *,
    va_world_mode: str = "legacy",
    planning_stride: int = 6,
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
    if Path(str(manifest.get("manifest_path", ""))).name != resolved_manifest.name:
        raise ValueError("World split manifest_path does not match the supplied file")

    metadata = payload.get("metadata") or {}
    if metadata.get("split_name") != "train":
        raise ValueError("visual World training requires metadata.split_name='train'")
    if metadata.get("split_contract") != manifest:
        raise ValueError("embedded split_contract differs from the external manifest")
    if metadata.get("split_manifest_sha256") != manifest_sha:
        raise ValueError("training payload split_manifest_sha256 mismatch")
    if Path(str(metadata.get("split_manifest_path", ""))).name != resolved_manifest.name:
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
    peer_horizon = int((payload.get("metadata") or {}).get("action_horizon", 6))
    if peer_mode and peer_horizon not in {6, 15}:
        raise ValueError("peer action horizon must be H6 or H15")
    expected_shape = (4, peer_horizon, 4) if peer_mode else (4, 48, 4)
    if peer_mode and planning_stride not in PEER_PLANNING_STRIDES:
        raise ValueError("peer planning_stride must be one of 1/2/3/6")
    expected_protocol = (
        f"peer_sync_h{peer_horizon}_p{planning_stride}_world_windows_v1"
        if peer_mode and (peer_horizon != 6 or planning_stride != 6)
        else PEER_SYNC_H6_CONTRACT
        if peer_mode
        else MANIFEST_CONTRACT
    )
    metadata_contract = metadata.get("contract")
    manifest_protocol = (manifest.get("data_protocol") or {}).get("contract")
    if tuple(actions.shape[1:]) != expected_shape:
        label = "peer_sync_h6" if peer_mode else "legacy visual World"
        raise ValueError(
            f"{label} training requires T={expected_shape[0]}/H={expected_shape[1]}/"
            f"A={expected_shape[2]}, got {tuple(actions.shape[1:])}"
        )
    if peer_mode:
        if metadata_contract != expected_protocol:
            raise ValueError(
                f"peer_sync_h6 requires metadata.contract={expected_protocol!r}"
            )
        if metadata.get("logged_action_chunk") != f"full_h{peer_horizon}":
            raise ValueError(
                f"peer mode requires the full logged H{peer_horizon} action chunk"
            )
        if int(metadata.get("control_stride", -1)) != planning_stride:
            raise ValueError(
                "peer_sync_h6 data control_stride must equal planning_stride "
                f"({planning_stride})"
            )
        if planning_stride != 6 and int(
            metadata.get("planning_stride", -1)
        ) != planning_stride:
            raise ValueError(
                "high-frequency peer data metadata.planning_stride must equal "
                f"{planning_stride}"
            )
        expected_offsets = [
            index * planning_stride for index in range(expected_shape[0])
        ]
        if planning_stride != 6 and metadata.get("decision_offsets") != expected_offsets:
            raise ValueError(
                "high-frequency peer data decision_offsets must be adjacent "
                f"planning decisions: {expected_offsets}"
            )
        for key in ("parent_identity", "source_identities", "output_identity"):
            if not metadata.get(key):
                raise ValueError(f"peer_sync_h6 requires metadata.{key}")
        if peer_horizon == 15:
            target_valid = payload.get("world_target_valid_mask")
            target_refs = payload.get("world_target_frame_refs")
            if (
                not isinstance(target_valid, Tensor)
                or target_valid.dtype != torch.bool
                or tuple(target_valid.shape) != tuple(actions.shape[:2])
            ):
                raise ValueError("H15 requires world_target_valid_mask bool [N,T]")
            if not isinstance(target_refs, (list, tuple)) or len(target_refs) != len(actions):
                raise ValueError("H15 requires one world_target_frame_ref per sample")
            if metadata.get("world_target_horizon") != 15:
                raise ValueError("H15 requires world_target_horizon=15")
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
    if not actual_tasks:
        raise ValueError("visual World joint training requires at least one task")

    splits = manifest.get("splits") or {}
    train_contract = splits.get("train") or {}
    eval_contract = splits.get("eval") or {}
    if Path(str(train_contract.get("output_path", ""))).name != resolved_data.name:
        raise ValueError("World split train output_path does not match the World dataset")
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

    frame_refs = payload.get("frame_refs")
    if not isinstance(frame_refs, (list, tuple)) or len(frame_refs) != len(actions):
        raise ValueError("visual World training requires one frame_ref per row")
    if peer_mode and peer_horizon == 15:
        target_refs = payload["world_target_frame_refs"]
        for current_ref, target_ref in zip(frame_refs, target_refs, strict=True):
            if (
                not isinstance(current_ref, (list, tuple))
                or len(current_ref) < 2
                or not isinstance(target_ref, (list, tuple))
                or len(target_ref) < 2
                or target_ref[0] != current_ref[0]
                or target_ref[1] != current_ref[1]
            ):
                raise ValueError(
                    "H15 world_target_frame_ref task/episode mismatch"
                )
    names_by_task: dict[int, set[str]] = {}
    for task_id, ref in zip(task_ids.tolist(), frame_refs, strict=True):
        if not isinstance(ref, (list, tuple)) or not ref:
            raise ValueError("visual World frame_ref must start with a task name")
        names_by_task.setdefault(int(task_id), set()).add(str(ref[0]))
    inconsistent_names = {
        task_id: sorted(names)
        for task_id, names in names_by_task.items()
        if len(names) != 1
    }
    if inconsistent_names:
        raise ValueError(
            "visual World task ids map to inconsistent frame names: "
            f"{inconsistent_names}"
        )
    expected_names = {
        task_id: next(iter(names_by_task[task_id])) for task_id in actual_tasks
    }
    manifest_tasks = {
        int(item["task_id"]): str(item.get("task_name"))
        for item in manifest.get("tasks", [])
    }
    if manifest_tasks != expected_names:
        raise ValueError(
            "World split task names differ from payload frame refs: "
            f"{manifest_tasks} != {expected_names}"
        )

    transition_rule = manifest.get("transition_rule") or {}
    explicit_target_valid = payload.get("world_target_valid_mask")
    transition_prefix = (
        int(metadata.get("world_target_horizon", -1))
        if explicit_target_valid is not None
        else planning_stride if peer_mode else 6
    )
    if int(transition_rule.get("current_action_prefix_steps", -1)) != (
        transition_prefix
    ):
        raise ValueError("World split transition prefix does not match World horizon")
    transition = (
        explicit_target_valid
        if explicit_target_valid is not None
        else split_transition_mask(
            action_valid,
            prefix_steps=planning_stride if peer_mode else 6,
        )
    )
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
    if sorted(task_contracts) != actual_tasks:
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
    source_candidate = resolved_data.parent / Path(str(source.get("path", ""))).name
    if source_candidate.exists():
        source_path = source_candidate.resolve(strict=True)
    else:
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


def validate_peer_data_isolation(
    va_payload: dict,
    world_payload: dict,
    *,
    planning_stride: int | None = None,
) -> dict[str, object]:
    """Prove that peer VA and World supervision use disjoint episode rows."""

    def integer_vector(payload: dict, key: str, stream: str) -> Tensor:
        value = payload.get(key)
        if (
            not isinstance(value, Tensor)
            or value.ndim != 1
            or value.dtype == torch.bool
            or value.is_floating_point()
        ):
            raise ValueError(f"{stream} {key} must be an integer [N] tensor")
        return value.to(torch.int64)

    va_episode = integer_vector(va_payload, "episode_id", "VA")
    world_episode = integer_vector(world_payload, "episode_id", "World")
    va_task = integer_vector(va_payload, "instruction_id", "VA")
    world_task = integer_vector(world_payload, "instruction_id", "World")
    va_episodes = {int(value) for value in va_episode.tolist()}
    world_episodes = {int(value) for value in world_episode.tolist()}
    overlap = sorted(va_episodes & world_episodes)
    if overlap:
        raise ValueError(
            "peer VA/World datasets must be episode-disjoint; overlapping "
            f"episode_id values: {overlap[:12]}"
        )
    va_tasks = sorted(int(value) for value in torch.unique(va_task).tolist())
    world_tasks = sorted(int(value) for value in torch.unique(world_task).tolist())
    if va_tasks != world_tasks:
        raise ValueError(
            "peer VA/World datasets must cover the same task ids: "
            f"VA={va_tasks}, World={world_tasks}"
        )
    for key in ("actions", "proprio", "language_hidden"):
        va_value = va_payload.get(key)
        world_value = world_payload.get(key)
        if not isinstance(va_value, Tensor) or not isinstance(world_value, Tensor):
            raise ValueError(f"peer VA/World datasets both require tensor {key}")
        if tuple(va_value.shape[1:]) != tuple(world_value.shape[1:]):
            raise ValueError(
                f"peer VA/World {key} schema mismatch: "
                f"{tuple(va_value.shape[1:])} vs {tuple(world_value.shape[1:])}"
            )
    if planning_stride is not None:
        for stream, payload in (("VA", va_payload), ("World", world_payload)):
            metadata = payload.get("metadata") or {}
            if int(metadata.get("control_stride", -1)) != planning_stride:
                raise ValueError(
                    f"{stream} data control_stride must equal planning_stride "
                    f"({planning_stride})"
                )
            if planning_stride != 6 and int(
                metadata.get("planning_stride", -1)
            ) != planning_stride:
                raise ValueError(
                    f"{stream} data metadata.planning_stride must equal "
                    f"{planning_stride}"
                )
    return {
        "contract": PEER_DATA_ISOLATION_CONTRACT,
        "va_episode_count": len(va_episodes),
        "world_episode_count": len(world_episodes),
        "task_ids": va_tasks,
    }


def validate_peer_resume_weights_contract(
    contract: dict,
    *,
    planning_stride: int,
    migrating_peer_world: bool = False,
    migrating_prefix_tail_flow: bool = False,
    action_horizon: int | None = None,
    world_horizon: int | None = None,
    deployment_execution_horizon: int | None = None,
    peer_flow_topology: str | None = None,
    assembly_metric_role_contract: str | None = None,
) -> dict[str, object] | None:
    """Validate peer weights-only initialization and name known migrations.

    Exact resume is intentionally handled by
    :func:`validate_visual_world_resume_contract`, which requires the current
    contracts verbatim.  This validator permits only the two shape-compatible
    weights-only semantic transitions needed by the current run: final-only to
    all-stage readout supervision, and the absent legacy assembly role contract
    to the explicitly supplied reward-center contract.
    """

    expected = {
        "peer_training_mode": "joint_dual_stream",
        "peer_world_topology": (
            PEER_LEGACY_TOPOLOGY_CONTRACT
            if migrating_peer_world
            else PEER_WORLD_TOPOLOGY_CONTRACT
        ),
        "peer_gradient_boundary": (
            PEER_LEGACY_GRADIENT_BOUNDARY_CONTRACT
            if migrating_peer_world
            else PEER_GRADIENT_BOUNDARY_CONTRACT
        ),
        "peer_data_isolation": PEER_DATA_ISOLATION_CONTRACT,
        "peer_dual_stream_optimizer": PEER_DUAL_STREAM_OPTIMIZER_CONTRACT,
        "peer_world_action_source": PEER_WORLD_ACTION_SOURCE_CONTRACT,
        "planning_stride": int(planning_stride),
        "planning_hz": 80.0 / int(planning_stride),
        "peer_high_frequency_contract": (
            PEER_LEGACY_HIGH_FREQUENCY_CONTRACT
            if migrating_peer_world
            else PEER_HIGH_FREQUENCY_CONTRACT
        ),
    }
    migrations: list[dict[str, object]] = []
    if not migrating_peer_world:
        expected["peer_world_readout"] = PEER_WORLD_READOUT_CONTRACT
        expected["peer_flow_topology"] = (
            None if migrating_prefix_tail_flow else peer_flow_topology
        )
        if action_horizon is not None and world_horizon is not None:
            expected["world_action_source"] = (
                f"logged_h{int(action_horizon)}_world_horizon_"
                f"{int(world_horizon)}"
            )
        if deployment_execution_horizon is not None:
            expected["deployment_execution_horizon"] = int(
                planning_stride
                if migrating_prefix_tail_flow
                else deployment_execution_horizon
            )
        old_readout = (
            contract.get("peer_high_frequency_contract")
            == PEER_READOUT_V2_HIGH_FREQUENCY_CONTRACT
            and contract.get("peer_world_readout")
            == PEER_WORLD_READOUT_V2_CONTRACT
        )
        if old_readout:
            expected["peer_high_frequency_contract"] = (
                PEER_READOUT_V2_HIGH_FREQUENCY_CONTRACT
            )
            expected["peer_world_readout"] = PEER_WORLD_READOUT_V2_CONTRACT
            migrations.append(
                {
                    "kind": PEER_READOUT_V2_TO_V3_WEIGHTS_MIGRATION,
                    "source_high_frequency": PEER_READOUT_V2_HIGH_FREQUENCY_CONTRACT,
                    "source_readout": PEER_WORLD_READOUT_V2_CONTRACT,
                    "target_high_frequency": PEER_HIGH_FREQUENCY_CONTRACT,
                    "target_readout": PEER_WORLD_READOUT_CONTRACT,
                }
            )

    if assembly_metric_role_contract is not None:
        saved_assembly_contract = contract.get("assembly_metric_role_contract")
        if saved_assembly_contract is None:
            migrations.append(
                {
                    "kind": ASSEMBLY_METRIC_ROLE_WEIGHTS_MIGRATION,
                    "source": None,
                    "target": assembly_metric_role_contract,
                }
            )
        else:
            expected["assembly_metric_role_contract"] = assembly_metric_role_contract

    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "peer --resume-weights requires a joint dual-stream checkpoint "
            f"with the same physical/message contract: {mismatches}"
        )
    if not migrations:
        return None
    return {
        "contract": PEER_WEIGHTS_SEMANTIC_MIGRATION_CONTRACT,
        "migrations": migrations,
    }


def validate_visual_world_resume_contract(
    checkpoint: dict,
    split_identity: dict[str, object],
    action_ranking: dict[str, object] | None = None,
    static_constraint_weight: float = 4.0,
    migration_id: str | None = None,
    va_world_mode: str = "legacy",
    planning_stride: int = 6,
    late_stage_anchor_weight: float = 0.0,
    stage_weight_overrides: dict[int, float] | None = None,
    world_horizon: int | None = None,
    assembly_metric_role_contract: str | None = None,
) -> None:
    """Reject exact continuation from an old or differently split loss graph."""

    from va_compound.world_supervision import canonical_stage_weight_overrides

    contract = dict(checkpoint.get("training_contract") or {})
    if "world_late_stage_anchor" not in contract:
        contract["world_late_stage_anchor"] = world_late_stage_anchor_contract(0.0)
    if "world_stage_weight_overrides" not in contract:
        contract["world_stage_weight_overrides"] = {}
    else:
        contract["world_stage_weight_overrides"] = canonical_stage_weight_overrides(
            contract.get("world_stage_weight_overrides")
        )
    target_horizon = planning_stride if world_horizon is None else int(world_horizon)
    if target_horizon < planning_stride:
        raise ValueError("World target horizon cannot be shorter than planning_stride")
    expected = {
        "world_supervision": WORLD_SUPERVISION_CONTRACT,
        "world_transition": (
            f"explicit_endpoint_h{target_horizon}_v1"
            if target_horizon > planning_stride
            else WORLD_TRANSITION_CONTRACT
            if planning_stride == 6
            else f"current_first{planning_stride}_and_next_first_v1"
        ),
        "world_loss_weights": WORLD_LOSS_COMPONENT_WEIGHTS,
        "world_stage_auxiliary_decay": WORLD_STAGE_AUXILIARY_DECAY,
        "world_stage_auxiliary_floor": WORLD_STAGE_AUXILIARY_FLOOR,
        "world_stage_weight_overrides": canonical_stage_weight_overrides(
            stage_weight_overrides
        ),
        "world_late_stage_anchor": world_late_stage_anchor_contract(
            late_stage_anchor_weight
        ),
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
                "peer_training_mode": "joint_dual_stream",
                "peer_gradient_boundary": PEER_GRADIENT_BOUNDARY_CONTRACT,
                "peer_data_isolation": PEER_DATA_ISOLATION_CONTRACT,
                "peer_dual_stream_optimizer": PEER_DUAL_STREAM_OPTIMIZER_CONTRACT,
                "planning_stride": planning_stride,
                "planning_hz": 80.0 / planning_stride,
                "peer_high_frequency_contract": PEER_HIGH_FREQUENCY_CONTRACT,
            }
        )
    if assembly_metric_role_contract is not None:
        expected["assembly_metric_role_contract"] = assembly_metric_role_contract
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
