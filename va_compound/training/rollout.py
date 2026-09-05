"""Sequence rollout and logged-transition supervision for flow policies."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from va_compound import VACompoundPolicy
from va_compound.world.wmrm import wmrm_world_loss
from va_compound.world.world_contract import (
    PEER_PLANNING_STRIDES, WORLD_NO_REGRESSION, WORLD_STATIC_COPY_CONSTRAINT,
    WORLD_ACTION_RANKING, WORLD_LOSS_COMPONENT_WEIGHTS,
    WORLD_STAGE_AUXILIARY_DECAY, WORLD_STAGE_AUXILIARY_FLOOR,
)
from va_compound.world.world_supervision import (
    action_top10_oracle_straight_through_gap_loss, late_stage_anchor_loss,
    masked_reduction as masked_world_reduction, stage_supervision_weights,
    static_copy_anchor_loss, transition_mask as world_transition_mask,
    visual_no_regression_loss, visual_world_loss, _summarize_visual_world_metrics,
    _world_task_ids,
)
from va_compound.training.batch import feature_policy_autocast, feature_no_grad_decode_autocast

def wmrm_next_feature_target(
    model: VACompoundPolicy,
    batch: dict[str, Tensor],
    time_index: int,
    *,
    dense_evidence: dict[int, Tensor] | None = None,
    metric_g: Tensor | None = None,
) -> Tensor:
    """Next VA-cycle visual feature. Target encoder is stop-grad (JEPA-style)."""
    explicit_target = batch.get("world_target_map")
    if explicit_target is not None:
        if time_index >= explicit_target.shape[1]:
            raise ValueError("world_target_map lacks the requested decision index")
        return explicit_target[:, time_index].detach()
    nxt = time_index + 1
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
    wmrm_late_stage_anchor_weight: float = 0.0,
    wmrm_stage_weight_overrides: dict[int, float] | None = None,
    wmrm_feature_metric: str = "mse",
    wmrm_progress_ordinal_weight: float = 0.0,
    summarize_visual_world_metrics: bool = True,
    feature_autocast_bf16: bool = False,
    train_world_model: bool = True,
    compute_action_output: bool = True,
) -> tuple[Tensor, Tensor]:
    if world_action_rank_step < 0:
        raise ValueError("world_action_rank_step must be non-negative")
    if world_action_rank_stage not in {"final", "cycle"}:
        raise ValueError("world_action_rank_stage must be 'final' or 'cycle'")
    peer_world_mode = getattr(model.config, "va_world_mode", "legacy") == "peer_sync_h6"
    logged_peer_world_forward = peer_world_mode and visual_world_supervision
    if visual_world_supervision and not train_world_model:
        raise ValueError(
            "visual World supervision cannot run in the VA-only objective path"
        )
    if model.wmrm is not None and not peer_world_mode:
        raise ValueError("training supports peer World only")
    if wmrm_adep_enabled or wmrm_progress_ordinal_weight:
        raise ValueError("action-dependence and ordinal-progress training are retired")
    if any((model.config.plan_resampler, model.config.scene_teacher,
            model.config.local_slots, model.config.direct_head,
            model.config.c2_controller, model.config.future_predict)):
        raise ValueError("rollout supports slot-free conditional flow policies only")
    if any(value is not None for value in (
        text_backbone, scene_teacher, servo, servo_stats, dense_evidence,
        metric_tokens, action_dense_evidence, metric_g,
    )):
        raise ValueError("legacy encoder and servo training inputs are retired")
    language_cache = model.build_language_cache(
        batch["language_hidden"], batch.get("language_mask")
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
    peer_readout_loss_records: list[tuple[Tensor, Tensor, Tensor]] = []
    peer_readout_squared_error_records: list[tuple[Tensor, Tensor, Tensor]] = []
    visual_world_final_records: list[dict[str, Tensor]] = []
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
        explicit_validity = batch.get("world_target_valid_mask")
        transition_validity = (
            explicit_validity.to(dtype=torch.bool)
            if explicit_validity is not None
            else world_transition_mask(
                action_valid_mask,
                cycle_steps=model.wmrm.cycle_steps,
            )
        )
        rank_shuffle_actions = batch.get("world_rank_shuffle_action")
        rank_shuffle_validity = batch.get("world_rank_shuffle_mask")
        expected_actions = (
            batch["actions"].shape[0],
            transition_validity.shape[1],
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
    for time_index in range(batch["actions"].shape[1]):
        pre_step_visual_memory = visual_memory
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
                planning_stride = int(getattr(model.config, "planning_stride", 6))
                if (
                    model.config.action_horizon not in {6, 15, 50}
                    or planning_stride not in PEER_PLANNING_STRIDES
                    or cycle not in {planning_stride, model.config.action_horizon}
                ):
                    raise ValueError(
                        "peer rollout requires H6/H15/H50 prediction with World "
                        "horizon equal to the execution prefix or full chunk"
                    )
                if logged_peer_world_forward:
                    # The World dataset owns logged transition supervision. Feed
                    # the complete logged chunk to the model; the World horizon
                    # selects either its P-step prefix or the full candidate.
                    world_action = batch["actions"][
                        :, time_index, :cycle
                    ].clamp(-1.0, 1.0)
                elif getattr(model, "world_action_readout", None) is None:
                    raise ValueError("peer_sync_h6 requires world_action_readout")
        peer_stage_snapshots: list[tuple[tuple, dict]] = []
        original_peer_propose = None
        if peer_world_mode and visual_world_supervision:
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
                cross_modal_vision_layers=(
                    batch["dino_last4"][:, time_index]
                    if "dino_last4" in batch
                    else None
                ),
                cross_modal_language_layers=batch.get("qwen_last4"),
                visual_memory=pre_step_visual_memory,
                return_visual_memory=True,
                env_action=world_action,
                detach_wmrm_stage_state=bool(
                    getattr(model.config, "wmrm_detach_proposal_stage_state", False)
                ),
            )
        finally:
            if original_peer_propose is not None:
                model.wmrm.propose = original_peer_propose
        proposal_auxes = list(getattr(model, "last_wmrm_auxes", None) or ())
        if not proposal_auxes and getattr(model, "last_wmrm", None) is not None:
            proposal_auxes = [model.last_wmrm]
        proposal_pres = list(getattr(model, "last_wmrm_pre_actions", None) or ())
        proposal_last = getattr(model, "last_wmrm", None)
        if (
            train_world_model
            and model.wmrm is not None
            and (
                "world_target_map" in batch
                or time_index + 1 < batch["actions"].shape[1]
            )
        ):
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
                logged_chunk = batch["actions"][
                    :, time_index, : model.wmrm.cycle_steps
                ]
                if peer_world_mode:
                    logged_auxes = list(proposal_auxes)
                    logged_pres = proposal_pres
                    # H[:P] conditions the World transition, while this auxiliary
                    # always supervises the causal action readout against the full chunk.
                    if not proposal_pres:
                        raise RuntimeError(
                            "peer World stage did not expose its action snapshot"
                        )
                    expected_stages = model.config.wmrm_stage_count()
                    if not (
                        len(proposal_pres) == len(proposal_auxes) == expected_stages
                    ):
                        raise RuntimeError(
                            "peer World stage trace is incomplete: "
                            f"pre_actions={len(proposal_pres)}, "
                            f"auxes={len(proposal_auxes)}, expected={expected_stages}"
                        )
                    stage_readouts = [
                        model.world_action_readout(
                            pre_action[:, : model.wmrm.cycle_steps]
                            if model.config.action_horizon == 50
                            else pre_action
                        )
                        for pre_action in proposal_pres
                    ]
                    if any(readout is None for readout in stage_readouts):
                        raise RuntimeError(
                            "peer World stage did not expose deterministic readout"
                        )
                    logged_readout = logged_chunk.to(dtype=stage_readouts[0].dtype)
                    # Deployment conditions every World stage with its own readout.
                    # Supervise every such action identity, then average over stages
                    # so the auxiliary keeps the same total scale as the old final-only
                    # objective and remains checkpoint-compatible.
                    readout_error = torch.stack(
                        [
                            F.smooth_l1_loss(
                                readout,
                                logged_readout,
                                reduction="none",
                            ).mean(dim=(-1, -2))
                            for readout in stage_readouts
                        ],
                        dim=0,
                    ).mean(dim=0)
                    readout_squared = torch.stack(
                        [
                            (readout - logged_readout).square().mean(dim=(-1, -2))
                            for readout in stage_readouts
                        ],
                        dim=0,
                    ).mean(dim=0)
                    peer_readout_loss_records.append(
                        (task_ids, valid, readout_error)
                    )
                    peer_readout_squared_error_records.append(
                        (task_ids, valid, readout_squared)
                    )
                final_visual = None
                logged_visuals = []
                for inject_i, aux in enumerate(logged_auxes):
                    logged_map = aux.z_tokens
                    if logged_map is None or logged_map.shape != target.shape:
                        raise ValueError(
                            "logged-action World prediction must be the full DINO map: "
                            f"{None if logged_map is None else tuple(logged_map.shape)} "
                            f"vs {tuple(target.shape)}"
                        )
                    visual = visual_world_loss(
                        logged_map,
                        target,
                        current,
                        feature_metric=wmrm_feature_metric,
                    )
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
                    snapshot_args, snapshot_kwargs = peer_stage_snapshots[rank_stage]
                    snapshot_kwargs["env_action"] = rank_shuffle_actions[:, time_index]
                    shuffled_map = model.wmrm.propose(
                        *snapshot_args, **snapshot_kwargs
                    ).aux.z_tokens
                    if shuffled_map is None or shuffled_map.shape != target.shape:
                        raise ValueError(
                            "action-gap World prediction must be the full DINO map"
                        )
                    real_visual = logged_visuals[rank_stage]
                    shuffled_visual = visual_world_loss(
                        shuffled_map,
                        target,
                        current,
                        feature_metric=wmrm_feature_metric,
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
        if compute_action_output:
            velocity = model.flow_velocity(
                condition,
                noisy_actions[:, time_index],
                flow_time[:, time_index],
                semantic_context=None,
            )
        else:
            velocity = condition.new_zeros(
                (
                    condition.shape[0],
                    model.config.action_horizon,
                    model.config.action_dim,
                )
            )
        predicted_velocities.append(velocity)
        action_conditions.append(condition)
    out = (
        torch.stack(
            predicted_velocities,
            dim=1,
        ),
        torch.stack(action_conditions, dim=1),
    )
    model.last_wmrm_progress_loss = None
    if visual_world_stage_records:
        stage_weights = stage_supervision_weights(
            len(visual_world_stage_records),
            auxiliary_decay=WORLD_STAGE_AUXILIARY_DECAY,
            floor=WORLD_STAGE_AUXILIARY_FLOOR,
            overrides=wmrm_stage_weight_overrides,
        )

        def reduce_stage_records(
            records_by_stage: list[list[tuple[Tensor, Tensor, Tensor]]],
            task_id: int | None = None,
        ) -> Tensor:
            values: list[Tensor] = []
            masks: list[Tensor] = []
            weights: list[float] = []
            for stage_weight, records in zip(
                stage_weights, records_by_stage, strict=True
            ):
                for record_task_ids, valid, value in records:
                    values.append(value)
                    masks.append(
                        valid
                        if task_id is None
                        else valid & record_task_ids.eq(task_id)
                    )
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
            task_id: int | None = None,
        ) -> Tensor:
            if not records:
                return objective_world_loss * 0.0
            return masked_world_reduction(
                [value for _, _, value in records],
                [
                    valid
                    if task_id is None
                    else valid & record_task_ids.eq(task_id)
                    for record_task_ids, valid, _ in records
                ],
            )

        def reduce_readout_records(
            records: list[tuple[Tensor, Tensor, Tensor]],
            task_id: int | None = None,
        ) -> Tensor:
            if not records:
                return objective_world_loss * 0.0
            return masked_world_reduction(
                [value for _, _, value in records],
                [
                    valid
                    if task_id is None
                    else valid & record_task_ids.eq(task_id)
                    for record_task_ids, valid, _ in records
                ],
            )

        action_shuffle_loss = reduce_action_records(
            visual_world_action_shuffle_records
        )
        action_zero_loss = objective_world_loss * 0.0
        action_strong_loss = objective_world_loss * 0.0
        action_rank_loss = action_shuffle_loss
        if peer_readout_loss_records:
            readout_loss = reduce_readout_records(peer_readout_loss_records)
            readout_mse = reduce_readout_records(
                peer_readout_squared_error_records
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
        def reduce_single_stage(
            records: list[tuple[Tensor, Tensor, Tensor]],
            task_id: int | None = None,
        ) -> Tensor:
            if not records:
                return objective_world_loss * 0.0
            return masked_world_reduction(
                [value for _, _, value in records],
                [
                    valid
                    if task_id is None
                    else valid & record_task_ids.eq(task_id)
                    for record_task_ids, valid, _ in records
                ],
            )

        if float(wmrm_late_stage_anchor_weight) > 0.0:
            late_stage_anchor = late_stage_anchor_loss(
                [
                    reduce_single_stage(records)
                    for records in visual_world_objective_stage_records
                ],
                weight=float(wmrm_late_stage_anchor_weight),
            )
        else:
            late_stage_anchor = objective_world_loss * 0.0
        model.last_world_late_stage_anchor_loss = late_stage_anchor
        model.last_wmrm_loss = (
            objective_world_loss
            + float(wmrm_static_constraint_weight)
            * static_constraint_loss
            + float(WORLD_ACTION_RANKING["weight"]) * action_rank_loss
            + readout_loss
            + late_stage_anchor
        )
        model.last_wmrm_task_losses = {}
        for task_id in sorted(
            int(value)
            for value in torch.unique(batch["instruction_id"]).detach().cpu().tolist()
        ):
            task_objective = reduce_stage_records(
                visual_world_objective_stage_records, task_id
            )
            task_static = reduce_stage_records(
                visual_world_static_constraint_stage_records, task_id
            )
            task_rank = reduce_action_records(
                visual_world_action_shuffle_records, task_id
            )
            task_readout = reduce_readout_records(
                peer_readout_loss_records, task_id
            )
            task_late = (
                late_stage_anchor_loss(
                    [
                        reduce_single_stage(records, task_id)
                        for records in visual_world_objective_stage_records
                    ],
                    weight=float(wmrm_late_stage_anchor_weight),
                )
                if float(wmrm_late_stage_anchor_weight) > 0.0
                else task_objective * 0.0
            )
            model.last_wmrm_task_losses[task_id] = (
                task_objective
                + float(wmrm_static_constraint_weight) * task_static
                + float(WORLD_ACTION_RANKING["weight"]) * task_rank
                + task_readout
                + task_late
            )
        model.last_visual_world_metrics = (
            _summarize_visual_world_metrics(
                visual_world_final_records,
                visual_world_stage_records,
            )
            if summarize_visual_world_metrics
            else {}
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
        model.last_world_late_stage_anchor_loss = model.last_wmrm_loss * 0.0
        model.last_wmrm_progress_loss = model.last_wmrm_loss * 0.0
        model.last_wmrm_progress_counts = {}
        model.last_visual_world_metrics = {}
        model.last_wmrm_task_losses = {}
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
        model.last_world_late_stage_anchor_loss = None
        model.last_wmrm_progress_loss = None
        model.last_wmrm_progress_counts = {}
        model.last_visual_world_metrics = {}
        model.last_wmrm_task_losses = {}
    model.last_wmrm_adep_loss = None
    return out
