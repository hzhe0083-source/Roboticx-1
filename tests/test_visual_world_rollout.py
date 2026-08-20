"""End-to-end CPU contracts for visual-motion World rollout supervision."""

from __future__ import annotations

import copy

import pytest
import torch

import train as train_module
from train import prepare_visual_world_action_ranking, rollout_policy
from va_compound.model import VACompoundConfig, VACompoundPolicy
from va_compound.world_supervision import ActionTop10GapLoss


def _tiny_world_model(*, va_world_mode: str = "legacy") -> VACompoundPolicy:
    config = VACompoundConfig(
        language_dim=12,
        vision_dim=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        action_horizon=6,
        action_dim=4,
        proprio_dim=5,
        flow_layers=1,
        wmrm=True,
        wmrm_target="dino",
        wmrm_cycle_steps=6,
        wmrm_inject="all",
        wmrm_handshake=True,
        wmrm_map_size=2,
        wmrm_map_channels=8,
        wmrm_world_grid=2,
        wmrm_predictor="st_blocks",
        wmrm_predictor_depth=1,
        wmrm_predictor_width=16,
        wmrm_predictor_heads=4,
        main_vision_frames=1,
        main_vision_grid=2,
        va_world_mode=va_world_mode,
    )
    return VACompoundPolicy(config).eval()


def _rollout_batch(*, transitions_valid: bool) -> tuple[dict, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260818)
    batch_size, sequence = 2, 3
    base_vision = torch.randn(batch_size, 1, 4, 8)
    visual_delta = torch.randn(batch_size, sequence, 4, 8) * 0.2
    visual_delta[:, 0].zero_()
    vision_tokens = base_vision + visual_delta.cumsum(dim=1)
    batch = {
        "vision_tokens": vision_tokens,
        "proprio": torch.randn(batch_size, sequence, 5),
        "previous_action": torch.randn(batch_size, sequence, 4),
        "actions": torch.randn(batch_size, sequence, 6, 4).clamp(-1.0, 1.0),
        "action_valid_mask": torch.full(
            (batch_size, sequence, 6), transitions_valid, dtype=torch.bool
        ),
        "language_hidden": torch.randn(batch_size, 4, 12),
        "language_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "instruction_id": torch.tensor([0, 16]),
        "episode_id": torch.tensor([10, 20]),
    }
    batch["world_rank_shuffle_action"] = (
        -batch["actions"][:, :-1, :6]
    ).clone()
    batch["world_rank_shuffle_mask"] = torch.ones(
        batch_size, sequence - 1, dtype=torch.bool
    )
    noisy_actions = torch.randn(batch_size, sequence, 6, 4)
    flow_time = torch.rand(batch_size, sequence)
    return batch, noisy_actions, flow_time


def _run_rollout(*, transitions_valid: bool) -> VACompoundPolicy:
    model = _tiny_world_model()
    batch, noisy_actions, flow_time = _rollout_batch(
        transitions_valid=transitions_valid
    )
    velocities, conditions = rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )
    assert velocities.shape == (2, 3, 6, 4)
    assert conditions.shape == (2, 3, 6, 16)
    return model


def test_two_task_visual_world_rollout_emits_metrics_and_world_gradient() -> None:
    model = _run_rollout(transitions_valid=True)
    loss = model.last_wmrm_loss

    assert loss is not None and loss.requires_grad
    assert torch.isfinite(loss)
    assert set(model.last_visual_world_metrics) == {0, 16}
    required = {
        "world_all",
        "copy_all",
        "world_motion",
        "copy_motion",
        "world_top10",
        "copy_top10",
        "world_static",
        "copy_static",
        "motion_energy",
        "stage_losses",
    }
    for metrics in model.last_visual_world_metrics.values():
        assert required <= set(metrics)
        assert metrics["transitions"] == 2
        assert len(metrics["stage_losses"]) == model.config.num_layers
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics["stage_losses"])

    loss.backward()
    out_weight = model.wmrm.st_predictor.out_proj.weight
    assert out_weight.grad is not None
    assert torch.isfinite(out_weight.grad).all()
    assert float(out_weight.grad.norm()) > 0.0


def test_empty_visual_world_transition_mask_backprops_connected_zero() -> None:
    model = _run_rollout(transitions_valid=False)
    loss = model.last_wmrm_loss

    assert loss is not None and loss.requires_grad
    assert loss.item() == pytest.approx(0.0)
    assert set(model.last_visual_world_metrics) == {0, 16}
    for metrics in model.last_visual_world_metrics.values():
        assert metrics["transitions"] == 0
        for name in (
            "world_all",
            "copy_all",
            "world_motion",
            "copy_motion",
            "world_top10",
            "copy_top10",
            "world_static",
            "copy_static",
            "motion_energy",
        ):
            assert metrics[name] == pytest.approx(0.0)
        assert metrics["stage_losses"] == pytest.approx([0.0, 0.0])

    loss.backward()
    out_weight = model.wmrm.st_predictor.out_proj.weight
    assert out_weight.grad is not None
    assert torch.isfinite(out_weight.grad).all()
    torch.testing.assert_close(out_weight.grad, torch.zeros_like(out_weight.grad))


def test_final_logged_map_recomputes_from_predictor_input_belief() -> None:
    model = _tiny_world_model()
    batch, _, _ = _rollout_batch(transitions_valid=True)
    language_cache = model.build_language_cache(
        batch["language_hidden"], batch["language_mask"]
    )
    logged_action = batch["actions"][:, 0, : model.wmrm.cycle_steps]

    with torch.no_grad():
        model.encode_condition(
            batch["vision_tokens"][:, 0],
            batch["proprio"][:, 0],
            batch["previous_action"][:, 0],
            language_cache=language_cache,
            env_action=logged_action,
            detach_wmrm_stage_state=True,
        )
        auxes = list(model.last_wmrm_auxes)
        pre_actions = list(model.last_wmrm_pre_actions)
        final_aux = auxes[-1]
        assert final_aux.predict_belief is not None
        previous_map = auxes[-2].z_tokens.detach()
        _, _, _, recomputed = model.wmrm.predict_world(
            pre_actions[-1],
            final_aux.proprio,
            final_aux.predict_belief,
            final_aux.task_summary,
            dino_tokens=final_aux.dino_tokens,
            env_action=logged_action,
            previous_map=previous_map,
        )

    assert recomputed is not None
    torch.testing.assert_close(recomputed, final_aux.z_tokens, rtol=0.0, atol=0.0)


def test_action_ranking_donors_are_fixed_in_train_payload() -> None:
    payload = {
        "actions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
             [3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]
        )[:, None, None, :].expand(4, 2, 6, 4).clone(),
        "proprio": torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.1, 0.0]],
                [[10.0, 0.0]],
                [[10.1, 0.0]],
            ]
        ).expand(4, 2, 2).clone(),
        "instruction_id": torch.tensor([0, 0, 0, 0]),
        "episode_id": torch.tensor([1, 1, 2, 2]),
        "action_valid_mask": torch.ones(4, 2, 6, dtype=torch.bool),
    }
    identity = prepare_visual_world_action_ranking(payload)
    assert identity["world_action_donor_transitions"] == 4
    assert identity["world_action_rank_transitions"] == 4
    assert payload["world_rank_shuffle_action"].shape == (4, 1, 6, 4)
    assert payload["world_rank_shuffle_mask"].tolist() == [[True], [True], [True], [True]]
    # Re-running is byte-identical and uses no batch-local/random state.
    second = prepare_visual_world_action_ranking(payload)
    assert identity == second


def test_logged_world_branch_reuses_detached_proposal_entry_memory() -> None:
    model = _tiny_world_model()
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    calls = []
    original = model.encode_condition

    def record(*args, **kwargs):
        calls.append(
            {
                "memory": kwargs.get("visual_memory"),
                "skip_wmrm": bool(kwargs.get("skip_wmrm", False)),
                "logged": bool(kwargs.get("detach_wmrm_stage_state", False)),
                "grad_enabled": torch.is_grad_enabled(),
                "env_action": kwargs.get("env_action"),
            }
        )
        return original(*args, **kwargs)

    model.encode_condition = record
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )

    # Each supervised decision has proposal/main/logged forwards.  The
    # shuffled branch is now a direct matched-context predictor call, so it
    # does not add another encode_condition forward; zero is diagnostic-only.
    assert len(calls) == 8
    assert [call["memory"] for call in calls[:3]] == [None, None, None]
    proposal_memory = calls[3]["memory"]
    main_memory = calls[4]["memory"]
    logged_memory = calls[5]["memory"]
    assert proposal_memory is main_memory
    assert logged_memory is not proposal_memory
    assert calls[3]["skip_wmrm"] is True
    assert calls[5]["logged"] is True
    assert calls[5]["grad_enabled"] is True
    for proposal_layer, logged_layer in zip(
        proposal_memory.layers, logged_memory.layers, strict=True
    ):
        torch.testing.assert_close(
            proposal_layer, logged_layer, rtol=0.0, atol=0.0
        )
        assert logged_layer.grad_fn is None


def test_peer_readout_uses_world_transition_mask_and_final_stage_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_world_model(va_world_mode="peer_sync_h6")
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    batch["actions"].zero_()
    # The first transition has a complete current H6 but is invalid because the
    # next decision's first action is missing.  The second transition is valid.
    batch["action_valid_mask"].fill_(True)
    batch["action_valid_mask"][0, 1, 0] = False
    calls = 0

    def staged_readout(action: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        # Two peer stages per decision: stage 0 predicts 1, final stage predicts 2.
        stage_value = 1.0 if calls % model.config.num_layers == 1 else 2.0
        output = action.new_full(
            (action.shape[0], 6, model.config.action_dim), stage_value
        )
        output[0] *= 5.0
        return output

    monkeypatch.setattr(model.world_action_readout, "forward", staged_readout)
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )

    # Smooth-L1(2, 0) = 1.5.  Stage 0 must not enter the reduction, and the
    # invalid first transition for row 0 must use the same mask as World targets.
    assert model.last_world_action_readout_loss.item() == pytest.approx(1.5)
    assert model.last_world_action_readout_rmse.item() == pytest.approx(2.0)


def test_peer_rollout_rejects_adep_before_any_model_forward() -> None:
    model = _tiny_world_model(va_world_mode="peer_sync_h6")
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)

    with pytest.raises(ValueError, match="exact same immutable stage snapshot"):
        rollout_policy(
            model,
            batch,
            noisy_actions,
            flow_time,
            visual_world_supervision=True,
            wmrm_adep_enabled=True,
            flow_steps=2,
        )


def test_legacy_rollout_keeps_adep_available() -> None:
    model = _tiny_world_model(va_world_mode="legacy")
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        wmrm_adep_enabled=True,
        flow_steps=2,
    )
    assert model.last_wmrm_adep_loss is not None
    assert torch.isfinite(model.last_wmrm_adep_loss)


def test_peer_rollout_uses_one_main_encode_and_explicit_snapshot_overrides() -> None:
    model = _tiny_world_model(va_world_mode="peer_sync_h6")
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    encode_calls = 0
    proposal_actions: list[torch.Tensor] = []
    original_encode = model.encode_condition
    original_propose = model.wmrm.propose

    def record_encode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(*args, **kwargs)

    def record_propose(*args, **kwargs):
        proposal_actions.append(kwargs["env_action"].detach().clone())
        return original_propose(*args, **kwargs)

    model.encode_condition = record_encode
    model.wmrm.propose = record_propose
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )

    assert encode_calls == batch["actions"].shape[1]
    assert len(proposal_actions) == 14
    for time_index in range(2):
        offset = time_index * 6
        main = proposal_actions[offset : offset + 2]
        logged = proposal_actions[offset + 2 : offset + 4]
        assert not any(
            torch.equal(action, batch["actions"][:, time_index, :6])
            for action in main
        )
        assert all(
            torch.equal(action, batch["actions"][:, time_index, :6])
            for action in logged
        )
    assert model.last_world_action_readout_loss is not None
    assert model.last_world_action_readout_loss.requires_grad
    assert torch.isfinite(model.last_world_action_readout_rmse)


def test_action_gap_reduces_only_distinct_shuffle_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_world_model()
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    batch["world_rank_shuffle_mask"][0].fill_(True)
    batch["world_rank_shuffle_mask"][1].fill_(False)

    def fixed_ranking(real, shuffled, *args, **kwargs):
        connected_zero = real.top10_per_sample * 0.0
        return ActionTop10GapLoss(
            loss_per_sample=connected_zero + 1.0,
            error_gap_per_sample=connected_zero,
        )

    monkeypatch.setattr(
        train_module,
        "action_top10_oracle_straight_through_gap_loss",
        fixed_ranking,
    )
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )

    assert model.last_world_action_shuffle_loss.item() == pytest.approx(1.0)
    assert model.last_world_action_zero_loss.item() == pytest.approx(0.0)
    assert model.last_world_action_strong_loss.item() == pytest.approx(0.0)
    assert model.last_world_action_rank_loss.item() == pytest.approx(1.0)


def test_action_gap_caps_each_sample_before_masked_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_world_model()
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    batch["world_rank_shuffle_mask"].fill_(True)

    def fixed_ranking(real, shuffled, *args, **kwargs):
        connected_zero = real.top10_per_sample * 0.0
        per_sample = connected_zero + torch.tensor(
            [0.1, 1.0], device=connected_zero.device, dtype=connected_zero.dtype
        )
        return ActionTop10GapLoss(
            loss_per_sample=per_sample,
            error_gap_per_sample=connected_zero,
        )

    monkeypatch.setattr(
        train_module,
        "action_top10_oracle_straight_through_gap_loss",
        fixed_ranking,
    )
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
        wmrm_action_rank_per_sample_cap=0.2,
    )

    assert model.last_world_action_rank_loss.item() == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("stage_mode", "expected_previous_map_is_none"),
    [("final", [False, False]), ("cycle", [True, False])],
)
def test_action_gap_final_and_cycle_select_the_expected_stage(
    stage_mode: str,
    expected_previous_map_is_none: list[bool],
) -> None:
    model = _tiny_world_model()
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    original = model.wmrm.predict_world
    selected_previous_maps: list[bool] = []

    def record(*args, **kwargs):
        env_action = kwargs.get("env_action")
        if any(
            torch.equal(env_action, batch["world_rank_shuffle_action"][:, time])
            for time in range(2)
        ):
            selected_previous_maps.append(kwargs.get("previous_map") is None)
        return original(*args, **kwargs)

    model.wmrm.predict_world = record
    rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
        world_action_rank_step=0,
        world_action_rank_stage=stage_mode,
    )

    assert selected_previous_maps == expected_previous_map_is_none


def test_counterfactual_replay_preserves_post_logged_rng_state() -> None:
    model_with_rank = _tiny_world_model().train()
    model_without_rank = copy.deepcopy(model_with_rank)
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    batch_without_rank = {key: value for key, value in batch.items()}
    batch_without_rank["action_valid_mask"] = torch.zeros_like(
        batch["action_valid_mask"]
    )

    initial_rng = torch.get_rng_state()
    rollout_policy(
        model_with_rank,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )
    with_rank_rng = torch.get_rng_state()

    torch.set_rng_state(initial_rng)
    rollout_policy(
        model_without_rank,
        batch_without_rank,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
        flow_steps=2,
    )
    without_rank_rng = torch.get_rng_state()

    torch.testing.assert_close(with_rank_rng, without_rank_rng, rtol=0.0, atol=0.0)
