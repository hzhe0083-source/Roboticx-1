"""Contracts for delayed VA↔World state exchange."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from unittest import mock

import pytest
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy, VisualMemory
from va_compound.wmrm import (
    ExecutableActionReadout,
    WAM4VA,
    WAMProposal,
    WAMState,
    wmrm_world_loss,
)


def _block() -> WAM4VA:
    torch.manual_seed(7)
    return WAM4VA(
        16,
        world_dim=8,
        proprio_dim=3,
        num_heads=4,
        n_belief=2,
        n_evidence=2,
        cycle_steps=6,
        dino_dim=8,
        map_size=2,
        map_frames=2,
        map_grid=2,
        world_grid=2,
        env_action_dim=4,
    ).eval()


def _block_inputs():
    torch.manual_seed(11)
    return (
        torch.randn(3, 6, 16),
        torch.randn(3, 5, 16),
        torch.randn(3, 3),
        torch.randn(3, 8, 8),
        torch.randn(3, 6, 4),
    )


def _peer_config(**overrides) -> VACompoundConfig:
    base = dict(
        language_dim=12,
        vision_dim=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        action_horizon=6,
        action_dim=4,
        proprio_dim=3,
        dropout=0.0,
        wmrm=True,
        wmrm_cycle_steps=6,
        wmrm_inject="all",
        wmrm_map_size=2,
        wmrm_map_channels=8,
        wmrm_world_grid=2,
        main_vision_grid=2,
        main_vision_frames=2,
        main_vision_tokens=8,
        va_world_mode="peer_sync_h6",
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def _policy_inputs(config: VACompoundConfig):
    torch.manual_seed(31)
    return (
        torch.randn(2, 8, config.vision_dim),
        torch.randn(2, config.proprio_dim),
        torch.randn(2, config.action_dim),
        torch.randn(2, 5, config.language_dim),
        torch.ones(2, 5, dtype=torch.bool),
    )


def test_state_is_frozen_and_tensor_operations_cover_all_fields() -> None:
    belief = torch.randn(3, 2, 16, requires_grad=True)
    innovation = torch.randn(3, 2, 16, requires_grad=True)
    world_map = torch.randn(3, 8, 2, 2, requires_grad=True)
    state = WAMState(belief, innovation, world_map)

    with pytest.raises(FrozenInstanceError):
        state.belief = None
    assert not state.detach().belief.requires_grad
    assert state.to(dtype=torch.float64).belief.dtype == torch.float32
    selected = state.index_select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.belief, belief[[2, 0]])
    torch.testing.assert_close(selected.innovation, innovation[[2, 0]])
    torch.testing.assert_close(selected.world_map, world_map[[2, 0]])


def test_world_transition_is_state_only_and_does_not_mutate_snapshot() -> None:
    block = _block()
    action, vision, proprio, dino, env_action = _block_inputs()
    state = WAMState(
        belief=torch.randn(3, 2, 16),
        innovation=torch.randn(3, 2, 16),
        world_map=torch.randn(3, 8, 2, 2),
    )
    before = tuple(x.clone() for x in (state.belief, state.innovation, state.world_map))

    with torch.no_grad():
        transition = block.propose(
            action,
            vision,
            proprio,
            state=state,
            dino_tokens=dino,
            env_action=env_action,
        )

    assert isinstance(transition, WAMProposal)
    assert transition.world_message.ndim == 3
    assert not hasattr(transition, "action_delta")
    assert not hasattr(transition, "vision_delta")
    assert not hasattr(block, "mixed_residual")
    assert not hasattr(block, "mix_world_into_vision")
    for current, original in zip(
        (state.belief, state.innovation, state.world_map), before, strict=True
    ):
        torch.testing.assert_close(current, original, rtol=0.0, atol=0.0)

    detached = transition.detach()
    assert not detached.world_message.requires_grad
    converted = transition.to(dtype=torch.float64)
    assert converted.world_message.dtype == torch.float64
    selected = transition.index_select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.world_message, transition.world_message[[2, 0]])


def test_executable_action_readout_is_deterministic_and_bounded() -> None:
    torch.manual_seed(3)
    readout = ExecutableActionReadout(hidden_dim=16, action_dim=4)
    action = torch.randn(2, 6, 16)
    first = readout(action)
    second = readout(action)
    assert first.shape == (2, 6, 4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool((first >= -1.0).all() and (first <= 1.0).all())
    assert ExecutableActionReadout(hidden_dim=16, action_dim=4, horizon=15)(
        torch.randn(2, 15, 16)
    ).shape == (2, 15, 4)
    with pytest.raises(ValueError, match="positive"):
        ExecutableActionReadout(hidden_dim=16, action_dim=4, horizon=0)


def test_executable_action_readout_runtime_checks_default_on_and_can_skip() -> None:
    action = torch.zeros(2, 6, 16)
    action[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="readout input"):
        ExecutableActionReadout(16, action_dim=4)(action)

    unchecked = ExecutableActionReadout(
        16,
        action_dim=4,
        runtime_integrity_checks=False,
    )
    with mock.patch(
        "va_compound.world.wmrm._require_finite",
        side_effect=AssertionError("finite check entered"),
    ):
        output = unchecked(action)
    assert torch.isnan(output).any()


@pytest.mark.parametrize("planning_stride", [1, 2, 3, 6])
def test_peer_planning_stride_matches_world_cycle(planning_stride: int) -> None:
    config = _peer_config(
        planning_stride=planning_stride,
        wmrm_cycle_steps=planning_stride,
    )
    assert config.action_horizon == 6
    assert config.planning_stride == planning_stride
    assert config.wmrm_cycle_steps == planning_stride


@pytest.mark.parametrize("planning_stride", [0, 4, 5, 7])
def test_peer_rejects_unsupported_planning_stride(planning_stride: int) -> None:
    with pytest.raises(ValueError, match="planning_stride"):
        _peer_config(
            planning_stride=planning_stride,
            wmrm_cycle_steps=max(planning_stride, 1),
        )


def test_peer_allows_world_horizon_longer_than_execution_prefix() -> None:
    config = _peer_config(
        action_horizon=15, planning_stride=2, wmrm_cycle_steps=15
    )
    assert config.planning_stride == 2
    assert config.wmrm_cycle_steps == config.action_horizon == 15


def test_peer_supports_true_h15_p15_replanning() -> None:
    config = _peer_config(
        action_horizon=15,
        planning_stride=15,
        deployment_execution_horizon=15,
        wmrm_cycle_steps=15,
    )
    assert config.planning_stride == 15
    assert config.deployment_execution_horizon == 15
    assert config.wmrm_cycle_steps == config.action_horizon == 15


def test_peer_rejects_world_horizon_matching_neither_prefix_nor_chunk() -> None:
    with pytest.raises(ValueError, match="execution prefix or full action horizon"):
        _peer_config(planning_stride=2, wmrm_cycle_steps=5)


def test_peer_rejects_single_va_layer() -> None:
    with pytest.raises(ValueError, match="num_layers >= 2"):
        _peer_config(num_layers=1)


def test_peer_world_is_one_stage_shorter_and_last_va_consumes_final_map() -> None:
    model = VACompoundPolicy(_peer_config(num_layers=3)).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    proposes: list[int] = []
    messages: list[torch.Tensor] = []
    last_state: list[torch.Tensor | None] = []
    original_propose = model.wmrm.propose
    original_last = model.layers[-1].forward

    def record_world(*args, **kwargs):
        transition = original_propose(*args, **kwargs)
        proposes.append(int(kwargs["stage_index"]))
        messages.append(transition.world_message.detach().clone())
        return transition

    def record_last(*args, **kwargs):
        last_state.append(kwargs.get("state"))
        return original_last(*args, **kwargs)

    with (
        mock.patch.object(model.wmrm, "propose", side_effect=record_world),
        mock.patch.object(model.layers[-1], "forward", side_effect=record_last),
        torch.no_grad(),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )

    assert model.config.wmrm_stage_count() == 2
    assert model.wmrm.max_stages == 2
    assert len(model.last_wmrm_auxes) == 2
    assert proposes == [0, 1]
    assert last_state[0] is not None
    torch.testing.assert_close(last_state[0], messages[-1], rtol=0.0, atol=0.0)


def test_full_language_tokens_and_mask_reach_every_world_stage_read_only() -> None:
    model = VACompoundPolicy(
        _peer_config(num_layers=3, wmrm_full_language_tokens=True)
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    cache = model.build_language_cache(language, mask)
    before_tokens = cache.tokens.clone()
    before_mask = cache.attention_mask.clone()
    calls: list[dict] = []
    original = model.wmrm.propose

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    with mock.patch.object(model.wmrm, "propose", side_effect=record), torch.no_grad():
        _, memory = model.encode_condition(
            vision,
            proprio,
            previous,
            language_cache=cache,
            return_visual_memory=True,
        )

    assert len(calls) == model.config.wmrm_stage_count()
    for call in calls:
        assert call["language_keys"] is None
        assert call["language_tokens"] is cache.tokens
        assert call["language_mask"] is cache.attention_mask
        assert call["language_tokens"].shape == language.shape
        assert call["language_mask"].shape == mask.shape
    torch.testing.assert_close(cache.tokens, before_tokens, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cache.attention_mask, before_mask, rtol=0.0, atol=0.0)
    assert set(vars(memory.world_state)) == {"belief", "innovation", "world_map"}


def test_full_language_padding_is_invisible_to_va_and_world() -> None:
    model = VACompoundPolicy(
        _peer_config(wmrm_full_language_tokens=True)
    ).eval()
    vision, proprio, previous, language, _ = _policy_inputs(model.config)
    valid = language[:, :3]
    short_mask = torch.ones(valid.shape[:2], dtype=torch.bool)
    padded = torch.cat((valid, torch.randn(2, 4, model.config.language_dim) * 100.0), dim=1)
    padded_mask = torch.cat(
        (short_mask, torch.zeros(2, 4, dtype=torch.bool)), dim=1
    )

    with torch.no_grad():
        short_condition = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=valid,
            language_mask=short_mask,
        )
        short_summaries = [aux.task_summary.clone() for aux in model.last_wmrm_auxes]
        padded_condition = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=padded,
            language_mask=padded_mask,
        )
        padded_summaries = [aux.task_summary.clone() for aux in model.last_wmrm_auxes]

    torch.testing.assert_close(short_condition, padded_condition)
    assert len(short_summaries) == len(padded_summaries)
    for short, long in zip(short_summaries, padded_summaries, strict=True):
        torch.testing.assert_close(short, long)


def test_slot_free_policy_rejects_fixed_metric_inputs() -> None:
    model = VACompoundPolicy(
        _peer_config(
            wmrm_full_language_tokens=True,
            slot_free_policy=True,
        )
    ).eval()
    assert model.geometry_projection is None
    vision, proprio, previous, language, mask = _policy_inputs(model.config)

    with pytest.raises(ValueError, match="slot_free_policy"):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            metric_g=torch.zeros(2, 8),
        )
    with pytest.raises(ValueError, match="slot_free_policy"):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            metric_tokens=torch.zeros(2, 2, model.config.hidden_dim),
        )


def test_causal_h6_readout_sends_only_executed_prefix_to_world() -> None:
    model = VACompoundPolicy(
        _peer_config(planning_stride=2, wmrm_cycle_steps=2)
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    full_readouts: list[torch.Tensor] = []
    world_actions: list[torch.Tensor] = []
    original_readout = model.world_action_readout.forward
    original_propose = model.wmrm.propose

    def record_readout(action):
        readout = original_readout(action)
        full_readouts.append(readout.detach().clone())
        return readout

    def record_world(*args, **kwargs):
        world_actions.append(kwargs["env_action"].detach().clone())
        return original_propose(*args, **kwargs)

    with (
        mock.patch.object(model.world_action_readout, "forward", side_effect=record_readout),
        mock.patch.object(model.wmrm, "propose", side_effect=record_world),
        torch.no_grad(),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )

    assert len(full_readouts) == len(world_actions) == model.config.wmrm_stage_count()
    assert all(readout.shape == (2, 6, 4) for readout in full_readouts)
    for readout, world_action in zip(full_readouts, world_actions, strict=True):
        assert world_action.shape == (2, 2, 4)
        torch.testing.assert_close(world_action, readout[:, :2], rtol=0.0, atol=0.0)


def test_h15_world_uses_the_full_candidate_chunk() -> None:
    model = VACompoundPolicy(
        _peer_config(
            num_layers=3,
            action_horizon=15,
            planning_stride=2,
            wmrm_cycle_steps=15,
        )
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    world_actions: list[torch.Tensor] = []
    original = model.wmrm.propose

    def record(*args, **kwargs):
        world_actions.append(kwargs["env_action"].detach().clone())
        return original(*args, **kwargs)

    with mock.patch.object(model.wmrm, "propose", side_effect=record), torch.no_grad():
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )

    assert len(world_actions) == 2
    assert all(action.shape == (2, 15, 4) for action in world_actions)


def test_va_and_wam_read_the_same_pre_stage_snapshot() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    va_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    va_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    wam_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer in model.layers:
        original = layer.forward

        def record_layer(visual, action, *args, _original=original, **kwargs):
            va_inputs.append((visual.detach().clone(), action.detach().clone()))
            result = _original(visual, action, *args, **kwargs)
            va_outputs.append((result[0].detach().clone(), result[1].detach().clone()))
            return result

        layer.forward = record_layer

    original_propose = model.wmrm.propose

    def record_world(action, visual, prop, **kwargs):
        wam_inputs.append((visual.detach().clone(), action.detach().clone()))
        return original_propose(action, visual, prop, **kwargs)

    with mock.patch.object(model.wmrm, "propose", side_effect=record_world):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )

    assert len(va_inputs) == model.config.num_layers
    assert len(wam_inputs) == model.config.wmrm_stage_count()
    for va_in, wam_in in zip(va_inputs[: len(wam_inputs)], wam_inputs, strict=True):
        torch.testing.assert_close(va_in[0], wam_in[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(va_in[1], wam_in[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(va_inputs[1][0], va_outputs[0][0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(va_inputs[1][1], va_outputs[0][1], rtol=0.0, atol=0.0)


def test_world_message_is_next_va_layers_attention_kv() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    transitions: list[WAMProposal] = []
    layer_states: list[torch.Tensor | None] = []
    original_propose = model.wmrm.propose
    original_second = model.layers[1].forward

    def record_world(*args, **kwargs):
        transition = original_propose(*args, **kwargs)
        transitions.append(transition)
        return transition

    def record_second(*args, **kwargs):
        layer_states.append(kwargs.get("state"))
        return original_second(*args, **kwargs)

    with (
        mock.patch.object(model.wmrm, "propose", side_effect=record_world),
        mock.patch.object(model.layers[1], "forward", side_effect=record_second),
        torch.no_grad(),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )

    assert len(transitions) == model.config.wmrm_stage_count() and len(layer_states) == 1
    torch.testing.assert_close(
        layer_states[0], transitions[0].world_message, rtol=0.0, atol=0.0
    )


def test_flow_loss_trains_world_reader_but_not_visual_predictor() -> None:
    model = VACompoundPolicy(
        _peer_config(
            num_layers=3,
            wmrm_predictor="st_blocks",
            wmrm_predictor_depth=1,
            wmrm_predictor_width=16,
            wmrm_predictor_heads=4,
        )
    ).train()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    condition = model.encode_condition(
        vision,
        proprio,
        previous,
        language_hidden=language,
        language_mask=mask,
    )
    noisy_actions = torch.randn(2, model.config.action_horizon, model.config.action_dim)
    flow_time = torch.rand(2)
    velocity = model.flow_velocity(condition, noisy_actions, flow_time)
    torch.nn.functional.mse_loss(velocity, torch.ones_like(velocity)).backward()

    for projection in (model.layers[1].k_s, model.layers[1].u_s):
        grad = projection.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0

    publisher_grad = model.wmrm.dino_to_hid.weight.grad
    assert publisher_grad is not None
    assert torch.isfinite(publisher_grad).all()
    assert float(publisher_grad.abs().sum()) > 0.0
    assert model.wmrm.st_predictor.out_proj.weight.grad is None
    assert model.world_action_readout.proj.weight.grad is None


def test_h15_tail_flow_cannot_backpropagate_into_prefix_or_va_condition() -> None:
    model = VACompoundPolicy(
        _peer_config(
            action_horizon=15,
            planning_stride=2,
            deployment_execution_horizon=15,
            wmrm_cycle_steps=15,
            wmrm_predictor="st_blocks",
            wmrm_predictor_depth=1,
            wmrm_predictor_width=16,
            wmrm_predictor_heads=4,
        )
    ).train()
    condition = torch.randn(2, 15, model.config.hidden_dim, requires_grad=True)
    noisy = torch.randn(2, 15, model.config.action_dim)
    velocity = model.flow_velocity(condition, noisy, torch.rand(2))
    velocity[:, 6:].square().mean().backward()

    assert condition.grad is not None
    assert int(torch.count_nonzero(condition.grad)) == 0
    prefix_grads = [parameter.grad for parameter in model.flow_head.parameters()]
    assert all(grad is None or int(torch.count_nonzero(grad)) == 0 for grad in prefix_grads)
    tail_grads = [parameter.grad for parameter in model.tail_flow_head.parameters()]
    assert any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in tail_grads)


def test_h15_frozen_tail_teacher_can_train_capacity_condition() -> None:
    model = VACompoundPolicy(
        _peer_config(
            action_horizon=15,
            planning_stride=2,
            deployment_execution_horizon=15,
            wmrm_cycle_steps=15,
            wmrm_predictor="st_blocks",
            wmrm_predictor_depth=1,
            wmrm_predictor_width=16,
            wmrm_predictor_heads=4,
            tail_flow_condition_grad=True,
        )
    ).train()
    model.flow_head.requires_grad_(False)
    model.tail_flow_head.requires_grad_(False)
    condition = torch.randn(2, 15, model.config.hidden_dim, requires_grad=True)
    noisy = torch.randn(2, 15, model.config.action_dim)
    velocity = model.flow_velocity(condition, noisy, torch.rand(2))
    velocity[:, 6:].square().mean().backward()

    assert condition.grad is not None
    assert float(condition.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.flow_head.parameters())
    assert all(parameter.grad is None for parameter in model.tail_flow_head.parameters())


def test_h15_prefix_queries_and_visual_stream_cannot_read_tail_tokens() -> None:
    model = VACompoundPolicy(
        _peer_config(
            action_horizon=15,
            planning_stride=2,
            deployment_execution_horizon=15,
            wmrm_cycle_steps=15,
        )
    )
    allowed = model.layers[0]._role_mask(
        n_visual=3,
        n_memory=1,
        n_action=15,
        n_language=2,
        n_task=0,
        n_state=1,
        device=torch.device("cpu"),
    )
    action_key_start = 4
    tail_keys = slice(action_key_start + 6, action_key_start + 15)
    assert not bool(allowed[: 3 + 6, tail_keys].any())
    assert bool(allowed[3 + 6 : 3 + 15, tail_keys].all())


def test_h15_prefix_condition_is_invariant_to_tail_query_values() -> None:
    model = VACompoundPolicy(
        _peer_config(
            action_horizon=15,
            planning_stride=2,
            deployment_execution_horizon=15,
            wmrm_cycle_steps=15,
        )
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    with torch.no_grad():
        first = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            skip_wmrm=True,
        )
        model.action_queries[6:].add_(100.0 * torch.randn_like(model.action_queries[6:]))
        second = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            skip_wmrm=True,
        )
    torch.testing.assert_close(first[:, :6], second[:, :6], rtol=0.0, atol=1e-6)


def test_h15_prefix_tail_migration_clones_existing_flow_into_tail() -> None:
    from train import migrate_peer_h15_prefix_tail_flow_state
    from va_compound.world_contract import (
        PEER_GRADIENT_BOUNDARY_CONTRACT,
        PEER_WORLD_TOPOLOGY_CONTRACT,
    )

    old = VACompoundPolicy(
        _peer_config(
            action_horizon=15,
            planning_stride=2,
            deployment_execution_horizon=2,
            wmrm_cycle_steps=15,
        )
    )
    new = VACompoundPolicy(
        _peer_config(
            action_horizon=15,
            planning_stride=2,
            deployment_execution_horizon=15,
            wmrm_cycle_steps=15,
        )
    )
    checkpoint = {
        "config": {
            "action_horizon": 15,
            "va_world_mode": "peer_sync_h6",
            "wmrm_cycle_steps": 15,
        },
        "training_contract": {
            "peer_world_topology": PEER_WORLD_TOPOLOGY_CONTRACT,
            "peer_gradient_boundary": PEER_GRADIENT_BOUNDARY_CONTRACT,
        },
        "model": old.state_dict(),
    }
    migrated = migrate_peer_h15_prefix_tail_flow_state(new, checkpoint)
    new.load_state_dict(migrated, strict=True)
    for name, value in new.tail_flow_head.state_dict().items():
        torch.testing.assert_close(value, new.flow_head.state_dict()[name])


def test_world_side_loss_reaches_wam_and_va_publishers_but_not_future_target() -> None:
    model = VACompoundPolicy(
        _peer_config(
            num_layers=3,
            wmrm_predictor="st_blocks",
            wmrm_predictor_depth=1,
            wmrm_predictor_width=16,
            wmrm_predictor_heads=4,
        )
    ).train()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    model.encode_condition(
        vision,
        proprio,
        previous,
        language_hidden=language,
        language_mask=mask,
    )
    final = model.last_wmrm_auxes[-1]
    future_target = torch.randn_like(final.z_tokens, requires_grad=True)
    # A real future-map objective must cross WAM's belief/task conditioning and
    # causal action readout; synthetic losses on intermediate tensors can hide
    # an accidental detach inside the predictor.
    world_loss = wmrm_world_loss(final.z_tokens, future_target)
    world_loss.backward()

    for parameter in (
        model.wmrm.st_predictor.out_proj.weight,
        model.world_action_readout.proj.weight,
        model.vision_projection.weight,
        model.action_queries,
        model.layers[0].out_v.weight,
        model.layers[0].out_a.weight,
    ):
        grad = parameter.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0
    assert future_target.grad is None


def test_old_world8_migration_rebuilds_dynamics_but_keeps_policy_interface() -> None:
    from train import migrate_peer_world8_to_world7_state
    from va_compound.world_contract import (
        PEER_LEGACY_GRADIENT_BOUNDARY_CONTRACT,
        PEER_LEGACY_TOPOLOGY_CONTRACT,
    )

    model = VACompoundPolicy(
        _peer_config(
            num_layers=8,
            action_horizon=15,
            planning_stride=2,
            wmrm_cycle_steps=15,
            wmrm_predictor="st_blocks",
            wmrm_predictor_depth=1,
            wmrm_predictor_width=16,
            wmrm_predictor_heads=4,
        )
    )
    initial_out = model.wmrm.st_predictor.out_proj.weight.detach().clone()
    initial_stage = model.wmrm.stage_embed.weight.detach().clone()
    initial_tail_queries = model.action_queries[6:].detach().clone()
    state = dict(model.state_dict())
    old_queries = state["action_queries"][:6].clone()
    state["action_queries"] = old_queries
    hidden = state["wmrm.stage_embed.weight"].shape[1]
    state["wmrm.stage_embed.weight"] = torch.arange(
        8 * hidden, dtype=torch.float32
    ).reshape(8, hidden)
    state["wmrm.st_predictor.out_proj.weight"] = torch.full_like(
        state["wmrm.st_predictor.out_proj.weight"], 9.0
    )
    state["wmrm.dino_to_hid.weight"] = torch.full_like(
        state["wmrm.dino_to_hid.weight"], 3.0
    )
    state["world_action_readout.proj.weight"] = torch.full_like(
        state["world_action_readout.proj.weight"], 4.0
    )
    checkpoint = {
        "config": {"num_layers": 8, "action_horizon": 6},
        "training_contract": {
            "peer_world_topology": PEER_LEGACY_TOPOLOGY_CONTRACT,
            "peer_gradient_boundary": PEER_LEGACY_GRADIENT_BOUNDARY_CONTRACT,
        },
        "model": state,
    }

    migrated, reset = migrate_peer_world8_to_world7_state(model, checkpoint)
    assert "wmrm.stage_embed.weight" not in migrated
    assert "wmrm.st_predictor.out_proj.weight" not in migrated
    assert int(torch.count_nonzero(migrated["wmrm.dino_to_hid.weight"] - 3.0)) == 0
    missing, unexpected = model.load_state_dict(migrated, strict=False)
    assert set(missing) == reset
    assert not unexpected
    torch.testing.assert_close(model.wmrm.stage_embed.weight, initial_stage)
    torch.testing.assert_close(model.wmrm.st_predictor.out_proj.weight, initial_out)
    assert int(torch.count_nonzero(model.wmrm.dino_to_hid.weight - 3.0)) == 0
    assert int(torch.count_nonzero(model.world_action_readout.proj.weight - 4.0)) == 0
    torch.testing.assert_close(model.action_queries[:6], old_queries)
    torch.testing.assert_close(model.action_queries[6:], initial_tail_queries)


def test_va8_to_va16_migration_preserves_four_decision_policy_state() -> None:
    from train import migrate_peer_va8_to_va16_state

    common = dict(
        action_horizon=15,
        planning_stride=15,
        deployment_execution_horizon=15,
        wmrm_cycle_steps=15,
        wmrm_predictor="st_blocks",
        wmrm_predictor_depth=6,
        wmrm_predictor_width=12,
        wmrm_predictor_heads=3,
    )
    torch.manual_seed(41)
    source = VACompoundPolicy(_peer_config(num_layers=8, **common)).eval()
    checkpoint = {
        "config": dict(source.config.__dict__),
        "model": source.state_dict(),
    }
    torch.manual_seed(42)
    expanded_common = {
        **common,
        "wmrm_predictor_depth": 7,
        "wmrm_predictor_copies": 11,
    }
    expanded = VACompoundPolicy(
        _peer_config(
            num_layers=16,
            wmrm_stage_gate_start=7,
            **expanded_common,
        )
    ).eval()
    expanded.load_state_dict(
        migrate_peer_va8_to_va16_state(expanded, checkpoint), strict=True
    )

    for index in range(8, 16):
        for name in ("out_v", "out_a", "out_t"):
            assert int(torch.count_nonzero(getattr(expanded.layers[index], name).weight)) == 0
        for name in ("ffn_v", "ffn_a", "ffn_t"):
            assert int(torch.count_nonzero(getattr(expanded.layers[index], name)[-1].weight)) == 0
    assert int(torch.count_nonzero(expanded.wmrm_stage_scale)) == 0
    assert int(torch.count_nonzero(expanded.wmrm_belief_message_scale)) == 0
    assert len(expanded.wmrm.st_predictor_extra) == 10
    for predictor in (expanded.wmrm.st_predictor, *expanded.wmrm.st_predictor_extra):
        torch.testing.assert_close(
            predictor.in_proj.weight,
            source.wmrm.st_predictor.in_proj.weight,
        )
        for name in ("sa_o", "ca_o"):
            assert int(torch.count_nonzero(getattr(predictor.blocks[6], name).weight)) == 0
        assert int(torch.count_nonzero(predictor.blocks[6].ff[2].weight)) == 0

    source_memory = None
    expanded_memory = None
    for step in range(4):
        torch.manual_seed(100 + step)
        vision, proprio, previous, language, mask = _policy_inputs(source.config)
        with torch.no_grad():
            source_action, source_memory = source.encode_condition(
                vision,
                proprio,
                previous,
                language_hidden=language,
                language_mask=mask,
                visual_memory=source_memory,
                return_visual_memory=True,
            )
            expanded_action, expanded_memory = expanded.encode_condition(
                vision,
                proprio,
                previous,
                language_hidden=language,
                language_mask=mask,
                visual_memory=expanded_memory,
                return_visual_memory=True,
            )
        torch.testing.assert_close(expanded_action, source_action, rtol=0.0, atol=1e-6)
        for expanded_layer, source_layer in zip(
            expanded_memory.layers[:8], source_memory.layers, strict=True
        ):
            torch.testing.assert_close(expanded_layer, source_layer, rtol=0.0, atol=1e-6)
        for field in ("belief", "innovation", "world_map"):
            torch.testing.assert_close(
                getattr(expanded_memory.world_state, field),
                getattr(source_memory.world_state, field),
                rtol=0.0,
                atol=1e-6,
            )

    belief = torch.randn(2, expanded.wmrm.n_belief, expanded.config.hidden_dim, requires_grad=True)
    world_map = torch.randn(
        2,
        expanded.wmrm.dino_dim,
        expanded.wmrm.world_grid,
        expanded.wmrm.world_grid,
        requires_grad=True,
    )
    state = WAMState(belief=belief, world_map=world_map)
    with torch.no_grad():
        map_only = expanded.wmrm.encode_world_tokens(world_map)
        expanded.wmrm_belief_message_scale[0] = 1.0
    bridged = expanded._peer_world_message(state, 0)
    count = min(map_only.shape[1], belief.shape[1])
    torch.testing.assert_close(
        bridged[:, :count], map_only[:, :count] + belief[:, :count]
    )
    bridged.sum().backward()
    assert belief.grad is not None and float(belief.grad.abs().sum()) > 0.0
    assert world_map.grad is None


def test_capacity_phase2_policy_gradient_reaches_gate_not_world_predictor() -> None:
    config = _peer_config(
        num_layers=4,
        wmrm_stage_gate_start=1,
        capacity_stage_gate_policy_grad=True,
        wmrm_predictor="st_blocks",
        wmrm_predictor_depth=1,
        wmrm_predictor_width=16,
        wmrm_predictor_heads=4,
        wmrm_predictor_copies=2,
    )
    torch.manual_seed(42)
    model = VACompoundPolicy(config)
    with torch.no_grad():
        model.wmrm_stage_scale.fill_(0.01)
    proposal_maps = []
    original_propose = model.wmrm.propose

    def capture_proposal(*args, **kwargs):
        proposal = original_propose(*args, **kwargs)
        proposal.next_world_state.world_map.retain_grad()
        proposal_maps.append(proposal.next_world_state.world_map)
        return proposal

    vision, proprio, previous, language, mask = _policy_inputs(config)
    with mock.patch.object(model.wmrm, "propose", side_effect=capture_proposal):
        condition = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
    condition.square().sum().backward()

    assert model.wmrm_stage_scale.grad is not None
    assert float(model.wmrm_stage_scale.grad.abs().sum()) > 0.0
    assert all(
        world_map.grad is None or int(torch.count_nonzero(world_map.grad)) == 0
        for world_map in proposal_maps
    )
    assert all(
        parameter.grad is None or int(torch.count_nonzero(parameter.grad)) == 0
        for name, parameter in model.named_parameters()
        if name.startswith("wmrm.st_predictor")
    )


def test_capacity_gate_policy_gradient_is_off_by_default() -> None:
    config = _peer_config(num_layers=4, wmrm_stage_gate_start=1)
    assert config.capacity_stage_gate_policy_grad is False
    torch.manual_seed(42)
    model = VACompoundPolicy(config)
    with torch.no_grad():
        model.wmrm_stage_scale.fill_(0.01)
    vision, proprio, previous, language, mask = _policy_inputs(config)
    condition = model.encode_condition(
        vision,
        proprio,
        previous,
        language_hidden=language,
        language_mask=mask,
    )
    condition.square().sum().backward()
    assert model.wmrm_stage_scale.grad is None or int(
        torch.count_nonzero(model.wmrm_stage_scale.grad)
    ) == 0


def test_terminal_world_state_is_read_by_next_decisions_first_va_layer() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    with torch.no_grad():
        _, memory = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            return_visual_memory=True,
        )
    assert memory.world_state is not None and memory.world_state.world_map is not None
    expected = model.wmrm.encode_world_tokens(memory.world_state.world_map)
    seen: list[torch.Tensor | None] = []
    original = model.layers[0].forward

    def record_first(*args, **kwargs):
        seen.append(kwargs.get("state"))
        return original(*args, **kwargs)

    with mock.patch.object(model.layers[0], "forward", side_effect=record_first), torch.no_grad():
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            visual_memory=memory,
        )
    torch.testing.assert_close(seen[0], expected, rtol=0.0, atol=0.0)


def test_long_horizon_map_is_decision_local_but_belief_persists() -> None:
    model = VACompoundPolicy(
        _peer_config(
            num_layers=3,
            action_horizon=15,
            planning_stride=2,
            wmrm_cycle_steps=15,
            wmrm_predictor="st_blocks",
            wmrm_predictor_depth=1,
            wmrm_predictor_width=16,
            wmrm_predictor_heads=4,
        )
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    with torch.no_grad():
        _, memory = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            return_visual_memory=True,
        )
    assert memory.world_state is not None
    assert memory.world_state.world_map is not None
    assert memory.world_state.belief is not None
    seen: list[torch.Tensor | None] = []
    original = model.layers[0].forward

    def record_first(*args, **kwargs):
        seen.append(kwargs.get("state"))
        return original(*args, **kwargs)

    with mock.patch.object(model.layers[0], "forward", side_effect=record_first), torch.no_grad():
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            visual_memory=memory,
        )
    torch.testing.assert_close(
        seen[0], memory.world_state.belief, rtol=0.0, atol=0.0
    )


def test_explicit_env_action_overrides_readout_and_keeps_gradient() -> None:
    model = VACompoundPolicy(
        _peer_config(planning_stride=2, wmrm_cycle_steps=2)
    ).train()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    override = torch.randn(2, 6, 4, requires_grad=True)
    with mock.patch.object(
        model.world_action_readout,
        "forward",
        side_effect=AssertionError("readout must not run with explicit env_action"),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            env_action=override,
        )
    torch.testing.assert_close(
        model.last_wmrm.env_action, override[:, :2], rtol=0.0, atol=0.0
    )
    model.last_wmrm.z_spans.square().mean().backward()
    assert override.grad is not None and float(override.grad.abs().sum()) > 0.0
    assert int(torch.count_nonzero(override.grad[:, 2:])) == 0


def test_peer_logged_env_action_requires_complete_h6() -> None:
    model = VACompoundPolicy(
        _peer_config(planning_stride=2, wmrm_cycle_steps=2)
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    with pytest.raises(ValueError, match="complete H6"):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            env_action=torch.randn(2, 2, 4),
        )


def test_nonfinite_world_message_is_rejected_before_next_layer() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    original = model.wmrm.propose
    calls = 0

    def inject_nonfinite(*args, **kwargs):
        nonlocal calls
        calls += 1
        transition = original(*args, **kwargs)
        message = transition.world_message.clone()
        message[0, 0, 0] = float("nan")
        return replace(transition, world_message=message)

    with (
        mock.patch.object(model.wmrm, "propose", side_effect=inject_nonfinite),
        pytest.raises(FloatingPointError, match="World stage 0 transition.*world_message"),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
    assert calls == 1


def test_disabled_runtime_checks_skip_world_stage_finite_validation() -> None:
    model = VACompoundPolicy(
        _peer_config(runtime_integrity_checks=False)
    ).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    original = model.wmrm.propose

    def inject_nonfinite(*args, **kwargs):
        transition = original(*args, **kwargs)
        message = transition.world_message.clone()
        message[0, 0, 0] = float("nan")
        return replace(transition, world_message=message)

    with mock.patch.object(
        model.wmrm, "propose", side_effect=inject_nonfinite
    ), mock.patch.object(
        WAMProposal,
        "validate_finite",
        side_effect=AssertionError("stage finite check entered"),
    ), torch.no_grad():
        condition = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
    # The injected message can be the final published World state and therefore
    # need not feed back into this call's action condition.  Reaching this point
    # while ``validate_finite`` is patched to raise is the contract under test.
    assert condition.shape == (
        vision.shape[0],
        model.config.action_horizon,
        model.config.hidden_dim,
    )


def test_runtime_checks_toggle_preserves_finite_output_and_gradients() -> None:
    checked = VACompoundPolicy(
        _peer_config(wmrm_full_language_tokens=True)
    ).eval()
    unchecked = VACompoundPolicy(
        _peer_config(
            wmrm_full_language_tokens=True,
            runtime_integrity_checks=False,
        )
    ).eval()
    unchecked.load_state_dict(checked.state_dict(), strict=True)
    inputs = _policy_inputs(checked.config)

    def run(model):
        model.zero_grad(set_to_none=True)
        condition = model.encode_condition(
            inputs[0],
            inputs[1],
            inputs[2],
            language_hidden=inputs[3],
            language_mask=inputs[4],
        )
        condition.square().mean().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return condition.detach(), gradients

    checked_output, checked_gradients = run(checked)
    unchecked_output, unchecked_gradients = run(unchecked)
    torch.testing.assert_close(
        checked_output, unchecked_output, rtol=0.0, atol=0.0
    )
    assert checked_gradients.keys() == unchecked_gradients.keys()
    for name in checked_gradients:
        torch.testing.assert_close(
            checked_gradients[name],
            unchecked_gradients[name],
            rtol=0.0,
            atol=0.0,
        )


def test_visual_memory_world_state_detach_to_index_and_episode_reset() -> None:
    layers = tuple(torch.randn(3, 5, 16, requires_grad=True) for _ in range(2))
    state = WAMState(
        belief=torch.randn(3, 2, 16, requires_grad=True),
        innovation=torch.randn(3, 2, 16, requires_grad=True),
        world_map=torch.randn(3, 8, 2, 2, requires_grad=True),
    )
    memory = VisualMemory(layers=layers, world_state=state)
    assert not memory.detach().world_state.belief.requires_grad
    assert memory.to(dtype=torch.float64).world_state.belief.dtype == torch.float32
    selected = memory.index_select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.world_state.belief, state.belief[[2, 0]])
    assert VisualMemory(layers=selected.layers).world_state is None
