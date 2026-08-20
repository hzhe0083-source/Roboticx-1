"""Focused contracts for the peer-synchronous WAM core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from unittest import mock

from va_compound.model import VACompoundConfig, VACompoundPolicy, VisualMemory
from va_compound.wmrm import (
    ExecutableActionReadout,
    WAM4VA,
    WAMProposal,
    WAMState,
)


def _block() -> WAM4VA:
    torch.manual_seed(7)
    return WAM4VA(
        16,
        world_dim=8,
        rank=4,
        proprio_dim=3,
        mixer_dropout=0.0,
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


def _inputs():
    torch.manual_seed(11)
    return (
        torch.randn(3, 6, 16),
        torch.randn(3, 5, 16),
        torch.randn(3, 3),
        torch.randn(3, 8, 8),
        torch.randn(3, 6, 4),
    )


def test_state_is_frozen_and_tensor_operations_cover_all_fields() -> None:
    belief = torch.randn(3, 2, 16, requires_grad=True)
    innovation = torch.randn(3, 4, 16, requires_grad=True)
    world_map = torch.randn(3, 8, 2, 2, requires_grad=True)
    state = WAMState(belief, innovation, world_map)

    with pytest.raises(FrozenInstanceError):
        state.belief = None

    detached = state.detach()
    assert all(
        tensor is not None and not tensor.requires_grad
        for tensor in (detached.belief, detached.innovation, detached.world_map)
    )
    converted = state.to(dtype=torch.float64)
    assert converted.belief.dtype == torch.float32
    assert converted.innovation.dtype == torch.float32
    assert converted.world_map.dtype == torch.float32
    selected = state.index_select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.belief, belief[[2, 0]])
    torch.testing.assert_close(selected.innovation, innovation[[2, 0]])
    torch.testing.assert_close(selected.world_map, world_map[[2, 0]])


def test_propose_matches_legacy_forward_bitwise_and_does_not_mutate_snapshot() -> None:
    block = _block()
    action, vision, proprio, dino, env_action = _inputs()
    state = WAMState(
        belief=torch.randn(3, 2, 16),
        innovation=torch.randn(3, 2, 16),
        world_map=torch.randn(3, 8, 2, 2),
    )
    before = tuple(
        tensor.clone()
        for tensor in (state.belief, state.innovation, state.world_map)
    )

    with torch.no_grad():
        legacy = block(
            action,
            vision,
            proprio,
            belief=state.belief,
            prev_innovation=state.innovation,
            dino_tokens=dino,
            env_action=env_action,
            previous_map=state.world_map,
        )
        proposal = block.propose(
            action,
            vision,
            proprio,
            state=state,
            dino_tokens=dino,
            env_action=env_action,
        )

    assert isinstance(proposal, WAMProposal)
    torch.testing.assert_close(
        action + proposal.action_delta, legacy[0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        proposal.next_world_state.belief, legacy[2], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        proposal.next_world_state.innovation, legacy[3], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        proposal.next_world_state.world_map,
        legacy[1].z_tokens,
        rtol=0.0,
        atol=0.0,
    )
    expected_vision_delta = block.mix_world_into_vision(vision, legacy[1]) - vision
    torch.testing.assert_close(
        proposal.vision_delta, expected_vision_delta, rtol=0.0, atol=0.0
    )
    for current, original in zip(
        (state.belief, state.innovation, state.world_map), before, strict=True
    ):
        torch.testing.assert_close(current, original, rtol=0.0, atol=0.0)


def _run_peer_state_autocast_regression(device: torch.device, *, stages: int) -> None:
    config = _peer_config(num_layers=stages)
    model = VACompoundPolicy(config).to(device).eval()
    inputs = tuple(tensor.to(device) for tensor in _policy_inputs(config))
    vision, proprio, previous, language, mask = inputs
    stage_states: list[WAMState] = []
    original_propose = model.wmrm.propose

    def record_propose(*args, **kwargs):
        proposal = original_propose(*args, **kwargs)
        stage_states.append(proposal.next_world_state)
        return proposal

    memory = None
    with (
        mock.patch.object(model.wmrm, "propose", side_effect=record_propose),
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.bfloat16),
    ):
        for _ in range(2):
            condition, memory = model.encode_condition(
                vision,
                proprio,
                previous,
                language_hidden=language,
                language_mask=mask,
                visual_memory=memory,
                return_visual_memory=True,
            )

    assert len(stage_states) == stages * 2
    for state in stage_states:
        for tensor in (state.belief, state.innovation, state.world_map):
            assert tensor is not None
            assert tensor.device.type == device.type
            assert tensor.dtype == torch.float32
            assert bool(torch.isfinite(tensor).all())
    assert condition.device.type == device.type
    assert memory is not None and memory.world_state is stage_states[-1]


def test_peer_state_is_canonical_fp32_under_cpu_bf16_autocast_tiny() -> None:
    _run_peer_state_autocast_regression(torch.device("cpu"), stages=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_peer_state_is_canonical_fp32_across_eight_cuda_bf16_stages() -> None:
    _run_peer_state_autocast_regression(torch.device("cuda"), stages=8)


def test_project_out_is_explicit_fp32_under_bf16_autocast() -> None:
    block = _block()
    current = torch.randn(3, 2, 16)
    previous = torch.randn(3, 2, 16)
    expected = block._project_out(current, previous)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = block._project_out(current.bfloat16(), previous)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, block._project_out(current.bfloat16().float(), previous))
    assert bool(torch.isfinite(actual).all())
    assert expected.dtype == torch.float32


def test_propose_validates_state_contract_and_rejects_reuse_aux() -> None:
    block = _block()
    action, vision, proprio, dino, env_action = _inputs()
    valid = block.propose(
        action,
        vision,
        proprio,
        dino_tokens=dino,
        env_action=env_action,
    )
    invalid_states = (
        WAMState(belief=torch.randn(2, 2, 16)),
        WAMState(innovation=torch.randn(3, 3, 16)),
        WAMState(world_map=torch.randn(3, 8, 3, 2)),
        WAMState(belief=torch.randn(3, 2, 16, dtype=torch.float64)),
        WAMState(innovation=torch.randn(3, 2, 16, dtype=torch.bfloat16)),
    )
    for state in invalid_states:
        with pytest.raises(ValueError, match="WAMState"):
            block.propose(
                action,
                vision,
                proprio,
                state=state,
                dino_tokens=dino,
                env_action=env_action,
            )
    with pytest.raises(ValueError, match="reuse_aux is unsupported"):
        block.propose(
            action,
            vision,
            proprio,
            dino_tokens=dino,
            env_action=env_action,
            reuse_aux=valid.aux,
        )


def test_proposal_tensor_operations_include_auxiliary_payload() -> None:
    block = _block()
    action, vision, proprio, dino, env_action = _inputs()
    proposal = block.propose(
        action.requires_grad_(),
        vision.requires_grad_(),
        proprio,
        dino_tokens=dino,
        env_action=env_action,
    )

    detached = proposal.detach()
    assert not detached.action_delta.requires_grad
    assert not detached.vision_delta.requires_grad
    assert not detached.aux.z_hat.requires_grad
    converted = proposal.to(dtype=torch.float64)
    assert converted.action_delta.dtype == torch.float64
    assert converted.aux.z_hat.dtype == torch.float64
    selected = proposal.index_select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.action_delta, proposal.action_delta[[2, 0]])
    torch.testing.assert_close(selected.aux.z_hat, proposal.aux.z_hat[[2, 0]])


def test_executable_action_readout_is_deterministic_h6_and_bounded() -> None:
    torch.manual_seed(3)
    readout = ExecutableActionReadout(hidden_dim=16, action_dim=4)
    action = torch.randn(2, 6, 16)
    first = readout(action)
    second = readout(action)

    assert first.shape == (2, 6, 4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool((first >= -1.0).all())
    assert bool((first <= 1.0).all())
    with pytest.raises(ValueError, match="H6"):
        ExecutableActionReadout(hidden_dim=16, action_dim=4, horizon=5)
    with pytest.raises(ValueError, match="action must be"):
        readout(torch.randn(2, 5, 16))


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
        wmrm_handshake=True,
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


def test_readout_is_horizon_local_and_never_calls_flow() -> None:
    readout = ExecutableActionReadout(hidden_dim=16, action_dim=4)
    action = torch.randn(2, 6, 16)
    baseline = readout(action)
    changed = action.clone()
    changed[:, 3] += 10.0
    perturbed = readout(changed)

    torch.testing.assert_close(perturbed[:, :3], baseline[:, :3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(perturbed[:, 4:], baseline[:, 4:], rtol=0.0, atol=0.0)
    assert not torch.equal(perturbed[:, 3], baseline[:, 3])
    assert not any("flow" in name.lower() for name, _ in readout.named_modules())


def test_readout_rejects_nonfinite_input_and_output_at_boundary() -> None:
    readout = ExecutableActionReadout(hidden_dim=16, action_dim=4)
    action = torch.randn(2, 6, 16)
    action[0, 2, 3] = float("nan")
    with pytest.raises(FloatingPointError, match="readout input.*action.*NaN or Inf"):
        readout(action)

    with torch.no_grad():
        readout.proj.bias[1] = float("inf")
    with pytest.raises(FloatingPointError, match="readout output.*readout.*NaN or Inf"):
        readout(torch.zeros(2, 6, 16))


def test_visual_memory_world_state_detach_to_index_and_episode_reset() -> None:
    layers = tuple(torch.randn(3, 5, 16, requires_grad=True) for _ in range(2))
    state = WAMState(
        belief=torch.randn(3, 2, 16, requires_grad=True),
        innovation=torch.randn(3, 2, 16, requires_grad=True),
        world_map=torch.randn(3, 8, 2, 2, requires_grad=True),
    )
    memory = VisualMemory(layers=layers, world_state=state)

    detached = memory.detach()
    assert all(not layer.requires_grad for layer in detached.layers)
    assert detached.world_state is not None
    assert not detached.world_state.belief.requires_grad
    converted = memory.to(dtype=torch.float64)
    assert all(layer.dtype == torch.float64 for layer in converted.layers)
    assert converted.world_state.belief.dtype == torch.float32
    assert converted.world_state.innovation.dtype == torch.float32
    assert converted.world_state.world_map.dtype == torch.float32
    selected = memory.index_select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.layers[0], layers[0][[2, 0]])
    torch.testing.assert_close(selected.world_state.belief, state.belief[[2, 0]])

    # Episode reset is represented by dropping the immutable recurrent object.
    reset = VisualMemory(layers=selected.layers)
    assert reset.world_state is None


def test_peer_world_and_va_use_same_pre_stage_snapshot() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    va_stage: dict[str, torch.Tensor] = {}
    original_layer = model.layers[0].forward
    original_propose = model.wmrm.propose

    def record_layer(action_vision, action, *args, **kwargs):
        va_stage["vision_in"] = action_vision.detach().clone()
        va_stage["action_in"] = action.detach().clone()
        result = original_layer(action_vision, action, *args, **kwargs)
        va_stage["vision_out"] = result[0].detach().clone()
        va_stage["action_out"] = result[1].detach().clone()
        return result

    def record_propose(action, visual, prop, **kwargs):
        snapshots.append(
            (action.detach().clone(), visual.detach().clone(), prop.detach().clone())
        )
        return original_propose(action, visual, prop, **kwargs)

    with (
        mock.patch.object(model.layers[0], "forward", side_effect=record_layer),
        mock.patch.object(model.wmrm, "propose", side_effect=record_propose),
        torch.no_grad(),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            return_visual_memory=True,
        )

    world_action, world_vision, world_proprio = snapshots[0]
    torch.testing.assert_close(world_action, va_stage["action_in"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(world_vision, va_stage["vision_in"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(world_proprio, proprio, rtol=0.0, atol=0.0)
    assert not torch.equal(world_action, va_stage["action_out"])
    assert not torch.equal(world_vision, va_stage["vision_out"])


def test_legacy_default_off_is_bitwise_identical_to_explicit_legacy() -> None:
    default_config = _peer_config(va_world_mode="legacy")
    explicit_config = VACompoundConfig(**{**default_config.__dict__, "va_world_mode": "legacy"})
    torch.manual_seed(19)
    default = VACompoundPolicy(default_config).eval()
    torch.manual_seed(19)
    explicit = VACompoundPolicy(explicit_config).eval()
    assert default.world_action_readout is None
    assert explicit.world_action_readout is None
    explicit.load_state_dict(default.state_dict(), strict=True)
    vision, proprio, previous, language, mask = _policy_inputs(default_config)

    with torch.no_grad():
        cond_a, mem_a = default.encode_condition(
            vision, proprio, previous,
            language_hidden=language, language_mask=mask,
            return_visual_memory=True,
        )
        cond_b, mem_b = explicit.encode_condition(
            vision, proprio, previous,
            language_hidden=language, language_mask=mask,
            return_visual_memory=True,
        )
        noise = torch.randn(2, 6, 4)
        decoded_a = default.sample_actions(cond_a, steps=2, noise=noise)
        decoded_b = explicit.sample_actions(cond_b, steps=2, noise=noise)

    torch.testing.assert_close(cond_a, cond_b, rtol=0.0, atol=0.0)
    for left, right in zip(mem_a.layers, mem_b.layers, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    torch.testing.assert_close(decoded_a, decoded_b, rtol=0.0, atol=0.0)


def test_current_vision_changes_peer_condition_with_stale_memory_and_writes_off() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    with torch.no_grad():
        _, stale = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            return_visual_memory=True,
        )
        assert stale.world_state is not None
        changed = vision.clone()
        changed[:, -1] += 3.0
        first = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            visual_memory=stale,
            wmrm_action_write=False,
            wmrm_vision_write=False,
        )
        second = model.encode_condition(
            changed,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            visual_memory=stale,
            wmrm_action_write=False,
            wmrm_vision_write=False,
        )

    assert not torch.equal(first, second)


def test_peer_explicit_env_action_overrides_readout_and_keeps_gradient() -> None:
    model = VACompoundPolicy(_peer_config()).train()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    override = torch.randn(2, 6, 4, requires_grad=True)

    with (
        mock.patch.object(
            model.world_action_readout,
            "forward",
            side_effect=AssertionError("readout must not run when env_action is explicit"),
        ),
        mock.patch.object(model.wmrm, "propose", wraps=model.wmrm.propose) as propose,
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            env_action=override,
        )

    assert model.last_wmrm is not None
    assert all(call.kwargs["env_action"] is override for call in propose.call_args_list)
    torch.testing.assert_close(model.last_wmrm.env_action, override, rtol=0.0, atol=0.0)
    model.last_wmrm.z_spans.square().mean().backward()
    assert override.grad is not None
    assert torch.isfinite(override.grad).all()
    assert float(override.grad.abs().sum()) > 0.0


def test_peer_regularizers_and_sentinel_use_world_snapshot_action() -> None:
    model = VACompoundPolicy(_peer_config(num_layers=1)).train()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    proposed: list[torch.Tensor] = []
    regularized: list[torch.Tensor] = []
    original_propose = model.wmrm.propose
    original_kl = model.wmrm.pi_kl_from_aux
    original_med = model.wmrm.fm_condition_hinge

    def record_propose(action, *args, **kwargs):
        proposed.append(action)
        return original_propose(action, *args, **kwargs)

    def record_kl(action, aux):
        regularized.append(action)
        return original_kl(action, aux)

    def record_med(action, aux, norm, **kwargs):
        regularized.append(action)
        return original_med(action, aux, norm, **kwargs)

    with (
        mock.patch.object(model.wmrm, "propose", side_effect=record_propose),
        mock.patch.object(model.wmrm, "pi_kl_from_aux", side_effect=record_kl),
        mock.patch.object(model.wmrm, "fm_condition_hinge", side_effect=record_med),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )

    assert len(proposed) == len(model.last_wmrm_pre_actions) == 1
    assert len(regularized) == 2
    torch.testing.assert_close(model.last_wmrm_pre_actions[0], proposed[0], rtol=0.0, atol=0.0)
    for action in regularized:
        torch.testing.assert_close(action, proposed[0], rtol=0.0, atol=0.0)


def test_peer_current_vision_correction_is_applied_once_per_decision() -> None:
    model = VACompoundPolicy(_peer_config(num_layers=3)).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    original = model.peer_current_vision.forward
    calls = 0

    def count(value):
        nonlocal calls
        calls += 1
        return original(value)

    with mock.patch.object(model.peer_current_vision, "forward", side_effect=count):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
    assert calls == 1


def test_peer_rejects_invalid_world_state_before_layer_execution() -> None:
    model = VACompoundPolicy(_peer_config()).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    bad = VisualMemory(
        layers=tuple(torch.zeros(2, 8, 16) for _ in model.layers),
        world_state=WAMState(belief=torch.zeros(3, 2, 16)),
    )
    with (
        mock.patch.object(
            model.layers[0],
            "forward",
            side_effect=AssertionError("invalid state must fail before VA execution"),
        ),
        pytest.raises(ValueError, match="WAMState.belief"),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            visual_memory=bad,
        )


def test_peer_rejects_nonfinite_proposal_before_recurrent_commit() -> None:
    model = VACompoundPolicy(_peer_config(num_layers=2)).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    original_propose = model.wmrm.propose
    calls = 0

    def inject_nonfinite(*args, **kwargs):
        nonlocal calls
        calls += 1
        proposal = original_propose(*args, **kwargs)
        bad_delta = proposal.action_delta.clone()
        bad_delta[0, 0, 0] = float("nan")
        return replace(proposal, action_delta=bad_delta)

    with (
        mock.patch.object(model.wmrm, "propose", side_effect=inject_nonfinite),
        pytest.raises(
            FloatingPointError,
            match="peer stage 0 proposal.*action_delta.*NaN or Inf",
        ),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
    assert calls == 1


def test_peer_rejects_nonfinite_next_state_before_recurrent_commit() -> None:
    model = VACompoundPolicy(_peer_config(num_layers=2)).eval()
    vision, proprio, previous, language, mask = _policy_inputs(model.config)
    original_propose = model.wmrm.propose
    calls = 0

    def inject_nonfinite(*args, **kwargs):
        nonlocal calls
        calls += 1
        proposal = original_propose(*args, **kwargs)
        bad_belief = proposal.next_world_state.belief.clone()
        bad_belief[0, 0, 0] = float("inf")
        bad_state = replace(proposal.next_world_state, belief=bad_belief)
        return replace(proposal, next_world_state=bad_state)

    with (
        mock.patch.object(model.wmrm, "propose", side_effect=inject_nonfinite),
        pytest.raises(
            FloatingPointError,
            match="peer stage 0 proposal.*WAMState.belief.*NaN or Inf",
        ),
    ):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
    assert calls == 1
