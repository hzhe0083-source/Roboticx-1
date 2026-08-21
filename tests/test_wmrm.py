"""World-state prediction and VA attention-memory integration."""

from __future__ import annotations

import pytest
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy
from va_compound.wmrm import (
    WAMState,
    WAM4VA,
    action_dependency_scores,
    matched_no_fixed_point_perm,
    wmrm_world_loss,
)


def _tiny_config(**overrides) -> VACompoundConfig:
    base = dict(
        language_dim=24,
        vision_dim=20,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        action_horizon=5,
        action_dim=6,
        proprio_dim=9,
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def _inputs(config: VACompoundConfig):
    torch.manual_seed(7)
    return (
        torch.randn(2, 11, config.vision_dim),
        torch.randn(2, config.proprio_dim),
        torch.randn(2, config.action_dim),
        torch.randn(2, 7, config.language_dim),
        torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.bool,
        ),
    )


def _tiny_spatial_config() -> VACompoundConfig:
    return _tiny_config(
        vision_dim=8,
        main_vision_dim=8,
        num_layers=2,
        action_horizon=6,
        action_dim=4,
        wmrm=True,
        wmrm_cycle_steps=6,
        wmrm_inject="all",
        wmrm_predictor="st_blocks",
        wmrm_predictor_depth=1,
        wmrm_predictor_width=16,
        wmrm_predictor_heads=4,
        wmrm_map_size=4,
        wmrm_map_channels=8,
        wmrm_world_grid=4,
        main_vision_grid=4,
        main_vision_frames=2,
        main_vision_tokens=32,
    )


def _spatial_inputs(config: VACompoundConfig):
    torch.manual_seed(12)
    batch = 2
    tokens = config.main_vision_frames * config.main_vision_grid**2
    return (
        torch.randn(batch, tokens, config.vision_dim),
        torch.randn(batch, config.proprio_dim),
        torch.randn(batch, config.action_dim),
        torch.randn(batch, 5, config.language_dim),
        torch.ones(batch, 5, dtype=torch.bool),
        torch.rand(batch, config.wmrm_cycle_steps, config.action_dim) * 2.0 - 1.0,
    )


def test_encode_condition_stage_map_detach_gradient_contract() -> None:
    config = _tiny_spatial_config()
    model = VACompoundPolicy(config).eval()
    vision, proprio, previous, language, mask, logged_action = _spatial_inputs(config)

    def run(detach: bool | None):
        model.zero_grad(set_to_none=True)
        # Joint VA↔World gradients make the language cache part of the graph;
        # each independent forward/backward therefore owns a fresh cache graph.
        language_cache = model.build_language_cache(language, mask)
        kwargs = {} if detach is None else {"detach_wmrm_stage_state": detach}
        condition = model.encode_condition(
            vision,
            proprio,
            previous,
            language_cache=language_cache,
            env_action=logged_action,
            **kwargs,
        )
        maps = [aux.z_tokens for aux in model.last_wmrm_auxes]
        assert len(maps) == config.num_layers
        assert all(stage_map is not None for stage_map in maps)
        first_map, final_map = maps[0], maps[-1]
        first_map.retain_grad()
        final_map.square().mean().backward()
        first_grad = None if first_map.grad is None else first_map.grad.detach().clone()
        return (
            condition.detach().clone(),
            torch.stack([stage_map.detach().clone() for stage_map in maps]),
            first_grad,
        )

    default_condition, default_maps, default_grad = run(None)
    explicit_condition, explicit_maps, explicit_grad = run(False)
    detached_condition, detached_maps, detached_grad = run(True)

    torch.testing.assert_close(default_condition, explicit_condition, rtol=0.0, atol=0.0)
    torch.testing.assert_close(default_maps, explicit_maps, rtol=0.0, atol=0.0)
    torch.testing.assert_close(default_condition, detached_condition, rtol=0.0, atol=0.0)
    torch.testing.assert_close(default_maps, detached_maps, rtol=0.0, atol=0.0)
    assert default_grad is not None and float(default_grad.abs().sum()) > 0.0
    torch.testing.assert_close(default_grad, explicit_grad, rtol=0.0, atol=0.0)
    assert detached_grad is None or int(torch.count_nonzero(detached_grad)) == 0


def test_rollout_proposal_stage_detach_is_explicit_and_forward_identical() -> None:
    from train import rollout_policy

    base_config = _tiny_spatial_config()
    detached_config = _tiny_config(
        vision_dim=8,
        main_vision_dim=8,
        num_layers=2,
        action_horizon=6,
        action_dim=4,
        wmrm=True,
        wmrm_cycle_steps=6,
        wmrm_inject="all",
        wmrm_detach_proposal_stage_state=True,
        wmrm_predictor="st_blocks",
        wmrm_predictor_depth=1,
        wmrm_predictor_width=16,
        wmrm_predictor_heads=4,
        wmrm_map_size=4,
        wmrm_map_channels=8,
        wmrm_world_grid=4,
        main_vision_grid=4,
        main_vision_frames=2,
        main_vision_tokens=32,
    )
    legacy = VACompoundPolicy(base_config).eval()
    detached = VACompoundPolicy(detached_config).eval()
    detached.load_state_dict(legacy.state_dict(), strict=True)

    vision, proprio, previous, language, mask, actions = _spatial_inputs(base_config)
    sequence = 2
    batch = {
        "vision_tokens": vision[:, None].expand(-1, sequence, -1, -1).clone(),
        "proprio": proprio[:, None].expand(-1, sequence, -1).clone(),
        "previous_action": previous[:, None].expand(-1, sequence, -1).clone(),
        "actions": actions[:, None].expand(-1, sequence, -1, -1).clone(),
        "language_hidden": language,
        "language_mask": mask,
    }
    noisy = torch.randn(
        vision.shape[0], sequence, base_config.action_horizon, base_config.action_dim
    )
    flow_time = torch.rand(vision.shape[0], sequence)

    calls: dict[str, list[bool]] = {"legacy": [], "detached": []}
    for name, model in (("legacy", legacy), ("detached", detached)):
        original = model.encode_condition

        def record(*args, _name=name, _original=original, **kwargs):
            if not kwargs.get("skip_wmrm", False):
                calls[_name].append(bool(kwargs.get("detach_wmrm_stage_state", False)))
            return _original(*args, **kwargs)

        model.encode_condition = record

    legacy_outputs = rollout_policy(legacy, batch, noisy, flow_time, flow_steps=2)
    detached_outputs = rollout_policy(detached, batch, noisy, flow_time, flow_steps=2)

    assert calls == {"legacy": [False, False], "detached": [True, True]}
    for legacy_output, detached_output in zip(
        legacy_outputs, detached_outputs, strict=True
    ):
        torch.testing.assert_close(
            legacy_output, detached_output, rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(
        torch.stack([aux.z_tokens for aux in legacy.last_wmrm_auxes]),
        torch.stack([aux.z_tokens for aux in detached.last_wmrm_auxes]),
        rtol=0.0,
        atol=0.0,
    )


def test_cli_proposal_stage_detach_defaults_off_and_is_configurable() -> None:
    from train import parse_args

    assert parse_args([]).wmrm_detach_proposal_stage_state is False
    assert (
        parse_args(["--wmrm-detach-proposal-stage-state"])
        .wmrm_detach_proposal_stage_state
        is True
    )


def test_evaluator_world_forward_matches_full_logged_forward_stage_maps() -> None:
    from scripts.eval_wam4va_world_action import _world_forward

    config = _tiny_spatial_config()
    model = VACompoundPolicy(config).eval()
    vision, proprio, previous, language, mask, logged_action = _spatial_inputs(config)
    language_cache = model.build_language_cache(language, mask)
    noisy_actions = torch.randn(vision.shape[0], config.action_horizon, config.action_dim)
    flow_time = torch.rand(vision.shape[0])

    with torch.inference_mode():
        model(
            vision,
            proprio,
            previous,
            noisy_actions,
            flow_time,
            language_cache=language_cache,
            env_action=logged_action,
        )
        full_forward_maps = torch.stack(
            [aux.z_tokens for aux in model.last_wmrm_auxes]
        ).clone()
        evaluator_maps = _world_forward(
            model,
            vision,
            proprio,
            previous,
            language_cache,
            None,
            None,
            None,
            logged_action,
        )

    assert full_forward_maps.shape == evaluator_maps.shape
    torch.testing.assert_close(
        full_forward_maps, evaluator_maps, rtol=0.0, atol=0.0
    )


def test_peer_readout_is_peer_only_and_legacy_state_keys_are_unchanged() -> None:
    legacy = VACompoundPolicy(_tiny_config(wmrm=True))
    peer = VACompoundPolicy(
        _tiny_config(
            wmrm=True,
            action_horizon=6,
            action_dim=4,
            wmrm_cycle_steps=6,
            va_world_mode="peer_sync_h6",
        )
    )
    assert legacy.world_action_readout is None
    assert peer.world_action_readout is not None
    assert not any(key.startswith("world_action_readout.") for key in legacy.state_dict())
    assert any(key.startswith("world_action_readout.") for key in peer.state_dict())
    latent = torch.randn(2, 6, peer.config.hidden_dim)
    readout = peer.world_action_readout(latent)
    assert readout.shape == (2, 6, peer.config.action_dim)
    assert float(readout.detach().abs().max()) <= 1.0


def test_off_path_matches_legacy_encode() -> None:
    config = _tiny_config(wmrm=False)
    model = VACompoundPolicy(config).eval()
    vision, proprio, previous, language, mask = _inputs(config)
    with torch.no_grad():
        cond = model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    assert model.wmrm is None
    assert cond.shape == (2, config.action_horizon, config.hidden_dim)
    assert model.last_wmrm is None


def test_world_goal_is_sealed() -> None:
    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    vision, proprio, previous, language, mask = _inputs(model.config)
    with pytest.raises(ValueError, match="world_goal is sealed"):
        model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
            world_goal=torch.zeros(2, model.config.wmrm_world_dim),
        )


def test_no_action_shaped_head() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    assert not block.has_action_shaped_head(action_dim=6)
    for module in block.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.out_features != 6


def test_world_loss_is_stopgrad_target() -> None:
    z_hat = torch.randn(3, 8, requires_grad=True)
    z_future = torch.randn(3, 8, requires_grad=True)
    loss = wmrm_world_loss(z_hat, z_future)
    loss.backward()
    assert z_hat.grad is not None
    assert z_future.grad is None


def test_action_dependency_probe_detects_linear_action() -> None:
    torch.manual_seed(0)
    n = 80
    state = torch.randn(n, 4)
    action = torch.randn(n, 3)
    target = state[:, :2] + 0.7 * action[:, :2]
    scores = action_dependency_scores(state, action, target)
    assert scores["mse_state_action"] < scores["mse_state"]
    assert scores["mse_state_action"] < scores["mse_state_shuffled"]


def test_shared_eye_is_pre_va_projection() -> None:
    torch.manual_seed(0)
    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    vision, proprio, previous, language, mask = _inputs(model.config)
    with torch.no_grad():
        projected = model.vision_projection(vision)
        eye = model.project_shared_eye(vision)
        torch.testing.assert_close(eye, projected)
        model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
        evidence_a = model.last_wmrm.evidence.clone()
        model.encode_condition(
            vision + 1e-3,
            proprio,
            previous,
            language_hidden=language,
            language_mask=mask,
        )
        evidence_b = model.last_wmrm.evidence.clone()
    assert model.last_wmrm is not None
    # evidence is read from shared DINO/projection, not post-VA mixed vision
    assert model.last_wmrm.evidence.shape[0] == 2
    assert not torch.allclose(evidence_a, evidence_b)


def test_cli_mutex_direct_head() -> None:
    from train import parse_args, validate_args

    args = parse_args(["--wam4va", "--direct-head"])
    with pytest.raises(ValueError, match="direct-head"):
        validate_args(args)


def test_policy_wires_executable_action_dim() -> None:
    model = VACompoundPolicy(_tiny_config(wmrm=True, action_dim=4, action_horizon=6))
    assert model.wmrm is not None
    assert model.wmrm.env_action_dim == 4
    assert model.wmrm.cycle_steps == 6


def test_default_inject_is_every_layer() -> None:
    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    assert model.config.wmrm_inject == "all"
    vision, proprio, previous, language, mask = _inputs(model.config)
    with torch.no_grad():
        model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    assert len(model.last_wmrm_auxes) == model.config.num_layers
    assert all(aux.z_tokens is not None for aux in model.last_wmrm_auxes)


def test_world_transition_updates_belief_and_dedups_innovation() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    block.eval()
    action = torch.randn(2, 5, 32)
    vision = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    with torch.no_grad():
        first = block.propose(action, vision, proprio)
        second = block.propose(
            action, vision, proprio, state=first.next_world_state, stage_index=1
        )
    aux1, aux2 = first.aux, second.aux
    belief1 = first.next_world_state.belief
    belief2 = second.next_world_state.belief
    innov1 = first.next_world_state.innovation
    innov2 = second.next_world_state.innovation
    assert aux1.z_spans.shape == (2, 3, 8)
    assert aux1.progress.shape == (2, 4)
    assert belief1.shape == (2, 8, 32)
    assert not torch.equal(innov1, innov2)
    assert aux2.belief.shape == belief2.shape


def test_innovation_projection_reads_previous_as_stable_memory() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    previous = torch.randn(2, 8, 32, requires_grad=True)
    current = (2.0 * previous.detach() + 0.1 * torch.randn_like(previous)).requires_grad_()

    projected = block._project_out(current, previous)
    flat_projected = projected.flatten(1)
    flat_previous = previous.detach().flatten(1)
    overlap = (flat_projected * flat_previous).sum(dim=-1)
    torch.testing.assert_close(overlap, torch.zeros_like(overlap), atol=2e-4, rtol=0)

    projected.square().mean().backward()
    assert previous.grad is None
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_innovation_projection_skips_near_zero_previous_direction() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    current = torch.randn(2, 8, 32, requires_grad=True)
    previous = torch.full_like(current, 1e-7, requires_grad=True)

    projected = block._project_out(current, previous)
    torch.testing.assert_close(projected, current)
    projected.sum().backward()
    assert previous.grad is None
    torch.testing.assert_close(current.grad, torch.ones_like(current))


def test_innovation_projection_stays_finite_for_large_finite_memory() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    previous = torch.full((2, 8, 32), 1.1e18)
    current = torch.full((2, 8, 32), 1.5e18, requires_grad=True)

    projected = block._project_out(current, previous)
    assert torch.isfinite(projected).all()
    projected.mean().backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_language_keys_condition_world_transition_without_writing_va() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    block.eval()
    action = torch.randn(2, 5, 32)
    vision = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    language = torch.randn(2, 4, 32)
    with torch.no_grad():
        plain = block.propose(action, vision, proprio)
        conditioned = block.propose(
            action, vision, proprio, language_keys=language
        )
    aux_a, aux_b = plain.aux, conditioned.aux
    assert aux_b.task_summary.shape == (2, 32)
    assert not torch.equal(aux_a.task_summary, aux_b.task_summary)


def test_span_ids_cover_horizon_when_not_divisible() -> None:
    block = WAM4VA(
        32, world_dim=8, proprio_dim=9, n_spans=3, cycle_steps=8
    )
    ids = block._span_ids(8, torch.device("cpu"))
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    action = torch.randn(1, 8, 32)
    means = block._segment_means(action)
    assert len(means) == 3
    torch.testing.assert_close(means[0], action[:, :3].mean(dim=1))
    torch.testing.assert_close(means[2], action[:, 6:].mean(dim=1))


def test_action_dep_hinge_penalizes_action_independent_z() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    z = torch.randn(4, 8)
    z_shuf = z.clone()
    future = z + 0.01
    hinge = block.action_dep_hinge(z, z_shuf, future, margin=0.05)
    assert float(hinge.item()) == pytest.approx(0.05)


def test_world_span_heads_exist() -> None:
    block = WAM4VA(32, world_dim=8, proprio_dim=9)
    assert len(block.span_heads) == 3


def test_dino_target_is_raw_tokens() -> None:
    from train import wmrm_next_feature_target

    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    batch = {
        "vision_tokens": torch.randn(
            2, 3, 11, model.config.vision_dim, requires_grad=True
        ),
    }
    got = wmrm_next_feature_target(model, batch, 0)
    torch.testing.assert_close(got, batch["vision_tokens"][:, 1])
    assert got.shape == (2, 11, model.config.vision_dim)
    assert not got.requires_grad
    assert got.grad_fn is None


def test_dino_map_uses_last_frame_full_channels() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_channels=4,
        map_frames=2,
        map_grid=4,
    )
    first = torch.randn(2, 16, 8)
    last = torch.randn(2, 16, 8)
    tokens = torch.cat((first, last), dim=1)
    mapped = block.encode_dino_map(tokens)
    assert mapped is not None
    assert mapped.shape == (2, 8, 4, 4)
    expected = last.view(2, 4, 4, 8).permute(0, 3, 1, 2)
    torch.testing.assert_close(mapped, expected)
    assert not torch.allclose(mapped, first.view(2, 4, 4, 8).permute(0, 3, 1, 2))
    action = torch.randn(2, 5, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    _, _, _, z_map = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens
    )
    assert z_map is not None
    torch.testing.assert_close(z_map, mapped, rtol=0.0, atol=0.0)
    clip = block.encode_dino_clip(tokens)
    assert clip is not None
    assert clip.shape == (2, 2, 8, 4, 4)
    torch.testing.assert_close(clip[:, -1], mapped)


def test_world_pred_uses_executable_env_action() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_channels=4,
        map_frames=2,
        map_grid=4,
        env_action_dim=4,
    )
    action = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    tokens = torch.randn(2, 32, 8)
    env = torch.randn(2, 6, 4)
    _, _, _, z0 = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env
    )
    _, _, _, z1 = block.predict_world(
        action + 1.5, proprio, belief, task, dino_tokens=tokens, env_action=env
    )
    torch.testing.assert_close(z0, z1, rtol=0.0, atol=0.0)
    block.film_shift.weight.data.normal_(0, 0.2)
    _, _, _, z2 = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env
    )
    _, _, _, z3 = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env + 1.5
    )
    assert not torch.allclose(z2, z3)


def test_dino_residual_zero_init_copies_current() -> None:
    block = WAM4VA(
        32, world_dim=8, proprio_dim=9, dino_dim=16
    )
    action = torch.randn(2, 5, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    dino = torch.randn(2, 7, 16)
    _, _, _, z_tokens = block.predict_world(
        action, proprio, belief, task, dino_tokens=dino
    )
    assert z_tokens is not None
    torch.testing.assert_close(z_tokens, dino, rtol=0.0, atol=0.0)


def test_predict_world_uses_logged_env_not_va_hidden() -> None:
    torch.manual_seed(4)
    block = WAM4VA(
        32, world_dim=8, proprio_dim=9, cycle_steps=6, env_action_dim=4
    )
    block.eval()
    action = torch.randn(2, 12, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    env = torch.randn(2, 6, 4)
    with torch.no_grad():
        z1, *_ = block.predict_world(action, proprio, belief, task, env_action=env)
        z2, *_ = block.predict_world(action + 10, proprio, belief, task, env_action=env)
        z3, *_ = block.predict_world(action, proprio, belief, task, env_action=env.flip(1))
        env_tail = env.clone()
        env_tail[:, 3:] += 2
        z4, *_ = block.predict_world(action, proprio, belief, task, env_action=env_tail)
    torch.testing.assert_close(z1, z2)
    assert not torch.allclose(z1, z3)
    assert not torch.allclose(z1, z4)


def test_world_transition_publishes_spatial_memory_tokens() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_frames=2,
        map_grid=4,
        world_grid=4,
        env_action_dim=4,
    )
    action = torch.randn(2, 5, 32)
    vision = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    tokens = torch.randn(2, 32, 8)
    env = torch.randn(2, 6, 4)
    transition = block.propose(
        action, vision, proprio, dino_tokens=tokens, env_action=env
    )
    aux = transition.aux
    assert aux.world_tokens is not None
    assert aux.world_tokens.shape == (2, 16, 32)
    torch.testing.assert_close(transition.world_message, aux.world_tokens)
    assert aux.z_tokens is not None
    assert aux.z_tokens.shape == (2, 8, 4, 4)


def test_each_forward_emits_full_spatial_map() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_frames=2,
        map_grid=4,
        world_grid=4,
        env_action_dim=4,
        predictor="st_blocks",
        predictor_depth=2,
        predictor_width=32,
        predictor_heads=4,
    )
    action = torch.randn(2, 5, 32)
    vision = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    tokens = torch.randn(2, 32, 8)
    state = None
    env = torch.randn(2, 6, 4)
    maps = []
    with torch.no_grad():
        block.st_predictor.out_proj.weight.normal_(0, 0.05)
    for stage in range(3):
        transition = block.propose(
            action,
            vision,
            proprio,
            state=state,
            dino_tokens=tokens,
            env_action=env,
            stage_index=stage,
        )
        aux = transition.aux
        assert aux.z_tokens is not None
        assert aux.z_tokens.shape == (2, 8, 4, 4)
        assert aux.world_tokens is not None
        assert aux.world_tokens.shape == (2, 16, 32)
        maps.append(aux.z_tokens)
        state = transition.next_world_state
    assert not torch.allclose(maps[0], maps[-1])
    maps[-1].square().mean().backward()
    assert block.st_predictor.in_proj.weight.grad is not None
    assert float(block.st_predictor.in_proj.weight.grad.norm()) > 0


def test_wam_state_finite_validation_names_corrupt_recurrent_field() -> None:
    state = WAMState(
        belief=torch.zeros(2, 8, 32),
        innovation=torch.full((2, 8, 32), float("nan")),
    )
    with pytest.raises(
        FloatingPointError,
        match="proposal commit.*WAMState.innovation.*NaN or Inf",
    ):
        state.validate_finite(boundary="proposal commit")


def test_env_action_requires_exact_cycle_and_dimension() -> None:
    block = WAM4VA(
        32, world_dim=8, proprio_dim=9, cycle_steps=6, env_action_dim=4
    )
    tokens, flat = block.encode_env_action(
        torch.randn(2, 6, 4),
        batch=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert tokens.shape == (2, 6, 32)
    assert flat.shape == (2, 32)
    with pytest.raises(ValueError, match="exact shape"):
        block.encode_env_action(
            torch.randn(2, 5, 4),
            batch=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="exact shape"):
        block.encode_env_action(
            torch.randn(2, 6, 7),
            batch=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("cycle_steps", [1, 2, 3, 6])
def test_world_action_encoding_represents_exact_p_step_cycle(
    cycle_steps: int,
) -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        cycle_steps=cycle_steps,
        env_action_dim=4,
    )
    actions = torch.randn(2, cycle_steps, 4)
    tokens, flat = block.encode_env_action(
        actions,
        batch=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert tokens.shape == (2, cycle_steps, 32)
    assert flat.shape == (2, 32)


def test_world_memory_keeps_native_16x16() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=16,
        map_frames=1,
        map_grid=16,
        world_grid=16,
        env_action_dim=4,
    )
    mapped = block.encode_dino_map(torch.randn(1, 256, 8))
    tokens = block.encode_world_tokens(mapped)
    assert tokens.shape == (1, 256, 32)


def test_cli_mutex_vjepa_target_on_dino(tmp_path) -> None:
    from train import parse_args, validate_args

    data = tmp_path / "windows.pt"
    ckpt = tmp_path / "dino.pt"
    data.write_bytes(b"x")
    ckpt.write_bytes(b"x")
    args = parse_args(
        [
            "--wmrm",
            "--dino-main-vision",
            "--wmrm-target",
            "vjepa",
            "--data",
            str(data),
            "--main-vision-checkpoint",
            str(ckpt),
        ]
    )
    with pytest.raises(ValueError, match="vjepa"):
        validate_args(args)


def test_matched_no_fixed_point_perm_same_task() -> None:
    task_id = torch.tensor([0, 0, 1, 1])
    eye = torch.tensor([[0.0], [0.1], [10.0], [10.2]])
    proprio = torch.zeros(4, 2)
    perm = matched_no_fixed_point_perm(task_id, eye, proprio)
    assert perm.tolist() == [1, 0, 3, 2]
    assert (perm != torch.arange(4)).all()
    assert task_id[perm].eq(task_id).all()
    # Clustered B=3 must still be a derangement, not many-to-one nearest neighbor.
    eye3 = torch.tensor([[0.0], [0.1], [0.11]])
    perm3 = matched_no_fixed_point_perm(None, eye3, torch.zeros(3, 2))
    assert sorted(perm3.tolist()) == [0, 1, 2]
    assert (perm3 != torch.arange(3)).all()


def test_wmrm_target_vjepa_rejected_on_dino_config() -> None:
    with pytest.raises(ValueError, match="vjepa"):
        _tiny_config(
            wmrm=True,
            wmrm_target="vjepa",
            main_vision_backbone="dinov2_vitl14_reg4",
            main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
            main_vision_image_size=224,
            main_vision_dim=1024,
            main_vision_grid=8,
            main_vision_frames=4,
            main_vision_tokens=256,
        )


def test_cli_wmrm_only_requires_wam4va() -> None:
    from train import parse_args, validate_args

    args = parse_args(["--wmrm-only"])
    with pytest.raises(ValueError, match="wmrm-only"):
        validate_args(args)


def test_cli_wam4va_uses_state_exchange_without_writeback_flags() -> None:
    from train import parse_args, validate_args

    args = parse_args(["--wam4va"])
    validate_args(args)
    assert not hasattr(args, "wmrm_handshake")
    assert not hasattr(args, "wmrm_med_weight")
    assert not hasattr(args, "wmrm_pi_kl_weight")


def test_peer_joint_optimizer_keeps_va_world_and_flow_trainable() -> None:
    from argparse import Namespace

    from train import _feature_optimizer_groups

    model = VACompoundPolicy(
        _tiny_config(
            wmrm=True,
            action_horizon=6,
            action_dim=4,
            wmrm_cycle_steps=6,
            va_world_mode="peer_sync_h6",
        )
    )
    optimizer_args = Namespace(
        wmrm_only=False,
        va_only=False,
        lr=1e-4,
        action_vision_only=False,
        head_only=False,
        servo_only=False,
    )
    groups = _feature_optimizer_groups(optimizer_args, model, None)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert any(name.startswith("layers.") for name in trainable)
    assert any(name.startswith("flow_head.") for name in trainable)
    assert any(name.startswith("wmrm.") for name in trainable)
    assert any(name.startswith("world_action_readout.") for name in trainable)
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    assert grouped == {id(parameter) for parameter in model.parameters()}


def test_wmrm_only_freezes_va_and_flow() -> None:
    from argparse import Namespace

    from train import _feature_optimizer_groups

    model = VACompoundPolicy(_tiny_config(wmrm=True))
    args = Namespace(
        wmrm_only=True,
        va_only=False,
        lr=1e-4,
        action_vision_only=False,
        head_only=False,
    )
    groups = _feature_optimizer_groups(args, model, None)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith("wmrm.") for name in trainable)
    assert not any(name.startswith("flow_head.") for name in trainable)
    assert not any(name.startswith("layers.") for name in trainable)
    assert groups[0]["lr"] == 1e-4


def test_st_predictor_starts_near_last_frame_and_uses_proposal() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_frames=2,
        map_grid=4,
        world_grid=4,
        env_action_dim=4,
        predictor="st_blocks",
        predictor_depth=2,
        predictor_width=32,
        predictor_heads=4,
    )
    action = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    tokens = torch.randn(2, 32, 8)
    _, _, _, z_map = block.predict_world(action, proprio, belief, task, dino_tokens=tokens)
    current = block.encode_dino_map(tokens)
    initial_residual = z_map - current
    residual_mean = float(initial_residual.detach().abs().mean().item())
    assert 0.0 < residual_mean < 0.05
    assert z_map.shape == (2, 8, 4, 4)
    block.st_predictor.out_proj.weight.data.normal_(0, 0.05)
    env = torch.randn(2, 6, 4)
    _, _, _, z0 = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env
    )
    _, _, _, z1 = block.predict_world(
        action + 1, proprio, belief, task, dino_tokens=tokens, env_action=env
    )
    _, _, _, z2 = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env + 1
    )
    torch.testing.assert_close(z0, z1)
    assert not torch.allclose(z0, z2)
    loss = z0.square().mean()
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in block.st_predictor.parameters())
    assert block.env_step.weight.grad is not None
    assert float(block.env_step.weight.grad.norm()) > 0


def test_st_predictor_recurrent_map_uses_stable_residual_gradient() -> None:
    """Map values recur, while their nonlinear context Jacobian does not compound."""
    predictor = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=12,
        predictor="st_blocks",
        predictor_depth=2,
        predictor_width=24,
        predictor_heads=4,
        map_frames=4,
        map_size=2,
    ).st_predictor
    assert predictor is not None
    clip = torch.randn(2, 4, 12, 2, 2)
    cond = torch.randn(2, 5, 24)
    previous = torch.randn(2, 12, 2, 2, requires_grad=True)

    output = predictor(clip, cond, previous)
    output.sum().backward()

    # The recurrent value is still the exact residual base, so credit reaches
    # the prior stage through an identity path rather than a 32-deep product of
    # predictor Jacobians.  The current predictor itself remains trainable.
    torch.testing.assert_close(previous.grad, torch.ones_like(previous))
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in predictor.parameters()
    )


def test_belief_recurrence_uses_identity_gradient_and_trains_current_update() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_frames=2,
        map_grid=4,
        world_grid=4,
        env_action_dim=4,
        predictor="st_blocks",
        predictor_depth=1,
        predictor_width=32,
        predictor_heads=4,
    )
    previous_belief = torch.randn(2, 8, 32, requires_grad=True)
    previous_innovation = torch.randn(2, 8, 32)
    previous_map = torch.randn(2, 8, 4, 4)
    proposal = block.propose(
        torch.randn(2, 6, 32),
        torch.randn(2, 7, 32),
        torch.randn(2, 9),
        state=WAMState(
            belief=previous_belief,
            innovation=previous_innovation,
            world_map=previous_map,
        ),
        dino_tokens=torch.randn(2, 32, 8),
        env_action=torch.randn(2, 6, 4),
        stage_index=1,
    )
    cotangent = torch.randn_like(proposal.next_world_state.belief)
    (proposal.next_world_state.belief * cotangent).sum().backward()

    torch.testing.assert_close(previous_belief.grad, cotangent)
    assert block.belief_write.o.weight.grad is not None
    assert float(block.belief_write.o.weight.grad.abs().sum()) > 0
    assert block.evidence_from_belief.weight.grad is not None
    assert float(block.evidence_from_belief.weight.grad.abs().sum()) > 0


def test_world_loss_still_trains_belief_update_after_stable_recurrence() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_frames=2,
        map_grid=4,
        world_grid=4,
        env_action_dim=4,
        predictor="st_blocks",
        predictor_depth=1,
        predictor_width=32,
        predictor_heads=4,
    )
    proposal = block.propose(
        torch.randn(2, 6, 32),
        torch.randn(2, 7, 32),
        torch.randn(2, 9),
        state=WAMState(
            belief=torch.randn(2, 8, 32),
            innovation=torch.randn(2, 8, 32),
        ),
        dino_tokens=torch.randn(2, 32, 8),
        env_action=torch.randn(2, 6, 4),
    )
    assert proposal.aux.z_tokens is not None
    proposal.aux.z_tokens.square().mean().backward()

    assert block.belief_write.o.weight.grad is not None
    assert float(block.belief_write.o.weight.grad.abs().sum()) > 0
    assert block.evidence_from_belief.weight.grad is not None
    assert float(block.evidence_from_belief.weight.grad.abs().sum()) > 0


def test_previous_map_is_read_not_only_added() -> None:
    block = WAM4VA(
        32,
        world_dim=8,
        proprio_dim=9,
        dino_dim=8,
        map_size=4,
        map_frames=2,
        map_grid=4,
        world_grid=4,
        env_action_dim=4,
        predictor="st_blocks",
        predictor_depth=2,
        predictor_width=32,
        predictor_heads=4,
    )
    action = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    tokens = torch.randn(2, 32, 8)
    env = torch.randn(2, 6, 4)
    with torch.no_grad():
        block.st_predictor.out_proj.weight.normal_(0, 0.08)
    prev_a = torch.randn(2, 8, 4, 4)
    prev_b = prev_a + 1.5
    _, _, _, za = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env, previous_map=prev_a
    )
    _, _, _, zb = block.predict_world(
        action, proprio, belief, task, dino_tokens=tokens, env_action=env, previous_map=prev_b
    )
    assert not torch.allclose(za, zb)
    assert not torch.allclose(za - prev_a, zb - prev_b)
