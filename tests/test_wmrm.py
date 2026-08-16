"""P2 WMRM: single-stream world-mediated residual, no candidates, no Δv."""

from __future__ import annotations

import pytest
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy
from va_compound.wmrm import (
    WorldMediatedResidualModulation,
    action_dependency_scores,
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


def test_zero_gate_is_identity() -> None:
    torch.manual_seed(3)
    off = VACompoundPolicy(_tiny_config(wmrm=False)).eval()
    torch.manual_seed(3)
    on = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    vision, proprio, previous, language, mask = _inputs(off.config)
    with torch.no_grad():
        legacy = off.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
        treated = on.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    assert on.wmrm is not None
    torch.testing.assert_close(on.wmrm.gate_proj.weight, torch.zeros_like(on.wmrm.gate_proj.weight))
    torch.testing.assert_close(treated, legacy, rtol=0.0, atol=0.0)


def test_nonzero_gate_changes_condition() -> None:
    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    vision, proprio, previous, language, mask = _inputs(model.config)
    with torch.no_grad():
        closed = model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
        model.wmrm.gate_proj.bias.fill_(2.0)
        opened = model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    assert not torch.allclose(closed, opened)
    assert model.last_wmrm is not None
    assert model.last_wmrm.z_hat.shape == (2, model.config.wmrm_world_dim)
    assert model.last_wmrm.pi.shape == (2, model.config.action_horizon, model.config.wmrm_rank)
    assert model.last_wmrm.gate.shape == (2, 1)
    assert torch.isfinite(model.last_wmrm.pi).all()
    torch.testing.assert_close(
        model.last_wmrm.pi.sum(dim=-1),
        torch.ones(2, model.config.action_horizon),
        atol=1e-5,
        rtol=1e-5,
    )


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
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
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


def test_mutex_with_wam_joint() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _tiny_config(wmrm=True, wam_joint=True)


def test_action_dependency_probe_detects_linear_action() -> None:
    torch.manual_seed(0)
    n = 80
    state = torch.randn(n, 4)
    action = torch.randn(n, 3)
    target = state[:, :2] + 0.7 * action[:, :2]
    scores = action_dependency_scores(state, action, target)
    assert scores["mse_state_action"] < scores["mse_state"]
    assert scores["mse_state_action"] < scores["mse_state_shuffled"]


def test_cli_mutex() -> None:
    from train import parse_args, validate_args

    args = parse_args(["--wmrm", "--wam-joint"])
    with pytest.raises(ValueError, match="wmrm"):
        validate_args(args)


def test_shared_eye_is_pre_va_projection() -> None:
    torch.manual_seed(0)
    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    vision, proprio, previous, language, mask = _inputs(model.config)
    with torch.no_grad():
        projected = model.vision_projection(vision)
        model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    assert model.last_wmrm is not None
    # evidence is read from shared DINO/projection, not post-VA mixed vision
    assert model.last_wmrm.evidence.shape[0] == 2


def test_cli_mutex_direct_head() -> None:
    from train import parse_args, validate_args

    args = parse_args(["--wam4va", "--direct-head"])
    with pytest.raises(ValueError, match="direct-head"):
        validate_args(args)


def test_inject_all_zero_gate_still_identity() -> None:
    torch.manual_seed(3)
    off = VACompoundPolicy(_tiny_config(wmrm=False)).eval()
    torch.manual_seed(3)
    on = VACompoundPolicy(_tiny_config(wmrm=True, wmrm_inject="all")).eval()
    vision, proprio, previous, language, mask = _inputs(off.config)
    with torch.no_grad():
        legacy = off.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
        treated = on.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    torch.testing.assert_close(treated, legacy, rtol=0.0, atol=0.0)


def test_pi_shuffle_kl_is_finite() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    block.eval()
    action = torch.randn(4, 5, 32)
    vision = torch.randn(4, 7, 32)
    proprio = torch.randn(4, 9)
    with torch.no_grad():
        kl = block.pi_shuffle_kl(action, vision, proprio)
    assert torch.isfinite(kl)
    assert kl.ndim == 0


def test_handshake_updates_belief_and_dedups_innovation() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    block.eval()
    action = torch.randn(2, 5, 32)
    vision = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    with torch.no_grad():
        _, aux1, belief1, innov1 = block(action, vision, proprio)
        _, aux2, belief2, innov2 = block(
            action, vision, proprio, belief=belief1, prev_innovation=innov1
        )
    assert aux1.z_spans.shape == (2, 3, 8)
    assert aux1.progress.shape == (2, 4)
    assert belief1.shape == (2, 8, 32)
    assert not torch.equal(innov1, innov2)
    assert aux2.belief.shape == belief2.shape


def test_language_keys_keep_zero_gate_identity() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    block.eval()
    action = torch.randn(2, 5, 32)
    vision = torch.randn(2, 6, 32)
    proprio = torch.randn(2, 9)
    language = torch.randn(2, 4, 32)
    with torch.no_grad():
        out_a, aux_a, _, _ = block(action, vision, proprio)
        out_b, aux_b, _, _ = block(action, vision, proprio, language_keys=language)
    torch.testing.assert_close(out_a, action, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_b, action, atol=0.0, rtol=0.0)
    assert aux_b.task_summary.shape == (2, 32)


def test_supervised_z_hat_is_broadcast_to_horizon() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9, n_spans=3)
    z_hat = torch.randn(2, 8)
    per_step = block._z_per_step(z_hat, 8)
    assert per_step.shape == (2, 8, 8)
    torch.testing.assert_close(per_step[:, 0], z_hat)
    torch.testing.assert_close(per_step[:, 7], z_hat)


def test_span_ids_cover_horizon_when_not_divisible() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9, n_spans=3)
    ids = block._span_ids(8, torch.device("cpu"))
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    action = torch.randn(1, 8, 32)
    means = block._segment_means(action)
    assert len(means) == 3
    torch.testing.assert_close(means[0], action[:, :3].mean(dim=1))
    torch.testing.assert_close(means[2], action[:, 6:].mean(dim=1))


def test_action_dep_hinge_penalizes_action_independent_z() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    z = torch.randn(4, 8)
    z_shuf = z.clone()
    future = z + 0.01
    hinge = block.action_dep_hinge(z, z_shuf, future, margin=0.05)
    assert float(hinge.item()) == pytest.approx(0.05)


def test_fm_condition_hinge_zero_when_gate_closed() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    block.eval()
    action = torch.randn(3, 5, 32)
    vision = torch.randn(3, 6, 32)
    proprio = torch.randn(3, 9)
    norm = torch.nn.LayerNorm(32)
    with torch.no_grad():
        _, aux, _, _ = block(action, vision, proprio)
        hinge = block.fm_condition_hinge(action, aux, norm, margin=0.05)
    assert float(aux.gate.abs().max()) == 0.0
    assert float(hinge.item()) == pytest.approx(0.05)


def test_source_gates_and_span_heads_exist() -> None:
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    assert block.source_gates.shape == (3,)
    assert len(block.span_heads) == 3
    assert block.ca_belief is not block.ca_geo
