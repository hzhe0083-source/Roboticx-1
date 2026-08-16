"""P2 WMRM: single-stream world-mediated residual, no candidates, no Δv."""

from __future__ import annotations

import pytest
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy
from va_compound.wmrm import (
    WorldMediatedResidualModulation,
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
    block = WorldMediatedResidualModulation(
        32, world_dim=8, rank=4, proprio_dim=9, n_spans=3, cycle_steps=8
    )
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


def test_fm_condition_hinge_has_gate_grad_at_zero_init() -> None:
    torch.manual_seed(0)
    block = WorldMediatedResidualModulation(32, world_dim=8, rank=4, proprio_dim=9)
    torch.testing.assert_close(
        block.gate_proj.weight, torch.zeros_like(block.gate_proj.weight)
    )
    torch.testing.assert_close(
        block.gate_proj.bias, torch.zeros_like(block.gate_proj.bias)
    )
    action = torch.randn(3, 5, 32)
    vision = torch.randn(3, 6, 32)
    proprio = torch.randn(3, 9)
    norm = torch.nn.LayerNorm(32)
    _, aux, _, _ = block(action, vision, proprio)
    hinge = block.fm_condition_hinge(action, aux, norm, margin=0.05)
    hinge.backward()
    assert block.gate_proj.weight.grad is not None
    assert float(block.gate_proj.weight.grad.norm()) > 0
    assert float(hinge.item()) == pytest.approx(0.05)


def test_fm_condition_hinge_disables_dropout() -> None:
    torch.manual_seed(1)
    block = WorldMediatedResidualModulation(
        32, world_dim=8, rank=4, proprio_dim=9, mixer_dropout=0.9
    )
    block.train()
    action = torch.randn(3, 5, 32)
    vision = torch.randn(3, 6, 32)
    proprio = torch.randn(3, 9)
    norm = torch.nn.LayerNorm(32)
    with torch.no_grad():
        _, aux, _, _ = block(action, vision, proprio)
        values = []
        for _ in range(5):
            torch.manual_seed(11)
            values.append(block.fm_condition_hinge(action, aux, norm, margin=0.05))
    stacked = torch.stack(values)
    torch.testing.assert_close(stacked, stacked[:1].expand_as(stacked), rtol=0.0, atol=0.0)


def test_dino_target_is_raw_tokens() -> None:
    from train import wmrm_next_feature_target

    model = VACompoundPolicy(_tiny_config(wmrm=True)).eval()
    batch = {
        "vision_tokens": torch.randn(2, 3, 11, model.config.vision_dim),
    }
    got = wmrm_next_feature_target(model, batch, 0)
    torch.testing.assert_close(got, batch["vision_tokens"][:, 1])
    assert got.shape == (2, 11, model.config.vision_dim)


def test_dino_map_uses_last_frame_full_channels() -> None:
    block = WorldMediatedResidualModulation(
        32,
        world_dim=8,
        rank=4,
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


def test_dino_residual_zero_init_copies_current() -> None:
    block = WorldMediatedResidualModulation(
        32, world_dim=8, rank=4, proprio_dim=9, dino_dim=16
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


def test_predict_world_ignores_action_tail() -> None:
    torch.manual_seed(4)
    block = WorldMediatedResidualModulation(
        32, world_dim=8, rank=4, proprio_dim=9, cycle_steps=6
    )
    block.eval()
    action = torch.randn(2, 12, 32)
    proprio = torch.randn(2, 9)
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    with torch.no_grad():
        z1, *_ = block.predict_world(action, proprio, belief, task)
        action2 = action.clone()
        action2[:, 6:] += 10
        z2, *_ = block.predict_world(action2, proprio, belief, task)
        action3 = action.clone()
        action3[:, :6] += 1
        z3, *_ = block.predict_world(action3, proprio, belief, task)
    torch.testing.assert_close(z1, z2)
    assert not torch.allclose(z1, z3)


def test_action_dep_hinge_zero_WA_has_grad() -> None:
    torch.manual_seed(2)
    block = WorldMediatedResidualModulation(
        32, world_dim=8, rank=4, proprio_dim=9, cycle_steps=6
    )
    torch.nn.init.zeros_(block.world_from_action.weight)
    torch.nn.init.zeros_(block.world_from_action.bias)
    action = torch.randn(4, 8, 32)
    proprio = torch.randn(4, 9)
    belief = torch.randn(4, 8, 32)
    task = torch.randn(4, 32)
    z_hat, *_ = block.predict_world(action, proprio, belief, task)
    used = min(block.cycle_steps, action.shape[1])
    action_shuf = action.clone()
    perm = torch.randperm(action.shape[0])
    action_shuf[:, :used] = action[perm, :used]
    z_shuf, *_ = block.predict_world(action_shuf, proprio, belief, task)
    target = torch.randn_like(z_hat)
    loss = block.action_dep_hinge(z_hat, z_shuf, target)
    loss.backward()
    assert block.world_from_action.weight.grad is not None
    assert float(block.world_from_action.weight.grad.norm()) > 0


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


def test_cli_wmrm_only_forces_med_off(tmp_path) -> None:
    from train import parse_args, validate_args

    data = tmp_path / "windows.pt"
    data.write_bytes(b"x")
    args = parse_args(
        [
            "--wam4va",
            "--wmrm-only",
            "--wmrm-med-weight",
            "0.5",
            "--mtvj-train-metric-head",
            "--data",
            str(data),
        ]
    )
    validate_args(args)
    assert args.wmrm_med_weight == 0.0
    assert args.wmrm_adep_weight == 0.0
    assert args.wmrm_handshake is False
    assert args.mtvj_train_metric_head is False


def test_wmrm_only_freezes_va_and_flow() -> None:
    from argparse import Namespace

    from train import _feature_optimizer_groups

    model = VACompoundPolicy(_tiny_config(wmrm=True))
    args = Namespace(wmrm_only=True, lr=1e-4, action_vision_only=False, head_only=False)
    groups = _feature_optimizer_groups(args, model, None)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith("wmrm.") for name in trainable)
    assert not any(name.startswith("flow_head.") for name in trainable)
    assert not any(name.startswith("layers.") for name in trainable)
    assert groups[0]["lr"] == 1e-4


def test_handshake_off_ignores_va_action() -> None:
    torch.manual_seed(0)
    model = VACompoundPolicy(_tiny_config(wmrm=True, wmrm_handshake=False)).eval()
    vision, proprio, previous, language, mask = _inputs(model.config)
    with torch.no_grad():
        closed = model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
        model.wmrm.gate_proj.bias.fill_(2.0)
        still = model.encode_condition(
            vision, proprio, previous, language_hidden=language, language_mask=mask
        )
    torch.testing.assert_close(closed, still, rtol=0.0, atol=0.0)
    action = torch.randn(2, 5, 32)
    other = action + 3.0
    belief = torch.randn(2, 8, 32)
    task = torch.randn(2, 32)
    z1, *_ = model.wmrm.predict_world(action, proprio, belief, task)
    z2, *_ = model.wmrm.predict_world(other, proprio, belief, task)
    torch.testing.assert_close(z1, z2)
