import torch

from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.training.rollout import rollout_policy
from tests.test_visual_world_rollout import _rollout_batch


def config(**kwargs):
    values = dict(language_dim=12, vision_dim=8, hidden_dim=16, num_layers=3,
                  num_heads=4, action_horizon=50, planning_stride=15,
                  deployment_execution_horizon=15, action_dim=4, proprio_dim=5,
                  flow_layers=3, architecture_version="dual_tower_expert_v1",
                  fusion_pair_count=2, wmrm=True, va_world_mode="peer_sync_h6",
                  wmrm_cycle_steps=15, wmrm_map_size=2, wmrm_map_channels=8,
                  wmrm_world_grid=2, wmrm_predictor="st_blocks", wmrm_predictor_depth=1,
                  wmrm_predictor_width=16, wmrm_predictor_heads=4,
                  main_vision_grid=2, main_vision_frames=1)
    values.update(kwargs)
    return VACompoundConfig(**values)


def test_new_condition_state_and_world_memory():
    torch.manual_seed(11)
    model = VACompoundPolicy(config()).eval()
    args = (torch.randn(2, 4, 8), torch.randn(2, 5), torch.randn(2, 4))
    language = torch.randn(2, 5, 12)
    condition, memory = model.encode_condition(*args, language_hidden=language, return_visual_memory=True)
    assert condition.shape == (2, 3, 50, 16)
    assert len(model.last_wmrm_auxes) == 2
    assert memory.layers[0].shape[1] == 5
    assert memory.world_state.world_map.shape == (2, 8, 2, 2)
    condition.retain_grad()
    model.flow_velocity(condition, torch.randn(2, 50, 4), torch.rand(2)).square().mean().backward()
    assert all(torch.count_nonzero(condition.grad[:, i]) for i in range(3))
    assert model.state_projection.weight.grad.norm() > 0
    next_condition, next_memory = model.encode_condition(*args, language_hidden=language,
                                                        visual_memory=memory, return_visual_memory=True)
    assert next_condition.shape == condition.shape
    assert next_memory.world_state is not memory.world_state


def test_nested_expert_prefix_does_not_read_future_tail():
    torch.manual_seed(12)
    model = VACompoundPolicy(config()).eval()
    condition = torch.randn(2, 3, 50, 16)
    noise = torch.randn(2, 50, 4)
    time = torch.rand(2)
    expected = model.flow_velocity(condition, noise, time)
    changed, changed_noise = condition.clone(), noise.clone()
    changed[:, :, 15:] += 4
    changed_noise[:, 15:] -= 4
    actual = model.flow_velocity(changed, changed_noise, time)
    torch.testing.assert_close(actual[:, :15], expected[:, :15], rtol=0, atol=0)
    assert not torch.equal(actual[:, 15:], expected[:, 15:])
    changed[:, :, 6:] += 4
    changed_noise[:, 6:] -= 4
    actual = model.flow_velocity(changed, changed_noise, time)
    torch.testing.assert_close(actual[:, :6], expected[:, :6], rtol=0, atol=0)


def test_executed_suffix_trains_all_va_layers_without_future_tail_leakage():
    torch.manual_seed(18)
    model = VACompoundPolicy(config()).eval()
    condition = torch.randn(1, 3, 50, 16, requires_grad=True)
    velocity = model.flow_velocity(condition, torch.randn(1, 50, 4), torch.rand(1))
    gradient = torch.autograd.grad(velocity[:, 6:15].square().mean(), condition, retain_graph=True)[0]
    assert all(gradient[:, layer, :15].norm() > 0 for layer in range(3))
    assert torch.count_nonzero(gradient[:, :, 15:]) == 0
    tail_gradient = torch.autograd.grad(velocity[:, 15:].square().mean(), condition)[0]
    assert torch.count_nonzero(tail_gradient) == 0


def test_legacy_executed_suffix_keeps_checkpoint_gradient_contract():
    model = VACompoundPolicy(config(architecture_version="legacy")).eval()
    condition = torch.randn(1, 50, 16, requires_grad=True)
    velocity = model.flow_velocity(condition, torch.randn(1, 50, 4), torch.rand(1))
    gradient = torch.autograd.grad(velocity[:, 6:15].square().mean(), condition)[0]
    assert torch.count_nonzero(gradient) == 0


def test_new_expert_checkpoint_roundtrip_and_cached_sampling():
    model = VACompoundPolicy(config()).eval()
    restored = VACompoundPolicy(VACompoundConfig(**model.config.__dict__)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    condition = torch.randn(1, 3, 50, 16)
    noise = torch.randn(1, 50, 4)
    torch.testing.assert_close(model.sample_actions(condition, steps=2, noise=noise),
                               restored.sample_actions(condition, steps=2, noise=noise), rtol=0, atol=0)


def test_dual_tower_per_decision_language_and_peer_world_rollout():
    torch.manual_seed(42)
    cfg = config(
        action_horizon=6,
        planning_stride=6,
        deployment_execution_horizon=6,
        flow_layers=1,
        wmrm_cycle_steps=6,
        wmrm_full_language_tokens=True,
        wmrm_reads_fused_va_tokens=True,
    )
    model = VACompoundPolicy(cfg).train()
    batch, noisy_actions, flow_time = _rollout_batch(transitions_valid=True)
    batch_size, sequence = batch["actions"].shape[:2]
    language_len, language_dim = 4, cfg.language_dim
    language_hidden = torch.randn(
        batch_size, sequence, language_len, language_dim, requires_grad=True
    )
    batch["language_hidden"] = language_hidden
    batch["language_mask"] = torch.ones(
        batch_size, sequence, language_len, dtype=torch.bool
    )

    velocities, conditions = rollout_policy(
        model,
        batch,
        noisy_actions,
        flow_time,
        visual_world_supervision=True,
    )

    # Check shapes [B, T, 3, H, D]
    assert conditions.shape == (
        batch_size,
        sequence,
        3,
        cfg.action_horizon,
        cfg.hidden_dim,
    )
    assert velocities.shape == (
        batch_size,
        sequence,
        cfg.action_horizon,
        cfg.action_dim,
    )

    # Visual world loss finite and gradients
    assert torch.isfinite(model.last_wmrm_loss)

    loss = velocities.sum() + model.last_wmrm_loss
    loss.backward()

    # Differentiability to each language decision
    assert language_hidden.grad is not None
    for t in range(sequence):
        assert language_hidden.grad[:, t].norm() > 0

    # Differentiability to state projection
    assert model.state_projection.weight.grad is not None
    assert model.state_projection.weight.grad.norm() > 0
