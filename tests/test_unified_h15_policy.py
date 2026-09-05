import copy

import pytest
import torch

from tests.test_layerwise_expert_policy import config
from va_compound import VACompoundPolicy
from va_compound.policy.action_expert import LayerwiseActionExpert


def make_model():
    return VACompoundPolicy(config(architecture_version="dual_tower_h15_v1", action_horizon=15,
                                   world_state_supervision=True)).eval()


def test_one_h15_expert_without_six_step_seam():
    model = make_model()
    assert sum(isinstance(m, LayerwiseActionExpert) for m in model.modules()) == 1
    assert model.tail_action_expert is None and model.extension_action_expert is None
    assert all(layer.protected_action_prefixes == () for layer in model.layers)
    assert not any(n.startswith(("tail_action_expert.", "extension_action_expert.")) for n, _ in model.named_parameters())
    condition = torch.randn(1, 3, 15, 16, requires_grad=True)
    noisy = torch.randn(1, 15, 4)
    velocity = model.flow_velocity(condition, noisy, torch.tensor([.5]))
    velocity.square().mean().backward()
    assert (condition.grad.abs().sum(-1) > 0).all()
    changed = noisy.clone()
    changed[:, 6:] += 3
    other = model.flow_velocity(condition.detach(), changed, torch.tensor([.5]))
    assert not torch.equal(other[:, :6], velocity[:, :6])


def test_h15_next_update_checkpoint_parity():
    torch.manual_seed(8)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    condition = torch.randn(1, 3, 15, 16)
    noisy = torch.randn(1, 15, 4)
    def update(m, opt):
        opt.zero_grad(set_to_none=True)
        m.flow_velocity(condition, noisy, torch.tensor([.5])).square().mean().backward()
        opt.step()
    update(model, optimizer)
    restored = make_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    update(model, optimizer)
    update(restored, restored_opt)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_h15_episode_autocast_and_world_state_supervision():
    from tests.test_episode_world_training import batch, run
    from va_compound.training.episode_memory import EpisodeMemoryBank
    model = make_model()
    bank = EpisodeMemoryBank()
    for start, count, end in ((0, 2, False), (30, 1, True)):
        data = batch(count=count, start=start, end=end)
        data["actions"] = data["actions"][:, :, :15]
        data["action_valid_mask"] = data["action_valid_mask"][:, :, :15]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            velocity, conditions = run(model, data, bank)
        assert velocity.shape == (1, 3, 15, 4)
        assert conditions.shape == (1, 3, 3, 15, 16)
        model.last_wmrm_loss.backward()
        bank.commit()
    assert bank.entries == {}


def test_unified_version_rejects_h50():
    with pytest.raises(ValueError, match="unified H15"):
        config(architecture_version="dual_tower_h15_v1", action_horizon=50)
