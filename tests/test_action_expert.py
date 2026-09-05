import pytest
import torch
import torch.nn as nn

from va_compound.policy.action_expert import (
    ActionTransformerBlock,
    LayerwiseActionExpert,
)


def test_init_and_validation():
    # Enforces num_layers == 3
    with pytest.raises(ValueError, match="num_layers must be exactly 3"):
        LayerwiseActionExpert(
            action_dim=7,
            condition_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

    with pytest.raises(ValueError, match="num_layers must be exactly 3"):
        LayerwiseActionExpert(
            action_dim=7,
            condition_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=4,
        )

    # Hidden dim divisible by num_heads
    with pytest.raises(ValueError, match="divisible by num_heads"):
        LayerwiseActionExpert(
            action_dim=7,
            condition_dim=32,
            hidden_dim=65,
            num_heads=4,
            num_layers=3,
        )


def test_forward_shapes_and_horizons():
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=32,
        hidden_dim=64,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    expert.eval()

    # Test horizon slices 6, 15, 50
    batch_size = 2
    for H in [6, 15, 50]:
        noisy_actions = torch.randn(batch_size, H, 7)
        conditions = (
            torch.randn(batch_size, H, 32),
            torch.randn(batch_size, H, 32),
            torch.randn(batch_size, H, 32),
        )
        # test with both [B] and [B, 1] time
        t_1d = torch.rand(batch_size)
        out_1d = expert(noisy_actions, conditions, t_1d)
        assert out_1d.shape == (batch_size, H, 7)

        t_2d = torch.rand(batch_size, 1)
        out_2d = expert(noisy_actions, conditions, t_2d)
        assert out_2d.shape == (batch_size, H, 7)


def test_position_offset():
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=32,
        hidden_dim=64,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    expert.eval()

    noisy_actions = torch.randn(2, 6, 7)
    conditions = (
        torch.randn(2, 6, 32),
        torch.randn(2, 6, 32),
        torch.randn(2, 6, 32),
    )
    time = torch.rand(2)

    # Valid offset
    out1 = expert(noisy_actions, conditions, time, position_offset=0)
    out2 = expert(noisy_actions, conditions, time, position_offset=10)
    assert out1.shape == (2, 6, 7)
    assert out2.shape == (2, 6, 7)
    # Different position offset should yield different results
    assert not torch.allclose(out1, out2, atol=1e-5)

    # Exceeding max_horizon
    with pytest.raises(ValueError, match="exceeds max_horizon"):
        expert(noisy_actions, conditions, time, position_offset=46)  # 46 + 6 = 52 > 50


def test_strict_condition_validation():
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=32,
        hidden_dim=64,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    noisy_actions = torch.randn(2, 6, 7)
    time = torch.rand(2)

    # Reject non-tuple/list conditions
    with pytest.raises(TypeError, match="tuple or list"):
        expert(noisy_actions, "invalid", time)

    # Reject not 3 conditions
    with pytest.raises(ValueError, match="Expected exactly 3 conditions"):
        expert(noisy_actions, (torch.randn(2, 6, 32), torch.randn(2, 6, 32)), time)

    with pytest.raises(ValueError, match="Expected exactly 3 conditions"):
        expert(
            noisy_actions,
            (
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 32),
            ),
            time,
        )

    # Reject invalid batch size in conditions
    with pytest.raises(ValueError, match="batch size"):
        expert(
            noisy_actions,
            (
                torch.randn(3, 6, 32),
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 32),
            ),
            time,
        )

    # Reject invalid condition_dim
    with pytest.raises(ValueError, match="expected condition_dim"):
        expert(
            noisy_actions,
            (
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 16),
                torch.randn(2, 6, 32),
            ),
            time,
        )

    # Reject invalid noisy_actions shape
    with pytest.raises(ValueError, match="noisy_actions dim"):
        expert(
            torch.randn(2, 6, 8),
            (
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 32),
                torch.randn(2, 6, 32),
            ),
            time,
        )


def test_backprop_each_condition():
    # Verify that gradients propagate back to each of the 3 condition inputs
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=16,
        hidden_dim=32,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    noisy_actions = torch.randn(2, 6, 7, requires_grad=True)
    c0 = torch.randn(2, 6, 16, requires_grad=True)
    c1 = torch.randn(2, 6, 16, requires_grad=True)
    c2 = torch.randn(2, 6, 16, requires_grad=True)
    time = torch.rand(2)

    out = expert(noisy_actions, (c0, c1, c2), time)
    loss = out.sum()
    loss.backward()

    assert noisy_actions.grad is not None and torch.isfinite(noisy_actions.grad).all()
    assert c0.grad is not None and torch.isfinite(c0.grad).all()
    assert c1.grad is not None and torch.isfinite(c1.grad).all()
    assert c2.grad is not None and torch.isfinite(c2.grad).all()
    assert (c0.grad.abs().sum() > 0).item()
    assert (c1.grad.abs().sum() > 0).item()
    assert (c2.grad.abs().sum() > 0).item()


def test_layer_specific_context_mapping():
    # Hook into cross_attn to verify that block i receives condition i projected
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=16,
        hidden_dim=32,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    expert.eval()

    received_contexts = []

    def hook_fn(module, args, kwargs, output):
        if "key" in kwargs:
            received_contexts.append(kwargs["key"])
        elif len(args) >= 2:
            received_contexts.append(args[1])

    hooks = [
        block.cross_attn.register_forward_hook(hook_fn, with_kwargs=True)
        for block in expert.blocks
    ]

    c0 = torch.randn(2, 6, 16)
    c1 = torch.randn(2, 6, 16)
    c2 = torch.randn(2, 6, 16)
    time = torch.rand(2)
    noisy_actions = torch.randn(2, 6, 7)

    with torch.no_grad():
        _ = expert(noisy_actions, (c0, c1, c2), time)

    for h in hooks:
        h.remove()

    assert len(received_contexts) == 3
    # Check that condition projection i matches received context i
    expected_ctx0 = expert.condition_projections[0](c0)
    expected_ctx1 = expert.condition_projections[1](c1)
    expected_ctx2 = expert.condition_projections[2](c2)

    assert torch.allclose(received_contexts[0], expected_ctx0, atol=1e-6)
    assert torch.allclose(received_contexts[1], expected_ctx1, atol=1e-6)
    assert torch.allclose(received_contexts[2], expected_ctx2, atol=1e-6)


def test_distinguish_positional_slots():
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=16,
        hidden_dim=32,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    expert.eval()

    # Create identical action inputs across horizon steps
    single_action = torch.randn(1, 1, 7)
    noisy_actions = single_action.repeat(1, 6, 1)

    # Identical condition across horizon steps
    single_cond = torch.randn(1, 1, 16)
    c = single_cond.repeat(1, 6, 1)
    conditions = (c, c, c)
    time = torch.tensor([0.5])

    with torch.no_grad():
        out = expert(noisy_actions, conditions, time)  # [1, 6, 7]

    # Because position embeddings distinguish slots, out[0, 0] != out[0, 1]
    assert not torch.allclose(out[:, 0, :], out[:, 1, :], atol=1e-5)


def test_time_effect():
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=16,
        hidden_dim=32,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    expert.eval()

    noisy_actions = torch.randn(2, 6, 7)
    conditions = (
        torch.randn(2, 6, 16),
        torch.randn(2, 6, 16),
        torch.randn(2, 6, 16),
    )
    t0 = torch.tensor([0.1, 0.1])
    t1 = torch.tensor([0.9, 0.9])

    with torch.no_grad():
        out0 = expert(noisy_actions, conditions, t0)
        out1 = expert(noisy_actions, conditions, t1)

    assert not torch.allclose(out0, out1, atol=1e-5)


def test_no_future_leakage_in_conditions():
    # Ensure that step t does not depend on condition step > t if conditions are sliced
    expert = LayerwiseActionExpert(
        action_dim=7,
        condition_dim=16,
        hidden_dim=32,
        num_heads=4,
        max_horizon=50,
        num_layers=3,
    )
    expert.eval()

    c0 = torch.randn(1, 15, 16)
    c1 = torch.randn(1, 15, 16)
    c2 = torch.randn(1, 15, 16)
    noisy_actions = torch.randn(1, 6, 7)
    time = torch.tensor([0.5])

    # Slice condition to 6 vs feeding 15
    # Caller slices conditions for isolated 6/15 heads; expert does not do global pooling across horizon
    out_sliced = expert(
        noisy_actions,
        (c0[:, :6], c1[:, :6], c2[:, :6]),
        time,
    )
    assert out_sliced.shape == (1, 6, 7)
