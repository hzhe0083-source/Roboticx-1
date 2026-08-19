"""CPU tests for masked/prefix-weighted flow supervision."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from train import (
    effective_action_valid_fraction,
    feature_no_grad_decode_autocast,
    feature_policy_autocast,
    masked_flow_matching_loss,
    parse_args,
    validate_args,
)
from va_compound.model import FlowMatchingHead


def test_default_flow_loss_is_exact_legacy_mse() -> None:
    torch.manual_seed(3)
    predicted = torch.randn(2, 4, 48, 4)
    target = torch.randn_like(predicted)

    loss, prefix, tail = masked_flow_matching_loss(predicted, target)

    assert torch.equal(loss, F.mse_loss(predicted, target))
    torch.testing.assert_close(prefix, F.mse_loss(predicted[..., :6, :], target[..., :6, :]))
    torch.testing.assert_close(tail, F.mse_loss(predicted[..., 6:, :], target[..., 6:, :]))


def test_action_valid_mask_and_tail_weight_use_element_normalization() -> None:
    predicted = torch.zeros(1, 1, 8, 1)
    target = torch.tensor([[[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]]])
    valid = torch.tensor([[[1, 1, 1, 0, 0, 0, 1, 1]]], dtype=torch.bool)

    loss, prefix, tail = masked_flow_matching_loss(
        predicted,
        target,
        {"action_valid_mask": valid},
        prefix_steps=6,
        prefix_weight=1.0,
        tail_weight=0.1,
    )

    assert prefix.item() == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)
    assert tail.item() == pytest.approx((49.0 + 64.0) / 2.0)
    expected = (1.0 + 4.0 + 9.0 + 0.1 * (49.0 + 64.0)) / (3.0 + 0.2)
    assert loss.item() == pytest.approx(expected)


def test_horizon_mask_broadcasts_across_batch_sequence_and_action() -> None:
    predicted = torch.zeros(2, 3, 4, 2)
    target = torch.ones_like(predicted)
    horizon_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss, prefix, tail = masked_flow_matching_loss(
        predicted,
        target,
        {"horizon_mask": horizon_mask},
        prefix_steps=2,
    )

    assert loss.item() == pytest.approx(1.0)
    assert prefix.item() == pytest.approx(1.0)
    assert tail.item() == pytest.approx(1.0)


def test_effective_action_valid_fraction_combines_masks() -> None:
    reference = torch.zeros(1, 1, 4, 2)
    batch = {
        "action_valid_mask": torch.tensor([[[True, True, False, False]]]),
        "horizon_mask": torch.tensor([True, False, True, False]),
    }
    # Only horizon element 0 survives both masks; action_dim is broadcast.
    assert effective_action_valid_fraction(batch, reference).item() == pytest.approx(0.25)


def test_flow_loss_rejects_zero_valid_supervision() -> None:
    predicted = torch.zeros(1, 1, 4, 2)
    with pytest.raises(ValueError, match="zero valid"):
        masked_flow_matching_loss(
            predicted,
            torch.ones_like(predicted),
            {"horizon_mask": torch.zeros(4)},
        )


def test_flow_weight_argument_contract() -> None:
    args = parse_args(["--flow-prefix-weight", "1", "--flow-tail-weight", "0.1"])
    validate_args(args)

    args.flow_tail_weight = -0.1
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_args(args)


def test_feature_autocast_keeps_flow_trainable_after_no_grad_decode() -> None:
    torch.manual_seed(11)
    head = FlowMatchingHead(
        hidden_dim=64,
        action_dim=4,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        flow_cond="adaln",
    )
    condition = torch.randn(3, 12, 64)
    noisy_actions = torch.randn(3, 12, 4)
    flow_time = torch.rand(3)
    target = torch.randn(3, 12, 4)

    with feature_policy_autocast(torch.device("cpu"), enabled=True):
        with feature_no_grad_decode_autocast(
            torch.device("cpu"), enabled=True
        ):
            with torch.no_grad():
                head(condition, noisy_actions, flow_time)
        prediction = head(condition, noisy_actions, flow_time)
        loss = F.mse_loss(prediction.float(), target)
    loss.backward()

    for parameter in (
        head.action_projection.weight,
        head.velocity_head.weight,
        head.ada_mlps[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0
