"""CPU contracts for action-relevant visual World supervision."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from va_compound.world_supervision import (
    action_top10_gap_loss,
    action_top10_oracle_straight_through_gap_loss,
    action_top10_pairwise_loss,
    action_top10_ranking_loss,
    gated_static_copy_anchor_loss,
    masked_numerator_denominator,
    masked_reduction,
    stage_supervision_weights,
    static_copy_anchor_loss,
    transition_mask,
    visual_no_regression_loss,
    visual_static_relative_penalty,
    visual_static_stage_chain_loss,
    visual_world_loss,
)


def _flat_map(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def test_transition_mask_requires_current_six_and_next_first() -> None:
    valid = torch.ones(4, 3, 8, dtype=torch.bool)
    valid[0, 0, 6:] = False
    valid[1, 0, 5] = False
    valid[2, 1, 0] = False
    valid[3, 1, 5] = False

    expected = torch.tensor(
        [[True, True], [False, True], [False, False], [True, False]]
    )

    torch.testing.assert_close(transition_mask(valid), expected)
    torch.testing.assert_close(transition_mask(valid, time_index=0), expected[:, 0])


def test_single_patch_motion_uses_motion_patch_and_keeps_static_gradient() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0])
    target = _flat_map([2.0, 0.0, 0.0, 0.0, 0.0])
    prediction = _flat_map([0.0, 1.0, 1.0, 1.0, 1.0]).requires_grad_()

    result = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
        motion_clip=4.0,
    )

    expected_motion = torch.tensor([[[True, False, False, False, False]]])
    torch.testing.assert_close(result.topk_mask, expected_motion)
    torch.testing.assert_close(result.top10_mask, expected_motion)
    torch.testing.assert_close(result.static_mask, ~expected_motion)
    assert result.motion_energy_per_sample.item() == pytest.approx(0.8)
    assert result.all_per_sample.item() == pytest.approx(1.6)
    assert result.motion_per_sample.item() == pytest.approx(4.0)
    assert result.topk_per_sample.item() == pytest.approx(4.0)
    assert result.static_per_sample.item() == pytest.approx(1.0)
    assert result.loss_per_sample.item() == pytest.approx(3.4)
    assert float(result.motion_weights.max()) <= 4.0

    result.loss_per_sample.sum().backward()
    assert prediction.grad is not None
    assert torch.all(prediction.grad[..., 1:] > 0.0)


def test_all_static_map_is_finite_and_uses_uniform_motion_weights() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0])
    target = current.clone()
    prediction = _flat_map([2.0, 2.0, 2.0, 2.0])

    result = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.25,
        diagnostic_top10_fraction=0.25,
    )

    torch.testing.assert_close(result.motion_weights, torch.ones(1, 4))
    torch.testing.assert_close(result.motion_energy_per_sample, torch.zeros(1))
    assert int(result.topk_mask.sum()) == 1
    assert int(result.static_mask.sum()) == 3
    for value in (
        result.loss_per_sample,
        result.all_per_sample,
        result.motion_per_sample,
        result.topk_per_sample,
        result.top10_per_sample,
        result.static_per_sample,
    ):
        torch.testing.assert_close(value, torch.full((1,), 4.0))
        assert torch.isfinite(value).all()
    for value in (
        result.copy_all_per_sample,
        result.copy_motion_per_sample,
        result.copy_topk_per_sample,
        result.copy_top10_per_sample,
        result.copy_static_per_sample,
    ):
        torch.testing.assert_close(value, torch.zeros(1))


def test_empty_transition_reduction_returns_graph_connected_zero() -> None:
    values = torch.tensor([2.0, 5.0], requires_grad=True)
    empty = torch.zeros(2, dtype=torch.bool)

    loss = masked_reduction([values], [empty])

    assert loss.requires_grad
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


def test_empty_spatial_mask_returns_graph_connected_zero() -> None:
    prediction = _flat_map([1.0, 2.0]).requires_grad_()
    target = _flat_map([0.0, 0.0])
    current = target.clone()

    result = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=0.5,
    )

    assert not result.static_mask.any()
    assert result.static_per_sample.item() == pytest.approx(0.0)
    result.static_per_sample.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_single_valid_auxiliary_sample_uses_actual_denominator() -> None:
    values = torch.tensor([3.0, 7.0], requires_grad=True)
    mask = torch.tensor([False, True])

    numerator, denominator = masked_numerator_denominator(
        [values], [mask], weights=[0.25]
    )
    loss = masked_reduction([values], [mask], weights=[0.25])

    assert numerator.item() == pytest.approx(1.75)
    assert denominator.item() == pytest.approx(0.25)
    assert loss.item() == pytest.approx(7.0)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.0, 1.0]))


def test_topk_selection_and_normalization_are_isolated_per_sample() -> None:
    current = torch.zeros(2, 1, 1, 5)
    target = torch.tensor(
        [[[[1.0, 0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0, 2.0]]]]
    )
    prediction = torch.zeros_like(target)

    first = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )
    changed_target = target.clone()
    changed_target[1] = torch.tensor([[[0.0, 100.0, 0.0, 0.0, 0.0]]])
    second = visual_world_loss(
        prediction,
        changed_target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )

    assert first.topk_mask[0, 0].tolist() == [True, False, False, False, False]
    assert first.topk_mask[1, 0].tolist() == [False, False, False, False, True]
    assert second.topk_mask[1, 0].tolist() == [False, True, False, False, False]
    torch.testing.assert_close(second.topk_mask[0], first.topk_mask[0])
    torch.testing.assert_close(second.motion_weights[0], first.motion_weights[0])
    torch.testing.assert_close(second.loss_per_sample[0], first.loss_per_sample[0])


def test_copy_metrics_use_the_identical_masks_weights_and_reduction() -> None:
    current = _flat_map([0.5, 0.0, -1.0, 0.0, 2.0])
    target = _flat_map([1.5, 2.0, -1.0, 0.0, -1.0])
    prediction = current.clone()

    result = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.40,
        diagnostic_top10_fraction=0.20,
    )

    metric_pairs = (
        (result.all_per_sample, result.copy_all_per_sample),
        (result.motion_per_sample, result.copy_motion_per_sample),
        (result.topk_per_sample, result.copy_topk_per_sample),
        (result.top10_per_sample, result.copy_top10_per_sample),
        (result.static_per_sample, result.copy_static_per_sample),
    )
    for world, copy in metric_pairs:
        torch.testing.assert_close(world, copy, rtol=0.0, atol=0.0)

    copy_error = (current - target).square().mean(dim=1).flatten(1)
    expected_motion = (
        copy_error * result.motion_weights
    ).sum(dim=1) / result.motion_weights.sum(dim=1)
    expected_topk = copy_error[result.topk_mask.flatten(1)].mean().reshape(1)
    expected_top10 = copy_error[result.top10_mask.flatten(1)].mean().reshape(1)
    expected_static = copy_error[result.static_mask.flatten(1)].mean().reshape(1)
    torch.testing.assert_close(result.copy_motion_per_sample, expected_motion)
    torch.testing.assert_close(result.copy_topk_per_sample, expected_topk)
    torch.testing.assert_close(result.copy_top10_per_sample, expected_top10)
    torch.testing.assert_close(result.copy_static_per_sample, expected_static)


def test_target_permutation_changes_only_loss_side_oracles_and_stops_grad() -> None:
    observed = _flat_map([2.0, 0.0, 0.0, 0.0]).requires_grad_()
    prediction = observed * 1.0
    prediction_before = prediction.detach().clone()
    current = _flat_map([0.0, 0.0, 0.0, 0.0]).requires_grad_()
    target = _flat_map([2.0, 0.0, 0.0, 0.0]).requires_grad_()
    permuted_target = _flat_map([0.0, 0.0, 0.0, 2.0]).requires_grad_()

    original = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.25,
        diagnostic_top10_fraction=0.25,
    )
    permuted = visual_world_loss(
        prediction,
        permuted_target,
        current,
        topk_fraction=0.25,
        diagnostic_top10_fraction=0.25,
    )

    torch.testing.assert_close(prediction, prediction_before, rtol=0.0, atol=0.0)
    assert not torch.equal(original.topk_mask, permuted.topk_mask)
    assert original.loss_per_sample.item() == pytest.approx(0.0)
    assert permuted.loss_per_sample.item() > original.loss_per_sample.item()
    assert not original.motion_weights.requires_grad
    assert not original.topk_mask.requires_grad
    assert not permuted.top10_mask.requires_grad

    (original.loss_per_sample.sum() + permuted.loss_per_sample.sum()).backward()
    assert observed.grad is not None
    assert torch.count_nonzero(observed.grad) > 0
    assert target.grad is None
    assert permuted_target.grad is None
    assert current.grad is None


def test_visual_no_regression_hinges_are_per_sample_and_copy_detached() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0]).requires_grad_()
    target = _flat_map([2.0, 0.0, 0.0, 0.0, 0.0]).requires_grad_()
    prediction = _flat_map([0.0, 1.0, 1.0, 1.0, 1.0]).requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )

    guard = visual_no_regression_loss(visual)

    assert guard.all_hinge_per_sample.item() == pytest.approx(0.8)
    assert guard.static_hinge_per_sample.item() == pytest.approx(1.0)
    assert guard.loss_per_sample.item() == pytest.approx(0.9)
    # The v8 base objective remains exactly 0.25 all + 0.25 motion + 0.50 top20.
    assert visual.loss_per_sample.item() == pytest.approx(3.4)

    guard.loss_per_sample.sum().backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) > 0
    assert target.grad is None
    assert current.grad is None


def test_visual_no_regression_hinges_are_zero_at_copy_baseline() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0])
    target = _flat_map([2.0, 0.0, 0.0, 0.0])
    prediction = current.clone().requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.25,
        diagnostic_top10_fraction=0.25,
    )

    guard = visual_no_regression_loss(visual)

    torch.testing.assert_close(guard.loss_per_sample, torch.zeros(1))
    guard.loss_per_sample.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_static_relative_penalty_matches_gate_and_stops_oracle_gradients() -> None:
    current = torch.zeros(2, 1, 1, 5, requires_grad=True)
    target = torch.tensor(
        [
            [[[2.0, 1.0, 1.0, 1.0, 1.0]]],
            [[[2.0, 1.0, 1.0, 1.0, 1.0]]],
        ],
        requires_grad=True,
    )
    prediction = torch.tensor(
        [
            [[[2.0, 2.0, 2.0, 2.0, 2.0]]],
            [[[2.0, 2.1180339, 2.1180339, 2.1180339, 2.1180339]]],
        ],
        requires_grad=True,
    )
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )

    penalty = visual_static_relative_penalty(visual)

    torch.testing.assert_close(penalty.hinge_per_sample, torch.tensor([0.0, 0.2]))
    assert penalty.loss_per_sample[0].item() == pytest.approx(0.0)
    assert penalty.loss_per_sample[1].item() == pytest.approx(0.22, rel=1e-5)
    penalty.loss_per_sample.sum().backward()
    torch.testing.assert_close(prediction.grad[0], torch.zeros_like(prediction.grad[0]))
    assert torch.count_nonzero(prediction.grad[1]) == 4
    assert target.grad is None
    assert current.grad is None


def test_static_relative_penalty_is_finite_at_zero_copy_and_empty_static() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0])
    target = current.clone()
    prediction = _flat_map([0.01, 0.01, 0.01, 0.01, 0.01]).requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )
    penalty = visual_static_relative_penalty(visual)
    assert penalty.loss_per_sample.item() == pytest.approx(0.0051, rel=1e-5)
    assert torch.isfinite(penalty.loss_per_sample).all()

    empty_static = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=0.20,
    )
    empty_penalty = visual_static_relative_penalty(empty_static)
    assert empty_penalty.loss_per_sample.item() == pytest.approx(0.0)
    empty_penalty.loss_per_sample.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_static_copy_anchor_gate_is_per_sample_and_detached() -> None:
    current = torch.zeros(2, 1, 1, 5, requires_grad=True)
    target = torch.tensor(
        [
            [[[2.0, 1.0, 1.0, 1.0, 1.0]]],
            [[[2.0, 1.0, 1.0, 1.0, 1.0]]],
        ],
        requires_grad=True,
    )
    prediction = torch.tensor(
        [
            [[[0.0, 0.5, 0.5, 0.5, 0.5]]],
            [[[0.0, 3.0, 3.0, 3.0, 3.0]]],
        ],
        requires_grad=True,
    )
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )

    anchor = gated_static_copy_anchor_loss(prediction, current, visual)

    torch.testing.assert_close(anchor.anchor_per_sample, torch.tensor([0.25, 9.0]))
    torch.testing.assert_close(
        anchor.active_per_sample, torch.tensor([False, True])
    )
    torch.testing.assert_close(anchor.loss_per_sample, torch.tensor([0.0, 9.0]))
    assert not anchor.active_per_sample.requires_grad

    anchor.loss_per_sample.sum().backward()
    torch.testing.assert_close(prediction.grad[0], torch.zeros_like(prediction.grad[0]))
    assert torch.count_nonzero(prediction.grad[1]) == 4
    assert target.grad is None
    assert current.grad is None


def test_static_copy_anchor_has_zero_top20_gradient() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0]).requires_grad_()
    target = _flat_map([2.0, 0.0, 0.0, 0.0, 0.0]).requires_grad_()
    prediction = _flat_map([4.0, 1.0, 2.0, 3.0, 4.0]).requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )

    anchor = gated_static_copy_anchor_loss(prediction, current, visual)

    assert anchor.active_per_sample.item()
    assert anchor.anchor_per_sample.item() == pytest.approx(7.5)
    anchor.loss_per_sample.sum().backward()
    torch.testing.assert_close(
        prediction.grad[..., 0], torch.zeros_like(prediction.grad[..., 0])
    )
    assert torch.all(prediction.grad[..., 1:] > 0.0)
    assert target.grad is None
    assert current.grad is None


def test_static_copy_anchor_empty_mask_is_graph_connected_zero() -> None:
    prediction = _flat_map([1.0, 2.0]).requires_grad_()
    target = _flat_map([0.0, 0.0])
    current = target.clone().requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=0.5,
    )

    anchor = gated_static_copy_anchor_loss(prediction, current, visual)

    assert not visual.static_mask.any()
    assert anchor.loss_per_sample.requires_grad
    torch.testing.assert_close(anchor.anchor_per_sample, torch.zeros(1))
    torch.testing.assert_close(anchor.loss_per_sample, torch.zeros(1))
    anchor.loss_per_sample.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
    assert current.grad is None


def test_static_copy_anchor_is_always_on_and_excludes_top20() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0]).requires_grad_()
    prediction = _flat_map([5.0, 1.0, 2.0, 3.0, 4.0]).requires_grad_()
    static_mask = torch.tensor([[[False, True, True, True, True]]])

    anchor = static_copy_anchor_loss(prediction, current, static_mask)

    assert anchor.item() == pytest.approx(7.5)
    anchor.sum().backward()
    torch.testing.assert_close(
        prediction.grad[..., 0], torch.zeros_like(prediction.grad[..., 0])
    )
    assert torch.all(prediction.grad[..., 1:] > 0.0)
    assert current.grad is None


def test_static_copy_anchor_empty_mask_is_connected_zero() -> None:
    prediction = _flat_map([1.0, 2.0]).requires_grad_()
    current = torch.zeros_like(prediction).requires_grad_()
    anchor = static_copy_anchor_loss(
        prediction,
        current,
        torch.zeros(1, 1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(anchor, torch.zeros(1))
    anchor.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
    assert current.grad is None


def test_static_relative_penalty_uses_copy_gate_units_and_detaches_oracles() -> None:
    target = _flat_map([0.0, 0.0]).requires_grad_()
    current = _flat_map([1.0, 2.0]).requires_grad_()
    prediction = _flat_map([2.0**0.5, 0.0]).requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.5,
        diagnostic_top10_fraction=0.5,
    )

    penalty = visual_static_relative_penalty(
        visual, static_copy_ratio=1.05, eps=1e-6
    )

    assert penalty.hinge_per_sample.item() == pytest.approx(0.95, abs=1e-6)
    assert penalty.normalized_quadratic_per_sample.item() == pytest.approx(
        0.5 * 0.95**2 / (1.0 + 1e-6), abs=1e-6
    )
    penalty.loss_per_sample.sum().backward()
    assert prediction.grad[..., 0].abs().item() > 0.0
    torch.testing.assert_close(
        prediction.grad[..., 1], torch.zeros_like(prediction.grad[..., 1])
    )
    assert target.grad is None
    assert current.grad is None


def test_static_stage_chain_gates_copy_and_each_worsening_refinement() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0])
    target = _flat_map([2.0, 1.0, 1.0, 1.0, 1.0])

    def visual(static_error: float):
        prediction = _flat_map(
            [2.0] + [1.0 + static_error**0.5] * 4
        )
        return visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=0.20,
            diagnostic_top10_fraction=0.20,
        )

    stages = [visual(error) for error in (1.00, 1.02, 1.20, 1.30)]
    penalties = [visual_static_stage_chain_loss(stages[0])]
    penalties.extend(
        visual_static_stage_chain_loss(stage, previous.static_per_sample)
        for previous, stage in zip(stages[:-1], stages[1:], strict=True)
    )

    torch.testing.assert_close(
        torch.stack([penalty.boundary_per_sample for penalty in penalties]).flatten(),
        torch.tensor([1.05, 1.00, 1.02, 1.05]),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.stack([penalty.loss_per_sample for penalty in penalties]).flatten(),
        torch.tensor([0.00, 0.02, 0.18, 0.25]),
        rtol=0.0,
        atol=1e-6,
    )


def test_static_stage_chain_detaches_copy_and_previous_stage_boundaries() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0]).requires_grad_()
    target = _flat_map([2.0, 1.0, 1.0, 1.0, 1.0]).requires_grad_()
    prediction = _flat_map([2.0, 3.0, 3.0, 3.0, 3.0]).requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )
    copy_static = torch.tensor([1.0], requires_grad=True)
    previous_static = torch.tensor([0.75], requires_grad=True)
    visual = replace(visual, copy_static_per_sample=copy_static)

    first = visual_static_stage_chain_loss(visual)
    later = visual_static_stage_chain_loss(visual, previous_static)

    torch.testing.assert_close(first.boundary_per_sample, torch.tensor([1.05]))
    torch.testing.assert_close(later.boundary_per_sample, torch.tensor([0.75]))
    (first.loss_per_sample + later.loss_per_sample).sum().backward()
    assert torch.count_nonzero(prediction.grad) == 4
    assert copy_static.grad is None
    assert previous_static.grad is None
    assert target.grad is None
    assert current.grad is None


def test_static_stage_chain_empty_transition_is_graph_connected_zero() -> None:
    current = _flat_map([0.0, 0.0, 0.0, 0.0, 0.0])
    target = _flat_map([2.0, 1.0, 1.0, 1.0, 1.0])
    prediction = _flat_map([2.0, 3.0, 3.0, 3.0, 3.0]).requires_grad_()
    visual = visual_world_loss(
        prediction,
        target,
        current,
        topk_fraction=0.20,
        diagnostic_top10_fraction=0.20,
    )
    penalty = visual_static_stage_chain_loss(visual)

    loss = masked_reduction(
        [penalty.loss_per_sample], [torch.zeros(1, dtype=torch.bool)]
    )

    assert loss.requires_grad
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_static_stage_chain_sums_stages_before_single_valid_sample_reduction() -> None:
    current = torch.zeros(2, 1, 1, 5)
    target = torch.tensor(
        [
            [[[2.0, 1.0, 1.0, 1.0, 1.0]]],
            [[[2.0, 1.0, 1.0, 1.0, 1.0]]],
        ]
    )

    def visual(static_errors: tuple[float, float]):
        prediction = torch.tensor(
            [
                [[2.0] + [1.0 + static_errors[0] ** 0.5] * 4],
                [[2.0] + [1.0 + static_errors[1] ** 0.5] * 4],
            ]
        ).unsqueeze(2).requires_grad_()
        return prediction, visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=0.20,
            diagnostic_top10_fraction=0.20,
        )

    early_prediction, early = visual((0.80, 1.20))
    final_prediction, final = visual((0.90, 1.50))
    early_penalty = visual_static_stage_chain_loss(early)
    final_penalty = visual_static_stage_chain_loss(
        final, early.static_per_sample
    )
    summed_per_sample = early_penalty.loss_per_sample + final_penalty.loss_per_sample
    valid = torch.tensor([False, True])

    numerator, denominator = masked_numerator_denominator(
        [summed_per_sample], [valid]
    )
    loss = masked_reduction([summed_per_sample], [valid])

    assert numerator.item() == pytest.approx(0.60, abs=1e-6)
    assert denominator.item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(0.60, abs=1e-6)
    loss.backward()
    torch.testing.assert_close(
        early_prediction.grad[0], torch.zeros_like(early_prediction.grad[0])
    )
    torch.testing.assert_close(
        final_prediction.grad[0], torch.zeros_like(final_prediction.grad[0])
    )
    assert torch.count_nonzero(early_prediction.grad[1]) == 4
    assert torch.count_nonzero(final_prediction.grad[1]) == 4


def test_action_top10_ranking_matches_both_5pct_and_either_10pct_gates() -> None:
    target = _flat_map([0.0])
    current = target.clone()

    def visual(error: float):
        prediction = _flat_map([error**0.5])
        return visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    passing = action_top10_ranking_loss(visual(1.0), visual(1.05), visual(1.10))
    torch.testing.assert_close(
        passing.loss_per_sample, torch.zeros(1), rtol=0.0, atol=1e-6
    )

    no_strong_ablation = action_top10_ranking_loss(
        visual(1.0), visual(1.06), visual(1.08)
    )
    torch.testing.assert_close(
        no_strong_ablation.shuffle_hinge_per_sample, torch.zeros(1), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        no_strong_ablation.zero_hinge_per_sample, torch.zeros(1), atol=1e-6, rtol=0.0
    )
    assert no_strong_ablation.either_strong_hinge_per_sample.item() == pytest.approx(
        0.02, abs=1e-6
    )

    one_weak_ablation = action_top10_ranking_loss(
        visual(1.0), visual(1.04), visual(1.12)
    )
    assert one_weak_ablation.shuffle_hinge_per_sample.item() == pytest.approx(
        0.01, abs=1e-6
    )
    torch.testing.assert_close(
        one_weak_ablation.zero_hinge_per_sample, torch.zeros(1), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        one_weak_ablation.either_strong_hinge_per_sample,
        torch.zeros(1),
        atol=1e-6,
        rtol=0.0,
    )


def test_action_top10_pairwise_loss_passes_at_relative_margin() -> None:
    target = _flat_map([0.0])
    current = target.clone()

    def visual(error: float):
        return visual_world_loss(
            _flat_map([error**0.5]),
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    loss = action_top10_pairwise_loss(
        visual(1.0), visual(1.10), minimum_relative_degradation=0.10
    )

    torch.testing.assert_close(loss, torch.zeros(1), rtol=0.0, atol=1e-6)


def test_action_top10_pairwise_loss_reports_margin_shortfall() -> None:
    target = _flat_map([0.0])
    current = target.clone()

    def visual(error: float):
        return visual_world_loss(
            _flat_map([error**0.5]),
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    loss = action_top10_pairwise_loss(
        visual(1.0), visual(1.04), minimum_relative_degradation=0.10
    )

    assert loss.item() == pytest.approx(0.06, abs=1e-6)


def test_action_top10_pairwise_loss_has_expected_gradients() -> None:
    target = _flat_map([0.0]).requires_grad_()
    current = target.detach().clone().requires_grad_()
    real_prediction = _flat_map([1.0]).requires_grad_()
    counterfactual_prediction = _flat_map([1.0]).requires_grad_()

    def visual(prediction: torch.Tensor):
        return visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    loss = action_top10_pairwise_loss(
        visual(real_prediction),
        visual(counterfactual_prediction),
        minimum_relative_degradation=0.10,
    )
    loss.sum().backward()

    assert torch.all(real_prediction.grad > 0.0)
    assert torch.all(counterfactual_prediction.grad < 0.0)
    assert target.grad is None
    assert current.grad is None


def test_action_top10_gap_loss_keeps_margin_value_and_cancels_shared_gradient() -> None:
    target = _flat_map([0.0]).requires_grad_()
    current = target.detach().clone().requires_grad_()
    shared = torch.tensor(1.0, requires_grad=True)
    action_delta = torch.tensor(0.0, requires_grad=True)

    real = visual_world_loss(
        _flat_map([1.0]) * shared,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=1.0,
    )
    shuffled = visual_world_loss(
        _flat_map([1.0]) * shared + action_delta,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=1.0,
    )
    gap = action_top10_gap_loss(
        real, shuffled, minimum_relative_degradation=0.05
    )

    assert gap.loss_per_sample.item() == pytest.approx(0.05, abs=1e-6)
    gap.loss_per_sample.sum().backward()
    assert shared.grad.item() == pytest.approx(0.0, abs=1e-6)
    assert action_delta.grad.item() == pytest.approx(-2.0, abs=1e-6)
    assert target.grad is None
    assert current.grad is None


def test_action_top10_gap_loss_is_zero_after_relative_margin() -> None:
    target = _flat_map([0.0])
    current = target.clone()
    real = visual_world_loss(
        _flat_map([1.0]), target, current,
        topk_fraction=1.0, diagnostic_top10_fraction=1.0,
    )
    shuffled = visual_world_loss(
        _flat_map([1.05**0.5]), target, current,
        topk_fraction=1.0, diagnostic_top10_fraction=1.0,
    )

    gap = action_top10_gap_loss(
        real, shuffled, minimum_relative_degradation=0.05
    )

    assert gap.loss_per_sample.item() == pytest.approx(0.0, abs=1e-6)
    assert gap.error_gap_per_sample.item() == pytest.approx(0.05, abs=1e-6)


def test_action_oracle_straight_through_keeps_exact_value_and_motion_gradient() -> None:
    current = _flat_map([0.0, 0.0]).requires_grad_()
    target = _flat_map([2.0, 1.0]).requires_grad_()
    shared = _flat_map([0.5, 0.5]).requires_grad_()
    action_delta = torch.zeros_like(shared, requires_grad=True)
    real_prediction = shared
    shuffled_prediction = shared + action_delta
    real = visual_world_loss(
        real_prediction,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=1.0,
    )
    shuffled = visual_world_loss(
        shuffled_prediction,
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=1.0,
    )

    gap = action_top10_oracle_straight_through_gap_loss(
        real,
        shuffled,
        real_prediction,
        shuffled_prediction,
        target,
        current,
        minimum_relative_degradation=0.12,
    )

    expected = 0.12 * real.top10_per_sample.detach()
    torch.testing.assert_close(gap.loss_per_sample, expected)
    torch.testing.assert_close(gap.error_gap_per_sample, torch.zeros(1))
    gap.loss_per_sample.sum().backward()
    torch.testing.assert_close(shared.grad, torch.zeros_like(shared))
    torch.testing.assert_close(action_delta.grad, 2.0 * (target.detach() - current.detach()) / 2.0)
    assert target.grad is None
    assert current.grad is None


def test_action_top10_pairwise_loss_validates_margin_and_batch_shape() -> None:
    target = _flat_map([0.0])
    current = target.clone()
    single = visual_world_loss(
        _flat_map([1.0]),
        target,
        current,
        topk_fraction=1.0,
        diagnostic_top10_fraction=1.0,
    )
    batched_target = torch.zeros(2, 1, 1, 1)
    batched = visual_world_loss(
        torch.ones_like(batched_target),
        batched_target,
        batched_target,
        topk_fraction=1.0,
        diagnostic_top10_fraction=1.0,
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        action_top10_pairwise_loss(
            single, single, minimum_relative_degradation=-0.01
        )
    with pytest.raises(ValueError, match="per-sample shape mismatch"):
        action_top10_pairwise_loss(single, batched)


def test_action_top10_ranking_has_expected_gradients_and_detached_target() -> None:
    target = _flat_map([0.0]).requires_grad_()
    current = target.detach().clone().requires_grad_()
    real_prediction = _flat_map([1.0]).requires_grad_()
    shuffled_prediction = _flat_map([1.0]).requires_grad_()
    zero_prediction = _flat_map([1.12**0.5]).requires_grad_()

    def visual(prediction: torch.Tensor):
        return visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    ranking = action_top10_ranking_loss(
        visual(real_prediction), visual(shuffled_prediction), visual(zero_prediction)
    )
    ranking.loss_per_sample.sum().backward()

    assert torch.all(real_prediction.grad > 0.0)
    assert torch.all(shuffled_prediction.grad < 0.0)
    torch.testing.assert_close(zero_prediction.grad, torch.zeros_like(zero_prediction))
    assert target.grad is None
    assert current.grad is None


def test_action_top10_ranking_detached_margin_only_pushes_wrong_actions_away() -> None:
    target = _flat_map([0.0]).requires_grad_()
    current = target.detach().clone().requires_grad_()
    real_prediction = _flat_map([1.0]).requires_grad_()
    shuffled_prediction = _flat_map([1.0]).requires_grad_()
    zero_prediction = _flat_map([1.0]).requires_grad_()

    def visual(prediction: torch.Tensor):
        return visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    ranking = action_top10_ranking_loss(
        visual(real_prediction),
        visual(shuffled_prediction),
        visual(zero_prediction),
        detach_real_margin=True,
    )
    ranking.loss_per_sample.sum().backward()

    assert real_prediction.grad is None
    assert torch.all(shuffled_prediction.grad < 0.0)
    assert torch.all(zero_prediction.grad < 0.0)
    assert target.grad is None
    assert current.grad is None


def test_action_top10_ranking_default_keeps_legacy_real_margin_gradient() -> None:
    target = _flat_map([0.0])
    current = target.clone()

    def visual(error: float):
        return visual_world_loss(
            _flat_map([error**0.5]),
            target,
            current,
            topk_fraction=1.0,
            diagnostic_top10_fraction=1.0,
        )

    default = action_top10_ranking_loss(
        visual(1.0), visual(1.0), visual(1.12)
    )
    explicit_legacy = action_top10_ranking_loss(
        visual(1.0),
        visual(1.0),
        visual(1.12),
        detach_real_margin=False,
    )

    torch.testing.assert_close(
        default.loss_per_sample, explicit_legacy.loss_per_sample
    )
    torch.testing.assert_close(
        default.shuffle_hinge_per_sample,
        explicit_legacy.shuffle_hinge_per_sample,
    )
    torch.testing.assert_close(
        default.zero_hinge_per_sample, explicit_legacy.zero_hinge_per_sample
    )
    torch.testing.assert_close(
        default.either_strong_hinge_per_sample,
        explicit_legacy.either_strong_hinge_per_sample,
    )


def test_v9_penalties_keep_empty_transition_reduction_connected() -> None:
    target = _flat_map([0.0, 0.0])
    current = target.clone()
    real_prediction = _flat_map([1.0, 1.0]).requires_grad_()
    shuffled_prediction = _flat_map([1.0, 1.0]).requires_grad_()
    zero_prediction = _flat_map([1.0, 1.0]).requires_grad_()

    def visual(prediction: torch.Tensor):
        return visual_world_loss(
            prediction,
            target,
            current,
            topk_fraction=0.5,
            diagnostic_top10_fraction=0.5,
        )

    real = visual(real_prediction)
    guard = visual_no_regression_loss(real)
    ranking = action_top10_ranking_loss(
        real, visual(shuffled_prediction), visual(zero_prediction)
    )
    empty = torch.zeros(1, dtype=torch.bool)
    loss = masked_reduction(
        [guard.loss_per_sample, ranking.loss_per_sample], [empty, empty]
    )

    assert loss.requires_grad
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(real_prediction.grad, torch.zeros_like(real_prediction))
    torch.testing.assert_close(
        shuffled_prediction.grad, torch.zeros_like(shuffled_prediction)
    )
    torch.testing.assert_close(zero_prediction.grad, torch.zeros_like(zero_prediction))


def test_stage_weights_decay_toward_the_final_refinement() -> None:
    assert stage_supervision_weights(4, floor=0.0) == (0.25**3, 0.25**2, 0.25, 1.0)
    assert stage_supervision_weights(4) == (0.1, 0.1, 0.25, 1.0)
    assert stage_supervision_weights(8)[0] == 0.1
    assert stage_supervision_weights(8)[-1] == 1.0
    assert stage_supervision_weights(1) == (1.0,)
    assert stage_supervision_weights(0) == ()


def test_stage_reduction_uses_one_shared_numerator_and_denominator() -> None:
    early = torch.tensor([2.0, 100.0], requires_grad=True)
    final = torch.tensor([4.0, 8.0], requires_grad=True)
    early_mask = torch.tensor([True, False])
    final_mask = torch.tensor([True, True])
    weights = stage_supervision_weights(2)

    numerator, denominator = masked_numerator_denominator(
        [early, final], [early_mask, final_mask], weights
    )
    loss = masked_reduction([early, final], [early_mask, final_mask], weights)

    assert numerator.item() == pytest.approx(12.5)
    assert denominator.item() == pytest.approx(2.25)
    assert loss.item() == pytest.approx(12.5 / 2.25)
    loss.backward()
    torch.testing.assert_close(
        early.grad, torch.tensor([0.25 / 2.25, 0.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        final.grad, torch.tensor([1.0 / 2.25, 1.0 / 2.25]), rtol=1e-6, atol=1e-6
    )
