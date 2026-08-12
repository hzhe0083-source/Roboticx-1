import torch

from va_compound.metric_roi import (
    crop_to_full,
    full_to_crop,
    gt_crop_visibility,
    merge_roi_refinement,
    plan_metric_roi,
)


def _coarse():
    p = torch.tensor(
        [
            [[0.20, 0.20], [0.30, 0.30], [0.80, 0.80], [0.70, 0.70]],
            [[0.10, 0.20], [0.60, 0.50], [0.55, 0.65], [0.45, 0.55]],
        ],
        dtype=torch.float32,
    )
    vis = torch.tensor([[0.9, 0.8, 0.2, 0.3], [0.2, 0.2, 0.9, 0.8]])
    return p, vis


def test_plan_selects_visibility_pair_and_dynamic_size():
    p, vis = _coarse()
    selection = plan_metric_roi(p, vis, 384)

    assert selection.pair_index.tolist() == [0, 1]
    assert selection.pair_roles.tolist() == [[0, 1], [3, 2]]
    assert selection.role_mask.tolist() == [
        [True, True, False, False],
        [False, False, True, True],
    ]
    torch.testing.assert_close(selection.confidence, torch.tensor([0.72, 0.72]))
    # First pair spans 38.4 px -> min 96. Second spans 38.4 px -> min 96.
    torch.testing.assert_close(selection.roi[:, 2], torch.tensor([96.0, 96.0]))


def test_dynamic_crop_reaches_max_and_training_jitter_is_bounded():
    p, vis = _coarse()
    selection = plan_metric_roi(
        p,
        vis,
        384,
        training=True,
        center_jitter_px=8.0,
        size_jitter=0.1,
        generator=torch.Generator().manual_seed(7),
    )
    assert ((selection.roi[:, 2] >= 96.0) & (selection.roi[:, 2] <= 192.0)).all()
    half = selection.roi[:, 2] / 2
    assert (selection.roi[:, 0] >= half).all()
    assert (selection.roi[:, 0] <= 384 - half).all()
    assert (selection.roi[:, 1] >= half).all()
    assert (selection.roi[:, 1] <= 384 - half).all()

    far = p[:1].clone()
    far[0, 0], far[0, 1] = torch.tensor([0.1, 0.1]), torch.tensor([0.9, 0.9])
    far_selection = plan_metric_roi(far, vis[:1], 384)
    assert far_selection.roi[0, 2].item() == 192.0


def test_full_crop_coordinate_roundtrip_rectangular_image():
    p, vis = _coarse()
    selection = plan_metric_roi(p, vis, (320, 480), max_size=192)
    local = full_to_crop(p, selection.roi, (320, 480))
    recovered = crop_to_full(local, selection.roi, (320, 480))
    torch.testing.assert_close(recovered, p, atol=1e-6, rtol=0)


def test_gt_visibility_requires_original_visibility_and_inside_crop():
    roi = torch.tensor([[192.0, 192.0, 96.0]])
    gt = torch.tensor([[[0.5, 0.5], [0.4, 0.5], [0.9, 0.9], [0.5, 0.6]]])
    visible = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    crop_vis = gt_crop_visibility(gt, visible, roi, 384)
    assert crop_vis.tolist() == [[1.0, 0.0, 0.0, 1.0]]


def test_alpha_zero_is_exact_noop_and_visibility_is_preserved():
    p, vis = _coarse()
    selection = plan_metric_roi(p, vis, 384)
    wildly_different = torch.tensor(
        [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]
    )
    final, final_vis = merge_roi_refinement(
        p, vis, wildly_different, selection, 384, alpha=0.0
    )
    assert torch.equal(final, p)
    assert final_vis is vis


def test_merge_updates_only_selected_pair_and_clips_pixel_delta():
    p, vis = _coarse()
    selection = plan_metric_roi(p, vis, 384)
    local_coarse = full_to_crop(p, selection.roi, 384)
    batch = torch.arange(p.shape[0])[:, None]
    refined_pair = local_coarse[batch, selection.pair_roles] + 1.0
    final, _ = merge_roi_refinement(
        p, vis, refined_pair, selection, 384, alpha=1.0, max_delta_px=8.0
    )
    delta_px = (final - p) * 384
    assert (delta_px.abs() <= 8.0 + 1e-5).all()
    assert torch.equal(final[0, 2:], p[0, 2:])
    assert torch.equal(final[1, :2], p[1, :2])

