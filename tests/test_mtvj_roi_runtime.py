from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from va_compound.metric_roi import (
    crop_metric_roi_video,
    load_metric_roi_checkpoint,
    metric_head_state_sha256,
    metric_roi_checkpoint_identity,
    refine_metric_roi_positions,
)
from va_compound.metric_visual_head import LanguageMetricField


def _roi_artifact(path, coarse_sha: str = "coarse-sha") -> dict:
    ctor = {
        "lang_dim": 8,
        "h_dim": 4,
        "d_proj": 2,
        "n_roles": 4,
        "l2_norm": False,
        "learnable_temp": False,
        "temp_init": 10.0,
        "freeze_bias": False,
        "mode_readout": False,
    }
    head = LanguageMetricField(**ctor)
    coarse_head_state_sha = metric_head_state_sha256(head)
    config = {
        "training_state_version": 1,
        "steps_done": 10,
        "canonical_image_size": 384,
        "head_ctor": ctor,
        "role_pairs": [[0, 1], [3, 2]],
        "alpha_default": 0.0,
        "eval_alpha": 1.0,
        "min_roi_size": 96.0,
        "max_roi_size": 192.0,
        "distance_scale": 2.0,
        "max_delta_px": 32.0,
        "coarse_sha256": coarse_sha,
        "coarse_head_state_sha256": coarse_head_state_sha,
    }
    artifact = {
        "contract": "mt_vj_metric_roi_v1",
        "config": config,
        "coarse": {
            "sha256": coarse_sha,
            "coarse_head_state_sha256": coarse_head_state_sha,
            "contract": "mt_vj_metric_field_v1",
            "config": {},
        },
        "roi_metric_head": head.state_dict(),
    }
    torch.save(artifact, path)
    return artifact


def _coarse_identity(sha: str = "coarse-sha") -> dict:
    return {
        "sha256": sha,
        "size_bytes": 1,
        "contract": "mt_vj_metric_field_v1",
    }


def test_roi_loader_is_frozen_and_strictly_roundtrips_policy_state(tmp_path) -> None:
    path = tmp_path / "roi.pt"
    external = _roi_artifact(path)
    head = load_metric_roi_checkpoint(
        path,
        torch.device("cpu"),
        coarse_identity=_coarse_identity(),
        coarse_head_state_sha256=external["config"]["coarse_head_state_sha256"],
    )
    assert not head.training
    assert all(not parameter.requires_grad for parameter in head.parameters())

    identity = metric_roi_checkpoint_identity(path, external)
    restored = load_metric_roi_checkpoint(
        path,
        torch.device("cpu"),
        coarse_identity=_coarse_identity(),
        coarse_head_state_sha256=external["config"]["coarse_head_state_sha256"],
        policy_state=head.state_dict(),
        policy_config=external["config"],
        policy_identity=identity,
        policy_training_contract={"mtvj_roi_enabled": True},
    )
    for key, value in head.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    with pytest.raises(ValueError, match="coarse SHA"):
        load_metric_roi_checkpoint(
            path,
            torch.device("cpu"),
            coarse_identity=_coarse_identity("wrong"),
            coarse_head_state_sha256=external["config"]["coarse_head_state_sha256"],
        )

    changed_coarse = LanguageMetricField(**external["config"]["head_ctor"])
    changed_coarse.load_state_dict(external["roi_metric_head"], strict=True)
    with torch.no_grad():
        next(changed_coarse.parameters()).view(-1)[0].add_(1.0)
    with pytest.raises(ValueError, match="coarse head state SHA"):
        load_metric_roi_checkpoint(
            path,
            torch.device("cpu"),
            # External coarse file identity is unchanged, only the actual
            # policy-embedded metric-head weight differs.
            coarse_identity=_coarse_identity(),
            coarse_head_state_sha256=metric_head_state_sha256(changed_coarse),
        )


def test_roi_alpha_zero_is_exact_and_skips_every_extra_forward() -> None:
    class MustNotRun(nn.Module):
        def forward_hierarchical_dense(self, *args, **kwargs):
            raise AssertionError("alpha=0 must skip ROI V-JEPA")

    coarse = torch.randn(2, 4, 2)
    visibility = torch.rand(2, 4)
    final, final_visibility = refine_metric_roi_positions(
        coarse,
        visibility,
        torch.empty(2, 4, 3, 480, 480),
        MustNotRun(),
        MustNotRun(),
        torch.empty(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
        torch.empty(1152, 3),
        alpha=0.0,
    )
    assert final is coarse
    assert final_visibility is visibility
    assert torch.equal(final, coarse)


def test_canonical_384_roi_is_scaled_before_raw_480_crop() -> None:
    x = torch.linspace(0.0, 1.0, 480).view(1, 1, 1, 1, 480)
    raw = x.expand(1, 4, 3, 480, 480).contiguous()
    canonical_roi = torch.tensor([[192.0, 192.0, 96.0]])
    crop = crop_metric_roi_video(
        raw, canonical_roi, canonical_image_size=384
    )

    # 96 canonical pixels become 120 raw pixels: left edge is x=180, not 192.
    assert crop.shape == (1, 4, 3, 384, 384)
    assert crop[0, 0, 0, 192, 0].item() == pytest.approx(180.0 / 479.0, abs=0.003)
    assert crop[0, 0, 0, 192, 192].item() == pytest.approx(240.0 / 479.0, abs=0.003)


def test_dino_480_geometry_keeps_96_source_pixels_before_224_resize() -> None:
    x = torch.linspace(0.0, 1.0, 480).view(1, 1, 1, 1, 480)
    raw = x.expand(1, 2, 3, 480, 480).contiguous()
    roi_480 = torch.tensor([[240.0, 240.0, 96.0]])
    crop = crop_metric_roi_video(
        raw,
        roi_480,
        canonical_image_size=224,
        roi_geometry_size=480,
    )
    assert crop.shape == (1, 2, 3, 224, 224)
    # Native 480 geometry: 96 source pixels span approximately x=192..288.
    assert crop[0, 0, 0, 112, 0].item() == pytest.approx(192.0 / 479.0, abs=0.004)
    assert crop[0, 0, 0, 112, 112].item() == pytest.approx(240.0 / 479.0, abs=0.004)
