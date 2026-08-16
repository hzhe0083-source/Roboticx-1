from __future__ import annotations

import pytest
import torch

from pathlib import Path

from scripts.validate_task35_fm_checkpoint import (
    EXPECTED_DATA_SHA256,
    EXPECTED_DINO_WEIGHT_SHA256,
    EXPECTED_FEATURE_SHA256,
    EXPECTED_RAW_FRAMES_SHA256,
    EXPECTED_ROI_SHA256,
    validate_task35_fm_checkpoint,
)
from va_compound.metric_roi import DINO_METRIC_ROI_CONTRACT, TASK35_METRIC_ROLE_CONTRACT


def _checkpoint(step: int = 1000, **overrides) -> dict:
    payload = {
        "config": {
            "action_horizon": 6,
            "direct_head": False,
            "main_vision_backbone": "dinov2_vitl14_reg4",
            "main_vision_grid": 16,
            "main_vision_frames": 4,
            "main_vision_tokens": 1024,
            "main_vision_temporal": True,
            "metric_geometry_inject": True,
            "dino_dense_metric": True,
            "dense_readout_mtvj": True,
            "va_attention_backend": "auto",
            "hidden_dim": 512,
        },
        "model": {
            "main_vision_frame_embedding.weight": torch.randn(4, 1024),
            "geometry_projection.weight": torch.randn(512, 8),
            "geometry_projection.bias": torch.randn(512),
        },
        "training_contract": {
            "action_decoder": "conditional_flow_matching",
            "wam_enabled": False,
            "task35_precision_contract": True,
            "task35_metric_role_contract": TASK35_METRIC_ROLE_CONTRACT,
            "dino_roi_contract": DINO_METRIC_ROI_CONTRACT,
            "dino_roi_enabled": True,
            "dino_roi_alpha": 1.0,
            "metric_state_source": "p_times_visibility_flat",
            "metric_contract_version": 3,
            "task35_data_sha256": EXPECTED_DATA_SHA256,
            "task35_raw_frames_sha256": EXPECTED_RAW_FRAMES_SHA256,
            "task35_dino_feature_sha256": dict(EXPECTED_FEATURE_SHA256),
            "main_vision_checkpoint_sha256": EXPECTED_DINO_WEIGHT_SHA256,
            "mtvj_visual_aux_every": 10,
            "mtvj_visual_aux_batch": 8,
        },
        "mtvj_metric_head": {"dummy": torch.zeros(1)},
        "mtvj_metric_head_config": {
            "lang_dim": 2048,
            "h_dim": 1024,
            "d_proj": 192,
            "n_roles": 4,
            "l2_norm": True,
            "learnable_temp": True,
            "temp_init": 10.0,
            "freeze_bias": False,
            "mode_readout": True,
            "grid": 16,
        },
        "mtvj_relation_encoder": {"dummy": torch.zeros(1)},
        "dino_roi_checkpoint_identity": {"sha256": EXPECTED_ROI_SHA256},
        "global_step": step,
        "exact_resume_version": 2,
    }
    payload.update(overrides)
    return payload


def test_task35_fm_contract_accepts_complete_checkpoint() -> None:
    report = validate_task35_fm_checkpoint(
        _checkpoint(), expected_step=1000, load_modules=False
    )
    assert report["ok"] is True
    assert report["action_decoder"] == "conditional_flow_matching"
    assert report["geometry_l2"] > 0.0


def test_task35_fm_contract_rejects_direct_and_missing_geometry() -> None:
    direct = _checkpoint()
    direct["training_contract"]["action_decoder"] = "direct_head"
    with pytest.raises(ValueError, match="FM decoder"):
        validate_task35_fm_checkpoint(direct, load_modules=False)

    missing = _checkpoint()
    del missing["model"]["geometry_projection.weight"]
    with pytest.raises(ValueError, match="geometry projection"):
        validate_task35_fm_checkpoint(missing, load_modules=False)


def test_task35_fm_contract_rejects_zero_geometry_after_one_thousand_steps() -> None:
    payload = _checkpoint(step=1000)
    payload["model"]["geometry_projection.weight"] = torch.zeros(512, 8)
    with pytest.raises(ValueError, match="still all zeros"):
        validate_task35_fm_checkpoint(payload, load_modules=False)


def test_validator_and_slice_diag_default_to_no_cuda() -> None:
    root = Path(__file__).resolve().parent.parent
    validate = (root / "scripts" / "validate_task35_fm_checkpoint.py").read_text()
    slices = (root / "scripts" / "diag_task35_clean_recovery_slices.py").read_text()
    posttrain = (root / "scripts" / "run_task35_h6_posttrain_eval.sh").read_text()
    waiter = (root / "scripts" / "wait_validate_task35_fm_milestone.sh").read_text()
    assert 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")' in validate
    assert 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")' in slices
    assert posttrain.count("CUDA_VISIBLE_DEVICES=") >= 3
    assert "CUDA_VISIBLE_DEVICES=" in waiter
