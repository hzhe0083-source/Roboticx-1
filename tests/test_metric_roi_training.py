from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_metric_roi as trainer
import prepare_metaworld_metric as metric_data
from scripts.eval_metric_roi_holdout import (
    adoption_gate,
    paired_metrics,
    selection_diagnostics,
    validate_roi_checkpoint,
)
from va_compound.metric_roi import (
    MetricROI,
    load_metric_roi_checkpoint,
    metric_head_state_sha256,
)
from va_compound.metric_visual_head import LanguageMetricField


def _tiny_head_ctor() -> dict:
    return {
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


def _write_completed_runtime_metric(path):
    ctor = _tiny_head_ctor()
    head = LanguageMetricField(**ctor)
    checkpoint = {
        "contract": trainer.COARSE_CONTRACT,
        "config": {
            **ctor,
            "language_cache_available": True,
            "loc_only": False,
            "tasks": list(trainer.SUPPORTED_TASKS),
            "steps": 10000,
            "steps_done": 10000,
        },
        "metric_head": head.state_dict(),
    }
    torch.save(checkpoint, path)
    return checkpoint, head


def _selection() -> MetricROI:
    return MetricROI(
        roi=torch.tensor([[192.0, 192.0, 192.0]]),
        pair_index=torch.tensor([0]),
        pair_roles=torch.tensor([[0, 1]]),
        role_mask=torch.tensor([[True, True, False, False]]),
        confidence=torch.tensor([1.0]),
    )


def _output(*, corrupt_unselected: bool = False):
    p = torch.full((1, 4, 2), 0.5, requires_grad=True)
    logits = torch.zeros((1, 4), requires_grad=True)
    scores = torch.zeros((1, 4, 1152), requires_grad=True)
    offsets = torch.zeros((1, 4, 1152, 2), requires_grad=True)
    if corrupt_unselected:
        with torch.no_grad():
            p[:, 2:] = 99.0
            logits[:, 2:] = -99.0
            scores[:, 2:] = 99.0
            offsets[:, 2:] = 99.0
    return SimpleNamespace(
        p=p,
        visibility_logits=logits,
        scores=scores,
        offset_full=offsets,
    )


def test_roi_loss_ignores_every_unselected_role() -> None:
    keypoints = torch.tensor(
        [[[0.50, 0.50], [0.55, 0.55], [0.10, 0.10], [0.90, 0.90]]]
    )
    visibility = torch.ones((1, 4))
    base, base_parts = trainer.compute_roi_losses(
        _output(), keypoints, visibility, _selection()
    )
    corrupt, corrupt_parts = trainer.compute_roi_losses(
        _output(corrupt_unselected=True), keypoints, visibility, _selection()
    )
    torch.testing.assert_close(base, corrupt)
    assert base_parts == corrupt_parts
    base.backward()


def test_raw_480_preprocessing_reuses_runtime_crop_path(monkeypatch) -> None:
    frames = np.zeros((2, 4, 480, 480, 3), dtype=np.uint8)
    seen = {}

    def fake_prepare(value, device, *, image_size):
        seen["input_shape"] = value.shape
        seen["preserve_raw"] = image_size
        return torch.zeros((2, 4, 3, 480, 480), device=device)

    def fake_crop(value, roi, *, canonical_image_size):
        seen["raw_video_shape"] = tuple(value.shape)
        seen["roi"] = roi.clone()
        seen["canonical_image_size"] = canonical_image_size
        return torch.full((2, 4, 3, 384, 384), 0.5, device=value.device)

    monkeypatch.setattr(trainer, "prepare_metric_roi_video", fake_prepare)
    monkeypatch.setattr(trainer, "crop_metric_roi_video", fake_crop)
    roi = torch.tensor([[192.0, 192.0, 96.0], [160.0, 160.0, 128.0]])
    result = trainer.preprocess_raw_roi_frames(frames, roi, torch.device("cpu"))
    assert seen["input_shape"] == (2, 1, 4, 480, 480, 3)
    assert seen["preserve_raw"] is None
    assert seen["raw_video_shape"] == (2, 4, 3, 480, 480)
    assert seen["canonical_image_size"] == 384
    assert result.shape == (2, 4, 3, 384, 384)


def test_metric_batch_raw_frames_are_opt_in_and_keep_480(monkeypatch) -> None:
    class FakeEnv:
        def reset(self, **kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr(metric_data, "make_env", lambda *args, **kwargs: FakeEnv())

    def fake_sample(env, task, rng, w, *, include_raw_frames=False):
        record = {
            "frames": np.zeros((w, 384, 384, 3), dtype=np.uint8),
            "keypoints": np.zeros((4, 2), dtype=np.float32),
            "visibility": np.ones(4, dtype=np.float32),
            "surface_visible": np.ones(4, dtype=np.float32),
            "entity_visible": np.ones(4, dtype=np.float32),
            "in_frame": np.ones(4, dtype=np.float32),
            "relation": np.zeros(6, dtype=np.float32),
            "relation_aux": np.zeros(4, dtype=np.float32),
            "contact": np.float32(0),
            "world": np.zeros((4, 3), dtype=np.float32),
            "supported": True,
        }
        if include_raw_frames:
            record["raw_frames"] = np.zeros((w, 480, 480, 3), dtype=np.uint8)
        return record

    monkeypatch.setattr(metric_data, "_sample_one", fake_sample)
    plain = metric_data.make_metric_batch(
        metric_data.SUPPORTED_TASKS[0], np.random.default_rng(1), 2
    )
    raw = metric_data.make_metric_batch(
        metric_data.SUPPORTED_TASKS[0],
        np.random.default_rng(1),
        2,
        include_raw_frames=True,
    )
    assert "raw_frames" not in plain
    assert raw["raw_frames"].shape == (2, 4, 480, 480, 3)
    assert raw["meta"]["raw_frame_size"] == 480


def test_exact_resume_restores_optimizer_numpy_and_torch_rng() -> None:
    torch.manual_seed(17)
    rng = np.random.default_rng(23)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    (parameter.square()).backward()
    optimizer.step()
    state = trainer._rng_state(optimizer, rng, completed_steps=4)
    expected_numpy = rng.random(5)
    expected_torch = torch.rand(5)
    parameter2 = torch.nn.Parameter(torch.tensor(1.0))
    optimizer2 = torch.optim.Adam([parameter2], lr=1e-4)
    trainer.restore_training_state(state, optimizer2, rng)
    np.testing.assert_array_equal(rng.random(5), expected_numpy)
    torch.testing.assert_close(torch.rand(5), expected_torch, rtol=0, atol=0)
    assert optimizer2.state_dict()["state"]


def test_trainer_payload_roundtrips_runtime_loader_and_resume(tmp_path) -> None:
    ctor = _tiny_head_ctor()
    head = LanguageMetricField(**ctor)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    args = trainer.resolve_args(
        trainer.parse_args(["--coarse-checkpoint", "coarse.pt"]), None
    )
    coarse_sha = "coarse-sha"
    coarse_head_sha = metric_head_state_sha256(head)
    payload = trainer.build_checkpoint_payload(
        args,
        list(trainer.SUPPORTED_TASKS),
        4,
        head_ctor=ctor,
        coarse_sha256=coarse_sha,
        coarse_head_state_sha256=coarse_head_sha,
        coarse_checkpoint={"contract": trainer.COARSE_CONTRACT, "config": {}},
        roi_head=head,
        optimizer=optimizer,
        rng=np.random.default_rng(args.seed),
    )
    path = tmp_path / "roi.pt"
    torch.save(payload, path)
    loaded = load_metric_roi_checkpoint(
        path,
        torch.device("cpu"),
        coarse_identity={
            "sha256": coarse_sha,
            "size_bytes": 1,
            "contract": trainer.COARSE_CONTRACT,
        },
        coarse_head_state_sha256=coarse_head_sha,
    )
    assert loaded._mtvj_roi_config["min_roi_size"] == args.min_roi_size
    assert loaded._mtvj_roi_config["max_roi_size"] == args.max_roi_size

    resumed = trainer.resolve_args(
        trainer.parse_args(
            [
                "--coarse-checkpoint",
                "coarse.pt",
                "--resume",
                str(path),
                "--steps",
                "5004",
            ]
        ),
        payload,
    )
    assert resumed.min_roi_size == args.min_roi_size
    assert resumed.max_roi_size == args.max_roi_size
    assert resumed.steps == 5004


def test_final_policy_coarse_uses_embedded_head_and_runtime_v5_identity(
    tmp_path,
) -> None:
    runtime_path = tmp_path / "metric-v5.pt"
    runtime_checkpoint, runtime_head = _write_completed_runtime_metric(runtime_path)
    runtime_identity = trainer.metric_checkpoint_identity(
        runtime_path, runtime_checkpoint
    )
    policy_head = LanguageMetricField(**_tiny_head_ctor())
    policy_head.load_state_dict(runtime_head.state_dict(), strict=True)
    with torch.no_grad():
        next(policy_head.parameters()).view(-1)[0].add_(0.25)
    policy_path = tmp_path / "policy-100k.pt"
    torch.save(
        {
            "training_contract": {"metric_head_checkpointed": True},
            "mtvj_metric_head": policy_head.state_dict(),
            "mtvj_metric_head_config": _tiny_head_ctor(),
            "mtvj_metric_checkpoint_identity": runtime_identity,
        },
        policy_path,
    )

    resolved_path, coarse_checkpoint, source = trainer.load_coarse_source(
        coarse_checkpoint_path=None,
        coarse_policy_checkpoint_path=policy_path,
        runtime_metric_checkpoint_path=runtime_path,
    )
    assert resolved_path == runtime_path.resolve()
    assert source["kind"] == trainer.COARSE_SOURCE_POLICY
    assert source["runtime_metric_checkpoint"]["sha256"] == runtime_identity["sha256"]
    policy_head_sha = metric_head_state_sha256(policy_head)
    assert source["actual_coarse_head_state_sha256"] == policy_head_sha
    assert policy_head_sha != metric_head_state_sha256(runtime_head)
    for key, value in policy_head.state_dict().items():
        torch.testing.assert_close(
            coarse_checkpoint["metric_head"][key], value, rtol=0, atol=0
        )

    coarse, roi_head, ctor = trainer.build_frozen_coarse_and_roi(
        coarse_checkpoint, torch.device("cpu")
    )
    assert metric_head_state_sha256(coarse) == policy_head_sha
    args = trainer.resolve_args(
        trainer.parse_args(
            [
                "--coarse-policy-checkpoint",
                str(policy_path),
                "--runtime-metric-checkpoint",
                str(runtime_path),
            ]
        ),
        None,
    )
    payload = trainer.build_checkpoint_payload(
        args,
        list(trainer.SUPPORTED_TASKS),
        4,
        head_ctor=ctor,
        coarse_sha256=runtime_identity["sha256"],
        coarse_head_state_sha256=policy_head_sha,
        coarse_checkpoint=coarse_checkpoint,
        roi_head=roi_head,
        optimizer=torch.optim.Adam(roi_head.parameters(), lr=args.lr),
        rng=np.random.default_rng(args.seed),
        coarse_source=source["kind"],
        runtime_metric_identity=runtime_identity,
    )
    assert payload["coarse"]["source"] == trainer.COARSE_SOURCE_POLICY
    assert payload["coarse"]["runtime_metric_checkpoint"]["sha256"] == (
        runtime_identity["sha256"]
    )
    assert payload["coarse"]["contract"] == trainer.COARSE_CONTRACT
    assert payload["coarse"]["coarse_head_state_sha256"] == policy_head_sha
    validate_roi_checkpoint(
        payload,
        coarse_checkpoint,
        runtime_identity["sha256"],
        policy_head_sha,
        coarse_source=trainer.COARSE_SOURCE_POLICY,
    )
    payload["coarse"]["runtime_metric_checkpoint"]["sha256"] = "wrong-v5"
    with pytest.raises(ValueError, match="runtime metric checkpoint"):
        validate_roi_checkpoint(
            payload,
            coarse_checkpoint,
            runtime_identity["sha256"],
            policy_head_sha,
            coarse_source=trainer.COARSE_SOURCE_POLICY,
        )


def test_final_policy_coarse_rejects_wrong_runtime_or_actual_head(tmp_path) -> None:
    runtime_path = tmp_path / "metric-v5.pt"
    runtime_checkpoint, runtime_head = _write_completed_runtime_metric(runtime_path)
    runtime_identity = trainer.metric_checkpoint_identity(
        runtime_path, runtime_checkpoint
    )
    policy_path = tmp_path / "policy-100k.pt"
    policy = {
        "training_contract": {"metric_head_checkpointed": True},
        "mtvj_metric_head": runtime_head.state_dict(),
        "mtvj_metric_head_config": _tiny_head_ctor(),
        "mtvj_metric_checkpoint_identity": runtime_identity,
    }
    torch.save(policy, policy_path)
    wrong_runtime_path = tmp_path / "wrong-metric-v5.pt"
    wrong_checkpoint, wrong_head = _write_completed_runtime_metric(wrong_runtime_path)
    with torch.no_grad():
        next(wrong_head.parameters()).view(-1)[0].add_(1.0)
    wrong_checkpoint["metric_head"] = wrong_head.state_dict()
    torch.save(wrong_checkpoint, wrong_runtime_path)
    with pytest.raises(ValueError, match="external identity"):
        trainer.load_coarse_source(
            coarse_checkpoint_path=None,
            coarse_policy_checkpoint_path=policy_path,
            runtime_metric_checkpoint_path=wrong_runtime_path,
        )

    _, coarse_checkpoint, source = trainer.load_coarse_source(
        coarse_checkpoint_path=None,
        coarse_policy_checkpoint_path=policy_path,
        runtime_metric_checkpoint_path=runtime_path,
    )
    coarse, roi_head, ctor = trainer.build_frozen_coarse_and_roi(
        coarse_checkpoint, torch.device("cpu")
    )
    head_sha = metric_head_state_sha256(coarse)
    args = trainer.resolve_args(
        trainer.parse_args(["--coarse-checkpoint", str(runtime_path)]), None
    )
    payload = trainer.build_checkpoint_payload(
        args,
        list(trainer.SUPPORTED_TASKS),
        4,
        head_ctor=ctor,
        coarse_sha256=runtime_identity["sha256"],
        coarse_head_state_sha256=head_sha,
        coarse_checkpoint=coarse_checkpoint,
        roi_head=roi_head,
        optimizer=torch.optim.Adam(roi_head.parameters(), lr=args.lr),
        rng=np.random.default_rng(args.seed),
        coarse_source=source["kind"],
        runtime_metric_identity=runtime_identity,
    )
    with pytest.raises(ValueError, match="metric-head state"):
        validate_roi_checkpoint(
            payload,
            coarse_checkpoint,
            runtime_identity["sha256"],
            "different-policy-head-sha",
            coarse_source=trainer.COARSE_SOURCE_POLICY,
        )


def test_coarse_source_cli_contract_keeps_stage_v_default(tmp_path) -> None:
    runtime_path = tmp_path / "metric-v5.pt"
    runtime_checkpoint, _ = _write_completed_runtime_metric(runtime_path)
    _, normalized, source = trainer.load_coarse_source(
        coarse_checkpoint_path=runtime_path,
    )
    assert source["kind"] == trainer.COARSE_SOURCE_STAGE_V
    assert normalized["metric_head"].keys() == runtime_checkpoint["metric_head"].keys()
    with pytest.raises(ValueError, match="requires --runtime-metric-checkpoint"):
        trainer.load_coarse_source(
            coarse_checkpoint_path=None,
            coarse_policy_checkpoint_path=tmp_path / "policy.pt",
        )


def test_resume_rejects_task_order_or_version_drift() -> None:
    args = trainer.resolve_args(
        trainer.parse_args(["--coarse-checkpoint", "coarse.pt"]), None
    )
    config = trainer.build_checkpoint_config(
        args,
        list(trainer.SUPPORTED_TASKS),
        4,
        head_ctor={},
        coarse_sha256="sha",
        coarse_head_state_sha256="head-sha",
    )
    config["tasks"] = list(reversed(config["tasks"]))
    with pytest.raises(ValueError, match="task order"):
        trainer.resolve_args(
            trainer.parse_args(["--coarse-checkpoint", "coarse.pt"]),
            {"config": config},
        )


def test_cli_rejects_runtime_incompatible_geometry() -> None:
    with pytest.raises(ValueError, match="distance-scale must be positive"):
        trainer.resolve_args(
            trainer.parse_args(
                ["--coarse-checkpoint", "coarse.pt", "--distance-scale", "0"]
            ),
            None,
        )


def _good_pair_metrics():
    target = np.full((2, 4, 2), 0.5, dtype=np.float32)
    visibility = np.ones((2, 4), dtype=np.float32)
    coarse = target.copy()
    coarse[..., 0] += 12.0 / 384.0
    roi = target.copy()
    roi[..., 0] += 6.0 / 384.0
    coarse_vis = np.full((2, 4), 0.6, dtype=np.float32)
    roi_vis = np.full((2, 4), 0.9, dtype=np.float32)
    return paired_metrics(coarse, roi, target, visibility, coarse_vis, roi_vis)


def test_adoption_gate_enforces_all_four_requested_thresholds() -> None:
    metrics = _good_pair_metrics()
    grouped = {"hard": metrics, "very_hard": metrics}
    gate = adoption_gate(metrics, grouped)
    assert gate["adopt"] is True
    assert all(gate["checks"].values())

    regressed = _good_pair_metrics()
    regressed["roi"]["localization"]["roles"]["target"]["rmse_px"] = 20.0
    gate = adoption_gate(regressed, grouped)
    assert gate["adopt"] is False
    assert gate["checks"]["every_role_regression_at_most_5pct"] is False


def test_selection_diagnostics_reports_confidence_and_gt_crop_coverage() -> None:
    result = selection_diagnostics(
        {
            "selection_confidence": np.array([[0.1], [0.5], [0.9]]),
            "selected_gt_crop_visible": np.array([[0.0], [1.0], [2.0]]),
        }
    )
    assert result["confidence_mean"] == pytest.approx(0.5)
    assert result["usable_pair_fraction"] == pytest.approx(2 / 3)
    assert result["both_roles_crop_visible_fraction"] == pytest.approx(1 / 3)


def test_eval_requires_explicit_alpha_one_before_loading_any_model(tmp_path) -> None:
    args = SimpleNamespace(
        alpha=None,
        samples_per_task=1,
        batch_size=1,
        coarse_checkpoint=str(tmp_path / "missing-coarse.pt"),
        roi_checkpoint=str(tmp_path / "missing-roi.pt"),
        seed=1,
        device="cpu",
    )
    from scripts.eval_metric_roi_holdout import evaluate

    with pytest.raises(ValueError, match="explicit --alpha 1"):
        evaluate(args)
