"""CPU regression tests for the dual-stream visual aux loss (阶段 C, 2026-08-12).

覆盖：_mtvj_visual_aux_loss 的反传范围（只进 metric head，不进 backbone）、
语言缓存缺失 fail-fast、参数契约（--mtvj-visual-aux-every 前置条件）。
仿真生成用 monkeypatch 的 fake make_metric_batch（测试环境无 GL）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from train import (
    _mtvj_visual_aux_loss,
    _mtvj_visual_aux_sample,
    _prepare_mtvj_visual_aux_step,
    parse_args,
    validate_args,
)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_hierarchical_dense(self, inputs, out_layers):
        values = self.scale * torch.ones(inputs.shape[0], 1152, 4)
        return {5: values, 11: values + 1.0}


class FakeMetricHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))
        self.vis_w = nn.Parameter(torch.tensor(0.5))

    def forward(self, h5, h11, language_hidden, language_mask, coords):
        b = h5.shape[0]
        p = (self.anchor + 0.1 * torch.arange(b, dtype=h5.dtype).view(-1, 1, 1)).expand(
            -1, 4, 2
        )
        return SimpleNamespace(
            p=p,
            scores=torch.zeros(b, 4, 1152),
            offset_full=torch.zeros(b, 4, 1152, 2),
            visibility_logits=(self.vis_w * torch.ones(b, 4)),
        )


def _fake_make_metric_batch(task, rng, n):
    frames = np.zeros((n, 4, 32, 32, 3), dtype=np.uint8)
    kp = np.tile(
        np.array(
            [[0.2, 0.2], [0.8, 0.8], [0.5, 0.5], [0.3, 0.7]],
            dtype=np.float32,
        ),
        (n, 1, 1),
    )
    vis = np.ones((n, 4), dtype=np.float32)
    return {"frames": frames, "keypoints": kp, "visibility": vis}


def _lang_cache():
    from scripts.build_longtraj_features import ENV_TO_TASK

    return {
        ENV_TO_TASK["pick-place-v3"]: (
            torch.zeros(1, 3, 8, dtype=torch.float32),
            torch.ones(1, 3, dtype=torch.bool),
        )
    }


def test_mtvj_visual_aux_loss_backprops_only_into_metric_head(monkeypatch) -> None:
    monkeypatch.setattr(
        "prepare_metaworld_metric.make_metric_batch", _fake_make_metric_batch
    )
    backbone = FakeBackbone()
    metric_head = FakeMetricHead()
    rng = np.random.default_rng(0)
    loss_aux, parts = _mtvj_visual_aux_loss(
        backbone,
        metric_head,
        "pick-place-v3",
        rng,
        aux_batch=2,
        lang_aux_cache=_lang_cache(),
        device=torch.device("cpu"),
        loc_lambda=1.0,
        vis_lambda=0.5,
    )
    assert torch.isfinite(loss_aux)
    assert parts["rmse_px"] > 0.0
    assert set(("loc", "vis", "hinge", "pos", "offset")) <= set(parts)
    loss_aux.backward()
    assert backbone.scale.grad is None, "V-JEPA 必须保持无梯度"
    assert metric_head.anchor.grad is not None
    assert metric_head.anchor.grad.abs().sum() > 0
    assert metric_head.vis_w.grad is not None
    assert metric_head.vis_w.grad.abs().sum() > 0


def test_mtvj_visual_aux_loss_missing_language_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(
        "prepare_metaworld_metric.make_metric_batch", _fake_make_metric_batch
    )
    with pytest.raises(KeyError, match="语言缓存缺少"):
        _mtvj_visual_aux_loss(
            FakeBackbone(),
            FakeMetricHead(),
            "unknown-task-v3",
            np.random.default_rng(0),
            aux_batch=1,
            lang_aux_cache=_lang_cache(),
            device=torch.device("cpu"),
            loc_lambda=1.0,
            vis_lambda=0.5,
        )


def test_mtvj_visual_aux_argument_contract() -> None:
    base = [
        "--dense-readout-mtvj",
        "--metric-visual-checkpoint",
        "metric.pt",
        "--mtvj-train-metric-head",
        "--mtvj-train-relation",
        "--lr-mtvj-metric-head",
        "1e-6",
        "--mtvj-visual-aux-every",
        "8",
        "--task-sampling",
        "weighted",
        "--single-task",
        "--resume",
        "policy.pt",
    ]
    validate_args(parse_args(base))  # 合法组合

    no_head = parse_args(base)
    no_head.mtvj_train_metric_head = False
    with pytest.raises(ValueError, match="要求解冻视觉头"):
        validate_args(no_head)

    no_dense = parse_args(
        [
            "--metric-visual-checkpoint",
            "metric.pt",
            "--mtvj-train-metric-head",
            "--mtvj-visual-aux-every",
            "8",
            "--task-sampling",
            "weighted",
            "--resume",
            "policy.pt",
        ]
    )
    with pytest.raises(ValueError, match="requires --dense-readout-mtvj"):
        validate_args(no_dense)

    balanced = parse_args(base)
    balanced.task_sampling = "balanced"
    validate_args(balanced)

    no_weighted_or_balanced = parse_args(base)
    no_weighted_or_balanced.task_sampling = "uniform"
    with pytest.raises(ValueError, match=r"weighted\|balanced"):
        validate_args(no_weighted_or_balanced)

    with_sam = parse_args(base)
    with_sam.sam_rho = 0.5
    with pytest.raises(ValueError, match="forbids --sam-rho"):
        validate_args(with_sam)

    no_relation = parse_args(base)
    no_relation.mtvj_train_relation = False
    with pytest.raises(ValueError, match="requires --mtvj-train-relation"):
        validate_args(no_relation)

    no_single = parse_args(base)
    no_single.single_task = False
    with pytest.raises(ValueError, match="requires --single-task"):
        validate_args(no_single)


def test_mtvj_visual_aux_sample_maps_description_and_is_resume_stable() -> None:
    descriptions = ["easy instruction", "hard instruction"]
    weights = torch.tensor([0.5, 2.0], dtype=torch.float64)
    mapping = {
        "easy instruction": "easy-v3",
        "hard instruction": "hard-v3",
    }
    first_env, first_rng = _mtvj_visual_aux_sample(
        descriptions, weights, mapping, seed=7, global_step=71008
    )
    resumed_env, resumed_rng = _mtvj_visual_aux_sample(
        descriptions, weights, mapping, seed=7, global_step=71008
    )

    assert first_env == resumed_env
    assert first_rng.random() == resumed_rng.random()
    assert first_env in mapping.values()


def test_mtvj_visual_aux_sample_rejects_unmapped_description() -> None:
    with pytest.raises(KeyError, match="无法映射"):
        _mtvj_visual_aux_sample(
            ["unknown instruction"],
            torch.ones(1),
            {},
            seed=0,
            global_step=8,
        )


def test_visual_aux_cpu_batch_is_scheduled_once_and_consumed_once(monkeypatch) -> None:
    calls = []

    def fake_batch(task, rng, n):
        calls.append((task, n))
        return _fake_make_metric_batch(task, rng, n)

    monkeypatch.setattr("prepare_metaworld_metric.make_metric_batch", fake_batch)
    common = dict(
        task_descriptions=["Pick and place an object"],
        task_weights=torch.ones(1, dtype=torch.float64),
        env_by_description={"Pick and place an object": "pick-place-v3"},
        seed=9,
        every=10,
        aux_batch=2,
        include_raw_frames=False,
    )

    assert _prepare_mtvj_visual_aux_step(global_step=9, **common) is None
    prepared = _prepare_mtvj_visual_aux_step(global_step=10, **common)
    assert prepared is not None
    task, rng, sim_batch = prepared
    assert calls == [("pick-place-v3", 2)]

    loss, _ = _mtvj_visual_aux_loss(
        FakeBackbone(),
        FakeMetricHead(),
        task,
        rng,
        aux_batch=2,
        lang_aux_cache=_lang_cache(),
        device=torch.device("cpu"),
        loc_lambda=1.0,
        vis_lambda=0.5,
        sim_batch=sim_batch,
    )

    assert torch.isfinite(loss)
    assert calls == [("pick-place-v3", 2)]
