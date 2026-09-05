"""CPU regression tests for the MT-VJ train/eval runtime contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval_metaworld import (
    _load_mtvj_metric_checkpoint,
    _mtvj_metric_checkpoint_identity,
    _mtvj_visibility_gated_positions as _eval_visibility_gated_positions,
)
from scripts.build_longtraj_features import ENV_TO_TASK
from va_compound.vision.metric_runtime import (
    _mtvj_metric_head_constructor_config,
    _mtvj_metric_deltas,
    _mtvj_relation_tokens,
    _mtvj_metric_positions as _train_visibility_gated_positions,
)
from va_compound.backbones import pool_mtvj_coarse_tokens
from va_compound.metric_visual_head import LanguageMetricField, RelationStateEncoder


def test_metric_module_import_keeps_gl_backend_self_consistent() -> None:
    """``prepare_metaworld_metric`` 不能把 GL 后端设成与 MUJOCO_GL 矛盾的值。

    该模块在导入时设默认后端，而 ``--mtvj-visual-aux-every`` 是第一次真正建
    MuJoCo env 的地方——训练启动几十分钟后。曾经它无条件把 PYOPENGL_PLATFORM 设为
    egl，而所有启动脚本导出 MUJOCO_GL=osmesa，于是 ``mujoco.gl_context`` 拒绝导入，
    长训练在第 10 步崩掉。

    子进程必须复现启动脚本的完整环境，``LD_PRELOAD`` 那一项不是可选的：不带它
    PyOpenGL 加载不到系统 libOSMesa（conda 自带 libstdc++ 与之 ABI 冲突），失败方式
    不同、会掩盖这里要测的东西。
    """
    import os
    import subprocess
    import sys

    preload = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if not preload.exists():
        pytest.skip(f"launcher LD_PRELOAD not present: {preload}")
    probe = (
        "import os, prepare_metaworld_metric, mujoco;"
        "print(os.environ.get('MUJOCO_GL'), os.environ.get('PYOPENGL_PLATFORM'))"
    )
    env = dict(os.environ, MUJOCO_GL="osmesa", LD_PRELOAD=str(preload))
    env.pop("PYOPENGL_PLATFORM", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    backend, pyopengl = result.stdout.split()[-2:]
    assert backend == "osmesa"
    assert pyopengl != "egl"


def test_mtvj_h11_pool16_matches_historical_training_formula() -> None:
    h11 = torch.arange(2 * 1152 * 3, dtype=torch.float32).reshape(2, 1152, 3)
    expected = h11.reshape(2, 16, 72, 3).mean(dim=2)

    pooled = pool_mtvj_coarse_tokens(h11)

    torch.testing.assert_close(pooled, expected, rtol=0.0, atol=0.0)
    assert pooled.shape == (2, 16, 3)


def test_mtvj_h11_pool16_rejects_unequal_bins() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        pool_mtvj_coarse_tokens(torch.zeros(1, 1151, 4))


def test_mtvj_metric_delta_has_zero_first_decision() -> None:
    g = torch.tensor(
        [
            [[2.0, -1.0], [3.5, 4.0], [-2.0, 8.0]],
            [[7.0, 9.0], [6.0, 8.0], [6.5, 7.0]],
        ]
    )

    nu = _mtvj_metric_deltas(g)

    torch.testing.assert_close(nu[:, 0], torch.zeros_like(g[:, 0]))
    torch.testing.assert_close(nu[:, 1:], g[:, 1:] - g[:, :-1])


def test_mtvj_visibility_gate_nulls_invisible_coordinates_and_train_eval_match() -> None:
    p = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.4], [99.0, -99.0], [-7.0, 8.0]]]
    )
    out = SimpleNamespace(
        p=p, visibility=torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    )
    expected = torch.tensor([[0.1, 0.2, 0.15, 0.2, 0.0, 0.0, 0.0, 0.0]])
    train_g = _train_visibility_gated_positions(out)
    eval_g = _eval_visibility_gated_positions(out)
    torch.testing.assert_close(train_g, expected)
    torch.testing.assert_close(eval_g, expected)

    changed = SimpleNamespace(p=p.clone(), visibility=out.visibility)
    changed.p[:, 2:] = torch.tensor([[[1e6, -1e6], [-3e6, 4e6]]])
    torch.testing.assert_close(
        _train_visibility_gated_positions(changed), train_g, rtol=0.0, atol=0.0
    )


def test_mtvj_relation_tokens_backprop_only_through_action_path() -> None:
    relation = RelationStateEncoder(state_dim=8, d_model=16)
    relation.recon.requires_grad_(False)
    g = torch.randn(2, 4, 8)

    tokens = _mtvj_relation_tokens(g, relation)
    tokens.square().mean().backward()

    assert tokens.requires_grad
    assert relation.g_proj.weight.grad is not None
    assert relation.nu_proj.weight.grad is not None
    assert relation.norm.weight.grad is not None
    assert relation.recon.weight.grad is None
    assert g.grad is None


def test_mtvj_joint_step0_is_bitwise_identical_to_frozen_relation() -> None:
    torch.manual_seed(7)
    frozen = RelationStateEncoder(state_dim=8, d_model=16).eval()
    trainable = RelationStateEncoder(state_dim=8, d_model=16).train()
    trainable.load_state_dict(frozen.state_dict(), strict=True)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    trainable.recon.requires_grad_(False)
    g = torch.randn(3, 4, 8)

    with torch.no_grad():
        expected = _mtvj_relation_tokens(g, frozen)
    actual = _mtvj_relation_tokens(g, trainable)

    assert torch.equal(actual.detach(), expected)
    assert actual.requires_grad














def _write_external_metric_checkpoint(
    path,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
    **overrides,
) -> None:
    if tasks is None:
        tasks = tuple(ENV_TO_TASK)
    metric_config = {
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
    metric_config.update(overrides)
    constructor_config = {
        key: metric_config[key]
        for key in (
            "lang_dim", "h_dim", "d_proj", "n_roles", "l2_norm",
            "learnable_temp", "temp_init", "freeze_bias", "mode_readout",
        )
    }
    metric_head = LanguageMetricField(**constructor_config)
    legacy_relation = RelationStateEncoder(state_dim=6, d_model=16)
    torch.save(
        {
            "contract": "mt_vj_metric_field_v1",
            "config": {
                **metric_config,
                "d_model": 16,
                "tasks": list(tasks),
                "loc_only": metric_config.get("loc_only", False),
                "relation_encoder_trained": metric_config.get(
                    "relation_encoder_trained", True
                ),
                "training_state_version": metric_config.get(
                    "training_state_version", 2
                ),
                "steps_done": metric_config.get("steps_done", 49),
            },
            "metric_head": metric_head.state_dict(),
            "relation_encoder": legacy_relation.state_dict(),
        },
        path,
    )

def _eval_metric_contract() -> dict:
    return {
        "metric_tokens_enabled": True,
        "metric_state_source": "p_times_visibility_flat",
        "metric_state_dim": 8,
        "metric_d_model": 16,
        "metric_contract_version": 3,
        "metric_head_checkpointed": True,
    }

def _policy_metric_metadata(path, metric_head) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return (
        _mtvj_metric_head_constructor_config(metric_head),
        _mtvj_metric_checkpoint_identity(path, checkpoint),
    )

def test_eval_rejects_external_metric_fingerprint_change(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    changed = torch.load(metric_path, map_location="cpu", weights_only=True)
    changed["revision"] = 2
    torch.save(changed, metric_path)

    with pytest.raises(ValueError, match="fingerprint"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=policy_metric.state_dict(),
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )

def test_eval_prefers_main_policy_mtvj_states_over_legacy_external(
    tmp_path, capsys
) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    constructor_only = torch.load(metric_path, map_location="cpu", weights_only=True)
    constructor_only.pop("metric_head")
    constructor_only.pop("relation_encoder")
    torch.save(constructor_only, metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    with torch.no_grad():
        for parameter in policy_relation.parameters():
            parameter.fill_(0.125)
        for parameter in policy_metric.parameters():
            parameter.fill_(0.25)

    loaded_metric, loaded_relation = _load_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        policy_relation_state=policy_relation.state_dict(),
        policy_metric_state=policy_metric.state_dict(),
        policy_metric_config=policy_config,
        policy_metric_identity=policy_identity,
        policy_training_contract=_eval_metric_contract(),
    )

    for key, value in policy_metric.state_dict().items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    for key, value in policy_relation.state_dict().items():
        torch.testing.assert_close(
            loaded_relation.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    output = capsys.readouterr().out
    assert "metric head from main policy checkpoint" in output
    assert "constructor config from main policy checkpoint" in output

def test_eval_fails_if_contract_declares_missing_policy_metric_head(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    external_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, external_metric
    )

    with pytest.raises(ValueError, match="mtvj_metric_head"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )

def test_eval_strictly_rejects_policy_metric_head_shape_mismatch(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    broken_metric = dict(policy_metric.state_dict())
    first_key = next(iter(broken_metric))
    broken_metric[first_key] = broken_metric[first_key][:-1]

    with pytest.raises(ValueError, match="shape_mismatch"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=broken_metric,
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )

def test_eval_legacy_policy_uses_external_metric_head(tmp_path, capsys) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    external = torch.load(metric_path, map_location="cpu", weights_only=True)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)

    loaded_metric, _ = _load_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        policy_relation_state=policy_relation.state_dict(),
        policy_training_contract={},
    )

    for key, value in external["metric_head"].items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    assert "external metric checkpoint (legacy migration)" in capsys.readouterr().out

def test_eval_rejects_incompatible_main_policy_metric_contract(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)

    with pytest.raises(ValueError, match="metric 契约不兼容"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_training_contract={
                "metric_tokens_enabled": True,
                "metric_state_source": "relation",
                "metric_state_dim": 6,
                "metric_d_model": 16,
                "metric_contract_version": 1,
            },
        )

def test_eval_rejects_random_relation_rebuild_for_legacy_6d(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)

    with pytest.raises(ValueError, match="dense-only"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
        )
