from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from eval_metaworld import resolve_execution_horizon


ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "scripts" / "eval_mw_hard2_wam4va.sh"


def _args(*, execution_horizon=None, execute_steps=None) -> SimpleNamespace:
    return SimpleNamespace(
        execution_horizon=execution_horizon,
        execute_steps=execute_steps,
    )


def test_peer_defaults_execution_horizon_to_checkpoint_planning_stride() -> None:
    config = SimpleNamespace(va_world_mode="peer_sync_h6", planning_stride=2)
    assert resolve_execution_horizon(_args(), config) == 2
    assert resolve_execution_horizon(_args(execution_horizon=2), config) == 2
    assert resolve_execution_horizon(_args(execute_steps=2), config) == 2


def test_peer_rejects_execution_horizon_different_from_planning_stride() -> None:
    config = SimpleNamespace(va_world_mode="peer_sync_h6", planning_stride=2)
    with pytest.raises(ValueError, match="execution_horizon.*planning_stride"):
        resolve_execution_horizon(_args(execution_horizon=6), config)


def test_legacy_deployment_keeps_six_step_default() -> None:
    config = SimpleNamespace(va_world_mode="legacy", planning_stride=2)
    assert resolve_execution_horizon(_args(), config) == 6


def test_hard2_eval_launcher_is_fixed_to_peer_p2_h6() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(EVAL_RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = EVAL_RUNNER.read_text(encoding="utf-8")
    for token in (
        "hard2_peer_h6_p2_eval_v1.pt",
        "EXECUTION_HORIZON=2",
        '"contract": "peer_sync_h6_p2_world_windows_v1"',
        '"fps": 80',
        '"control_stride": 2',
        '"planning_stride": 2',
        '"action_horizon": 6',
        '"wmrm_cycle_steps": 2',
        '"flow_prefix_steps": 2',
        '"peer_training_mode": "joint_dual_stream"',
        '"peer_va_data_identity"',
        '"peer_world_data_identity"',
        '--execution-horizon "$EXECUTION_HORIZON"',
    ):
        assert token in text
    assert "--execute-steps" not in text


def test_eval_preflight_rejects_p6_flow_prefix_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong-flow-prefix.pt"
    features = tmp_path / "p2-eval.pt"
    identity = {"full_file_sha256": "a" * 64}
    torch.save(
        {
            "config": {
                "va_world_mode": "peer_sync_h6",
                "wmrm": True,
                "action_horizon": 6,
                "planning_stride": 2,
                "wmrm_cycle_steps": 2,
            },
            "training_contract": {
                "peer_training_mode": "joint_dual_stream",
                "peer_world_topology": "one_stage_delayed_bidirectional_state_kv_v1",
                "peer_gradient_boundary": "fully_differentiable_bidirectional_messages_v1",
                "peer_data_isolation": "separate_va_world_episode_datasets_per_step_v1",
                "peer_dual_stream_optimizer": (
                    "va_backward_then_world_backward_one_optimizer_step_v1"
                ),
                "peer_va_data_identity": identity,
                "peer_world_data_identity": identity,
            },
            "exact_run_contract": {
                "arguments": {
                    "control_stride": 2,
                    "planning_stride": 2,
                    "wmrm_cycle_steps": 2,
                    "flow_prefix_steps": 6,
                }
            },
        },
        checkpoint,
    )
    torch.save({}, features)
    launcher = EVAL_RUNNER.read_text(encoding="utf-8")
    preflight = launcher.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-c", preflight, str(checkpoint), str(features), "2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "flow_prefix_steps" in result.stderr
