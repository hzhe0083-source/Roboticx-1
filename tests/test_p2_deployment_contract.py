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


def _args(
    *,
    execution_horizon=None,
    execute_steps=None,
    allow_execution_horizon_ablation=False,
) -> SimpleNamespace:
    return SimpleNamespace(
        execution_horizon=execution_horizon,
        execute_steps=execute_steps,
        allow_execution_horizon_ablation=allow_execution_horizon_ablation,
    )


def test_peer_defaults_execution_horizon_to_checkpoint_planning_stride() -> None:
    config = SimpleNamespace(va_world_mode="peer_sync_h6", planning_stride=2)
    assert resolve_execution_horizon(_args(), config) == 2
    assert resolve_execution_horizon(_args(execution_horizon=2), config) == 2
    assert resolve_execution_horizon(_args(execute_steps=2), config) == 2


def test_peer_rejects_execution_horizon_different_from_planning_stride() -> None:
    config = SimpleNamespace(va_world_mode="peer_sync_h6", planning_stride=2)
    with pytest.raises(ValueError, match="deployment horizon"):
        resolve_execution_horizon(_args(execution_horizon=6), config)


def test_h15_peer_defaults_to_full_chunk_deployment() -> None:
    config = SimpleNamespace(
        va_world_mode="peer_sync_h6",
        planning_stride=2,
        deployment_execution_horizon=15,
    )
    assert resolve_execution_horizon(_args(), config) == 15
    assert resolve_execution_horizon(_args(execution_horizon=15), config) == 15
    with pytest.raises(ValueError, match="deployment horizon"):
        resolve_execution_horizon(_args(execution_horizon=2), config)


@pytest.mark.parametrize("execution_horizon", (2, 6, 15))
def test_h15_peer_allows_explicit_execution_horizon_ablation(
    execution_horizon: int,
) -> None:
    config = SimpleNamespace(
        va_world_mode="peer_sync_h6",
        planning_stride=2,
        deployment_execution_horizon=15,
    )
    args = _args(
        execution_horizon=execution_horizon,
        allow_execution_horizon_ablation=True,
    )
    assert resolve_execution_horizon(args, config) == execution_horizon


def test_execution_horizon_ablation_rejects_legacy_checkpoint() -> None:
    config = SimpleNamespace(va_world_mode="legacy", planning_stride=2)
    args = _args(
        execution_horizon=6,
        allow_execution_horizon_ablation=True,
    )
    with pytest.raises(ValueError, match="requires a peer_sync_h6 checkpoint"):
        resolve_execution_horizon(args, config)


def test_legacy_deployment_keeps_six_step_default() -> None:
    config = SimpleNamespace(va_world_mode="legacy", planning_stride=2)
    assert resolve_execution_horizon(_args(), config) == 6


def test_hard2_eval_launcher_is_fixed_to_h15_p15() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(EVAL_RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = EVAL_RUNNER.read_text(encoding="utf-8")
    for token in (
        "hard2_peer_h15_p2_eval_v2.pt",
        "EXECUTION_HORIZON=${EXECUTION_HORIZON:-15}",
        "PEER_WORLD_OFF=${PEER_WORLD_OFF:-0}",
        'data_contract = f"peer_sync_h{action_horizon}_p2_world_windows_v1"',
        "action_horizon not in {6, 15}",
        '"fps": 80',
        '"control_stride": 2',
        '"planning_stride": 2',
        "deployment_horizon != execution_horizon",
        '"action_horizon": action_horizon',
        '"wmrm_cycle_steps": world_horizon',
        '"flow_prefix_steps": 2',
        '"peer_training_mode": "joint_dual_stream"',
        '"peer_flow_topology"',
        '"peer_va_data_identity"',
        '"peer_world_data_identity"',
        '--execution-horizon "$EXECUTION_HORIZON"',
        'WORLD_ARGS=(--peer-world-off)',
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
                "peer_world_topology": "world_minus_one_same_endpoint_fixed_current_anchor_v2",
                "peer_gradient_boundary": "world_map_stopgrad_policy_projection_trainable_v1",
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
