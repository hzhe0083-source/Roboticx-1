from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh"


def _text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _block(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_runner_syntax_and_invalid_mode_fail_before_training() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)], cwd=ROOT, capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    invalid = subprocess.run(
        ["bash", str(RUNNER), "invalid"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PY": "/definitely/missing/python"},
    )
    assert invalid.returncode == 2
    assert "{prepare|preflight|joint}" in invalid.stderr


def test_prepare_builds_three_episode_disjoint_p2_data_sides() -> None:
    text = _text()
    prepare = _block(text, "prepare_data(){", "preflight(){")
    # The split family is selectable so an expanded rebuild cannot overwrite the
    # immutable v1 splits, but the default must still reproduce v1 byte-for-byte
    # and the checkpoint FAMILY must move with the tag so the two data families
    # never share a checkpoint namespace.
    assert "DATA_TAG=${DATA_TAG:-v1}" in text
    assert "SOURCE=data/hard2_peer_h6_p2_source_${DATA_TAG}.pt" in text
    assert "VA_TRAIN_DATA=data/hard2_peer_h6_p2_va_train_${DATA_TAG}.pt" in text
    assert "WORLD_TRAIN_DATA=data/hard2_peer_h6_p2_world_train_${DATA_TAG}.pt" in text
    assert "EVAL_DATA=data/hard2_peer_h6_p2_eval_${DATA_TAG}.pt" in text
    assert "WORLD_POOL=data/hard2_peer_h6_p2_world_pool_${DATA_TAG}.pt" in text
    assert "FAMILY=mw_hard2_va_world_state_exchange_joint_h6_p2_${DATA_TAG}" in text
    assert "--planning-stride 2" in prepare
    assert "--data-contract peer_sync_h6_p2_world_windows_v1" in prepare
    assert "--control-stride" not in prepare
    assert "--heldout-fraction 0.50 --seed 101" in prepare
    assert "--heldout-fraction 0.20 --seed 202" in prepare
    assert '--eval-output "$VA_TRAIN_DATA"' in prepare
    assert '--train-output "$WORLD_TRAIN_DATA"' in prepare
    preflight = _block(text, "preflight(){", "checkpoint_contract(){")
    assert '"$WORLD_POOL"' in preflight
    assert 'episodes[left] & episodes[right]' in preflight
    assert '(("va", "world"), ("va", "eval"), ("world", "eval"))' in preflight
    assert 'get("current_action_prefix_steps") != 2' in preflight


def test_peer_cli_requires_joint_dual_data_streams() -> None:
    preflight = _block(_text(), "preflight(){", "checkpoint_contract(){")
    assert '"--va-data", "va-unused.pt"' in preflight
    assert '"--world-data", "world-unused.pt"' in preflight
    assert '"--visual-world-supervision", "--world-split-manifest"' in preflight
    assert "validate_args(parse_args" in preflight
    assert "--va-world-mode\", \"peer_sync_h6" in preflight
    assert '"--planning-stride", "2", "--control-stride", "2"' in preflight
    assert '"--wmrm-cycle-steps", "2"' in preflight
    assert '"--flow-prefix-steps", "2"' in preflight
    assert '"--data"' not in preflight
    assert "--world-only" not in preflight
    assert "--va-only" not in preflight


def test_joint_run_uses_both_data_streams_without_phase_handoff() -> None:
    text = _text()
    joint = _block(text, "run_joint(){", "command -v flock")
    assert '--va-data "$VA_TRAIN_DATA" --world-data "$WORLD_TRAIN_DATA"' in joint
    assert '--visual-world-supervision --world-split-manifest "$WORLD_SPLIT_MANIFEST"' in joint
    assert "--va-world-mode peer_sync_h6 --planning-stride 2 --control-stride 2" in joint
    assert "--wmrm-cycle-steps 2" in joint
    assert "--flow-prefix-steps 2" in joint
    assert "--task-sampling balanced" in joint
    assert " train.py --data " not in joint
    assert "NGPUS=${NGPUS:-1}" in text
    assert "torch.distributed.run" in joint
    assert "--world-only" not in joint
    assert "--va-only" not in joint
    assert 'resume_args=(--resume-exact "$RESUME_EXACT")' in joint
    assert 'resume_args=(--resume-weights "$RESUME_WEIGHTS")' in joint
    assert "RESUME_EXACT and RESUME_WEIGHTS are mutually exclusive" in joint
    assert "--wmrm-stage-s5-weight" in joint
    assert "--wmrm-stage-s6-weight" in joint
    assert "--wmrm-late-stage-anchor-weight" in joint
    assert "--lr-wmrm-predictor" in joint
    assert "--wmrm-predictor-grad-clip" in joint
    assert "SOURCE_CHECKPOINT" not in text
    assert "run_phase" not in text
    assert "joint) preflight; run_joint" in text
    for legacy in ("--wam-joint", "wam_residual_fn", "JointWorldActionFlow"):
        assert legacy not in text


def test_checkpoint_contract_records_joint_gradient_and_data_protocol() -> None:
    block = _block(_text(), "checkpoint_contract(){", "require_no_active_train(){")
    assert '"peer_world_topology": "one_stage_delayed_world_minus_one_last_va_consume_v1"' in block
    assert '"peer_training_mode": "joint_dual_stream"' in block
    assert '"peer_gradient_boundary": "fully_differentiable_bidirectional_messages_v1"' in block
    assert '"peer_data_isolation": "separate_va_world_episode_datasets_per_step_v1"' in block
    assert '"peer_dual_stream_optimizer": "va_backward_then_world_backward_one_optimizer_step_v1"' in block
    assert '"peer_va_data_identity"' in block
    assert '"peer_world_data_identity"' in block
    assert '"peer_data_isolation_summary"' in block
    assert 'identity.get("full_file_sha256")' in block
    assert 'summary.get("task_ids") != [0, 16]' in block
    assert '"planning_stride": (config.get("planning_stride"), 2)' in block
    assert '"wmrm_cycle_steps": (config.get("wmrm_cycle_steps"), 2)' in block
    assert '"contract": "peer_sync_h6_p2_world_windows_v1"' in block
    assert '"fps": 80' in block
    assert '"control_stride": 2' in block
    assert '"planning_stride": 2' in block
    assert '"decision_offsets": [0, 2, 4, 6]' in block
    assert '"flow_prefix_steps": 2' in block
