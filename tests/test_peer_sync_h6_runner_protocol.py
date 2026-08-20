from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh"
HISTORICAL_H48 = ROOT / "scripts" / "run_mw_hard2_wam4va_visualmotion_joint_v1.sh"


def _text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _block(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_runner_syntax_and_invalid_mode_fail_before_any_training() -> None:
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
    assert "{prepare|preflight|smoke10|pilot300|20k}" in invalid.stderr
    assert "scratch-only formal lineage" not in invalid.stdout


def test_runner_has_approved_h6_data_and_unique_output_namespaces() -> None:
    text = _text()
    assert "SOURCE=data/hard2_peer_h6_source_v1.pt" in text
    assert "TRAIN_DATA=data/hard2_peer_h6_train_v1.pt" in text
    assert "EVAL_DATA=data/hard2_peer_h6_eval_v1.pt" in text
    assert "SPLIT_MANIFEST=data/hard2_peer_h6_split_v1.json" in text
    assert "DATA_FAMILY=" not in text
    assert "FAMILY=mw_hard2_wam4va_visualmotion_peer_sync_h6_v1" in text
    assert "checkpoints/${run_id}" in text
    assert "logs/${run_id}" in text
    assert "diagnostics/${run_id}" in text
    assert RUNNER != HISTORICAL_H48
    assert "run_mw_hard2_wam4va_visualmotion_joint_v1.sh" not in text


def test_h6_preparation_rewindows_raw_trajectories_with_exact_identities() -> None:
    text = _text()
    prepare = _block(text, "prepare_h6_data(){", "preflight_h6(){")
    assert "ASSEMBLY_RAW=data/metaworld_longtraj_assembly-v3.pt" in text
    assert "DOOR_RAW=data/metaworld_longtraj_door-unlock-v3.pt" in text
    assert "ALLTASK_H48_REF=data/metaworld_longtraj_windows_h48.pt" in text
    assert "EXPECTED_ASSEMBLY_SHA256=c61f3b2102dea781c9db2a73109472e6e181f46e33536879a5eab181ee190ea0" in text
    assert "EXPECTED_DOOR_SHA256=309726cd679753633bf9bb658635b890affcc666523cb530bab62db4d9699bf1" in text
    assert "EXPECTED_ALLTASK_H48_REF_SHA256=5adc69fce88cdfc5a62b0fa4e9da536d2368a81e6ebb5c23543bca2810ab19a4" in text
    assert "raw assembly SHA-256 mismatch" in text
    assert "raw door-unlock SHA-256 mismatch" in text
    assert "normalization/language reference SHA-256 mismatch" in text
    assert "refusing to overwrite immutable H6 data family" in prepare
    for token in (
        "scripts/build_longtraj_features.py",
        "--phase 1",
        "--horizon 6",
        "--data-contract peer_sync_h6_world_windows_v1",
        "--legacy-policy infer",
        '--input "$ASSEMBLY_RAW"',
        '--input "$DOOR_RAW"',
        '--ref "$ALLTASK_H48_REF"',
        '--output "$SOURCE"',
    ):
        assert token in prepare
    assert "[:, :, :6]" not in text
    assert "torch.load" not in prepare
    assert "PARENT_SOURCE" not in text
    assert "prepare) prepare_h6_data; preflight_h6" in text
    for mode in ("preflight", "smoke10", "pilot300", "20k"):
        case = text.split(f"  {mode})", 1)[1].split(";;", 1)[0]
        assert "prepare_h6_data" not in case


def test_preflight_requires_exact_h6_counts_identities_split_and_peer_cli() -> None:
    preflight = _block(_text(), "preflight_h6(){", "refuse_output_family(){")
    for token in (
        "peer_sync_h6_world_windows_v1",
        "'source': 891, 'train': 793, 'eval': 98",
        "tuple(actions.shape) != (expected_counts[name], 4, 6, 4)",
        "metadata.get('contract_version') != 1",
        "metadata.get('action_horizon') != 6",
        "metadata.get('logged_action_chunk') != 'full_h6'",
        "normalization/language reference identity mismatch",
        "raw source identities mismatch",
        "expected_assembly_sha",
        "expected_door_sha",
        "expected_ref_sha",
        "transition_mask(valid).any()",
        "[0, 16]",
        "manifest canonical SHA mismatch",
        "manifest source SHA mismatch",
        "split_manifest_sha256",
        "from train import parse_args, validate_args",
        "validate_args(args)",
        "--va-world-mode', 'peer_sync_h6",
        "--wmrm-adep-weight', '0",
        "args.resume_weights is not None",
        "args.resume_exact_contract_migration is not None",
    ):
        assert token in preflight


def test_training_is_scratch_first_then_exact_resume_only_with_no_migrations() -> None:
    text = _text()
    segment = _block(text, "run_segment(){", "run_lineage(){")
    lineage = _block(text, "run_lineage(){", "command -v flock")
    assert "local start=0 source_checkpoint=scratch" in lineage
    assert "first segment must start from scratch" in segment
    assert 'resume_args=(--resume-exact "$source_checkpoint")' in segment
    assert "continuation requires a distinct exact-resume source" in segment
    assert "exact-resume source checkpoint was modified" in segment
    assert "--resume-weights" not in text
    assert "--resume-exact-contract-migration" not in segment
    assert "--resume-exact-contract-migration" not in text[text.index('"$PY" -u -B train.py'):]
    assert "migrations are forbidden" in text


def test_peer_h6_training_contract_is_explicit_and_single_worker() -> None:
    text = _text()
    segment = _block(text, "run_segment(){", "run_lineage(){")
    launch = text[text.index('"$PY" -u -B train.py'):]
    assert '[[ -f "$DINO" ]] || fail "missing optional training-only DINO checkpoint' in segment
    assert '"$DINO"' not in text.split("prepare_h6_data(){", 1)[0].split("for path in", 1)[1].split("done", 1)[0]
    assert "--phase 2" not in text
    assert "--st-npy" not in text
    assert "--st-meta" not in text
    for token in (
        "--va-world-mode peer_sync_h6",
        "--wmrm-adep-weight 0",
        "--wmrm-cycle-steps 6",
        "--flow-prefix-steps 6",
        "--wmrm-inject all",
        "--num-workers 0",
        "--visual-world-supervision",
        "--world-split-manifest \"$SPLIT_MANIFEST\"",
    ):
        assert token in launch
    verify = _block(text, "verify_checkpoint(){", "milestone(){")
    assert "config.get('va_world_mode') != 'peer_sync_h6'" in verify
    assert "config.get('action_horizon') != 6" in verify
    assert "arguments.get('num_workers') != 0" in verify
    assert "arguments.get('wmrm_adep_weight') != 0.0" in verify


def test_milestones_are_immutable_and_short_modes_stop_short() -> None:
    text = _text()
    assert "refusing to overwrite immutable output family" in text
    assert "refusing to overwrite immutable milestone" in text
    smoke = text.split("  smoke10)", 1)[1].split(";;", 1)[0]
    pilot = text.split("  pilot300)", 1)[1].split(";;", 1)[0]
    formal = text.split("  20k)", 1)[1].split(";;", 1)[0]
    assert 'run_lineage "$run_id" 10' in smoke
    assert "20000" not in smoke
    assert 'run_lineage "$run_id" 50 300' in pilot
    assert "20000" not in pilot
    assert 'run_lineage "$run_id" "${GATE_STEPS[@]}" 20000' in formal
    assert "no automatic long continuation" in smoke
    assert "STOP at 300" in pilot
