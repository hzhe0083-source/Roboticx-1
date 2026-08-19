from __future__ import annotations

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/continue_mw_hard2_visualmotion_actionrankcap02_v1_formal.sh"


def _text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _block(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_formal_cap02_runner_has_valid_syntax_and_fails_before_launch() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    invalid = subprocess.run(
        ["bash", str(RUNNER), "unexpected-argument"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PY": "/definitely/missing/python"},
    )
    assert invalid.returncode == 2
    assert "usage: bash" in invalid.stderr
    assert "formal exact continuation:" not in invalid.stdout


def test_formal_cap02_runner_pins_exact_source_and_update_interval() -> None:
    text = _text()
    for token in (
        "SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_actionrankcap02_validation256_v1.pt",
        "EXPECTED_SOURCE_REALPATH=/home/ryan/Documents/robot/ORA0/checkpoints/",
        "EXPECTED_SOURCE_SHA256=f75693d9f5ac449e5a6627a03f8ab57e15ff89aad30047719db5a31227b9f334",
        "EXPECTED_EXACT_CONTRACT_SHA256=c1170d61398daf0687521a881340b85a5457e5020d0ce1a58d73406877ce0a52",
        "EXPECTED_TRAINING_CONTRACT_SHA256=732ff34c104a29c3c985e35b75ec149887a8a740e6a58e6cb13f7d67ff7c2cc5",
        "EXPECTED_DATASET_FINGERPRINT=41181fc115389d76abb00f054cbd8b318bd534204a27c08fd8163c928d662e45",
        "EXPECTED_SOURCE_STEP=12330",
        "TARGET_STEP=20000",
        "ADDITIONAL_STEPS=7670",
    ):
        assert token in text
    assert "formal_12330_to_20000" in text


def test_formal_cap02_runner_checks_full_exact_state_and_contracts() -> None:
    verify = _block(_text(), "verify_checkpoint(){", "require_no_active_train(){")
    for token in (
        '"model", "optimizer_state", "sampler_state", "rng_state"',
        'digest(payload["exact_run_contract"]) != expected_exact_digest',
        'digest(payload["training_contract"]) != expected_training_digest',
        'optimizer.get("kind") != "adamw"',
        'set(state_dict) != {"state", "param_groups"}',
        'not state_dict["state"]',
        'not torch.isfinite(value).all()',
        '"sampler_contract_version": 3',
        '"batch_size": 3',
        '"block_batches": 4',
        '"sampling_mode": "balanced"',
        '"active_tasks": [0, 16]',
        '"task_weights": [1.0, 1.0]',
        'set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}',
        'not rng["torch_cuda"]',
        '"wmrm_action_rank_per_sample_cap": 0.2',
        '"wmrm_static_constraint_weight": 2.0',
        '"wmrm_world_weight": 1.0',
        '"wmrm_detach_proposal_stage_state": True',
        '"world_action_rank_stage": "cycle"',
        'ranking.get("per_sample_cap") != 0.2',
        'static.get("weight") != 2.0',
    ):
        assert token in verify
    assert 'verify_checkpoint "$SOURCE" "$EXPECTED_SOURCE_STEP" source' in _text()
    assert 'verify_checkpoint "$SAVE" "$TARGET_STEP" final-rolling' in _text()
    assert 'verify_checkpoint "$archive" "$step" "archive-s${step}"' in _text()


def test_formal_cap02_runner_is_immutable_exclusive_and_machine_safe() -> None:
    text = _text()
    for token in (
        'exec 9>"$LOCK"',
        "flock -n 9",
        "source_sha_before=$(sha \"$SOURCE\")",
        "verify_source_unchanged_on_exit",
        "source immutable trap fired before launch",
        "source immutable trap fired after training",
        "exact train.py process scan: idle",
        "if any(Path(arg).name == 'train.py' for arg in argv):",
        "--query-compute-apps=pid,process_name,used_memory",
        "display processes allowed",
        "free < int(total * 0.85)",
        "available_bytes >= required_free_bytes",
        "EXPECTED_ARCHIVE_COUNT * checkpoint_bytes",
        "checkpoint_bytes + DISK_RESERVE_BYTES",
        "insufficient checkpoint disk space",
        "refusing to overwrite new output family",
        '[[ "$SAVE" != "$SOURCE" ]]',
    ):
        assert token in text
    launch = text.index('"$PY" -u -B train.py')
    assert text.index("verify_checkpoint \"$SOURCE\"") < launch
    assert text.index("require_no_active_train") < launch
    assert text.index("require_idle_gpu") < launch


def test_formal_cap02_runner_archives_every_global_500_boundary() -> None:
    text = _text()
    launch = text[text.index('"$PY" -u -B train.py'):]
    assert "SAVE_EVERY=500" in text
    assert "FIRST_SAVE_STEP=12500" in text
    assert "EXPECTED_ARCHIVE_COUNT=16" in text
    assert '--steps "$ADDITIONAL_STEPS" --save-every "$SAVE_EVERY"' in launch
    assert '--save-step-copies --save "$SAVE" --resume-exact "$SOURCE"' in launch
    assert "--resume-weights" not in text
    assert "--resume-exact-contract-migration" not in text
    assert "cp " not in text
    assert 'for ((step=FIRST_SAVE_STEP; step<=TARGET_STEP; step+=SAVE_EVERY))' in text
    assert 'archive=${SAVE%.pt}_s${step}.pt' in text
    assert 'archive_count == EXPECTED_ARCHIVE_COUNT' in text
    assert '"$(sha "$SAVE")" == "$(sha "${SAVE%.pt}_s${TARGET_STEP}.pt")"' in text
    expected_steps = list(range(12500, 20001, 500))
    assert len(expected_steps) == 16
    assert expected_steps[0] == 12500 and expected_steps[-1] == 20000


def test_formal_cap02_runner_preserves_endpoint_contract_without_migration() -> None:
    launch = _text()[_text().index('"$PY" -u -B train.py'):]
    for token in (
        "--wmrm-world-weight 1.0",
        "--wmrm-static-constraint-weight 2.0",
        "--wmrm-action-rank-per-sample-cap 0.2",
        "--wmrm-detach-proposal-stage-state",
        "--world-action-rank-stage cycle",
        "--wmrm-inject all",
        "--wmrm-target dino",
        "--wmrm-predictor st_blocks",
        "--wmrm-predictor-depth 6",
        "--batch-size 3",
        "--task-sampling balanced",
        "--task-locality-block-batches 4",
        "--flow-tail-weight 0.036",
        "--mtvj-visual-aux-every 10",
    ):
        assert token in launch


def test_formal_cap02_runner_checks_complete_tee_status_before_final_verify() -> None:
    text = _text()
    status = _block(text, 'pipeline_status=("${PIPESTATUS[@]}")', 'printf \'formal continuation complete:')
    for token in (
        '(${#pipeline_status[@]} == 2)',
        '"${pipeline_status[0]}" -eq 0',
        '"${pipeline_status[1]}" -eq 0',
        "log is incomplete",
        '[[ -f "$LOG" && -s "$LOG" ]]',
        'verify_checkpoint "$SAVE" "$TARGET_STEP" final-rolling',
        'verify_checkpoint "$archive" "$step" "archive-s${step}"',
        "checkpoint archive count mismatch",
        "final rolling checkpoint differs",
    ):
        assert token in status
    assert status.index('"${pipeline_status[1]}" -eq 0') < status.index(
        'verify_checkpoint "$SAVE" "$TARGET_STEP" final'
    )
