from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_visualmotion_action_rank_cap_02_validation_256_v1.sh"
ANALYZER = ROOT / "scripts/analyze_visualmotion_action_rank_cap_02_validation.py"
SPEC = importlib.util.spec_from_file_location("rank_cap_analyzer", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _line(
    step: int,
    *,
    grad: float = 2.0,
    flow: float = 0.1,
    world: float = 0.2,
    assembly_static: float = 0.01,
    unlock_static: float = 0.01,
    task: str | None = None,
) -> str:
    tasks = [task] if task is not None else ["assembly-v3", "door-unlock-v3"]
    values = {
        "assembly-v3": assembly_static,
        "door-unlock-v3": unlock_static,
    }
    section = " | ".join(
        f"{name}:all=.1/.2 static={values[name]}/{values[name]}" for name in tasks
    )
    return (
        f"step={step} mode=bidir_va flow={flow} world={world} grad={grad} "
        f"world_task[{section}]"
    )


def _records(**kwargs):
    text = "\n".join(
        _line(step, **kwargs)
        for step in range(MODULE.EXPECTED_START_STEP, MODULE.EXPECTED_END_STEP + 1)
    )
    return MODULE.parse_log_text(text)


def _checkpoint(cap: float | None = 0.2) -> dict:
    return {
        "model": {"weight": 1},
        "optimizer_state": {
            "kind": "adamw",
            "state_dict": {"state": {0: {}}, "param_groups": [{}]},
        },
        "sampler_state": {
            "sampler_contract_version": 3,
            "epoch": 28,
            "batch_cursor": 19,
            "seed": 0,
            "batch_size": 3,
            "block_batches": 4,
            "sampling_mode": "balanced",
            "dataset_fingerprint": MODULE.EXPECTED_DATASET_FINGERPRINT,
            "active_tasks": [0, 16],
            "task_weights": [1.0, 1.0],
        },
        "rng_state": {
            "python": 1,
            "numpy": 1,
            "torch_cpu": 1,
            "torch_cuda": [],
        },
        "exact_run_contract": {
            "contract_version": 1,
            "arguments": {
                "wmrm_action_rank_per_sample_cap": cap,
                "wmrm_static_constraint_weight": 2.0,
                "wmrm_world_weight": 1.0,
                "wmrm_detach_proposal_stage_state": True,
                "world_action_rank_stage": "cycle",
            },
            "model_config": {"wmrm_detach_proposal_stage_state": True},
            "optimizer": {"kind": "adamw"},
        },
        "training_contract": {
            "world_action_ranking": {
                "stage": "rotating_8stage_direct_matched_context",
                "per_sample_cap": cap,
            },
            "world_static_copy_constraint": {"weight": 2.0},
        },
        "exact_resume_version": 2,
        "global_step": MODULE.EXPECTED_END_STEP,
    }


def test_runner_protocol_is_nonlaunching_pinned_and_single_checkpoint() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        "EXPECTED_SOURCE_STEP=12074",
        "TARGET_STEP=12330",
        "UPDATES=256",
        "1f04ed66c9a53a1b2a26eaf14aa6ddd55a081fe762844d4e11b6fbaca9de046c",
        "MIGRATION_ID=wmrm_action_rank_cap_none_to_0_2_v1",
        "--wmrm-action-rank-per-sample-cap 0.2",
        "--wmrm-static-constraint-weight 2.0",
        "--wmrm-world-weight 1.0",
        "--save-every 0",
        "require_no_active_train",
        "require_idle_gpu",
        "available_kib >= 8 * 1024 * 1024",
        'exec 9>"$LOCK"',
    ):
        assert token in text
    assert "--save-step-copies" not in text
    assert "cp " not in text


def test_parser_requires_exact_256_contiguous_steps() -> None:
    records = _records()
    assert len(records) == 256
    assert records[0]["step"] == 12075
    assert records[-1]["step"] == 12330
    short = "\n".join(_line(step) for step in range(12075, 12330))
    with pytest.raises(MODULE.AnalysisError, match="update steps mismatch"):
        MODULE.parse_log_text(short)


def test_parser_rejects_nonfinite_and_error_markers() -> None:
    valid = "\n".join(_line(step) for step in range(12075, 12331))
    with pytest.raises(MODULE.AnalysisError, match="nonfinite"):
        MODULE.parse_log_text(valid.replace("grad=2.0", "grad=nan", 1))
    with pytest.raises(MODULE.AnalysisError, match="error marker"):
        MODULE.parse_log_text(valid + "\nTraceback (most recent call last):")


def test_safe_metrics_pass_all_gates() -> None:
    result = MODULE.analyze_records(_records())
    assert result["decision"] == "PASS"
    assert all(result["gates"].values())


def test_gradient_max_and_repeated_spike_gates_are_independent() -> None:
    records = _records()
    records[0]["grad"] = 51.0
    result = MODULE.analyze_records(records)
    assert not result["gates"]["grad_max_le_50"]
    records = _records()
    records[0]["grad"] = 21.0
    records[15]["grad"] = 21.0
    result = MODULE.analyze_records(records)
    assert not result["gates"]["repeated_grad_over_20_le_1_per_16"]


def test_per_task_final_median_and_early_trend_gates() -> None:
    records = _records()
    for record in records[-32:]:
        record["static_by_task"]["door-unlock-v3"] = 0.026
    result = MODULE.analyze_records(records)
    assert not result["gates"]["per_task_final32_static_median_le_0_025"]

    records = _records()
    for record in records[:32]:
        record["static_by_task"]["assembly-v3"] = 0.004
    for record in records[-32:]:
        record["static_by_task"]["assembly-v3"] = 0.009
    result = MODULE.analyze_records(records)
    assert not result["gates"]["per_task_final32_static_median_le_2x_early"]


def test_raw_emergency_gate_is_per_task_not_task_average() -> None:
    records = _records()
    records[100]["static_by_task"]["assembly-v3"] = 0.051
    records[100]["static_by_task"]["door-unlock-v3"] = 0.0
    result = MODULE.analyze_records(records)
    assert not result["gates"]["raw_static_emergency_max_le_0_05"]


def test_world_and_previous_static2_flow_trend_gates() -> None:
    records = _records()
    for record in records[:32]:
        record["world"] = 0.1
    for record in records[-32:]:
        record["world"] = 0.21
    result = MODULE.analyze_records(records)
    assert not result["gates"]["world_trend_le_2"]

    records = _records(flow=0.32)
    result = MODULE.analyze_records(records)
    assert not result["gates"]["flow_median_le_2x_previous_static2"]
    assert result["observed"]["previous_static2_flow_final32_median"] == 0.1563585


def test_checkpoint_contract_accepts_exact_cap02_payload() -> None:
    MODULE.validate_final_checkpoint(_checkpoint())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update(global_step=12329), "step mismatch"),
        (
            lambda payload: payload["exact_run_contract"]["arguments"].update(
                wmrm_action_rank_per_sample_cap=None
            ),
            "wmrm_action_rank_per_sample_cap mismatch",
        ),
        (
            lambda payload: payload["training_contract"]["world_action_ranking"].update(
                per_sample_cap=None
            ),
            "action-rank cap mismatch",
        ),
        (
            lambda payload: payload["exact_run_contract"]["arguments"].update(
                resume_exact_contract_migration="wmrm_action_rank_cap_none_to_0_2_v1"
            ),
            "migration selector persisted",
        ),
    ],
)
def test_checkpoint_contract_rejects_mismatch(mutate, match: str) -> None:
    payload = _checkpoint()
    mutate(payload)
    with pytest.raises(MODULE.AnalysisError, match=match):
        MODULE.validate_final_checkpoint(payload)
