from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_visualmotion_world_weight_ab_64_v1.sh"
ANALYZER = ROOT / "scripts" / "analyze_visualmotion_world_weight_ab.py"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("world_weight_ab_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analyzer()


def _line(
    step: int,
    *,
    grad: float,
    flow: float,
    world: float,
    static: float,
    task: str | None = None,
) -> str:
    task = task or ("assembly-v3" if step % 2 else "door-unlock-v3")
    return (
        f"step={step} mode=bidir_va contract=single task={task} "
        f"loss=0.4 flow={flow:.6g} world={world:.6g} grad={grad:.6g} "
        f"world_task[{task}:all=0.1/0.2 static={static:.6g}/0.01 n=4]"
    )


def _records(*, grad: float, flow: float, world: float, static: float):
    return [
        {"step": step, "grad": grad, "flow": flow, "world": world, "static": static, "tasks": ["assembly-v3", "door-unlock-v3"]}
        for step in range(12011, 12075)
    ]


def test_runner_static_protocol_is_sequential_exact_and_fail_closed() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)], cwd=ROOT, capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "EXPECTED_SOURCE_STEP=12010" in text
    assert "TARGET_STEP=12074" in text and "UPDATES=64" in text
    assert (
        "EXPECTED_SOURCE_SHA256=f580caa4c1588b2a9807f9b0ab746ac54259eaaa482cea16ce5001c30a382f11"
        in text
    )
    assert "MIGRATION_ID=wmrm_world_weight_1_to_0_5_v1" in text
    assert 'exec 9>"$LOCK"' in text and "flock -n 9" in text
    assert "available_kib >= 14 * 1024 * 1024" in text
    assert "require_no_active_train()" in text
    assert 'Path(arg).name == "train.py"' in text
    assert "--query-compute-apps=pid,process_name,used_memory" in text
    assert "display processes allowed" in text
    assert "--save-step-copies" not in text
    assert '--steps "$UPDATES" --save-every 0' in text
    assert '--save "$save" --resume-exact "$SOURCE"' in text
    assert text.index("require_no_active_train\nrequire_idle_gpu\nrun_arm A") > text.index("verify_source\n")
    assert text.index("verify_final A") < text.index("require_no_active_train\nrequire_idle_gpu\nrun_arm B")
    assert text.count("require_no_active_train\nrequire_idle_gpu\nrun_arm") == 2
    assert '--resume-exact-contract-migration "$MIGRATION_ID"' in text
    common = text.split("COMMON=(", 1)[1].split("\n)", 1)[0]
    assert "--wmrm-world-weight" not in common
    assert "--resume-exact-contract-migration" not in common
    for exact_arg in (
        "--world-action-rank-stage cycle",
        "--wmrm-detach-proposal-stage-state",
        "--task-sampling balanced --task-locality-block-batches 4",
        "--batch-size 3",
        "--seed 0",
        "--flow-tail-weight 0.036",
        "--mtvj-visual-aux-every 10",
    ):
        assert exact_arg in common
    assert "source hash trap fired before arm" in text
    assert "source hash trap fired after arm" in text
    assert 'exit "$analysis_status"' in text


def test_parser_tolerates_noise_ansi_and_static_target() -> None:
    lines = ["startup banner", "\x1b[32mready\x1b[0m"]
    lines.extend(
        _line(step, grad=2.0, flow=0.2, world=0.3, static=0.01)
        for step in range(12011, 12075)
    )
    lines.insert(20, "step=64 global_step=12074 periodic checkpoint saved")
    records = ANALYSIS.parse_log_text("\r".join(lines))
    assert len(records) == 64
    assert records[0] == {
        "step": 12011,
        "grad": 2.0,
        "flow": 0.2,
        "world": 0.3,
        "static": 0.01,
        "tasks": ["assembly-v3"],
    }
    assert records[-1]["step"] == 12074


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda lines: lines.pop(10), "update steps mismatch"),
        (lambda lines: lines.append(lines[-1]), "duplicate update"),
        (lambda lines: lines.__setitem__(5, lines[5].replace(" grad=2", " grad=nan")), "nonfinite"),
        (lambda lines: lines.append("RuntimeError: CUDA exploded"), "error marker"),
        (lambda lines: lines.__setitem__(7, lines[7].replace(" world=0.3", "")), "missing world"),
    ],
)
def test_parser_rejects_incomplete_duplicate_nonfinite_and_errors(mutation, match) -> None:
    lines = [
        _line(step, grad=2.0, flow=0.2, world=0.3, static=0.01)
        for step in range(12011, 12075)
    ]
    mutation(lines)
    with pytest.raises(ANALYSIS.AnalysisError, match=match):
        ANALYSIS.parse_log_text("\n".join(lines))


def test_parser_rejects_missing_second_task_and_duplicate_section() -> None:
    lines = [_line(step, grad=2.0, flow=0.2, world=0.3, static=0.01, task="assembly-v3") for step in range(12011, 12075)]
    with pytest.raises(ANALYSIS.AnalysisError, match="both tasks"):
        ANALYSIS.parse_log_text("\n".join(lines), expected_updates=64)
    lines = [_line(step, grad=2.0, flow=0.2, world=0.3, static=0.01) for step in range(12011, 12075)]
    lines[0] += " world_task[door-unlock-v3:all=0.1/0.2 static=0.01/0.01 n=4]"
    with pytest.raises(ANALYSIS.AnalysisError, match="exactly one world_task"):
        ANALYSIS.parse_log_text("\n".join(lines), expected_updates=64)


def test_parser_rejects_unknown_task_and_malformed_static_token() -> None:
    lines = [_line(step, grad=2.0, flow=0.2, world=0.3, static=0.01) for step in range(12011, 12075)]
    lines[0] = lines[0].replace("assembly-v3", "button-press-v3")
    with pytest.raises(ANALYSIS.AnalysisError, match="invalid world_task task"):
        ANALYSIS.parse_log_text("\n".join(lines))
    lines = [_line(step, grad=2.0, flow=0.2, world=0.3, static=0.01) for step in range(12011, 12075)]
    lines[0] = lines[0].replace("static=0.01/0.01", "static=0.01")
    with pytest.raises(ANALYSIS.AnalysisError, match="static=current/copy"):
        ANALYSIS.parse_log_text("\n".join(lines))


def test_checkpoint_validator_rejects_missing_contract_fields() -> None:
    payload = {
        "model": {"weight": 1},
        "optimizer_state": {"kind": "adamw", "state_dict": {"state": {1: {}}, "param_groups": [{}]}},
        "sampler_state": {**ANALYSIS.EXPECTED_SAMPLER, "epoch": 1, "batch_cursor": 2},
        "rng_state": {key: object() for key in ("python", "numpy", "torch_cpu", "torch_cuda")},
        "exact_run_contract": {"contract_version": 1, "data_identity": {"full_file_sha256": ANALYSIS.EXPECTED_DATA_SHA256, "identity_algorithm": ANALYSIS.EXPECTED_DATA_IDENTITY_ALGORITHM}, "arguments": {**ANALYSIS.EXPECTED_ARGUMENTS, "wmrm_world_weight": 1.0}, "model_config": ANALYSIS.EXPECTED_MODEL_CONFIG, "optimizer": {"kind": "adamw"}},
        "exact_resume_version": 2,
        "global_step": 12010,
    }
    with pytest.raises(ANALYSIS.AnalysisError, match="sampler_state.dataset_fingerprint"):
        bad = copy.deepcopy(payload)
        bad["sampler_state"].pop("dataset_fingerprint")
        ANALYSIS.validate_checkpoint_payload(bad, expected_step=12010, expected_weight=1.0, label="source")
    with pytest.raises(ANALYSIS.AnalysisError, match="exact optimizer contract"):
        bad = copy.deepcopy(payload)
        bad["exact_run_contract"]["optimizer"] = {"kind": "sgd"}
        ANALYSIS.validate_checkpoint_payload(bad, expected_step=12010, expected_weight=1.0, label="source")


def test_thresholds_pass_with_one_approved_improvement() -> None:
    a = _records(grad=20.0, flow=0.20, world=0.30, static=0.012)
    b = _records(grad=14.0, flow=0.21, world=0.28, static=0.011)
    result = ANALYSIS.analyze_records(a, b)
    assert result["decision"] == "PASS"
    assert result["gates"]["at_least_one_approved_improvement"] is True
    assert result["observed"]["improvement_passes"]["grad_median_25pct"] is True


def test_each_hard_gate_and_improvement_gate_can_force_no_go() -> None:
    a = _records(grad=10.0, flow=0.20, world=0.30, static=0.01)

    cases = []
    b = _records(grad=5.0, flow=0.20, world=0.20, static=0.01)
    b[0]["grad"] = 50.01
    cases.append((b, "b_grad_max_le_50"))

    b = _records(grad=5.0, flow=0.20, world=0.20, static=0.01)
    b[0]["grad"] = b[1]["grad"] = 21.0
    cases.append((b, "b_any_16_grad_over_20_count_le_1"))

    b = _records(grad=5.0, flow=0.20, world=0.20, static=0.01)
    for item in b[-32:]:
        item["flow"] = 0.231
    cases.append((b, "b_final32_flow_median_within_a_tolerance"))

    b = _records(grad=5.0, flow=0.20, world=0.20, static=0.01)
    b[-1]["static"] = 0.02001
    cases.append((b, "b_last32_static_max_le_0_02"))

    b = _records(grad=5.0, flow=0.20, world=0.20, static=0.01)
    for item in b[:16]:
        item["world"] = 0.10
    for item in b[-16:]:
        item["world"] = 0.201
    cases.append((b, "b_world_last16_over_first16_le_2"))

    b = _records(grad=9.0, flow=0.20, world=0.27, static=0.009)
    cases.append((b, "at_least_one_approved_improvement"))

    for candidate, gate in cases:
        result = ANALYSIS.analyze_records(a, candidate)
        assert result["decision"] == "NO-GO", gate
        assert result["gates"][gate] is False


def test_cli_writes_atomic_json_and_returns_zero_or_two(tmp_path: Path) -> None:
    a_log = tmp_path / "a.log"
    b_log = tmp_path / "b.log"
    a_log.write_text(
        "\n".join(_line(s, grad=20, flow=0.2, world=0.3, static=0.012) for s in range(12011, 12075)),
        encoding="utf-8",
    )
    b_log.write_text(
        "\n".join(_line(s, grad=14, flow=0.21, world=0.25, static=0.01) for s in range(12011, 12075)),
        encoding="utf-8",
    )
    a_report, b_report, pair = (tmp_path / name for name in ("a.json", "b.json", "pair.json"))
    status = ANALYSIS.main(
        [
            "--a-log", str(a_log), "--b-log", str(b_log),
            "--a-report", str(a_report), "--b-report", str(b_report),
            "--output", str(pair),
        ]
    )
    assert status == 0
    assert json.loads(pair.read_text())["passed"] is True
    assert not list(tmp_path.glob("*.tmp"))

    b_log.write_text(
        "\n".join(_line(s, grad=10, flow=0.3, world=0.3, static=0.03) for s in range(12011, 12075)),
        encoding="utf-8",
    )
    assert ANALYSIS.main(
        [
            "--a-log", str(a_log), "--b-log", str(b_log),
            "--a-report", str(a_report), "--b-report", str(b_report),
            "--output", str(pair),
        ]
    ) == 2
    assert json.loads(pair.read_text())["decision"] == "NO-GO"
