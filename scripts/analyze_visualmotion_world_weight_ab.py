#!/usr/bin/env python3
"""Analyze the fixed 64-update visual-motion World-weight A/B protocol."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Iterable

CONTRACT = "visualmotion_world_weight_ab_64_v1"
EXPECTED_UPDATES = 64
EXPECTED_TASKS = {"assembly-v3", "door-unlock-v3"}
EXPECTED_EXACT_CONTRACT_VERSION = 1
EXPECTED_DATA_SHA256 = "da16993bd2f336d4f771ed00b479646073cd99bfcfbd6ac0f9d693cbe098d9e1"
EXPECTED_DATA_IDENTITY_ALGORITHM = "full_payload_sha256+referenced_source_sample_v1"
EXPECTED_SAMPLER = {
    "sampler_contract_version": 3,
    "batch_size": 3,
    "block_batches": 4,
    "sampling_mode": "balanced",
    "seed": 0,
    "active_tasks": [0, 16],
    "dataset_fingerprint": "41181fc115389d76abb00f054cbd8b318bd534204a27c08fd8163c928d662e45",
    "task_weights": [1.0, 1.0],
}
EXPECTED_ARGUMENTS = {
    "batch_size": 3,
    "sequence_length": 4,
    "min_sequence_length": 4,
    "task_sampling": "balanced",
    "task_locality_block_batches": 4,
    "seed": 0,
    "flow_steps": 8,
    "flow_prefix_steps": 6,
    "flow_prefix_weight": 1.0,
    "flow_tail_weight": 0.036,
    "wmrm": True,
    "wmrm_inject": "all",
    "wmrm_target": "dino",
    "wmrm_cycle_steps": 6,
    "wmrm_detach_proposal_stage_state": True,
    "wmrm_map_size": 16,
    "wmrm_map_channels": 1024,
    "wmrm_world_grid": 16,
    "wmrm_predictor": "st_blocks",
    "wmrm_predictor_depth": 6,
    "wmrm_predictor_width": 384,
    "wmrm_predictor_heads": 12,
    "max_gradient_norm": None,
}
EXPECTED_MODEL_CONFIG = {
    "action_horizon": 48,
    "num_layers": 8,
    "wmrm": True,
    "wmrm_inject": "all",
    "wmrm_target": "dino",
    "wmrm_cycle_steps": 6,
    "wmrm_detach_proposal_stage_state": True,
    "wmrm_map_size": 16,
    "wmrm_map_channels": 1024,
    "wmrm_world_grid": 16,
    "wmrm_predictor": "st_blocks",
    "wmrm_predictor_depth": 6,
    "wmrm_predictor_width": 384,
    "wmrm_predictor_heads": 12,
}
METRICS = ("grad", "flow", "world", "static")
STEP_RE = re.compile(r"(?:^|\s)step=(?P<step>\d+)(?=\s)")
VALUE_RE = {
    name: re.compile(rf"(?:^|\s){name}=(?P<value>[^\s\]]+)")
    for name in ("grad", "flow", "world")
}
STATIC_RE = re.compile(
    r"\bstatic=(?P<value>[^/\s\]]+)/(?P<copy>[^\s\]]+)"
)
WORLD_TASK_SECTION_RE = re.compile(r"\bworld_task\[(?P<body>[^\]]*)\]")
WORLD_TASK_ENTRY_RE = re.compile(r"(?:^|\s\|\s)(?P<task>[^:\s|]+):(?P<body>.*?)(?=\s\|\s|$)")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NONFINITE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
ERROR_RE = re.compile(
    r"(?:Traceback \(most recent call last\):|"
    r"\b(?:RuntimeError|ValueError|AssertionError|MemoryError|"
    r"FileNotFoundError|CUDA error|ERROR)\s*:)",
    re.IGNORECASE,
)


class AnalysisError(ValueError):
    """The logs or checkpoints cannot support a protocol decision."""


def _require_exact_mapping(mapping: object, expected: dict[str, object], *, label: str) -> None:
    if not isinstance(mapping, dict):
        raise AnalysisError(f"{label} must be a mapping")
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise AnalysisError(f"{label}.{key}={mapping.get(key)!r} != {value!r}")


def validate_checkpoint_payload(
    payload: object, *, expected_step: int, expected_weight: float, label: str
) -> None:
    """Fail closed on every state and semantic required by the paired run."""
    if not isinstance(payload, dict):
        raise AnalysisError(f"{label} checkpoint payload must be a mapping")
    required = {
        "model", "optimizer_state", "sampler_state", "rng_state",
        "exact_run_contract", "exact_resume_version", "global_step",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise AnalysisError(f"{label} checkpoint missing required keys: {missing}")
    model_state = payload["model"]
    if not isinstance(model_state, dict) or not model_state:
        raise AnalysisError(f"{label} model state must be a nonempty mapping")
    if payload["exact_resume_version"] != 2 or payload["global_step"] != expected_step:
        raise AnalysisError(f"{label} exact-resume version or global step mismatch")

    optimizer_state = payload["optimizer_state"]
    if not isinstance(optimizer_state, dict) or optimizer_state.get("kind") != "adamw":
        raise AnalysisError(f"{label} optimizer kind must be adamw")
    state_dict = optimizer_state.get("state_dict")
    if not isinstance(state_dict, dict) or set(state_dict) != {"state", "param_groups"}:
        raise AnalysisError(f"{label} optimizer state_dict is incomplete")
    if not isinstance(state_dict["state"], dict) or not state_dict["state"]:
        raise AnalysisError(f"{label} AdamW state is empty")
    if not isinstance(state_dict["param_groups"], list) or not state_dict["param_groups"]:
        raise AnalysisError(f"{label} AdamW param_groups are empty")

    sampler = payload["sampler_state"]
    _require_exact_mapping(sampler, EXPECTED_SAMPLER, label=f"{label}.sampler_state")
    for key in ("epoch", "batch_cursor"):
        if type(sampler.get(key)) is not int or sampler[key] < 0:
            raise AnalysisError(f"{label}.sampler_state.{key} must be a nonnegative integer")

    rng = payload["rng_state"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise AnalysisError(f"{label} RNG state is incomplete")
    if any(rng[key] is None for key in rng):
        raise AnalysisError(f"{label} RNG state contains null components")

    contract = payload["exact_run_contract"]
    if not isinstance(contract, dict) or contract.get("contract_version") != EXPECTED_EXACT_CONTRACT_VERSION:
        raise AnalysisError(f"{label} exact contract version mismatch")
    identity = contract.get("data_identity")
    _require_exact_mapping(
        identity,
        {"full_file_sha256": EXPECTED_DATA_SHA256, "identity_algorithm": EXPECTED_DATA_IDENTITY_ALGORITHM},
        label=f"{label}.exact_run_contract.data_identity",
    )
    arguments = contract.get("arguments")
    _require_exact_mapping(arguments, EXPECTED_ARGUMENTS, label=f"{label}.exact_run_contract.arguments")
    if type(arguments.get("wmrm_world_weight")) is not float or arguments["wmrm_world_weight"] != expected_weight:
        raise AnalysisError(f"{label} endpoint World weight mismatch")
    if "resume_exact_contract_migration" in arguments:
        raise AnalysisError(f"{label} persisted operational migration selector")
    _require_exact_mapping(
        contract.get("model_config"), EXPECTED_MODEL_CONFIG,
        label=f"{label}.exact_run_contract.model_config",
    )
    optimizer_contract = contract.get("optimizer")
    if not isinstance(optimizer_contract, dict) or optimizer_contract.get("kind") != "adamw":
        raise AnalysisError(f"{label} exact optimizer contract must be adamw")


def _number(raw: str, *, field: str, step: int) -> float:
    try:
        value = float(raw.rstrip(",;"))
    except ValueError as exc:
        raise AnalysisError(f"step {step}: invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"step {step}: nonfinite {field}={raw!r}")
    return value


def parse_log_text(
    text: str, *, expected_start_step: int = 12011, expected_updates: int = EXPECTED_UPDATES
) -> list[dict[str, float | int]]:
    """Extract strict, consecutive update records while tolerating unrelated log lines."""
    clean = ANSI_RE.sub("", text.replace("\r", "\n"))
    if ERROR_RE.search(clean):
        raise AnalysisError("trainer error marker found in log")
    if NONFINITE_RE.search(clean):
        raise AnalysisError("nonfinite token found in log")

    records: list[dict[str, float | int]] = []
    seen: set[int] = set()
    for line in clean.splitlines():
        step_match = STEP_RE.search(line)
        if step_match is None or " mode=" not in line:
            continue
        step = int(step_match.group("step"))
        values: dict[str, float | int] = {"step": step}
        for field, pattern in VALUE_RE.items():
            match = pattern.search(line)
            if match is None:
                raise AnalysisError(f"step {step}: missing {field} metric")
            values[field] = _number(match.group("value"), field=field, step=step)
        sections = list(WORLD_TASK_SECTION_RE.finditer(line))
        if len(sections) != 1:
            raise AnalysisError(
                f"step {step}: expected exactly one world_task section, found {len(sections)}"
            )
        entries = list(WORLD_TASK_ENTRY_RE.finditer(sections[0].group("body")))
        if not entries:
            raise AnalysisError(f"step {step}: empty world_task section")
        line_tasks: set[str] = set()
        static_values: list[float] = []
        for entry in entries:
            task = entry.group("task")
            if task not in EXPECTED_TASKS:
                raise AnalysisError(f"step {step}: invalid world_task task {task!r}")
            if task in line_tasks:
                raise AnalysisError(f"step {step}: duplicate world_task task {task!r}")
            line_tasks.add(task)
            static_matches = list(STATIC_RE.finditer(entry.group("body")))
            if len(static_matches) != 1:
                raise AnalysisError(
                    f"step {step}: task {task} must contain exactly one static=current/copy token"
                )
            static_values.append(
                _number(static_matches[0].group("value"), field="static", step=step)
            )
            _number(static_matches[0].group("copy"), field="copy_static", step=step)
        values["static"] = float(statistics.fmean(static_values))
        values["tasks"] = sorted(line_tasks)
        if NONFINITE_RE.search(line):
            raise AnalysisError(f"step {step}: nonfinite token found in metric line")
        if step in seen:
            raise AnalysisError(f"duplicate update record for step {step}")
        seen.add(step)
        records.append(values)

    expected_steps = list(range(expected_start_step, expected_start_step + expected_updates))
    observed_steps = [int(record["step"]) for record in records]
    observed_tasks = {
        task for record in records for task in record.get("tasks", [])
    }
    if observed_tasks != EXPECTED_TASKS:
        raise AnalysisError(
            f"64-step arm must contain both tasks {sorted(EXPECTED_TASKS)}, "
            f"observed {sorted(observed_tasks)}"
        )
    if observed_steps != expected_steps:
        raise AnalysisError(
            f"update steps mismatch: expected {expected_steps[0]}..{expected_steps[-1]} "
            f"({expected_updates} records), observed {observed_steps[:3]}..{observed_steps[-3:]} "
            f"({len(observed_steps)} records)"
        )
    return records


def parse_log(path: Path, **kwargs: int) -> list[dict[str, float | int]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AnalysisError(f"cannot read log {path}: {exc}") from exc
    return parse_log_text(text, **kwargs)


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot summarize an empty metric")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "median": float(statistics.median(values)),
        "p95": _quantile(values, 0.95),
    }


def _relative_improvement(a: float, b: float) -> float:
    if a == 0.0:
        return 0.0 if b >= a else math.inf
    return (a - b) / a


def analyze_records(
    a_records: list[dict[str, float | int]],
    b_records: list[dict[str, float | int]],
) -> dict[str, object]:
    if len(a_records) != EXPECTED_UPDATES or len(b_records) != EXPECTED_UPDATES:
        raise AnalysisError("analysis requires exactly 64 records per arm")
    if [item["step"] for item in a_records] != [item["step"] for item in b_records]:
        raise AnalysisError("A/B update steps are not paired")

    arm_values: dict[str, dict[str, list[float]]] = {}
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for arm, records in (("A", a_records), ("B", b_records)):
        arm_values[arm] = {
            metric: [float(record[metric]) for record in records] for metric in METRICS
        }
        summaries[arm] = {
            metric: _summary(values) for metric, values in arm_values[arm].items()
        }

    a = arm_values["A"]
    b = arm_values["B"]
    max_over_20_in_16 = max(
        sum(value > 20.0 for value in b["grad"][start : start + 16])
        for start in range(EXPECTED_UPDATES - 16 + 1)
    )
    b_final32_flow_median = float(statistics.median(b["flow"][-32:]))
    a_final32_flow_median = float(statistics.median(a["flow"][-32:]))
    flow_limit = a_final32_flow_median + max(0.05 * a_final32_flow_median, 0.02)
    b_last32_static = b["static"][-32:]
    b_world_first16 = float(statistics.median(b["world"][:16]))
    b_world_last16 = float(statistics.median(b["world"][-16:]))
    world_ratio = (
        b_world_last16 / b_world_first16
        if b_world_first16 > 0.0
        else (0.0 if b_world_last16 == 0.0 else math.inf)
    )

    improvements = {
        "grad_median_25pct": _relative_improvement(
            summaries["A"]["grad"]["median"], summaries["B"]["grad"]["median"]
        ),
        "grad_p95_30pct": _relative_improvement(
            summaries["A"]["grad"]["p95"], summaries["B"]["grad"]["p95"]
        ),
        "static_median_25pct": _relative_improvement(
            float(statistics.median(a["static"][-32:])),
            float(statistics.median(b_last32_static)),
        ),
        "world_median_20pct": _relative_improvement(
            float(statistics.median(a["world"][-32:])),
            float(statistics.median(b["world"][-32:])),
        ),
    }
    improvement_passes = {
        "grad_median_25pct": improvements["grad_median_25pct"] >= 0.25,
        "grad_p95_30pct": improvements["grad_p95_30pct"] >= 0.30,
        "static_median_25pct": improvements["static_median_25pct"] >= 0.25,
        "world_median_20pct": improvements["world_median_20pct"] >= 0.20,
    }
    gates = {
        "finite_and_error_free": True,
        "b_grad_max_le_50": summaries["B"]["grad"]["max"] <= 50.0,
        "b_any_16_grad_over_20_count_le_1": max_over_20_in_16 <= 1,
        "b_final32_flow_median_within_a_tolerance": b_final32_flow_median <= flow_limit,
        "b_last32_static_max_le_0_02": max(b_last32_static) <= 0.02,
        "b_world_last16_over_first16_le_2": world_ratio <= 2.0,
        "at_least_one_approved_improvement": any(improvement_passes.values()),
    }
    passed = all(gates.values())
    return {
        "contract": CONTRACT,
        "decision": "PASS" if passed else "NO-GO",
        "passed": passed,
        "thresholds": {
            "b_grad_max": 50.0,
            "b_grad_over_20_max_count_in_any_16": 1,
            "b_final32_flow_median_vs_a": "A + max(5%, 0.02)",
            "b_last32_static_max": 0.02,
            "b_world_last16_over_first16_max": 2.0,
            "improvement_any": {
                "grad_median": 0.25,
                "grad_p95": 0.30,
                "static_final32_median": 0.25,
                "world_final32_median": 0.20,
            },
        },
        "gates": gates,
        "observed": {
            "b_grad_max": summaries["B"]["grad"]["max"],
            "b_grad_over_20_max_count_in_any_16": max_over_20_in_16,
            "a_final32_flow_median": a_final32_flow_median,
            "b_final32_flow_median": b_final32_flow_median,
            "b_final32_flow_limit": flow_limit,
            "b_last32_static_max": max(b_last32_static),
            "b_world_first16_median": b_world_first16,
            "b_world_last16_median": b_world_last16,
            "b_world_last16_over_first16": world_ratio,
            "relative_improvements": improvements,
            "improvement_passes": improvement_passes,
        },
        "summaries": summaries,
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _arm_report(
    arm: str, log: Path, records: list[dict[str, float | int]]
) -> dict[str, object]:
    return {
        "contract": CONTRACT,
        "arm": arm,
        "log": str(log.resolve()),
        "start_step": records[0]["step"],
        "end_step": records[-1]["step"],
        "updates": len(records),
        "metrics": {
            metric: _summary([float(record[metric]) for record in records])
            for metric in METRICS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-log", type=Path, required=True)
    parser.add_argument("--b-log", type=Path, required=True)
    parser.add_argument("--a-report", type=Path, required=True)
    parser.add_argument("--b-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=12011)
    args = parser.parse_args(argv)
    try:
        a_records = parse_log(args.a_log, expected_start_step=args.start_step)
        b_records = parse_log(args.b_log, expected_start_step=args.start_step)
        result = analyze_records(a_records, b_records)
        result["inputs"] = {
            "A": str(args.a_log.resolve()),
            "B": str(args.b_log.resolve()),
        }
        atomic_write_json(args.a_report, _arm_report("A", args.a_log, a_records))
        atomic_write_json(args.b_report, _arm_report("B", args.b_log, b_records))
        atomic_write_json(args.output, result)
    except (AnalysisError, OSError, TypeError, ValueError) as exc:
        print(f"FATAL: {exc}", file=os.sys.stderr)
        return 1
    print(f"{result['decision']}: {args.output}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
