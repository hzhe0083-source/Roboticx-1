#!/usr/bin/env python3
"""Fail-closed analyzer for the 256-update action-rank cap=0.2 validation."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

CONTRACT = "visualmotion_action_rank_cap_02_validation_256_v1"
EXPECTED_START_STEP = 12075
EXPECTED_END_STEP = 12330
EXPECTED_UPDATES = 256
EXPECTED_TASKS = {"assembly-v3", "door-unlock-v3"}
EXPECTED_DATASET_FINGERPRINT = (
    "41181fc115389d76abb00f054cbd8b318bd534204a27c08fd8163c928d662e45"
)
# Measured by the immediately preceding static2 64-update analyzer report.
PREVIOUS_STATIC2_FLOW_FINAL32_MEDIAN = 0.1563585

STEP_RE = re.compile(r"(?:^|\s)step=(?P<step>\d+)(?=\s)")
VALUE_RE = {
    name: re.compile(rf"(?:^|\s){name}=(?P<value>[^\s\]]+)")
    for name in ("grad", "flow", "world")
}
STATIC_RE = re.compile(r"\bstatic=(?P<value>[^/\s\]]+)/(?P<copy>[^\s\]]+)")
SECTION_RE = re.compile(r"\bworld_task\[(?P<body>[^\]]*)\]")
ENTRY_RE = re.compile(
    r"(?:^|\s\|\s)(?P<task>[^:\s|]+):(?P<body>.*?)(?=\s\|\s|$)"
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\):|\b(?:RuntimeError|ValueError|"
    r"AssertionError|MemoryError|CUDA error|ERROR):",
    re.IGNORECASE,
)
NONFINITE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


class AnalysisError(ValueError):
    """The validation evidence is missing, malformed, or unsafe."""


def _number(raw: str, field: str, step: int) -> float:
    try:
        value = float(raw.rstrip(",;"))
    except ValueError as exc:
        raise AnalysisError(f"step {step}: invalid {field}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"step {step}: nonfinite {field}")
    return value


def parse_log_text(text: str) -> list[dict[str, Any]]:
    """Parse exactly the 256 contiguous optimizer-update records."""
    text = ANSI_RE.sub("", text.replace("\r", "\n"))
    if ERROR_RE.search(text):
        raise AnalysisError("trainer error marker found in log")
    if NONFINITE_RE.search(text):
        raise AnalysisError("nonfinite token found in log")

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if match is None or " mode=" not in line:
            continue
        step = int(match.group("step"))
        record: dict[str, Any] = {"step": step}
        for field, pattern in VALUE_RE.items():
            found = pattern.search(line)
            if found is None:
                raise AnalysisError(f"step {step}: missing {field}")
            record[field] = _number(found.group("value"), field, step)

        sections = list(SECTION_RE.finditer(line))
        if len(sections) != 1:
            raise AnalysisError(f"step {step}: expected exactly one world_task section")
        static_by_task: dict[str, float] = {}
        for entry in ENTRY_RE.finditer(sections[0].group("body")):
            task = entry.group("task")
            if task not in EXPECTED_TASKS or task in static_by_task:
                raise AnalysisError(f"step {step}: invalid or duplicate task {task!r}")
            static_matches = list(STATIC_RE.finditer(entry.group("body")))
            if len(static_matches) != 1:
                raise AnalysisError(
                    f"step {step}: task {task} needs one static=current/copy metric"
                )
            static_by_task[task] = _number(
                static_matches[0].group("value"), "static", step
            )
            _number(static_matches[0].group("copy"), "copy_static", step)
        if not static_by_task:
            raise AnalysisError(f"step {step}: world_task section has no task metrics")
        if step in seen:
            raise AnalysisError(f"duplicate update record for step {step}")
        seen.add(step)
        record["static_by_task"] = static_by_task
        records.append(record)

    expected_steps = list(range(EXPECTED_START_STEP, EXPECTED_END_STEP + 1))
    if [record["step"] for record in records] != expected_steps:
        raise AnalysisError("update steps mismatch")
    observed_tasks = {
        task for record in records for task in record["static_by_task"]
    }
    if observed_tasks != EXPECTED_TASKS:
        raise AnalysisError("updates must collectively contain both expected tasks")
    return records


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{description} must be a mapping")
    return value


def validate_final_checkpoint(
    payload: object, *, expected_step: int = EXPECTED_END_STEP
) -> None:
    """Verify exact-resume and migrated semantic contracts on the final artifact."""
    checkpoint = _require_mapping(payload, "final checkpoint")
    required = {
        "model",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "exact_run_contract",
        "exact_resume_version",
        "global_step",
        "training_contract",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise AnalysisError(f"final checkpoint missing required keys: {missing}")
    if checkpoint["global_step"] != expected_step:
        raise AnalysisError("final checkpoint step mismatch")
    if checkpoint["exact_resume_version"] != 2:
        raise AnalysisError("final checkpoint exact-resume version mismatch")
    model = _require_mapping(checkpoint["model"], "model state")
    if not model:
        raise AnalysisError("model state must be nonempty")

    optimizer = _require_mapping(checkpoint["optimizer_state"], "optimizer state")
    if optimizer.get("kind") != "adamw":
        raise AnalysisError("final optimizer must be AdamW")
    optimizer_state = _require_mapping(optimizer.get("state_dict"), "AdamW state_dict")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise AnalysisError("final AdamW state_dict is incomplete")
    if not optimizer_state["state"] or not optimizer_state["param_groups"]:
        raise AnalysisError("final AdamW state must be nonempty")

    sampler = _require_mapping(checkpoint["sampler_state"], "sampler state")
    expected_sampler = {
        "sampler_contract_version": 3,
        "seed": 0,
        "batch_size": 3,
        "block_batches": 4,
        "sampling_mode": "balanced",
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "active_tasks": [0, 16],
        "task_weights": [1.0, 1.0],
    }
    for key, expected in expected_sampler.items():
        if sampler.get(key) != expected:
            raise AnalysisError(f"final sampler {key} mismatch")
    for key in ("epoch", "batch_cursor"):
        if type(sampler.get(key)) is not int or sampler[key] < 0:
            raise AnalysisError(f"final sampler {key} invalid")

    rng = _require_mapping(checkpoint["rng_state"], "RNG state")
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise AnalysisError("final RNG state keys are incomplete")
    if any(value is None for value in rng.values()):
        raise AnalysisError("final RNG state contains null state")

    exact = _require_mapping(checkpoint["exact_run_contract"], "exact contract")
    if exact.get("contract_version") != 1:
        raise AnalysisError("final exact contract version mismatch")
    arguments = _require_mapping(exact.get("arguments"), "exact contract arguments")
    expected_arguments = {
        "wmrm_action_rank_per_sample_cap": 0.2,
        "wmrm_static_constraint_weight": 2.0,
        "wmrm_world_weight": 1.0,
        "wmrm_detach_proposal_stage_state": True,
        "world_action_rank_stage": "cycle",
    }
    for key, expected in expected_arguments.items():
        if arguments.get(key) != expected:
            raise AnalysisError(f"final exact contract {key} mismatch")
    if "resume_exact_contract_migration" in arguments:
        raise AnalysisError("operational migration selector persisted in exact contract")
    model_config = _require_mapping(exact.get("model_config"), "model contract")
    if model_config.get("wmrm_detach_proposal_stage_state") is not True:
        raise AnalysisError("final model contract detach flag mismatch")
    exact_optimizer = _require_mapping(exact.get("optimizer"), "optimizer contract")
    if exact_optimizer.get("kind") != "adamw":
        raise AnalysisError("final exact optimizer contract must be AdamW")

    training = _require_mapping(checkpoint["training_contract"], "training contract")
    ranking = _require_mapping(training.get("world_action_ranking"), "ranking contract")
    if ranking.get("per_sample_cap") != 0.2:
        raise AnalysisError("training contract action-rank cap mismatch")
    if ranking.get("stage") != "rotating_8stage_direct_matched_context":
        raise AnalysisError("training contract action-rank stage mismatch")
    static_contract = _require_mapping(
        training.get("world_static_copy_constraint"), "static-copy contract"
    )
    if static_contract.get("weight") != 2.0:
        raise AnalysisError("training contract static weight mismatch")


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != EXPECTED_UPDATES:
        raise AnalysisError("analysis requires exactly 256 records")
    grad = [float(record["grad"]) for record in records]
    flow = [float(record["flow"]) for record in records]
    world = [float(record["world"]) for record in records]
    if not all(math.isfinite(value) for value in grad + flow + world):
        raise AnalysisError("records contain nonfinite aggregate metrics")

    repeated_grad = max(
        sum(value > 20.0 for value in grad[start : start + 16])
        for start in range(len(grad) - 15)
    )
    task_values = {
        task: [
            float(record["static_by_task"][task])
            for record in records
            if task in record["static_by_task"]
        ]
        for task in EXPECTED_TASKS
    }
    if any(not values for values in task_values.values()):
        raise AnalysisError("one or more tasks have no static observations")
    early_median = {
        task: statistics.median(values[:32]) for task, values in task_values.items()
    }
    final_median = {
        task: statistics.median(values[-32:]) for task, values in task_values.items()
    }
    static_ratio = {
        task: (
            final_median[task] / early_median[task]
            if early_median[task] > 0.0
            else (0.0 if final_median[task] == 0.0 else math.inf)
        )
        for task in EXPECTED_TASKS
    }
    early_world = statistics.median(world[:32])
    final_world = statistics.median(world[-32:])
    world_ratio = (
        final_world / early_world
        if early_world > 0.0
        else (0.0 if final_world == 0.0 else math.inf)
    )
    flow_final32_median = statistics.median(flow[-32:])
    raw_static_max = max(
        value
        for record in records
        for value in record["static_by_task"].values()
    )

    gates = {
        "finite_and_error_free": True,
        "grad_max_le_50": max(grad) <= 50.0,
        "repeated_grad_over_20_le_1_per_16": repeated_grad <= 1,
        "per_task_final32_static_median_le_0_025": all(
            value <= 0.025 for value in final_median.values()
        ),
        "per_task_final32_static_median_le_2x_early": all(
            value <= 2.0 for value in static_ratio.values()
        ),
        "raw_static_emergency_max_le_0_05": raw_static_max <= 0.05,
        "world_trend_le_2": world_ratio <= 2.0,
        "flow_final32_finite": all(math.isfinite(value) for value in flow[-32:]),
        "flow_median_le_2x_previous_static2": (
            flow_final32_median <= 2.0 * PREVIOUS_STATIC2_FLOW_FINAL32_MEDIAN
        ),
    }
    passed = all(gates.values())
    return {
        "contract": CONTRACT,
        "decision": "PASS" if passed else "NO-GO",
        "passed": passed,
        "gates": gates,
        "observed": {
            "grad_max": max(grad),
            "grad_over_20_max_count_in_16": repeated_grad,
            "early32_static_median_by_task": early_median,
            "final32_static_median_by_task": final_median,
            "final_to_early_static_ratio_by_task": static_ratio,
            "raw_static_max": raw_static_max,
            "world_final32_over_early32": world_ratio,
            "flow_final32_median": flow_final32_median,
            "previous_static2_flow_final32_median": (
                PREVIOUS_STATIC2_FLOW_FINAL32_MEDIAN
            ),
        },
        "updates": len(records),
    }


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import torch

        validate_final_checkpoint(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
        result = analyze_records(
            parse_log_text(args.log.read_text(encoding="utf-8", errors="replace"))
        )
        result["log"] = str(args.log.resolve())
        result["checkpoint"] = str(args.checkpoint.resolve())
        _atomic_write(args.report, result)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: {exc}", file=os.sys.stderr)
        return 1
    print(f"{result['decision']}: {args.report}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
