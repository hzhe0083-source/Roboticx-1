#!/usr/bin/env python3
"""Fail-closed analyzer for the 64-update static World-weight=2 validation."""
from __future__ import annotations
import argparse, json, math, os, re, statistics
from pathlib import Path

CONTRACT = "visualmotion_static_weight_2_validation_64_v1"
EXPECTED_UPDATES = 64
EXPECTED_TASKS = {"assembly-v3", "door-unlock-v3"}
EXPECTED_SAMPLER = {
    "sampler_contract_version": 3, "batch_size": 3, "block_batches": 4,
    "sampling_mode": "balanced", "seed": 0, "active_tasks": [0, 16],
    "dataset_fingerprint": "41181fc115389d76abb00f054cbd8b318bd534204a27c08fd8163c928d662e45",
    "task_weights": [1.0, 1.0],
}
STATIC2_CONTRACT_VALUES = {
    "wmrm_world_weight": 1.0,
    "wmrm_static_constraint_weight": 2.0,
    "wmrm_detach_proposal_stage_state": True,
}
STEP_RE = re.compile(r"(?:^|\s)step=(?P<step>\d+)(?=\s)")
VALUE_RE = {n: re.compile(rf"(?:^|\s){n}=(?P<value>[^\s\]]+)") for n in ("grad", "flow", "world")}
STATIC_RE = re.compile(r"\bstatic=(?P<value>[^/\s\]]+)/(?P<copy>[^\s\]]+)")
SECTION_RE = re.compile(r"\bworld_task\[(?P<body>[^\]]*)\]")
ENTRY_RE = re.compile(r"(?:^|\s\|\s)(?P<task>[^:\s|]+):(?P<body>.*?)(?=\s\|\s|$)")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_RE = re.compile(r"Traceback \(most recent call last\):|\b(?:RuntimeError|ValueError|AssertionError|MemoryError|CUDA error|ERROR):", re.I)
NONFINITE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z0-9_])", re.I)
class AnalysisError(ValueError): pass

def validate_final_checkpoint(payload: object, *, expected_step: int = 12074) -> None:
    if not isinstance(payload, dict): raise AnalysisError("final checkpoint payload must be a mapping")
    required = {"model", "optimizer_state", "sampler_state", "rng_state", "exact_run_contract", "exact_resume_version", "global_step"}
    missing = sorted(required - payload.keys())
    if missing: raise AnalysisError(f"final checkpoint missing required keys: {missing}")
    if payload["global_step"] != expected_step or payload["exact_resume_version"] != 2:
        raise AnalysisError("final checkpoint exact step/version mismatch")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise AnalysisError("final checkpoint model state must be nonempty")
    optimizer = payload["optimizer_state"]
    if not isinstance(optimizer, dict) or optimizer.get("kind") != "adamw": raise AnalysisError("final optimizer must be AdamW")
    state_dict = optimizer.get("state_dict")
    if not isinstance(state_dict, dict) or set(state_dict) != {"state", "param_groups"} or not state_dict["state"] or not state_dict["param_groups"]:
        raise AnalysisError("final AdamW state is incomplete")
    sampler = payload["sampler_state"]
    if not isinstance(sampler, dict): raise AnalysisError("final sampler state must be a mapping")
    for key, value in EXPECTED_SAMPLER.items():
        if sampler.get(key) != value: raise AnalysisError(f"final sampler {key} mismatch")
    for key in ("epoch", "batch_cursor"):
        if type(sampler.get(key)) is not int or sampler[key] < 0: raise AnalysisError(f"final sampler {key} invalid")
    rng = payload["rng_state"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"} or any(value is None for value in rng.values()):
        raise AnalysisError("final RNG state is incomplete")
    contract = payload["exact_run_contract"]
    if not isinstance(contract, dict) or contract.get("contract_version") != 1: raise AnalysisError("final exact contract version mismatch")
    arguments = contract.get("arguments")
    if not isinstance(arguments, dict): raise AnalysisError("final exact contract arguments missing")
    for key, value in STATIC2_CONTRACT_VALUES.items():
        if arguments.get(key) != value: raise AnalysisError(f"final exact contract {key} mismatch")
    if "resume_exact_contract_migration" in arguments: raise AnalysisError("final contract persisted operational migration selector")
    model_config = contract.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("wmrm_detach_proposal_stage_state") is not True:
        raise AnalysisError("final model contract detach must be true")
    optimizer_contract = contract.get("optimizer")
    if not isinstance(optimizer_contract, dict) or optimizer_contract.get("kind") != "adamw": raise AnalysisError("final exact optimizer contract must be AdamW")

def num(raw: str, field: str, step: int) -> float:
    try: value = float(raw.rstrip(",;"))
    except ValueError as exc: raise AnalysisError(f"step {step}: invalid {field}") from exc
    if not math.isfinite(value): raise AnalysisError(f"step {step}: nonfinite {field}")
    return value

def parse_log_text(text: str, *, expected_start_step: int = 12011) -> list[dict]:
    text = ANSI_RE.sub("", text.replace("\r", "\n"))
    if ERROR_RE.search(text): raise AnalysisError("trainer error marker found in log")
    if NONFINITE_RE.search(text): raise AnalysisError("nonfinite token found in log")
    records, seen = [], set()
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if not match or " mode=" not in line: continue
        step = int(match.group("step")); values = {"step": step}
        for field, pattern in VALUE_RE.items():
            found = pattern.search(line)
            if not found: raise AnalysisError(f"step {step}: missing {field} metric")
            values[field] = num(found.group("value"), field, step)
        sections = list(SECTION_RE.finditer(line))
        if len(sections) != 1: raise AnalysisError(f"step {step}: expected exactly one world_task section")
        tasks, static_by_task = set(), {}
        for entry in ENTRY_RE.finditer(sections[0].group("body")):
            task = entry.group("task")
            if task not in EXPECTED_TASKS: raise AnalysisError(f"step {step}: invalid world_task task {task!r}")
            if task in tasks: raise AnalysisError(f"step {step}: duplicate world_task task")
            tasks.add(task); matches = list(STATIC_RE.finditer(entry.group("body")))
            if len(matches) != 1: raise AnalysisError(f"step {step}: static=current/copy token required")
            static_by_task[task] = num(matches[0].group("value"), "static", step)
            num(matches[0].group("copy"), "copy_static", step)
        if not tasks: raise AnalysisError(f"step {step}: at least one task is required")
        values["static_by_task"], values["tasks"] = static_by_task, sorted(tasks)
        # Keep the legacy aggregate for report compatibility; gates use per-task values.
        values["static"] = statistics.fmean(static_by_task.values())
        if step in seen: raise AnalysisError(f"duplicate update record for step {step}")
        seen.add(step); records.append(values)
    expected = list(range(expected_start_step, expected_start_step + EXPECTED_UPDATES))
    if [r["step"] for r in records] != expected: raise AnalysisError("update steps mismatch")
    observed_tasks = {task for record in records for task in record["tasks"]}
    if observed_tasks != EXPECTED_TASKS:
        raise AnalysisError(f"updates must collectively contain both tasks {sorted(EXPECTED_TASKS)}")
    return records

def analyze_records(records: list[dict]) -> dict:
    if len(records) != EXPECTED_UPDATES: raise AnalysisError("analysis requires exactly 64 records")
    grad = [float(r["grad"]) for r in records]; flow = [float(r["flow"]) for r in records]
    world = [float(r["world"]) for r in records]
    window_max = max(sum(x > 20 for x in grad[i:i+16]) for i in range(49))
    world_first, world_last = statistics.median(world[:16]), statistics.median(world[-16:])
    ratio = world_last / world_first if world_first > 0 else (0.0 if world_last == 0 else math.inf)
    task_values = {task: [float(record["static_by_task"][task]) for record in records if task in record["static_by_task"]] for task in EXPECTED_TASKS}
    missing = [task for task, values in task_values.items() if not values]
    if missing: raise AnalysisError(f"missing static metrics for tasks: {missing}")
    task_last32_max = {task: max(values[-32:]) for task, values in task_values.items()}
    static_last32_max = max(task_last32_max.values())
    gates = {"finite_and_error_free": True, "grad_max_le_50": max(grad) <= 50,
             "any_16_grad_over_20_count_le_1": window_max <= 1,
             "final32_flow_finite": all(math.isfinite(x) for x in flow[-32:]),
             "last32_static_max_le_0_02": static_last32_max <= .02,
             "world_last16_over_first16_le_2": ratio <= 2}
    passed = all(gates.values())
    return {"contract": CONTRACT, "decision": "PASS" if passed else "NO-GO", "passed": passed,
            "gates": gates, "observed": {"grad_max": max(grad), "grad_over_20_max_in_16": window_max,
            "last32_static_max": static_last32_max, "last32_static_max_by_task": task_last32_max, "world_ratio": ratio,
            "flow_final32_median": statistics.median(flow[-32:])}, "updates": len(records)}

def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"); os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()

def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--log", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--report", type=Path, required=True); a = p.parse_args(argv)
    try:
        import torch
        validate_final_checkpoint(torch.load(a.checkpoint, map_location="cpu", weights_only=True))
        result = analyze_records(parse_log_text(a.log.read_text(encoding="utf-8", errors="replace")))
        result["log"] = str(a.log.resolve()); result["checkpoint"] = str(a.checkpoint.resolve()); atomic_write(a.report, result)
    except (OSError, ValueError, TypeError) as exc: print(f"FATAL: {exc}", file=os.sys.stderr); return 1
    print(f"{result['decision']}: {a.report}"); return 0 if result["passed"] else 2
if __name__ == "__main__": raise SystemExit(main())
