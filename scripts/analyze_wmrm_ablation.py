#!/usr/bin/env python
"""Summarize paired fixed-seed WMRM ablation JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MODE_MAP = {
    "normal": "normal",
    "action-write-off": "action-off",
    "vision-write-off": "vision-off",
    "both-write-off": "both-off",
    "proposal-only": "proposal-only",
}

MODE_PROVENANCE = {
    "normal": (True, True, False),
    "action-off": (False, True, False),
    "vision-off": (True, False, False),
    "both-off": (False, False, False),
    "proposal-only": (False, False, True),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("contract") != "metaworld_closed_loop_trials_v1":
        raise ValueError(f"unsupported result contract: {path}")
    required = ("checkpoint_sha256", "task_ids", "trials_per_task", "execute_steps", "horizon", "trials")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} missing fields: {missing}")
    return payload


def _trial_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["seed"])


def _mode(payload: dict[str, Any], path: Path) -> str:
    raw = payload.get("wmrm_ablation_mode")
    if raw in MODE_MAP:
        return MODE_MAP[raw]
    for mode in ("proposal-only", "action-off", "vision-off", "both-off", "normal"):
        if path.stem.endswith("_" + mode):
            return mode
    raise ValueError(f"cannot identify ablation mode for {path}")


def _validate_provenance(payload: dict[str, Any], path: Path, mode: str) -> None:
    fields = (
        "wmrm_action_write_enabled",
        "wmrm_vision_write_enabled",
        "wmrm_proposal_only",
    )
    expected = MODE_PROVENANCE[mode]
    actual = tuple(payload.get(field) for field in fields)
    if any(type(value) is not bool for value in actual) or actual != expected:
        raise ValueError(
            f"WMRM provenance mismatch for mode {mode}: {path}; "
            f"expected {dict(zip(fields, expected, strict=True))}, "
            f"got {dict(zip(fields, actual, strict=True))}"
        )


def _chunks(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("action_chunks")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.ndim == 3 and array.shape[0] > 0 else None


def analyze(paths: list[Path]) -> dict[str, Any]:
    loaded = [(path, _load(path)) for path in paths]
    by_mode: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in loaded:
        mode = _mode(payload, path)
        _validate_provenance(payload, path, mode)
        if mode in by_mode:
            raise ValueError(f"duplicate mode {mode}: {by_mode[mode][0]} and {path}")
        by_mode[mode] = (path, payload)
    if "normal" not in by_mode:
        raise ValueError("normal baseline JSON is required")

    base_path, base = by_mode["normal"]
    identity = (
        base["checkpoint_sha256"], tuple(base["task_ids"]), int(base["trials_per_task"]),
        int(base["execute_steps"]), int(base["horizon"]),
    )
    base_trials = {_trial_key(row): row for row in base["trials"]}
    if len(base_trials) != len(base["trials"]):
        raise ValueError("normal baseline contains duplicate task/seed pairs")

    result: dict[str, Any] = {
        "contract": "wmrm_fixed_checkpoint_ablation_summary_v1",
        "checkpoint_sha256": identity[0],
        "task_ids": list(identity[1]),
        "trials_per_task": identity[2],
        "execute_steps": identity[3],
        "horizon": identity[4],
        "baseline": str(base_path),
        "modes": [],
    }
    for mode in ("normal", "action-off", "vision-off", "both-off", "proposal-only"):
        if mode not in by_mode:
            continue
        path, payload = by_mode[mode]
        candidate_identity = (
            payload["checkpoint_sha256"], tuple(payload["task_ids"]), int(payload["trials_per_task"]),
            int(payload["execute_steps"]), int(payload["horizon"]),
        )
        if candidate_identity != identity:
            raise ValueError(f"protocol/checkpoint mismatch: {path}")
        trials = {_trial_key(row): row for row in payload["trials"]}
        if len(trials) != len(payload["trials"]):
            raise ValueError(f"{mode} contains duplicate task/seed pairs: {path}")
        if set(trials) != set(base_trials):
            raise ValueError(f"fixed seed set mismatch: {path}")

        task_rows = []
        for task_id in identity[1]:
            keys = sorted(key for key in trials if key[0] == int(task_id))
            wins = sum(bool(trials[key]["success"]) for key in keys)
            base_wins = sum(bool(base_trials[key]["success"]) for key in keys)
            task_rows.append({
                "task_id": int(task_id), "successes": wins, "trials": len(keys),
                "success_rate": wins / len(keys), "success_delta_vs_normal": wins - base_wins,
            })

        first_l1: list[float] = []
        chunk_l1: list[float] = []
        paired_chunks = 0
        for key in sorted(trials):
            base_chunks, candidate_chunks = _chunks(base_trials[key]), _chunks(trials[key])
            if base_chunks is None or candidate_chunks is None:
                continue
            if candidate_chunks.shape != base_chunks.shape:
                raise ValueError(
                    f"action chunk shape mismatch for mode {mode}, trial {key}: "
                    f"{path} has {candidate_chunks.shape}, baseline has {base_chunks.shape}"
                )
            diff = np.abs(candidate_chunks - base_chunks)
            first_l1.append(float(diff[0, 0].mean()))
            chunk_l1.append(float(diff.mean()))
            paired_chunks += 1

        mode_row: dict[str, Any] = {
            "mode": mode, "path": str(path), "per_task": task_rows,
            "successes": sum(bool(row["success"]) for row in trials.values()),
            "trials": len(trials),
        }
        mode_row["success_rate"] = mode_row["successes"] / mode_row["trials"]
        mode_row["action_divergence"] = {
            "supported": paired_chunks == len(trials),
            "paired_trials": paired_chunks,
            "first_action_l1_mean": float(np.mean(first_l1)) if first_l1 else None,
            "chunk_l1_mean": float(np.mean(chunk_l1)) if chunk_l1 else None,
        }
        result["modes"].append(mode_row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.inputs)
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
