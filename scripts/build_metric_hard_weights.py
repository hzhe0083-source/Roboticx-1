#!/usr/bin/env python
"""Build bounded all-49 visual hard-mining weights from one dev evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_longtraj_features import ENV_TO_TASK  # noqa: E402


FORMULA = (
    "clip(1 + I(task_rmse>15) + I(worst_role_rmse>30) + "
    "I(task_pck10<0.30) + I(visibility_brier>0.05), 1, 4)"
)


def task_weight(metrics: Mapping[str, Any]) -> tuple[int, dict[str, float]]:
    localization = metrics["localization"]
    aggregate = localization["aggregate"]
    role_rmse = [
        float(values["rmse_px"])
        for values in localization["roles"].values()
        if values.get("rmse_px") is not None
    ]
    if not role_rmse:
        raise ValueError("task has no visible role localization observations")
    values = {
        "task_rmse_px": float(aggregate["rmse_px"]),
        "worst_role_rmse_px": max(role_rmse),
        "task_pck10": float(aggregate["pck@10px"]),
        "visibility_brier": float(metrics["visibility"]["aggregate"]["brier"]),
    }
    weight = 1
    weight += values["task_rmse_px"] > 15.0
    weight += values["worst_role_rmse_px"] > 30.0
    weight += values["task_pck10"] < 0.30
    weight += values["visibility_brier"] > 0.05
    return min(max(int(weight), 1), 4), values


def build_weights(result: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    expected = list(ENV_TO_TASK)
    tasks = result.get("tasks")
    if tasks != expected:
        raise ValueError("evaluation tasks are not the canonical project all-49 order")
    if int(result.get("samples_per_task", 0)) != 50:
        raise ValueError("hard weights require exactly 50 dev samples per task")
    if int(result.get("batch_size", 0)) != 4:
        raise ValueError("hard weights require the preregistered batch_size=4")
    per_task = result.get("per_task")
    if not isinstance(per_task, Mapping) or set(per_task) != set(expected):
        raise ValueError("evaluation per_task keys do not match canonical all-49")

    weights: dict[str, int] = {}
    diagnostics: dict[str, Any] = {}
    for task in expected:
        weight, values = task_weight(per_task[task])
        weights[task] = weight
        diagnostics[task] = {"weight": weight, **values}
    return weights, diagnostics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    source = Path(args.evaluation).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    result = json.loads(raw)
    weights, diagnostics = build_weights(result)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(weights, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    meta = Path(args.meta).expanduser() if args.meta else output.with_suffix(".meta.json")
    metadata = {
        "contract": "mt_vj_visual_hard_weights_v1",
        "formula": FORMULA,
        "evaluation": str(source),
        "evaluation_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_seed": result.get("seed"),
        "evaluation_batch_size": result.get("batch_size"),
        "evaluation_samples_per_task": result.get("samples_per_task"),
        "weights_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "weight_sum": sum(weights.values()),
        "diagnostics": diagnostics,
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = {value: list(weights.values()).count(value) for value in range(1, 5)}
    print(f"hard weights saved: {output} sum={sum(weights.values())} counts={counts}")
    print(f"provenance saved: {meta}")


if __name__ == "__main__":
    main()
