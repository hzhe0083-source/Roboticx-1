#!/usr/bin/env python
"""Select the best reproducible FM VA only from validated 50-seed closed-loop JSONs.

Slice geometry and training loss cannot elect a winner. Missing or invalid
eval50 evidence is a hard failure, not a fallback to mechanism metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metaworld import validate_task35_eval50_payload


def load_eval50(path: Path) -> dict:
    payload = json.loads(path.read_text())
    validate_task35_eval50_payload(payload)
    if payload.get("task35_causal_ablation", "none") != "none":
        raise ValueError(f"{path} is a causal diagnostic, not an acceptance eval50")
    if not payload.get("task35_precision_contract"):
        raise ValueError(f"{path} is not a precision-contract eval50")
    return payload


def select_best_task35_fm(candidates: list[dict]) -> dict:
    """candidates: [{path, eval50, validated, sha256, step}]"""
    eligible = []
    rejected = []
    for item in candidates:
        eval50 = item.get("eval50")
        if eval50 is None:
            rejected.append({"path": item.get("path"), "reason": "missing eval50"})
            continue
        try:
            validate_task35_eval50_payload(eval50)
        except ValueError as exc:
            rejected.append({"path": item.get("path"), "reason": str(exc)})
            continue
        if eval50.get("task35_causal_ablation", "none") != "none":
            rejected.append({"path": item.get("path"), "reason": "causal diagnostic"})
            continue
        if not eval50.get("task35_precision_contract"):
            rejected.append({"path": item.get("path"), "reason": "not precision contract"})
            continue
        if item.get("validated") is False:
            rejected.append({"path": item.get("path"), "reason": "checkpoint not validated"})
            continue
        eligible.append(item)
    if not eligible:
        raise ValueError(
            "no reproducible FM VA: every candidate lacks a valid 50-seed eval50. "
            f"rejected={rejected}"
        )
    ranked = sorted(
        eligible,
        key=lambda item: (
            -int(item["eval50"]["successes"]),
            _mean_min_obj(item["eval50"]),
            int(item.get("step") or 10**9),
        ),
    )
    winner = ranked[0]
    return {
        "contract": "task35_best_fm_v1",
        "selected": {
            "path": winner.get("path"),
            "step": winner.get("step"),
            "sha256": winner.get("sha256") or winner["eval50"].get("checkpoint_sha256"),
            "successes": int(winner["eval50"]["successes"]),
            "completed_trials": int(winner["eval50"]["completed_trials"]),
            "success_rate": float(winner["eval50"]["success_rate"]),
            "mean_min_obj_to_target": _mean_min_obj(winner["eval50"]),
        },
        "ranked": [
            {
                "path": item.get("path"),
                "step": item.get("step"),
                "successes": int(item["eval50"]["successes"]),
                "success_rate": float(item["eval50"]["success_rate"]),
            }
            for item in ranked
        ],
        "rejected": rejected,
        "label": "supported",
    }


def _mean_min_obj(eval50: dict) -> float:
    distances = []
    for row in eval50.get("trials") or []:
        value = (row.get("stage") or {}).get("min_obj_to_target")
        if value is not None:
            distances.append(float(value))
    return float(sum(distances) / len(distances)) if distances else 1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval50", type=Path, nargs="*")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eval50:
        raise SystemExit("no eval50 JSON given; refusing to elect a winner from slices/loss")
    candidates = []
    for path in args.eval50:
        payload = load_eval50(path)
        candidates.append(
            {
                "path": payload.get("checkpoint"),
                "sha256": payload.get("checkpoint_sha256"),
                "eval50": payload,
                "validated": True,
            }
        )
    report = select_best_task35_fm(candidates)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
