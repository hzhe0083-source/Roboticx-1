#!/usr/bin/env python
"""Compare task35 closed-loop JSON files on identical episode seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[dict, dict[int, dict]]:
    payload = json.loads(path.read_text())
    if payload.get("contract") != "metaworld_closed_loop_trials_v1":
        raise ValueError(f"unsupported result contract in {path}")
    trials = {int(row["seed"]): row for row in payload["trials"]}
    if len(trials) != int(payload["completed_trials"]):
        raise ValueError(f"duplicate or missing trial seeds in {path}")
    return payload, trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_payload, base = load(args.baseline)
    rows = []
    for path in args.candidates:
        payload, candidate = load(path)
        if set(candidate) != set(base):
            raise ValueError(f"seed set mismatch: {path}")
        seeds = sorted(base)
        base_success = np.asarray([bool(base[s]["success"]) for s in seeds])
        cand_success = np.asarray([bool(candidate[s]["success"]) for s in seeds])
        base_distance = np.asarray(
            [float(base[s].get("stage", {}).get("min_obj_to_target", np.nan)) for s in seeds]
        )
        cand_distance = np.asarray(
            [float(candidate[s].get("stage", {}).get("min_obj_to_target", np.nan)) for s in seeds]
        )
        valid_distance = np.isfinite(base_distance) & np.isfinite(cand_distance)
        rows.append(
            {
                "path": str(path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "ablation": payload.get("task35_causal_ablation", "none"),
                "successes": int(cand_success.sum()),
                "success_rate": float(cand_success.mean()),
                "success_delta": int(cand_success.sum() - base_success.sum()),
                "discordant_improved": int((~base_success & cand_success).sum()),
                "discordant_worsened": int((base_success & ~cand_success).sum()),
                "paired_min_obj_to_target_delta_mean": (
                    float((cand_distance[valid_distance] - base_distance[valid_distance]).mean())
                    if valid_distance.any()
                    else None
                ),
                "paired_distance_improvement_fraction": (
                    float((cand_distance[valid_distance] < base_distance[valid_distance]).mean())
                    if valid_distance.any()
                    else None
                ),
            }
        )
    result = {
        "contract": "task35_paired_eval_comparison_v1",
        "baseline": {
            "path": str(args.baseline),
            "checkpoint_sha256": base_payload["checkpoint_sha256"],
            "successes": int(sum(bool(row["success"]) for row in base.values())),
            "trials": len(base),
        },
        "comparisons": rows,
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
