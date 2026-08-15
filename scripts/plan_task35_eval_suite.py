#!/usr/bin/env python
"""Plan which archived FM milestones still need a 50-seed closed-loop eval.

This is CPU-only. It never starts eval or claims a winner from slices/loss.
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

REQUIRED_MILESTONES = (3000, 6000, 9000, 12000, 15000, 18000, 20000)
SKIP_CLOSED_LOOP_STEPS = (1000, 2000)


def _eval50_path(logs_dir: Path, checkpoint: str) -> Path:
    return logs_dir / f"{Path(checkpoint).stem}_eval50.json"


def _valid_eval50(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        validate_task35_eval50_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not payload.get("task35_precision_contract"):
        return None
    if payload.get("task35_causal_ablation", "none") != "none":
        return None
    return payload


def plan_task35_eval_suite(
    candidates: list[dict],
    *,
    logs_dir: Path,
    require_all: bool = False,
    require_eval50: bool = False,
    required_steps: tuple[int, ...] = REQUIRED_MILESTONES,
) -> dict:
    to_eval: list[dict] = []
    already_done: list[dict] = []
    skipped: list[dict] = []
    for item in candidates:
        path = item.get("path")
        step = item.get("step")
        if not item.get("validated"):
            skipped.append({"path": path, "step": step, "reason": "not validated"})
            continue
        if step in SKIP_CLOSED_LOOP_STEPS:
            skipped.append(
                {
                    "path": path,
                    "step": step,
                    "reason": "early milestone kept for mechanism only",
                }
            )
            continue
        eval50_path = _eval50_path(logs_dir, str(path))
        payload = _valid_eval50(eval50_path)
        archive_sha = item.get("sha256")
        if (
            payload is not None
            and archive_sha
            and payload.get("checkpoint_sha256")
            and payload.get("checkpoint_sha256") != archive_sha
        ):
            payload = None
        row = {
            "path": path,
            "step": step,
            "sha256": archive_sha,
            "eval50": str(eval50_path),
        }
        if payload is None:
            reason = "missing eval50" if not eval50_path.is_file() else "invalid eval50"
            to_eval.append({**row, "reason": reason})
        else:
            already_done.append(
                {
                    **row,
                    "successes": int(payload["successes"]),
                    "completed_trials": int(payload["completed_trials"]),
                }
            )
    if require_all:
        have = {
            int(item["step"])
            for item in candidates
            if item.get("validated") and item.get("step") is not None
        }
        missing = [step for step in required_steps if step not in have]
        if missing:
            raise ValueError(
                "missing validated milestones: "
                + ",".join(str(step) for step in missing)
            )
    if require_eval50:
        missing_eval50 = [
            step
            for step in required_steps
            if step not in {int(row["step"]) for row in already_done if row.get("step") is not None}
        ]
        if missing_eval50:
            raise ValueError(
                "missing valid 50-seed eval50 for: "
                + ",".join(str(step) for step in missing_eval50)
            )
    return {
        "contract": "task35_eval_suite_plan_v1",
        "require_all": require_all,
        "require_eval50": require_eval50,
        "required_steps": list(required_steps),
        "to_eval": to_eval,
        "already_done": already_done,
        "skipped": skipped,
        "eval50_paths": [row["eval50"] for row in already_done],
        "planned_eval50_paths": [row["eval50"] for row in already_done + to_eval],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "logs" / "task35_fm_candidates.json",
    )
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--require-eval50",
        action="store_true",
        help="Fail unless every acceptance milestone already has a valid 50-seed JSON.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.candidates.read_text()) if args.candidates.is_file() else {}
    plan = plan_task35_eval_suite(
        payload.get("candidates") or [],
        logs_dir=args.logs_dir,
        require_all=args.require_all,
        require_eval50=args.require_eval50,
    )
    text = json.dumps(plan, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
