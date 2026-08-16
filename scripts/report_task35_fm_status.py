#!/usr/bin/env python
"""Assemble the current task35 FM evidence ledger. Closed-loop success is planned."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metaworld import validate_task35_eval50_payload
from scripts.plan_task35_eval_suite import REQUIRED_MILESTONES, SKIP_CLOSED_LOOP_STEPS
from scripts.select_task35_best_fm import select_best_task35_fm


def _eval50_ok(payload: dict) -> bool:
    try:
        validate_task35_eval50_payload(payload)
    except ValueError:
        return False
    return bool(payload.get("task35_precision_contract")) and payload.get(
        "task35_causal_ablation", "none"
    ) == "none"


def closed_loop_complete(rows: list[dict]) -> bool:
    """True only when every acceptance milestone has a valid 50-seed eval50."""
    by_step = {row.get("step"): row for row in rows}
    return all(
        (by_step.get(step) or {}).get("labels", {}).get("closed_loop") == "supported"
        for step in REQUIRED_MILESTONES
    )


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def evidence_row(candidate: dict) -> dict:
    slices = candidate.get("slices") or {}
    eval50 = load_json(Path("logs") / f"{Path(candidate['path']).stem}_eval50.json")
    holdout = load_json(
        Path("logs") / f"{Path(candidate['path']).stem}_metric_holdout2777.json"
    )
    step = candidate.get("step")
    return {
        "step": step,
        "sha256": candidate.get("sha256"),
        "validated": bool(candidate.get("validated")),
        "geometry_l2": candidate.get("geometry_l2"),
        "clean_visible": (slices.get("clean") or {}).get("pair_visible_fraction"),
        "clean_pair_px": (slices.get("clean") or {}).get("pegHead_hole_mean_px"),
        "recovery_visible": (slices.get("recovery") or {}).get("pair_visible_fraction"),
        "recovery_pair_px": (slices.get("recovery") or {}).get("pegHead_hole_mean_px"),
        "holdout_rmse_px": None if holdout is None else (holdout.get("aggregate") or {}).get("rmse_px"),
        "closed_loop_successes": None if eval50 is None else eval50.get("successes"),
        "closed_loop_trials": None if eval50 is None else eval50.get("completed_trials"),
        "acceptance_candidate": step not in SKIP_CLOSED_LOOP_STEPS,
        "labels": {
            "checkpoint_contract": "supported" if candidate.get("validated") else "planned",
            "slice_geometry": "partially supported" if slices else "planned",
            "holdout_metric": "supported" if holdout is not None else "planned",
            "closed_loop": (
                "skipped"
                if step in SKIP_CLOSED_LOOP_STEPS
                else (
                    "supported"
                    if eval50 is not None and _eval50_ok(eval50)
                    else "planned"
                )
            ),
        },
    }


def render_md(payload: dict) -> str:
    lines = [
        "# task35 FM status",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- training_step: {payload['training'].get('latest_step')}",
        f"- training_status: {payload['training'].get('status')}",
        f"- closed_loop_complete: {payload['closed_loop_complete']}",
        f"- best: {payload.get('best')}",
        "",
        "## candidates",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"- step {row['step']}: validated={row['validated']} "
            f"geomL2={row['geometry_l2']} "
            f"clean_vis={row['clean_visible']} clean_px={row['clean_pair_px']} "
            f"rec_vis={row['recovery_visible']} rec_px={row['recovery_pair_px']} "
            f"holdout={row['holdout_rmse_px']} "
            f"eval50={row['closed_loop_successes']}/{row['closed_loop_trials']}"
            f"{'' if row.get('acceptance_candidate', True) else ' (mechanism only)'}"
        )
        lines.append(
            f"  labels: contract={row['labels']['checkpoint_contract']}, "
            f"slices={row['labels']['slice_geometry']}, "
            f"holdout={row['labels']['holdout_metric']}, "
            f"closed_loop={row['labels']['closed_loop']}"
        )
    lines.extend(
        [
            "",
            "Closed-loop insertion remains planned until 12k/15k/18k/20k each have a 50-seed eval50.",
            "1k/2k/3k/6k/9k stay mechanism-only and cannot elect a winner.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "logs" / "task35_fm_candidates.json",
    )
    parser.add_argument(
        "--monitor",
        type=Path,
        default=root / "logs" / "task35_fm_train_monitor.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "logs" / "task35_fm_status.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=root / "logs" / "task35_fm_status.md",
    )
    return parser.parse_args()


def main() -> None:
    from datetime import datetime, timezone, timedelta

    args = parse_args()
    candidates = load_json(args.candidates) or {"candidates": []}
    monitor = load_json(args.monitor) or {}
    rows = [evidence_row(item) for item in candidates.get("candidates") or []]
    closed = closed_loop_complete(rows)
    best = None
    best_error = None
    selector_input = []
    for item, row in zip(candidates.get("candidates") or [], rows):
        eval50 = load_json(Path("logs") / f"{Path(item['path']).stem}_eval50.json")
        selector_input.append(
            {
                "path": item.get("path"),
                "step": item.get("step"),
                "sha256": item.get("sha256"),
                "validated": item.get("validated"),
                "eval50": eval50,
            }
        )
    try:
        best = select_best_task35_fm(selector_input)
    except ValueError as exc:
        best_error = str(exc)
    payload = {
        "contract": "task35_fm_status_v1",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "training": {
            "latest_step": monitor.get("latest_step"),
            "status": "RUNNING" if (monitor.get("trainer") or {}).get("alive") else "UNKNOWN",
            "latest_loss": monitor.get("latest_loss"),
        },
        "candidates": rows,
        "closed_loop_complete": closed,
        "best": None if best is None else best.get("selected"),
        "best_error": best_error,
        "note": "mechanism and provenance only until eval50 exists",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(args.output_json)
    markdown = render_md(payload)
    md_tmp = args.output_md.with_suffix(args.output_md.suffix + ".tmp")
    md_tmp.write_text(markdown)
    md_tmp.replace(args.output_md)
    print(markdown, end="")


if __name__ == "__main__":
    main()
