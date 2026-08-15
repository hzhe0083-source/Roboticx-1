#!/usr/bin/env python
"""List archived task35 FM checkpoints that already passed strict validation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(r"_step(\d+)\.pt$")


def infer_step(path: Path) -> int | None:
    match = STEP_RE.search(path.name)
    return None if match is None else int(match.group(1))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=root / "checkpoints",
    )
    parser.add_argument(
        "--pattern",
        default="task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step*.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "logs" / "task35_fm_candidates.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.root.glob(args.pattern)):
        if path.name.endswith(".tmp"):
            continue
        sidecar = Path(str(path) + ".sha256")
        sha = sidecar.read_text().split()[0] if sidecar.is_file() else None
        report = None
        search = [
            Path("logs") / f"{path.stem}_validate.json",
            Path("logs") / "task35_fm_checkpoint_validate_1k2k.json",
        ]
        for report_path in search:
            if not report_path.is_file():
                continue
            payload = json.loads(report_path.read_text())
            items = payload.get("reports", [payload])
            for item in items:
                if Path(item.get("path", "")).name == path.name or item.get("sha256") == sha:
                    report = item
                    break
            if report is not None:
                break
        slice_path = Path("logs") / f"{path.stem}_clean_recovery_slices.json"
        slices = json.loads(slice_path.read_text()) if slice_path.is_file() else None
        slice_summary = None
        if slices and slices.get("slices"):
            slice_summary = {
                layer: {
                    "n": (slices["slices"].get(layer) or {}).get("n"),
                    "pair_visible_fraction": (slices["slices"].get(layer) or {}).get(
                        "pair_visible_fraction"
                    ),
                    "pegHead_hole_mean_px": (
                        ((slices["slices"].get(layer) or {}).get("pegHead_hole_px") or {}).get(
                            "mean"
                        )
                    ),
                }
                for layer in ("clean", "recovery")
            }
        rows.append(
            {
                "path": str(path),
                "step": (
                    None
                    if report is None
                    else report.get("global_step")
                )
                or infer_step(path),
                "sha256": (None if report is None else report.get("sha256")) or sha,
                "validated": bool(report and report.get("ok")),
                "geometry_l2": None if report is None else report.get("geometry_l2"),
                "slices": slice_summary,
            }
        )
    payload = {"contract": "task35_fm_candidates_v1", "candidates": rows}
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
