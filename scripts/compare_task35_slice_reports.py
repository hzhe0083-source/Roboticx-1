#!/usr/bin/env python
"""Compare two task35 clean/recovery slice reports. Mechanism evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _distance(report: dict, layer: str) -> dict:
    return ((report.get("slices") or {}).get(layer) or {}).get("pegHead_hole_px") or {}


def _visible(report: dict, layer: str) -> float | None:
    slice_ = (report.get("slices") or {}).get(layer) or {}
    value = slice_.get("pair_visible_fraction")
    return None if value is None else float(value)


def compare_slice_reports(left: dict, right: dict) -> dict:
    if left.get("contract") != "task35_clean_recovery_slice_v1":
        raise ValueError("left report is not a task35 slice report")
    if right.get("contract") != "task35_clean_recovery_slice_v1":
        raise ValueError("right report is not a task35 slice report")
    layers = {}
    for layer in ("clean", "recovery", "all"):
        left_d = _distance(left, layer)
        right_d = _distance(right, layer)
        left_vis = _visible(left, layer)
        right_vis = _visible(right, layer)
        layers[layer] = {
            "left_n": left_d.get("n"),
            "right_n": right_d.get("n"),
            "left_mean_px": left_d.get("mean"),
            "right_mean_px": right_d.get("mean"),
            "delta_mean_px": (
                None
                if left_d.get("mean") is None or right_d.get("mean") is None
                else float(right_d["mean"]) - float(left_d["mean"])
            ),
            "left_visible": left_vis,
            "right_visible": right_vis,
            "delta_visible": (
                None if left_vis is None or right_vis is None else right_vis - left_vis
            ),
        }
    return {
        "contract": "task35_slice_compare_v1",
        "left": left.get("checkpoint"),
        "right": right.get("checkpoint"),
        "layers": layers,
        "note": "cached-window geometry only; not closed-loop insertion success",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_slice_reports(
        json.loads(args.left.read_text()),
        json.loads(args.right.read_text()),
    )
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
