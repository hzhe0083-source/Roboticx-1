#!/usr/bin/env python
"""Summarize the live task35 FM training log into milestone windows."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

STEP_RE = re.compile(
    r"^step=(?P<step>\d+)\s+.*?loss=(?P<loss>[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
)
AUX_RE = re.compile(r"aux_rmse=(?P<rmse>[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)px")
SAVE_RE = re.compile(r"global_step=(?P<step>\d+)\s+periodic checkpoint saved")
ARCHIVE_MILESTONES = (1000, 2000, 3000, 6000, 9000, 12000, 15000, 18000, 20000)


def parse_float(text: str) -> float:
    return float(text)


def window_stats(values: list[float]) -> dict | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return {
        "n": len(finite),
        "mean": float(mean(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "last": float(finite[-1]),
    }


def summarize_task35_fm_log(
    text: str,
    *,
    total_steps: int = 20000,
    window: int = 1000,
    checkpoint_stem: Path | None = None,
) -> dict:
    rows: list[tuple[int, float]] = []
    aux: list[tuple[int, float]] = []
    saves: list[int] = []
    alerts: list[str] = []
    for line in text.splitlines():
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            alerts.append("oom")
        save = SAVE_RE.search(line)
        if save:
            saves.append(int(save.group("step")))
        match = STEP_RE.match(line)
        if not match:
            continue
        step = int(match.group("step"))
        loss = parse_float(match.group("loss"))
        if not math.isfinite(loss):
            alerts.append(f"non_finite_step_{step}")
        rows.append((step, loss))
        aux_match = AUX_RE.search(line)
        if aux_match:
            aux.append((step, float(aux_match.group("rmse"))))
    if not rows:
        raise ValueError("no training steps parsed")
    latest_step, latest_loss = rows[-1]
    windows = []
    start = 1
    while start <= latest_step:
        end = min(start + window - 1, latest_step)
        losses = [loss for step, loss in rows if start <= step <= end]
        aux_vals = [rmse for step, rmse in aux if start <= step <= end]
        item = {
            "start": start,
            "end": end,
            "loss": window_stats(losses),
            "aux_rmse_px": window_stats(aux_vals),
        }
        if end in ARCHIVE_MILESTONES:
            item["milestone"] = end
            if checkpoint_stem is not None:
                dest = Path(f"{checkpoint_stem}_step{end}.pt")
                item["archived"] = dest.is_file() and Path(str(dest) + ".sha256").is_file()
        elif end % window == 0:
            # Live trainer overwrites the working checkpoint every 1000 steps.
            # Only ARCHIVE_MILESTONES are copied aside; 4000/5000/7000/... are
            # periodic saves, not missing archives.
            item["periodic_save_only"] = True
        windows.append(item)
        if end == latest_step:
            break
        start = end + 1
    return {
        "contract": "task35_fm_train_summary_v1",
        "latest_step": latest_step,
        "latest_loss": latest_loss,
        "total_steps": total_steps,
        "progress": latest_step / float(total_steps),
        "n_logged_steps": len(rows),
        "checkpoints_saved": saves,
        "alerts": sorted(set(alerts)),
        "windows": windows,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=root / "logs" / "task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1.log",
    )
    parser.add_argument(
        "--checkpoint-stem",
        type=Path,
        default=root / "checkpoints" / "task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "logs" / "task35_fm_train_summary.json",
    )
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--window", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_task35_fm_log(
        args.log.read_text(encoding="utf-8", errors="replace"),
        total_steps=args.total_steps,
        window=args.window,
        checkpoint_stem=args.checkpoint_stem,
    )
    text = json.dumps(summary, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
