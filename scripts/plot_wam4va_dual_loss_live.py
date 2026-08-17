#!/usr/bin/env python3
"""Live L_flow + L_world curves from the WAM4VA training log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(
    r"^step=(\d+)\s+.*?flow=([0-9.eE+-]+).*?world=([0-9.eE+-]+)",
    re.M,
)
HEADER_RE = re.compile(r"^===== .* =====", re.M)


def parse_latest_run(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = [m.start() for m in HEADER_RE.finditer(text)]
    chunk = text[starts[-1] :] if starts else text
    rows = [(int(s), float(f), float(w)) for s, f, w in STEP_RE.findall(chunk)]
    cut = 0
    for i in range(1, len(rows)):
        if rows[i][0] < rows[i - 1][0]:
            cut = i
    rows = rows[cut:]
    if not rows:
        return np.array([]), np.array([]), np.array([])
    steps = np.array([r[0] for r in rows], dtype=np.int64)
    flow = np.array([r[1] for r in rows], dtype=np.float64)
    world = np.array([r[2] for r in rows], dtype=np.float64)
    return steps, flow, world


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if y.size < 3 or window < 3:
        return y
    k = min(window, y.size if y.size % 2 == 1 else y.size - 1)
    k = max(k, 3)
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/mw_hard2_wam4va_10k20k.log"),
    )
    parser.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    parser.add_argument("--smooth", type=int, default=21)
    parser.add_argument("--copy", type=float, default=0.046, help="world copy floor")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("artifacts/wam4va_joint_dual_loss.png"),
    )
    args = parser.parse_args()

    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    fig.suptitle("WAM4VA joint  ·  L_flow + L_world  (live)", fontsize=13, fontweight="bold")
    (flow_raw,) = axes[0].plot([], [], color="#93c5fd", lw=1.0, alpha=0.45)
    (flow_s,) = axes[0].plot([], [], color="#1d4ed8", lw=2.0, label="flow")
    (world_raw,) = axes[1].plot([], [], color="#86efac", lw=1.0, alpha=0.45)
    (world_s,) = axes[1].plot([], [], color="#15803d", lw=2.0, label="world")
    copy_line = axes[1].axhline(args.copy, color="#a16207", ls="--", lw=1.2, label=f"copy ~{args.copy}")
    axes[0].set_ylabel("L_flow")
    axes[1].set_ylabel("L_world")
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
    status = fig.text(0.01, 0.01, "", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)

    last_n = -1
    while plt.fignum_exists(fig.number):
        if args.log.is_file():
            steps, flow, world = parse_latest_run(args.log.read_text(errors="replace"))
        else:
            steps = flow = world = np.array([])
        if steps.size and steps.size != last_n:
            last_n = int(steps.size)
            flow_raw.set_data(steps, flow)
            world_raw.set_data(steps, world)
            sf = smooth(flow, args.smooth)
            sw = smooth(world, args.smooth)
            flow_s.set_data(steps, sf)
            world_s.set_data(steps, sw)
            axes[0].relim()
            axes[0].autoscale_view()
            axes[1].relim()
            axes[1].autoscale_view()
            axes[1].set_ylim(bottom=min(0.0, float(world.min()) * 0.9))
            status.set_text(
                f"n={steps.size}  step={int(steps[-1])}  "
                f"flow={flow[-1]:.4f}  world={world[-1]:.4f}"
            )
            if args.save is not None:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.save, dpi=120)
        fig.canvas.draw_idle()
        plt.pause(max(args.interval, 0.2))


if __name__ == "__main__":
    main()
