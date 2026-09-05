#!/usr/bin/env python3
"""Plot total/FM/World loss, gradient norm, and World gates from a train log."""

from __future__ import annotations

import argparse
import math
import re
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(
    rf"^step=(?P<step>\d+).*?\bloss=(?P<loss>{NUMBER}) "
    rf"flow=(?P<flow>{NUMBER}).*?\bworld=(?P<world>{NUMBER}).*?"
    rf"\bgrad=(?P<grad>{NUMBER})"
)
GATE_RE = re.compile(
    rf"\bworld_gate_mean=(?P<mean>{NUMBER}) "
    rf"world_gate_max=(?P<max>{NUMBER})"
)


def parse_log(path: Path) -> list[dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {key: float(match.group(key)) for key in ("loss", "flow", "world", "grad")}
        gate = GATE_RE.search(line)
        if gate:
            row["gate_mean"] = float(gate.group("mean"))
            row["gate_max"] = float(gate.group("max"))
        if all(math.isfinite(value) for value in row.values()):
            rows[int(match.group("step"))] = row
    if not rows:
        raise SystemExit(f"no training rows found in {path}")
    return [{"step": float(step), **rows[step]} for step in sorted(rows)]


def moving_average(values: list[float], width: int) -> list[float]:
    window: deque[float] = deque()
    total = 0.0
    result = []
    for value in values:
        window.append(value)
        total += value
        if len(window) > width:
            total -= window.popleft()
        result.append(total / len(window))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")

    rows = parse_log(args.log)
    steps = [row["step"] for row in rows]
    output = args.output or args.log.with_suffix(".curves.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {"loss": "#555555", "flow": "#d62728", "world": "#1f77b4"}
    for key, label in (("loss", "total"), ("flow", "FM"), ("world", "World")):
        values = [row[key] for row in rows]
        axes[0].plot(steps, values, color=colors[key], alpha=0.18, linewidth=0.7)
        axes[0].plot(
            steps,
            moving_average(values, args.window),
            color=colors[key],
            linewidth=1.8,
            label=f"{label} MA{args.window}",
        )
    axes[0].set_ylabel("loss")
    axes[0].legend(ncol=3)

    grads = [row["grad"] for row in rows]
    axes[1].plot(steps, grads, color="#9467bd", alpha=0.2, linewidth=0.7)
    axes[1].plot(steps, moving_average(grads, args.window), color="#9467bd", linewidth=1.8)
    axes[1].set_ylabel("grad norm")

    gate_rows = [row for row in rows if "gate_mean" in row]
    if gate_rows:
        gate_steps = [row["step"] for row in gate_rows]
        axes[2].plot(
            gate_steps, [row["gate_mean"] for row in gate_rows], "o-", label="mean"
        )
        axes[2].plot(
            gate_steps, [row["gate_max"] for row in gate_rows], "o-", label="max"
        )
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "Gate is logged every 10 steps", ha="center", va="center")
    axes[2].set_ylabel("World gate")
    axes[2].set_xlabel("step")

    for axis in axes:
        axis.grid(alpha=0.25)
    progress = f" / {args.total_steps}" if args.total_steps else ""
    fig.suptitle(f"{args.log.name} — step {int(steps[-1])}{progress}")
    fig.tight_layout()
    temporary = output.with_suffix(output.suffix + ".tmp")
    fig.savefig(temporary, format="png", dpi=150)
    temporary.replace(output)
    plt.close(fig)

    latest = rows[-1]
    print(
        f"saved {output} | step={int(latest['step'])} "
        f"loss={latest['loss']:.6f} FM={latest['flow']:.6f} "
        f"World={latest['world']:.6f} grad={latest['grad']:.6f}"
    )


if __name__ == "__main__":
    main()
