#!/usr/bin/env python
"""闭环评估日志 → 可复现的成功率与 95% CI 汇总。

输入：eval_metaworld.py 输出日志（每任务行 `task <name>: <wins>/<trials>`，
汇总行 `CLOSED-LOOP SUCCESS: <total>/<trials> = <pct>%`）。
输出（stdout，一行一指标，便于 grep 与登记）：
  aggregate:  n/total = pct%（简单合并）
  macro_mean: 任务级平均成功率（49 任务同权）
  macro_ci95: 宏平均 ± 1.96·SE(任务率)（项目既有口径，2026-08-08 沿用的
              [22.6, 41.4] 即此口径：macro_mean ± 1.96·std/√n_tasks）
  wilson_ci95: 聚合比例 Wilson 区间（保守替代口径）

用法：
  python scripts/closedloop_ci.py logs/<eval>.log [logs/<eval2>.log ...]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

TASK_RE = re.compile(r"^task (.+): (\d+)/(\d+)$")
SUM_RE = re.compile(r"CLOSED-LOOP SUCCESS: (\d+)/(\d+)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def summarize(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    wins, trials = [], []
    for line in text.splitlines():
        m = TASK_RE.match(line)
        if m:
            wins.append(int(m.group(2)))
            trials.append(int(m.group(3)))
    if not wins:
        print(f"{path}: no per-task lines found")
        return
    n_tasks = len(wins)
    agg_k, agg_n = sum(wins), sum(trials)
    rates = [w / t for w, t in zip(wins, trials) if t > 0]
    macro = sum(rates) / len(rates)
    if len(rates) > 1:
        sd = (sum((r - macro) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5
        se = sd / math.sqrt(len(rates))
    else:
        se = 0.0
    lo_w, hi_w = wilson(agg_k, agg_n)
    print(f"{path.name}: tasks={n_tasks}")
    print(f"  aggregate: {agg_k}/{agg_n} = {agg_k / agg_n:.1%}")
    print(f"  macro_mean: {macro:.1%}")
    print(f"  macro_ci95: [{macro - 1.96 * se:.1%}, {macro + 1.96 * se:.1%}]")
    print(f"  wilson_ci95: [{lo_w:.1%}, {hi_w:.1%}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        summarize(Path(p))
