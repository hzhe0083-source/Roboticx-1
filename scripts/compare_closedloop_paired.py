#!/usr/bin/env python
"""Compare two fixed-seed closed-loop logs with a paired task bootstrap.

Both logs must come from ``eval_metaworld.py`` after it started emitting
``trial task=... trial=... seed=... success=...`` lines.  Trials are paired by
task/trial/seed, then task-level success-rate differences are bootstrapped.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRIAL_RE = re.compile(
    r"^trial task=(\d+) trial=(\d+) seed=(\d+) success=([01])$"
)


@dataclass(frozen=True)
class PairedComparison:
    baseline_rate: float
    candidate_rate: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    candidate_ci_low: float
    candidate_ci_high: float
    n_tasks: int
    n_trials: int

    @property
    def empirical_improvement(self) -> bool:
        return self.delta > 0.0

    @property
    def improvement_confirmed(self) -> bool:
        return self.delta_ci_low > 0.0

    @property
    def target_60_reached(self) -> bool:
        return self.candidate_rate > 0.60

    @property
    def target_60_confirmed(self) -> bool:
        """Strong task-generalization gate: candidate task CI is above 60%."""
        return self.candidate_ci_low > 0.60


def parse_trials(path: Path) -> dict[tuple[int, int, int], int]:
    rows: dict[tuple[int, int, int], int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRIAL_RE.match(line)
        if match is None:
            continue
        key = tuple(int(match.group(index)) for index in (1, 2, 3))
        if key in rows:
            raise ValueError(f"duplicate trial {key} in {path}")
        rows[key] = int(match.group(4))
    if not rows:
        raise ValueError(f"no trial-level results in {path}")
    return rows


def compare_logs(
    baseline_path: Path,
    candidate_path: Path,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedComparison:
    baseline = parse_trials(baseline_path)
    candidate = parse_trials(candidate_path)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())[:5]
        extra = sorted(candidate.keys() - baseline.keys())[:5]
        raise ValueError(
            "paired trial keys differ: "
            f"missing_from_candidate={missing}, extra_in_candidate={extra}"
        )
    keys = sorted(baseline)
    expected_keys = {
        (task_id, trial, 1000 * task_id + trial)
        for task_id in range(49)
        for trial in range(10)
    }
    if set(keys) != expected_keys:
        missing = sorted(expected_keys - set(keys))[:5]
        extra = sorted(set(keys) - expected_keys)[:5]
        raise ValueError(
            "official all49 gate requires exactly 49 tasks x 10 trials with "
            "seed=1000*task+trial: "
            f"missing={missing}, extra={extra}, rows={len(keys)}"
        )
    task_ids = sorted({key[0] for key in keys})
    task_baseline = np.asarray(
        [
            np.mean([baseline[key] for key in keys if key[0] == task_id])
            for task_id in task_ids
        ],
        dtype=float,
    )
    task_candidate = np.asarray(
        [
            np.mean([candidate[key] for key in keys if key[0] == task_id])
            for task_id in task_ids
        ],
        dtype=float,
    )
    task_deltas = np.asarray(
        [
            np.mean(
                [candidate[key] - baseline[key] for key in keys if key[0] == task_id]
            )
            for task_id in task_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    candidate_boots = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.integers(0, task_deltas.size, size=task_deltas.size)
        boots[index] = task_deltas[sampled].mean()
        candidate_boots[index] = task_candidate[sampled].mean()
    low, high = np.percentile(boots, [2.5, 97.5])
    candidate_low, candidate_high = np.percentile(candidate_boots, [2.5, 97.5])
    baseline_rate = float(np.mean([baseline[key] for key in keys]))
    candidate_rate = float(np.mean([candidate[key] for key in keys]))
    return PairedComparison(
        baseline_rate=baseline_rate,
        candidate_rate=candidate_rate,
        delta=candidate_rate - baseline_rate,
        delta_ci_low=float(low),
        delta_ci_high=float(high),
        candidate_ci_low=float(candidate_low),
        candidate_ci_high=float(candidate_high),
        n_tasks=len(task_ids),
        n_trials=len(keys),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired fixed-seed closed-loop improvement gate"
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    result = compare_logs(args.baseline, args.candidate, n_boot=args.bootstrap)
    print(f"baseline:  {result.baseline_rate:.1%}")
    print(f"candidate: {result.candidate_rate:.1%}")
    print(
        f"paired delta: {result.delta:+.1%} "
        f"[task-bootstrap 95% CI: {result.delta_ci_low:+.1%}, "
        f"{result.delta_ci_high:+.1%}]"
    )
    print(
        f"candidate task-bootstrap 95% CI: "
        f"[{result.candidate_ci_low:.1%}, {result.candidate_ci_high:.1%}]"
    )
    print(f"empirical improvement: {result.empirical_improvement}")
    print(f"improvement confirmed (CI low > 0): {result.improvement_confirmed}")
    print(f"candidate > 60%: {result.target_60_reached}")
    print(f"candidate CI low > 60%: {result.target_60_confirmed}")
    accepted = result.improvement_confirmed and result.target_60_confirmed
    print(f"FINAL GATE: {'ACCEPT CANDIDATE' if accepted else 'REJECT / KEEP BASELINE'}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
