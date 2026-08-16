"""统计口径工具：单任务二项区间与多任务宏平均 bootstrap。

口径约定：
- 单任务成功率：独立 trial 是采样单位，使用 Wilson score interval；只有一个
  task 时按 task bootstrap 会退化为伪窄的 ``[p, p]``，禁止这样报告。
- 多任务宏平均：先对每个任务（group）取样本均值，再对任务取平均；以任务为
  有放回重采样单元做 percentile bootstrap（默认 B=2000、固定 seed）。

用法：
    from stats_ci import binomial_wilson_ci, macro_bootstrap_ci
    est, lo, hi = binomial_wilson_ci(successes=1, trials=10)
    macro, lo, hi = macro_bootstrap_ci(values, group_ids, seed=0)
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "binomial_wilson_ci",
    "bootstrap_ci",
    "macro_bootstrap_ci",
    "fmt_ci",
]


def binomial_wilson_ci(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    """Binomial proportion with a two-sided Wilson score interval.

    Use this trial-level interval for a single task. Task-level bootstrap is
    degenerate with one task and must not be presented as uncertainty there.
    """
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes/trials must be integer counts")
    if int(successes) != successes or int(trials) != trials:
        raise ValueError("successes/trials must be integer counts")
    successes, trials = int(successes), int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denom
    )
    return p, max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(
    values,
    stat=np.mean,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """``stat`` 在 ``values`` 上的百分位 bootstrap 95% CI。

    重采样单元是单个样本；估计量默认均值。返回 (估计值, 下界, 上界)。
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat(rng.choice(values, size=values.size, replace=True))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(values)), float(lo), float(hi)


def macro_bootstrap_ci(
    values,
    group_ids,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """宏平均 + 以 group（任务）为重采样单元的 bootstrap 95% CI。

    每个 group 先取组内均值，再对 group 有放回重采样求均值——重采样单元与
    宏平均口径一致。返回 (宏平均, 下界, 上界)。
    """
    values = np.asarray(values, dtype=float)
    group_ids = np.asarray(group_ids)
    if values.size != group_ids.size:
        raise ValueError("values and group_ids must have the same length")
    if values.size == 0:
        raise ValueError("empty values")
    groups = np.asarray(
        [float(values[group_ids == g].mean()) for g in np.unique(group_ids)]
    )
    if groups.size == 0:
        raise ValueError("no groups")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(groups, size=groups.size, replace=True).mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(groups.mean()), float(lo), float(hi)


def fmt_ci(est: float, lo: float, hi: float, digits: int = 4) -> str:
    """格式化为 ``0.0436 [0.0418, 0.0454]``。"""
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
