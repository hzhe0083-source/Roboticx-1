"""统计口径工具：宏平均 + bootstrap 95% CI（固定种子）。

口径约定（对应 VA_COMPOUND_REPORT.md §3.1 / 规格阶段 3）：
- 宏平均：先对每个任务（group）取样本均值，再对任务取平均；样本数不均的
  任务不会主导总体数字。
- 95% CI：以任务为有放回重采样单元做 bootstrap（默认 B=2000、固定 seed
  保证可复现），取百分位区间 [2.5%, 97.5%]。bootstrap 对分布不作正态假设，
  二值成功率同样适用。

用法：
    from stats_ci import macro_bootstrap_ci, fmt_ci
    est, lo, hi = macro_bootstrap_ci(values, group_ids, n_boot=2000, seed=0)
    print(fmt_ci(est, lo, hi))
"""
from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_ci", "macro_bootstrap_ci", "fmt_ci"]


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
