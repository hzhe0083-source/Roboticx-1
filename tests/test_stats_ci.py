"""stats_ci 统计口径工具测试。"""
import numpy as np
import pytest

from stats_ci import binomial_wilson_ci, bootstrap_ci, fmt_ci, macro_bootstrap_ci


def test_binomial_wilson_ci_single_task_is_non_degenerate():
    est, lo, hi = binomial_wilson_ci(1, 10)
    assert est == pytest.approx(0.1)
    assert lo == pytest.approx(0.017876, abs=1e-6)
    assert hi == pytest.approx(0.404150, abs=1e-6)
    assert lo < est < hi


def test_binomial_wilson_ci_rejects_invalid_counts():
    with pytest.raises(ValueError):
        binomial_wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        binomial_wilson_ci(11, 10)
    with pytest.raises(ValueError):
        binomial_wilson_ci(0, 0)


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(7)
    values = rng.normal(0.5, 0.1, size=20000)
    est, lo, hi = bootstrap_ci(values, n_boot=2000, seed=0)
    assert est == pytest.approx(0.5, abs=0.005)
    assert lo < est < hi
    assert lo < 0.5 < hi  # 大样本下 CI 应覆盖真实均值


def test_macro_bootstrap_weights_groups_equally():
    # group A 有 100 个样本（值 1.0），group B 只有 1 个（值 0.0）。
    # 宏平均 = 0.5（任务等权），而不是样本级平均 0.99。
    values = np.concatenate([np.full(100, 1.0), np.full(1, 0.0)])
    groups = np.concatenate([np.zeros(100, dtype=int), np.ones(1, dtype=int)])
    est, lo, hi = macro_bootstrap_ci(values, groups, n_boot=2000, seed=0)
    assert est == pytest.approx(0.5)
    assert lo < 0.5 < hi


def test_macro_bootstrap_deterministic():
    rng = np.random.default_rng(11)
    values = rng.normal(size=80)
    groups = np.repeat(np.arange(8), 10)
    first = macro_bootstrap_ci(values, groups, seed=0)
    second = macro_bootstrap_ci(values, groups, seed=0)
    assert first == second


def test_macro_bootstrap_ci_recovers_true_macro_mean():
    rng = np.random.default_rng(5)
    n_groups = 20
    group_means = rng.uniform(0.2, 0.8, size=n_groups)
    values = []
    groups = []
    for g, mean in enumerate(group_means):
        n = int(rng.integers(5, 30))
        values.extend(rng.normal(mean, 0.05, size=n))
        groups.extend([g] * n)
    est, lo, hi = macro_bootstrap_ci(values, groups, n_boot=4000, seed=0)
    # 宏平均 = 各任务样本均值的均值（实现口径）
    expected = float(
        np.asarray(
            [np.mean(np.asarray(values)[np.asarray(groups) == g]) for g in range(n_groups)]
        ).mean()
    )
    assert est == pytest.approx(expected)
    assert lo <= est <= hi
    # 20 个任务、B=4000 下，CI 应覆盖生成分布的真实宏平均
    assert lo < group_means.mean() < hi


def test_fmt_ci():
    assert fmt_ci(0.04361, 0.04181, 0.04541) == "0.0436 [0.0418, 0.0454]"


def test_empty_values_rejected():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        macro_bootstrap_ci([], [])


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        macro_bootstrap_ci([1.0, 2.0], [0])
