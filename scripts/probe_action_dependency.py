#!/usr/bin/env python
"""G1 动作依赖探针：动作 (48×4) 能否线性预测 +6 几何变化 Δg。

输入两种模式：
  --cache-dir <dir>   读 WAM cache 记录（Task 5 的 WAMCacheDataset + manifest），
                      用 record["geo8"] 与 record["future_geo_target"]（k=+6 目标）
                      构造 Δg(+6) = g_future(+6) − g_current；split 严格走
                      wam_split_from_episode（episode 级，无泄漏）。
  --synthetic         无 cache 时合成数据自检（W_true 线性可预测 → 期望 PASS，
                      exit 0）；--synthetic --zero-correlation（Δg 与动作无关 →
                      期望 FAIL，exit 2）。

模型：岭回归 W，动作 [N,48,4] 展平 [N,192] → Δg [N,8]，闭式解
      W = (XᵀX + λI)⁻¹ XᵀY（float64 求解，CPU）。

三对照：
  (a) 场景不变基线 err_base：预测常数 0（另报 per-task 均值基线 err_task_mean 作参考）
  (b) 动作模型 err_action：训练集拟合 ridge，测试集 MSE
  (c) 打乱动作 err_shuffle：训练集动作行固定 seed 0 随机置换后同法训练，测试集 MSE

门 G1（spec §7）：
  improvement      err_base − err_action ≥ 0.10 × err_base
  shuffle_gain_loss (err_shuffle − err_action) ≥ 0.5 × (err_base − err_action)

输出：三误差与两门判定 PASS/FAIL。
exit code：0 = PASS（两门全过），2 = FAIL（任一门不过），3 = 数据不足。

用法：
  python scripts/probe_action_dependency.py --synthetic --seed 0
  python scripts/probe_action_dependency.py --synthetic --zero-correlation
  python scripts/probe_action_dependency.py --cache-dir <wam_cache_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTION_HORIZON = 48
ACTION_DIM = 4
X_DIM = ACTION_HORIZON * ACTION_DIM  # 192
GEO_DIM = 8


def _load_split_fn():
    """优先用 va_compound.wam_cache.wam_split_from_episode；未集成时用同规则回退。"""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from va_compound.wam_cache import wam_split_from_episode  # type: ignore

        return wam_split_from_episode
    except Exception as exc:  # noqa: BLE001 —— Task 5 可能尚未落地
        print(f"[warn] 无法导入 va_compound.wam_cache.wam_split_from_episode（{exc}）", file=sys.stderr)
        print("[warn] 使用本地同规则回退：episode_id % 10 -> 0..7 train, 8 val, 9 test", file=sys.stderr)

        def _fallback(episode_ids, task_ids):
            rem = torch.as_tensor(episode_ids, dtype=torch.long) % 10
            train = (rem >= 0) & (rem <= 7)
            val = rem == 8
            test = rem == 9
            return train, val, test

        return _fallback


def _build_synthetic(n: int, zero_correlation: bool, seed: int):
    """合成数据：X ~ N(0,1)，Δg = X@W_true + noise（zero_correlation 时 Δg 纯噪声）。"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, X_DIM, generator=gen)
    if zero_correlation:
        dy = 0.3 * torch.randn(n, GEO_DIM, generator=gen)
    else:
        w_true = torch.randn(X_DIM, GEO_DIM, generator=gen) / (X_DIM ** 0.5)
        dy = x @ w_true + 0.3 * torch.randn(n, GEO_DIM, generator=gen)
    # 合成 episode id：保证每样本一个 episode，split 与真实数据同规则
    episode_ids = torch.arange(n, dtype=torch.long) % 97
    task_ids = torch.arange(n, dtype=torch.long) % 49
    return x, dy, episode_ids.tolist(), task_ids.tolist()


def _load_cache_records(cache_dir: str):
    """防御式读取 WAM cache 全部记录（train/val/test 分片拼接）。返回
    (actions [N,48,4], dg [N,8], episode_ids [N], task_ids [N])。"""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from va_compound.wam_cache import WAMCacheDataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"va_compound.wam_cache 不可导入（{exc}）——Task 5 尚未落地时请用 --synthetic 自检"
        ) from exc

    root = Path(cache_dir)
    manifest = None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            from va_compound.wam_cache import WAMCacheManifest  # type: ignore

            manifest = WAMCacheManifest(**json.loads(manifest_path.read_text()))
        except Exception:  # noqa: BLE001
            manifest = None  # 让 dataset 自行加载

    actions, dgs, ep_ids, task_ids = [], [], [], []
    for split in ("train", "val", "test"):
        try:
            ds = WAMCacheDataset(str(root), manifest, split=split) if manifest is not None else WAMCacheDataset(str(root), split=split)  # noqa: E501
        except TypeError:
            ds = WAMCacheDataset(str(root), manifest) if manifest is not None else WAMCacheDataset(str(root))  # noqa: E501
        except Exception:
            continue  # 该 split 分片不存在或不可读
        for i in range(len(ds)):
            rec = ds[i]
            a = rec.get("actions")
            if a is None:
                continue
            a = torch.as_tensor(a, dtype=torch.float32)
            if a.shape != (ACTION_HORIZON, ACTION_DIM):
                continue
            g0 = rec.get("geo8", rec.get("geo_current"))
            fut = rec.get("future_geo_target", rec.get("future_geo"))
            if g0 is None or fut is None:
                continue
            g0 = torch.as_tensor(g0, dtype=torch.float32).reshape(-1)
            fut = torch.as_tensor(fut, dtype=torch.float32)
            if fut.ndim == 3:  # [3,2,8] → k=+6 的 g_future
                gf = fut[0, 0]
            elif fut.ndim == 2:
                gf = fut[0]
            else:
                gf = fut
            dg = (gf.reshape(-1) - g0)
            if g0.numel() != GEO_DIM or dg.numel() != GEO_DIM:
                continue
            if not torch.isfinite(dg).all():
                continue
            # 无效样本 mask（若有）
            valid = rec.get("valid", rec.get("mask", None))
            if valid is not None and torch.as_tensor(valid).numel() == 1 and not bool(torch.as_tensor(valid).item()):
                continue
            actions.append(a.reshape(-1))
            dgs.append(dg)
            ep_ids.append(int(rec.get("episode_id", len(ep_ids))))
            task_ids.append(int(rec.get("task_id", 0)))
    if not actions:
        raise RuntimeError(f"{cache_dir} 未读到任何有效记录")
    return (
        torch.stack(actions),
        torch.stack(dgs),
        ep_ids,
        task_ids,
    )


def fit_ridge(x_train: torch.Tensor, y_train: torch.Tensor, l2: float) -> torch.Tensor:
    """闭式解 W = (XᵀX + λI)⁻¹ XᵀY（float64）。返回 [192, 8]。"""
    x = x_train.to(torch.float64)
    y = y_train.to(torch.float64)
    gram = x.T @ x + l2 * torch.eye(x.shape[1], dtype=torch.float64)
    return torch.linalg.solve(gram, x.T @ y)


def mse(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(((pred.to(torch.float64) - y.to(torch.float64)) ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=str, default=None, help="WAM cache 目录（manifest.json + 分片）")
    ap.add_argument("--synthetic", action="store_true", help="合成数据自检（不读 cache）")
    ap.add_argument("--zero-correlation", action="store_true", help="合成模式：Δg 与动作无关的纯噪声（期望 FAIL）")
    ap.add_argument("--ridge-l2", type=float, default=1e-3, help="岭回归正则系数")
    ap.add_argument("--seed", type=int, default=0, help="数据/合成随机种子")
    ap.add_argument("--n", type=int, default=2048, help="合成样本数")
    args = ap.parse_args()

    if not args.synthetic and not args.cache_dir:
        ap.error("需要 --cache-dir 或 --synthetic 之一")

    if args.synthetic:
        x, dg, ep_ids, task_ids = _build_synthetic(args.n, args.zero_correlation, args.seed)
        src = f"synthetic (zero_correlation={args.zero_correlation}, seed={args.seed}, n={args.n})"
    else:
        try:
            x, dg, ep_ids, task_ids = _load_cache_records(args.cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[data-insufficient] 读取 cache 失败：{exc}", file=sys.stderr)
            return 3
        src = f"cache={args.cache_dir}"

    n = x.shape[0]
    train_mask, _, test_mask = _load_split_fn()(ep_ids, task_ids)
    train_mask = torch.as_tensor(train_mask, dtype=torch.bool)
    test_mask = torch.as_tensor(test_mask, dtype=torch.bool)
    if train_mask.shape[0] != n:
        train_mask = train_mask[:n]
        test_mask = test_mask[:n]
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    if n_train < X_DIM or n_test < 1:
        print(
            f"[data-insufficient] 需要 n_train>={X_DIM} 且 n_test>=1，"
            f"实际 n_train={n_train}, n_test={n_test}（总样本 {n}）",
            file=sys.stderr,
        )
        return 3

    x_train, y_train = x[train_mask], dg[train_mask]
    x_test, y_test = x[test_mask], dg[test_mask]
    task_test = torch.as_tensor(task_ids, dtype=torch.long)[test_mask]
    task_train = torch.as_tensor(task_ids, dtype=torch.long)[train_mask]

    # (b) 动作模型
    w_action = fit_ridge(x_train, y_train, args.ridge_l2)
    err_action = mse(x_test.to(torch.float64) @ w_action, y_test)

    # (c) 打乱动作：训练集动作行固定 seed 0 置换（可复现），同法训练
    shuffle_gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_train, generator=shuffle_gen)
    w_shuffle = fit_ridge(x_train[perm], y_train, args.ridge_l2)
    err_shuffle = mse(x_test.to(torch.float64) @ w_shuffle, y_test)

    # (a) 场景不变基线：预测常数 0；per-task 均值基线仅作参考
    err_base = mse(torch.zeros_like(y_test), y_test)
    mean_by_task = {}
    for t in task_train.unique().tolist():
        mean_by_task[t] = y_train[task_train == t].mean(dim=0)
    pred_task_mean = torch.stack(
        [mean_by_task.get(t, y_train.mean(dim=0)) for t in task_test.tolist()]
    )
    err_task_mean = mse(pred_task_mean, y_test)

    # 门 G1 判定
    improvement = err_base - err_action
    gate_improvement = improvement >= 0.10 * err_base
    shuffle_gain = err_shuffle - err_action
    gate_shuffle = shuffle_gain >= 0.5 * improvement

    print(f"G1 action-dependency probe: {src}")
    print(f"  n_train={n_train}  n_test={n_test}  ridge_l2={args.ridge_l2}")
    print(f"  err_base (constant 0)      = {err_base:.6f}")
    print(f"  err_task_mean (reference)  = {err_task_mean:.6f}")
    print(f"  err_action (ridge)         = {err_action:.6f}")
    print(f"  err_shuffle (seed-0 perm)  = {err_shuffle:.6f}")
    print(f"  improvement                = {improvement:.6f}  "
          f"(need >= {0.10 * err_base:.6f})  -> {'PASS' if gate_improvement else 'FAIL'}")
    print(f"  shuffle_gain               = {shuffle_gain:.6f}  "
          f"(need >= {0.5 * improvement:.6f})  -> {'PASS' if gate_shuffle else 'FAIL'}")

    if gate_improvement and gate_shuffle:
        print("G1 RESULT: PASS")
        return 0
    print("G1 RESULT: FAIL")
    return 2


if __name__ == "__main__":
    sys.exit(main())
