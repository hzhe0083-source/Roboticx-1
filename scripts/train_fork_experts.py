#!/usr/bin/env python
"""Fork 专家：从 lerobot parquet demos 为任务对训两个小 BC 专家（state→action）。

pair 生死门需要同一状态下两个任务的专家动作（严格 fork 契约）。现有 50
demos/任务（parquet 的 observation.state 39 维 + action 4 维原始值），
训一个小 MLP（39→256→256→4，MSE）即可作为分支专家。

用法：
  python scripts/train_fork_experts.py \
      --pair drawer-close-v3 drawer-open-v3 \
      --out /media/ryan/robot-data/fork_experts/ \
      [--epochs 30] [--hidden 256]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

ROOT = Path("/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50")
# metaworld v3 环境名 -> parquet 任务文本
ENV_TO_TASK = {
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "faucet-close-v3": "Rotate the faucet clockwise",
    "faucet-open-v3": "Rotate the faucet counter-clockwise",
    "window-close-v3": "Push and close a window",
    "window-open-v3": "Push and open a window",
    "door-close-v3": "Close a door with a revolving joint",
    "door-open-v3": "Open a door with a revolving joint",
    "peg-insert-side-v3": "Insert a peg sideways",
    "peg-unplug-side-v3": "Unplug a peg sideways",
}


class BCExpert(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 256, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_demos(task_text: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (states [N,39], actions [N,4])——全部 episodes 拼接。"""
    eps = pq.read_table(ROOT / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    wanted = []
    for e in eps:
        t = e.get("tasks")
        key = t[0] if isinstance(t, list) and t else str(t)
        if key == task_text:
            wanted.append(e)
    if not wanted:
        raise ValueError(f"task '{task_text}' 无 episodes")
    states, actions = [], []
    for ep in wanted:
        c, f = ep["data/chunk_index"], ep["data/file_index"]
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        path = sorted(glob.glob(str(ROOT / f"data/chunk-{c:03d}/*.parquet")))[f]
        t = pq.read_table(path, columns=["index", "observation.state", "action"])
        pos = {g: l for l, g in enumerate(t.column("index").to_pylist())}
        i0 = pos[start]
        n = end - start
        states.append(np.asarray(t.column("observation.state").to_pylist()[i0 : i0 + n], dtype=np.float32))
        actions.append(np.asarray(t.column("action").to_pylist()[i0 : i0 + n], dtype=np.float32))
    return np.concatenate(states), np.concatenate(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True, metavar=("ENV_A", "ENV_B"))
    ap.add_argument("--out", type=Path, default=Path("/media/ryan/robot-data/fork_experts"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    for env_name in args.pair:
        task_text = ENV_TO_TASK.get(env_name)
        if task_text is None:
            raise ValueError(f"未登记环境名 {env_name}（ENV_TO_TASK）")
        states, actions = load_demos(task_text)
        # 标准化（z-score，避免 39 维量纲差异）
        sm, ss = states.mean(0), states.std(0) + 1e-6
        am, astd = actions.mean(0), actions.std(0) + 1e-6
        xs = torch.from_numpy((states - sm) / ss).float()
        ys = torch.from_numpy((actions - am) / astd).float()
        model = BCExpert(hidden=args.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        n = len(xs)
        for epoch in range(args.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, args.batch_size):
                idx = perm[i : i + args.batch_size]
                pred = model(xs[idx])
                loss = nn.functional.mse_loss(pred, ys[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"{env_name} epoch {epoch+1}/{args.epochs} mse={tot/n:.5f}")
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "state_mean": sm, "state_std": ss,
                "action_mean": am, "action_std": astd,
                "task_text": task_text,
            },
            args.out / f"{env_name}.pt",
        )
        print(f"saved {args.out / (env_name + '.pt')}")


if __name__ == "__main__":
    main()
