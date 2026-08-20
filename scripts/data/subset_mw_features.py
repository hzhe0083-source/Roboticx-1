"""从 metaworld_features_v2_full.pt 按任务描述提取子集（VLA-RL pilot/MT10 评估用）。

用法:
  python scripts/data/subset_mw_features.py --tasks "Push and close a drawer,Reach a goal position,Push the puck to a goal" \
      --output data/mw_subset_pilot.pt
  python scripts/data/subset_mw_features.py --n 10 --output data/mw_subset_mt10.pt   # 取 metadata 前 10 任务

输出与原始文件同构（instruction_id 重编号为 0..k-1，metadata.tasks = 选中描述）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import json

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/metaworld_features_v2_full.pt"))
    parser.add_argument("--tasks", type=str, default="", help="逗号分隔的任务描述；与 --n 二选一")
    parser.add_argument("--n", type=int, default=0, help="取 metadata 前 N 任务（与 --tasks 二选一）")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = torch.load(args.features, map_location="cpu", weights_only=False)
    tasks = d["metadata"]["tasks"]

    if args.tasks:
        selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in selected if t not in tasks]
        if missing:
            raise ValueError(f"tasks not in metadata: {missing[:5]}")
        keep_indices = [tasks.index(t) for t in selected]
    elif args.n > 0:
        keep_indices = list(range(min(args.n, len(tasks))))
    else:
        raise ValueError("specify --tasks or --n")
    selected_tasks = [tasks[i] for i in keep_indices]

    keep_mask = torch.isin(d["instruction_id"], torch.tensor(keep_indices))
    print(f"source samples={d['instruction_id'].shape[0]} selected={int(keep_mask.sum())} "
          f"tasks={len(selected_tasks)}")

    out = {k: v for k, v in d.items() if k not in ("normalization", "metadata")}
    out["instruction_id"] = torch.full_like(d["instruction_id"], -1)
    for new_id, old_id in enumerate(keep_indices):
        out["instruction_id"][d["instruction_id"] == old_id] = new_id
    out = {k: v[keep_mask] for k, v in out.items()}
    out["normalization"] = d["normalization"]
    out["metadata"] = dict(d["metadata"])
    out["metadata"]["tasks"] = selected_tasks
    out["metadata"]["contract"] = d["metadata"].get("contract", "language_conditioned_mt50")

    torch.save(out, args.output)
    print(f"saved={args.output} size={args.output.stat().st_size / 1e9:.2f}GiB "
          f"shape={out['vision_tokens'].shape}")


if __name__ == "__main__":
    main()
