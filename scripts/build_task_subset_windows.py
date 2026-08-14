#!/usr/bin/env python
"""按任务 ID 抽取 longtraj windows 子集（2026-08-14 DINO-main 替换实验）。

用法：
    python scripts/build_task_subset_windows.py \
        --input data/metaworld_longtraj_windows_h48_fine2_dino_clean.pt \
        --tasks 35 \
        --output data/metaworld_longtraj_windows_h48_dino35_clean.pt

行为：保留所有张量键（按 instruction_id 行过滤）与 frame_refs 行子集；
normalization 原样继承（与 all49-clean 同源，避免闭环 proprio 错位）；
metadata 更新 subset 字段，其余不变（tasks 保持 49 项供评测按全局索引选任务）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tasks", type=str, required=True,
                        help="逗号分隔的全局任务 ID，如 35")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_ids = [int(t.strip()) for t in args.tasks.split(",") if t.strip()]
    if not task_ids:
        raise ValueError("--tasks must list at least one task id")

    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    ids = payload["instruction_id"]
    mask = torch.isin(ids, torch.tensor(task_ids))
    rows = mask.nonzero().flatten().tolist()
    if not rows:
        raise ValueError(f"no rows for tasks {task_ids} in {args.input}")
    n = len(rows)
    out = {}
    for key, value in payload.items():
        if key in ("normalization", "metadata"):
            out[key] = value
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value[rows].clone() if value.shape[0] == len(ids) else value
        elif key == "frame_refs" and isinstance(value, list):
            out[key] = [value[i] for i in rows]
        else:
            out[key] = value

    metadata = dict(payload.get("metadata", {}) or {})
    metadata["subset_task_ids"] = sorted(set(int(ids[i]) for i in rows))
    metadata["subset_task_names"] = [
        str(metadata["tasks"][tid]) for tid in metadata["subset_task_ids"]
    ]
    metadata["n_subset_windows"] = n
    metadata["subset_source"] = str(args.input.resolve())
    out["metadata"] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(
        f"wrote {args.output}: {n} windows, tasks="
        f"{metadata['subset_task_ids']} ({metadata['subset_task_names']})"
    )


if __name__ == "__main__":
    main()
