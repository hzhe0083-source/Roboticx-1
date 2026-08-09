#!/usr/bin/env python
"""把 49 任务语言缓存从 REF（metaworld_fullframe_executed.pt）补进 longtraj windows。

背景（2026-08-09）：build_longtraj_features.py phase1 的 windows 文件没有
language_hidden/language_mask（FeatureDataset REQUIRED keys），而 REF 的语言缓存
是按任务恒等的（同任务所有窗口 13 步编码完全相同）——直接提取每任务一行，
按 instruction_id 广播到 [n, 13, 2048] + [n, 13]，与 E6 冻结 Qwen 口径 100% 一致
（零重算）。

用法：python scripts/add_language_cache_to_longtraj.py [--horizon 8|48]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "metaworld_fullframe_executed.pt"


def main(horizon: int) -> None:
    path = ROOT / "data" / f"metaworld_longtraj_windows_h{horizon}.pt"
    win = torch.load(path, map_location="cpu", weights_only=False)
    if "language_hidden" in win:
        print(f"{path}: language_hidden 已存在（{tuple(win['language_hidden'].shape)}），跳过")
        return
    ref = torch.load(REF, map_location="cpu", weights_only=True)
    lh, lm = ref["language_hidden"], ref["language_mask"]
    n_tasks = len(ref["metadata"]["tasks"])
    task_lh = torch.stack([lh[(ref["instruction_id"] == t).nonzero()[0][0]]
                           for t in range(n_tasks)])
    task_lm = torch.stack([lm[(ref["instruction_id"] == t).nonzero()[0][0]]
                           for t in range(n_tasks)])
    inst = win["instruction_id"]
    win["language_hidden"] = task_lh[inst]  # [n, 13, 2048] fp16
    win["language_mask"] = task_lm[inst]    # [n, 13] bool
    torch.save(win, path)
    print(f"[out] {path}: language_hidden {tuple(win['language_hidden'].shape)} "
          f"{win['language_hidden'].dtype}, mask {tuple(win['language_mask'].shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=8)
    args = ap.parse_args()
    main(args.horizon)
