"""Prepare MetaWorld e2e video data (2026-08-06, user: V-JEPA full + Qwen top-layer e2e).

Outputs (data disk, memmap-friendly):
  <out>/video_frames.npy  uint8 memmap [N, T=4, W=4, 3, 384, 384]
  <out>/meta.pt           dict: instructions[str], proprio, previous_action,
                          actions, instruction_id, episode_id, normalization,
                          metadata{tasks, sequences_per_episode}

Usage:
  python prepare_metaworld_video.py --max-tasks 12 --output /media/ryan/robot-data/mw_e2e_pilot
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from prepare_metaworld import (
    decode_bytes,
    CONTROL_STRIDE,
    SEQUENCE_LENGTH,
    ACTION_HORIZON,
    VISION_WINDOW,
    VISION_STRIDE,
    MAX_CACHE_FRAMES,
)
from prepare_metaworld import robust_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare MW e2e video data (memmap)")
    p.add_argument("--root", type=Path,
                   default=Path("/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"))
    p.add_argument("--output", type=Path, required=True, help="output dir (video_frames.npy + meta.pt)")
    p.add_argument("--max-tasks", type=int, default=49)
    p.add_argument("--episodes-per-task", type=int, default=50)
    p.add_argument("--sequences-per-episode", type=int, default=4)
    p.add_argument("--image-size", type=int, default=384)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    tasks_by_text = {t["task"]: t for t in tasks}
    data_files = sorted(glob.glob(str(root / "data/chunk-000/*.parquet")))
    positions, arrays = {}, {}
    for path in data_files:
        table = pq.read_table(path)
        arr = table["observation.image"].to_numpy()
        positions[path] = table["index"].to_numpy()
        arrays[path] = arr
    global_rows = np.concatenate([positions[p] for p in data_files])
    global_file = {}
    for path in data_files:
        for g in positions[path]:
            global_file[g] = path

    def frame(row: int) -> np.ndarray:
        path = global_file[row]
        pos = positions[path]
        idx = int(np.searchsorted(pos, row))
        return decode_bytes(arrays[path][idx], args.image_size)

    plans = []  # (episode, task_text, start)
    for task_text, task in sorted(tasks_by_text.items())[: args.max_tasks]:
        ep_for_task = [e for e in episodes if e["task"] == task_text]
        if not ep_for_task:
            print(f"skip {task_text}: no episodes")
            continue
        ep = ep_for_task[0]
        n = len(ep["actions"])
        starts = [0]
        if args.sequences_per_episode > 1:
            step = n // args.sequences_per_episode
            starts += [step * k for k in range(1, args.sequences_per_episode)]
        for s in starts[: args.episodes_per_task]:
            plans.append((ep, task_text, s))
    print(f"tasks={len({p[1] for p in plans})} samples={len(plans)}")

    # 行号映射（与 prepare_metaworld 同契约：episode 内 frame 偏移）
    def row_of(episode: dict, local_frame: int) -> int:
        start = int(episode["index"])
        return start + local_frame

    # 数据收集（边解码边写 memmap，避免全量驻留）
    n = len(plans)
    N_FRAMES = n * SEQUENCE_LENGTH * VISION_WINDOW
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "video_frames.npy"
    mm = np.lib.format.open_memmap(
        str(frames_path), mode="w+",
        dtype=np.uint8,
        shape=(n, SEQUENCE_LENGTH, VISION_WINDOW, 3, args.image_size, args.image_size),
    )
    instructions: list[str] = []
    proprio_all, prev_all, act_all, iid_all, epid_all = [], [], [], [], []
    cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def get_frame(row: int) -> np.ndarray:
        hit = cache.get(row)
        if hit is not None:
            cache.move_to_end(row)
            return hit
        f = frame(row)
        cache[row] = f
        if len(cache) > MAX_CACHE_FRAMES:
            cache.popitem(last=False)
        return f

    for i, (ep, task_text, start) in enumerate(plans):
        for off in range(SEQUENCE_LENGTH):
            d = start + off * CONTROL_STRIDE
            for w in range(VISION_WINDOW):
                fidx = max(0, d - w * VISION_STRIDE)
                mm[i, off, w] = get_frame(row_of(ep, fidx))
        instructions.append(task_text)
        proprio_all.append([ep["observation.state"][d + o * CONTROL_STRIDE][:4]
                            for o in range(SEQUENCE_LENGTH)])
        prev = []
        for o in range(SEQUENCE_LENGTH):
            d0 = start + o * CONTROL_STRIDE
            prev.append(np.zeros(4) if d0 == 0 else np.asarray(ep["action"][d0 - 1]))
        prev_all.append(prev)
        act_all.append([[ep["action"][d + o * CONTROL_STRIDE + s]
                        for s in range(ACTION_HORIZON)]
                       for o in range(SEQUENCE_LENGTH)])
        iid_all.append(int(ep["task_index"]) if "task_index" in ep else i)
        epid_all.append(int(ep["index"]))
        if (i + 1) % 500 == 0:
            print(f"  sample {i + 1}/{n}", flush=True)

    mm.flush()
    # 归一化参数（与 features 管线一致：全部样本的 q01/q99）
    acts = np.concatenate([np.asarray(a).reshape(-1, 4) for a in act_all])
    states = np.concatenate([np.asarray(s).reshape(-1, 4) for s in proprio_all])
    aq01, aq99 = np.quantile(acts, [0.01, 0.99], axis=0)
    sq01, sq99 = np.quantile(states, [0.01, 0.99], axis=0)
    meta = {
        "instructions": instructions,
        "proprio": torch.as_tensor(np.asarray(proprio_all, dtype=np.float32)),
        "previous_action": torch.as_tensor(np.asarray(prev_all, dtype=np.float32)),
        "actions": torch.as_tensor(np.asarray(act_all, dtype=np.float32)),
        "instruction_id": torch.as_tensor(iid_all, dtype=torch.long),
        "episode_id": torch.as_tensor(epid_all, dtype=torch.long),
        "normalization": {
            "action_q01": torch.from_numpy(aq01.astype(np.float32)),
            "action_q99": torch.from_numpy(aq99.astype(np.float32)),
            "state_q01": torch.from_numpy(sq01.astype(np.float32)),
            "state_q99": torch.from_numpy(sq99.astype(np.float32)),
        },
        "metadata": {
            "tasks": sorted({p[1] for p in plans}),
            "sequences_per_episode": args.sequences_per_episode,
        },
    }
    torch.save(meta, out_dir / "meta.pt")
    size_gib = frames_path.stat().st_size / 1024**3
    print(f"saved -> {out_dir}  frames={frames_path.stat().st_size/1e9:.1f}GB meta.pt ok")


if __name__ == "__main__":
    main()
