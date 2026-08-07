"""Prepare LIBERO same-scene subsets as raw video frames + instructions.

Extension of prepare_libero.py for end-to-end fine-tuning: stores the causal
4-frame windows as uint8 pixels (pnpw_video.pt format) so V-JEPA and Qwen can
be trained online.  Three scenes x four tasks (12 instructions) are merged
into one dataset with a single Qwen re-encode.
"""
from __future__ import annotations

import argparse
import glob
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from prepare_pnpw_features import robust_normalize


def decode_bytes(row: dict) -> np.ndarray:
    raw = row.get("bytes")
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"missing image bytes at path={row.get('path')}")
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def preprocess_uint8(frames: list[np.ndarray], image_size: int) -> Tensor:
    """uint8 frames -> center-cropped uint8 [1,W,3,S,S] clip."""
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    batch, channels, height, width = video.shape
    if height < width:
        resized_height = image_size
        resized_width = round(width * image_size / height)
    else:
        resized_width = image_size
        resized_height = round(height * image_size / width)
    flat = F.interpolate(
        video,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - image_size) // 2
    left = (resized_width - image_size) // 2
    flat = flat[:, :, top : top + image_size, left : left + image_size]
    return flat.round().clamp(0, 255).to(torch.uint8).reshape(batch, channels, image_size, image_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LIBERO subsets as raw video frames")
    parser.add_argument("--dataset", type=Path, default=Path(
        "/home/ryan/Documents/robot/benchmark_data/raw/libero100/kevin_libero100_lerobot"
    ))
    parser.add_argument("--output", type=Path, default=Path("data/libero_video.pt"))
    parser.add_argument("--scenes", nargs="+", default=("LIVING_ROOM_SCENE2", "KITCHEN_SCENE2", "STUDY_SCENE1"))
    parser.add_argument("--tasks-per-scene", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--sequences-per-episode", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--control-stride", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--vision-window", type=int, default=4)
    parser.add_argument("--vision-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()

    by_task: dict[str, list[dict]] = {}
    for episode in episodes:
        raw_task = episode.get("tasks") or episode.get("task") or ""
        if isinstance(raw_task, list):
            raw_task = raw_task[0] if raw_task else ""
        by_task.setdefault(str(raw_task).strip(), []).append(episode)

    selected_tasks = []
    plans = []
    episode_counter = 0  # 全局轨迹序号（跨任务唯一，2026-08-05 数据合同）
    for scene in args.scenes:
        scene_id = args.scenes.index(scene)
        scene_tasks = sorted(t for t in by_task if t.startswith(scene))[: args.tasks_per_scene]
        if not scene_tasks:
            raise ValueError(f"no tasks with scene prefix {scene}")
        selected_tasks.extend(scene_tasks)
        for task in scene_tasks:
            for episode in by_task[task][: args.episodes_per_task]:
                length = int(episode["length"])
                required_span = (args.sequence_length - 1) * args.control_stride + (
                    args.action_horizon - 1
                )
                last_start = length - 1 - required_span
                if last_start < 0:
                    continue
                stride = max(1, last_start // max(args.sequences_per_episode, 1))
                for start in range(0, last_start + 1, stride)[: args.sequences_per_episode]:
                    plans.append((episode, task, start, episode_counter, scene_id))
                episode_counter += 1
    if not plans:
        raise ValueError("no plans produced")
    print(f"tasks={len(selected_tasks)} samples={len(plans)} episodes={episode_counter}")

    data_files = sorted(glob.glob(str(root / "data/chunk-000/*.parquet")))
    file_meta = []
    for path in data_files:
        table = pq.read_table(path, columns=["index"])
        meta = table.column("index").to_pylist()
        file_meta.append((path, meta))

    def global_row(episode: dict, local_frame: int) -> int:
        return int(episode["dataset_from_index"]) + local_frame

    needed_rows: dict[str, set[int]] = {}
    for episode, _task, start, _trid, _scene in plans:
        for offset in range(args.sequence_length):
            decision = start + offset * args.control_stride
            for frame in range(decision - (args.vision_window - 1) * args.vision_stride, decision + 1, args.vision_stride):
                row = global_row(episode, max(0, frame))
                for path, meta in file_meta:
                    if meta[0] <= row <= meta[-1]:
                        needed_rows.setdefault(path, set()).add(row)
                        break
            for step in range(args.action_horizon):
                row = global_row(episode, decision + step)
                for path, meta in file_meta:
                    if meta[0] <= row <= meta[-1]:
                        needed_rows.setdefault(path, set()).add(row)
                        break
            previous_row = max(0, global_row(episode, decision - 1))
            for path, meta in file_meta:
                if meta[0] <= previous_row <= meta[-1]:
                    needed_rows.setdefault(path, set()).add(previous_row)
                    break

    frame_cache: dict[int, np.ndarray] = {}
    for path, rows in needed_rows.items():
        table = pq.read_table(path, columns=["index", "observation.images.image"])
        index_col = table.column("index").to_pylist()
        arr = table.column("observation.images.image").combine_chunks().to_pylist()
        position = {g: local for local, g in enumerate(index_col)}
        for row in rows:
            frame_cache[row] = decode_bytes(arr[position[row]])
    print(f"frames decoded: {len(frame_cache)}")

    if args.dry_run:
        return

    # Actions/states with per-row normalized values.
    stat_actions, stat_states = [], []
    for path, _rows in needed_rows.items():
        table = pq.read_table(path, columns=["action", "observation.state"])
        stat_actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        stat_states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
    raw_actions = np.concatenate(stat_actions)
    raw_states = np.concatenate(stat_states)
    action_low, action_high = np.quantile(raw_actions, (0.01, 0.99), axis=0)
    state_low, state_high = np.quantile(raw_states, (0.01, 0.99), axis=0)

    norm_action: dict[int, np.ndarray] = {}
    norm_state: dict[int, np.ndarray] = {}
    for path, rows in needed_rows.items():
        table = pq.read_table(path, columns=["index", "action", "observation.state"])
        index_col = table.column("index").to_pylist()
        acts = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        position = {g: local for local, g in enumerate(index_col)}
        for row in rows:
            local = position[row]
            norm_action[row] = robust_normalize(acts[local][None], action_low, action_high)[0]
            norm_state[row] = robust_normalize(states[local][None], state_low, state_high)[0]

    task_to_id = {task: index for index, task in enumerate(selected_tasks)}
    video_sequences = []
    proprio_sequences = []
    previous_sequences = []
    action_sequences = []
    instruction_ids = []
    for episode, task, start, trid, scene_id in plans:
        task_id = task_to_id[task]
        windows = []
        for offset in range(args.sequence_length):
            decision = start + offset * args.control_stride
            indices = list(
                range(decision - (args.vision_window - 1) * args.vision_stride, decision + 1, args.vision_stride)
            )
            clip = preprocess_uint8(
                [frame_cache[global_row(episode, max(0, idx))] for idx in indices], args.image_size
            )
            windows.append(clip)
        video_sequences.append(torch.stack(windows))  # [T,W,3,S,S]
        proprio_sequences.append(
            torch.from_numpy(
                np.stack([norm_state[global_row(episode, start + offset * args.control_stride)] for offset in range(args.sequence_length)])
            )
        )
        previous_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        # 修复 P0-A（2026-08-05 审查）：episode 首决策（local offset 0 且 start=0）
                        # 之前取 global_row-1 = 上一条 episode 的末动作（跨 episode 泄漏），
                        # 且首决策 prev==actions[0] 自泄漏。统一用 0（归一化中点，与评估侧 last_norm 初值一致）。
                        np.zeros_like(norm_action[global_row(episode, start)])
                        if start + offset * args.control_stride == 0
                        else norm_action[global_row(episode, start + offset * args.control_stride) - 1]
                        for offset in range(args.sequence_length)
                    ]
                )
            )
        )
        action_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        np.stack(
                            [
                                norm_action[global_row(episode, start + offset * args.control_stride) + step]
                                for step in range(args.action_horizon)
                            ]
                        )
                        for offset in range(args.sequence_length)
                    ]
                )
            )
        )
        instruction_ids.append(task_id)

    payload = {
        "video_frames": torch.stack(video_sequences),
        "instructions": [task for _episode, task, _start, _trid, _scene in plans],
        "proprio": torch.stack(proprio_sequences),
        "previous_action": torch.stack(previous_sequences),
        "actions": torch.stack(action_sequences),
        "pair_id": torch.arange(len(plans), dtype=torch.long),
        "instruction_id": torch.tensor(instruction_ids, dtype=torch.long),
        # 数据合同元数据（2026-08-05 Codex v2 要求）：轨迹级溯源
        "trajectory_id": torch.tensor([p[3] for p in plans], dtype=torch.long),
        "start_frame": torch.tensor([p[2] for p in plans], dtype=torch.long),
        "scene_id": torch.tensor([p[4] for p in plans], dtype=torch.long),
        "episode_id": torch.arange(len(plans), dtype=torch.long),
        "normalization": {
            "action_q01": torch.from_numpy(action_low.astype(np.float32)),
            "action_q99": torch.from_numpy(action_high.astype(np.float32)),
            "state_q01": torch.from_numpy(state_low.astype(np.float32)),
            "state_q99": torch.from_numpy(state_high.astype(np.float32)),
        },
        "metadata": {
            "contract": "same_scene_multi_task_video",
            "tasks": selected_tasks,
            "scenes": list(args.scenes),
            "image_size": args.image_size,
            "pixel_dtype": "uint8",
            "control_stride": args.control_stride,
            "action_horizon": args.action_horizon,
            "vision_window": args.vision_window,
            "vision_stride": args.vision_stride,
            "sequences_per_episode": args.sequences_per_episode,
            "previous_action_boundary": "zero_at_episode_start",  # P0-A 修复（2026-08-05）
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gib = args.output.stat().st_size / (1024**3)
    print(f"saved={args.output.resolve()} size={size_gib:.2f}GiB shape={payload['video_frames'].shape}")


if __name__ == "__main__":
    main()
