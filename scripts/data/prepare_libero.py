"""Prepare a LIBERO-100 subset (same-scene, different-instruction tasks) as
frozen features for the VA pipeline.

LIBERO stores frames as PNG bytes inside the data parquet files (LeRobot v3),
so this script decodes them instead of reading mp4 videos.  The selected
subset keeps one scene prefix (e.g. LIVING_ROOM_SCENE2) with several tasks,
so the vision context is near-identical across tasks and only the language
instruction can explain the action difference -- the ideal probe for the
language stream.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import gc
import glob
import io
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from prepare_pnpw_features import (
    QwenTextBackbone,
    VJEPA21Backbone,
    robust_normalize,
    _fixed_list_to_numpy,
    clip_frame_indices,
)


def decode_bytes(row: dict) -> np.ndarray:
    raw = row.get("bytes")
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"missing image bytes at path={row.get('path')}")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image)


def preprocess_frames(frames: list[np.ndarray], image_size: int) -> Tensor:
    """uint8 frames [H,W,C] -> ImageNet-normalized [1,W,3,S,S] (uint8 in)."""
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    batch, channels, height, width = video.shape
    flat = video
    if height < width:
        resized_height = image_size
        resized_width = round(width * image_size / height)
    else:
        resized_width = image_size
        resized_height = round(height * image_size / width)
    flat = F.interpolate(
        flat,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - image_size) // 2
    left = (resized_width - image_size) // 2
    flat = flat[:, :, top : top + image_size, left : left + image_size]
    return flat.reshape(1, batch, channels, image_size, image_size).float().div_(255.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LIBERO-100 subset features")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/media/ryan/robot-data/datasets/benchmark_data/raw/libero100/kevin_libero100_lerobot"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/libero_features.pt"))
    parser.add_argument("--scene", type=str, default="LIVING_ROOM_SCENE2")
    parser.add_argument("--max-tasks", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=15)
    parser.add_argument("--sequences-per-episode", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--control-stride", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--vision-window", type=int, default=4)
    parser.add_argument("--vision-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    info = json.load(open(root / "meta/info.json"))
    fps = int(info["fps"])
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()

    # Group episodes by full task text; keep tasks sharing the scene prefix.
    by_task: dict[str, list[dict]] = {}
    for episode in episodes:
        raw_task = episode.get("tasks") or episode.get("task") or ""
        if isinstance(raw_task, list):
            raw_task = raw_task[0] if raw_task else ""
        task = str(raw_task).strip()
        by_task.setdefault(task, []).append(episode)
    if args.scene.upper() in ("ALL", "*"):
        scene_tasks = sorted(by_task)
        print(f"scene=ALL tasks={len(scene_tasks)}")
    else:
        scene_tasks = [task for task in by_task if task.startswith(args.scene)]
        scene_tasks.sort()
        if not scene_tasks:
            raise ValueError(f"no tasks with scene prefix {args.scene}")
    selected_tasks = scene_tasks[: args.max_tasks]
    print(f"scene={args.scene} tasks={len(selected_tasks)}")
    for task in selected_tasks:
        print(f"  episodes={len(by_task[task])}: {task[:70]}")

    # Sample plans: sequences at evenly spaced starts inside each episode.
    plans = []  # (episode, task, start)
    for task in selected_tasks:
        chosen = by_task[task][: args.episodes_per_task]
        for episode in chosen:
            length = int(episode["length"])
            required_span = (args.sequence_length - 1) * args.control_stride + (
                args.action_horizon - 1
            )
            last_start = length - 1 - required_span
            if last_start < 0:
                continue
            stride = max(1, last_start // max(args.sequences_per_episode, 1))
            starts = list(range(0, last_start + 1, stride))[: args.sequences_per_episode]
            for start in starts:
                plans.append((episode, task, start))
    if not plans:
        raise ValueError("no plans produced")
    print(f"samples={len(plans)}")

    # Map (episode) -> data row range and file sets.
    data_files = sorted(glob.glob(str(root / "data/chunk-000/*.parquet")))
    file_meta = []
    for path in data_files:
        table = pq.read_table(path, columns=["frame_index", "episode_index", "index"])
        meta = table.to_pylist()
        file_meta.append((path, meta, table.num_rows))

    def global_row(episode: dict, local_frame: int) -> int:
        return int(episode["dataset_from_index"]) + local_frame

    # Read required rows per file once, keyed by global row index, then
    # rebuild the cache using file-local positions (global rows are absolute).
    needed_rows: dict[str, set[int]] = {}
    for episode, _task, start in plans:
        for offset in range(args.sequence_length):
            decision = start + offset * args.control_stride
            indices = clip_frame_indices(
                decision,
                video_start_frame=0,
                window=args.vision_window,
                stride=args.vision_stride,
            )
            for frame in indices:
                row = global_row(episode, frame)
                for path, meta, _n in file_meta:
                    if meta[0]["index"] <= row <= meta[-1]["index"]:
                        needed_rows.setdefault(path, set()).add(row)
                        break
                else:
                    raise RuntimeError(f"row {row} not found in any data file")
            # Action chunk rows and the previous-action row of each decision
            # point must also be available for normalization lookup.
            for step in range(args.action_horizon):
                row = global_row(episode, decision + step)
                for path, meta, _n in file_meta:
                    if meta[0]["index"] <= row <= meta[-1]["index"]:
                        needed_rows.setdefault(path, set()).add(row)
                        break
            previous_row = max(0, global_row(episode, decision - 1))
            for path, meta, _n in file_meta:
                if meta[0]["index"] <= previous_row <= meta[-1]["index"]:
                    needed_rows.setdefault(path, set()).add(previous_row)
                    break

    frame_cache: dict[int, np.ndarray] = {}
    for path, rows in needed_rows.items():
        table = pq.read_table(path, columns=["index", "observation.images.image"])
        index_col = table.column("index").to_pylist()
        arr = table.column("observation.images.image").combine_chunks().to_pylist()
        position = {global_idx: local for local, global_idx in enumerate(index_col)}
        for row in rows:
            frame_cache[row] = decode_bytes(arr[position[row]])
    print(f"frames decoded: {len(frame_cache)}")

    if args.dry_run:
        return

    # Language: one encode per selected task.
    text_backbone = QwenTextBackbone.from_pretrained(
        device=args.device, dtype=args.model_dtype, local_files_only=True
    )
    language_hidden, language_mask = text_backbone.encode(selected_tasks)
    language_hidden = language_hidden.to(device="cpu", dtype=torch.float16)
    language_mask = language_mask.cpu()
    del text_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Vision features.
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=args.device,
        dtype=args.model_dtype,
        max_tokens=64,
        local_files_only=True,
    )
    flat_features: dict[int, Tensor] = {}
    batch_keys: list[int] = []
    batch_clips: list[list[np.ndarray]] = []

    def encode_pending() -> None:
        if not batch_keys:
            return
        inputs = preprocess_frames(batch_clips[0], args.image_size)
        for clip in batch_clips[1:]:
            inputs = torch.cat(
                (inputs, preprocess_frames(clip, args.image_size)), dim=0
            )
        with torch.inference_mode():
            flat, _ = vision_backbone.forward_variants(inputs)
        flat = flat.to(device="cpu", dtype=torch.float16)
        for key, token in zip(batch_keys, flat, strict=True):
            flat_features[key] = token.contiguous()
        batch_keys.clear()
        batch_clips.clear()

    seen_keys = set()
    for episode, _task, start in plans:
        for offset in range(args.sequence_length):
            key = global_row(episode, start + offset * args.control_stride)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            indices = clip_frame_indices(
                start + offset * args.control_stride,
                video_start_frame=0,
                window=args.vision_window,
                stride=args.vision_stride,
            )
            batch_keys.append(key)
            batch_clips.append([frame_cache[global_row(episode, idx)] for idx in indices])
            if len(batch_keys) == args.batch_size:
                encode_pending()
    encode_pending()
    del vision_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"vision features: {len(flat_features)}")

    # Actions/states: quantiles over all involved rows for the normalization
    # base, then per-row normalized values keyed by global row index.
    stat_actions = []
    stat_states = []
    for path, _rows in needed_rows.items():
        table = pq.read_table(path, columns=["action", "observation.state"])
        stat_actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        stat_states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
    raw_all_actions = np.concatenate(stat_actions, axis=0)
    raw_all_states = np.concatenate(stat_states, axis=0)
    action_low, action_high = np.quantile(raw_all_actions, (0.01, 0.99), axis=0)
    state_low, state_high = np.quantile(raw_all_states, (0.01, 0.99), axis=0)

    norm_action_by_row: dict[int, np.ndarray] = {}
    norm_state_by_row: dict[int, np.ndarray] = {}
    for path, rows in needed_rows.items():
        table = pq.read_table(path, columns=["index", "action", "observation.state"])
        index_col = table.column("index").to_pylist()
        acts = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        position = {global_idx: local for local, global_idx in enumerate(index_col)}
        for row in rows:
            local = position[row]
            norm_action_by_row[row] = robust_normalize(
                acts[local][None], action_low, action_high
            )[0]
            norm_state_by_row[row] = robust_normalize(
                states[local][None], state_low, state_high
            )[0]

    task_to_id = {task: index for index, task in enumerate(selected_tasks)}
    vision_sequences = []
    proprio_sequences = []
    previous_sequences = []
    action_sequences = []
    language_sequences = []
    mask_sequences = []
    instruction_ids = []
    for episode, task, start in plans:
        task_id = task_to_id[task]
        frames = torch.stack(
            [flat_features[global_row(episode, start + offset * args.control_stride)]
             for offset in range(args.sequence_length)]
        )
        vision_sequences.append(frames)
        proprio_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        norm_state_by_row[global_row(episode, start + offset * args.control_stride)]
                        for offset in range(args.sequence_length)
                    ]
                )
            )
        )
        previous_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        # 修复 P0-A（2026-08-05 审查）：episode 首决策 prev 用 0 代替跨 episode 泄漏
                        np.zeros_like(norm_action_by_row[global_row(episode, start)])
                        if start + offset * args.control_stride == 0
                        else norm_action_by_row[global_row(episode, start + offset * args.control_stride) - 1]
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
                                norm_action_by_row[global_row(episode, start + offset * args.control_stride) + step]
                                for step in range(args.action_horizon)
                            ]
                        )
                        for offset in range(args.sequence_length)
                    ]
                )
            )
        )
        language_sequences.append(language_hidden[task_id])
        mask_sequences.append(language_mask[task_id])
        instruction_ids.append(task_id)

    payload = {
        "vision_tokens": torch.stack(vision_sequences),
        "language_hidden": torch.stack(language_sequences),
        "language_mask": torch.stack(mask_sequences),
        "proprio": torch.stack(proprio_sequences),
        "previous_action": torch.stack(previous_sequences),
        "actions": torch.stack(action_sequences),
        "pair_id": torch.arange(len(plans), dtype=torch.long),
        "instruction_id": torch.tensor(instruction_ids, dtype=torch.long),
        "episode_id": torch.tensor([i for i in range(len(plans))], dtype=torch.long),
        "normalization": {
            "action_q01": torch.from_numpy(action_low.astype(np.float32)),
            "action_q99": torch.from_numpy(action_high.astype(np.float32)),
            "state_q01": torch.from_numpy(state_low.astype(np.float32)),
            "state_q99": torch.from_numpy(state_high.astype(np.float32)),
        },
        "metadata": {
            "contract": "same_scene_multi_task",
            "scene": args.scene,
            "tasks": selected_tasks,
            "fps": fps,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_mb = args.output.stat().st_size / (1024**2)
    print(f"saved={args.output.resolve()} size={size_mb:.1f}MiB shape={payload['vision_tokens'].shape}")


if __name__ == "__main__":
    main()
