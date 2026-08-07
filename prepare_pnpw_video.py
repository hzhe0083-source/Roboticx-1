"""Prepare PNPW for end-to-end fine-tuning: raw video frames + instruction text.

Unlike prepare_pnpw_features.py (frozen precomputed features), this script
stores the preprocessed video clips as uint8 pixels together with the raw
instruction strings, so training can run V-JEPA 2.1 and Qwen3.5 online with
gradients.  State/action tensors and normalization are identical to the
feature dataset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.nn import functional as F

from prepare_pnpw_features import (
    DEFAULT_DATASET,
    SamplePlan,
    VideoRequest,
    _fixed_list_to_numpy,
    _load_tables,
    build_sample_plans,
    clip_frame_indices,
)


@dataclass(frozen=True)
class _ClipRequest:
    key: tuple[int, int]
    endpoint: int
    frame_indices: tuple[int, ...]


def _preprocess_clips_uint8(
    clips: list[list[np.ndarray]], image_size: int
) -> Tensor:
    """Resize + center-crop exactly like the feature pipeline, keep uint8."""
    video = torch.from_numpy(np.stack(clips)).permute(0, 1, 4, 2, 3)
    batch, frames, channels, height, width = video.shape
    flat = video.reshape(batch * frames, channels, height, width)
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
    return flat.round().clamp(0, 255).to(torch.uint8).reshape(
        batch, frames, channels, image_size, image_size
    )


def extract_video_clips(
    root: Path,
    camera: str,
    episodes_by_id: dict[int, dict],
    decision_keys: list[tuple[int, int]],
    *,
    fps: int,
    vision_window: int,
    vision_stride: int,
    image_size: int,
) -> dict[tuple[int, int], Tensor]:
    """Decode each decision point's causal vision window as uint8 frames."""
    grouped: dict[tuple[int, int], list[_ClipRequest]] = defaultdict(list)
    prefix = f"videos/observation.images.{camera}"
    for episode_index, local_frame in decision_keys:
        episode = episodes_by_id[episode_index]
        chunk_index = int(episode[f"{prefix}/chunk_index"])
        file_index = int(episode[f"{prefix}/file_index"])
        start_timestamp = float(episode[f"{prefix}/from_timestamp"])
        start_frame = round(start_timestamp * fps)
        indices = clip_frame_indices(
            local_frame,
            video_start_frame=start_frame,
            window=vision_window,
            stride=vision_stride,
        )
        grouped[(chunk_index, file_index)].append(
            _ClipRequest(
                key=(episode_index, local_frame),
                endpoint=start_frame + local_frame,
                frame_indices=indices,
            )
        )

    output: dict[tuple[int, int], Tensor] = {}
    completed = 0
    max_history = (vision_window - 1) * vision_stride
    for (chunk_index, file_index), requests in sorted(grouped.items()):
        path = root / (
            f"videos/observation.images.{camera}/chunk-{chunk_index:03d}/"
            f"file-{file_index:03d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        by_endpoint: dict[int, list[_ClipRequest]] = defaultdict(list)
        for request in requests:
            by_endpoint[request.endpoint].append(request)
        last_endpoint = max(by_endpoint)
        frame_cache: dict[int, np.ndarray] = {}

        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                frame_cache[frame_index] = frame.to_ndarray(format="rgb24")
                oldest = frame_index - max_history
                for stale in [index for index in frame_cache if index < oldest]:
                    del frame_cache[stale]

                for request in by_endpoint.pop(frame_index, []):
                    try:
                        clip = [frame_cache[index] for index in request.frame_indices]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"missing causal clip frame {exc.args[0]} in {path}"
                        ) from exc
                    clip_tensor = _preprocess_clips_uint8([clip], image_size)[0]
                    output[request.key] = clip_tensor.contiguous()
                    completed += 1
                    if completed % 200 == 0:
                        print(f"camera={camera} clips={completed}/{len(decision_keys)}")
                if frame_index >= last_endpoint:
                    break

        if by_endpoint:
            missing = min(by_endpoint)
            raise RuntimeError(f"video {path} ended before requested frame {missing}")
    if len(output) != len(decision_keys):
        raise RuntimeError(
            f"camera {camera} produced {len(output)} clips for {len(decision_keys)} decisions"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PNPW raw clips for e2e training")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("data/pnpw_video.pt"))
    parser.add_argument("--cameras", nargs="+", default=("environment_1",))
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--control-stride", type=int, default=3)
    parser.add_argument("--sequence-stride", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--vision-window", type=int, default=4)
    parser.add_argument("--vision-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.vision_window < 2 or args.vision_window % 2:
        raise ValueError("vision_window must be an even number >=2")

    info, episodes, data = _load_tables(root)
    sequence_stride = args.sequence_stride or args.sequence_length * args.control_stride
    plans = build_sample_plans(
        episodes,
        sequence_length=args.sequence_length,
        control_stride=args.control_stride,
        sequence_stride=sequence_stride,
        action_horizon=args.action_horizon,
        action_stride=args.action_stride,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
    )
    if not plans:
        raise ValueError("no valid training sequences were produced")

    available_cameras = {
        key.removeprefix("observation.images.")
        for key, value in info["features"].items()
        if value.get("dtype") == "video"
    }
    unknown_cameras = set(args.cameras) - available_cameras
    if unknown_cameras:
        raise ValueError(f"unknown cameras: {sorted(unknown_cameras)}")

    tasks = list(dict.fromkeys(plan.task for plan in plans))
    decision_keys = list(
        dict.fromkeys(
            (plan.episode_index, frame)
            for plan in plans
            for frame in plan.decision_frames
        )
    )
    print(
        f"episodes={len(set(plan.episode_index for plan in plans))} samples={len(plans)} "
        f"decisions={len(decision_keys)} tasks={tasks} cameras={list(args.cameras)}"
    )
    if args.dry_run:
        return

    episodes_by_id = {int(episode["episode_index"]): episode for episode in episodes}
    clip_pairs = []
    for camera in args.cameras:
        clip_pairs.append(
            extract_video_clips(
                root,
                camera,
                episodes_by_id,
                decision_keys,
                fps=int(info["fps"]),
                vision_window=args.vision_window,
                vision_stride=args.vision_stride,
                image_size=args.image_size,
            )
        )

    raw_actions = _fixed_list_to_numpy(
        pq.read_table(root / "data/chunk-000/file-000.parquet", columns=["action"]),
        "action",
    )
    raw_states = _fixed_list_to_numpy(
        pq.read_table(root / "data/chunk-000/file-000.parquet", columns=["observation.state"]),
        "observation.state",
    )
    action_low, action_high = np.quantile(raw_actions, (0.01, 0.99), axis=0)
    state_low, state_high = np.quantile(raw_states, (0.01, 0.99), axis=0)
    scale = np.where(np.abs(action_high - action_low) < 1e-6, 1.0, action_high - action_low)
    normalized_actions = np.clip(2.0 * (raw_actions - action_low) / scale - 1.0, -1.0, 1.0).astype(np.float32)
    scale_s = np.where(np.abs(state_high - state_low) < 1e-6, 1.0, state_high - state_low)
    normalized_states = np.clip(2.0 * (raw_states - state_low) / scale_s - 1.0, -1.0, 1.0).astype(np.float32)

    task_to_id = {task: index for index, task in enumerate(tasks)}
    video_sequences = []
    proprio_sequences = []
    previous_sequences = []
    action_sequences = []
    instruction_texts = []
    instruction_ids = []
    episode_ids = []
    for plan in plans:
        episode = episodes_by_id[plan.episode_index]
        global_start = int(episode["dataset_from_index"])
        task_id = task_to_id[plan.task]
        frames = torch.stack(
            [
                torch.cat(
                    [features[(plan.episode_index, frame)] for features in clip_pairs],
                    dim=0,
                )
                for frame in plan.decision_frames
            ]
        )
        video_sequences.append(frames)
        proprio_sequences.append(
            torch.from_numpy(
                np.stack([normalized_states[global_start + frame] for frame in plan.decision_frames])
            )
        )
        # previous_action is the action of the previous global frame; the very
        # first dataset frame stands in for itself (single sample).
        previous_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        normalized_actions[max(0, global_start + frame - 1)]
                        for frame in plan.decision_frames
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
                                normalized_actions[global_start + frame + offset * args.action_stride]
                                for offset in range(args.action_horizon)
                            ]
                        )
                        for frame in plan.decision_frames
                    ]
                )
            )
        )
        instruction_texts.append(plan.task)
        instruction_ids.append(task_id)
        episode_ids.append(plan.episode_index)

    payload = {
        "video_frames": torch.stack(video_sequences),
        "instructions": instruction_texts,
        "proprio": torch.stack(proprio_sequences),
        "previous_action": torch.stack(previous_sequences),
        "actions": torch.stack(action_sequences),
        "pair_id": torch.arange(len(plans), dtype=torch.long),
        "instruction_id": torch.tensor(instruction_ids, dtype=torch.long),
        "episode_id": torch.tensor(episode_ids, dtype=torch.long),
        "normalization": {
            "action_q01": torch.from_numpy(action_low.astype(np.float32)),
            "action_q99": torch.from_numpy(action_high.astype(np.float32)),
            "state_q01": torch.from_numpy(state_low.astype(np.float32)),
            "state_q99": torch.from_numpy(state_high.astype(np.float32)),
        },
        "metadata": {
            "source": str(root),
            "contract": "single_task" if len(tasks) == 1 else "unpaired_multi_task",
            "tasks": tasks,
            "cameras": list(args.cameras),
            "fps": int(info["fps"]),
            "sequence_length": args.sequence_length,
            "control_stride": args.control_stride,
            "sequence_stride": sequence_stride,
            "action_horizon": args.action_horizon,
            "action_stride": args.action_stride,
            "vision_window": args.vision_window,
            "vision_stride": args.vision_stride,
            "image_size": args.image_size,
            "pixel_dtype": "uint8",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gib = args.output.stat().st_size / (1024**3)
    print(
        f"saved={args.output.resolve()} size={size_gib:.3f}GiB "
        f"shape={payload['video_frames'].shape}"
    )


if __name__ == "__main__":
    main()
