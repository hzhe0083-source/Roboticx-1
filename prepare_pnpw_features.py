from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.nn import functional as F

from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
DEFAULT_DATASET = Path("/home/ryan/.evostudio/workspace/embodied/datasets/local/PNPW")
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


@dataclass(frozen=True)
class SamplePlan:
    episode_index: int
    task: str
    decision_frames: tuple[int, ...]


@dataclass(frozen=True)
class VideoRequest:
    key: tuple[int, int]
    endpoint: int
    frame_indices: tuple[int, ...]


def build_sample_plans(
    episodes: list[dict],
    *,
    sequence_length: int,
    control_stride: int,
    sequence_stride: int,
    action_horizon: int,
    action_stride: int,
    max_episodes: int | None = None,
    max_samples: int | None = None,
) -> list[SamplePlan]:
    if min(sequence_length, control_stride, action_horizon, action_stride) < 1:
        raise ValueError("sequence/action lengths and strides must be positive")
    if sequence_stride < 1:
        raise ValueError("sequence_stride must be positive")

    plans = []
    required_span = (sequence_length - 1) * control_stride + (
        action_horizon - 1
    ) * action_stride
    selected_episodes = episodes if max_episodes is None else episodes[:max_episodes]
    for episode in selected_episodes:
        length = int(episode["length"])
        last_start = length - 1 - required_span
        if last_start < 0:
            continue
        tasks = episode.get("tasks") or []
        if not tasks:
            raise ValueError(f"episode {episode['episode_index']} has no task instruction")
        for start in range(0, last_start + 1, sequence_stride):
            plans.append(
                SamplePlan(
                    episode_index=int(episode["episode_index"]),
                    task=str(tasks[0]),
                    decision_frames=tuple(
                        start + offset * control_stride for offset in range(sequence_length)
                    ),
                )
            )
            if max_samples is not None and len(plans) >= max_samples:
                return plans
    return plans


def clip_frame_indices(
    local_frame: int,
    *,
    video_start_frame: int,
    window: int,
    stride: int,
) -> tuple[int, ...]:
    if local_frame < 0 or window < 2 or window % 2 or stride < 1:
        raise ValueError("V-JEPA clip requires an even window >=2 and a positive stride")
    return tuple(
        video_start_frame + max(0, local_frame - offset * stride)
        for offset in reversed(range(window))
    )


def robust_normalize(
    values: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    scale = high - low
    safe_scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
    normalized = 2.0 * (values - low) / safe_scale - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _fixed_list_to_numpy(table, column: str) -> np.ndarray:
    array = table[column].combine_chunks()
    width = array.type.list_size
    return array.values.to_numpy(zero_copy_only=False).reshape(-1, width).astype(np.float32)


def _preprocess_clips(clips: list[list[np.ndarray]], image_size: int) -> Tensor:
    video = torch.from_numpy(np.stack(clips)).permute(0, 1, 4, 2, 3).float().div_(255.0)
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
    flat = (flat - IMAGE_MEAN[None, :, None, None]) / IMAGE_STD[None, :, None, None]
    return flat.reshape(batch, frames, channels, image_size, image_size)


def _encode_clip_batch(
    backbone: VJEPA21Backbone,
    keys: list[tuple[int, int]],
    clips: list[list[np.ndarray]],
    output: dict[tuple[int, int], Tensor],
    output_spatial: dict[tuple[int, int], Tensor],
    *,
    image_size: int,
    feature_dtype: torch.dtype,
) -> None:
    inputs = _preprocess_clips(clips, image_size)
    with torch.inference_mode():
        flat, spatial = backbone.forward_variants(inputs)
    flat = flat.to(device="cpu", dtype=feature_dtype)
    spatial = spatial.to(device="cpu", dtype=feature_dtype)
    for key, flat_token, spatial_token in zip(keys, flat, spatial, strict=True):
        output[key] = flat_token.contiguous()
        output_spatial[key] = spatial_token.contiguous()


def extract_camera_features(
    root: Path,
    camera: str,
    episodes_by_id: dict[int, dict],
    decision_keys: list[tuple[int, int]],
    backbone: VJEPA21Backbone,
    *,
    fps: int,
    vision_window: int,
    vision_stride: int,
    image_size: int,
    batch_size: int,
    feature_dtype: torch.dtype,
) -> tuple[dict[tuple[int, int], Tensor], dict[tuple[int, int], Tensor]]:
    """Extract both the flat (A) and spatial (B) pooled features in one pass."""
    grouped: dict[tuple[int, int], list[VideoRequest]] = defaultdict(list)
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
            VideoRequest(
                key=(episode_index, local_frame),
                endpoint=start_frame + local_frame,
                frame_indices=indices,
            )
        )

    output: dict[tuple[int, int], Tensor] = {}
    output_spatial: dict[tuple[int, int], Tensor] = {}
    completed = 0
    max_history = (vision_window - 1) * vision_stride
    for (chunk_index, file_index), requests in sorted(grouped.items()):
        path = root / (
            f"videos/observation.images.{camera}/chunk-{chunk_index:03d}/"
            f"file-{file_index:03d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        by_endpoint: dict[int, list[VideoRequest]] = defaultdict(list)
        for request in requests:
            by_endpoint[request.endpoint].append(request)
        last_endpoint = max(by_endpoint)
        frame_cache: dict[int, np.ndarray] = {}
        batch_keys: list[tuple[int, int]] = []
        batch_clips: list[list[np.ndarray]] = []

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
                    batch_keys.append(request.key)
                    batch_clips.append(clip)
                    if len(batch_keys) == batch_size:
                        _encode_clip_batch(
                            backbone,
                            batch_keys,
                            batch_clips,
                            output,
                            output_spatial,
                            image_size=image_size,
                            feature_dtype=feature_dtype,
                        )
                        completed += len(batch_keys)
                        if completed % 100 < len(batch_keys):
                            print(f"camera={camera} encoded={completed}/{len(decision_keys)}")
                        batch_keys = []
                        batch_clips = []
                if frame_index >= last_endpoint:
                    break

        if by_endpoint:
            missing = min(by_endpoint)
            raise RuntimeError(f"video {path} ended before requested frame {missing}")
        if batch_keys:
            _encode_clip_batch(
                backbone,
                batch_keys,
                batch_clips,
                output,
                output_spatial,
                image_size=image_size,
                feature_dtype=feature_dtype,
            )
            completed += len(batch_keys)
            print(f"camera={camera} encoded={completed}/{len(decision_keys)}")
    if len(output) != len(decision_keys):
        raise RuntimeError(
            f"camera {camera} produced {len(output)} features for {len(decision_keys)} decisions"
        )
    if len(output_spatial) != len(decision_keys):
        raise RuntimeError(
            f"camera {camera} produced {len(output_spatial)} spatial features "
            f"for {len(decision_keys)} decisions"
        )
    return output, output_spatial


def _load_tables(root: Path) -> tuple[dict, list[dict], object]:
    with (root / "meta/info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    data = pq.read_table(
        root / "data/chunk-000/file-000.parquet",
        columns=["action", "observation.state", "episode_index", "frame_index"],
    )
    return info, episodes, data


def _feature_dtype(name: str) -> torch.dtype:
    choices = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in choices:
        raise ValueError(f"unsupported feature dtype: {name}")
    return choices[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert EvoStudio PNPW into VA features.pt")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("data/pnpw_features.pt"))
    parser.add_argument("--cameras", nargs="+", default=("environment_1",))
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--control-stride", type=int, default=3)
    parser.add_argument("--sequence-stride", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--vision-window", type=int, default=4)
    parser.add_argument("--vision-stride", type=int, default=2)
    parser.add_argument("--vision-token-budget", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.batch_size < 1 or args.image_size < 1:
        raise ValueError("batch size and image size must be positive")
    if args.vision_window < 2 or args.vision_window % 2:
        raise ValueError("vision_window must be an even number >=2")
    if args.vision_token_budget < 1:
        raise ValueError("vision_token_budget must be positive")

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
    if len(tasks) == 1:
        print("contract=single_task (this dataset cannot test language-conditioned task switching)")
    if args.dry_run:
        return

    feature_dtype = _feature_dtype(args.feature_dtype)
    text_backbone = QwenTextBackbone.from_pretrained(
        device=args.device,
        dtype=args.model_dtype,
        local_files_only=True,
    )
    language_hidden, language_mask = text_backbone.encode(tasks)
    language_hidden = language_hidden.to(device="cpu", dtype=feature_dtype)
    language_mask = language_mask.cpu()
    del text_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=args.device,
        dtype=args.model_dtype,
        max_tokens=args.vision_token_budget,
        local_files_only=True,
    )
    episodes_by_id = {int(episode["episode_index"]): episode for episode in episodes}
    camera_pairs = []
    for camera in args.cameras:
        camera_pairs.append(
            extract_camera_features(
                root,
                camera,
                episodes_by_id,
                decision_keys,
                vision_backbone,
                fps=int(info["fps"]),
                vision_window=args.vision_window,
                vision_stride=args.vision_stride,
                image_size=args.image_size,
                batch_size=args.batch_size,
                feature_dtype=feature_dtype,
            )
        )
    vision_grid = vision_backbone.patch_grid()
    del vision_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_actions = _fixed_list_to_numpy(data, "action")
    raw_states = _fixed_list_to_numpy(data, "observation.state")
    action_low, action_high = np.quantile(raw_actions, (0.01, 0.99), axis=0)
    state_low, state_high = np.quantile(raw_states, (0.01, 0.99), axis=0)
    normalized_actions = robust_normalize(raw_actions, action_low, action_high)
    normalized_states = robust_normalize(raw_states, state_low, state_high)

    task_to_id = {task: index for index, task in enumerate(tasks)}
    vision_sequences = []
    vision_spatial_sequences = []
    proprio_sequences = []
    previous_sequences = []
    action_sequences = []
    language_sequences = []
    mask_sequences = []
    instruction_ids = []
    episode_ids = []
    for plan in plans:
        episode = episodes_by_id[plan.episode_index]
        global_start = int(episode["dataset_from_index"])
        task_id = task_to_id[plan.task]
        flat_vision = torch.stack(
            [
                torch.cat(
                    [features[(plan.episode_index, frame)] for features, _ in camera_pairs],
                    dim=0,
                )
                for frame in plan.decision_frames
            ]
        )
        spatial_vision = torch.stack(
            [
                torch.cat(
                    [features[(plan.episode_index, frame)] for _, features in camera_pairs],
                    dim=0,
                )
                for frame in plan.decision_frames
            ]
        )
        vision_sequences.append(flat_vision)
        vision_spatial_sequences.append(spatial_vision)
        proprio_sequences.append(
            torch.from_numpy(
                np.stack([normalized_states[global_start + frame] for frame in plan.decision_frames])
            )
        )
        # previous_action is the action of the previous global frame.  For the
        # very first frame of the dataset (global index 0) there is none, so
        # the current action stands in -- that single sample aside, no goal
        # leak is introduced.
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
                                normalized_actions[
                                    global_start + frame + offset * args.action_stride
                                ]
                                for offset in range(args.action_horizon)
                            ]
                        )
                        for frame in plan.decision_frames
                    ]
                )
            )
        )
        language_sequences.append(language_hidden[task_id])
        mask_sequences.append(language_mask[task_id])
        instruction_ids.append(task_id)
        episode_ids.append(plan.episode_index)

    payload = {
        "vision_tokens": torch.stack(vision_sequences),
        "vision_tokens_spatial": torch.stack(vision_spatial_sequences),
        "language_hidden": torch.stack(language_sequences),
        "language_mask": torch.stack(mask_sequences),
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
            "vision_grid": list(vision_grid),
            "vision_raw_tokens_per_frame": int(np.prod(vision_grid)),
            "vision_pooling": {
                "flat": f"adaptive_avg_pool1d t*{int(np.prod(vision_grid))}->{args.vision_token_budget}",
                "spatial": f"time-mean + adaptive_avg_pool2d {vision_grid[0]}x{vision_grid[1]}->{args.vision_token_budget}",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gib = args.output.stat().st_size / (1024**3)
    print(f"saved={args.output.resolve()} size={size_gib:.3f}GiB shape={payload['vision_tokens'].shape}")
    print(
        f"variants=flat{tuple(payload['vision_tokens'].shape)}+spatial"
        f"{tuple(payload['vision_tokens_spatial'].shape)} grid={vision_grid}"
    )


if __name__ == "__main__":
    main()
