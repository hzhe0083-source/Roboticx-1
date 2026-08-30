#!/usr/bin/env python3
"""Convert the official Evo-1 MT50 LeRobot dataset into ORA raw episodes.

The input is a local snapshot of ``MINT-SJTU/Evo1_MetaWorld_Dataset``.  The
output directory contains 50 ``metaworld_longtraj_<env>.pt`` files, a raw
manifest, and an online-episode index.  Frames stay as unmodified corner2 RGB
views (JPEG storage only); DINO features are deliberately not precomputed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_longtraj_features import ENV_TO_TASK, _save_new
from scripts.build_online_episode_index import file_identity
from va_compound.longtraj_frames import ONLINE_EPISODE_CONTRACT


SOURCE_REPO = "MINT-SJTU/Evo1_MetaWorld_Dataset"
RAW_CONTRACT = "evo1_mt50_success_longtraj_v1"
MANIFEST_CONTRACT = "evo1_mt50_raw_sources_v1"
SHORT_EPISODE_PADDING = "repeat_last_mask_actions_v1"
OFFICIAL_DESCRIPTION_ALIASES = {
    "Push the puck back to a goal": ENV_TO_TASK["push-back-v3"],
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _video_to_jpegs(
    path: Path,
    *,
    expected_frames: int,
    expected_hw: tuple[int, int],
    jpeg_quality: int,
) -> list[bytes]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised in the real runtime
        raise RuntimeError("PyAV is required to decode the official AV1 videos") from exc

    frames: list[bytes] = []
    with av.open(str(path)) as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(f"{path}: expected one video stream, got {len(streams)}")
        for frame in container.decode(streams[0]):
            image = frame.to_image().convert("RGB")
            if image.size != (expected_hw[1], expected_hw[0]):
                raise ValueError(
                    f"{path}: frame size {image.size} != "
                    f"{(expected_hw[1], expected_hw[0])}"
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality)
            frames.append(buffer.getvalue())
    if len(frames) != expected_frames:
        raise ValueError(
            f"{path}: decoded {len(frames)} frames, expected {expected_frames}"
        )
    return frames


def _episode_paths(root: Path, info: dict, episode_index: int) -> tuple[Path, Path]:
    chunk_size = int(info["chunks_size"])
    fields = {
        "episode_chunk": episode_index // chunk_size,
        "episode_index": episode_index,
        "video_key": "observation.images.image",
    }
    return (
        root / str(info["data_path"]).format(**fields),
        root / str(info["video_path"]).format(**fields),
    )


def _convert_episode(
    dataset_root: Path,
    info: dict,
    episode_row: dict,
    task_index: int,
    *,
    jpeg_quality: int,
    decode_video: Callable[..., list[bytes]] | None = None,
) -> dict:
    episode_index = int(episode_row["episode_index"])
    expected_length = int(episode_row["length"])
    parquet_path, video_path = _episode_paths(dataset_root, info, episode_index)
    if not parquet_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(
            f"episode {episode_index}: missing parquet/video: "
            f"{parquet_path}, {video_path}"
        )
    table = pq.read_table(
        parquet_path,
        columns=[
            "observation.state",
            "action",
            "frame_index",
            "episode_index",
            "task_index",
        ],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"], dtype=np.int64)
    episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
    task_indices = np.asarray(table["task_index"], dtype=np.int64)
    if states.shape != (expected_length, 4) or actions.shape != (expected_length, 4):
        raise ValueError(
            f"{parquet_path}: expected ({expected_length},4) state/action, "
            f"got {states.shape}/{actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{parquet_path}: state/action contains NaN or Inf")
    if not np.array_equal(frame_indices, np.arange(expected_length)):
        raise ValueError(f"{parquet_path}: frame_index is not contiguous from zero")
    if not np.all(episode_indices == episode_index) or not np.all(task_indices == task_index):
        raise ValueError(f"{parquet_path}: episode_index/task_index columns disagree")

    image_shape = info["features"]["observation.images.image"]["shape"]
    frames = (decode_video or _video_to_jpegs)(
        video_path,
        expected_frames=expected_length,
        expected_hw=(int(image_shape[0]), int(image_shape[1])),
        jpeg_quality=jpeg_quality,
    )
    valid = np.ones(expected_length, dtype=bool)
    return {
        "source_episode_index": episode_index,
        "frames": frames,
        "actions": actions,
        "states": states,
        # The release contains successful demonstrations but no per-frame success
        # bit.  Treat every recorded expert action as pre-terminal supervision.
        "first_success": expected_length - 1,
        "success_frame": expected_length - 1,
        "frame_valid": valid.copy(),
        "action_executed": valid.copy(),
        "action_valid": valid.copy(),
        "action_supervision_valid": valid.copy(),
        "settle_mask": np.zeros(expected_length, dtype=bool),
        "recovery_mask": np.zeros(expected_length, dtype=bool),
        "perturbed": False,
        "n_perturb_events": 0,
    }


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    language_reference: Path,
    *,
    jpeg_quality: int = 90,
    episodes_per_task: int | None = None,
    resume: bool = False,
    decode_video: Callable[..., list[bytes]] | None = None,
) -> tuple[Path, Path]:
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=False)
    language_reference = language_reference.expanduser().resolve(strict=True)
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")
    if episodes_per_task is not None and episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    task_rows = _read_jsonl(dataset_root / "meta/tasks.jsonl")
    episode_rows = _read_jsonl(dataset_root / "meta/episodes.jsonl")
    if int(info.get("total_tasks", -1)) != 50 or len(task_rows) != 50:
        raise ValueError("official Evo-1 dataset must contain exactly 50 tasks")
    if len(episode_rows) != int(info.get("total_episodes", -1)):
        raise ValueError("meta/episodes.jsonl differs from info.total_episodes")
    if int(info.get("fps", -1)) != 30:
        raise ValueError(f"unexpected Evo-1 dataset fps={info.get('fps')!r}")
    image_feature = (info.get("features") or {}).get("observation.images.image") or {}
    if image_feature.get("shape") != [480, 480, 3]:
        raise ValueError(f"unexpected Evo-1 image shape: {image_feature.get('shape')!r}")
    for key in ("observation.state", "action"):
        feature = (info.get("features") or {}).get(key) or {}
        if feature.get("dtype") != "float32" or feature.get("shape") != [4]:
            raise ValueError(f"unexpected Evo-1 {key} feature: {feature!r}")

    official_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    if sorted(official_by_index) != list(range(50)):
        raise ValueError("official task indices must be contiguous 0..49")
    canonical_to_env = {description: env for env, description in ENV_TO_TASK.items()}
    official_to_env: dict[str, str] = {}
    for description in official_by_index.values():
        canonical = OFFICIAL_DESCRIPTION_ALIASES.get(description, description)
        if canonical not in canonical_to_env:
            raise ValueError(f"unknown official task description: {description!r}")
        official_to_env[description] = canonical_to_env[canonical]
    if len(set(official_to_env.values())) != 50:
        raise ValueError("official descriptions do not map bijectively onto MT50")

    reference = torch.load(language_reference, map_location="cpu", weights_only=True)
    reference_tasks = list((reference.get("metadata") or {}).get("tasks") or [])
    if len(reference_tasks) != 50 or set(reference_tasks) != set(ENV_TO_TASK.values()):
        raise ValueError("language reference must contain the canonical 50 descriptions")
    if reference_tasks[-1] != ENV_TO_TASK["push-back-v3"]:
        raise ValueError("language reference must keep push-back as task id 49")
    for key in ("language_hidden", "language_mask"):
        value = reference.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != 50:
            raise ValueError(f"language reference {key} must contain one row per task")
    normalization = reference.get("normalization") or {}
    for key in ("action_q01", "action_q99", "state_q01", "state_q99"):
        if not isinstance(normalization.get(key), torch.Tensor):
            raise ValueError(f"language reference lacks normalization.{key}")

    rows_by_env: dict[str, list[tuple[int, dict]]] = {env: [] for env in ENV_TO_TASK}
    for row in episode_rows:
        descriptions = list(row.get("tasks") or [])
        if len(descriptions) != 1 or descriptions[0] not in official_to_env:
            raise ValueError(f"episode has unknown/non-singleton task: {row!r}")
        env = official_to_env[descriptions[0]]
        source_task_id = next(
            task_id for task_id, description in official_by_index.items()
            if description == descriptions[0]
        )
        rows_by_env[env].append((source_task_id, row))
    if episodes_per_task is None:
        bad = {env: len(rows) for env, rows in rows_by_env.items() if len(rows) != 50}
        if bad or int(info["total_episodes"]) != 2500:
            raise ValueError(f"full official snapshot must contain 50 episodes/task: {bad}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict] = []
    index_tasks: list[dict | None] = [None] * 50
    index_episodes: list[dict] = []
    for env in [name for name in ENV_TO_TASK if name != "push-back-v3"] + ["push-back-v3"]:
        source_rows_for_task = sorted(
            rows_by_env[env], key=lambda item: int(item[1]["episode_index"])
        )
        if episodes_per_task is not None:
            source_rows_for_task = source_rows_for_task[:episodes_per_task]
        if not source_rows_for_task:
            raise ValueError(f"no source episodes selected for {env}")
        canonical_description = ENV_TO_TASK[env]
        task_id = reference_tasks.index(canonical_description)
        official_description = str(source_rows_for_task[0][1]["tasks"][0])
        raw_path = output_dir / f"metaworld_longtraj_{env}.pt"
        if raw_path.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite existing output: {raw_path}")
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            if raw.get("task") != env or len(raw.get("episodes") or []) != len(source_rows_for_task):
                raise ValueError(f"cannot resume incompatible raw output: {raw_path}")
            episodes = raw["episodes"]
        else:
            episodes = [
                _convert_episode(
                    dataset_root,
                    info,
                    row,
                    source_task_id,
                    jpeg_quality=jpeg_quality,
                    decode_video=decode_video,
                )
                for source_task_id, row in source_rows_for_task
            ]
            _save_new(
                {
                    "task": env,
                    "episodes": episodes,
                    "normalization": dict(normalization),
                    "metadata": {
                        "contract": RAW_CONTRACT,
                        "contract_version": 1,
                        "source_repo": SOURCE_REPO,
                        "source_description": official_description,
                        "canonical_description": canonical_description,
                        "source_fps": 30,
                        "camera_name": "corner2",
                        "frame_transform": "none",
                        "raw_frame_contract": "exact_decoded_longtraj_jpeg_480_v1",
                        "jpeg_quality": jpeg_quality,
                        "success_only": True,
                        "success_semantics": "official_success_demo_terminal_assumed",
                        "state_dim": 4,
                        "action_dim": 4,
                        "dino_preencoded": False,
                    },
                },
                raw_path,
            )

        identity = file_identity(raw_path)
        source_rows.append(
            {
                "task": env,
                "description": canonical_description,
                "source_description": official_description,
                "source_path": identity["path"],
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
                "episode_count": len(episodes),
            }
        )
        task_index_rows = []
        for local_episode_index, episode in enumerate(episodes):
            length = len(episode["actions"])
            task_index_rows.append(
                {
                    "task_id": task_id,
                    "task": env,
                    "episode_index": local_episode_index,
                    "episode_id": int(episode["source_episode_index"]),
                    "length": length,
                    # The flag-gated online loader allows every observation as
                    # a start and masks action labels beyond the terminal frame.
                    "valid_start_count": length,
                    "split": "train",
                }
            )
        index_episodes.extend(task_index_rows)
        index_tasks[task_id] = {
            "task_id": task_id,
            "task": env,
            "description": canonical_description,
            "source_description": official_description,
            "source_path": identity["path"],
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
            "source_episodes": len(episodes),
            "train_episodes": len(episodes),
            "eval_episodes": 0,
        }
        print(f"[evo1-convert] {env}: episodes={len(episodes)} -> {raw_path}", flush=True)

    if any(row is None for row in index_tasks):
        raise ValueError("converted tasks do not cover language task ids 0..49")
    manifest_path = output_dir / "evo1_mt50_raw_manifest.json"
    _write_json(
        {
            "contract": MANIFEST_CONTRACT,
            "contract_version": 1,
            "source_repo": SOURCE_REPO,
            "reference": file_identity(language_reference),
            "sources": source_rows,
        },
        manifest_path,
    )
    index_path = output_dir / "evo1_mt50_online_index.json"
    _write_json(
        {
            "contract": ONLINE_EPISODE_CONTRACT,
            "contract_version": 1,
            "sampling_protocol": {
                "storage": "full_episode_only",
                "crop_start": "online_epoch_seeded_uniform_without_replacement",
                "crop_start_stride": 1,
                "sequence_length": 4,
                "decision_stride": 15,
                "decision_offsets_relative_to_random_d": [0, 15, 30, 45],
                "action_horizon": 15,
                "world_target_horizon": 15,
                "world_target_offsets_relative_to_random_d": [15, 30, 45, 60],
                "previous_action": "raw_action_at_decision_minus_1",
                "short_episode_padding": SHORT_EPISODE_PADDING,
                "offline_windows": False,
            },
            "source_dataset": {
                "repo_id": SOURCE_REPO,
                "fps": 30,
                "camera_name": "corner2",
                "success_only": True,
                "state_dim": 4,
                "action_dim": 4,
            },
            "raw_manifest": file_identity(manifest_path),
            "language_reference": file_identity(language_reference),
            "selection": {
                "rule": "all_official_success_episodes_v1",
                "short_episode_policy": SHORT_EPISODE_PADDING,
            },
            "tasks": [row for row in index_tasks if row is not None],
            "episodes": sorted(index_episodes, key=lambda row: int(row["episode_id"])),
            "counts": {
                "source_episodes": len(index_episodes),
                "train_episodes": len(index_episodes),
                "eval_episodes": 0,
            },
        },
        index_path,
    )
    return manifest_path, index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language-reference", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        help="optional per-task smoke subset; omit for the canonical 50/task build",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, index = convert_dataset(
        args.dataset_root,
        args.output_dir,
        args.language_reference,
        jpeg_quality=args.jpeg_quality,
        episodes_per_task=args.episodes_per_task,
        resume=args.resume,
    )
    print(f"[ok] raw manifest: {manifest}", flush=True)
    print(f"[ok] online index: {index}", flush=True)


if __name__ == "__main__":
    main()
