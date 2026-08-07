"""Merge the local multi-task datasets into one paired-ready feature set.

Three tasks are available locally (all 12-dim actions, 30 FPS, same camera
rig): white-cube pick, unguent-to-box, toothbrush-into-cup.  Features are
extracted with the same frozen V-JEPA/Qwen pipeline as prepare_pnpw_features
but normalized jointly across tasks, so the language stream is no longer a
constant and language-conditioned action selection becomes learnable.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from prepare_pnpw_features import (
    DEFAULT_DATASET,
    build_sample_plans,
    extract_camera_features,
    robust_normalize,
    _fixed_list_to_numpy,
    _load_tables,
)


@dataclass(frozen=True)
class TaskSource:
    name: str
    root: Path


def discover_task_sources(datasets_dir: Path) -> list[TaskSource]:
    """All dataset directories under the local root, grouped by task text.

    Task text is read from the episodes table (same source as the feature
    pipeline); tasks.parquet may carry a different, longer phrasing.
    """
    sources = []
    for directory in sorted(datasets_dir.iterdir()):
        if not directory.is_dir():
            continue
        episodes_path = directory / "meta/episodes/chunk-000/file-000.parquet"
        if not episodes_path.is_file():
            continue
        episodes = pq.read_table(episodes_path).to_pylist()
        if not episodes:
            continue
        raw_task = episodes[0].get("tasks") or episodes[0].get("task") or ""
        if isinstance(raw_task, list):
            raw_task = raw_task[0] if raw_task else ""
        task = str(raw_task).strip()
        if not task:
            continue
        sources.append(TaskSource(name=task, root=directory))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare joint multi-task feature dataset")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASET.parent)
    parser.add_argument("--output", type=Path, default=Path("data/multi_task_features.pt"))
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--control-stride", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--vision-window", type=int, default=4)
    parser.add_argument("--vision-stride", type=int, default=2)
    parser.add_argument("--vision-token-budget", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-episodes-per-task", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--feature-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = discover_task_sources(args.datasets_dir)
    if not sources:
        raise ValueError(f"no EvoStudio datasets found under {args.datasets_dir}")
    tasks = list(dict.fromkeys(source.name for source in sources))
    print(f"tasks={tasks} sources={[s.root.name for s in sources]}")
    if args.dry_run:
        return

    from prepare_pnpw_features import QwenTextBackbone, VJEPA21Backbone

    # ---- global sample plans with unique (episode_index, frame) keys ----
    plans = []
    decision_keys = []
    task_of_sample = []
    episode_offset = 0
    episode_global_id = []
    per_source_episode_offsets = []
    fps = None
    for source in sources:
        root = source.root.resolve()
        info, episodes, _ = _load_tables(root)
        if fps is None:
            fps = int(info["fps"])
        elif int(info["fps"]) != fps:
            raise ValueError(f"fps mismatch in {root}: {info['fps']} vs {fps}")
        source_plans = build_sample_plans(
            episodes,
            sequence_length=args.sequence_length,
            control_stride=args.control_stride,
            sequence_stride=args.sequence_length * args.control_stride,
            action_horizon=args.action_horizon,
            action_stride=args.action_stride,
            max_episodes=args.max_episodes_per_task,
        )
        offset = episode_offset
        per_source_episode_offsets.append(offset)
        for plan in source_plans:
            shifted = type(plan)(
                episode_index=plan.episode_index + offset,
                task=source.name,
                decision_frames=plan.decision_frames,
            )
            plans.append(shifted)
            decision_keys.extend(
                (shifted.episode_index, frame) for frame in shifted.decision_frames
            )
            task_of_sample.append(tasks.index(source.name))
            episode_global_id.append(offset + plan.episode_index)
        episode_offset += len(episodes)
    decision_keys = list(dict.fromkeys(decision_keys))
    if not plans:
        raise ValueError("no valid training sequences were produced")
    task_ids = torch.tensor(task_of_sample, dtype=torch.long)
    print(f"samples={len(plans)} decisions={len(decision_keys)} tasks_ids={torch.unique(task_ids, return_counts=True)}")

    # ---- language encoding (one pass per task) ----
    text_backbone = QwenTextBackbone.from_pretrained(
        device=args.device,
        dtype=args.model_dtype,
        local_files_only=True,
    )
    language_hidden, language_mask = text_backbone.encode(tasks)
    language_hidden = language_hidden.to(device="cpu", dtype=torch.float16)
    language_mask = language_mask.cpu()
    del text_backbone
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- vision features (one pass per source, unique keys) ----
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=args.device,
        dtype=args.model_dtype,
        max_tokens=args.vision_token_budget,
        local_files_only=True,
    )
    flat_all: dict[tuple[int, int], torch.Tensor] = {}
    spatial_all: dict[tuple[int, int], torch.Tensor] = {}
    for source, offset in zip(sources, per_source_episode_offsets, strict=True):
        root = source.root.resolve()
        info, episodes, _ = _load_tables(root)
        episodes_by_id = {int(episode["episode_index"]) + offset: episode for episode in episodes}
        source_keys = [
            key for key in decision_keys
            if offset <= key[0] < offset + len(episodes)
        ]
        flat, spatial = extract_camera_features(
            root,
            "environment_1",
            episodes_by_id,
            source_keys,
            vision_backbone,
            fps=int(info["fps"]),
            vision_window=args.vision_window,
            vision_stride=args.vision_stride,
            image_size=args.image_size,
            batch_size=args.batch_size,
            feature_dtype=torch.float16,
        )
        flat_all.update(flat)
        spatial_all.update(spatial)
    vision_grid = vision_backbone.patch_grid()
    del vision_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(flat_all) != len(decision_keys):
        raise ValueError(f"missing flat features: {len(flat_all)}/{len(decision_keys)}")

    # ---- joint normalization across tasks ----
    raw_actions = []
    raw_states = []
    for source in sources:
        root = source.root.resolve()
        _, _, data = _load_tables(root)
        raw_actions.append(_fixed_list_to_numpy(data, "action"))
        raw_states.append(_fixed_list_to_numpy(data, "observation.state"))
    raw_actions = np.concatenate(raw_actions, axis=0)
    raw_states = np.concatenate(raw_states, axis=0)
    action_low, action_high = np.quantile(raw_actions, (0.01, 0.99), axis=0)
    state_low, state_high = np.quantile(raw_states, (0.01, 0.99), axis=0)
    normalized_actions = robust_normalize(raw_actions, action_low, action_high)
    normalized_states = robust_normalize(raw_states, state_low, state_high)

    # ---- per-sample assembly: map global episode key -> data table row ----
    global_start_by_episode = {}
    episode_cursor = 0
    source_start = 0
    for source in sources:
        root = source.root.resolve()
        info, episodes, _ = _load_tables(root)
        for index, episode in enumerate(episodes):
            global_start_by_episode[source_start + index] = episode_cursor
            episode_cursor += int(episode["length"])
        source_start += len(episodes)

    vision_sequences = []
    vision_spatial_sequences = []
    proprio_sequences = []
    previous_sequences = []
    action_sequences = []
    language_sequences = []
    mask_sequences = []
    instruction_ids = []
    for plan, task_id in zip(plans, task_ids.tolist(), strict=True):
        global_start = global_start_by_episode[plan.episode_index]
        flat_vision = torch.stack(
            [
                flat_all[(plan.episode_index, frame)]
                for frame in plan.decision_frames
            ]
        )
        spatial_vision = torch.stack(
            [
                spatial_all[(plan.episode_index, frame)]
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
        "episode_id": torch.tensor(episode_global_id, dtype=torch.long),
        "normalization": {
            "action_q01": torch.from_numpy(action_low.astype(np.float32)),
            "action_q99": torch.from_numpy(action_high.astype(np.float32)),
            "state_q01": torch.from_numpy(state_low.astype(np.float32)),
            "state_q99": torch.from_numpy(state_high.astype(np.float32)),
        },
        "metadata": {
            "contract": "multi_task",
            "tasks": tasks,
            "sources": [s.root.name for s in sources],
            "fps": fps,
            "sequence_length": args.sequence_length,
            "control_stride": args.control_stride,
            "action_horizon": args.action_horizon,
            "action_stride": args.action_stride,
            "vision_window": args.vision_window,
            "vision_stride": args.vision_stride,
            "vision_grid": list(vision_grid),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gib = args.output.stat().st_size / (1024**3)
    print(f"saved={args.output.resolve()} size={size_gib:.3f}GiB shape={payload['vision_tokens'].shape}")


if __name__ == "__main__":
    main()
