"""Merge DAgger shards and append them to an online episode index."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from scripts.build_online_episode_index import _valid_start_count, file_identity
from va_compound.longtraj_frames import ONLINE_EPISODE_CONTRACT


SHARD_NAME = re.compile(
    r"metaworld_longtraj_(?P<task>.+)_dagger_seed-?\d+_t\d+-\d+\.pt"
)


def _same_normalization(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_augmented_index(
    base_index_path: Path,
    dagger_dir: Path,
    output_path: Path,
    *,
    repeat: int = 4,
) -> dict:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    base_index_path = base_index_path.expanduser().resolve(strict=True)
    dagger_dir = dagger_dir.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    result = json.loads(base_index_path.read_text(encoding="utf-8"))
    if result.get("contract") != ONLINE_EPISODE_CONTRACT:
        raise ValueError("base index has the wrong online episode contract")
    tasks = {str(item["task"]): item for item in result.get("tasks") or []}
    reference = torch.load(
        Path(str(result["language_reference"]["path"])),
        map_location="cpu",
        weights_only=True,
    )
    normalization = reference["normalization"]

    shards_by_task: dict[str, list[Path]] = {}
    for path in sorted(dagger_dir.glob("*.pt")):
        match = SHARD_NAME.fullmatch(path.name)
        if match is not None:
            shards_by_task.setdefault(match.group("task"), []).append(path)
    unknown = sorted(set(shards_by_task) - set(tasks))
    if unknown:
        raise ValueError(f"DAgger shards contain unknown tasks: {unknown}")
    if not shards_by_task:
        raise ValueError(f"no DAgger shards found in {dagger_dir}")

    merged_dir = dagger_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    next_episode_id = max(int(row["episode_id"]) for row in result["episodes"]) + 1
    additional_sources = list(result.get("additional_sources") or [])
    appended = []
    unique_count = 0
    skipped_count = 0
    seen_seeds: set[tuple[str, int]] = set()

    for task in sorted(shards_by_task, key=lambda name: int(tasks[name]["task_id"])):
        episodes = []
        checkpoint = None
        for shard in shards_by_task[task]:
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            if payload.get("task") != task:
                raise ValueError(f"{shard}: payload task mismatch")
            if (payload.get("metadata") or {}).get("contract") != "current_policy_dagger_v1":
                raise ValueError(f"{shard}: wrong DAgger contract")
            if not _same_normalization(payload.get("normalization") or {}, normalization):
                raise ValueError(f"{shard}: normalization differs from base index")
            current_checkpoint = (payload.get("metadata") or {}).get("checkpoint")
            if checkpoint is None:
                checkpoint = current_checkpoint
            elif checkpoint != current_checkpoint:
                raise ValueError(f"{task}: DAgger shards use different checkpoints")
            for episode in payload.get("episodes") or []:
                seed_key = (task, int(episode["episode_seed"]))
                if seed_key in seen_seeds:
                    raise ValueError(f"duplicate DAgger episode seed: {seed_key}")
                seen_seeds.add(seed_key)
                valid_count = _valid_start_count(episode, f"{shard.name}:{seed_key[1]}")
                if valid_count == 0:
                    skipped_count += 1
                    continue
                episodes.append((episode, valid_count))
        if not episodes:
            continue

        merged_path = merged_dir / f"metaworld_longtraj_{task}_dagger_merged_v1.pt"
        _atomic_torch_save(
            {
                "task": task,
                "episodes": [episode for episode, _ in episodes],
                "n_episodes": len(episodes),
                "normalization": dict(normalization),
                "metadata": {
                    "contract": "current_policy_dagger_merged_v1",
                    "checkpoint": checkpoint,
                    "source_shards": [str(path.resolve()) for path in shards_by_task[task]],
                },
            },
            merged_path,
        )
        identity = file_identity(merged_path)
        additional_sources.append(identity)
        task_id = int(tasks[task]["task_id"])
        for episode_index, (episode, valid_count) in enumerate(episodes):
            unique_count += 1
            for _ in range(repeat):
                appended.append(
                    {
                        "task_id": task_id,
                        "task": task,
                        "episode_index": episode_index,
                        "episode_id": next_episode_id,
                        "length": len(episode["actions"]),
                        "valid_start_count": valid_count,
                        "split": "train",
                        "source_path": identity["path"],
                        "anchor_eligible": False,
                    }
                )
                next_episode_id += 1
        print(f"[dagger-index] {task}: unique={len(episodes)} logical={len(episodes) * repeat}")

    if not appended:
        raise ValueError("DAgger collection produced no usable recovery episodes")
    result["episodes"].extend(appended)
    result["additional_sources"] = additional_sources
    result["counts"] = {
        "source_episodes": len(result["episodes"]),
        "train_episodes": sum(row.get("split") == "train" for row in result["episodes"]),
        "eval_episodes": sum(row.get("split") == "eval" for row in result["episodes"]),
    }
    current_round = {
        "base_index": file_identity(base_index_path),
        "unique_episodes": unique_count,
        "logical_repeat": repeat,
        "logical_train_episodes": len(appended),
        "skipped_zero_start_episodes": skipped_count,
    }
    previous = result.get("dagger_augmentation") or {}
    rounds = list(previous.get("rounds") or ([previous] if previous else []))
    rounds.append(current_round)
    result["dagger_augmentation"] = {
        **current_round,
        "round_count": len(rounds),
        "rounds": rounds,
        "cumulative_unique_episodes": sum(
            int(item["unique_episodes"]) for item in rounds
        ),
        "cumulative_logical_train_episodes": sum(
            int(item["logical_train_episodes"]) for item in rounds
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        f"[dagger-index] wrote {output_path}: unique={unique_count}, "
        f"logical={len(appended)}, train={result['counts']['train_episodes']}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--dagger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_augmented_index(
        args.base_index,
        args.dagger_dir,
        args.output,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
