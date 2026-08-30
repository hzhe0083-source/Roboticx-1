"""Build the tiny episode-level index used by online random training.

This script never writes action chunks or frame windows.  It records only the
train/eval episode partition, raw-file identities, trajectory lengths, and the
number of legal arbitrary crop starts in each episode.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.build_longtraj_features import ENV_TO_TASK, resolve_episode_semantics
from va_compound.longtraj_frames import ONLINE_EPISODE_CONTRACT


HARD_TASKS = {"assembly-v3", "door-unlock-v3"}
PUSH_BACK_TASK = "push-back-v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _valid_start_count(episode: dict, source: str) -> int:
    semantics = resolve_episode_semantics(episode, source, legacy_policy="infer")
    length = len(episode["actions"])
    count = 0
    for start in range(max(0, length - 60)):
        decisions = start + np.arange(4) * 15
        targets = decisions[:, None] + np.arange(15)[None, :]
        valid = semantics["valid"][targets].copy()
        perturb_start = semantics["perturb_start"]
        if perturb_start is not None:
            valid &= ~(
                semantics["recovery"][targets]
                & (decisions[:, None] < int(perturb_start))
            )
        count += int(bool(valid.any()))
    return count


def inspect_task(
    task_id: int,
    source_index: int,
    task: str,
    source_path: str,
    expected_episodes: int,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("task") != task or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"{source}: invalid task payload for {task}")
    if len(payload["episodes"]) != expected_episodes:
        raise ValueError(
            f"{source}: {task} requires {expected_episodes} full episodes, "
            f"got {len(payload['episodes'])}"
        )
    episodes = []
    for episode_index, episode in enumerate(payload["episodes"]):
        lengths = {
            key: len(episode[key]) if episode.get(key) is not None else 0
            for key in ("frames", "actions", "states")
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{source}:episode[{episode_index}] length mismatch {lengths}")
        valid_start_count = _valid_start_count(
            episode, f"{source.name}:episode[{episode_index}]"
        )
        episodes.append(
            {
                "task_id": task_id,
                "task": task,
                "episode_index": episode_index,
                # Preserve phase1's historical file-order episode id so the
                # frozen all49 eval partition remains byte-for-byte addressable.
                "episode_id": source_index * 10000 + episode_index,
                "length": lengths["actions"],
                "valid_start_count": valid_start_count,
            }
        )
    return {"task_id": task_id, "task": task, "episodes": episodes}


def validate_index(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("contract") != ONLINE_EPISODE_CONTRACT:
        raise ValueError(f"unexpected online index contract: {payload.get('contract')!r}")
    tasks = list(payload.get("tasks") or [])
    episodes = list(payload.get("episodes") or [])
    counts = payload.get("counts") or {}
    if len(tasks) != 50 or [int(item["task_id"]) for item in tasks] != list(range(50)):
        raise ValueError("online index must contain exactly task ids 0..49")
    source_n = len(episodes)
    train_n = sum(item.get("split") == "train" for item in episodes)
    eval_n = sum(item.get("split") == "eval" for item in episodes)
    if counts != {"source_episodes": source_n, "train_episodes": train_n, "eval_episodes": eval_n}:
        raise ValueError("online index episode counts are inconsistent")
    declared_source_n = sum(int(item.get("source_episodes", -1)) for item in tasks)
    if declared_source_n != source_n or eval_n != 198 or train_n != source_n - 198:
        raise ValueError(
            "MT50 online index source/train/eval episode contract differs: "
            f"declared={declared_source_n}, actual={source_n}/{train_n}/{eval_n}"
        )
    identities = {(int(item["task_id"]), int(item["episode_index"])) for item in episodes}
    if len(identities) != source_n:
        raise ValueError("online index contains duplicate episodes")
    for task in tasks:
        source = Path(str(task["source_path"]))
        if not source.is_file() or source.stat().st_size != int(task["size_bytes"]):
            raise ValueError(f"raw full-episode source missing or changed: {source}")
    return payload


def build_index(
    raw_manifest_path: Path,
    existing_eval_path: Path,
    output_path: Path,
    *,
    workers: int = 6,
) -> dict[str, Any]:
    if output_path.exists():
        return validate_index(output_path)
    raw_manifest_path = raw_manifest_path.expanduser().resolve(strict=True)
    existing_eval_path = existing_eval_path.expanduser().resolve(strict=True)
    raw = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    sources = list(raw.get("sources") or [])
    if len(sources) != 50 or sources[-1].get("task") != PUSH_BACK_TASK:
        raise ValueError("raw manifest must contain canonical MT50 with push-back last")
    reference_identity = dict(raw.get("reference") or {})
    reference_path = Path(str(reference_identity.get("path", "")))
    if not reference_path.is_file():
        raise FileNotFoundError(f"language reference missing: {reference_path}")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    descriptions = list((reference.get("metadata") or {}).get("tasks") or [])
    if len(descriptions) != 50:
        raise ValueError("MT50 language reference must contain 50 task descriptions")

    task_rows: list[dict[str, Any] | None] = [None] * 50
    for source_index, item in enumerate(sources):
        task = str(item["task"])
        description = ENV_TO_TASK.get(task)
        if description is None or description not in descriptions:
            raise ValueError(f"raw task is absent from the language reference: {task}")
        task_id = descriptions.index(description)
        if task_rows[task_id] is not None:
            raise ValueError(f"duplicate language task id {task_id}: {task}")
        source = Path(str(item["source_path"])).expanduser().resolve(strict=True)
        # The immutable raw manifest already contains the full SHA-256.  Avoid
        # re-reading ~56 GiB here before the parallel episode scan.
        if source.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"raw source identity mismatch: {source}")
        task_rows[task_id] = {
            "task_id": task_id,
            "source_index": source_index,
            "task": task,
            "description": description,
            "source_path": str(source),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
            "expected_source_episodes": int(
                item.get("episode_count", 270 if task in HARD_TASKS else 60)
            ),
        }
    if any(item is None for item in task_rows):
        raise ValueError("raw sources do not map bijectively onto language task ids 0..49")
    resolved_tasks = [item for item in task_rows if item is not None]
    if resolved_tasks[49]["task"] != PUSH_BACK_TASK:
        raise ValueError("push-back must remain appended as language task id 49")

    old_eval = torch.load(existing_eval_path, map_location="cpu", weights_only=True)
    old_pairs = {
        (int(task_id), int(episode_id))
        for task_id, episode_id in zip(
            old_eval["instruction_id"].tolist(),
            old_eval["episode_id"].tolist(),
            strict=True,
        )
    }
    if len({episode for _, episode in old_pairs}) != 195:
        raise ValueError("existing all49 eval must contain exactly 195 episodes")
    if any(not 0 <= task_id < 49 for task_id, _ in old_pairs):
        raise ValueError("existing eval contains a non-all49 task id")
    eval_episode_ids = {episode_id for _, episode_id in old_pairs}
    eval_episode_ids.update(490000 + offset for offset in range(3))

    inspected: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(
                inspect_task,
                item["task_id"],
                item["source_index"],
                item["task"],
                item["source_path"],
                item["expected_source_episodes"],
            ): item
            for item in resolved_tasks
        }
        for future in as_completed(futures):
            result = future.result()
            inspected[int(result["task_id"])] = result
            unusable = sum(
                int(item["valid_start_count"]) == 0 for item in result["episodes"]
            )
            print(
                f"[online-index] {result['task']} "
                f"episodes={len(result['episodes'])} unusable={unusable}",
                flush=True,
            )

    unusable = [
        item
        for task_id in range(50)
        for item in inspected[task_id]["episodes"]
        if int(item["valid_start_count"]) == 0
    ]
    if unusable:
        details = ", ".join(
            f"{item['task']}[{item['episode_index']}]"
            for item in unusable[:30]
        )
        raise ValueError(
            f"{len(unusable)} full episodes have zero executable supervision and "
            f"must be replaced before the 60-episode index is valid: {details}"
        )

    episodes = []
    for task_id in range(50):
        rows = inspected[task_id]["episodes"]
        for episode in rows:
            episode["split"] = (
                "eval" if int(episode["episode_id"]) in eval_episode_ids else "train"
            )
            episodes.append(episode)
        resolved_tasks[task_id]["source_episodes"] = len(rows)
        resolved_tasks[task_id]["train_episodes"] = sum(
            item["split"] == "train" for item in rows
        )
        resolved_tasks[task_id]["eval_episodes"] = sum(
            item["split"] == "eval" for item in rows
        )

    payload = {
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
            "offline_windows": False,
        },
        "raw_manifest": file_identity(raw_manifest_path),
        "language_reference": file_identity(reference_path),
        "selection": {
            "existing_all49_eval": file_identity(existing_eval_path),
            "existing_all49_eval_episodes": 195,
            "push_back_eval_episode_ids": [490000, 490001, 490002],
            "rule": "frozen_all49_eval_plus_first3_push_back_v1",
        },
        "tasks": resolved_tasks,
        "episodes": episodes,
        "counts": {
            "source_episodes": len(episodes),
            "train_episodes": sum(item["split"] == "train" for item in episodes),
            "eval_episodes": sum(item["split"] == "eval" for item in episodes),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale online-index temporary: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output_path)
    return validate_index(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--existing-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = (
        validate_index(args.output)
        if args.validate_only
        else build_index(
            args.raw_manifest,
            args.existing_eval,
            args.output,
            workers=args.workers,
        )
    )
    counts = payload["counts"]
    print(
        "[ok] full episodes only: "
        f"source={counts['source_episodes']} train={counts['train_episodes']} "
        f"eval={counts['eval_episodes']}; offline_windows=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
