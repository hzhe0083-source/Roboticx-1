"""Recover the 50 mislabeled push-back episodes from the local LeRobot MT50.

The source contains 50 underlying task ids, but task id 37 (push-back-v3) was
assigned the same text/task_index as push-v3.  This converter selects by the
preserved underlying task id, fixes the task identity, clips actions to the
actions actually executed by MetaWorld, and emits the long-trajectory format.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


TASK = "push-back-v3"
SOURCE_TASK_ID = 37
SOURCE_TASK_INDEX = 40
SOURCE_EPISODES = tuple(range(2100, 2150))
FAILED_SOURCE_EPISODES = (2138, 2143, 2144)
RECOVERED_SOURCE_EPISODES = tuple(
    episode for episode in SOURCE_EPISODES if episode not in FAILED_SOURCE_EPISODES
)
SUPPLEMENT_EPISODES = 3
CONTRACT = "lerobot_mt50_push_back_47_plus_clean3_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jpeg(raw: bytes, quality: int) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=quality)
        return output.getvalue()


def validate_output(path: Path, clean_supplement: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes")
    metadata = payload.get("metadata") or {}
    if (
        payload.get("task") != TASK
        or not isinstance(episodes, list)
        or len(episodes) != 50
        or payload.get("n_episodes") != 50
        or metadata.get("contract") != CONTRACT
        or metadata.get("source_task_id") != SOURCE_TASK_ID
        or metadata.get("source_episode_indices") != list(SOURCE_EPISODES)
        or metadata.get("recovered_source_episode_indices")
        != list(RECOVERED_SOURCE_EPISODES)
        or metadata.get("rejected_no_success_episode_indices")
        != list(FAILED_SOURCE_EPISODES)
    ):
        raise ValueError(f"{path}: invalid recovered push-back payload")
    if [episode.get("source_episode_index") for episode in episodes[:47]] != list(
        RECOVERED_SOURCE_EPISODES
    ) or any(episode.get("source_episode_index") is not None for episode in episodes[47:]):
        raise ValueError(f"{path}: recovered source episode order differs")
    supplement = torch.load(clean_supplement, map_location="cpu", weights_only=False)
    expected_seeds = [
        episode.get("episode_seed")
        for episode in (supplement.get("episodes") or [])[:SUPPLEMENT_EPISODES]
    ]
    if (
        len(expected_seeds) != SUPPLEMENT_EPISODES
        or metadata.get("supplemental_episode_seeds") != expected_seeds
        or [episode.get("episode_seed") for episode in episodes[47:]]
        != expected_seeds
    ):
        raise ValueError(f"{path}: clean supplement differs")
    for index, episode in enumerate(episodes):
        n = len(episode.get("frames") or [])
        if (
            np.asarray(episode.get("actions")).shape != (n, 4)
            or np.asarray(episode.get("states")).shape != (n, 4)
            or int(episode.get("n_perturb_events", -1)) != 0
            or bool(episode.get("perturbed", True))
        ):
            raise ValueError(f"{path}: invalid recovered episode {index}")
    return payload


def extract_push_back_once(
    dataset: Path,
    output: Path,
    *,
    normalization_ref: Path,
    clean_supplement: Path,
    jpeg_quality: int = 90,
) -> Path:
    if output.exists():
        validate_output(output, clean_supplement)
        return output
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")

    import pyarrow.parquet as pq

    dataset = dataset.expanduser().resolve(strict=True)
    episode_meta_path = dataset / "meta/episodes/chunk-000/file-000.parquet"
    task_meta_path = dataset / "meta/tasks.parquet"
    episode_rows = pq.read_table(episode_meta_path).to_pylist()
    selected = [
        row
        for row in episode_rows
        if int(row["stats/task_id/min"][0]) == SOURCE_TASK_ID
    ]
    if [int(row["episode_index"]) for row in selected] != list(SOURCE_EPISODES):
        raise ValueError("LeRobot task id 37 is not the expected 50 push-back episodes")
    if any(
        row["tasks"] != ["Push the puck to a goal"]
        or int(row["stats/task_index/min"][0]) != SOURCE_TASK_INDEX
        for row in selected
    ):
        raise ValueError("the expected LeRobot push/push-back label collision changed")

    columns = [
        "index",
        "observation.image",
        "observation.state",
        "action",
        "next.success",
        "task_id",
        "task_index",
        "episode_index",
    ]
    episodes = []
    cached_file_index: int | None = None
    cached_table = None
    for item in selected:
        file_index = int(item["data/file_index"])
        if file_index != cached_file_index:
            data_path = dataset / f"data/chunk-000/file-{file_index:03d}.parquet"
            cached_table = pq.read_table(data_path, columns=columns)
            cached_file_index = file_index
        assert cached_table is not None
        first = int(cached_table["index"][0].as_py())
        start = int(item["dataset_from_index"]) - first
        length = int(item["length"])
        rows = cached_table.slice(start, length).to_pylist()
        episode_index = int(item["episode_index"])
        if (
            len(rows) != length
            or any(int(row["episode_index"]) != episode_index for row in rows)
            or any(int(row["task_id"]) != SOURCE_TASK_ID for row in rows)
            or any(int(row["task_index"]) != SOURCE_TASK_INDEX for row in rows)
        ):
            raise ValueError(f"LeRobot episode {episode_index} row identity differs")
        success = np.asarray([bool(row["next.success"]) for row in rows])
        hits = np.flatnonzero(success)
        if not len(hits):
            if episode_index not in FAILED_SOURCE_EPISODES:
                raise ValueError(
                    f"unexpected LeRobot push-back episode without success: {episode_index}"
                )
            continue
        first_success = int(hits[0])
        actions = np.clip(
            np.asarray([row["action"] for row in rows], dtype=np.float32), -1.0, 1.0
        )
        states = np.asarray(
            [row["observation.state"] for row in rows], dtype=np.float32
        )
        valid = np.arange(length) <= first_success
        episodes.append(
            {
                "frames": [
                    _jpeg(row["observation.image"]["bytes"], jpeg_quality)
                    for row in rows
                ],
                "actions": actions,
                "states": states,
                "first_success": first_success,
                "success_frame": first_success,
                "frame_valid": np.ones(length, dtype=bool),
                "action_executed": np.ones(length, dtype=bool),
                "action_valid": valid,
                "action_supervision_valid": valid,
                "settle_mask": np.zeros(length, dtype=bool),
                "recovery_mask": np.zeros(length, dtype=bool),
                "perturbed": False,
                "n_perturb_events": 0,
                "source_episode_index": episode_index,
            }
        )

    if len(episodes) != len(RECOVERED_SOURCE_EPISODES):
        raise ValueError(
            f"expected {len(RECOVERED_SOURCE_EPISODES)} successful LeRobot episodes, "
            f"got {len(episodes)}"
        )
    supplement_payload = torch.load(
        clean_supplement, map_location="cpu", weights_only=False
    )
    supplement = list(supplement_payload.get("episodes") or [])[:SUPPLEMENT_EPISODES]
    if (
        supplement_payload.get("task") != TASK
        or len(supplement) != SUPPLEMENT_EPISODES
        or any(
            bool(episode.get("perturbed", False))
            or int(episode.get("n_perturb_events", -1)) != 0
            or episode.get("first_success") is None
            for episode in supplement
        )
    ):
        raise ValueError(f"{clean_supplement}: invalid clean push-back supplement")
    episodes.extend({**episode, "source_episode_index": None} for episode in supplement)

    normalization = torch.load(
        normalization_ref, map_location="cpu", weights_only=True
    )["normalization"]
    payload = {
        "task": TASK,
        "n_episodes": 50,
        "episodes": episodes,
        "normalization": normalization,
        "metadata": {
            "contract": CONTRACT,
            "fps": 80,
            "source_task_id": SOURCE_TASK_ID,
            "source_task_index": SOURCE_TASK_INDEX,
            "source_mislabeled_text": "Push the puck to a goal",
            "corrected_text": "Pull a puck to a goal",
            "source_episode_indices": list(SOURCE_EPISODES),
            "recovered_source_episode_indices": list(RECOVERED_SOURCE_EPISODES),
            "rejected_no_success_episode_indices": list(FAILED_SOURCE_EPISODES),
            "supplemental_episode_seeds": [
                episode.get("episode_seed") for episode in supplement
            ],
            "clean_supplement": {
                "path": str(clean_supplement.expanduser().resolve(strict=True)),
                "sha256": _sha256(clean_supplement),
            },
            "action_contract": "executed-clip-fullframe",
            "image_contract": f"source_png_recompressed_jpeg_q{jpeg_quality}",
            "source_dataset": str(dataset),
            "source_metadata": {
                "episodes_sha256": _sha256(episode_meta_path),
                "tasks_sha256": _sha256(task_meta_path),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    validate_output(output, clean_supplement)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/root/private_data/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization-ref",
        type=Path,
        default=Path("/root/private_data/ORA0/data/longtraj_normalization_ref.pt"),
    )
    parser.add_argument(
        "--clean-supplement",
        type=Path,
        default=Path(
            "/root/ora0_all49_expand60_v1/push_back/shards/"
            "metaworld_longtraj_push-back-v3_clean_v1_shard0.pt"
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    output = extract_push_back_once(
        args.dataset,
        args.output,
        normalization_ref=args.normalization_ref,
        clean_supplement=args.clean_supplement,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"[ok] recovered 50 push-back episodes: {output}")


if __name__ == "__main__":
    main()
