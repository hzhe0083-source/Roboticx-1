from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch

from scripts.build_longtraj_features import ENV_TO_TASK
from scripts.convert_evo1_metaworld_dataset import (
    OFFICIAL_DESCRIPTION_ALIASES,
    SHORT_EPISODE_PADDING,
    convert_dataset,
)
from va_compound.longtraj_frames import OnlineLongTrajEpisodeDataset


def _jpeg() -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(stream, format="JPEG")
    return stream.getvalue()


def test_evo1_converter_preserves_mt50_and_short_episode_h50_contract(tmp_path) -> None:
    source = tmp_path / "official"
    output = tmp_path / "overlay"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    video_dir = source / "videos/chunk-000/observation.images.image"
    video_dir.mkdir(parents=True)

    canonical_tasks = [
        description
        for env, description in ENV_TO_TASK.items()
        if env != "push-back-v3"
    ] + [ENV_TO_TASK["push-back-v3"]]
    reverse_alias = {canonical: official for official, canonical in OFFICIAL_DESCRIPTION_ALIASES.items()}
    official_tasks = [reverse_alias.get(task, task) for task in canonical_tasks]
    (source / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": task}) + "\n"
            for index, task in enumerate(official_tasks)
        ),
        encoding="utf-8",
    )

    episode_rows = []
    episode_index = 0
    for task_index, task in enumerate(official_tasks):
        for length in (14, 75):
            episode_rows.append(
                {"episode_index": episode_index, "tasks": [task], "length": length}
            )
            values = np.arange(length * 4, dtype=np.float32).reshape(length, 4) / 100
            table = pa.table(
                {
                    "observation.state": pa.array(
                        values.tolist(), type=pa.list_(pa.float32(), 4)
                    ),
                    "action": pa.array(
                        (values + 0.25).tolist(), type=pa.list_(pa.float32(), 4)
                    ),
                    "frame_index": np.arange(length, dtype=np.int64),
                    "episode_index": np.full(length, episode_index, dtype=np.int64),
                    "task_index": np.full(length, task_index, dtype=np.int64),
                }
            )
            pq.write_table(
                table,
                source / f"data/chunk-000/episode_{episode_index:06d}.parquet",
            )
            (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"fixture")
            episode_index += 1
    (source / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    (source / "meta/info.json").write_text(
        json.dumps(
            {
                "total_tasks": 50,
                "total_episodes": len(episode_rows),
                "chunks_size": 1000,
                "fps": 30,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [4]},
                    "action": {"dtype": "float32", "shape": [4]},
                    "observation.images.image": {
                        "dtype": "video",
                        "shape": [480, 480, 3],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    reference = tmp_path / "language.pt"
    torch.save(
        {
            "metadata": {"tasks": canonical_tasks},
            "language_hidden": torch.zeros(50, 2, 3),
            "language_mask": torch.ones(50, 2, dtype=torch.bool),
            "normalization": {
                "action_q01": torch.full((4,), -2.0),
                "action_q99": torch.full((4,), 2.0),
                "state_q01": torch.full((4,), -2.0),
                "state_q99": torch.full((4,), 2.0),
            },
        },
        reference,
    )
    jpeg = _jpeg()

    def fake_decode(_path: Path, *, expected_frames: int, **_kwargs) -> list[bytes]:
        return [jpeg] * expected_frames

    manifest_path, index_path = convert_dataset(
        source,
        output,
        reference,
        episodes_per_task=2,
        decode_video=fake_decode,
    )
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(index_path.read_text())
    assert len(manifest["sources"]) == 50
    assert manifest["sources"][-1]["task"] == "push-back-v3"
    assert index["sampling_protocol"]["short_episode_padding"] == SHORT_EPISODE_PADDING
    assert index["counts"] == {
        "source_episodes": 100,
        "train_episodes": 100,
        "eval_episodes": 0,
    }
    assert {row["valid_start_count"] for row in index["episodes"]} == {14, 75}
    assert {row["task_id"] for row in index["episodes"]} == set(range(50))

    raw = torch.load(
        output / "metaworld_longtraj_assembly-v3.pt",
        map_location="cpu",
        weights_only=False,
    )
    short = raw["episodes"][0]
    assert short["states"].shape == short["actions"].shape == (14, 4)
    assert short["states"].dtype == short["actions"].dtype == np.float32
    assert short["first_success"] == 13
    assert len(short["frames"]) == 14
    assert raw["metadata"]["camera_name"] == "corner2"
    assert raw["metadata"]["dino_preencoded"] is False

    dataset = OnlineLongTrajEpisodeDataset(
        index_path,
        longtraj_dir=output,
        samples_per_episode=1,
        action_horizon=50,
    )
    item = dataset[0]
    assert item["actions"].shape == (4, 50, 4)
    assert item["frames"].shape[-3:] == (2, 2, 3)
    assert not bool(item["action_valid_mask"].all())
