from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.extract_lerobot_push_back import (
    FAILED_SOURCE_EPISODES,
    RECOVERED_SOURCE_EPISODES,
    extract_push_back_once,
)


pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")


def test_extracts_hidden_task37_as_fifty_push_back_episodes(tmp_path: Path):
    dataset = tmp_path / "lerobot"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    image = Image.new("RGB", (4, 4), (20, 40, 60))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    episode_indices = list(range(2100, 2150))
    meta = pa.table(
        {
            "episode_index": episode_indices,
            "data/file_index": [0] * 50,
            "dataset_from_index": list(range(50)),
            "length": [1] * 50,
            "tasks": [["Push the puck to a goal"]] * 50,
            "stats/task_id/min": [[37]] * 50,
            "stats/task_index/min": [[40]] * 50,
        }
    )
    pq.write_table(meta, dataset / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(
        pa.table({"task_index": [40], "task": ["Push the puck to a goal"]}),
        dataset / "meta/tasks.parquet",
    )
    data = pa.table(
        {
            "index": list(range(50)),
            "observation.image": [
                {"bytes": encoded.getvalue(), "path": "frame.png"}
            ]
            * 50,
            "observation.state": [[0.0, 0.1, 0.2, 0.3]] * 50,
            "action": [[2.0, -2.0, 0.5, 1.0]] * 50,
            "next.success": [
                episode not in FAILED_SOURCE_EPISODES
                for episode in episode_indices
            ],
            "task_id": [37] * 50,
            "task_index": [40] * 50,
            "episode_index": episode_indices,
        }
    )
    pq.write_table(data, dataset / "data/chunk-000/file-000.parquet")
    norm = tmp_path / "norm.pt"
    torch.save(
        {
            "normalization": {
                "action_q01": torch.zeros(4),
                "action_q99": torch.ones(4),
                "state_q01": torch.zeros(4),
                "state_q99": torch.ones(4),
            }
        },
        norm,
    )
    output = tmp_path / "push-back.pt"
    supplement = tmp_path / "clean-supplement.pt"
    torch.save(
        {
            "task": "push-back-v3",
            "episodes": [
                {
                    "frames": [encoded.getvalue()],
                    "actions": np.zeros((1, 4), dtype=np.float32),
                    "states": np.zeros((1, 4), dtype=np.float32),
                    "first_success": 0,
                    "perturbed": False,
                    "n_perturb_events": 0,
                    "episode_seed": seed,
                }
                for seed in (649000, 649001, 649002)
            ],
        },
        supplement,
    )

    extract_push_back_once(
        dataset, output, normalization_ref=norm, clean_supplement=supplement
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert payload["task"] == "push-back-v3"
    assert payload["n_episodes"] == 50
    assert [
        episode["source_episode_index"] for episode in payload["episodes"][:47]
    ] == list(RECOVERED_SOURCE_EPISODES)
    assert [episode["source_episode_index"] for episode in payload["episodes"][47:]] == [
        None,
        None,
        None,
    ]
    assert [episode["episode_seed"] for episode in payload["episodes"][47:]] == [
        649000,
        649001,
        649002,
    ]
    assert np.array_equal(
        payload["episodes"][0]["actions"][0], np.array([1.0, -1.0, 0.5, 1.0])
    )
    assert payload["episodes"][0]["frames"][0].startswith(b"\xff\xd8")
    assert payload["metadata"]["corrected_text"] == "Pull a puck to a goal"
    assert payload["metadata"]["rejected_no_success_episode_indices"] == list(
        FAILED_SOURCE_EPISODES
    )
