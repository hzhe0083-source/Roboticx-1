from pathlib import Path

import pytest
import torch

from scripts.finalize_all49_to60 import (
    PUSH_BACK_EVAL_EPISODES,
    build_push_back_merge_once,
    compact_mt50_language_reference,
    make_mt50_eval_payload,
    stable_old_prefix,
    validate_mt50_language_reference,
)
from scripts.extract_lerobot_push_back import CONTRACT as PUSH_BACK_BASE_CONTRACT
from scripts.extract_lerobot_push_back import (
    FAILED_SOURCE_EPISODES,
    RECOVERED_SOURCE_EPISODES,
)


def _payload(rows):
    n = len(rows)
    return {
        "actions": torch.tensor([[float(row[3])] for row in rows]),
        "instruction_id": torch.tensor([row[0] for row in rows]),
        "episode_id": torch.tensor([row[1] for row in rows]),
        "pair_id": torch.arange(n),
        "frame_refs": [(row[2], row[1], [[row[3]]]) for row in rows],
        "labels": [f"label-{row[3]}" for row in rows],
        "metadata": {"output_identity": {"shape": {"windows": n}}},
    }


def test_stable_old_prefix_preserves_every_old_aligned_value(tmp_path: Path):
    old_rows = [(0, 0, "task-a", 10), (1, 10000, "task-b", 20)]
    expanded_rows = [
        (0, 0, "task-a", 10),
        (0, 1, "task-a", 11),
        (1, 10000, "task-b", 20),
        (1, 10001, "task-b", 21),
    ]

    result = stable_old_prefix(
        _payload(expanded_rows),
        _payload(old_rows),
        output_path=tmp_path / "source.pt",
    )

    assert result["instruction_id"].tolist() == [0, 1, 0, 1]
    assert result["episode_id"].tolist() == [0, 10000, 1, 10001]
    assert result["pair_id"].tolist() == [0, 1, 2, 3]
    assert result["labels"][:2] == ["label-10", "label-20"]
    assert result["metadata"]["stable_old_source_prefix_rows"] == 2


def test_stable_old_prefix_rejects_missing_old_row(tmp_path: Path):
    with pytest.raises(ValueError, match="old source row is absent"):
        stable_old_prefix(
            _payload([(0, 0, "task-a", 10)]),
            _payload([(0, 9, "task-a", 99)]),
            output_path=tmp_path / "source.pt",
        )


def test_mt50_language_reference_preserves_old_ids_and_appends_push_back(
    tmp_path: Path,
):
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"base-reference")
    qwen_path = tmp_path / "Qwen3.5-2B"
    qwen_path.mkdir()
    old_hidden = torch.arange(49 * 4 * 3, dtype=torch.float16).reshape(49, 4, 3)
    old_mask = torch.ones(49, 4, dtype=torch.bool)
    base = {
        "normalization": {
            "action_q01": torch.zeros(4),
            "action_q99": torch.ones(4),
            "state_q01": torch.zeros(4),
            "state_q99": torch.ones(4),
        },
        "instruction_id": torch.arange(49),
        "language_hidden": old_hidden,
        "language_mask": old_mask,
        "metadata": {"tasks": [f"task-{index}" for index in range(49)]},
    }
    result = compact_mt50_language_reference(
        base,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([True, True]),
        base_reference_path=base_path,
        qwen_model=qwen_path,
    )

    validate_mt50_language_reference(result, base)
    assert result["metadata"]["tasks"][-1] == "Pull a puck to a goal"
    assert torch.equal(result["language_hidden"][:49], old_hidden)
    assert result["language_hidden"].shape == (50, 4, 3)
    assert result["language_mask"][49].tolist() == [True, True, False, False]
    assert result["instruction_id"].tolist() == list(range(50))


def test_mt50_eval_is_frozen_all49_plus_three_clean_push_back(tmp_path: Path):
    frozen_rows = [
        (task_id, task_id * 10000, f"task-{task_id}", task_id)
        for task_id in range(49)
    ]
    source_rows = [
        *frozen_rows,
        (0, 1, "task-0", 1000),
        *[
            (49, episode_id, "push-back-v3", 2000 + offset)
            for offset, episode_id in enumerate(PUSH_BACK_EVAL_EPISODES)
        ],
        (49, 490003, "push-back-v3", 2003),
    ]
    source = _payload(source_rows)
    source["metadata"]["tasks"] = [f"task-{index}" for index in range(49)] + [
        "Pull a puck to a goal"
    ]
    frozen = _payload(frozen_rows)
    frozen_path = tmp_path / "all49-eval.pt"
    frozen_path.write_bytes(b"immutable eval")

    result = make_mt50_eval_payload(
        source,
        frozen,
        frozen_eval_path=frozen_path,
        output_path=tmp_path / "mt50-eval.pt",
    )

    assert torch.equal(result["actions"][:49], frozen["actions"])
    assert result["instruction_id"].tolist() == list(range(49)) + [49, 49, 49]
    assert result["episode_id"][-3:].tolist() == list(PUSH_BACK_EVAL_EPISODES)
    assert result["metadata"]["frozen_all49_eval_rows"] == 49


def test_push_back_merge_is_exactly_50_recovered_plus_10_recovery(tmp_path: Path):
    base = tmp_path / "metaworld_longtraj_push-back-v3_lerobot50.pt"
    recovery = tmp_path / "metaworld_longtraj_push-back-v3_recovery_v1_shard0.pt"
    torch.save(
        {
            "task": "push-back-v3",
            "n_episodes": 50,
            "episodes": [
                {
                    "source_episode_index": source_index,
                    "n_perturb_events": 0,
                    "perturbed": False,
                }
                for source_index in RECOVERED_SOURCE_EPISODES
            ]
            + [
                {
                    "source_episode_index": None,
                    "episode_seed": seed,
                    "n_perturb_events": 0,
                    "perturbed": False,
                }
                for seed in (649000, 649001, 649002)
            ],
            "metadata": {
                "contract": PUSH_BACK_BASE_CONTRACT,
                "rejected_no_success_episode_indices": list(
                    FAILED_SOURCE_EPISODES
                ),
            },
        },
        base,
    )
    torch.save(
        {
            "task": "push-back-v3",
            "n_episodes": 10,
            "episodes": [
                {
                    "episode_seed": seed,
                    "n_perturb_events": 1,
                    "perturbed": True,
                }
                for seed in range(649030, 649040)
            ],
            "metadata": {},
        },
        recovery,
    )
    output = tmp_path / "frames" / "metaworld_longtraj_push-back-v3.pt"
    build_push_back_merge_once(base, recovery, output)
    merged = torch.load(output, map_location="cpu", weights_only=False)

    assert merged["n_episodes"] == 60
    assert [
        episode["source_episode_index"] for episode in merged["episodes"][:47]
    ] == list(RECOVERED_SOURCE_EPISODES)
    assert [
        episode["episode_seed"] for episode in merged["episodes"][47:50]
    ] == [649000, 649001, 649002]
    assert [episode["episode_seed"] for episode in merged["episodes"][50:]] == list(
        range(649030, 649040)
    )
    assert all(not episode["perturbed"] for episode in merged["episodes"][:50])
    assert all(episode["perturbed"] for episode in merged["episodes"][50:])
