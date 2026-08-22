from __future__ import annotations

import hashlib
import json

import pytest
import torch

from scripts.split_wam4va_episode_holdout import (
    PEER_SYNC_H15_P2_CONTRACT,
    PEER_SYNC_H15_P2_TRANSITION_RULE,
    PEER_SYNC_H6_P2_CONTRACT,
    PEER_SYNC_H6_P2_TRANSITION_RULE,
    TRANSITION_RULE,
    build_split_artifacts,
    build_split_plan,
    canonical_manifest_sha256,
)


def _payload(episodes_per_task: int = 10, windows_per_episode: int = 2) -> dict:
    task_ids = []
    episode_ids = []
    frame_refs = []
    names = {0: "assembly-v3", 16: "door-unlock-v3"}
    for task_id in (0, 16):
        for local_episode in range(episodes_per_task):
            episode_id = task_id * 10_000 + local_episode
            for window in range(windows_per_episode):
                task_ids.append(task_id)
                episode_ids.append(episode_id)
                frame_refs.append((names[task_id], local_episode, [[window]] * 3))

    n = len(task_ids)
    actions = torch.zeros(n, 4, 48, 4)
    valid = torch.ones(n, 4, 48, dtype=torch.bool)
    valid[::2, 0, 0] = False
    valid[::3, 2, 0] = False
    recovery = torch.zeros_like(valid)
    recovery[:, :, :2] = True
    descriptions = [f"unused-{index}" for index in range(17)]
    descriptions[0] = "Pick up a nut and place it onto a peg"
    descriptions[16] = "Unlock the door by rotating the lock counter-clockwise"
    return {
        "actions": actions,
        "previous_action": torch.zeros(n, 4, 4),
        "proprio": torch.zeros(n, 4, 4),
        "instruction_id": torch.tensor(task_ids, dtype=torch.long),
        "episode_id": torch.tensor(episode_ids, dtype=torch.long),
        "pair_id": torch.arange(n),
        "frame_refs": frame_refs,
        "action_valid_mask": valid,
        "recovery_mask": recovery,
        "metadata": {
            "tasks": descriptions,
            "n_subset_windows": n,
            "subset_task_ids": [0, 16],
        },
    }


def _peer_payload() -> dict:
    payload = _payload()
    n = len(payload["actions"])
    payload["actions"] = torch.zeros(n, 4, 6, 4)
    payload["action_valid_mask"] = torch.ones(n, 4, 6, dtype=torch.bool)
    payload["recovery_mask"] = torch.zeros(n, 4, 6, dtype=torch.bool)
    payload["metadata"].update(
        {
            "contract": "peer_sync_h6_world_windows_v1",
            "contract_version": 1,
            "logged_action_chunk": "full_h6",
            "parent_identity": {"path": "/parent", "sha256": "p"},
            "source_identities": [{"path": "/source", "sha256": "s"}],
            "output_identity": {"path": "/windows", "shape": {"action_horizon": 6}},
        }
    )
    return payload


def _peer_p2_payload() -> dict:
    payload = _peer_payload()
    payload["metadata"].update(
        {
            "contract": PEER_SYNC_H6_P2_CONTRACT,
            "fps": 80,
            "planning_stride": 2,
            "control_stride": 2,
            "sequence_length": 4,
            "decision_offsets": [0, 2, 4, 6],
            "action_horizon": 6,
            "action_label_offsets": [0, 1, 2, 3, 4, 5],
        }
    )
    return payload


def _peer_h15_p2_payload() -> dict:
    payload = _peer_p2_payload()
    n = len(payload["actions"])
    payload["actions"] = torch.zeros(n, 4, 15, 4)
    payload["action_valid_mask"] = torch.ones(n, 4, 15, dtype=torch.bool)
    payload["recovery_mask"] = torch.zeros(n, 4, 15, dtype=torch.bool)
    payload["world_target_valid_mask"] = torch.ones(n, 4, dtype=torch.bool)
    payload["world_target_frame_refs"] = list(payload["frame_refs"])
    payload["metadata"].update(
        {
            "contract": PEER_SYNC_H15_P2_CONTRACT,
            "logged_action_chunk": "full_h15",
            "action_horizon": 15,
            "action_label_offsets": list(range(15)),
            "world_target_horizon": 15,
            "world_target_offsets": [15, 17, 19, 21],
        }
    )
    return payload


def _direct_mask_stats(payload: dict) -> dict:
    valid = payload["action_valid_mask"]
    recovery = payload["recovery_mask"]
    transition = valid[:, :-1, :6].all(dim=-1) & valid[:, 1:, 0]
    return {
        "action_valid_true": int(valid.sum()),
        "action_valid_total": valid.numel(),
        "recovery_true": int(recovery.sum()),
        "recovery_total": recovery.numel(),
        "transition_true": int(transition.sum()),
        "transition_total": transition.numel(),
    }


def test_builds_task_stratified_split_and_shared_manifest(tmp_path) -> None:
    source = tmp_path / "source.pt"
    train_path = tmp_path / "train.pt"
    eval_path = tmp_path / "eval.pt"
    manifest_path = tmp_path / "split.json"
    payload = _payload()
    torch.save(payload, source)

    returned = build_split_artifacts(
        source, train_path, eval_path, manifest_path, seed=0
    )
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    eval_payload = torch.load(eval_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert returned == manifest
    assert canonical_manifest_sha256(manifest) == manifest["manifest_sha256"]
    assert manifest["manifest_id"].endswith(manifest["manifest_sha256"][:16])
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["source"]["size_bytes"] == source.stat().st_size
    assert manifest["transition_rule"] == TRANSITION_RULE
    assert manifest["validation"] == {
        "episode_single_task": True,
        "episode_disjoint": True,
        "rows_disjoint": True,
        "rows_exhaustive": True,
        "full_logged_action_chunk": True,
    }
    assert manifest["data_protocol"] == {
        "contract": "wam4va_episode_holdout_manifest_v1",
        "shape": {"sequence_length": 4, "action_horizon": 48, "action_dim": 4},
        "logged_action_chunk": "full_h48",
    }

    source_episodes = set(payload["episode_id"].tolist())
    train_episodes = set(train["episode_id"].tolist())
    eval_episodes = set(eval_payload["episode_id"].tolist())
    assert not train_episodes & eval_episodes
    assert train_episodes | eval_episodes == source_episodes
    assert len(train_episodes) == 18
    assert len(eval_episodes) == 2
    for task_id in (0, 16):
        task_eval = eval_payload["episode_id"][
            eval_payload["instruction_id"] == task_id
        ]
        assert len(torch.unique(task_eval)) == 1

    for name, split_payload, path in (
        ("train", train, train_path),
        ("eval", eval_payload, eval_path),
    ):
        metadata = split_payload["metadata"]
        assert metadata["split_name"] == name
        assert metadata["source_n_windows"] == len(payload["actions"])
        assert metadata["n_subset_windows"] == len(split_payload["actions"])
        assert metadata["split_windows"] == len(split_payload["actions"])
        assert metadata["split_contract"] == manifest
        assert metadata["split_manifest_id"] == manifest["manifest_id"]
        assert manifest["splits"][name]["output_path"] == str(path.resolve())

    assert [item["task_name"] for item in manifest["tasks"]] == [
        "assembly-v3",
        "door-unlock-v3",
    ]


def test_manifest_mask_stats_match_output_tensors_exactly(tmp_path) -> None:
    source = tmp_path / "source.pt"
    train_path = tmp_path / "train.pt"
    eval_path = tmp_path / "eval.pt"
    manifest_path = tmp_path / "split.json"
    torch.save(_payload(), source)
    manifest = build_split_artifacts(source, train_path, eval_path, manifest_path)

    for name, path in (("train", train_path), ("eval", eval_path)):
        split_payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = _direct_mask_stats(split_payload)
        stats = manifest["splits"][name]["mask_stats"]
        assert stats["action_valid"]["true"] == expected["action_valid_true"]
        assert stats["action_valid"]["total"] == expected["action_valid_total"]
        assert stats["recovery"]["true"] == expected["recovery_true"]
        assert stats["recovery"]["total"] == expected["recovery_total"]
        assert stats["transition"]["true"] == expected["transition_true"]
        assert stats["transition"]["total"] == expected["transition_total"]

        for task_item in manifest["splits"][name]["tasks"]:
            task_mask = split_payload["instruction_id"] == task_item["task_id"]
            task_payload = {
                "action_valid_mask": split_payload["action_valid_mask"][task_mask],
                "recovery_mask": split_payload["recovery_mask"][task_mask],
            }
            task_expected = _direct_mask_stats(task_payload)
            assert task_item["mask_stats"]["transition"]["true"] == task_expected[
                "transition_true"
            ]
            assert task_item["mask_stats"]["transition"]["total"] == task_expected[
                "transition_total"
            ]


def test_selection_is_deterministic_and_keeps_whole_episodes() -> None:
    payload = _payload(episodes_per_task=30, windows_per_episode=3)
    first = build_split_plan(payload, heldout_fraction=0.10, seed=0)
    second = build_split_plan(payload, heldout_fraction=0.10, seed=0)
    assert first["eval_episodes_by_task"] == second["eval_episodes_by_task"]
    assert torch.equal(first["train_indices"], second["train_indices"])
    assert all(len(episodes) == 3 for episodes in first["eval_episodes_by_task"].values())

    eval_rows = set(first["eval_indices"].tolist())
    for episode_id in torch.unique(payload["episode_id"]).tolist():
        rows = set(
            torch.nonzero(payload["episode_id"] == episode_id, as_tuple=False)
            .flatten()
            .tolist()
        )
        assert rows <= eval_rows or rows.isdisjoint(eval_rows)


def test_rejects_episode_id_shared_by_multiple_tasks() -> None:
    payload = _payload()
    task16_row = int(
        torch.nonzero(payload["instruction_id"] == 16, as_tuple=False)[0, 0]
    )
    payload["episode_id"][task16_row] = payload["episode_id"][0]
    with pytest.raises(ValueError, match="belongs to multiple tasks"):
        build_split_plan(payload)


def test_peer_h6_manifest_preserves_protocol_identities_and_episode_disjointness(tmp_path) -> None:
    source = tmp_path / "peer_source.pt"
    train_path = tmp_path / "peer_train.pt"
    eval_path = tmp_path / "peer_eval.pt"
    manifest_path = tmp_path / "peer_split.json"
    torch.save(_peer_payload(), source)

    manifest = build_split_artifacts(source, train_path, eval_path, manifest_path)
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    eval_payload = torch.load(eval_path, map_location="cpu", weights_only=True)

    assert manifest["data_protocol"]["contract"] == "peer_sync_h6_world_windows_v1"
    assert manifest["data_protocol"]["shape"] == {
        "sequence_length": 4, "action_horizon": 6, "action_dim": 4
    }
    assert manifest["data_protocol"]["logged_action_chunk"] == "full_h6"
    assert manifest["source"]["payload_parent_identity"]["path"] == "/parent"
    assert manifest["source"]["payload_source_identities"][0]["path"] == "/source"
    assert manifest["source"]["payload_output_identity"]["path"] == "/windows"
    assert train["metadata"]["output_identity"] == manifest["splits"]["train"]["output_identity"]
    assert eval_payload["metadata"]["output_identity"] == manifest["splits"]["eval"]["output_identity"]
    assert set(train["episode_id"].tolist()).isdisjoint(eval_payload["episode_id"].tolist())


def test_peer_h6_p2_split_preserves_cadence_and_uses_two_action_transition(tmp_path) -> None:
    source = tmp_path / "peer_p2_source.pt"
    train_path = tmp_path / "peer_p2_train.pt"
    eval_path = tmp_path / "peer_p2_eval.pt"
    manifest_path = tmp_path / "peer_p2_split.json"
    payload = _peer_p2_payload()
    # This action is outside the d -> d+2 transition prefix. It must not mask
    # the transition even though every decision retains a full H6 action label.
    payload["action_valid_mask"][0, 0, 4] = False
    torch.save(payload, source)

    manifest = build_split_artifacts(source, train_path, eval_path, manifest_path)
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    eval_payload = torch.load(eval_path, map_location="cpu", weights_only=True)

    assert manifest["data_protocol"] == {
        "contract": PEER_SYNC_H6_P2_CONTRACT,
        "shape": {"sequence_length": 4, "action_horizon": 6, "action_dim": 4},
        "logged_action_chunk": "full_h6",
        "fps": 80,
        "planning_stride": 2,
        "control_stride": 2,
        "decision_offsets": [0, 2, 4, 6],
        "action_label_offsets": [0, 1, 2, 3, 4, 5],
    }
    assert manifest["transition_rule"] == PEER_SYNC_H6_P2_TRANSITION_RULE
    assert manifest["source"]["mask_stats"]["transition"]["true"] == (
        len(payload["actions"]) * 3
    )
    assert manifest["source"]["payload_parent_identity"]["path"] == "/parent"
    for split_payload in (train, eval_payload):
        metadata = split_payload["metadata"]
        assert metadata["contract"] == PEER_SYNC_H6_P2_CONTRACT
        assert metadata["planning_stride"] == 2
        assert metadata["control_stride"] == 2
        assert metadata["decision_offsets"] == [0, 2, 4, 6]
        assert metadata["source_identities"] == [{"path": "/source", "sha256": "s"}]


def test_peer_h6_p2_split_rejects_incomplete_cadence_metadata() -> None:
    payload = _peer_p2_payload()
    payload["metadata"]["planning_stride"] = 6
    with pytest.raises(ValueError, match="requires metadata.planning_stride=2"):
        build_split_plan(payload)


def test_peer_h15_split_preserves_explicit_endpoint_contract(tmp_path) -> None:
    source = tmp_path / "peer_h15_source.pt"
    train_path = tmp_path / "peer_h15_train.pt"
    eval_path = tmp_path / "peer_h15_eval.pt"
    manifest_path = tmp_path / "peer_h15_split.json"
    payload = _peer_h15_p2_payload()
    payload["world_target_valid_mask"][0, 3] = False
    torch.save(payload, source)

    manifest = build_split_artifacts(
        source, train_path, eval_path, manifest_path
    )
    assert manifest["data_protocol"]["shape"] == {
        "sequence_length": 4,
        "action_horizon": 15,
        "action_dim": 4,
    }
    assert manifest["data_protocol"]["logged_action_chunk"] == "full_h15"
    assert manifest["data_protocol"]["world_target_horizon"] == 15
    assert manifest["transition_rule"] == PEER_SYNC_H15_P2_TRANSITION_RULE
    assert manifest["source"]["mask_stats"]["transition"]["true"] == (
        len(payload["actions"]) * 4 - 1
    )


def test_legacy_split_rejects_non_h48_and_peer_requires_complete_identity() -> None:
    legacy = _payload()
    legacy["actions"] = torch.zeros(len(legacy["actions"]), 4, 6, 4)
    legacy["action_valid_mask"] = torch.ones(len(legacy["actions"]), 4, 6, dtype=torch.bool)
    legacy["recovery_mask"] = torch.zeros_like(legacy["action_valid_mask"])
    with pytest.raises(ValueError, match="legacy H48 split requires exact T4/H48/A4"):
        build_split_plan(legacy)

    peer = _peer_payload()
    del peer["metadata"]["parent_identity"]
    with pytest.raises(ValueError, match="requires metadata.parent_identity"):
        build_split_plan(peer)


def test_refuses_overwrite_before_writing_any_other_output(tmp_path) -> None:
    source = tmp_path / "source.pt"
    train_path = tmp_path / "train.pt"
    eval_path = tmp_path / "eval.pt"
    manifest_path = tmp_path / "split.json"
    torch.save(_payload(), source)
    train_path.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_split_artifacts(source, train_path, eval_path, manifest_path)
    assert train_path.read_bytes() == b"do-not-overwrite"
    assert not eval_path.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))
