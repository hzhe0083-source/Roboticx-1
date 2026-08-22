"""Tests for scripts/split_libero.py (episode-level train/heldout split).

The v4 leak root cause was training episodes mixed into the eval set; this
split must guarantee disjoint train/heldout episode sets, per-task
stratification, seed reproducibility, and pair-group integrity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split_libero import _write, split_rows  # noqa: E402


def make_payload(
    n_tasks: int = 4,
    episodes_per_task: int = 10,
    rows_per_episode: int = 2,
    seq_len: int = 4,
    pair_groups: bool = False,
) -> dict:
    n = n_tasks * episodes_per_task * rows_per_episode
    episode_id = torch.arange(n // rows_per_episode).repeat_interleave(rows_per_episode)
    instruction_id = torch.arange(n_tasks).repeat_interleave(
        episodes_per_task * rows_per_episode
    )
    pair_id = torch.full((n,), -1, dtype=torch.long)
    if pair_groups:
        # 4-row groups (one row per task) like the real fork contract.
        for g, start in enumerate(range(0, n, n_tasks)):
            pair_id[start : start + n_tasks] = g
    return {
        "episode_id": episode_id,
        "instruction_id": instruction_id,
        "pair_id": pair_id,
        "vision_tokens": torch.zeros(n, seq_len, 4, 8, dtype=torch.float16),
        "proprio": torch.zeros(n, seq_len, 9),
        "previous_action": torch.zeros(n, seq_len, 7),
        "actions": torch.zeros(n, seq_len, 4, 7),
        "language_hidden": torch.zeros(n, 16, 8, dtype=torch.float16),
        "language_mask": torch.ones(n, 16, dtype=torch.bool),
        "metadata": {"tasks": [f"task{i}" for i in range(n_tasks)]},
    }


def test_episode_unit_counts_and_disjointness() -> None:
    p = make_payload()
    train, heldout, report, unit = split_rows(
        p["episode_id"], p["instruction_id"], p["pair_id"], 3, seed=0
    )
    assert unit == "episode"
    # 4 tasks x 10 episodes x 2 rows; 3 episodes/task held out.
    assert len(train) == 4 * 7 * 2
    assert len(heldout) == 4 * 3 * 2
    train_ep = set(p["episode_id"][torch.tensor(train)].tolist())
    heldout_ep = set(p["episode_id"][torch.tensor(heldout)].tolist())
    assert not (train_ep & heldout_ep)
    assert len(train_ep) == 4 * 7 and len(heldout_ep) == 4 * 3
    for t in range(4):
        assert report[t]["heldout_units"] == 3


def test_seed_reproducibility() -> None:
    p = make_payload()
    a, h1, _, _ = split_rows(p["episode_id"], p["instruction_id"], p["pair_id"], 3, seed=7)
    b, h2, _, _ = split_rows(p["episode_id"], p["instruction_id"], p["pair_id"], 3, seed=7)
    assert a == b and h1 == h2
    c, h3, _, _ = split_rows(p["episode_id"], p["instruction_id"], p["pair_id"], 3, seed=8)
    assert h1 != h3  # different seed -> different heldout


def test_pair_groups_never_torn() -> None:
    p = make_payload(pair_groups=True)  # 4-row groups spanning tasks
    train, heldout, _, unit = split_rows(
        p["episode_id"], p["instruction_id"], p["pair_id"], 3, seed=0
    )
    assert unit == "pair-group"
    for g in p["pair_id"].unique().tolist():
        rows = (p["pair_id"] == g).nonzero().flatten().tolist()
        in_train = all(r in set(train) for r in rows)
        in_heldout = all(r in set(heldout) for r in rows)
        assert in_train != in_heldout  # whole group on one side only
    # 5 groups/task (rows 0-19 per task, one 4-row group per pair), 3 held out
    assert len(heldout) == 4 * 3 * 4
    assert len(train) == 4 * 2 * 4


def test_insufficient_episodes_raises() -> None:
    p = make_payload(episodes_per_task=5)
    with pytest.raises(ValueError):
        split_rows(p["episode_id"], p["instruction_id"], p["pair_id"], 8, seed=0)


def test_write_slices_list_sequences(tmp_path: Path) -> None:
    """e2e payloads carry `instructions` as a list — it must be sliced too."""
    p = make_payload(rows_per_episode=1)
    p["instructions"] = [f"instr{i}" for i in range(len(p["actions"]))]
    train, held, _, _ = split_rows(
        p["episode_id"], p["instruction_id"], p["pair_id"], 2, seed=0
    )
    out_train = tmp_path / "train.pt"
    out_held = tmp_path / "held.pt"
    meta = {"protocol": "test", "seed": 0, "heldout_per_task": 2,
            "unit": "episode", "disjoint": True}
    _write(p, train, out_train, meta)
    _write(p, held, out_held, meta)
    t = torch.load(out_train, weights_only=True)
    h = torch.load(out_held, weights_only=True)
    assert len(t["instructions"]) == len(t["actions"]) == len(train)
    assert len(h["instructions"]) == len(h["actions"]) == len(held)
    # instruction text travels with its row
    assert all(t["instructions"][i] == f"instr{t['episode_id'][i].item()}"
               for i in range(len(train)))


def test_aligned_heldout_same_ordinals_per_task() -> None:
    """--aligned-heldout: every task holds out the same episode ordinals, so
    the r-th kept row of each task corresponds to the same original episode
    (row-position alignment needed by scripts/data/prepare_libero_paired.py)."""
    p = make_payload()
    train, held, report, unit = split_rows(
        p["episode_id"], p["instruction_id"], p["pair_id"], 3, seed=0,
        aligned_heldout=True,
    )
    assert unit == "episode"
    per_task_held_ordinals = []
    for t in range(4):
        eps = sorted(
            set(p["episode_id"][torch.tensor(held)].tolist())
            & set(range(t * 10, t * 10 + 10))
        )
        per_task_held_ordinals.append([e - t * 10 for e in eps])
    assert all(o == per_task_held_ordinals[0] for o in per_task_held_ordinals)
    # 3 episodes held out per task; train rows keep alignment: the r-th train
    # row of every task is the same ordinal.
    train_ordinals = {}
    for t in range(4):
        rows = [i for i in train if int(p["instruction_id"][i]) == t]
        ords = [int(p["episode_id"][i]) - t * 10 for i in rows]
        train_ordinals[t] = ords
    assert all(o == train_ordinals[0] for o in train_ordinals.values())


def test_rows_preserve_order_and_task_balance() -> None:
    p = make_payload(rows_per_episode=3)
    train, heldout, _, _ = split_rows(
        p["episode_id"], p["instruction_id"], p["pair_id"], 2, seed=1
    )
    # train rows keep ascending original order within each episode
    assert train == sorted(train)
    # each task keeps 3*(10-2) = 24 train rows
    train_task = p["instruction_id"][torch.tensor(train)]
    for t in range(4):
        assert int((train_task == t).sum()) == 24
