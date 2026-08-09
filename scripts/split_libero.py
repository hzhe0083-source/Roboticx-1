"""Episode-level train/heldout split for LIBERO payloads (leak-free rebuild).

Fixes the v4 leakage root cause (training episodes mixed into the eval set,
see artifacts/libero_va2_diagnosis.md): splits a payload by episode_id with
per-task stratification, so no episode appears in both outputs.  Downstream
training / eval scripts take the output files via --data unchanged — no
training-loop or eval-script changes needed.

Pair safety: if the payload's pair_id forms groups of size > 1 (e.g. the
4-row same-state fork groups produced by prepare_libero_paired.py), groups
are treated as indivisible units so a pair is never torn across the split.
For the base feature file (pair size 1) the unit is the episode.

Usage:
  python scripts/split_libero.py --input data/libero_3scene.pt \
      --heldout-per-task 8 --seed 0 \
      --out-train data/libero_3scene_train.pt \
      --out-heldout data/libero_3scene_heldout.pt

The split protocol (seed, heldout-per-task, per-task counts, disjointness)
is recorded in each output's metadata["split"] and printed to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-train", type=Path, required=True)
    p.add_argument("--out-heldout", type=Path, required=True)
    p.add_argument("--heldout-per-task", type=int, default=8,
                   help="episodes held out per task (must be < episodes per task)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aligned-heldout", action="store_true",
                   help="hold out the same episode ordinals for every task "
                        "(restores row-position alignment across tasks, so "
                        "prepare_libero_paired.py's k-th-row pairing stays valid "
                        "on the split output)")
    p.add_argument("--max-tasks", type=int, default=0,
                   help="only split the first N tasks (dry-run validation)")
    return p.parse_args()


def _unique_unit_ids(rows: list[int]) -> list[int]:
    return sorted(set(rows))


def split_rows(
    episode_id: torch.Tensor,
    instruction_id: torch.Tensor,
    pair_id: torch.Tensor | None,
    heldout_per_task: int,
    seed: int,
    max_tasks: int = 0,
    aligned_heldout: bool = False,
) -> tuple[list[int], list[int], dict, str]:
    """Return (train_rows, heldout_rows, report) for a per-task split.

    Units are pair groups when pairs exist (size > 1), otherwise episodes.
    Per task the units are seeded-shuffled and the last ``heldout_per_task``
    units are held out.
    """
    n = len(episode_id)
    unit = "episode"
    has_real_pairs = False
    if pair_id is not None:
        vals, counts = pair_id.unique(return_counts=True)
        has_real_pairs = bool(int(((vals > -1) & (counts > 1)).sum()))
    if has_real_pairs:
        # Real pair groups: unit = pair group (never tear a pair).
        unit = "pair-group"
        by_pair: dict[int, list[int]] = defaultdict(list)
        for i, g in enumerate(pair_id.tolist()):
            by_pair[g].append(i)
        units = sorted(by_pair.values(), key=lambda rows: min(rows))
        unit_task = [int(instruction_id[rows[0]]) for rows in units]
        unit_ep = [episode_id[rows[0]].item() for rows in units]
    else:
        # No real pairs: unit = episode.
        by_ep: dict[int, list[int]] = defaultdict(list)
        for i, e in enumerate(episode_id.tolist()):
            by_ep[e].append(i)
        units = sorted(by_ep.values(), key=lambda rows: min(rows))
        unit_task = [int(instruction_id[rows[0]]) for rows in units]
        unit_ep = [episode_id[rows[0]].item() for rows in units]

    n_tasks = int(instruction_id.max()) + 1
    rng = torch.Generator().manual_seed(seed)
    per_task: dict[int, list[int]] = defaultdict(list)
    for u in range(len(units)):
        per_task[unit_task[u]].append(u)
    if aligned_heldout:
        # One shared heldout ordinal set across tasks: restores row-position
        # alignment (the r-th kept row of every task is the same original
        # episode ordinal), which keeps prepare_libero_paired.py's k-th-row
        # pairing valid on split outputs.
        if unit != "episode":
            raise ValueError("--aligned-heldout requires episode-unit splits")
        counts = {
            len(per_task[t])
            for t in range(n_tasks)
            if not (max_tasks and t >= max_tasks)
        }
        if len(counts) != 1:
            raise ValueError(
                f"--aligned-heldout requires equal episodes per task, got {sorted(counts)}"
            )
        ep_per_task = counts.pop()
        if ep_per_task < heldout_per_task:
            raise ValueError(
                f"need >= {heldout_per_task} units per task, got {ep_per_task}"
            )
        # Unit index == episode ordinal per task (ascending episode order).
        perm = torch.randperm(ep_per_task, generator=rng).tolist()
        shared_heldout = set(perm[:heldout_per_task])
        heldout_units: set[int] = set()
        report: dict = {}
        for t in range(n_tasks):
            if max_tasks and t >= max_tasks:
                break
            h = [per_task[t][k] for k in sorted(shared_heldout)]
            heldout_units.update(h)
            report[t] = {
                "units": len(per_task[t]),
                "heldout_units": len(h),
                "heldout_episodes": sorted(unit_ep[u] for u in h),
            }
        train_rows = [
            r for u in range(len(units)) if u not in heldout_units for r in units[u]
        ]
        heldout_rows = [r for u in sorted(heldout_units) for r in units[u]]
        return train_rows, heldout_rows, report, unit
    for t in range(n_tasks):
        if max_tasks and t >= max_tasks:
            break
        if len(per_task[t]) < heldout_per_task:
            raise ValueError(
                f"task {t} has only {len(per_task[t])} units, need >= {heldout_per_task}"
            )
        perm = torch.randperm(len(per_task[t]), generator=rng).tolist()
        per_task[t] = [per_task[t][i] for i in perm]

    heldout_units: set[int] = set()
    report: dict = {}
    for t in range(n_tasks):
        if max_tasks and t >= max_tasks:
            break
        h = per_task[t][:heldout_per_task]
        heldout_units.update(h)
        report[t] = {
            "units": len(per_task[t]),
            "heldout_units": len(h),
            "heldout_episodes": sorted(unit_ep[u] for u in h),
        }

    train_rows = [r for u in range(len(units)) if u not in heldout_units for r in units[u]]
    heldout_rows = [r for u in sorted(heldout_units) for r in units[u]]
    return train_rows, heldout_rows, report, unit


def _write(payload: dict, rows: list[int], out: Path, split_meta: dict) -> None:
    n = len(payload.get("actions", []))
    idx = torch.tensor(rows, dtype=torch.long)
    kept: dict = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == n:
            kept[key] = value[idx]
        elif isinstance(value, (list, tuple)) and len(value) == n:
            # e.g. the e2e payload's `instructions` list of strings.
            kept[key] = [value[i] for i in rows]
        else:
            kept[key] = value
    kept = dict(kept)
    meta = dict(kept.get("metadata", {}) or {})
    meta["split"] = split_meta
    kept["metadata"] = meta
    torch.save(kept, out)
    print(f"saved: {out} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    for key in ("episode_id", "instruction_id", "actions"):
        if key not in payload:
            raise ValueError(f"payload missing {key!r}; not a LIBERO feature payload")

    episode_id = payload["episode_id"]
    instruction_id = payload["instruction_id"]
    pair_id = payload.get("pair_id")
    train_rows, heldout_rows, report, unit = split_rows(
        episode_id, instruction_id, pair_id, args.heldout_per_task, args.seed,
        max_tasks=args.max_tasks, aligned_heldout=args.aligned_heldout,
    )

    # Disjointness assertion (train ∩ heldout must be empty by construction).
    train_ep = set(episode_id[torch.tensor(train_rows)].tolist())
    heldout_ep = set(episode_id[torch.tensor(heldout_rows)].tolist())
    overlap = train_ep & heldout_ep
    if overlap:
        raise RuntimeError(f"split overlap on episodes: {sorted(overlap)[:10]}")
    all_ep = train_ep | heldout_ep
    if args.max_tasks:
        print(f"[dry-run limited to first {args.max_tasks} tasks]")
    print(f"rows: {len(train_rows)} train / {len(heldout_rows)} heldout "
          f"(episodes: {len(train_ep)} / {len(heldout_ep)})")
    for t in sorted(report):
        r = report[t]
        print(f"  task {t}: {r['units']} units, {r['heldout_units']} held out, "
              f"episodes={r['heldout_episodes']}")
    if not all_ep:
        raise SystemExit("empty split")

    split_meta = {
        "protocol": "per-task stratified seeded shuffle; last units held out",
        "seed": args.seed,
        "heldout_per_task": args.heldout_per_task,
        "unit": unit,
        "n_train_rows": len(train_rows),
        "n_heldout_rows": len(heldout_rows),
        "n_train_episodes": len(train_ep),
        "n_heldout_episodes": len(heldout_ep),
        "disjoint": True,
        "input_sha256": hashlib.sha256(
            str(Path(args.input).resolve()).encode()
        ).hexdigest()[:16],
    }
    _write(payload, train_rows, args.out_train, split_meta)
    _write(payload, heldout_rows, args.out_heldout, split_meta)
    print("split OK (train and heldout episodes disjoint)")


if __name__ == "__main__":
    main()
