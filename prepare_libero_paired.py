"""Build a paired LIBERO dataset with a cosine-gated same-state fork contract.

Data facts (2026-08-07 measured on data/libero_3scene.pt, 12 tasks × 30
episodes): within a scene, cross-instruction first states are *similar but
not identical* — feature-space cosine 0.993–0.997, proprio max-diff
0.03–0.15, previous_action exactly 0. Strict per-token equality does not
exist because the target object configuration is part of the task
definition.  This builder therefore pairs the k-th episode of every
instruction in a scene (same scene layout, near-identical robot pose) and
gates each group on:

  - first-state vision cosine >= --cosine       (default 0.99)
  - first-state proprio max-diff <= --proprio-atol (default 0.15)
  - first-state previous_action diff <= --prev-atol (default 1e-3)
  - decision-0 action mean-abs-delta >= --min-action-delta

Groups failing the gate are dropped; the report lists min cosine / max
proprio diff per surviving group so the paper can cite the exact residual
visual difference (documented confound, quantified).

Usage:
  python prepare_libero_paired.py --input data/libero_3scene.pt \
      --output data/libero_3scene_paired.pt --cosine 0.99
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default="data/libero_3scene.pt")
    p.add_argument("--output", type=Path, default="data/libero_3scene_paired.pt")
    p.add_argument("--cosine", type=float, default=0.99)
    p.add_argument("--proprio-atol", type=float, default=0.15)
    p.add_argument("--prev-atol", type=float, default=1e-3)
    p.add_argument("--min-action-delta", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)

    instr = payload["instruction_id"]
    n_tasks = int(instr.max()) + 1
    n = len(instr)
    if n % n_tasks:
        raise ValueError(f"{n} samples not divisible by {n_tasks} instructions")

    tasks = payload["metadata"]["tasks"]
    # Scene-restricted pairing: only instructions sharing a scene can fork
    # from near-identical states (cross-scene first frames differ, cos~0.91).
    scene_groups: dict[str, list[int]] = {}
    for k, t in enumerate(tasks):
        scene = "UNKNOWN"
        for key in ("LIVING_ROOM", "KITCHEN", "STUDY"):
            if key in t:
                scene = key
                break
        scene_groups.setdefault(scene, []).append(k)
    indices_all = [
        torch.nonzero(instr == k, as_tuple=False).flatten().tolist() for k in range(n_tasks)
    ]
    n_rows = len(indices_all[0])
    assert all(len(rows) == n_rows for rows in indices_all), "uneven episodes per instruction"

    vis = payload["vision_tokens"][:, 0].flatten(1).float()
    vis = vis / vis.norm(dim=-1, keepdim=True)
    prop = payload["proprio"][:, 0]
    prev = payload["previous_action"][:, 0]
    actions = payload["actions"]

    new_pair = torch.full((n,), -1, dtype=torch.long)
    group = 0
    kept_rows = 0
    report = []
    for scene, scene_ids in scene_groups.items():
        indices = [indices_all[k] for k in scene_ids]
        for row in zip(*indices):
            row = list(row)
            ref = row[0]
            min_cos = 1.0
            max_prop = 0.0
            max_prev = 0.0
            ok = True
            for other in row[1:]:
                c = float((vis[ref] * vis[other]).sum().item())
                dp = float((prop[ref] - prop[other]).abs().max().item())
                dv = float((prev[ref] - prev[other]).abs().max().item())
                min_cos = min(min_cos, c)
                max_prop = max(max_prop, dp)
                max_prev = max(max_prev, dv)
                if c < args.cosine or dp > args.proprio_atol or dv > args.prev_atol:
                    ok = False
            if ok:
                # Action difference must be identifiable within every instruction pair.
                for a in range(len(scene_ids)):
                    for b in range(a + 1, len(scene_ids)):
                        ad = float(
                            (actions[row[a], 0] - actions[row[b], 0]).abs().mean().item()
                        )
                        if ad < args.min_action_delta:
                            ok = False
                            break
                    if not ok:
                        break
            if not ok:
                continue
            for idx in row:
                new_pair[idx] = group
            group += 1
            kept_rows += 1
            report.append((scene, kept_rows, min_cos, max_prop, max_prev))

    n_kept = kept_rows * 4
    print(f"episodes per instruction: {n_rows}; kept rows: {kept_rows}/{n_rows} "
          f"(samples: {n_kept}/{n})")
    print("surviving group stats (scene, row, min cosine, max proprio diff, max prev diff):")
    for row in report:
        print(f"  {row[0]:12s} row {row[1]:3d}: cos>={row[2]:.4f} proprio<={row[3]:.4f} prev<={row[4]:.4f}")
    if kept_rows == 0:
        raise SystemExit("no groups survived the gate; lower the thresholds")

    payload["pair_id"] = new_pair
    # Drop unpaired rows entirely (keeps the contract validator clean).
    keep = new_pair >= 0
    kept: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == n:
            kept[key] = value[keep]
        else:
            kept[key] = value
    kept["metadata"] = dict(kept["metadata"])
    kept["metadata"]["pair_contract"] = {
        "cosine": args.cosine,
        "proprio_atol": args.proprio_atol,
        "prev_atol": args.prev_atol,
        "min_action_delta": args.min_action_delta,
    }
    out = Path(args.output)
    torch.save(kept, out)
    print(f"saved: {out} ({n_kept} samples, {kept_rows} groups)")


if __name__ == "__main__":
    main()
