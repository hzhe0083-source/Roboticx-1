"""v4 previous-action rebuild (2026-08-07): undo the v3 prev-fix leak.

v3 bug (scripts/migrations/prepare_prev_fix_v3.py): `fixed[:, 1:] = actions[:, :-1, -1]` sets the
previous action of decision t to the LAST step of chunk t-1.  With
CONTROL_STRIDE=6 and ACTION_HORIZON=8 the chunk windows overlap, so
actions[t-1, -1] is the SAME raw time step as actions[t, 1] -- a precise
future-action label leak (verified: equality rate 1.0000 on MW v3 data).
The model could copy the future from `previous_action`, which explains the
unrealistic open-loop MAE (0.0706) vs. the collapsed closed loop (~15-30%).

v2/v1 contract was already correct:
    previous_action[t=0] = 0            (episode start, matches deployment)
    previous_action[t>0] = actions[t-1, stride-1]
        (the last *executed* step of the previous chunk; for MW stride=6 this
         is actions[t-1, 5] -- verified equality 1.0000 on MW v2 / LIBERO v1)

v4 = v2/v1 contract.  t=0 is zeroed for all rows (matches deployment where
every episode starts with a zero previous action; mid-episode sampled
sequences lose only the first-frame value, which the policy cannot rely on
at deployment anyway).

Rebuilds, writing NEW files (originals untouched, prior checkpoints stay
reproducible against their own data versions):
    data/metaworld_features_v4.pt          (from metaworld_features_v2_full.pt, stride=6)
    data/libero_3scene_v4.pt               (from libero_3scene.pt, stride=3)
    data/libero_3scene_paired_v4.pt        (from libero_3scene_paired_v3.pt, stride=3)
    /tmp/smoke_pairs_v4.pt                 (from /tmp/smoke_pairs.pt, stride=6)

Usage:
  python scripts/migrations/prepare_prev_v4.py
  python scripts/migrations/prepare_prev_v4.py --input FILE --output FILE --stride N
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse

import torch

JOBS = [
    ("data/metaworld_features_v2_full.pt", "data/metaworld_features_v4.pt", 6),
    ("data/libero_3scene.pt", "data/libero_3scene_v4.pt", 3),
    ("data/libero_3scene_paired_v3.pt", "data/libero_3scene_paired_v4.pt", 3),
    ("/tmp/smoke_pairs.pt", "/tmp/smoke_pairs_v4.pt", 6),
]


def rebuild(input_path: str, output_path: str, stride: int) -> None:
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    prev = payload["previous_action"]
    actions = payload["actions"]
    if prev.ndim != 3 or actions.ndim != 4:
        raise ValueError(f"{input_path}: expected prev [N,T,D] actions [N,T,H,D], "
                         f"got {prev.shape} / {actions.shape}")
    if prev.shape[:2] != actions.shape[:2]:
        raise ValueError(f"{input_path}: shape mismatch {prev.shape} vs {actions.shape}")

    fixed = prev.clone()
    # Correct contract: previous decision's last EXECUTED step (stride-1).
    fixed[:, 1:] = actions[:, :-1, stride - 1]
    fixed[:, 0] = 0.0  # deployment: episode start has no previous action

    # Sanity: the fixed t>0 rows must NOT equal the future-step copy (t, 1).
    leak = float((fixed[:, 1:] == actions[:, 1:, 1]).float().mean().item())
    correct = float((fixed[:, 1:] == actions[:, :-1, stride - 1]).float().mean().item())
    t0_max = float(fixed[:, 0].abs().max().item())
    assert correct > 0.999, f"{input_path}: contract violated ({correct})"
    assert t0_max == 0.0, f"{input_path}: t=0 not zeroed ({t0_max})"

    payload["previous_action"] = fixed
    payload["metadata"] = dict(payload.get("metadata", {}))
    payload["metadata"]["previous_action_contract"] = "v4_prevfix_20260807"
    torch.save(payload, output_path)
    print(f"saved {output_path}  contract_rate={correct:.4f}  "
          f"future_leak_rate={leak:.4f}  t0_max={t0_max:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()
    if args.input is not None or args.output is not None or args.stride is not None:
        if not (args.input and args.output and args.stride):
            raise SystemExit("--input/--output/--stride must be given together")
        rebuild(args.input, args.output, args.stride)
    else:
        for src, dst, stride in JOBS:
            rebuild(src, dst, stride)
    print("v4 rebuild done")
