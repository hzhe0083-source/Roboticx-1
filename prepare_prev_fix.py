"""Previous-action contract fix (2026-08-07, 数据重建：prev 泄漏修复).

Measured artifact: in the precomputed feature files, ``previous_action`` at
decision t carries a windowing artifact that differs from the truly executed
previous action, and at t=0 it is nonzero although deployment feeds zeros at
episode start (eval_metaworld.py:201, eval_libero_closedloop.py:88).  The
deployment-consistent contract is:

    previous_action[t=0] = 0
    previous_action[t>0] = actions[t-1, -1]   (last executed raw step)

This script writes *new* files (v3) so all prior checkpoints/results remain
reproducible against the untouched originals.

Usage:
  python prepare_prev_fix.py --input data/libero_3scene.pt \
      --output data/libero_3scene_v3.pt
  python prepare_prev_fix.py --input data/metaworld_features_v2_full.pt \
      --output data/metaworld_features_v3_prevfix.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    prev = payload["previous_action"]
    actions = payload["actions"]
    if prev.ndim != 3 or actions.ndim != 4:
        raise ValueError("unexpected shapes; expected previous_action [N,T,D] actions [N,T,H,D]")
    if prev.shape[:2] != actions.shape[:2]:
        raise ValueError("previous_action and actions must share [N,T]")

    fixed = prev.clone()
    fixed[:, 0] = 0.0  # episode start: no previous action (deployment contract)
    fixed[:, 1:] = actions[:, :-1, -1]  # truly executed previous chunk step

    max_shift = float((fixed[:, 1:] - actions[:, :-1, -1]).abs().max().item())
    max_zero = float(fixed[:, 0].abs().max().item())
    payload["previous_action"] = fixed
    payload["metadata"] = dict(payload.get("metadata", {}))
    payload["metadata"]["previous_action_contract"] = "v3_prevfix_20260807"
    torch.save(payload, args.output)
    print(f"saved: {args.output}  (t=0 max |prev| = {max_zero:.2e}; "
          f"t>0 residual vs actions[t-1,-1] = {max_shift:.2e})")


if __name__ == "__main__":
    main()
