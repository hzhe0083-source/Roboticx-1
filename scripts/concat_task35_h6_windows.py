#!/usr/bin/env python
"""Concat two same-horizon task35 window files without touching elected 1807."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ELECTED = "metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt"


def concat(left: dict, right: dict) -> dict:
    n_left = int(left["actions"].shape[0])
    n_right = int(right["actions"].shape[0])
    if left["actions"].shape[-2:] != right["actions"].shape[-2:]:
        raise ValueError("action chunk shapes differ")
    tensor_keys = [
        key
        for key, value in left.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n_left
    ]
    right_keys = [
        key
        for key, value in right.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n_right
    ]
    if set(tensor_keys) != set(right_keys):
        raise ValueError(f"tensor schema differs: {set(tensor_keys) ^ set(right_keys)}")
    for key in left["normalization"]:
        if not torch.equal(left["normalization"][key], right["normalization"][key]):
            raise ValueError(f"normalization differs on {key}")
    output = {}
    for key in tensor_keys:
        value = right[key]
        if key in {"episode_id", "pair_id"}:
            offset = int(left[key].max().item()) + 1 if left[key].numel() else 0
            value = value + offset
        output[key] = torch.cat((left[key], value), dim=0)
    output["frame_refs"] = [*left["frame_refs"], *right["frame_refs"]]
    output["normalization"] = left["normalization"]
    output["metadata"] = {
        **left["metadata"],
        "n_concat_windows": n_left + n_right,
        "concat_parts": [n_left, n_right],
    }
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.name == ELECTED:
        raise FileExistsError(f"refusing to write elected dataset: {args.output}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    output = concat(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"stale temporary output exists: {tmp}")
    torch.save(output, tmp)
    tmp.replace(args.output)
    print(
        f"concat: {args.output} n={output['actions'].shape[0]} "
        f"parts={output['metadata']['concat_parts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
