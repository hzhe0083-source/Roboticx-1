#!/usr/bin/env python
"""Keep longtraj windows whose frame_refs fit the local JPEG episode files."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def episode_lengths(path: Path) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [len(ep["frames"]) for ep in payload["episodes"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--longtraj-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    windows = torch.load(args.input, map_location="cpu", weights_only=True)
    refs = windows["frame_refs"]
    ids = windows["instruction_id"]
    lens: dict[str, list[int]] = {}
    keep: list[int] = []
    dropped: dict[str, int] = {}
    for index, ref in enumerate(refs):
        task, ep_idx, frames = ref
        if task not in lens:
            path = args.longtraj_dir / f"metaworld_longtraj_{task}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            lens[task] = episode_lengths(path)
        ep = int(ep_idx)
        max_frame = max(int(step) for row in frames for step in row)
        ok = 0 <= ep < len(lens[task]) and max_frame < lens[task][ep]
        if ok:
            keep.append(index)
        else:
            dropped[task] = dropped.get(task, 0) + 1
    if not keep:
        raise SystemExit("no windows fit the local frame files")
    out = {}
    n_src = int(ids.shape[0])
    for key, value in windows.items():
        if key in ("normalization", "metadata"):
            out[key] = value
            continue
        if isinstance(value, torch.Tensor) and value.shape[0] == n_src:
            out[key] = value[keep].clone()
        elif key == "frame_refs":
            out[key] = [refs[i] for i in keep]
        else:
            out[key] = value
    metadata = dict(windows.get("metadata") or {})
    metadata["frame_fit_source"] = str(args.input.resolve())
    metadata["n_windows_before_frame_fit"] = n_src
    metadata["n_windows_dropped_frame_fit"] = dict(dropped)
    metadata["n_subset_windows"] = len(keep)
    out["metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(
        f"wrote {args.output}: kept {len(keep)}/{n_src}, dropped {dropped}"
    )


if __name__ == "__main__":
    main()
