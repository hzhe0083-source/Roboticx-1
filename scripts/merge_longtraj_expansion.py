"""Merge scripted-expert expansion shards into per-task longtraj frame files.

Why this exists: under the peer_sync_h6 contract, build_longtraj_features stores
a window's frame pointer as ``(env_name, episode_index_within_file)``.  Feeding
expansion shards to phase 1 as separate ``--input`` files therefore collapses
every shard onto the canonical ``metaworld_longtraj_{env}.pt`` key while each
shard restarts ``episode_index`` at zero, so windows silently decode another
episode's frames.  Merging the shards into one file per task restores a unique
index per episode and keeps the contract untouched.

Base episodes keep their original indices (they are appended to first), so
datasets built against the base file alone stay valid against the merged file.

Usage:
  python scripts/merge_longtraj_expansion.py \
      --base data/metaworld_longtraj_assembly-v3.pt \
      --base data/metaworld_longtraj_door-unlock-v3.pt \
      --shard-dir data/expand --shard-dir data/expand_clean \
      --out-dir data/frames_v2
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import torch

MERGE_CONTRACT = "longtraj_expansion_merge_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(path: Path) -> list:
    """Order shard0 < shard2 < shard10 instead of lexicographically."""
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("task", "episodes"):
        if key not in payload:
            raise ValueError(f"{path}: not a longtraj task payload (missing {key})")
    return payload


def merge_task(base_path: Path, shard_dirs: list[Path], out_dir: Path,
               *, overwrite: bool = False) -> Path:
    base = _load(base_path)
    task = base["task"]
    out_path = out_dir / f"metaworld_longtraj_{task}.pt"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {out_path}; pass --overwrite")

    shards: list[Path] = []
    for index, directory in enumerate(shard_dirs):
        if not directory.is_dir():
            raise NotADirectoryError(f"missing shard dir: {directory}")
        found = sorted(directory.glob("metaworld_longtraj_*.pt"), key=_natural_key)
        shards.extend(path for path in found)
        print(f"  shard dir {directory}: {len(found)} files")

    episodes = list(base["episodes"])
    provenance = [{
        "path": str(base_path.resolve()),
        "sha256": _sha256(base_path),
        "size_bytes": base_path.stat().st_size,
        "role": "base",
        "episode_range": [0, len(episodes)],
    }]
    for path in shards:
        payload = _load(path)
        if payload["task"] != task:
            continue
        start = len(episodes)
        episodes.extend(payload["episodes"])
        provenance.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "role": "expansion",
            "episode_range": [start, len(episodes)],
        })
        print(f"  + {path.name}: {len(payload['episodes'])} episodes "
              f"-> indices [{start}, {len(episodes)})")
        del payload

    merged = {key: value for key, value in base.items() if key != "episodes"}
    merged["episodes"] = episodes
    merged["n_episodes"] = len(episodes)
    metadata = dict(merged.get("metadata") or {})
    metadata["merge_contract"] = MERGE_CONTRACT
    metadata["merge_sources"] = provenance
    merged["metadata"] = metadata

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(merged, tmp_path)
    tmp_path.replace(out_path)
    print(f"[out] {out_path}: {len(episodes)} episodes "
          f"({len(provenance) - 1} expansion files merged)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="append", required=True, type=Path,
                        help="canonical per-task longtraj file (repeatable)")
    parser.add_argument("--shard-dir", action="append", default=[], type=Path,
                        help="directory of expansion shards (repeatable, ordered)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for base_path in args.base:
        print(f"merging {base_path.name}")
        merge_task(base_path, args.shard_dir, args.out_dir,
                   overwrite=args.overwrite)


if __name__ == "__main__":
    main()
