#!/usr/bin/env python
"""Add exact decoded 480px uint8 frames to an existing DINO feature cache.

This avoids repeatedly decoding entire clean/recovery JPEG containers when the
policy uses train-time ROI refinement.  Cache row identity is reused from
``index.pkl``; no vision features are recomputed or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from va_compound.longtraj_frames import LongTrajFramesDataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.expanduser().resolve(strict=True)
    cache = args.cache.expanduser().resolve(strict=True)
    destination = cache / "raw_frames.npy"
    meta_path = cache / "meta.json"
    meta = json.loads(meta_path.read_text())
    data_sha = sha256_file(data)
    if meta.get("dataset_sha256") != data_sha:
        raise ValueError(
            "existing DINO cache was built from a different dataset payload: "
            f"cache={meta.get('dataset_sha256')!r}, data={data_sha!r}"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    with (cache / "index.pkl").open("rb") as stream:
        index: dict[tuple[str, int, int], int] = pickle.load(stream)
    rows = [None] * len(index)
    for key, row in index.items():
        if row < 0 or row >= len(rows) or rows[row] is not None:
            raise ValueError("DINO cache index rows must be a dense unique 0..N-1 range")
        rows[row] = key
    dataset = LongTrajFramesDataset(data, include_frames=False)
    expected = {
        (task_file, int(ep_idx), int(frame))
        for task_file, ep_idx, windows in dataset.refs
        for window in windows
        for frame in window
    }
    if expected != set(index):
        raise ValueError(
            "DINO cache index does not exactly match dataset frame references: "
            f"missing={len(expected - set(index))}, extra={len(set(index) - expected)}"
        )

    temporary = cache / ".raw_frames.npy.tmp"
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), 480, 480, 3),
    )
    current_task = None
    decoded = None
    for row, (task_file, episode, frame) in enumerate(rows):
        if task_file != current_task:
            decoded = dataset._decode_task(task_file)
            current_task = task_file
        value = decoded[episode][frame]
        if value.shape != (480, 480, 3) or value.dtype != np.uint8:
            raise ValueError(
                f"decoded frame {rows[row]} must be uint8 [480,480,3], "
                f"got {value.shape}/{value.dtype}"
            )
        output[row] = value
        if (row + 1) % 1000 == 0 or row + 1 == len(rows):
            output.flush()
            print(f"cached raw frames {row + 1}/{len(rows)}", flush=True)
    output.flush()
    del output
    temporary.replace(destination)

    # Reopen and compare boundary rows to the source after the atomic rename.
    verified = np.load(destination, mmap_mode="r")
    for row in sorted({0, len(rows) // 2, len(rows) - 1}):
        task_file, episode, frame = rows[row]
        source = dataset._decode_task(task_file)[episode][frame]
        if not np.array_equal(verified[row], source):
            raise ValueError(f"raw frame cache verification failed on row {row}")

    meta.update(
        {
            "raw_frames": True,
            "raw_frame_shape": [len(rows), 480, 480, 3],
            "raw_frame_dtype": "uint8",
            "raw_frame_contract": "exact_decoded_longtraj_jpeg_480_v1",
            "raw_frames_sha256": sha256_file(destination),
        }
    )
    meta_tmp = cache / ".meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=1) + "\n")
    meta_tmp.replace(meta_path)
    print(
        f"raw frame cache written: {destination} "
        f"sha256={meta['raw_frames_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
