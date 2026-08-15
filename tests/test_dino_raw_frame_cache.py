from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from va_compound.longtraj_frames import LongTrajFramesDataset


def test_dino_cache_can_supply_exact_raw_roi_frames_without_jpeg_decode(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "data"
    source.mkdir()
    payload_path = tmp_path / "windows.pt"
    payload = {
        "actions": torch.zeros(1, 4, 6, 4),
        "previous_action": torch.zeros(1, 4, 4),
        "proprio": torch.zeros(1, 4, 4),
        "language_hidden": torch.zeros(1, 3, 8),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
        "instruction_id": torch.tensor([35]),
        "pair_id": torch.tensor([0]),
        "frame_refs": [("task35", 0, [[0, 1, 2, 3]] * 4)],
    }
    torch.save(payload, payload_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    keys = [("task35", 0, index) for index in range(4)]
    with (cache / "index.pkl").open("wb") as stream:
        pickle.dump({key: index for index, key in enumerate(keys)}, stream)
    raw = np.lib.format.open_memmap(
        cache / "raw_frames.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(4, 480, 480, 3),
    )
    for index in range(4):
        raw[index].fill(index * 10)
    raw.flush()
    raw_sha = hashlib.sha256((cache / "raw_frames.npy").read_bytes()).hexdigest()
    (cache / "meta.json").write_text(
        json.dumps(
            {
                "raw_frame_contract": "exact_decoded_longtraj_jpeg_480_v1",
                "raw_frame_shape": [4, 480, 480, 3],
                "raw_frame_dtype": "uint8",
                "raw_frames_sha256": raw_sha,
            }
        )
    )
    dataset = LongTrajFramesDataset(
        payload_path,
        longtraj_dir=source,
        feature_cache=cache,
        include_frames=True,
    )
    monkeypatch.setattr(
        dataset,
        "_decode_task",
        lambda *_: (_ for _ in ()).throw(AssertionError("JPEG decode must be skipped")),
    )
    item = dataset[0]
    assert item["frames"].shape == (4, 4, 480, 480, 3)
    assert item["frames"][:, :, 0, 0, 0].tolist() == [
        [0, 10, 20, 30],
        [0, 10, 20, 30],
        [0, 10, 20, 30],
        [0, 10, 20, 30],
    ]
