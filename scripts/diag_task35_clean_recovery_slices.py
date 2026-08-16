#!/usr/bin/env python
"""CPU diagnostic: compare task35 clean vs recovery geometry from a policy ckpt.

Uses the frozen DINO feature cache and the jointly trained metric head. This is
mechanism evidence for the pegHead-hole route, not closed-loop insertion success.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metaworld import _load_dino_metric_from_policy
from scripts.peek_task35_checkpoint_step import peek_task35_checkpoint_step
from scripts.validate_task35_fm_checkpoint import (
    EXPECTED_DATA_SHA256,
    EXPECTED_FEATURE_SHA256,
    EXPECTED_RAW_FRAMES_SHA256,
    sha256_file,
)
from va_compound.longtraj_frames import LongTrajFramesDataset
from va_compound.model import VACompoundConfig, dense_coords

ROLE_NAMES = ("tool", "pegGrasp", "hole", "pegHead")
PEGHEAD, HOLE = 3, 2


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def assemble_last_decision_dense(
    block11: np.ndarray,
    block23: np.ndarray,
    cache_rows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble [d-2, d] evidence the same way training does: 2×256 → 512 tokens."""
    if cache_rows.ndim != 2 or cache_rows.shape[1] != 2:
        raise ValueError(f"cache_rows must be [B,2] last-decision pair, got {cache_rows.shape}")
    batch = int(cache_rows.shape[0])
    flat = cache_rows.reshape(-1)
    mid = np.asarray(block11[flat]).reshape(batch, 2, 256, 1024)
    last = np.asarray(block23[flat]).reshape(batch, 2, 256, 1024)
    dense5 = torch.from_numpy(np.concatenate((mid[:, 0], mid[:, 1]), axis=1)).float()
    dense11 = torch.from_numpy(np.concatenate((last[:, 0], last[:, 1]), axis=1)).float()
    if dense5.shape != (batch, 512, 1024) or dense11.shape != (batch, 512, 1024):
        raise ValueError(f"dense evidence shape {tuple(dense5.shape)} / {tuple(dense11.shape)}")
    return dense5, dense11


def load_cache_blocks(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((cache_dir / "meta.json").read_text())
    if meta.get("dataset_sha256") != EXPECTED_DATA_SHA256:
        raise ValueError("cache dataset_sha256 does not match the task35 payload")
    if meta.get("raw_frames_sha256") != EXPECTED_RAW_FRAMES_SHA256:
        raise ValueError("cache raw-frame sha does not match the task35 contract")
    if meta.get("feature_sha256") != EXPECTED_FEATURE_SHA256:
        raise ValueError("cache feature sha metadata does not match the bound hashes")
    block11 = np.load(cache_dir / "block11.npy", mmap_mode="r")
    block23 = np.load(cache_dir / "block23.npy", mmap_mode="r")
    if block11.shape != block23.shape:
        raise ValueError("block11/block23 shape mismatch")
    return block11, block23


def slice_geometry(
    p: np.ndarray,
    visibility: np.ndarray,
    layers: np.ndarray,
) -> dict[str, dict]:
    """Aggregate last-decision geometry by clean/recovery layer."""
    if p.ndim != 3 or p.shape[-2:] != (4, 2):
        raise ValueError(f"p must be [N,4,2], got {p.shape}")
    if visibility.shape != p.shape[:2]:
        raise ValueError("visibility shape must match p[:, :, 0]")
    out: dict[str, dict] = {}
    for name, mask in (
        ("clean", layers == 0),
        ("recovery", layers == 1),
        ("all", np.ones(len(layers), dtype=bool)),
    ):
        selected = p[mask]
        vis = visibility[mask]
        if selected.size == 0:
            out[name] = {"n": 0}
            continue
        pair_visible = (vis[:, PEGHEAD] >= 0.5) & (vis[:, HOLE] >= 0.5)
        distance = np.linalg.norm(selected[:, PEGHEAD] - selected[:, HOLE], axis=-1) * 480.0
        visible_points = selected[pair_visible]
        role_std = {}
        for index, role in enumerate(ROLE_NAMES):
            pts = visible_points[:, index] * 480.0 if visible_points.size else np.empty((0, 2))
            role_std[role] = {
                "n": int(pts.shape[0]),
                "std_x_px": None if pts.size == 0 else float(pts[:, 0].std()),
                "std_y_px": None if pts.size == 0 else float(pts[:, 1].std()),
            }
        out[name] = {
            "n": int(mask.sum()),
            "pair_visible_fraction": float(pair_visible.mean()),
            "pegHead_hole_px": summarize(distance[pair_visible]),
            "role_std_px": role_std,
            "visibility_mean": {
                role: float(vis[:, index].mean()) for index, role in enumerate(ROLE_NAMES)
            },
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt"),
    )
    parser.add_argument(
        "--dino-feature-cache",
        type=Path,
        default=Path("data/dino35_h6_clean60_recovery30_cache_v1"),
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    found_step = peek_task35_checkpoint_step(args.checkpoint)
    if args.expected_step is not None and found_step != int(args.expected_step):
        raise ValueError(
            f"slice checkpoint global_step={found_step} != expected {args.expected_step}"
        )
    digest = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    loaded_step = int(checkpoint["global_step"])
    if loaded_step != found_step:
        raise ValueError(
            f"pickle global_step={loaded_step} != peeked {found_step}"
        )
    if args.expected_step is not None and loaded_step != int(args.expected_step):
        raise ValueError(
            f"loaded global_step={loaded_step} != expected {args.expected_step}"
        )
    config = VACompoundConfig(**checkpoint["config"])
    metric_head, _ = _load_dino_metric_from_policy(checkpoint, config, device)
    dataset = LongTrajFramesDataset(
        args.features,
        min_sequence_length=4,
        include_frames=False,
        feature_cache=args.dino_feature_cache,
    )
    payload = dataset.payload
    layers = payload["data_layer"].cpu().numpy()
    if dataset.cache_rows is None:
        raise RuntimeError("feature cache rows were not resolved")
    block11, block23 = load_cache_blocks(args.dino_feature_cache)
    coords = dense_coords(512, device=device)
    positions = []
    visibilities = []
    for start in range(0, len(dataset), args.batch):
        end = min(start + args.batch, len(dataset))
        rows = dataset.cache_rows[start:end, -1, 2:4]
        dense5, dense11 = assemble_last_decision_dense(block11, block23, rows)
        language = payload["language_hidden"][start:end].float()
        mask = payload["language_mask"][start:end]
        with torch.inference_mode():
            out = metric_head(dense5, dense11, language, mask, coords)
        positions.append(out.p.cpu().numpy())
        visibilities.append(torch.sigmoid(out.visibility_logits).cpu().numpy())
    report = {
        "contract": "task35_clean_recovery_slice_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": loaded_step,
        "sha256": digest,
        "n_windows": int(len(dataset)),
        "n_clean": int((layers == 0).sum()),
        "n_recovery": int((layers == 1).sum()),
        "role_order": list(ROLE_NAMES),
        "precision_pair": ["pegHead", "hole"],
        "slices": slice_geometry(
            np.concatenate(positions, axis=0),
            np.concatenate(visibilities, axis=0),
            layers,
        ),
        "note": "mechanism diagnostic on cached training windows; not closed-loop success",
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
