#!/usr/bin/env python
"""Atomically add the finalized perturbation schema to existing task35 v2 data.

This migration is intentionally metadata-only: JPEG frames, actions, states,
validity masks, episode ordering, seeds, and normalization are preserved.  It
accepts only already-pure clean/recovery files and writes a new file unless
``--in-place`` is explicitly requested.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate(payload: dict, *, layer: str, source: Path) -> dict:
    if layer not in {"clean", "recovery"}:
        raise ValueError(f"unknown layer {layer!r}")
    if payload.get("task") != "peg-insert-side-v3":
        raise ValueError(f"task35 migration received task={payload.get('task')!r}")
    metadata = payload.get("metadata")
    episodes = payload.get("episodes")
    if not isinstance(metadata, dict) or not isinstance(episodes, list) or not episodes:
        raise ValueError("payload lacks non-empty episodes/metadata")
    if metadata.get("contract") != "long_trajectory_scripted_v2":
        raise ValueError(f"unsupported contract={metadata.get('contract')!r}")

    expected_event = layer == "recovery"
    for index, episode in enumerate(episodes):
        event = episode.get("perturb_event")
        starts = episode.get("perturb_start") is not None
        perturbed = bool(episode.get("perturbed", False))
        observed = int(event is not None or starts or perturbed)
        if observed != int(expected_event):
            raise ValueError(
                f"episode[{index}] violates {layer} purity: perturbed={perturbed}, "
                f"event={event is not None}, start={starts}"
            )
        recovery = np.asarray(episode.get("recovery_mask", []), dtype=bool)
        if expected_event and recovery.sum() == 0:
            raise ValueError(f"episode[{index}] recovery layer has zero recovery actions")
        if not expected_event and recovery.sum() != 0:
            raise ValueError(f"episode[{index}] clean layer has recovery actions")
        episode["n_perturb_events"] = observed

    metadata["perturbation_data_present"] = expected_event
    metadata["episode_perturbation_contract"] = (
        "clean: n_perturb_events=0 for every episode; "
        "recovery: n_perturb_events>=1 for every episode"
    )
    metadata["schema_migration"] = {
        "contract": "task35_longtraj_schema_metadata_only_v1",
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "preserved_fields": (
            "frames/actions/states/masks/seeds/order/normalization byte objects "
            "loaded unchanged; only n_perturb_events and metadata fields added"
        ),
    }
    payload["n_episodes"] = len(episodes)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layer", choices=("clean", "recovery"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    if args.in_place == (args.output is not None):
        raise ValueError("choose exactly one of --output or --in-place")
    source = args.input.expanduser().resolve(strict=True)
    destination = source if args.in_place else args.output.expanduser().absolute()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload = migrate(payload, layer=args.layer, source=source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    torch.save(payload, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    migrate(verified, layer=args.layer, source=source)
    temporary.replace(destination)
    print(
        f"migrated {args.layer}: {source} -> {destination} "
        f"episodes={len(payload['episodes'])} sha256={sha256_file(destination)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
