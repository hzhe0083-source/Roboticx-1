#!/usr/bin/env python
"""Strictly merge task35 clean/recovery H6 windows for matched VA training."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(clean: dict, recovery: dict, clean_path: Path, recovery_path: Path) -> dict:
    if clean["actions"].shape[-2:] != (6, 4) or recovery["actions"].shape[-2:] != (6, 4):
        raise ValueError("both task35 layers must have exact H6 action chunks")
    if clean["metadata"].get("action_horizon") != 6 or recovery["metadata"].get("action_horizon") != 6:
        raise ValueError("both metadata contracts must declare action_horizon=6")
    if bool(clean["recovery_mask"].any()) or bool(clean["decision_recovery"].any()):
        raise ValueError("clean windows contain recovery labels")
    if not bool((recovery["recovery_mask"] & recovery["action_valid_mask"]).any()):
        raise ValueError("recovery windows contain no valid recovery targets")
    if clean["normalization"].keys() != recovery["normalization"].keys():
        raise ValueError("normalization key sets differ")
    for key in clean["normalization"]:
        if not torch.equal(clean["normalization"][key], recovery["normalization"][key]):
            raise ValueError(f"normalization differs on {key}")

    n_clean = int(clean["actions"].shape[0])
    n_recovery = int(recovery["actions"].shape[0])
    tensor_keys = [
        key
        for key, value in clean.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n_clean
    ]
    if set(tensor_keys) != {
        key
        for key, value in recovery.items()
        if isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] == n_recovery
    }:
        raise ValueError("clean/recovery per-row tensor schemas differ")

    output = {}
    for key in tensor_keys:
        left = clean[key]
        right = recovery[key]
        if key in {"episode_id", "pair_id"}:
            offset = int(left.max().item()) + 1 if left.numel() else 0
            right = right + offset
        output[key] = torch.cat((left, right), dim=0)
    output["frame_refs"] = [*clean["frame_refs"], *recovery["frame_refs"]]
    output["normalization"] = clean["normalization"]
    output["data_layer"] = torch.cat(
        (
            torch.zeros(n_clean, dtype=torch.uint8),
            torch.ones(n_recovery, dtype=torch.uint8),
        )
    )
    output["metadata"] = {
        **clean["metadata"],
        "contract": "task35_clean_recovery_h6_matched_va_v1",
        "action_horizon": 6,
        "task": "peg-insert-side-v3",
        "task_role_order": ["tool", "pegGrasp", "hole", "pegHead"],
        "roi_relation_pair": ["pegHead", "hole"],
        "n_clean_windows": n_clean,
        "n_recovery_windows": n_recovery,
        "layer_encoding": "data_layer: 0=clean, 1=recovery",
        "source_files": [str(clean_path.resolve()), str(recovery_path.resolve())],
        "source_sha256": [sha256_file(clean_path), sha256_file(recovery_path)],
        "matched_va_contract": (
            "deterministic and FM consume this exact payload/cache/seed/update budget"
        ),
    }
    if output["actions"].shape != (n_clean + n_recovery, 4, 6, 4):
        raise ValueError(f"unexpected merged actions shape {tuple(output['actions'].shape)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean_path = args.clean.expanduser().resolve(strict=True)
    recovery_path = args.recovery.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().absolute()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    clean = torch.load(clean_path, map_location="cpu", weights_only=True)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
    output = merge(clean, recovery, clean_path, recovery_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    torch.save(output, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=True)
    if verified["metadata"]["contract"] != "task35_clean_recovery_h6_matched_va_v1":
        raise ValueError("merged payload verification failed")
    temporary.replace(output_path)
    print(
        f"merged: {output_path} shape={tuple(output['actions'].shape)} "
        f"sha256={sha256_file(output_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
