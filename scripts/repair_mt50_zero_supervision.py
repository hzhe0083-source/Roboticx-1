"""Create repaired MT50 raw containers without mutating the original 60ep files."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPAIR_CONTRACT = "mt50_zero_supervision_episode_replacement_v1"
RAW_CONTRACT = "mt50_canonical_raw_sources_online_repaired_v3"
REPLACEMENTS = {
    "coffee-button-v3": (58, 59),
    "faucet-open-v3": (44, 54, 56),
    "faucet-close-v3": (51, 52, 57, 59),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _has_supervision(episode: dict) -> bool:
    valid = np.asarray(episode.get("action_supervision_valid", []), dtype=bool)
    return valid.shape == (len(episode.get("actions", [])),) and bool(valid.any())


def repair_task(
    task: str,
    source_path: str,
    replacement_path: str,
    output_path: str,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve(strict=True)
    replacement = Path(replacement_path).expanduser().resolve(strict=True)
    output = Path(output_path)
    indices = REPLACEMENTS[task]
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if (
            payload.get("task") != task
            or len(payload.get("episodes") or []) != 60
            or (payload.get("metadata") or {}).get("online_repair_contract")
            != REPAIR_CONTRACT
        ):
            raise ValueError(f"invalid existing repaired file: {output}")
        return identity(output)

    base = torch.load(source, map_location="cpu", weights_only=False)
    patch = torch.load(replacement, map_location="cpu", weights_only=False)
    episodes = list(base.get("episodes") or [])
    replacements = list(patch.get("episodes") or [])
    if base.get("task") != task or len(episodes) != 60:
        raise ValueError(f"{source}: expected 60 {task} episodes")
    if patch.get("task") != task or len(replacements) != len(indices):
        raise ValueError(f"{replacement}: replacement count/task mismatch")
    if any(_has_supervision(episodes[index]) for index in indices):
        raise ValueError(f"{source}: elected replacement indices are no longer empty")
    if any(not _has_supervision(episode) for episode in replacements):
        raise ValueError(f"{replacement}: replacement still has zero supervision")
    existing_seeds = {
        int(episode["episode_seed"])
        for episode in episodes
        if episode.get("episode_seed") is not None
    }
    new_seeds = [int(episode["episode_seed"]) for episode in replacements]
    if len(set(new_seeds)) != len(new_seeds) or existing_seeds & set(new_seeds):
        raise ValueError(f"{replacement}: duplicate episode seeds")
    for index, episode in zip(indices, replacements, strict=True):
        episodes[index] = episode

    repaired = {key: value for key, value in base.items() if key != "episodes"}
    repaired["episodes"] = episodes
    repaired["n_episodes"] = 60
    metadata = dict(repaired.get("metadata") or {})
    metadata.update(
        {
            "online_repair_contract": REPAIR_CONTRACT,
            "online_repair_indices": list(indices),
            "online_repair_source": identity(replacement),
            "online_repair_original": identity(source),
        }
    )
    repaired["metadata"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale repair temporary: {temporary}")
    torch.save(repaired, temporary)
    os.replace(temporary, output)
    return identity(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--replacement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    raw_path = args.raw_manifest.expanduser().resolve(strict=True)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    sources = list(raw.get("sources") or [])
    by_task = {str(item["task"]): item for item in sources}
    if len(sources) != 50 or not set(REPLACEMENTS).issubset(by_task):
        raise ValueError("input raw manifest is not the elected MT50 source")

    jobs = []
    for task in REPLACEMENTS:
        jobs.append(
            (
                task,
                str(by_task[task]["source_path"]),
                str(
                    args.replacement_dir
                    / f"metaworld_longtraj_{task}_online_repair_v1.pt"
                ),
                str(args.output_dir / f"metaworld_longtraj_{task}.pt"),
            )
        )
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(repair_task, *job) for job in jobs]
        repaired_identities = [future.result() for future in futures]
    repaired_by_task = dict(zip(REPLACEMENTS, repaired_identities, strict=True))

    entries = []
    for item in sources:
        task = str(item["task"])
        if task in repaired_by_task:
            repaired_identity = repaired_by_task[task]
            entries.append(
                {
                    **item,
                    "source_path": repaired_identity["path"],
                    "canonical_path": repaired_identity["path"],
                    "sha256": repaired_identity["sha256"],
                    "size_bytes": repaired_identity["size_bytes"],
                    "repair_contract": REPAIR_CONTRACT,
                }
            )
        else:
            entries.append(item)
    manifest = {
        **raw,
        "contract": RAW_CONTRACT,
        "parent_raw_manifest": identity(raw_path),
        "online_repair_contract": REPAIR_CONTRACT,
        "sources": entries,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.output_manifest.exists():
        if args.output_manifest.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("existing repaired raw manifest differs")
    else:
        temporary = args.output_manifest.with_name(f".{args.output_manifest.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.output_manifest)
    print(
        f"[ok] {REPAIR_CONTRACT}: replaced={sum(map(len, REPLACEMENTS.values()))} "
        f"source_episodes=3420 manifest={args.output_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
