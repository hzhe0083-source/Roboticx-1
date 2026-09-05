"""Build a portable online index containing only persisted DAgger episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.build_online_episode_index import file_identity
from scripts.build_longtraj_features import resolve_episode_semantics


OLD_ROOT = Path("/root/ora0_all49_expand60_v1")


def _atomic_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _padded_valid_start_count(episode: dict, source: str) -> int:
    semantics = resolve_episode_semantics(episode, source, legacy_policy="infer")
    length = len(episode["actions"])
    count = 0
    for start in range(length):
        decisions = start + np.arange(4) * 15
        targets = decisions[:, None] + np.arange(15)[None, :]
        in_bounds = targets < length
        safe_targets = np.minimum(targets, length - 1)
        valid = semantics["valid"][safe_targets].copy() & in_bounds
        perturb_start = semantics["perturb_start"]
        if perturb_start is not None:
            valid &= ~(
                semantics["recovery"][safe_targets]
                & (decisions[:, None] < int(perturb_start))
                & in_bounds
            )
        count += int(bool(valid.any()))
    return count


def build_recovery_only_index(
    source_index: Path,
    persisted_root: Path,
    sha256sums: Path,
    output: Path,
    *,
    expected_episodes: int = 970,
    base_index: Path | None = None,
) -> dict:
    source_index = source_index.expanduser().resolve(strict=True)
    persisted_root = persisted_root.expanduser().resolve(strict=True)
    sha256sums = sha256sums.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    manifest_path = output.with_name(output.stem + "_manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite recovery-only index artifacts")

    source = json.loads(source_index.read_text(encoding="utf-8"))
    episodes = [dict(row) for row in source.get("episodes") or [] if row.get("source_path")]
    if len(episodes) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} DAgger episodes, got {len(episodes)}")
    if len({int(row["episode_id"]) for row in episodes}) != len(episodes):
        raise ValueError("DAgger episode ids are not unique")

    recorded = {}
    for line in sha256sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        recorded[relative.strip()] = digest

    def remap(raw: str) -> Path:
        path = Path(raw)
        try:
            relative = path.relative_to(OLD_ROOT)
        except ValueError as error:
            raise ValueError(f"unexpected DAgger source root: {path}") from error
        mapped = persisted_root / relative
        if not mapped.is_file():
            raise FileNotFoundError(mapped)
        return mapped

    declared = {str(item["path"]): item for item in source.get("additional_sources") or []}
    identities = {}
    for row in episodes:
        old_path = str(row["source_path"])
        mapped = remap(old_path)
        relative = str(mapped.relative_to(persisted_root))
        identity = declared.get(old_path)
        if identity is None or recorded.get(relative) != identity.get("sha256"):
            raise ValueError(f"source identity mismatch: {mapped}")
        if mapped.stat().st_size != int(identity["size_bytes"]):
            raise ValueError(f"source size mismatch: {mapped}")
        row["source_path"] = str(mapped)
        identities[str(mapped)] = {
            "path": str(mapped),
            "sha256": str(identity["sha256"]),
            "size_bytes": int(identity["size_bytes"]),
        }

    task_counts = Counter(int(row["task_id"]) for row in episodes)
    tasks = []
    for item in source.get("tasks") or []:
        task_id = int(item["task_id"])
        candidates = [row for row in episodes if int(row["task_id"]) == task_id]
        if not candidates:
            raise ValueError(f"DAgger data misses task {task_id}")
        identity = identities[str(candidates[0]["source_path"])]
        tasks.append(
            {
                **item,
                "source_path": identity["path"],
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
                "source_episodes": task_counts[task_id],
                "train_episodes": task_counts[task_id],
                "eval_episodes": 0,
            }
        )
    if [int(item["task_id"]) for item in tasks] != list(range(50)):
        raise ValueError("recovery-only index must cover task ids 0..49")

    base = None
    if base_index is not None:
        base_index = base_index.expanduser().resolve(strict=True)
        base = json.loads(base_index.read_text(encoding="utf-8"))
        if base.get("contract") != source.get("contract"):
            raise ValueError("base and DAgger indexes use different contracts")
        base_protocol = dict(base.get("sampling_protocol") or {})
        source_protocol = dict(source.get("sampling_protocol") or {})
        base_protocol.pop("short_episode_padding", None)
        source_protocol.pop("short_episode_padding", None)
        if base_protocol != source_protocol:
            raise ValueError("base and DAgger sampling protocols differ")
        if [item["description"] for item in base.get("tasks") or []] != [
            item["description"] for item in tasks
        ]:
            raise ValueError("base and DAgger task descriptions differ")
        base_ids = {int(row["episode_id"]) for row in base.get("episodes") or []}
        recovery_ids = {int(row["episode_id"]) for row in episodes}
        if base_ids & recovery_ids:
            raise ValueError("base and DAgger episode ids overlap")
        if (base.get("sampling_protocol") or {}).get("short_episode_padding"):
            rows_by_source = defaultdict(list)
            for row in episodes:
                rows_by_source[str(row["source_path"])].append(row)
            for path, rows in rows_by_source.items():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                for row in rows:
                    episode_index = int(row["episode_index"])
                    episode = payload["episodes"][episode_index]
                    row["valid_start_count"] = _padded_valid_start_count(
                        episode, f"{row['task']}:episode[{episode_index}]"
                    )

    manifest = {
        "contract": "dagger_recovery_only_sources_v1",
        "source_index": file_identity(source_index),
        "base_index": None if base_index is None else file_identity(base_index),
        "sha256sums": file_identity(sha256sums),
        "sources": sorted(identities.values(), key=lambda item: item["path"]),
        "episodes": len(episodes),
        "tasks": len(tasks),
    }
    _atomic_json(manifest, manifest_path)
    combined_episodes = [
        *([] if base is None else [dict(row) for row in base.get("episodes") or []]),
        *episodes,
    ]
    result = {
        "contract": (base or source)["contract"],
        "contract_version": (base or source)["contract_version"],
        "counts": {
            "source_episodes": len(combined_episodes),
            "train_episodes": sum(row.get("split") == "train" for row in combined_episodes),
            "eval_episodes": sum(row.get("split") == "eval" for row in combined_episodes),
        },
        "episodes": combined_episodes,
        "language_reference": (base or source)["language_reference"],
        "raw_manifest": file_identity(manifest_path),
        "sampling_protocol": (base or source)["sampling_protocol"],
        "selection": {
            "rule": (
                "evo1_clean_plus_persisted_dagger_round1_round2_v1"
                if base is not None
                else "persisted_dagger_round1_round2_only_v1"
            )
        },
        "tasks": (base or {"tasks": tasks})["tasks"],
        "additional_sources": sorted(identities.values(), key=lambda item: item["path"]),
        "source_index": file_identity(source_index),
    }
    _atomic_json(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--persisted-root", type=Path, required=True)
    parser.add_argument("--sha256sums", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=970)
    parser.add_argument("--base-index", type=Path)
    args = parser.parse_args()
    result = build_recovery_only_index(
        args.source_index,
        args.persisted_root,
        args.sha256sums,
        args.output,
        expected_episodes=args.expected_episodes,
        base_index=args.base_index,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}: episodes={len(result['episodes'])} sha256={digest}")


if __name__ == "__main__":
    main()
