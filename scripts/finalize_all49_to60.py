"""Finalize the expanded data as full MetaWorld MT50 episodes.

The original 49 task ids stay unchanged and ``push-back-v3`` is appended as
task id 49.  Every ordinary task, including push-back, has exactly 60 source
episodes.  Assembly and door-unlock retain their existing 270 episodes.  The
old 49-task eval remains the episode-level holdout source.  No action/frame
windows are generated: training draws arbitrary overlapping starts online.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from scripts.build_longtraj_features import PEER_SYNC_H15_P15_CONTRACT, phase1
from scripts.extract_lerobot_push_back import (
    CONTRACT as PUSH_BACK_BASE_CONTRACT,
    FAILED_SOURCE_EPISODES as PUSH_BACK_FAILED_SOURCE_EPISODES,
    RECOVERED_SOURCE_EPISODES as PUSH_BACK_RECOVERED_SOURCE_EPISODES,
    extract_push_back_once,
)
from scripts.merge_longtraj_expansion import merge_task
from scripts.split_wam4va_episode_holdout import build_complement_artifacts


HARD_TASKS = {"assembly-v3", "door-unlock-v3"}
PUSH_BACK_TASK = "push-back-v3"
PUSH_BACK_TEXT = "Pull a puck to a goal"
BASE_RAW_CONTRACT = "all49_canonical_raw_sources_v1"
RAW_CONTRACT = "mt50_canonical_raw_sources_v2"
EXPANSION_CONTRACT = "mt50_ordinary_60ep_recovery_v2"
LANGUAGE_REFERENCE_CONTRACT = "mt50_language_normalization_reference_v2"
PUSH_BACK_EVAL_EPISODES = (490000, 490001, 490002)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def load_longtraj(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"{path}: invalid longtraj payload")
    return payload


def _tensor_dict_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                return False
            if not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def _base_language_rows(reference: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    tasks = list((reference.get("metadata") or {}).get("tasks") or [])
    if len(tasks) != 49 or len(set(tasks)) != 49 or PUSH_BACK_TEXT in tasks:
        raise ValueError("language reference must contain the canonical 49 tasks")
    instruction_id = torch.as_tensor(reference.get("instruction_id"), dtype=torch.long)
    hidden = reference.get("language_hidden")
    mask = reference.get("language_mask")
    if not isinstance(hidden, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("language reference lacks cached hidden states/masks")
    rows = []
    for task_id in range(49):
        matches = (instruction_id == task_id).nonzero(as_tuple=False).flatten()
        if not len(matches):
            raise ValueError(f"language reference lacks task id {task_id}")
        rows.append(int(matches[0]))
    index = torch.tensor(rows, dtype=torch.long)
    return hidden.index_select(0, index), mask.index_select(0, index)


def compact_mt50_language_reference(
    base: dict[str, Any],
    new_hidden: torch.Tensor,
    new_mask: torch.Tensor,
    *,
    base_reference_path: Path,
    qwen_model: Path,
) -> dict[str, Any]:
    """Append the real push-back language row without changing ids 0..48."""
    old_hidden, old_mask = _base_language_rows(base)
    if new_hidden.ndim == 3 and new_hidden.shape[0] == 1:
        new_hidden = new_hidden[0]
    if new_mask.ndim == 2 and new_mask.shape[0] == 1:
        new_mask = new_mask[0]
    if new_hidden.ndim != 2 or new_mask.ndim != 1:
        raise ValueError("encoded push-back language must be [L,D] plus [L]")
    if new_hidden.shape[0] != new_mask.shape[0]:
        raise ValueError("encoded push-back hidden/mask token counts differ")
    if new_hidden.shape[1] != old_hidden.shape[2]:
        raise ValueError("encoded push-back language dimension differs")
    if new_hidden.shape[0] > old_hidden.shape[1]:
        raise ValueError("push-back instruction exceeds the frozen cache length")

    padded_hidden = torch.zeros_like(old_hidden[0])
    padded_mask = torch.zeros_like(old_mask[0])
    length = new_hidden.shape[0]
    padded_hidden[:length] = new_hidden.to(dtype=old_hidden.dtype, device="cpu")
    padded_mask[:length] = new_mask.to(dtype=old_mask.dtype, device="cpu")
    tasks = [*(base.get("metadata") or {})["tasks"], PUSH_BACK_TEXT]
    return {
        "normalization": dict(base["normalization"]),
        "instruction_id": torch.arange(50, dtype=torch.long),
        "language_hidden": torch.cat((old_hidden, padded_hidden.unsqueeze(0))),
        "language_mask": torch.cat((old_mask, padded_mask.unsqueeze(0))),
        "metadata": {
            "contract": LANGUAGE_REFERENCE_CONTRACT,
            "tasks": tasks,
            "push_back_task_id": 49,
            "push_back_task": PUSH_BACK_TASK,
            "push_back_text": PUSH_BACK_TEXT,
            "base_reference": file_identity(base_reference_path),
            "qwen_model_path": str(qwen_model.expanduser().resolve(strict=True)),
        },
    }


def validate_mt50_language_reference(
    payload: dict[str, Any], base: dict[str, Any]
) -> None:
    old_hidden, old_mask = _base_language_rows(base)
    metadata = payload.get("metadata") or {}
    tasks = list(metadata.get("tasks") or [])
    if (
        metadata.get("contract") != LANGUAGE_REFERENCE_CONTRACT
        or len(tasks) != 50
        or tasks[-1] != PUSH_BACK_TEXT
        or int(metadata.get("push_back_task_id", -1)) != 49
    ):
        raise ValueError("invalid MT50 language-reference metadata")
    hidden, mask = payload.get("language_hidden"), payload.get("language_mask")
    if (
        not isinstance(hidden, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or hidden.shape != (50, *old_hidden.shape[1:])
        or mask.shape != (50, old_mask.shape[1])
        or not torch.equal(hidden[:49], old_hidden)
        or not torch.equal(mask[:49], old_mask)
        or not bool(mask[49].any())
    ):
        raise ValueError("invalid MT50 cached language tensors")
    if not torch.equal(
        torch.as_tensor(payload.get("instruction_id")), torch.arange(50)
    ):
        raise ValueError("MT50 language-reference ids must be exactly 0..49")
    if not _tensor_dict_equal(payload.get("normalization") or {}, base["normalization"]):
        raise ValueError("MT50 reference changed the frozen normalization")


def build_mt50_language_reference_once(
    base_reference_path: Path,
    output_path: Path,
    *,
    qwen_model: Path,
    language_device: str,
) -> Path:
    base_reference_path = base_reference_path.expanduser().resolve(strict=True)
    base = torch.load(base_reference_path, map_location="cpu", weights_only=True)
    if output_path.exists():
        validate_mt50_language_reference(
            torch.load(output_path, map_location="cpu", weights_only=True), base
        )
        return output_path
    qwen_model = qwen_model.expanduser().resolve(strict=True)

    from va_compound.vision.backbones import QwenTextBackbone

    backbone = QwenTextBackbone.from_pretrained(
        model_id=str(qwen_model),
        device=language_device,
        dtype="float16",
        max_length=64,
        local_files_only=True,
    )
    new_hidden, new_mask = backbone.encode([PUSH_BACK_TEXT])
    payload = compact_mt50_language_reference(
        base,
        new_hidden.cpu(),
        new_mask.cpu(),
        base_reference_path=base_reference_path,
        qwen_model=qwen_model,
    )
    validate_mt50_language_reference(payload, base)
    save_payload_once(output_path, payload)
    return output_path


def expected_shards(shard_root: Path, task_id: int, task: str) -> list[Path]:
    paths = [
        shard_root
        / task
        / f"metaworld_longtraj_{task}_recovery_v2_shard{shard}.pt"
        for shard in range(3)
    ]
    for shard, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing expansion shard: {path}")
        payload = load_longtraj(path)
        if payload.get("task") != task or len(payload["episodes"]) != 10:
            raise ValueError(f"{path}: expected task={task} with 10 episodes")
        expected = [600000 + task_id * 1000 + shard * 10 + offset for offset in range(10)]
        actual = [episode.get("episode_seed") for episode in payload["episodes"]]
        metadata = payload.get("metadata") or {}
        if metadata.get("pinned_episode_seeds") is not None and actual != expected:
            raise ValueError(f"{path}: episode seeds differ: {actual} != {expected}")
        if (
            len(set(actual)) != 10
            or any(not isinstance(seed, int) for seed in actual)
            or any(35000 <= seed < 35050 for seed in actual)
        ):
            raise ValueError(f"{path}: fallback seeds must be 10 unique non-eval seeds")
        if any(int(episode.get("n_perturb_events", 0)) < 1 for episode in payload["episodes"]):
            raise ValueError(f"{path}: every appended episode must contain recovery")
    return paths


def validate_push_back_sources(base_path: Path, recovery_path: Path) -> None:
    base = load_longtraj(base_path)
    recovery = load_longtraj(recovery_path)
    base_episodes = base["episodes"]
    recovery_episodes = recovery["episodes"]
    if (
        base.get("task") != PUSH_BACK_TASK
        or len(base_episodes) != 50
        or (base.get("metadata") or {}).get("contract") != PUSH_BACK_BASE_CONTRACT
        or [episode.get("source_episode_index") for episode in base_episodes[:47]]
        != list(PUSH_BACK_RECOVERED_SOURCE_EPISODES)
        or any(
            episode.get("source_episode_index") is not None
            for episode in base_episodes[47:]
        )
        or (base.get("metadata") or {}).get("rejected_no_success_episode_indices")
        != list(PUSH_BACK_FAILED_SOURCE_EPISODES)
        or any(bool(episode.get("perturbed", False)) for episode in base_episodes)
    ):
        raise ValueError(f"{base_path}: invalid recovered LeRobot push-back base")
    if (
        recovery.get("task") != PUSH_BACK_TASK
        or len(recovery_episodes) != 10
        or [episode.get("episode_seed") for episode in recovery_episodes]
        != list(range(649030, 649040))
        or any(
            int(episode.get("n_perturb_events", 0)) < 1
            for episode in recovery_episodes
        )
    ):
        raise ValueError(f"{recovery_path}: invalid push-back recovery supplement")


def _push_back_provenance(
    base_path: Path, recovery_path: Path
) -> list[dict[str, Any]]:
    return [
        {
            **file_identity(base_path),
            "mode": "recovered47_plus_clean3",
            "episode_range": [0, 50],
        },
        {**file_identity(recovery_path), "mode": "recovery", "episode_range": [50, 60]},
    ]


def validate_push_back_merge(
    output: Path, base_path: Path, recovery_path: Path
) -> None:
    validate_push_back_sources(base_path, recovery_path)
    payload = load_longtraj(output)
    episodes = payload["episodes"]
    if payload.get("task") != PUSH_BACK_TASK or len(episodes) != 60:
        raise ValueError(f"{output}: push-back must contain exactly 60 episodes")
    if [episode.get("source_episode_index") for episode in episodes[:47]] != list(
        PUSH_BACK_RECOVERED_SOURCE_EPISODES
    ) or any(
        episode.get("source_episode_index") is not None
        for episode in episodes[47:50]
    ):
        raise ValueError(f"{output}: recovered push-back episode order differs")
    if any(bool(ep.get("perturbed", False)) for ep in episodes[:50]):
        raise ValueError(f"{output}: first 50 push-back episodes must be original clean")
    if [episode.get("episode_seed") for episode in episodes[50:]] != list(
        range(649030, 649040)
    ) or any(int(ep.get("n_perturb_events", 0)) < 1 for ep in episodes[50:]):
        raise ValueError(f"{output}: final 10 push-back episodes must be recovery")
    actual = list((payload.get("metadata") or {}).get("merge_sources") or [])
    if actual != _push_back_provenance(base_path, recovery_path):
        raise ValueError(f"{output}: push-back merge provenance differs")


def build_push_back_merge_once(
    base_path: Path, recovery_path: Path, output: Path
) -> Path:
    validate_push_back_sources(base_path, recovery_path)
    if output.exists():
        validate_push_back_merge(output, base_path, recovery_path)
        return output
    base = load_longtraj(base_path)
    recovery = load_longtraj(recovery_path)
    merged = {key: value for key, value in base.items() if key != "episodes"}
    merged["task"] = PUSH_BACK_TASK
    merged["episodes"] = [*base["episodes"], *recovery["episodes"]]
    merged["n_episodes"] = 60
    metadata = dict(merged.get("metadata") or {})
    metadata.update(
        {
            "merge_contract": "push_back_47_lerobot_plus_3_clean_plus_10_recovery_v2",
            "merge_sources": _push_back_provenance(base_path, recovery_path),
            "episode_contract": "exactly_50_recovered_clean_then_10_recovery_v1",
        }
    )
    merged["metadata"] = metadata
    save_payload_once(output, merged)
    validate_push_back_merge(output, base_path, recovery_path)
    return output


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.expanduser().resolve(strict=True)
    link = link.expanduser().absolute()
    if os.path.lexists(link):
        if not link.is_file() or link.resolve(strict=True) != target:
            raise FileExistsError(f"refusing to replace mismatched frame path: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"stale symlink temporary: {temporary}")
    temporary.symlink_to(target)
    os.replace(temporary, link)


def validate_existing_merge(
    output: Path,
    *,
    task: str,
    base: Path,
    shards: list[Path],
) -> None:
    payload = load_longtraj(output)
    if payload.get("task") != task or len(payload["episodes"]) != 60:
        raise ValueError(f"{output}: expected task={task} with 60 episodes")
    sources = list((payload.get("metadata") or {}).get("merge_sources") or [])
    expected_paths = [base.expanduser().resolve(strict=True), *(
        path.expanduser().resolve(strict=True) for path in shards
    )]
    if [Path(str(item.get("path", ""))).resolve(strict=True) for item in sources] != expected_paths:
        raise ValueError(f"{output}: merge provenance paths differ")
    for item, path in zip(sources, expected_paths, strict=True):
        if item.get("sha256") != sha256_file(path):
            raise ValueError(f"{output}: merge provenance SHA differs for {path}")


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace different manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale manifest temporary: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def prepare_expanded_raw(
    base_manifest_path: Path,
    shard_root: Path,
    push_back_root: Path,
    frames_dir: Path,
    raw_manifest_path: Path,
    language_reference: Path,
) -> dict[str, Any]:
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("contract") != BASE_RAW_CONTRACT:
        raise ValueError("unexpected base raw identity contract")
    sources = list(base_manifest.get("sources") or [])
    tasks = [str(item.get("task") or "") for item in sources]
    if len(tasks) != 49 or len(set(tasks)) != 49 or not HARD_TASKS.issubset(tasks):
        raise ValueError("base raw manifest must contain the canonical 49 tasks")

    entries: list[dict[str, Any]] = []
    frames_dir.mkdir(parents=True, exist_ok=True)
    for task_id, (task, old_entry) in enumerate(zip(tasks, sources, strict=True)):
        base = Path(old_entry["source_path"]).expanduser().resolve(strict=True)
        canonical = frames_dir / f"metaworld_longtraj_{task}.pt"
        if task in HARD_TASKS:
            ensure_symlink(canonical, base)
            source = base
        else:
            base_payload = load_longtraj(base)
            if base_payload.get("task") != task or len(base_payload["episodes"]) != 30:
                raise ValueError(f"{base}: ordinary base must contain exactly 30 episodes")
            shards = expected_shards(shard_root, task_id, task)
            if canonical.exists():
                validate_existing_merge(
                    canonical, task=task, base=base, shards=shards
                )
            else:
                merge_task(base, [shard_root / task], frames_dir)
            source = canonical.resolve(strict=True)
        identity = file_identity(source)
        entries.append(
            {
                "task": task,
                "source_path": identity["path"],
                "canonical_path": str(canonical.expanduser().absolute()),
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )

    push_back_path = build_push_back_merge_once(
        push_back_root
        / "base"
        / "metaworld_longtraj_push-back-v3_lerobot47_clean3.pt",
        push_back_root
        / "shards"
        / "metaworld_longtraj_push-back-v3_recovery_v1_shard0.pt",
        frames_dir / f"metaworld_longtraj_{PUSH_BACK_TASK}.pt",
    )
    push_back_identity = file_identity(push_back_path)
    entries.append(
        {
            "task": PUSH_BACK_TASK,
            "source_path": push_back_identity["path"],
            "canonical_path": str(push_back_path.expanduser().absolute()),
            "size_bytes": push_back_identity["size_bytes"],
            "sha256": push_back_identity["sha256"],
        }
    )
    if len(entries) != 50 or entries[-1]["task"] != PUSH_BACK_TASK:
        raise ValueError("expanded raw order must preserve ids 0..48 and append push-back")

    manifest = {
        "contract": RAW_CONTRACT,
        "expansion_contract": EXPANSION_CONTRACT,
        "reference": file_identity(language_reference),
        "base_reference": base_manifest["reference"],
        "task_id_contract": "canonical_all49_unchanged_push_back_appended_as_49_v1",
        "sources": entries,
    }
    write_json_once(raw_manifest_path, manifest)
    return manifest


def _freeze(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def row_identity(payload: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        int(payload["instruction_id"][index]),
        int(payload["episode_id"][index]),
        _freeze(payload["frame_refs"][index]),
    )


def _aligned(value: Any, rows: int) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] == rows
    ) or isinstance(value, (list, tuple)) and len(value) == rows


def stable_old_prefix(
    expanded: dict[str, Any],
    old: dict[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    new_rows = len(expanded["actions"])
    old_rows = len(old["actions"])
    lookup: dict[tuple[Any, ...], int] = {}
    for index in range(new_rows):
        key = row_identity(expanded, index)
        if key in lookup:
            raise ValueError(f"expanded source has duplicate row identity: {key[:2]}")
        lookup[key] = index
    old_indices: list[int] = []
    for index in range(old_rows):
        key = row_identity(old, index)
        if key not in lookup:
            raise ValueError(f"old source row is absent from expansion: {key[:2]}")
        old_indices.append(lookup[key])
    old_set = set(old_indices)
    order = old_indices + [index for index in range(new_rows) if index not in old_set]
    if len(order) != new_rows or len(set(order)) != new_rows:
        raise ValueError("expanded row permutation is not bijective")
    order_tensor = torch.tensor(order, dtype=torch.long)

    reordered: dict[str, Any] = {}
    for key, value in expanded.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == new_rows:
            reordered[key] = value.index_select(0, order_tensor)
        elif isinstance(value, list) and len(value) == new_rows:
            reordered[key] = [value[index] for index in order]
        elif isinstance(value, tuple) and len(value) == new_rows:
            reordered[key] = tuple(value[index] for index in order)
        else:
            reordered[key] = value
    reordered["pair_id"] = torch.arange(new_rows, dtype=old["pair_id"].dtype)

    for key, old_value in old.items():
        if not _aligned(old_value, old_rows):
            continue
        new_value = reordered.get(key)
        if isinstance(old_value, torch.Tensor):
            if not isinstance(new_value, torch.Tensor) or not torch.equal(
                new_value[:old_rows], old_value
            ):
                raise ValueError(f"old source prefix differs on tensor {key}")
        else:
            if list(new_value[:old_rows]) != list(old_value):
                raise ValueError(f"old source prefix differs on sequence {key}")

    metadata = dict(reordered.get("metadata") or {})
    output_identity = dict(metadata.get("output_identity") or {})
    output_identity["path"] = str(output_path.expanduser().resolve(strict=False))
    output_identity.setdefault("shape", {})["windows"] = new_rows
    metadata["output_identity"] = output_identity
    metadata["expansion_contract"] = EXPANSION_CONTRACT
    metadata["stable_old_source_prefix_rows"] = old_rows
    reordered["metadata"] = metadata
    return reordered


def save_payload_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale payload temporary: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_source_once(
    raw_manifest: dict[str, Any],
    old_source_path: Path,
    source_path: Path,
) -> None:
    if source_path.exists():
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        old = torch.load(old_source_path, map_location="cpu", weights_only=True)
        stable_old_prefix(payload, old, output_path=source_path)
        return
    phase1_output = source_path.with_name(f".{source_path.name}.phase1.pt")
    if phase1_output.exists():
        raise FileExistsError(f"stale phase1 artifact: {phase1_output}")
    inputs = [Path(item["source_path"]) for item in raw_manifest["sources"]]
    phase1(
        15,
        input_paths=inputs,
        output_path=phase1_output,
        ref_path=Path(raw_manifest["reference"]["path"]),
        legacy_policy="infer",
        data_contract=PEER_SYNC_H15_P15_CONTRACT,
        planning_stride=15,
    )
    try:
        expanded = torch.load(phase1_output, map_location="cpu", weights_only=True)
        old = torch.load(old_source_path, map_location="cpu", weights_only=True)
        stable = stable_old_prefix(expanded, old, output_path=source_path)
        save_payload_once(source_path, stable)
    finally:
        if phase1_output.exists():
            phase1_output.unlink()


def _subset_aligned(payload: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    rows = len(payload["actions"])
    output: dict[str, Any] = {}
    selected = indices.tolist()
    for key, value in payload.items():
        if key == "metadata":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == rows:
            output[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == rows:
            output[key] = [value[index] for index in selected]
        elif isinstance(value, tuple) and len(value) == rows:
            output[key] = tuple(value[index] for index in selected)
        else:
            output[key] = value
    return output


def _assert_aligned_prefix(
    candidate: dict[str, Any], expected: dict[str, Any], *, prefix_rows: int
) -> None:
    expected_rows = len(expected["actions"])
    if prefix_rows != expected_rows:
        raise ValueError("aligned-prefix row count differs")
    for key, value in expected.items():
        if not _aligned(value, expected_rows):
            continue
        actual = candidate.get(key)
        if isinstance(value, torch.Tensor):
            if not isinstance(actual, torch.Tensor) or not torch.equal(
                actual[:prefix_rows], value
            ):
                raise ValueError(f"extended eval changed frozen tensor {key}")
        elif list(actual[:prefix_rows]) != list(value):
            raise ValueError(f"extended eval changed frozen sequence {key}")


def make_mt50_eval_payload(
    source: dict[str, Any],
    frozen_all49_eval: dict[str, Any],
    *,
    frozen_eval_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    old_episode_ids = {
        int(value) for value in frozen_all49_eval["episode_id"].tolist()
    }
    selected_episode_ids = old_episode_ids | set(PUSH_BACK_EVAL_EPISODES)
    indices = torch.tensor(
        [
            index
            for index, episode_id in enumerate(source["episode_id"].tolist())
            if int(episode_id) in selected_episode_ids
        ],
        dtype=torch.long,
    )
    if not len(indices):
        raise ValueError("expanded source produced an empty MT50 eval")
    output = _subset_aligned(source, indices)
    old_rows = len(frozen_all49_eval["actions"])
    _assert_aligned_prefix(output, frozen_all49_eval, prefix_rows=old_rows)
    push_task = output["instruction_id"] == 49
    push_episodes = set(output["episode_id"][push_task].tolist())
    if push_episodes != set(PUSH_BACK_EVAL_EPISODES):
        raise ValueError(
            f"MT50 eval must contain push-back episodes {PUSH_BACK_EVAL_EPISODES}"
        )
    task_ids = sorted(int(value) for value in torch.unique(output["instruction_id"]))
    if task_ids != list(range(50)):
        raise ValueError(f"MT50 eval must cover task ids 0..49, got {task_ids}")

    metadata = dict(source.get("metadata") or {})
    output_identity = dict(metadata.get("output_identity") or {})
    output_identity.update(
        {
            "path": str(output_path.expanduser().resolve(strict=False)),
            "shape": {
                **dict(output_identity.get("shape") or {}),
                "windows": len(output["actions"]),
            },
        }
    )
    metadata.update(
        {
            "output_identity": output_identity,
            "split_name": "eval",
            "mt50_eval_contract": "frozen_all49_plus_push_back_clean3_v1",
            "frozen_all49_eval": file_identity(frozen_eval_path),
            "frozen_all49_eval_rows": old_rows,
            "push_back_eval_episode_ids": list(PUSH_BACK_EVAL_EPISODES),
        }
    )
    output["metadata"] = metadata
    return output


def build_mt50_eval_once(
    source_path: Path,
    frozen_all49_eval_path: Path,
    output_path: Path,
) -> None:
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    frozen = torch.load(
        frozen_all49_eval_path, map_location="cpu", weights_only=True
    )
    expected = make_mt50_eval_payload(
        source,
        frozen,
        frozen_eval_path=frozen_all49_eval_path,
        output_path=output_path,
    )
    if output_path.exists():
        actual = torch.load(output_path, map_location="cpu", weights_only=True)
        rows = len(expected["actions"])
        _assert_aligned_prefix(actual, expected, prefix_rows=rows)
        metadata = actual.get("metadata") or {}
        if metadata.get("mt50_eval_contract") != "frozen_all49_plus_push_back_clean3_v1":
            raise ValueError("existing MT50 eval has the wrong contract")
        return
    save_payload_once(output_path, expected)


def validate_or_build_complement(
    source: Path,
    mt50_eval: Path,
    train: Path,
    manifest: Path,
    *,
    frozen_all49_eval: Path,
) -> None:
    if train.exists() != manifest.exists():
        raise FileExistsError("expanded train and split manifest must exist together")
    if not train.exists():
        eval_sha = sha256_file(mt50_eval)
        frozen_sha = sha256_file(frozen_all49_eval)
        build_complement_artifacts(source, mt50_eval, train, manifest)
        if sha256_file(mt50_eval) != eval_sha:
            raise RuntimeError("MT50 eval changed while building expanded train")
        if sha256_file(frozen_all49_eval) != frozen_sha:
            raise RuntimeError("frozen all49 eval changed while building MT50 train")
        return
    split = json.loads(manifest.read_text(encoding="utf-8"))
    if (split.get("selection") or {}).get("existing_eval_sha256") != sha256_file(mt50_eval):
        raise ValueError("expanded split is not bound to the MT50 eval")
    if (split.get("source") or {}).get("sha256") != sha256_file(source):
        raise ValueError("expanded split is not bound to the expanded source")
    counts = {
        "source": len(torch.load(source, map_location="cpu", weights_only=True)["actions"]),
        "train": len(torch.load(train, map_location="cpu", weights_only=True)["actions"]),
        "eval": len(torch.load(mt50_eval, map_location="cpu", weights_only=True)["actions"]),
    }
    if counts["source"] != counts["train"] + counts["eval"]:
        raise ValueError(f"expanded split window counts are not exhaustive: {counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-raw-manifest",
        type=Path,
        default=Path("/root/ora0_all49_data/all49_raw_canonical_identity_v1.json"),
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=Path("/root/ora0_all49_expand60_v1/shards"),
    )
    parser.add_argument(
        "--push-back-root",
        type=Path,
        default=Path("/root/ora0_all49_expand60_v1/push_back"),
    )
    parser.add_argument(
        "--lerobot-dataset",
        type=Path,
        default=Path(
            "/root/private_data/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
        ),
    )
    parser.add_argument(
        "--normalization-ref",
        type=Path,
        default=Path("/root/private_data/ORA0/data/longtraj_normalization_ref.pt"),
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("/root/ora0_all49_expand60_v1")
    )
    parser.add_argument(
        "--old-source",
        type=Path,
        default=Path("/root/ora0_all49_data/all49_peer_h15_p15_source_v1.pt"),
    )
    parser.add_argument(
        "--existing-eval",
        type=Path,
        default=Path("/root/ora0_all49_data/all49_peer_h15_p15_eval_v1.pt"),
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=Path("/root/private_data/newhost_env/models/Qwen3.5-2B"),
    )
    parser.add_argument("--language-device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = args.out_root / "frames_v2"
    data = args.out_root / "data_v2"
    raw_manifest_path = data / "mt50_raw_canonical_identity_60ep_v2.json"
    language_reference = data / "mt50_language_normalization_ref_v2.pt"
    online_index = data / "mt50_full_episode_online_index_v1.json"

    base_manifest = json.loads(
        args.base_raw_manifest.read_text(encoding="utf-8")
    )
    base_reference = Path(base_manifest["reference"]["path"])
    build_mt50_language_reference_once(
        base_reference,
        language_reference,
        qwen_model=args.qwen_model,
        language_device=args.language_device,
    )
    extract_push_back_once(
        args.lerobot_dataset,
        args.push_back_root
        / "base"
        / "metaworld_longtraj_push-back-v3_lerobot47_clean3.pt",
        normalization_ref=args.normalization_ref,
        clean_supplement=(
            args.push_back_root
            / "shards"
            / "metaworld_longtraj_push-back-v3_clean_v1_shard0.pt"
        ),
    )

    raw_manifest = prepare_expanded_raw(
        args.base_raw_manifest,
        args.shard_root,
        args.push_back_root,
        frames,
        raw_manifest_path,
        language_reference,
    )
    from scripts.build_online_episode_index import build_index

    index = build_index(
        raw_manifest_path,
        args.existing_eval,
        online_index,
        workers=6,
    )
    counts = index["counts"]
    online_samples = counts["train_episodes"] * 6
    steps_per_epoch = (online_samples + 47) // 48
    print(
        f"[ok] true MT50: push-back=60 episodes "
        f"(47 recovered LeRobot + 3 clean supplement + 10 recovery), "
        f"source={counts['source_episodes']}, train={counts['train_episodes']}, "
        f"eval={counts['eval_episodes']}, offline_windows=0, "
        f"online_samples/epoch={online_samples}, steps/epoch={steps_per_epoch}, "
        f"20 epochs={20 * steps_per_epoch}"
    )


if __name__ == "__main__":
    main()
