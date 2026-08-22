#!/usr/bin/env python3
"""Create the fixed episode-level WAM4VA train/eval split and manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor


MANIFEST_CONTRACT = "wam4va_episode_holdout_manifest_v1"
PEER_SYNC_H6_CONTRACT = "peer_sync_h6_world_windows_v1"
PEER_SYNC_H6_P2_CONTRACT = "peer_sync_h6_p2_world_windows_v1"
PEER_SYNC_H15_P2_CONTRACT = "peer_sync_h15_p2_world_windows_v1"
PEER_SYNC_H6_CONTRACTS = frozenset({
    PEER_SYNC_H6_CONTRACT,
    PEER_SYNC_H6_P2_CONTRACT,
    PEER_SYNC_H15_P2_CONTRACT,
})
MANIFEST_VERSION = 1
LEGACY_SHAPE = (4, 48, 4)
PEER_SHAPE = (4, 6, 4)
PEER_H15_SHAPE = (4, 15, 4)
TRANSITION_PREFIX_STEPS = 6
TRANSITION_RULE = {
    "contract": "wam4va_world_transition_mask_v1",
    "expression": (
        "action_valid_mask[:, t, :6].all(-1) & "
        "action_valid_mask[:, t + 1, 0]"
    ),
    "current_action_prefix_steps": TRANSITION_PREFIX_STEPS,
    "next_action_index": 0,
    "time_indices": "t=0..T-2",
}
PEER_SYNC_H6_P2_TRANSITION_PREFIX_STEPS = 2
PEER_SYNC_H6_P2_TRANSITION_RULE = {
    "contract": "wam4va_world_transition_mask_v1",
    "expression": (
        "action_valid_mask[:, t, :2].all(-1) & "
        "action_valid_mask[:, t + 1, 0]"
    ),
    "current_action_prefix_steps": PEER_SYNC_H6_P2_TRANSITION_PREFIX_STEPS,
    "next_action_index": 0,
    "time_indices": "t=0..T-2",
}
PEER_SYNC_H15_P2_TRANSITION_RULE = {
    "contract": "wam4va_explicit_endpoint_mask_v1",
    "expression": "world_target_valid_mask[:, t]",
    "current_action_prefix_steps": 15,
    "target_offset_steps": 15,
    "time_indices": "t=0..T-1",
}
MANIFEST_HASH_CONTRACT = (
    "sha256(canonical compact UTF-8 JSON with sorted keys and "
    "manifest_id/manifest_sha256 omitted)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 without loading the file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_sha256(manifest: dict) -> str:
    """Hash the canonical manifest content, excluding its self-identity fields."""
    canonical = dict(manifest)
    canonical.pop("manifest_id", None)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_protocol(payload: dict) -> tuple[str, tuple[int, int, int], int]:
    metadata = payload.get("metadata") or {}
    contract = metadata.get("contract")
    if contract in PEER_SYNC_H6_CONTRACTS:
        if int(metadata.get("contract_version", -1)) != 1:
            raise ValueError(f"{contract} requires contract_version=1")
        expected_horizon = 15 if contract == PEER_SYNC_H15_P2_CONTRACT else 6
        if metadata.get("logged_action_chunk") != f"full_h{expected_horizon}":
            raise ValueError(
                f"{contract} requires full logged H{expected_horizon} chunk"
            )
        for key in ("parent_identity", "source_identities", "output_identity"):
            if not metadata.get(key):
                raise ValueError(f"{contract} requires metadata.{key}")
        if contract in {PEER_SYNC_H6_P2_CONTRACT, PEER_SYNC_H15_P2_CONTRACT}:
            required = {
                "fps": 80,
                "planning_stride": 2,
                "control_stride": 2,
                "sequence_length": 4,
                "decision_offsets": [0, 2, 4, 6],
                "action_horizon": expected_horizon,
                "action_label_offsets": list(range(expected_horizon)),
            }
            if contract == PEER_SYNC_H15_P2_CONTRACT:
                required.update(
                    world_target_horizon=15,
                    world_target_offsets=[15, 17, 19, 21],
                )
            for key, expected in required.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"{contract} requires metadata.{key}="
                        f"{expected!r}, got {metadata.get(key)!r}"
                    )
            return (
                contract,
                PEER_H15_SHAPE if expected_horizon == 15 else PEER_SHAPE,
                expected_horizon if expected_horizon == 15
                else PEER_SYNC_H6_P2_TRANSITION_PREFIX_STEPS,
            )
        return contract, PEER_SHAPE, TRANSITION_PREFIX_STEPS
    return MANIFEST_CONTRACT, LEGACY_SHAPE, TRANSITION_PREFIX_STEPS


def _validate_payload(payload: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    required = (
        "actions",
        "instruction_id",
        "episode_id",
        "action_valid_mask",
        "recovery_mask",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"source payload is missing keys: {missing}")

    actions = payload["actions"]
    task = payload["instruction_id"]
    episode = payload["episode_id"]
    action_valid = payload["action_valid_mask"]
    recovery = payload["recovery_mask"]
    if not isinstance(actions, Tensor) or actions.ndim != 4:
        raise ValueError("actions must be a tensor with shape [N,T,H,A]")
    n, sequence, horizon, action_dim = actions.shape
    protocol, expected_shape, _ = _payload_protocol(payload)
    if n == 0:
        raise ValueError("actions require N>0")
    if (sequence, horizon, action_dim) != expected_shape:
        label = "peer" if protocol in PEER_SYNC_H6_CONTRACTS else "legacy H48"
        raise ValueError(
            f"{label} split requires exact T{expected_shape[0]}/H{expected_shape[1]}/"
            f"A{expected_shape[2]}, got T{sequence}/H{horizon}/A{action_dim}"
        )
    if protocol == PEER_SYNC_H15_P2_CONTRACT:
        target_valid = payload.get("world_target_valid_mask")
        target_refs = payload.get("world_target_frame_refs")
        if (
            not isinstance(target_valid, Tensor)
            or target_valid.dtype != torch.bool
            or tuple(target_valid.shape) != (n, sequence)
        ):
            raise ValueError("H15 world_target_valid_mask must be bool [N,T]")
        if not isinstance(target_refs, (list, tuple)) or len(target_refs) != n:
            raise ValueError("H15 requires one world_target_frame_ref per window")
    for name, value in (("instruction_id", task), ("episode_id", episode)):
        if (
            not isinstance(value, Tensor)
            or value.shape != (n,)
            or value.dtype == torch.bool
            or value.is_floating_point()
        ):
            raise ValueError(f"{name} must be an integer tensor with shape [{n}]")
    expected_mask_shape = actions.shape[:-1]
    for name, value in (
        ("action_valid_mask", action_valid),
        ("recovery_mask", recovery),
    ):
        if (
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or value.shape != expected_mask_shape
        ):
            raise ValueError(
                f"{name} must be bool with shape {tuple(expected_mask_shape)}"
            )
    return task.long(), episode.long(), action_valid, recovery


def _episode_rank(seed: int, task_id: int, episode_id: int) -> bytes:
    value = f"{MANIFEST_CONTRACT}\n{seed}\n{task_id}\n{episode_id}\n"
    return hashlib.sha256(value.encode("ascii")).digest()


def build_split_plan(
    payload: dict,
    *,
    heldout_fraction: float = 0.10,
    seed: int = 0,
) -> dict:
    """Plan a deterministic task-stratified split over whole episodes."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    task, episode, _, _ = _validate_payload(payload)

    episode_owner: dict[int, int] = {}
    groups: dict[int, dict[int, list[int]]] = {}
    for row, (task_id, episode_id) in enumerate(
        zip(task.tolist(), episode.tolist(), strict=True)
    ):
        task_id, episode_id = int(task_id), int(episode_id)
        owner = episode_owner.setdefault(episode_id, task_id)
        if owner != task_id:
            raise ValueError(
                f"episode_id={episode_id} belongs to multiple tasks: {owner}, {task_id}"
            )
        groups.setdefault(task_id, {}).setdefault(episode_id, []).append(row)

    eval_episodes_by_task: dict[int, list[int]] = {}
    train_episodes_by_task: dict[int, list[int]] = {}
    for task_id, by_episode in sorted(groups.items()):
        episodes = sorted(by_episode)
        if len(episodes) < 2:
            raise ValueError(
                f"task {task_id} needs at least two episodes, got {len(episodes)}"
            )
        heldout_count = int(math.floor(len(episodes) * heldout_fraction + 0.5))
        heldout_count = min(max(heldout_count, 1), len(episodes) - 1)
        ranked = sorted(
            episodes,
            key=lambda episode_id: (
                _episode_rank(seed, task_id, episode_id),
                episode_id,
            ),
        )
        eval_set = set(ranked[:heldout_count])
        eval_episodes_by_task[task_id] = sorted(eval_set)
        train_episodes_by_task[task_id] = sorted(set(episodes) - eval_set)

    train_rows: list[int] = []
    eval_rows: list[int] = []
    for task_id, by_episode in sorted(groups.items()):
        eval_set = set(eval_episodes_by_task[task_id])
        for episode_id, rows in sorted(by_episode.items()):
            (eval_rows if episode_id in eval_set else train_rows).extend(rows)
    train_rows.sort()
    eval_rows.sort()

    train_set, eval_set = set(train_rows), set(eval_rows)
    expected = set(range(len(task)))
    if train_set & eval_set:
        raise RuntimeError("train/eval row sets overlap")
    if train_set | eval_set != expected:
        raise RuntimeError("train/eval row sets are not exhaustive")
    train_episode_set = {
        int(episode[index]) for index in train_rows
    }
    eval_episode_set = {
        int(episode[index]) for index in eval_rows
    }
    if train_episode_set & eval_episode_set:
        raise RuntimeError("train/eval episode sets overlap")

    return {
        "train_indices": torch.tensor(train_rows, dtype=torch.long),
        "eval_indices": torch.tensor(eval_rows, dtype=torch.long),
        "train_episodes_by_task": train_episodes_by_task,
        "eval_episodes_by_task": eval_episodes_by_task,
        "all_episodes_by_task": {
            task_id: sorted(by_episode) for task_id, by_episode in sorted(groups.items())
        },
    }


def transition_mask(
    action_valid_mask: Tensor,
    prefix_steps: int = TRANSITION_PREFIX_STEPS,
) -> Tensor:
    """Return [N,T-1] validity for the contract's transition action prefix."""
    if (
        action_valid_mask.ndim != 3
        or action_valid_mask.dtype != torch.bool
        or action_valid_mask.shape[1] < 2
        or isinstance(prefix_steps, bool)
        or not isinstance(prefix_steps, int)
        or prefix_steps <= 0
        or action_valid_mask.shape[2] < prefix_steps
    ):
        raise ValueError(
            "action_valid_mask must be bool [N,T>=2,H>=prefix_steps] with "
            "positive integer prefix_steps"
        )
    current = action_valid_mask[:, :-1, :prefix_steps].all(dim=-1)
    next_first = action_valid_mask[:, 1:, 0]
    return current & next_first


def _binary_stats(mask: Tensor) -> dict[str, int]:
    true = int(mask.sum().item())
    total = int(mask.numel())
    return {"true": true, "false": total - true, "total": total}


def mask_stats(payload: dict, indices: Tensor) -> dict:
    """Return exact integer mask numerators/denominators for selected rows."""
    _, _, action_valid, recovery = _validate_payload(payload)
    selected_valid = action_valid.index_select(0, indices)
    selected_recovery = recovery.index_select(0, indices)
    _, _, transition_prefix_steps = _payload_protocol(payload)
    explicit = payload.get("world_target_valid_mask")
    transitions = (
        torch.as_tensor(explicit, dtype=torch.bool).index_select(0, indices)
        if explicit is not None
        else transition_mask(selected_valid, transition_prefix_steps)
    )
    return {
        "action_valid": _binary_stats(selected_valid),
        "recovery": _binary_stats(selected_recovery),
        "recovery_and_action_valid": _binary_stats(
            selected_recovery & selected_valid
        ),
        "transition": {
            **_binary_stats(transitions),
            "windows_with_any_true": int(transitions.any(dim=1).sum().item()),
            "windows_without_any_true": int((~transitions.any(dim=1)).sum().item()),
        },
    }


def _task_names(payload: dict, task: Tensor) -> dict[int, tuple[str, str]]:
    metadata = payload.get("metadata") or {}
    descriptions = list(metadata.get("tasks") or [])
    frame_refs = payload.get("frame_refs")
    names_by_task: dict[int, set[str]] = {}
    if isinstance(frame_refs, (list, tuple)) and len(frame_refs) == len(task):
        for task_id, ref in zip(task.tolist(), frame_refs, strict=True):
            if isinstance(ref, (list, tuple)) and ref:
                names_by_task.setdefault(int(task_id), set()).add(str(ref[0]))

    output = {}
    for task_id in sorted(int(value) for value in torch.unique(task)):
        names = names_by_task.get(task_id, set())
        if len(names) > 1:
            raise ValueError(
                f"task {task_id} has inconsistent frame_ref names: {sorted(names)}"
            )
        description = (
            str(descriptions[task_id])
            if 0 <= task_id < len(descriptions)
            else f"task-{task_id}"
        )
        task_name = next(iter(names), description)
        output[task_id] = (task_name, description)
    return output


def _indices_for_task(indices: Tensor, task: Tensor, task_id: int) -> Tensor:
    return indices[task.index_select(0, indices) == task_id]


def _selection_summary(
    payload: dict,
    indices: Tensor,
    task_names: dict[int, tuple[str, str]],
    *,
    output_path: Path,
) -> dict:
    task = payload["instruction_id"].long()
    episode = payload["episode_id"].long()
    per_task = []
    for task_id, (task_name, description) in sorted(task_names.items()):
        task_indices = _indices_for_task(indices, task, task_id)
        per_task.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "task_description": description,
                "episode_ids": sorted(
                    {int(value) for value in episode[task_indices].tolist()}
                ),
                "windows": int(task_indices.numel()),
                "mask_stats": mask_stats(payload, task_indices),
            }
        )
    resolved_output = str(output_path.expanduser().resolve(strict=False))
    return {
        "output_path": resolved_output,
        "output_identity": {
            "path": resolved_output,
            "windows": int(indices.numel()),
            "episode_ids": sorted({int(value) for value in episode[indices].tolist()}),
        },
        "episode_ids": sorted({int(value) for value in episode[indices].tolist()}),
        "episodes": sum(len(item["episode_ids"]) for item in per_task),
        "windows": int(indices.numel()),
        "mask_stats": mask_stats(payload, indices),
        "tasks": per_task,
    }


def _build_manifest(
    payload: dict,
    source: dict,
    plan: dict,
    *,
    train_output: Path,
    eval_output: Path,
    manifest_output: Path,
    heldout_fraction: float,
    seed: int,
) -> dict:
    task, episode, _, _ = _validate_payload(payload)
    data_protocol, expected_shape, transition_prefix_steps = _payload_protocol(payload)
    task_names = _task_names(payload, task)
    train = _selection_summary(
        payload, plan["train_indices"], task_names, output_path=train_output
    )
    eval_split = _selection_summary(
        payload, plan["eval_indices"], task_names, output_path=eval_output
    )
    tasks = []
    all_indices = torch.arange(len(task), dtype=torch.long)
    for task_id, (task_name, description) in sorted(task_names.items()):
        task_indices = _indices_for_task(all_indices, task, task_id)
        train_item = next(item for item in train["tasks"] if item["task_id"] == task_id)
        eval_item = next(
            item for item in eval_split["tasks"] if item["task_id"] == task_id
        )
        tasks.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "task_description": description,
                "source": {
                    "episode_ids": sorted(
                        {int(value) for value in episode[task_indices].tolist()}
                    ),
                    "episodes": len(plan["all_episodes_by_task"][task_id]),
                    "windows": int(task_indices.numel()),
                    "mask_stats": mask_stats(payload, task_indices),
                },
                "train": {
                    key: copy.deepcopy(train_item[key])
                    for key in ("episode_ids", "windows", "mask_stats")
                },
                "eval": {
                    key: copy.deepcopy(eval_item[key])
                    for key in ("episode_ids", "windows", "mask_stats")
                },
            }
        )

    manifest = {
        "contract": MANIFEST_CONTRACT,
        "contract_version": MANIFEST_VERSION,
        "manifest_hash_contract": MANIFEST_HASH_CONTRACT,
        "manifest_path": str(manifest_output.expanduser().resolve(strict=False)),
        "data_protocol": {
            "contract": data_protocol,
            "shape": {
                "sequence_length": expected_shape[0],
                "action_horizon": expected_shape[1],
                "action_dim": expected_shape[2],
            },
            "logged_action_chunk": (
                f"full_h{expected_shape[1]}"
                if data_protocol in PEER_SYNC_H6_CONTRACTS else "full_h48"
            ),
            **(
                {
                    key: copy.deepcopy((payload.get("metadata") or {})[key])
                    for key in (
                        "fps",
                        "planning_stride",
                        "control_stride",
                        "decision_offsets",
                        "action_label_offsets",
                        *(
                            ("world_target_horizon", "world_target_offsets")
                            if data_protocol == PEER_SYNC_H15_P2_CONTRACT
                            else ()
                        ),
                    )
                }
                if data_protocol in {
                    PEER_SYNC_H6_P2_CONTRACT,
                    PEER_SYNC_H15_P2_CONTRACT,
                }
                else {}
            ),
        },
        "source": {
            **source,
            "payload_output_identity": copy.deepcopy(
                (payload.get("metadata") or {}).get("output_identity")
            ),
            "payload_parent_identity": copy.deepcopy(
                (payload.get("metadata") or {}).get("parent_identity")
            ),
            "payload_source_identities": copy.deepcopy(
                (payload.get("metadata") or {}).get("source_identities")
            ),
            "n_windows": int(len(task)),
            "mask_stats": mask_stats(payload, all_indices),
        },
        "selection": {
            "unit": "episode_id",
            "stratify_by": "instruction_id",
            "heldout_fraction": float(heldout_fraction),
            "heldout_count_rule": "clamp(round_half_up(task_episodes*fraction),1,n-1)",
            "ranking": "sha256(contract,seed,task_id,episode_id)",
            "seed": int(seed),
        },
        "transition_rule": dict(
            PEER_SYNC_H15_P2_TRANSITION_RULE
            if data_protocol == PEER_SYNC_H15_P2_CONTRACT
            else PEER_SYNC_H6_P2_TRANSITION_RULE
            if data_protocol == PEER_SYNC_H6_P2_CONTRACT
            else TRANSITION_RULE
        ),
        "tasks": tasks,
        "splits": {"train": train, "eval": eval_split},
        "validation": {
            "episode_single_task": True,
            "episode_disjoint": True,
            "rows_disjoint": True,
            "rows_exhaustive": True,
            "full_logged_action_chunk": True,
        },
    }
    digest = canonical_manifest_sha256(manifest)
    manifest["manifest_id"] = f"wam4va-episode-holdout-v1-{digest[:16]}"
    manifest["manifest_sha256"] = digest
    return manifest


def _subset_payload(
    payload: dict,
    indices: Tensor,
    *,
    split_name: str,
    manifest: dict,
) -> dict:
    total = int(payload["actions"].shape[0])
    output = {}
    for key, value in payload.items():
        if key == "metadata":
            continue
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == total:
            output[key] = value.index_select(0, indices)
        elif isinstance(value, (list, tuple)) and len(value) == total:
            selected = [value[int(index)] for index in indices]
            output[key] = tuple(selected) if isinstance(value, tuple) else selected
        else:
            output[key] = value

    split = manifest["splits"][split_name]
    metadata = dict(payload.get("metadata") or {})
    for stale in (
        "split_contract",
        "split_name",
        "split_windows",
        "split_episode_ids",
        "split_task_counts",
        "source_n_windows",
    ):
        metadata.pop(stale, None)
    metadata.update(
        {
            "source_n_windows": total,
            "n_subset_windows": int(indices.numel()),
            "split_name": split_name,
            "split_windows": int(indices.numel()),
            "split_episode_ids": list(split["episode_ids"]),
            "split_task_counts": {
                int(item["task_id"]): int(item["windows"])
                for item in split["tasks"]
            },
            "split_manifest_path": manifest["manifest_path"],
            "split_manifest_id": manifest["manifest_id"],
            "split_manifest_sha256": manifest["manifest_sha256"],
            "split_contract": copy.deepcopy(manifest),
            "parent_identity": copy.deepcopy(manifest["source"]),
            "source_identities": copy.deepcopy(
                manifest["source"].get("payload_source_identities") or []
            ),
            "output_identity": copy.deepcopy(split["output_identity"]),
        }
    )
    output["metadata"] = metadata
    return output


def _target_and_temp(path: Path) -> tuple[Path, Path]:
    target = path.expanduser().resolve(strict=False)
    return target, target.with_name(f".{target.name}.tmp")


def _write_outputs_atomically(
    train_payload: dict,
    eval_payload: dict,
    manifest: dict,
    *,
    train_output: Path,
    eval_output: Path,
    manifest_output: Path,
) -> None:
    targets = [
        _target_and_temp(train_output),
        _target_and_temp(eval_output),
        _target_and_temp(manifest_output),
    ]
    resolved = [target for target, _ in targets]
    if len(set(resolved)) != len(resolved):
        raise ValueError("train, eval and manifest outputs must be distinct")
    for target, temporary in targets:
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {target}")
        if temporary.exists():
            raise FileExistsError(f"stale temporary output exists: {temporary}")
    for target, _ in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    temporaries = [temporary for _, temporary in targets]
    try:
        torch.save(train_payload, temporaries[0])
        torch.save(eval_payload, temporaries[1])
        temporaries[2].write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for (target, temporary) in targets:
            temporary.replace(target)
    finally:
        for temporary in temporaries:
            if temporary.exists():
                temporary.unlink()


def build_split_artifacts(
    input_path: Path,
    train_output: Path,
    eval_output: Path,
    manifest_output: Path,
    *,
    heldout_fraction: float = 0.10,
    seed: int = 0,
) -> dict:
    input_path = input_path.expanduser().resolve(strict=True)
    outputs = {
        path.expanduser().resolve(strict=False)
        for path in (train_output, eval_output, manifest_output)
    }
    if input_path in outputs:
        raise ValueError("input path cannot also be an output path")

    before = input_path.stat()
    source_sha256 = sha256_file(input_path)
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    after = input_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while reading: {input_path}")
    _validate_payload(payload)
    plan = build_split_plan(
        payload, heldout_fraction=heldout_fraction, seed=seed
    )
    source = {
        "path": str(input_path),
        "sha256": source_sha256,
        "size_bytes": int(after.st_size),
    }
    manifest = _build_manifest(
        payload,
        source,
        plan,
        train_output=train_output,
        eval_output=eval_output,
        manifest_output=manifest_output,
        heldout_fraction=heldout_fraction,
        seed=seed,
    )
    train_payload = _subset_payload(
        payload,
        plan["train_indices"],
        split_name="train",
        manifest=manifest,
    )
    eval_payload = _subset_payload(
        payload,
        plan["eval_indices"],
        split_name="eval",
        manifest=manifest,
    )
    _write_outputs_atomically(
        train_payload,
        eval_payload,
        manifest,
        train_output=train_output,
        eval_output=eval_output,
        manifest_output=manifest_output,
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_split_artifacts(
        args.input,
        args.train_output,
        args.eval_output,
        args.manifest_output,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
    )
    print(
        f"split {manifest['manifest_id']}: "
        f"train={manifest['splits']['train']['windows']} windows, "
        f"eval={manifest['splits']['eval']['windows']} windows; "
        f"manifest={manifest['manifest_path']}"
    )


if __name__ == "__main__":
    main()
