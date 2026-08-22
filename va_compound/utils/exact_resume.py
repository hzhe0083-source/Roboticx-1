"""Deterministic exact-resume contract and checkpoint-state serialization.

These helpers freeze every runtime semantic (CLI, data identity, model config,
optimizer) into a comparable ``exact_run_contract`` and serialize the training
state (optimizer/sampler/RNG) needed to continue a run bit-for-bit.

The module is intentionally light on imports: ``SAM``, the task samplers and
``validate_optimizer_update_state`` live in ``train.py``, so they are imported
lazily inside the few functions that touch them to avoid a circular import
(train.py imports this module at top level).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from train import (
        SAM,
        TaskLocalityWeightedSampler,
        TaskWeightedSampler,
    )


EXACT_RESUME_VERSION = 2
EXACT_RUN_CONTRACT_VERSION = 1
WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION = (
    "wmrm_detach_proposal_stage_state_v1"
)
WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION = "wmrm_world_weight_1_to_0_5_v1"
WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION = (
    "wmrm_static_constraint_weight_4_to_2_v1"
)
WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION = "wmrm_action_rank_cap_none_to_0_2_v1"

# These only control how long/where the already-defined run is executed. They
# cannot change the next stochastic optimizer update and therefore are allowed
# to differ when an exact run is continued.
_EXACT_RUN_OPERATIONAL_ARGS = {
    "steps",
    "save",
    "save_every",
    "save_step_copies",
    "resume",
    "resume_exact",
    "resume_weights",
    # This selects a narrowly controlled compatibility policy for validation; it
    # does not become part of the semantic run contract saved after migration.
    "resume_exact_contract_migration",
    # Content/config identity is recorded separately, so an identical external
    # metric checkpoint copied to another filename is still the same input.
    "metric_visual_checkpoint",
    "mtvj_roi_checkpoint",
    # One-shot initialization migration.  The resulting head constructor,
    # weights and content identity are checkpointed, so replaying this flag is
    # neither necessary nor allowed during an exact continuation.
    "replace_mtvj_metric_head_from_external",
    # Dataset paths are spelling only; full content identities are stored in
    # data_identity and peer_world.{va_data_identity,world_data_identity}.
    "data",
    "va_data",
    "world_data",
}


def _normalize_contract_value(value):
    """Convert runtime objects into deterministic weights-only-safe values."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, dict):
        return {
            str(key): _normalize_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported exact-run contract value {type(value).__name__}: {value!r}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampled_file_identity(path: Path) -> dict:
    """Cheap identity for referenced JPEG containers, honestly marked sampled.

    The windows payload itself gets a full SHA-256.  Its referenced longtraj
    containers can total tens of GiB, so we combine exact stat metadata with
    deterministic beginning/middle/end samples instead of pretending to have
    hashed every JPEG byte.
    """
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    block_bytes = 256 * 1024
    if stat.st_size <= 3 * block_bytes:
        offsets = [0]
    else:
        offsets = [0, max(0, (stat.st_size - block_bytes) // 2), stat.st_size - block_bytes]
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            block = stream.read(block_bytes)
            digest.update(int(offset).to_bytes(8, "little", signed=False))
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sampled_sha256": digest.hexdigest(),
        "sample_offsets": offsets,
        "sample_bytes": block_bytes,
    }


def _payload_schema(payload: dict) -> dict:
    """Record the explicit tensor/content schema protected by the file hash."""
    tensors = {}
    non_tensors = {}
    for key, value in sorted(payload.items()):
        if isinstance(value, Tensor):
            tensors[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
        elif key == "metadata" and isinstance(value, dict):
            # Metadata is small and changes data semantics (task names, strides,
            # horizon); store it explicitly as well as protecting it by SHA-256.
            non_tensors[key] = _normalize_contract_value(value)
        elif isinstance(value, (list, tuple)):
            non_tensors[key] = {
                "container": type(value).__name__,
                "length": len(value),
            }
        else:
            non_tensors[key] = {"type": type(value).__name__}
    return {"tensors": tensors, "non_tensors": non_tensors}


def build_dataset_content_identity(
    path: str | Path,
    payload: dict,
    *,
    longtraj_dir: str | Path | None = None,
) -> dict:
    """Strong identity for the MT-VJ windows payload, computed once per run.

    ``full_file_sha256`` covers actions, masks, frame references, language and
    metadata in the actual serialized payload—not only sampler task/episode IDs.
    Referenced longtraj JPEG containers use a clearly labelled sampled identity
    plus exact path/stat metadata to keep startup bounded.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    before_key = (before.st_size, before.st_mtime_ns)
    after_key = (after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise RuntimeError(f"dataset changed while fingerprinting: {resolved}")

    sources = []
    refs = payload.get("frame_refs")
    if isinstance(refs, (list, tuple)) and refs:
        root = (
            Path(longtraj_dir).expanduser().resolve(strict=False)
            if longtraj_dir is not None
            else resolved.parent
        )
        task_files = sorted({str(ref[0]) for ref in refs})
        for task_file in task_files:
            raw = Path(task_file)
            if raw.is_absolute():
                source = raw
            elif raw.suffix == ".pt" and (root / raw).exists():
                source = root / raw
            else:
                source = root / f"metaworld_longtraj_{task_file}.pt"
            if source.exists():
                sources.append(_sampled_file_identity(source))
            else:
                sources.append(
                    {
                        "path": str(source.resolve(strict=False)),
                        "missing": True,
                    }
                )

    return {
        "identity_algorithm": "full_payload_sha256+referenced_source_sample_v1",
        "resolved_path": str(resolved),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "full_file_sha256": digest,
        "payload_schema": _payload_schema(payload),
        "referenced_sources": sources,
    }


def _optimizer_contract(optimizer: torch.optim.Optimizer) -> dict:
    from train import SAM

    kind = "sam_adamw" if isinstance(optimizer, SAM) else "adamw"
    groups = []
    for group in optimizer.param_groups:
        parameters = list(group["params"])
        groups.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": list(group.get("betas", ())),
                "eps": float(group.get("eps", 0.0)),
                "amsgrad": bool(group.get("amsgrad", False)),
                "rho": float(group.get("rho", 0.0)),
                "parameter_count": len(parameters),
                "parameter_numel": int(sum(parameter.numel() for parameter in parameters)),
                "parameter_signature": [
                    {
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "requires_grad": bool(parameter.requires_grad),
                    }
                    for parameter in parameters
                ],
            }
        )
    return {"kind": kind, "groups": groups}


def exact_run_contract_mismatches(saved: dict, current: dict) -> list[tuple[str, object, object]]:
    """Return deterministic leaf-level differences for a clear failure message."""
    mismatches: list[tuple[str, object, object]] = []
    missing = "<missing>"

    def compare(left, right, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else str(key)
                if key not in left:
                    mismatches.append((child, missing, right[key]))
                elif key not in right:
                    mismatches.append((child, left[key], missing))
                else:
                    compare(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append((f"{path}.length", len(left), len(right)))
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if type(left) is not type(right) or left != right:
            mismatches.append((path, left, right))

    compare(_normalize_contract_value(saved), _normalize_contract_value(current), "")
    return mismatches


def _normalize_legacy_exact_run_contract(contract: dict) -> dict:
    """Fill only defaults that are known for contracts predating these fields."""
    normalized = _normalize_contract_value(contract)
    arguments = normalized.get("arguments")
    if isinstance(arguments, dict):
        arguments.setdefault("wmrm_detach_proposal_stage_state", False)
        arguments.setdefault("wmrm_static_constraint_weight", 4.0)
        arguments.setdefault("wmrm_action_rank_per_sample_cap", None)
        arguments.setdefault("wmrm_late_stage_anchor_weight", 0.0)
        arguments.setdefault("wmrm_stage_s5_weight", None)
        arguments.setdefault("wmrm_stage_s6_weight", None)
        arguments.setdefault("lr_wmrm_predictor", None)
        arguments.setdefault("wmrm_predictor_grad_clip", None)
        arguments.setdefault("max_gradient_norm", None)
    model_config = normalized.get("model_config")
    if isinstance(model_config, dict):
        model_config.setdefault("wmrm_detach_proposal_stage_state", False)
        # Checkpoints predating the peer core are exactly the legacy topology.
        # Normalizing this absent field is compatibility, not a migration to peer.
        model_config.setdefault("va_world_mode", "legacy")
    if isinstance(arguments, dict):
        arguments.setdefault("va_world_mode", "legacy")
    return normalized


def validate_exact_run_contract(
    saved: dict | None,
    current: dict,
    *,
    migration_id: str | None = None,
) -> None:
    if saved is None:
        raise ValueError(
            "--resume-exact checkpoint is missing exact_run_contract; "
            "use --resume for legacy weights-only loading"
        )
    normalized_saved = _normalize_legacy_exact_run_contract(saved)
    normalized_current = _normalize_legacy_exact_run_contract(current)
    mismatches = exact_run_contract_mismatches(normalized_saved, normalized_current)
    if migration_id == WMRM_WORLD_WEIGHT_1_TO_0_5_MIGRATION:
        saved_target, current_target = 1.0, 0.5
        allowed_paths = {"arguments.wmrm_world_weight"}
        saved_arguments = normalized_saved.get("arguments")
        current_arguments = normalized_current.get("arguments")
        saved_weight = (
            saved_arguments.get("wmrm_world_weight")
            if isinstance(saved_arguments, dict)
            else None
        )
        current_weight = (
            current_arguments.get("wmrm_world_weight")
            if isinstance(current_arguments, dict)
            else None
        )
        coherent = (
            isinstance(saved_arguments, dict)
            and isinstance(current_arguments, dict)
            and type(saved_weight) is float
            and type(current_weight) is float
            and saved_weight == saved_target
            and current_weight == current_target
            and {path for path, _, _ in mismatches} == allowed_paths
            and all(
                type(left) is float
                and type(right) is float
                and left == saved_target
                and right == current_target
                for path, left, right in mismatches
            )
        )
        if coherent:
            return
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        ) or f"no coherent old-{saved_target} to new-{current_target} transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if migration_id == WMRM_ACTION_RANK_CAP_NONE_TO_0_2_MIGRATION:
        allowed_paths = {"arguments.wmrm_action_rank_per_sample_cap"}
        saved_arguments = normalized_saved.get("arguments")
        current_arguments = normalized_current.get("arguments")
        saved_cap = (
            saved_arguments.get("wmrm_action_rank_per_sample_cap")
            if isinstance(saved_arguments, dict)
            else "<missing>"
        )
        current_cap = (
            current_arguments.get("wmrm_action_rank_per_sample_cap")
            if isinstance(current_arguments, dict)
            else "<missing>"
        )
        coherent = (
            isinstance(saved_arguments, dict)
            and isinstance(current_arguments, dict)
            and saved_cap is None
            and type(current_cap) is float
            and current_cap == 0.2
            and type(saved_arguments.get("wmrm_static_constraint_weight")) is float
            and saved_arguments.get("wmrm_static_constraint_weight") == 2.0
            and current_arguments.get("wmrm_static_constraint_weight") == 2.0
            and type(saved_arguments.get("wmrm_world_weight")) is float
            and saved_arguments.get("wmrm_world_weight") == 1.0
            and current_arguments.get("wmrm_world_weight") == 1.0
            and saved_arguments.get("wmrm_detach_proposal_stage_state") is True
            and current_arguments.get("wmrm_detach_proposal_stage_state") is True
            and {path for path, _, _ in mismatches} == allowed_paths
            and all(
                left is None and type(right) is float and right == 0.2
                for path, left, right in mismatches
            )
        )
        if coherent:
            return
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        ) or "no coherent static2/world1/detached action-rank cap None to 0.2 transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if migration_id == WMRM_STATIC_CONSTRAINT_WEIGHT_4_TO_2_MIGRATION:
        allowed_paths = {"arguments.wmrm_static_constraint_weight"}
        saved_arguments = normalized_saved.get("arguments")
        current_arguments = normalized_current.get("arguments")
        saved_weight = (
            saved_arguments.get("wmrm_static_constraint_weight")
            if isinstance(saved_arguments, dict)
            else None
        )
        current_weight = (
            current_arguments.get("wmrm_static_constraint_weight")
            if isinstance(current_arguments, dict)
            else None
        )
        coherent = (
            isinstance(saved_arguments, dict)
            and isinstance(current_arguments, dict)
            and type(saved_weight) is float
            and type(current_weight) is float
            and saved_weight == 4.0
            and current_weight == 2.0
            and saved_arguments.get("wmrm_world_weight") == 1.0
            and current_arguments.get("wmrm_world_weight") == 1.0
            and saved_arguments.get("wmrm_detach_proposal_stage_state") is True
            and current_arguments.get("wmrm_detach_proposal_stage_state") is True
            and {path for path, _, _ in mismatches} == allowed_paths
            and all(
                type(left) is float
                and type(right) is float
                and left == 4.0
                and right == 2.0
                for path, left, right in mismatches
            )
        )
        if coherent:
            return
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        ) or "no coherent static-constraint 4.0 to 2.0 transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if migration_id is not None:
        if migration_id != WMRM_DETACH_PROPOSAL_STAGE_STATE_MIGRATION:
            raise ValueError(
                "unsupported --resume-exact-contract-migration: "
                f"{migration_id!r}"
            )
        allowed_paths = {
            "arguments.wmrm_detach_proposal_stage_state",
            "model_config.wmrm_detach_proposal_stage_state",
        }
        # The migration changes one semantic flag, but the exact contract stores
        # it in two representations.  Normalize legacy omissions to False, then
        # require both sides to be coherent before allowing the transition.  In
        # particular, never accept a partial/contradictory transition where only
        # one representation changes (or where either contract is internally
        # inconsistent).
        saved_arguments = normalized_saved.get("arguments")
        saved_model_config = normalized_saved.get("model_config")
        current_arguments = normalized_current.get("arguments")
        current_model_config = normalized_current.get("model_config")
        detach_paths = (
            (saved_arguments, "arguments"),
            (saved_model_config, "model_config"),
            (current_arguments, "arguments"),
            (current_model_config, "model_config"),
        )
        if not all(isinstance(value, dict) for value, _ in detach_paths):
            details = "arguments and model_config must both be mappings"
        else:
            saved_values = (
                saved_arguments["wmrm_detach_proposal_stage_state"],
                saved_model_config["wmrm_detach_proposal_stage_state"],
            )
            current_values = (
                current_arguments["wmrm_detach_proposal_stage_state"],
                current_model_config["wmrm_detach_proposal_stage_state"],
            )
            coherent = (
                saved_values == (False, False)
                and current_values == (True, True)
                and {path for path, _, _ in mismatches} == allowed_paths
                and all(left is False and right is True for path, left, right in mismatches)
            )
            if coherent:
                return
            details = "; ".join(
                f"{path}: checkpoint={left!r}, runtime={right!r}"
                for path, left, right in mismatches[:12]
            ) or "no coherent old-false-both to new-true-both transition"
        raise ValueError(
            f"controlled exact-resume migration {migration_id!r} refused: {details}"
        )
    if mismatches:
        details = "; ".join(
            f"{path}: checkpoint={left!r}, runtime={right!r}"
            for path, left, right in mismatches[:12]
        )
        if len(mismatches) > 12:
            details += f"; ... and {len(mismatches) - 12} more"
        raise ValueError(f"--resume-exact exact_run_contract mismatch: {details}")


def capture_rng_state(*, include_cuda: bool = True) -> dict:
    """Capture global RNGs using only weights-only-safe primitives/tensors."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            # torch 2.2 cannot construct tensors from NumPy uint32 arrays.
            # MT19937 words fit losslessly in int64 and are cast back on restore.
            "state": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if include_cuda and torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNGs captured by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"exact resume RNG state missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("exact resume checkpoint contains CUDA RNG but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "exact resume CUDA device count mismatch: "
                f"checkpoint={len(cuda_states)} runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def exact_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    """Serialize AdamW, including SAM's real base optimizer state."""
    from train import SAM

    if isinstance(optimizer, SAM):
        if not isinstance(optimizer.base_optimizer, torch.optim.AdamW):
            raise TypeError("--resume-exact only supports SAM with an AdamW base optimizer")
        if any("pre_perturbation" in value for value in optimizer.state.values()):
            raise RuntimeError("cannot checkpoint exact state between SAM first_step/second_step")
        return {
            "kind": "sam_adamw",
            "state_dict": optimizer.base_optimizer.state_dict(),
            "rho": [float(group["rho"]) for group in optimizer.param_groups],
        }
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("--resume-exact currently supports AdamW (or SAM over AdamW) only")
    return {"kind": "adamw", "state_dict": optimizer.state_dict()}


def restore_exact_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved: dict,
) -> None:
    """Strictly restore an optimizer produced by :func:`exact_optimizer_state_dict`."""
    from train import SAM, validate_optimizer_update_state

    expected_kind = "sam_adamw" if isinstance(optimizer, SAM) else "adamw"
    if saved.get("kind") != expected_kind:
        raise ValueError(
            "exact resume optimizer mismatch: "
            f"checkpoint={saved.get('kind')!r} runtime={expected_kind!r}"
        )
    if isinstance(optimizer, SAM):
        saved_rho = [float(value) for value in saved.get("rho", [])]
        runtime_rho = [float(group["rho"]) for group in optimizer.param_groups]
        if saved_rho != runtime_rho:
            raise ValueError(
                f"exact resume SAM rho mismatch: {saved_rho!r} != {runtime_rho!r}"
            )
        optimizer.base_optimizer.load_state_dict(saved["state_dict"])
        optimizer.param_groups = optimizer.base_optimizer.param_groups
    else:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("--resume-exact currently supports AdamW only")
        optimizer.load_state_dict(saved["state_dict"])
    validate_optimizer_update_state(optimizer)


def build_exact_resume_state(
    optimizer: torch.optim.Optimizer,
    global_step: int,
    sampler: "TaskLocalityWeightedSampler | TaskWeightedSampler | None",
    exact_run_contract: dict,
) -> dict:
    """Build the training-state portion of an exact-resumable checkpoint."""
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    uses_cuda = any(
        parameter.is_cuda
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    return {
        "exact_resume_version": EXACT_RESUME_VERSION,
        "global_step": int(global_step),
        "optimizer_state": exact_optimizer_state_dict(optimizer),
        "sampler_state": sampler.state_dict() if sampler is not None else None,
        "rng_state": capture_rng_state(include_cuda=uses_cuda),
        "exact_run_contract": _normalize_contract_value(exact_run_contract),
    }


def restore_exact_resume_state(
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    sampler: "TaskLocalityWeightedSampler | TaskWeightedSampler | None",
    *,
    runtime_exact_run_contract: dict | None = None,
    migration_id: str | None = None,
    restore_rng: bool = True,
) -> int:
    """Restore optimizer/sampler and optionally RNG; return completed updates."""
    from train import TaskLocalityWeightedSampler

    required = {
        "exact_resume_version",
        "global_step",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "exact_run_contract",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "--resume-exact requires a new exact checkpoint; missing keys: "
            f"{sorted(missing)}. Use --resume for legacy weights-only loading."
        )
    if checkpoint["exact_resume_version"] != EXACT_RESUME_VERSION:
        raise ValueError(
            "unsupported exact resume version: "
            f"{checkpoint['exact_resume_version']} != {EXACT_RESUME_VERSION}"
        )
    if runtime_exact_run_contract is not None:
        # Contract comparison intentionally precedes every mutable restore below.
        validate_exact_run_contract(
            checkpoint["exact_run_contract"],
            runtime_exact_run_contract,
            migration_id=migration_id,
        )
    saved_sampler = checkpoint["sampler_state"]
    if saved_sampler is None:
        if isinstance(sampler, TaskLocalityWeightedSampler):
            raise ValueError(
                "--resume-exact checkpoint has sampler_state=None but the "
                "runtime built a sampler; refuse to invent locality state"
            )
        # TaskWeightedSampler or no sampler: 6k / this 6k→20k run stored None.
        # Keep epoch=0 rather than inferring a cursor from global_step.
    elif sampler is None:
        raise ValueError("--resume-exact requires a checkpointed sampler state")
    restore_exact_optimizer_state(optimizer, checkpoint["optimizer_state"])
    if saved_sampler is not None:
        sampler.load_state_dict(saved_sampler)
    global_step = int(checkpoint["global_step"])
    if global_step < 0:
        raise ValueError(f"invalid exact checkpoint global_step: {global_step}")
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return global_step


def final_checkpoint_save_due(
    save_path: Path | None,
    global_step: int,
    last_saved_global_step: int | None,
) -> bool:
    """Return whether the completed run still needs its final checkpoint save."""
    return save_path is not None and last_saved_global_step != global_step
