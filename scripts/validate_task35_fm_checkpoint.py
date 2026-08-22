#!/usr/bin/env python
"""CPU-only strict validator for task35 FM H6 VA checkpoints.

This is a mechanism / provenance gate, not closed-loop success evidence.
It never uses the GPU and does not start training or evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# CPU-only even if imported after a CUDA-capable interpreter starts.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metaworld import (
    MTVJ_METRIC_CONTRACT_VERSION,
    MTVJ_METRIC_STATE_SOURCE,
    _canonical_mtvj_metric_head_config,
    _load_dino_metric_from_policy,
)
from va_compound.metric_roi import DINO_METRIC_ROI_CONTRACT, TASK35_METRIC_ROLE_CONTRACT
from va_compound.model import VACompoundConfig, VACompoundPolicy

EXPECTED_DATA_SHA256 = "a27e4617da1c98cb326fbaefbb30183adf8761a3777dd83ceba7aa7845cdd9ec"
EXPECTED_RAW_FRAMES_SHA256 = (
    "d7699e9ef8e0ebfb1be3f120f990131dc360752b8f7034a39d3c18f33e9fd37e"
)
EXPECTED_FEATURE_SHA256 = {
    "block11.npy": "b8d3738dc1922a9276fa6e3173f126fdba2410aa46a1b9a1d57ea90b2280a471",
    "block23.npy": "ea740fb8b47ffe90f251904c79a6b1985ec04d2b41f7607c2a82b5398c2cedcc",
}
EXPECTED_ROI_SHA256 = "fca5ecd4e05b878bbe8930306727eaeb68744e4e66903e41ec75206664c47e5d"
EXPECTED_DINO_WEIGHT_SHA256 = (
    "c893d72294d4c327e631ff92f428dbc14c4f93cb5581b6c5f9d89bb5d17def27"
)
REQUIRED_KEYS = (
    "config",
    "model",
    "training_contract",
    "mtvj_metric_head",
    "mtvj_metric_head_config",
    "mtvj_relation_encoder",
    "dino_roi_checkpoint_identity",
    "global_step",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_state(state: dict[str, torch.Tensor], label: str) -> dict[str, int]:
    count = 0
    bad = []
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        count += int(tensor.numel())
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            bad.append(name)
    _require(not bad, f"{label} contains non-finite tensors: {bad[:8]}")
    return {"tensors": len(state), "params": count}


def validate_task35_fm_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_step: int | None = None,
    expected_data_sha256: str = EXPECTED_DATA_SHA256,
    expected_raw_frames_sha256: str = EXPECTED_RAW_FRAMES_SHA256,
    expected_feature_sha256: dict[str, str] | None = None,
    expected_roi_sha256: str = EXPECTED_ROI_SHA256,
    expected_dino_weight_sha256: str = EXPECTED_DINO_WEIGHT_SHA256,
    load_modules: bool = True,
) -> dict[str, Any]:
    """Validate a loaded task35 FM checkpoint in memory."""
    expected_feature_sha256 = expected_feature_sha256 or dict(EXPECTED_FEATURE_SHA256)
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    _require(not missing, f"checkpoint missing keys: {missing}")

    config = VACompoundConfig(**checkpoint["config"])
    contract = checkpoint.get("training_contract") or {}
    step = int(checkpoint["global_step"])
    if expected_step is not None:
        _require(step == int(expected_step), f"global_step {step} != {expected_step}")

    requirements = {
        "FM decoder": contract.get("action_decoder") == "conditional_flow_matching",
        "not Direct": config.direct_head is False and contract.get("action_decoder") != "direct_head",
        "World off": config.wmrm is False,
        "H6": int(config.action_horizon) == 6,
        "grid16": int(config.main_vision_grid) == 16,
        "four frames": int(config.main_vision_frames) == 4,
        "1024 tokens": int(config.main_vision_tokens) == 1024,
        "temporal": bool(config.main_vision_temporal),
        "geometry inject": bool(config.metric_geometry_inject),
        "DINO metric": bool(config.dino_dense_metric),
        "dense MT-VJ": bool(config.dense_readout_mtvj),
        "SDPA auto": config.va_attention_backend == "auto",
        "precision contract": contract.get("task35_precision_contract") is True,
        "role contract": contract.get("task35_metric_role_contract")
        == TASK35_METRIC_ROLE_CONTRACT,
        "ROI contract": contract.get("dino_roi_contract") == DINO_METRIC_ROI_CONTRACT,
        "ROI enabled": contract.get("dino_roi_enabled") is True
        and float(contract.get("dino_roi_alpha") or 0.0) == 1.0,
        "metric source": contract.get("metric_state_source") == MTVJ_METRIC_STATE_SOURCE,
        "metric version": int(contract.get("metric_contract_version") or 0)
        == MTVJ_METRIC_CONTRACT_VERSION,
        "data sha": contract.get("task35_data_sha256") == expected_data_sha256,
        "raw-frame sha": contract.get("task35_raw_frames_sha256")
        == expected_raw_frames_sha256,
        "feature sha": contract.get("task35_dino_feature_sha256")
        == expected_feature_sha256,
        "DINO weight sha": contract.get("main_vision_checkpoint_sha256")
        == expected_dino_weight_sha256,
        "ROI identity": (checkpoint.get("dino_roi_checkpoint_identity") or {}).get("sha256")
        == expected_roi_sha256,
        "aux cadence": int(contract.get("mtvj_visual_aux_every") or 0) > 0
        and int(contract.get("mtvj_visual_aux_batch") or 0) > 0,
        "exact resume": int(checkpoint.get("exact_resume_version") or 0) >= 1,
    }
    failed = [name for name, ok in requirements.items() if not ok]
    _require(not failed, "task35 FM checkpoint failed: " + ", ".join(failed))

    model_stats = _finite_state(checkpoint["model"], "policy")
    metric_stats = _finite_state(checkpoint["mtvj_metric_head"], "metric_head")
    relation_stats = _finite_state(checkpoint["mtvj_relation_encoder"], "relation")
    _require(
        "main_vision_frame_embedding.weight" in checkpoint["model"],
        "missing temporal frame embeddings",
    )
    _require(
        "geometry_projection.weight" in checkpoint["model"],
        "missing geometry projection",
    )
    geometry = checkpoint["model"]["geometry_projection.weight"]
    _require(
        tuple(geometry.shape) == (int(config.hidden_dim), 8),
        f"geometry projection shape {tuple(geometry.shape)} != ({config.hidden_dim}, 8)",
    )
    if step >= 1000:
        _require(
            bool(torch.count_nonzero(geometry)),
            "geometry projection is still all zeros after 1000 steps",
        )

    ctor = _canonical_mtvj_metric_head_config(
        checkpoint["mtvj_metric_head_config"], require_complete=True
    )
    _require(int(ctor["h_dim"]) == 1024 and int(ctor["grid"]) == 16, "metric head ctor mismatch")
    _require(int(ctor["n_roles"]) == 4, "metric head must have four roles")

    loaded = False
    if load_modules:
        model = VACompoundPolicy(config)
        incompatible = model.load_state_dict(checkpoint["model"], strict=True)
        _require(
            not incompatible.missing_keys and not incompatible.unexpected_keys,
            f"policy load mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}",
        )
        _require(model.main_vision_frame_embedding is not None, "temporal embedding not built")
        _require(model.geometry_projection is not None, "geometry projection not built")
        _load_dino_metric_from_policy(checkpoint, config, torch.device("cpu"))
        loaded = True

    return {
        "ok": True,
        "global_step": step,
        "action_decoder": contract.get("action_decoder"),
        "task": "peg-insert-side-v3",
        "loaded_modules": loaded,
        "model": model_stats,
        "metric_head": metric_stats,
        "relation_encoder": relation_stats,
        "geometry_l2": float(geometry.detach().float().square().mean().sqrt()),
        "checks": requirements,
    }


def validate_task35_fm_checkpoint_path(
    path: Path,
    *,
    expected_step: int | None = None,
    sidecar: bool = True,
    load_modules: bool = True,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    if sidecar:
        sidecar_path = Path(str(resolved) + ".sha256")
        if sidecar_path.is_file():
            recorded = sidecar_path.read_text().split()[0]
            _require(recorded == digest, f"sidecar SHA mismatch: {sidecar_path}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    report = validate_task35_fm_checkpoint(
        checkpoint, expected_step=expected_step, load_modules=load_modules
    )
    report["path"] = str(resolved)
    report["sha256"] = digest
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-module-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        validate_task35_fm_checkpoint_path(
            path,
            expected_step=args.expected_step,
            load_modules=not args.skip_module_load,
        )
        for path in args.checkpoints
    ]
    payload = {"contract": "task35_fm_checkpoint_validate_v1", "reports": reports}
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
