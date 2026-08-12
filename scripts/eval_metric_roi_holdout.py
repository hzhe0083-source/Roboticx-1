#!/usr/bin/env python
"""Paired all-49 held-out evaluation for the standalone MT-VJ ROI refiner.

Every coarse/ROI comparison uses the exact same freshly generated simulator
sample.  ``--alpha 1`` is mandatory so a report can never accidentally measure
the checkpoint's lossless default gate (alpha=0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from prepare_metaworld_metric import SUPPORTED_TASKS, make_metric_batch  # noqa: E402
from scripts.build_longtraj_features import ENV_TO_TASK  # noqa: E402
from scripts.eval_metric_visual_holdout import (  # noqa: E402
    compute_gated_state_metrics,
    compute_localization_metrics,
    compute_visibility_metrics,
)
from scripts.mt50_difficulty import DEFAULT_WEIGHT, TASK_WEIGHTS  # noqa: E402
from train_metric_roi import (  # noqa: E402
    COARSE_SOURCE_POLICY,
    COARSE_SOURCE_STAGE_V,
    CONTRACT,
    IMAGE_SIZE,
    build_frozen_coarse_and_roi,
    load_coarse_source,
    preprocess_raw_full_frames,
    preprocess_raw_roi_frames,
    sha256_file,
    validate_coarse_checkpoint,
)
from train_metric_visual import (  # noqa: E402
    build_language_cache,
    gather_language,
)
from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone  # noqa: E402
from va_compound.live_vjepa import _dense_coords  # noqa: E402
from va_compound.metric_roi import (  # noqa: E402
    gt_crop_visibility,
    merge_roi_refinement,
    metric_head_state_sha256,
    plan_metric_roi,
)


DIFFICULTY = {0.5: "easy", 1.0: "medium", 2.0: "hard", 3.0: "very_hard"}


def difficulty_for_task(task: str) -> str:
    weight = float(TASK_WEIGHTS.get(ENV_TO_TASK.get(task, task), DEFAULT_WEIGHT))
    return DIFFICULTY[weight]


def validate_roi_checkpoint(
    checkpoint: Mapping[str, Any],
    coarse_checkpoint: Mapping[str, Any],
    coarse_sha256: str,
    coarse_head_state_sha256: str,
    *,
    coarse_source: str = COARSE_SOURCE_STAGE_V,
) -> Mapping[str, Any]:
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError(
            f"ROI contract={checkpoint.get('contract')!r}; expected {CONTRACT!r}"
        )
    required = {"config", "coarse", "roi_metric_head"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"ROI checkpoint missing keys: {missing}")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("ROI config must be a mapping")
    coarse_meta = checkpoint["coarse"]
    if not isinstance(coarse_meta, Mapping):
        raise ValueError("ROI coarse identity must be a mapping")
    coarse_config = dict(validate_coarse_checkpoint(coarse_checkpoint))
    if coarse_meta.get("sha256") != coarse_sha256:
        raise ValueError("provided coarse checkpoint does not match ROI SHA-256 identity")
    if coarse_meta.get("config") != coarse_config:
        raise ValueError("provided coarse config does not match ROI training identity")
    if config.get("coarse_sha256") != coarse_sha256:
        raise ValueError("ROI config/coarse SHA-256 fields disagree")
    if coarse_meta.get("coarse_head_state_sha256") != coarse_head_state_sha256:
        raise ValueError("provided coarse metric-head state does not match ROI identity")
    if config.get("coarse_head_state_sha256") != coarse_head_state_sha256:
        raise ValueError("ROI config/coarse metric-head state SHA fields disagree")
    saved_source = coarse_meta.get(
        "source", config.get("coarse_source", COARSE_SOURCE_STAGE_V)
    )
    if config.get("coarse_source", saved_source) != saved_source:
        raise ValueError("ROI config/coarse source fields disagree")
    if saved_source != coarse_source:
        raise ValueError(
            f"provided coarse source {coarse_source!r} does not match ROI "
            f"training source {saved_source!r}"
        )
    if coarse_meta.get("contract") != coarse_checkpoint.get("contract"):
        raise ValueError("provided runtime metric contract does not match ROI identity")
    runtime_identity = coarse_meta.get("runtime_metric_checkpoint")
    if runtime_identity is None and saved_source == COARSE_SOURCE_POLICY:
        raise ValueError("policy-sourced ROI checkpoint lacks runtime metric identity")
    if runtime_identity is not None:
        if not isinstance(runtime_identity, Mapping):
            raise ValueError("ROI runtime metric identity must be a mapping")
        expected_runtime = {
            "sha256": coarse_sha256,
            "contract": coarse_checkpoint.get("contract"),
        }
        mismatched = {
            key: (runtime_identity.get(key), expected)
            for key, expected in expected_runtime.items()
            if runtime_identity.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                f"provided runtime metric checkpoint does not match ROI identity: {mismatched}"
            )
    if float(config.get("alpha_default", -1.0)) != 0.0:
        raise ValueError("ROI checkpoint must preserve alpha_default=0")
    if config.get("role_pairs") != [[0, 1], [3, 2]]:
        raise ValueError("ROI role-pair semantics are incompatible")
    if config.get("canonical_image_size") != IMAGE_SIZE:
        raise ValueError("ROI canonical image size must be 384")
    return config


def paired_metrics(
    coarse_p: np.ndarray,
    roi_p: np.ndarray,
    targets: np.ndarray,
    visibility: np.ndarray,
    coarse_visibility: np.ndarray,
    roi_visibility: np.ndarray,
) -> dict[str, Any]:
    return {
        "coarse": {
            "localization": compute_localization_metrics(coarse_p, targets, visibility),
            "visibility": compute_visibility_metrics(coarse_visibility, visibility),
            "gated_state": compute_gated_state_metrics(
                coarse_p, coarse_visibility, targets, visibility
            ),
        },
        "roi": {
            "localization": compute_localization_metrics(roi_p, targets, visibility),
            "visibility": compute_visibility_metrics(roi_visibility, visibility),
            "gated_state": compute_gated_state_metrics(
                roi_p, roi_visibility, targets, visibility
            ),
        },
    }


def selection_diagnostics(record: Mapping[str, np.ndarray]) -> dict[str, Any]:
    confidence = np.asarray(record["selection_confidence"], dtype=np.float64).reshape(-1)
    visible_count = np.asarray(
        record["selected_gt_crop_visible"], dtype=np.float64
    ).reshape(-1)
    return {
        "confidence_mean": float(confidence.mean()),
        "confidence_median": float(np.median(confidence)),
        "confidence_p10": float(np.quantile(confidence, 0.10)),
        "gt_crop_visible_roles_mean": float(visible_count.mean()),
        "usable_pair_fraction": float(np.mean(visible_count > 0.0)),
        "both_roles_crop_visible_fraction": float(np.mean(visible_count >= 2.0)),
    }


def _relative_reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return 1.0 - float(after) / float(before)


def adoption_gate(metrics: Mapping[str, Any], grouped: Mapping[str, Any]) -> dict[str, Any]:
    coarse_loc = metrics["coarse"]["localization"]
    roi_loc = metrics["roi"]["localization"]
    coarse_all = coarse_loc["aggregate"]
    roi_all = roi_loc["aggregate"]
    rmse_gain = _relative_reduction(coarse_all["rmse_px"], roi_all["rmse_px"])
    pck10_gain = (
        float(roi_all["pck@10px"]) - float(coarse_all["pck@10px"])
        if roi_all["pck@10px"] is not None and coarse_all["pck@10px"] is not None
        else None
    )
    role_reductions = {
        role: _relative_reduction(
            coarse_loc["roles"][role]["rmse_px"],
            roi_loc["roles"][role]["rmse_px"],
        )
        for role in coarse_loc["roles"]
    }
    role_ok = all(value is not None and value >= -0.05 for value in role_reductions.values())
    hard_group_reductions = {
        name: _relative_reduction(
            grouped[name]["coarse"]["localization"]["aggregate"]["rmse_px"],
            grouped[name]["roi"]["localization"]["aggregate"]["rmse_px"],
        )
        for name in ("hard", "very_hard")
    }
    hard_ok = all(
        value is not None and value >= 0.0 for value in hard_group_reductions.values()
    )
    coarse_gated = metrics["coarse"]["gated_state"]["aggregate"]["rmse_px"]
    roi_gated = metrics["roi"]["gated_state"]["aggregate"]["rmse_px"]
    checks = {
        "rmse_reduction_at_least_15pct": rmse_gain is not None and rmse_gain >= 0.15,
        "pck10_gain_at_least_10pp": pck10_gain is not None and pck10_gain >= 0.10,
        "hard_and_very_hard_do_not_regress": hard_ok,
        "every_role_regression_at_most_5pct": role_ok,
        "policy_gated_state_does_not_regress": roi_gated <= coarse_gated,
    }
    return {
        "adopt": all(checks.values()),
        "checks": checks,
        "rmse_reduction": rmse_gain,
        "pck10_gain": pck10_gain,
        "role_rmse_reduction": role_reductions,
        "difficulty_rmse_reduction": hard_group_reductions,
    }


def _concat(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([record[key] for record in records], axis=0) for key in records[0]}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.alpha is None or float(args.alpha) != 1.0:
        raise ValueError("paired ROI evaluation requires explicit --alpha 1")
    if args.samples_per_task <= 0 or args.batch_size <= 0:
        raise ValueError("samples-per-task and batch-size must be positive")
    roi_path = Path(args.roi_checkpoint).resolve()
    roi_sha = sha256_file(roi_path)
    coarse_path, coarse_checkpoint, coarse_source_record = load_coarse_source(
        coarse_checkpoint_path=getattr(args, "coarse_checkpoint", None),
        coarse_policy_checkpoint_path=getattr(
            args, "coarse_policy_checkpoint", None
        ),
        runtime_metric_checkpoint_path=getattr(
            args, "runtime_metric_checkpoint", None
        ),
    )
    coarse_sha = coarse_source_record["runtime_metric_checkpoint"]["sha256"]
    coarse_source = coarse_source_record["kind"]
    roi_checkpoint = torch.load(roi_path, map_location="cpu", weights_only=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    coarse_head, roi_head, ctor = build_frozen_coarse_and_roi(coarse_checkpoint, device)
    coarse_head_sha = metric_head_state_sha256(coarse_head)
    config = validate_roi_checkpoint(
        roi_checkpoint,
        coarse_checkpoint,
        coarse_sha,
        coarse_head_sha,
        coarse_source=coarse_source,
    )
    if coarse_head_sha != coarse_source_record["actual_coarse_head_state_sha256"]:
        raise RuntimeError("loaded coarse metric-head state SHA changed unexpectedly")
    if int(args.seed) == int(config.get("seed", -1)):
        raise ValueError("held-out seed must differ from ROI training seed")
    if dict(config.get("head_ctor", {})) != ctor:
        raise ValueError("ROI head constructor differs from coarse constructor")
    roi_head.load_state_dict(roi_checkpoint["roi_metric_head"], strict=True)
    roi_head.requires_grad_(False).eval()
    dtype = "float16" if device.type == "cuda" else "float32"
    vision = VJEPA21Backbone.from_pretrained(
        device=device, dtype=dtype, local_files_only=True
    )
    vision.freeze_all()
    vision.eval()
    text = QwenTextBackbone.from_pretrained(
        device=device, dtype=dtype, local_files_only=True
    )
    text.requires_grad_(False).eval()
    language_cache, ok = build_language_cache(
        text, [ENV_TO_TASK[task] for task in SUPPORTED_TASKS]
    )
    if not ok:
        raise RuntimeError("Qwen language cache degraded")
    del text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    coords = torch.from_numpy(_dense_coords()).to(device)

    per_task: dict[str, Any] = {}
    task_records: dict[str, dict[str, np.ndarray]] = {}
    with torch.inference_mode():
        for task_index, task in enumerate(SUPPORTED_TASKS):
            rng = np.random.default_rng(
                np.random.SeedSequence((int(args.seed), int(task_index), 0xA701))
            )
            chunks: list[dict[str, np.ndarray]] = []
            generated = 0
            while generated < args.samples_per_task:
                count = min(args.batch_size, args.samples_per_task - generated)
                batch = make_metric_batch(
                    task, rng, count, include_raw_frames=True
                )
                frames = np.asarray(batch["frames"])
                language_hidden, language_mask = gather_language(
                    language_cache, list(batch["language_text"]), device
                )
                h5, h11 = vision.encode_multi(
                    preprocess_raw_full_frames(
                        np.asarray(batch["raw_frames"]), device
                    ),
                    out_layers=(5, 11),
                )
                coarse = coarse_head(h5, h11, language_hidden, language_mask, coords)
                selection = plan_metric_roi(
                    coarse.p.clamp(0.0, 1.0),
                    coarse.visibility,
                    IMAGE_SIZE,
                    min_size=float(config["min_roi_size"]),
                    max_size=float(config["max_roi_size"]),
                    distance_scale=float(config["distance_scale"]),
                )
                crop_h5, crop_h11 = vision.encode_multi(
                    preprocess_raw_roi_frames(
                        np.asarray(batch["raw_frames"]), selection.roi, device
                    ),
                    out_layers=(5, 11),
                )
                refined = roi_head(
                    crop_h5, crop_h11, language_hidden, language_mask, coords
                )
                batch_index = torch.arange(count, device=device)[:, None]
                refined_pair = refined.p[batch_index, selection.pair_roles]
                final_p, _ = merge_roi_refinement(
                    coarse.p,
                    coarse.visibility,
                    refined_pair,
                    selection,
                    IMAGE_SIZE,
                    alpha=float(args.alpha),
                    max_delta_px=float(config["max_delta_px"]),
                )
                # Runtime keeps coarse visibility unchanged; ROI visibility BCE
                # is auxiliary supervision and must not inflate deployed metrics.
                final_visibility = coarse.visibility
                chunks.append(
                    {
                        "coarse_p": coarse.p.float().cpu().numpy(),
                        "roi_p": final_p.float().cpu().numpy(),
                        "target": np.asarray(batch["keypoints"], dtype=np.float32),
                        "visibility": np.asarray(batch["visibility"], dtype=np.float32),
                        "coarse_visibility": coarse.visibility.float().cpu().numpy(),
                        "roi_visibility": final_visibility.float().cpu().numpy(),
                        "selected": selection.role_mask.cpu().numpy().astype(np.float32),
                        "selection_confidence": selection.confidence.cpu().numpy()[:, None],
                        "selected_gt_crop_visible": (
                            gt_crop_visibility(
                                torch.from_numpy(np.asarray(batch["keypoints"])).to(device),
                                torch.from_numpy(np.asarray(batch["visibility"])).to(device),
                                selection.roi,
                                IMAGE_SIZE,
                            )
                            .mul(selection.role_mask)
                            .sum(dim=1, keepdim=True)
                            .cpu()
                            .numpy()
                        ),
                    }
                )
                generated += count
            record = _concat(chunks)
            task_records[task] = record
            values = paired_metrics(
                record["coarse_p"],
                record["roi_p"],
                record["target"],
                record["visibility"],
                record["coarse_visibility"],
                record["roi_visibility"],
            )
            values["difficulty"] = difficulty_for_task(task)
            values["selected_role_fraction"] = record["selected"].mean(axis=0).tolist()
            values["selection"] = selection_diagnostics(record)
            per_task[task] = values

    all_records = list(task_records.values())
    overall_record = _concat(all_records)
    overall = paired_metrics(
        overall_record["coarse_p"],
        overall_record["roi_p"],
        overall_record["target"],
        overall_record["visibility"],
        overall_record["coarse_visibility"],
        overall_record["roi_visibility"],
    )
    overall["selection"] = selection_diagnostics(overall_record)
    grouped: dict[str, Any] = {}
    for name in DIFFICULTY.values():
        records = [record for task, record in task_records.items() if difficulty_for_task(task) == name]
        merged = _concat(records)
        grouped[name] = paired_metrics(
            merged["coarse_p"],
            merged["roi_p"],
            merged["target"],
            merged["visibility"],
            merged["coarse_visibility"],
            merged["roi_visibility"],
        )
    gate = adoption_gate(overall, grouped)
    return {
        "contract": CONTRACT,
        "coarse_checkpoint": str(coarse_path),
        "coarse_sha256": coarse_sha,
        "coarse_source": coarse_source,
        "coarse_policy_checkpoint": (
            str(Path(args.coarse_policy_checkpoint).resolve())
            if coarse_source == COARSE_SOURCE_POLICY
            else None
        ),
        "runtime_metric_checkpoint": str(coarse_path),
        "coarse_head_state_sha256": coarse_head_sha,
        "roi_checkpoint": str(roi_path),
        "roi_sha256": roi_sha,
        "roi_steps_done": int(config.get("steps_done", 0)),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "samples_per_task": int(args.samples_per_task),
        "samples": int(args.samples_per_task * len(SUPPORTED_TASKS)),
        "overall": overall,
        "by_difficulty": grouped,
        "per_task": per_task,
        "adoption_gate": gate,
    }


def print_report(result: Mapping[str, Any]) -> None:
    coarse = result["overall"]["coarse"]
    roi = result["overall"]["roi"]
    c_loc = coarse["localization"]["aggregate"]
    r_loc = roi["localization"]["aggregate"]
    c_vis = coarse["visibility"]["aggregate"]
    r_vis = roi["visibility"]["aggregate"]
    print(
        f"paired ROI held-out | samples={result['samples']} seed={result['seed']} alpha={result['alpha']}"
    )
    print(
        f"overall RMSE {c_loc['rmse_px']:.2f}->{r_loc['rmse_px']:.2f}px | "
        f"PCK@10 {c_loc['pck@10px']:.1%}->{r_loc['pck@10px']:.1%} | "
        f"vis balanced-acc {c_vis['balanced_accuracy']:.1%}->{r_vis['balanced_accuracy']:.1%}"
    )
    selection = result["overall"]["selection"]
    print(
        f"selection confidence mean={selection['confidence_mean']:.3f} "
        f"p10={selection['confidence_p10']:.3f} | "
        f"usable GT crop={selection['usable_pair_fraction']:.1%} "
        f"both roles={selection['both_roles_crop_visible_fraction']:.1%}"
    )
    for name, values in result["by_difficulty"].items():
        before = values["coarse"]["localization"]["aggregate"]["rmse_px"]
        after = values["roi"]["localization"]["aggregate"]["rmse_px"]
        print(f"{name:10s} RMSE {before:.2f}->{after:.2f}px")
    gate = result["adoption_gate"]
    print("ADOPT" if gate["adopt"] else "REJECT", json.dumps(gate["checks"], ensure_ascii=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired held-out MT-VJ ROI evaluation")
    coarse = parser.add_mutually_exclusive_group(required=True)
    coarse.add_argument(
        "--coarse-checkpoint",
        help="completed Stage-V metric checkpoint (legacy/default coarse source)",
    )
    coarse.add_argument(
        "--coarse-policy-checkpoint",
        help="final policy whose embedded mtvj_metric_head supplies coarse weights",
    )
    parser.add_argument(
        "--runtime-metric-checkpoint",
        help="immutable external Stage-V metric file required by the final policy runtime",
    )
    parser.add_argument("--roi-checkpoint", required=True)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=120813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate(args)
    print_report(result)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"JSON -> {output}")


if __name__ == "__main__":
    main()
