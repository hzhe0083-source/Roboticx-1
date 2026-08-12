#!/usr/bin/env python
"""Evaluate an MT-VJ metric head on fresh simulator samples.

This is a read-only acceptance check for the stage-V checkpoint.  It rebuilds
the frozen V-JEPA/Qwen/head path used by ``train_metric_visual.py``, generates
new ``make_metric_batch`` samples from the requested seed, and reports visible
keypoint localization error.

Example::

    python scripts/eval_metric_visual_holdout.py \
      --checkpoint checkpoints/metric_field_doorlock.pt \
      --task door-lock-v3 --samples 100 --seed 120813 --device cuda
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before importing mujoco/metaworld through the data generator.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

ROLE_NAMES = ("tool", "object", "target", "interface")
ROLE_PAIRS = {"tool_object": (0, 1), "interface_target": (3, 2)}
CONTRACT = "mt_vj_metric_field_v1"
IMAGE_SIZE = 384


def _one_group_metrics(errors_px: np.ndarray) -> dict[str, int | float | None]:
    """Return scalar metrics for one one-dimensional Euclidean-error array."""
    errors = np.asarray(errors_px, dtype=np.float64)
    if errors.ndim != 1:
        raise ValueError(f"errors_px must be one-dimensional, got {errors.shape}")
    if errors.size == 0:
        return {
            "visible_count": 0,
            "rmse_px": None,
            "median_px": None,
            "pck@5px": None,
            "pck@10px": None,
        }
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("errors_px must contain finite, non-negative values")
    return {
        "visible_count": int(errors.size),
        "rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
        "median_px": float(np.median(errors)),
        "pck@5px": float(np.mean(errors <= 5.0)),
        "pck@10px": float(np.mean(errors <= 10.0)),
    }


def compute_localization_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    visibility: np.ndarray,
    *,
    image_size: int = IMAGE_SIZE,
    role_names: Sequence[str] = ROLE_NAMES,
) -> dict[str, Any]:
    """Compute per-role and aggregate visible-keypoint localization metrics.

    ``predictions`` and ``targets`` are normalized ``(y, x)`` coordinates with
    shape ``[N, R, 2]``.  The aggregate is computed over every visible point,
    rather than averaging role-level scores, so sparsely visible roles do not
    receive disproportionate weight.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"predictions must have shape [N, R, 2], got {pred.shape}")
    if true.shape != pred.shape:
        raise ValueError(f"targets shape {true.shape} does not match predictions {pred.shape}")
    if vis.shape != pred.shape[:2]:
        raise ValueError(f"visibility must have shape {pred.shape[:2]}, got {vis.shape}")
    if pred.shape[1] != len(role_names):
        raise ValueError(
            f"role_names has {len(role_names)} entries but predictions have {pred.shape[1]} roles"
        )
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("predictions and targets must be finite")
    if not np.isfinite(vis).all() or np.any((vis < 0.0) | (vis > 1.0)):
        raise ValueError("visibility must be finite and in [0, 1]")

    visible = vis >= 0.5
    errors = np.linalg.norm((pred - true) * float(image_size), axis=-1)
    roles = {
        str(name): _one_group_metrics(errors[:, role][visible[:, role]])
        for role, name in enumerate(role_names)
    }
    return {
        "roles": roles,
        "aggregate": _one_group_metrics(errors[visible]),
    }


def _one_visibility_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> dict[str, int | float | None]:
    """Binary visibility metrics without letting class imbalance hide failures."""
    prob = np.asarray(probabilities, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if prob.ndim != 1 or true.shape != prob.shape:
        raise ValueError("visibility probabilities/targets must be matching vectors")
    if prob.size == 0:
        return {
            "count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "visible_recall": None,
            "hidden_recall": None,
            "brier": None,
        }
    if not np.isfinite(prob).all() or np.any((prob < 0.0) | (prob > 1.0)):
        raise ValueError("visibility probabilities must be finite and in [0, 1]")
    if not np.isfinite(true).all() or np.any((true < 0.0) | (true > 1.0)):
        raise ValueError("visibility targets must be finite and in [0, 1]")
    truth = true >= 0.5
    predicted = prob >= 0.5
    n_visible = int(truth.sum())
    n_hidden = int((~truth).sum())
    visible_recall = (
        float((predicted & truth).sum() / n_visible) if n_visible else None
    )
    hidden_recall = (
        float(((~predicted) & (~truth)).sum() / n_hidden) if n_hidden else None
    )
    balanced = (
        0.5 * (visible_recall + hidden_recall)
        if visible_recall is not None and hidden_recall is not None
        else None
    )
    return {
        "count": int(prob.size),
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": balanced,
        "visible_recall": visible_recall,
        "hidden_recall": hidden_recall,
        "brier": float(np.mean(np.square(prob - true))),
    }


def compute_visibility_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    role_names: Sequence[str] = ROLE_NAMES,
) -> dict[str, Any]:
    """Report per-role and aggregate visibility quality at the policy threshold."""
    prob = np.asarray(probabilities, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if prob.ndim != 2 or prob.shape != true.shape:
        raise ValueError(
            "visibility probabilities/targets must share shape [N, R], got "
            f"{prob.shape} and {true.shape}"
        )
    if prob.shape[1] != len(role_names):
        raise ValueError("visibility role count does not match role_names")
    return {
        "roles": {
            str(name): _one_visibility_metrics(prob[:, role], true[:, role])
            for role, name in enumerate(role_names)
        },
        "aggregate": _one_visibility_metrics(prob.reshape(-1), true.reshape(-1)),
    }


def compute_gated_state_metrics(
    predictions: np.ndarray,
    predicted_visibility: np.ndarray,
    targets: np.ndarray,
    target_visibility: np.ndarray,
    *,
    image_size: int = IMAGE_SIZE,
    role_names: Sequence[str] = ROLE_NAMES,
) -> dict[str, Any]:
    """Measure the exact ``p * visibility`` state consumed by the VA policy."""
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    pred_vis = np.asarray(predicted_visibility, dtype=np.float64)
    true_vis = np.asarray(target_visibility, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[-1] != 2 or true.shape != pred.shape:
        raise ValueError("gated-state positions must share shape [N, R, 2]")
    if pred_vis.shape != pred.shape[:2] or true_vis.shape != pred.shape[:2]:
        raise ValueError("gated-state visibility must share shape [N, R]")
    if not all(np.isfinite(value).all() for value in (pred, true, pred_vis, true_vis)):
        raise ValueError("gated-state inputs must be finite")
    if np.any((pred_vis < 0.0) | (pred_vis > 1.0)) or np.any(
        (true_vis < 0.0) | (true_vis > 1.0)
    ):
        raise ValueError("gated-state visibility must be in [0, 1]")
    predicted_state = pred * pred_vis[..., None]
    target_state = true * true_vis[..., None]
    errors = np.linalg.norm(
        (predicted_state - target_state) * float(image_size), axis=-1
    )
    return {
        "roles": {
            str(name): _one_group_metrics(errors[:, role])
            for role, name in enumerate(role_names)
        },
        "aggregate": _one_group_metrics(errors.reshape(-1)),
    }


def compute_pair_visibility_coverage(visibility: np.ndarray) -> dict[str, float]:
    """Fraction of samples where each actionable role pair is jointly visible."""
    vis = np.asarray(visibility, dtype=np.float64)
    if vis.ndim != 2 or vis.shape[1] != len(ROLE_NAMES):
        raise ValueError(f"visibility must have shape [N, 4], got {vis.shape}")
    if vis.shape[0] == 0 or not np.isfinite(vis).all():
        raise ValueError("visibility coverage requires non-empty finite input")
    binary = vis >= 0.5
    pair_masks = {
        name: binary[:, left] & binary[:, right]
        for name, (left, right) in ROLE_PAIRS.items()
    }
    any_pair = np.logical_or.reduce(list(pair_masks.values()))
    return {
        **{name: float(mask.mean()) for name, mask in pair_masks.items()},
        "any_actionable_pair": float(any_pair.mean()),
    }


def validate_checkpoint_contract(checkpoint: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    """Fail fast on checkpoints that cannot produce a credible held-out result."""
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError(
            f"checkpoint contract={checkpoint.get('contract')!r}; expected {CONTRACT!r}"
        )
    missing = {"config", "metric_head", "relation_encoder"} - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")

    expected = {
        "h_dim": 768,
        "d_proj": 192,
        "n_roles": len(ROLE_NAMES),
        "lang_dim": 2048,
        "image_size": IMAGE_SIZE,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"checkpoint config {key}={config.get(key)!r}; expected {value!r}"
            )
    tasks = config.get("tasks")
    if not isinstance(tasks, (list, tuple)) or task not in tasks:
        raise ValueError(
            f"task {task!r} is absent from checkpoint training tasks {tasks!r}; "
            "refusing to label this as an in-domain held-out evaluation"
        )
    if config.get("language_cache_available") is not True:
        raise ValueError(
            "checkpoint was not trained with a verified language backbone "
            "(language_cache_available is not true)"
        )
    if not isinstance(checkpoint["metric_head"], Mapping):
        raise ValueError("checkpoint metric_head must be a state_dict mapping")
    return config


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full frozen inference path.  Imports stay lazy for CPU-only tests."""
    import torch

    from prepare_metaworld_metric import (
        SAMPLE_RNG_CONTRACT,
        SUPPORTED_TASKS,
        make_metric_batch,
    )
    from scripts.build_longtraj_features import ENV_TO_TASK
    from train_metric_visual import (
        build_language_cache,
        gather_language,
        preprocess_frames,
    )
    from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone
    from va_compound.live_vjepa import _dense_coords
    from va_compound.metric_visual_head import LanguageMetricField

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    requested_tasks = list(SUPPORTED_TASKS) if args.all_tasks else [args.task]
    unknown = [task for task in requested_tasks if task not in SUPPORTED_TASKS]
    if unknown:
        raise ValueError(f"unsupported metric task(s) {unknown!r}; supported={SUPPORTED_TASKS}")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    config = validate_checkpoint_contract(checkpoint, requested_tasks[0])
    for task in requested_tasks[1:]:
        validate_checkpoint_contract(checkpoint, task)
    train_seed = config.get("seed")
    if train_seed is not None and int(train_seed) == args.seed:
        raise ValueError(
            f"evaluation seed {args.seed} equals checkpoint training seed; choose a fresh seed"
        )

    device = _resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ctor_keys = set(inspect.signature(LanguageMetricField.__init__).parameters) - {"self"}
    ctor_kwargs = {key: value for key, value in config.items() if key in ctor_keys}
    metric_head = LanguageMetricField(**ctor_kwargs).to(device)
    metric_head.load_state_dict(checkpoint["metric_head"], strict=True)
    metric_head.requires_grad_(False).eval()

    vision_dtype = "float16" if device.type == "cuda" else "float32"
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device,
        dtype=vision_dtype,
        local_files_only=True,
    )
    vision_backbone.freeze_all()
    language_dtype = "float16" if device.type == "cuda" else "float32"
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device,
        dtype=language_dtype,
        local_files_only=True,
    )
    instructions = [ENV_TO_TASK[task] for task in requested_tasks]
    language_cache, language_ok = build_language_cache(text_backbone, instructions)
    if not language_ok:
        raise RuntimeError("language backbone degraded during evaluation")
    del text_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    coords = torch.from_numpy(_dense_coords()).to(device)

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    predicted_visibility: list[np.ndarray] = []
    per_task: dict[str, Any] = {}
    with torch.inference_mode():
        for task_index, task in enumerate(requested_tasks):
            # Per-task SeedSequence keeps samples stable for paired ROI checks.
            rng = np.random.default_rng(
                np.random.SeedSequence((int(args.seed), int(task_index), 0xE7A1))
            )
            task_predictions: list[np.ndarray] = []
            task_targets: list[np.ndarray] = []
            task_visibility: list[np.ndarray] = []
            task_predicted_visibility: list[np.ndarray] = []
            generated = 0
            while generated < args.samples:
                count = min(args.batch_size, args.samples - generated)
                batch = make_metric_batch(task, rng, count)
                if not bool(np.asarray(batch["supported"]).all()):
                    raise RuntimeError("data generator returned unsupported samples")
                video = preprocess_frames(np.asarray(batch["frames"]), device)
                h5, h11 = vision_backbone.encode_multi(video, out_layers=(5, 11))
                lang_hidden, lang_mask = gather_language(
                    language_cache, list(batch["language_text"]), device
                )
                output = metric_head(h5, h11, lang_hidden, lang_mask, coords)
                task_predictions.append(output.p.float().cpu().numpy())
                task_targets.append(np.asarray(batch["keypoints"], dtype=np.float32))
                task_visibility.append(np.asarray(batch["visibility"], dtype=np.float32))
                task_predicted_visibility.append(output.visibility.float().cpu().numpy())
                generated += count

            pred_task = np.concatenate(task_predictions, axis=0)
            true_task = np.concatenate(task_targets, axis=0)
            vis_task = np.concatenate(task_visibility, axis=0)
            pred_vis_task = np.concatenate(task_predicted_visibility, axis=0)
            per_task[task] = {
                "localization": compute_localization_metrics(pred_task, true_task, vis_task),
                "visibility": compute_visibility_metrics(pred_vis_task, vis_task),
                "gated_state": compute_gated_state_metrics(
                    pred_task, pred_vis_task, true_task, vis_task
                ),
                "pair_visibility_coverage": compute_pair_visibility_coverage(vis_task),
            }
            task_rmse = per_task[task]["localization"]["aggregate"]["rmse_px"]
            print(
                f"held-out task {task_index + 1}/{len(requested_tasks)} "
                f"{task}: RMSE={task_rmse:.2f}px",
                flush=True,
            )
            predictions.append(pred_task)
            targets.append(true_task)
            visibility.append(vis_task)
            predicted_visibility.append(pred_vis_task)

    pred_all = np.concatenate(predictions, axis=0)
    true_all = np.concatenate(targets, axis=0)
    vis_all = np.concatenate(visibility, axis=0)
    pred_vis_all = np.concatenate(predicted_visibility, axis=0)
    metrics = {
        "localization": compute_localization_metrics(pred_all, true_all, vis_all),
        "visibility": compute_visibility_metrics(pred_vis_all, vis_all),
        "gated_state": compute_gated_state_metrics(
            pred_all, pred_vis_all, true_all, vis_all
        ),
        "pair_visibility_coverage": compute_pair_visibility_coverage(vis_all),
    }
    return {
        "contract": CONTRACT,
        "sample_rng_contract": SAMPLE_RNG_CONTRACT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_steps_done": int(config.get("steps_done", 0)),
        "task": "all49" if args.all_tasks else args.task,
        "tasks": requested_tasks,
        "samples_per_task": args.samples,
        "samples": args.samples * len(requested_tasks),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "metrics": metrics,
        "per_task": per_task,
    }


def _format_value(value: float | int | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{100.0 * float(value):.1f}%"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.2f}"


def print_report(result: Mapping[str, Any]) -> None:
    print(
        f"MT-VJ held-out | task={result['task']} samples={result['samples']} "
        f"batch={result['batch_size']} seed={result['seed']} device={result['device']}"
    )
    print(
        f"checkpoint={result['checkpoint']} "
        f"steps_done={result['checkpoint_steps_done']}"
    )
    localization = result["metrics"]["localization"]
    print(f"{'role':<12} {'visible':>8} {'RMSE px':>10} {'median px':>11} {'PCK@5':>9} {'PCK@10':>9}")
    rows = list(localization["roles"].items()) + [
        ("aggregate", localization["aggregate"])
    ]
    for name, values in rows:
        print(
            f"{name:<12} "
            f"{_format_value(values['visible_count']):>8} "
            f"{_format_value(values['rmse_px']):>10} "
            f"{_format_value(values['median_px']):>11} "
            f"{_format_value(values['pck@5px'], percent=True):>9} "
            f"{_format_value(values['pck@10px'], percent=True):>9}"
        )
    vis = result["metrics"]["visibility"]["aggregate"]
    gated = result["metrics"]["gated_state"]["aggregate"]
    print(
        "visibility: "
        f"balanced_acc={_format_value(vis['balanced_accuracy'], percent=True)} "
        f"visible_recall={_format_value(vis['visible_recall'], percent=True)} "
        f"hidden_recall={_format_value(vis['hidden_recall'], percent=True)} "
        f"brier={_format_value(vis['brier'])}"
    )
    print(
        "policy gated-state: "
        f"RMSE={_format_value(gated['rmse_px'])}px "
        f"median={_format_value(gated['median_px'])}px"
    )
    coverage = result["metrics"]["pair_visibility_coverage"]
    print(
        "GT actionable-pair coverage: "
        f"tool-object={coverage['tool_object']:.1%} "
        f"interface-target={coverage['interface_target']:.1%} "
        f"either={coverage['any_actionable_pair']:.1%}"
    )
    if len(result.get("per_task", {})) > 1:
        worst = sorted(
            result["per_task"],
            key=lambda task: result["per_task"][task]["localization"]["aggregate"]["rmse_px"],
            reverse=True,
        )[:5]
        print(
            "worst localization tasks: "
            + ", ".join(
                f"{task}={result['per_task'][task]['localization']['aggregate']['rmse_px']:.2f}px"
                for task in worst
            )
        )
        low_coverage = [
            task
            for task, values in result["per_task"].items()
            if values["pair_visibility_coverage"]["any_actionable_pair"] < 0.5
        ]
        print(
            "tasks with <50% actionable-pair GT coverage: "
            + (", ".join(low_coverage) if low_coverage else "none")
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="stage-V metric checkpoint")
    parser.add_argument("--task", default="door-lock-v3")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="evaluate all 49 supported tasks; --samples is then per-task",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True, help="fresh held-out simulator seed")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--json", type=str, default=None, help="optional JSON result path")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print_report(result)
    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json={output.resolve()}")


if __name__ == "__main__":
    main()
