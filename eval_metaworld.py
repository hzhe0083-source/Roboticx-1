"""Closed-loop evaluation on MetaWorld MT50 (language-conditioned).

Protocol: 10 episodes per task, four difficulty tiers.  The policy receives
the corner2 camera image (480x480, resized to 384), the state
(--state-take 截取：4 = EEF xyz + gripper，现状默认；8 = Evo-1 官方评测口径；
39 = 完整 obs；0 = proprio 恒零), the previous action, and the task
language condition.

C²-IRF v2 部署（设计文档 c2irf_v2_vision_ablation.md）：
- ``--servo-ablation``：Step 2 低秩伺服四消融（zero-gain / gain-shuffle /
  wrong-role / open-loop），挂 c2 plan/feedback 节奏；
- ``--fovea``：Step 3 foveal 双速率部署（plan_due 全图 dense 重读 + ROI，
  feedback 步 foveal crop 局部关系更新 + 伺服修正，servo 新息超阈值立即
  提前全局刷新）。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_pnpw_features import QwenTextBackbone

_DEBUG_FA_DONE: dict[str, bool] = {}
_ALIGN_ACTS: list | None = None
from va_compound.backbones import (
    QwenSemanticBackbone,
    TimmActionVisionBackbone,
    VJEPA21Backbone,
    pool_mtvj_coarse_tokens,
)
from va_compound.model import (
    ControllerParams,
    VACompoundConfig,
    VACompoundPolicy,
)
from va_compound.fovea import (
    FoveaPrefixEncoder,
    apply_unified_crop,
    compute_roi,
    crop_to_full_cov,
    crop_to_full_norm,
    full_to_crop_norm,
)
from va_compound.local_control_slots import (
    MultiModeReadout,
    build_va_vision_input,
)
from va_compound.metric_roi import (
    DINO_METRIC_ROI_CONTRACT,
    TASK35_METRIC_ROLE_CONTRACT,
    load_dino_metric_roi_checkpoint,
    load_metric_roi_checkpoint,
    metric_head_state_sha256,
    prepare_metric_roi_video,
    refine_metric_roi_positions,
    refine_metric_roi_positions_dino,
)
from scripts.mt50_difficulty import (
    MT50_BENCHMARK_TASK_TO_GROUP,
    canonical_mt50_benchmark_env,
    summarize_mt50_benchmark_trials,
)

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

MTVJ_METRIC_HEAD_CONFIG_KEYS = (
    "lang_dim",
    "h_dim",
    "d_proj",
    "n_roles",
    "l2_norm",
    "learnable_temp",
    "temp_init",
    "freeze_bias",
    "mode_readout",
    "grid",
)
# 缺省即向后兼容的构造键（旧 checkpoint 无此键 → 填默认值，不视为不完整）。
_MTVJ_METRIC_HEAD_OPTIONAL_KEYS = frozenset({"grid"})
_MTVJ_METRIC_HEAD_CONFIG_DEFAULTS = {
    "lang_dim": 2048,
    "h_dim": 768,
    "d_proj": 192,
    "n_roles": 4,
    "l2_norm": False,
    "learnable_temp": False,
    "temp_init": 10.0,
    "freeze_bias": False,
    "mode_readout": False,
    "grid": 24,
}
MTVJ_METRIC_STATE_SOURCE = "p_times_visibility_flat"
MTVJ_METRIC_CONTRACT_VERSION = 3
MTVJ_LEGACY_METRIC_STATE_SOURCE = "p_flat"
MTVJ_LEGACY_METRIC_CONTRACT_VERSION = 2


def _mtvj_metric_positions(
    out, source: str = MTVJ_METRIC_STATE_SOURCE
) -> torch.Tensor:
    """Train/eval-identical state selector, including faithful v2 baselines."""
    p = out.p
    if p.ndim != 3 or p.shape[-2:] != (4, 2):
        raise ValueError(f"MT-VJ out.p must be [N, 4, 2], got {tuple(p.shape)}")
    if source == MTVJ_LEGACY_METRIC_STATE_SOURCE:
        return p.reshape(p.shape[0], 8)
    if source != MTVJ_METRIC_STATE_SOURCE:
        raise ValueError(f"unknown MT-VJ metric state source: {source!r}")
    visibility = out.visibility
    if visibility.shape != p.shape[:-1]:
        raise ValueError(
            "MT-VJ out.visibility must match out.p roles: "
            f"{tuple(visibility.shape)} != {tuple(p.shape[:-1])}"
        )
    return (p * visibility.unsqueeze(-1)).reshape(p.shape[0], 8)


_mtvj_visibility_gated_positions = _mtvj_metric_positions


def select_eval_tasks(
    all_tasks: list[str], task_ids: str | None, max_tasks: int
) -> list[tuple[int, str]]:
    """Select tasks while preserving their global metadata IDs."""
    if task_ids is not None:
        indices = [int(token) for token in task_ids.split(",")]
    else:
        indices = list(range(min(max_tasks, len(all_tasks))))
    for index in indices:
        if index < 0 or index >= len(all_tasks):
            raise ValueError(
                f"--task-ids index {index} out of range 0..{len(all_tasks) - 1}"
            )
    return [(index, all_tasks[index]) for index in indices]


def validate_language_features(
    features: dict[str, Any], language_features: dict[str, Any]
) -> None:
    """Require the compact language cache to share the evaluation normalization."""
    normal = features.get("normalization") or {}
    language_normal = language_features.get("normalization") or {}
    if not normal or normal.keys() != language_normal.keys() or any(
        not isinstance(normal[key], torch.Tensor)
        or not isinstance(language_normal[key], torch.Tensor)
        or not torch.equal(normal[key], language_normal[key])
        for key in normal
    ):
        raise ValueError("--language-features normalization differs from --features")
    tasks = list((language_features.get("metadata") or {}).get("tasks") or [])
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("--language-features must contain unique metadata.tasks")
    raw_instruction_id = language_features.get("instruction_id")
    if not isinstance(raw_instruction_id, torch.Tensor):
        raise ValueError("--language-features is missing tensor instruction_id")
    instruction_id = raw_instruction_id.to(dtype=torch.long).reshape(-1)
    if set(instruction_id.tolist()) != set(range(len(tasks))):
        raise ValueError("--language-features must cover every metadata task id")


def validate_mt50_benchmark_protocol(
    args: argparse.Namespace,
    selected_tasks: list[tuple[int, str]],
    descriptions_to_env: dict[str, str],
) -> None:
    """Fail fast before rollout when the formal EvoMind-compatible contract drifts."""
    if not args.mt50_benchmark:
        return
    selected_envs = {
        canonical_mt50_benchmark_env(str(descriptions_to_env.get(text, "")))
        for _, text in selected_tasks
    }
    requirements = {
        "50 canonical tasks": selected_envs == set(MT50_BENCHMARK_TASK_TO_GROUP),
        "10 trials per task": args.trials_per_task == 10,
        "400-step horizon": args.horizon == 400,
        "H15 execution": (
            args.execution_horizon == 15
            and not args.allow_execution_horizon_ablation
        ),
        "shared reset seeds 4042-4051": args.episode_seed_base == 4042,
        "structured output": args.output_json is not None,
        "unaligned environment resets": not args.align_init,
    }
    failed = [name for name, ok in requirements.items() if not ok]
    if failed:
        raise ValueError("MT50 benchmark protocol failed: " + ", ".join(failed))


METAWORLD_TASK_CONFIG = (
    Path(__file__).resolve().parent / "metaworld_config.json"
    if (Path(__file__).resolve().parent / "metaworld_config.json").is_file()
    else Path(
        "/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json"
    )
)


def load_metaworld_description_to_env(path: Path = METAWORLD_TASK_CONFIG) -> dict[str, str]:
    import json

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ValueError(f"MetaWorld task config missing: {resolved}")
    config = json.loads(resolved.read_text())
    descriptions = config.get("TASK_DESCRIPTIONS")
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError(f"{resolved} has no TASK_DESCRIPTIONS")
    mapping = {str(value): str(key) for key, value in descriptions.items()}
    # Native MetaWorld gives push-v3 and push-back-v3 the same description.
    # The recovered MT50 source deliberately names push-back "Pull..." so the
    # two task ids remain unambiguous.
    if "push-v3" in descriptions:
        mapping[str(descriptions["push-v3"])] = "push-v3"
    if "push-back-v3" in descriptions:
        mapping["Pull a puck to a goal"] = "push-back-v3"
    return mapping


def require_task35_peg_insert_side(
    selected_tasks: list[tuple[int, str]],
    descriptions_to_env: dict[str, str],
) -> str:
    """Fail before GPU work if --task-ids 35 is not peg-insert-side-v3."""
    if len(selected_tasks) != 1 or selected_tasks[0][0] != 35:
        raise ValueError("task35 evaluation must select exactly --task-ids 35")
    task_text = selected_tasks[0][1]
    env_name = descriptions_to_env.get(task_text)
    if env_name != "peg-insert-side-v3":
        raise ValueError(
            "task35 evaluation mapped "
            f"{task_text!r} -> {env_name!r}, expected peg-insert-side-v3"
        )
    return env_name


def evaluation_episode_seed(
    global_task_id: int, trial: int, *, base_seed: int | None = None
) -> int:
    """Use a shared benchmark reset sequence or the legacy task-specific sequence."""
    if base_seed is not None:
        return int(base_seed) + int(trial)
    return 1000 * int(global_task_id) + int(trial)


def parse_trial_range(spec: str | None, trials_per_task: int) -> tuple[int, int]:
    """Parse ``START:END`` into a half-open trial slice for sharded evaluation.

    Seeds come from ``evaluation_episode_seed(task, trial)`` alone, so a shard
    reproduces exactly the trials a serial run would produce for those indices.
    """
    if spec is None:
        return 0, int(trials_per_task)
    text = spec.strip()
    if text.count(":") != 1:
        raise ValueError(f"--trial-range must be START:END, got {spec!r}")
    start_text, stop_text = (part.strip() for part in text.split(":"))
    start = 0 if not start_text else int(start_text)
    stop = int(trials_per_task) if not stop_text else int(stop_text)
    if not 0 <= start < stop <= int(trials_per_task):
        raise ValueError(
            f"--trial-range {spec!r} must satisfy "
            f"0 <= start < stop <= --trials-per-task ({trials_per_task})"
        )
    return start, stop


TASK35_EVAL50_SEEDS = tuple(range(35000, 35050))
MT50_FORMAL_SEEDS = tuple(range(4042, 4052))
DAGGER_POST_SUCCESS_STEPS = 60


def dagger_takeover_step(
    task_id: int,
    episode_seed: int,
    minimum: int,
    maximum: int,
    *,
    stride: int = 15,
) -> int:
    """Choose a reproducible plan-boundary takeover without touching global RNG."""
    if stride < 1 or minimum < 0 or maximum < minimum:
        raise ValueError("invalid DAgger takeover range/stride")
    first = ((int(minimum) + stride - 1) // stride) * stride
    last = (int(maximum) // stride) * stride
    if first > last:
        raise ValueError("DAgger takeover range contains no plan boundary")
    choices = range(first, last + 1, stride)
    digest = hashlib.sha256(f"{task_id}:{episode_seed}:dagger".encode()).digest()
    return choices[int.from_bytes(digest[:8], "little") % len(choices)]


def build_dagger_episode(
    *,
    episode_seed: int,
    takeover_step: int,
    prefix_keep: int,
    frames: list[np.ndarray],
    actions: list[np.ndarray],
    states: list[np.ndarray],
    action_success: list[bool],
    action_source: list[str],
    metric_state: list[np.ndarray] | None = None,
    metric_state_valid: list[bool] | None = None,
) -> dict[str, Any] | None:
    """Trim one model-prefix/expert-suffix rollout into the raw v2 contract."""
    lengths = {
        len(frames), len(actions), len(states), len(action_success), len(action_source)
    }
    if len(lengths) != 1:
        raise ValueError(f"DAgger rollout timeline mismatch: {sorted(lengths)}")
    if prefix_keep < 0 or takeover_step < 0:
        raise ValueError("invalid DAgger prefix/takeover")
    if takeover_step >= len(actions):
        return None
    succeeded = np.flatnonzero(np.asarray(action_success, dtype=bool))
    if not len(succeeded) or int(succeeded[0]) < takeover_step:
        return None

    first_success_abs = int(succeeded[0])
    trim = max(0, int(takeover_step) - int(prefix_keep))
    first_success = first_success_abs - trim
    takeover = int(takeover_step) - trim
    n = len(actions) - trim
    frame_valid = np.ones(n, dtype=bool)
    action_executed = np.ones(n, dtype=bool)
    supervision = np.zeros(n, dtype=bool)
    supervision[takeover:first_success + 1] = True
    recovery = supervision.copy()
    settle = np.zeros(n, dtype=bool)
    event = {
        "start": takeover,
        "end": takeover,
        "kind": "current_policy_takeover",
        "magnitude_m": 0.0,
        "magnitude_mm": 0.0,
        "applied": True,
    }
    episode: dict[str, Any] = {
        "episode_seed": int(episode_seed),
        "frames": np.stack(frames[trim:]),
        "actions": np.asarray(actions[trim:], dtype=np.float32),
        "states": np.asarray(states[trim:], dtype=np.float32),
        "first_success": first_success,
        "success_frame": first_success,
        "action_success": np.asarray(action_success[trim:], dtype=bool),
        "frame_valid": frame_valid,
        "action_executed": action_executed,
        "action_source": action_source[trim:],
        "settle_mask": settle,
        "action_valid": supervision,
        "action_supervision_valid": supervision,
        "recovery_mask": recovery,
        "perturbed": True,
        "n_perturb_events": 1,
        "perturb_start": takeover,
        "perturb_end": takeover,
        "perturb_kind": "current_policy_takeover",
        "perturb_magnitude": 0.0,
        "perturb_magnitude_mm": 0.0,
        "perturb_event": event,
        "dagger_model_prefix_steps": takeover,
        "dagger_takeover_step": int(takeover_step),
    }
    if metric_state is not None:
        if metric_state_valid is None or len(metric_state) != len(actions):
            raise ValueError("DAgger metric timeline mismatch")
        episode["metric_state"] = np.asarray(metric_state[trim:], dtype=np.float32)
        episode["metric_state_valid"] = np.asarray(
            metric_state_valid[trim:], dtype=bool
        )
    return episode


def validate_dagger_args(args: argparse.Namespace) -> None:
    if args.dagger_output_dir is None:
        return
    if args.episode_seed_base is None:
        raise ValueError("--dagger-output-dir requires --episode-seed-base")
    if args.peer_eval_trace:
        raise ValueError("DAgger collection is incompatible with --peer-eval-trace")
    if args.dagger_prefix_keep < 0:
        raise ValueError("--dagger-prefix-keep must be non-negative")
    # Never train on either formal acceptance split.
    requested = set(
        range(args.episode_seed_base, args.episode_seed_base + args.trials_per_task)
    )
    protected = set(MT50_FORMAL_SEEDS) | set(TASK35_EVAL50_SEEDS)
    overlap = sorted(requested & protected)
    if overlap:
        raise ValueError(f"DAgger seeds overlap protected evaluation seeds: {overlap}")
    dagger_takeover_step(
        0,
        args.episode_seed_base,
        args.dagger_takeover_min,
        args.dagger_takeover_max,
    )


def cached_task_language(
    features: dict[str, Any],
    device: torch.device,
    *,
    task_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the training-cache language tokens instead of loading Qwen on GPU."""
    if "language_hidden" not in features or "language_mask" not in features:
        raise ValueError("features missing cached language_hidden/language_mask")
    if "instruction_id" not in features:
        raise ValueError("features missing instruction_id")
    instruction_id = torch.as_tensor(features["instruction_id"]).reshape(-1)
    keep = instruction_id == int(task_id)
    if not bool(keep.any()):
        raise ValueError(f"features have no instruction_id={task_id} language cache")
    hidden_rows = features["language_hidden"][keep]
    mask_rows = features["language_mask"][keep]
    if hidden_rows.ndim != 3 or mask_rows.shape[:2] != hidden_rows.shape[:2]:
        raise ValueError(
            f"cached language has unexpected shape {tuple(hidden_rows.shape)} / {tuple(mask_rows.shape)}"
        )
    if not torch.equal(hidden_rows, hidden_rows[:1].expand_as(hidden_rows)):
        raise ValueError(f"cached language_hidden is not identical for task {task_id}")
    if not torch.equal(mask_rows, mask_rows[:1].expand_as(mask_rows)):
        raise ValueError(f"cached language_mask is not identical for task {task_id}")
    return hidden_rows[:1].to(device=device), mask_rows[:1].to(device=device)


def cached_task35_language(
    features: dict[str, Any],
    device: torch.device,
    *,
    task_id: int = 35,
) -> tuple[torch.Tensor, torch.Tensor]:
    return cached_task_language(features, device, task_id=task_id)


def want_vjepa_dense_backbone(config: Any, args: Any) -> bool:
    """Load frozen V-JEPA dense evidence only for non-DINO MT-VJ policies.

    Task35 DINO-metric checkpoints keep ``dense_readout_mtvj=True`` because the
    VA dense K/V layers exist, but those layers consume DINO block11/block23
    tokens. Auto-enabling the V-JEPA tower would mix two evidence stacks and
    OOM a 16 GiB laptop during eval50.
    """
    dino_metric = bool(getattr(config, "dino_dense_metric", False))
    dino_main = getattr(config, "main_vision_backbone", "vjepa") != "vjepa"
    cli_vjepa = bool(getattr(args, "dense_readout_mtvj", False))
    has_metric_ckpt = getattr(args, "metric_visual_checkpoint", None) is not None
    if dino_metric:
        if cli_vjepa or has_metric_ckpt:
            raise ValueError(
                "DINO-metric checkpoint 的 dense/metric 栈由 DINO 特征驱动；"
                "禁止 --dense-readout-mtvj / --metric-visual-checkpoint "
                "（V-JEPA 路径）混用"
            )
        return False
    if dino_main:
        # DINO-main already forbids the V-JEPA CLI flags. Do not auto-enable
        # a second tower just because the config still has dense_readout_mtvj.
        return False
    return cli_vjepa or bool(getattr(config, "dense_readout_mtvj", False))


def validate_task35_eval50_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed check for a 50-seed task35 closed-loop JSON."""
    if payload.get("contract") != "metaworld_closed_loop_trials_v1":
        raise ValueError("eval50 payload has the wrong contract")
    trials = payload.get("trials") or []
    seeds = [int(row["seed"]) for row in trials]
    expected = list(TASK35_EVAL50_SEEDS)
    requirements = {
        "task35 only": payload.get("task_ids") == [35],
        "50 completed trials": int(payload.get("completed_trials") or 0) == 50,
        "50 trial records": len(trials) == 50,
        "unique seeds": len(set(seeds)) == 50,
        "paired seeds 35000-35049": seeds == expected,
        "execute_steps 6": int(payload.get("execute_steps") or 0) == 6,
        "horizon 500": int(payload.get("horizon") or 0) == 500,
        "WAM4VA state exchange off": payload.get("wmrm_state_exchange") is False,
        "FM decoder": payload.get("action_decoder") == "conditional_flow_matching",
        "peg-insert-side-v3": payload.get("env_name") == "peg-insert-side-v3",
        "success count matches": int(payload.get("successes") or 0)
        == sum(bool(row.get("success")) for row in trials),
        "language source recorded": payload.get("language_source")
        in {"task35_features_cache", "feature_language_cache", "qwen_text_backbone"},
    }
    ablation = payload.get("task35_causal_ablation", "none")
    if payload.get("task35_precision_contract"):
        requirements["acceptance not ablated"] = ablation == "none"
    elif ablation != "none":
        requirements["diagnostic ablation"] = ablation in {
            "temporal-reverse",
            "geometry-zero",
            "geometry-shuffle",
            "roi-off",
            "dense-zero",
        }
    else:
        raise ValueError("eval50 payload is neither precision acceptance nor a causal diagnostic")
    missing = [name for name, ok in requirements.items() if not ok]
    if missing:
        raise ValueError("task35 eval50 payload failed: " + ", ".join(missing))
    return {"ok": True, "seeds": seeds, "successes": int(payload["successes"])}


def task35_ablation_frames(
    frames: list[np.ndarray], ablation: str
) -> list[np.ndarray]:
    """Apply the fixed frame permutation used by the task35 temporal diagnostic."""
    return list(reversed(frames)) if ablation == "temporal-reverse" else frames


def task35_ablation_dense(
    dense_evidence: dict[int, torch.Tensor], ablation: str
) -> dict[int, torch.Tensor]:
    """Zero both DINO dense levels without changing their shape or dtype."""
    if ablation != "dense-zero":
        return dense_evidence
    return {
        layer: torch.zeros_like(evidence)
        for layer, evidence in dense_evidence.items()
    }


def task35_ablation_geometry(metric_g: torch.Tensor, ablation: str) -> torch.Tensor:
    """Ablate only the direct 8-D geometry route, preserving relation tokens."""
    if ablation == "geometry-zero":
        return torch.zeros_like(metric_g)
    if ablation == "geometry-shuffle":
        # Keep each role's (x,y) pair intact while swapping tool↔pegGrasp and
        # hole↔pegHead.  This is deterministic and identical across trials.
        return metric_g[:, [2, 3, 0, 1, 6, 7, 4, 5]]
    return metric_g


def _canonical_mtvj_metric_head_config(
    config: dict | None,
    *,
    require_complete: bool = False,
) -> dict:
    raw = dict(config or {})
    if require_complete:
        required = set(MTVJ_METRIC_HEAD_CONFIG_KEYS) - _MTVJ_METRIC_HEAD_OPTIONAL_KEYS
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "主 checkpoint 缺少完整 mtvj_metric_head_config："
                f"missing={missing}"
            )
    values = {
        key: raw.get(key, default)
        for key, default in _MTVJ_METRIC_HEAD_CONFIG_DEFAULTS.items()
    }
    for key in ("lang_dim", "h_dim", "d_proj", "n_roles", "grid"):
        values[key] = int(values[key])
    for key in (
        "l2_norm",
        "learnable_temp",
        "freeze_bias",
        "mode_readout",
    ):
        values[key] = bool(values[key])
    values["temp_init"] = float(values["temp_init"])
    return values


def _mtvj_metric_checkpoint_identity(path: Path, checkpoint: dict) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(resolved.stat().st_size),
        "contract": checkpoint.get("contract"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mtvj_metric_identity_mismatches(saved: dict, current: dict) -> dict:
    return {
        key: (saved.get(key), current.get(key))
        for key in ("sha256", "size_bytes", "contract")
        if saved.get(key) != current.get(key)
    }

try:
    from va_compound.live_vjepa import _dense_coords, _slot_coords as _stage_coords
except Exception:  # pragma: no cover - 环境裁剪场景
    _dense_coords = None
    _stage_coords = None


def _apply_local_vision(model, tokens, language_cache):
    """Stage A/B：local_slots 读出路径（direct288 恒等；slots 走槽 cross-attn）。

    训练侧 rollout_policy 用 build_local_vision(st, coords, role_queries) 喂
    encode_condition；闭环评估必须走同一路径，否则 288-token checkpoint 的
    槽/坐标读出被跳过（闭环数字失真）。``config.dense_readout``（Step 0）时
    坐标切换为 [1152, 3] 全量 patch 网格（与 288 池化槽坐标互斥）。
    """
    if not model.config.local_slots:
        return tokens
    if _stage_coords is None or _dense_coords is None:
        raise RuntimeError("local_slots eval requires va_compound.live_vjepa")
    role_queries = (
        getattr(language_cache, "role_queries", None)
        if language_cache is not None
        else None
    )
    coords_arr = (
        _dense_coords() if getattr(model.config, "dense_readout", False) else _stage_coords()
    )
    coords = torch.from_numpy(coords_arr).to(
        device=tokens.device, dtype=tokens.dtype
    )
    return model.build_local_vision(tokens, coords, role_queries)


def _load_mtvj_metric_checkpoint(
    path: Path,
    device,
    config,
    policy_relation_state: dict[str, torch.Tensor] | None = None,
    policy_metric_state: dict[str, torch.Tensor] | None = None,
    policy_metric_config: dict | None = None,
    policy_metric_identity: dict | None = None,
    policy_training_contract: dict[str, Any] | None = None,
) -> tuple[nn.Module, nn.Module]:
    """MT-VJ（契约 §2/§7）：加载并冻结 LanguageMetricField + RelationStateEncoder。

    与 train.py ``_load_mtvj_metric_checkpoint`` 同构：checkpoint 契约
    ``{"config": {...}, "metric_head": state_dict, "relation_encoder": state_dict,
    "contract": "mt_vj_metric_field_v1"}``；ctor 参数按 checkpoint config 签名
    过滤注入（缺省用契约默认值）；两模块置 eval + requires_grad_(False)（冻结，
    闭环 no_grad 只读）；启动即校验 relation encoder d_model == hidden_dim
    （metric_tokens 加入每层 action cross-attention 需同维，fail-fast）。
    """
    from va_compound.metric_visual_head import (
        LanguageMetricField,
        RelationStateEncoder,
    )

    policy_contract = policy_training_contract or {}
    if policy_contract.get("metric_head_checkpointed") is True:
        missing_main = [
            key
            for key, value in (
                ("mtvj_metric_head", policy_metric_state),
                ("mtvj_metric_head_config", policy_metric_config),
                ("mtvj_metric_checkpoint_identity", policy_metric_identity),
            )
            if value is None
        ]
        if missing_main:
            raise ValueError(
                "主 checkpoint 声明 metric_head_checkpointed=True，"
                f"但缺少 {missing_main}；拒绝静默回退外部旧 metric head"
            )
    if policy_metric_state is not None and (
        policy_metric_config is None or policy_metric_identity is None
    ):
        raise ValueError(
            "主 checkpoint 含 mtvj_metric_head，但缺少完整构造配置或外部来源指纹"
        )
    if policy_metric_config is not None and policy_metric_state is None:
        raise ValueError(
            "主 checkpoint 含 mtvj_metric_head_config 但缺少 mtvj_metric_head"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # 外部 checkpoint 始终提供模块构造 config；其权重仅用于旧 policy 的
    # 单向迁移。新 policy 已随主 checkpoint 保存精确 runtime 权重。
    required_external_states = []
    if policy_metric_state is None:
        required_external_states.append("metric_head")
    if policy_relation_state is None:
        required_external_states.append("relation_encoder")
    for key in required_external_states:
        if key not in ckpt:
            raise ValueError(
                f"--metric-visual-checkpoint {path} 缺少键 {key!r}（契约 §2）"
            )
    contract = ckpt.get("contract")
    if contract is not None and contract != "mt_vj_metric_field_v1":
        raise ValueError(
            f"--metric-visual-checkpoint contract={contract!r} != "
            f"'mt_vj_metric_field_v1'（阶段 V checkpoint 不匹配）"
        )
    external_ctor_config = _canonical_mtvj_metric_head_config(ckpt.get("config"))
    current_identity = _mtvj_metric_checkpoint_identity(path, ckpt)
    if policy_metric_state is not None:
        ctor_config = _canonical_mtvj_metric_head_config(
            policy_metric_config,
            require_complete=True,
        )
        config_mismatch = (
            {"checkpoint": ctor_config, "external": external_ctor_config}
            if ctor_config != external_ctor_config
            else None
        )
        identity_mismatch = _mtvj_metric_identity_mismatches(
            policy_metric_identity or {}, current_identity
        )
        if config_mismatch or identity_mismatch:
            raise ValueError(
                "评测使用的 MT-VJ 外部 checkpoint 与训练保存时不一致："
                f"constructor={config_mismatch}, fingerprint={identity_mismatch}"
            )
        metric_state = policy_metric_state
        metric_source = "main policy checkpoint"
        constructor_source = "main policy checkpoint"
    else:
        ctor_config = external_ctor_config
        metric_state = ckpt["metric_head"]
        metric_source = "external metric checkpoint (legacy migration)"
        constructor_source = str(path)

    def strict_load_state(
        module: nn.Module,
        state: dict[str, torch.Tensor],
        *,
        module_name: str,
        source: str,
    ) -> None:
        expected = module.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        bad_shapes = {
            key: (tuple(state[key].shape), tuple(expected[key].shape))
            for key in set(expected) & set(state)
            if tuple(state[key].shape) != tuple(expected[key].shape)
        }
        if missing or unexpected or bad_shapes:
            raise ValueError(
                f"MT-VJ {module_name} 与保存的构造配置不兼容；"
                f"source={source}, missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}, shape_mismatch={bad_shapes}"
            )
        module.load_state_dict(state, strict=True)

    metric_head = LanguageMetricField(**ctor_config).to(device)
    strict_load_state(
        metric_head,
        metric_state,
        module_name="metric head",
        source=metric_source,
    )
    # metric tokens 输入使用 visibility-gated out.p（8D）。主 policy checkpoint 中的
    # projection 优先；只有形状兼容时才允许回退到外部 metric checkpoint。
    # 禁止评测时随机重建，否则训练和考试会读取两套不同的 token 语义。
    has_relation_contract = any(
        key in policy_contract
        for key in (
            "metric_tokens_enabled",
            "metric_state_source",
            "metric_state_dim",
            "metric_d_model",
            "metric_contract_version",
        )
    )
    runtime_metric_source = policy_contract.get(
        "metric_state_source", MTVJ_LEGACY_METRIC_STATE_SOURCE
    )
    runtime_metric_version = int(
        policy_contract.get(
            "metric_contract_version", MTVJ_LEGACY_METRIC_CONTRACT_VERSION
        )
    )
    if policy_relation_state is not None and has_relation_contract:
        common_expected = {
            "metric_tokens_enabled": True,
            "metric_state_dim": 8,
            "metric_d_model": config.hidden_dim,
        }
        mismatched = {
            key: (policy_contract.get(key), expected)
            for key, expected in common_expected.items()
            if policy_contract.get(key) != expected
        }
        allowed_runtime = {
            (MTVJ_LEGACY_METRIC_STATE_SOURCE, MTVJ_LEGACY_METRIC_CONTRACT_VERSION),
            (MTVJ_METRIC_STATE_SOURCE, MTVJ_METRIC_CONTRACT_VERSION),
        }
        if mismatched or (runtime_metric_source, runtime_metric_version) not in allowed_runtime:
            raise ValueError(
                "主 checkpoint 的 MT-VJ metric 契约不兼容："
                f"fields={mismatched}, source/version="
                f"{runtime_metric_source!r}/{runtime_metric_version}"
            )
    metric_head._mtvj_metric_state_source = runtime_metric_source
    metric_head._mtvj_metric_contract_version = runtime_metric_version
    metric_head._mtvj_current_external_checkpoint_identity = dict(current_identity)
    relation_encoder = RelationStateEncoder(
        state_dim=8,
        d_model=(
            config.hidden_dim
            if policy_relation_state is not None
            else int((ckpt.get("config") or {}).get("d_model", 512))
        ),
    ).to(device)
    relation_source = (
        "main policy checkpoint"
        if policy_relation_state is not None
        else "external metric checkpoint (legacy migration)"
    )
    try:
        strict_load_state(
            relation_encoder,
            policy_relation_state
            if policy_relation_state is not None
            else ckpt["relation_encoder"],
            module_name="relation encoder",
            source=relation_source,
        )
    except ValueError as exc:
        raise ValueError(
            "MT-VJ relation encoder 与当前 8D out.p 契约不兼容。"
            "评测不能随机重建；请使用包含 mtvj_relation_encoder 的主 checkpoint，"
            "或移除 --metric-visual-checkpoint 运行明确的 dense-only benchmark。"
            f"详情：{exc}"
        ) from exc
    for module in (metric_head, relation_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    state_dim = 8
    with torch.no_grad():
        probe_g, _ = relation_encoder(
            torch.zeros(1, state_dim, device=device),
            torch.zeros(1, state_dim, device=device),
        )
    if probe_g.shape[-1] != config.hidden_dim:
        raise ValueError(
            f"relation encoder d_model={probe_g.shape[-1]} != "
            f"VACompoundConfig.hidden_dim={config.hidden_dim}"
            "（metric_tokens 加入每层 action cross-attention 需同维）"
        )
    print(
        "eval: 冻结 metric head "
        f"（params={sum(p.numel() for p in metric_head.parameters()):,}）"
        "+ relation encoder "
        f"（params={sum(p.numel() for p in relation_encoder.parameters()):,}）"
        f" from {relation_source}; metric head from {metric_source}; "
        f"constructor config from {constructor_source}"
    )
    return metric_head, relation_encoder


def _load_dino_metric_from_policy(ckpt: dict, config, device):
    """DINO-metric（2026-08-15）：从主 checkpoint 严格重建 DINO metric 栈。

    与训练 _build_dino_metric_stack 同构：LanguageMetricField 构造配置取
    ckpt["mtvj_metric_head_config"]（h_dim=1024、grid=16），权重取
    ckpt["mtvj_metric_head"]；RelationStateEncoder(state_dim=8,
    d_model=config.hidden_dim) 取 ckpt["mtvj_relation_encoder"]。评测禁止
    随机重建——缺失即 fail-fast。
    """
    from va_compound.metric_visual_head import LanguageMetricField, RelationStateEncoder

    metric_state = ckpt.get("mtvj_metric_head")
    metric_config = ckpt.get("mtvj_metric_head_config")
    relation_state = ckpt.get("mtvj_relation_encoder")
    if metric_state is None or metric_config is None or relation_state is None:
        raise ValueError(
            "DINO-metric checkpoint 缺少 mtvj_metric_head / "
            "mtvj_metric_head_config / mtvj_relation_encoder；评测不能随机重建"
        )
    ctor_config = _canonical_mtvj_metric_head_config(metric_config, require_complete=True)
    if int(ctor_config["h_dim"]) != int(config.main_vision_dim) or int(
        ctor_config["grid"]
    ) != 16:
        raise ValueError(
            "DINO-metric checkpoint 的 metric head 构造配置与 DINO 特征不兼容："
            f"h_dim={ctor_config['h_dim']} (期望 {config.main_vision_dim}), "
            f"grid={ctor_config['grid']} (期望 16)"
        )
    metric_head = LanguageMetricField(
        lang_dim=int(ctor_config["lang_dim"]),
        h_dim=int(ctor_config["h_dim"]),
        d_proj=int(ctor_config["d_proj"]),
        n_roles=int(ctor_config["n_roles"]),
        l2_norm=bool(ctor_config["l2_norm"]),
        learnable_temp=bool(ctor_config["learnable_temp"]),
        temp_init=float(ctor_config["temp_init"]),
        freeze_bias=bool(ctor_config["freeze_bias"]),
        mode_readout=bool(ctor_config["mode_readout"]),
        grid=int(ctor_config["grid"]),
    ).to(device)
    metric_head.load_state_dict(metric_state, strict=True)
    metric_head._mtvj_metric_state_source = (
        ckpt.get("training_contract", {}) or {}
    ).get("metric_state_source", MTVJ_LEGACY_METRIC_STATE_SOURCE)
    relation_encoder = RelationStateEncoder(
        state_dim=8, d_model=int(config.hidden_dim)
    ).to(device)
    relation_encoder.load_state_dict(relation_state, strict=True)
    for module in (metric_head, relation_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    print(
        f"eval: DINO-metric 冻结 metric head（grid={metric_head.grid}, "
        f"dense_tokens={metric_head.dense_tokens}, "
        f"params={sum(p.numel() for p in metric_head.parameters()):,}）+ "
        f"relation encoder（params={sum(p.numel() for p in relation_encoder.parameters()):,}）"
        " 从主 checkpoint 严格恢复",
        flush=True,
    )
    return metric_head, relation_encoder


def _load_dino_roi_head(path: Path, device):
    """Strict task35 DINO ROI v2 loader shared with policy training."""
    roi_head = load_dino_metric_roi_checkpoint(path, device)
    print(
        f"eval: task35 DINO ROI 精修头就绪（grid={roi_head.grid}, "
        f"params={sum(p.numel() for p in roi_head.parameters()):,}, "
        f"contract={DINO_METRIC_ROI_CONTRACT}, roles={TASK35_METRIC_ROLE_CONTRACT}, "
        f"roi_config={roi_head._mtvj_roi_config}）",
        flush=True,
    )
    return roi_head


def _mtvj_metric_tokens(
    metric_head,
    relation_encoder,
    dense_evidence: dict[int, torch.Tensor],
    language_hidden: torch.Tensor,
    language_mask: torch.Tensor,
    coords: torch.Tensor,
    g_prev: torch.Tensor | None,
    device,
    *,
    roi_head: nn.Module | None = None,
    roi_backbone: nn.Module | None = None,
    roi_video: torch.Tensor | None = None,
    roi_alpha: float = 0.0,
    roi_dino: bool = False,
    trace_out: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单决策 metric_tokens（与 train.py ``_mtvj_online_encode`` 的 metric 分支同构）。

    - g_t = ``out.p * out.visibility`` 展平的8维门控图像坐标；
    - ν_t = g_t − g_{t−1}（首决策 ν≡0，与训练 T 序列首决策一致）；
    - z_g, z_nu = RelationStateEncoder(g, ν) → metric_tokens [B=1, 2, d_model]。
    """
    head_dtype = next(metric_head.parameters()).dtype
    out = metric_head(
        dense_evidence[5].to(dtype=head_dtype),
        dense_evidence[11].to(dtype=head_dtype),
        language_hidden.to(device=device, dtype=head_dtype),
        language_mask.to(device=device),
        coords.to(device=device, dtype=head_dtype),
    )
    if roi_head is not None and roi_alpha != 0.0:
        if roi_backbone is None or roi_video is None:
            raise ValueError("MT-VJ ROI runtime requires raw video and frozen backbone")
        if roi_dino:
            out.p, out.visibility = refine_metric_roi_positions_dino(
                out.p,
                out.visibility,
                roi_video,
                roi_backbone,
                roi_head,
                language_hidden.to(device=device),
                language_mask.to(device=device),
                coords,
                alpha=roi_alpha,
            )
        else:
            out.p, out.visibility = refine_metric_roi_positions(
                out.p,
                out.visibility,
                roi_video,
                roi_backbone,
                roi_head,
                language_hidden.to(device=device),
                language_mask.to(device=device),
                coords,
                alpha=roi_alpha,
            )
    # 与 train.py _mtvj_online_encode 完全同构：不可见角色坐标归零。
    g = _mtvj_metric_positions(
        out,
        getattr(
            metric_head,
            "_mtvj_metric_state_source",
            MTVJ_LEGACY_METRIC_STATE_SOURCE,
        ),
    ).detach()[0]
    if trace_out is not None:
        trace_out.update(
            {
                "role_order": ["tool", "object", "target", "interface"],
                "predicted_yx": out.p[0].detach().float().cpu().tolist(),
                "visibility": out.visibility[0].detach().float().cpu().tolist(),
                "operative_metric_g": g.detach().float().cpu().tolist(),
            }
        )
    nu = torch.zeros_like(g) if g_prev is None else g - g_prev
    if trace_out is not None:
        trace_out["metric_nu"] = nu.detach().float().cpu().tolist()
    z_g, z_nu = relation_encoder(g[None], nu[None])
    metric_tokens = torch.stack((z_g, z_nu), dim=1)  # [1, 2, d_model]
    return metric_tokens, g.detach()


def _decision_mtvj_context(
    *,
    frames,
    clip: torch.Tensor,
    mtvj_backbone,
    metric_head,
    relation_encoder,
    language_hidden: torch.Tensor,
    language_mask: torch.Tensor,
    coords_mtvj: torch.Tensor | None,
    metric_g_prev: torch.Tensor | None,
    device,
    local_slots: bool,
    roi_head: nn.Module | None = None,
    roi_alpha: float = 0.0,
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """One closed-loop decision of MT-VJ dense + metric context.

    Returns ``(dense_kwargs, metric_g, vision_override, metric_g_prev_next)``.
    ``vision_override`` is Pool16(H11) when local slots are off, else None
    (caller keeps the slot / coarse vision already computed).
    """
    dense_evidence = mtvj_backbone.forward_hierarchical_dense(clip.unsqueeze(0))
    dense_evidence = {layer: ev.float() for layer, ev in dense_evidence.items()}
    vision_override = None
    if not local_slots:
        vision_override = pool_mtvj_coarse_tokens(dense_evidence[11])
    metric_tokens = None
    metric_g = None
    if metric_head is not None:
        roi_video = None
        if roi_head is not None and roi_alpha != 0.0:
            raw_window = np.stack(frames, axis=0)[None, None]
            roi_video = prepare_metric_roi_video(
                raw_window,
                device,
                image_size=None,
            )
        metric_tokens, metric_g = _mtvj_metric_tokens(
            metric_head,
            relation_encoder,
            dense_evidence,
            language_hidden,
            language_mask,
            coords_mtvj,
            metric_g_prev,
            device,
            roi_head=roi_head,
            roi_backbone=mtvj_backbone,
            roi_video=roi_video,
            roi_alpha=roi_alpha if roi_head is not None else 0.0,
        )
    dense_kwargs = {
        "dense_evidence": dense_evidence,
        "metric_tokens": metric_tokens,
        "metric_g": (metric_g[None] if metric_g is not None else None),
    }
    return dense_kwargs, metric_g, vision_override, (
        metric_g if metric_g is not None else metric_g_prev
    )


VISION_WINDOW = 4
OBSERVATION_STRIDE = 2
LEGACY_TRAINING_CONTROL_STRIDE = 6
LEGACY_EXECUTION_HORIZON = 6
EXPECTED_WMRM_WORLD_HORIZON = 6
ACTION_HORIZON = 8
SUPPORTED_EXECUTION_HORIZONS = (1, 2, 3, 6, 15)


@dataclasses.dataclass(frozen=True)
class Plan:
    """A decoded action plan indexed by absolute environment time."""

    start_step: int
    actions: np.ndarray

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if self.start_step < 0:
            raise ValueError("plan start_step must be non-negative")
        if actions.ndim != 2 or actions.shape[0] < 1:
            raise ValueError("plan actions must have shape [time, action_dim]")
        object.__setattr__(self, "actions", actions)

    @property
    def stop_step(self) -> int:
        return self.start_step + self.actions.shape[0]

    def action_at(self, step: int) -> np.ndarray:
        if not self.start_step <= step < self.stop_step:
            raise IndexError(
                f"step {step} is outside plan [{self.start_step}, {self.stop_step})"
            )
        return self.actions[step - self.start_step]


class SynchronousPlanQueue:
    """Absolute-time queue with synchronous, hard plan replacement."""

    def __init__(self, execution_horizon: int) -> None:
        if execution_horizon not in SUPPORTED_EXECUTION_HORIZONS:
            raise ValueError(
                "execution_horizon must be one of "
                f"{SUPPORTED_EXECUTION_HORIZONS}, got {execution_horizon}"
            )
        self.execution_horizon = execution_horizon
        self._plan: Plan | None = None

    def needs_plan(self, step: int) -> bool:
        return self._plan is None or step >= self._plan.stop_step

    def replace(self, step: int, actions: np.ndarray) -> Plan:
        decoded = np.asarray(actions)
        if decoded.ndim != 2 or decoded.shape[0] < self.execution_horizon:
            raise ValueError(
                "decoded plan must have shape [time, action_dim] and contain at least "
                f"execution_horizon={self.execution_horizon} actions"
            )
        self._plan = Plan(step, decoded[: self.execution_horizon])
        return self._plan

    @property
    def plan(self) -> Plan | None:
        """The active plan, exposed read-only for telemetry and tests."""
        return self._plan

    def action_at(self, step: int) -> np.ndarray:
        if self._plan is None:
            raise RuntimeError("no active plan")
        return self._plan.action_at(step)


def _action_trace_metrics(world_raw: np.ndarray, flow_raw: np.ndarray) -> dict[str, Any]:
    """Compare peer readouts before and after the execution clip."""
    world = np.asarray(world_raw, dtype=float)
    flow = np.asarray(flow_raw, dtype=float)
    if world.shape != flow.shape or world.ndim != 2 or world.shape[1] != 4:
        raise ValueError("peer trace actions must share shape [horizon, 4]")
    world_clipped = np.clip(world, -1.0, 1.0)
    flow_clipped = np.clip(flow, -1.0, 1.0)
    pre = world - flow
    post = world_clipped - flow_clipped
    world_sat = np.abs(world) > 1.0
    flow_sat = np.abs(flow) > 1.0
    return {
        "world_raw": world.tolist(),
        "flow_raw": flow.tolist(),
        "world_clipped": world_clipped.tolist(),
        "flow_clipped": flow_clipped.tolist(),
        "distance": {
            "xyz_preclip_l2": np.linalg.norm(pre[:, :3], axis=1).tolist(),
            "xyz_postclip_l2": np.linalg.norm(post[:, :3], axis=1).tolist(),
            "gripper_preclip_abs": np.abs(pre[:, 3]).tolist(),
            "gripper_postclip_abs": np.abs(post[:, 3]).tolist(),
        },
        "saturation_disagreement": {
            "xyz": np.any(world_sat[:, :3] != flow_sat[:, :3], axis=1).tolist(),
            "gripper": (world_sat[:, 3] != flow_sat[:, 3]).tolist(),
        },
    }


def _assembly_trace_state(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Read reward-authoritative assembly geometry and success."""
    target = np.asarray(env._target_pos, dtype=float).reshape(3)
    nut = np.asarray(env._get_site_pos("RoundNut"), dtype=float).reshape(3)
    grasp_handle = np.asarray(
        env._get_site_pos("RoundNut-8"), dtype=float
    ).reshape(3)
    metric_object = np.asarray(env._get_pos_objects(), dtype=float).reshape(-1, 3)[0]
    tcp = np.asarray(env.tcp_center, dtype=float).reshape(3)
    delta = target - nut
    xy_radius = float(np.linalg.norm(delta[:2]))
    delta_z = float(delta[2])
    reward_z_condition = bool(delta_z > 0.0)

    def optional_reward_term(name: str) -> float | None:
        value = info.get(name)
        return None if value is None else float(value)

    return {
        "target_pos": target.tolist(),
        "round_nut": nut.tolist(),
        "round_nut_reward_site_xyz": nut.tolist(),
        "grasp_handle_roundnut8_xyz": grasp_handle.tolist(),
        "metric_object": metric_object.tolist(),
        "tcp": tcp.tolist(),
        "xy_radius": xy_radius,
        "z_gap": delta_z,
        "aligned": bool(xy_radius < 0.02),
        # Backward-compatible alias. This is only the reward's vertical sign
        # test; it does not mean that the nut is grasped, in contact, or hooked.
        "hooked": reward_z_condition,
        "reward_z_condition": reward_z_condition,
        "tcp_to_metric_object": float(np.linalg.norm(tcp - metric_object)),
        "tcp_to_grasp_handle": float(np.linalg.norm(tcp - grasp_handle)),
        "metric_object_to_target": float(np.linalg.norm(target - metric_object)),
        "near_object": optional_reward_term("near_object"),
        "grasp_success": optional_reward_term("grasp_success"),
        "grasp_reward": optional_reward_term("grasp_reward"),
        "in_place_reward": optional_reward_term("in_place_reward"),
        "obj_to_target": optional_reward_term("obj_to_target"),
        "success": bool(info.get("success", False)),
    }


def _validate_peer_eval_trace(
    *, output_json: Path | None, va_world_mode: str, peer_world_off: bool
) -> None:
    if output_json is None:
        raise ValueError("--peer-eval-trace requires --output-json")
    if va_world_mode != "peer_sync_h6":
        raise ValueError("--peer-eval-trace requires a peer_sync_h6 checkpoint")
    if peer_world_off:
        raise ValueError("--peer-eval-trace is incompatible with --peer-world-off")


def _peer_world_trace_stages(model: Any) -> list[dict[str, Any]]:
    """Return every peer stage's predicted and operative executable action.

    ``readout`` is the deterministic action decoded from that stage's input VA
    tokens. ``operative`` is copied from the WAM auxiliary and is therefore the
    authoritative action that actually conditioned the World predictor.  They
    are equal at deployment, but differ in the logged-action training branch.
    """
    stages = sorted(model._wmrm_inject_layers())
    pre_actions = list(getattr(model, "last_wmrm_pre_actions", None) or ())
    auxes = list(getattr(model, "last_wmrm_auxes", None) or ())
    if not stages or len(pre_actions) != len(stages) or len(auxes) != len(stages):
        raise RuntimeError("peer trace has no aligned World stage snapshots")
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for stage, pre_action, aux in zip(stages, pre_actions, auxes, strict=True):
            readout = model.world_action_readout(
                pre_action[:, : model.wmrm.cycle_steps]
                if model.config.action_horizon == 50
                else pre_action
            )
            operative = getattr(aux, "env_action", None)
            if operative is None or tuple(operative.shape) != tuple(readout.shape):
                raise RuntimeError(
                    f"peer trace stage {stage} has no full-horizon operative action"
                )
            records.append(
                {
                    "stage": int(stage),
                    "readout": readout[0].detach().cpu().numpy().copy(),
                    "operative": operative[0].detach().cpu().numpy().copy(),
                }
            )
    return records


def _peer_world_trace_readout(model: Any) -> tuple[int, np.ndarray]:
    """Backward-compatible final-stage view used by existing trace analysis."""
    final = _peer_world_trace_stages(model)[-1]
    return int(final["stage"]), np.asarray(final["readout"]).copy()


def _peer_trace_stage_metrics(
    stages: list[dict[str, Any]], flow: np.ndarray
) -> list[dict[str, Any]]:
    """Serialize all stages, using the predictor's operative action as primary."""
    result = []
    stage_count = len(stages)
    for ordinal, record in enumerate(stages):
        readout = np.asarray(record["readout"], dtype=float)
        operative = np.asarray(record["operative"], dtype=float)
        operative_vs_flow = _action_trace_metrics(
            operative, np.asarray(flow)[: operative.shape[0]]
        )
        readout_vs_operative = _action_trace_metrics(readout, operative)
        result.append(
            {
                "stage_ordinal": int(ordinal),
                "stage": int(record["stage"]),
                "va_layer_index": int(record["stage"]),
                "stage_count": int(stage_count),
                "operative_matches_readout": bool(
                    np.array_equal(operative, readout)
                ),
                "operative_raw": operative.tolist(),
                "readout_raw": readout.tolist(),
                "operative_vs_flow": operative_vs_flow,
                "readout_vs_operative": readout_vs_operative,
            }
        )
    return result


def _peer_world_effect_metrics(
    world_on_flow: np.ndarray, world_off_flow: np.ndarray
) -> dict[str, Any]:
    """Compare two Flow chunks generated from the same state and noise."""
    metrics = _action_trace_metrics(world_on_flow, world_off_flow)
    return {
        "world_on_flow_raw": metrics["world_raw"],
        "world_off_flow_raw": metrics["flow_raw"],
        "world_on_flow_clipped": metrics["world_clipped"],
        "world_off_flow_clipped": metrics["flow_clipped"],
        "distance": metrics["distance"],
        "saturation_disagreement": metrics["saturation_disagreement"],
    }


def _condition_trace_summary(condition: torch.Tensor, prefix_steps: int = 6) -> dict[str, Any]:
    """Compact decision-level features for held-out representation probes."""
    if condition.ndim != 3 or condition.shape[0] != 1:
        raise ValueError("trace condition must have shape [1, horizon, hidden]")
    horizon = condition.shape[1]
    prefix = min(max(int(prefix_steps), 1), horizon)
    value = condition[0].detach().float().cpu()
    result = {
        "horizon": int(horizon),
        "hidden_dim": int(value.shape[-1]),
        "token0": value[0].tolist(),
        "all_mean": value.mean(dim=0).tolist(),
        "prefix_mean": value[:prefix].mean(dim=0).tolist(),
    }
    if prefix < horizon:
        result["tail_mean"] = value[prefix:].mean(dim=0).tolist()
    return result


def _assembly_metric_oracle(env: Any) -> dict[str, Any]:
    """Project simulator-authoritative metric roles into the rendered camera."""
    from prepare_metaworld_metric import (
        RENDER_SIZE,
        keypoint_world_positions,
        project_points,
    )

    world = keypoint_world_positions(env, "assembly-v3")
    if world is None or np.asarray(world).shape != (4, 3):
        raise RuntimeError("assembly metric oracle did not expose four 3-D roles")
    pixels_xy, depth = project_points(env, np.asarray(world, dtype=float))
    normalized_yx = np.stack(
        (pixels_xy[:, 1] / RENDER_SIZE, pixels_xy[:, 0] / RENDER_SIZE), axis=1
    )
    reward_center = np.asarray(env._get_site_pos("RoundNut"), dtype=float).reshape(3)
    grasp_handle = np.asarray(env._get_site_pos("RoundNut-8"), dtype=float).reshape(3)
    extra_world = np.stack((grasp_handle, reward_center))
    extra_xy, extra_depth = project_points(env, extra_world)
    extra_yx = np.stack(
        (extra_xy[:, 1] / RENDER_SIZE, extra_xy[:, 0] / RENDER_SIZE), axis=1
    )
    return {
        "role_order": ["tool", "object", "target", "interface"],
        "world_xyz": np.asarray(world, dtype=float).tolist(),
        "image_yx": normalized_yx.tolist(),
        "camera_depth": np.asarray(depth, dtype=float).tolist(),
        "assembly_extra": {
            "role_order": ["grasp_handle_roundnut8", "reward_center_roundnut"],
            "world_xyz": extra_world.tolist(),
            "image_yx": extra_yx.tolist(),
            "camera_depth": np.asarray(extra_depth, dtype=float).tolist(),
            "separation_m": float(np.linalg.norm(grasp_handle - reward_center)),
        },
    }


def _append_peer_trace_token(
    decision: dict[str, Any],
    *,
    token: int,
    env_step: int,
    normalized_command: np.ndarray,
    denormalized_command: np.ndarray,
    pre_tcp: np.ndarray,
    post_tcp: np.ndarray,
    reward: float,
    pre_assembly: dict[str, Any],
    assembly: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> None:
    flow_raw = np.asarray(decision["flow_raw"], dtype=float)
    if token != len(decision["tokens"]) or not 0 <= token < flow_raw.shape[0]:
        raise RuntimeError(f"peer trace token {token} is not the next planned token")
    normalized = np.asarray(normalized_command, dtype=float)
    if not np.allclose(normalized, np.clip(flow_raw[token], -1.0, 1.0)):
        raise RuntimeError("peer trace normalized command does not match planned Flow token")
    decision["tokens"].append(
        {
            "decision": int(decision["decision"]),
            "token": int(token),
            "env_step": int(env_step),
            "pre_tcp": np.asarray(pre_tcp, dtype=float).reshape(3).tolist(),
            "post_tcp": np.asarray(post_tcp, dtype=float).reshape(3).tolist(),
            "normalized_command": normalized.tolist(),
            "denormalized_command": np.asarray(denormalized_command, dtype=float).tolist(),
            "reward": float(reward),
            "pre_step_assembly": pre_assembly,
            "post_step_assembly": assembly,
            **assembly,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal": bool(assembly["success"] or terminated or truncated),
        }
    )
    decision["executed_token_count"] = len(decision["tokens"])


def resolve_execution_horizon(
    args: argparse.Namespace,
    config: Any | None = None,
) -> int:
    explicit = getattr(args, "execution_horizon", None)
    legacy = getattr(args, "execute_steps", None)
    if explicit is not None and legacy is not None and explicit != legacy:
        raise ValueError(
            "--execution-horizon and legacy --execute-steps disagree: "
            f"{explicit} != {legacy}"
        )
    peer_sync = (
        config is not None
        and getattr(config, "va_world_mode", "legacy") == "peer_sync_h6"
    )
    allow_ablation = bool(
        getattr(args, "allow_execution_horizon_ablation", False)
    )
    if allow_ablation and not peer_sync:
        raise ValueError(
            "--allow-execution-horizon-ablation requires a peer_sync_h6 checkpoint"
        )
    planning_stride = int(
        getattr(config, "planning_stride", LEGACY_EXECUTION_HORIZON)
    )
    if peer_sync and planning_stride not in SUPPORTED_EXECUTION_HORIZONS:
        raise ValueError(
            "peer_sync_h6 checkpoint planning_stride must be one of "
            f"{SUPPORTED_EXECUTION_HORIZONS}, got {planning_stride}"
        )
    deployment_horizon = int(
        getattr(config, "deployment_execution_horizon", 0) or planning_stride
    )
    default = deployment_horizon if peer_sync else LEGACY_EXECUTION_HORIZON
    value = default if explicit is None and legacy is None else int(
        explicit if explicit is not None else legacy
    )
    if value not in SUPPORTED_EXECUTION_HORIZONS:
        raise ValueError(
            f"--execution-horizon must be one of {SUPPORTED_EXECUTION_HORIZONS}"
        )
    if peer_sync and value != deployment_horizon and not allow_ablation:
        raise ValueError(
            "peer_sync_h6 requires execution_horizon == checkpoint deployment "
            f"horizon ({value} != {deployment_horizon})"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop MetaWorld MT50 eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="metaworld_features.pt for normalization")
    parser.add_argument(
        "--language-features",
        type=Path,
        default=None,
        help="Optional compact task/language cache. Use the MT50 cache to append "
        "push-back while --features keeps the frozen normalization/cadence.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vision-pooling",
        choices=("flat", "spatial", "spatiotemporal", "dense"),
        default=None,
        help="在线 V-JEPA 池化（覆盖 training_contract；Stage A/B 288-token "
        "checkpoint 必须为 spatiotemporal，Step 0 dense_readout checkpoint "
        "必须为 dense（1152 patch 不池化），否则闭环数字失真）",
    )
    parser.add_argument(
        "--dense-readout-mtvj",
        action="store_true",
        help="MT-VJ 在线 dense 解码路径（契约 §7，与训练 §6 同构）：每决策点同一 "
        "4 帧历史窗 [d-6, d-4, d-2, d] → 冻结 V-JEPA forward_hierarchical_dense "
        "→ dense_evidence 注入 encode_condition（config.dense_readout_mtvj 强制 "
        "True；ckpt 无 dense 权重时零初始化初始等价）。与 c2/recovery 部署互斥"
        "（servo plan 分支已接同一 dense/metric 上下文）",
    )
    parser.add_argument(
        "--action-vision-checkpoint",
        type=Path,
        default=None,
        help="Local timm-compatible weights for a checkpoint-trained DINO "
        "action tower. Required iff config.action_vision_backbone is enabled.",
    )
    parser.add_argument(
        "--main-vision-checkpoint",
        type=Path,
        default=None,
        help="DINO-main replacement（2026-08-14 用户决策）：本地 timm 权重。"
        "Required iff config.main_vision_backbone != 'vjepa'（DINOv2 替换 "
        "V-JEPA 作为 VA 主视觉；V-JEPA/dense/metric 路径保留但禁用）。",
    )
    parser.add_argument(
        "--metric-visual-checkpoint",
        type=Path,
        default=None,
        help="MT-VJ metric 视觉预训练 checkpoint（契约 §2：config/metric_head/"
        "relation_encoder/contract='mt_vj_metric_field_v1'）。提供时每决策点经 "
        "LanguageMetricField（g_t = out.p 四角色坐标展平）+ RelationStateEncoder "
        "（ν_t = g_t − g_{t−1}）生成 metric_tokens 注入模型；缺省 None（回退 "
        "metric_tokens=None，与训练无 metric head 分支一致）",
    )
    parser.add_argument(
        "--mtvj-dense-only-ablation",
        action="store_true",
        help="显式关闭 checkpoint 期待的 MT-VJ metric tokens，只评测 dense H5/H11 "
        "action readout 消融；普通 benchmark 禁止静默关闭 metric 路径。",
    )
    parser.add_argument(
        "--mtvj-roi-checkpoint",
        type=Path,
        default=None,
        help="可选 MT-VJ 原图 ROI 精修 checkpoint；默认关闭。",
    )
    parser.add_argument(
        "--mtvj-roi-alpha",
        type=float,
        default=None,
        help="ROI 有界残差融合系数 [0,1]；启用 ROI checkpoint 时必须显式给出。",
    )
    parser.add_argument(
        "--dino-roi-checkpoint",
        type=Path,
        default=None,
        help="可选 DINO 原图 ROI 精修 checkpoint（contract=dino_metric_roi_v1）；"
        "默认关闭。只对 dino_dense_metric 主 checkpoint 有效。",
    )
    parser.add_argument(
        "--dino-roi-alpha",
        type=float,
        default=None,
        help="DINO ROI 有界残差融合系数 [0,1]；启用 DINO ROI checkpoint 时"
        "必须显式给出。",
    )
    parser.add_argument(
        "--dino-feature-cache",
        type=Path,
        default=None,
        help="task35 precision provenance check only: cache directory whose "
        "block11/block23 SHA-256 identities must match the policy checkpoint; "
        "closed-loop frames remain encoded online.",
    )
    parser.add_argument(
        "--record-action-chunks",
        action="store_true",
        help="Include decoded normalized chunks in per-trial JSON for paired divergence analysis.",
    )
    parser.add_argument(
        "--peer-eval-trace",
        action="store_true",
        help=(
            "Opt-in per-decision/token JSON trace for peer World/Flow assembly "
            "diagnostics; requires --output-json and a peer checkpoint."
        ),
    )
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument(
        "--episode-seed-base",
        type=int,
        default=None,
        help="Use the shared reset sequence BASE..BASE+trials-1 for every task. "
        "Formal MT50 acceptance uses 4042; omitted preserves legacy task-specific seeds.",
    )
    parser.add_argument(
        "--trial-range",
        type=str,
        default=None,
        help="分片评测的 trial 半开区间 START:END（缺省 = 全部）。每个 trial 开头都用 "
        "evaluation_episode_seed 重新播种 numpy/env/torch，跨 trial 无 RNG 依赖，"
        "因此分片结果与串行逐 trial 完全一致。用于把 trial 拆到多进程并行：闭环"
        "评测 GPU 利用率仅 0~6%%，瓶颈是单线程软件渲染（容器无 /dev/dri，EGL 退回 "
        "llvmpipe，硬件渲染不可用），并行是唯一的大幅加速手段。",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="实时显示闭环 corner2 RGB 画面；按 q 可提前关闭显示窗口",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Atomically write per-trial success/stage telemetry and aggregate CI.",
    )
    parser.add_argument(
        "--dagger-output-dir",
        type=Path,
        default=None,
        help="Collect current-policy states followed by scripted-expert takeover.",
    )
    parser.add_argument("--dagger-takeover-min", type=int, default=45)
    parser.add_argument("--dagger-takeover-max", type=int, default=120)
    parser.add_argument(
        "--dagger-prefix-keep",
        type=int,
        default=45,
        help="Model-controlled steps retained before expert takeover.",
    )
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="逗号分隔的任务索引子集（从 features metadata.tasks 里选，8 任务诊断用）；"
        "缺省 = 前 --max-tasks 个",
    )
    parser.add_argument(
        "--plan-refresh",
        type=int,
        default=0,
        help="Plan-Cache 闭环刷新间隔：每 R 个决策用当前场景重建语言缓存 "
        "（0 = 仅 episode 首帧建一次；仅在 checkpoint config 含 plan_resampler "
        "或 scene_teacher 时可用）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="每 episode 最大步数；400 与 Evo-1/FabriVLA MT50 报告口径一致。"
        "旧诊断若需要环境上限须显式传 500。",
    )
    parser.add_argument(
        "--mt50-benchmark",
        action="store_true",
        help="严格要求 50 tasks × 10 trials、400 steps、共享 seeds 4042..4051，"
        "并输出 EvoMind 四难度桶等权平均。",
    )
    parser.add_argument(
        "--align-init",
        action="store_true",
        help="reset 后对齐数据首帧 init（物体+target），验证模型在训练分布上的闭环能力",
    )
    parser.add_argument(
        "--debug-first-action",
        action="store_true",
        help="打印首次决策的模型动作（与 --align-init 联用可对比数据专家动作）",
    )
    parser.add_argument(
        "--task35-precision-contract",
        action="store_true",
        help="最终 task35 评测 fail-fast：H6/grid16/temporal/geometry/ROI、"
        "execute=6、50 trials、stage telemetry、WAM4VA state exchange off。",
    )
    parser.add_argument(
        "--debug-stage-metrics",
        action="store_true",
        help="逐 trial 汇报 MetaWorld 阶段指标（max near/grasp/in_place、min obj_to_target），"
        "只读诊断，不改变动作或成功判定。",
    )
    parser.add_argument(
        "--task35-causal-ablation",
        choices=("none", "temporal-reverse", "geometry-zero", "geometry-shuffle", "roi-off", "dense-zero"),
        default="none",
        help="task35 FM causal diagnostic; changes exactly one inference signal while "
        "keeping checkpoint and episode seeds fixed. Non-none runs are diagnostics, "
        "not precision acceptance.",
    )
    parser.add_argument(
        "--peer-world-off",
        action="store_true",
        help=(
            "causal closed-loop control: bypass peer World state exchange while "
            "keeping checkpoint, Flow sampling, observations, and episode seeds fixed"
        ),
    )
    parser.add_argument(
        "--flow-samples",
        type=int,
        default=1,
        help="每次决策独立采样 K 个 flow action chunk 后取均值（默认 1，原行为）；"
        "仅用于诊断 FM 采样方差。",
    )
    parser.add_argument(
        "--memory-reset-every",
        type=int,
        default=0,
        help="每 N 个决策点将递归视觉记忆置 None（0 = 不重置）。训练只展开 T=4 次而部署"
        "连续递归几十次——此参数将部署记忆截断到与训练一致的递归深度，零训练代价的"
        "契约缺口对照（2026-08-06 Codex 判决顺序第 3 步）",
    )
    parser.add_argument(
        "--world-reset-every",
        type=int,
        default=4,
        help="每 N 个决策点只重置 world_state（WAMState）为 None，保留 VA 视觉记忆"
        "（0 = 不重置）。训练 sequence_length=4 时 world_state 只跨 4 步累积"
        "（每步 8 stage propose）；peer_sync_h6 长 horizon 闭环把同一段无约束 belief "
        "循环推到数千次 propose 而发散到 NaN。此参数把部署的 world 递归深度截断回"
        "训练分布，是零训练代价的分布对齐（对应 --memory-reset-every 但只动 world）。",
    )
    parser.add_argument(
        "--world-map-reset-every",
        type=int,
        default=0,
        help="每 N 个决策点只把 world_map 置 None、保留 belief（0 = 不重置）。"
        "wmrm.py 的 map 基底是 previous_map，跨 stage 与跨决策点持续累积，一集 2000 "
        "次 propose 全在自己上一次预测上加残差、从不重锚真实观测；--world-reset-every "
        "压住了这个发散，但代价是 belief 每 4 个决策点一起被清空。此参数用 N=1 把 map "
        "的开环深度截断到单个决策点的 8 个 stage，同时让 belief 跨整集存活。",
    )
    parser.add_argument(
        "--prev-zero",
        action="store_true",
        help="把 previous_action 输入恒置零（归一化 0）。previous_action 训练用真值、"
        "闭环用模型自身输出（自激）——此参数将闭环 prev 改为恒零，零训练代价的"
        "自激对照（2026-08-06 Codex 16-task panel 条件）",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        choices=SUPPORTED_EXECUTION_HORIZONS,
        help=(
            "部署时每次硬替换执行的动作数；仅支持 1/2/3/6/15。"
            "peer_sync_h6 默认且必须等于 checkpoint deployment horizon；"
            "legacy 默认 6。"
        ),
    )
    parser.add_argument(
        "--allow-execution-horizon-ablation",
        action="store_true",
        help=(
            "仅用于闭环评测消融：允许执行步数不同于 peer checkpoint 的部署"
            "步数。不会改写 checkpoint 的训练/部署合同。"
        ),
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=None,
        help="legacy alias for --execution-horizon; retained for old launchers.",
    )
    parser.add_argument(
        "--direct-head",
        choices=("auto", "on", "off"),
        default="auto",
        help="C²-VA Stage A 解码器：auto = 从 checkpoint config 读 direct_head"
        "（默认）；on/off = 强制 Direct Head / flow matching（消融对照）。"
        "direct 时经 decode_actions 一次前向，flow 时仍走 32 步 Euler 采样",
    )
    parser.add_argument(
        "--plan-stride",
        type=int,
        default=None,
        help="C² 部署（Codex 修正 5）：VA 生成 {ū,c̄,K} 的重规划间隔（原始步）。"
        "默认 6，与该 legacy checkpoint 的训练控制步幅一致；token 0..5 顺序消费后重规划",
    )
    parser.add_argument(
        "--feedback-stride",
        type=int,
        default=None,
        help="C² 部署：c_current 刷新间隔（原始步，V-JEPA → P）。默认 1——每原始步"
        "刷新并消费下一个 token；>1 时中间步保持上一 token 动作",
    )
    parser.add_argument(
        "--c2-oracle-ref",
        action="store_true",
        help="C² 消融：用 ground-truth 参考替代预测 c̄（c̄ ≡ c_current，e ≡ 0）——"
        "测参考零误差上界（K 修正空转，仅名义 ū 执行）",
    )
    parser.add_argument(
        "--c2-zero-gain",        action="store_true",
        help="C² 消融：增益 K 恒置零（等价 Stage A 名义执行，go/no-go 对照）",
    )
    parser.add_argument(
        "--c2-gain-scale",
        type=float,
        default=1.0,
        help="部署时对 K 的缩放系数（damping 消融：<1 减弱反馈修正）",
    )
    parser.add_argument(
        "--c2-error-threshold",
        type=float,
        default=0.0,
        help="误差死区门控：‖e‖ < 阈值时跳过 K 修正（只执行名义动作）",
    )
    parser.add_argument(
        "--c2-recovery-eval",
        type=Path,
        default=None,
        help="C² 恢复评估：从 v6b 的 held-out 扰动分支初始状态出发闭环，"
        "测'拉回'成功率（需要按钮任务 c2 checkpoint）",
    )
    parser.add_argument(
        "--c2-recovery-split",
        choices=("train", "heldout"),
        default="heldout",
        help="恢复评估用 v6b 的哪个 split（go/no-go 用 held-out 扰动种子）",
    )
    parser.add_argument(
        "--state-take",
        type=int,
        default=4,
        choices=STATE_TAKE_VALUES,
        help="proprio 截取协议（protocol_verification_evo_fabri.md §3.2 三协议）："
        "4 = EEF xyz + gripper（现状默认，与 Evo-1/FabriVLA 训练口径一致）；"
        "8 = Evo-1 官方评测输入（+obj1 pos3 + quat_w；后 4 维为训练分布外输入，"
        "经零初始化投影扩展，模型行为不变——协议复刻）；39 = 完整 MetaWorld "
        "obs；0 = proprio 恒零（RGB-only 协议）",
    )
    parser.add_argument(
        "--servo-ablation",
        choices=("none", "zero-gain", "gain-shuffle", "wrong-role", "open-loop"),
        default="none",
        help="C²-IRF v2 Step 2 伺服四消融（设计文档 §七 Step 2）：none = 正常低秩"
        "伺服；zero-gain = β≡0（增益恒零，仅名义执行）；gain-shuffle = 部署时"
        "固定种子随机打乱 K 行/列（增益语义破坏）；wrong-role = 角色循环移位"
        "（关系状态角色错位）；open-loop = 不施加任何伺服修正。需要含 servo "
        "权重的 c2 checkpoint（local_slots + multi_mode 读出）",
    )
    parser.add_argument(
        "--fovea",
        action="store_true",
        help="C²-IRF v2 Step 3 foveal 双速率部署（设计 §三.3）：plan_due 全图 dense "
        "重读 + ROI 计算，feedback 步 foveal crop 局部关系更新 + 伺服修正；"
        "servo 新息超阈值（innovation_flag）立即提前全局刷新。需要 c2 + "
        "multi_mode + servo 权重",
    )
    parser.add_argument(
        "--fovea-nu-thresh",
        type=float,
        default=0.1,
        help="新息幅度 |ν| 阈值（归一化关系坐标；fovea 立即刷新，设计 §三.3）",
    )
    parser.add_argument(
        "--fovea-h-thresh",
        type=float,
        default=0.7,
        help="配对熵 H(w) 阈值（位；fovea 立即刷新，设计 §三.3）",
    )
    parser.add_argument(
        "--fovea-vis-thresh",
        type=float,
        default=0.3,
        help="最小可见度阈值（fovea 立即刷新，设计 §三.3）",
    )
    return parser.parse_args()


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)[None]
    if tensor.shape[-1] != image_size:
        tensor = F.interpolate(
            tensor, size=(image_size, image_size), mode="bicubic",
            align_corners=False, antialias=True,  # 与 prepare_metaworld.py 管线一致
        )
    return (tensor - IMAGE_MEAN) / IMAGE_STD


@torch.inference_mode()
def _action_vision_encode_window(
    frames: list[np.ndarray],
    backbone: TimmActionVisionBackbone,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Encode ``[d-4,d]`` into the action tower's two temporal patch grids."""
    if len(frames) != VISION_WINDOW:
        raise ValueError(
            f"action vision expects {VISION_WINDOW} history frames, got {len(frames)}"
        )
    images = torch.cat(
        [
            preprocess(frames[1], backbone.image_size),
            preprocess(frames[3], backbone.image_size),
        ],
        dim=0,
    ).to(device)
    hierarchical = backbone.forward_hierarchical_dense(images)
    return {
        layer: tokens.reshape(1, -1, tokens.shape[-1])
        for layer, tokens in hierarchical.items()
    }


@torch.inference_mode()
def _main_vision_encode_window(
    frames: list[np.ndarray],
    backbone: TimmActionVisionBackbone,
    device: torch.device,
    *,
    grid: int,
    window: int,
    return_dense: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """DINO-main replacement: [1, window*grid*grid, dim] tokens per decision.

    Mirrors train.py:_dino_main_online_encode exactly: the same complete
    history window, the same bicubic resize and ImageNet normalization, the
    same final-block patch-grid average pooling.

    ``return_dense=True``（DINO-metric）：额外返回 dense evidence
    ``{5: [1, 512, D], 11: [1, 512, D]}``（block11/block23 帧 [d-2,d] patch，
    前 256 = d-2、后 256 = d，与训练同序）。
    """
    if len(frames) != VISION_WINDOW:
        raise ValueError(
            f"dino-main expects {VISION_WINDOW} history frames, got {len(frames)}"
        )
    # Encode one frame at a time so ViT-L/14-reg4 + VA + FM fit a 16 GiB laptop
    # GPU. The per-frame contract is identical to a batched 4-frame encode.
    hierarchical_parts = [
        backbone.forward_hierarchical_dense(
            preprocess(frames[i], backbone.image_size).to(device)
        )
        for i in range(window)
    ]
    hierarchical = {
        layer: torch.cat([part[layer] for part in hierarchical_parts], dim=0)
        for layer in hierarchical_parts[0]
    }
    tokens = hierarchical[11].float()  # [window, 256, D]
    if tokens.shape[-2] != 256 or tokens.shape[-1] != backbone.feature_dim:
        raise RuntimeError(
            "dino-main expects 256 patch tokens per frame, got "
            f"{tuple(tokens.shape)}"
        )
    dim = tokens.shape[-1]
    tokens = tokens.reshape(window, 16, 16, dim).permute(0, 3, 1, 2)
    tokens = F.adaptive_avg_pool2d(tokens, (grid, grid))
    tokens = tokens.permute(0, 2, 3, 1).reshape(1, window * grid * grid, dim)
    if not return_dense:
        return tokens
    # 帧 [d-2, d] = 窗口内索引 2, 3（时间升序 → 前 256 = d-2）。
    dense_evidence = {
        layer: torch.cat((ev[2], ev[3]), dim=0).float()[None]
        for layer, ev in hierarchical.items()
    }
    for layer, evidence in dense_evidence.items():
        if evidence.shape[-2] != 512:
            raise RuntimeError(
                f"dino-metric dense evidence {layer} must be 512 tokens, got "
                f"{tuple(evidence.shape)}"
            )
    return tokens, dense_evidence


def plan_refresh_due(decision_count: int, plan_refresh: int) -> bool:
    """Plan-Cache 缓存重建时机：首决策（episode 开始）必须建；之后每 R 决策一次。

    ``decision_count`` 从 1 开始计数（每 episode 第一个决策 = 1）。
    """
    if decision_count == 1:
        return True
    return plan_refresh > 0 and (decision_count - 1) % plan_refresh == 0


def _reset_world_state(memory):
    """只把 ``VisualMemory.world_state`` 置 None，保留 VA 视觉记忆。

    ``VisualMemory`` 是 frozen dataclass，必须用 ``dataclasses.replace`` 重建。
    ``world_state=None`` 会在下一次 ``encode_condition`` 里触发 ``WAMState()``
    重新初始化（model.py 对 None 有兜底）。peer_sync_h6 的 belief 是无约束
    循环，训练 sequence_length=4 只验证了 4 步窗口；长 horizon 闭环把它推到
    数千次 propose 而发散到 NaN，这里把 world 递归深度截断回训练分布。
    """
    if memory is None or getattr(memory, "world_state", None) is None:
        return memory
    return dataclasses.replace(memory, world_state=None)


def _reset_world_map(memory):
    """只把 ``WAMState.world_map`` 置 None，保留 belief。

    ``world_map`` 的残差基底是 ``clip[:, -1] if previous_map is None else
    previous_map``（wmrm.py），所以置 None 让下一个 stage 重新锚定到真实 DINO 帧。
    ``--world-reset-every`` 清整个 ``WAMState``，belief 一起没了（250 决策点 / 4 =
    每集 62 次），而 belief 正是 WAM 存在的理由。此参数只截断 map 的开环深度。
    """
    if memory is None:
        return memory
    world_state = getattr(memory, "world_state", None)
    if world_state is None or getattr(world_state, "world_map", None) is None:
        return memory
    return dataclasses.replace(
        memory, world_state=dataclasses.replace(world_state, world_map=None)
    )


def c2_schedule(
    step: int,
    plan_step: int | None,
    plan_stride: int,
    feedback_stride: int,
    horizon: int,
) -> tuple[bool, bool, int]:
    """C² 部署节奏（Codex 修正 5：plan 与 feedback 解耦）。

    Returns (plan_due, feedback_due, token_index)：
    - plan_due：距上次规划 >= plan_stride（或尚无规划）；token 用尽也强制重规划；
    - feedback_due：距规划步为 feedback_stride 整数倍（plan 步本身必刷新）；
    - token_index：自规划以来应消费的 token（= 距规划步 / feedback_stride）。
    """
    if plan_stride < 1 or feedback_stride < 1:
        raise ValueError("plan/feedback stride must be positive")
    if plan_step is None or step - plan_step >= plan_stride:
        return True, True, 0
    token_index = (step - plan_step) // feedback_stride
    if token_index >= horizon:
        return True, True, 0
    feedback_due = (step - plan_step) % feedback_stride == 0
    return False, feedback_due, token_index


# ---------------------------------------------------------------------------
# C²-IRF v2 评估协议：--state-take / --servo-ablation / --fovea
# （设计文档 c2irf_v2_vision_ablation.md §三/§七；protocol_verification_evo_fabri.md §3.2）
# ---------------------------------------------------------------------------

STATE_TAKE_VALUES = (0, 4, 8, 39)
# 4 = EEF xyz + gripper（现状默认，与 Evo-1/FabriVLA 训练口径一致）；
# 8 = Evo-1 官方评测输入（+obj1 pos3 + quat_w，后 4 维训练分布外）；
# 39 = 完整 MetaWorld obs；0 = proprio 恒零（RGB-only 协议）。

# MetaWorld（gymnasium 版）39 维 obs 布局的物理范围表
# （protocol_verification_evo_fabri.md §1.2）：仅供 --state-take 8/39 的
# 第 4 维起（训练分布外维度）归一化使用；前 4 维永远用数据统计
# （state_q01/q99），与旧行为一致。布局：0-2 eef pos，3 gripper，
# 4-6 obj1 pos，7-10 obj1 quat，11-13 obj2 pos，14-17 obj2 quat，
# 18-20 prev eef，21 prev gripper，22-28 prev obj1（pos3+quat4），
# 29-35 prev obj2，36-38 goal pos。
_STATE_LAYOUT_Q01 = np.zeros(39, dtype=np.float32)
_STATE_LAYOUT_Q99 = np.zeros(39, dtype=np.float32)
for _i in range(39):
    if _i in (0, 4, 11, 18, 22, 29, 36):  # x 坐标（eef/obj1/obj2/goal 及上一帧）
        _lo, _hi = -0.5, 0.5
    elif _i in (1, 5, 12, 19, 23, 30, 37):  # y 坐标
        _lo, _hi = 0.35, 0.95
    elif _i in (2, 6, 13, 20, 24, 31, 38):  # z 坐标
        _lo, _hi = 0.0, 0.55
    elif _i in (3, 21):  # gripper 开度
        _lo, _hi = 0.0, 1.0
    else:  # 四元数分量（7-10, 14-17, 25-28, 32-35）
        _lo, _hi = -1.0, 1.0
    _STATE_LAYOUT_Q01[_i], _STATE_LAYOUT_Q99[_i] = _lo, _hi


def state_take_normalize(
    obs: np.ndarray, take: int, sq01: np.ndarray, scale_s: np.ndarray
) -> np.ndarray:
    """--state-take 截取 + 归一化 → [-1, 1]（纯函数；take=4 与旧行为逐位一致）。

    - 0：恒零 4 维（RGB-only 协议，proprio 无信息）；
    - 4：obs[:4] 用数据 q01/q99 归一化（现状默认，公式逐字不变）；
    - 8/39：数据统计覆盖的维度（本仓库 features 为前 4 维）用数据 q01/q99，
      其余为训练分布外输入，用 39 维布局物理范围表（_STATE_LAYOUT_Q01/Q99）
      归一化——数值只进入零初始化扩展投影（无学习贡献，模型输出不变），
      纯协议复刻（protocol_verification_evo_fabri.md §3.2）。
    """
    if take == 0:
        return np.zeros(4, dtype=np.float32)
    if take > obs.shape[0]:
        raise ValueError(f"--state-take {take} 超出 obs 维度 {obs.shape[0]}")
    n_stats = min(take, sq01.shape[0])
    state = np.clip(
        2.0 * (obs[:n_stats] - sq01[:n_stats]) / scale_s[:n_stats] - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)
    if n_stats < take:
        span = _STATE_LAYOUT_Q99[n_stats:take] - _STATE_LAYOUT_Q01[n_stats:take]
        span = np.where(np.abs(span) < 1e-6, 1.0, span)
        ext = np.clip(
            2.0 * (obs[n_stats:take] - _STATE_LAYOUT_Q01[n_stats:take]) / span - 1.0,
            -1.0,
            1.0,
        ).astype(np.float32)
        state = np.concatenate([state, ext])
    return state


def extend_state_projection(model, take: int) -> None:
    """--state-take > 4：把 state_projection 输入宽度扩展到 take + action_dim。

    零初始化扩展：前（proprio_dim+action_dim）列拷贝原权重，扩展列权重零 →
    分布外维度的模型输出逐位不变（与 Evo-1 训练 4 维、评测喂 8 维的 OOD
    输入同构，protocol_verification_evo_fabri.md E3）。仅替换 eval 会话内的
    模块属性，不动 model.py 与训练；``config.proprio_dim == take``（未来用
    take 维训练的 checkpoint）时无需扩展。
    """
    proj = model.state_projection
    target_in = take + model.config.action_dim
    if proj.in_features == target_in:
        return
    if proj.in_features != model.config.proprio_dim + model.config.action_dim:
        raise ValueError(
            f"state_projection 输入宽度 {proj.in_features} 与 proprio_dim+action_dim"
            f"（{model.config.proprio_dim}+{model.config.action_dim}）不符，无法扩展"
        )
    if take < model.config.proprio_dim:
        return  # take=0 仍走 proprio_dim 维零向量
    ext = nn.Linear(target_in, proj.out_features)
    with torch.no_grad():
        ext.weight.zero_()
        ext.weight[:, : proj.in_features].copy_(proj.weight)
        ext.bias.copy_(proj.bias)
    model.state_projection = ext.to(
        device=proj.weight.device, dtype=proj.weight.dtype
    )


def validate_servo_args(args, config) -> None:
    """servo/fovea 参数校验（独立成函数便于测试；--state-take 由 argparse 限制）。

    servo 消融与 fovea 是 C²-IRF v2 部署机制：需要 local_slots + multi_mode
    读出（mu/cov/vis/slots 由多模式读出提供）；servo 训练与 c2_controller/
    direct_head 互斥（train.py --servo 校验：修正作用于 flow 输出），故 servo
    部署需要 flow checkpoint（direct 也拒：--servo 与 --direct-head 互斥）。
    """
    servo_on = args.servo_ablation != "none" or args.fovea
    if servo_on and config.c2_controller:
        raise ValueError(
            "--servo-ablation/--fovea 需要 flow checkpoint：servo 训练与 "
            "c2_controller 互斥（train.py --servo 校验，修正作用于 flow 输出）"
        )
    if servo_on and getattr(config, "direct_head", False):
        raise ValueError(
            "--servo-ablation/--fovea 需要 flow checkpoint：servo 训练与 "
            "direct_head 互斥（train.py --servo 校验）"
        )
    if servo_on and not (
        getattr(config, "local_slots", False)
        and getattr(config, "multi_mode", False)
        and not getattr(config, "local_slots_direct288", False)
    ):
        raise ValueError(
            "--servo-ablation/--fovea 需要 local_slots + multi_mode 读出"
            "（mu/cov/vis/slots 由多模式读出提供；local_slots_direct288 跳过 reader）"
        )
    if args.fovea and not getattr(config, "dense_readout", False):
        raise ValueError(
            "--fovea 需要 dense_readout checkpoint（foveal H11 固定输出 1152 token）"
        )
    if servo_on and (args.c2_oracle_ref or args.c2_zero_gain):
        # 与 main() 既有的非 c2 校验同一口径（--c2-* 需要 c2 checkpoint）。
        raise ValueError(
            "--c2-oracle-ref/--c2-zero-gain 需要 c2 checkpoint，与 servo 部署互斥"
        )
    if min(args.fovea_nu_thresh, args.fovea_h_thresh, args.fovea_vis_thresh) < 0:
        raise ValueError("--fovea-*-thresh 必须非负")


MODE_PLAN = "plan"          # 全图 dense 重读（plan_due；fovea 时同时重算 ROI）
MODE_FOVEAL = "foveal"      # ROI crop → 冻结前缀 → 局部关系更新（feedback_due，仅 fovea）
MODE_FEEDBACK = "feedback"  # 全图重读（feedback_due，非 fovea——与现状一致）
MODE_HOLD = "hold"          # 无视觉计算，保持上一动作


def fovea_schedule(
    step: int,
    plan_step: int | None,
    plan_stride: int,
    feedback_stride: int,
    horizon: int,
    *,
    fovea: bool,
    innovation_flag: bool = False,
) -> tuple[str, bool, int]:
    """C²-IRF v2 双速率 fovea 部署节奏（设计 §三.3；纯函数）。

    挂 ``c2_schedule(plan_due, feedback_due)``：plan_due 或上一步 servo 新息
    超阈值（innovation_flag=True，立即提前全局刷新）→ MODE_PLAN；feedback_due
    → fovea 时 MODE_FOVEAL（40Hz 局部关系更新），否则 MODE_FEEDBACK（全图
    重读，现状）；其余 MODE_HOLD。correction_due：PLAN/FOVEAL/FEEDBACK 步
    都施加伺服修正（40Hz 修正节奏）。返回 (mode, correction_due, token_index)
    ——与 c2_schedule 的 token_index 约定一致（自规划以来应消费的 token）。
    """
    if plan_stride < 1 or feedback_stride < 1:
        raise ValueError("plan/feedback stride must be positive")
    plan_due, feedback_due, token_index = c2_schedule(
        step, plan_step, plan_stride, feedback_stride, horizon
    )
    if plan_due or innovation_flag:
        return MODE_PLAN, True, 0
    if feedback_due:
        return (MODE_FOVEAL if fovea else MODE_FEEDBACK), True, token_index
    return MODE_HOLD, False, token_index


def fovea_refresh_due(
    *,
    nu_norm: float | None = None,
    mode_entropy: float | None = None,
    vis_min: float | None = None,
    nu_thresh: float = 0.1,
    h_thresh: float = 0.7,
    vis_thresh: float = 0.3,
) -> bool:
    """foveal 立即刷新判定（设计 §三.3）：|ν| > τ_ν 或 H(w) > τ_H 或 v < τ_v。

    任一超阈值即刷新。servo 暴露 innovation_flag 时 eval 直接采用其标志
    （阈值评估在 servo 内，设计 §一.1 新息机制）；本函数供回退与单元测试。
    None 项不参与比较。
    """
    if nu_norm is not None and nu_norm > nu_thresh:
        return True
    if mode_entropy is not None and mode_entropy > h_thresh:
        return True
    if vis_min is not None and vis_min < vis_thresh:
        return True
    return False


def vis_entropy(vis: torch.Tensor) -> float:
    """角色可见度分布熵 H(p)，p = vis/Σvis（配对假设熵 H(w) 的可用代理）。"""
    p = vis.float().clamp_min(0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    entropy = -(p * (p + 1e-12).log()).sum(-1)
    return float(entropy.mean())


def select_roi_pair(
    mu: torch.Tensor, cov: torch.Tensor, vis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """多模式读出 [B,K,2,3] → 活动交互对（compute_roi 的 [B,2,3] 输入）。

    设计 §三.1"只为当前活动交互关系生成一个 ROI"：角色对取可见度乘积
    vis_i·vis_j 最大的两角色（遮挡/缺物角色不参与）；每角色取最强峰模式
    （mode 0，topk 序）。返回 (mu_pair [B,2,3], cov_pair [B,2,3,3])。
    """
    if mu.ndim != 4 or mu.shape[1] < 2 or mu.shape[2] != 2 or mu.shape[3] != 3:
        raise ValueError(f"mu 必须为 [B, K≥2, 2, 3]，got {tuple(mu.shape)}")
    if tuple(cov.shape) != tuple(mu.shape[:-1]) + (3, 3):
        raise ValueError(f"cov 必须为 [B, K, 2, 3, 3]，got {tuple(cov.shape)}")
    if tuple(vis.shape) != tuple(mu.shape[:2]):
        raise ValueError(f"vis 必须为 [B, K]，got {tuple(vis.shape)}")
    mu, cov, vis = mu.detach().float(), cov.detach().float(), vis.detach().float()
    batch, n_roles = mu.shape[0], mu.shape[1]
    pairs = [(i, j) for i in range(n_roles) for j in range(i + 1, n_roles)]
    best: list[tuple[int, int]] = []
    for b in range(batch):
        scores = [float(vis[b, i] * vis[b, j]) for i, j in pairs]
        best.append(pairs[int(np.argmax(scores))])  # argmax 取首个（确定性）
    mu_pair = torch.stack(
        [torch.stack([mu[b, i, 0], mu[b, j, 0]]) for b, (i, j) in enumerate(best)]
    )  # [B, 2, 3]（两角色各自的最强峰模式）
    cov_pair = torch.stack(
        [torch.stack([cov[b, i, 0], cov[b, j, 0]]) for b, (i, j) in enumerate(best)]
    )  # [B, 2, 3, 3]
    return mu_pair, cov_pair


def build_multimode_stream(
    model, readout: MultiModeReadout, dense_tokens: torch.Tensor, role_queries: torch.Tensor
) -> torch.Tensor:
    """servo/fovea 路径的 31-token 视觉流（与 build_local_vision multi_mode 分支一致）。

    复刻 ``VACompoundPolicy.build_local_vision`` 的 multi_mode 尾部
    （coarse → slots_flat → relations → vis_cond），唯一差异：readout 由调用
    方提供（reader 可带 prev_mu 跟踪先验，避免双跑导致视觉流与伺服读出
    不一致）。servo 路径外不使用；与 build_local_vision 的逐位一致性由
    tests/test_eval_servo.py 断言。
    """
    target_dtype = model.vision_projection.weight.dtype
    dense = dense_tokens.to(dtype=target_dtype)
    coarse = model.coarse_pool(dense.transpose(1, 2)).transpose(1, 2)  # [B, C, D]
    slots_flat = readout.slots.reshape(dense.shape[0], -1, model.config.vision_dim)
    relations = model.relation_tokens(readout.slots[:, :, 0], readout.mu[:, :, 0])
    stream = build_va_vision_input(coarse, slots_flat, relations)
    vis_cond = model.vis_conditioner(readout.vis.to(dtype=target_dtype))
    return stream + vis_cond[:, None, :]


class ServoRuntime:
    """C²-IRF v2 Step 2 伺服修正运行时（eval 侧薄包装，设计 §二/§五 MVP）。

    控制器形态（以 ``va_compound/servo.py`` 的 ``InteractionServo`` 为准，
    Agent E 交付；接口偏差以其 docstring 为准）：
    - 真实前向接口（InteractionServo）：``controller(readout, proprio, lang_cond,
      a_prev, g_prev) -> ServoOutput``——``correction`` [B, A] 已含阶段幅度上限、
      假设混合权重与 β 信任缩放（评估侧 ``a = clip(a_base + correction, −1, 1)``）；
      ``innovation_flag`` [B] float {0,1}（|ν|/H(w)/vis 阈值，设计 §三.3）；
      ``g`` [B, G] 跨决策由本运行时维护（g_prev）；
    - 旧契约接口（仅测试假控制器）：``relation_state(mu, cov, vis) -> g_t`` +
      ``gain`` 属性（低秩 learned gain，可辨识性纪律：只称 learned gain）。

    消融（--servo-ablation，设计 §七 Step 2）：
    - zero-gain：β≡0——最终修正恒置零（感知照常、增益归零，仅名义执行）；
    - gain-shuffle：固定种子随机打乱 K 行/列（U 行=动作维、V 行=关系维，
      增益语义破坏、尺度保留）；
    - wrong-role：角色循环移位（关系状态角色错位）；
    - open-loop：跳过伺服前向与修正（纯名义执行）。
    """

    def __init__(
        self,
        controller: Any,
        ablation: str,
        *,
        innovation_fn: Callable | None = None,
        seed: int = 0,
        nu_thresh: float = 0.1,
        h_thresh: float = 0.7,
        vis_thresh: float = 0.3,
    ) -> None:
        if ablation not in ("none", "zero-gain", "gain-shuffle", "wrong-role", "open-loop"):
            raise ValueError(f"未知消融：{ablation}")
        self.controller = controller
        self.ablation = ablation
        self.innovation_fn = innovation_fn
        self.nu_thresh = nu_thresh
        self.h_thresh = h_thresh
        self.vis_thresh = vis_thresh
        # 真实 InteractionServo 无 relation_state 属性 → 走前向接口。
        self._forward_interface = not hasattr(controller, "relation_state")
        params_fn = getattr(controller, "parameters", None)
        params = list(params_fn()) if params_fn is not None else []
        self.device = params[0].device if params else torch.device("cpu")
        self.prev_mu: torch.Tensor | None = None  # reader 跟踪先验 [B,K,2,3]
        self.prev_g: torch.Tensor | None = None  # 上一关系状态 g [B,G]（ν 依赖）
        if ablation == "gain-shuffle":
            self._shuffle_gain(seed)

    def _gain(self) -> torch.Tensor:
        gain = getattr(self.controller, "gain", None)
        if gain is None:
            raise ValueError("servo 控制器未暴露 gain（旧契约 [A, D_rel]）")
        return gain() if callable(gain) else gain

    def _shuffle_gain(self, seed: int) -> None:
        """部署时固定种子随机打乱 K 的行与列（消融：增益语义破坏，尺度保留）。

        InteractionServo 的 K = κ·U·Vᵀ 由 U/V 实时计算——打乱 U 行（动作维）
        与 V 行（关系维），此后每次 gain() 都返回行列打乱的 K；旧契约控制器
        直接置换缓存的 gain 张量。
        """
        rng = np.random.default_rng(seed)
        inner = getattr(self.controller, "servo", None)
        if inner is not None and hasattr(inner, "U") and hasattr(inner, "V"):
            rows = rng.permutation(inner.U.shape[0])
            cols = rng.permutation(inner.V.shape[0])
            with torch.no_grad():
                inner.U.data.copy_(inner.U.data[rows])
                inner.V.data.copy_(inner.V.data[cols])
            return
        gain = self._gain()
        rows = rng.permutation(gain.shape[0])
        cols = rng.permutation(gain.shape[1])
        with torch.no_grad():
            gain.copy_(gain[rows][:, cols])  # 高级索引返回拷贝，原地安全

    def _maybe_role_permute(
        self, readout: MultiModeReadout
    ) -> MultiModeReadout:
        """wrong-role：角色索引循环移位 (i+1) % K（交互语义破坏）。"""
        if self.ablation != "wrong-role":
            return readout
        n_roles = readout.mu.shape[1]
        perm = [(i + 1) % n_roles for i in range(n_roles)]
        return MultiModeReadout(
            readout.slots[:, perm],
            readout.mu[:, perm],
            readout.cov[:, perm],
            readout.vis[:, perm],
            readout.weights[:, perm],
        )

    @staticmethod
    def _as_tensor(x, device: torch.device):
        """None → None；np 数组 → float32 tensor；一维 → 批维 [1, D]（评估单实例）。"""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
        x = x.to(device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    def correct(
        self,
        readout: MultiModeReadout,
        proprio: Any = None,
        lang_cond: torch.Tensor | None = None,
        a_prev: Any = None,
    ) -> tuple[np.ndarray | None, bool]:
        """单步伺服修正 → (Δa [A] 或 None, innovation_flag)。

        open-loop → (None, False)（不施加任何修正）。真实接口：前向
        InteractionServo（correction 已含 β/阶段上限/假设混合），g_prev 由
        运行时跨决策维护；旧契约：g_t = relation_state(mu, cov, vis) →
        Δa = K·(g* − g_t)（g*≡0）。zero-gain 在输出处把修正恒置零。
        """
        if self.ablation == "open-loop":
            self.prev_mu = readout.mu
            return None, False
        readout_p = self._maybe_role_permute(readout)
        if self._forward_interface:
            if proprio is None or lang_cond is None:
                raise ValueError(
                    "InteractionServo 前向需要 proprio [B,4] 与 lang_cond [B,L]"
                )
            out = self.controller(
                readout_p,
                self._as_tensor(proprio, self.device),
                lang_cond.to(device=self.device, dtype=torch.float32),
                a_prev=self._as_tensor(a_prev, self.device),
                g_prev=self.prev_g,
            )
            correction = out.correction[0].detach().cpu().numpy().astype(
                np.float32
            ).ravel()
            flag = bool(out.innovation_flag.reshape(-1)[0].item())
            self.prev_g = out.g.detach()
        else:
            g_t = self.controller.relation_state(readout_p.mu, readout_p.cov, readout_p.vis)
            if g_t.ndim != 2:
                raise ValueError(
                    f"relation_state 必须返回 [B, D_rel]，got {tuple(g_t.shape)}"
                )
            correction = -(self._gain() @ g_t[0]).detach().cpu().numpy().astype(
                np.float32
            ).ravel()
            flag = self._innovation_flag(readout_p, g_t)
            self.prev_g = g_t.detach()
        if self.ablation == "zero-gain":
            correction = np.zeros_like(correction)  # β≡0：修正恒零
        self.prev_mu = readout.mu
        return correction, bool(flag)

    def _innovation_flag(self, readout: MultiModeReadout, g_t: torch.Tensor) -> bool:
        """新息标志（旧契约路径）：innovation() 优先；否则阈值回退。

        真实 InteractionServo 的 innovation_flag 已含 |ν|/H(w)/vis 阈值
        （设计 §三.3），不走此路径。
        """
        if self.innovation_fn is not None:
            _nu, flag = self.innovation_fn(
                readout.mu, readout.cov, readout.vis, self.prev_mu
            )
            if isinstance(flag, torch.Tensor):
                flag = flag.item()
            return bool(flag)
        nu_norm = float((g_t - self.prev_g).norm()) if self.prev_g is not None else 0.0
        return fovea_refresh_due(
            nu_norm=nu_norm,
            mode_entropy=vis_entropy(readout.vis),
            vis_min=float(readout.vis.min()),
            nu_thresh=self.nu_thresh,
            h_thresh=self.h_thresh,
            vis_thresh=self.vis_thresh,
        )


def _servo_vision(model, tokens: torch.Tensor, language_cache, coords_arr, prev_mu=None):
    """servo 路径视觉：一次 reader 调用产出 (MultiModeReadout, 31-token 流)。

    reader 带 prev_mu 跟踪先验（设计 §二.3 b_track）；视觉流与伺服读出
    共用同一次读出（build_multimode_stream），保证闭环自洽。
    """
    if not (
        getattr(model.config, "local_slots", False)
        and getattr(model.config, "multi_mode", False)
    ):
        raise ValueError("servo 需要 local_slots + multi_mode 读出")
    role_queries = getattr(language_cache, "role_queries", None)
    if role_queries is None:
        raise ValueError("servo 需要带 role_queries 的语言缓存（local_slots checkpoint）")
    coords = torch.from_numpy(coords_arr).to(device=tokens.device, dtype=tokens.dtype)
    dense = tokens.to(dtype=model.vision_projection.weight.dtype)
    readout = model.slot_reader(dense, role_queries, coords, prev_mu=prev_mu)
    stream = build_multimode_stream(model, readout, dense, role_queries)
    return readout, stream


def _foveal_tokens(
    frames: list[np.ndarray],
    roi: torch.Tensor,
    device: torch.device,
    vision_backbone=None,
    fovea_encoder=None,
    full_encoder: bool = True,
) -> torch.Tensor:
    """foveal 反馈步：ROI crop（渲染分辨率）→ 统一 resize 384 → 编码。

    - 同一 4 帧窗口共用同一仿射 crop（apply_unified_crop，防假运动，§三.2）；
    - ROI 在渲染像素空间（compute_roi(image_size=渲染高)），放大倍数
      = 384/roi_size（96px crop → 4px/patch ≈ 亚厘米，§三.2）；
    - 审查 P0-2：单决策窗口 [W,R,R,3] → [B=1,T=1,W,R,R,3] 六维，
      取 crops[0,0] 还原四帧；
    - 审查 P0-4：默认 ``full_encoder=True`` 走完整 V-JEPA（H11，pooling="dense"）
      ——与 reader/servo 训练的特征层一致（前缀 blocks[:2] 特征未在训练中
      出现过，直接喂 H11 reader 属域错配）；FoveaPrefixEncoder 保留为显式
      ``full_encoder=False``（待 foveal adapter 训练接线后启用）。
    """
    window = np.stack(frames)  # [W, R, R, 3] uint8
    render_size = window.shape[1]
    crops = apply_unified_crop(
        window[None, None], roi, image_size=render_size
    )  # [1, 1, W, R, R, 3]
    tensor = (
        torch.from_numpy(np.ascontiguousarray(crops[0, 0]))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
    )  # [W, 3, R, R]
    if tensor.shape[-1] != 384:
        tensor = F.interpolate(
            tensor, size=(384, 384), mode="bicubic",
            align_corners=False, antialias=True,  # 与 preprocess 管线一致
        )
    tensor = (tensor - IMAGE_MEAN) / IMAGE_STD
    inp = tensor.to(device)[None]  # [1, W, 3, 384, 384]
    if full_encoder:
        if vision_backbone is None:
            raise ValueError("full_encoder=True 需要 vision_backbone")
        return vision_backbone(inp, pooling="dense")  # [1, 1152, D]（H11，与训练一致）
    if fovea_encoder is None:
        raise ValueError("full_encoder=False 需要 fovea_encoder")
    return fovea_encoder(inp)  # [1, 1152, D]（冻结前缀 blocks[:2]）


def _load_servo_controller(model, ckpt: dict, device, args) -> ServoRuntime | None:
    """按契约加载 servo 运行时（无权重且未请求 → None；请求但不可用 → 报错）。

    权重来源（Agent E 训练侧写入，以 servo.py docstring / train.py 构造为准）：
    - ``ckpt["servo"]``：``InteractionServo`` 独立 state_dict（构造参数对齐
      train.py Step 2：vision_dim/lang_dim/action_dim 取模型 config，
      rank/dls/dls_lambda 存 training_contract 的 servo_rank/servo_dls/
      servo_lambda）；
    - ``model.servo``：servo 已并入 VA 政策（随 ``ckpt["model"]`` 加载）。
    """
    requested = args.servo_ablation != "none" or args.fovea
    controller = getattr(model, "servo", None)
    servo_sd = ckpt.get("servo")
    if controller is None and servo_sd is not None:
        try:
            from va_compound.servo import InteractionServo
        except ImportError as exc:  # Agent E 尚未交付
            raise ValueError(
                "va_compound/servo.py 尚未交付（Wave 2 Agent E）；"
                "--servo-ablation/--fovea 需先集成 servo 模块"
            ) from exc
        contract = ckpt.get("training_contract", {}) or {}
        controller = InteractionServo(
            vision_dim=model.config.vision_dim,
            lang_dim=model.config.hidden_dim,
            action_dim=model.config.action_dim,
            rank=int(contract.get("servo_rank", 2)),
            dls=bool(contract.get("servo_dls", False)),
            dls_lambda=float(contract.get("servo_lambda", 1e-2)),
        )
        controller.load_state_dict(servo_sd)
        controller.to(device).eval()
    if controller is None:
        if requested:
            raise ValueError(
                "--servo-ablation/--fovea 需要含 servo 权重的 checkpoint"
                "（ckpt['servo'] 或 model.servo）"
            )
        return None
    innovation_fn = getattr(controller, "innovation", None)
    return ServoRuntime(
        controller,
        args.servo_ablation,
        innovation_fn=innovation_fn,
        nu_thresh=args.fovea_nu_thresh,
        h_thresh=args.fovea_h_thresh,
        vis_thresh=args.fovea_vis_thresh,
    )


def run_c2_recovery_eval(
    args,
    device,
    model,
    vision_backbone,
    features,
    sq01,
    scale_s,
    aq01,
    aq99,
    vision_pooling="flat",
) -> None:
    """C² 恢复评估（Codex go/no-go ③）：从 v6b held-out 扰动分支初始状态出发
    闭环，测"拉回"成功率。分支状态由 prepare_mw_recovery.py 的 snapshot
    完整恢复（qpos/qvel/mocap/act/time/_prev_obs/target）。"""
    import json

    import metaworld

    if not model.config.c2_controller:
        raise ValueError("--c2-recovery-eval requires a c2 checkpoint")
    rec = torch.load(args.c2_recovery_eval, map_location="cpu", weights_only=True)
    starts = [s for s in rec["recovery_start"] if s["split"] == args.c2_recovery_split]
    if not starts:
        raise ValueError(f"no recovery branches with split={args.c2_recovery_split}")
    tasks = features["metadata"]["tasks"]
    if "Press a button" not in tasks:
        raise ValueError("recovery eval requires the button-press task in features")
    task_index = tasks.index("Press a button")
    row = int((features["instruction_id"] == task_index).nonzero()[0][0])
    hidden = features["language_hidden"][row : row + 1].to(device)
    language_mask = features["language_mask"][row : row + 1].to(device)
    language_cache = model.build_language_cache(hidden, language_mask)

    config_path = METAWORLD_TASK_CONFIG
    mw_config = json.load(open(config_path))
    if "button-press-v3" not in mw_config["TASK_DESCRIPTIONS"]:
        raise ValueError("button-press-v3 missing from metaworld_config.json")
    mt1 = metaworld.MT1("button-press-v3", seed=42)
    env = mt1.train_classes["button-press-v3"](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env._freeze_rand_vec = False

    from prepare_mw_recovery import restore_env

    wins = 0
    trials = min(len(starts), args.trials_per_task)
    for branch in starts[:trials]:
        env.reset(seed=int(branch["reset_seed"]))
        restore_env(env, branch["snapshot"])
        prev_action = branch["prev_action"]
        last_norm = (
            prev_action.numpy()
            if isinstance(prev_action, torch.Tensor)
            else np.asarray(prev_action, dtype=np.float32)
        )
        memory = None
        success = False
        plan_step = None
        c2_token = 0
        c2_params = None
        frame_buffer = []
        obs = env._get_obs()
        for step in range(args.horizon):
            img = env.render()
            frame_buffer.append(img)
            if step == 0:
                while len(frame_buffer) < (VISION_WINDOW - 1) * LEGACY_TRAINING_CONTROL_STRIDE + 1:
                    frame_buffer.insert(0, img)
            if len(frame_buffer) > (VISION_WINDOW - 1) * LEGACY_TRAINING_CONTROL_STRIDE + 1:
                frame_buffer.pop(0)
            indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
            frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
            clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
            plan_due, feedback_due, _ = c2_schedule(
                step, plan_step, args.plan_stride, args.feedback_stride, ACTION_HORIZON
            )
            if plan_due:
                state = state_take_normalize(obs, args.state_take, sq01, scale_s)
                proprio = torch.tensor(state, device=device)[None, None]
                previous = torch.tensor(
                    last_norm, dtype=torch.float32, device=device
                )[None, None]
                with torch.inference_mode():
                    tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                    c_current = model.control_projector(tokens)
                    cond, memory = model.encode_condition(
                        tokens,
                        proprio[0],
                        previous[0],
                        language_cache=language_cache,
                        visual_memory=memory,
                        return_visual_memory=True,
                    )
                    c2_params = model.controller_params(cond, c_current)
                    if args.c2_zero_gain:
                        c2_params = ControllerParams(
                            c2_params.nominal,
                            c2_params.reference,
                            torch.zeros_like(c2_params.gain),
                        )
                plan_step = step
                c2_token = 0
            if feedback_due and c2_token < ACTION_HORIZON and c2_params is not None:
                with torch.inference_mode():
                    if step != plan_step:
                        tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                    c_current = model.control_projector(tokens)
                    if args.c2_oracle_ref:
                        norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                    else:
                        error = c_current[0] - c2_params.reference[0, c2_token]
                        if (
                            args.c2_error_threshold > 0.0
                            and float(error.norm()) < args.c2_error_threshold
                        ):
                            norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                        else:
                            norm_action = (
                                c2_params.nominal[0, c2_token]
                                - c2_params.gain[0, c2_token] @ error
                            ).cpu().numpy()
                norm_action = np.clip(norm_action, -1.0, 1.0)
                c2_token += 1
            else:
                norm_action = last_norm
            action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
            obs, reward, terminated, truncated, info = env.step(action)
            last_norm = norm_action
            if info.get("success"):
                success = True
                break
            if terminated or truncated:
                break
        wins += int(success)
        print(
            f"  recovery branch seed={branch['seed']} kind={branch['kind']}: "
            f"{'SUCCESS' if success else 'FAIL'}"
        )
    env.close()
    print(
        f"\nC2 RECOVERY SUCCESS: {wins}/{trials} = {wins / trials:.1%} "
        f"(split={args.c2_recovery_split}, oracle_ref={args.c2_oracle_ref}, "
        f"zero_gain={args.c2_zero_gain})"
    )


def restore_text_backbone(
    ckpt: dict,
    device: torch.device,
    language_dtype: str = "float16",
    language_max_length: int = 64,
) -> QwenTextBackbone | QwenSemanticBackbone:
    """P0-3：按 training_contract 恢复文本分支（普通 / semantic adapter）。

    - 普通 ckpt：QwenTextBackbone + contract.lora_rank 建 LoRA + 加载
      qwen_state_dict / lora（旧行为不变）；
    - semantic ckpt（contract.semantic_adapter=True）：QwenSemanticBackbone
      （按 semantic_lora_rank / semantic_top_layers / semantic_lora_suffixes
      构造，构造器内部建 LoRA）+ 加载 qwen_state_dict / lora /
      semantic_gate。旧实现用 contract.lora_rank（semantic 模式下恒 0）建
      LoRA 且不构造 wrapper/门控，semantic checkpoint 完全无法恢复。
    返回加载完毕的 backbone（eval 模式）。
    """
    contract = ckpt.get("training_contract", {}) or {}
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device,
        dtype=language_dtype,
        local_files_only=True,
        max_length=int(contract.get("language_max_length", language_max_length)),
    )
    if contract.get("semantic_adapter"):
        text_backbone = QwenSemanticBackbone(
            text_backbone,
            lora_rank=int(contract.get("semantic_lora_rank", 8)),
            lora_alpha=float(contract.get("semantic_lora_alpha", 32.0)),
            top_layers=int(contract.get("semantic_top_layers", 4)),
            lora_suffixes=tuple(
                s.strip()
                for s in str(
                    contract.get(
                        "semantic_lora_suffixes",
                        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    )
                ).split(",")
                if s.strip()
            ),
        )
    elif ckpt.get("lora"):
        # 旧路径（--lora-rank > 0）：按 contract.lora_rank 建 LoRA 再复制权重
        # （语义路径的 LoRA 由 QwenSemanticBackbone 构造器按 semantic_lora_rank
        # 建立，P0-3）。
        from va_compound.backbones import apply_lora

        rank = int(contract.get("lora_rank", 32))
        apply_lora(text_backbone.text_model, rank=rank)
    qwen_state = {
        k.removeprefix("text_backbone.").removeprefix("text_model."): v
        for k, v in (ckpt.get("qwen_state_dict") or {}).items()
    }
    if qwen_state:
        missing, unexpected = text_backbone.text_model.load_state_dict(
            qwen_state, strict=False
        )
        print(f"eval: qwen loaded missing={len(missing)} unexpected={len(unexpected)}")
    if ckpt.get("lora"):
        own = dict(text_backbone.text_model.named_parameters())
        for name, value in ckpt["lora"].items():
            clean = name.removeprefix("text_backbone.").removeprefix("text_model.")
            if clean in own:
                own[clean].data.copy_(value)
    gate = getattr(text_backbone, "gate", None)
    if gate is not None and ckpt.get("semantic_gate"):
        gate.load_state_dict(ckpt["semantic_gate"])
    text_backbone.text_model.eval()
    return text_backbone


def build_plan_language_cache(
    model,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    scene_summary: torch.Tensor,
    *,
    instruction: str | None = None,
    text_backbone=None,
    scene_teacher=None,
    compiler=None,
    scene_tokens: torch.Tensor | None = None,
    semantic_history: torch.Tensor | None = None,
    scene_delta: torch.Tensor | None = None,
):
    """Build the VA language cache with scene-conditioned plan tokens appended.

    ``hidden``/``mask`` are the single-task language slice [1, L, D]; the
    scene summary is the global mean of the current vision window tokens.
    With ``plan_resampler`` the policy's PlanResampler produces the plan
    tokens; with ``scene_teacher`` the frozen Qwen readout path is used;
    with a ``SemanticCompiler``（P0-3）the semantic readout tokens are
    compiled from the window ``scene_tokens``（semantic_history / scene_delta
    首决策为零向量，与训练 rollout t=0 一致）。
    """
    if model.config.plan_resampler:
        return model.build_plan_cache(scene_summary, hidden, mask)
    if compiler is not None:
        if (
            instruction is None
            or text_backbone is None
            or scene_tokens is None
            or semantic_history is None
            or scene_delta is None
        ):
            raise ValueError(
                "compile cache build requires the Qwen text backbone + "
                "scene tokens/history/delta"
            )
        semantic, _ = compiler(
            text_backbone,
            [instruction],
            scene_tokens,
            semantic_history,
            scene_delta,
        )
        semantic = semantic.to(dtype=hidden.dtype)
        extended = torch.cat((hidden, semantic), dim=1)
        extended_mask = torch.cat(
            (
                mask,
                torch.ones(
                    semantic.shape[:2], dtype=torch.bool, device=semantic.device
                ),
            ),
            dim=1,
        )
        return model.build_language_cache(extended, extended_mask)
    if model.config.scene_teacher:
        if instruction is None or text_backbone is None or scene_teacher is None:
            raise ValueError("scene-teacher cache build requires the Qwen text backbone")
        plan, _ = text_backbone.encode_with_scene(
            [instruction],
            scene_summary,
            scene_teacher.scene_projector,
            scene_teacher.readout_tokens,
        )
        plan = plan.to(dtype=hidden.dtype)
        extended = torch.cat((hidden, plan), dim=1)
        extended_mask = torch.cat(
            (mask, torch.ones(plan.shape[:2], dtype=torch.bool, device=plan.device)),
            dim=1,
        )
        return model.build_language_cache(extended, extended_mask)
    return model.build_language_cache(hidden, mask)


def main() -> None:
    args = parse_args()
    if args.flow_samples < 1:
        raise ValueError("--flow-samples must be >= 1")
    validate_dagger_args(args)
    torch.manual_seed(0)  # 固定 flow 采样噪声（口径要求：重跑可复现，2026-08-05 审查补充）
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt.pop("optimizer_state", None)
    config = VACompoundConfig(**ckpt["config"])
    args.execution_horizon = resolve_execution_horizon(args, config)
    checkpoint_deployment_horizon = int(
        getattr(config, "deployment_execution_horizon", 0)
        or getattr(config, "planning_stride", LEGACY_EXECUTION_HORIZON)
    )
    if args.peer_world_off and config.va_world_mode != "peer_sync_h6":
        raise ValueError("--peer-world-off requires a peer_sync_h6 checkpoint")
    if args.peer_eval_trace:
        _validate_peer_eval_trace(
            output_json=args.output_json,
            va_world_mode=config.va_world_mode,
            peer_world_off=args.peer_world_off,
        )
    # Keep the old field available to existing validation/reporting callers.
    args.execute_steps = args.execution_horizon
    policy_contract = ckpt.get("training_contract", {}) or {}
    if (
        getattr(config, "va_world_mode", "legacy") == "peer_sync_h6"
        and int(getattr(config, "planning_stride", LEGACY_EXECUTION_HORIZON)) == 2
    ):
        required_p2_contract = {
            "peer_training_mode": "joint_dual_stream",
            "peer_world_topology": "world_minus_one_same_endpoint_fixed_current_anchor_v2",
            "peer_gradient_boundary": "world_map_stopgrad_policy_projection_trainable_v1",
            "peer_data_isolation": "separate_va_world_episode_datasets_per_step_v1",
            "peer_dual_stream_optimizer": (
                "va_backward_then_world_backward_one_optimizer_step_v1"
            ),
        }
        if checkpoint_deployment_horizon == 15:
            required_p2_contract["peer_flow_topology"] = (
                "h6_prefix_h9_tail_one_way_detached_flow_v1"
            )
        p2_contract_mismatch = {
            key: (policy_contract.get(key), value)
            for key, value in required_p2_contract.items()
            if policy_contract.get(key) != value
        }
        for identity_key in ("peer_va_data_identity", "peer_world_data_identity"):
            identity = policy_contract.get(identity_key)
            if not isinstance(identity, dict) or not identity.get("full_file_sha256"):
                p2_contract_mismatch[identity_key] = (identity, "strong identity")
        exact_arguments = (ckpt.get("exact_run_contract") or {}).get(
            "arguments"
        ) or {}
        expected_p2_arguments = {
            "control_stride": 2,
            "planning_stride": 2,
            "wmrm_cycle_steps": int(config.wmrm_cycle_steps),
            "deployment_execution_horizon": checkpoint_deployment_horizon,
            "flow_prefix_steps": 2,
        }
        for key, value in expected_p2_arguments.items():
            if exact_arguments.get(key) != value:
                p2_contract_mismatch[f"exact_run_contract.arguments.{key}"] = (
                    exact_arguments.get(key),
                    value,
                )
        if p2_contract_mismatch:
            raise ValueError(
                f"peer P2 checkpoint training contract mismatch: {p2_contract_mismatch}"
            )
    action_vision_enabled = config.action_vision_backbone != "none"
    if action_vision_enabled:
        if args.action_vision_checkpoint is None:
            raise ValueError(
                "checkpoint requires an action vision tower; provide "
                "--action-vision-checkpoint"
            )
        # Keep the HF cache symlink suffix so timm selects safetensors loading;
        # ``resolve()`` would produce an extensionless blob path.
        action_checkpoint = args.action_vision_checkpoint.expanduser().absolute()
        if not action_checkpoint.is_file():
            raise FileNotFoundError(
                f"action vision checkpoint is missing: {action_checkpoint}"
            )
        expected_action_contract = {
            "action_vision_backbone": config.action_vision_backbone,
            "action_vision_model_id": config.action_vision_model_id,
            "action_vision_image_size": config.action_vision_image_size,
            "action_vision_feature_dim": config.action_vision_dim,
            "action_vision_output_layers": list(config.action_vision_layers),
            "action_vision_frame_indices": [1, 3],
        }
        mismatches = {
            key: (policy_contract.get(key), expected)
            for key, expected in expected_action_contract.items()
            if policy_contract.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"action vision checkpoint contract mismatch: {mismatches}"
            )
        expected_sha = policy_contract.get("action_vision_checkpoint_sha256")
        actual_sha = _sha256_file(action_checkpoint)
        if not expected_sha or actual_sha != expected_sha:
            raise ValueError(
                "action vision external-weight SHA mismatch: "
                f"policy={expected_sha!r}, runtime={actual_sha!r}"
            )
    elif args.action_vision_checkpoint is not None:
        raise ValueError(
            "--action-vision-checkpoint was provided but this policy has no "
            "trained action vision branch"
        )
    # DINO-main replacement contract（2026-08-14 用户决策）：冻结 DINOv2 替换
    # V-JEPA 主视觉。V-JEPA/dense/metric 路径保留在代码中（flag 关闭即禁用）。
    main_vision_dino = getattr(config, "main_vision_backbone", "vjepa") != "vjepa"
    if main_vision_dino:
        if args.main_vision_checkpoint is None:
            raise ValueError(
                "checkpoint uses a DINO main vision backbone; provide "
                "--main-vision-checkpoint"
            )
        main_checkpoint = args.main_vision_checkpoint.expanduser().absolute()
        if not main_checkpoint.is_file():
            raise FileNotFoundError(
                f"main vision checkpoint is missing: {main_checkpoint}"
            )
        expected_main_contract = {
            "main_vision_model_id": getattr(config, "main_vision_model_id", None),
            "main_vision_image_size": getattr(
                config, "main_vision_image_size", None
            ),
            "main_vision_feature_dim": getattr(config, "main_vision_dim", None),
            "main_vision_grid": getattr(config, "main_vision_grid", None),
            "main_vision_frames": getattr(config, "main_vision_frames", None),
            "main_vision_tokens": getattr(config, "main_vision_tokens", None),
        }
        main_mismatches = {
            key: (policy_contract.get(key), expected)
            for key, expected in expected_main_contract.items()
            if policy_contract.get(key) != expected
        }
        if main_mismatches:
            raise ValueError(
                f"main vision checkpoint contract mismatch: {main_mismatches}"
            )
        expected_main_sha = policy_contract.get("main_vision_checkpoint_sha256")
        actual_main_sha = _sha256_file(main_checkpoint)
        if not expected_main_sha or actual_main_sha != expected_main_sha:
            raise ValueError(
                "main vision external-weight SHA mismatch: "
                f"policy={expected_main_sha!r}, runtime={actual_main_sha!r}"
            )
        if (
            args.dense_readout_mtvj
            or args.metric_visual_checkpoint is not None
            or args.mtvj_dense_only_ablation
            or args.fovea
        ):
            raise ValueError(
                "DINO-main replacement: V-JEPA dense/metric/fovea eval flags "
                "must stay off (--dense-readout-mtvj/--metric-visual-checkpoint/"
                "--mtvj-dense-only-ablation/--fovea)"
            )
    elif args.main_vision_checkpoint is not None:
        raise ValueError(
            "--main-vision-checkpoint was provided but this policy uses the "
            "V-JEPA main vision backbone"
        )
    checkpoint_uses_mtvj = want_vjepa_dense_backbone(config, args)
    # DINO-metric checkpoint 的 dense 层由 DINO block11/block23 证据驱动，
    # 不建 V-JEPA 骨干/路径（--dense-readout-mtvj CLI 保持关闭）。
    dino_roi_expected = policy_contract.get("dino_roi_enabled") is True
    if args.dino_roi_checkpoint is None:
        if args.dino_roi_alpha is not None:
            raise ValueError("--dino-roi-alpha requires --dino-roi-checkpoint")
        if dino_roi_expected:
            raise ValueError(
                "checkpoint was trained with task35 DINO ROI; pass the exact "
                "--dino-roi-checkpoint instead of silently disabling it"
            )
    else:
        if not getattr(config, "dino_dense_metric", False):
            raise ValueError(
                "--dino-roi-checkpoint 只对 dino_dense_metric 主 checkpoint 有效"
            )
        if (
            args.dino_roi_alpha is None
            or not np.isfinite(args.dino_roi_alpha)
            or not 0.0 <= args.dino_roi_alpha <= 1.0
        ):
            raise ValueError(
                "--dino-roi-checkpoint requires finite --dino-roi-alpha in [0,1]"
            )
        if dino_roi_expected:
            saved_alpha = policy_contract.get("dino_roi_alpha")
            if saved_alpha is None or float(saved_alpha) != float(args.dino_roi_alpha):
                raise ValueError(
                    "--dino-roi-alpha must match policy training: "
                    f"policy={saved_alpha!r}, runtime={args.dino_roi_alpha!r}"
                )
    if checkpoint_uses_mtvj and not args.dense_readout_mtvj:
        args.dense_readout_mtvj = True
        print(
            "eval: checkpoint config.dense_readout_mtvj=True；自动启用同构 MT-VJ "
            "H5/H11 dense 路径",
            flush=True,
        )
    metric_expected = bool(policy_contract.get("metric_tokens_enabled", False))
    roi_expected = policy_contract.get("mtvj_roi_enabled") is True
    if args.mtvj_roi_checkpoint is None:
        if args.mtvj_roi_alpha is not None:
            raise ValueError("--mtvj-roi-alpha requires --mtvj-roi-checkpoint")
        if roi_expected:
            raise ValueError(
                "checkpoint requires --mtvj-roi-checkpoint; refusing to silently "
                "disable the trained ROI runtime"
            )
    else:
        if (
            args.mtvj_roi_alpha is None
            or not np.isfinite(args.mtvj_roi_alpha)
            or not 0.0 <= args.mtvj_roi_alpha <= 1.0
        ):
            raise ValueError(
                "--mtvj-roi-checkpoint requires finite --mtvj-roi-alpha in [0,1]"
            )
        if args.metric_visual_checkpoint is None:
            raise ValueError(
                "--mtvj-roi-checkpoint requires --metric-visual-checkpoint"
            )
        if roi_expected:
            saved_alpha = policy_contract.get("mtvj_roi_alpha")
            if saved_alpha is None or float(saved_alpha) != float(args.mtvj_roi_alpha):
                raise ValueError(
                    "--mtvj-roi-alpha must exactly match the policy checkpoint: "
                    f"policy={saved_alpha!r}, runtime={args.mtvj_roi_alpha!r}"
                )
    if args.mtvj_dense_only_ablation and not args.dense_readout_mtvj:
        raise ValueError("--mtvj-dense-only-ablation requires an MT-VJ checkpoint/path")
    if (
        metric_expected
        and args.metric_visual_checkpoint is None
        and not args.mtvj_dense_only_ablation
        and not getattr(config, "dino_dense_metric", False)
    ):
        raise ValueError(
            "checkpoint 训练时启用了 MT-VJ metric tokens；评测必须提供 "
            "--metric-visual-checkpoint，或显式指定 --mtvj-dense-only-ablation"
        )
    # 2026-08-09：ACTION_HORIZON 从 checkpoint config 读（E7 H=48），不再硬编码 8。
    global ACTION_HORIZON
    ACTION_HORIZON = int(getattr(config, "action_horizon", 8))
    if args.task35_precision_contract and args.task35_causal_ablation != "none":
        raise ValueError(
            "--task35-precision-contract is acceptance-only; causal ablations must "
            "run without it and be reported as diagnostics"
        )
    if args.task35_causal_ablation != "none":
        ablation_requirements = {
            "task35 only": args.task_ids == "35",
            "50 trials": args.trials_per_task == 50,
            "execute_steps 6": args.execute_steps == 6,
            "horizon 500": args.horizon == 500,
            "stage telemetry": args.debug_stage_metrics,
            "WAM4VA state exchange off": not getattr(config, "wmrm", False),
            "FM checkpoint": policy_contract.get("action_decoder")
            == "conditional_flow_matching",
            "FM decoder not overridden": args.direct_head in {"auto", "off"},
            "DINO metric": getattr(config, "dino_dense_metric", False),
            "H6": ACTION_HORIZON == 6,
        }
        missing_ablation = [
            name for name, enabled in ablation_requirements.items() if not enabled
        ]
        if missing_ablation:
            raise ValueError(
                "task35 causal ablation missing fixed protocol: "
                + ", ".join(missing_ablation)
            )
        print(
            f"TASK35 CAUSAL ABLATION: {args.task35_causal_ablation} "
            "(diagnostic only; not acceptance evidence)",
            flush=True,
        )
    if args.task35_precision_contract:
        expected_data_sha = policy_contract.get("task35_data_sha256")
        actual_data_sha = _sha256_file(args.features.expanduser().absolute())
        expected_feature_sha = policy_contract.get("task35_dino_feature_sha256") or {}
        actual_feature_sha = {}
        if args.dino_feature_cache is not None:
            cache_root = args.dino_feature_cache.expanduser().absolute()
            actual_feature_sha = {
                name: _sha256_file(cache_root / name)
                for name in ("block11.npy", "block23.npy")
            }
        requirements = {
            "checkpoint precision contract": policy_contract.get(
                "task35_precision_contract"
            )
            is True,
            "exact matched H6 features": bool(expected_data_sha)
            and actual_data_sha == expected_data_sha,
            "raw-frame identity recorded": bool(
                policy_contract.get("task35_raw_frames_sha256")
            ),
            "DINO feature cache provided": args.dino_feature_cache is not None,
            "DINO feature identities match": set(expected_feature_sha)
            == {"block11.npy", "block23.npy"}
            and actual_feature_sha == expected_feature_sha,
            "FM decoder": policy_contract.get("action_decoder")
            == "conditional_flow_matching",
            "fused VA attention": getattr(
                config, "va_attention_backend", "manual"
            )
            == "auto",
            "H6": ACTION_HORIZON == 6,
            "DINO metric": getattr(config, "dino_dense_metric", False),
            "grid16": getattr(config, "main_vision_grid", None) == 16,
            "four frames": getattr(config, "main_vision_frames", None) == 4,
            "temporal": getattr(config, "main_vision_temporal", False),
            "geometry": getattr(config, "metric_geometry_inject", False),
            "visual auxiliary grounding": int(
                policy_contract.get("mtvj_visual_aux_every", 0)
            )
            > 0,
            "visual auxiliary raw pixels": policy_contract.get(
                "mtvj_visual_aux_pixel_contract"
            )
            == "true_simulator_render_480_to_dino224_v1",
            "ROI": dino_roi_expected and args.dino_roi_checkpoint is not None,
            "ROI alpha 1": args.dino_roi_alpha == 1.0,
            "execute_steps 6": args.execute_steps == 6,
            "horizon 500": args.horizon == 500,
            "50 trials": args.trials_per_task == 50,
            "task35 only": args.task_ids == "35",
            "stage telemetry": args.debug_stage_metrics,
            "WAM4VA state exchange off": not getattr(config, "wmrm", False),
            "FM decoder not overridden": args.direct_head in {"auto", "off"},
        }
        missing = [name for name, enabled in requirements.items() if not enabled]
        if missing:
            raise ValueError(
                "--task35-precision-contract missing required settings: "
                + ", ".join(missing)
            )
    # Codex P0-4（2026-08-10）：flow Euler 步数从 training_contract 读
    # （train.py:2561 保存位置），不再硬编码 32 或误读顶层键。
    flow_steps = int(
        (ckpt.get("training_contract", {}) or {}).get("flow_steps")
        or ckpt.get("flow_steps")
        or 8
    )
    print(f"eval: action_horizon={ACTION_HORIZON} (from checkpoint config), "
          f"flow_steps={flow_steps} (from checkpoint contract)")
    # spatial-pooling ckpt 评估：vision_pooling 存在 training_contract 而非 config。
    vision_pooling = str(
        (ckpt.get("training_contract", {}) or {}).get("vision_pooling", "flat")
    )
    if args.vision_pooling is not None:
        vision_pooling = args.vision_pooling
    if config.local_slots:
        if getattr(config, "dense_readout", False):
            # Step 0 dense readout：1152 patch 不池化，池化模式必须为 dense。
            if vision_pooling != "dense":
                print(
                    f"eval: config.dense_readout=True 但 vision_pooling={vision_pooling}；"
                    "强制 dense（1152 patch 不池化，与训练一致）"
                )
                vision_pooling = "dense"
        elif vision_pooling != "spatiotemporal":
            # Stage A/B：local_slots 训练（ST288/live）必为 spatiotemporal 288 token，
            # 旧 contract 可能漏记；强制对齐避免 flat 64-token 闭环失真。
            print(
                f"eval: config.local_slots=True 但 vision_pooling={vision_pooling}；"
                "强制 spatiotemporal（288 token，与训练一致）"
            )
            vision_pooling = "spatiotemporal"
    if args.direct_head != "auto":
        if (
            args.direct_head == "on"
            and (
                args.task35_precision_contract
                or args.task35_causal_ablation != "none"
            )
        ):
            raise ValueError(
                "task35 FM evaluation refuses --direct-head on; "
                "acceptance and causal diagnostics must stay on the FM decoder"
            )
        config = dataclasses.replace(config, direct_head=args.direct_head == "on")
    # MT-VJ（契约 §5/§7）：--dense-readout-mtvj 强制打开 model 的 dense 层
    # （与 train.py _mtvj_config_kwargs 同构）。ckpt 无 dense 权重时 W_o 零初始化
    # → 初始输出逐位等价（下方非严格加载 + 警告）。
    dense_forced = False
    if args.dense_readout_mtvj:
        if not getattr(config, "dense_readout_mtvj", False):
            dense_forced = True
            config = dataclasses.replace(config, dense_readout_mtvj=True)
            print(
                "eval: --dense-readout-mtvj 强制 config.dense_readout_mtvj=True"
                "（零初始化 dense 层初始等价）"
            )
    if args.metric_visual_checkpoint is not None and not args.dense_readout_mtvj:
        raise ValueError("--metric-visual-checkpoint 需要 --dense-readout-mtvj")
    # P0-3：semantic-compiler ckpt 同样按需逐决策重建语言缓存（场景条件化）。
    has_compile = ckpt.get("semantic_compiler") is not None
    has_plan = config.plan_resampler or config.scene_teacher or has_compile
    if args.plan_refresh < 0:
        raise ValueError("--plan-refresh must be >= 0")
    if args.plan_refresh > 0 and not has_plan:
        raise ValueError(
            "--plan-refresh requires a checkpoint with plan_resampler, "
            "scene_teacher, or a semantic compiler"
        )
    model = VACompoundPolicy(config).eval().to(device)
    model.runtime_execution_horizon = args.execution_horizon
    ckpt_direct_head = bool(ckpt["config"].get("direct_head", False))

    if ckpt_direct_head == config.direct_head and not dense_forced:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected or missing:
            raise RuntimeError(
                "checkpoint/model mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
    elif dense_forced:
        # --dense-readout-mtvj 强制：ckpt 训练时未开 dense 层 → dense 权重缺失，
        # 非严格加载（W_o 零初始化 → 初始输出与无 dense 路径逐位一致）。
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"eval: --dense-readout-mtvj 强制 dense 层（ckpt config "
            f"dense_readout_mtvj={bool(ckpt['config'].get('dense_readout_mtvj', False))}）；"
            f"非严格加载 missing={len(missing)} unexpected={len(unexpected)}"
        )
    else:
        # --direct-head on/off 强制换解码器（消融）：另一头的 head 未训练，
        # 用非严格加载 + 显式警告。
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"eval: forced decoder override direct_head={config.direct_head} "
            f"(ckpt had {ckpt_direct_head}); missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    assert config.proprio_dim == 4 and config.action_dim == 4, "expect 4D MetaWorld config"
    if config.c2_controller and not config.direct_head:
        raise ValueError("c2 checkpoint requires direct_head in the config")
    if config.c2_controller and args.plan_stride is not None and args.plan_stride < 1:
        raise ValueError("--plan-stride must be positive")
    if config.c2_controller and args.feedback_stride is not None and args.feedback_stride < 1:
        raise ValueError("--feedback-stride must be positive")
    if not config.c2_controller and (args.c2_oracle_ref or args.c2_zero_gain):
        raise ValueError("--c2-oracle-ref/--c2-zero-gain require a c2 checkpoint")
    if config.c2_controller and not config.direct_head:
        raise ValueError("c2 controller requires the direct head decoder")
    validate_servo_args(args, config)
    if args.state_take != 4:
        # --state-take 8/39：零初始化扩展 proprio 投影（训练 4 维权重不变，
        # 分布外维度零贡献——协议复刻，protocol_verification_evo_fabri.md E3）。
        extend_state_projection(model, args.state_take)
        print(
            f"eval: state_projection extended to "
            f"{args.state_take + config.action_dim} inputs (--state-take {args.state_take})"
        )
    # C²-IRF v2：servo 运行时（--servo-ablation/--fovea；flow checkpoint +
    # ckpt["servo"] 权重）。放在节奏默认值之前（servo 也用 plan/feedback 节奏）。
    servo_runtime = _load_servo_controller(model, ckpt, device, args)
    dense_coords_arr = None
    if servo_runtime is not None:
        if _stage_coords is None or _dense_coords is None:
            raise RuntimeError("servo 路径需要 va_compound.live_vjepa 坐标")
        dense_coords_arr = _dense_coords() if config.dense_readout else _stage_coords()
        print(
            f"eval: servo runtime active "
            f"(ablation={args.servo_ablation}, fovea={args.fovea})"
        )
    if args.dense_readout_mtvj and (
        config.c2_controller or args.c2_recovery_eval is not None
    ):
        raise ValueError(
            "--dense-readout-mtvj（MT-VJ §7）与 c2/recovery 部署互斥"
        )
    print(
        f"eval: action decoder = "
        f"{'c2_controller (ū/c̄/K contraction)' if config.c2_controller else ('direct_head (MLP->tanh)' if config.direct_head else f'flow_matching (Euler steps={flow_steps})')}"
    )

    features = torch.load(args.features, map_location="cpu", weights_only=True)
    language_features = features
    if args.language_features is not None:
        language_features = torch.load(
            args.language_features, map_location="cpu", weights_only=True
        )
        validate_language_features(features, language_features)
        print(
            f"eval: task/language cache = {args.language_features} "
            f"({len(language_features['metadata']['tasks'])} tasks)",
            flush=True,
        )
    feature_metadata = features.get("metadata") or {}
    try:
        training_control_stride = int(feature_metadata["control_stride"])
        control_hz = int(feature_metadata["fps"])
        training_prediction_horizon = int(feature_metadata["action_horizon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "feature metadata must define integer control_stride, fps, and action_horizon"
        ) from exc
    raw_planning_stride = feature_metadata.get("planning_stride")
    try:
        training_planning_stride = (
            training_control_stride
            if raw_planning_stride is None
            else int(raw_planning_stride)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("feature metadata planning_stride must be an integer") from exc
    h50_p15 = (
        int(config.action_horizon) == 50
        and training_prediction_horizon == 15
        and int(getattr(config, "planning_stride", 0)) == 15
        and int(getattr(config, "deployment_execution_horizon", 0)) == 15
        and int(getattr(config, "wmrm_cycle_steps", 0)) == 15
    )
    if training_prediction_horizon != config.action_horizon and not h50_p15:
        raise ValueError(
            "feature/checkpoint action horizon mismatch: "
            f"{training_prediction_horizon} != {config.action_horizon}"
        )
    checkpoint_planning_stride = int(
        getattr(config, "planning_stride", LEGACY_EXECUTION_HORIZON)
    )
    if getattr(config, "va_world_mode", "legacy") == "peer_sync_h6":
        if (
            checkpoint_planning_stride != LEGACY_EXECUTION_HORIZON
            and raw_planning_stride is None
        ):
            raise ValueError(
                "P2 peer feature metadata must explicitly define planning_stride"
            )
        peer_cadence = {
            "fps": (control_hz, 80),
            "control_stride": (
                training_control_stride,
                checkpoint_planning_stride,
            ),
            "planning_stride": (
                training_planning_stride,
                checkpoint_planning_stride,
            ),
            "execution_horizon": (
                args.execution_horizon,
                (
                    args.execution_horizon
                    if args.allow_execution_horizon_ablation
                    else checkpoint_deployment_horizon
                ),
            ),
            "wmrm_cycle_steps": (
                int(getattr(config, "wmrm_cycle_steps", 0)),
                training_prediction_horizon,
            ),
            "action_horizon": (
                int(getattr(config, "action_horizon", 0)),
                int(getattr(config, "action_horizon", 0)),
            ),
        }
        cadence_mismatch = {
            key: values
            for key, values in peer_cadence.items()
            if values[0] != values[1]
        }
        if cadence_mismatch:
            raise ValueError(
                f"peer_sync_h6 planning cadence mismatch: {cadence_mismatch}"
            )
    training_planning_hz = float(control_hz) / float(checkpoint_planning_stride)
    deployment_planning_hz = float(control_hz) / float(args.execution_horizon)
    print(
        "eval: deployment cadence "
        f"fps={control_hz}, training_stride={checkpoint_planning_stride}, "
        f"training_hz={training_planning_hz:g}, "
        f"deployment_hz={deployment_planning_hz:g}, "
        f"prediction_horizon={int(config.action_horizon)}, "
        f"execution_horizon={args.execution_horizon}, "
        f"checkpoint_execution_horizon={checkpoint_deployment_horizon}, "
        f"execution_ablation={int(args.allow_execution_horizon_ablation)}"
    )
    all_tasks = list(language_features["metadata"]["tasks"])
    selected_tasks = select_eval_tasks(all_tasks, args.task_ids, args.max_tasks)
    if not selected_tasks:
        raise ValueError("no tasks selected for evaluation")
    task_indices = [index for index, _ in selected_tasks]
    tasks = [text for _, text in selected_tasks]
    descriptions_to_env = load_metaworld_description_to_env()
    validate_mt50_benchmark_protocol(args, selected_tasks, descriptions_to_env)
    if args.task35_precision_contract or args.task35_causal_ablation != "none":
        env_name = require_task35_peg_insert_side(selected_tasks, descriptions_to_env)
        print(
            f"eval: preflight task35 -> {env_name} "
            f"(id=35, text={selected_tasks[0][1]!r})",
            flush=True,
        )
    sq01 = features["normalization"]["state_q01"].numpy()
    sq99 = features["normalization"]["state_q99"].numpy()
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    # 动作反归一化参数（模型输出 norm -> 环境原始动作）
    aq01 = features["normalization"]["action_q01"].numpy()
    aq99 = features["normalization"]["action_q99"].numpy()

    if config.c2_controller or servo_runtime is not None:
        # C² / 伺服部署默认节奏：plan_stride=6（VA 生成 {ū,c̄,K} / 解码名义
        # chunk 一次），feedback_stride=1（每原始步刷新读出并消费 token，
        # Codex 修正 5）。
        args.plan_stride = (
            args.plan_stride
            if args.plan_stride is not None
            else LEGACY_EXECUTION_HORIZON
        )
        args.feedback_stride = (
            args.feedback_stride if args.feedback_stride is not None else 1
        )
        print(
            f"eval: {'c2' if config.c2_controller else 'servo'} "
            f"plan_stride={args.plan_stride} feedback_stride={args.feedback_stride} "
            f"oracle_ref={args.c2_oracle_ref} zero_gain={args.c2_zero_gain}"
        )
    vision_backbone = None
    main_vision_backbone = None
    if main_vision_dino:
        # DINO-main replacement（2026-08-14 用户决策）：冻结 DINOv2 替换 V-JEPA
        # 主视觉（V-JEPA 路径保留未删除，flag 关闭即禁用）。
        from va_compound.backbones import TimmActionVisionBackbone

        main_vision_backbone = TimmActionVisionBackbone.from_pretrained(
            device=device,
            dtype="float16",
            model_id=config.main_vision_model_id,
            image_size=config.main_vision_image_size,
            feature_dim=config.main_vision_dim,
            output_layers=(11, 23),  # 与训练侧同一 canonical mid/final 契约
            checkpoint_path=main_checkpoint,
            local_files_only=True,
        )
        if policy_contract.get("main_vision_joint_trained"):
            trained_main_state = ckpt.get("main_vision_state_dict")
            if trained_main_state is None:
                raise ValueError(
                    "checkpoint declares trained DINO-main but lacks "
                    "main_vision_state_dict"
                )
            main_vision_backbone.model.load_state_dict(
                trained_main_state, strict=True
            )
            print("eval: loaded trained DINO-main state from checkpoint")
        main_vision_backbone.freeze_all()
        print(
            f"eval: DINO-main frozen {config.main_vision_backbone} replaces "
            f"V-JEPA ({config.main_vision_image_size}px, "
            f"dim={config.main_vision_dim}, {config.main_vision_tokens} "
            f"tokens/decision, params={sum(p.numel() for p in main_vision_backbone.parameters()):,})",
            flush=True,
        )
    else:
        vision_backbone = VJEPA21Backbone.from_pretrained(
            device=device,
            dtype="float16",
            max_tokens=144 if vision_pooling == "spatiotemporal" else 64,
            local_files_only=True,
        )
        if ckpt.get("vjepa_state_dict"):
            # e2e checkpoint：V-JEPA 被微调过，必须加载训练后权重（2026-08-06 P0 #4）
            vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
            print("eval: loaded vjepa_state_dict from checkpoint")
        vision_backbone.freeze_all()
    # --fovea：foveal 前缀编码器共享同一冻结 V-JEPA 实例（无新权重——
    # 显存纪律：1152 只读、V-JEPA 冻结）。
    fovea_encoder = FoveaPrefixEncoder(vision_backbone.model) if args.fovea else None
    # MT-VJ（契约 §6/§7）：冻结 fp16 独立 V-JEPA 实例做 dense evidence——
    # 与 train.py _maybe_build_mtvj_backbone 同构（不加载 vjepa_state_dict，
    # e2e checkpoint 的 dense 证据同样来自冻结原版权重，与训练一致）。
    mtvj_backbone = None
    action_vision_backbone = None
    metric_head = None
    relation_encoder = None
    roi_head = None
    coords_mtvj = None
    coords_dino_metric = None
    if getattr(config, "dino_dense_metric", False):
        # DINO-metric（2026-08-15）：metric 栈从主 checkpoint 严格重建
        # （无外部 --metric-visual-checkpoint；V-JEPA metric 权重不兼容）。
        from va_compound.model import dense_coords

        metric_head, relation_encoder = _load_dino_metric_from_policy(
            ckpt, config, device
        )
        coords_dino_metric = dense_coords(512, device=device)
        if args.dino_roi_checkpoint is not None:
            roi_head = _load_dino_roi_head(args.dino_roi_checkpoint, device)
            expected_identity = ckpt.get("dino_roi_checkpoint_identity")
            actual_identity = getattr(roi_head, "_dino_roi_identity", None)
            if expected_identity is not None:
                mismatches = {
                    key: (expected_identity.get(key), actual_identity.get(key))
                    for key in ("sha256", "size_bytes", "contract")
                    if expected_identity.get(key) != actual_identity.get(key)
                }
                if mismatches:
                    raise ValueError(
                        f"DINO ROI checkpoint identity mismatch: {mismatches}"
                    )
    if args.dense_readout_mtvj:
        mtvj_backbone = VJEPA21Backbone.from_pretrained(
            device=device,
            dtype="float16",
            max_tokens=144,
            local_files_only=True,
        )
        mtvj_backbone.freeze_all()
        print(
            "eval: MT-VJ 冻结 V-JEPA dense evidence 骨干就绪（fp16，"
            f"params={sum(p.numel() for p in mtvj_backbone.parameters()):,}）"
        )
        if args.metric_visual_checkpoint is not None:
            if _dense_coords is None:
                raise RuntimeError(
                    "metric 路径需要 va_compound.live_vjepa._dense_coords"
                )
            metric_head, relation_encoder = _load_mtvj_metric_checkpoint(
                args.metric_visual_checkpoint,
                device,
                config,
                policy_relation_state=ckpt.get("mtvj_relation_encoder"),
                policy_metric_state=ckpt.get("mtvj_metric_head"),
                policy_metric_config=ckpt.get("mtvj_metric_head_config"),
                policy_metric_identity=ckpt.get(
                    "mtvj_metric_checkpoint_identity"
                ),
                policy_training_contract=ckpt.get("training_contract", {}) or {},
            )
            coords_mtvj = torch.from_numpy(_dense_coords())
            if args.mtvj_roi_checkpoint is not None:
                coarse_identity = getattr(
                    metric_head, "_mtvj_current_external_checkpoint_identity", None
                )
                if not isinstance(coarse_identity, dict):
                    raise ValueError("MT-VJ metric head lacks its external coarse identity")
                roi_head = load_metric_roi_checkpoint(
                    args.mtvj_roi_checkpoint,
                    device,
                    coarse_identity=coarse_identity,
                    coarse_head_state_sha256=metric_head_state_sha256(metric_head),
                    policy_state=ckpt.get("mtvj_roi_head"),
                    policy_config=ckpt.get("mtvj_roi_config"),
                    policy_identity=ckpt.get("mtvj_roi_checkpoint_identity"),
                    policy_training_contract=policy_contract,
                )
                print(
                    "eval: frozen MT-VJ ROI head loaded "
                    f"(alpha={args.mtvj_roi_alpha}, "
                    f"params={sum(p.numel() for p in roi_head.parameters()):,})",
                    flush=True,
                )
        else:
            print(
                "eval: MT-VJ dense-only benchmark（metric tokens disabled"
                f"{'；这是主模型的显式消融' if args.mtvj_dense_only_ablation else ''}）"
            )
    if action_vision_enabled:
        action_vision_backbone = TimmActionVisionBackbone.from_pretrained(
            device=device,
            dtype="float16",
            model_id=config.action_vision_model_id,
            image_size=config.action_vision_image_size,
            feature_dim=config.action_vision_dim,
            output_layers=tuple(config.action_vision_layers),
            checkpoint_path=action_checkpoint,
            local_files_only=True,
        )
        action_vision_backbone.freeze_all()
        print(
            f"eval: action vision {config.action_vision_backbone} ready "
            f"({config.action_vision_image_size}px, dim={config.action_vision_dim}, "
            f"params={sum(p.numel() for p in action_vision_backbone.parameters()):,})",
            flush=True,
        )
    if args.c2_recovery_eval is not None:
        if not config.c2_controller:
            raise ValueError("--c2-recovery-eval requires a c2 checkpoint")
        run_c2_recovery_eval(
            args,
            device,
            model,
            vision_backbone,
            features,
            sq01,
            scale_s,
            aq01,
            aq99,
            vision_pooling=vision_pooling,
        )
        return

    # P0-3：统一恢复路径（普通 LoRA / semantic adapter 都按 training_contract
    # 构造并加载 qwen_state_dict / lora / semantic_gate）。
    # task35 FM acceptance/causal can reuse the training language cache and
    # skip Qwen entirely: the 15k checkpoint has no qwen/lora/plan weights.
    compiler = None
    if has_compile:
        from va_compound.backbones import SemanticCompiler

        compiler = SemanticCompiler(
            language_dim=config.language_dim,
            vision_dim=config.vision_dim,
            history_in_dim=config.hidden_dim,
            n_readout=int(
                ckpt.get("training_contract", {}).get("compile_n_readout", 16)
            ),
        ).to(device)
        compiler.load_state_dict(ckpt["semantic_compiler"])
        compiler.eval()
        print("eval: semantic_compiler loaded from checkpoint")
    can_use_feature_language = (
        not has_plan
        and not ckpt.get("qwen_state_dict")
        and not ckpt.get("lora")
        and not config.scene_teacher
        and "language_hidden" in language_features
        and "language_mask" in language_features
        and "instruction_id" in language_features
    )
    text_backbone = None
    if can_use_feature_language:
        hidden_rows = []
        mask_rows = []
        for task_id, _ in selected_tasks:
            hid, msk = cached_task_language(
                language_features, device, task_id=int(task_id)
            )
            hidden_rows.append(hid)
            mask_rows.append(msk)
        hidden = torch.cat(hidden_rows, dim=0)
        mask = torch.cat(mask_rows, dim=0)
        print(
            f"eval: using cached language_hidden for task_ids={task_indices}; Qwen not loaded",
            flush=True,
        )
    else:
        text_backbone = restore_text_backbone(ckpt, device, language_dtype="float16")
        if isinstance(text_backbone, QwenSemanticBackbone):
            # P0-3：semantic adapter ckpt——语言 hidden 用 fused 嵌入
            # （prior + g ⊙ (adapted − prior)），不是裸冻结先验。
            with torch.no_grad():
                prior, mask = text_backbone.encode_prior(tasks)
                adapted, _ = text_backbone.encode_adapted(tasks)
                hidden = text_backbone.fused_embedding(prior, adapted)
        else:
            hidden, mask = text_backbone.encode(tasks)

    scene_teacher = None
    if config.scene_teacher:
        if ckpt.get("scene_teacher") is None:
            raise ValueError("scene-teacher checkpoint has no scene_teacher weights")
        from va_compound.backbones import SceneTeacher

        scene_teacher = SceneTeacher(
            language_dim=config.language_dim, vision_dim=config.vision_dim
        ).to(device)
        scene_teacher.load_state_dict(ckpt["scene_teacher"])
        scene_teacher.eval()
        print("eval: scene_teacher loaded from checkpoint")
    if has_plan:
        # Plan-Cache：缓存按 episode 逐任务懒构建（首帧场景 → plan tokens），
        # --plan-refresh R 控制后续重建；Qwen 仅在 scene_teacher / compiler 下常驻。
        task_caches: list | None = [None] * len(tasks)
        if text_backbone is not None and not config.scene_teacher and compiler is None:
            del text_backbone
            text_backbone = None
    else:
        task_caches = [
            model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
            for i in range(len(tasks))
        ]
        if text_backbone is not None:
            del text_backbone
            text_backbone = None

    import metaworld

    dagger_compress_frames: Callable[[np.ndarray], list[bytes]] | None = None
    dagger_get_policy: Callable[[str], Any] | None = None
    if args.dagger_output_dir is not None:
        from scripts.collect_long_trajectories import compress_frames, get_policy

        dagger_compress_frames = compress_frames
        dagger_get_policy = get_policy
        args.dagger_output_dir.mkdir(parents=True, exist_ok=True)

    per_task = {}
    trial_records: list[dict[str, Any]] = []
    completed_trials = 0
    trial_start, trial_stop = parse_trial_range(
        args.trial_range, args.trials_per_task
    )
    trials_in_shard = trial_stop - trial_start
    if (trial_start, trial_stop) != (0, args.trials_per_task):
        print(
            f"eval: trial shard [{trial_start}, {trial_stop}) of "
            f"{args.trials_per_task} per task",
            flush=True,
        )
    for local_task_index, (global_task_index, task_text) in enumerate(selected_tasks):
        env_name = descriptions_to_env.get(task_text)
        if args.peer_eval_trace and env_name != "assembly-v3":
            raise ValueError("--peer-eval-trace currently supports assembly-v3 only")
        if env_name is None:
            message = f"task {task_text[:40]} has no env_name mapping"
            if args.task35_precision_contract:
                raise ValueError(message)
            print(f"{message}: SKIP")
            continue
        # 采集同款环境：MT1(env_name, seed=42)，corner2 相机位置修正，
        # 物体随机化不冻结（每次 reset 随机 init，与训练数据分布一致）
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]  # corner2 位置（lerobot 采集同款）
        env._freeze_rand_vec = False
        dagger_episodes: list[dict[str, Any]] = []
        dagger_policy = (
            None if dagger_get_policy is None else dagger_get_policy(env_name)
        )
        dagger_path = (
            None
            if args.dagger_output_dir is None
            else args.dagger_output_dir
            / (
                f"metaworld_longtraj_{env_name}_dagger_"
                f"seed{args.episode_seed_base}_t{trial_start}-{trial_stop}.pt"
            )
        )
        if dagger_path is not None and dagger_path.exists():
            raise FileExistsError(f"refusing to overwrite DAgger data: {dagger_path}")
        wins = 0
        for trial in range(trial_start, trial_stop):
            episode_seed = evaluation_episode_seed(
                global_task_index,
                trial,
                base_seed=args.episode_seed_base,
            )
            # MetaWorld v3 的 reset_model 在部分版本仍读全局 NumPy RNG；
            # 仅传 env.reset(seed=...) 不足以固定任务布局。显式同步后，基线与
            # 候选的同一 trial 才是真正同初态配对。
            np.random.seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)  # 固定环境种子（口径要求）
            # 每个 trial 独立重置 flow 噪声：否则某模型提前成功会
            # 少消耗随机数，使后续 trial 与基线失去配对可比性。
            torch.manual_seed(1_000_000 + episode_seed)
            if args.align_init:
                # 对齐数据首帧（物体+target），把闭环拉回训练分布
                from mw_expert_replay import align_objects, load_episode_rows, load_episodes

                ep = next(
                    (e for e in load_episodes() if task_text in str(e.get("tasks"))),
                    None,
                )
                if ep is not None:
                    o0 = np.asarray(
                        load_episode_rows(ep)[0]["observation.environment_state"],
                        dtype=float,
                    )
                    align_report = align_objects(env, o0, env._get_obs())
                    # align_objects mutates MuJoCo state after reset.  Refresh
                    # the observation and expose the alignment report so an
                    # align-init diagnostic cannot silently run unaligned.
                    obs = env._get_obs()
                    print(
                        f"align-init task={global_task_index} trial={trial}: "
                        f"{align_report}",
                        flush=True,
                    )
                    if args.debug_first_action:
                        global _ALIGN_ACTS
                        feat = torch.load(args.features, map_location="cpu", weights_only=True)
                        tid = global_task_index
                        idx = int((feat["instruction_id"] == tid).nonzero()[0][0])
                        _ALIGN_ACTS = feat["actions"][idx].numpy()
            frame_buffer = []
            last_norm = np.zeros(4)  # 归一化动作（模型输入）
            chunk = (
                None
                if servo_runtime is not None
                else np.zeros((ACTION_HORIZON, 4))
            )
            memory = None
            metric_g_prev = None  # MT-VJ：上一决策的 g_t（ν_t = g_t − g_{t−1}，每 trial 重置）
            success = False
            stage_metrics = {
                "near_object": 0.0,
                "grasp_success": 0.0,
                "grasp_reward": 0.0,
                "in_place_reward": 0.0,
                "obj_to_target": float("inf"),
                "best_obj_step": -1,
            }
            decision_count = 0  # 2026-08-06：--memory-reset-every 的决策计数器
            recorded_chunks: list[list[list[float]]] = []
            peer_eval_trace: list[dict[str, Any]] = []
            active_peer_decision: dict[str, Any] | None = None
            last_env_info: dict[str, Any] = {}
            plan_queue = SynchronousPlanQueue(args.execution_horizon)
            takeover_step = (
                None
                if dagger_policy is None
                else dagger_takeover_step(
                    global_task_index,
                    episode_seed,
                    args.dagger_takeover_min,
                    args.dagger_takeover_max,
                )
            )
            dagger_frames: list[np.ndarray] = []
            dagger_actions: list[np.ndarray] = []
            dagger_states: list[np.ndarray] = []
            dagger_action_success: list[bool] = []
            dagger_action_source: list[str] = []
            dagger_metric_state: list[np.ndarray] = []
            dagger_metric_valid: list[bool] = []
            dagger_first_success_step: int | None = None
            plan_step = None  # C²/伺服：上次规划的原始步
            c2_token = 0  # C²/伺服：自规划以来消费的 token 索引
            c2_params = None  # C²：缓存的 {ū, c̄, K}
            readout = None  # servo：最近一次 MultiModeReadout
            roi = None  # --fovea：最近一次 plan 的 ROI（渲染像素空间 [1,3]）
            innovation_flag = False  # servo 新息标志（True → 下一步立即全局刷新）
            servo_lang_cond = None  # servo 语言条件（plan 时取 role queries 均值）
            servo_first = True  # 首决策 a_prev=None（ν≡0，servo.py 契约）
            for step in range(args.horizon):
                img = env.render()  # 数据图像与本地渲染一致（实测 MAE 0.48 vs flip 55，勿加 flip）
                if args.show_window:
                    import cv2

                    title = "MetaWorld closed-loop (q to stop)"
                    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        args.show_window = False
                frame_buffer.append(img)
                if step == 0:
                    # 2026-08-06 评估缺陷修复：原实现首决策前执行 chunk 的初始零值
                    # （归一化零 → 反归一化 (aq99+aq01)/2 = [2.45, 2.27, -1.37, 0]，
                    # 环境裁剪后 [1, 1, -1, 0]——实测把机械手提前移动 ~4.3cm，
                    # 精细抓取任务直接进入分布外状态）。
                    # 用首帧重复填充窗口使 step 0 立即推理：与训练首决策窗口同分布
                    # （prepare 的 clip_frame_indices 以 max(0, d-offset*stride)
                    # 钳制，episode 首窗口本身由重复帧组成）。
                    while len(frame_buffer) < (VISION_WINDOW - 1) * LEGACY_TRAINING_CONTROL_STRIDE + 1:
                        frame_buffer.insert(0, img)
                if len(frame_buffer) > (VISION_WINDOW - 1) * LEGACY_TRAINING_CONTROL_STRIDE + 1:
                    frame_buffer.pop(0)
                # 与训练一致的时间升序 [d-6, d-4, d-2, d]（clip_frame_indices 返回
                # video_start + max(0, d - offset*stride)，offset 升序 → 最老帧在前）
                # 修复（2026-08-05 多 agent 审查）：旧代码 range(-1,-2*W,-2) 是降序 [d,d-2,...]，
                # 与训练数据方向相反，V-JEPA 时序注意力对帧序敏感 → MW 闭环数字无效
                indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
                frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
                if config.c2_controller:
                    # C² 部署（Codex 修正 5）：plan_stride 步重规划一次 {ū,c̄,K}；
                    # 每 feedback_stride 步刷新并消费 token。C²-IRF v2 扩展：
                    # fovea 时 plan_due 全图重读 + ROI，feedback 步 foveal crop
                    # 局部关系更新；servo 新息超阈值立即提前全局刷新。
                    clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
                    mode, correction_due, _ = fovea_schedule(
                        step, plan_step, args.plan_stride, args.feedback_stride,
                        ACTION_HORIZON, fovea=args.fovea,
                        innovation_flag=innovation_flag,
                    )
                    if mode == MODE_PLAN:
                        if (
                            args.memory_reset_every > 0
                            and decision_count > 0
                            and decision_count % args.memory_reset_every == 0
                        ):
                            memory = None
                        decision_count += 1
                        state = state_take_normalize(obs, args.state_take, sq01, scale_s)
                        proprio = torch.tensor(state, device=device)[None, None]
                        previous = torch.tensor(
                            np.zeros(4, dtype=np.float32) if args.prev_zero else last_norm,
                            dtype=torch.float32, device=device,
                        )[None, None]
                        with torch.inference_mode():
                            tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                            if servo_runtime is not None:
                                # servo 路径：一次 reader 调用（带 prev_mu 跟踪先验）
                                # 同时产出 MultiModeReadout 与 31-token 视觉流。
                                readout, vision_in = _servo_vision(
                                    model, tokens, task_caches[local_task_index],
                                    dense_coords_arr, prev_mu=servo_runtime.prev_mu,
                                )
                                if args.fovea:
                                    # ROI 在渲染像素空间（480×480）；同一仿射
                                    # 应用于整个 4 帧窗口（§三.1/§三.2）。
                                    render_size = frame_buffer[-1].shape[0]
                                    if render_size % 16:
                                        raise ValueError(
                                            f"--fovea 需要渲染尺寸能被 16 整除，"
                                            f"got {render_size}"
                                        )
                                    roi = compute_roi(
                                        *select_roi_pair(
                                            readout.mu, readout.cov, readout.vis
                                        ),
                                        image_size=render_size,
                                    ).detach().cpu().float()
                            else:
                                vision_in = _apply_local_vision(
                                    model, tokens, task_caches[local_task_index]
                                )
                            c_current = model.control_projector(tokens)
                            cond, memory = model.encode_condition(
                                vision_in,
                                proprio[0],
                                previous[0],
                                language_cache=task_caches[local_task_index],
                                visual_memory=memory,
                                return_visual_memory=True,
                            )
                            c2_params = model.controller_params(cond, c_current)
                            if args.c2_gain_scale != 1.0:
                                c2_params = ControllerParams(
                                    c2_params.nominal,
                                    c2_params.reference,
                                    c2_params.gain * args.c2_gain_scale,
                                )
                            if args.c2_zero_gain:
                                c2_params = ControllerParams(
                                    c2_params.nominal,
                                    c2_params.reference,
                                    torch.zeros_like(c2_params.gain),
                                )
                        plan_step = step
                        c2_token = 0
                    if correction_due and c2_token < ACTION_HORIZON and c2_params is not None:
                        if servo_runtime is not None:
                            # 注：validate_servo_args 保证 c2 checkpoint 下
                            # servo_runtime 恒为 None（servo 训练与 c2 互斥）；
                            # 此分支保留以备未来放开组合。
                            with torch.inference_mode():
                                if step != plan_step:
                                    if args.fovea:
                                        # foveal 局部更新：ROI crop → 编码 →
                                        # 局部关系状态（40Hz，§三.3）。审查 P0-4：
                                        # 默认完整 V-JEPA（H11）编码 crop，与
                                        # reader 训练特征层一致；P0-3：prev_mu
                                        # 先 full→crop 变换再进 reader，reader
                                        # 输出的 mu/cov 逆变换回全图再送 servo
                                        # （crop 归一化坐标 ≠ 全图坐标）。
                                        render_size = frame_buffer[-1].shape[0]
                                        tokens = _foveal_tokens(
                                            frames, roi, device,
                                            vision_backbone=vision_backbone,
                                            fovea_encoder=fovea_encoder,
                                        )
                                        prev_crop = (
                                            full_to_crop_norm(
                                                servo_runtime.prev_mu, roi,
                                                render_size,
                                            )
                                            if servo_runtime.prev_mu is not None
                                            else None
                                        )
                                        readout, _ = _servo_vision(
                                            model, tokens, task_caches[local_task_index],
                                            dense_coords_arr, prev_mu=prev_crop,
                                        )
                                        readout = MultiModeReadout(
                                            readout.slots,
                                            crop_to_full_norm(
                                                readout.mu, roi, render_size
                                            ),
                                            crop_to_full_cov(
                                                readout.cov, roi, render_size
                                            ),
                                            readout.vis,
                                            readout.weights,
                                        )
                                    else:
                                        tokens = vision_backbone(
                                            clip.unsqueeze(0), pooling=vision_pooling
                                        )
                                        readout, _ = _servo_vision(
                                            model, tokens, task_caches[local_task_index],
                                            dense_coords_arr,
                                            prev_mu=servo_runtime.prev_mu,
                                        )
                                state_norm = state_take_normalize(
                                    obs, args.state_take, sq01, scale_s
                                )
                                correction, innovation_flag = servo_runtime.correct(
                                    readout,
                                    state_norm[:4],
                                    task_caches[local_task_index].role_queries.mean(dim=1),
                                    a_prev=None if servo_first else last_norm,
                                )
                                nominal = c2_params.nominal[0, c2_token].cpu().numpy()
                                if correction is None:
                                    norm_action = nominal
                                else:
                                    norm_action = np.clip(
                                        nominal + correction, -1.0, 1.0
                                    )
                            servo_first = False
                            c2_token += 1
                        else:
                            with torch.inference_mode():
                                if step != plan_step:
                                    # feedback 刷新：重新编码当前窗口 → c_current。
                                    tokens = vision_backbone(
                                        clip.unsqueeze(0), pooling=vision_pooling
                                    )
                                c_current = model.control_projector(tokens)
                                if args.c2_oracle_ref:
                                    # 参考零误差上界：c̄ ≡ c_current（e ≡ 0，K 空转）。
                                    norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                                else:
                                    error = c_current[0] - c2_params.reference[0, c2_token]
                                    if (
                                        args.c2_error_threshold > 0.0
                                        and float(error.norm()) < args.c2_error_threshold
                                    ):
                                        norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                                    else:
                                        norm_action = (
                                            c2_params.nominal[0, c2_token]
                                            - c2_params.gain[0, c2_token] @ error
                                        ).cpu().numpy()
                            norm_action = np.clip(norm_action, -1.0, 1.0)
                            c2_token += 1
                    else:
                        # 非刷新步：保持上一动作（feedback_stride > 1 时）。
                        norm_action = last_norm
                elif servo_runtime is not None:
                    # 伺服部署（flow checkpoint + servo 权重；--servo-ablation/
                    # --fovea）：c2_schedule 节奏——plan_due 全图重读 + 解码名义
                    # chunk（ā），feedback 步伺服修正（fovea 时 foveal crop 局部
                    # 关系更新）。修正 = clip(ā + correction)（correction 已含
                    # 阶段上限/假设混合/β，va_compound/servo.py 契约）。
                    mode, correction_due, _ = fovea_schedule(
                        step, plan_step, args.plan_stride, args.feedback_stride,
                        ACTION_HORIZON, fovea=args.fovea,
                        innovation_flag=innovation_flag,
                    )
                    if mode == MODE_PLAN:
                        if (
                            args.memory_reset_every > 0
                            and decision_count > 0
                            and decision_count % args.memory_reset_every == 0
                        ):
                            memory = None
                        decision_count += 1
                        state = state_take_normalize(obs, args.state_take, sq01, scale_s)
                        proprio = torch.tensor(state, device=device)[None, None]
                        previous = torch.tensor(
                            np.zeros(4, dtype=np.float32) if args.prev_zero else last_norm,
                            dtype=torch.float32, device=device,
                        )[None, None]
                        # 审查 P0-1：servo 分支此前未构造 frames/clip（该变量只在
                        # C²/普通分支创建），首个 plan 步即 UnboundLocalError。
                        # 与 C² 分支同一时间升序窗口 [d-6, d-4, d-2, d]。
                        indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
                        frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
                        clip = torch.cat(
                            [preprocess(f, 384) for f in frames], dim=0
                        ).to(device)
                        with torch.inference_mode():
                            tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                            readout, vision_in = _servo_vision(
                                model, tokens, task_caches[local_task_index],
                                dense_coords_arr, prev_mu=servo_runtime.prev_mu,
                            )
                            # servo 语言条件：role queries 均值（与 train.py
                            # servo_correction_t0 同一构造）。
                            servo_lang_cond = (
                                task_caches[local_task_index].role_queries.mean(dim=1)
                            )
                            if args.fovea:
                                # ROI 在渲染像素空间（480×480）；同一仿射应用于
                                # 整个 4 帧窗口（§三.1/§三.2）。
                                render_size = frame_buffer[-1].shape[0]
                                if render_size % 16:
                                    raise ValueError(
                                        f"--fovea 需要渲染尺寸能被 16 整除，"
                                        f"got {render_size}"
                                    )
                                roi = compute_roi(
                                    *select_roi_pair(
                                        readout.mu, readout.cov, readout.vis
                                    ),
                                    image_size=render_size,
                                ).detach().cpu().float()
                            servo_dense_kwargs: dict = {}
                            if args.dense_readout_mtvj and mtvj_backbone is not None:
                                (
                                    servo_dense_kwargs,
                                    _,
                                    servo_vision,
                                    metric_g_prev,
                                ) = _decision_mtvj_context(
                                    frames=frames,
                                    clip=clip,
                                    mtvj_backbone=mtvj_backbone,
                                    metric_head=metric_head,
                                    relation_encoder=relation_encoder,
                                    language_hidden=hidden[
                                        local_task_index : local_task_index + 1
                                    ],
                                    language_mask=mask[
                                        local_task_index : local_task_index + 1
                                    ],
                                    coords_mtvj=coords_mtvj,
                                    metric_g_prev=metric_g_prev,
                                    device=device,
                                    local_slots=bool(model.config.local_slots),
                                    roi_head=roi_head,
                                    roi_alpha=(
                                        float(args.mtvj_roi_alpha)
                                        if roi_head is not None
                                        else 0.0
                                    ),
                                )
                                if servo_vision is not None:
                                    vision_in = servo_vision
                            if action_vision_backbone is not None:
                                servo_dense_kwargs["action_dense_evidence"] = (
                                    _action_vision_encode_window(
                                        frames, action_vision_backbone, device
                                    )
                                )
                            # 训练侧 flow_semantic 时槽/Pool16 输出作为 flow
                            # head 逐层 cross-attn 语义上下文（同常规路径）。
                            semantic_ctx = (
                                vision_in
                                if (
                                    getattr(config, "flow_semantic", False)
                                    and not config.direct_head
                                )
                                else None
                            )
                            cond, memory = model.encode_condition(
                                vision_in,
                                proprio[0],
                                previous[0],
                                language_cache=task_caches[local_task_index],
                                visual_memory=memory,
                                return_visual_memory=True,
                                **servo_dense_kwargs,
                            )
                            if args.flow_samples == 1:
                                decoded = model.decode_actions(
                                    cond,
                                    steps=flow_steps,
                                    semantic_context=semantic_ctx,
                                )
                            else:
                                decoded = torch.stack(
                                    [
                                        model.decode_actions(
                                            cond,
                                            steps=flow_steps,
                                            semantic_context=semantic_ctx,
                                        )
                                        for _ in range(args.flow_samples)
                                    ]
                                ).mean(dim=0)
                            chunk = decoded[0].cpu().numpy()
                        plan_step = step
                        c2_token = 0
                    if correction_due and c2_token < ACTION_HORIZON and chunk is not None:
                        with torch.inference_mode():
                            if step != plan_step:
                                if args.fovea:
                                    # foveal 局部更新：ROI crop → 编码 → 局部
                                    # 关系状态（40Hz，§三.3）。审查 P0-4：默认
                                    # 完整 V-JEPA（H11）编码 crop（与 reader
                                    # 训练特征层一致）；P0-3：prev_mu 先
                                    # full→crop 变换再进 reader，reader 输出
                                    # mu/cov 逆变换回全图再送 servo（crop 归一化
                                    # 坐标 ≠ 全图坐标）。
                                    render_size = frame_buffer[-1].shape[0]
                                    tokens = _foveal_tokens(
                                        frames, roi, device,
                                        vision_backbone=vision_backbone,
                                        fovea_encoder=fovea_encoder,
                                    )
                                    prev_crop = (
                                        full_to_crop_norm(
                                            servo_runtime.prev_mu, roi, render_size
                                        )
                                        if servo_runtime.prev_mu is not None
                                        else None
                                    )
                                    readout, _ = _servo_vision(
                                        model, tokens, task_caches[local_task_index],
                                        dense_coords_arr, prev_mu=prev_crop,
                                    )
                                    readout = MultiModeReadout(
                                        readout.slots,
                                        crop_to_full_norm(
                                            readout.mu, roi, render_size
                                        ),
                                        crop_to_full_cov(
                                            readout.cov, roi, render_size
                                        ),
                                        readout.vis,
                                        readout.weights,
                                    )
                                else:
                                    feedback_clip = torch.cat(
                                        [preprocess(f, 384) for f in frames], dim=0
                                    ).to(device)
                                    tokens = vision_backbone(
                                        feedback_clip.unsqueeze(0), pooling=vision_pooling
                                    )
                                    readout, _ = _servo_vision(
                                        model, tokens, task_caches[local_task_index],
                                        dense_coords_arr,
                                        prev_mu=servo_runtime.prev_mu,
                                    )
                            state_norm = state_take_normalize(
                                obs, args.state_take, sq01, scale_s
                            )
                            correction, innovation_flag = servo_runtime.correct(
                                readout,
                                state_norm[:4],
                                servo_lang_cond,
                                a_prev=None if servo_first else last_norm,
                            )
                            nominal = chunk[c2_token]
                            if correction is None:
                                norm_action = nominal
                            else:
                                norm_action = np.clip(
                                    nominal + correction, -1.0, 1.0
                                )
                        servo_first = False
                        c2_token += 1
                    else:
                        norm_action = last_norm
                elif plan_queue.needs_plan(step) and len(frame_buffer) >= VISION_WINDOW:
                    if (
                        args.memory_reset_every > 0
                        and decision_count > 0
                        and decision_count % args.memory_reset_every == 0
                    ):
                        memory = None  # 契约缺口对照：截断递归记忆到训练深度
                    if (
                        args.world_reset_every > 0
                        and decision_count > 0
                        and decision_count % args.world_reset_every == 0
                    ):
                        # peer_sync_h6 长 horizon：world_state 对齐训练 4 步窗口，
                        # 只重置 WAMState（保留 VA 视觉记忆），避免 belief 发散。
                        memory = _reset_world_state(memory)
                    if (
                        args.world_map_reset_every > 0
                        and decision_count > 0
                        and decision_count % args.world_map_reset_every == 0
                    ):
                        memory = _reset_world_map(memory)
                    decision_count += 1
                    # 与训练一致的时间升序 [d-6, d-4, d-2, d]（clip_frame_indices 返回
                    # video_start + max(0, d - offset*stride)，offset 升序 → 最老帧在前）
                    # 修复（2026-08-05 多 agent 审查）：旧代码 range(-1,-2*W,-2) 是降序 [d,d-2,...]，
                    # 与训练数据方向相反，V-JEPA 时序注意力对帧序敏感 → MW 闭环数字无效
                    clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
                    with torch.inference_mode():
                        if main_vision_backbone is not None:
                            # DINO-main replacement：冻结 DINOv2 特征替换 V-JEPA
                            # 主视觉（与训练 _dino_main_online_encode 同构）。
                            # DINO-metric：同一窗口附带 block11/block23 两帧
                            # [d-2,d] patch evidence（与训练 return_dense 同构）。
                            dino_dense_evidence = None
                            if getattr(config, "dino_dense_metric", False):
                                encode_frames = task35_ablation_frames(
                                    frames, args.task35_causal_ablation
                                )
                                tokens, dino_dense_evidence = _main_vision_encode_window(
                                    encode_frames,
                                    main_vision_backbone,
                                    device,
                                    grid=config.main_vision_grid,
                                    window=config.main_vision_frames,
                                    return_dense=True,
                                )
                            else:
                                tokens = _main_vision_encode_window(
                                    frames,
                                    main_vision_backbone,
                                    device,
                                    grid=config.main_vision_grid,
                                    window=config.main_vision_frames,
                                )
                        else:
                            tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                    if has_plan and plan_refresh_due(decision_count, args.plan_refresh):
                        # Plan-Cache：用当前窗口场景（vision 全局均值）重建该任务缓存
                        scene_summary = tokens.mean(dim=1)  # [1, vision_dim]
                        task_caches[local_task_index] = build_plan_language_cache(
                            model,
                            hidden[local_task_index : local_task_index + 1].to(device),
                            mask[local_task_index : local_task_index + 1].to(device),
                            scene_summary,
                            instruction=(
                                tasks[local_task_index]
                                if config.scene_teacher or compiler is not None
                                else None
                            ),
                            # plan_resampler 分支不访问 text_backbone；短路避免 NameError
                            text_backbone=(
                                text_backbone
                                if config.scene_teacher or compiler is not None
                                else None
                            ),
                            scene_teacher=scene_teacher,
                            compiler=compiler,
                            scene_tokens=tokens if compiler is not None else None,
                            semantic_history=(
                                torch.zeros(
                                    1,
                                    compiler.history_in_dim,
                                    device=device,
                                    dtype=torch.float32,
                                )
                                if compiler is not None
                                else None
                            ),
                            scene_delta=(
                                torch.zeros(
                                    1, config.vision_dim, device=device
                                )
                                if compiler is not None
                                else None
                            ),
                        )
                    state = state_take_normalize(obs, args.state_take, sq01, scale_s)
                    proprio = torch.tensor(state, device=device)[None, None]
                    previous = torch.tensor(
                        np.zeros(4, dtype=np.float32) if args.prev_zero else last_norm,
                        dtype=torch.float32, device=device,
                    )[None, None]
                    with torch.inference_mode():
                        vision_in = _apply_local_vision(
                            model, tokens, task_caches[local_task_index]
                        )
                        dense_kwargs = {}
                        metric_g = None
                        metric_trace: dict[str, Any] | None = (
                            {} if args.peer_eval_trace else None
                        )
                        if getattr(config, "dino_dense_metric", False):
                            # DINO-metric 在线解码：dense evidence（本决策窗口
                            # 已编码）+ metric tokens（g_t / ν_t，与训练
                            # _dino_metric_tokens 同构；coords=512 两帧 16×16）。
                            if dino_dense_evidence is None:
                                raise RuntimeError(
                                    "dino_dense_metric 已启用但本决策无 dense evidence"
                                )
                            dino_roi_video = None
                            if roi_head is not None and args.dino_roi_alpha:
                                # ROI 原图输入与训练同构：窗口帧 [d-2,d] 双时间片
                                # [1,2,3,H,W] NCHW 0-1 未归一化（refine 内部做
                                # ImageNet 归一化 + 裁剪上采样；crop 契约 NCHW）。
                                dino_roi_video = (
                                    torch.from_numpy(
                                        np.stack(encode_frames[2:4], axis=0)[None]
                                    )
                                    .float()
                                    .div_(255.0)
                                    .permute(0, 1, 4, 2, 3)
                                    .to(device)
                                )
                            metric_dense_evidence = task35_ablation_dense(
                                dino_dense_evidence, args.task35_causal_ablation
                            )
                            dino_metric_tokens, metric_g = _mtvj_metric_tokens(
                                metric_head,
                                relation_encoder,
                                metric_dense_evidence,
                                hidden[local_task_index : local_task_index + 1],
                                mask[local_task_index : local_task_index + 1],
                                coords_dino_metric,
                                metric_g_prev,
                                device,
                                roi_head=(
                                    None
                                    if args.task35_causal_ablation == "roi-off"
                                    else roi_head
                                ),
                                roi_backbone=(
                                    main_vision_backbone
                                    if roi_head is not None
                                    and args.task35_causal_ablation != "roi-off"
                                    else None
                                ),
                                roi_video=dino_roi_video,
                                roi_alpha=(
                                    float(args.dino_roi_alpha)
                                    if roi_head is not None
                                    and args.task35_causal_ablation != "roi-off"
                                    else 0.0
                                ),
                                roi_dino=True,
                                trace_out=metric_trace,
                            )
                            metric_g_prev = metric_g
                            metric_g_policy = task35_ablation_geometry(
                                metric_g[None], args.task35_causal_ablation
                            )
                            dense_kwargs = {
                                "dense_evidence": metric_dense_evidence,
                                "metric_tokens": dino_metric_tokens,
                                "metric_g": metric_g_policy,
                            }
                        if args.dense_readout_mtvj:
                            # MT-VJ 在线 dense 解码（契约 §7，与训练 §6
                            # _mtvj_online_encode 同构）：同一 4 帧历史窗。
                            (
                                dense_kwargs,
                                metric_g,
                                vision_override,
                                metric_g_prev,
                            ) = _decision_mtvj_context(
                                frames=frames,
                                clip=clip,
                                mtvj_backbone=mtvj_backbone,
                                metric_head=metric_head,
                                relation_encoder=relation_encoder,
                                language_hidden=hidden[
                                    local_task_index : local_task_index + 1
                                ],
                                language_mask=mask[
                                    local_task_index : local_task_index + 1
                                ],
                                coords_mtvj=coords_mtvj,
                                metric_g_prev=metric_g_prev,
                                device=device,
                                local_slots=bool(model.config.local_slots),
                                roi_head=roi_head,
                                roi_alpha=(
                                    float(args.mtvj_roi_alpha)
                                    if roi_head is not None
                                    else 0.0
                                ),
                            )
                            if vision_override is not None:
                                vision_in = vision_override
                        if action_vision_backbone is not None:
                            dense_kwargs["action_dense_evidence"] = (
                                _action_vision_encode_window(
                                    frames, action_vision_backbone, device
                                )
                            )
                        world_action = None
                        proposal_noise = None
                        decoded_proposal = None
                        peer_checkpoint = (
                            getattr(config, "va_world_mode", "legacy")
                            == "peer_sync_h6"
                        )
                        if getattr(model, "wmrm", None) is not None:
                            expected_world_horizon = (
                                int(config.wmrm_cycle_steps)
                                if peer_checkpoint
                                else EXPECTED_WMRM_WORLD_HORIZON
                            )
                            if (
                                model.wmrm.cycle_steps
                                != expected_world_horizon
                            ):
                                raise ValueError(
                                    "WAM4VA world_horizon must match its deployment "
                                    "planning cycle: "
                                    f"{model.wmrm.cycle_steps} != "
                                    f"{expected_world_horizon}"
                                )
                            if args.flow_samples != 1:
                                raise ValueError(
                                    "WAM4VA closed loop requires --flow-samples 1"
                                )
                            if not peer_checkpoint:
                                # Legacy WMRM deployment needs a preliminary flow
                                # proposal as the executable World-action input.
                                proposal_cond, _ = model.encode_condition(
                                    vision_in,
                                    proprio[0],
                                    previous[0],
                                    language_cache=task_caches[local_task_index],
                                    visual_memory=memory,
                                    return_visual_memory=True,
                                    skip_wmrm=True,
                                    **dense_kwargs,
                                )
                                proposal_noise = torch.randn(
                                    proposal_cond.shape[0],
                                    config.action_horizon,
                                    config.action_dim,
                                    device=proposal_cond.device,
                                    dtype=proposal_cond.dtype,
                                )
                                decoded_proposal = model.decode_actions(
                                    proposal_cond,
                                    steps=flow_steps,
                                    noise=proposal_noise,
                                    semantic_context=(
                                        vision_in
                                        if (
                                            getattr(config, "flow_semantic", False)
                                            and not config.direct_head
                                        )
                                        else None
                                    ),
                                )
                                cycle = model.wmrm.cycle_steps
                                if decoded_proposal.shape[1] < cycle:
                                    raise ValueError(
                                        f"decoded action horizon {decoded_proposal.shape[1]} "
                                        f"is shorter than WAM4VA cycle {cycle}"
                                    )
                                world_action = decoded_proposal[:, :cycle].clamp(-1.0, 1.0)
                        pre_decision_memory = memory
                        cond, memory = model.encode_condition(
                            vision_in,
                            proprio[0],
                            previous[0],
                            language_cache=task_caches[local_task_index],
                            visual_memory=memory,
                            return_visual_memory=True,
                            env_action=world_action,
                            skip_wmrm=args.peer_world_off,
                            **dense_kwargs,
                        )
                        peer_stage_records = (
                            _peer_world_trace_stages(model)
                            if args.peer_eval_trace
                            else None
                        )
                        # 训练侧 flow_semantic 时槽输出（vision_in）作为 flow
                        # head 逐层 cross-attn 语义上下文；闭环必须传同一路径，
                        # 否则语义通道静默回退为 action_condition（数字失真）。
                        semantic_ctx = (
                            vision_in
                            if (
                                getattr(config, "flow_semantic", False)
                                and not config.direct_head
                            )
                            else None
                        )
                        if args.peer_eval_trace and proposal_noise is None:
                            # This is exactly the random draw sample_actions would
                            # make. Making it explicit preserves the baseline while
                            # allowing a same-state, same-noise World-off decode.
                            proposal_noise = torch.randn(
                                cond.shape[0],
                                config.action_horizon,
                                config.action_dim,
                                device=cond.device,
                                dtype=cond.dtype,
                            )
                        if args.flow_samples == 1:
                            decoded = model.decode_actions(
                                cond,
                                steps=flow_steps,
                                noise=proposal_noise,
                                semantic_context=semantic_ctx,
                            )
                        else:
                            decoded = torch.stack(
                                [
                                    model.decode_actions(
                                        cond,
                                        steps=flow_steps,
                                        semantic_context=semantic_ctx,
                                    )
                                    for _ in range(args.flow_samples)
                                ]
                            ).mean(dim=0)
                        chunk = decoded[0].cpu().numpy()
                        if args.peer_eval_trace:
                            if not all(
                                np.array_equal(row["readout"], row["operative"])
                                for row in peer_stage_records
                            ):
                                raise RuntimeError(
                                    "deployment World stage did not consume its own readout"
                                )
                            world_stage = int(peer_stage_records[-1]["stage"])
                            world_readout = np.asarray(
                                peer_stage_records[-1]["readout"]
                            )
                            world_off_cond = model.encode_condition(
                                vision_in,
                                proprio[0],
                                previous[0],
                                language_cache=task_caches[local_task_index],
                                visual_memory=pre_decision_memory,
                                skip_wmrm=True,
                                **dense_kwargs,
                            )
                            world_off_chunk = model.decode_actions(
                                world_off_cond,
                                steps=flow_steps,
                                noise=proposal_noise,
                                semantic_context=semantic_ctx,
                            )[0].cpu().numpy()
                            candidate_action = np.clip(
                                world_off_chunk, -1.0, 1.0
                            )
                            candidate_tensor = torch.as_tensor(
                                candidate_action,
                                device=cond.device,
                                dtype=cond.dtype,
                            )[None]
                            candidate_cond = model.encode_condition(
                                vision_in,
                                proprio[0],
                                previous[0],
                                language_cache=task_caches[local_task_index],
                                visual_memory=pre_decision_memory,
                                env_action=candidate_tensor,
                                **dense_kwargs,
                            )
                            candidate_stage_records = _peer_world_trace_stages(model)
                            candidate_chunk = model.decode_actions(
                                candidate_cond,
                                steps=flow_steps,
                                noise=proposal_noise,
                                semantic_context=semantic_ctx,
                            )[0].cpu().numpy()
                            active_peer_decision = {
                                "decision": int(decision_count - 1),
                                "env_step_start": int(step),
                                "world_stage": world_stage,
                                **_action_trace_metrics(world_readout, chunk),
                                "world_stages": _peer_trace_stage_metrics(
                                    peer_stage_records, chunk
                                ),
                                "same_noise_counterfactual": {
                                    "history_note": (
                                        "current-decision World bypass; pre-decision memory "
                                        "still contains prior World influence"
                                    ),
                                    "flow_current_world_bypass_raw": world_off_chunk.tolist(),
                                    "flow_candidate_conditioned_raw": candidate_chunk.tolist(),
                                    "candidate_action_clipped": candidate_action.tolist(),
                                    "world_on_vs_current_bypass": _peer_world_effect_metrics(
                                        chunk, world_off_chunk
                                    ),
                                    "candidate_conditioned_vs_current_bypass": (
                                        _peer_world_effect_metrics(
                                            candidate_chunk, world_off_chunk
                                        )
                                    ),
                                    "candidate_world_stages": _peer_trace_stage_metrics(
                                        candidate_stage_records, candidate_chunk
                                    ),
                                },
                                "condition_summary": {
                                    "world_on": _condition_trace_summary(cond),
                                    "current_world_bypass": _condition_trace_summary(
                                        world_off_cond
                                    ),
                                    "candidate_conditioned": _condition_trace_summary(
                                        candidate_cond
                                    ),
                                },
                                "flow_noise": proposal_noise[0]
                                .detach()
                                .float()
                                .cpu()
                                .tolist(),
                                "action_affine": {
                                    "q01": np.asarray(aq01, dtype=float).tolist(),
                                    "q99": np.asarray(aq99, dtype=float).tolist(),
                                },
                                "pre_decision_assembly": _assembly_trace_state(env, {}),
                                "metric_perception": metric_trace,
                                "metric_oracle": _assembly_metric_oracle(env),
                                "denormalized_command": (
                                    np.clip(chunk, -1.0, 1.0)
                                    * (aq99 - aq01) / 2
                                    + (aq99 + aq01) / 2
                                ).astype(float).tolist(),
                                "executed_token_count": 0,
                                "tokens": [],
                            }
                            peer_eval_trace.append(active_peer_decision)
                        # Standard evaluation uses an absolute-time synchronous plan;
                        # execution_horizon controls the hard replacement prefix while
                        # peer WMRM consumes the same checkpoint planning prefix.
                        plan_queue.replace(step, chunk)
                        if args.record_action_chunks:
                            recorded_chunks.append(chunk.astype(float).tolist())
                        if args.debug_first_action and not _DEBUG_FA_DONE.get("x"):
                            _DEBUG_FA_DONE["x"] = True
                            print(f"DEBUG first chunk0={np.round(chunk[0], 4)}")
                            if args.align_init and _ALIGN_ACTS is not None:
                                dc = step // LEGACY_EXECUTION_HORIZON
                                if dc < len(_ALIGN_ACTS):
                                    ref = _ALIGN_ACTS[dc][0]
                                    print(
                                        "DEBUG data act0:",
                                        np.round(ref, 4),
                                        "mae:",
                                        round(float(np.abs(chunk[0] - ref).mean()), 4),
                                    )
                # 模型输出为归一化动作：与训练标签一致裁剪到 [-1,1]（robust_normalize
                # 存盘即 clip），再反归一化到环境原始动作空间；prev 反馈同样用裁剪值。
                # servo 部署的 norm_action 由伺服分支给出（clip(ā + correction)），
                # 不能再用 chunk 覆盖。
                if not config.c2_controller and servo_runtime is None:
                    if plan_queue.needs_plan(step):
                        raise RuntimeError(
                            f"synchronous action plan missing at environment step {step}"
                        )
                    norm_action = np.clip(plan_queue.action_at(step), -1.0, 1.0)

                action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
                if takeover_step is not None and step >= takeover_step:
                    action = np.clip(
                        np.asarray(dagger_policy.get_action(obs), dtype=np.float32),
                        -1.0,
                        1.0,
                    )
                    scale = np.asarray(aq99 - aq01, dtype=np.float32)
                    norm_action = np.clip(
                        2.0 * (action - aq01) / scale - 1.0,
                        -1.0,
                        1.0,
                    )
                if takeover_step is not None:
                    dagger_frames.append(np.asarray(img, dtype=np.uint8))
                    dagger_actions.append(np.asarray(action, dtype=np.float32).copy())
                    dagger_states.append(np.asarray(obs[:4], dtype=np.float32).copy())
                    dagger_action_source.append(
                        "expert_takeover" if step >= takeover_step else "current_policy"
                    )
                    if env_name == "door-lock-v3":
                        lock_pos = np.asarray(obs[4:7], dtype=np.float32)
                        lock_target = np.asarray(
                            getattr(env, "_target_pos", np.full(3, np.nan)),
                            dtype=np.float32,
                        ).reshape(-1)[:3]
                        valid_metric = (
                            lock_pos.shape == (3,)
                            and lock_target.shape == (3,)
                            and np.isfinite(lock_pos).all()
                            and np.isfinite(lock_target).all()
                        )
                        dagger_metric_state.append(
                            np.concatenate((lock_pos, lock_target))
                            if valid_metric
                            else np.zeros(6, dtype=np.float32)
                        )
                        dagger_metric_valid.append(bool(valid_metric))
                pre_tcp = (
                    np.asarray(env.tcp_center, dtype=float).reshape(3).copy()
                    if args.peer_eval_trace
                    else None
                )
                pre_assembly = (
                    _assembly_trace_state(env, last_env_info)
                    if args.peer_eval_trace
                    else None
                )
                obs, reward, terminated, truncated, info = env.step(action)
                if takeover_step is not None:
                    dagger_action_success.append(bool(info.get("success")))
                if args.peer_eval_trace:
                    if active_peer_decision is None or plan_queue.plan is None:
                        raise RuntimeError("peer trace token has no active decision plan")
                    token = int(step - plan_queue.plan.start_step)
                    if not 0 <= token < args.execution_horizon:
                        raise RuntimeError(
                            f"peer trace token {token} is outside execution horizon"
                        )
                    _append_peer_trace_token(
                        active_peer_decision,
                        token=token,
                        env_step=step,
                        normalized_command=norm_action,
                        denormalized_command=action,
                        pre_tcp=pre_tcp,
                        post_tcp=env.tcp_center,
                        reward=reward,
                        pre_assembly=pre_assembly,
                        assembly=_assembly_trace_state(env, info),
                        terminated=terminated,
                        truncated=truncated,
                    )
                last_env_info = dict(info)
                last_norm = norm_action
                if args.debug_stage_metrics:
                    for metric_name in (
                        "near_object",
                        "grasp_success",
                        "grasp_reward",
                        "in_place_reward",
                    ):
                        value = float(info.get(metric_name, 0.0))
                        stage_metrics[metric_name] = max(
                            stage_metrics[metric_name], value
                        )
                    obj_to_target = float(info.get("obj_to_target", float("inf")))
                    if obj_to_target < stage_metrics["obj_to_target"]:
                        stage_metrics["obj_to_target"] = obj_to_target
                        stage_metrics["best_obj_step"] = step
                if info.get("success"):
                    success = True
                    if takeover_step is None or step < takeover_step:
                        break
                    if dagger_first_success_step is None:
                        dagger_first_success_step = step
                if (
                    dagger_first_success_step is not None
                    and step - dagger_first_success_step >= DAGGER_POST_SUCCESS_STEPS
                ):
                    break
                if terminated or truncated:
                    break
            dagger_collected = False
            if takeover_step is not None:
                episode = build_dagger_episode(
                    episode_seed=episode_seed,
                    takeover_step=takeover_step,
                    prefix_keep=args.dagger_prefix_keep,
                    frames=dagger_frames,
                    actions=dagger_actions,
                    states=dagger_states,
                    action_success=dagger_action_success,
                    action_source=dagger_action_source,
                    metric_state=(
                        dagger_metric_state if env_name == "door-lock-v3" else None
                    ),
                    metric_state_valid=(
                        dagger_metric_valid if env_name == "door-lock-v3" else None
                    ),
                )
                if episode is not None:
                    if dagger_compress_frames is None:
                        raise RuntimeError("DAgger frame compressor was not loaded")
                    episode["frames"] = dagger_compress_frames(episode["frames"])
                    dagger_episodes.append(episode)
                    dagger_collected = True
            wins += int(success)
            completed_trials += 1
            record: dict[str, Any] = {
                "task_id": int(global_task_index),
                "task": task_text,
                "env_name": env_name,
                "difficulty": MT50_BENCHMARK_TASK_TO_GROUP.get(
                    canonical_mt50_benchmark_env(env_name)
                ),
                "trial": int(trial),
                "seed": int(episode_seed),
                "success": bool(success),
            }
            if args.debug_stage_metrics:
                record["stage"] = {
                    "near_object": float(stage_metrics["near_object"]),
                    "grasp_success": float(stage_metrics["grasp_success"]),
                    "grasp_reward": float(stage_metrics["grasp_reward"]),
                    "in_place_reward": float(stage_metrics["in_place_reward"]),
                    "min_obj_to_target": float(stage_metrics["obj_to_target"]),
                    "best_obj_step": int(stage_metrics["best_obj_step"]),
                }
            if args.record_action_chunks:
                record["action_chunks"] = recorded_chunks
            if args.peer_eval_trace:
                record["peer_eval_trace"] = peer_eval_trace
            if takeover_step is not None:
                record["dagger_takeover_step"] = int(takeover_step)
                record["dagger_collected"] = bool(dagger_collected)
            trial_records.append(record)
            print(
                f"trial task={global_task_index} trial={trial} seed={episode_seed} "
                f"success={int(success)}"
            )
            if args.debug_stage_metrics:
                print(
                    f"stage task={global_task_index} trial={trial} "
                    f"near={stage_metrics['near_object']:.3f} "
                    f"grasp={stage_metrics['grasp_success']:.3f} "
                    f"grasp_reward={stage_metrics['grasp_reward']:.3f} "
                    f"in_place={stage_metrics['in_place_reward']:.3f} "
                    f"min_obj_to_target={stage_metrics['obj_to_target']:.4f} "
                    f"best_step={stage_metrics['best_obj_step']}",
                    flush=True,
                )
        per_task[task_text[:40]] = wins
        print(f"task {task_text[:40]}: {wins}/{trials_in_shard}")
        if dagger_path is not None:
            payload = {
                "episodes": dagger_episodes,
                "task": env_name,
                "n_episodes": len(dagger_episodes),
                "normalization": dict(features["normalization"]),
                "metadata": {
                    "contract": "current_policy_dagger_v1",
                    "fps": int(control_hz),
                    "control_stride": 1,
                    "checkpoint": str(args.checkpoint.expanduser().resolve()),
                    "episode_seed_base": int(args.episode_seed_base),
                    "trial_range": [int(trial_start), int(trial_stop)],
                    "takeover_range": [
                        int(args.dagger_takeover_min),
                        int(args.dagger_takeover_max),
                    ],
                    "prefix_keep": int(args.dagger_prefix_keep),
                    "post_success_steps": DAGGER_POST_SUCCESS_STEPS,
                    "supervision_contract": (
                        "current-policy prefix invalid; scripted-expert suffix valid "
                        "through first success"
                    ),
                },
            }
            temporary = dagger_path.with_suffix(dagger_path.suffix + ".tmp")
            if temporary.exists():
                raise FileExistsError(f"stale DAgger temporary output: {temporary}")
            torch.save(payload, temporary)
            temporary.replace(dagger_path)
            print(
                f"DAgger saved: {dagger_path} ({len(dagger_episodes)} episodes)",
                flush=True,
            )
        env.close()

    if args.task35_precision_contract:
        expected_task_text = selected_tasks[0][1]
        if len(selected_tasks) != 1 or set(per_task) != {expected_task_text[:40]}:
            raise RuntimeError(
                "task35 precision evaluation did not complete exactly one selected task"
            )
        if completed_trials != 50:
            raise RuntimeError(
                f"task35 precision evaluation completed {completed_trials} trials, expected 50"
            )
        if [int(row["seed"]) for row in trial_records] != list(TASK35_EVAL50_SEEDS):
            raise RuntimeError("task35 precision evaluation did not use seeds 35000-35049")
        env_name = descriptions_to_env.get(expected_task_text)
        if env_name != "peg-insert-side-v3":
            raise RuntimeError(
                "task35 precision evaluation mapped to "
                f"{env_name!r}, expected peg-insert-side-v3"
            )
    total = sum(per_task.values())
    trials = completed_trials
    if trials <= 0:
        raise RuntimeError("closed-loop evaluation completed no trials")
    print(f"\nCLOSED-LOOP SUCCESS: {total}/{trials} = {total / trials:.1%}")
    ci_kind = None
    ci_estimate = None
    ci_low = None
    ci_high = None
    if per_task:
        scores = np.asarray([w / trials_in_shard for w in per_task.values()])
        if len(scores) == 1:
            # 单任务无法按 task bootstrap：唯一 task 被重复抽样只会产生 [p,p]
            # 的伪窄区间。此时不确定性单位应是独立 trial，使用 Wilson 区间。
            from va_compound.statistics import binomial_wilson_ci

            est, lo, hi = binomial_wilson_ci(total, trials)
            ci_kind, ci_estimate, ci_low, ci_high = "wilson", est, lo, hi
            print(
                f"single-task success: {est:.1%} "
                f"[95% Wilson CI: {lo:.1%}, {hi:.1%}] (n_trials={trials})"
            )
        else:
            # 多任务宏平均：以 task 为有放回重采样单元（固定种子）。
            from va_compound.statistics import macro_bootstrap_ci

            group_ids = np.arange(len(scores))
            est, lo, hi = macro_bootstrap_ci(
                scores, group_ids, n_boot=2000, seed=0
            )
            ci_kind, ci_estimate, ci_low, ci_high = "task_bootstrap", est, lo, hi
            print(
                f"macro (per-task avg): {est:.1%} "
                f"[95% task-bootstrap CI: {lo:.1%}, {hi:.1%}] "
                f"(n_tasks={len(scores)})"
            )
    mt50_benchmark = summarize_mt50_benchmark_trials(trial_records)
    if mt50_benchmark["complete_mt50"]:
        print("MT50 DIFFICULTY BUCKETS:")
        for name, values in mt50_benchmark["groups"].items():
            print(
                f"  {name}: {values['success_rate']:.1%} "
                f"({values['successes']}/{values['trials']}, "
                f"n_tasks={values['n_tasks']})"
            )
        print(
            f"EVOMIND FOUR-TIER AVERAGE: "
            f"{mt50_benchmark['bucket_average']:.1%}"
        )
    if args.output_json is not None:
        import json

        result = {
            "contract": "metaworld_closed_loop_trials_v1",
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint.expanduser().absolute()),
            "features": str(args.features.expanduser().resolve()),
            "language_features": (
                None
                if args.language_features is None
                else str(args.language_features.expanduser().resolve())
            ),
            "task_ids": [int(index) for index, _ in selected_tasks],
            "trials_per_task": int(args.trials_per_task),
            "episode_seed_base": args.episode_seed_base,
            "seed_protocol": (
                "shared_base_plus_trial"
                if args.episode_seed_base is not None
                else "legacy_task_times_1000_plus_trial"
            ),
            "completed_trials": int(completed_trials),
            "successes": int(total),
            "success_rate": float(total / trials),
            "ci": {
                "kind": ci_kind,
                "estimate": None if ci_estimate is None else float(ci_estimate),
                "low_95": None if ci_low is None else float(ci_low),
                "high_95": None if ci_high is None else float(ci_high),
            },
            "mt50_benchmark": mt50_benchmark,
            "execute_steps": int(args.execution_horizon),
            "prediction_horizon": int(config.action_horizon),
            "execution_horizon": int(args.execution_horizon),
            "checkpoint_deployment_execution_horizon": int(
                checkpoint_deployment_horizon
            ),
            "execution_horizon_ablation": bool(
                args.allow_execution_horizon_ablation
            ),
            "world_horizon": (
                int(model.wmrm.cycle_steps)
                if getattr(model, "wmrm", None) is not None
                else None
            ),
            "flow_solver_steps": int(flow_steps),
            "source_fps": int(control_hz),
            "control_hz": int(control_hz),
            "training_control_stride": int(training_control_stride),
            "control_stride": int(training_control_stride),
            "planning_stride": int(checkpoint_planning_stride),
            "planning_hz": float(deployment_planning_hz),
            "training_planning_hz": float(training_planning_hz),
            "deployment_planning_hz": float(deployment_planning_hz),
            "world_transition_stride": int(training_control_stride),
            "observation_stride": int(OBSERVATION_STRIDE),
            "memory_reset_every": int(args.memory_reset_every),
            "wmrm_mode": (
                "ablated"
                if args.peer_world_off
                else "state-exchange"
                if getattr(model, "wmrm", None) is not None
                else "disabled"
            ),
            "peer_world_off": bool(args.peer_world_off),
            "horizon": int(args.horizon),
            "flow_samples": int(args.flow_samples),
            "action_decoder": (
                "c2_controller"
                if config.c2_controller
                else (
                    "direct_head"
                    if config.direct_head
                    else "conditional_flow_matching"
                )
            ),
            "wmrm_state_exchange": (
                getattr(model, "wmrm", None) is not None
                and not args.peer_world_off
            ),
            "task35_precision_contract": bool(args.task35_precision_contract),
            "task35_causal_ablation": args.task35_causal_ablation,
            "dino_feature_cache": (
                None
                if args.dino_feature_cache is None
                else str(args.dino_feature_cache.expanduser().resolve())
            ),
            "env_name": descriptions_to_env.get(selected_tasks[0][1])
            if len(selected_tasks) == 1
            else None,
            "language_source": (
                (
                    "task35_features_cache"
                    if task_indices == [35]
                    else "feature_language_cache"
                )
                if can_use_feature_language
                else "qwen_text_backbone"
            ),
            "trials": trial_records,
        }
        if args.task35_precision_contract or args.task35_causal_ablation != "none":
            validate_task35_eval50_payload(result)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output_json)
        print(f"structured results saved: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
