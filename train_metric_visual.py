"""MT-VJ 阶段 V：控制度量视觉预训练（契约 §4，2026-08-10）。

冻结 V-JEPA 2.1（fp16）+ 冻结 Qwen（语言缓存只算一次），只训
LanguageMetricField（2.2M 参数，Adam lr=1e-3）。仿真器自动生成随机观测
（prepare_metaworld_metric.make_metric_batch），真值：角色像素位置 / 可见度 /
关系状态 / 接触。

loss = CE(heatmap, Gaussian 标签(σ=2px)) + Huber(p̂, p*) + 1.0·Huber(ĝ, g*)
       + BCE(visibility)
位置类损失按可见度掩码（不可见角色不监督位置，只监督可见度）；g* 关系用世界
坐标（米）[‖eef−obj‖, ‖obj−target‖, axis_alignment, depth]。

每 1000 步打印 train RMSE（px，可见角色，图像坐标 ×384）。checkpoint 契约：
{"config": {...}, "metric_head": state_dict, "relation_encoder": state_dict,
 "optimizer": ..., RNG states..., "contract": "mt_vj_metric_field_v1"}。
loc-only 时 relation encoder 不训练且在 config 中诚实标记；非 loc-only
时则与 metric head 联合训练。

用法：
    python train_metric_visual.py --steps 20000 --batch-size 8
    python train_metric_visual.py --steps 5 --batch-size 2 --device cpu --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_metaworld_metric import (
    SAMPLE_RNG_CONTRACT,
    SUPPORTED_TASKS,
    make_metric_batch,
)
from scripts.build_longtraj_features import ENV_TO_TASK
from scripts.mt50_difficulty import DEFAULT_WEIGHT, TASK_WEIGHTS
from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone
from va_compound.live_vjepa import _dense_coords
from va_compound.metric_visual_head import (
    D_PROJ,
    H_DIM,
    HEATMAP_GRID,
    N_ROLES,
    LanguageMetricField,
    RelationStateEncoder,
)

IMAGE_SIZE = 384
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
HEATMAP_SIGMA_PX = 2.0  # Gaussian 标签 σ（像素）
RELATION_LAMBDA = 1.0
REL_RECON_LAMBDA = 0.1  # 拍板 3A（2026-08-10）：relation encoder z_g 重建辅助权重
ALIAS_CONSISTENCY_LAMBDA = 0.05
ALIAS_COORD_TOL = 1e-5
GEOMETRY_CONSISTENCY_LAMBDA = 1.0
METRIC_VISIBILITY_CONTRACT = "entity_aware_object_interface_v1"
METRIC_LOSS_CONTRACT = "hinge_pos_offset_geom_alias_vis_v1"
DEFAULT_TASKS = ",".join(SUPPORTED_TASKS)
CONTRACT = "mt_vj_metric_field_v1"
TRAINING_STATE_VERSION = 2


# These defaults are intentionally centralized: resume resolution and fresh-run
# parsing must use the same values.  Constructor/loss settings are immutable
# across resume; operational settings (tasks/target steps/batch/lr) may change.
_SEMANTIC_DEFAULTS: dict[str, Any] = {
    "task_sampling": "weighted",
    "l2_norm": False,
    "learnable_temp": None,  # fresh run: follows l2_norm
    "temp_init": 10.0,
    "no_bias": False,
    "sigma_px": 2.0,
    "loc_only": False,
    "offset_supervision": False,
    "grad_accum": 1,
    "mode_readout": False,
    "hinge_loss": False,
    "hinge_margin": 0.1,
}
_SEMANTIC_CONFIG_KEYS = {
    "task_sampling": "task_sampling",
    "l2_norm": "l2_norm",
    "learnable_temp": "learnable_temp",
    "temp_init": "temp_init",
    "no_bias": "freeze_bias",
    "sigma_px": "sigma_px",
    "loc_only": "loc_only",
    "offset_supervision": "offset_supervision",
    "grad_accum": "grad_accum",
    "mode_readout": "mode_readout",
    "hinge_loss": "hinge_loss",
    "hinge_margin": "hinge_margin",
}
# Historical v4 checkpoints recorded most constructor/loss flags but omitted
# loc_only/offset_supervision/grad_accum.  In general, any absent semantic field
# below must be supplied explicitly because it cannot be inferred from tensors.
_LEGACY_EXPLICIT_FIELDS = set(_SEMANTIC_CONFIG_KEYS) - {"learnable_temp"}


def _bool_flag(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    """BooleanOptionalAction with an unspecified (None) state for resume."""
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=None,
        **kwargs,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT-VJ 阶段 V：控制度量视觉预训练")
    parser.add_argument("--tasks", default=None,
                        help="逗号分隔的 metaworld v3 任务名")
    parser.add_argument(
        "--task-sampling",
        choices=("weighted", "balanced"),
        default=None,
        help="weighted 使用 easy/medium/hard/very-hard=0.5/1/2/3；"
        "balanced 仅用于等量对照",
    )
    parser.add_argument(
        "--task-weights-json",
        default=None,
        help="all-49 显式 hard-mining 权重 JSON（每任务整数 1..4）；"
        "权重与文件 SHA 写入 checkpoint，exact resume 不可漂移",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save", default="checkpoints/metric_field.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="训练前验证投影/标签（深度一致性 + 标注图）")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--allow-zero-language-smoke",
        action="store_true",
        help="仅冒烟：允许 Qwen 加载失败时回退随机 role query（正式训练必须 fail-fast，Codex P1-4）",
    )
    parser.add_argument(
        "--data-workers",
        type=int,
        default=4,
        help="阶段 V 数据生成并行 worker 数（make_metric_batch 每样本建 env+渲染，"
        "串行 ~4s/步不可行；4 worker 并行预取 → ~1s/步，2026-08-10）",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 续训（自愈用：mujoco 渲染 GL 上下文长期运行失效 → "
        "看护检测卡死后 --resume 重启，2026-08-10）",
    )
    source_group.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="新契约迁移：仅 strict 加载 metric/relation 权重；"
        "Adam、RNG、step 全部从零开始，必须显式指定新 --seed",
    )
    parser.add_argument(
        "--allow-legacy-optimizer-reset",
        action="store_true",
        help="仅旧 checkpoint 一次性迁移：明确接受 Adam/RNG 无法恢复。",
    )
    # ---- v2（2026-08-10，三方评审后）----
    _bool_flag(parser, "--l2-norm", help="query/d11 逐行 L2 归一化 → cosine 分数")
    _bool_flag(
        parser,
        "--learnable-temp",
        help="是否使用可学习温度（默认跟随 --l2-norm）",
    )
    bias_group = parser.add_mutually_exclusive_group()
    bias_group.add_argument(
        "--no-bias",
        dest="no_bias",
        action="store_true",
        help="冻结 spatial_bias（防边缘分布捷径，评审一致要求）",
    )
    bias_group.add_argument(
        "--train-bias",
        dest="no_bias",
        action="store_false",
        help="显式训练 spatial_bias（用于旧 checkpoint 迁移契约）",
    )
    parser.set_defaults(no_bias=None)
    parser.add_argument("--temp-init", type=float, default=None,
                        help="可学习温度初值（l2_norm 时分数 = temp·cos）")
    parser.add_argument("--sigma-px", type=float, default=None,
                        help="高斯标签 σ（像素）；v2 建议 ≥3-4（σ=2 在 patch 交叉点 "
                        "有 clamp 归一化伪影，target 只和 0.45）")
    _bool_flag(
        parser,
        "--loc-only",
        help="只训定位（CE+坐标），跳过 relation/vis/relation-encoder",
    )
    _bool_flag(
        parser,
        "--offset-supervision",
        help="直接监督 GT patch 的 offset：δ* = p* − p_center（SmoothL1）",
    )
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="梯度累积步数（batch 4 太小时建议 ≥8）")
    parser.add_argument("--fixed-data", type=str, default=None,
                        help="tiny-set 模式：固定数据集 .pt（make_metric_batch 输出的 dict），"
                        "特征一次性预计算，循环内只训 head（过拟合门）")
    _bool_flag(
        parser,
        "--mode-readout",
        help="v3 模式读出：NMS 全局峰 + 局部 5×5 soft-argmax + 峰 offset "
        "（探针实证：全网格期望读出在近乎全平的余弦面上 ≈ 均匀分布）",
    )
    _bool_flag(
        parser,
        "--hinge-loss",
        help="v4 max-margin 目标替代 CE（探针实证：CE 在平坦余弦面上 "
        "收敛到边缘分布，hinge 2000 步达 8.5px）",
    )
    parser.add_argument("--hinge-margin", type=float, default=None)
    return parser.parse_args(argv)


def _same_semantic(left: Any, right: Any) -> bool:
    """Compare checkpoint/CLI semantic scalars without bool/int confusion."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _legacy_flag_hint(field: str) -> str:
    hints = {
        "task_sampling": "--task-sampling weighted / balanced",
        "l2_norm": "--l2-norm / --no-l2-norm",
        "temp_init": "--temp-init VALUE",
        "no_bias": "--no-bias / --train-bias",
        "sigma_px": "--sigma-px VALUE",
        "loc_only": "--loc-only / --no-loc-only",
        "offset_supervision": "--offset-supervision / --no-offset-supervision",
        "grad_accum": "--grad-accum VALUE",
        "mode_readout": "--mode-readout / --no-mode-readout",
        "hinge_loss": "--hinge-loss / --no-hinge-loss",
        "hinge_margin": "--hinge-margin VALUE",
    }
    return hints[field]


def validate_initialization_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a weights-only migration source without accepting its train state."""
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError(
            f"init checkpoint contract={checkpoint.get('contract')!r}; expected {CONTRACT!r}"
        )
    missing = {"config", "metric_head", "relation_encoder"} - set(checkpoint)
    if missing:
        raise ValueError(f"init checkpoint missing required keys: {sorted(missing)}")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("init checkpoint config must be a mapping")
    for key in ("metric_head", "relation_encoder"):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"init checkpoint {key} must be a state_dict mapping")
    return config


def checkpoint_file_identity(
    path: str | os.PathLike[str], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Record immutable provenance for a weights-only initialization source."""
    checkpoint_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    config = checkpoint.get("config")
    source_steps = (
        int(config.get("steps_done", 0)) if isinstance(config, Mapping) else 0
    )
    return {
        "path": str(checkpoint_path),
        "sha256": digest.hexdigest(),
        "size_bytes": checkpoint_path.stat().st_size,
        "contract": checkpoint.get("contract"),
        "steps_done": source_steps,
        "sample_rng_contract": (
            config.get("sample_rng_contract") if isinstance(config, Mapping) else None
        ),
        "metric_visibility_contract": (
            config.get("metric_visibility_contract")
            if isinstance(config, Mapping)
            else None
        ),
        "metric_loss_contract": (
            config.get("metric_loss_contract") if isinstance(config, Mapping) else None
        ),
    }


def load_initialization_weights(
    metric_head: nn.Module,
    relation_encoder: nn.Module,
    checkpoint: Mapping[str, Any],
) -> None:
    """Strictly load model weights; optimizer/RNG/step are intentionally ignored."""
    validate_initialization_checkpoint(checkpoint)
    metric_head.load_state_dict(checkpoint["metric_head"], strict=True)
    relation_encoder.load_state_dict(checkpoint["relation_encoder"], strict=True)


def _validate_task_weights(payload: Mapping[str, Any]) -> dict[str, int]:
    expected = set(SUPPORTED_TASKS)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "task weights must contain exactly the canonical all-49 keys; "
            f"missing={missing}, extra={extra}"
        )
    weights: dict[str, int] = {}
    for task in SUPPORTED_TASKS:
        value = payload[task]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
            raise ValueError(
                f"task weight for {task!r} must be an integer in [1,4], got {value!r}"
            )
        weights[task] = int(value)
    return weights


def load_task_weights_json(
    path: str | os.PathLike[str],
) -> tuple[dict[str, int], str, str]:
    """Load canonical bounded weights and hash the exact JSON bytes."""
    weights_path = Path(path).expanduser().resolve()
    raw = weights_path.read_bytes()

    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate task weight key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid task weights JSON {weights_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("task weights JSON root must be an object")
    weights = _validate_task_weights(payload)
    return weights, hashlib.sha256(raw).hexdigest(), str(weights_path)


def _resolve_task_weights(
    args: argparse.Namespace,
    resume_config: Mapping[str, Any] | None,
) -> None:
    """Resolve explicit weights and make file content part of resume identity."""
    loaded = (
        load_task_weights_json(args.task_weights_json)
        if args.task_weights_json is not None
        else None
    )
    if resume_config is None:
        if loaded is None:
            args.task_weights = None
            args.task_weights_sha256 = None
            args.task_weights_source = None
        else:
            (
                args.task_weights,
                args.task_weights_sha256,
                args.task_weights_source,
            ) = loaded
    else:
        saved_payload = resume_config.get("task_weights")
        saved_sha = resume_config.get("task_weights_sha256")
        saved_source = resume_config.get("task_weights_source")
        if saved_payload is None:
            if loaded is not None:
                raise ValueError(
                    "resume semantic mismatch: checkpoint has no explicit task weights"
                )
            args.task_weights = None
            args.task_weights_sha256 = None
            args.task_weights_source = None
        else:
            if not isinstance(saved_payload, Mapping):
                raise ValueError("checkpoint task_weights must be a mapping")
            saved_weights = _validate_task_weights(saved_payload)
            if not isinstance(saved_sha, str) or len(saved_sha) != 64:
                raise ValueError("checkpoint with task_weights lacks a valid file SHA256")
            if loaded is not None and (loaded[0] != saved_weights or loaded[1] != saved_sha):
                raise ValueError(
                    "resume semantic mismatch for task weights or task-weights JSON SHA256"
                )
            args.task_weights = saved_weights
            args.task_weights_sha256 = saved_sha
            args.task_weights_source = loaded[2] if loaded is not None else saved_source

    if args.task_weights is not None:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        if tasks != list(SUPPORTED_TASKS):
            raise ValueError("--task-weights-json requires canonical all-49 task order")
        if args.task_sampling != "weighted":
            raise ValueError("explicit task weights require --task-sampling weighted")


def resolve_training_args(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None = None,
    init_checkpoint: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Resolve fresh/resume settings and forbid semantic drift.

    Tasks, target micro-steps, batch size and LR are operational overrides.  All
    LanguageMetricField constructor and loss/update settings are immutable.  A
    legacy checkpoint may be migrated only when every missing semantic is made
    explicit on the command line.
    """
    if checkpoint is not None and init_checkpoint is not None:
        raise ValueError("resume and init checkpoint are mutually exclusive")
    config: dict[str, Any] = {}
    if checkpoint is not None:
        if checkpoint.get("contract") != CONTRACT:
            raise ValueError(
                f"resume contract={checkpoint.get('contract')!r}; expected {CONTRACT!r}"
            )
        config = dict(checkpoint.get("config", {}))
    init_config: Mapping[str, Any] = {}
    if init_checkpoint is not None:
        init_config = validate_initialization_checkpoint(init_checkpoint)
    version = int(config.get("training_state_version", 0))
    is_legacy = checkpoint is not None and version < TRAINING_STATE_VERSION
    args._legacy_resume = is_legacy
    args.initialization_source = (
        config.get("initialization_source") if checkpoint is not None else None
    )

    # Operational values inherit from the checkpoint unless explicitly changed.
    checkpoint_tasks = config.get("tasks")
    if args.tasks is None:
        if checkpoint_tasks:
            args.tasks = ",".join(checkpoint_tasks) if isinstance(checkpoint_tasks, list) else str(checkpoint_tasks)
        else:
            args.tasks = DEFAULT_TASKS
    args.steps = int(args.steps if args.steps is not None else config.get("steps", 20000))
    args.batch_size = int(
        args.batch_size if args.batch_size is not None else config.get("batch_size", 8)
    )
    args.lr = float(args.lr if args.lr is not None else config.get("lr", 1e-3))
    args.fixed_data = args.fixed_data if args.fixed_data is not None else config.get("fixed_data")

    checkpoint_seed = config.get("seed")
    if init_checkpoint is not None:
        if args.seed is None:
            raise ValueError("--init-checkpoint requires an explicit new --seed")
        source_seed = init_config.get("seed")
        if source_seed is not None and int(args.seed) == int(source_seed):
            raise ValueError(
                f"--init-checkpoint requires a new seed; source seed is {source_seed}"
            )
        args.seed = int(args.seed)
    elif checkpoint is None:
        args.seed = int(0 if args.seed is None else args.seed)
    elif checkpoint_seed is None:
        # Historical runs used argparse's seed=0 default.  This cannot restore
        # the old stream, but the required legacy-reset gate below makes that
        # loss of exactness explicit.
        args.seed = int(0 if args.seed is None else args.seed)
    else:
        if args.seed is not None and int(args.seed) != int(checkpoint_seed):
            raise ValueError(
                f"--seed={args.seed} would change resume identity; checkpoint uses {checkpoint_seed}"
            )
        args.seed = int(checkpoint_seed)

    missing_legacy: list[str] = []
    for attr, key in _SEMANTIC_CONFIG_KEYS.items():
        cli_value = getattr(args, attr)
        if init_checkpoint is not None and attr != "task_sampling":
            if key not in init_config:
                raise ValueError(
                    f"init checkpoint config missing semantic field {key!r}"
                )
            init_value = init_config[key]
            if cli_value is not None and not _same_semantic(cli_value, init_value):
                raise ValueError(
                    f"init checkpoint semantic mismatch for {attr}: "
                    f"CLI={cli_value!r}, source={init_value!r}"
                )
            setattr(args, attr, init_value)
            continue
        if checkpoint is None:
            value = _SEMANTIC_DEFAULTS[attr] if cli_value is None else cli_value
            if attr == "learnable_temp" and value is None:
                value = bool(args.l2_norm)
            setattr(args, attr, value)
            continue

        if key in config:
            checkpoint_value = config[key]
        elif attr == "task_sampling" and is_legacy:
            # Every historical metric run used the hard-coded balanced cycle.
            checkpoint_value = "balanced"
        elif attr == "learnable_temp" and "l2_norm" in config:
            # All historical trainers tied learnable_temp to l2_norm.
            checkpoint_value = bool(config["l2_norm"])
        elif is_legacy and attr in _LEGACY_EXPLICIT_FIELDS:
            if cli_value is None:
                missing_legacy.append(attr)
                continue
            checkpoint_value = cli_value
        else:
            raise ValueError(
                f"checkpoint training_state_version={version} missing semantic field {key!r}"
            )

        if cli_value is not None and not _same_semantic(cli_value, checkpoint_value):
            raise ValueError(
                f"resume semantic mismatch for {attr}: CLI={cli_value!r}, "
                f"checkpoint={checkpoint_value!r}; start a new run to change it"
            )
        setattr(args, attr, checkpoint_value)

    if missing_legacy:
        hints = ", ".join(_legacy_flag_hint(field) for field in missing_legacy)
        raise ValueError(
            "legacy checkpoint omits training semantics that cannot be inferred: "
            f"{', '.join(missing_legacy)}. Re-run with explicit matching flags: {hints}"
        )
    if is_legacy and not getattr(args, "allow_legacy_optimizer_reset", False):
        raise ValueError(
            "legacy checkpoint has no Adam/RNG state. Re-run once with "
            "--allow-legacy-optimizer-reset to acknowledge a non-exact migration"
        )
    if checkpoint is not None:
        static_losses = {
            "relation_lambda": RELATION_LAMBDA,
            "relation_recon_lambda": REL_RECON_LAMBDA,
            "alias_consistency_lambda": ALIAS_CONSISTENCY_LAMBDA,
            "alias_coord_tolerance": ALIAS_COORD_TOL,
            "geometry_consistency_lambda": GEOMETRY_CONSISTENCY_LAMBDA,
            "sample_rng_contract": SAMPLE_RNG_CONTRACT,
            "metric_visibility_contract": METRIC_VISIBILITY_CONTRACT,
            "metric_loss_contract": METRIC_LOSS_CONTRACT,
        }
        for key, expected in static_losses.items():
            if key not in config:
                if not is_legacy:
                    raise ValueError(f"versioned checkpoint missing loss field {key!r}")
                continue  # historical value is known from the v4 trainer
            if not _same_semantic(config[key], expected):
                raise ValueError(
                    f"resume loss mismatch for {key}: checkpoint={config[key]!r}, "
                    f"current trainer={expected!r}"
                )
        if not is_legacy and bool(config.get("relation_encoder_trained")) != (
            not bool(args.loc_only)
        ):
            raise ValueError(
                "checkpoint relation_encoder_trained contradicts loc_only contract"
            )
    if init_checkpoint is not None and bool(
        init_config.get("relation_encoder_trained")
    ) != (not bool(args.loc_only)):
        raise ValueError(
            "init checkpoint relation_encoder_trained contradicts inherited loc_only contract"
        )

    _resolve_task_weights(args, config if checkpoint is not None else None)

    if args.steps < 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("--steps must be >=0; --batch-size and --lr must be >0")
    args.grad_accum = int(args.grad_accum)
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be >0")
    if args.steps % args.grad_accum != 0:
        raise ValueError(
            f"--steps ({args.steps}) must be divisible by --grad-accum ({args.grad_accum}); "
            "checkpoints are written only after complete optimizer updates"
        )
    if args.sigma_px <= 0 or args.hinge_margin < 0:
        raise ValueError("--sigma-px must be >0 and --hinge-margin must be >=0")
    return args


def build_training_state(
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    completed_steps: int,
    grad_accum: int,
) -> dict[str, Any]:
    """Exact-resume state captured only at an optimizer boundary."""
    if completed_steps % grad_accum != 0:
        raise ValueError("checkpoint attempted inside a gradient-accumulation window")
    return {
        "optimizer": optimizer.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "optimizer_steps_done": completed_steps // grad_accum,
    }


def build_checkpoint_config(
    args: argparse.Namespace,
    tasks: list[str],
    completed_steps: int,
    *,
    language_cache_available: bool,
    task_schedule_origin: int = 0,
) -> dict[str, Any]:
    """Serializable, complete constructor/loss/update contract."""
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "tasks": tasks,
        "task_schedule": (
            "step_derived_shuffled_explicit_weighted_cycles_v1"
            if args.task_weights is not None
            else (
                "step_derived_shuffled_difficulty_weighted_cycles_v1"
                if args.task_sampling == "weighted"
                else "step_derived_shuffled_balanced_cycles_v1"
            )
        ),
        "task_sampling": args.task_sampling,
        "task_weights": args.task_weights,
        "task_weights_sha256": args.task_weights_sha256,
        "task_weights_source": args.task_weights_source,
        "task_schedule_origin": int(task_schedule_origin),
        "initialization_source": args.initialization_source,
        "seed": args.seed,
        "steps": args.steps,
        "steps_done": completed_steps,
        "optimizer_steps_done": completed_steps // args.grad_accum,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "fixed_data": args.fixed_data,
        "h_dim": H_DIM,
        "d_proj": D_PROJ,
        "n_roles": N_ROLES,
        "lang_dim": 2048,
        "image_size": IMAGE_SIZE,
        "heatmap_sigma_px": args.sigma_px,
        "relation_lambda": RELATION_LAMBDA,
        "relation_recon_lambda": REL_RECON_LAMBDA,
        "alias_consistency_lambda": ALIAS_CONSISTENCY_LAMBDA,
        "alias_coord_tolerance": ALIAS_COORD_TOL,
        "geometry_consistency_lambda": GEOMETRY_CONSISTENCY_LAMBDA,
        "sample_rng_contract": SAMPLE_RNG_CONTRACT,
        "metric_visibility_contract": METRIC_VISIBILITY_CONTRACT,
        "metric_loss_contract": METRIC_LOSS_CONTRACT,
        "relation_encoder_trained": not args.loc_only,
        "state_dim": 6,
        "language_cache_available": language_cache_available,
        # LanguageMetricField constructor contract.
        "l2_norm": args.l2_norm,
        "learnable_temp": args.learnable_temp,
        "temp_init": args.temp_init,
        "freeze_bias": args.no_bias,
        "mode_readout": args.mode_readout,
        # Loss/update contract.
        "sigma_px": args.sigma_px,
        "loc_only": args.loc_only,
        "offset_supervision": args.offset_supervision,
        "grad_accum": args.grad_accum,
        "hinge_loss": args.hinge_loss,
        "hinge_margin": args.hinge_margin,
    }


def restore_training_state(
    checkpoint: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    *,
    completed_steps: int,
    grad_accum: int,
    requested_lr: float,
) -> None:
    """Restore Adam and RNG exactly for versioned checkpoints."""
    expected_updates = completed_steps // grad_accum
    if int(checkpoint.get("optimizer_steps_done", -1)) != expected_updates:
        raise ValueError(
            "checkpoint optimizer_steps_done does not match steps_done/grad_accum"
        )
    required = {"optimizer", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"versioned checkpoint missing exact-resume state: {missing}")
    optimizer.load_state_dict(checkpoint["optimizer"])
    # LR is an explicitly allowed operational override; Adam moments remain exact.
    for group in optimizer.param_groups:
        group["lr"] = float(requested_lr)
    rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_states = checkpoint["cuda_rng_state_all"]
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def balanced_task_for_step(tasks: Sequence[str], step: int, seed: int) -> str:
    """Deterministic shuffled cycles with exact per-task balance.

    Every contiguous cycle of ``len(tasks)`` micro-steps visits every task once.
    The cycle permutation is derived only from ``seed`` and the absolute step,
    so exact resume needs no hidden scheduler cursor.
    """
    if not tasks:
        raise ValueError("balanced task schedule requires at least one task")
    if step < 0:
        raise ValueError("task schedule step must be non-negative")
    cycle, offset = divmod(int(step), len(tasks))
    cycle_seed = np.random.SeedSequence((int(seed), cycle))
    order = np.random.default_rng(cycle_seed).permutation(len(tasks))
    return str(tasks[int(order[offset])])


def task_for_step(
    tasks: Sequence[str],
    step: int,
    seed: int,
    sampling: str = "weighted",
    task_weights: Mapping[str, int] | None = None,
) -> str:
    """Deterministic balanced or bounded difficulty-weighted task schedule.

    Weighted mode uses explicit bounded integer slots when supplied; otherwise
    it expands the existing 0.5/1/2/3 policy weights into 1/2/4/6 slots. Every
    task remains present. The schedule depends only on absolute step and seed,
    so exact resume has no hidden cursor.
    """
    if sampling == "balanced":
        if task_weights is not None:
            raise ValueError("balanced task schedule cannot use explicit weights")
        return balanced_task_for_step(tasks, step, seed)
    if sampling != "weighted":
        raise ValueError(f"unknown task sampling mode: {sampling!r}")
    if not tasks:
        raise ValueError("weighted task schedule requires at least one task")
    if step < 0:
        raise ValueError("task schedule step must be non-negative")

    slots: list[int] = []
    for index, task in enumerate(tasks):
        if task_weights is not None:
            if task not in task_weights:
                raise ValueError(f"explicit task weights missing {task!r}")
            repeats = task_weights[task]
            if (
                isinstance(repeats, bool)
                or not isinstance(repeats, int)
                or not 1 <= repeats <= 4
            ):
                raise ValueError(
                    f"explicit task weight for {task!r} must be integer 1..4"
                )
        else:
            task_text = ENV_TO_TASK.get(str(task), str(task))
            weight = float(TASK_WEIGHTS.get(task_text, DEFAULT_WEIGHT))
            repeats = int(round(2.0 * weight))
            if repeats <= 0 or not math.isclose(repeats / 2.0, weight):
                raise ValueError(f"unsupported difficulty weight {weight} for {task!r}")
        slots.extend([index] * repeats)
    cycle, offset = divmod(int(step), len(slots))
    cycle_seed = np.random.SeedSequence((int(seed), cycle, 1))
    order = np.random.default_rng(cycle_seed).permutation(len(slots))
    return str(tasks[slots[int(order[offset])]])


def preprocess_frames(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 [B, 4, 384, 384, 3] → 归一化 [B, 4, 3, 384, 384]（与 eval 一致）。"""
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 1, 4, 2, 3)
    tensor = tensor.float().div_(255.0)
    return (tensor - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)


def build_language_cache(
    text_backbone: QwenTextBackbone | None, texts: list[str]
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], bool]:
    """语言缓存只算一次：唯一任务文本 → (hidden [1, L, 2048] fp16, mask [1, L])。

    Qwen 不可用时回退：语言输入置零 + 打印警告（role query 保持随机初始化）。
    """
    unique = sorted(set(texts))
    cache = {}
    if text_backbone is None:
        print("WARNING: QwenTextBackbone 不可用——回退 role query 随机初始化，"
              "语言输入置零（契约 §4 允许的退化路径）")
        for text in unique:
            cache[text] = (torch.zeros(1, 1, 2048, dtype=torch.float16),
                           torch.ones(1, 1, dtype=torch.bool))
        return cache, False
    hidden, mask = text_backbone.encode(unique)  # [T, L, 2048] / [T, L]
    for i, text in enumerate(unique):
        cache[text] = (hidden[i : i + 1].cpu().to(torch.float16), mask[i : i + 1].cpu())
    return cache, True


def release_text_backbone(
    text_backbone: QwenTextBackbone | None, device: torch.device
) -> None:
    """Drop Qwen after its CPU cache is complete and return GPU memory to PyTorch."""
    del text_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return None


def gather_language(
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.cat([cache[t][0] for t in texts], dim=0).to(device)
    mask = torch.cat([cache[t][1] for t in texts], dim=0).to(device)
    return hidden, mask


def _gen_batch_worker(task: str, seed: int, n: int) -> dict:
    """阶段 V 数据生成 worker（模块顶层：ProcessPoolExecutor 需可 pickle）。"""
    import numpy as np
    from prepare_metaworld_metric import make_metric_batch
    return make_metric_batch(task, np.random.default_rng(seed), n)


def gaussian_targets(
    keypoints: torch.Tensor, sigma_px: float = HEATMAP_SIGMA_PX,
    grid: int = HEATMAP_GRID, image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """keypoints [B, R, 2]（0-1, y,x）→ [B, R, grid, grid] 高斯标签（每图归一化）。

    Codex P1-2（2026-08-10）：用 patch 中心 (i+0.5)/grid 对齐，避免 keypoint=1
    时中心落在网格外（旧式 keypoint*grid 在边界处 target 总和 ~1.6e-22，CE 消失）。
    """
    sigma = sigma_px / (image_size / grid)  # 像素 → 网格单位
    yc = keypoints[..., 0:1] * grid - 0.5  # [B, R, 1]（patch 中心坐标系）
    xc = keypoints[..., 1:2] * grid - 0.5
    yy = torch.arange(grid, device=keypoints.device, dtype=keypoints.dtype)
    xx = torch.arange(grid, device=keypoints.device, dtype=keypoints.dtype)
    dist2 = (yy.view(1, 1, 1, grid) - yc.unsqueeze(-1)) ** 2 + (
        xx.view(1, 1, grid, 1) - xc.unsqueeze(-1)
    ) ** 2  # [B, R, grid, grid]
    target = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def _alias_consistency_loss(
    out,
    keypoints: torch.Tensor,
    tolerance: float = ALIAS_COORD_TOL,
) -> tuple[torch.Tensor, int]:
    """Keep object/interface predictions equal only when their GT point aliases."""
    if tolerance < 0:
        raise ValueError("alias coordinate tolerance must be non-negative")
    alias = (
        (keypoints[:, 1] - keypoints[:, 3]).abs().amax(dim=-1) <= tolerance
    ).to(keypoints.dtype)
    count = int(alias.sum().detach())
    denominator = alias.sum().clamp_min(1.0)
    position = F.smooth_l1_loss(
        out.p[:, 1], out.p[:, 3], reduction="none"
    ).mean(dim=-1)
    visibility = F.smooth_l1_loss(
        out.visibility_logits[:, 1],
        out.visibility_logits[:, 3],
        reduction="none",
    )
    scores = getattr(out, "scores", None)
    if scores is None:
        scores = out.log_heatmap.flatten(start_dim=2)
    score = F.smooth_l1_loss(
        scores[:, 1], scores[:, 3], reduction="none"
    ).mean(dim=-1)
    loss = ((position + visibility + score) * alias).sum() / denominator
    return loss, count


def _pair_geometry_consistency_loss(
    out,
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Match the two generic role-pair displacement vectors when observable."""
    predicted = torch.stack(
        (out.p[:, 0] - out.p[:, 1], out.p[:, 3] - out.p[:, 2]), dim=1
    )
    target = torch.stack(
        (
            keypoints[:, 0] - keypoints[:, 1],
            keypoints[:, 3] - keypoints[:, 2],
        ),
        dim=1,
    )
    pair_visible = torch.stack(
        (visibility[:, 0] * visibility[:, 1], visibility[:, 3] * visibility[:, 2]),
        dim=1,
    )
    error = F.smooth_l1_loss(predicted, target, reduction="none").sum(dim=-1)
    loss = (error * pair_visible).sum() / pair_visible.sum().clamp_min(1.0)
    return loss, int(pair_visible.sum().detach())


def compute_losses(out, keypoints, visibility, relation, sigma_px: float = HEATMAP_SIGMA_PX,
                   loc_only: bool = False, offset_supervision: bool = False,
                   hinge: bool = False, hinge_margin: float = 0.1,
                   alias_consistency_weight: float = ALIAS_CONSISTENCY_LAMBDA,
                   alias_coord_tolerance: float = ALIAS_COORD_TOL,
                   geometry_consistency_weight: float = GEOMETRY_CONSISTENCY_LAMBDA,
                   image_size: int = IMAGE_SIZE,
                   ) -> tuple[torch.Tensor, dict]:
    """v4（2026-08-10，探针实证）：``hinge`` 用 max-margin 目标替代 CE——
    max(s_GT) 必须超过 max(其余 patch) 至少 ``hinge_margin``。CE 在近乎全平的
    余弦面上梯度 ≈ mean(全图特征) − f_target，被静态背景主导 → 收敛到边缘
    分布（探针：三种读出 × 两种初始化全部 49-58px）；hinge 梯度 = −f_target +
    f_best_other 逐样本相干，2000 步即达 8.5px（scripts/diag_trained_linear_probe.py
    与 hinge 探针实证）。其余组件同前：CE/Huber 位置/offset 按可见度掩码。"""
    device = keypoints.device
    if alias_consistency_weight < 0 or not math.isfinite(alias_consistency_weight):
        raise ValueError("alias_consistency_weight must be finite and non-negative")
    if geometry_consistency_weight < 0 or not math.isfinite(
        geometry_consistency_weight
    ):
        raise ValueError("geometry_consistency_weight must be finite and non-negative")
    vis = visibility  # [B, R] float
    n_vis = vis.sum().clamp_min(1.0)
    # DINO-metric（2026-08-16）：网格从输出推导（V-JEPA 24×24，DINO
    # 16×16），不再写死 HEATMAP_GRID。旧训练/ROI 测试夹具可能只带
    # log_heatmap，hinge-only 夹具则只带 2*grid^2 scores；三者语义等价。
    heatmap = getattr(out, "heatmap", None)
    log_heatmap = getattr(out, "log_heatmap", None)
    scores = getattr(out, "scores", None)
    if heatmap is not None:
        grid = int(heatmap.shape[-1])
        grid_shape = tuple(heatmap.shape)
        square = heatmap.shape[-2] == grid
    elif log_heatmap is not None:
        grid = int(log_heatmap.shape[-1])
        grid_shape = tuple(log_heatmap.shape)
        square = log_heatmap.shape[-2] == grid
    elif scores is not None:
        per_frame = int(scores.shape[-1]) // 2
        grid = math.isqrt(per_frame)
        grid_shape = tuple(scores.shape)
        square = scores.shape[-1] == 2 * grid * grid
    else:
        raise ValueError("metric output must provide heatmap, log_heatmap, or scores")
    if grid < 2 or not square:
        raise ValueError(
            "metric output must encode a square spatial grid, "
            f"got shape={grid_shape} inferred_grid={grid}"
        )
    yi = torch.clamp(torch.floor(keypoints[..., 0] * grid).long(), 0, grid - 1)
    xi = torch.clamp(torch.floor(keypoints[..., 1] * grid).long(), 0, grid - 1)
    idx = yi * grid + xi  # [B, R]（片内位置；两片坐标相同）

    parts: dict[str, float] = {}
    if hinge:
        s = out.scores  # [B, R, 1152]
        s_gt = torch.maximum(
            s.gather(-1, idx.unsqueeze(-1)).squeeze(-1),
            s.gather(-1, (idx + grid * grid).unsqueeze(-1)).squeeze(-1),
        )  # [B, R]
        mask_excl = torch.ones_like(s, dtype=torch.bool)
        mask_excl.scatter_(-1, idx.unsqueeze(-1), False)
        mask_excl.scatter_(-1, (idx + grid * grid).unsqueeze(-1), False)
        s_other = s.masked_fill(~mask_excl, -1e9).max(dim=-1).values  # [B, R]
        h = F.relu(hinge_margin - (s_gt - s_other))
        loss_hinge = (h * vis).sum() / n_vis
        parts["hinge"] = loss_hinge.item()
        loss_cls = loss_hinge
    else:
        targets = gaussian_targets(
            keypoints, sigma_px=sigma_px, grid=grid, image_size=image_size
        )
        ce_per = -(targets * out.log_heatmap).sum(dim=(-2, -1))  # [B, R]
        loss_cls = (ce_per * vis).sum() / n_vis
        parts["ce"] = loss_cls.item()

    # Huber 位置（归一化图像坐标 → 像素乘 384 在 RMSE 中体现）
    pos_per = F.smooth_l1_loss(out.p, keypoints, reduction="none").sum(dim=-1)  # [B, R]
    loss_pos = (pos_per * vis).sum() / n_vis
    parts["pos"] = loss_pos.item()

    # v2：GT patch 的直接 offset 监督（δ* = p* − p_center，归一化坐标）
    loss_offset = torch.zeros((), device=device)
    if offset_supervision:
        gt_center = torch.stack(((yi + 0.5) / grid, (xi + 0.5) / grid), dim=-1)
        delta_star = keypoints - gt_center  # [B, R, 2]
        off = out.offset_full[:, :, : grid * grid]  # [B, R, 576, 2]（t=0 片）
        idx4 = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        off_gt = off.gather(dim=2, index=idx4).squeeze(2)  # [B, R, 2]
        loss_offset = (
            F.smooth_l1_loss(off_gt, delta_star, reduction="none").sum(dim=-1) * vis
        ).sum() / n_vis
        parts["offset"] = loss_offset.item()

    if not loc_only:
        # 关系只在其视觉端点均可见时监督，避免让网络回归画面里不存在的
        # simulator-only target。前两维=tool/object；中两维=progress/target；
        # axis/depth 依赖两组端点。
        pair_tool_object = vis[:, 0] * vis[:, 1]
        pair_progress_target = vis[:, 3] * vis[:, 2]
        both_pairs = pair_tool_object * pair_progress_target
        relation_mask = torch.stack(
            (
                pair_tool_object,
                pair_tool_object,
                pair_progress_target,
                pair_progress_target,
                both_pairs,
                pair_progress_target,
            ),
            dim=-1,
        )
        relation_error = F.smooth_l1_loss(
            out.relation, relation, reduction="none"
        )
        loss_rel = (relation_error * relation_mask).sum() / relation_mask.sum().clamp_min(1.0)
        # BCE 可见度
        loss_vis = F.binary_cross_entropy_with_logits(
            out.visibility_logits, vis, reduction="mean"
        )
        total = (
            loss_cls
            + loss_pos
            + loss_offset
            + RELATION_LAMBDA * loss_rel
            + loss_vis
        )
        parts.update({"rel": loss_rel.item(), "vis": loss_vis.item()})
    else:
        total = loss_cls + loss_pos + loss_offset
    geometry_loss, geometry_count = _pair_geometry_consistency_loss(
        out, keypoints, visibility
    )
    total = total + geometry_consistency_weight * geometry_loss
    alias_loss, alias_count = _alias_consistency_loss(
        out, keypoints, tolerance=alias_coord_tolerance
    )
    total = total + alias_consistency_weight * alias_loss
    parts.update(
        {
            "alias": alias_loss.item(),
            "alias_weighted": (alias_consistency_weight * alias_loss).item(),
            "alias_n": float(alias_count),
            "geom": geometry_loss.item(),
            "geom_weighted": (geometry_consistency_weight * geometry_loss).item(),
            "geom_n": float(geometry_count),
        }
    )
    return total, parts


def verify_labels(tasks: list[str], rng: np.random.Generator) -> None:
    """Render one real sample per task and validate the public label contract."""
    from prepare_metaworld_metric import make_visualization

    role_visible = np.zeros(4, dtype=np.int64)
    first_batch = None
    for task in tasks:
        batch = make_metric_batch(task, rng, 1)
        if not bool(batch["supported"][0]):
            raise RuntimeError(f"verify: {task} unexpectedly lacks metric supervision")
        for key in (
            "keypoints",
            "visibility",
            "surface_visible",
            "entity_visible",
            "in_frame",
            "relation",
        ):
            if not np.isfinite(np.asarray(batch[key])).all():
                raise RuntimeError(f"verify: {task} produced non-finite {key}")
        role_visible += np.asarray(batch["visibility"][0], dtype=np.int64)
        print(
            f"verify {task}: visible={batch['visibility'][0].astype(int).tolist()} "
            f"in_frame={batch['in_frame'][0].astype(int).tolist()}",
            flush=True,
        )
        if first_batch is None:
            first_batch = batch
    if first_batch is not None:
        make_visualization(first_batch, "/tmp/metric_batch_verify.png")
    print(
        f"verify: training-visible counts [tool,object,target,progress]="
        f"{role_visible.tolist()}/{len(tasks)}; /tmp/metric_batch_verify.png",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    resume_checkpoint = None
    init_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    if args.init_checkpoint:
        init_checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=True
        )
        validate_initialization_checkpoint(init_checkpoint)
    args = resolve_training_args(args, resume_checkpoint, init_checkpoint)
    if init_checkpoint is not None:
        args.initialization_source = checkpoint_file_identity(
            args.init_checkpoint, init_checkpoint
        )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--tasks 不能为空")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- V-JEPA 2.1（冻结，fp16） ----
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True,
    )
    vision_backbone.freeze_all()
    coords = torch.from_numpy(_dense_coords()).to(device)  # [1152, 3]（[-1,1]，head 内部归一化）

    # ---- Qwen 语言缓存（冻结，只算一次；失败则退化） ----
    text_backbone = None
    if device.type == "cuda":
        lang_dtype, lang_device = "float16", device
    else:
        lang_dtype, lang_device = "float32", device
    try:
        text_backbone = QwenTextBackbone.from_pretrained(
            device=lang_device, dtype=lang_dtype, local_files_only=True,
        )
        print(f"train: QwenTextBackbone 就绪（{lang_dtype} on {lang_device}）")
    except Exception as exc:  # noqa: BLE001
        # Codex P1-4（2026-08-10）：正式训练 fail-fast——静默回退会训出
        # 无语言角色的 checkpoint（20k 步后才暴露）。仅显式 smoke 允许退化。
        if not args.allow_zero_language_smoke:
            raise RuntimeError(
                f"Qwen 加载失败且未指定 --allow-zero-language-smoke（正式训练 fail-fast）：{exc!r}"
            ) from exc
        print(f"WARNING: Qwen 加载失败（{exc!r}），回退 role query 随机初始化（仅 smoke）")
    language_cache, language_cache_available = build_language_cache(
        text_backbone, [ENV_TO_TASK[t] for t in tasks]
    )
    if args.fixed_data is None:
        text_backbone = release_text_backbone(text_backbone, device)

    # ---- 模型（v2：loc_only 只训 metric head；relation encoder 恒创建以保契约） ----
    metric_head = LanguageMetricField(
        l2_norm=args.l2_norm,
        learnable_temp=args.learnable_temp,
        temp_init=args.temp_init,
        freeze_bias=args.no_bias,
        mode_readout=args.mode_readout,
    ).to(device)
    # 拍板 3A（2026-08-10）：阶段 V 一起训练 RelationStateEncoder——metric tokens
    # 应为监督学习的关系编码（Codex P1-3），而非随机线性映射。loc_only 时仅
    # 随机初始化保存（checkpoint 契约 §2 要求该键），不进优化器。
    relation_encoder = RelationStateEncoder(state_dim=6).to(device)
    optimizer_params = list(metric_head.parameters())
    if args.loc_only:
        relation_encoder.eval()
        print("train: loc-only 模式——relation/vis 损失跳过，relation encoder 仅随机保存", flush=True)
    else:
        relation_encoder.train()
        optimizer_params += list(relation_encoder.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    n_params = sum(p.numel() for p in metric_head.parameters())
    print(f"train: device={device} tasks={tasks} steps={args.steps} "
          f"batch_size={args.batch_size} lr={args.lr} metric_head_params={n_params / 1e6:.2f}M")

    if args.verify:
        verify_labels(tasks, rng)

    # ---- 训练（2026-08-10 最终版：单进程串行） ----
    # 多进程数据生成被证实不可用：mujoco offscreen 渲染在 worker 进程
    # （fork 或 spawn）中崩溃/退化（BrokenProcessPool + GL 上下文问题），
    # 反复出现"卡住-变慢-消失"。回到单进程串行：每步 = 生成 ~7s + 训练 0.5s，
    # 因此步数由 20k 降到 5k（数据无限生成，metric head 2M 参数 4 万样本足够）。
    rmse_sum = 0.0
    rmse_count = 0
    start_step = 0
    if init_checkpoint is not None:
        load_initialization_weights(metric_head, relation_encoder, init_checkpoint)
        print(
            "train: strict weights-only initialization from "
            f"{args.init_checkpoint} (fresh Adam/RNG, steps_done=0, seed={args.seed})",
            flush=True,
        )
    elif resume_checkpoint is not None:
        # Constructor/loss semantics were resolved before model construction;
        # state loading is therefore strict.  No randomly initialized parameter
        # may slip into a resumed run.
        metric_head.load_state_dict(resume_checkpoint["metric_head"], strict=True)
        relation_encoder.load_state_dict(
            resume_checkpoint["relation_encoder"], strict=True
        )
        start_step = int(resume_checkpoint.get("config", {}).get("steps_done", 0))
        if start_step % args.grad_accum != 0:
            raise ValueError(
                f"checkpoint steps_done={start_step} is not an optimizer boundary "
                f"for grad_accum={args.grad_accum}"
            )
        if start_step > args.steps:
            raise ValueError(
                f"checkpoint steps_done={start_step} exceeds target --steps={args.steps}"
            )
        if args._legacy_resume:
            print(
                "WARNING: legacy metric checkpoint migration: weights restored strictly; "
                "Adam/RNG restart was explicitly acknowledged.",
                flush=True,
            )
        print(
            f"train: resume from {args.resume}（steps_done={start_step}, "
            f"optimizer_updates={start_step // args.grad_accum}）",
            flush=True,
        )
    if resume_checkpoint is None:
        task_schedule_origin = 0
    elif args._legacy_resume:
        # The legacy run used random task draws. Start the new balanced schedule
        # at the migration boundary so exactly 49*k added steps yield k/task.
        task_schedule_origin = start_step
    else:
        task_schedule_origin = int(
            resume_checkpoint.get("config", {}).get("task_schedule_origin", 0)
        )

    # 2026-08-10 修复：checkpoint_payload 必须在训练循环前定义——周期保存
    # （循环内调用）此前因定义在循环后被 UnboundLocalError 崩掉（丢进度）。
    def checkpoint_payload(completed_steps: int) -> dict:
        payload = {
            "config": build_checkpoint_config(
                args,
                tasks,
                completed_steps,
                language_cache_available=language_cache_available,
                task_schedule_origin=task_schedule_origin,
            ),
            "metric_head": metric_head.state_dict(),
            "relation_encoder": relation_encoder.state_dict(),
            "contract": CONTRACT,
        }
        payload.update(
            build_training_state(
                optimizer, rng, completed_steps, args.grad_accum
            )
        )
        return payload

    # ---- 数据源（v2）：tiny 固定集（特征一次性预计算，head-only）或仿真流 ----
    fixed = None
    if args.fixed_data:
        fixed = torch.load(args.fixed_data, map_location="cpu", weights_only=False)
        n_fixed = len(fixed["frames"])
        video = preprocess_frames(np.asarray(fixed["frames"]), device)
        with torch.no_grad():
            h5_f, h11_f = vision_backbone.encode_multi(video, out_layers=(5, 11))
        del video
        lang_cache_f, _ = build_language_cache(
            text_backbone, [str(t) for t in fixed["language_text"]]
        )
        text_backbone = release_text_backbone(text_backbone, device)
        kp_f = torch.from_numpy(np.asarray(fixed["keypoints"])).to(device)
        vis_f = torch.from_numpy(np.asarray(fixed["visibility"])).to(device)
        rel_f = torch.from_numpy(np.asarray(fixed["relation"])).to(device)
        print(f"train: tiny 固定集 {n_fixed} 样本，特征预计算完成（head-only）", flush=True)

    # Restore RNG after every one-time setup operation (including fixed-data
    # feature precomputation), otherwise an exact resume would consume random
    # numbers that the uninterrupted run did not consume at this point.
    if resume_checkpoint is not None and not args._legacy_resume:
        restore_training_state(
            resume_checkpoint,
            optimizer,
            rng,
            completed_steps=start_step,
            grad_accum=args.grad_accum,
            requested_lr=args.lr,
        )
    optimizer.zero_grad(set_to_none=True)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    last_checkpoint_step = start_step
    for step in range(start_step, args.steps):
        if fixed is not None:
            idx = torch.randperm(n_fixed, device=device)[: args.batch_size]
            h5, h11 = h5_f[idx], h11_f[idx]
            texts = [str(fixed["language_text"][int(i)]) for i in idx.cpu()]
            lang_hidden, lang_mask = gather_language(lang_cache_f, texts, device)
            keypoints, visibility, relation = kp_f[idx], vis_f[idx], rel_f[idx]
        else:
            task = task_for_step(
                tasks,
                step - task_schedule_origin,
                args.seed,
                args.task_sampling,
                args.task_weights,
            )
            batch = make_metric_batch(task, rng, args.batch_size)
            video = preprocess_frames(batch["frames"], device)  # [B,4,3,384,384]
            with torch.no_grad():
                h5, h11 = vision_backbone.encode_multi(video, out_layers=(5, 11))
                lang_hidden, lang_mask = gather_language(
                    language_cache, batch["language_text"], device
                )
            keypoints = torch.from_numpy(batch["keypoints"]).to(device)
            visibility = torch.from_numpy(batch["visibility"]).to(device)
            relation = torch.from_numpy(batch["relation"]).to(device)

        out = metric_head(h5, h11, lang_hidden, lang_mask, coords)
        loss, parts = compute_losses(
            out, keypoints, visibility, relation,
            sigma_px=args.sigma_px, loc_only=args.loc_only,
            offset_supervision=args.offset_supervision,
            hinge=args.hinge_loss, hinge_margin=args.hinge_margin,
        )
        if not args.loc_only and relation_encoder is not None:
            # 拍板 3A（2026-08-10）：relation encoder 重建监督——z_g 须保留 g_t 信息
            # （Codex P1-3；ν 分支无历史依赖，留阶段 A 监督）。
            g_true = relation
            nu_zero = torch.zeros_like(g_true)
            z_g, _ = relation_encoder(g_true, nu_zero)
            g_recon = relation_encoder.recon(z_g)
            loss = loss + REL_RECON_LAMBDA * F.mse_loss(g_recon, g_true)
        loss = loss / max(args.grad_accum, 1)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss 非有限值 @ step {step}: {loss.item()}")

        loss.backward()
        optimizer_boundary = (step + 1) % args.grad_accum == 0
        if optimizer_boundary:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # train RMSE（px）：可见角色的位置误差 ×384
        err_px = (out.p.detach() - keypoints) * IMAGE_SIZE  # [B, R, 2]
        sq = (err_px ** 2).sum(dim=-1) * visibility  # [B, R]
        rmse_sum += float(sq.sum())
        rmse_count += int(visibility.sum())

        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            rmse = math.sqrt(rmse_sum / max(rmse_count, 1))
            parts_str = " ".join(f"{k} {v:.4f}" for k, v in parts.items())
            temp_str = ""
            if args.l2_norm and hasattr(metric_head, "temperature"):
                temp_str = f" temp={float(metric_head.temperature.detach()):.2f}"
            print(
                f"step {step + 1}/{args.steps} loss {loss.item():.4f} "
                f"({parts_str}){temp_str} "
                f"train RMSE {rmse:.2f} px  vis_mean {float(visibility.mean()):.2f}",
                flush=True,
            )
        # 周期保存（2026-08-10）：nohup 重定向下 print 会被块缓冲（进度不可见），
        # 且中途崩溃会丢全部训练——每 500 步原子落盘一次（自愈 resume 依赖）。
        # Only an optimizer boundary is recoverable because partial accumulated
        # gradients are deliberately not serialized.  Save at the first boundary
        # at least 500 micro-steps after the previous checkpoint.
        if optimizer_boundary and (step + 1 - last_checkpoint_step) >= 500:
            torch.save(
                checkpoint_payload(step + 1),
                Path(args.save).with_suffix(".pt.tmp"),
            )
            Path(args.save).with_suffix(".pt.tmp").replace(args.save)
            last_checkpoint_step = step + 1
            print(f"  checkpoint @ step {step + 1} → {args.save}", flush=True)

    # ---- 保存 checkpoint（契约 §2/§4；checkpoint_payload 已定义在循环前） ----
    checkpoint = checkpoint_payload(args.steps)
    torch.save(checkpoint, args.save)
    print(f"train: checkpoint saved -> {args.save} ({os.path.getsize(args.save) / 2**20:.1f} MiB)")

    # 验证 weights_only=True 可加载
    loaded = torch.load(args.save, map_location="cpu", weights_only=True)
    assert loaded["contract"] == CONTRACT
    assert loaded["config"]["training_state_version"] == TRAINING_STATE_VERSION
    assert {
        "config",
        "metric_head",
        "relation_encoder",
        "contract",
        "optimizer",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "optimizer_steps_done",
    } <= set(loaded)
    print("train: checkpoint 验证通过（weights_only=True 可加载）")


if __name__ == "__main__":
    main()
