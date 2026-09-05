"""Metric evaluation helpers retained for standalone diagnostic tools."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from va_compound.metric_roi import (
    ASSEMBLY_METRIC_ROLE_CONTRACT, DINO_METRIC_ROI_CONTRACT,
    TASK35_METRIC_ROLE_CONTRACT, METRIC_ROI_CONTRACT_VERSION,
    load_dino_metric_roi_checkpoint, metric_head_state_sha256,
    prepare_metric_roi_video, refine_metric_roi_positions_dino,
)
from va_compound.utils.exact_resume import _sha256_file

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

def _mtvj_metric_positions(out, source: str = MTVJ_METRIC_STATE_SOURCE) -> Tensor:
    """Return the declared 8-D state; v2 stays reproducible, v3 gates visibility."""
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

def _canonical_mtvj_metric_head_config(
    config: dict | None,
    *,
    require_complete: bool = False,
) -> dict:
    """Return the complete, weights-only-safe LanguageMetricField ctor contract.

    ``grid``（2026-08-15 新增）是向后兼容的缺省键：旧 checkpoint 无此键时
    填默认 24（V-JEPA），require_complete 不要求其存在。
    """
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

def _build_dino_metric_stack(
    device: torch.device,
    config: VACompoundConfig,
    *,
    train_metric_head: bool,
    train_relation: bool,
    saved_ctor_config: dict | None = None,
) -> tuple[nn.Module, nn.Module]:
    """DINO-metric（2026-08-15）：从零构建 LanguageMetricField + RelationStateEncoder。

    V-JEPA 的 metric head（768 维 / 1152 token / 24×24 网格）与 DINO 不兼容，
    不复用其权重；此处按 DINO 特征从零构建：h_dim=1024、grid=16（原生
    224px/14 patch，不插值）、两帧 512 token。v3 模式读出（V-JEPA 探针实证的
    更优读出）。训练语义与 V-JEPA 联合微调同构：rel_mlp 仅辅助不动、
    temperature（非 l2_norm 时）冻结、recon 冻结。
    """
    from va_compound.metric_visual_head import (
        LanguageMetricField,
        RelationStateEncoder,
    )

    ctor_config = _canonical_mtvj_metric_head_config(saved_ctor_config or {})
    if saved_ctor_config is not None:
        # 续训：严格使用主 checkpoint 的构造配置（否则权重形状不匹配）。
        if int(ctor_config["h_dim"]) != int(config.main_vision_dim) or int(
            ctor_config["grid"]
        ) != 16:
            raise ValueError(
                "DINO-metric resume 的 mtvj_metric_head_config 与 DINO 特征不兼容："
                f"h_dim={ctor_config['h_dim']} (期望 {config.main_vision_dim}), "
                f"grid={ctor_config['grid']} (期望 16)"
            )
    else:
        # 与生产 V-JEPA metric 栈（metric_field_v6，mode_readout/l2_norm/
        # learnable_temp）同一读出语义，只是视觉输入换 DINO（1024 维、16 网格）。
        ctor_config.update(
            h_dim=int(config.main_vision_dim),
            grid=16,
            l2_norm=True,
            learnable_temp=True,
            mode_readout=True,
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
    relation_encoder = RelationStateEncoder(
        state_dim=8, d_model=int(config.hidden_dim)
    ).to(device)
    metric_head._mtvj_metric_state_source = MTVJ_METRIC_STATE_SOURCE
    metric_head._mtvj_metric_contract_version = MTVJ_METRIC_CONTRACT_VERSION
    metric_head._mtvj_metric_head_source = "dino-metric-from-scratch"
    # 无外部 metric checkpoint：来源指纹为 DINO 从零训练的合成标记（save
    # checkpoint 的指纹门控对 dino 来源豁免）。
    metric_head._mtvj_external_checkpoint_identity = {
        "source": "dino-metric-from-scratch",
        "sha256": "none-dino-metric-from-scratch",
    }
    metric_head.train(train_metric_head)
    for name, parameter in metric_head.named_parameters():
        action_connected = not name.startswith("rel_mlp.")
        if name == "temperature" and not metric_head.l2_norm:
            action_connected = False
        if name == "spatial_bias" and metric_head.freeze_bias:
            action_connected = False
        parameter.requires_grad_(train_metric_head and action_connected)
    relation_encoder.train(train_relation)
    for name, parameter in relation_encoder.named_parameters():
        if name.startswith("recon."):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(train_relation)
    print(
        "dino-metric: LanguageMetricField 从零构建 "
        f"(h_dim={metric_head.h_dim}, grid={metric_head.grid}, "
        f"dense_tokens={metric_head.dense_tokens}, "
        f"mode_readout={metric_head.mode_readout}) + "
        f"RelationStateEncoder(state_dim=8, d_model={config.hidden_dim})；"
        f"train_metric_head={train_metric_head}, train_relation={train_relation}",
        flush=True,
    )
    return metric_head, relation_encoder

def _dino_metric_tokens(
    metric_head: nn.Module,
    relation_encoder: nn.Module,
    dense_evidence: dict[int, Tensor],
    batch: dict[str, Tensor],
    device: torch.device,
    *,
    train_metric_head: bool = False,
    roi_head: nn.Module | None = None,
    roi_backbone: nn.Module | None = None,
    roi_frames=None,
    roi_alpha: float = 0.0,
) -> tuple[Tensor | None, Tensor | None]:
    """DINO-metric dense evidence → ``(metric_tokens, metric_g)``。

    与 ``_mtvj_online_encode`` 同构（仅 token 数/网格不同）：language hidden
    沿 T 复制 → coarse LanguageMetricField → 可选、训练/评测同构的 task35 ROI
    refinement → ``g_t = p * visibility [B,T,8]`` → RelationStateEncoder
    ``[B,T,2,d_model]``。冻结时 detach；联合训练时 coarse localization 保留梯度。
    """
    if metric_head is None or relation_encoder is None:
        return None, None
    from va_compound.model import dense_coords

    batch_size, sequence_length, _, _ = dense_evidence[11].shape
    head_dtype = next(metric_head.parameters()).dtype
    language_hidden = batch["language_hidden"].to(device=device, dtype=head_dtype)
    language_mask = batch.get("language_mask")
    if language_mask is None:
        language_mask = torch.ones(
            language_hidden.shape[:2], dtype=torch.bool, device=device
        )
    else:
        language_mask = language_mask.to(device=device)
    coords = dense_coords(512, device=device, dtype=head_dtype)

    language_flat = language_hidden.repeat_interleave(sequence_length, dim=0)
    mask_flat = language_mask.repeat_interleave(sequence_length, dim=0)

    def run_metric_head():
        flat = {
            layer: evidence.reshape(
                batch_size * sequence_length, -1, evidence.shape[-1]
            ).to(dtype=head_dtype)
            for layer, evidence in dense_evidence.items()
        }
        return metric_head(
            flat[5],
            flat[11],
            language_flat,
            mask_flat,
            coords,
        )

    def apply_roi(out):
        if roi_head is None or roi_alpha == 0.0:
            return out
        if roi_backbone is None or roi_frames is None:
            raise ValueError(
                "DINO ROI policy training requires the frozen backbone and raw frames"
            )
        frames_np = (
            roi_frames.detach().cpu().numpy()
            if isinstance(roi_frames, Tensor)
            else np.asarray(roi_frames)
        )
        if frames_np.ndim != 6 or frames_np.shape[:3] != (
            batch_size,
            sequence_length,
            4,
        ):
            raise ValueError(
                "DINO ROI frames must be [B,T,4,H,W,3], got "
                f"{tuple(frames_np.shape)}"
            )
        raw = (
            torch.from_numpy(
                np.ascontiguousarray(frames_np[:, :, (2, 3)]).reshape(
                    batch_size * sequence_length,
                    2,
                    *frames_np.shape[3:],
                )
            )
            .float()
            .div_(255.0)
            .permute(0, 1, 4, 2, 3)
            .to(device)
        )
        out.p, out.visibility = refine_metric_roi_positions_dino(
            out.p,
            out.visibility,
            raw,
            roi_backbone,
            roi_head,
            language_flat,
            mask_flat,
            coords,
            alpha=roi_alpha,
        )
        return out

    if train_metric_head:
        trainable = [p for p in metric_head.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "--mtvj-train-metric-head 已开启但 DINO metric head 没有可训练参数"
            )
        out = apply_roi(run_metric_head())
        g = _mtvj_metric_positions(
            out,
            getattr(
                metric_head,
                "_mtvj_metric_state_source",
                MTVJ_LEGACY_METRIC_STATE_SOURCE,
            ),
        ).reshape(batch_size, sequence_length, -1)
    else:
        with torch.no_grad():
            out = apply_roi(run_metric_head())
            g = (
                _mtvj_metric_positions(
                    out,
                    getattr(
                        metric_head,
                        "_mtvj_metric_state_source",
                        MTVJ_LEGACY_METRIC_STATE_SOURCE,
                    ),
                )
                .reshape(batch_size, sequence_length, -1)
                .detach()
            )
    return _mtvj_relation_tokens(g, relation_encoder), g

def _mtvj_metric_deltas(g: Tensor) -> Tensor:
    """Return causal metric deltas, with no invented predecessor at t=0."""
    if g.ndim != 3:
        raise ValueError(f"MT-VJ metric state must be [B, T, D], got {tuple(g.shape)}")
    nu = torch.zeros_like(g)
    if g.shape[1] > 1:
        nu[:, 1:] = g[:, 1:] - g[:, :-1]
    return nu

def _mtvj_relation_tokens(g: Tensor, relation_encoder: nn.Module) -> Tensor:
    """Encode metric positions while preserving every enabled upstream gradient."""
    if g.ndim != 3:
        raise ValueError(f"MT-VJ metric state must be [B, T, D], got {tuple(g.shape)}")
    batch_size, sequence_length, _ = g.shape
    nu = _mtvj_metric_deltas(g)
    z_g, z_nu = relation_encoder(
        g.reshape(batch_size * sequence_length, -1),
        nu.reshape(batch_size * sequence_length, -1),
    )
    return torch.stack((z_g, z_nu), dim=1).reshape(
        batch_size, sequence_length, 2, -1
    )

def _validate_dino_roi_resume_contract(
    resume_checkpoint: dict | None,
    *,
    runtime_checkpoint: Path | None,
    runtime_alpha: float | None,
    runtime_identity: dict | None = None,
) -> bool:
    """Fail fast when a resumed DINO policy changes its ROI distribution.

    Returns whether the saved policy used DINO ROI. Identity is checked after
    the runtime artifact has been loaded, while path/alpha checks can run first.
    """
    if resume_checkpoint is None:
        return False
    training_contract = resume_checkpoint.get("training_contract") or {}
    resume_dino_roi = training_contract.get("dino_roi_enabled") is True
    if not resume_dino_roi:
        return False
    if runtime_checkpoint is None:
        raise ValueError(
            "resume checkpoint requires --dino-roi-checkpoint; refusing to "
            "silently continue on a different geometry distribution"
        )
    saved_alpha = training_contract.get("dino_roi_alpha")
    if saved_alpha is None or runtime_alpha is None or float(saved_alpha) != float(
        runtime_alpha
    ):
        raise ValueError(
            "--dino-roi-alpha must exactly match resume training: "
            f"policy={saved_alpha!r}, runtime={runtime_alpha!r}"
        )
    if runtime_identity is None:
        return True
    expected_identity = resume_checkpoint.get("dino_roi_checkpoint_identity")
    if not isinstance(expected_identity, dict):
        raise ValueError("resume checkpoint declares DINO ROI but lacks its identity")
    mismatches = {
        key: (expected_identity.get(key), runtime_identity.get(key))
        for key in ("sha256", "size_bytes", "contract")
        if expected_identity.get(key) != runtime_identity.get(key)
    }
    if mismatches:
        raise ValueError(f"DINO ROI checkpoint identity mismatch on resume: {mismatches}")
    return True


def _mtvj_metric_head_constructor_config(metric_head: nn.Module) -> dict:
    """Read every constructor semantic from a live LanguageMetricField."""
    missing = [
        key for key in MTVJ_METRIC_HEAD_CONFIG_KEYS if not hasattr(metric_head, key)
    ]
    if missing:
        raise ValueError(
            "metric head 无法保存完整构造契约："
            f"missing attributes={missing}"
        )
    return _canonical_mtvj_metric_head_config(
        {key: getattr(metric_head, key) for key in MTVJ_METRIC_HEAD_CONFIG_KEYS},
        require_complete=True,
    )
