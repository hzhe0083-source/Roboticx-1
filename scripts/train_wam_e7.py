#!/usr/bin/env python
"""Standalone trainer for E7 WAM M0 (Task 6 of docs/superpowers/plans/2026-08-13-e7-wam-m0.md).

Trains `JointWorldActionFlow` (va_compound/wam.py, Task 1) against the frozen
base action-flow policy plus the WAM cache (va_compound/wam_cache.py, Task 5),
using the joint loss from design doc §4:

    L = 1.0*L_action + 0.5*L_VJ + 1.0*L_geo + 0.1*L_consistency   (L_consistency = 0 placeholder)

- L_action: masked_flow_matching_loss(v_base + dv, target, prefix_steps=6,
  prefix_weight=1.0, tail_weight=0.036) — copied/imported from train.py.
- L_VJ / L_geo: per-horizon smooth-L1 with cache-provided weights
  w = horizon_weight * (~perturbed_future) * valid_future; horizons with zero
  weight leave the denominator and are counted in the `excluded=` log field.
- One shared flow time tau (design §3.1: action / spatial-latent / geo tokens
  share the same flow time): x_t = (1-tau)*noise + tau*target,
  v_target = target - noise on all three paths.
- Targets are taken from the cache as stored, no re-derivation:
  latent target = future_latent_target (already Delta = z(d+k) - z(d));
  geo target = future_geo_target (slice0 = g_future absolute, slice1 = nu).
- Record schema: every record (real cache or synthetic) is checked against
  va_compound.wam_cache.assert_record_schema (whitelist keys/types/shapes).

Determinism contract (bitwise resume):
- Deterministic cyclic record order (no shuffling): step s consumes rows
  ((s*grad_accum*batch_size + i) % n_train) for i in 0..eff-1, so any step
  count is in-bounds and resume reproduces the uninterrupted run bitwise.
- Checkpoint stores wam_model / optimizer_state / global_step / rng_state /
  exact_run_contract (saved after optimizer.step, i.e. post-step state), so a
  --resume continuation reproduces the uninterrupted run bitwise.
- Stateless warmup+cosine LR schedule (function of completed steps only).

Modes:
- --smoke: synthetic CPU records + BaseStub frozen base (v = 0.1*x_t), fp32,
  asserts per-term gradients non-zero, finite losses, consistency == 0.
  base_ckpt_sha256 recorded as "smoke".
- --fake-cache: builds a synthetic cache via build_wam_cache(None, ...), then
  runs WAMCacheDataset + the real JointWorldActionFlow for a few CPU steps —
  end-to-end proof that the cache schema feeds the trainer (M0 substitute for
  the real-cache end-to-end run).

Gates:
- --cache-dir requires manifest with contract_version == 1 and non-empty
  data_sha256 + base_ckpt_sha256 (skipped only for --fake-cache, which builds
  its own synthetic cache).
- Training without --base-ckpt is rejected unless --smoke/--fake-cache
  (real cache + BaseStub combination is forbidden).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import sys
import tempfile
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Task 1 interface: va_compound/wam.py (implemented by a parallel agent).
# Call strictly by the pinned signature. If the module has not landed yet,
# use a compact temporary stand-in with the exact interface so this trainer
# is self-testable; the real module is picked up automatically once it exists.
# ---------------------------------------------------------------------------
try:
    from va_compound.wam import (  # noqa: F401
        JointWorldActionFlow,
        WAMConfig,
        WAMSceneVelocity,
    )

    _USING_WAM_STUB = False
except Exception:  # module missing or mid-edit by the Task 1 agent
    _USING_WAM_STUB = True

    from dataclasses import dataclass  # noqa: E402

    @dataclass(frozen=True)
    class WAMConfig:  # temporary stand-in mirror of Task 1's config (smaller for fast CPU smoke)
        hidden_dim: int = 256
        num_layers: int = 2
        num_heads: int = 8
        ffn_hidden: int = 512
        cond_dim: int = 64
        vision_dim: int = 768
        geo_dim: int = 8
        n_scene_tokens: int = 16
        n_geo_tokens: int = 2
        horizons: tuple = (6, 24, 48)
        action_horizon: int = 48
        action_dim: int = 4
        qk_norm: bool = True
        use_swiglu: bool = True
        dropout: float = 0.0

    @dataclass
    class WAMSceneVelocity:
        latent: Tensor  # [B, 3, 16, 768]
        geo: Tensor  # [B, 3, 2, 8]

    class _StubAdaLNLayer(nn.Module):
        """Zero-init AdaLN residual block: identity at init (zero-init discipline)."""

        def __init__(self, hidden: int, cond_dim: int, ffn_hidden: int) -> None:
            super().__init__()
            self.adaln = nn.Linear(cond_dim, 6 * hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, ffn_hidden), nn.SiLU(), nn.Linear(ffn_hidden, hidden)
            )
            nn.init.zeros_(self.adaln.weight)
            nn.init.zeros_(self.adaln.bias)
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)

        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            s1, s2, g1, s3, s4, g2 = torch.chunk(self.adaln(t), 6, dim=-1)
            h = x * (1.0 + s1[:, None, :]) + s2[:, None, :]
            h = F.silu(h) * (1.0 + s3[:, None, :]) + s4[:, None, :]
            return x + g2[:, None, :] * self.ffn(h)

    class JointWorldActionFlow(nn.Module):
        """Temporary compact stand-in (replaced by va_compound.wam at integration).

        Token layout: 48 action + 3*16 latent + 3*2 geo = 102 tokens.
        Action residual head zero-init (dv == 0 at init); scene heads normal init.
        """

        def __init__(self, config: WAMConfig) -> None:
            super().__init__()
            self.config = config
            hidden = config.hidden_dim
            self.act_in = nn.Linear(4, hidden)
            self.cond_in = nn.Linear(512, hidden)
            self.lat_in = nn.Linear(768, hidden)
            self.geo_in = nn.Linear(8, hidden)
            self.time_mlp = nn.Sequential(nn.Linear(1, config.cond_dim), nn.SiLU())
            self.layers = nn.ModuleList(
                _StubAdaLNLayer(hidden, config.cond_dim, config.ffn_hidden)
                for _ in range(config.num_layers)
            )
            self.action_norm = nn.LayerNorm(hidden)
            self.action_head = nn.Linear(hidden, 4)  # zero-init
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
            self.latent_heads = nn.ModuleList(nn.Linear(hidden, 768) for _ in range(3))
            self.geo_heads = nn.ModuleList(nn.Linear(hidden, 8) for _ in range(3))

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def forward(
            self,
            *,
            action_condition: Tensor,
            va_layers: tuple,
            spatial_tokens: Tensor,
            geo_tokens: Tensor,
            noisy_actions: Tensor,
            noisy_scene_latents: Tensor,
            noisy_scene_geo: Tensor,
            flow_time: Tensor,
        ):
            batch = noisy_actions.shape[0]
            # Shape guards mirror the pinned interface.
            assert action_condition.shape == (batch, 48, 512)
            assert noisy_actions.shape == (batch, 48, 4)
            assert noisy_scene_latents.shape == (batch, 3, 16, 768)
            assert noisy_scene_geo.shape == (batch, 3, 2, 8)
            assert flow_time.shape == (batch,)
            assert isinstance(va_layers, tuple) and len(va_layers) >= 1
            a = self.act_in(noisy_actions) + self.cond_in(action_condition)
            latent = self.lat_in(noisy_scene_latents).reshape(batch, 3 * 16, self.config.hidden_dim)
            geo = self.geo_in(noisy_scene_geo).reshape(batch, 3 * 2, self.config.hidden_dim)
            x = torch.cat([a, latent, geo], dim=1)  # [B, 102, hidden]
            t = self.time_mlp(flow_time[:, None])
            for layer in self.layers:
                x = layer(x, t)
            dv = self.action_head(self.action_norm(x[:, :48]))
            latent_v = torch.stack(
                [head(x[:, 48 + k * 16: 48 + (k + 1) * 16]) for k, head in enumerate(self.latent_heads)],
                dim=1,
            )
            geo_v = torch.stack(
                [head(x[:, 96 + k * 2: 96 + (k + 1) * 2]) for k, head in enumerate(self.geo_heads)],
                dim=1,
            )
            return dv, WAMSceneVelocity(latent=latent_v, geo=geo_v)

# ---------------------------------------------------------------------------
# Task 5 interface: va_compound/wam_cache.py (implemented by a parallel agent).
# ---------------------------------------------------------------------------
try:
    from va_compound.wam_cache import (  # noqa: F401
        WAMCacheDataset,
        WAMCacheManifest,
        assert_record_schema,
    )

    _CACHE_AVAILABLE = True
except Exception:  # module missing or mid-edit by the Task 5 agent
    _CACHE_AVAILABLE = False

    def assert_record_schema(record: dict) -> dict:
        """Minimal mirror of va_compound.wam_cache.assert_record_schema so the
        trainer stays self-testable while Task 5's module is absent. The real
        whitelist in wam_cache.RECORD_SCHEMA is authoritative."""
        spec = {
            "action_condition": (48, 512),
            "va_layers": "va_layers",
            "spatial16": (16, 768),
            "geo8": (8,),
            "actions": (48, 4),
            "future_latent_target": (3, 16, 768),
            "future_geo_target": (3, 2, 8),
            "action_valid": (48,),
            "perturbed_future": (3,),
            "valid_future": (3,),
            "horizon_weight": (3,),
            "episode_id": None,
            "task_id": None,
            "task_file": None,
            "ep_idx": None,
            "decision_frame": None,
        }
        if not isinstance(record, dict) or set(record) != set(spec):
            raise ValueError(f"record keys != Task 5 whitelist: {sorted(set(record) ^ set(spec))}")
        for key, shape in spec.items():
            value = record[key]
            if key == "va_layers":
                if len(value) != 8 or any(tuple(t.shape) != (16, 512) for t in value):
                    raise ValueError(f"{key}: expected 8x[16,512]")
            elif shape is not None and tuple(value.shape) != shape:
                raise ValueError(f"{key}: shape {tuple(value.shape)} != {shape}")
        return record

# ---------------------------------------------------------------------------
# Flow-matching loss from train.py (source: train.py lines 1435-1524, 31).
# Preferred path imports it; if train.py is mid-edit/broken, use the verbatim
# self-contained copy below so this trainer never depends on import side effects.
# ---------------------------------------------------------------------------
try:
    from train import ACTION_MASK_KEYS, masked_flow_matching_loss  # noqa: F401
except Exception:
    ACTION_MASK_KEYS = ("action_valid_mask", "horizon_mask")

    def _expand_action_mask(mask: Tensor, reference: Tensor, name: str) -> Tensor:
        """Broadcast a common action/horizon mask shape to ``[B,T,H,A]``. (verbatim train.py)"""
        if reference.ndim != 4:
            raise ValueError("flow tensors must have shape [batch, sequence, horizon, action_dim]")
        batch, sequence, horizon, action_dim = reference.shape
        shape = tuple(mask.shape)
        if shape == tuple(reference.shape):
            expanded = mask
        elif shape == (batch, sequence, horizon):
            expanded = mask.unsqueeze(-1)
        elif shape == (batch, sequence):
            expanded = mask[:, :, None, None]
        elif shape == (batch, horizon):
            expanded = mask[:, None, :, None]
        elif shape == (sequence, horizon):
            expanded = mask[None, :, :, None]
        elif shape == (horizon,):
            expanded = mask[None, None, :, None]
        else:
            raise ValueError(
                f"{name} shape {shape} cannot broadcast to flow tensor "
                f"{(batch, sequence, horizon, action_dim)}"
            )
        expanded = expanded.to(device=reference.device, dtype=reference.dtype).expand_as(reference)
        if not bool(torch.isfinite(expanded).all()) or bool((expanded < 0).any()):
            raise ValueError(f"{name} must contain finite non-negative weights")
        return expanded

    def masked_flow_matching_loss(
        predicted_velocity: Tensor,
        target_velocity: Tensor,
        batch: dict | None = None,
        *,
        prefix_steps: int = 6,
        prefix_weight: float = 1.0,
        tail_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Flow MSE with optional validity masks and prefix/tail weighting. (verbatim train.py)"""
        if predicted_velocity.shape != target_velocity.shape or predicted_velocity.ndim != 4:
            raise ValueError(
                "predicted_velocity and target_velocity must share shape "
                "[batch, sequence, horizon, action_dim]"
            )
        if prefix_steps <= 0:
            raise ValueError("prefix_steps must be positive")
        if prefix_weight < 0.0 or tail_weight < 0.0:
            raise ValueError("prefix/tail flow weights must be non-negative")

        squared_error = (predicted_velocity - target_velocity).square()
        validity = torch.ones_like(squared_error)
        if batch is not None:
            for key in ACTION_MASK_KEYS:
                mask = batch.get(key)
                if mask is not None:
                    validity = validity * _expand_action_mask(mask, squared_error, key)

        split = min(prefix_steps, squared_error.shape[-2])

        def region_mean(error: Tensor, weight: Tensor) -> Tensor:
            denominator = weight.sum()
            if not bool(denominator > 0):
                return error.new_zeros(())
            return (error * weight).sum() / denominator

        prefix_loss = region_mean(
            squared_error[..., :split, :], validity[..., :split, :]
        )
        tail_loss = region_mean(
            squared_error[..., split:, :], validity[..., split:, :]
        )
        horizon_weights = squared_error.new_full((squared_error.shape[-2],), tail_weight)
        horizon_weights[:split] = prefix_weight
        element_weights = validity * horizon_weights.view(1, 1, -1, 1)
        denominator = element_weights.sum()
        if not bool(denominator > 0):
            raise ValueError("flow loss has zero valid weighted action elements")

        has_mask = batch is not None and any(batch.get(key) is not None for key in ACTION_MASK_KEYS)
        if not has_mask and prefix_weight == 1.0 and tail_weight == 1.0:
            loss = F.mse_loss(predicted_velocity, target_velocity)
        else:
            loss = (squared_error * element_weights).sum() / denominator
        return loss, prefix_loss, tail_loss


# ---------------------------------------------------------------------------
# Frozen base policy.
# ---------------------------------------------------------------------------
class BaseStub(nn.Module):
    """Frozen base velocity stand-in for --smoke: v = 0.1 * x_t (matches plan Task 6)."""

    def flow_velocity(
        self,
        action_condition: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        return 0.1 * noisy_actions


def load_base_policy(path: str, device: torch.device) -> nn.Module:
    """Load a real VACompoundPolicy from a base checkpoint and freeze it."""
    from va_compound.model import VACompoundConfig, VACompoundPolicy

    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg_dict = {
        k: v
        for k, v in payload["config"].items()
        if k in VACompoundConfig.__dataclass_fields__
    }
    config = VACompoundConfig(**cfg_dict)
    if config.action_horizon != 48 or config.action_dim != 4:
        raise ValueError(
            f"base checkpoint {path} has action_horizon={config.action_horizon}, "
            "action_dim={config.action_dim}; E7 WAM requires 48x4"
        )
    policy = VACompoundPolicy(config)
    policy.load_state_dict(payload["model"])
    policy.to(device)
    policy.eval()
    policy.requires_grad_(False)
    print(f"base policy loaded from {path} ({sum(p.numel() for p in policy.parameters())} params, frozen)")
    return policy


# ---------------------------------------------------------------------------
# Synthetic smoke dataset (deterministic per index; stateless w.r.t. access order).
# Record schema + target semantics match the real cache exactly (same whitelist
# via assert_record_schema, Δlatent targets, [g_future, nu] geo targets):
#   future_latent_target = z(d+k) - z(d)          (Delta, not absolute)
#   future_geo_target    = [g_future, nu]          (slice0 absolute, slice1 delta)
# ---------------------------------------------------------------------------
_SYNTH_HORIZON_WEIGHTS = (1.0, 0.5, 0.25)


class SyntheticWAMDataset:
    def __init__(self, n: int, seed: int = 1234) -> None:
        self.n = int(n)

        def per_index(i: int):
            gen = torch.Generator().manual_seed(seed + i)
            return gen

        self._gen = per_index

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        gen = self._gen(index)
        action_condition = torch.randn(48, 512, generator=gen)
        va_layers = tuple(torch.randn(16, 512, generator=gen) for _ in range(8))
        spatial16 = torch.randn(16, 768, generator=gen)          # z(d)
        geo8 = torch.randn(8, generator=gen)                     # g(d)
        actions = torch.randn(48, 4, generator=gen)
        z_future = torch.randn(3, 16, 768, generator=gen)        # z(d+k) absolute
        future_latent_target = z_future - spatial16              # Δlatent（cache 语义）
        g_future = torch.randn(3, 8, generator=gen)              # g(d+k) absolute
        nu = g_future - geo8                                     # ν = g_future − g_current
        future_geo_target = torch.stack([g_future, nu], dim=1)   # [3,2,8]
        return {
            "action_condition": action_condition,
            "va_layers": va_layers,
            "spatial16": spatial16,
            "geo8": geo8,
            "actions": actions,
            "future_latent_target": future_latent_target,
            "future_geo_target": future_geo_target,
            "action_valid": torch.ones(48, dtype=torch.bool),
            "perturbed_future": torch.zeros(3, dtype=torch.bool),
            "valid_future": torch.ones(3, dtype=torch.bool),
            "horizon_weight": torch.tensor(_SYNTH_HORIZON_WEIGHTS, dtype=torch.float32),
            "episode_id": index,
            "task_id": index % 3,
            "task_file": f"fake-task-{index % 3:02d}",
            "ep_idx": 0,
            "decision_frame": index,
        }


def make_dataset(args: argparse.Namespace):
    if args.cache_dir:
        if not _CACHE_AVAILABLE:
            raise SystemExit(
                "--cache-dir requires va_compound/wam_cache.py (Task 5 not integrated yet)"
            )
        manifest_path = os.path.join(args.cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise SystemExit(f"no manifest.json under {args.cache_dir}")
        manifest = WAMCacheManifest.load(manifest_path)
        # contract_version gate: trainer only accepts the schema it was built for.
        if int(manifest.contract_version) != 1:
            raise SystemExit(
                f"cache manifest contract_version={manifest.contract_version} != 1: "
                "cache/trainer schema mismatch (rebuild the cache)"
            )
        # SHA gate: real training requires both data and base SHA; --fake-cache
        # builds its own synthetic cache (no real data/base SHAs to check).
        if not args.fake_cache and (
            not manifest.data_sha256 or not manifest.base_ckpt_sha256
        ):
            raise SystemExit(
                "cache manifest is missing data_sha256 or base_ckpt_sha256: "
                "real training requires both (rebuild the cache after the base freeze)"
            )
        return WAMCacheDataset(args.cache_dir, manifest, split="train")
    print("synthetic WAM records (--smoke path, deterministic per index)")
    n_needed = args.steps * args.grad_accum * args.batch_size
    return SyntheticWAMDataset(max(n_needed, 1))


def collate_records(records: list[dict]) -> dict:
    out: dict = {}
    for record in records:
        assert_record_schema(record)  # 白名单键/类型/形状，与 cache 写入同契约
    for key in sorted(records[0]):
        if key == "va_layers":
            n_layers = len(records[0][key])
            out[key] = tuple(
                torch.stack([record[key][i] for record in records])
                for i in range(n_layers)
            )
        elif isinstance(records[0][key], torch.Tensor):
            out[key] = torch.stack([record[key] for record in records])
        else:
            out[key] = [record[key] for record in records]  # provenance，不进 loss
    return out


def batch_to(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for key, value in batch.items():
        if isinstance(value, tuple):
            out[key] = tuple(t.to(device) for t in value)
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value  # provenance 列表等非张量字段，不进 loss
    return out


# ---------------------------------------------------------------------------
# Flow sampling + joint loss (design §4).
# ---------------------------------------------------------------------------
def compute_losses(
    wam: nn.Module,
    base: nn.Module,
    batch: dict,
    consistency_weight: float = 0.0,
) -> dict[str, Tensor]:
    actions = batch["actions"]  # [B,48,4]
    n_batch = actions.shape[0]
    device = actions.device
    dtype = actions.dtype

    # 单一共享 flow time τ（设计 §3.1：动作/空间 latent/几何三路同 τ）：
    # x_t = (1-τ)·noise + τ·target, v_target = target − noise。
    noise = torch.randn_like(actions)
    tau = torch.rand(n_batch, device=device, dtype=dtype)
    tau_b = tau[:, None, None]
    x_t = (1.0 - tau_b) * noise + tau_b * actions
    target_v = actions - noise

    # 场景 latent：cache 已存 Δlatent = z(d+k) − z(d)，直接作 target，
    # 不再减当前（fill_record_latents 内已做差分）。
    delta_z = batch["future_latent_target"]  # [B,3,16,768]
    scene_noise = torch.randn_like(delta_z)
    tau_s = tau[:, None, None, None]
    x_scene = (1.0 - tau_s) * scene_noise + tau_s * delta_z
    v_scene_target = delta_z - scene_noise

    # 场景 geo：cache 已存 [g_future, ν]（slice0 绝对 g(d+k)、slice1 差分
    # ν = g_future − g_current），两条 slice 分别做直线路径，不再手算
    # g(d+k) − geo8。
    geo_target = batch["future_geo_target"]  # [B,3,2,8]
    geo_noise = torch.randn_like(geo_target)
    tau_g = tau[:, None, None, None]
    x_geo = (1.0 - tau_g) * geo_noise + tau_g * geo_target
    v_geo_target = geo_target - geo_noise

    with torch.no_grad():
        v_base = base.flow_velocity(batch["action_condition"], x_t, tau)

    dv, scene_v = wam(
        action_condition=batch["action_condition"],
        va_layers=batch["va_layers"],
        spatial_tokens=batch["spatial16"],
        geo_tokens=batch["geo8"],
        noisy_actions=x_t,
        noisy_scene_latents=x_scene,
        noisy_scene_geo=x_geo,
        flow_time=tau,
    )

    v_pred = v_base + dv
    # WAM flow has no sequence axis; the loss contract is [B, T, H, A] with T=1.
    loss_action = masked_flow_matching_loss(
        v_pred.unsqueeze(1),
        target_v.unsqueeze(1),
        {"action_valid_mask": batch["action_valid"]},
        prefix_steps=6,
        prefix_weight=1.0,
        tail_weight=0.036,
    )[0]

    # 未来目标 mask 入 loss：w = horizon_weight · (~perturbed_future) · valid_future
    # （两者都来自 cache 白名单字段；w=0 的跨度退出分母并计入 excluded）。
    horizon_w = batch["horizon_weight"]  # [B,3]
    future_w = (
        horizon_w
        * (~batch["perturbed_future"]).float()
        * batch["valid_future"].float()
    )  # [B,3]

    def masked_smooth_l1(pred: Tensor, target: Tensor, w: Tensor) -> Tensor:
        se = F.smooth_l1_loss(pred, target, reduction="none").mean(
            dim=tuple(range(1, pred.ndim))
        )  # [B]
        denom = w.sum()
        if not bool(denom > 0):
            return pred.new_zeros(())
        return (se * w).sum() / denom

    loss_vj = torch.zeros((), device=device, dtype=dtype)
    loss_geo = torch.zeros((), device=device, dtype=dtype)
    for k in range(3):
        loss_vj = loss_vj + masked_smooth_l1(
            scene_v.latent[:, k], v_scene_target[:, k], future_w[:, k]
        )
        loss_geo = loss_geo + masked_smooth_l1(
            scene_v.geo[:, k], v_geo_target[:, k], future_w[:, k]
        )
    loss_vj = loss_vj / 3.0
    loss_geo = loss_geo / 3.0
    excluded = int((future_w <= 0).sum().item())

    # Placeholder: first-round consistency term is 0 (enabled post G1/G2 gates).
    loss_consistency = loss_action.new_zeros(())
    total = (
        1.0 * loss_action
        + 0.5 * loss_vj
        + 1.0 * loss_geo
        + 0.1 * consistency_weight * loss_consistency
    )
    return {
        "action": loss_action,
        "vj": loss_vj,
        "geo": loss_geo,
        "consistency": loss_consistency,
        "total": total,
        "excluded": excluded,
    }


def _assert_nonzero_grads(wam: nn.Module, losses: dict[str, Tensor]) -> None:
    """Smoke gate: each of the three losses must produce non-zero gradients."""
    for name in ("action", "vj", "geo"):
        wam.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=True)
        total = 0.0
        for param in wam.parameters():
            if param.grad is not None:
                total = total + param.grad.detach().abs().sum().float()
        assert float(total) > 0.0, f"{name} loss produced zero gradient (M0 smoke gate failed)"
    wam.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Optimizer / schedule / contract / checkpoint.
# ---------------------------------------------------------------------------
def lr_at(completed_steps: int, lr: float, warmup: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 0. Stateless function of completed steps."""
    s = float(completed_steps)
    if s < warmup:
        return lr * (s + 1.0) / float(warmup)
    denom = max(float(total_steps) - float(warmup), 1.0)
    progress = min((s - float(warmup)) / denom, 1.0)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(wam: nn.Module, lr: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(wam.parameters(), lr=lr, weight_decay=1e-4)


def build_contract(args: argparse.Namespace, wam_config, base_sha: str) -> dict:
    return {
        "contract_version": 1,
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "save_every": int(args.save_every),
        "horizons": list(wam_config.horizons),
        "consistency_weight": float(args.consistency_weight),
        "base_ckpt": args.base_ckpt if args.base_ckpt else "smoke",
        "base_ckpt_sha256": base_sha,
        "weight_decay": 1e-4,
        "seed": int(args.seed),
    }


def validate_contract(saved: dict, current: dict) -> None:
    mismatches = [key for key in current if saved.get(key) != current[key]]
    if mismatches:
        raise ValueError(
            f"resume contract mismatch on {sorted(mismatches)}; "
            "exact resume requires identical hyperparameters"
        )


def sha256_file(path: str | None) -> str:
    if not path:
        return "smoke"  # --smoke/--fake-cache 无真实基座：SHA 记为 "smoke"
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng() -> dict:
    state: dict = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_checkpoint(wam, optimizer, global_step, wam_config, contract, base_sha) -> dict:
    return {
        "wam_config": asdict(wam_config),
        "wam_model": {key: value.detach().cpu() for key, value in wam.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "global_step": int(global_step),
        "base_ckpt_sha256": base_sha,
        "exact_run_contract": contract,
        "rng_state": capture_rng(),
    }


def save_checkpoint(path: str, payload: dict) -> None:
    """Atomic write: tmp file + os.replace (train.py save_checkpoint pattern)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def run_training(
    args: argparse.Namespace,
    wam: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    base: nn.Module,
    *,
    start_step: int,
    rng_state: dict | None,
    check_grads: bool,
    out_ckpt: str,
    contract: dict,
    base_sha: str,
    wam_config,
) -> dict[int, Tensor]:
    """Run steps [start_step, args.steps); returns {completed_step: total_loss_tensor}."""
    if rng_state is not None:
        restore_rng(rng_state)
    validate = args.smoke or args.fake_cache  # CPU validation modes
    if validate:
        device = torch.device("cpu")
        autocast_ctx = contextlib.nullcontext()
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )

    n_train = len(dataset)
    if n_train < 1:
        raise SystemExit("train split is empty (no records for training)")
    eff_batch = args.grad_accum * args.batch_size

    step_losses: dict[int, Tensor] = {}
    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.lr, args.warmup, args.steps)

        acc = {name: 0.0 for name in ("total", "action", "vj", "geo", "consistency")}
        acc_excluded = 0
        for micro in range(args.grad_accum):
            # 确定性循环采样：step s 消费行号 (s*eff_batch + i) % n_train。
            # 无 shuffle、永不越界、可精确续训（resume 从 start_step 续吃同一序列）。
            idxs = [
                (step * eff_batch + micro * args.batch_size + j) % n_train
                for j in range(args.batch_size)
            ]
            batch = collate_records([dataset[idx] for idx in idxs])
            batch = batch_to(batch, device)
            with autocast_ctx:
                losses = compute_losses(wam, base, batch, args.consistency_weight)
                if validate:
                    for name in ("action", "vj", "geo", "total"):
                        assert bool(torch.isfinite(losses[name])), f"{name} loss not finite in smoke"
                    assert float(losses["consistency"]) == 0.0, "consistency placeholder must be 0"
            if check_grads and step == start_step and micro == 0:
                _assert_nonzero_grads(wam, losses)
            (losses["total"] / args.grad_accum).backward()
            for name in acc:
                acc[name] += losses[name].detach().float()
            acc_excluded += int(losses["excluded"])

        grad_norm = float(torch.nn.utils.clip_grad_norm_(wam.parameters(), 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        means = {name: value / args.grad_accum for name, value in acc.items()}
        print(
            f"step={completed} loss={means['total']:.6f} action={means['action']:.6f} "
            f"vj={means['vj']:.6f} geo={means['geo']:.6f} "
            f"consistency={means['consistency']:.6f} grad={grad_norm:.6f} "
            f"excluded={acc_excluded}",
            flush=True,
        )
        step_losses[completed] = means["total"].clone()
        if completed % args.save_every == 0:
            save_checkpoint(
                out_ckpt,
                make_checkpoint(wam, optimizer, completed, wam_config, contract, base_sha),
            )
    return step_losses


# ---------------------------------------------------------------------------
# Modes: normal / self-check resume.
# ---------------------------------------------------------------------------
def self_check_resume(args: argparse.Namespace) -> int:
    """In-process bitwise resume check: uninterrupted run vs fresh load from disk."""
    if not args.smoke:
        raise SystemExit("--self-check-resume requires --smoke")
    if args.resume:
        raise SystemExit("--self-check-resume manages resume itself; drop --resume")
    if args.save_every >= args.steps or args.steps % args.save_every == 0:
        raise SystemExit(
            "--self-check-resume needs save_every < steps and steps % save_every != 0 "
            "(need at least one step after the last save)"
        )

    dataset = make_dataset(args)
    base = BaseStub()
    base_sha = sha256_file(args.base_ckpt)

    # Phase A: uninterrupted run (also runs the smoke gradient gates).
    torch.manual_seed(args.seed)
    wam_a = JointWorldActionFlow(WAMConfig())
    optimizer_a = build_optimizer(wam_a, args.lr)
    contract_a = build_contract(args, WAMConfig(), base_sha)
    losses_a = run_training(
        args, wam_a, optimizer_a, dataset, base,
        start_step=0, rng_state=None, check_grads=True,
        out_ckpt=args.out, contract=contract_a, base_sha=base_sha,
        wam_config=WAMConfig(),
    )
    last_saved = (args.steps // args.save_every) * args.save_every
    ref_step = last_saved + 1
    ref_loss = losses_a[ref_step]

    # Phase B: fresh objects, restore everything from the on-disk checkpoint.
    checkpoint = torch.load(args.out, map_location="cpu", weights_only=True)
    assert int(checkpoint["global_step"]) == last_saved
    wam_config_b = WAMConfig(**checkpoint["wam_config"])
    wam_b = JointWorldActionFlow(wam_config_b)
    wam_b.load_state_dict(checkpoint["wam_model"])
    optimizer_b = build_optimizer(wam_b, args.lr)
    optimizer_b.load_state_dict(checkpoint["optimizer_state"])
    validate_contract(checkpoint["exact_run_contract"], contract_a)
    losses_b = run_training(
        args, wam_b, optimizer_b, dataset, base,
        start_step=last_saved, rng_state=checkpoint["rng_state"], check_grads=False,
        out_ckpt=args.out, contract=contract_a, base_sha=base_sha,
        wam_config=wam_config_b,
    )
    resumed_loss = losses_b[ref_step]

    bitwise = bool(torch.equal(ref_loss, resumed_loss))
    print(f"self-check resume: reference step={ref_step} loss={ref_loss.item():.9f} "
          f"resumed loss={resumed_loss.item():.9f} bitwise_equal={bitwise}")
    if not bitwise:
        print("SELF-CHECK RESUME: FAIL")
        return 1
    print("SELF-CHECK RESUME: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    cli_args = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="E7 WAM M0 standalone trainer (Task 6)")
    parser.add_argument("--smoke", action="store_true", help="synthetic CPU run with BaseStub")
    parser.add_argument("--fake-cache", action="store_true",
                        help="build synthetic cache via build_wam_cache(None, ...) and train "
                             "a few CPU steps through WAMCacheDataset + real JointWorldActionFlow "
                             "(cache->trainer schema end-to-end check)")
    parser.add_argument("--base-ckpt", type=str, default=None, help="frozen base VACompoundPolicy checkpoint")
    parser.add_argument("--cache-dir", type=str, default=None, help="WAM cache directory (Task 5)")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--out", type=str, default="checkpoints/wam_e7.pt")
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="placeholder (L_consistency = 0 in M0)")
    parser.add_argument("--resume", type=str, default=None, help="resume from a WAM checkpoint")
    parser.add_argument("--self-check-resume", action="store_true",
                        help="in-process bitwise resume self test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(cli_args)

    if args.batch_size < 1 or args.grad_accum < 1 or args.steps < 1:
        raise SystemExit("batch-size/grad-accum/steps must be positive")

    if args.self_check_resume:
        return self_check_resume(args)

    # Gate: 训练（非 --smoke/--fake-cache）必须给 --base-ckpt；
    # 真实 cache + BaseStub 组合被直接拒绝。
    if not args.base_ckpt and not args.smoke and not args.fake_cache:
        raise SystemExit(
            "--base-ckpt is required for training (real cache + BaseStub is forbidden); "
            "use --smoke/--fake-cache for BaseStub validation runs"
        )

    # --fake-cache：build_wam_cache(None, ...) 合成 cache → 走真实
    # WAMCacheDataset + 真实 JointWorldActionFlow 的 CPU 端到端。
    if args.fake_cache:
        if args.resume:
            raise SystemExit("--fake-cache does not support --resume")
        if not _CACHE_AVAILABLE:
            raise SystemExit("--fake-cache requires va_compound/wam_cache.py (Task 5)")
        from va_compound.wam_cache import build_wam_cache

        if not args.cache_dir:
            args.cache_dir = tempfile.mkdtemp(prefix="wam_fake_cache_")
        if "--steps" not in cli_args:
            args.steps = 3
            print("--fake-cache: steps defaulting to 3 (CPU schema end-to-end check)")
        manifest = build_wam_cache(None, args.cache_dir, base_ckpt=None)
        print(
            f"--fake-cache: built {manifest.n_anchors} synthetic records at "
            f"{args.cache_dir} (contract={manifest.contract} "
            f"v{manifest.contract_version}, action_axis_units={manifest.action_axis_units})"
        )

    if _USING_WAM_STUB:
        print("WARNING: va_compound/wam.py not importable; using temporary smoke stand-in")
    torch.manual_seed(args.seed)
    dataset = make_dataset(args)  # 含 contract_version + SHA 门禁（cache-dir 路径）
    if args.smoke or args.fake_cache:
        base = BaseStub()
        base_sha = "smoke"
    else:
        base = load_base_policy(
            args.base_ckpt,
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )
        base_sha = sha256_file(args.base_ckpt)
    if args.smoke and args.base_ckpt is not None:
        print("NOTE: --smoke forces BaseStub; --base-ckpt ignored for the base velocity")

    smoke_device = torch.device("cpu")
    train_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = smoke_device if (args.smoke or args.fake_cache) else train_device

    wam_config = WAMConfig()
    wam = JointWorldActionFlow(wam_config).to(device)
    optimizer = build_optimizer(wam, args.lr)
    contract = build_contract(args, wam_config, base_sha)
    start_step = 0
    rng_state = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        wam_config = WAMConfig(**checkpoint["wam_config"])
        wam = JointWorldActionFlow(wam_config).to(device)
        wam.load_state_dict(checkpoint["wam_model"])
        optimizer = build_optimizer(wam, args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        contract = build_contract(args, wam_config, base_sha)
        validate_contract(checkpoint["exact_run_contract"], contract)
        start_step = int(checkpoint["global_step"])
        rng_state = checkpoint["rng_state"]
        print(f"resuming from {args.resume} at step {start_step}")

    run_training(
        args, wam, optimizer, dataset, base,
        start_step=start_step, rng_state=rng_state,
        check_grads=args.smoke and start_step == 0,
        out_ckpt=args.out, contract=contract, base_sha=base_sha,
        wam_config=wam_config,
    )
    if args.smoke:
        print(f"smoke completed: {args.steps} steps, gradient/finiteness gates passed")
    elif args.fake_cache:
        print(
            f"fake-cache completed: {args.steps} steps through WAMCacheDataset + "
            f"{type(wam).__name__} (cache->trainer schema E2E ok)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
