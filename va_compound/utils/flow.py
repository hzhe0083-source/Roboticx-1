"""Flow-matching sampling and loss helpers.

Pure tensor functions used by the training loop: probability-path sampling,
prefix/tail-weighted flow loss, and the shared-source counterfactual pair
loss. They depend only on ``torch`` (plus a lazy import of the frozen V-JEPA
encoder for perturbed frames), so they are safe to keep independent of the
trainer.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


ACTION_MASK_KEYS = ("action_valid_mask", "horizon_mask")


def _expand_action_mask(mask: Tensor, reference: Tensor, name: str) -> Tensor:
    """Broadcast a common action/horizon mask shape to ``[B,T,H,A]``."""
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
    """Flow MSE with optional validity masks and prefix/tail weighting.

    Returns ``(training_loss, raw_prefix_mse, raw_tail_mse)``.  The diagnostic
    terms use validity masks but intentionally exclude prefix/tail scalar
    weights, so changing ``tail_weight`` does not hide tail quality.
    """
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

    # Preserve the exact legacy reduction when no masking/reweighting is active.
    has_mask = batch is not None and any(batch.get(key) is not None for key in ACTION_MASK_KEYS)
    if not has_mask and prefix_weight == 1.0 and tail_weight == 1.0:
        loss = F.mse_loss(predicted_velocity, target_velocity)
    else:
        loss = (squared_error * element_weights).sum() / denominator
    return loss, prefix_loss, tail_loss


def effective_action_valid_fraction(batch: dict | None, reference: Tensor) -> Tensor:
    """Return the supervised fraction after combining all optional masks.

    Reporting this next to loss prevents a lower number caused by masking bad
    H48 targets from being mistaken for genuine policy improvement.
    """
    validity = torch.ones_like(reference)
    if batch is not None:
        for key in ACTION_MASK_KEYS:
            mask = batch.get(key)
            if mask is not None:
                validity = validity * _expand_action_mask(mask, reference, key)
    return validity.mean()


def sample_flow_matching_inputs(
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a straight probability path from Gaussian noise to expert actions.
    """
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, sequence, horizon, action_dim]")
    noise = torch.randn_like(actions)
    flow_time = torch.rand(
        actions.shape[:2],
        device=actions.device,
        dtype=actions.dtype,
    )
    tau = flow_time[:, :, None, None]
    noisy_actions = (1.0 - tau) * noise + tau * actions
    target_velocity = actions - noise
    return noisy_actions, flow_time, target_velocity


def sample_flow_matching_inputs_paired(
    actions: Tensor,
    is_perturbed: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """clean/perturbed 配对行共享同一 (τ, ε)（设计 §六.2：两条监督间的动作差异
    不被随机 flow noise 淹没）。

    配对布局契约（mix_perturb_batch）：perturbed 行 k 的配对 clean 行 = 前一行
    （[c0,p0,c1,p1,...]），故 ``is_perturbed`` 中第 k 个 True 行直接复制前一行的
    (τ, ε)。其余行为独立采样（与 ``sample_flow_matching_inputs`` 语义一致）。
    """
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, sequence, horizon, action_dim]")
    if is_perturbed.ndim != 1 or is_perturbed.shape[0] != actions.shape[0]:
        raise ValueError("is_perturbed must be [batch] bool 且与 actions 同批")
    noise = torch.randn_like(actions)
    flow_time = torch.rand(
        actions.shape[:2],
        device=actions.device,
        dtype=actions.dtype,
    )
    p_idx = torch.nonzero(is_perturbed, as_tuple=False).flatten()
    if p_idx.numel():
        c_idx = p_idx - 1
        if bool((c_idx < 0).any()) or bool(is_perturbed[c_idx].any()):
            raise ValueError(
                "配对布局契约破坏：perturbed 行的前一行必须是 clean 行"
            )
        # 共享 (τ, ε)：复制配对 clean 行（randn 后原地覆盖，无梯度历史）。
        noise[p_idx] = noise[c_idx]
        flow_time[p_idx] = flow_time[c_idx]
    tau = flow_time[:, :, None, None]
    noisy_actions = (1.0 - tau) * noise + tau * actions
    target_velocity = actions - noise
    return noisy_actions, flow_time, target_velocity


def mix_perturb_batch(
    clean: dict[str, Tensor],
    perturbed: dict[str, Tensor],
    p_vision: Tensor,
    m: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """clean [n_clean] 与 perturbed [m] 交错配对 → [B=n_clean+m]（--perturb-data
    混合加载；clean:perturbed 比例由 ``m = round(B·ratio)`` 决定）。

    布局 ``[c0,p0,c1,p1,..., tail]``：paired 段 2m 行（clean 前 m 行与全部
    perturbed 行交错），tail = clean[m:]（不丢行）；``p_vision`` [m, T, N, D]
    为 perturbed 行的视觉 token（与训练路径同构，dtype 以 p_vision 为准）。
    返回 (mixed, is_perturbed [B] bool)——is_perturbed 供
    ``sample_flow_matching_inputs_paired`` 共享 (τ, ε)。
    """
    n_clean = int(clean["actions"].shape[0])
    if not (1 <= m <= n_clean):
        raise ValueError(f"m={m} 需满足 1 ≤ m ≤ clean 行数 {n_clean}")
    if int(perturbed["actions"].shape[0]) != m:
        raise ValueError(
            f"perturbed 行数 {perturbed['actions'].shape[0]} != m={m}"
        )
    mixed: dict[str, Tensor] = {}
    for key, value in clean.items():
        if not isinstance(value, Tensor):
            mixed[key] = value
            continue
        if key in ("vision_tokens", "vision_tokens_st"):
            continue  # 视觉键单独拼接（paired 段用 p_vision）
        if key in perturbed:
            head = torch.stack([value[:m], perturbed[key]], dim=1).flatten(0, 1)
            mixed[key] = torch.cat([head, value[m:]], dim=0)
        elif key == "coords":
            # 全局坐标常量（行无关）：扩展行数保持一致。
            mixed[key] = value[0:1].expand(n_clean + m, -1, -1).contiguous()
        else:
            raise ValueError(
                f"混批契约破坏：perturbed 批缺少键 {key!r}"
            )
    for key in ("vision_tokens", "vision_tokens_st"):
        if key in clean:
            head = torch.stack(
                [clean[key][:m].to(dtype=p_vision.dtype), p_vision], dim=1
            ).flatten(0, 1)
            mixed[key] = torch.cat(
                [head, clean[key][m:].to(dtype=p_vision.dtype)], dim=0
            )
    is_perturbed = torch.zeros(n_clean + m, dtype=torch.bool)
    is_perturbed[1 : 2 * m : 2] = True
    return mixed, is_perturbed


def sample_pair_intervention(
    actions: Tensor,
    partner: Tensor,
    probe_tau_max: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared-source counterfactual flow intervention (2026-08-07 upgrade).

    For each genuine same-state pair (i, j), all non-language flow inputs are
    identical: the probe x = (1-tau)*eps + tau*mid_{ij} with mid_{ij} =
    (a_i + a_j)/2 and a per-pair tau ~ U[0, probe_tau_max].  Only the language
    condition differs, so any velocity difference at the probe is attributable
    to language.  tau=0 is included (x = eps, the canonical shared source);
    tau in (0, 0.5] adds the shared midpoint probe that constrains the
    mid-flow field.  Targets are the linear-FM vector field values
    u_i = (a_i - x) / (1 - tau).
    """
    if actions.ndim != 4:
        raise ValueError("actions must have shape [batch, sequence, horizon, action_dim]")
    if partner.shape != (actions.shape[0],):
        raise ValueError("partner must have shape [batch]")

    batch = actions.shape[0]
    device = actions.device
    dtype = actions.dtype
    shared_noise = torch.randn_like(actions[:, 0])
    # Per-pair shared tau (identical for both partners).
    shared_tau = torch.rand(batch, device=device, dtype=dtype) * probe_tau_max
    mid = torch.zeros_like(actions[:, 0])
    for left in range(batch):
        right = int(partner[left])
        if left < right:
            shared_noise[right] = shared_noise[left]
            shared_tau[right] = shared_tau[left]
            mid[left] = 0.5 * (actions[left, 0] + actions[right, 0])
            mid[right] = mid[left]
    # Pairs without a genuine partner get tau=0 with x=eps (well-defined).
    tau_b = shared_tau[:, None, None]
    probe = (1.0 - tau_b) * shared_noise + tau_b * mid
    denom = (1.0 - tau_b).clamp_min(1e-6)
    target_velocity = (actions[:, 0] - probe) / denom
    return probe, shared_tau, target_velocity


def paired_partner_indices(pair_id: Tensor, instruction_id: Tensor) -> Tensor:
    """Return one different-instruction partner for every batch row."""
    if pair_id.ndim != 1 or instruction_id.shape != pair_id.shape:
        raise ValueError("pair_id and instruction_id must have matching shape [B]")
    partner = torch.empty_like(pair_id, dtype=torch.long)
    for value in torch.unique(pair_id).tolist():
        indices = torch.nonzero(pair_id == value, as_tuple=False).flatten()
        if indices.numel() != 2:
            raise ValueError("every training batch must contain exactly two rows per pair_id")
        first, second = indices[0], indices[1]
        if instruction_id[first] == instruction_id[second]:
            raise ValueError("paired batch rows must use different instruction_id values")
        partner[first] = second
        partner[second] = first
    return partner


def semantic_pair_loss(
    predicted: Tensor,
    target: Tensor,
    partner: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared-source counterfactual pair loss (2026-08-07 upgrade).

    Genuine pairs only (partner[i] > i, each pair counted once):
      abs_loss  = L(v(C_i, x, tau), u_i) + L(v(C_j, x, tau), u_j)
      delta     = L(v(C_i, x, tau) - v(C_j, x, tau), u_i - u_j)
    with x/tau shared within the pair (see sample_pair_intervention), so
    absolute terms pin both endpoints of the language condition and the
    delta term pins the language-caused flow bifurcation.  Samples without a
    genuine partner contribute zero (pair_id=arange MW data keeps pair=0).
    """
    if predicted.ndim != 3 or target.shape != predicted.shape:
        raise ValueError("paired velocities must have shape [batch, horizon, action_dim]")
    left = torch.nonzero(partner > torch.arange(partner.shape[0], device=partner.device), as_tuple=False).flatten()
    if left.numel() == 0:
        return (
            predicted.new_zeros(()),
            predicted.new_zeros(()),
            predicted.new_zeros(()),
        )
    right = partner[left]
    abs_loss = F.smooth_l1_loss(predicted[left], target[left]) + F.smooth_l1_loss(
        predicted[right], target[right]
    )
    delta_loss = F.smooth_l1_loss(
        predicted[left] - predicted[right], target[left] - target[right]
    )
    loss = abs_loss + delta_loss
    predicted_magnitude = (predicted[left] - predicted[right]).detach().abs().mean()
    target_magnitude = (target[left] - target[right]).detach().abs().mean()
    return loss, predicted_magnitude, target_magnitude


def semantic_pair_loss_legacy(
    predicted: Tensor,
    target: Tensor,
    partner: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Old L_pair (ablation): delta-only smooth-L1 on partner differences.

    Kept bit-compatible with the pre-2026-08-07 behavior: full-batch indexing
    through ``partner`` (self-pairs contribute zero deltas).
    """
    if predicted.ndim != 3 or target.shape != predicted.shape:
        raise ValueError("paired velocities must have shape [batch, horizon, action_dim]")
    predicted_delta = predicted - predicted[partner]
    target_delta = target - target[partner]
    loss = F.smooth_l1_loss(predicted_delta, target_delta)
    predicted_magnitude = predicted_delta.detach().abs().mean()
    target_magnitude = target_delta.detach().abs().mean()
    return loss, predicted_magnitude, target_magnitude
