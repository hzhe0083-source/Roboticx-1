"""WAM4VA: world-action model for a VA policy.

WAM supplies future/world information; VA still emits the only action.
Implementation name was WMRM (residual modulation). Public name is WAM4VA.
No candidate actions, no Δv head.

Each handshake:
    evidence = CA(queries, vision)              # spatial readout
    innov = evidence - predict(belief)          # innovation
    innov = innov - Proj_{prev}(innov)          # same-round dedup
    belief = belief + CA(belief, innov)
    z_spans = world heads on action segments    # temporal structure
    C_b, C_g, C_p = separate CA to belief/geo/progress
    A' = A + q * (Σ π_j U_j + g_b C_b + g_g C_g + g_p C_p)

q and source gates are zero-init, so A' ≡ A at step 0.
world_goal is sealed. Futures only enter ``wmrm_world_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WMRMAux:
    z_hat: Tensor
    z_spans: Tensor
    pi: Tensor
    gate: Tensor
    progress: Tensor
    belief: Tensor
    innovation: Tensor
    task_summary: Tensor


class _CrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        batch, n_q, dim = query.shape
        n_k = key_value.shape[1]
        heads = self.num_heads
        q = self.q(query).view(batch, n_q, heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(batch, n_k, heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).view(batch, n_k, heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(batch, n_q, dim)
        return self.o(out)


class WAM4VA(nn.Module):
    """World-action model that only modulates VA's action stream."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        world_dim: int = 8,
        rank: int = 4,
        proprio_dim: int = 9,
        mixer_dropout: float = 0.3,
        num_heads: int = 4,
        n_belief: int = 8,
        n_evidence: int = 8,
        n_spans: int = 3,
        n_progress: int = 4,
        n_task_queries: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if world_dim < 1:
            raise ValueError("world_dim must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")
        if proprio_dim < 1:
            raise ValueError("proprio_dim must be positive")
        if not 0.0 <= mixer_dropout < 1.0:
            raise ValueError("mixer_dropout must be in [0, 1)")
        if n_spans < 1 or n_belief < 1 or n_evidence < 1 or n_progress < 1:
            raise ValueError("belief/evidence/span/progress sizes must be positive")
        if n_task_queries < 1:
            raise ValueError("n_task_queries must be positive")
        self.hidden_dim = hidden_dim
        self.world_dim = world_dim
        self.rank = rank
        self.proprio_dim = proprio_dim
        self.mixer_dropout = mixer_dropout
        self.n_belief = n_belief
        self.n_evidence = n_evidence
        self.n_spans = n_spans
        self.n_progress = n_progress
        self.n_task_queries = n_task_queries

        self.belief_tokens = nn.Parameter(torch.zeros(n_belief, hidden_dim))
        self.evidence_queries = nn.Parameter(torch.empty(n_evidence, hidden_dim))
        nn.init.normal_(self.evidence_queries, std=0.02)
        self.evidence_read = _CrossAttn(hidden_dim, num_heads)
        self.evidence_from_belief = nn.Linear(hidden_dim, hidden_dim)
        self.belief_write = _CrossAttn(hidden_dim, num_heads)

        self.world_from_action = nn.Linear(hidden_dim, hidden_dim)
        self.world_from_state = nn.Linear(proprio_dim, hidden_dim)
        self.world_from_belief = nn.Linear(hidden_dim, hidden_dim)
        self.task_queries = nn.Parameter(torch.empty(n_task_queries, hidden_dim))
        nn.init.normal_(self.task_queries, std=0.02)
        self.task_attention = _CrossAttn(hidden_dim, num_heads)
        self.world_from_task = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.world_from_task.weight)
        nn.init.zeros_(self.world_from_task.bias)
        self.mix_stage = nn.Linear(hidden_dim, rank)
        nn.init.zeros_(self.mix_stage.weight)
        nn.init.zeros_(self.mix_stage.bias)
        self.span_heads = nn.ModuleList(nn.Linear(hidden_dim, world_dim) for _ in range(n_spans))
        self.progress_head = nn.Linear(hidden_dim * 2, n_progress)

        self.geo_to_token = nn.Linear(world_dim, hidden_dim)
        self.progress_to_token = nn.Linear(n_progress, hidden_dim)
        self.ca_belief = _CrossAttn(hidden_dim, num_heads)
        self.ca_geo = _CrossAttn(hidden_dim, num_heads)
        self.ca_progress = _CrossAttn(hidden_dim, num_heads)
        self.source_gates = nn.Parameter(torch.zeros(3))

        context_dim = hidden_dim + hidden_dim + proprio_dim
        self.basis = nn.Linear(context_dim, rank * hidden_dim)
        self.mix_world = nn.Linear(world_dim, rank)
        self.mix_action = nn.Linear(hidden_dim, rank)
        self.gate_proj = nn.Linear(world_dim, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        self.innov_overlap = 0.5

    def has_action_shaped_head(self, action_dim: int) -> bool:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.out_features == action_dim:
                return True
        return False

    def _project_out(self, current: Tensor, previous: Tensor) -> Tensor:
        """Remove prev only when cosine overlap exceeds ``innov_overlap``."""
        flat_prev = previous.reshape(previous.shape[0], -1)
        flat_cur = current.reshape(current.shape[0], -1)
        cosine = (
            F.normalize(flat_cur, dim=-1) * F.normalize(flat_prev, dim=-1)
        ).sum(dim=-1, keepdim=True)
        denom = flat_prev.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        coeff = (flat_cur * flat_prev).sum(dim=-1, keepdim=True) / denom
        drop = (cosine.abs() > self.innov_overlap).to(dtype=current.dtype)
        while drop.ndim < current.ndim:
            drop = drop.unsqueeze(-1)
        return current - drop * (coeff * flat_prev).view_as(current)

    def _z_per_step(self, z_spans: Tensor, horizon: int) -> Tensor:
        n_spans = z_spans.shape[1]
        step_ids = (
            torch.arange(horizon, device=z_spans.device, dtype=torch.long) * n_spans
        ) // max(horizon, 1)
        return z_spans[:, step_ids.clamp(max=n_spans - 1), :]

    def _pi(
        self,
        z_per_step: Tensor,
        action: Tensor,
        task_summary: Tensor | None = None,
    ) -> Tensor:
        world_logits = F.dropout(
            self.mix_world(z_per_step), p=self.mixer_dropout, training=self.training
        )
        action_logits = F.dropout(
            self.mix_action(action), p=self.mixer_dropout, training=self.training
        )
        logits = world_logits * action_logits
        if task_summary is not None:
            logits = logits + self.mix_stage(task_summary)[:, None, :]
        return torch.softmax(logits, dim=-1)

    def _segment_means(self, action: Tensor) -> list[Tensor]:
        horizon = action.shape[1]
        spans = []
        for index in range(self.n_spans):
            start = index * horizon // self.n_spans
            end = (index + 1) * horizon // self.n_spans
            if end <= start:
                end = start + 1
            spans.append(action[:, start:end].mean(dim=1))
        return spans

    def forward(
        self,
        action: Tensor,
        vision: Tensor,
        proprio: Tensor,
        *,
        belief: Tensor | None = None,
        prev_innovation: Tensor | None = None,
        language_keys: Tensor | None = None,
        world_goal: Tensor | None = None,
    ) -> tuple[Tensor, WMRMAux, Tensor, Tensor]:
        if world_goal is not None:
            raise ValueError(
                "world_goal is sealed: realized futures must not enter the "
                "forward mixer; pass them only to wmrm_world_loss"
            )
        if action.ndim != 3 or action.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"action must be [B, H, {self.hidden_dim}], got {tuple(action.shape)}"
            )
        if vision.ndim != 3 or vision.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"vision must be [B, N, {self.hidden_dim}], got {tuple(vision.shape)}"
            )
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        if action.shape[0] != vision.shape[0] or action.shape[0] != proprio.shape[0]:
            raise ValueError("action, vision, and proprio batch sizes must match")

        batch, horizon, hidden = action.shape
        queries = self.evidence_queries[None].expand(batch, -1, -1)
        evidence = self.evidence_read(queries, vision.detach())
        if belief is None:
            belief = self.belief_tokens[None].expand(batch, -1, -1)
        elif belief.shape != (batch, self.n_belief, hidden):
            raise ValueError(
                f"belief must be [B, {self.n_belief}, {hidden}], got {tuple(belief.shape)}"
            )
        predicted = self.evidence_from_belief(belief)
        innovation = evidence - predicted
        if prev_innovation is not None:
            innovation = self._project_out(innovation, prev_innovation)
        belief = belief + self.belief_write(belief, innovation)

        if language_keys is None:
            task_summary = torch.zeros(batch, hidden, device=action.device, dtype=action.dtype)
        else:
            if language_keys.ndim != 3 or language_keys.shape[0] != batch:
                raise ValueError(
                    f"language_keys must be [B, T, {hidden}], got {tuple(language_keys.shape)}"
                )
            if language_keys.shape[-1] != hidden:
                raise ValueError(
                    f"language_keys last dim must be {hidden}, got {language_keys.shape[-1]}"
                )
            task_tokens = self.task_attention(
                self.task_queries[None].expand(batch, -1, -1),
                language_keys.to(dtype=action.dtype),
            )
            task_summary = task_tokens.mean(dim=1)

        state = proprio.to(dtype=action.dtype)
        belief_pool = belief.mean(dim=1)
        task_cond = self.world_from_task(task_summary)
        span_latents = []
        for segment, head in zip(self._segment_means(action), self.span_heads):
            fused = torch.tanh(
                self.world_from_action(segment)
                + self.world_from_state(state)
                + self.world_from_belief(belief_pool)
                + task_cond
            )
            span_latents.append(head(fused))
        z_spans = torch.stack(span_latents, dim=1)
        z_hat = z_spans.mean(dim=1)
        progress = self.progress_head(torch.cat((belief_pool, task_summary), dim=-1))

        geo_tokens = self.geo_to_token(z_spans)
        progress_tokens = self.progress_to_token(progress)[:, None, :]
        source = torch.stack(
            (
                self.ca_belief(action, belief),
                self.ca_geo(action, geo_tokens),
                self.ca_progress(action, progress_tokens),
            ),
            dim=0,
        )
        gates = (1.0 + torch.tanh(self.source_gates)).view(3, 1, 1, 1)
        sourced = (gates * source).sum(dim=0)

        context = torch.cat(
            (
                action,
                evidence.mean(dim=1, keepdim=True).expand(-1, horizon, -1),
                state[:, None, :].expand(-1, horizon, -1),
            ),
            dim=-1,
        )
        bases = F.normalize(
            self.basis(context).view(batch, horizon, self.rank, hidden),
            dim=-1,
        )
        z_per_step = self._z_per_step(z_spans, horizon)
        pi = self._pi(z_per_step, action, task_summary)
        mixed = (pi.unsqueeze(-1) * bases).sum(dim=2) + sourced
        gate = torch.tanh(self.gate_proj(z_hat))
        updated = action + gate.unsqueeze(-1) * mixed
        aux = WMRMAux(
            z_hat=z_hat,
            z_spans=z_spans,
            pi=pi,
            gate=gate.detach(),
            progress=progress,
            belief=belief,
            innovation=innovation,
            task_summary=task_summary,
        )
        return updated, aux, belief, innovation

    def pi_shuffle_kl(
        self,
        action: Tensor,
        vision: Tensor,
        proprio: Tensor,
    ) -> Tensor:
        """KL(π(z_hat) || π(shuffle z_hat)). Near 0 means mixer ignores the world."""
        _, aux, _, _ = self.forward(action, vision, proprio)
        return self.pi_kl_from_aux(action, aux)

    def pi_kl_from_aux(self, action: Tensor, aux: WMRMAux) -> Tensor:
        perm = torch.randperm(aux.z_spans.shape[0], device=aux.z_spans.device)
        z_real = self._z_per_step(aux.z_spans, action.shape[1])
        z_shuf = self._z_per_step(aux.z_spans[perm], action.shape[1])
        pi_real = self._pi(z_real, action, aux.task_summary).clamp_min(1e-8)
        pi_shuf = self._pi(z_shuf, action, aux.task_summary).clamp_min(1e-8)
        return (pi_real * (pi_real.log() - pi_shuf.log())).sum(dim=-1).mean()


WorldMediatedResidualModulation = WAM4VA


def wmrm_world_loss(z_hat: Tensor, z_future: Tensor) -> Tensor:
    """Supervise world prediction. ``z_future`` is loss-side only (stop-grad)."""
    if z_hat.shape != z_future.shape:
        raise ValueError(
            f"z_hat/z_future shape mismatch: {tuple(z_hat.shape)} vs {tuple(z_future.shape)}"
        )
    return F.mse_loss(z_hat, z_future.detach())


def action_dependency_scores(
    state: Tensor,
    action: Tensor,
    target: Tensor,
    *,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Linear probe: target ~ state vs state+action vs state+shuffled action."""
    if state.ndim != 2 or action.ndim != 2 or target.ndim != 2:
        raise ValueError("state, action, target must be [N, D] matrices")
    if state.shape[0] != action.shape[0] or state.shape[0] != target.shape[0]:
        raise ValueError("row counts must match")
    n = int(state.shape[0])
    if n < 10:
        raise ValueError("need at least 10 rows")
    split = max(n - max(n // 5, 1), 1)

    def _fit_mse(features: Tensor) -> float:
        train_x = features[:split]
        test_x = features[split:]
        train_y = target[:split]
        test_y = target[split:]
        gram = train_x.T @ train_x
        gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        weights = torch.linalg.solve(gram, train_x.T @ train_y)
        pred = test_x @ weights
        return float((pred - test_y).square().mean().item())

    ones = torch.ones(n, 1, device=state.device, dtype=state.dtype)
    gen = torch.Generator(device=state.device)
    gen.manual_seed(0)
    perm = torch.randperm(n, generator=gen)
    shuffled = action[perm]
    return {
        "mse_state": _fit_mse(torch.cat((ones, state), dim=1)),
        "mse_state_action": _fit_mse(torch.cat((ones, state, action), dim=1)),
        "mse_state_shuffled": _fit_mse(torch.cat((ones, state, shuffled), dim=1)),
    }
