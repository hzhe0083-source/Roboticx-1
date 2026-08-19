"""WAM4VA: world-action model for a VA policy.

WAM predicts the next-cycle last-frame DINO map and writes that spatial
world into VA's action stream. VA/FM remain the only action emitter.

Handshake:
    A' = A + q * mixed(world_tokens, A)
    q is zero-init, so A' ≡ A at step 0.

World prediction never reads VA's latent A. It reads the logged env-action
chunk (full cycle, not a mean) plus the T-frame DINO clip. Futures only
enter ``wmrm_world_loss``.
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
    evidence: Tensor
    proprio: Tensor
    z_tokens: Tensor | None = None
    dino_tokens: Tensor | None = None
    env_action: Tensor | None = None
    world_tokens: Tensor | None = None
    vision_gate: Tensor | None = None
    # Belief state exactly at the predictor input. ``belief`` above is the
    # post-prediction/handshake state and is therefore not valid for an
    # action-counterfactual re-evaluation of the final World stage.
    predict_belief: Tensor | None = None


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

    def zero_output(self) -> None:
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)


class _WorldPredictorBlock(nn.Module):
    """One V-JEPA-2-AC-style latent block: causal SA + cond CA + FFN."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("predictor width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.sa_o = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cq = nn.Linear(dim, dim)
        self.ck = nn.Linear(dim, dim)
        self.cv = nn.Linear(dim, dim)
        self.ca_o = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def _heads(self, tensor: Tensor) -> Tensor:
        batch, tokens, dim = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, tensor: Tensor) -> Tensor:
        batch, heads, tokens, head_dim = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, tokens, heads * head_dim)

    def forward(self, tokens: Tensor, cond: Tensor, causal_mask: Tensor) -> Tensor:
        qkv = self.qkv(self.norm1(tokens))
        query, key, value = qkv.chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(query),
            self._heads(key),
            self._heads(value),
            attn_mask=causal_mask,
        )
        tokens = tokens + self.sa_o(self._merge(attended))
        query = self._heads(self.cq(self.norm2(tokens)))
        key = self._heads(self.ck(cond))
        value = self._heads(self.cv(cond))
        crossed = F.scaled_dot_product_attention(query, key, value)
        tokens = tokens + self.ca_o(self._merge(crossed))
        return tokens + self.ff(self.norm3(tokens))


class _DeepWorldPredictor(nn.Module):
    """Full-grid spatiotemporal residual predictor. Initial output copies last frame."""

    def __init__(
        self,
        dino_dim: int,
        *,
        width: int,
        depth: int,
        num_heads: int,
        map_frames: int,
        map_size: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("predictor depth must be positive")
        if width < 1:
            raise ValueError("predictor width must be positive")
        self.dino_dim = dino_dim
        self.width = width
        self.map_frames = map_frames
        self.map_size = map_size
        self.in_proj = nn.Linear(dino_dim, width)
        self.time_embed = nn.Embedding(map_frames, width)
        self.row_embed = nn.Embedding(map_size, width)
        self.col_embed = nn.Embedding(map_size, width)
        self.blocks = nn.ModuleList(
            _WorldPredictorBlock(width, num_heads) for _ in range(depth)
        )
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, dino_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _causal_mask(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        patches = self.map_size * self.map_size
        frames = torch.arange(self.map_frames, device=device).repeat_interleave(patches)
        allowed = frames[None, :] <= frames[:, None]
        mask = torch.zeros(
            self.map_frames * patches,
            self.map_frames * patches,
            device=device,
            dtype=dtype,
        )
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def forward(self, clip: Tensor, cond: Tensor, previous_map: Tensor | None = None) -> Tensor:
        """Predict/refine endpoint map; ``previous_map`` is a differentiable stage state."""
        if clip.ndim != 5:
            raise ValueError(f"clip must be [B, T, D, H, W], got {tuple(clip.shape)}")
        batch, frames, dim, height, width = clip.shape
        if (frames, dim, height, width) != (
            self.map_frames,
            self.dino_dim,
            self.map_size,
            self.map_size,
        ):
            raise ValueError(
                "clip shape must be "
                f"[B, {self.map_frames}, {self.dino_dim}, {self.map_size}, {self.map_size}], "
                f"got {tuple(clip.shape)}"
            )
        predictor_clip = clip
        if previous_map is not None:
            if previous_map.shape != (batch, dim, height, width):
                raise ValueError(f"previous_map shape mismatch: {tuple(previous_map.shape)}")
            predictor_clip = torch.cat((clip[:, :-1], previous_map[:, None]), dim=1)
        tokens = predictor_clip.permute(0, 1, 3, 4, 2).reshape(
            batch, frames * height * width, dim
        )
        hidden = self.in_proj(tokens)
        time = torch.arange(frames, device=clip.device).repeat_interleave(height * width)
        rows = torch.arange(height, device=clip.device).repeat_interleave(width).repeat(frames)
        cols = torch.arange(width, device=clip.device).repeat(height * frames)
        hidden = hidden + self.time_embed(time) + self.row_embed(rows) + self.col_embed(cols)
        mask = self._causal_mask(clip.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, cond, mask)
        last = hidden[:, -height * width :]
        residual = self.out_proj(self.out_norm(last))
        delta = residual.view(batch, height, width, dim).permute(0, 3, 1, 2)
        base = clip[:, -1] if previous_map is None else previous_map
        return base + delta


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
        cycle_steps: int = 6,
        condition_on_action: bool = True,
        dino_dim: int | None = None,
        map_size: int = 16,
        map_channels: int = 32,
        map_frames: int = 4,
        map_grid: int = 16,
        env_action_dim: int = 7,
        world_grid: int = 16,
        predictor: str = "legacy",
        predictor_depth: int = 6,
        predictor_width: int = 384,
        predictor_heads: int = 12,
        max_stages: int = 8,
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
        if cycle_steps < 1:
            raise ValueError("cycle_steps must be positive")
        if world_grid < 1:
            raise ValueError("world_grid must be positive")
        if predictor not in {"legacy", "st_blocks"}:
            raise ValueError("predictor must be legacy|st_blocks")
        if predictor_depth < 1:
            raise ValueError("predictor_depth must be positive")
        if predictor_width < 1:
            raise ValueError("predictor_width must be positive")
        if predictor_heads < 1:
            raise ValueError("predictor_heads must be positive")
        if max_stages < 1:
            raise ValueError("max_stages must be positive")
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
        self.cycle_steps = cycle_steps
        # Kept for CLI/config compatibility. World prediction never reads VA A.
        self.condition_on_action = condition_on_action
        self.dino_dim = dino_dim
        self.world_grid = world_grid
        self.predictor = predictor
        self.predictor_depth = predictor_depth
        self.predictor_width = predictor_width
        self.predictor_heads = predictor_heads
        self.max_stages = max_stages
        self.stage_embed = nn.Embedding(max_stages, hidden_dim)
        nn.init.zeros_(self.stage_embed.weight)

        self.belief_tokens = nn.Parameter(torch.zeros(n_belief, hidden_dim))
        self.evidence_queries = nn.Parameter(torch.empty(n_evidence, hidden_dim))
        nn.init.normal_(self.evidence_queries, std=0.02)
        self.evidence_read = _CrossAttn(hidden_dim, num_heads)
        self.evidence_from_belief = nn.Linear(hidden_dim, hidden_dim)
        self.belief_write = _CrossAttn(hidden_dim, num_heads)
        self.belief_from_world = _CrossAttn(hidden_dim, num_heads)
        self.belief_from_world.zero_output()
        self.vision_from_world = _CrossAttn(hidden_dim, num_heads)
        self.legacy_ungated_vision = False
        self.vision_gate_proj = nn.Linear(world_dim, 1)
        nn.init.zeros_(self.vision_gate_proj.weight)
        nn.init.zeros_(self.vision_gate_proj.bias)

        self.world_from_env = nn.Linear(hidden_dim, hidden_dim)
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
        if map_size < 1 or map_channels < 1 or map_frames < 1 or map_grid < 1:
            raise ValueError("map_size/channels/frames/grid must be positive")
        self.map_size = map_size
        self.map_channels = map_channels
        self.map_frames = map_frames
        self.map_grid = map_grid
        if env_action_dim < 1:
            raise ValueError("env_action_dim must be positive")
        self.env_action_dim = env_action_dim
        self.env_step = nn.Linear(env_action_dim, hidden_dim)
        self.env_time = nn.Parameter(torch.zeros(cycle_steps, hidden_dim))
        self.env_seq = nn.Linear(cycle_steps * env_action_dim, hidden_dim)
        if dino_dim is not None:
            if dino_dim < 1:
                raise ValueError("dino_dim must be positive")
            self.cond_to_dino = nn.Linear(hidden_dim, dino_dim)
            self.dino_pred = nn.Linear(dino_dim, dino_dim)
            nn.init.zeros_(self.dino_pred.weight)
            nn.init.zeros_(self.dino_pred.bias)
            self.token_readout = nn.Linear(dino_dim, world_dim)
            self.dino_to_hid = nn.Linear(dino_dim, hidden_dim)
            self.hid_to_dino = nn.Linear(hidden_dim, dino_dim)
            nn.init.zeros_(self.hid_to_dino.weight)
            nn.init.zeros_(self.hid_to_dino.bias)
            self.action_read = _CrossAttn(hidden_dim, num_heads)
            self.action_read.zero_output()
            self.map_readout = nn.Linear(hidden_dim, world_dim)
            if predictor == "legacy":
                self.frame_spatial_dw = nn.Conv2d(
                    dino_dim, dino_dim, kernel_size=3, padding=1, groups=dino_dim
                )
                self.frame_spatial_pw = nn.Conv2d(dino_dim, dino_dim, kernel_size=1)
                self.temporal_mix = nn.Parameter(torch.zeros(dino_dim, map_frames))
                with torch.no_grad():
                    self.temporal_mix[:, -1] = 1.0
                self.film_scale = nn.Linear(hidden_dim, dino_dim)
                self.film_shift = nn.Linear(hidden_dim, dino_dim)
                nn.init.zeros_(self.film_scale.weight)
                nn.init.zeros_(self.film_shift.weight)
                nn.init.ones_(self.film_scale.bias)
                nn.init.zeros_(self.film_shift.bias)
                self.map_dw1 = nn.Conv2d(
                    dino_dim, dino_dim, kernel_size=3, padding=1, groups=dino_dim
                )
                self.map_pw1 = nn.Conv2d(dino_dim, dino_dim, kernel_size=1)
                self.map_dw2 = nn.Conv2d(
                    dino_dim, dino_dim, kernel_size=3, padding=1, groups=dino_dim
                )
                self.map_pw2 = nn.Conv2d(dino_dim, dino_dim, kernel_size=1)
                nn.init.zeros_(self.map_pw2.weight)
                nn.init.zeros_(self.map_pw2.bias)
            else:
                self.frame_spatial_dw = None
                self.frame_spatial_pw = None
                self.temporal_mix = None
                self.film_scale = None
                self.film_shift = None
                self.map_dw1 = None
                self.map_pw1 = None
                self.map_dw2 = None
                self.map_pw2 = None
            self.z_query = nn.Parameter(torch.empty(1, world_dim))
            nn.init.normal_(self.z_query, std=0.02)
            self.z_read = _CrossAttn(world_dim, num_heads)
            self.proposal_to_pred = nn.Linear(hidden_dim, predictor_width)
            self.belief_to_pred = nn.Linear(hidden_dim, predictor_width)
            self.fused_to_pred = nn.Linear(hidden_dim, predictor_width)
            self.st_predictor = (
                _DeepWorldPredictor(
                    dino_dim,
                    width=predictor_width,
                    depth=predictor_depth,
                    num_heads=predictor_heads,
                    map_frames=map_frames,
                    map_size=map_size,
                )
                if predictor == "st_blocks"
                else None
            )
        else:
            self.cond_to_dino = None
            self.dino_pred = None
            self.token_readout = None
            self.frame_spatial_dw = None
            self.frame_spatial_pw = None
            self.temporal_mix = None
            self.dino_to_hid = None
            self.hid_to_dino = None
            self.action_read = None
            self.film_scale = None
            self.film_shift = None
            self.map_dw1 = None
            self.map_pw1 = None
            self.map_dw2 = None
            self.map_pw2 = None
            self.map_readout = None
            self.z_query = None
            self.z_read = None
            self.proposal_to_pred = None
            self.belief_to_pred = None
            self.fused_to_pred = None
            self.st_predictor = None
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
        drop = (cosine > self.innov_overlap).to(dtype=current.dtype)
        while drop.ndim < current.ndim:
            drop = drop.unsqueeze(-1)
        return current - drop * (coeff * flat_prev).view_as(current)

    def _span_ids(self, horizon: int, device: torch.device) -> Tensor:
        n_spans = self.n_spans
        return (
            torch.arange(horizon, device=device, dtype=torch.long) * n_spans
        ) // max(horizon, 1)

    def _segment_means(self, action: Tensor) -> list[Tensor]:
        used = min(self.cycle_steps, action.shape[1])
        action = action[:, :used]
        horizon = action.shape[1]
        ids = self._span_ids(horizon, action.device)
        spans = []
        for index in range(self.n_spans):
            mask = ids == index
            if not bool(mask.any()):
                mask = ids == ids[-1]
            spans.append(action[:, mask].mean(dim=1))
        return spans

    def _z_per_step(self, z_hat: Tensor, horizon: int) -> Tensor:
        """Broadcast the supervised z_hat; do not feed unsupervised span residuals."""
        if z_hat.ndim != 2:
            raise ValueError(f"z_hat must be [B, G], got {tuple(z_hat.shape)}")
        return z_hat[:, None, :].expand(-1, horizon, -1)

    def _pi(
        self,
        z_per_step: Tensor,
        action: Tensor,
        task_summary: Tensor | None = None,
        *,
        apply_dropout: bool | None = None,
    ) -> Tensor:
        use_dropout = self.training if apply_dropout is None else apply_dropout
        world_logits = F.dropout(
            self.mix_world(z_per_step), p=self.mixer_dropout, training=use_dropout
        )
        action_logits = F.dropout(
            self.mix_action(action), p=self.mixer_dropout, training=use_dropout
        )
        logits = world_logits * action_logits
        if task_summary is not None:
            logits = logits + self.mix_stage(task_summary)[:, None, :]
        return torch.softmax(logits, dim=-1)

    def encode_env_action(self, env_action: Tensor | None, *, batch: int, dtype: torch.dtype, device: torch.device) -> tuple[Tensor | None, Tensor]:
        """Return (step tokens [B, C, H], ordered flat cond [B, H])."""
        if env_action is None:
            return None, torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        if env_action.ndim != 3:
            raise ValueError(
                f"env_action must be [B, C, A], got {tuple(env_action.shape)}"
            )
        expected = (batch, self.cycle_steps, self.env_action_dim)
        if tuple(env_action.shape) != expected:
            raise ValueError(
                f"env_action must have exact shape {expected}, got {tuple(env_action.shape)}"
            )
        steps = env_action.to(device=device, dtype=dtype)
        tokens = self.env_step(steps) + self.env_time.to(dtype=dtype)
        flat = self.env_seq(steps.flatten(1))
        return tokens, flat

    def _world_condition(
        self,
        proprio: Tensor,
        belief: Tensor,
        task_summary: Tensor,
        env_tokens: Tensor | None,
        env_flat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = proprio.to(dtype=env_flat.dtype)
        belief_pool = belief.mean(dim=1)
        task_cond = self.world_from_task(task_summary)
        if env_tokens is None:
            segments = [
                torch.zeros(
                    env_flat.shape[0],
                    self.hidden_dim,
                    device=env_flat.device,
                    dtype=env_flat.dtype,
                )
                for _ in range(self.n_spans)
            ]
        else:
            segments = self._segment_means(env_tokens)
        span_latents = []
        fused_sum = env_flat.new_zeros(env_flat.shape[0], self.hidden_dim)
        for segment, head in zip(segments, self.span_heads):
            fused = torch.tanh(
                self.world_from_env(segment)
                + self.world_from_state(state)
                + self.world_from_belief(belief_pool)
                + task_cond
                + env_flat
            )
            span_latents.append(head(fused))
            fused_sum = fused_sum + fused
        z_spans = torch.stack(span_latents, dim=1)
        progress = self.progress_head(torch.cat((belief_pool, task_summary), dim=-1))
        return fused_sum / max(self.n_spans, 1), z_spans, progress, belief_pool

    def encode_dino_clip(self, dino_tokens: Tensor) -> Tensor | None:
        """All T frames as [B, T, D, map_size, map_size]. Untrained reshape/pool."""
        if (
            self.dino_dim is None
            or dino_tokens.ndim != 3
            or dino_tokens.shape[-1] != self.dino_dim
        ):
            return None
        batch = dino_tokens.shape[0]
        grid, dim = self.map_grid, self.dino_dim
        patches = grid * grid
        expected = self.map_frames * patches
        if dino_tokens.shape[1] != expected:
            return None
        frames = dino_tokens.view(batch, self.map_frames, grid, grid, dim)
        clip = frames.permute(0, 1, 4, 2, 3).contiguous()
        if (grid, grid) == (self.map_size, self.map_size):
            return clip
        pooled = F.adaptive_avg_pool2d(clip.flatten(0, 1), (self.map_size, self.map_size))
        return pooled.view(batch, self.map_frames, dim, self.map_size, self.map_size)

    def encode_dino_map(self, dino_tokens: Tensor) -> Tensor | None:
        """Last-frame DINO grid [B, D, map_size, map_size]. Loss target stays this."""
        clip = self.encode_dino_clip(dino_tokens)
        if clip is None:
            return None
        return clip[:, -1]

    def encode_world_tokens(self, z_map: Tensor) -> Tensor:
        """Handshake tokens [B, H*W, hidden]. Keep native 16x16; do not pool it down."""
        if z_map.ndim != 4:
            raise ValueError(f"z_map must be [B, D, H, W], got {tuple(z_map.shape)}")
        height, width = z_map.shape[-2], z_map.shape[-1]
        if (height, width) != (self.world_grid, self.world_grid) and (
            height > self.world_grid or width > self.world_grid
        ):
            z_map = F.adaptive_avg_pool2d(z_map, (self.world_grid, self.world_grid))
        tokens = z_map.flatten(2).transpose(1, 2)
        return self.dino_to_hid(tokens)

    def _proposal_tokens(self, action: Tensor) -> Tensor:
        cycle = min(self.cycle_steps, action.shape[1])
        return action[:, :cycle]

    def predict_world(
        self,
        action: Tensor,
        proprio: Tensor,
        belief: Tensor,
        task_summary: Tensor,
        dino_tokens: Tensor | None = None,
        env_action: Tensor | None = None,
        previous_map: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Predict the next DINO map from configured executable actions, not VA hidden A."""
        if env_action is None:
            env_tokens, env_flat = None, action.new_zeros(action.shape[0], self.hidden_dim)
        else:
            env_tokens, env_flat = self.encode_env_action(
                env_action,
                batch=action.shape[0],
                dtype=action.dtype,
                device=action.device,
            )
        fused, z_spans, progress, _ = self._world_condition(
            proprio, belief.detach(), task_summary.detach(), env_tokens, env_flat
        )
        if dino_tokens is None or (
            self.st_predictor is None and self.map_dw1 is None and self.dino_pred is None
        ):
            return z_spans.mean(dim=1), z_spans, progress, None
        if dino_tokens.ndim != 3 or dino_tokens.shape[0] != proprio.shape[0]:
            raise ValueError(
                f"dino_tokens must be [B, N, dino_dim], got {tuple(dino_tokens.shape)}"
            )
        if dino_tokens.shape[-1] != self.dino_dim:
            raise ValueError(
                f"dino_tokens last dim must be {self.dino_dim}, got {dino_tokens.shape[-1]}"
            )
        clip = self.encode_dino_clip(dino_tokens)
        if clip is not None and self.st_predictor is not None:
            cond_parts = [
                self.belief_to_pred(belief.detach()),
                self.fused_to_pred(fused)[:, None, :],
            ]
            if env_tokens is not None:
                cond_parts.insert(0, self.proposal_to_pred(env_tokens))
            cond = torch.cat(cond_parts, dim=1)
            if self.training:
                z_map = torch.utils.checkpoint.checkpoint(
                    self.st_predictor,
                    clip,
                    cond,
                    previous_map,
                    use_reentrant=False,
                )
            else:
                z_map = self.st_predictor(clip, cond, previous_map)
            world_tokens = self.encode_world_tokens(z_map)
            kv = self.map_readout(world_tokens)
            query = self.z_query[None].expand(action.shape[0], -1, -1)
            z_hat = self.z_read(query, kv).squeeze(1)
            return z_hat, z_spans, progress, z_map
        if clip is not None and self.map_dw1 is not None:
            batch, frames, dim, height, width = clip.shape
            per_frame = []
            for time in range(frames):
                frame = clip[:, time]
                per_frame.append(
                    torch.relu(self.frame_spatial_pw(self.frame_spatial_dw(frame)))
                )
            stacked = torch.stack(per_frame, dim=1)
            mix = self.temporal_mix.to(dtype=stacked.dtype)
            context = torch.einsum("dt,btdhw->bdhw", mix, stacked)
            last_frame = clip[:, -1] if previous_map is None else previous_map
            spatial = context.flatten(2).transpose(1, 2)
            hid = self.dino_to_hid(spatial)
            if env_tokens is not None:
                hid = hid + self.action_read(hid, env_tokens)
            context = context + self.hid_to_dino(hid).transpose(1, 2).view(
                batch, dim, height, width
            )
            scale = self.film_scale(fused)[:, :, None, None]
            shift = self.film_shift(fused)[:, :, None, None]
            conditioned = context * scale + shift
            hidden = torch.relu(self.map_pw1(self.map_dw1(conditioned)))
            hidden = hidden * scale + shift
            delta = self.map_pw2(self.map_dw2(hidden))
            z_map = last_frame + delta + shift
            world_tokens = self.encode_world_tokens(z_map)
            kv = self.map_readout(world_tokens)
            query = self.z_query[None].expand(batch, -1, -1)
            z_hat = self.z_read(query, kv).squeeze(1)
            return z_hat, z_spans, progress, z_map
        tokens = dino_tokens.to(dtype=belief.dtype)
        cond = self.cond_to_dino(fused)
        z_tokens = tokens + self.dino_pred(torch.tanh(tokens + cond[:, None, :]))
        kv = self.token_readout(z_tokens)
        query = self.z_query[None].expand(proprio.shape[0], -1, -1)
        z_hat = self.z_read(query, kv).squeeze(1)
        return z_hat, z_spans, progress, z_tokens

    def mixed_residual(
        self,
        action: Tensor,
        z_hat: Tensor,
        task_summary: Tensor,
        evidence: Tensor,
        belief: Tensor,
        proprio: Tensor,
        progress: Tensor,
        *,
        apply_dropout: bool | None = None,
        world_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (mixed, gate, pi). Geo source is spatial world tokens when present."""
        horizon = action.shape[1]
        hidden = action.shape[-1]
        state = proprio.to(dtype=action.dtype)
        if world_tokens is None:
            geo_tokens = self.geo_to_token(z_hat)[:, None, :]
        else:
            geo_tokens = world_tokens
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
            self.basis(context).view(action.shape[0], horizon, self.rank, hidden),
            dim=-1,
        )
        pi = self._pi(
            self._z_per_step(z_hat, horizon),
            action,
            task_summary,
            apply_dropout=apply_dropout,
        )
        mixed = (pi.unsqueeze(-1) * bases).sum(dim=2) + sourced
        gate = torch.tanh(self.gate_proj(z_hat))
        return mixed, gate, pi

    def fm_condition_hinge(
        self,
        action: Tensor,
        aux: WMRMAux,
        action_norm: nn.Module,
        *,
        margin: float = 0.05,
    ) -> Tensor:
        """Forces z into FM input; signed shift so q=0 has nonzero gate_proj grad."""
        perm = torch.randperm(aux.z_hat.shape[0], device=aux.z_hat.device)
        z = aux.z_hat
        z_shuf = aux.z_hat[perm]
        world = aux.world_tokens
        world_shuf = None if world is None else world[perm]
        mixed, gate, _ = self.mixed_residual(
            action,
            z,
            aux.task_summary,
            aux.evidence,
            aux.belief,
            aux.proprio,
            aux.progress,
            apply_dropout=False,
            world_tokens=world,
        )
        mixed_alt, gate_alt, _ = self.mixed_residual(
            action,
            z_shuf,
            aux.task_summary,
            aux.evidence,
            aux.belief,
            aux.proprio,
            aux.progress,
            apply_dropout=False,
            world_tokens=world_shuf,
        )
        cond = action_norm(action + gate.unsqueeze(-1) * mixed)
        cond_alt = action_norm(action + gate_alt.unsqueeze(-1) * mixed_alt)
        delta_m = (mixed - mixed_alt).flatten(1)
        d_m = (delta_m / (delta_m.norm(dim=-1, keepdim=True) + 1e-6)).detach()
        shift = ((cond - cond_alt).flatten(1) * d_m).sum(dim=-1).mean()
        return torch.relu(cond.new_tensor(margin) - shift)

    def action_dep_hinge(
        self,
        z_hat: Tensor,
        z_hat_shuf: Tensor,
        z_future: Tensor,
        *,
        margin: float = 0.05,
    ) -> Tensor:
        real = F.mse_loss(z_hat, z_future.detach())
        shuf = F.mse_loss(z_hat_shuf, z_future.detach())
        return torch.relu(z_hat.new_tensor(margin) - (shuf - real))

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
        dino_tokens: Tensor | None = None,
        env_action: Tensor | None = None,
        reuse_aux: WMRMAux | None = None,
        stage_index: int = 0,
        previous_map: Tensor | None = None,
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
        # === DIAGNOSTIC: gradient norm prints ===
        if self.training:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    norm = param.grad.norm().item()
                    if norm > 1e4:
                        print(f"DIAG: {name} grad norm BEFORE mixed_residual: {norm:.2e}", flush=True)
        # === END DIAG ===

        queries = self.evidence_queries[None].expand(batch, -1, -1)
        evidence = self.evidence_read(queries, vision)
        if belief is None:
            belief = self.belief_tokens[None].expand(batch, -1, -1)
        elif belief.shape != (batch, self.n_belief, hidden):
            raise ValueError(
                f"belief must be [B, {self.n_belief}, {hidden}], got {tuple(belief.shape)}"
            )
        stage = int(stage_index)
        if not 0 <= stage < self.max_stages:
            raise ValueError(f"stage_index must be in [0, {self.max_stages}), got {stage}")
        belief = belief + self.stage_embed.weight[stage]
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

        if reuse_aux is not None:
            z_hat = reuse_aux.z_hat
            z_spans = reuse_aux.z_spans
            progress = reuse_aux.progress
            z_tokens = reuse_aux.z_tokens
            world_tokens = reuse_aux.world_tokens
            predict_belief = (
                reuse_aux.predict_belief
                if reuse_aux.predict_belief is not None
                else reuse_aux.belief
            )
        else:
            predict_belief = belief.detach()
            z_hat, z_spans, progress, z_tokens = self.predict_world(
                action,
                proprio,
                belief,
                task_summary,
                dino_tokens=dino_tokens,
                env_action=env_action,
                previous_map=previous_map,
            )
            world_tokens = None
            if (
                z_tokens is not None
                and z_tokens.ndim == 4
                and self.dino_to_hid is not None
            ):
                world_tokens = self.encode_world_tokens(z_tokens)
        if world_tokens is not None:
            belief = belief + self.belief_from_world(belief, world_tokens)
        elif self.dino_to_hid is not None:
            # Intermediate VA↔WM exchanges use the recurrent WM belief as
            # spatially addressable memory; the final layer replaces it with
            # the predicted DINO map tokens.
            world_tokens = belief
        # === DIAGNOSTIC: after predict_world ===
        if self.training:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    norm = param.grad.norm().item()
                    if norm > 1e4:
                        print(f"DIAG: {name} grad norm AFTER predict_world: {norm:.2e}", flush=True)
        # === END DIAG ===

        mixed, gate, pi = self.mixed_residual(
            action,
            z_hat,
            task_summary,
            evidence,
            belief,
            proprio,
            progress,
            world_tokens=world_tokens,
        )
        updated = action + gate.unsqueeze(-1) * mixed
        vision_gate = torch.tanh(self.vision_gate_proj(z_hat))
        aux = WMRMAux(
            z_hat=z_hat,
            z_spans=z_spans,
            pi=pi,
            gate=gate.detach(),
            progress=progress,
            belief=belief,
            innovation=innovation,
            task_summary=task_summary,
            evidence=evidence,
            proprio=proprio.to(dtype=action.dtype),
            z_tokens=z_tokens,
            dino_tokens=None if dino_tokens is None else dino_tokens.to(dtype=action.dtype),
            env_action=(
                None
                if env_action is None
                else env_action.to(device=action.device, dtype=action.dtype)
            ),
            world_tokens=world_tokens,
            vision_gate=vision_gate,
            predict_belief=predict_belief,
        )
        return updated, aux, belief, innovation

    def mix_world_into_vision(self, vision: Tensor, aux: WMRMAux) -> Tensor:
        """WM → next VA: gated spatial world message, zero at initialization."""
        if aux.world_tokens is None:
            return vision
        message = self.vision_from_world(vision, aux.world_tokens)
        if self.legacy_ungated_vision:
            return vision + message
        gate = aux.vision_gate
        if gate is None:
            return vision
        return vision + gate.unsqueeze(-1) * message

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
        perm = torch.randperm(aux.z_hat.shape[0], device=aux.z_hat.device)
        z_real = self._z_per_step(aux.z_hat, action.shape[1])
        z_shuf = self._z_per_step(aux.z_hat[perm], action.shape[1])
        pi_real = self._pi(
            z_real, action, aux.task_summary, apply_dropout=False
        ).clamp_min(1e-8)
        pi_shuf = self._pi(
            z_shuf, action, aux.task_summary, apply_dropout=False
        ).clamp_min(1e-8)
        return (pi_real * (pi_real.log() - pi_shuf.log())).sum(dim=-1).mean()


WorldMediatedResidualModulation = WAM4VA


def matched_no_fixed_point_perm(
    task_id: Tensor | None,
    eye_mean: Tensor,
    proprio: Tensor,
) -> Tensor:
    """Same-task nearest assignment: a derangement when B>=2.

    Nearest-neighbor alone can map many rows onto one donor. This returns a
    bijection with no fixed points, preferring low distance and same task.
    """
    if eye_mean.shape[0] != proprio.shape[0]:
        raise ValueError("eye_mean and proprio batch sizes must match")
    batch = int(eye_mean.shape[0])
    device = eye_mean.device
    if batch <= 1:
        return torch.arange(batch, device=device, dtype=torch.long)
    if task_id is not None and int(task_id.shape[0]) != batch:
        raise ValueError("task_id batch size must match eye_mean")

    eye_flat = eye_mean.reshape(batch, -1).to(dtype=torch.float32)
    prop_flat = proprio.reshape(batch, -1).to(dtype=torch.float32)
    feat = torch.cat(
        (F.normalize(eye_flat, dim=-1), F.normalize(prop_flat, dim=-1)),
        dim=-1,
    )
    cost = torch.cdist(feat, feat)
    finite = cost.isfinite()
    span = (
        cost[finite].max() - cost[finite].min()
        if bool(finite.any())
        else torch.tensor(1.0, device=device)
    )
    big = (span.abs() + 1.0) * 10.0
    cost = torch.where(finite, cost, big)
    cost.fill_diagonal_(float("inf"))
    if task_id is not None:
        same = task_id[:, None] == task_id[None, :]
        cost = torch.where(same, cost, cost + big)

    remaining = set(range(batch))
    perm = [-1] * batch
    for index in range(batch):
        if not remaining:
            break
        order = torch.argsort(cost[index]).tolist()
        partner = None
        for cand in order:
            if cand in remaining and cand != index:
                partner = cand
                break
        if partner is None:
            continue
        perm[index] = partner
        remaining.remove(partner)
    leftover = list(remaining)
    for index, partner in enumerate(perm):
        if partner < 0:
            perm[index] = leftover.pop(0)
    for index, partner in enumerate(perm):
        if partner != index:
            continue
        swap = 0 if index != 0 else 1
        perm[index], perm[swap] = perm[swap], perm[index]
    return torch.tensor(perm, device=device, dtype=torch.long)


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
