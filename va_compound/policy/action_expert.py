import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ActionTransformerBlock(nn.Module):
    """Transformer block with Pre-LN:

    1. Action Self-Attention + residual (once)
    2. Additive time modulation before LN (optional/configurable)
    3. Cross-Attention (Action Q, Layer Condition K/V) + residual (once)
    4. FFN + residual (once)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_dim

        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        time_mod: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Args:

            x: [B, H, Dexpert]
            context: [B, H_ctx, Dexpert] (matching layer condition)
            time_mod: [B, Dexpert] or [B, 1, Dexpert] additive modulation
            action_mask: Optional key_padding_mask for self-attention [B, H]
            context_mask: Optional key_padding_mask for cross-attention [B, H_ctx]

        Returns:
            [B, H, Dexpert]
        """
        if time_mod is not None:
            if time_mod.ndim == 2:
                time_mod_expanded = time_mod.unsqueeze(1)
            else:
                time_mod_expanded = time_mod
        else:
            time_mod_expanded = 0.0

        normed_sa = self.self_attn_norm(x + time_mod_expanded)
        sa_out, _ = self.self_attn(
            query=normed_sa,
            key=normed_sa,
            value=normed_sa,
            key_padding_mask=action_mask,
            need_weights=False,
        )
        x = x + sa_out

        normed_ca = self.cross_attn_norm(x + time_mod_expanded)
        ca_out, _ = self.cross_attn(
            query=normed_ca,
            key=context,
            value=context,
            key_padding_mask=context_mask,
            need_weights=False,
        )
        x = x + ca_out

        normed_ffn = self.ffn_norm(x + time_mod_expanded)
        ffn_out = self.ffn(normed_ffn)
        x = x + ffn_out

        return x


class LayerwiseActionExpert(nn.Module):
    """3-layer pre-norm action expert that conditions on 3 layerwise VA tokens.

    Fixed math contract:
    - Input: noisy_actions [B, H, A]
    - Conditions: tuple/list of exactly 3 [B, H, Dva] tensors
    - Time: [B] or [B, 1]
    - Output: [B, H, A] velocity
    - Enforces num_layers == 3
    - Explicit learned action position embeddings up to max_horizon (default 50)
    - Supports horizon slices (e.g. 6, 15, 50) and optional position_offset
    - No global condition mean leaking future horizons
    - No whole Transformer residual nested in outer gate
    """

    def __init__(
        self,
        action_dim: int,
        condition_dim: int,
        hidden_dim: int,
        num_heads: int,
        max_horizon: int = 50,
        num_layers: int = 3,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers != 3:
            raise ValueError(f"num_layers must be exactly 3, got {num_layers}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if max_horizon <= 0:
            raise ValueError(f"max_horizon must be positive, got {max_horizon}")

        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_horizon = max_horizon
        self.num_layers = num_layers

        self.action_projection = nn.Linear(action_dim, hidden_dim)

        self.action_pos_embed = nn.Embedding(max_horizon, hidden_dim)

        self.condition_projections = nn.ModuleList(
            [nn.Linear(condition_dim, hidden_dim) for _ in range(num_layers)]
        )

        half_dim = hidden_dim // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half_dim, dtype=torch.float32)
            / denominator
        )
        self.register_buffer("time_frequencies", frequencies, persistent=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                ActionTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, action_dim)

    def _time_embedding(self, time: Tensor, dtype: torch.dtype) -> Tensor:
        if time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        if time.ndim != 1:
            raise ValueError(f"time must have shape [B] or [B, 1], got {time.shape}")

        angles = time.float()[:, None] * self.time_frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        elif embedding.shape[-1] > self.hidden_dim:
            embedding = embedding[:, : self.hidden_dim]
        return embedding.to(dtype=dtype)

    def forward(
        self,
        noisy_actions: Tensor,
        conditions: Union[Sequence[Tensor], Tuple[Tensor, ...]],
        time: Tensor,
        position_offset: int = 0,
        action_mask: Optional[Tensor] = None,
        context_masks: Optional[Union[Sequence[Optional[Tensor]], Tuple[Optional[Tensor], ...]]] = None,
    ) -> Tensor:
        """Forward pass of LayerwiseActionExpert.

        Args:
            noisy_actions: [B, H, A] noisy actions
            conditions: tuple/list of exactly 3 tensors [B, H_cond, Dva]
            time: [B] or [B, 1] flow/diffusion time
            position_offset: optional integer offset for positional embeddings
            action_mask: optional [B, H] mask for action self-attention
            context_masks: optional tuple/list of masks for each condition layer

        Returns:
            velocity: [B, H, A] action velocity
        """
        if not isinstance(conditions, (tuple, list)):
            raise TypeError(
                f"conditions must be a tuple or list of exactly {self.num_layers} tensors, got {type(conditions)}"
            )
        if len(conditions) != self.num_layers:
            raise ValueError(
                f"Expected exactly {self.num_layers} conditions, got {len(conditions)}"
            )

        if noisy_actions.ndim != 3:
            raise ValueError(
                f"noisy_actions must have shape [B, H, A], got ndim={noisy_actions.ndim} ({noisy_actions.shape})"
            )

        batch_size, horizon, action_dim = noisy_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"noisy_actions dim {action_dim} does not match expected action_dim {self.action_dim}"
            )

        if position_offset < 0:
            raise ValueError(f"position_offset must be non-negative, got {position_offset}")
        if position_offset + horizon > self.max_horizon:
            raise ValueError(
                f"Requested slice [{position_offset}:{position_offset + horizon}] exceeds max_horizon {self.max_horizon}"
            )

        for i, cond in enumerate(conditions):
            if not isinstance(cond, Tensor):
                raise TypeError(f"condition[{i}] must be a Tensor, got {type(cond)}")
            if cond.ndim != 3:
                raise ValueError(
                    f"condition[{i}] must have shape [B, H_cond, Dva], got ndim={cond.ndim} ({cond.shape})"
                )
            if cond.shape[0] != batch_size:
                raise ValueError(
                    f"condition[{i}] batch size {cond.shape[0]} does not match noisy_actions batch size {batch_size}"
                )
            if cond.shape[2] != self.condition_dim:
                raise ValueError(
                    f"condition[{i}] dim {cond.shape[2]} does not match expected condition_dim {self.condition_dim}"
                )

        if context_masks is not None:
            if len(context_masks) != self.num_layers:
                raise ValueError(
                    f"context_masks must have length {self.num_layers}, got {len(context_masks)}"
                )

        x = self.action_projection(noisy_actions)  # [B, H, Dexpert]

        pos_indices = torch.arange(
            position_offset,
            position_offset + horizon,
            dtype=torch.long,
            device=noisy_actions.device,
        )
        pos_emb = self.action_pos_embed(pos_indices).unsqueeze(0)  # [1, H, Dexpert]
        x = x + pos_emb

        raw_t_emb = self._time_embedding(time, dtype=noisy_actions.dtype)  # [B, Dexpert]
        t_mod = self.time_mlp(raw_t_emb)  # [B, Dexpert]

        x = x + t_mod.unsqueeze(1)

        for layer_idx in range(self.num_layers):
            block = self.blocks[layer_idx]
            cond_proj = self.condition_projections[layer_idx]
            cond_layer = cond_proj(conditions[layer_idx])  # [B, H_cond, Dexpert]

            ctx_mask = context_masks[layer_idx] if context_masks is not None else None
            x = block(
                x,
                context=cond_layer,
                time_mod=t_mod,
                action_mask=action_mask,
                context_mask=ctx_mask,
            )

        x = self.output_norm(x)
        velocity = self.velocity_head(x)  # [B, H, A]

        return velocity
