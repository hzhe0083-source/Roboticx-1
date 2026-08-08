"""Language-programmed local control slots (PULSE-VA round 1).

Two-stage readout replacing ``mean(64 flat tokens) -> button PCA``:

1. ``LanguageRoleCompiler``: fixed role seeds ``R [K, hidden]`` + a gated
   language cross-attention instance ``Q_L = R + g_L * CrossAttn(R, H_L)``
   (``g_L`` init ``sigma(-2) ~ 0.12`` so role identity survives early
   training).  Computed once per instruction and cached.
2. ``LocalControlSlotReader``: ``Q_L`` cross-attends dense spatiotemporal
   V-JEPA tokens ``[B, N, 768]`` with a zero-init Fourier coordinate bypass
   (reader-side metric coordinates only; frozen V-JEPA untouched), producing
   ``slots [B, K, 768]``, attention weights and slot centers ``[B, K, 3]``
   (normalized t/y/x).
3. ``RelationTokens``: three explicit relation tokens (EO/OT/IC) built from
   slot pairs and their center differences, giving the VA nominal path an
   explicit relative-geometry channel.

No new training losses: everything is supervised by the existing action loss
(L_FM / L_act).  Stage A runs slots + nominal direct head; the C² control
chart (P_slot over g_t) is fit in Stage B from recovery data.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ROLE_NAMES = (
    "actor_tool",
    "manipuland",
    "target",
    "interface",
    "constraint",
    "phase",
)
N_SLOTS = len(ROLE_NAMES)


def fourier_encode(coords: torch.Tensor, num_bands: int = 4) -> torch.Tensor:
    """[N, 3] normalized coords -> [N, 3 + 2*3*num_bands] Fourier features."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N, 3], got {tuple(coords.shape)}")
    bands = 2.0 ** torch.arange(num_bands, device=coords.device, dtype=coords.dtype)
    # [N, 3, B]
    scaled = coords.unsqueeze(-1) * bands.view(1, 1, -1)
    feats = torch.cat(
        [torch.sin(scaled), torch.cos(scaled)], dim=-1
    ).reshape(coords.shape[0], 3 * 2 * num_bands)
    return torch.cat([coords, feats], dim=-1)


class LanguageRoleCompiler(nn.Module):
    """Fixed role seeds instantiated by the instruction (one-shot, cached)."""

    def __init__(
        self,
        hidden_dim: int,
        language_dim: int,
        n_role: int = N_SLOTS,
        num_heads: int = 8,
        gate_init: float = -2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.n_role = n_role
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        # Learned role seeds: identity of each slot (actor/manipuland/...).
        self.role_seeds = nn.Parameter(
            torch.empty(n_role, hidden_dim).normal_(0.0, 0.02, generator=gen)
        )
        # Language side: layer-0 flat key -> hidden space.
        self.lang_proj = nn.Linear(language_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    def set_role_seeds(self, seeds: torch.Tensor) -> None:
        """Optional init from frozen-Qwen role-description embeddings [K, hidden]."""
        if tuple(seeds.shape) != (self.n_role, self.role_seeds.shape[-1]):
            raise ValueError(
                f"role seeds must be [{self.n_role}, {self.role_seeds.shape[-1]}], "
                f"got {tuple(seeds.shape)}"
            )
        with torch.no_grad():
            self.role_seeds.copy_(seeds.to(dtype=self.role_seeds.dtype))

    def set_role_description_embeddings(self, desc_emb: torch.Tensor) -> None:
        """Init from raw Qwen role-description embeddings [K, language_dim]:
        project through lang_proj (language -> hidden) then copy as seeds."""
        if tuple(desc_emb.shape) != (self.n_role, self.lang_proj.in_features):
            raise ValueError(
                f"desc embeddings must be [{self.n_role}, {self.lang_proj.in_features}], "
                f"got {tuple(desc_emb.shape)}"
            )
        with torch.no_grad():
            projected = self.lang_proj(
                desc_emb.to(
                    dtype=self.lang_proj.weight.dtype,
                    device=self.lang_proj.weight.device,
                )
            )
            self.role_seeds.copy_(projected)

    def forward(
        self, language_key: torch.Tensor, language_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[B, L, language_dim] -> [B, K, hidden] instantiated role queries."""
        batch = language_key.shape[0]
        keys = self.lang_proj(language_key)  # [B, L, hidden]
        query = self.query_norm(self.role_seeds.unsqueeze(0).expand(batch, -1, -1))
        delta, _ = self.cross_attn(
            query,
            keys,
            keys,
            key_padding_mask=(
                ~language_mask.bool() if language_mask is not None else None
            ),
        )
        gate = torch.sigmoid(self.gate_logit)
        return query + gate * delta


class LocalControlSlotReader(nn.Module):
    """Role queries -> dense V-JEPA tokens with zero-init coordinate bypass.

    Outputs slots [B, K, 768] (vision dim), per-head mean attention weights
    [B, K, N] and slot centers [B, K, 3] (normalized t/y/x in [-1, 1]).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        hidden_dim: int = 512,
        num_slots: int = N_SLOTS,
        num_heads: int = 8,
        pos_dim: int = 27,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        # Reader-side metric coordinates; zero-init keeps the frozen-feature
        # behavior at initialization (coordinate channel silent until trained).
        self.pos_proj = nn.Linear(pos_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.pos_proj.weight)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.read_gate_logit = nn.Parameter(torch.tensor(gate_init))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.to_vision = nn.Linear(hidden_dim, vision_dim)

    def forward(
        self,
        dense_tokens: torch.Tensor,  # [B, N, vision_dim]
        role_queries: torch.Tensor,  # [B, K, hidden_dim]
        coords: torch.Tensor,        # [N, 3] normalized t/y/x
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = fourier_encode(coords.to(dtype=dense_tokens.dtype)).to(
            device=dense_tokens.device, dtype=dense_tokens.dtype
        )
        visual = self.vision_proj(self.vision_norm(dense_tokens))
        visual = visual + self.pos_proj(pos)[None]  # zero-init -> identity at start
        delta, weights = self.cross_attn(
            self.query_norm(role_queries),
            visual,
            visual,
            need_weights=True,
            average_attn_weights=False,
        )
        gate = torch.sigmoid(self.read_gate_logit)
        slots = role_queries + gate * delta
        slots = slots + self.ffn(self.output_norm(slots))
        slots = self.to_vision(slots)  # [B, K, vision_dim]
        weights = weights.float().mean(dim=1)  # [B, K, N]
        centers = torch.einsum("bkn,nc->bkc", weights, coords.float())  # [B, K, 3]
        return slots, weights, centers


class RelationTokens(nn.Module):
    """EO / OT / IC relation tokens from slot pairs + center differences."""

    def __init__(self, vision_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * vision_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_vision = nn.Linear(hidden_dim, vision_dim)

    def forward(
        self,
        slots: torch.Tensor,   # [B, K, vision_dim]
        centers: torch.Tensor, # [B, K, 3]
    ) -> torch.Tensor:
        # Indices: 0=actor, 1=manipuland, 2=target, 3=interface.
        pairs = (
            ((0, 1), (1, 2), (3, 2)),  # EO, OT, IC
        )[0]
        rels = []
        for i, j in pairs:
            feat = torch.cat(
                [slots[:, i], slots[:, j], centers[:, i] - centers[:, j]], dim=-1
            )
            rels.append(self.to_vision(self.pair_mlp(feat)))  # [B, vision_dim]
        return torch.stack(rels, dim=1)  # [B, 3, vision_dim]


def build_va_vision_input(
    coarse: torch.Tensor,  # [B, n_coarse, vision_dim] (pooled context)
    slots: torch.Tensor,   # [B, K, vision_dim]
    relations: torch.Tensor,  # [B, 3, vision_dim]
) -> torch.Tensor:
    """25-token VA vision stream: [coarse; slots; relations]."""
    return torch.cat([coarse, slots, relations], dim=1)
