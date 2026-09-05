"""Dual-tower cross-attention fusion module.

Implements bidirectional cross-attention fusion between vision and language
representations at fixed snapshot depth, preserving native dimensions and residuals.
"""

from typing import Tuple
import torch
import torch.nn as nn


class DualTowerFusionPair(nn.Module):
    """Bidirectional cross-attention fusion between vision and language towers.

    Contract:
      - Inputs:
          vision: [B, N, Dv] (float tensor)
          language: [B, L, Dl] (float tensor)
          language_mask: [B, L] (bool tensor, True where valid token, False where padding)
      - Both modalities are normalized via native LayerNorms and projected to hidden_dim.
      - Both cross-attentions read the ORIGINAL snapshot (simultaneous bidirectional fusion).
      - Vision reads language (key_padding_mask=~language_mask).
      - Language reads vision (vision has no padding tokens).
      - Outputs are projected back to native widths via zero-initialized linear projections
        and added to the original residual tensors.
      - Masked language padding positions preserve their original input representation exactly.
      - All-padding language rows reject with ValueError to prevent NaN in softmax.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if min(vision_dim, language_dim, hidden_dim, num_heads) < 1:
            raise ValueError("dimensions and num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.vision_dim = int(vision_dim)
        self.language_dim = int(language_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.vision_norm = nn.LayerNorm(self.vision_dim)
        self.language_norm = nn.LayerNorm(self.language_dim)

        self.vision_proj = nn.Linear(self.vision_dim, self.hidden_dim)
        self.language_proj = nn.Linear(self.language_dim, self.hidden_dim)

        self.v2l_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.l2v_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )

        self.vision_out_proj = nn.Linear(self.hidden_dim, self.vision_dim)
        self.language_out_proj = nn.Linear(self.hidden_dim, self.language_dim)

        nn.init.zeros_(self.vision_out_proj.weight)
        nn.init.zeros_(self.vision_out_proj.bias)
        nn.init.zeros_(self.language_out_proj.weight)
        nn.init.zeros_(self.language_out_proj.bias)

    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass performing simultaneous bidirectional cross-attention.

        Args:
            vision: [B, N, Dv] float tensor.
            language: [B, L, Dl] float tensor.
            language_mask: [B, L] bool tensor where True indicates valid tokens and
                           False indicates padding tokens.

        Returns:
            new_vision: [B, N, Dv] float tensor.
            new_language: [B, L, Dl] float tensor.
        """
        if vision.dim() != 3:
            raise ValueError(f"vision must be 3D [B, N, Dv], got shape {tuple(vision.shape)}")
        if language.dim() != 3:
            raise ValueError(f"language must be 3D [B, L, Dl], got shape {tuple(language.shape)}")
        if language_mask.dim() != 2:
            raise ValueError(f"language_mask must be 2D [B, L], got shape {tuple(language_mask.shape)}")

        B, N, Dv = vision.shape
        Bl, L, Dl = language.shape
        Bm, Lm = language_mask.shape

        if B != Bl or B != Bm:
            raise ValueError(
                f"Batch dimensions mismatch: vision B={B}, language B={Bl}, language_mask B={Bm}"
            )
        if L != Lm:
            raise ValueError(
                f"Sequence length mismatch: language L={L}, language_mask L={Lm}"
            )
        if Dv != self.vision_dim:
            raise ValueError(
                f"vision feature dim mismatch: expected {self.vision_dim}, got {Dv}"
            )
        if Dl != self.language_dim:
            raise ValueError(
                f"language feature dim mismatch: expected {self.language_dim}, got {Dl}"
            )
        if language_mask.dtype != torch.bool:
            raise ValueError(
                f"language_mask must be torch.bool, got {language_mask.dtype}"
            )

        valid_per_row = language_mask.sum(dim=-1)
        if (valid_per_row == 0).any():
            invalid_indices = (valid_per_row == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"All-padding language row(s) detected at indices {invalid_indices}. "
                "Each row must have at least one valid language token."
            )

        v_orig = vision
        l_orig = language

        v_h = self.vision_proj(self.vision_norm(v_orig))
        l_h = self.language_proj(self.language_norm(l_orig))

        lang_key_padding_mask = ~language_mask

        v_delta_h, _ = self.v2l_attn(
            query=v_h,
            key=l_h,
            value=l_h,
            key_padding_mask=lang_key_padding_mask,
            need_weights=False,
        )
        new_vision = v_orig + self.vision_out_proj(v_delta_h)

        l_delta_h, _ = self.l2v_attn(
            query=l_h,
            key=v_h,
            value=v_h,
            need_weights=False,
        )
        l_delta = self.language_out_proj(l_delta_h)

        mask_expanded = language_mask.unsqueeze(-1)
        new_language = torch.where(mask_expanded, l_orig + l_delta, l_orig)

        return new_vision, new_language


class MultiLayerDualTowerFusion(nn.Module):
    """Container owning multiple DualTowerFusionPair modules."""

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_pairs: int = 6,
    ) -> None:
        super().__init__()
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        self.num_pairs = int(num_pairs)
        self.pairs = nn.ModuleList([
            DualTowerFusionPair(
                vision_dim=vision_dim,
                language_dim=language_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(self.num_pairs)
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> DualTowerFusionPair:
        return self.pairs[index]

    def forward_pair(
        self,
        index: int,
        vision: torch.Tensor,
        language: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass using the fusion pair at index with bounds checking."""
        if not (0 <= index < self.num_pairs):
            raise IndexError(
                f"Pair index {index} out of bounds for MultiLayerDualTowerFusion with {self.num_pairs} pairs"
            )
        return self.pairs[index](vision=vision, language=language, language_mask=language_mask)
