"""C²-IRF v2 multi-layer output and high-frequency residual heads.

Building blocks for the value-side K/V decomposition (design doc
``artifacts/c2irf_v2_vision_ablation.md`` §5):

    V_t = P_V([H_t^11, R_t^5, delta_t H_t^11, phi(p)])
    R_t^5 = P_5(H_t^5) - Up(Pool(P_5(H_t^5)))

This module only provides the trainable components and concat helper; wiring
happens at the Step 1/2 integration site. V-JEPA weights stay frozen and the
H5/H11 activations are read-only evidence. Memory is negligible
(``[B, 1152, 768]`` x 2 ~= 3.4 MiB/sample in FP16).

Class <-> contract mapping (checked statically by name during integration):

- ``high_freq_residual(H5, grid=(24, 24))`` -> ``HighFreqResidual(in_dim,
  aux_dim=128, grid=(24, 24))(H5)`` (grid is a constructor attribute, not a
  per-call argument);
- ``residual_value_concat(H11, R5)`` -> ``ResidualValueConcat(h11_dim,
  value_dim)(H11, R5)``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HighFreqResidual(nn.Module):
    """High-frequency residual R5 from H5: project, downsample, upsample, subtract.

    ``P_5`` linearly projects H5 ``[B, N, in_dim]`` to ``aux_dim`` (default 128).
    Each time slice is reshaped to a 2D grid and 2x2 average-pooled (24x24 ->
    12x12), then nearest-neighbor upsampled back to ``grid`` (``Up`` is the
    adjoint of ``Pool``: each pooled value is replicated to its 2x2 cell) and
    subtracted element-wise. A constant spatial field has zero residual while
    edges/texture are preserved, so the residual is used purely as value
    enrichment, not as the sub-centimeter localization mechanism (design doc §5).

    Output ``[B, N, aux_dim]`` with ``N = t_grid * h * w``; token order matches
    the dense-readout t->y->x grid (aligned with ``live_vjepa._dense_coords``).
    """

    def __init__(
        self,
        in_dim: int,
        aux_dim: int = 128,
        grid: tuple[int, int] = (24, 24),
    ) -> None:
        super().__init__()
        if in_dim < 1 or aux_dim < 1:
            raise ValueError("in_dim/aux_dim must be positive")
        h, w = grid
        if min(h, w) < 1:
            raise ValueError("grid dimensions must be positive")
        if h % 2 or w % 2:
            raise ValueError("grid must be even in both dimensions for a 2×2 pool")
        self.grid = (h, w)
        self.proj = nn.Linear(in_dim, aux_dim)  # P_5

    def forward(self, H5: Tensor) -> Tensor:
        if H5.ndim != 3:
            raise ValueError(
                f"H5 must have shape [B, N, in_dim], got {tuple(H5.shape)}"
            )
        h, w = self.grid
        spatial = h * w
        if H5.shape[1] % spatial:
            raise ValueError(
                f"token count {H5.shape[1]} is not a multiple of the spatial "
                f"grid {self.grid}"
            )
        batch = H5.shape[0]
        t_grid = H5.shape[1] // spatial
        projected = self.proj(H5)  # [B, N, aux_dim]
        view = projected.reshape(batch, t_grid, h, w, -1).permute(0, 1, 4, 2, 3)
        pooled = F.avg_pool2d(
            view.reshape(batch * t_grid, -1, h, w), kernel_size=2, stride=2
        )
        up = F.interpolate(pooled, size=(h, w), mode="nearest")  # Up(Pool(·))
        up = up.reshape(batch, t_grid, -1, h, w).permute(0, 1, 3, 4, 2)
        return projected - up.reshape(batch, -1, projected.shape[-1])


class ResidualValueConcat(nn.Module):
    """H11 projection concatenated with R5 to form the V channel.

    Completes the first half of ``V_t = P_V([H_t^11, R_t^5, delta_t H_t^11,
    phi(p)])``: ``P_V(H11)`` (``h11_dim`` -> ``value_dim``) concatenated with
    ``R5``. The temporal delta ``delta_t H_t^11`` and coordinate ``phi(p)`` are
    appended by the caller at the Step 1/2 integration site. Output
    ``[B, N, value_dim + R5.shape[-1]]``.
    """

    def __init__(self, h11_dim: int = 768, value_dim: int = 128) -> None:
        super().__init__()
        if h11_dim < 1 or value_dim < 1:
            raise ValueError("h11_dim/value_dim must be positive")
        self.proj = nn.Linear(h11_dim, value_dim)  # P_V

    def forward(self, H11: Tensor, R5: Tensor) -> Tensor:
        if H11.ndim != 3 or R5.ndim != 3:
            raise ValueError("H11/R5 must have shape [B, N, D]")
        if H11.shape[:2] != R5.shape[:2]:
            raise ValueError(
                f"H11 {tuple(H11.shape[:2])} and R5 {tuple(R5.shape[:2])} "
                f"must share the batch/token axes"
            )
        return torch.cat((self.proj(H11), R5), dim=-1)
