"""C²-IRF v2 Step 4：多层输出与高频残差（H⁵/H¹¹ residual head）。

设计文档（artifacts/c2irf_v2_vision_ablation.md §五）：K/V 分解的 value 侧

    V_t = P_V([H_t¹¹, R_t⁵, Δ_t H_t¹¹, φ(p)])
    R_t⁵ = P_5(H_t⁵) − Up(Pool(P_5(H_t⁵)))

本模块只提供可训练部件与拼接帮助函数，不接线（Step 1/2 集成时消费）；
V-JEPA 权重保持冻结（backbones.VJEPA21Backbone.freeze_all），H⁵/H¹¹ 只作
只读 evidence。显存（§九）：[B, 1152, 768] × 2 ≈ 3.38 MiB/sample（FP16），
可忽略。

契约伪代码与类名的对应（集成时静态检查以类名为准）：

- ``high_freq_residual(H5, grid=(24, 24))`` → ``HighFreqResidual(in_dim,
  aux_dim=128, grid=(24, 24))(H5)``（grid 是 token 网格的构造属性，不是
  每次调用的参数）；
- ``residual_value_concat(H11, R5)`` → ``ResidualValueConcat(h11_dim,
  value_dim)(H11, R5)``。
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HighFreqResidual(nn.Module):
    """P_5 → 2×2 avg pool → 上采样 → 差值：H⁵ 的高频残差 R⁵。

    ``P_5`` 把 H⁵ [B, N, in_dim] 线性投影到 ``aux_dim``（默认 128）；每个
    时间片按 2D 网格做 2×2 avg pool（24×24 → 12×12），最近邻上采样回
    ``grid``（Up 是 Pool 的伴随：每个池化值复制到其 2×2 胞），逐元素相减。
    常数空间场残差为 0，边缘/纹理保留——高频残差只是 value enrichment，
    不再被包装成亚厘米定位机制（设计文档 §五）。

    输出 [B, N, aux_dim]，N = t_grid × h × w，token 顺序与 dense 读出同一
    t→y→x 网格（live_vjepa._dense_coords 对齐）。
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
    """H¹¹ 投影 + R⁵ 拼接 → V 通道（Step 1/2 消费；本次只提供部件不接线）。

    设计公式 V_t = P_V([H_t¹¹, R_t⁵, Δ_t H_t¹¹, φ(p)])：本部件完成
    P_V(H¹¹)（``h11_dim`` → ``value_dim``）+ concat(R⁵) 的 V 通道前半段；
    时间差 Δ_tH¹¹ 与坐标 φ(p) 由 Step 1/2 在调用侧追加。输出
    [B, N, value_dim + R5.shape[-1]]。
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
