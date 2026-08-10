"""MT-VJ 阶段 V：语言角色度量场（契约 §2，2026-08-10）。

LanguageMetricField：语言角色查询 → 全 patch 分数 + patch 内连续 offset →
连续位置 / 可见度 / 关系状态。训练时冻结 V-JEPA 与 Qwen，只训本模块
（train_metric_visual.py）。本文件同时实现契约 §2 的 MetricFieldOutput /
RelationStateEncoder / MicroRefiner，供阶段 A（train.py）与 §8 测试使用。

设计（artifacts/mt_vj_design.md §2）：
- 分数 s_{r,n} = q_r^T W_K D_n + b_r(t,y,x)，D=W11·H11, G=W5·H5 投影到 D_PROJ；
- 位置 p̂_r = Σ_n softmax(s)(p_n + δ_n)，δ_n = ½tanh(f_offset(D_n, G_n, q_r))；
- heatmap [B, r, 24, 24] = 对 t 轴求和后的 softmax 概率（CE 监督，σ=2px 高斯）；
- visibility：attention 加权特征 → sigmoid；relation ĝ [B, 4]：回归 g*。

坐标约定：``coords`` [1152, 3]（t, y, x）。契约规定 y/x 为 0-1 归一化网格；
``va_compound.live_vjepa._dense_coords()`` 实际返回 [-1, 1]（与 patch 网格
一致），此处自动检测并把 [-1, 1] 映射到 [0, 1]——两种输入都接受，p̂ 恒为
[0, 1] 图像坐标（y, x 序）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# 契约公共常量（与 artifacts/mt_vj_contract.md 一致）
ROLE_NAMES = ("tool", "object", "target", "interface")
N_ROLES = 4
DENSE_TOKENS = 1152  # 2×24×24
H_DIM = 768  # V-JEPA 特征维
D_PROJ = 192  # 投影维
HEATMAP_GRID = 24  # 空间网格（2×24×24 → 24×24 heatmap）


@dataclass
class MetricFieldOutput:
    p: Tensor          # [B, N_ROLES, 2] 连续位置（图像坐标，归一化 0-1，y,x 序）
    visibility: Tensor  # [B, N_ROLES] sigmoid
    offset: Tensor     # [B, N_ROLES, 2] patch 内偏移（诊断）
    heatmap: Tensor    # [B, N_ROLES, 24, 24] 分数图（诊断/可视化；t 求和后的概率）
    relation: Tensor   # [B, 6] ĝ_t = [p_eef−p_obj(2), p_obj−p_target(2), axis_cos, depth_m]
                       # （Codex P0-5 拍板 2026-08-10：统一 6 维，支持显式轴向/深度主张）
    visibility_logits: Tensor  # [B, N_ROLES] BCE 用 logits（诊断/训练，附加字段）
    log_heatmap: Tensor  # [B, N_ROLES, 24, 24] log P（数值稳定，CE 监督用；附加字段）


def _normalize_coords(coords: Tensor) -> Tensor:
    """coords [.., 3]（t, y, x）→ [.., 3]：把 [-1, 1] 的 y/x 映射到 [0, 1]。

    [-1, 1] 网格（_dense_coords）与 0-1 归一化网格（契约）都接受：仅当 y/x
    绝对值最大值 ≤ 1.01 时视为 [-1, 1] 并平移缩放；t 轴原样保留。
    """
    coords = coords.to(torch.float32)
    yx = coords[..., 1:]
    if yx.abs().max() <= 1.01:
        yx = (yx + 1.0) / 2.0
    return torch.cat((coords[..., :1], yx), dim=-1)


class LanguageMetricField(nn.Module):
    """语言角色查询 → 全 patch 分数 + patch 内连续 offset → 连续位置/可见度/关系。

    forward 内所有输入（h5/h11 fp16、language_hidden fp16）统一转 fp32 计算，
    参数保持 fp32；冻结的 V-JEPA/Qwen 输出只作只读 evidence。
    """

    def __init__(self, lang_dim: int = 2048, h_dim: int = H_DIM,
                 d_proj: int = D_PROJ, n_roles: int = N_ROLES) -> None:
        super().__init__()
        self.lang_dim = lang_dim
        self.h_dim = h_dim
        self.d_proj = d_proj
        self.n_roles = n_roles
        self.grid = HEATMAP_GRID

        # 语言 → 角色查询：语言 token 掩码均值池 → 每角色一个 d_proj 查询。
        self.lang_pool = nn.Linear(lang_dim, n_roles * d_proj)
        # 视觉证据投影（D=W11·H11, G=W5·H5）
        self.w_k11 = nn.Linear(h_dim, d_proj)
        self.w_k5 = nn.Linear(h_dim, d_proj)
        # 空间偏置 b_r(t,y,x)：[n_roles, 2, 24, 24]
        self.spatial_bias = nn.Parameter(torch.zeros(n_roles, 2, self.grid, self.grid))
        # 连续 offset：δ_n = ½tanh(f_offset([D_n, G_n, q_r]))
        self.offset_mlp = nn.Sequential(
            nn.Linear(3 * d_proj, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 2),
        )
        # 可见度：Σπ·D + mean D + mean G → 每角色 1 个 logit
        self.vis_mlp = nn.Sequential(
            nn.Linear(3 * d_proj, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 1),
        )
        # 关系状态 ĝ [B, 4]：attention 加权 D/G 特征
        self.rel_mlp = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 6),
        )
        # 初始等价：softmax 前分数由空间偏置决定（0），保证训练前 p̂ ≈ 网格中心。
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        h5: Tensor,
        h11: Tensor,
        language_hidden: Tensor,
        language_mask: Tensor,
        coords: Tensor,
    ) -> MetricFieldOutput:
        """h5/h11: [B, 1152, 768]；language_hidden: [B, L, lang_dim]；
        language_mask: [B, L] bool；coords: [1152, 3]（t,y,x）。"""
        batch = h5.shape[0]
        dtype = torch.float32
        h5 = h5.to(dtype)
        h11 = h11.to(dtype)
        language_hidden = language_hidden.to(dtype)
        coords = _normalize_coords(coords.to(dtype))  # [1152, 3]

        # ---- 角色查询：掩码均值池 → q_r ----
        mask = language_mask.to(dtype)  # [B, L]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        lang_pool = (language_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, lang_dim]
        query = self.lang_pool(lang_pool).view(batch, self.n_roles, self.d_proj)  # [B, r, d]

        # ---- 视觉证据投影 ----
        d11 = self.w_k11(h11)  # [B, 1152, d]
        d5 = self.w_k5(h5)     # [B, 1152, d]

        # ---- 分数 s_{r,n} = q_r·D_n/√d + b_r(t,y,x)（√d 缩放稳定 softmax） ----
        scale = 1.0 / math.sqrt(self.d_proj)
        scores = torch.einsum("brd,bnd->brn", query, d11) * scale  # [B, r, 1152]
        bias = self.spatial_bias.reshape(self.n_roles, -1)  # [r, 1152]
        scores = scores + bias.unsqueeze(0)

        # ---- 连续 offset δ_{r,n} = ½tanh(f_offset([D_n, G_n, q_r])) ----
        offsets = torch.empty(
            batch, self.n_roles, DENSE_TOKENS, 2, dtype=dtype, device=h11.device
        )
        for role in range(self.n_roles):
            cat_in = torch.cat(
                (
                    d11,                                   # [B, 1152, d]
                    d5,                                    # [B, 1152, d]
                    query[:, role : role + 1].expand(-1, DENSE_TOKENS, -1),  # [B, 1152, d]
                ),
                dim=-1,
            )  # [B, 1152, 3d]
            offsets[:, role] = 0.5 * torch.tanh(self.offset_mlp(cat_in)) / self.grid

        # ---- softmax 概率（含 t 轴）→ 位置 + heatmap + log-heatmap ----
        probs = F.softmax(scores, dim=-1)  # [B, r, 1152]
        patch_center = coords[:, 1:]       # [1152, 2]（0-1，y,x）
        expected = patch_center.unsqueeze(0).unsqueeze(0) + offsets  # [B, r, 1152, 2]
        p_hat = (probs.unsqueeze(-1) * expected).sum(dim=2)  # [B, r, 2]
        scores_grid = scores.view(batch, self.n_roles, 2, self.grid, self.grid)
        heatmap = probs.view(batch, self.n_roles, 2, self.grid, self.grid).sum(dim=2)
        # log P(y,x) = logsumexp_t(s) − log Z：数值稳定，CE 监督用（不经概率钳制，
        # 避免 clamp 区梯度为 0 导致 CE 卡死）。
        log_heatmap = torch.logsumexp(scores_grid, dim=2) - torch.logsumexp(
            scores.view(batch, self.n_roles, -1), dim=-1, keepdim=True
        ).unsqueeze(-1)  # [B, r, 24, 24]

        # ---- 可见度 + 关系 ----
        attn_feat = (probs.unsqueeze(-1) * d11.unsqueeze(1)).sum(dim=2)  # [B, r, d] ΣπD
        vis_feat = torch.cat(
            (attn_feat, d11.mean(dim=1, keepdim=True).expand(-1, self.n_roles, -1),
             d5.mean(dim=1, keepdim=True).expand(-1, self.n_roles, -1)),
            dim=-1,
        )  # [B, r, 3d]
        vis_logits = self.vis_mlp(vis_feat).squeeze(-1)  # [B, r]
        rel_feat = torch.cat((attn_feat.mean(dim=1), d5.mean(dim=1)), dim=-1)  # [B, 2d]
        relation = self.rel_mlp(rel_feat)  # [B, 6]（拍板 2A：2D差+axis+depth）

        return MetricFieldOutput(
            p=p_hat,
            visibility=torch.sigmoid(vis_logits),
            offset=offsets.mean(dim=2),  # 诊断：patch 均值偏移
            heatmap=heatmap,
            relation=relation,
            visibility_logits=vis_logits,
            log_heatmap=log_heatmap,
        )


class RelationStateEncoder(nn.Module):
    """g_t + ν_t → 两个 d_model token（z_g, z_nu），加入每层 action cross-attention。

    阶段 A（train.py）集成使用；阶段 V 不训练本模块（checkpoint 契约仍要求
    保存其 state_dict，随机初始化即可，config 中标注 relation_encoder_trained）。
    """

    def __init__(self, state_dim: int = 6, d_model: int = 512) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.g_proj = nn.Linear(state_dim, d_model)
        self.nu_proj = nn.Linear(state_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        # 拍板 3A（2026-08-10）：阶段 V 用重建辅助 loss 监督 z_g 保留 g_t 信息
        # （Codex P1-3：随机冻结的 encoder = 随机线性映射，违背"监督关系 token"）。
        self.recon = nn.Linear(d_model, state_dim)

    def forward(self, g: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
        """g/nu: [B, state_dim] → z_g/z_nu: [B, d_model]（LayerNorm 后）。"""
        z_g = self.norm(self.g_proj(g))
        z_nu = self.norm(self.nu_proj(nu))
        return z_g, z_nu


class MicroRefiner(nn.Module):
    """原像素 ROI 精修（0.5-1M 参数）：[B, 3, roi, roi] → [B, 4]
    （δp_y, δp_x, δz, contact）。阶段 V 后续扩展（同批 simulator 标签训练，
    不进策略 loss）；此处按契约 §2 提供前向形状。"""

    def __init__(self, roi: int = 96) -> None:
        super().__init__()
        self.roi = roi
        channels = (32, 64, 128)
        layers = []
        in_c = 3
        for out_c in channels:
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64), nn.ReLU(inplace=True), nn.Linear(64, 4)
        )

    def forward(self, roi_images: Tensor) -> Tensor:
        """roi_images: [B, 3, roi, roi]（归一化输入）→ [B, 4]。"""
        feat = self.features(roi_images).flatten(1)
        return self.head(feat)
