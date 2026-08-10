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
    offset_full: Tensor | None = None  # [B, N_ROLES, 1152, 2] 逐 patch offset（直接监督用）
    scores: Tensor | None = None  # [B, N_ROLES, 1152] 原始分数（hinge 监督用；附加字段）


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
                 d_proj: int = D_PROJ, n_roles: int = N_ROLES,
                 l2_norm: bool = False, learnable_temp: bool = False,
                 temp_init: float = 10.0, freeze_bias: bool = False,
                 mode_readout: bool = False) -> None:
        """v3（2026-08-10，探针实证后）：``mode_readout`` 用模式读出替代全网格
        期望读出——heatmap（片求和）全局峰 + 局部 5×5 soft-argmax（+峰 patch
        的 offset）。实测 V-JEPA h11 余弦面近乎全平（576 片中 575 片相似度
        >0.5×max），全网格期望 ≈ 均匀分布 → 预测钉死网格质心（40-80px）；
        argmax / 模式读出在同一查询下即达 8-10px（scripts/diag_probe_oracle.py
        与 diag_trained_linear_probe.py 实证）。v1 默认参数行为逐字节不变。"""
        super().__init__()
        self.lang_dim = lang_dim
        self.h_dim = h_dim
        self.d_proj = d_proj
        self.n_roles = n_roles
        self.grid = HEATMAP_GRID
        self.l2_norm = l2_norm
        self.learnable_temp = learnable_temp
        self.freeze_bias = freeze_bias
        self.temp_init = temp_init
        self.mode_readout = mode_readout

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
        if self.learnable_temp:
            # 可学习温度：s = temp·cos(q,d)（l2_norm 时）或 temp·(q·d/√d)（纯 v1）
            self.temperature = nn.Parameter(torch.tensor(float(temp_init)))
        if self.freeze_bias:
            self.spatial_bias.requires_grad_(False)

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

        # ---- 分数 s_{r,n}（v1: q·d/√d；v2: L2 归一化 + 温度 → cosine 相似度）----
        if self.l2_norm:
            qn = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            dn = d11 / d11.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            temp = (
                self.temperature
                if self.learnable_temp
                else torch.tensor(self.temp_init, dtype=dtype, device=h11.device)
            )
            scores = torch.einsum("brd,bnd->brn", qn, dn) * temp  # [B, r, 1152]
        else:
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
        scores_grid = scores.view(batch, self.n_roles, 2, self.grid, self.grid)
        heatmap = probs.view(batch, self.n_roles, 2, self.grid, self.grid).sum(dim=2)
        # log P(y,x) = logsumexp_t(s) − log Z：数值稳定，CE 监督用（不经概率钳制，
        # 避免 clamp 区梯度为 0 导致 CE 卡死）。
        log_heatmap = torch.logsumexp(scores_grid, dim=2) - torch.logsumexp(
            scores.view(batch, self.n_roles, -1), dim=-1, keepdim=True
        ).unsqueeze(-1)  # [B, r, 24, 24]

        if self.mode_readout:
            # ---- v3 模式读出（探针实证，2026-08-10）：片求和 heatmap 全局峰 +
            # 局部 5×5 soft-argmax + 峰 patch 的 offset。全网格期望读出在近乎
            # 全平的余弦面上 ≈ 均匀分布 → 钉死网格质心（40-80px）；模式读出
            # 同一查询即达 8-10px。 ----
            hm = heatmap  # [B, r, 24, 24]
            flat_hm = hm.view(batch, self.n_roles, -1)
            peak_idx = flat_hm.argmax(dim=-1)  # [B, r]
            py = peak_idx // self.grid
            px = peak_idx % self.grid
            dy = torch.arange(-2, 3, device=hm.device, dtype=torch.long)
            dx = torch.arange(-2, 3, device=hm.device, dtype=torch.long)
            # 5×5 窗口（边界 clamp；多峰 NMS/top-2 留后续）
            yy = (py.unsqueeze(-1).unsqueeze(-1) + dy.view(1, 1, 5, 1)).clamp(0, self.grid - 1)
            xx = (px.unsqueeze(-1).unsqueeze(-1) + dx.view(1, 1, 1, 5)).clamp(0, self.grid - 1)
            yyb = yy.expand(-1, -1, 5, 5)
            xxb = xx.expand(-1, -1, 5, 5)
            # 扁平索引一次 gather（两次 gather 会作用在中间 5×5 结果上越界）
            win_idx = (yyb * self.grid + xxb).view(batch, self.n_roles, -1)  # [B, r, 25]
            w = flat_hm.gather(-1, win_idx).view(batch, self.n_roles, 5, 5)  # [B, r, 5, 5]
            w = w / w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
            cy = (yyb.float() + 0.5) / self.grid
            cx = (xxb.float() + 0.5) / self.grid
            p_mode = torch.stack(
                ((w * cy).sum(dim=(-2, -1)), (w * cx).sum(dim=(-2, -1))), dim=-1
            )  # [B, r, 2]（y,x 0-1）
            # 峰 patch 的 offset（取概率更高的时间片；两片坐标相同）
            ps_flat = probs.view(batch, self.n_roles, 2, self.grid * self.grid)
            idx2 = (py * self.grid + px).unsqueeze(-1)  # [B, r, 1]
            p0 = ps_flat[:, :, 0].gather(-1, idx2).squeeze(-1)
            p1 = ps_flat[:, :, 1].gather(-1, idx2).squeeze(-1)
            slice_sel = (p1 > p0).long()  # [B, r]
            off_idx = slice_sel * (self.grid * self.grid) + peak_idx
            off_peak = offsets.gather(
                2, off_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            ).squeeze(2)  # [B, r, 2]
            p_hat = p_mode + off_peak
        else:
            patch_center = coords[:, 1:]       # [1152, 2]（0-1，y,x）
            expected = patch_center.unsqueeze(0).unsqueeze(0) + offsets  # [B, r, 1152, 2]
            p_hat = (probs.unsqueeze(-1) * expected).sum(dim=2)  # [B, r, 2]

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
            offset_full=offsets,  # [B, r, 1152, 2]（v2 直接监督：δ* = p* − p_center）
            scores=scores,  # [B, r, 1152]（v4 hinge 监督）
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
