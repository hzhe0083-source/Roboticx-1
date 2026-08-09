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

Step 1 (``multi_mode=True``, C²-IRF v2 设计 §二.3/§七): multi-mode readout.
Each role query produces one spatiotemporal heatmap [B, K, 2, grid, grid]
(2 time slices); max_pool2d local NMS picks the top-2 peaks, and each peak
runs a local 5×5 soft-argmax (sub-pixel, no averaging across distant
candidates) yielding mu/cov/z per mode.  A learned NULL key/value lets the
query select "occluded / absent" (vis = 1 − P(NULL)), and the addressing
logit gains a coordinate bias ``b_coord(p)`` plus a tracking prior
``b_track(p; prev_mu)`` (gamma init 0.01).  ``key_aux`` is reserved for
Step 4 (H⁵ residual addressing).  ``multi_mode=False`` keeps the legacy
forward bit-for-bit identical.  No new training losses.

No new training losses: everything is supervised by the existing action loss
(L_FM / L_act).  Stage A runs slots + nominal direct head; the C² control
chart (P_slot over g_t) is fit in Stage B from recovery data.
"""
from __future__ import annotations

import math
from typing import NamedTuple

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


class MultiModeReadout(NamedTuple):
    """Step 1 多模式读出结果（``multi_mode=True`` 时 ``forward`` 的返回契约）。

    - ``slots`` [B, K, 2, D]：每模式内容 z（遮挡时退化为 NULL 值）；
    - ``mu``   [B, K, 2, 3]：每模式局部 soft-argmax 位置（归一化 t/y/x）；
    - ``cov``  [B, K, 2, 3, 3]：每模式峰邻域协方差（含数值下限）；
    - ``vis``  [B, K]：可见度 = 1 − P(NULL)（无 visibility loss）；
    - ``weights`` [B, K, 2, N]：每模式局部 soft-argmax 权重（稀疏，每模式
      仅 5×5 窗口内非零、窗口内和为 1）。
    """

    slots: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    vis: torch.Tensor
    weights: torch.Tensor


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

    ``multi_mode=False``（默认）：输出 slots [B, K, 768]（vision dim）、
    per-head mean attention weights [B, K, N] 和 slot centers [B, K, 3]
    （normalized t/y/x in [-1, 1]）——与旧版逐位一致。

    ``multi_mode=True``（Step 1）：每角色一个查询产生 spatiotemporal heatmap，
    局部 NMS top-2 峰 + 5×5 局部 soft-argmax → ``MultiModeReadout``（详见该
    类 docstring）。寻址 logit = qᵀK/√head_dim + b_coord(p) + b_track(p; μ̂)，
    另加 learned NULL 键值（vis = 1 − P(∅)）。
    """

    def __init__(
        self,
        vision_dim: int = 768,
        hidden_dim: int = 512,
        num_slots: int = N_SLOTS,
        num_heads: int = 8,
        pos_dim: int = 27,
        gate_init: float = -2.0,
        multi_mode: bool = False,
        track_gamma_init: float = 0.01,
        track_sigma: float = 0.25,
        cov_floor: float = 1e-4,
        aux_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.multi_mode = multi_mode
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        # Reader-side metric coordinates; zero-init keeps the frozen-feature
        # behavior at initialization (coordinate channel silent until trained).
        self.pos_proj = nn.Linear(pos_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.pos_proj.weight)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.to_vision = nn.Linear(hidden_dim, vision_dim)
        if multi_mode:
            # Step 1（C²-IRF v2 §七 Step 1）多模式读出参数：
            # 旧路径专用模块（cross_attn/门控/FFN）不构造——多模式用手动寻址
            # logit + 局部 soft-argmax，z=ΣπV 直读（无门控残差/FFN，避免死参数）。
            # - coord_bias：每 patch 标量坐标偏置 b_coord(p)（零初始化 → 初始
            #   静默，与 pos_proj 同一零门控纪律）；
            # - track_gamma：跟踪先验强度 b_track(p; prev_mu)（0.01 起步，
            #   设计文档 §八：不制造梯度死区的小门控）；
            # - null_key/null_value：learned NULL 键值——遮挡/缺物时查询可选
            #   NULL，可见度 vis = 1 − P(∅)（无需 visibility loss）。
            self.cross_attn = None
            self.read_gate_logit = None
            self.output_norm = None
            self.ffn = None
            self.coord_bias = nn.Linear(pos_dim, 1)
            nn.init.zeros_(self.coord_bias.weight)
            nn.init.zeros_(self.coord_bias.bias)
            self.track_gamma = nn.Parameter(torch.tensor(track_gamma_init))
            self.track_sigma = track_sigma
            self.cov_floor = cov_floor
            self.null_key = nn.Parameter(torch.empty(hidden_dim).normal_(0.0, 0.02))
            self.null_value = nn.Parameter(torch.empty(vision_dim).normal_(0.0, 0.02))
            # Step 4（C²-IRF v2 §二.2）：K⁵ 浅层寻址项 ℓ += γ_r·q_sᵀK⁵_n。
            # aux_proj（branch 内部）正常初始化；aux_query（输出投影）零初始化 +
            # aux_gamma=0.01 → 初始项≡0 但 aux_query 首步即有梯度（γ·aux_k≠0），
            # 避免双零初始化互锁梯度死区（设计 §八：门控 0.01 + 输出投影零初始化）。
            self.aux_proj = nn.Linear(aux_dim, hidden_dim)
            self.aux_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.zeros_(self.aux_query.weight)
            self.aux_gamma = nn.Parameter(torch.tensor(0.01))
        else:
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

    def forward(
        self,
        dense_tokens: torch.Tensor,  # [B, N, vision_dim]
        role_queries: torch.Tensor,  # [B, K, hidden_dim]
        coords: torch.Tensor,        # [N, 3] normalized t/y/x
        key_aux: torch.Tensor | None = None,  # Step 4 K⁵ 高频残差寻址项 [B, N, aux_dim]
        prev_mu: torch.Tensor | None = None,  # [B, K, 2, 3] 跟踪先验
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | MultiModeReadout:
        """角色查询读出 dense tokens。

        ``multi_mode=False`` 返回旧契约 (slots, weights, centers)；``True``
        返回 ``MultiModeReadout``。``key_aux`` 为 Step 4（H⁵ 残差寻址）项：
        寻址 logit 增加 γ_r·q_sᵀK⁵ 浅层项（γ_r=0.01、q_s 零初始化 → 初始静默）；
        仅 multi_mode 路径支持。``prev_mu`` 仅多模式路径支持。
        """
        if not self.multi_mode:
            if key_aux is not None:
                raise NotImplementedError(
                    "key_aux（Step 4 K⁵ 残差寻址项）仅 multi_mode 读出支持"
                )
            if prev_mu is not None:
                raise NotImplementedError("prev_mu（跟踪先验）仅 multi_mode 读出支持")
            return self._forward_legacy(dense_tokens, role_queries, coords)
        return self._forward_multimode(dense_tokens, role_queries, coords, prev_mu, key_aux)

    def _forward_legacy(
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

    def _forward_multimode(
        self,
        dense_tokens: torch.Tensor,  # [B, N, vision_dim]
        role_queries: torch.Tensor,  # [B, K, hidden_dim]
        coords: torch.Tensor,        # [N, 3] normalized t/y/x
        prev_mu: torch.Tensor | None,
        key_aux: torch.Tensor | None = None,  # Step 4 K⁵ 高频残差 [B, N, aux_dim]
    ) -> MultiModeReadout:
        """Step 1 多模式读出（设计文档 §二.3/§七 Step 1）。

        每角色一个查询 → 全量寻址 logit（含 NULL）→ 逐角色 heatmap
        [B, K, 2, grid, grid]（2 时间片 × grid²，t→y→x 行优先）→
        max_pool2d(5×5) 局部 NMS 取 top-2 峰 → 每峰 5×5 邻域局部 soft-argmax
        （跨 patch 亚像素）得 μ/Σ/z，权重回填为稀疏 [B, K, 2, N]。
        可见度 vis = 1 − P(NULL)，槽内容在遮挡时退化为学习到的 NULL 值。
        无辅助 loss；全部由既有 action loss（L_FM）监督。
        """
        B, N, D = dense_tokens.shape
        K = role_queries.shape[1]
        n_slice = N // 2
        grid = math.isqrt(n_slice)
        if grid * grid != n_slice:
            raise ValueError(
                f"multi_mode 需要 N=2*grid²（如 1152=2×24² 或 288=2×12²），got {N}"
            )
        dev, dtype = dense_tokens.device, dense_tokens.dtype
        pos = fourier_encode(coords.to(dtype=dtype)).to(device=dev, dtype=dtype)  # [N, pos_dim]
        # K/V = 内容投影 + 零初始化坐标旁路（与 legacy 相同构造；Step 4 并入 K⁵）。
        visual = self.vision_proj(self.vision_norm(dense_tokens)) + self.pos_proj(pos)[None]
        q = self.query_norm(role_queries)  # [B, K, hidden]
        scale = math.sqrt(self.hidden_dim // self.num_heads)  # 与 MHA 相同的缩放
        logits = torch.einsum("bkh,bnh->bkn", q, visual) / scale  # [B, K, N]
        # Step 4（C²-IRF v2 §二.2）：浅层 K⁵ 寻址项 ℓ += γ_r·q_sᵀK⁵_n。
        # aux_proj 零初始化 + aux_query 零初始化 + aux_gamma=0.01 → 初始寻址项
        # ≡0（渐进打开，不破坏深层语义主导）。
        if key_aux is not None:
            if tuple(key_aux.shape[:2]) != (B, N):
                raise ValueError(
                    f"key_aux 必须为 [B, {N}, aux_dim]，got {tuple(key_aux.shape)}"
                )
            aux_k = self.aux_proj(key_aux.to(dtype=dtype, device=dev))  # [B, N, hidden]
            q_aux = self.aux_query(q)  # [B, K, hidden]（零初始化 → 初始静默）
            logits = logits + self.aux_gamma * torch.einsum(
                "bkh,bnh->bkn", q_aux, aux_k
            ) / scale
        # b_coord(p)：每 patch 标量坐标偏置（零初始化 → 训练初期静默）。
        logits = logits + self.coord_bias(pos)[..., 0][None, None, :]
        if prev_mu is not None:
            if tuple(prev_mu.shape) != (B, K, 2, 3):
                raise ValueError(
                    f"prev_mu 必须为 [B, K, 2, 3]，got {tuple(prev_mu.shape)}"
                )
            # b_track(p; prev_mu)：对每个旧模式中心的高斯先验（γ=0.01 起步），
            # 取两模式中最近者；t/y/x 均为归一化坐标（σ=0.25 ≈ 3 patch @24 网格）。
            p = coords.to(device=dev, dtype=dtype)[None, None, None]  # [1,1,1,N,3]
            d2 = ((p - prev_mu[:, :, :, None]) ** 2).sum(-1)  # [B,K,2,N]
            prior = (-d2 / (2.0 * self.track_sigma**2)).exp().amax(dim=2)  # [B,K,N]
            logits = logits + self.track_gamma * prior
        # NULL：learned 键值；softmax 覆盖 N patch + 1 NULL → P(∅) 即遮挡概率。
        null_logit = torch.einsum("bkh,h->bk", q, self.null_key) / scale  # [B,K]
        w = torch.softmax(
            torch.cat([logits, null_logit.unsqueeze(-1)], dim=-1), dim=-1
        )  # [B, K, N+1]
        p_null = w[..., -1:]                      # [B, K, 1]
        vis = (1.0 - p_null).squeeze(-1)          # [B, K]
        heatmap = w[..., :-1].reshape(B, K, 2, grid, grid)

        # 局部 NMS：max_pool2d(5×5) 相等即局部极大候选；每角色跨两时间片取 top-2 峰。
        G2 = grid * grid
        h3 = heatmap  # [B, K, 2, G, G]（2 = 时间片）
        pooled = F.max_pool2d(
            h3.reshape(B, K * 2, grid, grid), kernel_size=5, stride=1, padding=2
        ).reshape(B, K, 2, grid, grid)
        mask = h3 == pooled
        cand = torch.where(
            mask.reshape(B, K, 2 * G2),
            h3.reshape(B, K, 2 * G2),
            torch.full((B, K, 2 * G2), float("-inf"), device=dev, dtype=dtype),
        )
        vals, idx = cand.topk(2, dim=-1)  # 平坦索引（含时间片），[B, K, 2]
        # 极端单峰热图（候选 < 2）时第二峰回退到全局最大（两个假设同位置）。
        second = torch.where(vals[..., 1].isfinite(), idx[..., 1], idx[..., 0])
        peaks = torch.stack([idx[..., 0], second], dim=-1)  # [B, K, 2]
        t = peaks // G2  # 时间片 0/1
        rem = peaks % G2
        y = rem // grid
        x = rem % grid
        # 5×5 邻域局部 soft-argmax：NMS 用概率保持排序；局部权重直接从
        # 原始 logits 做 float32 softmax，既保持概率比，也避免 NULL 占优时
        # patch 概率下溢为 0 导致 0/0 或权重和小于 1。
        logit_h3 = logits.reshape(B, K, 2, grid, grid)
        h_pad = F.pad(
            logit_h3.reshape(B, K * 2, grid, grid),
            (2, 2, 2, 2),
            value=float("-inf"),
        ).reshape(B, K, 2, grid + 4, grid + 4)
        offs = torch.arange(5, device=dev)
        b_i = torch.arange(B, device=dev).view(B, 1, 1, 1, 1)
        k_i = torch.arange(K, device=dev).view(1, K, 1, 1, 1)
        wy = y[..., None, None] + offs.view(1, 1, 1, 5, 1)  # [B,K,2,5,1]
        wx = x[..., None, None] + offs.view(1, 1, 1, 1, 5)  # [B,K,2,1,5]
        windows = h_pad[b_i, k_i, t[..., None, None], wy, wx]  # [B,K,2,5,5]
        pi = F.softmax(
            windows.reshape(B, K, 2, 25), dim=-1, dtype=torch.float32
        ).to(dtype=dtype)  # [B,K,2,25]
        # 窗口 patch 的全局索引（越界位置 π=0，clamp 后对 μ/Σ/z 无贡献）。
        gy = (y[..., None, None] + offs.view(1, 1, 1, 5, 1) - 2).clamp(0, grid - 1)
        gx = (x[..., None, None] + offs.view(1, 1, 1, 1, 5) - 2).clamp(0, grid - 1)
        n_idx = (t[..., None, None] * G2 + gy * grid + gx).reshape(B, K, 2, 25)
        p_win = coords.to(device=dev, dtype=dtype)[n_idx]  # [B,K,2,25,3]
        v_win = visual[torch.arange(B, device=dev).view(B, 1, 1, 1), n_idx]  # [B,K,2,25,hidden]
        mu = torch.einsum("bkjp,bkjpc->bkjc", pi, p_win)  # [B,K,2,3]
        d = p_win - mu.unsqueeze(-2)
        cov = torch.einsum("bkjpc,bkjpe->bkjce", pi.unsqueeze(-1) * d, d) + (
            self.cov_floor * torch.eye(3, device=dev, dtype=dtype)
        )  # [B,K,2,3,3]
        z_local = torch.einsum("bkjp,bkjph->bkjh", pi, v_win)  # [B,K,2,hidden]
        z_local = self.to_vision(z_local)  # [B,K,2,D]
        # 遮挡时模式内容退化为学习到的 NULL 值（"空"向量，防下游误读）。
        p_null_m = p_null.view(B, K, 1, 1)
        slots = (1.0 - p_null_m) * z_local + p_null_m * self.null_value.view(1, 1, 1, D)
        # 稀疏回填：每模式仅窗口内非零、窗口内和 = 1。
        # scatter_add_：角点窗口有 clamp 后重复索引，必须累加而非 last-write-wins。
        weights = torch.zeros(B, K, 2, N, device=dev, dtype=dtype)
        weights.scatter_add_(-1, n_idx, pi)
        return MultiModeReadout(
            slots=slots,  # [B, K, 2, D]
            mu=mu,
            cov=cov,
            vis=vis,
            weights=weights,
        )


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
    slots: torch.Tensor,   # [B, K, vision_dim]（multi_mode 时为调用方展平的 [B, K*2, D]）
    relations: torch.Tensor,  # [B, 3, vision_dim]
) -> torch.Tensor:
    """VA 视觉流拼接：[coarse; slots; relations]。

    默认 25-token（16 coarse + 6 槽 + 3 关系）；multi_mode 时调用方把模式
    槽展平为 [B, K*2, D] 传入 → 16 + 12 + 3 = 31 tokens（设计文档 §九：
    1152 patch 不进 VA 自注意力，此处即压缩后的全部工作空间）。
    """
    return torch.cat([coarse, slots, relations], dim=1)
