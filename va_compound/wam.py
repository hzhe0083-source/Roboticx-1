"""E7 WAM v1：联合世界动作流（JointWorldActionFlow）。

独立模块（约 60M）：联合去噪 102 个生成 token —— 48 个动作残差 token、
3 跨度 × 16 个未来 V-JEPA 空间 latent token、3 跨度 × 2 个未来几何 token，
输出动作残差速度与场景速度（latent / 几何）。设计依据
docs/superpowers/specs/2026-08-13-e7-wam-design.md §3.1。

- 逐层旁路耦合（方案 C）：第 i 层 cross-attn 只读 VA 记忆快照
  va_layers[min(i * n_va // num_layers, n_va - 1)]，并与 spatial/geo 条件
  拼接为 K/V = [16 va, 16 spatial, 1 geo] 共 33 token（每层独立投影参数，
  K/V 各一套，bias=False，投影正常初始化、受 W_O 零屏障保护）。
- 零初始化纪律：AdaLN 每层 Linear(256, 6d) 权重+bias 全零；CA W_O 全零；
  动作残差速度头全零（训练起点 Δv ≡ 0）；场景头正常初始化。
  CA 出口用残差门 gate2 = 1.0 + ada_gate2——避免与零初始化 W_O 构成双重
  零屏障使 CA 永久死亡（SA gate1 与 FFN ffn_gate 维持零门，详见
  _WAMBlock.forward 注释）。
- 时间条件 slim AdaLN：t_emb = SiLU(Linear(256)(sinusoidal(flow_time)))，
  每层 Linear(256, 6*512) chunk(6) → scale1,shift1,gate1,scale2,shift2,gate2。
- 无 eval() 时间随机性；CPU 确定性。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WAMConfig:
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_hidden: int = 1000          # SwiGLU intermediate；1408 会让加入每层 K/V 条件投影后破 65M 预算（1000 → 实测 ~64.7M）
    cond_dim: int = 256             # slim AdaLN MLP input
    vision_dim: int = 768           # V-JEPA H11 latent dim
    geo_dim: int = 8                # MT-VJ p_times_visibility_flat
    n_scene_tokens: int = 16        # per-horizon spatial tokens (4x4 pool of last time slice)
    n_geo_tokens: int = 2           # per-horizon geometry tokens (g_future, nu)
    horizons: tuple = (6, 24, 48)
    action_horizon: int = 48
    action_dim: int = 4
    qk_norm: bool = True
    use_swiglu: bool = True
    dropout: float = 0.0


def wam_config_from_state(state: dict | None, **fallback) -> WAMConfig:
    """Rebuild ``WAMConfig`` from a checkpoint ``wam_config`` dict.

    ``dataclasses.asdict`` + ``torch.save`` may turn ``horizons`` into a list.
    Unknown keys are ignored so older sidecars still load. ``fallback`` fills
    missing fields (used when a blob has weights but no saved config).
    """
    payload = dict(fallback)
    if state:
        payload.update(state)
    if "horizons" in payload and not isinstance(payload["horizons"], tuple):
        payload["horizons"] = tuple(payload["horizons"])
    fields = {item.name for item in dataclasses.fields(WAMConfig)}
    return WAMConfig(**{key: value for key, value in payload.items() if key in fields})


@dataclass
class WAMSceneVelocity:
    latent: Tensor  # [B, 3, 16, 768] velocity in latent space
    geo: Tensor     # [B, 3, 2, 8]    velocity in geometry space


class _RMSNorm(nn.Module):
    """带可学习权重的 RMSNorm（F.rms_norm 包装，pre-norm 用）。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _WAMBlock(nn.Module):
    """SA → CA(读 VA 快照) → FFN，AdaLN-Zero 调制前两个子层，三出口门控。

    chunk(6) = scale1,shift1,gate1,scale2,shift2,gate2 只覆盖 SA/CA 两个
    调制子层；FFN 出口 gate3 无 chunk 槽位，用独立的零初始化标量参数
    （见模块 docstring 说明）。
    """

    def __init__(self, config: WAMConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        if d % config.num_heads != 0:
            raise ValueError(
                f"hidden_dim {d} 必须能被 num_heads {config.num_heads} 整除"
            )
        self.hidden_dim = d
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = config.qk_norm
        self.dropout = config.dropout

        self.norm1 = _RMSNorm(d)
        self.sa_q = nn.Linear(d, d)
        self.sa_k = nn.Linear(d, d)
        self.sa_v = nn.Linear(d, d)
        self.sa_o = nn.Linear(d, d)

        self.norm2 = _RMSNorm(d)
        self.ca_q = nn.Linear(d, d)
        # CA K/V 条件：每层独立投影参数，K/V 各一套，bias=False；
        # K/V = [16 va 快照, 16 spatial, 1 geo] 共 33 token（拼接见 forward）。
        # 这些投影正常初始化——受下方 W_O 零屏障保护，不破坏初始等价。
        self.proj_va_k = nn.Linear(d, d, bias=False)
        self.proj_va_v = nn.Linear(d, d, bias=False)
        self.proj_spatial_k = nn.Linear(config.vision_dim, d, bias=False)
        self.proj_spatial_v = nn.Linear(config.vision_dim, d, bias=False)
        self.proj_geo_k = nn.Linear(config.geo_dim, d, bias=False)
        self.proj_geo_v = nn.Linear(config.geo_dim, d, bias=False)
        self.ca_o = nn.Linear(d, d)
        # 旁路 CA 出口零初始化：训练起点世界分支注入为零。
        nn.init.zeros_(self.ca_o.weight)
        nn.init.zeros_(self.ca_o.bias)

        self.norm3 = _RMSNorm(d)
        if config.use_swiglu:
            self.w1 = nn.Linear(d, config.ffn_hidden)
            self.w2 = nn.Linear(config.ffn_hidden, d)
            self.w3 = nn.Linear(d, config.ffn_hidden)
            self.gelu_ffn = None
        else:
            # GELU 2048 对照开关（hidden*4，与 SwiGLU 1408 的 2/3*4h 对等）。
            self.w1 = self.w2 = self.w3 = None
            self.gelu_ffn = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, d),
            )
        # FFN 出口 gate3（AdaLN chunk(6) 无其槽位，独立零初始化）。
        self.ffn_gate = nn.Parameter(torch.zeros(1, 1, d))

        # AdaLN slim：Linear(256, 6d) 权重+bias 全零。
        self.ada = nn.Linear(config.cond_dim, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def _attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out_proj: nn.Linear,
        apply_qk_norm: bool,
    ) -> Tensor:
        B, N, _ = q.shape
        nh, hd = self.num_heads, self.head_dim
        q = q.view(B, N, nh, hd).transpose(1, 2)
        k = k.view(B, k.shape[1], nh, hd).transpose(1, 2)
        v = v.view(B, v.shape[1], nh, hd).transpose(1, 2)
        if self.qk_norm and apply_qk_norm:
            # 苏式 per-head QK RMSNorm；统计量在 fp32 计算再回落原 dtype，
            # 避免 BF16 下 q/k 的 RMS 统计精度不足（与仓库 qk_norm 惯例一致）。
            q = F.rms_norm(q.float(), (q.shape[-1],)).to(q.dtype)
            k = F.rms_norm(k.float(), (k.shape[-1],)).to(k.dtype)
        # logits 用 fp32，softmax 后回到原 dtype（与 FlowMatchingHead 一致）。
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        w = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        if self.dropout > 0.0:
            w = F.dropout(w, self.dropout, self.training)
        out = torch.matmul(w, v)
        out = out.transpose(1, 2).reshape(B, N, nh * hd)
        return out_proj(out)

    def forward(
        self,
        hidden: Tensor,
        va_kv: Tensor,
        spatial_tokens: Tensor,
        geo_tokens: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        mod = self.ada(t_emb).unsqueeze(1)  # [B, 1, 6d]
        scale1, shift1, gate1, scale2, shift2, gate2 = mod.chunk(6, dim=-1)

        # 子层 1：自注意力（102 token 全互看）。零门 gate1：SA 内部没有
        # 零初始化输出投影（sa_o 正常初始化），单零屏障下 gate1/sa_* 梯度
        # 第一步即非零，零门安全。
        h = self.norm1(hidden)
        h = h * (1.0 + scale1) + shift1
        h = self._attn(self.sa_q(h), self.sa_k(h), self.sa_v(h), self.sa_o, True)
        hidden = hidden + gate1 * h

        # 子层 2：交叉注意力（只读 VA 层快照 + spatial/geo 条件），
        # K/V = [16 va, 16 spatial, 1 geo] 共 33 token。
        h = self.norm2(hidden)
        h = h * (1.0 + scale2) + shift2
        k = torch.cat(
            (
                self.proj_va_k(va_kv),
                self.proj_spatial_k(spatial_tokens),
                self.proj_geo_k(geo_tokens[:, None, :]),
            ),
            dim=1,
        )
        v = torch.cat(
            (
                self.proj_va_v(va_kv),
                self.proj_spatial_v(spatial_tokens),
                self.proj_geo_v(geo_tokens[:, None, :]),
            ),
            dim=1,
        )
        h = self._attn(self.ca_q(h), k, v, self.ca_o, True)
        # CA 出口用残差门 gate2 = 1.0 + ada_gate2（ada 零初始化 → 初始 gate2 = 1）。
        # 为什么 CA 用残差门而 SA/FFN 用零门：CA 的 W_O 被设计纪律要求零
        # 初始化，若 gate2 也零初始化则构成双重零屏障——gate2 梯度
        # = out·dL/dh = (h·W_O)·dL/dh ≡ 0，W_O/K/V/Q 梯度也恒 0，CA 分支
        # 永久死亡。残差门下初始贡献 = 1·(h·W_O=0) = 0（初始等价不变式
        # 保住），且 W_O 第一步即有梯度、K/V/Q 随 W_O 打开后获得梯度。
        hidden = hidden + (1.0 + gate2) * h

        # 子层 3：SwiGLU FFN（无 AdaLN 调制，出口 gate3）。
        h = self.norm3(hidden)
        if self.gelu_ffn is not None:
            h = self.gelu_ffn(h)
        else:
            h = self.w2(F.silu(self.w1(h)) * self.w3(h))
        hidden = hidden + self.ffn_gate * h
        return hidden


class JointWorldActionFlow(nn.Module):
    """联合生成 48 步动作残差速度与 3 跨度的场景速度（latent + 几何）。

    102 生成 token：48 动作 + 3×16 latent + 3×2 geo；类型嵌入
    （action/space/geo）+ 组内位置嵌入（动作 index / 4×4 空间 / 跨度 id），
    无 RoPE。共享同一 flow time τ。
    """

    # 逐层 VA 读取映射：n_va=8 / num_layers=12 下 va_index_for_layer(i) 的
    # 结果序列（min(i*n_va//num_layers, n_va-1)）。运行路径用同一公式动态
    # 计算（va_index_for_layer）；类属性供测试直接断言映射。
    VA_IDX_MAP = (0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7)

    @staticmethod
    def va_index_for_layer(i: int, n_va: int, num_layers: int) -> int:
        """第 i 层 CA 读取的 VA 层索引：min(i * n_va // num_layers, n_va - 1)。"""
        return min(i * n_va // num_layers, n_va - 1)

    def __init__(self, config: WAMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        n_horiz = len(config.horizons)
        if d % config.num_heads != 0:
            raise ValueError(
                f"hidden_dim {d} 必须能被 num_heads {config.num_heads} 整除"
            )
        if config.cond_dim % 2 != 0:
            raise ValueError("cond_dim 必须为偶数（sin/cos 各占一半）")

        # 输入嵌入。
        self.embed_x = nn.Linear(config.action_dim, d)   # noisy_actions 4→512
        self.cond_proj = nn.Linear(d, d)                 # action_condition（只读）
        self.embed_l = nn.Linear(config.vision_dim, d)   # latent 768→512
        self.embed_g = nn.Linear(config.geo_dim, d)      # geo 8→512

        # 类型嵌入（3 类）+ 组内位置嵌入。
        self.type_emb = nn.Parameter(torch.empty(3, d))
        self.pos_action = nn.Parameter(torch.empty(config.action_horizon, d))
        self.pos_spatial = nn.Parameter(torch.empty(config.n_scene_tokens, d))
        self.pos_horizon = nn.Parameter(torch.empty(n_horiz, d))
        for p in (self.type_emb, self.pos_action, self.pos_spatial, self.pos_horizon):
            nn.init.normal_(p, std=0.02)

        # 时间条件：sinusoidal(flow_time) → Linear(256)→SiLU → t_emb。
        half = config.cond_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("time_frequencies", frequencies, persistent=False)
        self.time_mlp = nn.Linear(config.cond_dim, config.cond_dim)

        self.blocks = nn.ModuleList(
            _WAMBlock(config) for _ in range(config.num_layers)
        )

        # 输出头：动作残差速度头零初始化；场景头正常初始化。
        self.action_norm = _RMSNorm(d)
        self.action_head = nn.Linear(d, config.action_dim)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        self.latent_heads = nn.ModuleList(
            nn.Linear(d, config.vision_dim) for _ in range(n_horiz)
        )
        self.geo_heads = nn.ModuleList(
            nn.Linear(d, config.geo_dim) for _ in range(n_horiz)
        )

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        *,
        action_condition: Tensor,          # [B, 48, 512] 只读
        va_layers: tuple,                  # tuple of n_va tensors [B, 16, 512]，只读
        spatial_tokens: Tensor,            # [B, 16, 768] 当前 4×4 pool（每层 CA K/V 条件）
        geo_tokens: Tensor,                # [B, 8] 当前 p*vis flat（每层 CA K/V 条件）
        noisy_actions: Tensor,             # [B, 48, 4] 动作路径 x_t
        noisy_scene_latents: Tensor,       # [B, 3, 16, 768] 场景路径 x_t
        noisy_scene_geo: Tensor,           # [B, 3, 2, 8]
        flow_time: Tensor,                 # [B] in [0, 1]
    ) -> tuple[Tensor, WAMSceneVelocity]:
        cfg = self.config
        n_horiz = len(cfg.horizons)
        if len(va_layers) == 0:
            raise ValueError("va_layers 不能为空（每层 CA 需要至少 1 个 VA 快照）")
        dtype = self.embed_x.weight.dtype
        d = cfg.hidden_dim
        B = action_condition.shape[0]

        # 时间嵌入（fp32 角度 → 目标 dtype）。
        t = flow_time.float()
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError("flow_time must have shape [batch] or [batch, 1]")
        angles = t[:, None] * self.time_frequencies[None]
        t_sin = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if t_sin.shape[-1] < cfg.cond_dim:
            t_sin = F.pad(t_sin, (0, cfg.cond_dim - t_sin.shape[-1]))
        t_emb = F.silu(self.time_mlp(t_sin.to(dtype)))  # [B, cond_dim]

        # 102 token 组装：48 动作 + 3×16 latent + 3×2 geo。
        a = self.embed_x(noisy_actions.to(dtype)) + self.cond_proj(
            action_condition.to(dtype)
        )
        a = a + self.type_emb[0] + self.pos_action  # [B, 48, d]
        l = self.embed_l(noisy_scene_latents.to(dtype))  # [B, 3, 16, d]
        l = l + self.type_emb[1] + self.pos_spatial[None, None] + self.pos_horizon[None, :, None]
        g = self.embed_g(noisy_scene_geo.to(dtype))  # [B, 3, 2, d]
        g = g + self.type_emb[2] + self.pos_horizon[None, :, None]
        hidden = torch.cat(
            (a, l.reshape(B, -1, d), g.reshape(B, -1, d)), dim=1
        )  # [B, 102, d]

        n_va = len(va_layers)
        spatial_tokens = spatial_tokens.to(dtype)
        geo_tokens = geo_tokens.to(dtype)
        for i, block in enumerate(self.blocks):
            va_idx = self.va_index_for_layer(i, n_va, cfg.num_layers)
            hidden = block(
                hidden, va_layers[va_idx].to(dtype), spatial_tokens, geo_tokens, t_emb
            )

        # 动作残差速度头（零初始化 → 训练起点 Δv ≡ 0）。
        action_tokens = hidden[:, : cfg.action_horizon]
        dv = self.action_head(self.action_norm(action_tokens))  # [B, 48, 4]

        # 场景速度头：每跨度独立 head。
        l_start = cfg.action_horizon
        l_end = l_start + n_horiz * cfg.n_scene_tokens
        lat = hidden[:, l_start:l_end].reshape(B, n_horiz, cfg.n_scene_tokens, cfg.hidden_dim)
        geo = hidden[:, l_end:l_end + n_horiz * cfg.n_geo_tokens].reshape(
            B, n_horiz, cfg.n_geo_tokens, cfg.hidden_dim
        )
        lat_vel = torch.stack(
            [head(lat[:, h]) for h, head in enumerate(self.latent_heads)], dim=1
        )  # [B, 3, 16, vision_dim]
        geo_vel = torch.stack(
            [head(geo[:, h]) for h, head in enumerate(self.geo_heads)], dim=1
        )  # [B, 3, 2, geo_dim]

        return dv, WAMSceneVelocity(latent=lat_vel, geo=geo_vel)
