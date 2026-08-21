"""WAM4VA: recurrent world state for a VA policy.

VA and WAM keep independent states.  At stage ``i`` both peers read the
committed state from stage ``i-1``.  WAM publishes world-memory tokens that
the next VA layer reads as attention K/V; it never adds a correction to VA's
visual or action outputs.  VA/Flow remain the only action emitter.

World prediction reads the executable action chunk plus the T-frame DINO
clip.  Realized futures enter only the world-model loss.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import functional as F


# 循环状态数值稳定性（无参数，兼容 strict checkpoint 加载）。
#
# belief 与 world_map 是跨 8 个 stage × T 个决策点持久化的循环状态。两者原来
# 都是无约束的纯加法残差（``belief = belief + f``、``z_map = base + delta``），
# 在 sequence_length=4 的短窗口训练下勉强有限，但长 horizon 闭环（horizon=500，
# 约 4000 次 propose）会把同一段非收缩递推推到 NaN；训练后期 world_action_rank
# 的尖峰震荡也源于早期 stage 的循环状态幅度单调漂移。
#
# 收缩只作用于前向值：用 straight-through 让梯度对 base/delta 恒等，保住
# ``belief = belief + f`` 原有的 identity credit path 契约（早期 stage 通过
# 恒等路径收到不衰减的梯度），同时前向的 ``retention<1`` 让循环有不动点。
_BELIEF_RETENTION = 0.9  # belief 循环的遗忘保留系数（<1 即收缩，防发散）
_MAP_RETENTION = 0.9     # world_map 循环 stage 的保留系数（<1 使循环有不动点）
_MAP_DELTA_GAIN = 0.5    # world_map 残差的固定缩放（8-stage 累加不失控）


def _st_scale_residual(
    base: Tensor,
    delta: Tensor,
    *,
    base_scale: float,
    delta_scale: float,
) -> Tensor:
    """Straight-through 缩放残差：前向 ``base_scale*base + delta_scale*delta``。

    梯度对 ``base`` 和 ``delta`` 均恒等（I），因此循环的 identity credit path
    设计契约不变——早期 stage 通过恒等路径收到不衰减的梯度，而缩放只改变前向
    值，用于抑制跨 stage / 跨决策点的无界累积。``delta`` 本身在调用点用
    ``base.detach()`` 计算，所以非线性 reader 的 Jacobian 不会跨 stage 连乘。
    """
    scaled = base_scale * base + delta_scale * delta
    return (base + delta) + (scaled - base - delta).detach()


def _require_finite(tensor: Tensor, name: str, *, boundary: str) -> None:
    """Fail at peer boundaries before non-finite values enter recurrent state."""
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(
            f"{boundary}: {name} contains NaN or Inf values; "
            "check the immediately upstream peer computation"
        )


def _tensor_to(
    tensor: Tensor | None,
    device: torch.device | str | torch.dtype | None,
    dtype: torch.dtype | None,
) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class WAMState:
    """Immutable recurrent World snapshot used by peer-synchronous stages."""

    belief: Tensor | None = None
    innovation: Tensor | None = None
    world_map: Tensor | None = None

    def detach(self) -> "WAMState":
        return WAMState(
            belief=None if self.belief is None else self.belief.detach(),
            innovation=None if self.innovation is None else self.innovation.detach(),
            world_map=None if self.world_map is None else self.world_map.detach(),
        )

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WAMState":
        if isinstance(device, torch.dtype) and dtype is None:
            dtype, device = device, None
        # Recurrent state has one storage contract regardless of the surrounding
        # module/feature dtype. ``dtype`` is accepted for Tensor.to API parity but
        # intentionally does not alter persisted state precision.
        return WAMState(
            belief=_tensor_to(self.belief, device, torch.float32),
            innovation=_tensor_to(self.innovation, device, torch.float32),
            world_map=_tensor_to(self.world_map, device, torch.float32),
        )

    def index_select(self, index: Tensor) -> "WAMState":
        if index.ndim != 1:
            raise ValueError(f"index must be one-dimensional, got {tuple(index.shape)}")

        def select(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None
            return tensor.index_select(0, index.to(device=tensor.device, dtype=torch.long))

        return WAMState(
            belief=select(self.belief),
            innovation=select(self.innovation),
            world_map=select(self.world_map),
        )

    def validate_for(
        self,
        *,
        batch: int,
        hidden_dim: int,
        n_belief: int,
        n_evidence: int,
        dino_dim: int | None,
        map_size: int,
        device: torch.device,
    ) -> None:
        """Reject snapshots outside the canonical device-local FP32 contract."""
        specs = (
            ("belief", self.belief, (batch, n_belief, hidden_dim)),
            ("innovation", self.innovation, (batch, n_evidence, hidden_dim)),
            (
                "world_map",
                self.world_map,
                None if dino_dim is None else (batch, dino_dim, map_size, map_size),
            ),
        )
        for name, tensor, expected in specs:
            if tensor is None:
                continue
            if expected is None:
                raise ValueError(f"WAMState.{name} requires dino_dim to be configured")
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"WAMState.{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
            if tensor.device != device:
                raise ValueError(
                    f"WAMState.{name} device must be {device}, got {tensor.device}"
                )
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"WAMState.{name} dtype must be torch.float32, got {tensor.dtype}"
                )

    def validate_finite(self, *, boundary: str = "WAM recurrent state") -> None:
        for name, tensor in (
            ("belief", self.belief),
            ("innovation", self.innovation),
            ("world_map", self.world_map),
        ):
            if tensor is not None:
                _require_finite(tensor, f"WAMState.{name}", boundary=boundary)


@dataclass(frozen=True)
class WMRMAux:
    z_hat: Tensor
    z_spans: Tensor
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
    # Belief state exactly at the predictor input. ``belief`` above is the
    # post-prediction state and is therefore not valid for an
    # action-counterfactual re-evaluation of the final World stage.
    predict_belief: Tensor | None = None

    def _map_tensors(self, transform) -> "WMRMAux":
        return WMRMAux(
            **{
                field.name: (
                    transform(value) if isinstance(value, Tensor) else value
                )
                for field in fields(self)
                for value in (getattr(self, field.name),)
            }
        )

    def detach(self) -> "WMRMAux":
        return self._map_tensors(Tensor.detach)

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WMRMAux":
        if isinstance(device, torch.dtype) and dtype is None:
            dtype, device = device, None
        return self._map_tensors(lambda tensor: tensor.to(device=device, dtype=dtype))

    def index_select(self, index: Tensor) -> "WMRMAux":
        if index.ndim != 1:
            raise ValueError(f"index must be one-dimensional, got {tuple(index.shape)}")
        return self._map_tensors(
            lambda tensor: tensor.index_select(
                0, index.to(device=tensor.device, dtype=torch.long)
            )
        )


@dataclass(frozen=True)
class WAMProposal:
    """Independent World transition computed from one peer snapshot."""

    next_world_state: WAMState
    world_message: Tensor
    aux: WMRMAux

    def detach(self) -> "WAMProposal":
        return WAMProposal(
            next_world_state=self.next_world_state.detach(),
            world_message=self.world_message.detach(),
            aux=self.aux.detach(),
        )

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WAMProposal":
        if isinstance(device, torch.dtype) and dtype is None:
            dtype, device = device, None
        return WAMProposal(
            next_world_state=self.next_world_state.to(device=device, dtype=dtype),
            world_message=self.world_message.to(device=device, dtype=dtype),
            aux=self.aux.to(device=device, dtype=dtype),
        )

    def index_select(self, index: Tensor) -> "WAMProposal":
        if index.ndim != 1:
            raise ValueError(f"index must be one-dimensional, got {tuple(index.shape)}")

        def select(tensor: Tensor) -> Tensor:
            return tensor.index_select(0, index.to(device=tensor.device, dtype=torch.long))

        return WAMProposal(
            next_world_state=self.next_world_state.index_select(index),
            world_message=select(self.world_message),
            aux=self.aux.index_select(index),
        )

    def validate_finite(self, *, boundary: str = "WAM proposal") -> None:
        _require_finite(self.world_message, "world_message", boundary=boundary)
        self.next_world_state.validate_finite(boundary=boundary)


class ExecutableActionReadout(nn.Module):
    """Deterministic bounded H6 action belief for the World peer."""

    def __init__(self, hidden_dim: int, action_dim: int = 4, horizon: int = 6) -> None:
        super().__init__()
        if hidden_dim < 1 or action_dim < 1:
            raise ValueError("hidden_dim and action_dim must be positive")
        if horizon != 6:
            raise ValueError(f"ExecutableActionReadout requires H6, got H{horizon}")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, action: Tensor) -> Tensor:
        expected = (self.horizon, self.hidden_dim)
        if action.ndim != 3 or tuple(action.shape[1:]) != expected:
            raise ValueError(
                f"action must be [B, {self.horizon}, {self.hidden_dim}], "
                f"got {tuple(action.shape)}"
            )
        _require_finite(action, "action", boundary="executable action readout input")
        logits = self.proj(action)
        _require_finite(logits, "readout", boundary="executable action readout output")
        readout = torch.tanh(logits)
        _require_finite(readout, "readout", boundary="executable action readout output")
        return readout


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
    """Full-grid spatiotemporal residual predictor with a near-copy start."""

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
        # Keep the initial prediction close to the last frame without closing
        # the World-loss Jacobian to the condition encoder on the first update.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
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
            # The recurrent map remains the differentiable residual base below,
            # so later World losses still train every earlier stage.  Treat its
            # second use as predictor context as read-only memory: otherwise the
            # 6-block nonlinear Jacobian is multiplied through all 8 stages and
            # 4 decisions, which can make an otherwise finite forward pass have
            # an arbitrarily large backward pass.
            predictor_clip = torch.cat(
                (clip[:, :-1], previous_map.detach()[:, None]), dim=1
            )
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
        if previous_map is None:
            # 首决策点 stage 0 的锚是真实 DINO 最后一帧，保持语义不衰减。
            return _st_scale_residual(
                base, delta, base_scale=1.0, delta_scale=_MAP_DELTA_GAIN
            )
        # 循环项（上一 stage / 上一决策点的 map）：8 个 stage 复用同一个
        # predictor，每个 stage 都在上一 stage 的 map 上累加 delta，且跨决策点
        # 经 WAMState.world_map 持续累积。纯加法对应训练日志里前几个 stage 能量
        # 6→26→47 的单调抬升，长 horizon 下无界。保留系数 <1 使循环有不动点，
        # 固定缩放把每阶段 refine 量压回有界范围；straight-through 保证 base 的
        # 恒等梯度不因收缩而衰减。
        return _st_scale_residual(
            base, delta, base_scale=_MAP_RETENTION, delta_scale=_MAP_DELTA_GAIN
        )


class WAM4VA(nn.Module):
    """World-action model that publishes recurrent attention memory to VA."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        world_dim: int = 8,
        proprio_dim: int = 9,
        num_heads: int = 4,
        n_belief: int = 8,
        n_evidence: int = 8,
        n_spans: int = 3,
        n_progress: int = 4,
        n_task_queries: int = 4,
        cycle_steps: int = 6,
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
        if proprio_dim < 1:
            raise ValueError("proprio_dim must be positive")
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
        self.proprio_dim = proprio_dim
        self.n_belief = n_belief
        self.n_evidence = n_evidence
        self.n_spans = n_spans
        self.n_progress = n_progress
        self.n_task_queries = n_task_queries
        self.cycle_steps = cycle_steps
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

        self.world_from_env = nn.Linear(hidden_dim, hidden_dim)
        self.world_from_state = nn.Linear(proprio_dim, hidden_dim)
        self.world_from_belief = nn.Linear(hidden_dim, hidden_dim)
        self.task_queries = nn.Parameter(torch.empty(n_task_queries, hidden_dim))
        nn.init.normal_(self.task_queries, std=0.02)
        self.task_attention = _CrossAttn(hidden_dim, num_heads)
        self.world_from_task = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.world_from_task.weight)
        nn.init.zeros_(self.world_from_task.bias)
        self.span_heads = nn.ModuleList(nn.Linear(hidden_dim, world_dim) for _ in range(n_spans))
        self.progress_head = nn.Linear(hidden_dim * 2, n_progress)
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
        if previous.shape != current.shape:
            raise ValueError(
                "previous innovation shape must match current innovation, got "
                f"{tuple(previous.shape)} vs {tuple(current.shape)}"
            )
        if previous.device != current.device:
            raise ValueError(
                "previous innovation device must match current innovation, got "
                f"{previous.device} vs {current.device}"
            )
        # This recurrent comparison is numerically sensitive and must not inherit
        # the feature path's incidental autocast dtype.
        with torch.autocast(device_type=current.device.type, enabled=False):
            current_fp32 = current.float()
            # ``previous`` is recurrent memory, not a quantity that this local
            # projection should optimize through.  Differentiating the
            # scale-invariant projection with respect to a near-zero previous
            # innovation produces a 1 / ||previous|| Jacobian and can explode
            # across the 8-stage x T recurrence.  Its value still participates
            # in the forward pass; only that ill-conditioned backward edge is
            # removed.
            previous_fp32 = previous.detach().float()
            flat_prev = previous_fp32.reshape(previous_fp32.shape[0], -1)
            flat_cur = current_fp32.reshape(current_fp32.shape[0], -1)
            # Scale before any square/dot reduction.  Deployment carries this
            # state for far longer than one training window, so even finite
            # values can otherwise overflow their FP32 energy to Inf.
            prev_scale = flat_prev.abs().amax(dim=-1, keepdim=True)
            cur_scale = flat_cur.abs().amax(dim=-1, keepdim=True)
            safe_prev_scale = prev_scale.clamp_min(1e-20)
            safe_cur_scale = cur_scale.clamp_min(1e-20)
            scaled_prev = flat_prev / safe_prev_scale
            scaled_cur = flat_cur / safe_cur_scale
            prev_energy = scaled_prev.square().sum(dim=-1, keepdim=True)
            cur_energy = scaled_cur.square().sum(dim=-1, keepdim=True)
            dot = (scaled_cur * scaled_prev).sum(dim=-1, keepdim=True)
            energy_floor = 1e-8
            cosine = dot / (cur_energy * prev_energy).sqrt().clamp_min(
                energy_floor
            )
            coeff = dot / prev_energy.clamp_min(energy_floor)
            original_prev_energy = prev_scale.square() * prev_energy
            drop = (original_prev_energy > energy_floor) & (
                cosine > self.innov_overlap
            )
            projection = (
                coeff * scaled_prev * cur_scale
            ).view_as(current_fp32)
            while drop.ndim < current_fp32.ndim:
                drop = drop.unsqueeze(-1)
            # ``where`` is intentional: a false branch must stay clean even if
            # an extreme true-branch projection ever overflows.  Multiplying a
            # false 0 by NaN/Inf would still contaminate the output.
            return current_fp32 - torch.where(
                drop, projection, torch.zeros_like(projection)
            )

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
        """Predict the P-step-later DINO map from exactly P executable actions."""
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
            proprio, belief, task_summary, env_tokens, env_flat
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
                self.belief_to_pred(belief),
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

    def _forward_from_snapshot(
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
    ) -> tuple[WMRMAux, Tensor, Tensor]:
        if world_goal is not None:
            raise ValueError(
                "world_goal is sealed: realized futures must not enter the "
                "forward mixer; pass them only to wmrm_world_loss"
            )
        if reuse_aux is not None:
            raise ValueError(
                "reuse_aux is unsupported: reusing predictor outputs with newly "
                "computed evidence/belief creates a hybrid snapshot"
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

        batch, _, hidden = action.shape

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
        # Belief is a recurrent residual state across 8 stages x T decisions.
        # Read the incoming value as memory, then publish a differentiable
        # residual update.  This keeps an identity credit path to earlier
        # stages while preventing the shared nonlinear reader Jacobians from
        # being multiplied through the full recurrent chain.
        belief_context = belief.detach()
        predicted = self.evidence_from_belief(belief_context)
        innovation = evidence - predicted
        if prev_innovation is not None:
            innovation = self._project_out(innovation, prev_innovation)
        belief_update = self.belief_write(belief_context, innovation)
        belief = _st_scale_residual(
            belief,
            belief_update,
            base_scale=_BELIEF_RETENTION,
            delta_scale=1.0 - _BELIEF_RETENTION,
        )

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
            # The predicted map has its own differentiable loss/message paths.
            # Here it is read as recurrent memory so the belief recurrence also
            # remains an identity residual rather than another I + J product.
            belief_update = self.belief_from_world(
                belief.detach(), world_tokens.detach()
            )
            belief = _st_scale_residual(
                belief,
                belief_update,
                base_scale=_BELIEF_RETENTION,
                delta_scale=1.0 - _BELIEF_RETENTION,
            )
        elif self.dino_to_hid is not None:
            # Intermediate VA↔WM exchanges use the recurrent WM belief as
            # spatially addressable memory; the final layer replaces it with
            # the predicted DINO map tokens.
            world_tokens = belief
        aux = WMRMAux(
            z_hat=z_hat,
            z_spans=z_spans,
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
            predict_belief=predict_belief,
        )
        return aux, belief, innovation

    def propose(
        self,
        action: Tensor,
        vision: Tensor,
        proprio: Tensor,
        *,
        state: WAMState | None = None,
        language_keys: Tensor | None = None,
        dino_tokens: Tensor | None = None,
        env_action: Tensor | None = None,
        reuse_aux: WMRMAux | None = None,
        stage_index: int = 0,
    ) -> WAMProposal:
        """Compute a World proposal without mutating or advancing the snapshot."""
        snapshot = WAMState() if state is None else state
        snapshot.validate_for(
            batch=action.shape[0],
            hidden_dim=self.hidden_dim,
            n_belief=self.n_belief,
            n_evidence=self.n_evidence,
            dino_dim=self.dino_dim,
            map_size=self.map_size,
            device=action.device,
        )
        aux, belief, innovation = self._forward_from_snapshot(
            action,
            vision,
            proprio,
            belief=snapshot.belief,
            prev_innovation=snapshot.innovation,
            language_keys=language_keys,
            dino_tokens=dino_tokens,
            env_action=env_action,
            reuse_aux=reuse_aux,
            stage_index=stage_index,
            previous_map=snapshot.world_map,
        )
        next_world_map = (
            aux.z_tokens
            if aux.z_tokens is not None and aux.z_tokens.ndim == 4
            else snapshot.world_map
        )
        # Canonicalize only the persisted peer snapshot. Transient proposal and
        # auxiliary tensors retain their feature-path dtype for legacy math.
        state_belief = belief.float()
        state_innovation = innovation.float()
        state_world_map = None if next_world_map is None else next_world_map.float()
        return WAMProposal(
            next_world_state=WAMState(
                belief=state_belief,
                innovation=state_innovation,
                world_map=state_world_map,
            ),
            world_message=(
                aux.world_tokens
                if aux.world_tokens is not None
                else belief
            ),
            aux=aux,
        )


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
