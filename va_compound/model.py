from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AttentionMode = Literal["bidir_va", "uni_a"]


@dataclass(frozen=True)
class VACompoundConfig:
    language_dim: int = 2048
    vision_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    action_horizon: int = 8
    action_dim: int = 7
    proprio_dim: int = 14
    flow_layers: int = 2
    dropout: float = 0.0
    mode: AttentionMode = "bidir_va"
    # Su Shen-inspired upgrade (default off, backward compatible).
    qk_norm: bool = False  # per-head RMSNorm on Q/K before the dot product
    # 源测度校正注意力（SMC-Attn，2026-08-05 原创设计）：softmax 前对每个来源减去
    # log N_s，使比较的是"平均证据"而非"token 数量 × 平均证据"——同一来源被复制/细分
    # 不再获得额外票数。flat = 原始共享 softmax（向后兼容）。
    attention_variant: str = "flat"  # "flat" | "smc"
    # Qwen-conditioned action queries（2026-08-06 GPT 方案 A）：语言摘要经 MLP 生成
    # 每 horizon 步的 action-query 偏移（zero-init，初始等价于静态 query）。Qwen 从
    # "被动被读的 K/V"升级为"决定动作查询找什么"的慢脑；VA 每步用视觉/状态修正。
    action_query_cond: bool = False
    # 因果分解记忆（2026-08-07 审阅落地②）：evidence（受保护证据记忆，仅 K/V、
    # 门控更新自 视觉+状态+旧证据，动作/语言不能写入）+ task（动态任务工作区，
    # 读写流，残差门控）。memory_split=False 时行为与旧版完全一致。
    memory_split: bool = False
    evidence_tokens: int = 16
    task_tokens: int = 8
    # 未来 latent 预测（2026-08-07 审阅落地③，可选）：轻量 P_psi(E,T,C) 预测
    # 下一决策点的冻结 V-JEPA 特征（stop-grad target），使 Action→Vision 反向
    # 通路学习"执行后状态变化"而非普通特征混合。训练 loss 见 train.py。
    future_predict: bool = False
    # 顺序式 A→V→A 耦合（2026-08-07 审阅落地④）：每 N 层使用
    # proposal→reorganize→correction 三遍注意力；0 = 全层同步联合（旧行为）。
    sequential_coupling: int = 0
    # flow 头深度条件（2026-08-07 审阅落地④）：每层 AdaLN-Zero + cross-attn
    # 注入条件（"entry" 为旧行为：仅入口相加）。
    flow_cond: str = "entry"
    # EVSM：证据验证的暂存记忆（2026-08-07 Codex 主推，可选）：动作提议写入
    # 暂存 task_spec，下一决策用 FutureLatentPredictor 的预测与真实视觉比较，
    # q = sigma((kappa - stopgrad(delta)) / temp) 决定提交或回滚。零新增参数
    # （复用 future_predict 通道），仅改变任务记忆的提交时序。
    evsm: bool = False
    evsm_kappa: float = 0.02  # delta 阈值：delta < kappa 才高概率提交
    evsm_temp: float = 0.005  # 门控温度（q 曲线陡峭度）
    # Plan-Cache（2026-08-07 Codex 评审 MVP，方案 B）：场景条件化 plan tokens。
    # PlanResampler 用场景摘要（vision_tokens 全局均值）+ 语言 hidden 生成 8 个
    # plan tokens，cat 到语言序列后再 build_language_cache——plan tokens 自然进入
    # 既有 per-layer K/V 缓存，encode_condition 与 VA attention 完全不动。
    plan_resampler: bool = False
    # Plan-Cache 方案 A（Qwen 看场景 teacher）：训练时用 encode_with_scene 在线
    # 计算 readout plan hidden（冻结 Qwen 带梯度），同样 cat 进语言缓存。
    scene_teacher: bool = False
    # C²-VA Stage A（2026-08-07）：Direct Head 替代 Flow。确定性 2 层 MLP →
    # tanh，直接回归归一化 executed 动作标签（v5 数据：denorm→clip(raw,-1,1)→
    # 重新归一化，一个执行动作只对应一个标签）。默认 False 保持 flow 路径。
    direct_head: bool = False
    # C²-VA Stage B（2026-08-07 Codex 评审版）：收缩控制型 VA。每个 Action
    # Token = {名义动作 ū, 期望视觉状态 c̄, 反馈增益 K}，执行
    # a_i = clip(ū_i − K_i·(c_current − c̄_i), −1, 1)。c_current 来自冻结的
    # PCA 视觉控制投影 P（recovery 差空间 top-16 PCA + whitening）。
    # 训练用恢复残差损失监督 K；L_contract 仅为 held-out 指标（离线 successor
    # 下对 K 无梯度，λc 默认 0）。需要 direct_head；默认 False 完全兼容。
    c2_controller: bool = False
    c2_control_dim: int = 16  # 控制状态维度（P 与 c̄ 的投影维度）

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.flow_layers < 1:
            raise ValueError("flow_layers must be positive")
        if self.mode not in ("bidir_va", "uni_a"):
            raise ValueError(f"unsupported attention mode: {self.mode}")
        if self.attention_variant not in ("flat", "smc"):
            raise ValueError(f"unsupported attention variant: {self.attention_variant}")
        if self.sequential_coupling < 0:
            raise ValueError("sequential_coupling must be >= 0")
        if self.flow_cond not in ("entry", "adaln"):
            raise ValueError(f"unsupported flow conditioning: {self.flow_cond}")
        if self.c2_controller and not self.direct_head:
            raise ValueError("c2_controller requires direct_head")
        if self.c2_control_dim < 1:
            raise ValueError("c2_control_dim must be positive")


@dataclass(frozen=True)
class LayerLanguageCache:
    key: Tensor
    value: Tensor

    def detach(self) -> "LayerLanguageCache":
        return LayerLanguageCache(self.key.detach(), self.value.detach())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LayerLanguageCache":
        return LayerLanguageCache(
            self.key.to(device=device, dtype=dtype),
            self.value.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class LanguageCache:
    layers: tuple[LayerLanguageCache, ...]
    attention_mask: Tensor

    def detach(self) -> "LanguageCache":
        return LanguageCache(
            layers=tuple(layer.detach() for layer in self.layers),
            attention_mask=self.attention_mask.detach(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LanguageCache":
        return LanguageCache(
            layers=tuple(layer.to(device=device, dtype=dtype) for layer in self.layers),
            attention_mask=self.attention_mask.to(device=device, dtype=torch.bool),
        )


@dataclass(frozen=True)
class VisualMemory:
    """One goal-conditioned visual state per VA layer from the previous step.

    With ``memory_split=True`` the recurrent state is the causal-decomposed
    pair (evidence, task) instead of per-layer visual snapshots: ``layers``
    is then empty and ``evidence``/``task`` carry the two memory streams.

    EVSM extension (``evsm=True``): ``task`` is the *committed* task state,
    ``task_spec`` is the speculative proposal from the previous decision
    (action intent written to scratch, not yet verified by evidence), and
    ``pending_future`` is the future-latent prediction issued alongside
    ``task_spec``.  At the next decision the model compares ``pending_future``
    against the observed vision latent: on agreement ``task_spec`` is
    committed, otherwise it is rolled back (intent can propose; only evidence
    can commit).
    """

    layers: tuple[Tensor, ...]
    evidence: Tensor | None = None
    task: Tensor | None = None
    task_spec: Tensor | None = None
    pending_future: Tensor | None = None
    gate: float | None = None  # EVSM commit-gate mean at this step (diagnostics)

    def __init__(
        self,
        layers: tuple[Tensor, ...],
        evidence: Tensor | None = None,
        task: Tensor | None = None,
        task_spec: Tensor | None = None,
        pending_future: Tensor | None = None,
        gate: float | None = None,
    ) -> None:
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "task_spec", task_spec)
        object.__setattr__(self, "pending_future", pending_future)
        object.__setattr__(self, "gate", gate)

    def detach(self) -> "VisualMemory":
        return VisualMemory(
            layers=tuple(layer.detach() for layer in self.layers),
            evidence=self.evidence.detach() if self.evidence is not None else None,
            task=self.task.detach() if self.task is not None else None,
            task_spec=self.task_spec.detach() if self.task_spec is not None else None,
            pending_future=(
                self.pending_future.detach() if self.pending_future is not None else None
            ),
            gate=self.gate,
        )

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "VisualMemory":
        if isinstance(device, torch.dtype) and dtype is None:
            dtype, device = device, None
        return VisualMemory(
            layers=tuple(layer.to(device=device, dtype=dtype) for layer in self.layers),
            evidence=self.evidence.to(device=device, dtype=dtype) if self.evidence is not None else None,
            task=self.task.to(device=device, dtype=dtype) if self.task is not None else None,
            task_spec=self.task_spec.to(device=device, dtype=dtype) if self.task_spec is not None else None,
            pending_future=(
                self.pending_future.to(device=device, dtype=dtype)
                if self.pending_future is not None
                else None
            ),
            gate=self.gate,
        )


class EvidenceGate(nn.Module):
    """Protected evidence memory update (2026-08-07).

    E_t = (1-g) E_{t-1} + g E_tilde, where E_tilde is a cross-attention
    readout of E_{t-1} over [V_t, S_t] (current frozen vision + robot state).
    Action proposals and language never appear in the update inputs, so
    executed-state evidence cannot be overwritten by intent (anti
    confirmation-bias).  First step (no previous evidence): full overwrite.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_evidence: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(
        self, evidence: Tensor, vision: Tensor, state: Tensor, overwrite: bool = False
    ) -> Tensor:
        B, n_ev, H = evidence.shape
        # Readout: E queries over [V_t; S_t] keys.
        q = self._heads(self.q(self.norm_q(evidence)))  # [B, heads, n_ev, hd]
        context = torch.cat((vision, state[:, None]), dim=1)
        k = self._heads(self.k(self.norm_k(context)))
        v = self._heads(self.v(context))
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
            H // self.num_heads
        )
        weights = torch.softmax(scores, dim=-1).to(dtype=evidence.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        update = self._from_heads(torch.matmul(weights, v))
        tilde = evidence + self.out(update)
        tilde = tilde + self.ffn(self.norm(tilde))
        if overwrite:
            return tilde  # first step: full overwrite (no previous evidence)
        # Gate on [E_prev, E_tilde, innovation].
        g = torch.sigmoid(
            self.gate(
                torch.cat((evidence, tilde, tilde - evidence), dim=-1)
            )
        )
        return (1.0 - g) * evidence + g * tilde

    def _heads(self, x: Tensor) -> Tensor:
        B, N, H = x.shape
        return x.view(B, N, self.num_heads, H // self.num_heads).transpose(1, 2)

    def _from_heads(self, x: Tensor) -> Tensor:
        B, Hh, N, hd = x.shape
        return x.transpose(1, 2).reshape(B, N, Hh * hd)


class TaskResampler(nn.Module):
    """Language-conditioned task-workspace initialization (2026-08-07).

    T_0 = task_queries + MLP(mask-weighted language summary), run once per
    episode so the workspace starts from the language contract.
    """

    def __init__(self, hidden_dim: int, n_task: int) -> None:
        super().__init__()
        self.task_queries = nn.Parameter(torch.empty(n_task, hidden_dim))
        nn.init.normal_(self.task_queries, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, language_key: Tensor, language_mask: Tensor) -> Tensor:
        """language_key: [B, tokens, hidden] (layer-0 projected key, flattened)."""
        denom = language_mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        summary = (language_key * language_mask[:, :, None]).sum(1) / denom
        return self.task_queries[None] + self.mlp(summary[:, None, :])


class TaskGate(nn.Module):
    """Gated residual update of the task workspace (2026-08-07).

    T_t = T_{t-1} + g^T (T_hat - T_{t-1}), g^T from [T_prev, T_hat, diff].
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, prev: Tensor, hat: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(torch.cat((prev, hat, hat - prev), dim=-1)))
        return prev + g * (hat - prev)


class PlanResampler(nn.Module):
    """Scene-conditioned plan tokens (2026-08-07 Plan-Cache MVP, 方案 B).

    A fixed set of learned plan queries reads the scene summary (global mean
    of the frozen V-JEPA tokens) together with the language hidden states via
    a single cross-attention layer + FFN.  The output plan tokens are appended
    to the language sequence before ``build_language_cache``, so they enter the
    existing per-layer K/V cache naturally and neither ``encode_condition``
    nor the VA attention needs to change.
    """

    def __init__(
        self,
        language_dim: int,
        vision_dim: int,
        n_plan: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_plan = n_plan
        self.num_heads = num_heads
        self.dropout = dropout
        self.plan_queries = nn.Parameter(torch.empty(n_plan, language_dim))
        nn.init.normal_(self.plan_queries, std=0.02)
        self.scene_proj = nn.Linear(vision_dim, language_dim)
        self.norm_q = nn.LayerNorm(language_dim)
        self.norm_k = nn.LayerNorm(language_dim)
        self.q = nn.Linear(language_dim, language_dim)
        self.k = nn.Linear(language_dim, language_dim)
        self.v = nn.Linear(language_dim, language_dim)
        self.out = nn.Linear(language_dim, language_dim)
        self.norm = nn.LayerNorm(language_dim)
        self.ffn = nn.Sequential(
            nn.Linear(language_dim, 2 * language_dim),
            nn.GELU(),
            nn.Linear(2 * language_dim, language_dim),
        )

    def forward(
        self,
        scene_summary: Tensor,
        language_hidden: Tensor,
        language_mask: Tensor | None = None,
    ) -> Tensor:
        """scene_summary: [B, vision_dim]; language_hidden: [B, L, language_dim].

        Returns plan tokens [B, n_plan, language_dim] (in the dtype of the
        module parameters; callers cast to the language dtype when caching).
        """
        if scene_summary.ndim != 2 or scene_summary.shape[0] != language_hidden.shape[0]:
            raise ValueError("scene_summary must have shape [batch, vision_dim]")
        if language_hidden.ndim != 3:
            raise ValueError("language_hidden must have shape [batch, tokens, language_dim]")
        batch, length, dim = language_hidden.shape
        target_dtype = self.q.weight.dtype
        scene = self.scene_proj(scene_summary.to(dtype=target_dtype))[:, None]  # [B, 1, D]
        kv = torch.cat((scene, language_hidden.to(dtype=target_dtype)), dim=1)  # [B, L+1, D]
        if language_mask is None:
            mask = torch.ones(batch, length + 1, dtype=torch.bool, device=kv.device)
        else:
            if language_mask.shape != (batch, length):
                raise ValueError("language_mask must match [batch, language_tokens]")
            mask = torch.cat(
                (torch.ones(batch, 1, dtype=torch.bool, device=kv.device), language_mask),
                dim=1,
            )
        query = self._heads(self.q(self.norm_q(self.plan_queries[None].expand(batch, -1, -1))))
        key = self._heads(self.k(self.norm_k(kv)))
        value = self._heads(self.v(kv))
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
            dim // self.num_heads
        )
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(dtype=target_dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        plan = self._from_heads(torch.matmul(weights, value))
        plan = self.out(plan)
        plan = plan + self.ffn(self.norm(plan))
        return plan

    def _heads(self, x: Tensor) -> Tensor:
        B, N, H = x.shape
        return x.view(B, N, self.num_heads, H // self.num_heads).transpose(1, 2)

    def _from_heads(self, x: Tensor) -> Tensor:
        B, Hh, N, hd = x.shape
        return x.transpose(1, 2).reshape(B, N, Hh * hd)


class FutureLatentPredictor(nn.Module):
    """Lightweight future-latent predictor (2026-08-07 审阅落地③).

    Maps pooled (evidence, task workspace, action condition) to the mean of
    the NEXT decision's frozen V-JEPA features.  The target comes from the
    precomputed feature bank (stop-gradient by construction), so the loss
    measures whether the action condition carries predictive power about
    executed-state change.  Optional module: with ``future_predict=False``
    the architecture is unchanged.
    """

    def __init__(self, hidden_dim: int, vision_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vision_dim),
        )

    def forward(
        self,
        action_condition: Tensor,
        evidence: Tensor | None,
        task: Tensor | None,
    ) -> Tensor:
        parts = [action_condition.mean(dim=1)]
        if evidence is not None:
            parts.append(evidence.mean(dim=1))
        if task is not None:
            parts.append(task.mean(dim=1))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)  # [B, vision_dim]

    @staticmethod
    def future_loss(predicted: Tensor, target: Tensor) -> Tensor:
        """1 - cosine similarity on normalized vectors (scale-agnostic)."""
        p = F.normalize(predicted, dim=-1)
        t = F.normalize(target.detach(), dim=-1)
        return (1.0 - (p * t).sum(-1)).mean()


class VACouplingLayer(nn.Module):
    """Layer with visual/action streams, optionally task stream + extra K/V.

    Streams (query sets): V, A, and (with memory_split) T.
    K/V sources: V, memory (visual snapshot or protected evidence), A, T,
    language, and (with memory_split) an explicit robot-state token.
    Evidence tokens are K/V-only by construction: attention can read them but
    no query group can write into them (protection against action-intent
    contamination, 2026-08-07).
    """
    """One shared-attention layer for visual and action tokens.

    Key layout is [current vision, previous visual memory, action, language].
    The memory is absent on the first step. In ``uni_a``
    mode visual queries can only read visual keys; action queries still read
    every stream. This keeps the parameter budget matched while removing the
    proposed Memory/Action/Language -> Vision paths.
    """

    def __init__(
        self,
        hidden_dim: int,
        language_dim: int,
        num_heads: int,
        dropout: float,
        mode: AttentionMode,
        *,
        qk_norm: bool = False,
        attention_variant: str = "flat",
        sequential: bool = False,
    ) -> None:
        super().__init__()
        self.sequential = sequential
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mode = mode
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.attention_variant = attention_variant
        self.save_attention = False  # 可选调试开关：记录 attention weights（默认关闭，不影响训练）

        self.norm_v_attn = nn.LayerNorm(hidden_dim)
        self.norm_m_attn = nn.LayerNorm(hidden_dim)
        self.norm_a_attn = nn.LayerNorm(hidden_dim)
        self.norm_l = nn.LayerNorm(language_dim)

        self.q_v = nn.Linear(hidden_dim, hidden_dim)
        self.k_v = nn.Linear(hidden_dim, hidden_dim)
        self.u_v = nn.Linear(hidden_dim, hidden_dim)
        self.k_m = nn.Linear(hidden_dim, hidden_dim)
        self.u_m = nn.Linear(hidden_dim, hidden_dim)
        self.q_a = nn.Linear(hidden_dim, hidden_dim)
        self.k_a = nn.Linear(hidden_dim, hidden_dim)
        self.u_a = nn.Linear(hidden_dim, hidden_dim)
        self.k_l = nn.Linear(language_dim, hidden_dim)
        self.u_l = nn.Linear(language_dim, hidden_dim)
        # memory_split 扩展（2026-08-07）：task 流 + 显式 state K/V 源。
        self.norm_t_attn = nn.LayerNorm(hidden_dim)
        self.q_t = nn.Linear(hidden_dim, hidden_dim)
        self.k_t = nn.Linear(hidden_dim, hidden_dim)
        self.u_t = nn.Linear(hidden_dim, hidden_dim)
        self.out_t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t_ffn = nn.LayerNorm(hidden_dim)
        self.ffn_t = self._make_ffn(hidden_dim, dropout)
        self.norm_s_attn = nn.LayerNorm(hidden_dim)
        self.k_s = nn.Linear(hidden_dim, hidden_dim)
        self.u_s = nn.Linear(hidden_dim, hidden_dim)

        self.out_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_a = nn.Linear(hidden_dim, hidden_dim)

        self.norm_v_ffn = nn.LayerNorm(hidden_dim)
        self.norm_a_ffn = nn.LayerNorm(hidden_dim)
        self.ffn_v = self._make_ffn(hidden_dim, dropout)
        self.ffn_a = self._make_ffn(hidden_dim, dropout)

    @staticmethod
    def _make_ffn(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _to_heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x: Tensor) -> Tensor:
        batch, _, tokens, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, tokens, self.hidden_dim)

    def project_language(self, hidden: Tensor) -> LayerLanguageCache:
        hidden = hidden.to(dtype=self.norm_l.weight.dtype)
        hidden = self.norm_l(hidden)
        return LayerLanguageCache(
            key=self._to_heads(self.k_l(hidden)),
            value=self._to_heads(self.u_l(hidden)),
        )

    def _role_mask(
        self,
        n_visual: int,
        n_memory: int,
        n_action: int,
        n_language: int,
        n_task: int,
        n_state: int,
        device: torch.device,
    ) -> Tensor:
        n_query = n_visual + n_action + n_task
        n_key = n_visual + n_memory + n_action + n_task + n_language + n_state
        allowed = torch.ones((n_query, n_key), dtype=torch.bool, device=device)
        if self.mode == "uni_a":
            allowed[:n_visual] = False
            allowed[:n_visual, :n_visual] = True
        return allowed

    def forward(
        self,
        visual: Tensor,
        action: Tensor,
        language: LayerLanguageCache,
        language_mask: Tensor,
        visual_memory: Tensor | None = None,
        task: Tensor | None = None,
        evidence: Tensor | None = None,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        visual_norm = self.norm_v_attn(visual)
        action_norm = self.norm_a_attn(action)

        query_parts = [self._to_heads(self.q_v(visual_norm)), self._to_heads(self.q_a(action_norm))]
        key_parts = [self._to_heads(self.k_v(visual_norm))]
        value_parts = [self._to_heads(self.u_v(visual_norm))]
        n_memory = 0
        memory_tokens: Tensor | None = visual_memory
        if evidence is not None:
            memory_tokens = evidence  # memory_split: protected evidence replaces snapshots
        if memory_tokens is not None:
            memory_norm = self.norm_m_attn(memory_tokens)
            key_parts.append(self._to_heads(self.k_m(memory_norm)))
            value_parts.append(self._to_heads(self.u_m(memory_norm)))
            n_memory = memory_tokens.shape[1]
        n_task = 0
        if task is not None:
            task_norm = self.norm_t_attn(task)
            query_parts.append(self._to_heads(self.q_t(task_norm)))
            key_parts.append(self._to_heads(self.k_t(task_norm)))
            value_parts.append(self._to_heads(self.u_t(task_norm)))
            n_task = task.shape[1]
        key_parts.extend((self._to_heads(self.k_a(action_norm)), language.key))
        value_parts.extend((self._to_heads(self.u_a(action_norm)), language.value))
        n_state = 0
        if state is not None:
            state_norm = self.norm_s_attn(state)
            key_parts.append(self._to_heads(self.k_s(state_norm)))
            value_parts.append(self._to_heads(self.u_s(state_norm)))
            n_state = state.shape[1]
        query = torch.cat(query_parts, dim=2)
        key = torch.cat(key_parts, dim=2)
        value = torch.cat(value_parts, dim=2)

        n_visual = visual.shape[1]
        n_action = action.shape[1]
        n_language = language_mask.shape[1]
        role_mask = self._role_mask(
            n_visual,
            n_memory,
            n_action,
            n_language,
            n_task,
            n_state,
            visual.device,
        )
        key_mask = torch.cat(
            (
                torch.ones(
                    (visual.shape[0], n_visual + n_memory + n_task + n_action),
                    dtype=torch.bool,
                    device=visual.device,
                ),
                language_mask.to(device=visual.device, dtype=torch.bool),
                torch.ones(
                    (visual.shape[0], n_state),
                    dtype=torch.bool,
                    device=visual.device,
                ),
            ),
            dim=1,
        )
        allowed = role_mask[None, None] & key_mask[:, None, None]

        if self.qk_norm:
            query = F.rms_norm(query, (query.shape[-1],))
            key = F.rms_norm(key, (key.shape[-1],))
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scale
        if self.attention_variant == "smc":
            # SMC-Attn：源测度校正（2026-08-05 原创）。
            # softmax 前对每个来源减去 log N_s → 来源总质量 ∝ 平均证据而非
            # token 数量 × 平均证据；复制/细分同一来源不再稀释其他来源。
            # 只作用于动作 query 行（视觉 query 保持 uni_a 语义），L 计数按
            # 实际有效 mask 计算（padding 不计入）。
            n_lang = key.shape[2] - (n_visual + n_memory + n_action + n_task + n_state)
            log_n_v = math.log(max(1, n_visual))
            log_n_m = math.log(max(1, n_memory))
            log_n_a = math.log(max(1, n_action))
            log_n_t = math.log(max(1, n_task))
            log_n_s = math.log(max(1, n_state))
            log_n_l = torch.log(
                language_mask.to(scores.dtype).sum(-1).clamp(min=1.0)
            )  # [B]
            log_measure = torch.zeros(
                (scores.shape[0], key.shape[2]),
                device=scores.device,
                dtype=scores.dtype,
            )
            off = 0
            for n_group, log_n_group in (
                (n_visual, log_n_v),
                (n_memory, log_n_m),
                (n_action, log_n_a),
                (n_task, log_n_t),
                (n_state, log_n_s),
            ):
                log_measure[:, off : off + n_group] = log_n_group
                off += n_group
            log_measure[:, off:] = log_n_l[:, None]
            scores[:, :, n_visual:, :] -= log_measure[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        self.last_max_logit = float(
            scores.detach().masked_fill(~allowed, float("-inf")).amax()
        )
        weights = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        update = self._from_heads(torch.matmul(weights, value))

        n_query_v, n_query_a, n_query_t = n_visual, n_action, n_task
        if n_query_t:
            update_v, update_a, update_t = update.split(
                (n_query_v, n_query_a, n_query_t), dim=1
            )
        else:
            update_v, update_a = update.split((n_query_v, n_query_a), dim=1)
            update_t = None
        visual = visual + self.out_v(update_v)
        action = action + self.out_a(update_a)
        visual = visual + self.ffn_v(self.norm_v_ffn(visual))
        action = action + self.ffn_a(self.norm_a_ffn(action))
        task_out: Tensor | None = None
        if n_query_t:
            task = task + self.out_t(update_t)
            task = task + self.ffn_t(self.norm_t_ffn(task))
            task_out = task
        if self.save_attention:
            self._last_attention = (
                weights.detach().clone(),  # [B, heads, n_query, n_key]
                n_visual,
                n_memory,
                n_action,
                n_language,
                n_task,
                n_state,
            )
        return visual, action, task_out

    def _attend(
        self,
        q_tokens: Tensor,
        q_proj: nn.Linear,
        q_norm: nn.LayerNorm,
        groups: list[tuple[Tensor, nn.Linear, nn.Linear, nn.LayerNorm]],
        language: LayerLanguageCache | None,
        language_mask: Tensor | None,
    ) -> Tensor:
        """Generic masked multi-head attention over K/V groups + language."""
        q = self._to_heads(q_proj(q_norm(q_tokens)))
        key_parts: list[Tensor] = []
        value_parts: list[Tensor] = []
        n_other = 0
        for tokens, k_proj, u_proj, norm in groups:
            key_parts.append(self._to_heads(k_proj(norm(tokens))))
            value_parts.append(self._to_heads(u_proj(tokens)))
            n_other += tokens.shape[1]
        if language is not None:
            key_parts.append(language.key)
            value_parts.append(language.value)
        key = torch.cat(key_parts, dim=2)
        value = torch.cat(value_parts, dim=2)
        scores = torch.matmul(q.float(), key.float().transpose(-1, -2)) * self.scale
        if language is not None and language_mask is not None:
            key_mask = torch.cat(
                (
                    torch.ones(
                        (q_tokens.shape[0], n_other),
                        dtype=torch.bool,
                        device=q_tokens.device,
                    ),
                    language_mask.to(device=q_tokens.device, dtype=torch.bool),
                ),
                dim=1,
            )
            scores = scores.masked_fill(
                ~key_mask[:, None, None], torch.finfo(scores.dtype).min
            )
        weights = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return self._from_heads(torch.matmul(weights, value))

    def forward_sequential(
        self,
        visual: Tensor,
        action: Tensor,
        language: LayerLanguageCache,
        language_mask: Tensor,
        visual_memory: Tensor | None = None,
        task: Tensor | None = None,
        evidence: Tensor | None = None,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Sequential A->V/T->A coupling (2026-08-07 审阅落地④).

        Pass 1 (action proposal):  A_half <- A + Attn(A; [V, M, T, L, S])
        Pass 2 (reorganize):       V',T' <- V,T + Attn([V,T]; [V, M, A_half, T, L, S])
        Pass 3 (correction):       A' <- A_half + Attn(A_half; [V', M, A_half, T', L, S])

        Each pass applies the layer's own FFN.  Evidence stays K/V-only.
        """
        memory_tokens = evidence if evidence is not None else visual_memory
        base_groups: list[tuple[Tensor, nn.Linear, nn.Linear, nn.LayerNorm]] = []
        base_groups.append((visual, self.k_v, self.u_v, self.norm_v_attn))
        if memory_tokens is not None:
            base_groups.append((memory_tokens, self.k_m, self.u_m, self.norm_m_attn))
        if task is not None:
            base_groups.append((task, self.k_t, self.u_t, self.norm_t_attn))
        if state is not None:
            base_groups.append((state, self.k_s, self.u_s, self.norm_s_attn))

        # Pass 1: action proposal reads context only (no self-attention on A).
        proposal = self._attend(
            action, self.q_a, self.norm_a_attn, base_groups, language, language_mask
        )
        action_half = action + self.out_a(proposal)
        action_half = action_half + self.ffn_a(self.norm_a_ffn(action_half))

        # Pass 2: vision + task reorganize under the action hypothesis.
        groups_2 = base_groups + [(action_half, self.k_a, self.u_a, self.norm_a_attn)]
        update_v = self._attend(
            visual, self.q_v, self.norm_v_attn, groups_2, language, language_mask
        )
        visual_new = visual + self.out_v(update_v)
        visual_new = visual_new + self.ffn_v(self.norm_v_ffn(visual_new))
        task_new: Tensor | None = None
        if task is not None:
            update_t = self._attend(
                task, self.q_t, self.norm_t_attn, groups_2, language, language_mask
            )
            task_new = task + self.out_t(update_t)
            task_new = task_new + self.ffn_t(self.norm_t_ffn(task_new))

        # Pass 3: action correction reads the reorganized visual/task state.
        groups_3: list[tuple[Tensor, nn.Linear, nn.Linear, nn.LayerNorm]] = [
            (visual_new, self.k_v, self.u_v, self.norm_v_attn)
        ]
        if memory_tokens is not None:
            groups_3.append((memory_tokens, self.k_m, self.u_m, self.norm_m_attn))
        groups_3.append((action_half, self.k_a, self.u_a, self.norm_a_attn))
        if task_new is not None:
            groups_3.append((task_new, self.k_t, self.u_t, self.norm_t_attn))
        if state is not None:
            groups_3.append((state, self.k_s, self.u_s, self.norm_s_attn))
        update_corr = self._attend(
            action_half, self.q_a, self.norm_a_attn, groups_3, language, language_mask
        )
        action_new = action_half + self.out_a(update_corr)
        action_new = action_new + self.ffn_a(self.norm_a_ffn(action_new))
        return visual_new, action_new, task_new


class DirectActionHead(nn.Module):
    """Direct action decoder（C²-VA Stage A，2026-08-07）。

    ``action_condition`` [B, horizon, hidden_dim] → 2 层 MLP（hidden 256，
    GELU）→ Linear → [B, horizon, action_dim]，输出经 tanh 钳制到 (-1, 1)。
    训练标签是归一化 executed 动作（∈[-1,1]，v5 数据），一次前向得到完整
    chunk，替代 flow matching 的 32 步 ODE 采样；确定性解码（无采样噪声）。
    """

    def __init__(self, hidden_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, action_condition: Tensor) -> Tensor:
        if action_condition.ndim != 3:
            raise ValueError(
                "action_condition must have shape [batch, horizon, hidden_dim]"
            )
        return self.net(action_condition)


@dataclass(frozen=True)
class ControllerParams:
    """C² 控制器参数 {ū, c̄, K}（Codex 评审 2026-08-07 布局）。

    - nominal:    ū [B, H, A]——名义动作（DirectActionHead 输出，Stage A 兼容）；
    - reference:  c̄ [B, H, C]——每 token 期望视觉状态（= sg(c_anchor) + Δc，
      Δc_0 ≡ 0，即 token 0 的参考 ≡ 当前投影 → e_0 = 0）；
    - gain:       K [B, H, A, C]——反馈增益（全量矩阵，不做低秩；零初始化）。
    """

    nominal: Tensor  # [B, H, A]
    reference: Tensor  # [B, H, C]
    gain: Tensor  # [B, H, A, C]

    def apply_controller(
        self,
        c_current: Tensor,
        *,
        oracle_reference: bool = False,
    ) -> Tensor:
        """a_i = clip(ū_i − K_i·e_i, −1, 1)，e_i = c_current − c̄_i。

        ``oracle_reference=True``：c̄ ≡ c_current（e ≡ 0），测量"参考零误差"
        上界（部署消融 --c2-oracle-ref）。
        """
        if c_current.ndim != 2:
            raise ValueError("c_current must have shape [batch, control_dim]")
        if c_current.shape[0] != self.nominal.shape[0]:
            raise ValueError("c_current batch size must match the controller params")
        if oracle_reference:
            e = torch.zeros_like(self.reference)
        else:
            e = c_current[:, None, :] - self.reference
        correction = torch.einsum("bhc,bhac->bha", e, self.gain)
        return torch.clamp(self.nominal - correction, -1.0, 1.0)


class ControllableProjection(nn.Module):
    """C² 视觉控制投影 P：z [B, vision_dim] → c [B, control_dim]（冻结）。

    c = Linear(LayerNorm(mean_tokens))。权重由 prepare_mw_recovery.py 从
    recovery 差空间（z_perturbed − z_nominal）top-16 PCA + whitening 计算，
    经 ``set_pca`` 写入后整体冻结；c_current、c̄ 目标（v6a/v6b）共享同一 P。
    首轮不端到端学 P（避免 P→0 与 P→αP, K→K/α 退化，Codex 判决）。
    """

    def __init__(self, vision_dim: int = 768, control_dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.linear = nn.Linear(vision_dim, control_dim)

    def set_pca(self, weight: Tensor, bias: Tensor) -> None:
        """写入 PCA-whitening 权重（[C, vision_dim] / [C]）并冻结整个投影。"""
        expected = (self.linear.out_features, self.linear.in_features)
        if tuple(weight.shape) != expected:
            raise ValueError(f"pca weight must have shape {expected}, got {tuple(weight.shape)}")
        if tuple(bias.shape) != (self.linear.out_features,):
            raise ValueError(
                f"pca bias must have shape {(self.linear.out_features,)}, got {tuple(bias.shape)}"
            )
        with torch.no_grad():
            self.linear.weight.copy_(weight.to(device=self.linear.weight.device))
            self.linear.bias.copy_(bias.to(device=self.linear.bias.device))
        self.requires_grad_(False).eval()

    def forward(self, vision_tokens: Tensor) -> Tensor:
        if vision_tokens.ndim != 3:
            raise ValueError("vision_tokens must have shape [batch, tokens, vision_dim]")
        z = vision_tokens.mean(dim=1)
        return self.linear(self.norm(z.to(dtype=self.linear.weight.dtype)))


class C2ActionHead(nn.Module):
    """C²-VA Stage B 控制头（Codex 评审 2026-08-07 版）。

    ``action_condition`` [B, horizon, hidden_dim] → (delta, gain)：
    - delta：reference_head 输出的 Δc [B, H, C]，Δc_0 硬置 0（token 0 的
      参考 ≡ 当前投影）；c̄ = sg(c_anchor) + Δc（anchor 由调用方提供）；
    - gain：gain_head 输出的 K [B, H, A, C]，最后层 weight/bias 全零 →
      初始 K ≡ 0，行为严格等于 Stage A Direct Head（ū 由外部 direct_head
      提供，Stage A checkpoint 的权重直接复用）。
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        control_dim: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.control_dim = control_dim
        self.reference_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, control_dim),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim * control_dim),
        )
        # K 全零初始化：初始严格等价 Direct Head（防初始扰动，Codex 修正 3）。
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def forward(self, action_condition: Tensor) -> tuple[Tensor, Tensor]:
        if action_condition.ndim != 3:
            raise ValueError(
                "action_condition must have shape [batch, horizon, hidden_dim]"
            )
        batch, horizon, _ = action_condition.shape
        delta = self.reference_head(action_condition)  # [B, H, C]
        delta[:, 0] = 0.0  # Δc_0 ≡ 0（硬设，Codex 修正 2）
        gain = self.gain_head(action_condition).view(
            batch, horizon, self.action_dim, self.control_dim
        )
        return delta, gain


class FlowMatchingHead(nn.Module):
    """Small conditional vector field over a complete action chunk.

    The expensive VA condition is computed once per observation.  Only this
    head is evaluated repeatedly by the ODE solver.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        flow_cond: str = "entry",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.flow_cond_mode = flow_cond
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, action_dim)
        # 深度条件（2026-08-07）：AdaLN-Zero（6 调制量/层）+ 条件 cross-attn。
        if self.flow_cond_mode != "entry":
            self.ada_mlps = nn.ModuleList(
                nn.Linear(2 * hidden_dim, 6 * hidden_dim)
                for _ in range(num_layers)
            )
            self.ca_q = nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            )
            self.ca_k = nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            )
            self.ca_v = nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            )
            self.ca_out = nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            )
            self.ca_norms = nn.ModuleList(
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            )
            for mlp in self.ada_mlps:
                nn.init.zeros_(mlp.weight)
                nn.init.zeros_(mlp.bias)

        half_dim = hidden_dim // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half_dim, dtype=torch.float32) / denominator
        )
        self.register_buffer("time_frequencies", frequencies, persistent=False)

    def _time_embedding(self, flow_time: Tensor, dtype: torch.dtype) -> Tensor:
        if flow_time.ndim == 2 and flow_time.shape[1] == 1:
            flow_time = flow_time[:, 0]
        if flow_time.ndim != 1:
            raise ValueError("flow_time must have shape [batch] or [batch, 1]")
        angles = flow_time.float()[:, None] * self.time_frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return embedding.to(dtype=dtype)

    def forward(
        self,
        action_condition: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
    ) -> Tensor:
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape [batch, horizon, action_dim]")
        if noisy_actions.shape[:2] != action_condition.shape[:2]:
            raise ValueError("noisy_actions batch/horizon must match action_condition")
        time_embedding = self._time_embedding(flow_time, action_condition.dtype)
        if time_embedding.shape[0] != action_condition.shape[0]:
            raise ValueError("flow_time batch size must match action_condition")

        dtype = action_condition.dtype
        action = self.action_projection(noisy_actions.to(dtype=dtype))
        time = self.time_projection(time_embedding)[:, None]
        if self.flow_cond_mode == "entry":
            hidden = action_condition + action + time
            for layer in self.layers:
                hidden = layer(hidden)
        else:
            # adaln（标准 AdaLN-Zero，DiT 风格）：入口不含条件，条件完全经
            # 每层 scale/shift/gate 调制 + cross-attention 注入；零初始化
            # gate=0 → 起点为无条件流场，条件通道从零开始学习。
            hidden = action + time
            global_cond = torch.cat(
                (
                    time_embedding,
                    action_condition.mean(dim=1),
                ),
                dim=-1,
            )
            for layer, ada_mlp, w_q, w_k, w_v, w_o, ca_norm in zip(
                self.layers,
                self.ada_mlps,
                self.ca_q,
                self.ca_k,
                self.ca_v,
                self.ca_out,
                self.ca_norms,
                strict=True,
            ):
                mod = ada_mlp(global_cond).unsqueeze(1)  # [B,1,6H]
                scale1, shift1, gate1, scale2, shift2, gate2 = mod.chunk(6, dim=-1)
                # 子层 1：transformer block（AdaLN-Zero 调制 + 门控残差）
                h = hidden * (1.0 + scale1) + shift1
                h = layer(h)
                hidden = hidden + gate1 * h
                # 子层 2：条件 cross-attention（每层重新注入 action_condition）
                h = ca_norm(hidden) * (1.0 + scale2) + shift2
                q = self._ca_heads(w_q(h))
                k = self._ca_heads(w_k(action_condition))
                v = self._ca_heads(w_v(action_condition))
                scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (
                    (self.hidden_dim // self.num_heads) ** 0.5
                )
                w = torch.softmax(scores, dim=-1).to(dtype=h.dtype)
                update = self._ca_from_heads(torch.matmul(w, v))
                update = w_o(update)
                hidden = hidden + gate2 * update
        return self.velocity_head(self.output_norm(hidden))

    def _ca_heads(self, x: Tensor) -> Tensor:
        B, N, H = x.shape
        return x.view(B, N, self.num_heads, H // self.num_heads).transpose(1, 2)

    def _ca_from_heads(self, x: Tensor) -> Tensor:
        B, Hh, N, hd = x.shape
        return x.transpose(1, 2).reshape(B, N, Hh * hd)


class VACompoundPolicy(nn.Module):
    def __init__(self, config: VACompoundConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_projection = nn.Linear(config.vision_dim, config.hidden_dim)
        self.state_projection = nn.Linear(config.proprio_dim + config.action_dim, config.hidden_dim)
        self.action_queries = nn.Parameter(torch.empty(config.action_horizon, config.hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        if config.action_query_cond:
            # 语言摘要（cache 第 0 层投影 key 的 mask 加权均值）→ 每 horizon 步 query 偏移。
            # 最后一层 zero-init：训练开始时等价于静态 action_queries（不破坏现有行为）。
            self.lang_to_query = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.action_horizon * config.hidden_dim),
            )
            nn.init.zeros_(self.lang_to_query[-1].weight)
            nn.init.zeros_(self.lang_to_query[-1].bias)

        self.layers = nn.ModuleList(
            VACouplingLayer(
                hidden_dim=config.hidden_dim,
                language_dim=config.language_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                mode=config.mode,
                qk_norm=config.qk_norm,
                attention_variant=config.attention_variant,
                sequential=(
                    config.sequential_coupling > 0
                    and (index + 1) % config.sequential_coupling == 0
                ),
            )
            for index in range(config.num_layers)
        )
        if config.memory_split:
            self.evidence_init = nn.Parameter(
                torch.empty(1, config.evidence_tokens, config.hidden_dim)
            )
            nn.init.normal_(self.evidence_init, std=0.02)
            self.evidence_gate = EvidenceGate(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                n_evidence=config.evidence_tokens,
                dropout=config.dropout,
            )
            self.task_resampler = TaskResampler(
                hidden_dim=config.hidden_dim, n_task=config.task_tokens
            )
            self.task_gate = TaskGate(config.hidden_dim)
            self.task_to_action = nn.Linear(config.hidden_dim, config.hidden_dim)
        if config.future_predict:
            self.future_predictor = FutureLatentPredictor(
                hidden_dim=config.hidden_dim,
                vision_dim=config.vision_dim,
            )
        if config.plan_resampler:
            self.plan_resampler = PlanResampler(
                language_dim=config.language_dim,
                vision_dim=config.vision_dim,
                n_plan=8,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        self.action_norm = nn.LayerNorm(config.hidden_dim)
        self.flow_head = FlowMatchingHead(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            num_heads=config.num_heads,
            num_layers=config.flow_layers,
            dropout=config.dropout,
            flow_cond=config.flow_cond,
        )
        if config.direct_head:
            self.direct_head = DirectActionHead(
                hidden_dim=config.hidden_dim,
                action_dim=config.action_dim,
            )
        if config.c2_controller:
            self.c2_head = C2ActionHead(
                hidden_dim=config.hidden_dim,
                action_dim=config.action_dim,
                control_dim=config.c2_control_dim,
            )
            self.control_projector = ControllableProjection(
                vision_dim=config.vision_dim,
                control_dim=config.c2_control_dim,
            )

    def build_language_cache(
        self,
        language_hidden: Tensor,
        language_mask: Tensor | None = None,
        *,
        detach: bool = False,
    ) -> LanguageCache:
        if language_hidden.ndim != 3:
            raise ValueError("language_hidden must have shape [batch, tokens, language_dim]")
        if language_hidden.shape[-1] != self.config.language_dim:
            raise ValueError(
                f"expected language_dim={self.config.language_dim}, got {language_hidden.shape[-1]}"
            )
        if language_mask is None:
            language_mask = torch.ones(
                language_hidden.shape[:2], dtype=torch.bool, device=language_hidden.device
            )
        elif language_mask.shape != language_hidden.shape[:2]:
            raise ValueError("language_mask must match [batch, language_tokens]")
        caches = tuple(layer.project_language(language_hidden) for layer in self.layers)
        cache = LanguageCache(layers=caches, attention_mask=language_mask.bool())
        return cache.detach() if detach else cache

    def build_plan_cache(
        self,
        scene_summary: Tensor,
        language_hidden: Tensor,
        language_mask: Tensor | None = None,
        *,
        detach: bool = False,
    ) -> LanguageCache:
        """Plan-Cache 方案 B：场景条件化 plan tokens 接入语言缓存。

        Appends the PlanResampler output (8 plan tokens) to the language
        sequence and builds the VA language cache; the original language mask
        is kept (padding stays masked) and the plan tokens are always visible.
        """
        if not self.config.plan_resampler:
            raise ValueError("plan_resampler is disabled in the config")
        plan = self.plan_resampler(scene_summary, language_hidden, language_mask)
        plan = plan.to(dtype=language_hidden.dtype)
        extended = torch.cat((language_hidden, plan), dim=1)
        if language_mask is None:
            extended_mask = torch.ones(
                extended.shape[:2], dtype=torch.bool, device=extended.device
            )
        else:
            extended_mask = torch.cat(
                (
                    language_mask,
                    torch.ones(
                        plan.shape[:2], dtype=torch.bool, device=plan.device
                    ),
                ),
                dim=1,
            )
        return self.build_language_cache(extended, extended_mask, detach=detach)

    def encode_condition(
        self,
        vision_tokens: Tensor,
        proprio: Tensor,
        previous_action: Tensor,
        *,
        language_hidden: Tensor | None = None,
        language_mask: Tensor | None = None,
        language_cache: LanguageCache | None = None,
        visual_memory: VisualMemory | None = None,
        return_visual_memory: bool = False,
    ) -> Tensor | tuple[Tensor, VisualMemory]:
        if (language_hidden is None) == (language_cache is None):
            raise ValueError("provide exactly one of language_hidden or language_cache")
        if vision_tokens.ndim != 3:
            raise ValueError("vision_tokens must have shape [batch, tokens, vision_dim]")

        target_dtype = self.vision_projection.weight.dtype
        vision = self.vision_projection(vision_tokens.to(dtype=target_dtype))
        state = torch.cat((proprio, previous_action), dim=-1).to(dtype=target_dtype)
        state = self.state_projection(state)

        if language_cache is None:
            language_cache = self.build_language_cache(language_hidden, language_mask)

        action = self.action_queries[None].expand(vision.shape[0], -1, -1) + state[:, None]
        if self.config.action_query_cond:
            # Qwen-conditioned action queries：语言摘要（第 0 层投影 key 的 mask 加权均值）
            # 经 MLP 生成每 horizon 步偏移，zero-init 保证初始等价于静态 query。
            first_key = language_cache.layers[0].key  # [B, heads, tokens, head_dim]
            B = vision.shape[0]
            flat_key = first_key.transpose(1, 2).reshape(
                B, -1, self.config.hidden_dim
            )  # [B, tokens, hidden]
            mask = language_cache.attention_mask  # [B, tokens]
            denom = mask.float().sum(-1, keepdim=True).clamp_min(1.0)  # [B, 1]
            summary = (flat_key * mask[:, :, None]).sum(1) / denom  # [B, hidden]
            summary = summary.to(dtype=target_dtype)
            offset = self.lang_to_query(summary).view(
                B, self.config.action_horizon, self.config.hidden_dim
            )
            action = action + offset
        if len(language_cache.layers) != len(self.layers):
            raise ValueError("language cache does not match the number of VA layers")
        if language_cache.attention_mask.shape[0] != vision.shape[0]:
            raise ValueError("language cache batch size does not match vision batch size")
        for layer_cache in language_cache.layers:
            if layer_cache.key.shape != layer_cache.value.shape:
                raise ValueError("cached language key/value shapes do not match")
            if layer_cache.key.shape[:2] != (vision.shape[0], self.config.num_heads):
                raise ValueError("cached language batch size or head count is invalid")
            if layer_cache.key.shape[2] != language_cache.attention_mask.shape[1]:
                raise ValueError("cached language token count does not match its mask")
            if layer_cache.key.device != vision.device or layer_cache.key.dtype != vision.dtype:
                raise ValueError(
                    "language cache device/dtype must match the policy; call cache.to(device, dtype)"
                )

        if visual_memory is not None:
            if self.config.memory_split:
                if visual_memory.evidence is not None and (
                    visual_memory.evidence.ndim != 3
                    or visual_memory.evidence.shape[0] != vision.shape[0]
                    or visual_memory.evidence.shape[-1] != self.config.hidden_dim
                ):
                    raise ValueError("evidence memory shape is invalid")
                if visual_memory.task is not None and (
                    visual_memory.task.ndim != 3
                    or visual_memory.task.shape[0] != vision.shape[0]
                    or visual_memory.task.shape[-1] != self.config.hidden_dim
                ):
                    raise ValueError("task memory shape is invalid")
            else:
                if len(visual_memory.layers) != len(self.layers):
                    raise ValueError("visual memory does not match the number of VA layers")
                for memory in visual_memory.layers:
                    if memory.ndim != 3 or memory.shape[0] != vision.shape[0]:
                        raise ValueError("visual memory must have shape [batch, tokens, hidden_dim]")
                    if memory.shape[-1] != self.config.hidden_dim:
                        raise ValueError("visual memory hidden dimension is invalid")
                    if memory.device != vision.device or memory.dtype != vision.dtype:
                        raise ValueError(
                            "visual memory device/dtype must match the policy; call memory.to(device, dtype)"
                        )

        if self.config.memory_split:
            # ---- causal-decomposed memory: protected evidence + task workspace ----
            prev_evidence = None if visual_memory is None else visual_memory.evidence
            prev_task = None if visual_memory is None else visual_memory.task
            prev_spec = None if visual_memory is None else visual_memory.task_spec
            prev_future = None if visual_memory is None else visual_memory.pending_future
            # ---- EVSM: only evidence can commit (2026-08-07) ----
            # The previous decision's action proposal was written to scratch
            # (task_spec) together with a future-latent prediction.  Compare
            # that prediction against the vision we actually observe now:
            # agreement commits the proposal, disagreement rolls it back.
            # stop-grad is applied to the *entire* delta (predictor output AND
            # observed vision) so the gate never back-propagates into the
            # future predictor or (E2E) the vision backbone.
            if self.config.evsm and prev_spec is not None and prev_future is not None:
                observed = vision_tokens.mean(dim=1)  # frozen V-JEPA mean, [B, vision_dim]
                delta = (
                    1.0
                    - F.cosine_similarity(
                        F.normalize(prev_future.float(), dim=-1),
                        F.normalize(observed.float(), dim=-1),
                        dim=-1,
                    )
                ).detach()
                q = torch.sigmoid(
                    (self.config.evsm_kappa - delta) / self.config.evsm_temp
                )
                q = q.to(dtype=target_dtype).view(-1, 1, 1)
                if prev_task is None:
                    # Speculative proposal without a committed baseline is an
                    # abnormal state: fall back to the language-initialized
                    # workspace (fail-safe), never a blind commit.
                    commit = None
                else:
                    commit = q * prev_spec.to(dtype=target_dtype) + (1.0 - q) * prev_task
            else:
                commit = prev_task
            if prev_evidence is None:
                evidence = self.evidence_init.expand(vision.shape[0], -1, -1)
                evidence = self.evidence_gate(evidence, vision, state, overwrite=True)
            else:
                evidence = self.evidence_gate(prev_evidence, vision, state)
            if commit is None:
                first_key = language_cache.layers[0].key  # [B, heads, tokens, head_dim]
                flat_key = first_key.transpose(1, 2).reshape(
                    vision.shape[0], -1, self.config.hidden_dim
                )
                task = self.task_resampler(
                    flat_key.to(dtype=target_dtype),
                    language_cache.attention_mask,
                )
            else:
                task = commit
            task_hat = task
            for index, (layer, layer_cache) in enumerate(
                zip(self.layers, language_cache.layers, strict=True)
            ):
                if layer.sequential:
                    vision, action, task_hat = layer.forward_sequential(
                        vision,
                        action,
                        layer_cache,
                        language_cache.attention_mask,
                        evidence=evidence,
                        task=task_hat,
                        state=state[:, None],
                    )
                else:
                    vision, action, task_hat = layer(
                        vision,
                        action,
                        layer_cache,
                        language_cache.attention_mask,
                        evidence=evidence,
                        task=task_hat,
                        state=state[:, None],
                    )
            # The VA layers propose a speculative task update; with EVSM it
            # goes to scratch (task_spec) and is only committed after evidence
            # verification at the next decision.  Without EVSM it commits
            # immediately (original behavior).
            spec = self.task_gate(task, task_hat)
            task_out = task if self.config.evsm else spec
            action = action + self.task_to_action(spec.mean(dim=1, keepdim=True))
            action_condition = self.action_norm(action)
            if return_visual_memory:
                if self.config.evsm and self.config.future_predict:
                    pending_future = self.future_predictor(action_condition, evidence, spec)
                else:
                    pending_future = None
                return action_condition, VisualMemory(
                    layers=(),
                    evidence=evidence,
                    task=task_out,
                    task_spec=spec if self.config.evsm else None,
                    pending_future=pending_future,
                    gate=(
                        float(q.mean().item())
                        if self.config.evsm and prev_spec is not None and prev_future is not None
                        else None
                    ),
                )
            return action_condition

        next_memory = []
        for index, (layer, layer_cache) in enumerate(
            zip(self.layers, language_cache.layers, strict=True)
        ):
            previous_visual = None if visual_memory is None else visual_memory.layers[index]
            if layer.sequential:
                vision, action, _ = layer.forward_sequential(
                    vision,
                    action,
                    layer_cache,
                    language_cache.attention_mask,
                    visual_memory=previous_visual,
                )
            else:
                vision, action, _ = layer(
                    vision,
                    action,
                    layer_cache,
                    language_cache.attention_mask,
                    visual_memory=previous_visual,
                )
            next_memory.append(vision)
        action_condition = self.action_norm(action)
        if return_visual_memory:
            return action_condition, VisualMemory(layers=tuple(next_memory))
        return action_condition

    def flow_velocity(
        self,
        action_condition: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
    ) -> Tensor:
        expected = (
            action_condition.shape[0],
            self.config.action_horizon,
            self.config.hidden_dim,
        )
        if action_condition.shape != expected:
            raise ValueError(
                "action_condition must have shape "
                f"[batch, {self.config.action_horizon}, {self.config.hidden_dim}]"
            )
        if noisy_actions.shape != expected[:2] + (self.config.action_dim,):
            raise ValueError(
                "noisy_actions must have shape "
                f"[batch, {self.config.action_horizon}, {self.config.action_dim}]"
            )
        return self.flow_head(action_condition, noisy_actions, flow_time)

    def forward(
        self,
        vision_tokens: Tensor,
        proprio: Tensor,
        previous_action: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
        *,
        language_hidden: Tensor | None = None,
        language_mask: Tensor | None = None,
        language_cache: LanguageCache | None = None,
        visual_memory: VisualMemory | None = None,
        return_visual_memory: bool = False,
    ) -> Tensor | tuple[Tensor, VisualMemory]:
        encoded = self.encode_condition(
            vision_tokens,
            proprio,
            previous_action,
            language_hidden=language_hidden,
            language_mask=language_mask,
            language_cache=language_cache,
            visual_memory=visual_memory,
            return_visual_memory=return_visual_memory,
        )
        if return_visual_memory:
            action_condition, next_memory = encoded
            velocity = self.flow_velocity(action_condition, noisy_actions, flow_time)
            return velocity, next_memory
        return self.flow_velocity(encoded, noisy_actions, flow_time)

    @torch.no_grad()
    def sample_actions(
        self,
        action_condition: Tensor,
        *,
        steps: int = 8,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Integrate noise at tau=0 to an action chunk at tau=1 with Euler steps."""
        if steps < 1:
            raise ValueError("flow sampling steps must be positive")
        expected = (
            action_condition.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )
        if noise is None:
            actions = torch.randn(
                expected,
                device=action_condition.device,
                dtype=action_condition.dtype,
            )
        elif noise.shape != expected:
            raise ValueError(f"noise must have shape {expected}")
        else:
            actions = noise.to(device=action_condition.device, dtype=action_condition.dtype)

        step_size = 1.0 / steps
        for index in range(steps):
            flow_time = torch.full(
                (action_condition.shape[0],),
                index * step_size,
                device=action_condition.device,
                dtype=action_condition.dtype,
            )
            actions = actions + step_size * self.flow_velocity(
                action_condition,
                actions,
                flow_time,
            )
        return actions

    def decode_actions(
        self,
        action_condition: Tensor,
        *,
        steps: int = 8,
        noise: Tensor | None = None,
        c_current: Tensor | None = None,
    ) -> Tensor:
        """从 action_condition 解码归一化动作 chunk（C²-VA 统一入口）。

        ``c2_controller=True``：C² 收缩解码——ū = DirectActionHead，
        c̄ = sg(c_anchor) + Δc，K = gain_head；a = clip(ū − K·(c_current − c̄))。
        ``c_current``（[B, c2_control_dim]，P 投影后的当前视觉状态）必须提供，
        否则报错；``steps``/``noise`` 忽略。``direct_head=True``（无 c2）：
        确定性 DirectActionHead，``steps``/``noise`` 忽略。否则走现有 flow
        Euler 采样（等价于 ``sample_actions``，现有路径不变）。
        """
        if self.config.c2_controller:
            if c_current is None:
                raise ValueError(
                    "c2_controller requires c_current: project the current vision "
                    "tokens with control_projector first"
                )
            return self.controller_params(action_condition, c_current).apply_controller(
                c_current
            )
        if self.config.direct_head:
            return self.direct_head(action_condition)
        return self.sample_actions(action_condition, steps=steps, noise=noise)

    def controller_params(
        self,
        action_condition: Tensor,
        c_current: Tensor,
    ) -> ControllerParams:
        """生成 C² 控制器参数 {ū, c̄, K}

        ū = DirectActionHead；c̄ = sg(c_anchor) + Δc（Δc_0 ≡ 0，anchor 取
        当前投影 → token 0 参考 ≡ 当前状态，e_0 = 0）；K = gain_head 输出。
        """
        expected_dim = self.config.c2_control_dim
        if c_current.ndim != 2 or c_current.shape[1] != expected_dim:
            raise ValueError(
                f"c_current must have shape [batch, {expected_dim}], "
                f"got {tuple(c_current.shape)}"
            )
        if c_current.shape[0] != action_condition.shape[0]:
            raise ValueError(
                "c_current batch size must match action_condition "
                f"({c_current.shape[0]} vs {action_condition.shape[0]})"
            )
        nominal = self.direct_head(action_condition)
        delta, gain = self.c2_head(action_condition)
        anchor = c_current.detach()  # sg(c_anchor)：Δc 学相对当前观测的增量
        reference = anchor[:, None, :] + delta
        return ControllerParams(nominal=nominal, reference=reference, gain=gain)

    @staticmethod
    def flow_matching_loss(predicted_velocity: Tensor, target_velocity: Tensor) -> Tensor:
        return F.mse_loss(predicted_velocity, target_velocity)

    def sample_flow_trajectory(
        self,
        action_condition: Tensor,
        *,
        steps: int = 8,
        noise: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> list[Tensor]:
        """Stochastic Euler trajectory; returns path [x_0, ..., x_K].

        Each transition is x_{k+1} = x_k + (1/K) v(x_k, t_k) + sigma_k * eps_k,
        matching the ReinFlow-lite augmented Markov policy.  sigma has shape
        [K, action_dim] (per-step per-dim transition noise); None gives the
        deterministic Euler path (identical to sample_actions).
        """
        if steps < 1:
            raise ValueError("flow sampling steps must be positive")
        expected = (
            action_condition.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )
        if noise is None:
            x = torch.randn(
                expected, device=action_condition.device, dtype=action_condition.dtype
            )
        elif noise.shape != expected:
            raise ValueError(f"noise must have shape {expected}")
        else:
            x = noise.to(device=action_condition.device, dtype=action_condition.dtype)
        if sigma is not None and sigma.shape != (steps, self.config.action_dim):
            raise ValueError(f"sigma must have shape [steps, action_dim]={steps, self.config.action_dim}")

        step_size = 1.0 / steps
        path = [x]
        for index in range(steps):
            flow_time = torch.full(
                (action_condition.shape[0],),
                index * step_size,
                device=action_condition.device,
                dtype=action_condition.dtype,
            )
            mu = x + step_size * self.flow_velocity(action_condition, x, flow_time)
            if sigma is not None:
                mu = mu + sigma[index] * torch.randn_like(mu)
            x = mu
            path.append(x)
        return path

    def flow_trajectory_log_prob(
        self,
        path: list[Tensor],
        action_condition: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """Log-prob of the sampled denoising path under the flow-noisy
        transition model: sum_k log N(x_{k+1}; x_k + (1/K) v(x_k, t_k),
        sigma_k^2 I).  Returns [batch] log-probs, differentiable in theta.
        """
        steps = len(path) - 1
        if sigma.shape != (steps, self.config.action_dim):
            raise ValueError(f"sigma must have shape [steps, action_dim]={steps, self.config.action_dim}")
        step_size = 1.0 / steps
        device = action_condition.device
        dtype = action_condition.dtype
        log_prob = torch.zeros(
            (action_condition.shape[0],), device=device, dtype=dtype
        )
        log_2pi = math.log(2.0 * math.pi)
        horizon = self.config.action_horizon
        for index in range(steps):
            flow_time = torch.full(
                (action_condition.shape[0],),
                index * step_size,
                device=device,
                dtype=dtype,
            )
            mu = path[index] + step_size * self.flow_velocity(
                action_condition, path[index], flow_time
            )
            sigma_k = sigma[index]  # [action_dim]
            diff = path[index + 1] - mu  # [B, H, A]
            log_var = 2.0 * sigma_k.log()
            # per-step normalizer over [H, A] entries; without the H factors
            # the constant terms would not cancel once sigma is trainable
            log_prob = log_prob - 0.5 * (
                log_2pi * horizon * self.config.action_dim
                + log_var.sum() * horizon
                + (diff.pow(2) / sigma_k.pow(2)).sum(dim=(1, 2))
            )
        return log_prob
