from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from va_compound.local_control_slots import fourier_encode


AttentionMode = Literal["bidir_va", "uni_a"]

# MT-VJ 公共常量（artifacts/mt_vj_contract.md §公共常量，2026-08-10）：
# dense evidence 投影维（768 → 192，带宽降 4 倍，0.42MiB/decision）。
D_PROJ = 192
# 坐标正弦嵌入维：fourier_encode(coords, num_bands=4) = 3 + 2*3*4 = 27。
_COORD_DIM = 27


def dense_coords(
    n_tokens: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """[N, 3] 归一化 (t, y, x) 网格坐标（N = 2*grid²，t→y→x 序）。

    与 live_vjepa._slot_coords / tests/test_dense_readout.build_dense_coords_smoke
    同一生成公式；MT-VJ dense readout 用它构造坐标正弦嵌入（K_dense 加项）。
    """
    if n_tokens % 2:
        raise ValueError(f"dense token 数必须为 2*grid²（2 时间片），got {n_tokens}")
    grid = math.isqrt(n_tokens // 2)
    if grid * grid != n_tokens // 2:
        raise ValueError(f"dense token 数必须为 2*grid²，got {n_tokens}")
    half = (grid - 1) / 2
    rows = [
        (t * 2.0 - 1.0, (y - half) / half, (x - half) / half)
        for t in range(2)
        for y in range(grid)
        for x in range(grid)
    ]
    return torch.tensor(rows, dtype=dtype or torch.float32, device=device)


@dataclass(frozen=True)
class VACompoundConfig:
    language_dim: int = 2048
    vision_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    action_horizon: int = 8
    # Number of executable actions consumed before the next policy query.
    planning_stride: int = 6
    # Intended closed-loop execution chunk. 0 keeps the legacy behavior of
    # executing ``planning_stride`` actions. Training windows may still be
    # sampled every planning_stride frames while an H15 policy deploys P15.
    deployment_execution_horizon: int = 0
    action_dim: int = 7
    proprio_dim: int = 14
    world_state_supervision: bool = False
    world_state_loss_weight: float = 1.0
    flow_layers: int = 2
    # Optional smaller action-decoder width. 0 keeps the historical shared width.
    flow_hidden_dim: int = 0
    dropout: float = 0.0
    mode: AttentionMode = "bidir_va"
    # Su Shen-inspired upgrade (default off, backward compatible).
    qk_norm: bool = False  # per-head RMSNorm on Q/K before the dot product
    # 源测度校正注意力（SMC-Attn，2026-08-05 原创设计）：softmax 前对每个来源减去
    # log N_s，使比较的是"平均证据"而非"token 数量 × 平均证据"——同一来源被复制/细分
    # 不再获得额外票数。flat = 原始共享 softmax（向后兼容）。
    attention_variant: str = "flat"  # "flat" | "smc"
    # Memory-efficient fused SDPA for the shared VA attention.  ``auto`` uses
    # SDPA only on the mathematically compatible flat/non-dual/no-debug path;
    # ``manual`` preserves the legacy explicit FP32 score tensor.  The option
    # is serialized because it changes floating-point reduction order even
    # though the attention equation is unchanged.
    va_attention_backend: str = "manual"  # "manual" | "auto"
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
    # WAM4VA：独立世界状态通过下一层 attention K/V 与 VA 交换信息。
    wmrm: bool = False
    # Give World the same complete Qwen token sequence that VA reads.  Legacy
    # checkpoints keep the old four-query language summary unless explicitly
    # enabled, so the new parameter set is an intentional architecture change.
    wmrm_full_language_tokens: bool = False
    # Feed each World stage the current VA layer's language-grounded visual
    # output. False preserves the legacy peer contract's pre-VA snapshot.
    wmrm_reads_fused_va_tokens: bool = False
    wmrm_world_dim: int = 8
    # all = 每层更新 World state；last = 仅末层；even = 奇数层+末层。
    # peer_sync_h6 still requires this flag to be "all", but World is one
    # layer shorter: propose on VA layers 0..N-2, last VA layer consumes only.
    wmrm_inject: str = "all"
    # dino: 预测下一决策 DINO/投影 feature；vjepa: 下一决策 H11 池化；metric: 旧几何。
    wmrm_target: str = "dino"
    wmrm_cycle_steps: int = 6  # action-conditioned World target horizon
    # Training proposal only: stop gradients between successive WMRM stage maps while
    # preserving their forward values. Logged/evaluator forwards choose explicitly.
    wmrm_detach_proposal_stage_state: bool = False
    wmrm_map_size: int = 16
    wmrm_map_channels: int = 32
    wmrm_world_grid: int = 16  # attention memory keeps native DINO grid; do not pool 16→4
    wmrm_predictor: str = "legacy"
    wmrm_predictor_depth: int = 6
    wmrm_predictor_width: int = 384
    wmrm_predictor_heads: int = 12
    wmrm_predictor_copies: int = 1
    # Optional warm-start gate for newly appended peer World stages. Stages
    # before this index keep their checkpoint behavior; later proposals start
    # as residual no-ops and learn to open during the depth experiment.
    wmrm_stage_gate_start: int | None = None
    # Fail-fast tensor-content checks are valuable in tests/debug runs, but each
    # CUDA truth-value check synchronizes the host inside the recurrent hot path.
    runtime_integrity_checks: bool = True
    # VA↔World execution topology. ``peer_sync_h6`` uses one pre-stage
    # snapshot for both state transitions and a deterministic action-chunk readout.
    # World has N-1 stages so the last VA layer reads the final DINO map.
    va_world_mode: str = "legacy"  # "legacy" | "peer_sync_h6"
    # 顺序式 A→V→A 耦合（2026-08-07 审阅落地④）：每 N 层使用
    # proposal→reorganize→correction 三遍注意力；0 = 全层同步联合（旧行为）。
    sequential_coupling: int = 0
    # flow 头深度条件（2026-08-07 审阅落地④）：每层 AdaLN-Zero + cross-attn
    # 注入条件（"entry" 为旧行为：仅入口相加）。
    flow_cond: str = "entry"
    # Evo-1-style VA→Flow readout: three parallel cross-attention blocks read
    # the final three VA action streams once per decision. Their zero-initialized
    # residual gates keep the initial condition exactly equal to the last VA layer.
    va_last3_cross_attn: bool = False
    # Zero-gated bidirectional DINO↔Qwen bridge. The default four pairs keep
    # old checkpoints loadable; new runs may fuse more tail layers one-to-one.
    dino_qwen_cross_modal_bridge: bool = False
    dino_qwen_cross_modal_layers: int = 4
    # Capacity warm-up only: let the frozen H9 Flow teacher pass its loss
    # through the action condition into newly appended VA layers.
    tail_flow_condition_grad: bool = False
    # Capacity phase 2 only: let policy loss train the new World stage gates
    # without sending policy gradients into World-state predictions.
    capacity_stage_gate_policy_grad: bool = False
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
    # 第二轮完整版架构重构（2026-08-08，默认全关，现有行为逐字节不变）：
    # learned role queries 替代语言 mask-weighted mean 摘要（TaskResampler 与
    # action_query_cond 共享同一 RoleQueryResampler 实例）。
    role_query: bool = False
    role_query_tokens: int = 16
    # 双注意力（非 sequential 层）：动作 query 拆 physical（无语言列）与
    # semantic（仅语言列）两条注意力，融合门 g_A 初始 < 0.2。
    dual_attention: bool = False
    # flow head 逐层读语义上下文（compile readout tokens；flow_cond=adaln
    # 时经 cross-attn 注入）。
    flow_semantic: bool = False
    # PULSE-VA（2026-08-08 Pro 审阅落地）：语言编程局部控制槽。Stage A 用
    # dense spatiotemporal tokens [B,N,768]（2×12×12=288）+ 归一化坐标旁路 →
    # 6 固定角色槽（语言实例化）→ 3 关系 token → 25-token VA 视觉流。
    # 训练损失不变（仅 action loss）；C² 控制图在 Stage B 另行拟合。
    local_slots: bool = False
    local_slot_k: int = 6
    local_slot_tokens: int = 288  # 每决策 dense token 数（2×12×12；dense_readout 时为 1152）
    local_coarse: int = 16        # 粗上下文 token 数
    # 消融格：direct288 = 288 token 直送 VA（无槽）；fixed_query = 槽只用固定
    # 角色种子（无语言交叉注意）——与 language-slot 对照验证"语言定义"增益。
    local_slots_direct288: bool = False
    local_slots_fixed_query: bool = False
    # Step 0（2026-08-09，C²-IRF v2 设计 §七）：dense readout——跳过 V-JEPA 池化，
    # 角色查询直接读出 1152 个 patch token（2×24×24），coarse 仍从 1152 avg-pool
    # 到 16。1152 只作槽 cross-attention 的 K/V（Q=6），不进 VA 自注意力（§九）。
    dense_readout: bool = False
    # Step 1（2026-08-09，C²-IRF v2 设计 §七 Step 1）：多模式读出——每角色
    # heatmap（2 时间片 × grid²）局部 NMS 取 top-2 峰 + 5×5 局部 soft-argmax
    # （跨 patch 亚像素 μ/Σ/z）+ learned NULL 键值（遮挡时查询选 NULL，
    # vis=1−P(∅)）+ 寻址偏置 b_coord / b_track（γ=0.01）。视觉流变为
    # 16 coarse + 12 modes + 3 relations = 31 tokens；vis 经零初始化投影注入。
    # 与 --dense-readout 兼容（1152 网格）；288 网格同样可用（消融）。
    multi_mode: bool = False
    # MT-VJ Stage A dense action readout（2026-08-10 契约 §5）：每层
    # VACouplingLayer 注入 dense K/V cross-attention（1152 patch 只做 K/V，
    # query 仅 action tokens）+ metric relation tokens；W_o 严格零初始化 →
    # 初始输出与无 dense 路径逐位一致。与 local_slots/dense_readout 正交：
    # 直接消费 forward_hierarchical_dense 的 {5, 11} 证据，不经角色读出。
    dense_readout_mtvj: bool = False
    # OpenVLA-style additive action vision：冻结 DINOv2 ViT-L/14-reg4 @224
    # 提供独立 dense K/V，保留既有 V-JEPA base + MT-VJ 路径。每层独立
    # action_dense_out 严格 zero-init，因此旧 E7 checkpoint 非严格恢复后
    # 初始策略逐位不变。字符串同时作为 checkpoint 内的视觉塔契约。
    action_vision_backbone: str = "none"  # e.g. "dinov2_vitl14_reg4"
    action_vision_model_id: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    action_vision_dim: int = 1024
    action_vision_image_size: int = 224
    action_vision_layers: tuple[int, int] = (11, 23)
    # DINO-main replacement（2026-08-14 用户决策）：冻结 DINOv2 REPLACES V-JEPA
    # 作为 VA 主视觉骨干，VA/FM/投影从零可训练。V-JEPA/dense/metric 代码保留
    # 在仓库中（flag 关闭即禁用，不删除）。默认 "vjepa" 使全部旧路径逐位不变。
    main_vision_backbone: str = "vjepa"  # "vjepa" | "dinov2_vitl14_reg4"
    main_vision_model_id: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    main_vision_image_size: int = 224
    main_vision_dim: int = 1024
    main_vision_grid: int = 8    # 每帧 16x16 patch 网格池化到 grid x grid
    main_vision_frames: int = 4  # 每决策消费的窗口帧数 [d-6,d-4,d-2,d]
    main_vision_tokens: int = 256  # = grid*grid*frames
    # DINO 四帧显式时序：raw 1024-D patch token 进入 vision_projection 前，
    # 按 frame-major 布局加 learned slot embedding。默认关闭使旧 checkpoint/路径
    # 逐位不变；新 task35 run 开启，打破原来的四帧集合置换不变性。
    main_vision_temporal: bool = False
    main_vision_temporal_scale: float = 1.0
    # DINO-metric（2026-08-15 用户决策）：DINO-main 下接回 MT-VJ dense + metric
    # 全栈。dense evidence = DINO block11(g)/block23(d) 两帧 [d-2,d] patch
    # （2×16×16=512 token，1024 维）+ Δt；LanguageMetricField 以 h_dim=1024、
    # grid=16 从零训练。复用 dense_readout_mtvj 的逐层 dense K/V 机制。
    dino_dense_metric: bool = False
    # Formal slot-free policy contract.  Legacy metric/local-slot experiments
    # remain loadable when this is false, but a slot-free run cannot silently
    # route simulator-derived geometry into VA/World/action.
    slot_free_policy: bool = False
    # 8-D ``p * visibility`` geometry 直接进入 state/action query 条件，绕过
    # 512 dense + 2 metric token 的共享 softmax 稀释。Linear 严格 zero-init，
    # 从旧架构迁移时初始行为等价；默认关闭保持旧 state_dict 键集合不变。
    metric_geometry_inject: bool = False
    metric_geometry_dim: int = 8
    # 架构版本：'legacy' 保持既有行为与 state_dict；'dual_tower_expert_v1' 启用双塔融合与逐层动作专家。
    architecture_version: str = "legacy"
    fusion_pair_count: int = 6

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.flow_layers < 1:
            raise ValueError("flow_layers must be positive")
        flow_hidden_dim = self.flow_hidden_dim or self.hidden_dim
        if flow_hidden_dim < 1 or flow_hidden_dim % self.num_heads:
            raise ValueError("flow_hidden_dim must be positive and divisible by num_heads")
        if self.mode not in ("bidir_va", "uni_a"):
            raise ValueError(f"unsupported attention mode: {self.mode}")
        if self.attention_variant not in ("flat", "smc"):
            raise ValueError(f"unsupported attention variant: {self.attention_variant}")
        if self.va_attention_backend not in ("manual", "auto"):
            raise ValueError(
                f"unsupported VA attention backend: {self.va_attention_backend}"
            )
        if self.sequential_coupling < 0:
            raise ValueError("sequential_coupling must be >= 0")
        if self.sequential_coupling and self.mode == "uni_a":
            raise ValueError("sequential_coupling does not support uni_a role masking")
        if self.sequential_coupling and self.va_world_mode == "peer_sync_h6":
            raise ValueError(
                "sequential_coupling does not support protected action prefixes"
            )
        if self.flow_cond not in ("entry", "adaln"):
            raise ValueError(f"unsupported flow conditioning: {self.flow_cond}")
        if self.va_last3_cross_attn and self.num_layers < 3:
            raise ValueError("va_last3_cross_attn requires num_layers >= 3")
        if self.va_last3_cross_attn and self.direct_head:
            raise ValueError("va_last3_cross_attn requires the Flow action head")
        if self.dino_qwen_cross_modal_bridge and self.dino_dense_metric:
            raise ValueError(
                "dino_qwen_cross_modal_bridge is incompatible with dino_dense_metric"
            )
        if self.dino_qwen_cross_modal_layers < 1:
            raise ValueError("dino_qwen_cross_modal_layers must be positive")
        if (
            self.dino_qwen_cross_modal_bridge
            and self.num_layers < self.dino_qwen_cross_modal_layers
        ):
            raise ValueError(
                "dino_qwen_cross_modal_bridge requires enough VA layers"
            )
        if self.c2_controller and not self.direct_head:
            raise ValueError("c2_controller requires direct_head")
        if self.c2_control_dim < 1:
            raise ValueError("c2_control_dim must be positive")
        if self.role_query_tokens < 1:
            raise ValueError("role_query_tokens must be positive")
        if self.dense_readout and not self.local_slots:
            raise ValueError("dense_readout requires local_slots (角色查询读出路径)")
        if self.dense_readout and self.local_slots_direct288:
            raise ValueError(
                "dense_readout 与 local_slots_direct288 互斥：1152 token 直送 VA "
                "会在 VA 内做 1152×1152 自注意力（设计文档 §九 明确禁止）"
            )
        if self.dense_readout and self.local_slot_tokens != 1152:
            raise ValueError(
                f"dense_readout 需要 local_slot_tokens=1152（2×24×24 patch），"
                f"got {self.local_slot_tokens}"
            )
        if self.multi_mode and not self.local_slots:
            raise ValueError("multi_mode requires local_slots (角色查询读出路径)")
        if self.multi_mode and self.local_slots_direct288:
            raise ValueError(
                "multi_mode 与 local_slots_direct288 互斥：direct288 无角色读出路径"
            )
        if not isinstance(self.action_vision_backbone, str) or not self.action_vision_backbone:
            raise ValueError("action_vision_backbone must be a non-empty string")
        if not isinstance(self.action_vision_model_id, str) or not self.action_vision_model_id:
            raise ValueError("action_vision_model_id must be a non-empty string")
        if self.action_vision_dim < 1:
            raise ValueError("action_vision_dim must be positive")
        if self.action_vision_image_size < 1:
            raise ValueError("action_vision_image_size must be positive")
        if (
            len(self.action_vision_layers) != 2
            or self.action_vision_layers[0] < 0
            or self.action_vision_layers[0] >= self.action_vision_layers[1]
        ):
            raise ValueError("action_vision_layers must be two increasing block indices")
        if self.main_vision_backbone != "vjepa":
            if not self.main_vision_model_id:
                raise ValueError("main_vision_model_id must be non-empty")
            if self.main_vision_image_size < 1 or self.main_vision_dim < 1:
                raise ValueError("main vision tower spec must be complete")
            if not (1 <= self.main_vision_grid <= 16):
                raise ValueError("main_vision_grid must be in [1, 16]")
            if self.main_vision_frames < 1:
                raise ValueError("main_vision_frames must be positive")
            expected_tokens = (
                self.main_vision_grid * self.main_vision_grid * self.main_vision_frames
            )
            if self.main_vision_tokens != expected_tokens:
                raise ValueError(
                    "main_vision_tokens must equal grid*grid*frames, got "
                    f"{self.main_vision_tokens} vs {expected_tokens}"
                )
        if self.main_vision_temporal:
            if self.main_vision_backbone == "vjepa":
                raise ValueError(
                    "main_vision_temporal requires a non-V-JEPA main vision backbone"
                )
            if not math.isfinite(self.main_vision_temporal_scale):
                raise ValueError("main_vision_temporal_scale must be finite")
        if self.metric_geometry_dim < 1:
            raise ValueError("metric_geometry_dim must be positive")
        if self.metric_geometry_inject and not (
            self.dino_dense_metric or self.dense_readout_mtvj
        ):
            raise ValueError(
                "metric_geometry_inject requires dino_dense_metric or dense_readout_mtvj"
            )
        if self.dino_dense_metric:
            if self.main_vision_backbone == "vjepa":
                raise ValueError(
                    "dino_dense_metric requires main_vision_backbone != 'vjepa'"
                )
            if not self.dense_readout_mtvj:
                raise ValueError(
                    "dino_dense_metric requires dense_readout_mtvj "
                    "（逐层 dense K/V 机制复用）"
                )
        if self.wmrm_full_language_tokens and not self.wmrm:
            raise ValueError("wmrm_full_language_tokens requires wmrm=true")
        if self.wmrm_reads_fused_va_tokens and not self.wmrm:
            raise ValueError("wmrm_reads_fused_va_tokens requires wmrm=true")
        if self.slot_free_policy and (
            self.metric_geometry_inject
            or self.dino_dense_metric
            or self.local_slots
        ):
            raise ValueError(
                "slot_free_policy forbids metric geometry, metric relation tokens, "
                "and local fixed-slot readers"
            )
        if self.wmrm and self.memory_split:
            raise ValueError("wmrm is mutually exclusive with memory_split (P2 last-layer hook)")
        if self.wmrm and (self.direct_head or self.c2_controller):
            raise ValueError("wmrm is mutually exclusive with direct_head/c2_controller")
        if self.wmrm_world_dim < 1:
            raise ValueError("wmrm_world_dim must be positive")
        if self.wmrm_inject not in ("last", "all", "even"):
            raise ValueError("wmrm_inject must be last|all|even")
        if self.wmrm_target not in ("dino", "vjepa", "metric"):
            raise ValueError("wmrm_target must be dino|vjepa|metric")
        if self.wmrm_cycle_steps < 1:
            raise ValueError("wmrm_cycle_steps must be positive")
        if self.planning_stride < 1:
            raise ValueError("planning_stride must be positive")
        deployment_horizon = (
            self.deployment_execution_horizon or self.planning_stride
        )
        if self.wmrm_target == "vjepa" and self.main_vision_backbone != "vjepa":
            raise ValueError(
                "vjepa world target requires V-JEPA main vision "
                "(do not infer backbone from dense key 11)"
            )
        if self.wmrm_map_size < 1:
            raise ValueError("wmrm_map_size must be positive")
        if self.wmrm_map_channels < 1:
            raise ValueError("wmrm_map_channels must be positive")
        if self.wmrm_world_grid < 1:
            raise ValueError("wmrm_world_grid must be positive")
        if self.wmrm_predictor not in {"legacy", "st_blocks"}:
            raise ValueError("wmrm_predictor must be legacy|st_blocks")
        if self.wmrm_predictor_depth < 1:
            raise ValueError("wmrm_predictor_depth must be positive")
        if self.wmrm_predictor_width < 1:
            raise ValueError("wmrm_predictor_width must be positive")
        if self.wmrm_predictor_heads < 1:
            raise ValueError("wmrm_predictor_heads must be positive")
        if self.wmrm_predictor_copies < 1:
            raise ValueError("wmrm_predictor_copies must be positive")
        if self.wmrm_predictor != "st_blocks" and self.wmrm_predictor_copies != 1:
            raise ValueError("wmrm_predictor_copies requires st_blocks")
        if self.wmrm_predictor_width % self.wmrm_predictor_heads != 0:
            raise ValueError("wmrm_predictor_width must be divisible by wmrm_predictor_heads")
        if self.va_world_mode not in {"legacy", "peer_sync_h6"}:
            raise ValueError("va_world_mode must be legacy|peer_sync_h6")
        if self.wmrm_stage_gate_start is not None:
            if self.va_world_mode != "peer_sync_h6":
                raise ValueError("wmrm_stage_gate_start requires peer_sync_h6")
            if not 0 < self.wmrm_stage_gate_start < self.num_layers - 1:
                raise ValueError(
                    "wmrm_stage_gate_start must be inside the peer World stage range"
                )
        if self.va_world_mode == "peer_sync_h6":
            if not self.wmrm:
                raise ValueError("peer_sync_h6 requires wmrm=true")
            if self.action_horizon not in {6, 15, 50}:
                raise ValueError("peer mode requires action_horizon in {6, 15, 50}")
            if self.planning_stride not in {1, 2, 3, 4, 6, 15}:
                raise ValueError(
                    "peer_sync_h6 requires planning_stride in {1, 2, 3, 4, 6, 15}"
                )
            if not 1 <= deployment_horizon <= self.action_horizon:
                raise ValueError(
                    "planning_stride/deployment_execution_horizon must be within "
                    "the action horizon"
                )
            if deployment_horizon not in {
                self.planning_stride,
                self.action_horizon,
            }:
                raise ValueError(
                    "peer deployment must execute either the training stride "
                    "or the complete action horizon"
                )
            if (
                deployment_horizon == self.action_horizon
                and self.wmrm_cycle_steps != self.action_horizon
            ):
                raise ValueError(
                    "full-chunk peer deployment requires World to predict the "
                    "same endpoint horizon"
                )
            if self.wmrm_cycle_steps not in {
                self.planning_stride,
                self.action_horizon,
            }:
                raise ValueError(
                    "peer mode requires wmrm_cycle_steps to equal either the "
                    "execution prefix or full action horizon"
                )
            if self.action_dim < 1:
                raise ValueError("peer_sync_h6 requires a positive action_dim")
            if self.num_layers < 2:
                raise ValueError(
                    "peer_sync_h6 requires num_layers >= 2 so World can be one "
                    "layer shorter than VA"
                )
            if self.wmrm_inject != "all":
                raise ValueError("peer_sync_h6 requires wmrm_inject=all")
        if self.architecture_version not in ("legacy", "dual_tower_expert_v1"):
            raise ValueError(
                f"unsupported architecture_version: {self.architecture_version}"
            )
        if self.architecture_version == "dual_tower_expert_v1":
            if self.num_layers < 3:
                raise ValueError(
                    "dual_tower_expert_v1 requires num_layers >= 3"
                )
            if self.direct_head or self.c2_controller:
                raise ValueError(
                    "dual_tower_expert_v1 rejects direct_head and c2_controller"
                )
            if self.dino_qwen_cross_modal_bridge:
                raise ValueError(
                    "dual_tower_expert_v1 rejects dino_qwen_cross_modal_bridge"
                )
            if self.va_last3_cross_attn:
                raise ValueError(
                    "dual_tower_expert_v1 rejects va_last3_cross_attn"
                )
            if self.local_slots:
                raise ValueError(
                    "dual_tower_expert_v1 rejects local_slots"
                )
            if self.memory_split:
                raise ValueError(
                    "dual_tower_expert_v1 rejects memory_split"
                )
            if self.flow_semantic or self.future_predict:
                raise ValueError("dual_tower_expert_v1 rejects legacy semantic and future routes")
            if self.fusion_pair_count < 1:
                raise ValueError(
                    "fusion_pair_count must be positive"
                )

    def wmrm_stage_count(self) -> int:
        """World ``propose()`` steps per encode. Peer is one shorter than VA."""
        if not self.wmrm:
            return 0
        if self.va_world_mode == "peer_sync_h6":
            return self.num_layers - 1
        return self.num_layers


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
    role_queries: Tensor | None = None  # [B, K, hidden] PULSE-VA 角色查询（一次性缓存）
    # The original full Qwen sequence.  VA consumes its per-layer projections
    # above; full-language World consumes this same read-only sequence directly.
    tokens: Tensor | None = None

    def detach(self) -> "LanguageCache":
        return LanguageCache(
            layers=tuple(layer.detach() for layer in self.layers),
            attention_mask=self.attention_mask.detach(),
            role_queries=(
                self.role_queries.detach() if self.role_queries is not None else None
            ),
            tokens=self.tokens.detach() if self.tokens is not None else None,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LanguageCache":
        return LanguageCache(
            layers=tuple(layer.to(device=device, dtype=dtype) for layer in self.layers),
            attention_mask=self.attention_mask.to(device=device, dtype=torch.bool),
            role_queries=(
                self.role_queries.to(device=device, dtype=dtype)
                if self.role_queries is not None
                else None
            ),
            tokens=(
                self.tokens.to(device=device, dtype=dtype)
                if self.tokens is not None
                else None
            ),
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
    world_state: object | None = None  # WAMState; object avoids importing optional WMRM here

    def __init__(
        self,
        layers: tuple[Tensor, ...],
        evidence: Tensor | None = None,
        task: Tensor | None = None,
        task_spec: Tensor | None = None,
        pending_future: Tensor | None = None,
        gate: float | None = None,
        world_state: object | None = None,
    ) -> None:
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "task_spec", task_spec)
        object.__setattr__(self, "pending_future", pending_future)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "world_state", world_state)

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
            world_state=(
                self.world_state.detach() if self.world_state is not None else None
            ),
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
            world_state=(
                self.world_state.to(device=device, dtype=dtype)
                if self.world_state is not None
                else None
            ),
        )

    def index_select(self, index: Tensor) -> "VisualMemory":
        if index.ndim != 1:
            raise ValueError(f"index must be one-dimensional, got {tuple(index.shape)}")

        def select(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None
            return tensor.index_select(
                0, index.to(device=tensor.device, dtype=torch.long)
            )

        return VisualMemory(
            layers=tuple(select(layer) for layer in self.layers),
            evidence=select(self.evidence),
            task=select(self.task),
            task_spec=select(self.task_spec),
            pending_future=select(self.pending_future),
            gate=self.gate,
            world_state=(
                self.world_state.index_select(index)
                if self.world_state is not None
                else None
            ),
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


class RoleQueryResampler(nn.Module):
    """Learned role queries over the language flat key（第二轮架构重构 2026-08-08）。

    ``n_role`` 个 learned role queries（hidden 空间，normal σ=0.02）对 layer-0
    投影后的语言 flat key ``[B, L, hidden_dim]``（即 TaskResampler /
    action_query_cond 使用的形式）做 masked multi-head cross-attention + FFN，
    输出 ``[B, n_role, hidden_dim]`` role tokens。``config.role_query=True`` 时
    policy 构造一个共享实例（``policy.role_resampler``）：TaskResampler 与
    action_query_cond 分支各自取 role 输出的 token 均值作为摘要（实现自定：
    取均值保持 [B, hidden] 摘要形状，与旧路径的 MLP 结构完全复用）。
    """

    def __init__(
        self,
        hidden_dim: int,
        language_dim: int,
        n_role: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_role < 1:
            raise ValueError("n_role must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.n_role = n_role
        self.num_heads = num_heads
        self.dropout = dropout
        self.role_queries = nn.Parameter(torch.empty(n_role, hidden_dim))
        nn.init.normal_(self.role_queries, std=0.02)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, language_key: Tensor, language_mask: Tensor) -> Tensor:
        """language_key: [B, L, hidden]; language_mask: [B, L] 布尔/数值。

        Returns role tokens [B, n_role, hidden] in the language_key dtype
        (padding positions are masked out of the attention weights).
        """
        if language_key.ndim != 3:
            raise ValueError("language_key must have shape [batch, tokens, hidden_dim]")
        batch, length, dim = language_key.shape
        if tuple(language_mask.shape) != (batch, length):
            raise ValueError("language_mask must match [batch, language_tokens]")
        target_dtype = self.q.weight.dtype
        key = language_key.to(dtype=target_dtype)
        query = self._heads(
            self.q(self.norm_q(self.role_queries[None].expand(batch, -1, -1)))
        )
        k = self._heads(self.k(self.norm_k(key)))
        v = self._heads(self.v(key))
        scores = torch.matmul(query.float(), k.float().transpose(-1, -2)) / math.sqrt(
            dim // self.num_heads
        )
        scores = scores.masked_fill(
            ~language_mask.bool()[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=-1).to(dtype=target_dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        out = self._from_heads(torch.matmul(weights, v))
        out = self.out(out)
        out = out + self.ffn(self.norm(out))
        # P0-高优：全 False mask 时 softmax(-inf) 是均匀分布而非零——语言序列
        # 全被遮蔽（如空序列）时 role 输出必须严格为零。
        valid = language_mask.bool().any(dim=-1, keepdim=True)  # [B, 1]
        out = out * valid[:, None, :]
        return out.to(dtype=language_key.dtype)

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

    ``role_resampler``（第二轮架构重构，config.role_query=True 时传入）非 None
    时，摘要改为 RoleQueryResampler 输出的 token 均值（role tokens 已通过
    masked cross-attention 聚合语言；取均值保持 [B, hidden] 摘要形状，MLP 结构
    复用）。否则走旧版 mask-weighted mean（逐字节不变）。
    """

    def __init__(
        self,
        hidden_dim: int,
        n_task: int,
        role_resampler: RoleQueryResampler | None = None,
    ) -> None:
        super().__init__()
        self.task_queries = nn.Parameter(torch.empty(n_task, hidden_dim))
        nn.init.normal_(self.task_queries, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.role_resampler = role_resampler

    def forward(self, language_key: Tensor, language_mask: Tensor) -> Tensor:
        """language_key: [B, tokens, hidden] (layer-0 projected key, flattened)."""
        if self.role_resampler is not None:
            role_out = self.role_resampler(language_key, language_mask)
            summary = role_out.mean(dim=1)
        else:
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


@dataclass(frozen=True)
class DenseReadoutInput:
    """MT-VJ 每决策 dense evidence（契约 §5）：策略级共享投影一次、逐层复用。

    - ``d`` [B, N, D_PROJ]：proj(H11)；``g`` [B, N, D_PROJ]：proj(H5)；
    - ``t`` [B, N, D_PROJ]：proj(ΔtH11)（H11 两时间片之差，按时间片复制回 N
      与 D/G 对齐，t→y→x 序）；
    - ``coord_raw`` [N, _COORD_DIM]：正弦坐标嵌入（K/V 两侧共用）；
    - ``coord_k`` [N, hidden]：coord 嵌入投影——K_dense = W_K·D + coord_k；
    - ``metric_tokens`` [B, 2, hidden] | None：RelationStateEncoder 的
      (z_g, z_ν)，拼接到每层 dense K/V 后（query 仍只有 action tokens）。
    """

    d: Tensor
    g: Tensor
    t: Tensor
    coord_raw: Tensor
    coord_k: Tensor
    metric_tokens: Tensor | None = None


class DenseEvidenceProjector(nn.Module):
    """MT-VJ dense evidence 共享投影（2026-08-10 契约 §5，策略级共享）。

    D = proj_d(H11), G = proj_g(H5), T = proj_t(ΔtH11)：768 → D_PROJ=192 的
    可训练投影（0.42MiB/decision vs 768D 1.69MiB，带宽降 4 倍）。ΔtH11 =
    H11 两时间片（1152 = 2×576，t→y→x 序）逐 patch 之差，即时序创新项。
    coord_k 把正弦坐标嵌入（[N, _COORD_DIM]）投影到 hidden 空间，作为
    K_dense 的坐标加项（正弦、非参数；仅此小投影可训练）。

    仅 ``dense_readout_mtvj=True`` 时构造。这里不做零初始化——只有每层的
    W_o 输出投影严格零初始化（初始等价纪律）。
    """

    def __init__(self, vision_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj_d = nn.Linear(vision_dim, D_PROJ)
        self.proj_g = nn.Linear(vision_dim, D_PROJ)
        self.proj_t = nn.Linear(vision_dim, D_PROJ)
        self.coord_k = nn.Linear(_COORD_DIM, hidden_dim)

    def forward(
        self,
        dense_evidence: dict[int, Tensor],
        metric_tokens: Tensor | None,
    ) -> DenseReadoutInput:
        dtype = self.proj_d.weight.dtype
        h5 = dense_evidence[5]
        h11 = dense_evidence[11]
        if h5.ndim != 3 or h11.ndim != 3:
            raise ValueError(
                "dense_evidence[5]/[11] 必须为 [B, N, vision_dim] 3D 张量，"
                f"got {tuple(h5.shape)} / {tuple(h11.shape)}"
            )
        if h5.shape[:2] != h11.shape[:2]:
            raise ValueError(
                "dense_evidence[5] 与 [11] 的 batch/token 数必须一致，"
                f"got {tuple(h5.shape)} vs {tuple(h11.shape)}"
            )
        if h11.shape[-1] != self.proj_d.in_features:
            raise ValueError(
                f"dense_evidence[11] 最后一维必须等于 vision_dim="
                f"{self.proj_d.in_features}，got {h11.shape[-1]}"
            )
        batch, n_tokens, _ = h11.shape
        if n_tokens % 2:
            raise ValueError(
                f"dense token 数必须为偶数（2 时间片），got {n_tokens}"
            )
        if metric_tokens is not None and (
            metric_tokens.ndim != 3
            or metric_tokens.shape[0] != batch
            or metric_tokens.shape[1] != 2
            or metric_tokens.shape[2] != self.coord_k.out_features
        ):
            raise ValueError(
                "metric_tokens 必须为 [batch, 2, hidden_dim]，"
                f"got {tuple(metric_tokens.shape)}"
            )
        h5 = h5.to(dtype=dtype)
        h11 = h11.to(dtype=dtype)
        half = n_tokens // 2
        d = self.proj_d(h11)  # [B, N, D_PROJ]
        g = self.proj_g(h5)  # [B, N, D_PROJ]
        t_diff = h11[:, half:] - h11[:, :half]  # ΔtH11 [B, N/2, vision]
        t = torch.cat((self.proj_t(t_diff), self.proj_t(t_diff)), dim=1)  # [B, N, D_PROJ]
        coords = dense_coords(n_tokens, device=h11.device).to(dtype=dtype)
        coord_raw = fourier_encode(coords)  # [N, _COORD_DIM]
        coord_k = self.coord_k(coord_raw)  # [N, hidden]
        return DenseReadoutInput(
            d=d,
            g=g,
            t=t,
            coord_raw=coord_raw,
            coord_k=coord_k,
            metric_tokens=(
                metric_tokens.to(dtype=dtype) if metric_tokens is not None else None
            ),
        )


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
        attention_backend: str = "manual",
        sequential: bool = False,
        dual_attention: bool = False,
        dense_readout_mtvj: bool = False,
        action_dense_readout: bool = False,
        protected_action_prefix: int | tuple[int, ...] = 0,
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
        if attention_backend not in ("manual", "auto"):
            raise ValueError(f"unsupported VA attention backend: {attention_backend}")
        self.attention_backend = attention_backend
        prefixes = (
            (protected_action_prefix,)
            if isinstance(protected_action_prefix, int) and protected_action_prefix
            else ()
            if isinstance(protected_action_prefix, int)
            else tuple(protected_action_prefix)
        )
        if any(prefix <= 0 for prefix in prefixes) or tuple(sorted(set(prefixes))) != prefixes:
            raise ValueError(
                "protected_action_prefix must contain increasing positive boundaries"
            )
        self.protected_action_prefixes = prefixes
        # 双注意力（第二轮架构重构 2026-08-08）：仅非 sequential 层（policy
        # 构造时 sequential 层传 False）。动作 query 的 physical 更新不含语言列，
        # 语言列走独立 semantic 注意力；融合门 g_A = σ(G([A_mean, lang_mean]))。
        # 末层 zero-init + 负偏置 → 初始 g ≈ σ(−2) ≈ 0.119 < 0.2（训练起点
        # 接近纯 physical，语义通道从零开始学习）。
        self.dual_attention = dual_attention
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

        if dual_attention:
            self.sem_gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.sem_gate[-1].weight)
            nn.init.zeros_(self.sem_gate[-1].bias)
            self.sem_gate[-1].bias.data.fill_(-2.0)
        # MT-VJ dense action readout（2026-08-10 契约 §5）：每层独立
        # K/V/query 投影 + 严格零初始化的 W_o。D=proj(H11) 等共享投影由
        # 策略级 DenseEvidenceProjector 完成一次，这里只做逐层投影与
        # cross-attention。1152 永远只做 K/V，query 仅 action tokens。
        self.dense_readout_mtvj = dense_readout_mtvj
        if dense_readout_mtvj:
            self.dense_q = nn.Linear(hidden_dim, hidden_dim)
            self.dense_k = nn.Linear(D_PROJ, hidden_dim)
            self.dense_v = nn.Linear(3 * D_PROJ + _COORD_DIM, hidden_dim)
            self.dense_out = nn.Linear(hidden_dim, hidden_dim)
            # W_o 严格零初始化：A_out = A_base + W_o·z ≡ A_base（初始等价）。
            nn.init.zeros_(self.dense_out.weight)
            nn.init.zeros_(self.dense_out.bias)
            self.metric_k = nn.Linear(hidden_dim, hidden_dim)
            self.metric_v = nn.Linear(hidden_dim, hidden_dim)
        # DINOv2 action evidence is an independent additive residual. It does
        # not replace or share parameters with the V-JEPA/MT-VJ dense branch.
        self.action_dense_readout = action_dense_readout
        if action_dense_readout:
            self.action_dense_q = nn.Linear(hidden_dim, hidden_dim)
            self.action_dense_k = nn.Linear(D_PROJ, hidden_dim)
            self.action_dense_v = nn.Linear(3 * D_PROJ + _COORD_DIM, hidden_dim)
            self.action_dense_out = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.action_dense_out.weight)
            nn.init.zeros_(self.action_dense_out.bias)

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

    def _dense_update(
        self, action_norm: Tensor, dense_input: DenseReadoutInput
    ) -> Tensor:
        """MT-VJ dense readout（契约 §5）：z = CrossAttn(A, K_dense, V_dense)。

        K_dense = W_K·D + coord_emb；V_dense = W_V·[D, G, T, coord_emb]；
        A_out = A_base + W_o·z（W_o 严格零初始化 → 初始输出与无 dense 路径
        逐位一致）。1152 patch + 2 metric tokens 只出现在 K/V 侧，query 仅
        action tokens（≤48），绝无 1152×1152 自注意力。``action_norm`` 是
        本层输入 action 的 norm_a_attn 输出（与 base attention 同一 query）。
        """
        d, g, t = dense_input.d, dense_input.g, dense_input.t
        coord_raw, coord_k = dense_input.coord_raw, dense_input.coord_k
        k_dense = self.dense_k(d) + coord_k[None]  # [B, N, hidden]
        v_dense = self.dense_v(
            torch.cat((d, g, t, coord_raw[None].expand(d.shape[0], -1, -1)), dim=-1)
        )
        if dense_input.metric_tokens is not None:
            metric = dense_input.metric_tokens.to(dtype=self.dense_q.weight.dtype)
            k_dense = torch.cat((k_dense, self.metric_k(metric)), dim=1)
            v_dense = torch.cat((v_dense, self.metric_v(metric)), dim=1)
        q = self._to_heads(self.dense_q(action_norm))
        k = self._to_heads(k_dense)
        v = self._to_heads(v_dense)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        z = self._from_heads(torch.matmul(weights, v))
        return self.dense_out(z)

    def _action_dense_update(
        self, action_norm: Tensor, dense_input: DenseReadoutInput
    ) -> Tensor:
        """DINOv2 action-only dense cross-attention residual.

        The evidence layout matches ``DenseEvidenceProjector`` (two temporal
        patch grids). No metric tokens are mixed into this independent tower.
        """
        d, g, t = dense_input.d, dense_input.g, dense_input.t
        coord_raw, coord_k = dense_input.coord_raw, dense_input.coord_k
        k_dense = self.action_dense_k(d) + coord_k[None]
        v_dense = self.action_dense_v(
            torch.cat((d, g, t, coord_raw[None].expand(d.shape[0], -1, -1)), dim=-1)
        )
        q = self._to_heads(self.action_dense_q(action_norm))
        k = self._to_heads(k_dense)
        v = self._to_heads(v_dense)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        z = self._from_heads(torch.matmul(weights, v))
        return self.action_dense_out(z)

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
        for protected_prefix in self.protected_action_prefixes:
            if not protected_prefix < n_action:
                raise ValueError(
                    "protected_action_prefix must be shorter than the action horizon"
                )
            action_key_start = n_visual + n_memory + n_task
            tail_keys = slice(
                action_key_start + protected_prefix,
                action_key_start + n_action,
            )
            # Base visual/prefix/task queries cannot read the H9 extension.
            # Tail queries remain free to read both the protected prefix and tail.
            allowed[:, tail_keys] = False
            tail_queries = slice(
                n_visual + protected_prefix,
                n_visual + n_action,
            )
            allowed[tail_queries, tail_keys] = True
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
        dense_input: DenseReadoutInput | None = None,
        action_dense_input: DenseReadoutInput | None = None,
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
        # The task35 grid16 path has thousands of shared-attention queries per
        # sample.  Materializing FP32 [B,H,Q,K] logits/weights dominates memory
        # (hundreds of MiB per layer).  Fused SDPA evaluates the same flat
        # masked attention without retaining that quadratic tensor.  SMC,
        # dual-attention and attention capture need explicit logits and stay on
        # the legacy path.  ``dropout_p`` must be zero in eval because SDPA
        # applies dropout solely from its argument, independent of module mode.
        use_sdpa = (
            self.attention_backend == "auto"
            and self.attention_variant == "flat"
            and not self.dual_attention
            and not self.save_attention
        )
        weights: Tensor | None
        if use_sdpa:
            update_heads = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
            # No production consumer reads this debug-only scalar.  Avoid a
            # second quadratic QK matmul merely to populate it.
            self.last_max_logit = None
            weights = None
            sem_update = None
        else:
            scores = (
                torch.matmul(query.float(), key.float().transpose(-1, -2))
                * self.scale
            )
            if self.attention_variant == "smc":
                # SMC-Attn：源测度校正（2026-08-05 原创）。
                # softmax 前对每个来源减去 log N_s → 来源总质量 ∝ 平均证据而非
                # token 数量 × 平均证据；复制/细分同一来源不再稀释其他来源。
                # 只作用于动作 query 行（视觉/task query 不做源测度校正），L 计数按
                # 实际有效 mask 计算（padding 不计入）。
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
                    (n_task, log_n_t),
                    (n_action, log_n_a),
                ):
                    log_measure[:, off : off + n_group] = log_n_group
                    off += n_group
                log_measure[:, off : off + n_language] = log_n_l[:, None]
                off += n_language
                log_measure[:, off : off + n_state] = log_n_s
                action_start = n_visual
                action_end = n_visual + n_action
                scores[:, :, action_start:action_end, :] -= log_measure[:, None, None, :]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            self.last_max_logit = float(
                scores.detach().masked_fill(~allowed, float("-inf")).amax()
            )
            if self.dual_attention:
                weights, sem_update = self._dual_attention(
                    scores,
                    query,
                    key,
                    value,
                    action,
                    language_mask,
                    n_visual,
                    n_memory,
                    n_action,
                    n_language,
                    n_task,
                    n_state,
                )
            else:
                weights = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
                sem_update = None
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            update_heads = torch.matmul(weights, value)
        update = self._from_heads(update_heads)

        n_query_v, n_query_a, n_query_t = n_visual, n_action, n_task
        if n_query_t:
            update_v, update_a, update_t = update.split(
                (n_query_v, n_query_a, n_query_t), dim=1
            )
        else:
            update_v, update_a = update.split((n_query_v, n_query_a), dim=1)
            update_t = None
        if sem_update is not None:
            # 双注意力：g_A 已乘入 sem_update（见 _dual_attention），动作行
            # 更新 = physical 更新 + g_A ⊙ semantic 更新，两者共用 out_a 投影。
            update_a = update_a + sem_update
        visual = visual + self.out_v(update_v)
        action = action + self.out_a(update_a)
        if dense_input is not None:
            # MT-VJ：A_out = A_base + W_o·z（W_o 零初始化 → 初始严格等价）。
            action = action + self._dense_update(action_norm, dense_input)
        if action_dense_input is not None:
            action = action + self._action_dense_update(
                action_norm, action_dense_input
            )
        visual = visual + self.ffn_v(self.norm_v_ffn(visual))
        action = action + self.ffn_a(self.norm_a_ffn(action))
        task_out: Tensor | None = None
        if n_query_t:
            task = task + self.out_t(update_t)
            task = task + self.ffn_t(self.norm_t_ffn(task))
            task_out = task
        if self.save_attention:
            if weights is None:
                raise RuntimeError("attention capture unexpectedly used fused SDPA")
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

    def _dual_attention(
        self,
        scores: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        action: Tensor,
        language_mask: Tensor,
        n_visual: int,
        n_memory: int,
        n_action: int,
        n_language: int,
        n_task: int,
        n_state: int,
    ) -> tuple[Tensor, Tensor]:
        """双注意力拆分（第二轮架构重构 2026-08-08，仅非 sequential 层）。

        Key 布局 [visual, memory, task, action, language, state]（与 forward
        中 key_parts 一致；语言列区间 = visual+memory+task+action 之后）。
        返回 (weights, sem_update)：
        - ``weights``：physical 权重——动作 query 行在共享 softmax 中屏蔽
          语言列（physical 更新不读语言），其余 query 行（视觉/task）保持
          原共享路径（含语言列，不动；uni_a 的视觉行规则照旧）；
        - ``sem_update``：动作 query 对语言列的独立注意力更新 [B, n_action,
          hidden]，已乘融合门 g_A = σ(G([A_mean, lang_mean]))，G 为小 MLP
          （2*hidden → hidden → 1），末层 zero-init + bias=-2 → 初始 g ≈ 0.119
          < 0.2。``lang_mean`` 是语言 key 的 mask 加权均值（按 [B, L, hidden]
          还原）。
        """
        lang_start = n_visual + n_memory + n_task + n_action
        lang_end = lang_start + n_language
        action_start = n_visual
        action_end = n_visual + n_action
        physical = scores.clone()
        physical[:, :, action_start:action_end, lang_start:lang_end] = torch.finfo(
            scores.dtype
        ).min
        weights = torch.softmax(physical, dim=-1).to(dtype=value.dtype)
        lang_key = key[:, :, lang_start:lang_end]
        lang_value = value[:, :, lang_start:lang_end]
        sem_scores = torch.matmul(
            query[:, :, action_start:action_end].float(),
            lang_key.float().transpose(-1, -2),
        ) * self.scale
        lang_mask = language_mask.to(device=query.device, dtype=torch.bool)
        sem_scores = sem_scores.masked_fill(
            ~lang_mask[:, None, None, :], torch.finfo(sem_scores.dtype).min
        )
        sem_weights = torch.softmax(sem_scores, dim=-1).to(dtype=value.dtype)
        sem_weights = F.dropout(sem_weights, p=self.dropout, training=self.training)
        sem_update = self._from_heads(torch.matmul(sem_weights, lang_value))
        # P0-高优：全 False 语言 mask 时 softmax(-inf) 是均匀分布而非零——
        # semantic 更新必须严格为零（语言列全被遮蔽时没有可读的语义）。
        valid = lang_mask.any(dim=-1)  # [B]
        sem_update = sem_update * valid[:, None, None]
        flat_lang = lang_key.transpose(1, 2).reshape(
            query.shape[0], -1, self.hidden_dim
        )
        denom = language_mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        lang_mean = (flat_lang * language_mask[:, :, None]).sum(1) / denom
        gate_dtype = next(self.sem_gate.parameters()).dtype
        gate = torch.sigmoid(
            self.sem_gate(
                torch.cat(
                    (
                        action.mean(dim=1).to(dtype=gate_dtype),
                        lang_mean.to(dtype=gate_dtype),
                    ),
                    dim=-1,
                )
            )
        ).to(dtype=sem_update.dtype)  # [B, 1]
        return weights, sem_update * gate[:, None, :]

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
            # Sequential attention intentionally keeps raw-token values; the flat
            # path projects normalized values. Preserve this established behavior.
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
        dense_input: DenseReadoutInput | None = None,
        action_dense_input: DenseReadoutInput | None = None,
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
        if dense_input is not None:
            # MT-VJ：dense readout 注入 Pass 3（correction）的 action 输出。
            action_new = action_new + self._dense_update(
                self.norm_a_attn(action_half), dense_input
            )
        if action_dense_input is not None:
            action_new = action_new + self._action_dense_update(
                self.norm_a_attn(action_half), action_dense_input
            )
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


class EvoCrossAttentionBlock(nn.Module):
    """Evo-1 cross-attention + FFN block adapted to VA hidden states."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        attended, _ = self.attn(
            self.norm1(query),
            context,
            context,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden = query + attended
        return hidden + self.ff(self.norm2(hidden))


class VALast3CrossAttentionReadout(nn.Module):
    """Read all final-three VA action streams without changing the base at init."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        action_horizon: int,
        protected_action_prefixes: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.readers = nn.ModuleList(
            EvoCrossAttentionBlock(hidden_dim, num_heads) for _ in range(3)
        )
        self.gates = nn.Parameter(torch.zeros(3))
        blocked = torch.zeros(
            action_horizon, action_horizon, dtype=torch.bool
        )
        for prefix in protected_action_prefixes:
            if not 0 < prefix < action_horizon:
                raise ValueError("protected action prefix must be inside the horizon")
            blocked[:, prefix:] = True
            blocked[prefix:, prefix:] = False
        self.register_buffer("attn_mask", blocked, persistent=False)

    def forward(self, layer_actions: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        if len(layer_actions) != 3:
            raise ValueError("VA last-three readout requires exactly three layer states")
        query = layer_actions[-1]
        output = query
        for gate, reader, context in zip(
            self.gates, self.readers, layer_actions, strict=True
        ):
            update = reader(query, context, attn_mask=self.attn_mask) - query
            output = output + gate.to(dtype=query.dtype) * update
        return output


class DinoQwenCrossModalBridge(nn.Module):
    """Zero-gated DINO/Qwen tail-layer pairs; no cross-layer averaging."""

    def __init__(
        self,
        language_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_pairs: int = 4,
    ) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.vision_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_pairs)
        )
        self.language_norms = nn.ModuleList(
            nn.LayerNorm(language_dim) for _ in range(num_pairs)
        )
        self.language_projections = nn.ModuleList(
            nn.Linear(language_dim, hidden_dim) for _ in range(num_pairs)
        )
        self.vision_readers = nn.ModuleList(
            EvoCrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_pairs)
        )
        self.language_readers = nn.ModuleList(
            EvoCrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_pairs)
        )
        self.language_delta_k = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_pairs)
        )
        self.language_delta_v = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_pairs)
        )
        self.gates = nn.Parameter(torch.zeros(num_pairs, 2))

    def _to_heads(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.view(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        vision_layers: Tensor,
        language_layers: Tensor,
        language_mask: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[LayerLanguageCache, ...]]:
        if vision_layers.ndim != 4 or vision_layers.shape[1] != self.num_pairs:
            raise ValueError(
                f"cross-modal bridge requires DINO [B,{self.num_pairs},N,H]"
            )
        if language_layers.ndim != 4 or language_layers.shape[1] != self.num_pairs:
            raise ValueError(
                f"cross-modal bridge requires Qwen [B,{self.num_pairs},L,D]"
            )
        if vision_layers.shape[0] != language_layers.shape[0]:
            raise ValueError("cross-modal DINO/Qwen batch sizes differ")
        if language_mask.shape != language_layers.shape[:1] + language_layers.shape[2:3]:
            raise ValueError("cross-modal bridge language mask shape mismatch")
        key_padding_mask = ~language_mask.bool()
        visual_updates = []
        language_deltas = []
        for index in range(self.num_pairs):
            vision = self.vision_norms[index](vision_layers[:, index])
            language = self.language_projections[index](
                self.language_norms[index](language_layers[:, index])
            )
            visual_update = (
                self.vision_readers[index](vision, language, key_padding_mask)
                - vision
            )
            visual_update = self.gates[index, 0].to(vision.dtype) * visual_update
            language_update = (
                self.language_readers[index](language, vision + visual_update)
                - language
            )
            language_update = (
                self.gates[index, 1].to(language.dtype)
                * language_update
                * language_mask[:, :, None].to(dtype=language.dtype)
            )
            visual_updates.append(visual_update)
            language_deltas.append(
                LayerLanguageCache(
                    self._to_heads(self.language_delta_k[index](language_update)),
                    self._to_heads(self.language_delta_v[index](language_update)),
                )
            )
        return tuple(visual_updates), tuple(language_deltas)


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
        semantic_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.flow_cond_mode = flow_cond
        # 第二轮架构重构：语义上下文（compile readout tokens，语言空间
        # language_dim）→ hidden 空间的逐层 cross-attn k/v 投影。None = 关闭
        # （flow_semantic=False 时 policy 不传；传入的 semantic_context 必须
        # 是 hidden_dim 空间）。
        self.semantic_in_dim = semantic_in_dim
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
        if semantic_in_dim is not None:
            self.semantic_proj = nn.Linear(semantic_in_dim, hidden_dim)

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
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        """``semantic_context``（第二轮架构重构 2026-08-08）: 语义上下文
        [B, M, D]（compile readout tokens，D = semantic_in_dim，语言空间）;
        adaln 分支先经 ``semantic_proj`` 投影到 hidden，再与 action_condition
        拼接为 cross-attn 的 k/v（逐层读语义上下文）。None 时与旧行为逐位
        一致。entry 分支无逐层 cross-attn，传入该参数会显式报错。
        """
        if self.flow_cond_mode == "entry" and semantic_context is not None:
            raise ValueError("semantic_context requires flow_cond='adaln'")
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape [batch, horizon, action_dim]")
        if noisy_actions.shape[:2] != action_condition.shape[:2]:
            raise ValueError("noisy_actions batch/horizon must match action_condition")
        if semantic_context is not None:
            context_dim = (
                self.semantic_in_dim
                if self.semantic_in_dim is not None
                else action_condition.shape[-1]
            )
            if (
                semantic_context.ndim != 3
                or semantic_context.shape[0] != action_condition.shape[0]
                or semantic_context.shape[-1] != context_dim
            ):
                raise ValueError(
                    "semantic_context must have shape [batch, tokens, hidden_dim] "
                    "matching action_condition"
                )
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
            # 每个 horizon 槽必须携带自己的 action condition；否则纯噪声
            # 采样时整个 Flow 对槽位置换等变，无法区分第 1..H 步。AdaLN
            # 仍负责全局调制，cross-attention 仍负责逐层读取完整条件序列。
            hidden = action + action_condition + time
            global_cond = torch.cat(
                (
                    time_embedding,
                    action_condition.mean(dim=1),
                ),
                dim=-1,
            )
            if semantic_context is None:
                cond_kv = action_condition
            else:
                context = semantic_context.to(dtype=action_condition.dtype)
                if self.semantic_in_dim is not None:
                    context = self.semantic_proj(context)
                cond_kv = torch.cat((action_condition, context), dim=1)
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
                # 子层 2：条件 cross-attention（每层重新注入 action_condition；
                # flow_semantic 时拼接语义上下文）
                h = ca_norm(hidden) * (1.0 + scale2) + shift2
                q = self._ca_heads(w_q(h))
                k = self._ca_heads(w_k(cond_kv))
                v = self._ca_heads(w_v(cond_kv))
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
        # Training sequences advance at planning_stride. Evaluators may switch
        # this to the checkpoint's full-chunk deployment horizon.
        self.runtime_execution_horizon = config.planning_stride
        # Runtime-only switch: excluded from state_dict and managed by training scripts.
        self.runtime_dino_qwen_bridge_enabled = True
        self.vision_projection = nn.Linear(config.vision_dim, config.hidden_dim)
        self.state_projection = nn.Linear(config.proprio_dim + config.action_dim, config.hidden_dim)
        if config.architecture_version == "dual_tower_expert_v1":
            from va_compound.vision.dual_tower_fusion import MultiLayerDualTowerFusion

            # Learned type embedding table for vision (0) vs robot state (1) tokens
            self.state_type_embed = nn.Embedding(2, config.hidden_dim)
            nn.init.normal_(self.state_type_embed.weight, std=0.02)
            self.dual_tower_fusion = MultiLayerDualTowerFusion(
                vision_dim=config.vision_dim,
                language_dim=config.language_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_pairs=config.fusion_pair_count,
            )
        else:
            self.state_type_embed = None
            self.dual_tower_fusion = None
        # Optional treatment-only modules must not consume the parent RNG stream:
        # otherwise a temporal/geometry ablation changes every common parameter
        # constructed below it and ceases to be a matched initialization.  Their
        # own weights are still deterministic under the caller's seed, while the
        # RNG state seen by action_queries/VA/decoder is restored exactly.
        with torch.random.fork_rng(devices=[]):
            self.main_vision_frame_embedding = (
                nn.Embedding(config.main_vision_frames, config.vision_dim)
                if config.main_vision_temporal
                else None
            )
            if self.main_vision_frame_embedding is not None:
                nn.init.normal_(self.main_vision_frame_embedding.weight, std=0.02)
            self.geometry_projection = (
                nn.Linear(config.metric_geometry_dim, config.hidden_dim)
                if config.metric_geometry_inject
                else None
            )
            if self.geometry_projection is not None:
                nn.init.zeros_(self.geometry_projection.weight)
                nn.init.zeros_(self.geometry_projection.bias)
            self.wmrm = None
        if config.wmrm:
            from va_compound.wmrm import WAM4VA

            with torch.random.fork_rng(devices=[]):
                self.wmrm = WAM4VA(
                    config.hidden_dim,
                    language_dim=(
                        config.language_dim
                        if config.wmrm_full_language_tokens
                        else None
                    ),
                    world_dim=config.wmrm_world_dim,
                    proprio_dim=config.proprio_dim,
                    num_heads=config.num_heads,
                    cycle_steps=config.wmrm_cycle_steps,
                    dino_dim=config.vision_dim,
                    map_size=config.wmrm_map_size,
                    map_channels=config.wmrm_map_channels,
                    map_frames=config.main_vision_frames,
                    map_grid=config.main_vision_grid,
                    env_action_dim=config.action_dim,
                    world_grid=config.wmrm_world_grid,
                    predictor=config.wmrm_predictor,
                    predictor_depth=config.wmrm_predictor_depth,
                    predictor_width=config.wmrm_predictor_width,
                    predictor_heads=config.wmrm_predictor_heads,
                    predictor_copies=config.wmrm_predictor_copies,
                    max_stages=config.wmrm_stage_count(),
                    runtime_integrity_checks=config.runtime_integrity_checks,
                )
        else:
            self.wmrm = None
        if config.world_state_supervision:
            if config.architecture_version != "dual_tower_expert_v1" or self.wmrm is None:
                raise ValueError("state transition supervision requires joint World architecture")
            if not math.isfinite(config.world_state_loss_weight) or config.world_state_loss_weight < 0:
                raise ValueError("World state loss weight must be finite and nonnegative")
            self.wmrm.state_delta_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.proprio_dim),
            )
        if config.va_world_mode == "peer_sync_h6":
            from va_compound.wmrm import ExecutableActionReadout

            with torch.random.fork_rng(devices=[]):
                self.world_action_readout = ExecutableActionReadout(
                    config.hidden_dim,
                    action_dim=config.action_dim,
                    horizon=(
                        config.wmrm_cycle_steps
                        if config.action_horizon == 50
                        else config.action_horizon
                    ),
                    runtime_integrity_checks=config.runtime_integrity_checks,
                )
        else:
            self.world_action_readout = None
        if config.wmrm_stage_gate_start is None:
            self.register_parameter("wmrm_stage_scale", None)
            self.register_parameter("wmrm_belief_message_scale", None)
        else:
            self.wmrm_stage_scale = nn.Parameter(
                torch.zeros(
                    config.wmrm_stage_count() - config.wmrm_stage_gate_start
                )
            )
            self.wmrm_belief_message_scale = nn.Parameter(
                torch.zeros(config.wmrm_stage_count())
            )
        self.last_wmrm = None
        self.action_queries = nn.Parameter(torch.empty(config.action_horizon, config.hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        # 第二轮架构重构：role query 共享实例（config.role_query=True 时构造）。
        # TaskResampler（memory_split 的语言初始化）与 action_query_cond 分支
        # 共用同一实例，把语言 mask-weighted mean 摘要替换为 role tokens 输出。
        self.role_resampler = (
            RoleQueryResampler(
                hidden_dim=config.hidden_dim,
                language_dim=config.language_dim,
                n_role=config.role_query_tokens,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            if config.role_query
            else None
        )

        # PULSE-VA：语言编程局部控制槽（config.local_slots=True 时构造）。
        # Stage A 只训练这些模块 + VA + direct head；C² 控制图 Stage B 再拟合。
        if config.local_slots:
            from va_compound.local_control_slots import (
                LanguageRoleCompiler,
                LocalControlSlotReader,
                RelationTokens,
            )

            self.role_compiler = LanguageRoleCompiler(
                hidden_dim=config.hidden_dim,
                language_dim=config.language_dim,
                n_role=config.local_slot_k,
                num_heads=config.num_heads,
            )
            self.slot_reader = LocalControlSlotReader(
                vision_dim=config.vision_dim,
                hidden_dim=config.hidden_dim,
                num_slots=config.local_slot_k,
                num_heads=config.num_heads,
                multi_mode=config.multi_mode,
            )
            self.relation_tokens = RelationTokens(vision_dim=config.vision_dim)
            self.coarse_pool = nn.AdaptiveAvgPool1d(config.local_coarse)
            # Step 1：模式可见度条件向量（vis [B, K] → vision 空间广播加到
            # 31-token 视觉流；零初始化 → 初始静默，不破坏 dense_readout 行为）。
            self.vis_conditioner = (
                nn.Linear(config.local_slot_k, config.vision_dim)
                if config.multi_mode
                else None
            )
            if self.vis_conditioner is not None:
                nn.init.zeros_(self.vis_conditioner.weight)
                nn.init.zeros_(self.vis_conditioner.bias)
        else:
            self.role_compiler = None
            self.slot_reader = None
            self.relation_tokens = None
            self.coarse_pool = None
            self.vis_conditioner = None

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

        protected_action_prefixes = (
            (6, 15)
            if config.va_world_mode == "peer_sync_h6"
            and config.action_horizon == 50
            and (config.deployment_execution_horizon or config.planning_stride) == 15
            else (6,)
            if config.va_world_mode == "peer_sync_h6"
            and config.action_horizon == 15
            and (config.deployment_execution_horizon or config.planning_stride) == 15
            else ()
        )
        self.layers = nn.ModuleList(
            VACouplingLayer(
                hidden_dim=config.hidden_dim,
                language_dim=config.language_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                mode=config.mode,
                qk_norm=config.qk_norm,
                attention_variant=config.attention_variant,
                attention_backend=config.va_attention_backend,
                sequential=(
                    config.sequential_coupling > 0
                    and (index + 1) % config.sequential_coupling == 0
                ),
                # 双注意力只作用于非 sequential 层（sequential 层保持旧路径）。
                dual_attention=(
                    config.dual_attention
                    and not (
                        config.sequential_coupling > 0
                        and (index + 1) % config.sequential_coupling == 0
                    )
                ),
                dense_readout_mtvj=config.dense_readout_mtvj,
                action_dense_readout=(config.action_vision_backbone != "none"),
                protected_action_prefix=protected_action_prefixes,
            )
            for index in range(config.num_layers)
        )
        with torch.random.fork_rng(devices=[]):
            self.va_last3_readout = (
                VALast3CrossAttentionReadout(
                    config.hidden_dim,
                    config.num_heads,
                    config.action_horizon,
                    protected_action_prefixes,
                )
                if config.va_last3_cross_attn
                else None
            )
            self.dino_qwen_bridge = (
                DinoQwenCrossModalBridge(
                    config.language_dim,
                    config.hidden_dim,
                    config.num_heads,
                    config.dino_qwen_cross_modal_layers,
                )
                if config.dino_qwen_cross_modal_bridge
                else None
            )
        # MT-VJ（2026-08-10 契约 §5）：dense evidence 共享投影（D/G/T：
        # 768 → 192 + 坐标嵌入投影），每层 dense K/V 复用其输出。
        # DINO-metric（2026-08-15）：DINO-main 下同一投影器以
        # vision_dim=main_vision_dim（1024）构造，消费 block11/block23
        # 两帧 [d-2,d] patch evidence（512 token）。
        if config.dense_readout_mtvj:
            projector_vision_dim = (
                config.main_vision_dim
                if config.main_vision_backbone != "vjepa"
                else config.vision_dim
            )
            self.dense_evidence_proj = DenseEvidenceProjector(
                vision_dim=projector_vision_dim,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.dense_evidence_proj = None
        if config.action_vision_backbone != "none":
            self.action_dense_evidence_proj = DenseEvidenceProjector(
                vision_dim=config.action_vision_dim,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.action_dense_evidence_proj = None
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
                hidden_dim=config.hidden_dim,
                n_task=config.task_tokens,
                role_resampler=self.role_resampler,
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
        flow_hidden_dim = config.flow_hidden_dim or config.hidden_dim
        self.flow_condition_projection = (
            nn.Identity()
            if flow_hidden_dim == config.hidden_dim or config.architecture_version == "dual_tower_expert_v1"
            else nn.Linear(config.hidden_dim, flow_hidden_dim)
        )
        flow_head_kwargs = dict(
            hidden_dim=flow_hidden_dim,
            action_dim=config.action_dim,
            num_heads=config.num_heads,
            num_layers=config.flow_layers,
            dropout=config.dropout,
            flow_cond=config.flow_cond,
            # flow_semantic：语义上下文来源随路径——e2e/compile 在语言空间
            # （2048），live+local_slots 是槽输出（vision 空间 768）；
            # head 内部统一经 semantic_proj 投影到 hidden 供逐层 cross-attn。
            semantic_in_dim=(
                (config.vision_dim if config.local_slots else config.language_dim)
                if config.flow_semantic
                else None
            ),
        )
        if config.architecture_version == "dual_tower_expert_v1":
            from va_compound.policy.action_expert import LayerwiseActionExpert

            expert_kwargs = dict(
                action_dim=config.action_dim,
                condition_dim=config.hidden_dim,
                hidden_dim=flow_hidden_dim,
                num_heads=config.num_heads,
                max_horizon=config.action_horizon,
                num_layers=3,
                dropout=config.dropout,
            )
            self.action_expert = LayerwiseActionExpert(**expert_kwargs)
            self.tail_action_expert = (
                LayerwiseActionExpert(**expert_kwargs)
                if config.va_world_mode == "peer_sync_h6"
                and config.action_horizon in {15, 50}
                and (config.deployment_execution_horizon or config.planning_stride)
                == 15
                else None
            )
            self.extension_action_expert = (
                LayerwiseActionExpert(**expert_kwargs)
                if config.va_world_mode == "peer_sync_h6"
                and config.action_horizon == 50
                and (config.deployment_execution_horizon or config.planning_stride) == 15
                else None
            )
            self.flow_head = None
            self.tail_flow_head = None
            self.extension_flow_head = None
        else:
            self.action_expert = None
            self.tail_action_expert = None
            self.extension_action_expert = None
            self.flow_head = FlowMatchingHead(**flow_head_kwargs)
            self.tail_flow_head = (
                FlowMatchingHead(**flow_head_kwargs)
                if config.va_world_mode == "peer_sync_h6"
                and config.action_horizon in {15, 50}
                and (config.deployment_execution_horizon or config.planning_stride)
                == 15
                else None
            )
            self.extension_flow_head = (
                FlowMatchingHead(**flow_head_kwargs)
                if config.va_world_mode == "peer_sync_h6"
                and config.action_horizon == 50
                and (config.deployment_execution_horizon or config.planning_stride) == 15
                else None
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
        role_queries = None
        if self.config.local_slots and not self.config.local_slots_direct288:
            if self.config.local_slots_fixed_query:
                # 消融：纯固定角色种子（无语言实例化），按 batch 展开。
                role_queries = self.role_compiler.role_seeds.unsqueeze(0).expand(
                    language_hidden.shape[0], -1, -1
                )
            else:
                role_queries = self.role_compiler(
                    language_hidden.to(dtype=self.role_compiler.lang_proj.weight.dtype),
                    language_mask.bool(),
                )
            if detach:
                role_queries = role_queries.detach()
        cache = LanguageCache(
            layers=caches,
            attention_mask=language_mask.bool(),
            role_queries=role_queries,
            tokens=language_hidden,
        )
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

    def build_local_vision(
        self,
        dense_tokens: Tensor,   # [B, N, vision_dim] spatiotemporal tokens（288 或 dense 1152）
        coords: Tensor,         # [N, 3] normalized t/y/x
        role_queries: Tensor,   # [B, K, hidden]
    ) -> Tensor:
        """PULSE-VA Stage A readout: 16 coarse + K slots + 3 relations -> 25 tokens.

        对任意 ``N`` 生效：coarse 经 ``coarse_pool`` 自适应池化到 16；槽由
        cross-attention（Q=K 个角色查询 × N 个 dense keys）读出，N=1152
        （Step 0 ``dense_readout``）时该注意力仍在 K/V 侧、不进 VA 自注意力
        （设计文档 §九），VA 视觉流恒为 25 tokens。

        ``multi_mode``（Step 1）时读出的 2 模式/角色槽展平为 [B, K*2, D]，
        视觉流变为 16 coarse + 12 modes + 3 relations = 31 tokens；模式可见度
        vis 经 ``vis_conditioner`` 零初始化投影广播注入。关系 token 用每角色
        最强模式（mode 0，峰评分最高）——多模式配对假设（§二.4）留给 Step 2
        伺服；跟踪先验 prev_mu 由 reader 接收，闭环逐决策传递留待 Step 2/3。

        Ablation cell: ``local_slots_direct288`` returns the 288 dense tokens
        unchanged (no slots) to isolate the pooling-resolution gain.
        """
        if self.config.local_slots_direct288:
            return dense_tokens.to(dtype=self.vision_projection.weight.dtype)
        if self.slot_reader is None or self.relation_tokens is None or self.coarse_pool is None:
            raise ValueError("local_slots modules not built (config.local_slots=False)")
        if coords.ndim != 2 or coords.shape[0] != dense_tokens.shape[1]:
            raise ValueError(
                f"coords 必须为 [N, 3] 且 N == dense token 数（{dense_tokens.shape[1]}），"
                f"got {tuple(coords.shape)}（288 槽坐标与 1152 dense 读出不可混用）"
            )
        target_dtype = self.vision_projection.weight.dtype
        from va_compound.local_control_slots import build_va_vision_input

        dense = dense_tokens.to(dtype=target_dtype)
        coarse = self.coarse_pool(dense.transpose(1, 2)).transpose(1, 2)  # [B, C, D]
        role_queries = role_queries.to(dtype=target_dtype)
        if self.config.multi_mode:
            readout = self.slot_reader(dense, role_queries, coords)  # MultiModeReadout
            slots_flat = readout.slots.reshape(
                dense.shape[0], -1, self.config.vision_dim
            )  # [B, K*2, D]（角色 → 模式 k 主序展平）
            relations = self.relation_tokens(
                readout.slots[:, :, 0], readout.mu[:, :, 0]
            )  # 最强峰模式
            stream = build_va_vision_input(coarse, slots_flat, relations)  # [B, 31, D]
            # vis 作为附加条件向量注入：零初始化投影（初始静默）广播加到视觉流。
            vis_cond = self.vis_conditioner(readout.vis.to(dtype=target_dtype))
            return stream + vis_cond[:, None, :]
        slots, _, centers = self.slot_reader(dense, role_queries, coords)
        relations = self.relation_tokens(slots, centers)
        return build_va_vision_input(coarse, slots, relations)  # [B, 25, D]

    def _wmrm_inject_layers(self) -> set[int]:
        n_layers = len(self.layers)
        if self.config.va_world_mode == "peer_sync_h6":
            return set(range(n_layers - 1))
        mode = self.config.wmrm_inject
        if mode == "last":
            return {n_layers - 1}
        if mode == "all":
            return set(range(n_layers))
        return {index for index in range(n_layers) if index % 2 == 1 or index == n_layers - 1}

    def _peer_world_message(
        self,
        world_state: object,
        stage_index: int,
        *,
        detach_world_map: bool = True,
    ) -> Tensor:
        """Publish the spatial map plus a zero-gated residual belief bridge."""
        world_map = world_state.world_map
        if world_map is None:
            raise ValueError("peer World message requires a predicted world map")
        message = self.wmrm.encode_world_tokens(
            world_map.detach() if detach_world_map else world_map
        )
        scale = self.wmrm_belief_message_scale
        belief = world_state.belief
        if scale is not None and belief is not None:
            count = min(message.shape[1], belief.shape[1])
            residual = belief[:, :count].to(dtype=message.dtype)
            residual = residual * scale[stage_index].to(dtype=message.dtype)
            message = torch.cat(
                (message[:, :count] + residual, message[:, count:]), dim=1
            )
        return message

    def project_shared_eye(self, vision_tokens: Tensor) -> Tensor:
        """Pre-VA DINO/main tokens: optional frame embedding then vision_projection."""
        target_dtype = self.vision_projection.weight.dtype
        vision_input = vision_tokens.to(dtype=target_dtype)
        if self.main_vision_frame_embedding is not None:
            expected_tokens = (
                self.config.main_vision_frames
                * self.config.main_vision_grid
                * self.config.main_vision_grid
            )
            if vision_input.shape[1] != expected_tokens:
                raise ValueError(
                    "temporal DINO main vision expects "
                    f"{expected_tokens} frame-major tokens, got {vision_input.shape[1]}"
                )
            patches_per_frame = self.config.main_vision_grid ** 2
            frame_ids = torch.arange(
                self.config.main_vision_frames, device=vision_input.device
            ).repeat_interleave(patches_per_frame)
            frame_embedding = self.main_vision_frame_embedding(frame_ids).to(
                dtype=target_dtype
            )
            vision_input = vision_input + (
                float(self.config.main_vision_temporal_scale)
                * frame_embedding.unsqueeze(0)
            )
        return self.vision_projection(vision_input)

    def encode_condition(
        self,
        vision_tokens: Tensor,
        proprio: Tensor,
        previous_action: Tensor,
        *,
        language_hidden: Tensor | None = None,
        language_mask: Tensor | None = None,
        language_cache: LanguageCache | None = None,
        cross_modal_vision_layers: Tensor | None = None,
        cross_modal_language_layers: Tensor | None = None,
        visual_memory: VisualMemory | None = None,
        return_visual_memory: bool = False,
        dense_evidence: dict[int, Tensor] | None = None,
        metric_tokens: Tensor | None = None,
        action_dense_evidence: dict[int, Tensor] | None = None,
        metric_g: Tensor | None = None,
        world_goal: Tensor | None = None,
        env_action: Tensor | None = None,
        skip_wmrm: bool = False,
        detach_wmrm_stage_state: bool = False,
    ) -> Tensor | tuple[Tensor, VisualMemory]:
        if (language_hidden is None) == (language_cache is None):
            raise ValueError("provide exactly one of language_hidden or language_cache")
        if vision_tokens.ndim != 3:
            raise ValueError("vision_tokens must have shape [batch, tokens, vision_dim]")
        if self.config.slot_free_policy and (
            metric_g is not None or metric_tokens is not None
        ):
            raise ValueError(
                "slot_free_policy rejects metric_g and metric_tokens at the policy boundary"
            )
        if self.config.architecture_version == "dual_tower_expert_v1":
            if (
                cross_modal_vision_layers is not None
                or cross_modal_language_layers is not None
            ):
                raise ValueError(
                    "dual_tower_expert_v1 rejects legacy cross_modal_* inputs"
                )

        target_dtype = self.vision_projection.weight.dtype
        vision = self.project_shared_eye(vision_tokens)
        state = torch.cat((proprio, previous_action), dim=-1).to(dtype=target_dtype)
        state = self.state_projection(state)
        if self.geometry_projection is not None:
            if metric_g is None:
                raise ValueError(
                    "metric_geometry_inject=True requires metric_g for every decision"
                )
            expected = (vision.shape[0], self.config.metric_geometry_dim)
            if metric_g.shape != expected:
                raise ValueError(
                    f"metric_g must have shape {expected}, got {tuple(metric_g.shape)}"
                )
            if not torch.isfinite(metric_g).all():
                raise ValueError("metric_g must contain only finite values")
            state = state + self.geometry_projection(
                metric_g.to(device=state.device, dtype=target_dtype)
            )

        if language_cache is None:
            language_cache = self.build_language_cache(language_hidden, language_mask)

        bridge_visual_updates = None
        bridge_language_deltas = None
        if self.dino_qwen_bridge is not None and self.runtime_dino_qwen_bridge_enabled:
            if cross_modal_vision_layers is None or cross_modal_language_layers is None:
                raise ValueError(
                    "enabled DINO/Qwen bridge requires layer-specific streams"
                )
            if (
                cross_modal_vision_layers.ndim != 4
                or cross_modal_vision_layers.shape[1]
                != self.dino_qwen_bridge.num_pairs
                or cross_modal_vision_layers.shape[0] != vision.shape[0]
            ):
                raise ValueError("cross_modal_vision_layers has the wrong shape")
            projected_layers = torch.stack(
                [
                    self.project_shared_eye(cross_modal_vision_layers[:, index])
                    for index in range(self.dino_qwen_bridge.num_pairs)
                ],
                dim=1,
            )
            bridge_visual_updates, bridge_language_deltas = self.dino_qwen_bridge(
                projected_layers,
                cross_modal_language_layers,
                language_cache.attention_mask,
            )

        if self.config.architecture_version == "dual_tower_expert_v1":
            # Explicit robot state token: append projected state token with learned type embedding
            # to vision tokens BEFORE the VA loop. Keep WM dino raw anchor unchanged.
            state_token = state.unsqueeze(1)  # [B, 1, hidden_dim]
            vision_type = self.state_type_embed(
                torch.zeros(vision.shape[1], dtype=torch.long, device=vision.device)
            ).unsqueeze(0)  # [1, N_vis, hidden_dim]
            state_type = self.state_type_embed(
                torch.ones(1, dtype=torch.long, device=state.device)
            ).unsqueeze(0)  # [1, 1, hidden_dim]
            vision = torch.cat((vision + vision_type, state_token + state_type), dim=1)
            action = self.action_queries[None].expand(vision.shape[0], -1, -1)
        else:
            action = self.action_queries[None].expand(vision.shape[0], -1, -1) + state[:, None]
        if self.config.action_query_cond:
            # Qwen-conditioned action queries：语言摘要（第 0 层投影 key 的 mask 加权均值）
            # 经 MLP 生成每 horizon 步偏移，zero-init 保证初始等价于静态 query。
            # role_query（第二轮架构重构）开启时摘要改为共享 RoleQueryResampler
            # 输出的 token 均值（role tokens 经 masked cross-attention 聚合语言）。
            first_key = language_cache.layers[0].key  # [B, heads, tokens, head_dim]
            B = vision.shape[0]
            flat_key = first_key.transpose(1, 2).reshape(
                B, -1, self.config.hidden_dim
            )  # [B, tokens, hidden]
            mask = language_cache.attention_mask  # [B, tokens]
            if self.config.role_query:
                role_out = self.role_resampler(flat_key.to(dtype=target_dtype), mask)
                summary = role_out.mean(dim=1).to(dtype=target_dtype)
            else:
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

        def layerwise_bridge(
            index: int, current_vision: Tensor, current_cache: LayerLanguageCache
        ) -> tuple[Tensor, LayerLanguageCache]:
            if (
                bridge_visual_updates is None
                or index >= self.dino_qwen_bridge.num_pairs
            ):
                return current_vision, current_cache
            pair = index
            delta = bridge_language_deltas[pair]
            return current_vision + bridge_visual_updates[pair].to(
                dtype=current_vision.dtype
            ), LayerLanguageCache(
                current_cache.key + delta.key.to(dtype=current_cache.key.dtype),
                current_cache.value + delta.value.to(dtype=current_cache.value.dtype),
            )

        # MT-VJ dense readout（契约 §5）：dense_evidence 非 None 时共享投影
        # 一次，每层用各自 W_K/W_V 构建 dense K/V。None（False 或 True 但
        # 未传）→ 与现有行为逐位一致。
        dense_input = None
        if self.config.dense_readout_mtvj and dense_evidence is not None:
            if not (5 in dense_evidence and 11 in dense_evidence):
                raise ValueError("dense_evidence 必须包含 key 5（H5）与 11（H11）")
            dense_input = self.dense_evidence_proj(dense_evidence, metric_tokens)
        action_dense_input = None
        if (
            self.config.action_vision_backbone != "none"
            and action_dense_evidence is not None
        ):
            if not (5 in action_dense_evidence and 11 in action_dense_evidence):
                raise ValueError(
                    "action_dense_evidence 必须包含 canonical key 5（中层）与 11（末层）"
                )
            action_dense_input = self.action_dense_evidence_proj(
                action_dense_evidence, None
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
            last_action_layers: list[Tensor] = []
            for index, (layer, layer_cache) in enumerate(
                zip(self.layers, language_cache.layers, strict=True)
            ):
                vision, layer_cache = layerwise_bridge(index, vision, layer_cache)
                if layer.sequential:
                    vision, action, task_hat = layer.forward_sequential(
                        vision,
                        action,
                        layer_cache,
                        language_cache.attention_mask,
                        evidence=evidence,
                        task=task_hat,
                        state=state[:, None],
                        dense_input=dense_input,
                        action_dense_input=action_dense_input,
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
                        dense_input=dense_input,
                        action_dense_input=action_dense_input,
                    )
                if (self.va_last3_readout is not None or self.config.architecture_version == "dual_tower_expert_v1") and index >= len(self.layers) - 3:
                    last_action_layers.append(action)
            # The VA layers propose a speculative task update; with EVSM it
            # goes to scratch (task_spec) and is only committed after evidence
            # verification at the next decision.  Without EVSM it commits
            # immediately (original behavior).
            spec = self.task_gate(task, task_hat)
            task_out = task if self.config.evsm else spec
            if self.va_last3_readout is not None:
                action = self.va_last3_readout(tuple(last_action_layers))
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
        peer_mode = self.config.va_world_mode == "peer_sync_h6"
        if self.wmrm is not None:
            from va_compound.wmrm import WAMState

            world_state = (
                visual_memory.world_state
                if visual_memory is not None and visual_memory.world_state is not None
                else WAMState()
            )
            if not isinstance(world_state, WAMState):
                raise ValueError("visual memory world_state must be a WAMState")
            if (
                peer_mode
                and self.wmrm.cycle_steps > self.runtime_execution_horizon
                and world_state.world_map is not None
            ):
                # A long-horizon candidate is stale after receding-horizon P-step
                # execution. Keep cognitive belief, but rebuild the physical map
                # from the new observation at every decision.
                world_state = WAMState(
                    belief=world_state.belief,
                    innovation=world_state.innovation,
                    world_map=None,
                )
            world_state.validate_for(
                batch=action.shape[0],
                hidden_dim=self.config.hidden_dim,
                n_belief=self.wmrm.n_belief,
                n_evidence=self.wmrm.n_evidence,
                dino_dim=self.wmrm.dino_dim,
                map_size=self.wmrm.map_size,
                device=action.device,
            )
            if world_goal is not None:
                raise ValueError(
                    "world_goal is sealed: realized futures must not enter the World forward"
                )
            if skip_wmrm:
                world_message = None
            elif world_state.world_map is not None:
                world_message = (
                    self._peer_world_message(
                        world_state, self.config.wmrm_stage_count() - 1
                    )
                    if peer_mode
                    else self.wmrm.encode_world_tokens(world_state.world_map)
                ).to(device=vision.device, dtype=target_dtype)
            elif world_state.belief is not None:
                world_message = world_state.belief.to(
                    device=vision.device, dtype=target_dtype
                )
            else:
                world_message = None
        else:
            world_state = None
            world_message = None
        self.last_wmrm = None
        self.last_wmrm_auxes: list = []
        self.last_wmrm_pre_actions: list = []
        self.last_world_state_predictions: list = []
        inject_layers = (
            self._wmrm_inject_layers() if self.wmrm is not None else set()
        )
        last_action_layers: list[Tensor] = []
        for index, (layer, layer_cache) in enumerate(
            zip(self.layers, language_cache.layers, strict=True)
        ):
            vision, layer_cache = layerwise_bridge(index, vision, layer_cache)
            previous_visual = None if visual_memory is None else visual_memory.layers[index]
            snapshot_action = action
            snapshot_vision = vision
            if layer.sequential:
                va_vision, va_action, _ = layer.forward_sequential(
                    snapshot_vision,
                    snapshot_action,
                    layer_cache,
                    language_cache.attention_mask,
                    visual_memory=previous_visual,
                    state=world_message,
                    dense_input=dense_input,
                    action_dense_input=action_dense_input,
                )
            else:
                va_vision, va_action, _ = layer(
                    snapshot_vision,
                    snapshot_action,
                    layer_cache,
                    language_cache.attention_mask,
                    visual_memory=previous_visual,
                    state=world_message,
                    dense_input=dense_input,
                    action_dense_input=action_dense_input,
                )
            vision, action = va_vision, va_action
            next_memory.append(vision)
            if (self.va_last3_readout is not None or self.config.architecture_version == "dual_tower_expert_v1") and index >= len(self.layers) - 3:
                last_action_layers.append(action)
            if (
                self.wmrm is not None
                and not skip_wmrm
                and index in inject_layers
            ):
                mask = language_cache.attention_mask
                if self.config.wmrm_full_language_tokens:
                    language_tokens = language_cache.tokens
                    if language_tokens is None:
                        raise ValueError(
                            "wmrm_full_language_tokens requires raw tokens in LanguageCache"
                        )
                    language_keys = None
                    world_language_mask = mask
                else:
                    lang_key = layer_cache.key
                    language_keys = lang_key.transpose(1, 2).reshape(
                        lang_key.shape[0], lang_key.shape[2], self.config.hidden_dim
                    )
                    if mask is not None:
                        language_keys = language_keys * mask.to(
                            dtype=language_keys.dtype
                        )[:, :, None]
                    language_tokens = None
                    world_language_mask = mask
                # Peer inputs expose one complete causal/logged action chunk.
                # The World endpoint matches that full candidate horizon, while
                # deployment executes only planning_stride actions before replanning.
                # The LIBERO path gives World the current VA layer's VL-fused
                # output. Legacy checkpoints keep the pre-VA peer snapshot.
                # The next VA layer reads the new map; the final VA layer only
                # consumes the last map and does not open another World stage.
                if peer_mode:
                    full_env_action = (
                        self.world_action_readout(
                            snapshot_action[:, : self.wmrm.cycle_steps]
                            if self.config.action_horizon == 50
                            else snapshot_action
                        )
                        if env_action is None
                        else env_action
                    )
                    expected_env_action = (
                        snapshot_action.shape[0],
                        self.wmrm.cycle_steps
                        if self.config.action_horizon == 50
                        else self.config.action_horizon,
                        self.config.action_dim,
                    )
                    if tuple(full_env_action.shape) != expected_env_action:
                        raise ValueError(
                            "peer_sync_h6 env_action must be a complete action tensor "
                            f"with shape {expected_env_action}, got "
                            f"{tuple(full_env_action.shape)}"
                        )
                    snapshot_env_action = full_env_action[:, : self.wmrm.cycle_steps]
                else:
                    snapshot_env_action = env_action
                proposal = self.wmrm.propose(
                    snapshot_action,
                    (
                        va_vision
                        if self.config.wmrm_reads_fused_va_tokens
                        else snapshot_vision
                    ),
                    proprio.to(dtype=snapshot_action.dtype),
                    state=world_state,
                    language_keys=language_keys,
                    language_tokens=language_tokens,
                    language_mask=world_language_mask,
                    dino_tokens=vision_tokens.to(dtype=target_dtype),
                    env_action=snapshot_env_action,
                    stage_index=index,
                )
                if self.config.runtime_integrity_checks:
                    proposal.validate_finite(
                        boundary=f"World stage {index} transition"
                    )
                aux = proposal.aux
                next_world_state = proposal.next_world_state
                gated_policy_state = None
                if (
                    self.wmrm_stage_scale is not None
                    and index >= self.config.wmrm_stage_gate_start
                ):
                    if (
                        world_state.belief is None
                        or world_state.innovation is None
                        or world_state.world_map is None
                        or next_world_state.belief is None
                        or next_world_state.innovation is None
                        or next_world_state.world_map is None
                    ):
                        raise ValueError(
                            "gated peer World stages require a complete prior and proposal state"
                        )
                    scale = self.wmrm_stage_scale[
                        index - self.config.wmrm_stage_gate_start
                    ]
                    if self.config.capacity_stage_gate_policy_grad:
                        # Policy/FM may decide how much of this stage to use, but
                        # cannot turn the World predictor into a control scratchpad.
                        gated_policy_state = WAMState(
                            belief=world_state.belief.detach(),
                            innovation=world_state.innovation.detach(),
                            world_map=torch.lerp(
                                world_state.world_map.detach(),
                                next_world_state.world_map.detach(),
                                scale,
                            ),
                        )
                    world_state = WAMState(
                        belief=torch.lerp(
                            world_state.belief, next_world_state.belief, scale
                        ),
                        innovation=torch.lerp(
                            world_state.innovation,
                            next_world_state.innovation,
                            scale,
                        ),
                        world_map=torch.lerp(
                            world_state.world_map,
                            next_world_state.world_map,
                            scale,
                        ),
                    )
                else:
                    world_state = next_world_state
                # Keep the predicted map semantically honest: Flow may train the
                # map-to-policy projection and VA state readers, but not rewrite
                # the future-DINO predictor into a control scratchpad. World loss
                # and recurrent World state remain fully differentiable.
                published_message = (
                    self._peer_world_message(
                        gated_policy_state,
                        index,
                        detach_world_map=False,
                    )
                    if gated_policy_state is not None
                    else self._peer_world_message(world_state, index)
                    if peer_mode and world_state.world_map is not None
                    else proposal.world_message
                )
                if self.config.world_state_supervision and torch.is_grad_enabled():
                    state_message = self._peer_world_message(
                        world_state, index, detach_world_map=False
                    )
                    self.last_world_state_predictions.append(
                        self.wmrm.state_delta_head(state_message.mean(dim=1))
                    )
                world_message = published_message.to(
                    device=vision.device, dtype=target_dtype
                )
                if detach_wmrm_stage_state:
                    world_state = world_state.detach()
                self.last_wmrm = aux
                self.last_wmrm_auxes.append(aux)
                self.last_wmrm_pre_actions.append(snapshot_action)
        if self.va_last3_readout is not None:
            action = self.va_last3_readout(tuple(last_action_layers))
        action_condition = (
            torch.stack([self.action_norm(value) for value in last_action_layers], dim=1)
            if self.config.architecture_version == "dual_tower_expert_v1"
            else self.action_norm(action)
        )
        if return_visual_memory:
            return action_condition, VisualMemory(
                layers=tuple(next_memory),
                world_state=world_state,
            )
        return action_condition

    def flow_velocity(
        self,
        action_condition: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        """Predict velocity using the checkpoint's action decoder contract."""
        if self.config.architecture_version == "dual_tower_expert_v1":
            expected = (noisy_actions.shape[0], 3, self.config.action_horizon, self.config.hidden_dim)
            if tuple(action_condition.shape) != expected:
                raise ValueError(f"layerwise action condition must have shape {expected}")
            if tuple(noisy_actions.shape) != (expected[0], expected[2], self.config.action_dim):
                raise ValueError("noisy action shape does not match policy horizon")
            if semantic_context is not None:
                raise ValueError("layerwise Expert reads VA conditions, not separate semantic context")

            def predict(head, context, horizon):
                return head(noisy_actions[:, :horizon], tuple(context[:, i, :horizon] for i in range(3)), flow_time)

            if self.tail_action_expert is None:
                return predict(self.action_expert, action_condition, self.config.action_horizon)
            prefix = predict(self.action_expert, action_condition, 6)
            tail_condition = action_condition if self.config.tail_flow_condition_grad else action_condition.detach()
            # The deployed P15 prefix must train its VA conditions; forward
            # prefix isolation does not require detaching the executed suffix.
            middle = predict(self.tail_action_expert, action_condition, 15)[:, 6:15]
            if self.config.action_horizon == 50:
                tail = predict(self.extension_action_expert, tail_condition, 50)[:, 15:]
                return torch.cat((prefix, middle, tail), dim=1)
            return torch.cat((prefix, middle), dim=1)
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
        flow_condition = self.flow_condition_projection(action_condition)
        if noisy_actions.shape != expected[:2] + (self.config.action_dim,):
            raise ValueError(
                "noisy_actions must have shape "
                f"[batch, {self.config.action_horizon}, {self.config.action_dim}]"
            )
        if self.tail_flow_head is None:
            return self.flow_head(
                flow_condition, noisy_actions, flow_time, semantic_context
            )

        # The migrated H6 controller remains a closed prefix path. The H9
        # extension may read the complete candidate chunk, but its loss owns
        # only tail_flow_head parameters and cannot overwrite VA/H6 features.
        prefix_horizon = 6
        prefix_velocity = self.flow_head(
            flow_condition[:, :prefix_horizon],
            noisy_actions[:, :prefix_horizon],
            flow_time,
            semantic_context,
        )
        tail_condition = (
            flow_condition
            if self.config.tail_flow_condition_grad
            else flow_condition.detach()
        )
        tail_semantic = semantic_context
        if tail_semantic is not None and not self.config.tail_flow_condition_grad:
            tail_semantic = tail_semantic.detach()
        if self.config.action_horizon == 50:
            if self.extension_flow_head is None:
                raise RuntimeError("H50 nested Flow requires extension_flow_head")
            # Preserve the complete pretrained H15 decoder at migration time:
            # neither the H35 noise nor its conditions enter the old outputs.
            mid_velocity = self.tail_flow_head(
                tail_condition[:, :15],
                noisy_actions[:, :15],
                flow_time,
                tail_semantic,
            )[:, prefix_horizon:15]
            new_tail_velocity = self.extension_flow_head(
                tail_condition,
                noisy_actions,
                flow_time,
                tail_semantic,
            )[:, 15:]
            return torch.cat(
                (prefix_velocity, mid_velocity, new_tail_velocity), dim=1
            )
        tail_velocity = self.tail_flow_head(
            tail_condition,
            noisy_actions,
            flow_time,
            tail_semantic,
        )[:, prefix_horizon:]
        return torch.cat((prefix_velocity, tail_velocity), dim=1)

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
        semantic_context: Tensor | None = None,
        dense_evidence: dict[int, Tensor] | None = None,
        metric_tokens: Tensor | None = None,
        action_dense_evidence: dict[int, Tensor] | None = None,
        metric_g: Tensor | None = None,
        world_goal: Tensor | None = None,
        env_action: Tensor | None = None,
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
            dense_evidence=dense_evidence,
            metric_tokens=metric_tokens,
            action_dense_evidence=action_dense_evidence,
            metric_g=metric_g,
            world_goal=world_goal,
            env_action=env_action,
        )
        if return_visual_memory:
            action_condition, next_memory = encoded
            velocity = self.flow_velocity(
                action_condition,
                noisy_actions,
                flow_time,
                semantic_context=semantic_context,
            )
            return velocity, next_memory
        return self.flow_velocity(
            encoded, noisy_actions, flow_time, semantic_context=semantic_context
        )

    @torch.no_grad()
    def sample_actions(
        self,
        action_condition: Tensor,
        *,
        steps: int = 8,
        noise: Tensor | None = None,
        semantic_context: Tensor | None = None,
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
            v = self.flow_velocity(
                action_condition,
                actions,
                flow_time,
                semantic_context=semantic_context,
            )
            actions = actions + step_size * v
        return actions

    def decode_actions(
        self,
        action_condition: Tensor,
        *,
        steps: int = 8,
        noise: Tensor | None = None,
        c_current: Tensor | None = None,
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        """从 action_condition 解码归一化动作 chunk（C²-VA 统一入口）。

        ``c2_controller=True``：C² 收缩解码——ū = DirectActionHead，
        c̄ = sg(c_anchor) + Δc，K = gain_head；a = clip(ū − K·(c_current − c̄))。
        ``c_current``（[B, c2_control_dim]，P 投影后的当前视觉状态）必须提供，
        否则报错；``steps``/``noise``/``semantic_context`` 忽略（C² 控制参数
        由 direct_head/c2_head 生成，不读语义上下文）。``direct_head=True``
        （无 c2）：确定性 DirectActionHead，``steps``/``noise``/``semantic_context``
        忽略。否则走现有 flow Euler 采样（等价于 ``sample_actions``，
        ``semantic_context`` 透传，现有路径不变）。
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
        return self.sample_actions(
            action_condition,
            steps=steps,
            noise=noise,
            semantic_context=semantic_context,
        )

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
        semantic_context: Tensor | None = None,
    ) -> list[Tensor]:
        """Stochastic Euler trajectory; returns path [x_0, ..., x_K].

        Each transition is x_{k+1} = x_k + (1/K) v(x_k, t_k) + sigma_k * eps_k,
        matching the ReinFlow-lite augmented Markov policy.  sigma has shape
        [K, action_dim] (per-step per-dim transition noise); None gives the
        deterministic Euler path (identical to sample_actions).
        ``semantic_context``（第二轮架构重构）透传给 flow_velocity。
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
            mu = x + step_size * self.flow_velocity(
                action_condition, x, flow_time, semantic_context=semantic_context
            )
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
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        """Log-prob of the sampled denoising path under the flow-noisy
        transition model: sum_k log N(x_{k+1}; x_k + (1/K) v(x_k, t_k),
        sigma_k^2 I).  Returns [batch] log-probs, differentiable in theta.
        ``semantic_context``（第二轮架构重构）透传给 flow_velocity。
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
                action_condition,
                path[index],
                flow_time,
                semantic_context=semantic_context,
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
