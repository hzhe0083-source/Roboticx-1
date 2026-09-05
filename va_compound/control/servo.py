"""C²-IRF v2 Step 2：双新息中央凹交互伺服（Dual-Innovation Foveal Interaction Servo）。

设计文档：``artifacts/c2irf_v2_vision_ablation.md`` §四（局部控制状态）/§五（响应
矩阵 MVP）/§二.4（假设配对混合）。

== 模块定位（MVP，阻尼最小二乘作开关） ==
任务误差 r 驱动受限动作修正；模型新息 ν 驱动增益缩放与中央凹重读触发；显式关系
状态 g 由多模式读出（``MultiModeReadout``，Wave 1 Step 1 契约）+ proprio 显式组装，
语言条件给出有界期望关系 g*（``W_g`` 零初始化 → 初始 g*≡0，即"零对齐"期望关系，
可辨识性保证：训练起点修正 ≡ 0，参考无任意退化空间）。低秩增益只称 learned
gain（设计 §一.4 路线 A），不声称物理 Jacobian。

== 接口契约（Agent E 交付；Agent F 按此集成 eval_metaworld.py，偏差以本 docstring 为准） ==

构造::

    servo = InteractionServo(
        vision_dim=768, lang_dim=512, relation_dim=16, action_dim=4,
        rank=2, dls=False, dls_lambda=1e-2, kappa_max=0.25, delta_max=0.25,
        ...
    )

前向::

    out = servo(readout, proprio, lang_cond, a_prev, g_prev)

- ``readout``: ``MultiModeReadout``（mu [B,K,2,3] 归一化 t/y/x、cov [B,K,2,3,3]、
  vis [B,K]、slots [B,K,2,D]；K=6 角色序 actor_tool/manipuland/target/interface/
  constraint/phase，关系组装只用角色 0/1/2）；
- ``proprio``: [B,4] v5 归一化状态（前 3 = eef xyz，第 4 = gripper 开度）；
- ``lang_cond``: [B,L] 语言条件（role queries 均值或专门 relation query；L=lang_dim）；
- ``a_prev``: [B,4] 上一执行动作（首决策 None）；
- ``g_prev``: [B,G] 上一决策的关系状态（调用方跨决策维护；首决策 None → ν≡0）；
- 返回 ``ServoOutput``（字段见类定义）。

评估侧动作修正：``a = clip(a_base + out.correction, -1, 1)``（correction 已含
阶段幅度上限、假设混合权重与 β 信任缩放）。``out.innovation_flag == 1`` 时触发
中央凹提前全局刷新（设计 §三.3：|ν| > τ_ν 或 H(w) > τ_H 或 vis < τ_v）。

== 训练侧约定（train.py） ==
- 修正加在 flow 速度输出（直路径 FM 的 v 目标 = a−noise，v 修正等价最终动作空间
  修正 a = clip(a_base + βΔa)）；损失保持 L_FM 形式不变，修正路径可微回传；
- 配对采样（clean/perturbed 共享 (τ,ε)）见 train.py ``sample_flow_matching_inputs_paired``。

无新增辅助 loss；全部由既有动作损失（L_FM）监督。
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from va_compound.local_control_slots import MultiModeReadout

# 角色序（与 local_control_slots.ROLE_NAMES 一致）：关系组装只用前三个角色。
ROLE_TOOL = 0        # actor_tool
ROLE_MANIPULAND = 1  # manipuland
ROLE_TARGET = 2      # target

# 阶段枚举（stage 输出与 ZeroInitReference 的 phase 独热编码共用）。
STAGE_COARSE = 0        # 距离远：无修正
STAGE_APPROACHING = 1   # 接近：幅度上限 5–10%
STAGE_PRE_CONTACT = 2   # 接触前：幅度上限 10–20%
STAGE_CONTACT = 3       # 接触：幅度上限 20–30%（需高可见度）
STAGE_UNCERTAIN = 4     # 遮挡/低可见度：无修正
N_STAGES = 5

# 显式关系状态维度（设计 §四：12–24 维，取 16）。
G_DIM = 16


def stage_from_distance(
    d: Tensor,
    vis: Tensor,
    *,
    d_coarse: float,
    d_approaching: float,
    d_pre_contact: float,
    vis_threshold: float,
) -> Tensor:
    """距离启发式 + 可见度 → 阶段 [B] long（0..4）。

    ``d`` 为归一化坐标平面距离（y,x），``vis`` 为参与角色最小可见度。
    顺序：coarse(d>d_coarse) → approaching(d>d_ap) → pre_contact(d>d_pc) →
    contact(d≤d_pc)；任何角色 vis < τ_v → uncertain（遮挡时几何不可信，
    阶段幅度上限归零）。
    """
    device = d.device
    coarse = torch.tensor(STAGE_COARSE, device=device)
    approaching = torch.tensor(STAGE_APPROACHING, device=device)
    pre_contact = torch.tensor(STAGE_PRE_CONTACT, device=device)
    contact = torch.tensor(STAGE_CONTACT, device=device)
    uncertain = torch.tensor(STAGE_UNCERTAIN, device=device)
    stage = torch.where(
        d <= d_pre_contact,
        contact,
        torch.where(
            d <= d_approaching,
            pre_contact,
            torch.where(d <= d_coarse, approaching, coarse),
        ),
    )
    return torch.where(vis < vis_threshold, uncertain, stage)


class RelationStateProjector(nn.Module):
    """显式关系状态 g_t [B, G]（G=16，设计 §四）。

    从 ``MultiModeReadout``（每角色主模式选择后的 mu/cov/vis）+ proprio 显式
    组装，附一个零初始化小投影（初始静默 → g ≡ 显式几何，几何先验不被随机
    投影污染）。组件（归一化坐标域，y=μ[1]、x=μ[2]、z=t=μ[0]）：

    - [0:2]  μ_tool − μ_manip (y,x)
    - [2:4]  μ_manip − μ_target (y,x)
    - [4:6]  μ_tool − μ_target (y,x)
    - [6]    log 尺度比 log(√trΣ_manip / √trΣ_target)
    - [7]    log 尺度比 log(√trΣ_tool / √trΣ_manip)
    - [8:10] 窗口内模式位移差 (μ_m,m0−μ_m,m1) − (μ_t,m0−μ_t,m1) (y,x)
             （两 top-2 峰的位置差近似窗口内位移；静态单峰时 ≈ 0）
    - [10]   vis_manip·vis_target（可见度加权）
    - [11]   vis_tool·vis_manip
    - [12]   z_manip − z_target（归一化时间坐标差）
    - [13]   q_gripper（proprio[3]）
    - [14]   eef 高度（proprio[2]）
    - [15]   ‖μ_tool − μ_manip‖（y,x 平面距离）
    """

    def __init__(self, relation_dim: int = G_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.relation_dim = relation_dim
        # 可训练小投影：g = g_explicit + proj(g_explicit)；输出零初始化 →
        # 初始完全等于显式组装（"可训练投影少量"，设计 §四）。
        # 已知权衡（设计 §八：不要制造梯度死区）：输出零初始化下残差分支首层
        # （proj[0]）初始梯度 ≡ 0，第一步后随 proj[2] 离开零点自动解除——
        # 与 §八 方案二（门控 0.01 + 输出投影零初始化）同一取舍，可辨识性优先。
        self.proj = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, relation_dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        mu: Tensor,     # [B, K, 2, 3] 归一化 t/y/x（2 = top-2 模式）
        cov: Tensor,    # [B, K, 2, 3, 3] 每模式峰邻域协方差
        vis: Tensor,    # [B, K] 可见度 = 1 − P(NULL)
        proprio: Tensor,  # [B, 4] v5 归一化状态（前 3 = eef xyz，第 4 = gripper）
        modes: Tensor | None = None,  # [K] long 每角色选用的模式（默认全 0）
    ) -> Tensor:
        """→ g [B, G]（显式组装 + 零初始化投影）。"""
        B, K = vis.shape
        if mu.ndim != 4 or tuple(mu.shape[:2]) != (B, K) or mu.shape[2:] != (2, 3):
            raise ValueError(
                f"mu 必须为 [B, K, 2, 3]，got {tuple(mu.shape)}"
            )
        if tuple(cov.shape) != (B, K, 2, 3, 3):
            raise ValueError(f"cov 必须为 [B, K, 2, 3, 3]，got {tuple(cov.shape)}")
        if proprio.ndim != 2 or proprio.shape[0] != B or proprio.shape[1] != 4:
            raise ValueError(f"proprio 必须为 [B, 4]，got {tuple(proprio.shape)}")
        if modes is None:
            modes = torch.zeros(K, dtype=torch.long, device=mu.device)
        modes = modes.to(device=mu.device, dtype=torch.long)
        if modes.shape != (K,):
            raise ValueError(f"modes 必须为 [K]={K}，got {tuple(modes.shape)}")
        if B == 0 or K < 3:
            raise ValueError(f"servo 需要 K>=3（tool/manip/target），got K={K}")
        g_idx = torch.arange(K, device=mu.device)
        mu0 = mu[:, g_idx, modes]      # [B, K, 3] 主模式位置（逐角色模式选择）
        cov0 = cov[:, g_idx, modes]    # [B, K, 3, 3] 主模式协方差
        # 各向同性 σ² 代理：tr(Σ)/3（归一化坐标域，含 cov_floor 数值下限）。
        tr = cov0.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0  # [B, K]
        tool, manip, target = ROLE_TOOL, ROLE_MANIPULAND, ROLE_TARGET
        tm = mu0[:, tool, 1:3] - mu0[:, manip, 1:3]      # [B, 2]
        mt = mu0[:, manip, 1:3] - mu0[:, target, 1:3]
        tt = mu0[:, tool, 1:3] - mu0[:, target, 1:3]
        dmu_manip = mu[:, manip, 0, 1:3] - mu[:, manip, 1, 1:3]
        dmu_target = mu[:, target, 0, 1:3] - mu[:, target, 1, 1:3]
        dmu_diff = dmu_manip - dmu_target                 # [B, 2]
        safe = 1e-6
        log_scale_mt = 0.5 * (
            tr[:, manip].clamp_min(safe).log() - tr[:, target].clamp_min(safe).log()
        )[:, None]
        log_scale_tm = 0.5 * (
            tr[:, tool].clamp_min(safe).log() - tr[:, manip].clamp_min(safe).log()
        )[:, None]
        vis_mt = (vis[:, manip] * vis[:, target])[:, None]
        vis_tm = (vis[:, tool] * vis[:, manip])[:, None]
        z_diff = (mu0[:, manip, 0] - mu0[:, target, 0])[:, None]
        gripper = proprio[:, 3:4]
        eef_z = proprio[:, 2:3]
        dist_tm = tm.norm(dim=-1, keepdim=True)
        g = torch.cat(
            [
                tm, mt, tt,
                log_scale_mt, log_scale_tm,
                dmu_diff,
                vis_mt, vis_tm,
                z_diff,
                gripper, eef_z,
                dist_tm,
            ],
            dim=-1,
        )  # [B, 16]
        return g + self.proj(g)

    def relation_weights(
        self,
        cov: Tensor,      # [B, K, 2, 3, 3]（与 forward 同源）
        vis: Tensor,      # [B, K]
        modes: Tensor | None = None,
        floor: float = 1e-4,
    ) -> Tensor:
        """DLS 不确定性权重 w_g = vis_pair/(σ²+ε)（设计 §一.2，W = diag(w_g)）。

        仅依赖协方差与可见度（与 μ 无关）；各组件按角色对映射各向同性 σ² 代理
        （σ²_a + σ²_b）；vis 乘积组件与 proprio 组件取恒 1。返回 [B, G]。
        """
        B, K = vis.shape
        if modes is None:
            modes = torch.zeros(K, dtype=torch.long, device=cov.device)
        modes = modes.to(device=cov.device, dtype=torch.long)
        g_idx = torch.arange(K, device=cov.device)
        cov0 = cov[:, g_idx, modes]  # [B, K, 3, 3] 逐角色主模式协方差
        sig2 = cov0.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0  # [B, K]
        # (start, end, 角色对|None)：16 个组件的不确定性映射。
        specs = (
            (0, 2, (ROLE_TOOL, ROLE_MANIPULAND)),
            (2, 4, (ROLE_MANIPULAND, ROLE_TARGET)),
            (4, 6, (ROLE_TOOL, ROLE_TARGET)),
            (6, 7, (ROLE_MANIPULAND, ROLE_TARGET)),
            (7, 8, (ROLE_TOOL, ROLE_MANIPULAND)),
            (8, 10, (ROLE_MANIPULAND, ROLE_TARGET)),
            (10, 11, None),   # vis 乘积本身即置信度
            (11, 12, None),
            (12, 13, (ROLE_MANIPULAND, ROLE_TARGET)),
            (13, 14, None),   # gripper
            (14, 15, None),   # eef 高度
            (15, 16, (ROLE_TOOL, ROLE_MANIPULAND)),
        )
        w = torch.ones(B, self.relation_dim, device=cov.device, dtype=cov.dtype)
        for start, end, pair in specs:
            if pair is None:
                continue
            a, b = pair
            denom = sig2[:, a] + sig2[:, b] + floor
            w[:, start:end] = (
                (vis[:, a] * vis[:, b]).unsqueeze(-1).expand(-1, end - start)
                / denom.unsqueeze(-1)
            )
        return w


class ZeroInitReference(nn.Module):
    """期望关系 g* = δ_max·tanh(W_g[lang_cond, phase])（设计 §四）。

    ``W_g`` 零初始化 → 初始 g* ≡ 0（"零对齐"期望关系）：训练起点修正 ≈ 0，
    只有数据证明需要偏移时才打开（可辨识性：参考无任意退化空间，比任意
    reference head 更容易辨识）。``phase`` 为阶段独热（N_STAGES=5）。
    """

    def __init__(
        self,
        lang_dim: int,
        relation_dim: int = G_DIM,
        delta_max: float = 0.25,
        n_phase: int = N_STAGES,
    ) -> None:
        super().__init__()
        self.delta_max = delta_max
        self.n_phase = n_phase
        self.proj = nn.Linear(lang_dim + n_phase, relation_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, lang_cond: Tensor, phase: Tensor) -> Tensor:
        """lang_cond [B, L]、phase [B] long → g* [B, G]（|g*|∞ ≤ δ_max）。"""
        if lang_cond.ndim != 2:
            raise ValueError(f"lang_cond 必须为 [B, L]，got {tuple(lang_cond.shape)}")
        if phase.ndim != 1 or phase.shape[0] != lang_cond.shape[0]:
            raise ValueError("phase 必须为 [B] long 且与 lang_cond 同批")
        phase_oh = F.one_hot(phase, num_classes=self.n_phase).to(
            dtype=lang_cond.dtype
        )
        return self.delta_max * torch.tanh(
            self.proj(torch.cat([lang_cond, phase_oh], dim=-1))
        )


class LowRankServo(nn.Module):
    """低秩有界增益伺服（设计 §五 MVP；只称 learned gain，不称物理 Jacobian）。

    ``K = κ·U·Vᵀ``（U [A,r]、V [G,r]，rank r=2/4），``κ = κ_max·tanh(ρ)``；
    ``ρ`` 零初始化 → κ≡0 → 训练起点修正 ≡ 0（与 g* 零初始化同一可辨识性保证）。
    修正 ``Δa = K·r``，按阶段幅度上限 clip：|Δa|∞ ≤ cap(stage)
    （coarse 0 / approaching 5–10% / pre-contact 10–20% / contact 20–30% /
    uncertain 0，归一化动作域比例）。

    ``dls=True``（--servo-dls 开关）：阻尼最小二乘
    ``Δa = (K W Kᵀ + λI)⁻¹ K W r``（W = diag(w_g) 由视觉协方差与可见度决定，
    设计 §一.2；K 为学习增益转置充当 J，仅需解 4×4 线性系统），再按阶段上限
    clip。κ≡0 时 rhs ≡ 0 → Δa ≡ 0（与直乘路径同一起点）。
    """

    def __init__(
        self,
        relation_dim: int = G_DIM,
        action_dim: int = 4,
        rank: int = 2,
        kappa_max: float = 0.25,
        dls: bool = False,
        dls_lambda: float = 1e-2,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank 必须为正")
        self.kappa_max = float(kappa_max)
        self.dls = dls
        self.dls_lambda = float(dls_lambda)
        self.U = nn.Parameter(torch.empty(action_dim, rank))
        self.V = nn.Parameter(torch.empty(relation_dim, rank))
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)
        self.rho = nn.Parameter(torch.zeros(()))  # κ 零初始化 → 起点修正 ≡ 0
        # 阶段幅度上限（归一化动作域；coarse/uncertain 恒 0）。
        self.register_buffer(
            "caps",
            torch.tensor([0.0, 0.075, 0.15, 0.25, 0.0]),
            persistent=False,
        )

    def gain(self) -> Tensor:
        """当前低秩增益 K [A, G] = κ·U·Vᵀ（κ = κ_max·tanh(ρ)，|κ| ≤ κ_max）。"""
        kappa = self.kappa_max * torch.tanh(self.rho)
        return kappa * (self.U @ self.V.t())

    def correction(
        self,
        r: Tensor,          # [B, G] 任务误差
        stage: Tensor,      # [B] long 阶段
        w: Tensor | None = None,  # [B, G] DLS 不确定性权重（dls 模式必需）
    ) -> Tensor:
        """→ Δa [B, A]（阶段上限 clip 后；dls 模式解 4×4 线性系统）。"""
        if r.ndim != 2 or r.shape[1] != self.V.shape[0]:
            raise ValueError(
                f"r 必须为 [B, {self.V.shape[0]}]，got {tuple(r.shape)}"
            )
        if stage.ndim != 1 or stage.shape[0] != r.shape[0]:
            raise ValueError("stage 必须为 [B] long 且与 r 同批")
        K = self.gain()  # [A, G]
        if self.dls:
            if w is None:
                raise ValueError("dls 模式需要不确定性权重 w [B, G]")
            if tuple(w.shape) != tuple(r.shape):
                raise ValueError(
                    f"w 必须为 [B, G]={tuple(r.shape)}，got {tuple(w.shape)}"
                )
            KW = K.unsqueeze(0) * w.unsqueeze(1)  # [B, A, G]（行乘 w）
            system = KW @ K.t()  # [B, A, A]（= K W Kᵀ，4×4）
            eye = torch.eye(
                system.shape[-1], device=K.device, dtype=K.dtype
            )
            delta = torch.linalg.solve(
                system + self.dls_lambda * eye, KW @ r.unsqueeze(-1)
            ).squeeze(-1)  # [B, A]
        else:
            delta = r @ K.t()  # [B, A] = K·r
        cap = self.caps[stage].unsqueeze(-1)  # [B, 1]
        return delta.clamp(-cap, cap)


class HypothesisPairMixer(nn.Module):
    """manipuland/target 各 top-2 模式 → 4 配对假设的混合网络（设计 §二.4）。

    ``w_ij = softmax_4(f(q_relation, z_o,i, z_t,j, μ_o,i − μ_t,j))``；假设熵
    H(w) > τ_H 时调用方降 β / 触发重读（InteractionServo 输出 hyp_entropy /
    innovation_flag）。槽内容先经瓶颈投影（D→64）再进 MLP（防 2×768 大输入）。
    """

    def __init__(
        self,
        vision_dim: int = 768,
        lang_dim: int = 512,
        bottleneck: int = 64,
    ) -> None:
        super().__init__()
        self.z_proj = nn.Linear(vision_dim, bottleneck)
        self.net = nn.Sequential(
            nn.Linear(2 * bottleneck + 3 + lang_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1, bias=False),  # 无 bias：softmax 对 logit 常值平移不变，
            # bias 恒无梯度（死参数）——不构造。
        )

    def forward(self, lang_cond: Tensor, slots: Tensor, mu: Tensor) -> Tensor:
        """lang_cond [B, L]、slots [B, K, 2, D]、mu [B, K, 2, 3] → w [B, 4]。"""
        logits = []
        for i in (0, 1):
            for j in (0, 1):
                z_o = self.z_proj(slots[:, ROLE_MANIPULAND, i])
                z_t = self.z_proj(slots[:, ROLE_TARGET, j])
                d = mu[:, ROLE_MANIPULAND, i] - mu[:, ROLE_TARGET, j]
                logits.append(
                    self.net(torch.cat([lang_cond, z_o, z_t, d], dim=-1))
                )
        return torch.softmax(torch.stack(logits, dim=1).squeeze(-1), dim=1)  # [B, 4]


class ServoOutput(NamedTuple):
    """``InteractionServo.forward`` 的输出契约。

    - ``g`` [B, G]：加权关系状态 g_w = Σ_ij w_ij·g_ij（跨决策由调用方维护 g_prev）；
    - ``g_star`` [B, G]：期望关系（零初始化 → 训练起点 ≡ 0，即对齐关系）；
    - ``task_error`` [B, G]：r = g* − g_w（任务误差，负责产生动作修正）；
    - ``innovation`` [B, G]：ν = g_w − (g_prev + Kᵀ·a_prev)（模型新息，负责
      增益缩放 / 重读触发；g_prev 缺失时 ≡ 0）；
    - ``innovation_flag`` [B]：float {0,1} = (‖ν‖>τ_ν) | (H(w)>τ_H) | (vis<τ_v)，
      1 时触发中央凹提前全局刷新；
    - ``correction`` [B, A]：最终动作修正 β·Σ_ij w_ij·Δa_ij（逐假设阶段上限
      已乘入；评估侧 a = clip(a_base + correction, −1, 1)）；
    - ``beta`` [B]：信任缩放 s_ν·s_H·vis_min（新息/假设熵/可见度共同决定）；
    - ``stage`` [B] long：主导（argmax w）假设的阶段 0..4；
    - ``hyp_weights`` [B, 4]：假设混合权重 w_ij；
    - ``hyp_entropy`` [B]：H(w)（自然对数）。
    """

    g: Tensor
    g_star: Tensor
    task_error: Tensor
    innovation: Tensor
    innovation_flag: Tensor
    correction: Tensor
    beta: Tensor
    stage: Tensor
    hyp_weights: Tensor
    hyp_entropy: Tensor


class InteractionServo(nn.Module):
    """双新息交互伺服总装（C²-IRF v2 Step 2；设计 §四/§五/§二.4 最小完整算法）。

    forward 流程：
      1) manipuland/target 各 top-2 模式 → 4 配对假设；每假设独立组装 g_ij
         （RelationStateProjector，模式选择 tool=0 / manip=i / target=j）与
         DLS 不确定性权重 w_unc_ij（relation_weights）；
      2) w_ij = HypothesisPairMixer(lang, slots, mu)；g_w = Σ w_ij·g_ij；
         主导阶段（argmax w 假设的距离启发式 + vis）→ g* = ZeroInitReference；
      3) r_ij = g* − g_ij；Δa_ij = LowRankServo(r_ij, stage_ij, w_unc_ij)
         （逐假设阶段上限）；Δa = Σ_ij w_ij·Δa_ij；
      4) ν = g_w − (g_prev + Kᵀ·a_prev)（K = 当前低秩增益，MVP 近似 J；
         g_prev/a_prev 首决策 None → ν≡0）；
         β = exp(−‖ν‖²/2σ_ν²)·exp(−relu(H−τ_H)/σ_H)·vis_min；correction = β·Δa；
      5) innovation_flag = (‖ν‖>τ_ν) | (H>τ_H) | (vis_min<τ_v)。

    参数（默认值即设计 §五 建议区间中点）：coarse 0 / approaching 7.5% /
    pre-contact 15% / contact 25% / uncertain 0；κ_max=0.25；δ_max=0.25。
    """

    def __init__(
        self,
        vision_dim: int = 768,
        lang_dim: int = 512,
        relation_dim: int = G_DIM,
        action_dim: int = 4,
        rank: int = 2,
        dls: bool = False,
        dls_lambda: float = 1e-2,
        kappa_max: float = 0.25,
        delta_max: float = 0.25,
        hyp_bottleneck: int = 64,
        nu_scale: float = 0.1,
        nu_threshold: float = 0.15,
        ent_threshold: float = 0.9,
        ent_scale: float = 0.2,
        vis_threshold: float = 0.3,
        d_coarse: float = 0.5,
        d_approaching: float = 0.3,
        d_pre_contact: float = 0.15,
    ) -> None:
        super().__init__()
        self.relation_dim = relation_dim
        self.action_dim = action_dim
        self.nu_scale = float(nu_scale)
        self.nu_threshold = float(nu_threshold)
        self.ent_threshold = float(ent_threshold)
        self.ent_scale = float(ent_scale)
        self.vis_threshold = float(vis_threshold)
        self.d_coarse = float(d_coarse)
        self.d_approaching = float(d_approaching)
        self.d_pre_contact = float(d_pre_contact)
        self.projector = RelationStateProjector(relation_dim=relation_dim)
        self.reference = ZeroInitReference(
            lang_dim=lang_dim, relation_dim=relation_dim, delta_max=delta_max
        )
        self.mixer = HypothesisPairMixer(
            vision_dim=vision_dim, lang_dim=lang_dim, bottleneck=hyp_bottleneck
        )
        self.servo = LowRankServo(
            relation_dim=relation_dim,
            action_dim=action_dim,
            rank=rank,
            kappa_max=kappa_max,
            dls=dls,
            dls_lambda=dls_lambda,
        )

    def forward(
        self,
        readout: MultiModeReadout,
        proprio: Tensor,     # [B, 4]
        lang_cond: Tensor,   # [B, L]
        a_prev: Tensor | None = None,  # [B, 4] 上一执行动作（首决策 None）
        g_prev: Tensor | None = None,  # [B, G] 上一决策关系状态（None → ν≡0）
    ) -> ServoOutput:
        """→ ServoOutput（详见类定义与模块 docstring 的接口契约）。"""
        B = proprio.shape[0]
        mu, cov, vis, slots = readout.mu, readout.cov, readout.vis, readout.slots
        K = int(mu.shape[1])
        if K < 3:
            raise ValueError(
                f"servo 需要 K>=3（actor_tool/manipuland/target），got K={K}"
            )
        if tuple(mu.shape) != (B, K, 2, 3):
            raise ValueError(
                f"mu 必须为 [B, {K}, 2, 3]，got {tuple(mu.shape)}"
            )
        if tuple(cov.shape) != (B, K, 2, 3, 3):
            raise ValueError(
                f"cov 必须为 [B, {K}, 2, 3, 3]，got {tuple(cov.shape)}"
            )
        if tuple(vis.shape) != (B, K):
            raise ValueError(f"vis 必须为 [B, {K}]，got {tuple(vis.shape)}")
        if slots.ndim != 4 or tuple(slots.shape[:2]) != (B, K):
            raise ValueError(
                f"slots 必须为 [B, {K}, 2, D]，got {tuple(slots.shape)}"
            )
        if proprio.shape != (B, 4):
            raise ValueError(f"proprio 必须为 [B, 4]，got {tuple(proprio.shape)}")
        if lang_cond.ndim != 2 or lang_cond.shape[0] != B:
            raise ValueError(
                f"lang_cond 必须为 [B, L]，got {tuple(lang_cond.shape)}"
            )
        if (a_prev is None) != (g_prev is None):
            raise ValueError("a_prev 与 g_prev 必须同时给出或同时为 None（ν 依赖两者）")
        if a_prev is not None and tuple(a_prev.shape) != (B, self.action_dim):
            raise ValueError(
                f"a_prev 必须为 [B, {self.action_dim}]，got {tuple(a_prev.shape)}"
            )
        if g_prev is not None and tuple(g_prev.shape) != (B, self.relation_dim):
            raise ValueError(
                f"g_prev 必须为 [B, {self.relation_dim}]，got {tuple(g_prev.shape)}"
            )

        # 1) 4 配对假设：独立 g_ij / 阶段 / DLS 权重 / 修正。
        g_ij_list: list[Tensor] = []
        d_ij_list: list[Tensor] = []
        stage_ij_list: list[Tensor] = []
        w_unc_ij_list: list[Tensor] = []
        vis_min = vis[:, :3].min(dim=-1).values  # [B] 参与角色最小可见度
        for i in (0, 1):
            for j in (0, 1):
                modes = torch.zeros(K, dtype=torch.long, device=mu.device)
                modes[ROLE_MANIPULAND] = i
                modes[ROLE_TARGET] = j
                g_ij_list.append(self.projector(mu, cov, vis, proprio, modes))
                # 阶段启发式距离：tool(mode 0) → manipuland(mode i)，y,x 平面。
                d_ij = (
                    mu[:, ROLE_TOOL, 0, 1:3] - mu[:, ROLE_MANIPULAND, i, 1:3]
                ).norm(dim=-1)
                d_ij_list.append(d_ij)
                stage_ij_list.append(
                    stage_from_distance(
                        d_ij,
                        vis_min,
                        d_coarse=self.d_coarse,
                        d_approaching=self.d_approaching,
                        d_pre_contact=self.d_pre_contact,
                        vis_threshold=self.vis_threshold,
                    )
                )
                w_unc_ij_list.append(
                    self.projector.relation_weights(cov, vis, modes)
                )

        # 2) 假设混合 → 主导阶段 → 期望关系（零初始化）。
        w = self.mixer(lang_cond, slots, mu)  # [B, 4]
        g_ij = torch.stack(g_ij_list, dim=1)  # [B, 4, G]
        g = (w.unsqueeze(-1) * g_ij).sum(dim=1)  # [B, G] 加权关系状态
        stage_ij = torch.stack(stage_ij_list, dim=1)  # [B, 4]
        dominant = w.argmax(dim=1)  # [B]
        stage = stage_ij.gather(1, dominant[:, None]).squeeze(1)  # [B]
        g_star = self.reference(lang_cond, stage)  # [B, G]（初始 ≡ 0）

        # 3) 任务误差 + 逐假设受限修正 → 加权混合。
        r_ij = g_star.unsqueeze(1) - g_ij  # [B, 4, G]
        delta_ij = torch.stack(
            [
                self.servo.correction(
                    r_ij[:, k],
                    stage_ij[:, k],
                    w=w_unc_ij_list[k] if self.servo.dls else None,
                )
                for k in range(4)
            ],
            dim=1,
        )  # [B, 4, A]
        correction_raw = (w.unsqueeze(-1) * delta_ij).sum(dim=1)  # [B, A]

        # 4) 模型新息 ν（MVP：低秩增益转置近似 J）与 β 信任缩放。
        gain = self.servo.gain()  # [A, G]
        if a_prev is not None and g_prev is not None:
            nu = g - (g_prev + a_prev @ gain)  # Kᵀ·a_prev → [B, G]
        else:
            nu = torch.zeros_like(g)
        nu_norm = nu.norm(dim=-1)  # [B]
        s_nu = (-nu_norm.square() / (2.0 * self.nu_scale**2)).exp()
        H = -(w * (w + 1e-12).log()).sum(dim=-1)  # [B] 假设熵
        s_H = (-torch.relu(H - self.ent_threshold) / self.ent_scale).exp()
        beta = s_nu * s_H * vis_min
        correction = beta[:, None] * correction_raw
        flag = (
            (nu_norm > self.nu_threshold)
            | (H > self.ent_threshold)
            | (vis_min < self.vis_threshold)
        ).to(dtype=proprio.dtype)  # [B] float {0,1}
        return ServoOutput(
            g=g,
            g_star=g_star,
            task_error=g_star - g,
            innovation=nu,
            innovation_flag=flag,
            correction=correction,
            beta=beta,
            stage=stage,
            hyp_weights=w,
            hyp_entropy=H,
        )
