"""C²-IRF v2 Step 2：双新息伺服（va_compound/servo.py + train.py 接入）单元测试。

覆盖（全部 CPU 可跑，对应 Agent E 交付契约）：
- 关系状态组装：g [B,16] 形状 + 零初始化投影下与显式几何逐位一致（组件映射）；
- 可辨识性：ZeroInitReference 初始 g*≡0、|g*|∞≤δ_max；κ 零初始化 → 训练起点
  修正 ≡ 0（直乘与 DLS 两路径）；
- 低秩增益：K=κ·U·Vᵀ 形状/秩 ≤ r、κ 有界；
- 阶段幅度上限：coarse/uncertain 恒 0、approaching/pre-contact/contact 逐级上限；
- --servo-dls 开关：Δa=(KWKᵀ+λI)⁻¹KWr 与手解 4×4 线性系统逐位一致；
- 假设配对混合：w 归一、熵范围、H(w)>τ_H → 重读 flag；新息 flag 阈值；
- 配对共享噪声：sample_flow_matching_inputs_paired（clean/perturbed 共享 (τ,ε)）
  + mix_perturb_batch 布局 [c0,p0,c1,p1,...]；
- --servo-only 冻结断言：VA/flow requires_grad=False，reader/relation 可训练；
- 梯度回传：κ>0 后 loss 反向 → servo 全参数 + reader 参数梯度非空；
- validate_args：--servo 前置约束（--multi-mode 缺失报错）。
"""
from __future__ import annotations

import argparse

import pytest
import torch

from train import (
    _feature_optimizer_groups,
    mix_perturb_batch,
    sample_flow_matching_inputs_paired,
    validate_args,
)
from va_compound.local_control_slots import MultiModeReadout
from va_compound.model import VACompoundConfig, VACompoundPolicy
from va_compound.servo import (
    G_DIM,
    InteractionServo,
    LowRankServo,
    RelationStateProjector,
    ServoOutput,
    STAGE_APPROACHING,
    STAGE_COARSE,
    STAGE_CONTACT,
    STAGE_PRE_CONTACT,
    STAGE_UNCERTAIN,
    ZeroInitReference,
    stage_from_distance,
)


def make_cov(B: int, K: int, device="cpu") -> torch.Tensor:
    """[B, K, 2, 3, 3] 半正定协方差（小随机 + 数值下限）。"""
    A = torch.randn(B, K, 2, 3, 3, device=device) * 0.01
    return A @ A.transpose(-1, -2) + 0.02 * torch.eye(3, device=device)


def make_readout(B: int = 2, K: int = 6, D: int = 64, seed: int = 0) -> MultiModeReadout:
    """随机 MultiModeReadout（mu 归一化 [-1,1]，vis∈[0.5,1]）。"""
    torch.manual_seed(seed)
    mu = torch.rand(B, K, 2, 3) * 2 - 1
    cov = make_cov(B, K)
    vis = torch.rand(B, K) * 0.5 + 0.5
    slots = torch.randn(B, K, 2, D)
    return MultiModeReadout(
        slots=slots, mu=mu, cov=cov, vis=vis, weights=torch.zeros(B, K, 2, 16)
    )


def make_geometry_readout(
    B: int = 2,
    K: int = 6,
    D: int = 64,
    d: float = 0.05,
    vis_value: float = 1.0,
) -> MultiModeReadout:
    """可控几何：tool(mode0) 与 manipuland 两模式距离 d（y,x 平面），vis 统一。"""
    torch.manual_seed(0)
    mu = torch.rand(B, K, 2, 3) * 2 - 1
    mu[:, 0, :, 1:3] = 0.3
    mu[:, 1, :, 1:3] = 0.3 + d
    cov = make_cov(B, K)
    vis = torch.full((B, K), vis_value)
    slots = torch.randn(B, K, 2, D)
    return MultiModeReadout(
        slots=slots, mu=mu, cov=cov, vis=vis, weights=torch.zeros(B, K, 2, 16)
    )


def make_proprio(B: int = 2) -> torch.Tensor:
    return torch.rand(B, 4)


def make_lang(B: int = 2, L: int = 32) -> torch.Tensor:
    return torch.randn(B, L)


def test_relation_state_shape_and_explicit_components():
    """g [B, G] 形状 + 零初始化投影下与显式组装逐位一致（组件映射正确）。"""
    B, K = 3, 6
    readout = make_readout(B, K, seed=1)
    proprio = make_proprio(B)
    proj = RelationStateProjector(relation_dim=G_DIM)
    g = proj(readout.mu, readout.cov, readout.vis, proprio)
    assert g.shape == (B, G_DIM)
    # 零初始化投影 → g ≡ 显式几何（逐组件核对映射）。
    mu, vis = readout.mu, readout.vis
    tr = (
        readout.cov[:, torch.arange(K), 0]
        .diagonal(dim1=-2, dim2=-1)
        .sum(-1)
        / 3.0
    )  # 模式 0 协方差迹代理
    expected = torch.cat(
        [
            mu[:, 0, 0, 1:3] - mu[:, 1, 0, 1:3],   # tool−manip (y,x)
            mu[:, 1, 0, 1:3] - mu[:, 2, 0, 1:3],   # manip−target (y,x)
            mu[:, 0, 0, 1:3] - mu[:, 2, 0, 1:3],   # tool−target (y,x)
            0.5
            * (tr[:, 1].clamp_min(1e-6).log() - tr[:, 2].clamp_min(1e-6).log())[:, None],
            0.5
            * (tr[:, 0].clamp_min(1e-6).log() - tr[:, 1].clamp_min(1e-6).log())[:, None],
            (mu[:, 1, 0, 1:3] - mu[:, 1, 1, 1:3])
            - (mu[:, 2, 0, 1:3] - mu[:, 2, 1, 1:3]),
            (vis[:, 1] * vis[:, 2])[:, None],
            (vis[:, 0] * vis[:, 1])[:, None],
            (mu[:, 1, 0, 0] - mu[:, 2, 0, 0])[:, None],
            proprio[:, 3:4],
            proprio[:, 2:3],
            (mu[:, 0, 0, 1:3] - mu[:, 1, 0, 1:3]).norm(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    assert torch.allclose(g, expected, atol=1e-6)


def test_zero_init_reference_identifiability():
    """g* 零初始化：初始 ≡ 0（对齐关系），且恒有 |g*|∞ ≤ δ_max。"""
    ref = ZeroInitReference(lang_dim=8, relation_dim=G_DIM, delta_max=0.25)
    lang = torch.randn(2, 8)
    phase = torch.tensor([STAGE_CONTACT, STAGE_APPROACHING])
    g_star = ref(lang, phase)
    assert torch.allclose(g_star, torch.zeros_like(g_star), atol=1e-7)
    # 权重扰动后仍受 δ_max 界约束。
    with torch.no_grad():
        ref.proj.weight.normal_(0.0, 1.0)
    g_star = ref(lang, phase)
    assert float(g_star.detach().abs().max()) <= 0.25 + 1e-6
    assert g_star.shape == (2, G_DIM)


def test_servo_correction_zero_at_init():
    """κ 零初始化：训练起点修正 ≡ 0（直乘与 DLS 两路径；任务误差非零也无关）。"""
    readout = make_geometry_readout(d=0.05, vis_value=1.0)
    proprio = make_proprio()
    lang = make_lang()
    for dls in (False, True):
        servo = InteractionServo(vision_dim=64, lang_dim=32, dls=dls)
        out = servo(readout, proprio, lang, a_prev=None, g_prev=None)
        assert isinstance(out, ServoOutput)
        assert torch.allclose(out.correction, torch.zeros_like(out.correction), atol=1e-7)
        assert torch.allclose(out.g_star, torch.zeros_like(out.g_star), atol=1e-7)
        assert out.g.shape == (2, G_DIM)
        assert out.correction.shape == (2, 4)
        assert out.stage.dtype == torch.long


def test_low_rank_gain_rank_and_bound():
    """K = κ·U·Vᵀ：秩 ≤ r、|κ| ≤ κ_max、ρ=0 → K ≡ 0。"""
    servo = LowRankServo(relation_dim=G_DIM, action_dim=4, rank=2, kappa_max=0.25)
    K = servo.gain()
    assert K.shape == (4, G_DIM)
    assert torch.allclose(K, torch.zeros_like(K), atol=1e-7)  # ρ=0 → κ=0
    with torch.no_grad():
        servo.rho.fill_(3.0)
    K = servo.gain()
    rank = int(torch.linalg.matrix_rank(K).item())
    assert rank <= 2
    kappa = 0.25 * torch.tanh(torch.tensor(3.0))
    assert torch.allclose(K, kappa * (servo.U @ servo.V.t()), atol=1e-6)
    assert float(kappa.abs()) < 0.25


def test_stage_from_distance_and_caps():
    """阶段启发式 + 逐阶段幅度上限（coarse/uncertain 恒 0）。"""
    d = torch.tensor([0.9, 0.4, 0.2, 0.05, 0.05])
    vis = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.1])
    stage = stage_from_distance(
        d,
        vis,
        d_coarse=0.5,
        d_approaching=0.3,
        d_pre_contact=0.15,
        vis_threshold=0.3,
    )
    assert stage.tolist() == [
        STAGE_COARSE,
        STAGE_APPROACHING,
        STAGE_PRE_CONTACT,
        STAGE_CONTACT,
        STAGE_UNCERTAIN,
    ]
    # InteractionServo：接触（d=0.05）→ |Δa| ≤ 0.25；coarse（d=0.9）与
    # uncertain（vis 低）→ 修正恒 0。ent_threshold 调高使 s_H=1（ν=0、vis=1
    # 时 β=1），单独验证阶段上限。
    servo = InteractionServo(
        vision_dim=64, lang_dim=32, ent_threshold=10.0
    )
    with torch.no_grad():
        servo.servo.rho.fill_(2.0)  # κ≈κ_max
    proprio = make_proprio()
    lang = make_lang()
    contact = servo(make_geometry_readout(d=0.05, vis_value=1.0), proprio, lang)
    assert float(contact.correction.abs().max()) <= 0.25 + 1e-5
    coarse = servo(make_geometry_readout(d=0.9, vis_value=1.0), proprio, lang)
    assert torch.allclose(coarse.correction, torch.zeros_like(coarse.correction), atol=1e-7)
    assert bool((coarse.stage == STAGE_COARSE).all())
    uncertain = servo(make_geometry_readout(d=0.05, vis_value=0.1), proprio, lang)
    assert torch.allclose(
        uncertain.correction, torch.zeros_like(uncertain.correction), atol=1e-7
    )
    assert bool((uncertain.stage == STAGE_UNCERTAIN).all())


def test_dls_switch_matches_manual_4x4_solve():
    """--servo-dls：Δa = (KWKᵀ+λI)⁻¹KWr 与手解 4×4 线性系统逐位一致。"""
    torch.manual_seed(0)
    B, G = 3, G_DIM
    servo = LowRankServo(
        relation_dim=G, action_dim=4, rank=2, kappa_max=0.25, dls=True, dls_lambda=1e-2
    )
    with torch.no_grad():
        servo.rho.fill_(2.0)
    r = torch.randn(B, G)
    stage = torch.full((B,), STAGE_CONTACT, dtype=torch.long)
    w = torch.rand(B, G) + 0.1
    delta = servo.correction(r, stage, w=w)
    assert delta.shape == (B, 4)
    K = servo.gain()
    KW = K.unsqueeze(0) * w.unsqueeze(1)  # [B, 4, G]
    M = KW @ K.t() + 0.01 * torch.eye(4)
    expected = torch.linalg.solve(M, KW @ r.unsqueeze(-1)).squeeze(-1)
    cap = 0.25
    assert torch.allclose(delta, expected.clamp(-cap, cap), atol=1e-6)
    # 非 dls 模式忽略 w（直乘 K·r 后 clip）。
    direct = LowRankServo(relation_dim=G, action_dim=4, rank=2)
    with torch.no_grad():
        direct.rho.fill_(2.0)
    assert torch.allclose(
        direct.correction(r, stage),
        (r @ direct.gain().t()).clamp(-cap, cap),
        atol=1e-6,
    )


def test_hypothesis_mixing_and_innovation_flag():
    """假设权重归一 + 熵范围；H>τ_H 或 ‖ν‖>τ_ν → innovation_flag=1。"""
    servo = InteractionServo(vision_dim=64, lang_dim=32, nu_threshold=0.15, ent_threshold=0.9)
    readout = make_geometry_readout(d=0.05, vis_value=1.0)
    proprio = make_proprio()
    lang = make_lang()
    out = servo(readout, proprio, lang, a_prev=None, g_prev=None)
    w = out.hyp_weights
    assert w.shape == (2, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-6)
    H = out.hyp_entropy
    assert float(H.min()) >= 0.0 and float(H.max()) <= 1.3863 + 1e-3
    # 新息：g_prev 远离当前 g → ‖ν‖ 大 → flag=1（vis=1、H 可能低）。
    servo_hi = InteractionServo(
        vision_dim=64, lang_dim=32, nu_threshold=0.15, ent_threshold=10.0
    )
    g_prev = out.g + 5.0
    a_prev = torch.randn(2, 4)
    flagged = servo_hi(readout, proprio, lang, a_prev=a_prev, g_prev=g_prev)
    assert bool(flagged.innovation_flag.all().item())
    # 一致状态（g_prev = g）→ ν≈0；随机混合权重下 H≈ln4>0.9 → 默认阈值 flag=1。
    consistent = servo(readout, proprio, lang, a_prev=torch.zeros(2, 4), g_prev=out.g)
    assert bool((consistent.innovation_flag > 0).any().item())  # H(w) 项触发


def test_paired_shared_noise():
    """clean/perturbed 配对行共享同一 (τ, ε)（设计 §六.2）；非配对行独立。"""
    torch.manual_seed(0)
    actions = torch.randn(6, 4, 8, 4)
    is_perturbed = torch.tensor([False, True, False, True, False, False])
    noisy, flow_time, target = sample_flow_matching_inputs_paired(actions, is_perturbed)
    assert noisy.shape == actions.shape
    # 配对行 1↔0、3↔2：共享 τ（flow_time 恒等）与 ε（actions−target ≈ 噪声，
    # 浮点减法往返仅差舍入；target 差异即动作差异，噪声抵消）。
    assert torch.equal(flow_time[1], flow_time[0])
    assert torch.equal(flow_time[3], flow_time[2])
    assert torch.allclose(actions[1] - target[1], actions[0] - target[0], atol=1e-6)
    assert torch.allclose(actions[3] - target[3], actions[2] - target[2], atol=1e-6)
    assert torch.allclose(target[1] - target[0], actions[1] - actions[0], atol=1e-6)
    # 非配对行 4/5：flow_time 相同概率极低（独立采样）。
    assert not torch.equal(flow_time[4], flow_time[3])
    # 无 perturbed 行 → 与默认采样器等价结构（无配对复制）。
    plain = sample_flow_matching_inputs_paired(
        actions, torch.zeros(6, dtype=torch.bool)
    )
    assert plain[0].shape == actions.shape


def test_mix_perturb_batch_layout():
    """混批布局 [c0,p0,c1,p1,...] + is_perturbed 掩码 + 视觉替换。"""
    n_clean, m, T, N, D = 4, 2, 4, 64, 32
    total = n_clean + m
    clean = {
        "vision_tokens": torch.randn(n_clean, T, N, D),
        "vision_tokens_st": torch.randn(n_clean, T, N, D),
        "proprio": torch.randn(n_clean, T, 4),
        "previous_action": torch.randn(n_clean, T, 4),
        "actions": torch.randn(n_clean, T, 8, 4),
        "pair_id": torch.arange(n_clean, dtype=torch.long),
        "instruction_id": torch.zeros(n_clean, dtype=torch.long),
        "coords": torch.randn(n_clean, N, 3),
    }
    perturbed = {
        "vision_tokens": torch.randn(m, T, N, D),
        "proprio": torch.randn(m, T, 4),
        "previous_action": torch.randn(m, T, 4),
        "actions": torch.randn(m, T, 8, 4),
        "pair_id": torch.zeros(m, dtype=torch.long),
        "instruction_id": torch.zeros(m, dtype=torch.long),
    }
    p_vision = torch.randn(m, T, N, D)
    mixed, is_perturbed = mix_perturb_batch(clean, perturbed, p_vision, m)
    assert mixed["actions"].shape == (total, T, 8, 4)
    assert is_perturbed.tolist() == [False, True, False, True, False, False]
    # 配对段：clean 行 0、1 与 perturbed 行交错；tail = clean 行 [m:n_clean]。
    assert torch.equal(mixed["actions"][0], clean["actions"][0])
    assert torch.equal(mixed["actions"][1], perturbed["actions"][0])
    assert torch.equal(mixed["actions"][2], clean["actions"][1])
    assert torch.equal(mixed["actions"][3], perturbed["actions"][1])
    assert torch.equal(mixed["actions"][4], clean["actions"][2])
    assert torch.equal(mixed["actions"][5], clean["actions"][3])
    # 视觉：paired 段 perturbed 行 = p_vision；coords 扩展行数。
    assert torch.equal(mixed["vision_tokens"][1], p_vision[0])
    assert torch.equal(mixed["vision_tokens"][3], p_vision[1])
    assert torch.equal(mixed["vision_tokens"][4], clean["vision_tokens"][2])
    assert torch.equal(mixed["vision_tokens_st"][1], p_vision[0])
    assert mixed["coords"].shape == (total, N, 3)


def _build_multi_mode_policy() -> VACompoundPolicy:
    config = VACompoundConfig(
        language_dim=8,
        vision_dim=64,
        hidden_dim=64,
        action_horizon=4,
        action_dim=4,
        proprio_dim=4,
        num_layers=2,
        num_heads=4,
        local_slots=True,
        multi_mode=True,
        local_slot_tokens=288,
    )
    return VACompoundPolicy(config)


def test_servo_only_freezes_va_flow():
    """--servo-only：VA/flow/入口投影 requires_grad=False，reader/relation 可训练。"""
    model = _build_multi_mode_policy()
    args = argparse.Namespace(
        head_only=False, servo_only=True, lr_slot=None, lr=1e-4
    )
    groups = _feature_optimizer_groups(args, model, None)
    frozen = [
        name
        for name, param in model.named_parameters()
        if not param.requires_grad
    ]
    trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    assert any(name.startswith("flow_head.") for name in frozen)
    assert any(name.startswith("layers.") for name in frozen)
    assert any(name.startswith("vision_projection.") for name in frozen)
    assert all(name.startswith(("role_compiler.", "slot_reader.", "relation_tokens.", "vis_conditioner.")) for name in trainable)
    assert len(groups) == 1
    assert groups[0]["params"]  # 可训练参数非空


def test_gradient_flow_through_servo_and_reader():
    """κ>0 后损失反向：servo 全参数（U/V/ρ/W_g/投影/混合网）与 reader 参数梯度非空。

    用零 token（均匀热图 → 角色读出近距离 → contact 阶段）保证确定性；
    投影残差先离开零点（模拟第一步优化，设计 §八 死区一步后解除）。"""
    torch.manual_seed(0)
    from va_compound.local_control_slots import LocalControlSlotReader

    B, K, D, N = 2, 6, 64, 2 * 12 * 12
    reader = LocalControlSlotReader(
        vision_dim=D, hidden_dim=64, num_slots=K, num_heads=4, multi_mode=True
    )
    coords = torch.rand(N, 3) * 2 - 1
    tokens = torch.zeros(B, N, D)  # 均匀热图 → mu 近距离 → contact 阶段（确定性）
    queries = torch.randn(B, K, 64)
    servo = InteractionServo(vision_dim=D, lang_dim=64, action_dim=4)
    with torch.no_grad():
        servo.servo.rho.fill_(2.0)  # 开启修正路径（初始 κ=0 时 Δa_ij≡0 无梯度）
        # 模拟第一步优化后：投影残差输出离开零点（proj[0] 死区解除）。
        servo.projector.proj[2].weight.normal_(0.0, 0.1)
    readout = reader(tokens, queries, coords)
    proprio = make_proprio(B)
    lang = make_lang(B, L=64)
    out = servo(readout, proprio, lang, a_prev=None, g_prev=None)
    assert bool((out.stage == STAGE_CONTACT).all())  # 确定性接触阶段
    assert torch.allclose(out.g_star, torch.zeros_like(out.g_star), atol=1e-7)
    assert float(out.correction.abs().max()) > 0.0
    loss = (out.correction - torch.ones(B, 4)).pow(2).mean()
    loss.backward()
    named = {name: param for name, param in servo.named_parameters()}
    for name in ("servo.U", "servo.V", "servo.rho", "reference.proj.weight"):
        assert named[name].grad is not None and float(named[name].grad.abs().sum()) > 0, name
    assert float(servo.projector.proj[0].weight.grad.abs().sum()) > 0  # 死区已解除
    assert float(servo.projector.proj[2].weight.grad.abs().sum()) > 0
    assert servo.mixer.net[0].weight.grad is not None
    assert float(servo.mixer.net[0].weight.grad.abs().sum()) > 0
    # reader 可见度路径：vis → g → r → Δa（null_key 参与寻址 logit → vis）。
    reader_grads = [
        param.grad
        for param in reader.parameters()
        if param.grad is not None and float(param.grad.abs().sum()) > 0
    ]
    assert reader_grads, "reader 参数必须通过 g 组装路径收到梯度"
    assert (
        reader.null_key.grad is not None
        and float(reader.null_key.grad.abs().sum()) > 0
    )


def test_servo_validate_args_requires_multi_mode():
    """--servo 前置校验：无 --multi-mode 报错；--servo-dls 无 --servo 报错。"""
    from train import parse_args

    # 真实 parser 默认值打底，避免每次 validate_args 新读一个字段就要手工补夹具。
    base = vars(parse_args([]))
    base.update(
        steps=1, flow_steps=8, lr=1e-4, pair_loss_weight=0.0, batch_size=2,
        single_task=True, mode="bidir_va", attention_variant="flat",
        action_query_cond=False, memory_split=False, evidence_tokens=16,
        task_tokens=8, future_predict=False, sequential_coupling=0,
        flow_cond="entry", flow_layers=2, evsm=False, plan_resampler=False,
        scene_teacher=False, direct_head=False, c2_controller=False,
        role_query=False, role_query_tokens=16, dual_attention=False,
        flow_semantic=False, compile_task=False, compile_every=4,
        compile_n_scene=16, compile_n_readout=16, semantic_adapter=False,
        semantic_lora_rank=8, semantic_top_layers=4, semantic_anchor_weight=0.0,
        semantic_geometry_weight=0.0, semantic_anchor_layers="",
        semantic_act_grad_scale=0.1, semantic_lora_suffixes="q_proj",
        language_max_length=64, vision_pooling="flat", num_workers=0,
        fork_data=None, fork_k=83, fork_skip_contract=False,
        sequence_length=4, min_sequence_length=4, pair_start_atol=0.0,
        pair_start_cosine=0.0, min_pair_action_delta=1e-3,
        pair_probe_tau_max=0.5, pair_mode="shared_cf", va_layers=2,
        lr_slot=None, lr_va=None, head_only=False, prev_dropout=0.0,
        sam_rho=0.0, seed=0, resume=None, device="cpu", save=None,
        save_every=0, data=None, live_vjepa=False, live_root="",
        control_stride=6, sequences_per_episode=4, phase_bins=0,
        phase_seed=0, success_only=False, sliding_window=False,
        frame_aug=False, lr_vision=3e-6, vision_unfreeze_last=0,
        vision_unfreeze_all=False, e2e_data=None, lora_rank=0, lora_alpha=32.0,
        unfreeze_blocks=None, qwen_unfreeze_blocks=0, qwen_lr=1e-5,
        lora_lr=1e-4, vision_lr=1e-5, language_dtype="bfloat16",
        vision_dtype="bfloat16", e2e_pooling="flat", local_slots_data=None,
        dense_readout=False, multi_mode=False, role_seeds=None,
        local_slots_direct288=False, lang_fixed_vector=False,
        local_slots_fixed_query=False, c2_v6a="", c2_v6b="",
        c2_lambda_f=0.1, c2_lambda_r=1.0, c2_lambda_c=0.0,
        c2_recovery_ratio=0.25, c2_unfreeze_stage_a=False,
        c2_contract_every=500, c2_contract_rho6=0.8, qk_norm=False,
        training_stage=None, lr_servo=None, perturb_data=None,
        servo_perturb_ratio=0.5,
    )
    base.pop("multi_mode")
    base.pop("local_slots_data")
    for _key in ("servo", "servo_only", "servo_dls", "servo_rank", "servo_lambda"):
        base.pop(_key, None)
    args = argparse.Namespace(**base, servo=True, servo_only=False,
                              servo_dls=False, servo_rank=2, servo_lambda=1e-2,
                              multi_mode=False, local_slots_data=None)
    with pytest.raises(ValueError, match="--servo requires --multi-mode"):
        validate_args(args)
    args = argparse.Namespace(**base, servo=False, servo_only=False,
                              servo_dls=True, servo_rank=2, servo_lambda=1e-2,
                              multi_mode=False, local_slots_data=None)
    with pytest.raises(ValueError, match="--servo-dls requires --servo"):
        validate_args(args)
    # --servo-only 隐含启用 --servo（合法路径：multi_mode + local_slots_data）。
    args = argparse.Namespace(
        **base,
        servo=False,
        servo_only=True,
        servo_dls=False,
        servo_rank=2,
        servo_lambda=1e-2,
        multi_mode=True,
        local_slots_data="data/x.pt",
    )
    validate_args(args)
    assert args.servo
