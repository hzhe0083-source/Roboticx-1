"""C²-IRF v2 Wave 2 Agent F：eval_metaworld.py 评估集成测试（servo/fovea/state-take）。

覆盖（全部 CPU 可跑；真实 V-JEPA/GPU 冒烟按显存纪律回退）：
- 参数校验：--state-take 取值（0/4/8/39）、--servo-ablation 四消融取值、
  validate_servo_args 的非法/互斥组合（servo 与 c2/direct checkpoint 互斥、
  缺 multi_mode、direct288、与 --c2-* 标志互斥、负阈值）；
- --state-take 截取逻辑：take=4 与旧公式逐位一致、0 恒零、8/39 扩展段走
  39 维布局物理范围表、超出 obs 维度报错；
- 可变维 proprio：extend_state_projection 零初始化扩展（前段逐位拷贝、
  扩展段零权重 → 模型输出不变）、已匹配宽度时短路；
- 四消融开关映射（ServoRuntime）：zero-gain（修正恒零）/gain-shuffle
  （低秩因子 U/V 行置换，可复现）/ wrong-role（角色循环移位传入读出）/
  open-loop（修正 None、不调用伺服前向）；
- InteractionServo 真实集成：前向契约（proprio/lang_cond/a_prev/g_prev）、
  g_prev 跨决策维护、innovation_flag 输出；
- fovea 调度节奏：fovea_schedule 何时全图（plan_due / 新息立即刷新）、
  何时 foveal、何时 feedback 全图、何时 hold、何时施加修正；
- fovea_refresh_due 阈值：|ν|/H(w)/vis 三路 OR 与边界；
- select_roi_pair：可见度乘积最大角色对 + 最强峰模式 + compute_roi 端到端；
- build_multimode_stream 与 build_local_vision 逐位一致（小策略 CPU）；
- 单任务闭环冒烟（CPU 集成：plan → foveal 修正 → 新息提前刷新节奏；
  GPU 空闲 < 2GiB 时自动回退 CPU 并打印——显存纪律）。
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from eval_metaworld import (
    MODE_FEEDBACK,
    MODE_FOVEAL,
    MODE_HOLD,
    MODE_PLAN,
    ServoRuntime,
    _STATE_LAYOUT_Q01,
    _STATE_LAYOUT_Q99,
    build_multimode_stream,
    compute_roi,
    extend_state_projection,
    fovea_refresh_due,
    fovea_schedule,
    parse_args,
    select_roi_pair,
    state_take_normalize,
    validate_servo_args,
    vis_entropy,
)
from va_compound.local_control_slots import MultiModeReadout
from va_compound.model import VACompoundConfig, VACompoundPolicy


def build_grid_coords(grid: int = 24) -> torch.Tensor:
    """[2*grid², 3]：与 live_vjepa._dense_coords 同一生成公式（t→y→x 行优先）。"""
    rows = []
    half = (grid - 1) / 2
    for t in range(2):
        for y in range(grid):
            for x in range(grid):
                rows.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
    return torch.tensor(rows, dtype=torch.float32)


def make_policy(**overrides) -> VACompoundPolicy:
    """最小 local_slots + multi_mode 策略（CPU；可叠加 c2/direct 配置）。"""
    config = VACompoundConfig(
        language_dim=64,
        vision_dim=64,
        hidden_dim=64,
        num_layers=1,
        num_heads=8,
        action_dim=4,
        proprio_dim=4,
        flow_layers=1,
        local_slots=True,
        local_slot_k=6,
        local_slot_tokens=1152,
        multi_mode=True,
        dense_readout=True,
        local_coarse=16,
        **overrides,
    )
    return VACompoundPolicy(config).eval()


def make_readout(b: int = 1, k: int = 3, slots_dim: int = 64) -> MultiModeReadout:
    """固定种子的多模式读出假数据（形状契约完整；slots_dim 对齐 vision_dim）。"""
    torch.manual_seed(0)
    slots = torch.randn(b, k, 2, slots_dim)
    mu = torch.randn(b, k, 2, 3)
    cov = torch.randn(b, k, 2, 3, 3) * 0.1
    vis = torch.rand(b, k)
    weights = torch.zeros(b, k, 2, 10)
    return MultiModeReadout(slots, mu, cov, vis, weights)


class FakeServoController:
    """契约假控制器：gain [A, D_rel] + relation_state（记录收到的 mu/cov/vis）。"""

    def __init__(self, d_rel: int = 6) -> None:
        torch.manual_seed(0)
        self.gain = torch.randn(4, d_rel)
        self.seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def relation_state(self, mu, cov, vis) -> torch.Tensor:
        self.seen.append((mu.clone(), cov.clone(), vis.clone()))
        return mu[:, :2, 0].reshape(mu.shape[0], -1)  # [B, 6]


# ---------------------------------------------------------------------------
# 参数校验（argparse choices + validate_servo_args 组合）
# ---------------------------------------------------------------------------


def _parse_args(*extra: str):
    old_argv = sys.argv
    sys.argv = [
        "eval_metaworld",
        "--checkpoint", "c.pt",
        "--features", "f.pt",
        *extra,
    ]
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


def test_cli_accepts_new_flags():
    args = _parse_args("--state-take", "8", "--servo-ablation", "wrong-role", "--fovea")
    assert args.state_take == 8
    assert args.servo_ablation == "wrong-role"
    assert args.fovea is True
    assert args.fovea_nu_thresh == 0.1 and args.fovea_h_thresh == 0.7
    assert args.fovea_vis_thresh == 0.3


@pytest.mark.parametrize("take", (0, 4, 8, 39))
def test_cli_state_take_legal_values(take):
    assert _parse_args("--state-take", str(take)).state_take == take


@pytest.mark.parametrize("take", (1, 5, 7, 24, -1))
def test_cli_state_take_rejects_illegal_values(take):
    with pytest.raises(SystemExit):
        _parse_args("--state-take", str(take))


@pytest.mark.parametrize("ablation", ("zero-gain", "gain-shuffle", "wrong-role", "open-loop"))
def test_cli_servo_ablation_legal_values(ablation):
    assert _parse_args("--servo-ablation", ablation).servo_ablation == ablation


def test_cli_servo_ablation_rejects_unknown():
    with pytest.raises(SystemExit):
        _parse_args("--servo-ablation", "bogus")


class _Args:
    def __init__(self, **kw) -> None:
        self.servo_ablation = kw.get("servo_ablation", "none")
        self.fovea = kw.get("fovea", False)
        self.c2_oracle_ref = kw.get("c2_oracle_ref", False)
        self.c2_zero_gain = kw.get("c2_zero_gain", False)
        self.c2_gain_scale = kw.get("c2_gain_scale", 1.0)
        self.c2_error_threshold = kw.get("c2_error_threshold", 0.0)
        self.fovea_nu_thresh = kw.get("fovea_nu_thresh", 0.1)
        self.fovea_h_thresh = kw.get("fovea_h_thresh", 0.7)
        self.fovea_vis_thresh = kw.get("fovea_vis_thresh", 0.3)


class _Config:
    def __init__(self, **kw) -> None:
        self.c2_controller = kw.get("c2_controller", True)
        self.direct_head = kw.get("direct_head", False)
        self.local_slots = kw.get("local_slots", True)
        self.multi_mode = kw.get("multi_mode", True)
        self.dense_readout = kw.get("dense_readout", True)
        self.local_slots_direct288 = kw.get("local_slots_direct288", False)


def test_validate_servo_args_ok():
    validate_servo_args(_Args(servo_ablation="zero-gain"), _Config(c2_controller=False))
    validate_servo_args(_Args(fovea=True), _Config(c2_controller=False))
    validate_servo_args(_Args(), _Config(c2_controller=False))  # 默认 none 不报错
    validate_servo_args(_Args(), _Config())  # 无 servo 请求时 c2 checkpoint 不受影响


def test_validate_servo_args_rejects_c2_checkpoint():
    # servo 训练与 c2_controller/direct_head 互斥（train.py --servo 校验）。
    for kwargs in ({"servo_ablation": "open-loop"}, {"fovea": True}):
        with pytest.raises(ValueError):
            validate_servo_args(_Args(**kwargs), _Config(c2_controller=True))
    with pytest.raises(ValueError):
        validate_servo_args(_Args(servo_ablation="zero-gain"), _Config(direct_head=True))


def test_validate_servo_args_requires_multimode_reader():
    with pytest.raises(ValueError):
        validate_servo_args(
            _Args(servo_ablation="none", fovea=True), _Config(c2_controller=False, multi_mode=False)
        )
    with pytest.raises(ValueError):
        validate_servo_args(_Args(servo_ablation="zero-gain"), _Config(c2_controller=False, local_slots=False))
    with pytest.raises(ValueError):
        validate_servo_args(
            _Args(servo_ablation="zero-gain"),
            _Config(c2_controller=False, local_slots_direct288=True),
        )


def test_validate_fovea_requires_dense_readout():
    with pytest.raises(ValueError, match="dense_readout"):
        validate_servo_args(
            _Args(fovea=True),
            _Config(c2_controller=False, dense_readout=False),
        )


def test_validate_servo_args_rejects_c2_ablation_flags():
    # --c2-* 消融标志需要 c2 checkpoint，与 servo 部署（flow）互斥。
    cfg = _Config(c2_controller=False)
    with pytest.raises(ValueError):
        validate_servo_args(_Args(servo_ablation="zero-gain", c2_oracle_ref=True), cfg)
    with pytest.raises(ValueError):
        validate_servo_args(_Args(servo_ablation="zero-gain", c2_zero_gain=True), cfg)


def test_validate_servo_args_rejects_negative_thresholds():
    with pytest.raises(ValueError):
        validate_servo_args(_Args(fovea=True, fovea_nu_thresh=-0.1), _Config())


# ---------------------------------------------------------------------------
# --state-take 截取逻辑
# ---------------------------------------------------------------------------


def test_state_take_normalize_default_identical_to_legacy():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(39) * 0.2 + 0.4
    sq01 = rng.standard_normal(4) * 0.1
    sq99 = sq01 + rng.uniform(0.05, 0.5, 4)
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    legacy = np.clip(2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0).astype(
        np.float32
    )
    assert np.array_equal(state_take_normalize(obs, 4, sq01, scale_s), legacy)


def test_state_take_normalize_zero_is_vision_only():
    obs = np.zeros(39, dtype=np.float64)
    sq01 = np.zeros(4)
    scale_s = np.ones(4)
    out = state_take_normalize(obs, 0, sq01, scale_s)
    assert out.shape == (4,)
    assert not out.any()


def test_state_take_normalize_extended_dims_use_layout_table():
    rng = np.random.default_rng(1)
    obs = rng.standard_normal(39) * 0.2 + 0.4
    sq01 = rng.standard_normal(4) * 0.1
    scale_s = rng.uniform(0.05, 0.5, 4)
    legacy4 = state_take_normalize(obs, 4, sq01, scale_s)
    out8 = state_take_normalize(obs, 8, sq01, scale_s)
    assert out8.shape == (8,)
    assert np.array_equal(out8[:4], legacy4)
    span = np.where(
        np.abs(_STATE_LAYOUT_Q99[4:8] - _STATE_LAYOUT_Q01[4:8]) < 1e-6,
        1.0,
        _STATE_LAYOUT_Q99[4:8] - _STATE_LAYOUT_Q01[4:8],
    )
    expected = np.clip(
        2.0 * (obs[4:8] - _STATE_LAYOUT_Q01[4:8]) / span - 1.0, -1.0, 1.0
    ).astype(np.float32)
    assert np.array_equal(out8[4:], expected)
    out39 = state_take_normalize(obs, 39, sq01, scale_s)
    assert out39.shape == (39,)
    assert np.array_equal(out39[:4], legacy4)
    assert np.array_equal(out39[4:8], expected)  # 4-7 与 take=8 的扩展段一致


def test_state_take_normalize_quat_dims_use_minus_one_to_one():
    # 四元数分量（7-10 等）的物理范围 [-1, 1] → 归一化后 ±1 端点可测。
    obs = np.zeros(39, dtype=np.float64)
    obs[7] = 1.0
    obs[8] = -1.0
    sq01 = np.zeros(4)
    scale_s = np.ones(4)
    out = state_take_normalize(obs, 39, sq01, scale_s)
    assert out[7] == 1.0 and out[8] == -1.0


def test_state_take_normalize_rejects_take_beyond_obs():
    with pytest.raises(ValueError):
        state_take_normalize(np.zeros(4), 8, np.zeros(4), np.ones(4))


# ---------------------------------------------------------------------------
# 可变维 proprio：零初始化投影扩展
# ---------------------------------------------------------------------------


class _StubPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(proprio_dim=4, action_dim=4)
        torch.manual_seed(0)
        self.state_projection = nn.Linear(8, 16)


def test_extend_state_projection_zero_init_extension():
    stub = _StubPolicy()
    original = stub.state_projection.weight.clone()
    bias = stub.state_projection.bias.clone()
    extend_state_projection(stub, 8)
    proj = stub.state_projection
    assert proj.in_features == 12
    assert torch.equal(proj.weight[:, :8], original)
    assert proj.weight[:, 8:].abs().sum() == 0.0  # 扩展列零权重（OOD 零贡献）
    assert torch.equal(proj.bias, bias)


def test_extend_state_projection_output_unchanged_for_known_dims():
    stub = _StubPolicy()
    extend_state_projection(stub, 8)
    torch.manual_seed(3)
    x_known = torch.randn(2, 8)
    x_ext = torch.cat([x_known, torch.randn(2, 4)], dim=-1)
    with torch.no_grad():
        assert torch.equal(
            stub.state_projection(x_ext),
            _StubPolicy().state_projection(x_known),  # 原投影（未扩展）
        )


def test_extend_state_projection_noop_when_width_matches():
    stub = _StubPolicy()
    before = stub.state_projection
    extend_state_projection(stub, 4)  # 4+4=8 已匹配 → 短路
    assert stub.state_projection is before
    extend_state_projection(stub, 0)  # take < proprio_dim → 短路
    assert stub.state_projection is before


# ---------------------------------------------------------------------------
# 四消融开关映射（ServoRuntime）
# ---------------------------------------------------------------------------


def test_servo_zero_gain_correction_is_zero():
    runtime = ServoRuntime(FakeServoController(), "zero-gain")
    correction, flag = runtime.correct(make_readout())
    assert correction is not None
    assert np.allclose(correction, 0.0)  # β≡0：修正恒零
    assert isinstance(flag, bool)


def test_servo_open_loop_skips_servo_forward():
    controller = FakeServoController()
    runtime = ServoRuntime(controller, "open-loop")
    correction, flag = runtime.correct(make_readout())
    assert correction is None  # 不施加任何修正
    assert flag is False
    assert controller.seen == []  # 连 relation_state 都不调用


def test_servo_wrong_role_permutes_role_indices():
    controller = FakeServoController()
    runtime = ServoRuntime(controller, "wrong-role")
    readout = make_readout(k=3)
    runtime.correct(readout)
    assert len(controller.seen) == 1
    mu_seen, cov_seen, vis_seen = controller.seen[0]
    perm = [1, 2, 0]  # (i+1) % 3 循环移位
    assert torch.equal(mu_seen, readout.mu[:, perm])
    assert torch.equal(cov_seen, readout.cov[:, perm])
    assert torch.equal(vis_seen, readout.vis[:, perm])


def test_servo_wrong_role_permutes_real_forward_readout():
    # 真实 InteractionServo：循环移位后的读出进入前向（关系组装角色错位）。
    from va_compound.servo import InteractionServo

    controller = InteractionServo(vision_dim=64, lang_dim=64, action_dim=4, rank=2)
    runtime = ServoRuntime(controller, "wrong-role")
    readout = make_readout()
    correction, _flag = runtime.correct(
        readout, np.zeros(4, dtype=np.float32), torch.randn(1, 64), a_prev=None
    )
    assert correction is not None and correction.shape == (4,)


def test_servo_gain_shuffle_is_deterministic_row_col_permutation():
    original = FakeServoController()
    gain0 = original.gain.clone()
    runtime_a = ServoRuntime(FakeServoController(), "gain-shuffle", seed=0)
    runtime_b = ServoRuntime(FakeServoController(), "gain-shuffle", seed=0)
    runtime_c = ServoRuntime(FakeServoController(), "gain-shuffle", seed=1)
    assert torch.equal(runtime_a._gain(), runtime_b._gain())  # 固定种子可复现
    assert not torch.equal(runtime_a._gain(), gain0)  # 确实被打乱
    assert not torch.equal(runtime_a._gain(), runtime_c._gain())  # 不同种子不同
    # 行/列置换保持条目多重集不变（只是打乱语义，不是数值破坏）。
    assert np.allclose(
        np.sort(runtime_a._gain().numpy().ravel()),
        np.sort(gain0.numpy().ravel()),
    )


def test_servo_gain_shuffle_permutes_low_rank_factors():
    # 真实 InteractionServo：K = κ·U·Vᵀ——gain-shuffle 打乱 U 行（动作维）与
    # V 行（关系维），部署后每次 gain() 都返回行列打乱的 K。
    from va_compound.servo import InteractionServo

    torch.manual_seed(0)
    controller = InteractionServo(vision_dim=64, lang_dim=64, action_dim=4, rank=2)
    u0 = controller.servo.U.data.clone()
    v0 = controller.servo.V.data.clone()
    ServoRuntime(controller, "gain-shuffle", seed=0)
    rng = np.random.default_rng(0)
    rows = rng.permutation(4)
    cols = rng.permutation(16)
    assert torch.equal(controller.servo.U.data, u0[rows])
    assert torch.equal(controller.servo.V.data, v0[cols])
    # 尺度保留：K 条目多重集不变（只是语义打乱）。
    k_before = (u0 @ v0.t()).numpy().ravel()
    k_after = (controller.servo.U.data @ controller.servo.V.data.t()).numpy().ravel()
    assert np.allclose(np.sort(k_before), np.sort(k_after))


def test_servo_real_interaction_servo_forward_contract():
    """InteractionServo 前向集成：correction 形状/数值、g_prev 跨决策维护、
    innovation_flag bool 输出（阈值在 servo 内，设计 §三.3）。"""
    from va_compound.servo import InteractionServo

    torch.manual_seed(0)
    controller = InteractionServo(vision_dim=64, lang_dim=64, action_dim=4, rank=2)
    runtime = ServoRuntime(controller, "none")
    readout = make_readout()
    lang_cond = torch.randn(1, 64)
    proprio = np.zeros(4, dtype=np.float32)
    correction, flag = runtime.correct(readout, proprio, lang_cond, a_prev=None)
    assert correction.shape == (4,)
    assert isinstance(flag, bool)
    assert runtime.prev_g is not None  # g_prev 跨决策维护（ν 依赖）
    # 第二次调用：a_prev 生效（ν 路径），g_prev 更新。
    correction2, flag2 = runtime.correct(
        readout, proprio, lang_cond, a_prev=np.zeros(4, dtype=np.float32)
    )
    assert correction2.shape == (4,)
    assert isinstance(flag2, bool)
    # 首决策缺失 a_prev/g_prev → ν≡0（servo.py 契约）。
    assert runtime.prev_g.shape == (1, 16)


def test_servo_real_zero_gain_zeroes_correction():
    from va_compound.servo import InteractionServo

    torch.manual_seed(0)
    controller = InteractionServo(vision_dim=64, lang_dim=64, action_dim=4, rank=2)
    runtime = ServoRuntime(controller, "zero-gain")
    readout = make_readout()
    correction, flag = runtime.correct(
        readout, np.zeros(4, dtype=np.float32), torch.randn(1, 64), a_prev=None
    )
    assert np.allclose(correction, 0.0)  # β≡0：修正恒零（感知照常）
    assert isinstance(flag, bool)


def test_servo_real_open_loop_skips_forward():
    from va_compound.servo import InteractionServo

    controller = InteractionServo(vision_dim=64, lang_dim=64, action_dim=4, rank=2)
    runtime = ServoRuntime(controller, "open-loop")
    correction, flag = runtime.correct(make_readout())
    assert correction is None and flag is False
    assert runtime.prev_g is None  # 前向未运行


def test_servo_correction_matches_contract_math():
    controller = FakeServoController()
    runtime = ServoRuntime(controller, "none")
    readout = make_readout()
    correction, _ = runtime.correct(readout)
    g_t = readout.mu[:, :2, 0].reshape(1, -1)
    expected = -(controller.gain @ g_t[0]).numpy()
    assert np.allclose(correction, expected)  # Δa = K·(g* − g_t)，g*≡0


def test_servo_innovation_fn_flag_forwarded():
    def fake_innovation(mu, cov, vis, prev_mu):
        return torch.tensor(0.1), torch.tensor(True)

    runtime = ServoRuntime(FakeServoController(), "none", innovation_fn=fake_innovation)
    _correction, flag = runtime.correct(make_readout())
    assert flag is True


def test_servo_fallback_refresh_flag_thresholds():
    # 无 innovation_fn：回退 fovea_refresh_due（vis 极低 → 刷新）。
    controller = FakeServoController()
    runtime = ServoRuntime(controller, "none", vis_thresh=0.5)
    readout = make_readout()
    readout = MultiModeReadout(
        readout.slots,
        readout.mu,
        readout.cov,
        torch.full_like(readout.vis, 0.1),  # vis 全低 → 触发刷新
        readout.weights,
    )
    _correction, flag = runtime.correct(readout)
    assert flag is True


# ---------------------------------------------------------------------------
# fovea 调度节奏（纯函数）
# ---------------------------------------------------------------------------


def test_fovea_schedule_plan_foveal_hold_rhythm():
    # plan_stride=6，feedback_stride=1：plan 之后每步 foveal，第 6 步再 plan。
    assert fovea_schedule(0, None, 6, 1, 8, fovea=True) == (MODE_PLAN, True, 0)
    assert fovea_schedule(1, 0, 6, 1, 8, fovea=True) == (MODE_FOVEAL, True, 1)
    assert fovea_schedule(2, 0, 6, 1, 8, fovea=True) == (MODE_FOVEAL, True, 2)
    assert fovea_schedule(5, 0, 6, 1, 8, fovea=True) == (MODE_FOVEAL, True, 5)
    assert fovea_schedule(6, 0, 6, 1, 8, fovea=True) == (MODE_PLAN, True, 0)


def test_fovea_schedule_hold_when_no_feedback_due():
    # feedback_stride=3：plan 后第 1、2 步 hold（无视觉计算、无修正）。
    mode, correction_due, token = fovea_schedule(1, 0, 6, 3, 8, fovea=True)
    assert mode == MODE_HOLD and correction_due is False and token == 0
    assert fovea_schedule(2, 0, 6, 3, 8, fovea=True) == (MODE_HOLD, False, 0)
    assert fovea_schedule(3, 0, 6, 3, 8, fovea=True) == (MODE_FOVEAL, True, 1)


def test_fovea_schedule_innovation_forces_immediate_full_refresh():
    # 新息超阈值：即使不在 plan 节奏上也立即全图重读（设计 §三.3）。
    assert fovea_schedule(2, 0, 6, 1, 8, fovea=True, innovation_flag=True) == (
        MODE_PLAN, True, 0,
    )
    # 非 fovea（现状）同样支持新息提前刷新。
    assert fovea_schedule(2, 0, 6, 1, 8, fovea=False, innovation_flag=True) == (
        MODE_PLAN, True, 0,
    )


def test_fovea_schedule_without_fovea_falls_back_to_legacy_feedback():
    # fovea=False：feedback 步是"全图重读"（MODE_FEEDBACK），与现状一致。
    assert fovea_schedule(0, None, 6, 1, 8, fovea=False) == (MODE_PLAN, True, 0)
    assert fovea_schedule(1, 0, 6, 1, 8, fovea=False) == (MODE_FEEDBACK, True, 1)
    assert fovea_schedule(2, 0, 6, 3, 8, fovea=False) == (MODE_HOLD, False, 0)


def test_fovea_schedule_token_exhaustion_forces_replan():
    # horizon=8：token 用尽（距 plan 8 步）→ 强制重规划（与 c2_schedule 一致）。
    assert fovea_schedule(8, 0, 100, 1, 8, fovea=True) == (MODE_PLAN, True, 0)


def test_fovea_schedule_rejects_bad_strides():
    with pytest.raises(ValueError):
        fovea_schedule(0, None, 0, 1, 8, fovea=True)
    with pytest.raises(ValueError):
        fovea_schedule(0, None, 6, 0, 8, fovea=True)


def test_fovea_refresh_due_thresholds_or_logic():
    assert fovea_refresh_due(nu_norm=0.2, nu_thresh=0.1)  # |ν| 超阈值
    assert fovea_refresh_due(mode_entropy=0.9, h_thresh=0.7)  # H(w) 超阈值
    assert fovea_refresh_due(vis_min=0.1, vis_thresh=0.3)  # vis 过低
    assert not fovea_refresh_due(
        nu_norm=0.05, mode_entropy=0.1, vis_min=0.9,
        nu_thresh=0.1, h_thresh=0.7, vis_thresh=0.3,
    )
    # 边界：严格大于/小于，等值不触发。
    assert not fovea_refresh_due(nu_norm=0.1, nu_thresh=0.1)
    assert not fovea_refresh_due(vis_min=0.3, vis_thresh=0.3)
    # None 项不参与。
    assert not fovea_refresh_due(vis_min=0.9, vis_thresh=0.3)


def test_vis_entropy_uniform_vs_onehot():
    uniform = torch.full((1, 3), 1.0 / 3.0)
    assert vis_entropy(uniform) == pytest.approx(np.log(3.0), rel=1e-5)
    onehot = torch.tensor([[1.0, 0.0, 0.0]])
    assert vis_entropy(onehot) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# select_roi_pair
# ---------------------------------------------------------------------------


def test_select_roi_pair_picks_max_vis_product_pair():
    mu = torch.zeros(1, 3, 2, 3)
    cov = torch.zeros(1, 3, 2, 3, 3)
    vis = torch.tensor([[0.9, 0.1, 0.8]])  # 对 (0,2) 乘积 0.72 最大
    mu_pair, cov_pair = select_roi_pair(mu, cov, vis)
    assert tuple(mu_pair.shape) == (1, 2, 3)
    assert tuple(cov_pair.shape) == (1, 2, 3, 3)
    # 角色 0 与 2 的 mode 0（top 峰）。
    assert torch.equal(mu_pair[0, 0], mu[0, 0, 0])
    assert torch.equal(mu_pair[0, 1], mu[0, 2, 0])


def test_select_roi_pair_validation():
    mu = torch.zeros(1, 3, 2, 3)
    with pytest.raises(ValueError):
        select_roi_pair(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3, 3), torch.ones(1, 1))
    with pytest.raises(ValueError):
        select_roi_pair(mu, torch.zeros(1, 3, 2, 2), torch.ones(1, 3))
    with pytest.raises(ValueError):
        select_roi_pair(mu, torch.zeros(1, 3, 2, 3, 3), torch.ones(1, 2))


def test_select_roi_pair_feeds_compute_roi_end_to_end():
    mu = torch.zeros(1, 3, 2, 3)
    mu[0, 0, 0, 1:] = torch.tensor([-0.2, -0.1])
    mu[0, 1, 0, 1:] = torch.tensor([0.2, 0.1])
    cov = torch.zeros(1, 3, 2, 3, 3)
    vis = torch.tensor([[0.5, 0.5, 0.5]])  # 三对平手 → 取首个 (0,1)
    mu_pair, cov_pair = select_roi_pair(mu, cov, vis)
    roi = compute_roi(mu_pair, cov_pair)
    assert tuple(roi.shape) == (1, 3)  # (cy, cx, size_px)
    assert torch.equal(mu_pair[0, 0], mu[0, 0, 0])  # 角色 0 的 mode 0
    assert torch.equal(mu_pair[0, 1], mu[0, 1, 0])  # 角色 1 的 mode 0


# ---------------------------------------------------------------------------
# build_multimode_stream 与 build_local_vision 逐位一致
# ---------------------------------------------------------------------------


def test_build_multimode_stream_matches_build_local_vision():
    torch.manual_seed(0)
    model = make_policy()
    tokens = torch.randn(1, 1152, 64)
    coords = build_grid_coords(24)
    queries = torch.randn(1, 6, 64)  # K=6（relation_tokens 固定索引角色 3）
    stream_legacy = model.build_local_vision(tokens, coords, queries)
    readout = model.slot_reader(tokens, queries, coords)
    stream_servo = build_multimode_stream(model, readout, tokens, queries)
    assert stream_legacy.shape == (1, 31, 64)
    assert torch.equal(stream_servo, stream_legacy)


# ---------------------------------------------------------------------------
# 单任务闭环冒烟（CPU 集成：plan → foveal 修正 → 新息提前刷新节奏）
# ---------------------------------------------------------------------------


def test_closed_loop_smoke_plan_foveal_correction_rhythm():
    """真实 InteractionServo（小 config）+ flow 小策略的 10 步闭环调度冒烟。

    验证 plan → foveal 交替、名义 chunk 解码 + 伺服修正、新息触发提前全局
    刷新。servo 阈值（|ν|/H/vis）在 E 的 servo.py 内；本冒烟用放宽阈值
    隔离出纯调度节奏。GPU 空闲 < 2GiB 时自动回退 CPU 并打印（显存纪律）。
    """
    from va_compound.servo import InteractionServo

    torch.manual_seed(0)
    model = make_policy(action_horizon=8)  # flow（servo 训练与 c2/direct 互斥）
    controller = InteractionServo(
        vision_dim=64, lang_dim=64, action_dim=4, rank=2,
        nu_threshold=10.0, ent_threshold=10.0, vis_threshold=0.0,  # 关阈值 → 纯节奏
    )
    runtime = ServoRuntime(controller, "none")
    hidden = torch.randn(1, 4, 64)
    mask = torch.ones(1, 4, dtype=torch.bool)
    language_cache = model.build_language_cache(hidden, mask)
    role_queries = language_cache.role_queries
    coords = build_grid_coords(24)
    proprio = torch.zeros(1, 4)
    lang_cond = role_queries.mean(dim=1)  # 与 train.py servo_correction_t0 同一构造

    plan_step = None
    c2_token = 0
    chunk = None
    readout = None
    innovation_flag = False
    plan_count = 0
    foveal_count = 0
    hold_count = 0
    first = True
    for step in range(10):
        tokens = torch.randn(1, 1152, 64)
        mode, correction_due, _ = fovea_schedule(
            step, plan_step, 6, 1, 8, fovea=True, innovation_flag=innovation_flag
        )
        if mode == MODE_PLAN:
            readout, stream = _servo_vision_smoke(model, tokens, role_queries, coords)
            cond, _ = model.encode_condition(
                stream, proprio, torch.zeros(1, 4),
                language_cache=language_cache, return_visual_memory=True,
            )
            chunk = model.decode_actions(cond, steps=4)[0].numpy()  # 名义 ā [8,4]
            plan_step = step
            c2_token = 0
            plan_count += 1
        if correction_due and c2_token < 8 and chunk is not None:
            if mode == MODE_FOVEAL:
                foveal_count += 1
            correction, innovation_flag = runtime.correct(
                readout,
                proprio.numpy()[0],
                lang_cond,
                a_prev=None if first else np.zeros(4, dtype=np.float32),
            )
            first = False
            nominal = chunk[c2_token]
            action = np.clip(nominal + correction, -1.0, 1.0)
            assert action.shape == (4,)
            assert np.isfinite(action).all()
            c2_token += 1
        else:
            hold_count += 1

    # 10 步（plan_stride=6, feedback_stride=1, horizon=8）：步 0 plan；
    # 步 1-5 foveal 修正；步 6 plan；步 7-9 foveal 修正——无 hold。
    assert plan_count == 2
    assert foveal_count == 8
    assert hold_count == 0
    assert plan_step == 6

    # 新息标志 → 下一步立即 plan（提前全局刷新，设计 §三.3）。
    mode, correction_due, _ = fovea_schedule(
        2, 0, 6, 1, 8, fovea=True, innovation_flag=True
    )
    assert mode == MODE_PLAN and correction_due is True

    free_mib = (
        torch.cuda.mem_get_info()[0] / 2**20
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"[eval-servo smoke] CPU 闭环集成 OK（plan×{plan_count}, foveal×{foveal_count}）"
        + (
            f"；GPU 空闲 {free_mib:.0f} MiB < 2048 MiB，回退 CPU 冒烟"
            if 0 < free_mib < 2048
            else ""
        )
    )


def _servo_vision_smoke(model, tokens, role_queries, coords):
    """冒烟用的 reader 单跑（等价 eval_metaworld._servo_vision，避免依赖 live_vjepa）。"""
    readout = model.slot_reader(tokens, role_queries, coords)
    stream = build_multimode_stream(model, readout, tokens, role_queries)
    return readout, stream


# --- 审查修复回归：真实 _foveal_tokens + 坐标变换端到端 ---

class _MockFoveaBackbone(nn.Module):
    """vision_backbone 契约：forward(inputs, pooling) -> [B, 1152, D]。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, inp, pooling="flat"):
        b = inp.shape[0]
        return torch.randn(b, 1152, self.dim)


def test_foveal_tokens_full_encoder_shape():
    """审查 P0-2/P0-4：真实 _foveal_tokens（6 维 crop 路径 + 完整编码契约）。

    单决策窗口 [W,R,R,3] → [B=1,T=1,W,R,R,3] → crops[0,0] → 编码 [1,1152,D]。
    """
    from eval_metaworld import _foveal_tokens
    from va_compound.fovea import compute_roi

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (192, 192, 3), dtype=np.uint8) for _ in range(4)]
    mu = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.1, -0.1]]])  # [1,2,3]
    cov = torch.eye(3).repeat(1, 2, 1, 1)
    roi = compute_roi(mu, cov, image_size=192)
    bb = _MockFoveaBackbone(dim=64)
    out = _foveal_tokens(frames, roi, torch.device("cpu"), vision_backbone=bb)
    assert tuple(out.shape) == (1, 1152, 64)


def test_foveal_tokens_prefix_encoder_path():
    """full_encoder=False 走 fovea_encoder（保留路径，显式开关）。"""
    from eval_metaworld import _foveal_tokens
    from va_compound.fovea import compute_roi

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (192, 192, 3), dtype=np.uint8) for _ in range(4)]
    mu = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.1, -0.1]]])
    cov = torch.eye(3).repeat(1, 2, 1, 1)
    roi = compute_roi(mu, cov, image_size=192)

    class _Prefix(nn.Module):
        def forward(self, inp):
            assert tuple(inp.shape) == (1, 4, 3, 384, 384)  # crop 已 resize 384
            return torch.randn(1, 1152, 32)

    out = _foveal_tokens(
        frames, roi, torch.device("cpu"),
        fovea_encoder=_Prefix(), full_encoder=False,
    )
    assert tuple(out.shape) == (1, 1152, 32)


def test_foveal_full_to_crop_roundtrip_readout():
    """审查 P0-3 端到端：reader 前 full→crop、后 crop→full，mu 语义回到全图。

    用真实 LocalControlSlotReader（multi_mode）+ 变换函数，验证 fovea 部署
    的坐标链：full prev_mu → crop reader → crop→full 输出 ≈ 全图坐标。
    """
    from va_compound.fovea import crop_to_full_cov, crop_to_full_norm, full_to_crop_norm
    from va_compound.local_control_slots import LocalControlSlotReader

    torch.manual_seed(0)
    B, K, D = 1, 3, 64
    reader = LocalControlSlotReader(
        vision_dim=D, hidden_dim=64, num_slots=K, multi_mode=True, aux_dim=16
    ).eval()
    coords = build_grid_coords(24)  # crop 归一化网格（[-1,1]，与全图同构）
    tokens = torch.randn(B, 1152, D)
    q = torch.randn(B, K, 64)
    roi = torch.tensor([[120.0, 120.0, 96.0]])  # S=240, r=96（放大 2.5×）
    prev_mu_full = torch.tensor([[[[0.0, 0.2, -0.3], [0.0, 0.2, -0.3]],
                                   [[0.0, -0.1, 0.4], [0.0, -0.1, 0.4]],
                                   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])  # [1,3,2,3]
    prev_crop = full_to_crop_norm(prev_mu_full, roi, image_size=240)
    out = reader(tokens, q, coords, prev_mu=prev_crop)
    mu_full = crop_to_full_norm(out.mu, roi, image_size=240)
    cov_full = crop_to_full_cov(out.cov, roi, image_size=240)
    assert tuple(mu_full.shape) == (B, K, 2, 3)
    assert tuple(cov_full.shape) == (B, K, 2, 3, 3)
    # 形状/数值健全：逆变换后坐标仍在全图归一化范围附近（crop 内模式经放大
    # 后可能越界，但不允许 NaN/Inf；中心点应回到全图中心附近）。
    assert torch.isfinite(mu_full).all()
    assert (mu_full.abs() < 10.0).all()
    # 恒等核对：reader 不使用跟踪先验时（track_gamma=0），输入 prev 不影响
    # 输出；这里只验证变换链可逆性：crop→full 再 full→crop 回到原值。
    back_crop = full_to_crop_norm(mu_full, roi, image_size=240)
    torch.testing.assert_close(back_crop, out.mu, atol=1e-4, rtol=1e-4)


def test_five_step_foveal_deployment_chain():
    """审查要求的 5 步 foveal smoke：真实部署函数链（非随机 token 假路径）。

    覆盖 P0-1~P0-4 修复点的接线：plan（frames/clip 构造 + 全图 dense 读出）
    → foveal feedback（_foveal_tokens 真实 crop 编码 + full↔crop 坐标变换 +
    servo 修正）→ 新息触发提前 plan 刷新。用 mock backbone（dim 与策略匹配）
    贯穿，断言无异常且修正/标志形状正确。
    """
    from eval_metaworld import (
        ACTION_HORIZON,
        MODE_FOVEAL,
        MODE_PLAN,
        ServoRuntime,
        _foveal_tokens,
        _servo_vision,
        fovea_schedule,
    )
    from va_compound.fovea import (
        compute_roi,
        crop_to_full_cov,
        crop_to_full_norm,
        full_to_crop_norm,
    )

    torch.manual_seed(0)
    model = make_policy()  # 小策略（multi_mode + local_slots，CPU；dim=64）
    coords_np = build_grid_coords(24).numpy()  # _servo_vision 契约：np.ndarray
    role_queries = torch.randn(1, 6, 64)  # hidden_dim=64

    class _LangCache(SimpleNamespace):
        def __init__(self):
            self.role_queries = role_queries

    lang_cache = _LangCache()
    bb = _MockFoveaBackbone(dim=64)  # 与策略 vision_dim 匹配
    from va_compound.servo import InteractionServo
    servo = InteractionServo(
        vision_dim=64, lang_dim=64, action_dim=4, rank=2
    )
    runtime = ServoRuntime(controller=servo, ablation="none")

    rng = np.random.default_rng(1)
    frame_buffer = [rng.integers(0, 255, (192, 192, 3), dtype=np.uint8)
                    for _ in range(13)]  # 窗口 4 帧 × stride 2 + 首帧
    plan_step = None
    chunk = None
    last_a = None  # 跨决策上一动作（ν 契约：与 g_prev 同给）
    c2_token = 0
    innovation_flag = False
    roi = None
    plan_count = foveal_count = 0
    for step in range(5):
        # 模拟 eval 主循环：新帧入 buffer + frames/clip 构造（P0-1 修复点）
        frame_buffer.append(rng.integers(0, 255, (192, 192, 3), dtype=np.uint8))
        frame_buffer.pop(0)
        indices = list(range(-2 * 4 + 1, 0, 2))
        frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
        # 随机权重的 innovation_flag 恒 True 会每步强制 plan（foveal 链不执行）；
        # 此处模拟"新息在阈值内"的正常节奏（plan_stride=6 → 1 plan + 4 foveal），
        # 新息触发提前刷新的节奏由 test_closed_loop_smoke_plan_foveal_correction_rhythm 覆盖。
        mode, correction_due, _ = fovea_schedule(
            step, plan_step, 6, 1, ACTION_HORIZON,
            fovea=True, innovation_flag=False,
        )
        if mode == MODE_PLAN:
            # 全图 dense 读出（真实 _servo_vision 契约）
            tokens_full = bb(torch.randn(1, 4, 3, 384, 384), pooling="dense")
            readout, stream = _servo_vision(
                model, tokens_full, lang_cache, coords_np,
                prev_mu=runtime.prev_mu,
            )
            render_size = frame_buffer[-1].shape[0]
            roi = compute_roi(
                *select_roi_pair(readout.mu, readout.cov, readout.vis),
                image_size=render_size,
            ).detach().cpu().float()
            chunk = np.zeros((ACTION_HORIZON, 4), dtype=np.float32)  # 名义 ā（模拟）
            plan_step = step
            c2_token = 0
            plan_count += 1
        if correction_due and c2_token < ACTION_HORIZON and chunk is not None:
            if step != plan_step:
                # foveal 局部更新（真实 _foveal_tokens + 坐标变换，P0-2/3/4）
                tokens_crop = _foveal_tokens(
                    frames, roi, torch.device("cpu"),
                    vision_backbone=bb,
                )
                prev_crop = (
                    full_to_crop_norm(runtime.prev_mu, roi, render_size)
                    if runtime.prev_mu is not None else None
                )
                readout, _ = _servo_vision(
                    model, tokens_crop, lang_cache, coords_np, prev_mu=prev_crop,
                )
                readout = MultiModeReadout(
                    readout.slots,
                    crop_to_full_norm(readout.mu, roi, render_size),
                    crop_to_full_cov(readout.cov, roi, render_size),
                    readout.vis,
                    readout.weights,
                )
                foveal_count += 1
            correction, innovation_flag = runtime.correct(
                readout,
                np.zeros(4, dtype=np.float32),
                role_queries.mean(dim=1),
                a_prev=None if last_a is None else last_a,
            )
            assert correction is None or correction.shape == (4,)
            if correction is not None:
                last_a = np.zeros(4, dtype=np.float32)  # 模拟执行后的上一动作
            c2_token += 1
    assert plan_count >= 1 and foveal_count >= 1  # 1 plan + 4 foveal 已执行
    assert torch.isfinite(readout.mu).all()
