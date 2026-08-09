"""Step 1 多模式读出（--multi-mode，C²-IRF v2 设计 §七 Step 1）单元测试。

覆盖（全部 CPU 可跑）：
- 形状/梯度：MultiModeReadout 全契约（slots/mu/cov/vis/weights）+ 全部参数
  梯度 + 288（12×12）与 1152（24×24）两种网格；
- 假中点修复：两个相距远的强峰各产生独立 μ，不再输出两者之间的中间中心
  （旧全局加权平均路径的中心 ≈ 中点，作为对照）；
- NULL 遮挡选择：查询对齐 NULL 键 → vis≈0、槽内容退化为 NULL 值；
- 旧行为一致：multi_mode=False 与参考实现逐位一致、state_dict 键不变、
  prev_mu/key_aux 非 None 时显式报错；
- 模型集成：config 校验 + 31-token 视觉流（16 coarse + 12 modes + 3 relations）
  + vis 条件注入零初始化。
"""
from __future__ import annotations

import pytest
import torch

from va_compound.local_control_slots import (
    LocalControlSlotReader,
    MultiModeReadout,
    fourier_encode,
)
from va_compound.model import VACompoundConfig, VACompoundPolicy

N_DENSE = 2 * 24 * 24  # 1152


def build_grid_coords(grid: int = 24) -> torch.Tensor:
    """[2*grid², 3]：与 live_vjepa._dense_coords 同一生成公式（t→y→x 行优先）。"""
    rows = []
    half = (grid - 1) / 2
    for t in range(2):
        for y in range(grid):
            for x in range(grid):
                rows.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
    return torch.tensor(rows, dtype=torch.float32)


def make_controlled_reader(*, multi_mode: bool) -> LocalControlSlotReader:
    """固定投影的可控 reader：K_n = P·tokens_n（norm/query 恒等、pos 旁路零）。

    测试据此直接构造任意期望的寻址 logit（如两个相距远的强峰）。
    """
    reader = LocalControlSlotReader(
        vision_dim=64, hidden_dim=64, num_slots=1, multi_mode=multi_mode
    )
    with torch.no_grad():
        reader.vision_norm.weight.fill_(1.0)
        reader.vision_norm.bias.zero_()
        reader.query_norm.weight.fill_(1.0)
        reader.query_norm.bias.zero_()
        P = torch.randn(64, 64)
        reader.vision_proj.weight.copy_(P)
        reader.vision_proj.bias.zero_()
        assert reader.pos_proj.weight.abs().sum() == 0  # 默认零初始化
    return reader


def controlled_tokens(P: torch.Tensor, targets: dict[int, torch.Tensor]) -> torch.Tensor:
    """tokens [1, N, 64]：第 idx 个 patch 的 K_n = targets[idx]（P† 最小范数解）。"""
    pinv = torch.linalg.pinv(P)
    tokens = torch.zeros(1, N_DENSE, 64)
    for idx, v in targets.items():
        tokens[0, idx] = pinv @ v
    return tokens


def test_multimode_shapes_and_gradients():
    torch.manual_seed(0)
    B, K, D = 2, 3, 32
    reader = LocalControlSlotReader(vision_dim=D, hidden_dim=64, num_slots=K, multi_mode=True)
    coords = build_grid_coords(24)
    tokens = torch.randn(B, N_DENSE, D)
    queries = torch.randn(B, K, 64)
    out = reader(tokens, queries, coords)
    assert isinstance(out, MultiModeReadout)
    assert out.slots.shape == (B, K, 2, D)
    assert out.mu.shape == (B, K, 2, 3)
    assert out.cov.shape == (B, K, 2, 3, 3)
    assert out.vis.shape == (B, K)
    assert out.weights.shape == (B, K, 2, N_DENSE)
    # 可见度严格在 (0,1)（softmax 有限输入）；权重每模式局部归一化且稀疏（≤5×5）。
    assert bool((out.vis > 0).all() and (out.vis < 1).all())
    assert torch.allclose(out.weights.sum(-1), torch.ones(B, K, 2), atol=1e-5)
    assert int(out.weights[0, 0, 0].count_nonzero()) <= 25
    # μ 落在坐标范围；协方差对称正定（含数值下限）。
    assert bool((out.mu.abs() <= 1.05).all())
    assert torch.allclose(out.cov, out.cov.transpose(-1, -2), atol=1e-6)
    # 跟踪先验：prev_mu 传递后形状不变、track_gamma 收到梯度。
    prev = torch.randn(B, K, 2, 3)
    out2 = reader(tokens, queries, coords, prev_mu=prev)
    assert out2.mu.shape == (B, K, 2, 3)
    loss = sum(t.mean() for t in out2)
    loss.backward()
    for name, p in reader.named_parameters():
        if name.startswith(("aux_proj", "aux_query", "aux_gamma")):
            continue  # key_aux（K⁵）路径参数，仅在传入 key_aux 时入计算图（单独测试覆盖）
        assert p.grad is not None, f"multi_mode 参数无梯度: {name}"
    # key_aux 单模式路径必须显式报错（防静默忽略）——见 test_key_aux_requires_multimode。
    with pytest.raises(ValueError, match="N=2"):
        reader(torch.randn(B, 100, D), queries, build_grid_coords(24)[:100])


def test_multimode_288_grid():
    """288 token（12×12 网格）同样可用——multi_mode 不强制 dense_readout。"""
    torch.manual_seed(1)
    reader = LocalControlSlotReader(vision_dim=32, hidden_dim=64, num_slots=2, multi_mode=True)
    coords = build_grid_coords(12)
    out = reader(torch.randn(1, 288, 32), torch.randn(1, 2, 64), coords)
    assert out.weights.shape == (1, 2, 2, 288)
    assert torch.allclose(out.weights.sum(-1), torch.ones(1, 2, 2), atol=1e-5)


def test_fake_midpoint_fix():
    """两个相距远的强峰 → 两个独立 μ；旧全局加权平均会落在中间（假中点）。"""
    torch.manual_seed(2)
    reader = make_controlled_reader(multi_mode=True)
    coords = build_grid_coords(24)
    # 两个远距峰：t=0 切片 (4,4) 与 t=1 切片 (19,19)。
    idx_a, idx_b = 4 * 24 + 4, 576 + 19 * 24 + 19
    p_a, p_b = coords[idx_a], coords[idx_b]
    v_a = torch.randn(64)
    v_b = torch.randn(64)
    v_b = v_b - (v_b @ v_a) * v_a
    v_a, v_b = v_a / v_a.norm(), v_b / v_b.norm()
    tokens = controlled_tokens(
        reader.vision_proj.weight,
        {idx_a: v_a, idx_b: v_b},  # 其余 patch：K=0 → 全 logit 0（弱背景）
    )
    q = (v_a + v_b).view(1, 1, 64)  # 与两个峰同时匹配
    out = reader(tokens, q, coords)
    mu0, mu1 = out.mu[0, 0, 0], out.mu[0, 0, 1]
    # 每峰局部 soft-argmax（对称 5×5 窗口）→ μ 精确落在峰 patch 上。
    assert torch.allclose(mu0, p_a, atol=0.05), (mu0, p_a)
    assert torch.allclose(mu1, p_b, atol=0.05), (mu1, p_b)
    midpoint = (p_a + p_b) / 2
    assert (mu0 - midpoint).norm() > 0.5 and (mu1 - midpoint).norm() > 0.5
    # 对照：旧路径的全局加权平均中心（centers = Σw·p，无局部 NMS）≈ 中点——
    # 正是设计文档 §二.3 要修的"假中点"。
    logits = torch.einsum("bkh,bnh->bkn", q, tokens @ reader.vision_proj.weight.T) / (
        (64 // 8) ** 0.5
    )
    w_old = torch.softmax(logits, dim=-1)
    old_center = torch.einsum("bkn,nc->bc", w_old, coords)[0]
    assert (old_center - midpoint).norm() < 0.1, (old_center, midpoint)


def test_null_occlusion_selection():
    """查询对齐 NULL 键 → 可见度≈0、槽内容退化为 NULL 值（遮挡选择）。"""
    torch.manual_seed(4)
    reader = make_controlled_reader(multi_mode=True)
    coords = build_grid_coords(24)
    q = torch.randn(1, 1, 64)
    tokens = torch.zeros(1, N_DENSE, 64)  # 全零 → patch logit 全 0
    with torch.no_grad():
        # NULL 键与查询强对齐（30·q/|q| → null logit 15 >> 0）。
        reader.null_key.copy_(30.0 * q[0, 0] / q[0, 0].norm())
    out = reader(tokens, q, coords)
    assert out.vis[0, 0].item() < 0.01, out.vis[0, 0].item()
    # vis≈0 → 模式内容 = (1−vis)·z_local + vis·V_null ≈ V_null（学习到的"空"向量）。
    assert torch.allclose(out.slots[0, 0, 0], reader.null_value, atol=0.05)
    assert torch.allclose(out.slots[0, 0, 1], reader.null_value, atol=0.05)
    with torch.no_grad():
        reader.null_key.copy_(-30.0 * q[0, 0] / q[0, 0].norm())
    out2 = reader(tokens, q, coords)
    assert out2.vis[0, 0].item() > 0.99, out2.vis[0, 0].item()


def test_local_weights_stay_normalized_when_null_dominates():
    """NULL 概率≈1 时 patch 概率可下溢；局部模式仍必须有限且和为 1。"""
    torch.manual_seed(41)
    reader = make_controlled_reader(multi_mode=True)
    coords = build_grid_coords(24)
    q = torch.randn(1, 1, 64)
    tokens = torch.zeros(1, N_DENSE, 64)
    with torch.no_grad():
        reader.null_key.copy_(1000.0 * q[0, 0] / q[0, 0].norm())
    out = reader(tokens, q, coords)
    assert out.vis[0, 0].item() == 0.0
    assert torch.isfinite(out.weights).all()
    assert torch.isfinite(out.mu).all()
    assert torch.allclose(out.weights.sum(-1), torch.ones(1, 1, 2), atol=1e-6)


def test_multimode_false_matches_legacy():
    """multi_mode=False：与旧 forward 逐位一致（参考实现复刻 + 参数键不变）。"""
    torch.manual_seed(7)
    B, N, K, D = 3, 288, 6, 768
    reader = LocalControlSlotReader(vision_dim=D, hidden_dim=512, num_slots=K)
    tokens = torch.randn(B, N, D)
    queries = torch.randn(B, K, 512)
    coords = torch.rand(N, 3)
    slots, weights, centers = reader(tokens, queries, coords)

    # 参考实现：旧 forward 的逐位复刻。
    pos = fourier_encode(coords.to(dtype=tokens.dtype))
    visual = reader.vision_proj(reader.vision_norm(tokens)) + reader.pos_proj(pos)[None]
    delta, w = reader.cross_attn(
        reader.query_norm(queries),
        visual,
        visual,
        need_weights=True,
        average_attn_weights=False,
    )
    gate = torch.sigmoid(reader.read_gate_logit)
    ref_slots = reader.to_vision(queries + gate * delta + reader.ffn(reader.output_norm(queries + gate * delta)))
    ref_weights = w.float().mean(dim=1)
    ref_centers = torch.einsum("bkn,nc->bkc", ref_weights, coords.float())
    torch.testing.assert_close(slots, ref_slots, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(centers, ref_centers, rtol=1e-6, atol=1e-6)

    # 参数键与旧版完全一致（老 checkpoint 严格加载不受影响，无新增参数）。
    legacy_keys = {
        "vision_norm.weight", "vision_norm.bias",
        "vision_proj.weight", "vision_proj.bias",
        "pos_proj.weight",
        "query_norm.weight", "query_norm.bias",
        "cross_attn.in_proj_weight", "cross_attn.in_proj_bias",
        "cross_attn.out_proj.weight", "cross_attn.out_proj.bias",
        "read_gate_logit",
        "output_norm.weight", "output_norm.bias",
        "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
        "to_vision.weight", "to_vision.bias",
    }
    assert set(reader.state_dict()) == legacy_keys
    # 多模式专属参数在旧路径上不存在。
    assert not hasattr(reader, "null_key")
    # prev_mu 在单模式路径必须显式报错（不接受静默忽略）。
    with pytest.raises(NotImplementedError, match="prev_mu"):
        reader(tokens, queries, coords, prev_mu=torch.randn(B, K, 2, 3))


def test_tracking_prior_shifts_peaks():
    """b_track(p; prev_mu)：强 γ 下寻址峰被拉向上一决策的模式中心。"""
    torch.manual_seed(6)
    # track_gamma_init=10：全零 tokens（patch logit 全 0）时先验主导热图。
    reader = make_controlled_reader(multi_mode=True)
    with torch.no_grad():
        reader.track_gamma.fill_(10.0)
    coords = build_grid_coords(24)
    tokens = torch.zeros(1, N_DENSE, 64)
    q = torch.randn(1, 1, 64)
    prev_mu = torch.tensor([[[[-1.0, 0.13, 0.13], [-1.0, -0.65, 0.65]]]])
    out = reader(tokens, q, coords, prev_mu=prev_mu)
    # 第一模式被先验拉到 (t=-1, y=5, x=5) 附近（σ=0.25 → 误差 < ~2 patch）。
    assert torch.allclose(out.mu[0, 0, 0], prev_mu[0, 0, 0], atol=0.2), out.mu[0, 0, 0]
    # 第二模式同样跟随其先验中心（同时间片，避免跨片高斯惩罚）。
    assert torch.allclose(out.mu[0, 0, 1], prev_mu[0, 0, 1], atol=0.2), out.mu[0, 0, 1]
    # 无先验时（prev_mu=None）热图均匀，μ 与先验无关（对照）。
    out0 = reader(tokens, q, coords)
    assert (out0.mu[0, 0, 0] - prev_mu[0, 0, 0]).norm() > 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 冒烟可选")
def test_multimode_cuda_smoke():
    """小 batch CUDA 冒烟：gather/scatter_add 路径在 GPU 上可用。"""
    torch.manual_seed(8)
    reader = LocalControlSlotReader(
        vision_dim=32, hidden_dim=64, num_slots=2, multi_mode=True
    ).cuda()
    tokens = torch.randn(1, N_DENSE, 32, device="cuda")
    queries = torch.randn(1, 2, 64, device="cuda")
    coords = build_grid_coords(24).cuda()
    out = reader(tokens, queries, coords)
    assert out.slots.shape == (1, 2, 2, 32)
    assert torch.allclose(out.weights.sum(-1), torch.ones(1, 2, 2, device="cuda"), atol=1e-5)
    sum(t.mean() for t in out).backward()
    assert reader.null_key.grad is not None


def make_mm_config(**overrides) -> VACompoundConfig:
    base = dict(
        language_dim=1536,
        vision_dim=768,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        action_horizon=6,
        action_dim=4,
        proprio_dim=4,
        direct_head=True,
        local_slots=True,
        dense_readout=True,
        local_slot_tokens=N_DENSE,
        multi_mode=True,
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def test_config_validation():
    with pytest.raises(ValueError, match="local_slots"):
        VACompoundConfig(
            language_dim=1536, vision_dim=768, hidden_dim=256, num_layers=1,
            num_heads=4, action_horizon=4, action_dim=4, proprio_dim=4,
            multi_mode=True,  # local_slots 默认 False
        )
    with pytest.raises(ValueError, match="互斥"):
        make_mm_config(local_slots_direct288=True)
    # 288 网格（12×12）同样允许——multi_mode 不强制 dense_readout。
    cfg = make_mm_config(dense_readout=False, local_slot_tokens=288)
    assert cfg.multi_mode


def test_model_multimode_stream_31_tokens_and_vis():
    """1152 dense tokens → 16 coarse + 12 modes + 3 relations = 31-token 流。"""
    torch.manual_seed(3)
    model = VACompoundPolicy(make_mm_config()).eval()
    B = 2
    batch = {
        "vision_tokens_st": torch.randn(B, 4, N_DENSE, 768),
        "coords": build_grid_coords(24),
        "language_hidden": torch.randn(B, 8, 1536),
        "language_mask": torch.ones(B, 8, dtype=torch.bool),
        "proprio": torch.randn(B, 4, 4),
        "previous_action": torch.randn(B, 4, 4),
        "actions": torch.randn(B, 4, 6, 4),
    }
    cache = model.build_language_cache(
        batch["language_hidden"], batch["language_mask"]
    )
    assert cache.role_queries is not None
    vision = model.build_local_vision(
        batch["vision_tokens_st"][:, 0], batch["coords"], cache.role_queries
    )
    assert vision.shape == (B, 31, 768)  # 16 coarse + 12 modes + 3 relations
    with torch.inference_mode():
        cond, _ = model.encode_condition(
            vision,
            batch["proprio"][:, 0],
            batch["previous_action"][:, 0],
            language_cache=cache,
            return_visual_memory=True,
        )
        pred = model.decode_actions(cond)
    assert pred.shape == (B, 6, 4)
    # vis 条件注入投影零初始化（初始静默，不改变 dense_readout 初始行为）。
    assert bool((model.vis_conditioner.weight == 0).all())
    assert bool((model.vis_conditioner.bias == 0).all())


def test_model_multimode_gradients():
    """动作 loss 梯度回传到多模式 reader 全部参数（含 NULL/偏置/先验）。"""
    torch.manual_seed(5)
    model = VACompoundPolicy(make_mm_config(hidden_dim=128, num_layers=1)).train()
    B = 1
    batch = {
        "vision_tokens_st": torch.randn(B, 4, N_DENSE, 768),
        "coords": build_grid_coords(24),
        "language_hidden": torch.randn(B, 8, 1536),
        "language_mask": torch.ones(B, 8, dtype=torch.bool),
        "proprio": torch.randn(B, 4, 4),
        "previous_action": torch.randn(B, 4, 4),
        "actions": torch.randn(B, 4, 6, 4),
    }
    cache = model.build_language_cache(
        batch["language_hidden"], batch["language_mask"]
    )
    vision = model.build_local_vision(
        batch["vision_tokens_st"][:, 0], batch["coords"], cache.role_queries
    )
    cond, _ = model.encode_condition(
        vision,
        batch["proprio"][:, 0],
        batch["previous_action"][:, 0],
        language_cache=cache,
        return_visual_memory=True,
    )
    pred = model.decode_actions(cond)
    loss = (pred - batch["actions"][:, 0, : pred.shape[-2]]).pow(2).mean()
    loss.backward()
    for name in (
        "slot_reader.coord_bias.weight",
        "slot_reader.null_key",
        "slot_reader.null_value",
        "vis_conditioner.weight",
        "vis_conditioner.bias",
    ):
        param = dict(model.named_parameters())[name]
        assert param.grad is not None, f"no gradient through {name}"


# --- Step 4 key_aux（K⁵ 残差寻址项）集成测试 ---

def _make_reader_multimode(aux_dim: int = 128):
    return LocalControlSlotReader(
        vision_dim=768, hidden_dim=512, num_slots=6, multi_mode=True, aux_dim=aux_dim
    ).eval()


def test_key_aux_initial_silent():
    """γ_r·q_sᵀK⁵ 项初始必须 ≡0（aux_query 零初始化 + aux_proj 零初始化）：
    传入 key_aux 与不传的结果逐位一致。"""
    reader = _make_reader_multimode()
    torch.manual_seed(0)
    tokens = torch.randn(1, 288, 768)
    q = torch.randn(1, 6, 512)
    coords = build_grid_coords(12)
    aux = torch.randn(1, 288, 128)
    out_with = reader(tokens, q, coords, key_aux=aux)
    out_without = reader(tokens, q, coords)
    assert torch.equal(out_with.slots, out_without.slots)
    assert torch.equal(out_with.mu, out_without.mu)
    assert torch.equal(out_with.vis, out_without.vis)
    # γ 本身存在且初始 0.01（小门控，非梯度死区）
    assert abs(reader.aux_gamma.item() - 0.01) < 1e-6


def test_key_aux_trained_changes_addressing():
    """aux_query 非零后寻址项生效：logit 变化会移动读出中心。"""
    reader = _make_reader_multimode()
    torch.manual_seed(1)
    tokens = torch.randn(1, 288, 768)
    q = torch.randn(1, 6, 512)
    coords = build_grid_coords(12)
    aux = torch.randn(1, 288, 128)
    with torch.no_grad():
        # 模拟训练后：aux_query 非零、γ 增大 → 寻址项 = γ·q_sᵀK⁵ ≠ 0 → 输出变化
        reader.aux_query.weight.normal_(0.0, 0.1)
        reader.aux_gamma.fill_(1.0)
        reader.aux_proj.weight.normal_(0.0, 0.1)  # branch 内部正常初始化（首步即活）
    out_pert = reader(tokens, q, coords, key_aux=aux)
    out_base = reader(tokens, q, coords)
    assert not torch.equal(out_pert.slots, out_base.slots)
    assert (out_pert.mu - out_base.mu).abs().max() > 0


def test_key_aux_shape_mismatch_raises():
    reader = _make_reader_multimode()
    tokens = torch.randn(1, 288, 768)
    q = torch.randn(1, 6, 512)
    coords = build_grid_coords(12)
    with pytest.raises(ValueError):
        reader(tokens, q, coords, key_aux=torch.randn(1, 100, 128))


def test_key_aux_requires_multimode():
    """单模式路径传 key_aux 必须显式报错（防静默忽略）。"""
    reader = LocalControlSlotReader(vision_dim=768, hidden_dim=512, num_slots=6).eval()
    tokens = torch.randn(1, 288, 768)
    q = torch.randn(1, 6, 512)
    coords = build_grid_coords(12)
    with pytest.raises(NotImplementedError):
        reader(tokens, q, coords, key_aux=torch.randn(1, 288, 128))


def test_key_aux_gradient_flow():
    """key_aux 路径梯度：aux_proj/aux_query/aux_gamma 均有非零梯度。"""
    reader = _make_reader_multimode()
    torch.manual_seed(2)
    tokens = torch.randn(1, 288, 768)
    q = torch.randn(1, 6, 512)
    coords = build_grid_coords(12)
    aux = torch.randn(1, 288, 128)
    out = reader(tokens, q, coords, key_aux=aux)
    loss = out.slots.pow(2).mean()
    loss.backward()
    grads = {n: p.grad for n, p in reader.named_parameters()
             if n.startswith(("aux_proj", "aux_query", "aux_gamma"))}
    # 零初始化输出投影（aux_query）阻断整条寻址支路：首步只有 aux_query 自身
    # 收到非零梯度（∂/∂q_aux = γ·aux_k，γ=0.01 非零）；aux_gamma/aux_proj 的
    # 梯度都乘 q_aux≡0 → 首步允许零（设计 §八 已注明，一步后自动解除）。
    assert grads["aux_query.weight"] is not None and grads["aux_query.weight"].abs().sum() > 0
    # 解除验证：模拟一步更新（aux_query 非零）后整条支路必须全活。
    with torch.no_grad():
        reader.aux_query.weight.normal_(0.0, 0.1)
    reader.zero_grad()
    loss2 = reader(tokens, q, coords, key_aux=aux).slots.pow(2).mean()
    loss2.backward()
    for name in ("aux_proj.weight", "aux_query.weight", "aux_gamma"):
        g = dict(reader.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, f"aux 支路未解除: {name}"
