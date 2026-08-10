"""MT-VJ 契约 §8 测试（artifacts/mt_vj_contract.md §8）。

只新建本文件，不修改其他文件。用例按契约 §8 的五组划分：

1. ``VJEPA21Backbone.forward_hierarchical_dense``：输出形状 / 与 ``_encode``
   及 ``encode_multi`` 的一致性（fake backbone 恒运行，真实 V-JEPA 在本地
   checkpoint 可用时运行）；
2. ``LanguageMetricField``：前向形状、p ∈ [0,1]；
3. ``MicroRefiner``：前向形状；
4. ``dense_readout_mtvj=True`` 且 W_o 全零时与 ``False`` 输出逐位一致
   （随机小模型）；
5. metric checkpoint 保存/加载 roundtrip（``weights_only=True``）。

并行 agent 未落地的实现用 ``pytest.skip``（带原因）跳过：模块/方法/字段
缺失时用例标记为 skipped 而非 failed，并行 agent 完成后自动激活、本文件
无需改动。
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from va_compound.backbones import VJEPA21Backbone, VJEPA21_CHECKPOINT_NAME
from va_compound.model import VACompoundConfig, VACompoundPolicy

# 契约 §公共常量
DENSE_TOKENS = 2 * 24 * 24  # 1152
N_ROLES = 4
H_DIM = 768
LANG_DIM = 2048
D_MODEL = 512  # RelationStateEncoder 的 d_model
IMAGE_SIZE = 384
GRID = 24
ROI = 96


def _dense_coords(grid: int = GRID) -> torch.Tensor:
    """[2*grid*grid, 3]：与 va_compound.live_vjepa._dense_coords() 同一公式
    （t∈{-1,1} 外层循环 → y → x，坐标归一化到 [-1,1]），避免引入 live_vjepa
    的重型 import 链。"""
    half = (grid - 1) / 2
    rows = []
    for t in range(2):
        for y in range(grid):
            for x in range(grid):
                rows.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
    return torch.tensor(rows, dtype=torch.float32)


def _exact_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """逐位一致比较（唯一例外：-0.0 视为等于 +0.0，allclose atol=rtol=0）。"""
    return bool(torch.allclose(a, b, rtol=0.0, atol=0.0))


# ---------------------------------------------------------------------------
# 并行 agent 未落地的能力探测（缺失 → skip，落地后自动激活）
# ---------------------------------------------------------------------------

def _requires_forward_hierarchical_dense() -> None:
    if not hasattr(VJEPA21Backbone, "forward_hierarchical_dense"):
        pytest.skip(
            "VJEPA21Backbone.forward_hierarchical_dense 尚未实现"
            "（backbones agent 并行开发中，实现后本用例自动生效）"
        )


def _metric_visual_head():
    return pytest.importorskip(
        "va_compound.metric_visual_head",
        reason="va_compound/metric_visual_head.py 尚未实现（metric agent 并行开发中）",
    )


def _requires_dense_readout_mtvj() -> None:
    if "dense_readout_mtvj" not in VACompoundConfig.__dataclass_fields__:
        pytest.skip(
            "VACompoundConfig.dense_readout_mtvj 尚未实现"
            "（model agent 并行开发中，实现后本用例自动生效）"
        )


# ---------------------------------------------------------------------------
# 1) forward_hierarchical_dense —— fake backbone（恒运行，快且确定）
# ---------------------------------------------------------------------------

class _MarkerBlock(nn.Module):
    """向全部 token 加一个层标记常数（区分层输出，模拟每层变换）。"""

    def __init__(self, marker: float) -> None:
        super().__init__()
        self.marker = marker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.marker


class _FakeVJEPA(nn.Module):
    """按官方 V-JEPA 2.1 forward 语义的伪模型（12 blocks，384→24×24 patch 网格）。

    官方契约（与 test_multi_layer._FakeVJEPA 同构，自包含不跨文件依赖）：
    ``self.out_layers`` 非 None 时按 block 升序收集 ``self.norm(x)`` 并返回
    列表（跳过末尾 norm）；None 时返回 ``norm(x)``。每个 token 第 0 特征列
    编码位置（值 = 16*位置），用于验证 t→y→x 扁平顺序。
    """

    def __init__(self, num_blocks: int = 12, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = 16
        self.tubelet_size = 2
        self.num_frames = 4
        self.img_height = IMAGE_SIZE
        self.img_width = IMAGE_SIZE
        self.out_layers = None
        self.norm = nn.Identity()
        self.blocks = nn.ModuleList(
            [_MarkerBlock(0.1 * (i + 1)) for i in range(num_blocks)]
        )
        self.scale = nn.Parameter(torch.tensor(1.0))  # 可训练参数占位

    def forward(self, videos: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        batch = videos.shape[0]
        x = torch.arange(2 * GRID * GRID * self.dim, dtype=torch.float32).view(
            1, DENSE_TOKENS, self.dim
        )
        x = x.repeat(batch, 1, 1) * self.scale
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))
        if self.out_layers is not None:
            return outs
        return self.norm(x)


def _fake_videos(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 4, 3, IMAGE_SIZE, IMAGE_SIZE)


def _fake_backbone() -> VJEPA21Backbone:
    backbone = VJEPA21Backbone(_FakeVJEPA(), max_tokens=64)
    backbone.freeze_all()  # 与 from_pretrained 生产路径等价
    return backbone


class TestForwardHierarchicalDense:
    def test_default_layers_shape_and_keys(self) -> None:
        _requires_forward_hierarchical_dense()
        backbone = _fake_backbone()
        out = backbone.forward_hierarchical_dense(_fake_videos())
        assert isinstance(out, dict)
        assert set(out.keys()) == {5, 11}
        for layer, tokens in out.items():
            assert tuple(tokens.shape) == (2, DENSE_TOKENS, 16)

    def test_layer11_matches_encode(self) -> None:
        """H¹¹ 与 _encode 逐位一致（官方 out_layers 收集与默认 forward 均对
        block 11 的输出做同一次 norm，_encode 等价于 out_layers=(11,) 池化前）。"""
        _requires_forward_hierarchical_dense()
        backbone = _fake_backbone()
        videos = _fake_videos()
        out = backbone.forward_hierarchical_dense(videos)
        assert _exact_equal(out[11], backbone._encode(videos))

    def test_layer5_matches_encode_multi(self) -> None:
        """H⁵ 与 encode_multi(video, (5,))[0] 逐位一致（同一官方机制）。"""
        _requires_forward_hierarchical_dense()
        backbone = _fake_backbone()
        videos = _fake_videos()
        out = backbone.forward_hierarchical_dense(videos)
        h5 = backbone.encode_multi(videos, out_layers=(5,))[0]
        assert _exact_equal(out[5], h5)

    def test_token_order_then_y_then_x(self) -> None:
        """输出必须按 t→y→x 序扁平：位置 p 的第 0 特征列 = 16p + 层标记和
        （非该顺序的重排会破坏严格递增性）。"""
        _requires_forward_hierarchical_dense()
        backbone = _fake_backbone()
        out = backbone.forward_hierarchical_dense(_fake_videos(1))
        flat = out[5][0, :, 0]  # [1152]
        assert bool((torch.diff(flat) > 0).all()), (
            "dense token 必须按 t→y→x 顺序扁平（相邻位置严格递增）"
        )
        # block 0..5 的 marker 累加和 = 0.1*(1+...+6) = 2.1；fp32 多次相加放宽 atol。
        expected = 16.0 * torch.arange(DENSE_TOKENS, dtype=torch.float32) + 2.1
        assert torch.allclose(flat, expected, atol=5e-2)

    def test_custom_out_layers(self) -> None:
        _requires_forward_hierarchical_dense()
        backbone = _fake_backbone()
        out = backbone.forward_hierarchical_dense(_fake_videos(1), out_layers=(11,))
        assert set(out.keys()) == {11}
        assert tuple(out[11].shape) == (1, DENSE_TOKENS, 16)

    def test_existing_paths_unchanged_after_call(self) -> None:
        """调用后 model.out_layers 恢复、既有 forward/_encode 行为不变。"""
        _requires_forward_hierarchical_dense()
        backbone = _fake_backbone()
        model = backbone.model
        assert model.out_layers is None
        videos = _fake_videos(2)
        backbone.forward_hierarchical_dense(videos)
        assert model.out_layers is None
        tokens = backbone(videos, pooling="dense")
        assert tuple(tokens.shape) == (2, DENSE_TOKENS, 16)


# ---------------------------------------------------------------------------
# 1b) forward_hierarchical_dense —— 真实 V-JEPA 2.1（本地 checkpoint，GPU）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vjepa21_backbone() -> VJEPA21Backbone:
    """真实 V-JEPA 2.1 ViT-B（local_files_only）。缺失/加载失败 → skip。

    checkpoint 为 1.66GB，加载约 30-60s；模块级 fixture 只加载一次。
    """
    _requires_forward_hierarchical_dense()
    hub_dir = Path(torch.hub.get_dir())
    ckpt = hub_dir / "checkpoints" / VJEPA21_CHECKPOINT_NAME
    if not ckpt.is_file():
        pytest.skip(
            f"本地 V-JEPA 2.1 checkpoint 缺失（{ckpt}）；先运行 prepare_models.py。"
            "真实模型用例跳过，形状契约由 fake-backbone 用例覆盖"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    try:
        return VJEPA21Backbone.from_pretrained(
            device=device, dtype=dtype, local_files_only=True
        )
    except Exception as exc:  # 加载失败（文件损坏/内存不足等）→ skip 而非 fail
        pytest.skip(f"V-JEPA 2.1 加载失败：{exc!r}")


def test_real_vjepa_forward_hierarchical_dense(vjepa21_backbone: VJEPA21Backbone) -> None:
    """真实模型 + torch.randn(1,4,3,384,384) 小输入：形状与 _encode 一致性。"""
    backbone = vjepa21_backbone
    video = torch.randn(1, 4, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone.forward_hierarchical_dense(video)
    assert isinstance(out, dict)
    assert set(out.keys()) == {5, 11}
    for layer, tokens in out.items():
        assert tuple(tokens.shape) == (1, DENSE_TOKENS, H_DIM)
        assert torch.isfinite(tokens).all()
    # 官方 out_layers 语义：out_layers=(11,) 收集的 norm(x) 与默认 forward 的
    # 最终 norm(x) 是同一计算 → 逐位一致（GPU 上同形状同 kernel，确定性）。
    assert _exact_equal(out[11], backbone._encode(video))
    h5 = backbone.encode_multi(video, out_layers=(5,))[0]
    assert _exact_equal(out[5], h5)


# ---------------------------------------------------------------------------
# 2) LanguageMetricField —— 前向形状、p ∈ [0,1]
# ---------------------------------------------------------------------------

def metric_field_inputs(batch: int = 2, n_lang: int = 5) -> dict:
    mask = torch.ones(batch, n_lang, dtype=torch.bool)
    mask[1, -1] = False  # 带一个 padding token，覆盖 mask 生效路径
    return {
        "h5": torch.randn(batch, DENSE_TOKENS, H_DIM),
        "h11": torch.randn(batch, DENSE_TOKENS, H_DIM),
        "language_hidden": torch.randn(batch, n_lang, LANG_DIM),
        "language_mask": mask,
        "coords": _dense_coords(),
    }


class TestLanguageMetricField:
    def test_forward_shapes_and_bounds(self) -> None:
        mm = _metric_visual_head()
        if not hasattr(mm, "LanguageMetricField"):
            pytest.skip("LanguageMetricField 尚未实现（metric agent 并行开发中）")
        head = mm.LanguageMetricField()  # 契约默认参数
        inputs = metric_field_inputs()
        with torch.inference_mode():
            out = head(**inputs)
        assert tuple(out.p.shape) == (2, N_ROLES, 2)
        assert tuple(out.visibility.shape) == (2, N_ROLES)
        assert tuple(out.offset.shape) == (2, N_ROLES, 2)
        assert tuple(out.heatmap.shape) == (2, N_ROLES, GRID, GRID)
        assert tuple(out.relation.shape) == (2, 6)  # 拍板 2A（2026-08-10）：[p_eef−p_obj(2), p_obj−p_target(2), axis_cos, depth_m]
        # 连续位置是图像坐标 0-1（y,x 序）；patch 中心 ± ½patch 偏移的浮点
        # 舍入用 1e-3 容差——坐标漏做 0-1 归一化会超出 ~0.5，必然触发。
        assert (out.p >= -1e-3).all(), "p 越界（< 0）"
        assert (out.p <= 1 + 1e-3).all(), "p 越界（> 1）"
        assert (out.visibility > 0).all() and (out.visibility < 1).all()
        assert torch.isfinite(out.offset).all()
        assert torch.isfinite(out.heatmap).all()
        assert torch.isfinite(out.relation).all()


# ---------------------------------------------------------------------------
# 3) MicroRefiner —— 前向形状
# ---------------------------------------------------------------------------

class TestMicroRefiner:
    def test_forward_shape(self) -> None:
        mm = _metric_visual_head()
        if not hasattr(mm, "MicroRefiner"):
            pytest.skip("MicroRefiner 尚未实现（metric agent 并行开发中）")
        refiner = mm.MicroRefiner()  # 契约默认 roi=96
        roi_images = torch.randn(2, 3, ROI, ROI)
        with torch.inference_mode():
            out = refiner(roi_images)
        assert tuple(out.shape) == (2, 4)  # δp_y, δp_x, δz, contact
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4) dense_readout_mtvj=True（W_o 全零）vs False 逐位一致
# ---------------------------------------------------------------------------

def _mtvj_config(**overrides) -> VACompoundConfig:
    """随机小模型 config，维度与契约 §5 字面形状一致：

    - ``vision_dim=768``：dense_evidence 按 [B, 1152, 768] 原维传入
      （model agent 的 DenseEvidenceProjector 按 config.vision_dim 投影，
      与生产配置相同，避免小维度歧义）；
    - ``hidden_dim=512``：metric_tokens [B, 2, 512]（契约 d_model=512，
      实现按 hidden_dim 投影，生产配置下二者相同）。

    带上 local_slots/dense_readout 以满足并行 model agent 可能添加的任何
    配置校验；等价性测试只走 encode_condition 层栈，与 local_slots 读出
    路径无关。"""
    base = dict(
        language_dim=64,
        vision_dim=H_DIM,  # 768：dense_evidence 原维
        hidden_dim=D_MODEL,  # 512：metric_tokens 的 d_model
        num_layers=2,
        num_heads=4,
        action_horizon=6,
        action_dim=4,
        proprio_dim=4,
        direct_head=True,
        local_slots=True,
        dense_readout=True,
        local_slot_tokens=DENSE_TOKENS,
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def _policy_inputs(batch: int = 2) -> dict:
    horizon = 6
    return {
        "vision": torch.randn(batch, 8, H_DIM),
        "proprio": torch.randn(batch, 4),
        "previous": torch.randn(batch, 4),
        "language": torch.randn(batch, 5, 64),
        "mask": torch.ones(batch, 5, dtype=torch.bool),
        "noisy": torch.randn(batch, horizon, 4),
        "flow_time": torch.rand(batch),
        # 契约 §5 约定形状：dense_evidence {5:[B,1152,768], 11:[B,1152,768]}
        # 原维（config.vision_dim=768 与生产一致）；metric_tokens
        # [B, 2, d_model=512]（RelationStateEncoder 输出，实现按 hidden_dim 投影）。
        "dense_evidence": {
            5: torch.randn(batch, DENSE_TOKENS, H_DIM),
            11: torch.randn(batch, DENSE_TOKENS, H_DIM),
        },
        "metric_tokens": torch.randn(batch, 2, D_MODEL),
    }


def _call_policy(
    model: VACompoundPolicy,
    inputs: dict,
    *,
    dense_evidence: torch.Tensor | None = None,
    metric_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """经 encode_condition 调层栈；若 agent 只把 dense 参数挂在 forward 上则
    退回 forward（flow 速度输出同样确定可比）。两者都未接入 → skip。"""
    base = {
        "vision_tokens": inputs["vision"],
        "proprio": inputs["proprio"],
        "previous_action": inputs["previous"],
        "language_hidden": inputs["language"],
        "language_mask": inputs["mask"],
    }
    extra = {}
    if dense_evidence is not None:
        extra = {"dense_evidence": dense_evidence, "metric_tokens": metric_tokens}
    if "dense_evidence" in inspect.signature(model.encode_condition).parameters:
        return model.encode_condition(**base, **extra)
    if "dense_evidence" in inspect.signature(model.forward).parameters:
        return model.forward(
            **base,
            noisy_actions=inputs["noisy"],
            flow_time=inputs["flow_time"],
            **extra,
        )
    pytest.skip(
        "dense_evidence/metric_tokens 尚未接入 encode_condition/forward"
        "（model agent 并行开发中，实现后本用例自动生效）"
    )


class TestDenseReadoutMtvj:
    def test_zero_init_equivalence(self) -> None:
        """dense_readout_mtvj=True（W_o 严格全零）时输出与 False 逐位一致。

        契约 §5：W_o 零初始化 → z 分支贡献恒 0；同时做正控制——扰动新增的
        全零参数（W_o）后输出必须改变，证明 dense 分支确实被消费。
        """
        _requires_dense_readout_mtvj()
        torch.manual_seed(1234)
        model_on = VACompoundPolicy(_mtvj_config(dense_readout_mtvj=True)).eval()
        torch.manual_seed(1234)
        model_off = VACompoundPolicy(_mtvj_config()).eval()
        # 基础路径参数逐位一致（构造 RNG 交错不影响：从 on 复制共享键）。
        shared = {
            key: value
            for key, value in model_on.state_dict().items()
            if key in model_off.state_dict()
        }
        model_off.load_state_dict(shared, strict=False)

        inputs = _policy_inputs()
        dense_kwargs = {
            "dense_evidence": inputs["dense_evidence"],
            "metric_tokens": inputs["metric_tokens"],
        }
        with torch.inference_mode():
            base_off = _call_policy(model_off, inputs)
            base_on = _call_policy(model_on, inputs)
            dense_on = _call_policy(model_on, inputs, **dense_kwargs)
        assert _exact_equal(base_off, base_on), (
            "dense_readout_mtvj=True 未传 dense 输入时也必须与 False 逐位一致"
        )
        assert _exact_equal(base_off, dense_on), (
            "W_o 必须严格零初始化：dense 分支（K_dense/V_dense 随机）初始输出"
            "必须与 False 路径逐位一致"
        )

        # 正控制：dense 分支必须真实存在且被消费。
        new_names = set(model_on.state_dict()) - set(model_off.state_dict())
        assert new_names, "dense_readout_mtvj=True 必须新增 dense 分支参数（W_K/W_V/W_o 等）"
        zero_new = [
            name
            for name in new_names
            if float(model_on.state_dict()[name].abs().sum()) == 0.0
        ]
        assert zero_new, "dense 分支必须存在严格全零参数（契约：W_o 零初始化）"
        with torch.no_grad():
            for name in zero_new:
                model_on.state_dict()[name].normal_()
        with torch.inference_mode():
            perturbed = _call_policy(model_on, inputs, **dense_kwargs)
        assert not _exact_equal(base_off, perturbed), (
            "扰动 W_o 后 dense 输入必须改变输出（验证 dense 分支被消费而非被忽略）"
        )


# ---------------------------------------------------------------------------
# 5) metric checkpoint 保存/加载 roundtrip（weights_only=True）
# ---------------------------------------------------------------------------

class TestMetricCheckpoint:
    def test_roundtrip_weights_only(self, tmp_path: Path) -> None:
        mm = _metric_visual_head()
        if not hasattr(mm, "LanguageMetricField"):
            pytest.skip("LanguageMetricField 尚未实现（metric agent 并行开发中）")
        config = {"lang_dim": LANG_DIM, "h_dim": H_DIM, "d_proj": 192, "n_roles": N_ROLES}
        head = mm.LanguageMetricField(**config)
        ckpt = {
            "config": config,
            "metric_head": head.state_dict(),
            "contract": "mt_vj_metric_field_v1",
        }
        # RelationStateEncoder 与 LanguageMetricField 同文件、由同一 agent 落地；
        # 若尚未落地则只 roundtrip metric_head（§8 的核心要求）。
        has_relation = hasattr(mm, "RelationStateEncoder")
        relation = None
        if has_relation:
            relation = mm.RelationStateEncoder(state_dim=4, d_model=D_MODEL)
            ckpt["relation_encoder"] = relation.state_dict()

        path = tmp_path / "metric_field.pt"
        torch.save(ckpt, path)
        loaded = torch.load(path, weights_only=True)
        assert loaded["contract"] == "mt_vj_metric_field_v1"
        assert set(loaded["config"]) == set(config)

        head2 = mm.LanguageMetricField(**loaded["config"])
        head2.load_state_dict(loaded["metric_head"], strict=True)
        inputs = metric_field_inputs()
        with torch.inference_mode():
            out1 = head(**inputs)
            out2 = head2(**inputs)
        for name in ("p", "visibility", "offset", "heatmap", "relation"):
            assert _exact_equal(getattr(out1, name), getattr(out2, name)), name

        if has_relation:
            relation2 = mm.RelationStateEncoder(state_dim=4, d_model=D_MODEL)
            relation2.load_state_dict(loaded["relation_encoder"], strict=True)
            g = torch.randn(2, 4)
            nu = torch.randn(2, 4)
            with torch.inference_mode():
                z1a, z1b = relation(g, nu)
                z2a, z2b = relation2(g, nu)
            assert z1a.shape == z2a.shape and z1a.shape[-1] == D_MODEL
            assert _exact_equal(z1a, z2a) and _exact_equal(z1b, z2b)


# ---------------------------------------------------------------------------
# 辅助契约校验（当前即可运行）
# ---------------------------------------------------------------------------

def test_dense_coords_matches_live_vjepa() -> None:
    """本文件内联的坐标公式必须与 va_compound.live_vjepa._dense_coords()
    逐位一致（metric head 测试用的 coords 约定）。"""
    from va_compound.live_vjepa import _dense_coords as live_dense_coords

    live = torch.from_numpy(live_dense_coords())  # live_vjepa 返回 np.ndarray
    assert _exact_equal(_dense_coords(), live)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
