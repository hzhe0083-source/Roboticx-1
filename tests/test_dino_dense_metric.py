"""DINO-metric 路径测试（2026-08-15 用户决策：DINO-main 接回 MT-VJ dense+metric）。

覆盖：
1. _dino_main_online_encode(return_dense=True) 的 dense evidence 形状/帧序
   （前 256 = d-2、后 256 = d；block5→key5、block11→key11）；
2. 训练/评测两帧 dense evidence 输出一致；
3. LanguageMetricField(grid=16, h_dim=1024) 前向形状 + spatial_bias 网格；
4. 旧 V-JEPA metric head 构造契约（无 grid 键）默认 grid=24 逐字节兼容；
5. DenseEvidenceProjector(vision_dim=1024) 512-token evidence + Δt 方向；
6. VACompoundConfig dino_dense_metric 组合校验；
7. _main_vision_config_kwargs 新 flag → dense_readout_mtvj/dino_dense_metric。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from va_compound.model import (
    DenseEvidenceProjector,
    VACompoundConfig,
    dense_coords,
)


class FakeDinoBackbone:
    """假冻结塔：block 特征按 patch 位置编码（内容无关）。"""

    image_size = 224
    feature_dim = 1024

    def forward_hierarchical_dense(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        n = int(images.shape[0])
        tokens = torch.zeros(n, 256, self.feature_dim)
        tokens[..., 0] = torch.arange(256).view(1, 256)
        return {5: tokens + 1.0, 11: tokens + 11.0}


def _frames(b=1, t=1, w=4):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(b, t, w, 224, 224, 3), dtype=np.uint8)


def _dino_dense_config(**overrides) -> VACompoundConfig:
    base = dict(
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        main_vision_image_size=224,
        main_vision_dim=1024,
        main_vision_grid=8,
        main_vision_frames=4,
        main_vision_tokens=256,
        dense_readout_mtvj=True,
        dino_dense_metric=True,
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def test_dino_dense_evidence_shapes_and_frame_order() -> None:
    from train import _dino_main_online_encode

    backbone = FakeDinoBackbone()
    frames = _frames(2, 2, 4)  # B=2, T=2, W=4
    tokens, dense = _dino_main_online_encode(
        frames, backbone, torch.device("cpu"),
        encode_batch=3, grid=8, window=4, return_dense=True,
    )
    assert tuple(tokens.shape) == (2, 2, 256, 1024)
    assert sorted(dense) == [5, 11]
    for layer in (5, 11):
        assert tuple(dense[layer].shape) == (2, 2, 512, 1024)
    # 位置编码通道：前 256 个 token 是帧 d-2（w=2）→ channel0 = 0..255；
    # 后 256 个是帧 d（w=3）→ 同样 0..255（块内重排为帧内索引）。
    ev = dense[11][0, 0]
    first_half = ev[:256, 0]
    second_half = ev[256:, 0]
    assert torch.equal(first_half, second_half)
    assert first_half[0].item() == pytest.approx(11.0)  # 第 0 个 patch + 11
    assert first_half[255].item() == pytest.approx(255.0 + 11.0)
    # 两层内容区分（key5 = block11 特征）。
    assert torch.equal(dense[5][0, 0] - 1.0, dense[11][0, 0] - 11.0)


def test_train_eval_dense_evidence_equivalence() -> None:
    from train import _dino_main_online_encode
    from eval_metaworld import _main_vision_encode_window

    backbone = FakeDinoBackbone()
    frames = _frames(1, 1, 4)
    _, train_dense = _dino_main_online_encode(
        frames, backbone, torch.device("cpu"),
        encode_batch=2, grid=8, window=4, return_dense=True,
    )
    frame_list = [frames[0, 0, i] for i in range(4)]
    _, eval_dense = _main_vision_encode_window(
        frame_list, backbone, torch.device("cpu"),
        grid=8, window=4, return_dense=True,
    )
    for layer in (5, 11):
        assert torch.equal(train_dense[layer][0, 0], eval_dense[layer][0])


def test_metric_head_grid16_forward_shapes() -> None:
    from va_compound.metric_visual_head import LanguageMetricField

    head = LanguageMetricField(lang_dim=32, h_dim=1024, d_proj=64, grid=16)
    assert head.dense_tokens == 512
    assert tuple(head.spatial_bias.shape) == (4, 2, 16, 16)
    b = 2
    h5 = torch.randn(b, 512, 1024)
    h11 = torch.randn(b, 512, 1024)
    lang = torch.randn(b, 5, 32)
    mask = torch.ones(b, 5, dtype=torch.bool)
    coords = dense_coords(512)
    out = head(h5, h11, lang, mask, coords)
    assert tuple(out.p.shape) == (b, 4, 2)
    assert tuple(out.heatmap.shape) == (b, 4, 16, 16)
    assert tuple(out.visibility.shape) == (b, 4)
    assert tuple(out.offset_full.shape) == (b, 4, 512, 2)
    # 零初始化空间偏置 → 初始 p̂ 接近网格中心（mode_readout 下 5×5 窗口内）。
    head_mode = LanguageMetricField(
        lang_dim=32, h_dim=1024, d_proj=64, grid=16, mode_readout=True
    )
    out_mode = head_mode(h5, h11, lang, mask, coords)
    assert torch.all((out_mode.p >= 0.0) & (out_mode.p <= 1.0))


def test_metric_head_grid16_wrong_tokens_rejected() -> None:
    from va_compound.metric_visual_head import LanguageMetricField

    head = LanguageMetricField(lang_dim=32, h_dim=1024, d_proj=64, grid=16)
    with pytest.raises(ValueError, match="512"):
        head(torch.randn(1, 1152, 1024), torch.randn(1, 1152, 1024),
             torch.randn(1, 5, 32), torch.ones(1, 5, dtype=torch.bool),
             dense_coords(1152))


def test_legacy_metric_contract_defaults_to_grid24() -> None:
    from train import (
        _canonical_mtvj_metric_head_config,
        _mtvj_metric_head_constructor_config,
    )
    from va_compound.metric_visual_head import LanguageMetricField

    # 旧 checkpoint config（无 grid 键）→ 默认 24，行为逐字节不变。
    legacy = {
        "lang_dim": 2048, "h_dim": 768, "d_proj": 192, "n_roles": 4,
        "l2_norm": False, "learnable_temp": False, "temp_init": 10.0,
        "freeze_bias": False, "mode_readout": False,
    }
    canonical = _canonical_mtvj_metric_head_config(legacy, require_complete=True)
    assert canonical["grid"] == 24
    head = LanguageMetricField(**canonical)
    # 构造语义可完整回读（含 grid），供 checkpoint 保存。
    roundtrip = _mtvj_metric_head_constructor_config(head)
    assert roundtrip["grid"] == 24
    assert tuple(head.spatial_bias.shape) == (4, 2, 24, 24)


def test_dense_projector_dino_dim_and_dt_direction() -> None:
    proj = DenseEvidenceProjector(vision_dim=1024, hidden_dim=512)
    b = 2
    h5 = torch.randn(b, 512, 1024)
    h11 = torch.randn(b, 512, 1024)
    metric_tokens = torch.randn(b, 2, 512)
    out = proj({5: h5, 11: h11}, metric_tokens)
    assert tuple(out.d.shape) == (b, 512, 192)
    assert tuple(out.g.shape) == (b, 512, 192)
    assert tuple(out.t.shape) == (b, 512, 192)
    assert tuple(out.coord_k.shape) == (512, 512)
    # Δt = 后 256（d 帧）− 前 256（d-2 帧），按片复制回 512。
    expected_t = proj.proj_t(h11[:, 256:] - h11[:, :256])
    assert torch.allclose(out.t[:, :256], expected_t, atol=1e-6)
    assert torch.allclose(out.t[:, 256:], expected_t, atol=1e-6)


def test_dense_projector_rejects_wrong_dim() -> None:
    proj = DenseEvidenceProjector(vision_dim=1024, hidden_dim=512)
    with pytest.raises(ValueError, match="vision_dim"):
        proj({5: torch.randn(1, 512, 768), 11: torch.randn(1, 512, 768)}, None)


def test_config_dino_dense_metric_validation() -> None:
    _dino_dense_config()  # 合法组合不抛
    with pytest.raises(ValueError, match="main_vision_backbone"):
        _dino_dense_config(main_vision_backbone="vjepa")
    with pytest.raises(ValueError, match="dense_readout_mtvj"):
        _dino_dense_config(dense_readout_mtvj=False)


def test_main_vision_config_kwargs_dino_metric() -> None:
    from train import _main_vision_config_kwargs

    class Args:
        dino_main_vision = True
        dino_dense_metric = True
        main_vision_grid = 8
        main_vision_frames = 4

    kwargs = _main_vision_config_kwargs(Args())
    assert kwargs["dense_readout_mtvj"] is True
    assert kwargs["dino_dense_metric"] is True
    assert kwargs["main_vision_dim"] == 1024

    class ArgsNoMetric:
        dino_main_vision = True
        dino_dense_metric = False
        main_vision_grid = 8
        main_vision_frames = 4

    kwargs = _main_vision_config_kwargs(ArgsNoMetric())
    assert "dense_readout_mtvj" not in kwargs
    assert "dino_dense_metric" not in kwargs
