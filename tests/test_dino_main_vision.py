"""DINO-main replacement 路径测试（2026-08-14 用户决策：DINOv2 替换 V-JEPA）。

覆盖：
1. VACompoundConfig main_vision_* 字段默认值与往返/校验；
2. _dino_main_online_encode 的 patch 网格平均池化保持行主序空间结构
   （假塔返回位置编码 token，验证 16x16 → grid x grid 分组正确）；
3. 训练侧 _dino_main_online_encode 与评测侧 _main_vision_encode_window
   对同一决策窗口输出一致（形状与展平对齐）。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy


class FakeDinoBackbone:
    """假冻结塔：返回按 patch 网格位置编码的 token（输入内容无关）。"""

    image_size = 224
    feature_dim = 1024

    def forward_hierarchical_dense(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        n = int(images.shape[0])
        tokens = torch.zeros(n, 256, self.feature_dim)
        # channel 0 = 行主序线性索引；channel 1 = 行号；channel 2 = 列号。
        tokens[..., 0] = torch.arange(256).view(1, 256)
        tokens[..., 1] = torch.arange(16).repeat_interleave(16).view(1, 256)
        tokens[..., 2] = torch.arange(16).repeat(16).view(1, 256)
        return {5: tokens.clone(), 11: tokens}

    def forward_final_dense(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_hierarchical_dense(images)[11]

    def forward_last_four_mean_dense(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_final_dense(images) + 1000.0

    def forward_last_four_dense(self, images: torch.Tensor) -> torch.Tensor:
        base = self.forward_final_dense(images)
        return torch.stack([base + 1000.0 * index for index in range(4)], dim=1)

    def forward_last_layers_dense(
        self, images: torch.Tensor, count: int
    ) -> torch.Tensor:
        base = self.forward_final_dense(images)
        return torch.stack(
            [base + 1000.0 * index for index in range(count)], dim=1
        )


def _dino_config(**overrides) -> VACompoundConfig:
    base = dict(
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        main_vision_image_size=224,
        main_vision_dim=1024,
        main_vision_grid=8,
        main_vision_frames=4,
        main_vision_tokens=256,
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def test_main_vision_config_defaults_and_roundtrip() -> None:
    default = VACompoundConfig()
    assert default.main_vision_backbone == "vjepa"
    assert default.vision_dim == 768  # 旧路径逐位不变
    dino = _dino_config()
    restored = VACompoundConfig(**dino.__dict__)
    assert restored.main_vision_backbone == "dinov2_vitl14_reg4"
    assert restored.main_vision_tokens == 256


def test_main_vision_config_validation() -> None:
    with pytest.raises(ValueError, match="grid\\*grid\\*frames"):
        _dino_config(main_vision_tokens=255)
    with pytest.raises(ValueError, match="main_vision_grid"):
        _dino_config(main_vision_grid=0)
    with pytest.raises(ValueError, match="main_vision_frames"):
        _dino_config(main_vision_frames=0)
    with pytest.raises(ValueError, match="main_vision_model_id"):
        _dino_config(main_vision_model_id="")
    with pytest.raises(ValueError, match="incompatible"):
        _dino_config(
            dino_qwen_cross_modal_bridge=True,
            dino_dense_metric=True,
        )


def test_online_encoder_selects_dino_last_four_mean() -> None:
    from va_compound.vision.encoding import _dino_main_online_encode

    frames = np.zeros((1, 1, 4, 32, 32, 3), dtype=np.uint8)
    base = _dino_main_online_encode(
        frames,
        FakeDinoBackbone(),
        torch.device("cpu"),
        encode_batch=4,
        grid=16,
        window=4,
    )
    fused = _dino_main_online_encode(
        frames,
        FakeDinoBackbone(),
        torch.device("cpu"),
        encode_batch=4,
        grid=16,
        window=4,
        last_four_mean=True,
    )
    torch.testing.assert_close(fused, base + 1000.0)


def test_online_encoder_keeps_dino_last_four_layers_separate() -> None:
    from va_compound.vision.encoding import _dino_main_online_encode

    frames = np.zeros((1, 1, 4, 32, 32, 3), dtype=np.uint8)
    base, layers = _dino_main_online_encode(
        frames,
        FakeDinoBackbone(),
        torch.device("cpu"),
        encode_batch=4,
        grid=16,
        window=4,
        return_last_four=True,
    )
    assert layers.shape == (1, 1, 4, 1024, 1024)
    torch.testing.assert_close(base, layers[:, :, -1].float())
    for index in range(1, 4):
        torch.testing.assert_close(
            layers[:, :, index] - layers[:, :, index - 1],
            torch.full_like(layers[:, :, index], 1000.0),
        )


def test_online_encoder_keeps_dino_last_six_layers_separate() -> None:
    from va_compound.vision.encoding import _dino_main_online_encode

    frames = np.zeros((1, 1, 4, 32, 32, 3), dtype=np.uint8)
    base, layers = _dino_main_online_encode(
        frames,
        FakeDinoBackbone(),
        torch.device("cpu"),
        encode_batch=4,
        grid=16,
        window=4,
        return_last_layers=6,
    )
    assert layers.shape == (1, 1, 6, 1024, 1024)
    torch.testing.assert_close(base, layers[:, :, -1].float())


def test_temporal_embedding_breaks_frame_permutation_invariance() -> None:
    config = _dino_config(
        language_dim=12,
        vision_dim=1024,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
        main_vision_grid=2,
        main_vision_tokens=16,
        main_vision_temporal=True,
    )
    torch.manual_seed(7)
    model = VACompoundPolicy(config).eval()
    # The four frame blocks contain distinct content.  Without an explicit frame
    # code, reversing whole blocks is a pure token permutation to attention.
    vision = torch.stack(
        [torch.full((4, 1024), float(frame)) for frame in range(4)], dim=0
    ).reshape(1, 16, 1024)
    reversed_vision = vision.reshape(1, 4, 4, 1024).flip(1).reshape_as(vision)
    language = torch.randn(1, 3, 12)
    mask = torch.ones(1, 3, dtype=torch.bool)
    cache = model.build_language_cache(language, mask)
    kwargs = dict(
        proprio=torch.zeros(1, 5),
        previous_action=torch.zeros(1, 4),
        language_cache=cache,
    )
    full = model.encode_condition(vision, **kwargs)
    reversed_condition = model.encode_condition(reversed_vision, **kwargs)
    assert not torch.allclose(full, reversed_condition, atol=1e-7, rtol=1e-7)


def test_temporal_embedding_disabled_preserves_old_state_dict_contract() -> None:
    config = _dino_config(
        language_dim=12,
        vision_dim=1024,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
    )
    model = VACompoundPolicy(config)
    assert model.main_vision_frame_embedding is None
    assert not any(
        key.startswith("main_vision_frame_embedding.") for key in model.state_dict()
    )


def test_dino_main_pool_preserves_row_major_grid() -> None:
    from va_compound.vision.encoding import _dino_main_online_encode

    backbone = FakeDinoBackbone()
    # [B=1, T=1, W=4, H=224, W=224, 3] 随机帧（假塔忽略内容）
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(1, 1, 4, 224, 224, 3), dtype=np.uint8)
    tokens = _dino_main_online_encode(
        frames, backbone, torch.device("cpu"),
        encode_batch=2, grid=8, window=4,
    )
    assert tuple(tokens.shape) == (1, 1, 256, 1024)
    grid = tokens.reshape(1, 1, 4, 8, 8, 1024)
    # 每帧第 0 通道 = 行主序线性索引经 2x2 块平均；行/列通道校验空间结构。
    for f in range(4):
        frame = grid[0, 0, f]
        for gy in range(8):
            for gx in range(8):
                block_idx = []
                for dy in range(2):
                    for dx in range(2):
                        r, c = 2 * gy + dy, 2 * gx + dx
                        block_idx.append(r * 16 + c)
                assert frame[gy, gx, 0].item() == pytest.approx(float(np.mean(block_idx)))
                assert frame[gy, gx, 1].item() == pytest.approx(2 * gy + 0.5)
                assert frame[gy, gx, 2].item() == pytest.approx(2 * gx + 0.5)


class ContentSensitiveFakeDino:
    """Fake tower that records per-image mean so frame order is testable."""

    image_size = 224
    feature_dim = 1024

    def forward_hierarchical_dense(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        n = int(images.shape[0])
        tokens = torch.zeros(n, 256, self.feature_dim)
        mean = images.float().mean(dim=(1, 2, 3))
        tokens[..., 0] = mean.view(n, 1)
        tokens[..., 3] = images[:, 0, 0, 0].float().view(n, 1)
        return {5: tokens.clone(), 11: tokens}


def test_eval_one_frame_encode_matches_batched_window() -> None:
    from eval_metaworld import _main_vision_encode_window, preprocess

    backbone = ContentSensitiveFakeDino()
    frames = [
        np.full((224, 224, 3), fill, dtype=np.uint8) for fill in (10, 40, 80, 160)
    ]
    tokens, dense = _main_vision_encode_window(
        frames,
        backbone,
        torch.device("cpu"),
        grid=16,
        window=4,
        return_dense=True,
    )
    batched = backbone.forward_hierarchical_dense(
        torch.cat([preprocess(frame, 224) for frame in frames], dim=0)
    )
    assert torch.allclose(
        tokens[0].reshape(4, 256, 1024)[..., 0], batched[11][..., 0], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(dense[11][0, :256, 0], batched[11][2, :, 0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(dense[11][0, 256:, 0], batched[11][3, :, 0], atol=1e-6, rtol=1e-6)
    assert float(tokens[0, 0, 0]) != float(tokens[0, 256, 0])


def test_train_eval_main_encode_equivalence() -> None:
    from va_compound.vision.encoding import _dino_main_online_encode
    from eval_metaworld import _main_vision_encode_window

    backbone = FakeDinoBackbone()
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, size=(1, 1, 4, 224, 224, 3), dtype=np.uint8)
    train_tokens = _dino_main_online_encode(
        frames, backbone, torch.device("cpu"),
        encode_batch=2, grid=8, window=4,
    )[0, 0]  # [256, 1024]
    # 评测侧输入：单决策 4 帧列表（与闭环 frame_buffer 切片一致）。
    frame_list = [frames[0, 0, i] for i in range(4)]
    eval_tokens = _main_vision_encode_window(
        frame_list, backbone, torch.device("cpu"), grid=8, window=4
    )[0]  # [256, 1024]
    assert torch.equal(train_tokens, eval_tokens)


def test_main_vision_vision_dim_override_via_kwargs() -> None:
    from va_compound.training.model_setup import _main_vision_config_kwargs

    class Args:
        dino_main_vision = True
        main_vision_grid = 4
        main_vision_frames = 4
        main_vision_temporal = False
        main_vision_temporal_scale = 1.0
        dino_dense_metric = False

    kwargs = _main_vision_config_kwargs(Args())
    assert kwargs["main_vision_backbone"] == "dinov2_vitl14_reg4"
    assert kwargs["main_vision_dim"] == 1024
    assert kwargs["main_vision_tokens"] == 64

    class ArgsOff:
        dino_main_vision = False

    assert _main_vision_config_kwargs(ArgsOff()) == {}


def test_timm_dino_full_unfreeze_enables_gradients_and_checkpointing() -> None:
    from va_compound.backbones import TimmActionVisionBackbone

    class TinyTimm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 4)
            self.grad_checkpointing = False

        def set_grad_checkpointing(self, enabled: bool) -> None:
            self.grad_checkpointing = enabled

        def get_intermediate_layers(self, images, **_kwargs):
            tokens = self.projection(images.mean(dim=(-1, -2)))[:, None, :]
            prefixes = tokens[:, :0]
            return [(tokens, prefixes), (tokens + 1.0, prefixes)]

    model = TinyTimm()
    backbone = TimmActionVisionBackbone(
        model,
        model_id="tiny",
        image_size=2,
        feature_dim=4,
        output_layers=(0, 1),
    )
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    backbone.unfreeze_all()
    assert model.grad_checkpointing
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    output = backbone.forward_hierarchical_dense(torch.randn(2, 3, 2, 2))[11]
    output.sum().backward()
    assert model.projection.weight.grad is not None
