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

from va_compound.model import VACompoundConfig


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


def test_dino_main_pool_preserves_row_major_grid() -> None:
    from train import _dino_main_online_encode

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


def test_train_eval_main_encode_equivalence() -> None:
    from train import _dino_main_online_encode
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
    from train import _main_vision_config_kwargs

    class Args:
        dino_main_vision = True
        main_vision_grid = 4
        main_vision_frames = 4

    kwargs = _main_vision_config_kwargs(Args())
    assert kwargs["main_vision_backbone"] == "dinov2_vitl14_reg4"
    assert kwargs["main_vision_dim"] == 1024
    assert kwargs["main_vision_tokens"] == 64

    class ArgsOff:
        dino_main_vision = False

    assert _main_vision_config_kwargs(ArgsOff()) == {}
