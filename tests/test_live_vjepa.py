"""PULSE-VA Stage B：live-vjepa 路径单元测试（encode 形状 / 槽坐标 / 参数校验）。"""
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from va_compound.live_vjepa import (
    N_DENSE_TOKENS,
    N_TOKENS,
    SEQUENCE_LENGTH,
    VISION_WINDOW,
    _dense_coords,
    _slot_coords,
    encode_live_frames,
)


class _FakeBackbone:
    """伪 V-JEPA：_encode → [B, 2*grid*grid, D] 扁平 tokens（真实 2.1 契约）；
    _pool(spatiotemporal) → [B, 288, D]。"""

    def __init__(self, dim: int = 768, grid: int = 12) -> None:
        self.dim = dim
        self.grid = grid
        self.device = torch.device("cpu")
        self.dtype = torch.float16

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        B = inputs.shape[0]
        return torch.randn(B, 2 * self.grid * self.grid, self.dim, dtype=self.dtype)

    def _pool(self, raw: torch.Tensor, pooling: str) -> torch.Tensor:
        assert pooling == "spatiotemporal"
        return raw.reshape(raw.shape[0], N_TOKENS, self.dim)


class TestSlotCoords:
    def test_shape_and_range(self) -> None:
        coords = _slot_coords()
        assert coords.shape == (N_TOKENS, 3)
        assert coords.dtype == np.float32
        assert coords[:, :].min() >= -1.0 - 1e-6
        assert coords[:, :].max() <= 1.0 + 1e-6
        # 与 ST288 提取一致：(t∈{-1,1}, y, x)，t 外层循环
        assert int((coords[:, 0] == -1.0).sum()) == 144
        assert int((coords[:, 0] == 1.0).sum()) == 144
        assert np.allclose(coords[:3], [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0 + 2.0 / 11.0], [-1.0, -1.0, -1.0 + 4.0 / 11.0]])

    def test_matches_prepare_build_coords(self) -> None:
        """与 prepare_mw_local_features.build_coords 逐位一致（Codex P0-5 回归）。"""
        import subprocess
        import sys

        # 直接内联同一实现（避免 import 副作用）：语义同源，数值必须相等。
        half = (12 - 1) / 2
        expected = []
        for t in range(2):
            for y in range(12):
                for x in range(12):
                    expected.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
        assert np.allclose(_slot_coords(), np.asarray(expected, dtype=np.float32))


class TestEncodeLiveFrames:
    def test_shapes(self) -> None:
        B, T, W, S = 2, SEQUENCE_LENGTH, VISION_WINDOW, 384
        frames = (np.random.rand(B, T, W, S, S, 3) * 255).astype(np.uint8)
        backbone = _FakeBackbone()
        out = encode_live_frames(frames, backbone, torch.device("cpu"))
        assert tuple(out.shape) == (B, T, N_TOKENS, 768)

    def test_single_sample(self) -> None:
        T, W, S = SEQUENCE_LENGTH, VISION_WINDOW, 384
        frames = np.zeros((1, T, W, S, S, 3), dtype=np.uint8)
        out = encode_live_frames(frames, _FakeBackbone(dim=512), torch.device("cpu"))
        assert tuple(out.shape) == (1, T, N_TOKENS, 512)

    def test_requires_no_grad_when_frozen(self) -> None:
        """冻结 backbone：输出应无梯度要求（编码为纯前向）。"""
        B, T, W, S = 1, SEQUENCE_LENGTH, VISION_WINDOW, 384
        frames = np.zeros((B, T, W, S, S, 3), dtype=np.uint8)
        out = encode_live_frames(frames, _FakeBackbone(), torch.device("cpu"))
        assert not out.requires_grad


class TestDenseReadout:
    """Step 0 dense readout：不池化分支 + [1152, 3] 全量 patch 网格坐标。"""

    def test_dense_coords_shape_range_order(self) -> None:
        coords = _dense_coords()
        assert coords.shape == (N_DENSE_TOKENS, 3)
        assert coords.dtype == np.float32
        assert coords.min() >= -1.0 - 1e-6
        assert coords.max() <= 1.0 + 1e-6
        # (t∈{-1,1}, y, x)，t 外层循环：每个时间片 24×24=576 patch。
        assert int((coords[:, 0] == -1.0).sum()) == N_DENSE_TOKENS // 2
        assert int((coords[:, 0] == 1.0).sum()) == N_DENSE_TOKENS // 2
        # 与 _slot_coords(grid=24) 逐位一致（同一生成器，仅网格尺寸不同）。
        assert np.allclose(coords, _slot_coords(grid=24))
        # y/x 步长为 2/23（24 格归一化），与 12 格（2/11）同一公式。
        assert np.allclose(
            coords[:3],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0 + 2.0 / 23.0], [-1.0, -1.0, -1.0 + 4.0 / 23.0]],
        )

    def test_encode_live_frames_dense(self) -> None:
        B, T, W, S = 2, SEQUENCE_LENGTH, VISION_WINDOW, 384
        frames = (np.random.rand(B, T, W, S, S, 3) * 255).astype(np.uint8)
        out = encode_live_frames(
            frames, _FakeBackbone(grid=24), torch.device("cpu"), dense=True
        )
        assert tuple(out.shape) == (B, T, N_DENSE_TOKENS, 768)

    def test_dense_single_sample(self) -> None:
        T, W, S = SEQUENCE_LENGTH, VISION_WINDOW, 384
        frames = np.zeros((1, T, W, S, S, 3), dtype=np.uint8)
        out = encode_live_frames(
            frames, _FakeBackbone(dim=512, grid=24), torch.device("cpu"), dense=True
        )
        assert tuple(out.shape) == (1, T, N_DENSE_TOKENS, 512)

    def test_dense_rejects_wrong_layout(self) -> None:
        """raw token 数不是 1152（如 288 路径的 12×12 池化网格）必须报错。"""
        B, T, W, S = 1, SEQUENCE_LENGTH, VISION_WINDOW, 384
        frames = np.zeros((B, T, W, S, S, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="raw tokens"):
            encode_live_frames(
                frames, _FakeBackbone(grid=12), torch.device("cpu"), dense=True
            )
