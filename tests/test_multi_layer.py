"""C²-IRF v2 Step 4：多层输出（encode_multi）。

设计文档（artifacts/c2irf_v2_vision_ablation.md §五/§七/§九）：H⁵/H¹¹ 一次
前向取回（官方原生 out_layers，返回列表顺序与 out_layers 一致）；V-JEPA 权重
冻结，多层输出只作只读 evidence。全部用例 CPU 可跑——伪官方 V-JEPA 模拟
blocks/norm/out_layers 契约，不加载真实 checkpoint。

高频残差部件（``HighFreqResidual``/``ResidualValueConcat``）已归档到
``archives/c2irf_v2_step4/``。
"""
from __future__ import annotations

import unittest
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from torch import nn

from va_compound.backbones import VJEPA21Backbone
from va_compound.live_vjepa import (
    N_DENSE_TOKENS,
    N_TOKENS,
    SEQUENCE_LENGTH,
    VISION_WINDOW,
    encode_live_frames,
)

DIM = 16  # 伪 V-JEPA 特征维（越小越快，CPU 即可跑）


class _MarkerBlock(nn.Module):
    """向全部 token 加一个层标记常数（区分层输出，模拟每层变换）。"""

    def __init__(self, marker: float) -> None:
        super().__init__()
        self.marker = marker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.marker


class _FakeVJEPA(nn.Module):
    """按官方 V-JEPA 2.1 forward 语义的伪模型（12 blocks，384→24×24 patch 网格）。

    官方契约：``self.out_layers`` 非 None 时按 block 升序收集
    ``self.norm(x)`` 并返回列表（跳过末尾 norm）；norm 用 Identity 代替
    LayerNorm（排序语义不变，数值断言更直接）。
    """

    def __init__(self, num_blocks: int = 12, dim: int = DIM) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = 16
        self.tubelet_size = 2
        self.num_frames = 4
        self.img_height = 384
        self.img_width = 384
        self.out_layers = None
        self.norm = nn.Identity()
        self.blocks = nn.ModuleList(
            [_MarkerBlock(0.1 * (i + 1)) for i in range(num_blocks)]
        )
        self.scale = nn.Parameter(torch.tensor(1.0))  # 可训练参数占位（冻结断言用）

    def forward(self, videos: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        batch = videos.shape[0]
        x = torch.arange(2 * 24 * 24 * self.dim, dtype=torch.float32).view(
            1, 1152, self.dim
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


def make_videos(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 4, 3, 16, 16)


class TestEncodeMulti(unittest.TestCase):
    def _backbone(self) -> VJEPA21Backbone:
        backbone = VJEPA21Backbone(_FakeVJEPA())
        backbone.freeze_all()  # 生产路径（from_pretrained）等价
        return backbone

    def test_returns_list_with_shapes_and_order(self) -> None:
        """默认 (5, 11)：返回列表、逐层 [B, 1152, D]、顺序 = H⁵ 在前 H¹¹ 在后。"""
        backbone = self._backbone()
        layers = backbone.encode_multi(make_videos(2))
        assert isinstance(layers, list)
        assert len(layers) == 2
        assert all(tuple(layer.shape) == (2, 1152, DIM) for layer in layers)
        # block 11 后的 marker 累加和比 block 5 后大 5.7（0.1*(78−21)）；
        # fp32 累加有 ~1e-4 舍入，放宽 atol。
        diff = layers[1] - layers[0]
        assert torch.allclose(diff, torch.full_like(diff, 0.1 * (78 - 21)), atol=1e-3)

    def test_matches_official_forward_semantics(self) -> None:
        """与官方 forward 语义逐位一致：按 block 升序收集 norm(x) 并返回列表。"""
        backbone = self._backbone()
        model = backbone.model
        videos = make_videos(2)
        got = backbone.encode_multi(videos, out_layers=(5, 11))
        x = (
            torch.arange(2 * 24 * 24 * DIM, dtype=torch.float32)
            .view(1, 1152, DIM)
            .repeat(2, 1, 1)
            * model.scale
        )
        expected = []
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i in (5, 11):
                expected.append(model.norm(x))
        assert len(expected) == 2
        for a, b in zip(got, expected):
            assert torch.allclose(a, b)

    def test_non_ascending_out_layers_reordered_to_given_order(self) -> None:
        """契约"返回值顺序与 out_layers 一致"：非升序 (11, 5) 返回 [H¹¹, H⁵]。"""
        backbone = self._backbone()
        layers = backbone.encode_multi(make_videos(1), out_layers=(11, 5))
        assert len(layers) == 2
        diff = layers[0] - layers[1]
        assert torch.allclose(diff, torch.full_like(diff, 0.1 * (78 - 21)), atol=1e-3)

    def test_set_out_layers_canonicalized_ascending(self) -> None:
        """无序集合 {11, 5} 规范为升序：无论集合迭代顺序，返回 [H⁵, H¹¹]。"""
        backbone = self._backbone()
        layers = backbone.encode_multi(make_videos(1), out_layers={11, 5})
        diff = layers[0] - layers[1]
        assert torch.allclose(diff, torch.full_like(diff, -0.1 * (78 - 21)), atol=1e-3)

    def test_out_layers_restored_and_default_path_unchanged(self) -> None:
        """临时改写 model.out_layers 后恢复：缺省 forward/_encode 逐字节不变。"""
        backbone = self._backbone()
        model = backbone.model
        assert model.out_layers is None
        videos = make_videos(2)
        before = backbone(videos, pooling="dense")
        assert isinstance(before, torch.Tensor)
        assert tuple(before.shape) == (2, 1152, DIM)
        backbone.encode_multi(videos, out_layers=(5, 11))
        assert model.out_layers is None  # 临时改写已恢复
        assert torch.equal(backbone(videos, pooling="dense"), before)
        assert torch.equal(backbone._encode(videos), before)

    def test_weights_frozen_and_unchanged(self) -> None:
        """冻结断言：encode_multi 不改任何 V-JEPA 权重，requires_grad 保持 False。"""
        backbone = VJEPA21Backbone(_FakeVJEPA())
        model = backbone.model
        params = list(model.parameters())
        assert all(p.requires_grad for p in params)  # 前置：伪模型参数默认可训练
        model.requires_grad_(False).eval()
        snapshot = [p.detach().clone() for p in params]
        with torch.no_grad():
            layers = backbone.encode_multi(make_videos(2), out_layers=(5, 11))
        assert all(not layer.requires_grad for layer in layers)
        for p, snap in zip(params, snapshot):
            assert torch.equal(p.detach(), snap)  # 权重未被改动
            assert not p.requires_grad  # 冻结保持

    def test_validation_errors(self) -> None:
        backbone = self._backbone()
        videos = make_videos(2)
        with pytest.raises(ValueError, match="non-empty"):
            backbone.encode_multi(videos, out_layers=())
        with pytest.raises(ValueError, match="duplicates"):
            backbone.encode_multi(videos, out_layers=(5, 5))
        with pytest.raises(ValueError, match="ints"):
            backbone.encode_multi(videos, out_layers=(5, True))
        with pytest.raises(ValueError, match=r"in \[0, 11\]"):
            backbone.encode_multi(videos, out_layers=(12,))
        with pytest.raises(ValueError, match="even number"):
            backbone.encode_multi(torch.randn(2, 3, 3, 16, 16))  # 帧数非偶




class _FakeBackboneMulti:
    """encode_multi 透传伪 backbone：每层输出为常数 layer index（可验序）。"""

    def __init__(self, dim: int = 768, grid: int = 24) -> None:
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

    def encode_multi(
        self, inputs: torch.Tensor, out_layers: Sequence[int] = (5, 11)
    ) -> list[torch.Tensor]:
        B = inputs.shape[0]
        return [
            torch.full(
                (B, 2 * self.grid * self.grid, self.dim), float(layer), dtype=self.dtype
            )
            for layer in out_layers
        ]


def make_frames(batch: int = 2) -> np.ndarray:
    T, W, S = SEQUENCE_LENGTH, VISION_WINDOW, 384
    return (np.random.rand(batch, T, W, S, S, 3) * 255).astype(np.uint8)


class TestEncodeLiveFramesMulti(unittest.TestCase):
    def test_out_layers_dense_returns_list_in_order(self) -> None:
        """dense + out_layers：返回列表，逐层 [B, T, 1152, D]，顺序与 out_layers 一致。"""
        frames = make_frames(2)
        out = encode_live_frames(
            frames, _FakeBackboneMulti(grid=24), torch.device("cpu"),
            dense=True, out_layers=(5, 11),
        )
        assert isinstance(out, list) and len(out) == 2
        assert tuple(out[0].shape) == (2, SEQUENCE_LENGTH, N_DENSE_TOKENS, 768)
        assert float(out[0][0, 0, 0, 0]) == 5.0
        assert float(out[1][0, 0, 0, 0]) == 11.0

    def test_out_layers_spatiotemporal_pooled(self) -> None:
        """非 dense + out_layers：逐层折叠成 [B, T, 288, D]（ST288 池化规则）。"""
        frames = make_frames(2)
        out = encode_live_frames(
            frames, _FakeBackboneMulti(grid=12), torch.device("cpu"), out_layers=(5, 11)
        )
        assert isinstance(out, list) and len(out) == 2
        assert tuple(out[0].shape) == (2, SEQUENCE_LENGTH, N_TOKENS, 768)

    def test_default_path_unchanged(self) -> None:
        """out_layers=None：返回单个 Tensor（既有行为逐字节不变）。"""
        frames = make_frames(2)
        out = encode_live_frames(frames, _FakeBackboneMulti(grid=12), torch.device("cpu"))
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (2, SEQUENCE_LENGTH, N_TOKENS, 768)


if __name__ == "__main__":
    unittest.main()
