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


class TestArgValidation:
    def _args(self, **overrides) -> argparse.Namespace:
        from train import parse_args

        # 真实 parser 默认值打底，避免每次 validate_args 新读一个字段就要手工补夹具。
        base = vars(parse_args([]))
        base.update(
            steps=100, flow_steps=4, lr=3e-5, pair_loss_weight=1.0,
            single_task=True, batch_size=8, evsm=False, memory_split=False,
            future_predict=False, future_predict_weight=0.1,
            evsm_temp=1.0, evsm_kappa=1.0, compile_task=False, compile_every=4,
            compile_n_readout=16, language_max_length=64,
            scene_teacher=False, plan_resampler=False, training_stage="",
            semantic_adapter=False, semantic_anchor_weight=0.0,
            semantic_geometry_weight=0.0, semantic_act_grad_scale=0.1,
            semantic_top_layers=4, semantic_lora_suffixes="q_proj",
            dual_attention=False, flow_semantic=False, flow_cond="entry",
            action_query_cond=False, role_query=False, role_query_tokens=16,
            sequential_coupling=0, qwen_unfreeze_blocks=0, lora_rank=0,
            c2_lambda_c=0.1, c2_recovery_ratio=0.25,
            direct_head=True, c2_controller=False, data=argparse.Namespace(),
            e2e_data=None, local_slots_data=None, live_vjepa=False,
            dense_readout=False, multi_mode=False, local_slots_direct288=False,
            fork_data=None, fork_k=83, fork_skip_contract=False,
            live_root=argparse.Namespace(), vision_unfreeze_all=False,
            vision_unfreeze_last=0, vision_pooling="spatiotemporal",
            sam_rho=0.0,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_live_requires_data_single_direct(self) -> None:
        from train import validate_args

        with pytest.raises(ValueError, match="requires --data"):
            validate_args(self._args(live_vjepa=True, data=None))
        # 2026-08-08：flow 是合法 live 模式（π0 式 flow action expert），
        # direct_head 不再强制；这里验证 flow 组合通过、无 direct 时不报错。
        validate_args(
            self._args(live_vjepa=True, direct_head=False, flow_cond="adaln")
        )

    def test_live_mutually_exclusive(self) -> None:
        from train import validate_args

        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_args(self._args(live_vjepa=True, local_slots_data=argparse.Namespace()))
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_args(self._args(live_vjepa=True, c2_controller=True))

    def test_unfreeze_flags_mutually_exclusive(self) -> None:
        from train import validate_args

        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_args(
                self._args(live_vjepa=True, vision_unfreeze_all=True, vision_unfreeze_last=4)
            )
        # 合法组合不抛错
        validate_args(self._args(live_vjepa=True, vision_unfreeze_all=True))
        validate_args(self._args(live_vjepa=True, vision_unfreeze_last=4))

    def test_dense_readout_requires_local_slots_path(self) -> None:
        from train import validate_args

        # 无 live / 预计算路径 → 拒绝
        with pytest.raises(ValueError, match="requires --live-vjepa or --local-slots-data"):
            validate_args(self._args(dense_readout=True))
        # live 与预计算两条合法路径都通过
        validate_args(self._args(dense_readout=True, live_vjepa=True))
        validate_args(self._args(dense_readout=True, local_slots_data=argparse.Namespace()))

    def test_dense_readout_conflicts_direct288(self) -> None:
        from train import validate_args

        # §九：1152 token 直送 VA 会在 VA 内做 1152×1152 自注意力，必须互斥。
        with pytest.raises(ValueError, match="互斥"):
            validate_args(
                self._args(
                    dense_readout=True, live_vjepa=True, local_slots_direct288=True
                )
            )
        with pytest.raises(ValueError, match="互斥"):
            validate_args(
                self._args(
                    dense_readout=True,
                    local_slots_data=argparse.Namespace(),
                    local_slots_direct288=True,
                )
            )
