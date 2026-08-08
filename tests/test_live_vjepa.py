"""PULSE-VA Stage B：live-vjepa 路径单元测试（encode 形状 / 槽坐标 / 参数校验）。"""
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from va_compound.live_vjepa import (
    N_TOKENS,
    SEQUENCE_LENGTH,
    VISION_WINDOW,
    _slot_coords,
    encode_live_frames,
)


class _FakeBackbone:
    """伪 V-JEPA：_encode → [B, 2, 12, 12, D]；_pool(spatiotemporal) → [B, 288, D]。"""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.device = torch.device("cpu")
        self.dtype = torch.float16

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        B = inputs.shape[0]
        return torch.randn(B, 2, 12, 12, self.dim, dtype=self.dtype)

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


class TestArgValidation:
    def _args(self, **overrides) -> argparse.Namespace:
        base = dict(
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
