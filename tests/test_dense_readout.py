"""Step 0 dense readout（--dense-readout）端到端 smoke：1152 patch → 25-token VA 视觉流。

设计文档（artifacts/c2irf_v2_vision_ablation.md §七 Step 0）：跳过 V-JEPA 池化，
角色查询直接读出 2×24×24=1152 个 patch token；coarse 从 1152 avg-pool 到 16，
VA 视觉流仍为 16 coarse + 6 角色槽 + 3 关系 token = 25 tokens（§九：1152 不进
VA 自注意力，只在槽 cross-attention 的 K/V 侧）。
"""
from __future__ import annotations

import unittest

import pytest
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy

N_DENSE = 2 * 24 * 24  # 1152


def build_dense_coords_smoke(n_grid: int = 24) -> torch.Tensor:
    """[2*n_grid*n_grid, 3]：与 live_vjepa._dense_coords 同一生成公式。"""
    rows = []
    half = (n_grid - 1) / 2
    for t in range(2):
        for y in range(n_grid):
            for x in range(n_grid):
                rows.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
    return torch.tensor(rows, dtype=torch.float32)


def make_dense_batch(samples: int = 2) -> dict:
    torch.manual_seed(11)
    sequence = 4
    dense = torch.randn(samples, sequence, N_DENSE, 768)
    proprio = torch.randn(samples, sequence, 4)
    previous = torch.randn(samples, sequence, 4)
    language = torch.randn(samples, 8, 1536)
    return {
        "vision_tokens_st": dense,
        "coords": build_dense_coords_smoke(),
        "language_hidden": language,
        "language_mask": torch.ones(samples, 8, dtype=torch.bool),
        "proprio": proprio,
        "previous_action": previous,
        "actions": torch.randn(samples, sequence, 6, 4),
    }


def dense_config(**overrides) -> VACompoundConfig:
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
    )
    base.update(overrides)
    return VACompoundConfig(**base)


class DenseConfigValidationTests(unittest.TestCase):
    def test_requires_local_slots(self) -> None:
        with pytest.raises(ValueError, match="local_slots"):
            VACompoundConfig(dense_readout=True, local_slots=False)

    def test_conflicts_direct288(self) -> None:
        # §九：1152 token 直送 VA = VA 内 1152×1152 自注意力，禁止。
        with pytest.raises(ValueError, match="互斥"):
            dense_config(local_slots_direct288=True)

    def test_requires_1152_tokens(self) -> None:
        with pytest.raises(ValueError, match="1152"):
            dense_config(local_slot_tokens=288)

    def test_default_config_unchanged(self) -> None:
        """默认配置（无 dense_readout）行为不变：不触发任何新校验。"""
        cfg = VACompoundConfig(
            language_dim=1536, vision_dim=768, hidden_dim=256, num_layers=1,
            num_heads=4, action_horizon=4, action_dim=4, proprio_dim=4,
            local_slots=True, local_slot_tokens=288,
        )
        assert not cfg.dense_readout
        assert cfg.local_slot_tokens == 288


class DenseReadoutForwardTests(unittest.TestCase):
    def test_vision_stream_25_tokens_and_decode(self) -> None:
        """1152 dense tokens → 16 coarse + 6 槽 + 3 关系 = 25；VA 前向/解码形状。"""
        model = VACompoundPolicy(dense_config()).eval()
        batch = make_dense_batch()
        cache = model.build_language_cache(
            batch["language_hidden"], batch["language_mask"]
        )
        assert cache.role_queries is not None
        assert cache.role_queries.shape == (2, 6, 256)
        vision = model.build_local_vision(
            batch["vision_tokens_st"][:, 0],
            batch["coords"],
            cache.role_queries,
        )
        assert vision.shape == (2, 25, 768)
        with torch.inference_mode():
            cond, _ = model.encode_condition(
                vision,
                batch["proprio"][:, 0],
                batch["previous_action"][:, 0],
                language_cache=cache,
                return_visual_memory=True,
            )
            pred = model.decode_actions(cond)
        assert pred.shape == (2, 6, 4)

    def test_gradients_flow_to_slot_modules(self) -> None:
        """动作 loss 的梯度必须回传到角色编译器/槽读出器/关系 token（1152 K/V 侧）。"""
        model = VACompoundPolicy(dense_config(hidden_dim=128, num_layers=1)).train()
        batch = make_dense_batch()
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
        for name, module in (
            ("role_compiler", model.role_compiler),
            ("slot_reader", model.slot_reader),
            ("relation_tokens", model.relation_tokens),
        ):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads, f"no gradients through {name}"
            assert all(bool(g.abs().sum() > 0) for g in grads), f"zero grad in {name}"

    def test_gradients_flow_to_dense_input(self) -> None:
        """dense token 输入本身带梯度（live 路径解冻 V-JEPA 时依赖此通道）。"""
        model = VACompoundPolicy(dense_config(hidden_dim=128, num_layers=1)).train()
        batch = make_dense_batch()
        cache = model.build_language_cache(
            batch["language_hidden"], batch["language_mask"]
        )
        dense = batch["vision_tokens_st"][:, 0].clone().requires_grad_(True)
        vision = model.build_local_vision(dense, batch["coords"], cache.role_queries)
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
        assert dense.grad is not None
        assert bool(dense.grad.abs().sum() > 0)

    def test_coords_must_match_token_count(self) -> None:
        """坐标与 token 数不匹配时 build_local_vision 必须报错（防静默错位）。"""
        model = VACompoundPolicy(dense_config()).eval()
        batch = make_dense_batch()
        cache = model.build_language_cache(
            batch["language_hidden"], batch["language_mask"]
        )
        wrong_coords = build_dense_coords_smoke(n_grid=12)  # 288 坐标
        with pytest.raises(ValueError, match="coords"):
            model.build_local_vision(
                batch["vision_tokens_st"][:, 0], wrong_coords, cache.role_queries
            )


if __name__ == "__main__":
    unittest.main()
