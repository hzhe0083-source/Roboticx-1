"""MT-VJ dense action readout smoke（artifacts/mt_vj_contract.md §5 / §8）。

验收点：
1. ``dense_readout_mtvj=True``（W_o 严格全零）与 False 在同一随机输入下
   输出逐元素一致（atol 1e-6）——覆盖普通 VA 路径与 memory_split +
   sequential（forward_sequential 注入）路径，metric_tokens 传与不传两种
   情形；
2. dense_evidence 未传（True 但未传）时行为与现状完全一致；False 模型
   收到 dense_evidence 时忽略、不报错；
3. W_o 随机初始化后形状正确且输出确实改变（dense 路径是活的）；
4. 训练模式下梯度能回传到全部 dense 参数（W_o 非零后）。
5. 配置向后兼容：默认 False、序列化往返、dataclasses.replace。

注：本文件独立命名（§8 的 tests/test_mt_vj.py 由多个并行 agent 共享，
避免并行写入冲突；集成时可把本类并入）。
"""
from __future__ import annotations

import dataclasses
import unittest

import pytest
import torch
from torch import nn

from va_compound.model import D_PROJ, VACompoundConfig, VACompoundPolicy

N_DENSE = 2 * 24 * 24  # 1152


def base_config(**overrides) -> VACompoundConfig:
    cfg = dict(
        language_dim=1536,
        vision_dim=768,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        action_horizon=6,
        action_dim=4,
        proprio_dim=4,
        direct_head=True,
    )
    cfg.update(overrides)
    return VACompoundConfig(**cfg)


def copy_common_params(dst: VACompoundPolicy, src: VACompoundPolicy) -> None:
    """把 src 的公共参数（不含 dense 专属参数）拷给 dst——两个模型共享权重。"""
    common = {k: v for k, v in src.state_dict().items() if k in dst.state_dict()}
    dst.load_state_dict(common)


def make_inputs(samples: int = 2, metric_dim: int = 256, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    return {
        "vision_tokens": torch.randn(samples, 16, 768),
        "proprio": torch.randn(samples, 4),
        "previous_action": torch.randn(samples, 4),
        "language_hidden": torch.randn(samples, 8, 1536),
        "language_mask": torch.ones(samples, 8, dtype=torch.bool),
        "dense_evidence": {
            5: torch.randn(samples, N_DENSE, 768),
            11: torch.randn(samples, N_DENSE, 768),
        },
        "metric_tokens": torch.randn(samples, 2, metric_dim),
        "noisy_actions": torch.randn(samples, 6, 4),
        "flow_time": torch.rand(samples),
    }


def assert_all_dense_out_zero(model: VACompoundPolicy) -> None:
    for layer in model.layers:
        assert torch.count_nonzero(layer.dense_out.weight) == 0
        assert torch.count_nonzero(layer.dense_out.bias) == 0


def run_encode(model: VACompoundPolicy, batch: dict) -> tuple:
    cache = model.build_language_cache(
        batch["language_hidden"], batch["language_mask"]
    )
    out = model.encode_condition(
        batch["vision_tokens"],
        batch["proprio"],
        batch["previous_action"],
        language_cache=cache,
        return_visual_memory=True,
        dense_evidence=batch["dense_evidence"],
        metric_tokens=batch["metric_tokens"],
    )
    cond, mem = out
    return cond, model.decode_actions(cond), mem


class DenseReadoutMtvjConfigTests(unittest.TestCase):
    def test_default_false_and_roundtrip(self) -> None:
        cfg = base_config()
        assert cfg.dense_readout_mtvj is False
        # checkpoint config dict 往返 + dataclasses.replace 向后兼容。
        assert VACompoundConfig(**cfg.__dict__).dense_readout_mtvj is False
        on = base_config(dense_readout_mtvj=True)
        assert VACompoundConfig(**on.__dict__).dense_readout_mtvj is True
        assert (
            dataclasses.replace(on, dense_readout_mtvj=False).dense_readout_mtvj
            is False
        )

    def test_false_model_has_no_dense_params(self) -> None:
        model = VACompoundPolicy(base_config()).eval()
        assert model.dense_evidence_proj is None
        for layer in model.layers:
            assert not hasattr(layer, "dense_out")

    def test_true_model_builds_dense_params(self) -> None:
        model = VACompoundPolicy(base_config(dense_readout_mtvj=True)).eval()
        assert model.dense_evidence_proj is not None
        for layer in model.layers:
            assert tuple(layer.dense_k.weight.shape) == (256, D_PROJ)
            # V 侧输入 [D, G, T, coord_emb] = 3*192 + 27
            assert tuple(layer.dense_v.weight.shape) == (256, 3 * D_PROJ + 27)
            assert tuple(layer.dense_out.weight.shape) == (256, 256)
            assert tuple(layer.metric_k.weight.shape) == (256, 256)
        assert_all_dense_out_zero(model)


class DenseReadoutMtvjEquivalenceTests(unittest.TestCase):
    """False vs True（W_o 全零）同一输入 → 输出逐元素一致（atol 1e-6）。"""

    def _assert_equal(self, on, off, batch, memory_split: bool) -> None:
        cond_on, pred_on, mem_on = run_encode(on, batch)
        cond_off, pred_off, mem_off = run_encode(off, batch)
        for name, a, b in (
            ("action_condition", cond_on, cond_off),
            ("decoded_actions", pred_on, pred_off),
        ):
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), f"{name} 不一致"
        if memory_split:
            for name, a, b in (
                ("evidence", mem_on.evidence, mem_off.evidence),
                ("task", mem_on.task, mem_off.task),
            ):
                assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), (
                    f"memory.{name} 不一致"
                )
        else:
            for a, b in zip(mem_on.layers, mem_off.layers):
                assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)

    def _pair(self, **overrides) -> tuple:
        on = VACompoundPolicy(base_config(dense_readout_mtvj=True, **overrides)).eval()
        off = VACompoundPolicy(base_config(**overrides)).eval()
        copy_common_params(off, on)
        assert_all_dense_out_zero(on)
        return on, off

    def test_equivalence_plain_va(self) -> None:
        on, off = self._pair()
        with torch.inference_mode():
            self._assert_equal(on, off, make_inputs(), memory_split=False)

    def test_equivalence_memory_split_sequential(self) -> None:
        # memory_split 分支 + sequential 层（forward_sequential 注入路径）。
        on, off = self._pair(memory_split=True, sequential_coupling=1)
        with torch.inference_mode():
            self._assert_equal(on, off, make_inputs(), memory_split=True)

    def test_equivalence_metric_tokens_none(self) -> None:
        on, off = self._pair()
        batch = make_inputs()
        batch["metric_tokens"] = None
        with torch.inference_mode():
            self._assert_equal(on, off, batch, memory_split=False)

    def test_equivalence_dense_evidence_none(self) -> None:
        """True 但未传 dense_evidence → 与现状完全一致（不注入、不报错）。"""
        on, off = self._pair()
        batch = make_inputs()
        batch["dense_evidence"] = None
        batch["metric_tokens"] = None
        with torch.inference_mode():
            self._assert_equal(on, off, batch, memory_split=False)

    def test_false_model_ignores_dense_input(self) -> None:
        """False 模型收到 dense_evidence 时忽略（参数不参与，逐位不变）。"""
        off = VACompoundPolicy(base_config()).eval()
        batch = make_inputs()
        with torch.inference_mode():
            cond_a, pred_a, _ = run_encode(off, batch)
            batch["dense_evidence"] = None
            batch["metric_tokens"] = None
            cond_b, pred_b, _ = run_encode(off, batch)
        assert torch.allclose(cond_a, cond_b, atol=1e-6, rtol=1e-6)
        assert torch.allclose(pred_a, pred_b, atol=1e-6, rtol=1e-6)

    def test_equivalence_forward_end_to_end(self) -> None:
        """完整 forward（flow 路径 + dense 注入）在 W_o 全零时也逐元素一致。"""
        on, off = self._pair()
        batch = make_inputs()
        with torch.inference_mode():
            vel_on = on(
                batch["vision_tokens"],
                batch["proprio"],
                batch["previous_action"],
                batch["noisy_actions"],
                batch["flow_time"],
                language_hidden=batch["language_hidden"],
                language_mask=batch["language_mask"],
                dense_evidence=batch["dense_evidence"],
                metric_tokens=batch["metric_tokens"],
            )
            vel_off = off(
                batch["vision_tokens"],
                batch["proprio"],
                batch["previous_action"],
                batch["noisy_actions"],
                batch["flow_time"],
                language_hidden=batch["language_hidden"],
                language_mask=batch["language_mask"],
                dense_evidence=batch["dense_evidence"],
                metric_tokens=batch["metric_tokens"],
            )
        assert torch.allclose(vel_on, vel_off, atol=1e-6, rtol=1e-6)


class DenseReadoutMtvjActiveTests(unittest.TestCase):
    """W_o 随机化后：形状正确、输出确实改变、梯度可回传。"""

    def _randomize_wo(self, model: VACompoundPolicy) -> None:
        with torch.no_grad():
            for layer in model.layers:
                nn.init.normal_(layer.dense_out.weight, std=0.05)
                nn.init.normal_(layer.dense_out.bias, std=0.05)

    def test_randomized_wo_shape_and_effect(self) -> None:
        on = VACompoundPolicy(base_config(dense_readout_mtvj=True)).eval()
        off = VACompoundPolicy(base_config()).eval()
        copy_common_params(off, on)
        self._randomize_wo(on)
        batch = make_inputs()
        with torch.inference_mode():
            cond_on, pred_on, _ = run_encode(on, batch)
            cond_off, pred_off, _ = run_encode(off, batch)
        assert tuple(pred_on.shape) == (2, 6, 4)  # 形状正确
        assert tuple(cond_on.shape) == (2, 6, 256)
        # W_o 非零 → dense 路径确实改变输出（far 大于 atol）。
        assert not torch.allclose(cond_on, cond_off, atol=1e-6, rtol=1e-6)
        assert (cond_on - cond_off).abs().max() > 1e-3

    def test_gradients_flow_to_dense_params(self) -> None:
        model = VACompoundPolicy(
            base_config(dense_readout_mtvj=True, num_layers=1, hidden_dim=128)
        ).train()
        self._randomize_wo(model)
        batch = make_inputs(metric_dim=128, seed=3)
        cond, _, _ = run_encode(model, batch)
        loss = cond.pow(2).mean()
        loss.backward()
        named = {
            "dense_q": model.layers[0].dense_q.weight,
            "dense_k": model.layers[0].dense_k.weight,
            "dense_v": model.layers[0].dense_v.weight,
            "dense_out": model.layers[0].dense_out.weight,
            "metric_k": model.layers[0].metric_k.weight,
            "metric_v": model.layers[0].metric_v.weight,
            "proj_d": model.dense_evidence_proj.proj_d.weight,
            "proj_g": model.dense_evidence_proj.proj_g.weight,
            "proj_t": model.dense_evidence_proj.proj_t.weight,
            "coord_k": model.dense_evidence_proj.coord_k.weight,
        }
        for name, p in named.items():
            assert p.grad is not None and bool(p.grad.abs().sum() > 0), (
                f"no gradient through {name}"
            )


class DenseReadoutMtvjValidationTests(unittest.TestCase):
    def _model(self) -> VACompoundPolicy:
        return VACompoundPolicy(base_config(dense_readout_mtvj=True)).eval()

    def test_missing_key_raises(self) -> None:
        model = self._model()
        batch = make_inputs()
        bad = dict(batch["dense_evidence"])
        del bad[5]
        with pytest.raises(ValueError, match="key 5"):
            run_encode(model, {**batch, "dense_evidence": bad})

    def test_batch_mismatch_raises(self) -> None:
        model = self._model()
        batch = make_inputs()
        wrong = {5: torch.randn(3, N_DENSE, 768), 11: torch.randn(3, N_DENSE, 768)}
        with pytest.raises(ValueError):
            run_encode(model, {**batch, "dense_evidence": wrong})

    def test_metric_tokens_shape_raises(self) -> None:
        model = self._model()
        batch = make_inputs()
        with pytest.raises(ValueError, match="metric_tokens"):
            run_encode(model, {**batch, "metric_tokens": torch.randn(2, 3, 256)})


if __name__ == "__main__":
    unittest.main()
