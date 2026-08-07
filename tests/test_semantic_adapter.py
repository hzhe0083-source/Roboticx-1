"""第三种方案（2026-08-07 落地）测试。

覆盖:apply_lora(top_layers) 只包装顶部 N 层;encode/encode_trainable/
encode_with_scene 的 output_layers 中间层支持;QwenSemanticBackbone(冻结先验 +
顶部层 LoRA + zero-init 门控融合 + anchor/geometry 约束的数值正确性);
EndToEndPolicy 融合 cache;train.py 的 --semantic-* 参数校验与默认路径兼容。
全部用轻量 fake 小模型（2 层 32 维 decoder 桩），不加载真实 Qwen/V-JEPA，
不碰 GPU。
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from train import E2EDataset, parse_args, validate_args
from va_compound.backbones import (
    LoRALinear,
    QwenSemanticBackbone,
    QwenTextBackbone,
    SceneTeacher,
    VJEPA21Backbone,
    apply_lora,
)
from va_compound.end_to_end import EndToEndPolicy, build_e2e_policy
from va_compound.model import LanguageCache, VACompoundConfig, VACompoundPolicy


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.arange(5).repeat(batch, 1),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]).repeat(batch, 1),
        }


class FakeDecoderLayer(nn.Module):
    """Fake Qwen decoder layer using all seven LoRA-able projections."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim)
        self.self_attn.k_proj = nn.Linear(dim, dim)
        self.self_attn.v_proj = nn.Linear(dim, dim)
        self.self_attn.o_proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, dim)
        self.mlp.up_proj = nn.Linear(dim, dim)
        self.mlp.down_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.self_attn.q_proj(x)
        k = self.self_attn.k_proj(x)
        v = self.self_attn.v_proj(x)
        attn = self.self_attn.o_proj(q * k + v)
        gated = self.mlp.down_proj(F.gelu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))
        return x + attn + gated


class FakeDecoder(nn.Module):
    """Fake Qwen decoder: hidden_states = (embed, h0, h1, ..., norm(hN-1)).

    Mirrors transformers decoder semantics: layer ``i``'s hidden state is
    ``hidden_states[i + 1]`` (pre-norm), the last hidden state is the final
    norm output.
    """

    def __init__(self, dim: int = 32, vocab: int = 16, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([FakeDecoderLayer(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
        **_kwargs,
    ):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states.append(x)
        return SimpleNamespace(
            last_hidden_state=x,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )


class FakeVideoModel(nn.Module):
    """Minimal V-JEPA stub (patch grid + blocks/norms_block for unfreeze_last)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.patch_size = 16
        self.tubelet_size = 2
        self.num_frames = 4
        self.img_height = 384
        self.img_width = 384
        self.blocks = nn.ModuleList([nn.Module() for _ in range(6)])
        self.norms_block = nn.ModuleList([nn.LayerNorm(16) for _ in range(6)])

    def forward(self, videos):
        batch = videos.shape[0]
        tokens = torch.arange(2 * 24 * 24 * 16, dtype=torch.float32).view(1, 1152, 16)
        return tokens.repeat(batch, 1, 1) * self.scale


def make_backbone(num_layers: int = 2, dim: int = 32) -> QwenTextBackbone:
    return QwenTextBackbone(FakeTokenizer(), FakeDecoder(dim=dim, num_layers=num_layers))


def tiny_config(**overrides) -> VACompoundConfig:
    values = dict(
        language_dim=32,
        vision_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
    )
    values.update(overrides)
    return VACompoundConfig(**values)


class ApplyLoraTopLayersTests(unittest.TestCase):
    def test_top_layers_wraps_only_the_last_n_layers(self):
        decoder = FakeDecoder(num_layers=3)
        count = apply_lora(decoder, rank=4, top_layers=1)
        self.assertEqual(count, 7)  # q/k/v/o + gate/up/down of the last layer only
        self.assertIsInstance(decoder.layers[2].self_attn.q_proj, LoRALinear)
        self.assertIsInstance(decoder.layers[2].self_attn.o_proj, LoRALinear)
        self.assertIsInstance(decoder.layers[2].mlp.down_proj, LoRALinear)
        self.assertIsInstance(decoder.layers[1].self_attn.q_proj, nn.Linear)
        self.assertIsInstance(decoder.layers[0].self_attn.q_proj, nn.Linear)
        self.assertIsInstance(decoder.layers[0].mlp.gate_proj, nn.Linear)
        # LoRA 参数只存在于最后一层。
        names = [name for name, _ in decoder.named_parameters() if "lora_" in name]
        self.assertEqual(len(names), 7 * 2)
        self.assertTrue(all(name.startswith("layers.2.") for name in names))

    def test_top_layers_zero_keeps_whole_model_behavior(self):
        decoder = FakeDecoder(num_layers=3)
        count = apply_lora(decoder, rank=4, top_layers=0)
        self.assertEqual(count, 21)  # 3 layers * 7 projections
        self.assertIsInstance(decoder.layers[0].self_attn.q_proj, LoRALinear)

    def test_top_layers_validation_errors(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_lora(FakeDecoder(num_layers=3), rank=4, top_layers=-1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            apply_lora(FakeDecoder(num_layers=3), rank=4, top_layers=4)
        with self.assertRaisesRegex(ValueError, "layers"):
            apply_lora(nn.Sequential(nn.Linear(4, 4)), rank=4, top_layers=1)


class EncodeOutputLayersTests(unittest.TestCase):
    def test_encode_returns_dict_keyed_by_layer_index(self):
        backbone = make_backbone(num_layers=2)
        hidden, mask = backbone.encode(
            ["pick red cup", "push blue cup"], output_layers=[0, 1]
        )
        self.assertEqual(set(hidden), {0, 1})
        self.assertEqual(hidden[0].shape, (2, 5, 32))
        self.assertEqual(hidden[1].shape, (2, 5, 32))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(hidden[0].requires_grad)  # encode is no_grad
        # dict values are the true per-layer hidden states (embedding excluded)
        decoder = backbone.text_model
        tokens = torch.arange(5)[None].repeat(2, 1)
        x = decoder.embed_tokens(tokens)
        expected_h0 = decoder.layers[0](x)
        torch.testing.assert_close(hidden[0], expected_h0)
        # 层 1 是 final norm 之前的输出，与 post-norm last_hidden 不同。
        last, _ = backbone.encode(["a", "b"])
        self.assertFalse(torch.allclose(hidden[1], last))

    def test_encode_trainable_returns_grad_dict(self):
        backbone = make_backbone(num_layers=2)
        hidden, _ = backbone.encode_trainable(["a"], output_layers=[1])
        self.assertEqual(set(hidden), {1})
        self.assertTrue(hidden[1].requires_grad)

    def test_default_none_returns_last_hidden_tuple(self):
        backbone = make_backbone(num_layers=2)
        last, mask = backbone.encode(["a", "b"])
        self.assertEqual(last.shape, (2, 5, 32))
        self.assertFalse(last.requires_grad)

    def test_output_layers_validation(self):
        backbone = make_backbone(num_layers=2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            backbone.encode(["a"], output_layers=[2])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            backbone.encode(["a"], output_layers=[])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            backbone.encode(["a"], output_layers=[-1])

    def test_encode_with_scene_output_layers(self):
        backbone = make_backbone(num_layers=2, dim=12)
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(2, 16)
        layers, mask = backbone.encode_with_scene(
            ["a", "b"],
            scene,
            teacher.scene_projector,
            teacher.readout_tokens,
            output_layers=[0, 1],
        )
        self.assertEqual(set(layers), {0, 1})
        self.assertEqual(layers[1].shape, (2, 5 + 8 + 8, 12))  # 全序列 [B, L+K+N, D]
        self.assertEqual(mask.shape, (2, 21))
        self.assertTrue(layers[1].requires_grad)
        # 默认路径不变：仍返回 readout 位置 plan。
        plan, full_mask = backbone.encode_with_scene(
            ["a", "b"], scene, teacher.scene_projector, teacher.readout_tokens
        )
        self.assertEqual(plan.shape, (2, 8, 12))
        self.assertEqual(full_mask.shape, (2, 21))


class SemanticBackboneTests(unittest.TestCase):
    def test_base_frozen_and_lora_count(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(
            backbone, lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        self.assertEqual(adapter.lora_layer_count, 7)
        self.assertEqual(adapter.top_layers, 1)
        self.assertEqual(adapter.num_layers, 2)
        self.assertEqual(adapter.anchor_layers, (1,))
        # base（非 LoRA）冻结；LoRA + 门控可训练
        for name, parameter in adapter.text_backbone.text_model.named_parameters():
            if "lora_a" in name or "lora_b" in name:
                self.assertTrue(parameter.requires_grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)
        self.assertTrue(all(p.requires_grad for p in adapter.gate.parameters()))
        # text_model 兼容 shim（save_checkpoint/resume 直接访问 text_backbone.text_model）
        self.assertIs(adapter.text_model, adapter.text_backbone.text_model)
        # 只有最后一层挂了 LoRA。
        self.assertIsInstance(backbone.text_model.layers[1].self_attn.q_proj, LoRALinear)
        self.assertNotIsInstance(backbone.text_model.layers[0].self_attn.q_proj, LoRALinear)

    def test_top_layers_zero_attaches_no_lora(self):
        adapter = QwenSemanticBackbone(make_backbone(num_layers=2), lora_rank=4, top_layers=0)
        self.assertEqual(adapter.lora_layer_count, 0)

    def test_prior_and_adapted_equal_at_init_fused_equals_prior(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(
            backbone, lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        prior, mask = adapter.encode_prior(["pick red cup", "push blue cup"])
        adapted, _ = adapter.encode_adapted(["pick red cup", "push blue cup"])
        self.assertEqual(mask.shape, (2, 5))
        self.assertFalse(prior.requires_grad)
        self.assertTrue(adapted.requires_grad)
        # lora_b 零初始化 → 初始 adapted == prior
        torch.testing.assert_close(adapted, prior, rtol=0, atol=1e-6)
        fused = adapter.fused_embedding(prior, adapted)
        # 门控 ≈ 0 → fused == prior（契约：误差 < 1e-3；实际逐位相等）
        torch.testing.assert_close(fused, prior, rtol=0, atol=1e-6)
        self.assertLess(float((fused - prior).abs().max()), 1e-3)

    def test_fused_embedding_gate_approx_zero_and_shape_check(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1)
        prior = torch.randn(2, 5, 32)
        adapted = prior + 0.1 * torch.randn_like(prior)
        fused = adapter.fused_embedding(prior, adapted)
        self.assertEqual(fused.shape, prior.shape)
        # g ≈ −0.01（小负偏置破零梯度死点）⊙ 残差 → fused−prior 很小
        self.assertLess(float((fused - prior).detach().abs().max()), 0.05)
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.fused_embedding(prior, adapted[:, :4])

    def test_anchor_loss_numerics(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1, anchor_layers=(0,))
        identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # 平行（identical）→ 0
        self.assertEqual(adapter.anchor_loss({0: identity}, {0: identity}).item(), 0.0)
        # 行交换（每元素差 ±1）→ MSE = 1.0
        rotated = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(adapter.anchor_loss({0: identity}, {0: rotated}).item(), 1.0)
        # 先归一化：模长不同的平行向量 → 0
        scaled = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(adapter.anchor_loss({0: scaled}, {0: identity}).item(), 0.0)
        # 全取反 → 每 token 归一化后差 2n，MSE = 2.0
        negated = adapter.anchor_loss({0: identity}, {0: -identity}).item()
        self.assertAlmostEqual(negated, 2.0, places=5)

    def test_anchor_loss_multi_layer_mean(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1, anchor_layers=(0, 1))
        identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        rotated = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = adapter.anchor_loss({0: identity, 1: identity}, {0: identity, 1: rotated})
        self.assertAlmostEqual(loss.item(), 0.5)

    def test_anchor_loss_stopgrad_and_empty_and_missing(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1, anchor_layers=(0,))
        prior = torch.randn(2, 4, requires_grad=True)
        adapted = torch.randn(2, 4, requires_grad=True)
        loss = adapter.anchor_loss({0: prior}, {0: adapted})
        loss.backward()
        self.assertIsNone(prior.grad)  # prior 侧 stop-grad
        self.assertIsNotNone(adapted.grad)
        # anchor_layers 为空 → 0
        empty = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1
        )
        identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(empty.anchor_loss({0: identity}, {0: identity}).item(), 0.0)
        # 层缺失 → 报错
        with self.assertRaisesRegex(ValueError, "missing"):
            adapter.anchor_loss({}, {})

    def test_geometry_loss_numerics(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1)
        identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # 相同 → 0
        self.assertEqual(adapter.geometry_loss(identity, identity).item(), 0.0)
        # 坍塌（两行相同）→ G 差 [[0,1],[1,0]] → 行均值 0.5
        collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(adapter.geometry_loss(identity, collapsed).item(), 0.5)
        # 正交旋转保持 G 矩阵 → 0
        rotated = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        self.assertAlmostEqual(adapter.geometry_loss(identity, rotated).item(), 0.0, places=6)
        # 归一化：模长缩放不改变 G
        self.assertAlmostEqual(
            adapter.geometry_loss(2 * identity, identity).item(), 0.0, places=6
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.geometry_loss(identity, torch.randn(3, 2))

    def test_geometry_loss_mask_weighting_and_stopgrad(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1)
        prior = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        adapted = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]]
        )
        # mask 只选 token 1：prior 两行都是 [0,1] → G=[[1,1],[1,1]]；adapted
        # 行为 [0,1] 与 [1,0] → G=[[1,0],[0,1]]；差方行均值 (0.5+0.5)/2 = 0.5。
        mask = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)
        masked = adapter.geometry_loss(prior, adapted, mask)
        self.assertAlmostEqual(masked.item(), 0.5, places=5)
        # 全 1 mask 与无 mask 等价。
        ones = torch.ones(2, 2, dtype=torch.bool)
        self.assertAlmostEqual(
            adapter.geometry_loss(prior, adapted, ones).item(),
            adapter.geometry_loss(prior, adapted).item(),
            places=6,
        )
        with self.assertRaisesRegex(ValueError, "mask"):
            adapter.geometry_loss(prior, adapted, torch.ones(3, 2, dtype=torch.bool))
        # prior 侧 stop-grad
        prior_grad = prior.clone().requires_grad_(True)
        adapted_grad = adapted.clone().requires_grad_(True)
        adapter.geometry_loss(prior_grad, adapted_grad, mask).backward()
        self.assertIsNone(prior_grad.grad)
        self.assertIsNotNone(adapted_grad.grad)

    def test_gradients_flow_to_lora_and_gate(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(
            backbone, lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        # 扰动一个 lora_b 使 adapted != prior（否则零梯度死点：初始门控 ≈ 0
        # 且 adapted == prior，fused 路径对门控/LoRA 的梯度恒为 0）。
        bumped = [
            parameter
            for name, parameter in adapter.named_parameters()
            if "lora_b" in name
        ][0]
        bumped.data.normal_(mean=0.0, std=0.5)
        prior_last, mask = adapter.encode_prior(["hi there"])
        adapted_last, _ = adapter.encode_adapted(["hi there"])
        fused = adapter.fused_embedding(prior_last, adapted_last)
        prior_layers, _ = adapter.encode_prior_states(["hi there"], [1])
        adapted_layers, _ = adapter.encode_adapted_states(["hi there"], [1])
        loss = (
            fused.float().square().mean()
            + adapter.anchor_loss(prior_layers, adapted_layers)
            + adapter.geometry_loss(prior_last, adapted_last, mask)
        )
        loss.backward()
        lora_grads = [
            parameter.grad
            for name, parameter in adapter.named_parameters()
            if "lora_a" in name or "lora_b" in name
        ]
        self.assertTrue(
            any(g is not None and float(g.abs().sum()) > 0 for g in lora_grads)
        )
        self.assertIsNotNone(adapter.gate[0].weight.grad)
        self.assertIsNotNone(adapter.gate[-1].weight.grad)
        self.assertFalse(prior_last.requires_grad)
        self.assertTrue(adapted_last.requires_grad)
        # 冻结 base 不拿梯度。
        self.assertIsNone(backbone.text_model.layers[0].self_attn.q_proj.weight.grad)

    def test_prior_states_forward_is_always_no_grad(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(backbone, lora_rank=4, top_layers=1)
        layers, mask = adapter.encode_prior_states(["a", "b"], [0, 1])
        self.assertFalse(layers[0].requires_grad)
        self.assertFalse(layers[1].requires_grad)
        self.assertEqual(mask.dtype, torch.bool)
        adapted_layers, _ = adapter.encode_adapted_states(["a", "b"], [0, 1])
        # 冻结区域（层 0）无梯度；LoRA 区域（顶部层 1）带梯度。
        self.assertFalse(adapted_layers[0].requires_grad)
        self.assertTrue(adapted_layers[1].requires_grad)


class EndToEndSemanticTests(unittest.TestCase):
    def test_semantic_cache_builds_from_fused_embedding(self):
        config = tiny_config()
        policy = VACompoundPolicy(config)
        vision = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(
            backbone, lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        e2e = EndToEndPolicy(text_backbone=adapter, vision_backbone=vision, policy=policy)
        cache = e2e.build_language_cache(["a", "a", "b"])
        self.assertIsInstance(cache, LanguageCache)
        self.assertEqual(len(cache.layers), 1)
        # cache 按样本数构建（去重只作用于编码）；初始 fused == prior，与
        # 直接 build_language_cache(fused[indices]) 逐位一致。
        self.assertEqual(tuple(cache.attention_mask.shape), (3, 5))
        prior, mask = adapter.encode_prior(["a", "b"])
        adapted, _ = adapter.encode_adapted(["a", "b"])
        fused = adapter.fused_embedding(prior, adapted)
        lookup = {"a": 0, "b": 1}
        indices = torch.tensor([lookup[text] for text in ["a", "a", "b"]], dtype=torch.long)
        reference = policy.build_language_cache(fused[indices], mask[indices])
        torch.testing.assert_close(cache.layers[0].key, reference.layers[0].key)
        torch.testing.assert_close(cache.layers[0].value, reference.layers[0].value)
        # 默认路径行为不变。
        plain = EndToEndPolicy(text_backbone=backbone, vision_backbone=vision, policy=policy)
        cache_plain = plain.build_language_cache(["a", "a", "b"])
        self.assertEqual(tuple(cache_plain.attention_mask.shape), (3, 5))

    def test_prior_adapted_states_match_public_encode(self):
        backbone = make_backbone(num_layers=2)
        adapter = QwenSemanticBackbone(
            backbone, lora_rank=4, top_layers=1, anchor_layers=(0, 1)
        )
        states_p, mask_p = adapter.encode_prior_states(["a", "b"], [0, 1])
        states_e, _ = backbone.encode(["a", "b"], output_layers=[0, 1])
        torch.testing.assert_close(states_p[0], states_e[0])
        torch.testing.assert_close(states_p[1], states_e[1])
        self.assertFalse(states_p[1].requires_grad)
        self.assertEqual(mask_p.shape, (2, 5))
        states_a, _ = adapter.encode_adapted_states(["a", "b"], [1])
        self.assertTrue(states_a[1].requires_grad)

    def test_build_e2e_policy_semantic_wiring_and_counts(self):
        config = tiny_config()
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fake_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            e2e, counts = build_e2e_policy(
                config=config,
                device=torch.device("cpu"),
                semantic_adapter=True,
                semantic_lora_rank=4,
                semantic_top_layers=1,
            )
        self.assertIsInstance(e2e.text_backbone, QwenSemanticBackbone)
        self.assertEqual(counts["semantic_lora_layers"], 7)
        self.assertEqual(counts["semantic_top_layers"], 1)
        self.assertEqual(counts["lora_layers"], 7)
        # 默认 anchor 集合 = 被适配的顶部层（2 层取后 1 层）。
        self.assertEqual(e2e.text_backbone.anchor_layers, (1,))
        # 显式 anchor 集合透传（复用同一个 fake 会重复包装，用新实例）。
        fresh_qwen = make_backbone(num_layers=2)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fresh_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            e2e2, counts2 = build_e2e_policy(
                config=config,
                device=torch.device("cpu"),
                semantic_adapter=True,
                semantic_top_layers=1,
                semantic_anchor_layers=(0,),
            )
        self.assertEqual(e2e2.text_backbone.anchor_layers, (0,))
        self.assertEqual(counts2["semantic_lora_layers"], 7)

    def test_build_e2e_policy_default_path_unchanged(self):
        config = tiny_config()
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fake_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            e2e, counts = build_e2e_policy(config=config, device=torch.device("cpu"))
        self.assertIsInstance(e2e.text_backbone, QwenTextBackbone)
        self.assertNotIsInstance(e2e.text_backbone, QwenSemanticBackbone)
        self.assertEqual(counts["semantic_lora_layers"], 0)
        self.assertEqual(counts["semantic_top_layers"], 0)
        self.assertEqual(counts["lora_layers"], 0)

    def test_build_e2e_policy_rejects_semantic_with_global_lora(self):
        config = tiny_config()
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fake_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                build_e2e_policy(
                    config=config,
                    device=torch.device("cpu"),
                    semantic_adapter=True,
                    lora_rank=4,
                )

    def test_end_to_end_policy_rollout_with_semantic_adapter(self):
        config = tiny_config()
        adapter = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        vision = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        policy = VACompoundPolicy(config)
        e2e = EndToEndPolicy(text_backbone=adapter, vision_backbone=vision, policy=policy)
        frames = torch.randint(0, 256, (2, 4, 2, 3, 16, 16), dtype=torch.uint8)
        proprio = torch.randn(2, 4, config.proprio_dim)
        previous_action = torch.randn(2, 4, config.action_dim)
        noisy = torch.randn(2, 4, config.action_horizon, config.action_dim)
        flow_time = torch.rand(2, 4)
        target = torch.randn(2, 4, config.action_horizon, config.action_dim)
        predicted, _, _ = e2e.rollout(
            frames, ["a", "b"], proprio, previous_action, noisy, flow_time
        )
        self.assertEqual(
            predicted.shape, (2, 4, config.action_horizon, config.action_dim)
        )
        loss = policy.flow_matching_loss(predicted, target)
        loss.backward()
        self.assertIsNotNone(adapter.gate[-1].weight.grad)


class TrainArgSemanticTests(unittest.TestCase):
    def test_semantic_adapter_requires_e2e_data(self):
        args = parse_args(["--semantic-adapter"])
        with self.assertRaisesRegex(ValueError, "e2e-data"):
            validate_args(args)

    def test_semantic_adapter_conflicts_with_global_lora(self):
        args = parse_args(
            ["--semantic-adapter", "--e2e-data", "x.pt", "--lora-rank", "8"]
        )
        with self.assertRaisesRegex(ValueError, "lora-rank"):
            validate_args(args)

    def test_semantic_adapter_conflicts_with_qwen_unfreeze(self):
        args = parse_args(
            [
                "--semantic-adapter",
                "--e2e-data",
                "x.pt",
                "--qwen-unfreeze-blocks",
                "2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "qwen-unfreeze-blocks"):
            validate_args(args)

    def test_semantic_adapter_valid_combo_passes(self):
        args = parse_args(
            ["--semantic-adapter", "--e2e-data", "x.pt", "--single-task"]
        )
        validate_args(args)  # 不抛异常

    def test_semantic_top_layers_must_be_positive(self):
        args = parse_args(
            [
                "--semantic-adapter",
                "--e2e-data",
                "x.pt",
                "--semantic-top-layers",
                "0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "semantic-top-layers"):
            validate_args(args)

    def test_semantic_loss_weights_must_be_non_negative(self):
        for flag in ("--semantic-anchor-weight", "--semantic-geometry-weight"):
            args = parse_args([flag + "=-0.1"])
            with self.assertRaisesRegex(ValueError, "non-negative"):
                validate_args(args)

    def test_non_semantic_defaults_unchanged(self):
        args = parse_args([])
        self.assertFalse(args.semantic_adapter)
        self.assertEqual(args.semantic_lora_rank, 8)
        self.assertEqual(args.semantic_top_layers, 4)
        self.assertEqual(args.semantic_anchor_weight, 0.0)
        self.assertEqual(args.semantic_geometry_weight, 0.0)
        self.assertEqual(args.semantic_anchor_layers, "")
        # 非 semantic 模式允许全局 LoRA，校验不报错
        args = parse_args(["--lora-rank", "8"])
        validate_args(args)

    def test_e2e_dataset_still_constructs_without_semantic_flags(self):
        payload = {
            "video_frames": torch.randint(0, 256, (2, 4, 2, 3, 8, 8), dtype=torch.uint8),
            "instructions": ["pick red cup", "push blue cup"],
            "proprio": torch.randn(2, 4, 5),
            "previous_action": torch.randn(2, 4, 4),
            "actions": torch.randn(2, 4, 3, 4),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e2e.pt"
            torch.save(payload, path)
            dataset = E2EDataset(path, min_sequence_length=4)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["instruction"], "pick red cup")


if __name__ == "__main__":
    unittest.main()
