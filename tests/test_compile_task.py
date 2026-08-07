"""compile-task（SemanticCompiler，2026-08-07）测试。

覆盖:encode_with_scene 的 extra_embeds 拼接契约;SemanticCompiler 前向
（形状/mask/各种 K/历史与变化输入/参数可训练）;EndToEndPolicy 的
compile_semantic 与 rollout 重编译时机（可计数 fake text_backbone）;
build_e2e_policy 的 compile_task 接线与 parameter_groups 分组;train.py
--compile-*/--training-stage 校验矩阵;checkpoint 的 semantic_compiler 键
与 Stage A → semantic 的 strict=False 迁移。
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

from train import parse_args, save_checkpoint, validate_args
from va_compound.backbones import (
    QwenSemanticBackbone,
    QwenTextBackbone,
    SceneTeacher,
    SemanticCompiler,
    VJEPA21Backbone,
    pool_flat_tokens,
)
from va_compound.end_to_end import EndToEndPolicy, build_e2e_policy, parameter_groups
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
    """Fake Qwen decoder: hidden_states = (embed, h0, h1, ..., norm(hN-1))."""

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


class CountingBackbone(QwenTextBackbone):
    """QwenTextBackbone 桩：记录 encode_trainable / encode_with_scene 调用次数。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_trainable_calls = 0
        self.encode_with_scene_calls = 0

    def encode_trainable(self, instructions, output_layers=None):
        self.encode_trainable_calls += 1
        return super().encode_trainable(instructions, output_layers=output_layers)

    def encode_with_scene(
        self,
        instructions,
        scene_summary,
        scene_projector,
        readout_tokens,
        n_scene=8,
        output_layers=None,
        extra_embeds=None,
    ):
        self.encode_with_scene_calls += 1
        return super().encode_with_scene(
            instructions,
            scene_summary,
            scene_projector,
            readout_tokens,
            n_scene=n_scene,
            output_layers=output_layers,
            extra_embeds=extra_embeds,
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


def make_counting_backbone(num_layers: int = 2, dim: int = 32) -> CountingBackbone:
    return CountingBackbone(FakeTokenizer(), FakeDecoder(dim=dim, num_layers=num_layers))


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


def make_compile_e2e(
    backbone: QwenTextBackbone | None = None,
    compiler: SemanticCompiler | None = None,
    **config_overrides,
) -> tuple[EndToEndPolicy, VACompoundConfig]:
    config = tiny_config(**config_overrides)
    policy = VACompoundPolicy(config)
    vision = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
    backbone = backbone or make_backbone(num_layers=2)
    compiler = compiler or SemanticCompiler(
        language_dim=config.language_dim,
        vision_dim=config.vision_dim,
        n_scene=8,
        n_hist=2,
        n_delta=2,
        n_readout=8,
        hidden=24,
    )
    e2e = EndToEndPolicy(
        text_backbone=backbone,
        vision_backbone=vision,
        policy=policy,
        compiler=compiler,
        n_scene_tokens=16,
    )
    return e2e, config


def rollout_inputs(sequence: int, config: VACompoundConfig) -> dict:
    return dict(
        frames=torch.randint(0, 256, (2, sequence, 2, 3, 16, 16), dtype=torch.uint8),
        instructions=["pick red cup", "push blue cup"],
        proprio=torch.randn(2, sequence, config.proprio_dim),
        previous_action=torch.randn(2, sequence, config.action_dim),
        noisy_actions=torch.randn(2, sequence, config.action_horizon, config.action_dim),
        flow_time=torch.rand(2, sequence),
    )


class EncodeWithSceneExtraTests(unittest.TestCase):
    def test_extra_embeds_order_mask_and_shapes(self):
        backbone = make_backbone(num_layers=2, dim=12)
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(2, 16)
        extra_a = torch.randn(2, 2, 12)
        extra_b = torch.randn(2, 3, 12)
        captured = {}
        original = backbone.text_model.forward

        def spy(**kwargs):
            captured["embeds"] = kwargs["inputs_embeds"]
            captured["mask"] = kwargs["attention_mask"]
            return original(**kwargs)

        backbone.text_model.forward = spy
        try:
            plan, mask = backbone.encode_with_scene(
                ["a", "b"],
                scene,
                teacher.scene_projector,
                teacher.readout_tokens,
                extra_embeds=[extra_a, extra_b],
            )
        finally:
            backbone.text_model.forward = original
        self.assertEqual(tuple(plan.shape), (2, 8, 12))
        # mask = L(5) + n_scene(8) + extra(2+3) + n_readout(8)
        self.assertEqual(tuple(mask.shape), (2, 5 + 8 + 2 + 3 + 8))
        self.assertEqual(mask.dtype, torch.bool)
        # 拼接顺序：指令 → 场景 → extra(列表序) → readout
        input_ids = backbone.tokenizer(["a", "b"])["input_ids"]
        instr_embeds = backbone.text_model.embed_tokens(input_ids)
        scene_embeds = teacher.scene_projector(scene).view(2, 8, 12)
        readout = teacher.readout_tokens[None].expand(2, -1, -1)
        expected = torch.cat((instr_embeds, scene_embeds, extra_a, extra_b, readout), 1)
        torch.testing.assert_close(captured["embeds"], expected)
        expected_mask = torch.cat(
            (
                torch.tensor([[1, 1, 1, 1, 0]]).repeat(2, 1),
                torch.ones(2, 8 + 2 + 3 + 8, dtype=torch.int64),
            ),
            dim=1,
        )
        torch.testing.assert_close(captured["mask"], expected_mask)

    def test_extra_none_identical_to_legacy_behavior(self):
        backbone = make_backbone(num_layers=2, dim=12)
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(2, 16)
        plan_a, mask_a = backbone.encode_with_scene(
            ["a", "b"], scene, teacher.scene_projector, teacher.readout_tokens
        )
        plan_b, mask_b = backbone.encode_with_scene(
            ["a", "b"],
            scene,
            teacher.scene_projector,
            teacher.readout_tokens,
            extra_embeds=None,
        )
        self.assertTrue(torch.equal(plan_a, plan_b))
        self.assertTrue(torch.equal(mask_a, mask_b))
        # 空列表同样等价
        plan_c, mask_c = backbone.encode_with_scene(
            ["a", "b"],
            scene,
            teacher.scene_projector,
            teacher.readout_tokens,
            extra_embeds=[],
        )
        self.assertTrue(torch.equal(plan_a, plan_c))
        self.assertTrue(torch.equal(mask_a, mask_c))

    def test_extra_embeds_validation(self):
        backbone = make_backbone(num_layers=2, dim=12)
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(1, 16)
        with self.assertRaisesRegex(ValueError, "language hidden dim"):
            backbone.encode_with_scene(
                ["a"], scene, teacher.scene_projector, teacher.readout_tokens,
                extra_embeds=[torch.randn(1, 2, 10)],
            )
        with self.assertRaisesRegex(ValueError, "batch"):
            backbone.encode_with_scene(
                ["a"], scene, teacher.scene_projector, teacher.readout_tokens,
                extra_embeds=[torch.randn(2, 2, 12)],
            )
        with self.assertRaisesRegex(ValueError, r"\[B, n, language_dim\]"):
            backbone.encode_with_scene(
                ["a"], scene, teacher.scene_projector, teacher.readout_tokens,
                extra_embeds=[torch.randn(1, 2)],
            )

    def test_extra_with_output_layers_returns_full_sequence(self):
        backbone = make_backbone(num_layers=2, dim=12)
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(1, 16)
        extra = torch.randn(1, 2, 12)
        layers, mask = backbone.encode_with_scene(
            ["a"], scene, teacher.scene_projector, teacher.readout_tokens,
            output_layers=[0, 1], extra_embeds=[extra],
        )
        self.assertEqual(set(layers), {0, 1})
        self.assertEqual(tuple(layers[1].shape), (1, 5 + 8 + 2 + 8, 12))
        self.assertEqual(tuple(mask.shape), (1, 23))
        self.assertTrue(layers[1].requires_grad)


class SemanticCompilerTests(unittest.TestCase):
    def setUp(self):
        self.backbone = make_backbone(num_layers=2)  # language_dim 32
        self.compiler = SemanticCompiler(
            language_dim=32,
            vision_dim=16,
            n_scene=8,
            n_hist=2,
            n_delta=2,
            n_readout=8,
            hidden=24,
        )

    def test_forward_shapes_and_mask(self):
        scene = torch.randn(2, 20, 16)  # K=20 != n_scene=8
        history = torch.randn(2, 16)
        delta = torch.randn(2, 16)
        plan, mask = self.compiler(self.backbone, ["a", "b"], scene, history, delta)
        self.assertEqual(tuple(plan.shape), (2, 8, 32))
        # mask = L(5) + n_scene(8) + n_hist(2) + n_delta(2) + n_readout(8)
        self.assertEqual(tuple(mask.shape), (2, 5 + 8 + 2 + 2 + 8))
        self.assertTrue(plan.requires_grad)
        self.assertIsInstance(plan, torch.Tensor)

    def test_scene_tokens_various_k(self):
        for k in (1, 4, 8, 16, 33):
            scene = torch.randn(2, k, 16)
            plan, mask = self.compiler(
                self.backbone, ["a", "b"], scene, torch.randn(2, 16),
                torch.randn(2, 16),
            )
            self.assertEqual(tuple(plan.shape), (2, 8, 32))
            self.assertEqual(mask.shape[1], 5 + 8 + 2 + 2 + 8)

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "scene_tokens"):
            self.compiler(
                self.backbone, ["a"], torch.randn(2, 8, 12), torch.randn(2, 16),
                torch.randn(2, 16),
            )
        with self.assertRaisesRegex(ValueError, "semantic_history"):
            self.compiler(
                self.backbone, ["a"], torch.randn(2, 8, 16), torch.randn(2, 12),
                torch.randn(2, 16),
            )
        with self.assertRaisesRegex(ValueError, "scene_delta"):
            self.compiler(
                self.backbone, ["a"], torch.randn(2, 8, 16), torch.randn(2, 16),
                torch.randn(3, 16),
            )
        with self.assertRaisesRegex(ValueError, "batch"):
            self.compiler(
                self.backbone, ["a"], torch.randn(2, 8, 16), torch.randn(3, 16),
                torch.randn(2, 16),
            )

    def test_parameters_trainable(self):
        # 冻结 Qwen base（与真实训练一致：encode_with_scene 只训 projector/readout）
        self.backbone.text_model.requires_grad_(False)
        scene = torch.randn(2, 16, 16)
        plan, _ = self.compiler(
            self.backbone, ["a", "b"], scene, torch.randn(2, 16), torch.randn(2, 16)
        )
        loss = plan.square().mean()
        loss.backward()
        for name, parameter in self.compiler.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
        # 冻结的 Qwen base 不拿梯度
        self.assertIsNone(self.backbone.text_model.embed_tokens.weight.grad)
        self.assertIsNone(self.backbone.text_model.layers[0].self_attn.q_proj.weight.grad)

    def test_readout_tokens_initialization(self):
        self.assertEqual(tuple(self.compiler.readout_tokens.shape), (8, 32))
        self.assertTrue(self.compiler.readout_tokens.requires_grad)
        self.assertLess(float(self.compiler.readout_tokens.std()), 0.1)

    def test_forward_accepts_semantic_backbone_wrapper(self):
        adapter = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        plan, mask = self.compiler(
            adapter, ["a"], torch.randn(1, 16, 16), torch.randn(1, 16),
            torch.randn(1, 16),
        )
        self.assertEqual(tuple(plan.shape), (1, 8, 32))
        self.assertEqual(mask.shape[1], 5 + 8 + 2 + 2 + 8)


class EndToEndCompileTests(unittest.TestCase):
    def test_compile_semantic_builds_extended_cache(self):
        e2e, config = make_compile_e2e()
        visual_tokens = torch.randn(2, 64, 16)
        history = torch.randn(2, 16)
        delta = torch.randn(2, 16)
        cache = e2e.compile_semantic(["a", "a", "b"], visual_tokens, history, delta)
        self.assertIsInstance(cache, LanguageCache)
        n_readout = e2e.compiler.n_readout
        self.assertEqual(tuple(cache.attention_mask.shape), (3, 5 + n_readout))
        self.assertEqual(cache.layers[0].key.shape[2], 5 + n_readout)
        # 与手工参考逐位一致：语言 hidden（encode_trainable，按去重索引展开）+
        # compiler 语义 token（同样展开到样本批）
        hidden, mask = e2e.text_backbone.encode_trainable(["a", "b"])
        scene_tokens = pool_flat_tokens(visual_tokens, e2e.n_scene_tokens)
        semantic, _ = e2e.compiler(
            e2e.text_backbone, ["a", "b"], scene_tokens, history, delta
        )
        lookup = {"a": 0, "b": 1}
        indices = torch.tensor(
            [lookup[text] for text in ["a", "a", "b"]], dtype=torch.long
        )
        extended = torch.cat((hidden[indices], semantic[indices]), dim=1)
        extended_mask = torch.cat(
            (
                mask[indices],
                torch.ones(3, n_readout, dtype=torch.bool),
            ),
            dim=1,
        )
        reference = e2e.policy.build_language_cache(extended, extended_mask)
        torch.testing.assert_close(cache.layers[0].key, reference.layers[0].key)
        torch.testing.assert_close(cache.layers[0].value, reference.layers[0].value)
        torch.testing.assert_close(cache.attention_mask, reference.attention_mask)

    def test_compile_semantic_requires_compiler_and_shapes(self):
        config = tiny_config()
        policy = VACompoundPolicy(config)
        vision = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        e2e = EndToEndPolicy(
            text_backbone=make_backbone(num_layers=2),
            vision_backbone=vision,
            policy=policy,
        )
        with self.assertRaisesRegex(ValueError, "SemanticCompiler"):
            e2e.compile_semantic(["a"], torch.randn(2, 64, 16), torch.randn(2, 16),
                                 torch.randn(2, 16))
        e2e_with, _ = make_compile_e2e()
        with self.assertRaisesRegex(ValueError, "semantic_history"):
            e2e_with.compile_semantic(["a"], torch.randn(2, 64, 16),
                                      torch.randn(2, 8), torch.randn(2, 16))
        with self.assertRaisesRegex(ValueError, "visual_tokens"):
            e2e_with.compile_semantic(["a"], torch.randn(2, 16),
                                      torch.randn(2, 16), torch.randn(2, 16))

    def _count_rollout(self, sequence, compile_every, counting):
        e2e, config = make_compile_e2e(backbone=counting)
        inputs = rollout_inputs(sequence, config)
        predicted, _, cache = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
            compile_every=compile_every,
        )
        return predicted, cache, e2e, config

    def test_rollout_compile_timing_counts(self):
        # compile_every=0：只构建一次语言 cache（1 次 encode_trainable，0 次编译）
        counting = make_counting_backbone()
        predicted, cache, _, config = self._count_rollout(8, 0, counting)
        self.assertEqual(counting.encode_trainable_calls, 1)
        self.assertEqual(counting.encode_with_scene_calls, 0)
        self.assertEqual(predicted.shape, (2, 8, 3, 4))
        self.assertEqual(cache.attention_mask.shape[1], 5)  # 未扩展
        # compile_every=4、T=4：仅 t=0 重编译 → 1 次编译
        counting = make_counting_backbone()
        predicted, cache, _, config = self._count_rollout(4, 4, counting)
        self.assertEqual(counting.encode_trainable_calls, 2)
        self.assertEqual(counting.encode_with_scene_calls, 1)
        self.assertEqual(cache.attention_mask.shape[1], 5 + 8)
        # compile_every=4、T=8：t=0 与 t=4 → 2 次编译
        counting = make_counting_backbone()
        predicted, cache, _, config = self._count_rollout(8, 4, counting)
        self.assertEqual(counting.encode_trainable_calls, 3)
        self.assertEqual(counting.encode_with_scene_calls, 2)
        self.assertEqual(predicted.shape, (2, 8, 3, 4))
        # compile_every=1、T=8：每步编译 → 8 次编译
        counting = make_counting_backbone()
        predicted, _, _, config = self._count_rollout(8, 1, counting)
        self.assertEqual(counting.encode_trainable_calls, 9)
        self.assertEqual(counting.encode_with_scene_calls, 8)

    def test_rollout_external_compiler_overrides_self(self):
        config = tiny_config()
        policy = VACompoundPolicy(config)
        vision = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        e2e = EndToEndPolicy(
            text_backbone=make_backbone(num_layers=2),
            vision_backbone=vision,
            policy=policy,
        )  # 无 compiler
        compiler = SemanticCompiler(language_dim=32, vision_dim=16)
        inputs = rollout_inputs(4, config)
        predicted, _, cache = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
            compile_every=4, compiler=compiler,
        )
        self.assertEqual(cache.attention_mask.shape[1], 5 + compiler.n_readout)
        self.assertEqual(predicted.shape, (2, 4, 3, 4))

    def test_rollout_with_memory_split_uses_task_history(self):
        e2e, config = make_compile_e2e(memory_split=True)
        inputs = rollout_inputs(4, config)
        predicted, _, cache = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
            compile_every=1,
        )
        self.assertEqual(predicted.shape, (2, 4, 3, 4))
        self.assertEqual(cache.attention_mask.shape[1], 5 + 8)
        # 梯度到达 compiler（memory.task 路径被真正走过）
        loss = predicted.square().mean()
        loss.backward()
        self.assertIsNotNone(e2e.compiler.history_projector[0].weight.grad)
        self.assertIsNotNone(e2e.compiler.readout_tokens.grad)

    def test_rollout_default_path_unchanged(self):
        e2e, config = make_compile_e2e(backbone=make_counting_backbone())
        inputs = rollout_inputs(4, config)
        predicted, _, cache = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
        )
        self.assertEqual(cache.attention_mask.shape[1], 5)
        self.assertEqual(predicted.shape, (2, 4, 3, 4))


class BuildE2ECompileTests(unittest.TestCase):
    def test_compile_task_wiring_and_counts(self):
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
                compile_task=True,
                compile_every=4,
                n_scene_tokens=16,
            )
        self.assertIsInstance(e2e.compiler, SemanticCompiler)
        self.assertEqual(e2e.n_scene_tokens, 16)
        self.assertEqual(counts["compile_every"], 4)
        self.assertEqual(counts["n_scene_tokens"], 16)

    def test_default_no_compiler_and_default_counts(self):
        config = tiny_config()
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fake_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            e2e, counts = build_e2e_policy(config=config, device=torch.device("cpu"))
        self.assertIsNone(e2e.compiler)
        self.assertEqual(e2e.n_scene_tokens, 16)
        self.assertEqual(counts["compile_every"], 4)
        self.assertEqual(counts["n_scene_tokens"], 16)

    def test_compiler_params_land_in_policy_group(self):
        e2e, _ = make_compile_e2e()
        groups = parameter_groups(
            e2e, lora_lr=1e-5, vision_lr=1e-6, policy_lr=1e-4, qwen_lr=1e-7
        )
        policy_params = set(groups[0]["params"])
        for name, parameter in e2e.compiler.named_parameters():
            self.assertIn(parameter, policy_params, name)
        # 每个参数恰好出现在一组
        all_params = [p for group in groups for p in group["params"]]
        self.assertEqual(len(all_params), len(set(all_params)))


class TrainArgCompileTests(unittest.TestCase):
    def test_compile_task_requires_e2e_data(self):
        args = parse_args(["--compile-task"])
        with self.assertRaisesRegex(ValueError, "e2e-data"):
            validate_args(args)

    def test_compile_every_must_be_positive(self):
        args = parse_args(["--compile-every", "0"])
        with self.assertRaisesRegex(ValueError, "compile-every"):
            validate_args(args)

    def test_compile_task_conflicts_with_scene_teacher(self):
        args = parse_args(
            [
                "--compile-task", "--e2e-data", "x.pt",
                "--scene-teacher", "--data", "d.pt",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_args(args)

    def test_compile_task_conflicts_with_plan_resampler(self):
        args = parse_args(["--compile-task", "--e2e-data", "x.pt", "--plan-resampler"])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_args(args)

    def test_stage_a_requires_compile_task(self):
        args = parse_args(["--training-stage", "a"])
        with self.assertRaisesRegex(ValueError, "compile-task"):
            validate_args(args)

    def test_stage_a_forbids_semantic_adapter(self):
        args = parse_args(
            ["--training-stage", "a", "--compile-task", "--e2e-data", "x.pt",
             "--semantic-adapter"]
        )
        with self.assertRaisesRegex(ValueError, "semantic-adapter"):
            validate_args(args)

    def test_stage_a_anchor_geometry_must_be_zero(self):
        for flag in ("--semantic-anchor-weight", "--semantic-geometry-weight"):
            args = parse_args(
                ["--training-stage", "a", "--compile-task", "--e2e-data", "x.pt",
                 flag, "0.1"]
            )
            with self.assertRaisesRegex(ValueError, "anchor-weight"):
                validate_args(args)

    def test_stage_a_valid_passes(self):
        args = parse_args(
            ["--training-stage", "a", "--compile-task", "--e2e-data", "x.pt",
             "--single-task"]
        )
        validate_args(args)  # 不抛异常

    def test_stage_b_requires_semantic_adapter(self):
        args = parse_args(["--training-stage", "b"])
        with self.assertRaisesRegex(ValueError, "semantic-adapter"):
            validate_args(args)

    def test_stage_b_valid_passes(self):
        args = parse_args(
            ["--training-stage", "b", "--semantic-adapter", "--e2e-data", "x.pt",
             "--single-task"]
        )
        validate_args(args)

    def test_stage_c_requires_semantic_adapter(self):
        args = parse_args(["--training-stage", "c"])
        with self.assertRaisesRegex(ValueError, "semantic-adapter"):
            validate_args(args)

    def test_stage_c_valid_passes(self):
        args = parse_args(
            ["--training-stage", "c", "--semantic-adapter", "--e2e-data", "x.pt",
             "--single-task"]
        )
        validate_args(args)

    def test_no_stage_skips_stage_validation(self):
        args = parse_args([])
        self.assertIsNone(args.training_stage)
        validate_args(args)
        # compile-task 不带 stage 也允许
        args = parse_args(["--compile-task", "--e2e-data", "x.pt", "--single-task"])
        validate_args(args)

    def test_compile_defaults(self):
        args = parse_args([])
        self.assertFalse(args.compile_task)
        self.assertEqual(args.compile_every, 4)
        self.assertEqual(args.compile_n_scene, 16)
        self.assertIsNone(args.training_stage)


class CompileCheckpointTests(unittest.TestCase):
    def test_payload_carries_semantic_compiler(self):
        e2e, config = make_compile_e2e()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ck.pt"
            args = parse_args(["--save", str(path)])
            save_checkpoint(args, config, None, e2e)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertIn("semantic_compiler", ckpt)
        expected = set(e2e.compiler.state_dict().keys())
        self.assertEqual(set(ckpt["semantic_compiler"].keys()), expected)
        fresh = SemanticCompiler(language_dim=32, vision_dim=16, hidden=24)
        fresh.load_state_dict(ckpt["semantic_compiler"])
        for key, value in fresh.state_dict().items():
            torch.testing.assert_close(value, e2e.compiler.state_dict()[key])

    def test_payload_none_without_compiler(self):
        config = tiny_config()
        policy = VACompoundPolicy(config)
        vision = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        e2e = EndToEndPolicy(
            text_backbone=make_backbone(num_layers=2),
            vision_backbone=vision,
            policy=policy,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ck.pt"
            args = parse_args(["--save", str(path)])
            save_checkpoint(args, config, None, e2e)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertIn("semantic_compiler", ckpt)
        self.assertIsNone(ckpt["semantic_compiler"])

    def test_resume_branch_tolerates_missing_semantic_compiler(self):
        # 镜像 main() resume 分支的取值逻辑：老 ckpt 无该键 → 跳过，不崩溃
        resume_ckpt = {"model": {}}
        self.assertIsNone(resume_ckpt.get("semantic_compiler"))
        e2e, _ = make_compile_e2e()
        if resume_ckpt.get("semantic_compiler"):
            own_compiler = getattr(e2e, "compiler", None)
            if own_compiler is not None:
                own_compiler.load_state_dict(resume_ckpt["semantic_compiler"], strict=False)

    def test_stage_a_state_loads_into_semantic_model_missing_only_absent_keys(self):
        # Stage A（compile）ckpt 的 qwen_state_dict：裸 Qwen 参数，无 LoRA 键
        stage_a = make_backbone(num_layers=2)
        stage_a.text_model.requires_grad_(False)
        stage_a.text_model.layers[0].requires_grad_(True).train()
        qwen_state = {
            name.removeprefix("text_model."): parameter.detach().cpu()
            for name, parameter in stage_a.named_parameters()
            if parameter.requires_grad
            and "lora_a" not in name
            and "lora_b" not in name
        }
        self.assertFalse(any("lora" in key for key in qwen_state))
        # 加载到 semantic（stage b/c）模型：strict=False 通过，
        # 缺失键 = Stage A 没有的键（顶部层 LoRA 等），无 unexpected
        adapter = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1, anchor_layers=(1,)
        )
        missing, unexpected = adapter.text_model.load_state_dict(qwen_state, strict=False)
        self.assertEqual(unexpected, [])
        self.assertEqual(
            set(missing),
            set(adapter.text_model.state_dict().keys()) - set(qwen_state.keys()),
        )
        lora_names = {
            name.removeprefix("text_backbone.text_model.")
            for name, _ in adapter.named_parameters()
            if "lora_a" in name or "lora_b" in name
        }
        self.assertTrue(lora_names.issubset(missing))
        # 存在的键确实恢复：layer 0 权重与 Stage A 逐位一致
        torch.testing.assert_close(
            adapter.text_model.layers[0].self_attn.q_proj.weight,
            stage_a.text_model.layers[0].self_attn.q_proj.weight,
        )


if __name__ == "__main__":
    unittest.main()
