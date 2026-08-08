"""第二轮完整版架构重构（2026-08-08）测试。

覆盖:
- RoleQueryResampler:输出形状 / mask 正确 / role_query 开启时 TaskResampler 与
  action_query_cond 的摘要与 mask-weighted mean 不同且形状兼容;
- dual_attention:初始融合门 g < 0.2;语言 key 置零 vs 随机时 physical 更新
  逐位不变、semantic 更新变化;与 sequential_coupling 组合时 sequential 层
  走旧路径（逐位一致）;
- FlowMatchingHead semantic_context:None 与旧输出逐位一致;给定 context 时
  形状/维度正确;entry 模式忽略;
- SemanticCompiler execution_error 输入（有/无）与 history_in_dim 参数化
  （512 维历史可用）;build_e2e_policy 的 history_in_dim=config.hidden_dim /
  compile_n_readout / language_max_length / semantic_lora_suffixes 接线;
- QwenSemanticBackbone lora_suffixes 子集（q/o）时 lora_layer_count 减少;
- train.py validate_args 新参数矩阵与默认值;η_act 梯度缩放（lora 缩放、
  非 lora 不动）;
- rollout flow_semantic 的 semantic_context 传递（compile_every>0 收到非 None,
  compile_every=0 恒 None）。

全部用轻量 fake 小模型（2 层 32 维 decoder 桩），不加载真实 Qwen/V-JEPA，
不碰 GPU。
"""
import unittest
from unittest import mock

import torch
from torch import nn

from train import parse_args, scale_semantic_lora_grads, validate_args
from va_compound.backbones import (
    QwenSemanticBackbone,
    QwenTextBackbone,
    SemanticCompiler,
    VJEPA21Backbone,
)
from va_compound.end_to_end import EndToEndPolicy, build_e2e_policy
from va_compound.model import (
    FlowMatchingHead,
    LayerLanguageCache,
    RoleQueryResampler,
    VACouplingLayer,
    VACompoundConfig,
    VACompoundPolicy,
)
from torch.nn import functional as F


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
        return type(
            "O",
            (),
            {
                "last_hidden_state": x,
                "hidden_states": tuple(hidden_states) if output_hidden_states else None,
            },
        )()


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


def rollout_inputs(sequence: int, config: VACompoundConfig) -> dict:
    return dict(
        frames=torch.randint(0, 256, (2, sequence, 2, 3, 16, 16), dtype=torch.uint8),
        instructions=["pick red cup", "push blue cup"],
        proprio=torch.randn(2, sequence, config.proprio_dim),
        previous_action=torch.randn(2, sequence, config.action_dim),
        noisy_actions=torch.randn(2, sequence, config.action_horizon, config.action_dim),
        flow_time=torch.rand(2, sequence),
    )


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


def head_cache(lang_key: torch.Tensor, lang_value: torch.Tensor, num_heads: int):
    """[B, L, D] key/value -> LayerLanguageCache(头空间 [B, heads, L, hd])。"""
    batch, length, dim = lang_key.shape
    head_dim = dim // num_heads
    return LayerLanguageCache(
        key=lang_key.view(batch, length, num_heads, head_dim).transpose(1, 2),
        value=lang_value.view(batch, length, num_heads, head_dim).transpose(1, 2),
    )


class RoleQueryResamplerTests(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        resampler = RoleQueryResampler(
            hidden_dim=16, language_dim=32, n_role=4, num_heads=4
        )
        out = resampler(torch.randn(2, 6, 16), torch.ones(2, 6, dtype=torch.bool))
        self.assertEqual(tuple(out.shape), (2, 4, 16))
        # 输出保持输入 dtype（与旧 mask-weighted mean 路径一致，避免下游混合精度断裂）
        out_bf16 = resampler(
            torch.randn(2, 6, 16, dtype=torch.bfloat16),
            torch.ones(2, 6, dtype=torch.bool),
        )
        self.assertEqual(out_bf16.dtype, torch.bfloat16)

    def test_mask_excludes_padding(self):
        resampler = RoleQueryResampler(
            hidden_dim=16, language_dim=32, n_role=4, num_heads=4
        )
        key = torch.randn(2, 6, 16)
        mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
        )
        with torch.no_grad():
            out = resampler(key, mask)
            # 改 padding 位置的值 → 输出不变（softmax 权重被 mask 为 0）
            key2 = key.clone()
            key2[0, 5, :] = 1e3
            out2 = resampler(key2, mask)
            torch.testing.assert_close(out[0], out2[0])
            # 改有效位置的值 → 输出变化
            key3 = key.clone()
            key3[1, 2, :] = -1e3
            out3 = resampler(key3, mask)
            self.assertFalse(torch.allclose(out[1], out3[1]))

    def test_all_false_mask_outputs_strict_zero(self):
        # P0-高优：全 False mask 时 softmax(-inf) 是均匀分布而非零——语言序列
        # 全被遮蔽时 role 输出必须严格为零。
        resampler = RoleQueryResampler(
            hidden_dim=16, language_dim=32, n_role=4, num_heads=4
        )
        key = torch.randn(2, 6, 16)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        with torch.no_grad():
            out = resampler(key, mask)
        self.assertEqual(tuple(out.shape), (2, 4, 16))
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        # 混合批：一行全 False → 该行严格零，另一行正常
        mask_mixed = torch.tensor(
            [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool
        )
        out_mixed = resampler(key, mask_mixed)
        torch.testing.assert_close(out_mixed[0], torch.zeros_like(out_mixed[0]), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(out_mixed[1]).all())

    def test_task_resampler_role_path_differs_from_mean(self):
        config = tiny_config(memory_split=True, role_query=True, role_query_tokens=8)
        policy = VACompoundPolicy(config)
        key = torch.randn(2, 6, 16)
        mask = torch.ones(2, 6, dtype=torch.bool)
        out = policy.task_resampler(key, mask)
        self.assertEqual(tuple(out.shape), (2, 8, 16))  # n_task=8，形状兼容
        # 同模型手工均值参考：role 路径 ≠ mask-weighted mean 路径
        denom = mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        summary = (key * mask[:, :, None]).sum(1) / denom
        manual = policy.task_resampler.task_queries[None] + policy.task_resampler.mlp(
            summary[:, None, :]
        )
        self.assertFalse(torch.allclose(out, manual))
        # TaskResampler 与 policy 共享同一 RoleQueryResampler 实例
        self.assertIs(policy.task_resampler.role_resampler, policy.role_resampler)

    def test_action_query_cond_role_summary_differs(self):
        config = tiny_config(action_query_cond=True, role_query=True, role_query_tokens=8)
        policy = VACompoundPolicy(config)
        lang = torch.randn(2, 6, 32)
        mask = torch.ones(2, 6, dtype=torch.bool)
        cache = policy.build_language_cache(lang, mask)
        flat_key = cache.layers[0].key.transpose(1, 2).reshape(2, 6, 16)
        role_out = policy.role_resampler(flat_key, mask)
        self.assertEqual(tuple(role_out.shape), (2, 8, 16))
        denom = mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        mean_summary = (flat_key * mask[:, :, None]).sum(1) / denom
        self.assertFalse(torch.allclose(role_out.mean(dim=1), mean_summary))
        # encode_condition 走通，输出形状兼容
        cond = policy.encode_condition(
            torch.randn(2, 5, 16),
            torch.randn(2, 5),
            torch.randn(2, 4),
            language_cache=cache,
        )
        self.assertEqual(tuple(cond.shape), (2, 3, 16))


class DualAttentionTests(unittest.TestCase):
    def test_initial_gate_below_0_2(self):
        layer = VACouplingLayer(
            hidden_dim=16,
            language_dim=32,
            num_heads=4,
            dropout=0.0,
            mode="bidir_va",
            dual_attention=True,
        )
        action = torch.randn(2, 3, 16)
        lang_mean = torch.randn(2, 16)
        gate = torch.sigmoid(
            layer.sem_gate(torch.cat((action.mean(1), lang_mean), dim=-1))
        )
        self.assertLess(float(gate.detach().mean()), 0.2)
        # 末层 zero-init + bias=-2 → 任意输入下门都在 ~0.119
        for _ in range(5):
            gate = torch.sigmoid(
                layer.sem_gate(
                    torch.cat(
                        (torch.randn(2, 3, 16).mean(1), torch.randn(2, 16)), dim=-1
                    )
                )
            )
            self.assertLess(float(gate.detach().mean()), 0.2)

    def test_physical_update_independent_of_language(self):
        config = tiny_config(dual_attention=True)
        policy = VACompoundPolicy(config)
        layer = policy.layers[0]
        layer.eval()
        visual = torch.randn(2, 5, 16)
        action = torch.randn(2, 3, 16)
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            # 语言 value 全零 → semantic 贡献恒 0 → 动作行输出只剩 physical
            # （不读语言列）→ 语言 key 置零 vs 随机时动作输出逐位一致。
            zero_key = torch.zeros(2, 6, 16)
            v1, a1, _ = layer(
                visual, action, head_cache(zero_key, zero_key, 4), mask
            )
            rand_key = torch.randn(2, 6, 16)
            v2, a2, _ = layer(
                visual, action, head_cache(rand_key, zero_key, 4), mask
            )
            torch.testing.assert_close(a1, a2)
            # key 相同但 value 随机 → semantic 更新变化（动作行输出变化）
            rand_val = torch.randn(2, 6, 16)
            v3, a3, _ = layer(
                visual, action, head_cache(rand_key, rand_val, 4), mask
            )
            self.assertFalse(torch.allclose(a1, a3))
            # 视觉行仍走共享路径（含语言列）→ key 变化时视觉输出变化
            self.assertFalse(torch.allclose(v1, v2))

    def test_all_false_language_mask_semantic_update_strict_zero(self):
        # P0-高优：全 False 语言 mask 时 semantic 分支必须严格零（旧实现
        # softmax(-inf) → 均匀分布，输出垃圾值）。
        config = tiny_config(dual_attention=True)
        policy = VACompoundPolicy(config)
        layer = policy.layers[0]
        layer.eval()
        visual = torch.randn(2, 5, 16)
        action = torch.randn(2, 3, 16)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        with torch.no_grad():
            v1, a1, _ = layer(
                visual, action, head_cache(torch.randn(2, 6, 16), torch.randn(2, 6, 16), 4), mask
            )
            # 与语言列缺失（长度 0）对照：动作行输出与"无 semantic 贡献"一致
            # ——即语言 key/value 置零（semantic 更新为零）+ 物理路径不读语言。
            v2, a2, _ = layer(
                visual, action, head_cache(torch.zeros(2, 6, 16), torch.zeros(2, 6, 16), 4), mask
            )
            torch.testing.assert_close(a1, a2)
        # 有有效语言时同一权重下动作行应不同（semantic 更新非零）
        mask_ok = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            _, a3, _ = layer(
                visual, action, head_cache(torch.randn(2, 6, 16), torch.randn(2, 6, 16), 4), mask_ok
            )
        self.assertFalse(torch.allclose(a1, a3))

    def test_sequential_layers_keep_legacy_path(self):
        config = tiny_config(dual_attention=True, sequential_coupling=2, num_layers=4)
        policy = VACompoundPolicy(config)
        # 非 sequential 层拆双注意力；sequential 层保持旧路径（不构造 sem_gate）
        self.assertTrue(policy.layers[0].dual_attention)
        self.assertTrue(hasattr(policy.layers[0], "sem_gate"))
        self.assertFalse(policy.layers[1].dual_attention)
        self.assertFalse(hasattr(policy.layers[1], "sem_gate"))
        self.assertTrue(policy.layers[2].dual_attention)
        self.assertFalse(policy.layers[3].dual_attention)
        # 相同权重下 sequential 层输出与无 dual 基线逐位一致
        layer_dual = VACouplingLayer(
            hidden_dim=16,
            language_dim=32,
            num_heads=4,
            dropout=0.0,
            mode="bidir_va",
            sequential=True,
            dual_attention=True,
        )
        layer_base = VACouplingLayer(
            hidden_dim=16,
            language_dim=32,
            num_heads=4,
            dropout=0.0,
            mode="bidir_va",
            sequential=True,
        )
        missing, unexpected = layer_dual.load_state_dict(
            layer_base.state_dict(), strict=False
        )
        # target=layer_dual 有 sem_gate 键而 source=layer_base 没有 → missing
        self.assertEqual(
            set(missing),
            {"sem_gate.0.weight", "sem_gate.0.bias", "sem_gate.2.weight", "sem_gate.2.bias"},
        )
        self.assertEqual(unexpected, [])
        lang_hidden = torch.randn(2, 6, 32)
        cache = layer_base.project_language(lang_hidden)
        mask = torch.ones(2, 6, dtype=torch.bool)
        visual = torch.randn(2, 5, 16)
        action = torch.randn(2, 3, 16)
        layer_dual.eval()
        layer_base.eval()
        with torch.no_grad():
            out_a = layer_dual.forward_sequential(visual, action, cache, mask)
            out_b = layer_base.forward_sequential(visual, action, cache, mask)
        for x, y in zip(out_a, out_b):
            torch.testing.assert_close(x, y)


class FlowSemanticTests(unittest.TestCase):
    def test_none_identical_to_legacy(self):
        head = FlowMatchingHead(
            hidden_dim=16, action_dim=4, num_heads=4, num_layers=2, dropout=0.0,
            flow_cond="adaln", semantic_in_dim=32,
        )
        cond = torch.randn(2, 3, 16)
        noise = torch.randn(2, 3, 4)
        time = torch.rand(2)
        out_a = head(cond, noise, time)
        out_b = head(cond, noise, time, semantic_context=None)
        torch.testing.assert_close(out_a, out_b)

    def test_context_changes_output_and_shape(self):
        head = FlowMatchingHead(
            hidden_dim=16, action_dim=4, num_heads=4, num_layers=2, dropout=0.0,
            flow_cond="adaln", semantic_in_dim=32,
        )
        cond = torch.randn(2, 3, 16)
        noise = torch.randn(2, 3, 4)
        time = torch.rand(2)
        out_none = head(cond, noise, time)
        ctx = torch.randn(2, 5, 32)  # 语言空间（language_dim）
        out_ctx = head(cond, noise, time, semantic_context=ctx)
        self.assertEqual(tuple(out_ctx.shape), (2, 3, 4))
        # AdaLN 门零初始化：训练起点无条件（语义通道从零学），输出与无 ctx 一致
        torch.testing.assert_close(out_ctx, out_none)
        # 扰动调制网络后，semantic_context 确实影响输出（cat 进 cross-attn k/v）
        for mlp in head.ada_mlps:
            nn.init.normal_(mlp.weight, std=0.5)
        out_ctx2 = head(cond, noise, time, semantic_context=ctx)
        self.assertFalse(torch.allclose(out_ctx2, head(cond, noise, time)))
        with self.assertRaisesRegex(ValueError, "semantic_context"):
            head(cond, noise, time, semantic_context=torch.randn(2, 5, 8))
        with self.assertRaisesRegex(ValueError, "semantic_context"):
            head(cond, noise, time, semantic_context=torch.randn(4, 5, 32))

    def test_entry_mode_ignores_context(self):
        head = FlowMatchingHead(
            hidden_dim=16, action_dim=4, num_heads=4, num_layers=2, dropout=0.0,
            flow_cond="entry",
        )
        cond = torch.randn(2, 3, 16)
        noise = torch.randn(2, 3, 4)
        time = torch.rand(2)
        ctx = torch.randn(2, 5, 16)
        torch.testing.assert_close(
            head(cond, noise, time, semantic_context=ctx), head(cond, noise, time)
        )

    def test_policy_passthrough(self):
        policy = VACompoundPolicy(tiny_config(flow_cond="adaln", flow_semantic=True))
        # AdaLN 零初始化时无条件起点；扰动后验证 semantic_context 透传生效
        for mlp in policy.flow_head.ada_mlps:
            nn.init.normal_(mlp.weight, std=0.5)
        cond = torch.randn(2, 3, 16)
        noise = torch.randn(2, 3, 4)
        time = torch.rand(2)
        ctx = torch.randn(2, 5, 32)  # compile readout tokens（语言空间）
        self.assertFalse(
            torch.allclose(
                policy.flow_velocity(cond, noise, time),
                policy.flow_velocity(cond, noise, time, semantic_context=ctx),
            )
        )
        self.assertFalse(
            torch.allclose(
                policy.sample_actions(cond, steps=3),
                policy.sample_actions(cond, steps=3, semantic_context=ctx),
            )
        )
        self.assertFalse(
            torch.allclose(
                policy.decode_actions(cond, steps=3),
                policy.decode_actions(cond, steps=3, semantic_context=ctx),
            )
        )


class SemanticCompilerRound2Tests(unittest.TestCase):
    def setUp(self):
        self.backbone = make_backbone(num_layers=2)
        self.compiler = SemanticCompiler(
            language_dim=32,
            vision_dim=16,
            n_scene=8,
            n_hist=2,
            n_delta=2,
            n_readout=8,
            hidden=24,
            n_err=2,
            error_in_dim=8,
        )

    def test_execution_error_optional_and_shapes(self):
        scene = torch.randn(2, 20, 16)
        history = torch.randn(2, 16)
        delta = torch.randn(2, 16)
        plan_a, mask_a = self.compiler(self.backbone, ["a", "b"], scene, history, delta)
        plan_b, mask_b = self.compiler(
            self.backbone, ["a", "b"], scene, history, delta, execution_error=None
        )
        torch.testing.assert_close(plan_a, plan_b)
        torch.testing.assert_close(mask_a, mask_b)
        # 有执行误差：mask 多 n_err=2 个 token（插在 delta 之后、readout 之前）
        err = torch.randn(2, 8)
        plan_c, mask_c = self.compiler(
            self.backbone, ["a", "b"], scene, history, delta, execution_error=err
        )
        self.assertEqual(tuple(plan_c.shape), (2, 8, 32))
        self.assertEqual(mask_c.shape[1], mask_a.shape[1] + 2)
        self.assertEqual(mask_c.shape[1], 5 + 8 + 2 + 2 + 2 + 8)
        self.assertTrue(plan_c.requires_grad)

    def test_execution_error_validation(self):
        scene = torch.randn(2, 16, 16)
        history = torch.randn(2, 16)
        delta = torch.randn(2, 16)
        with self.assertRaisesRegex(ValueError, "execution_error"):
            self.compiler(
                self.backbone, ["a"], scene, history, delta,
                execution_error=torch.randn(2, 4),
            )
        with self.assertRaisesRegex(ValueError, "batch"):
            self.compiler(
                self.backbone, ["a"], scene, history, delta,
                execution_error=torch.randn(3, 8),
            )

    def test_error_projector_trainable_without_error_input(self):
        # 无 execution_error 时 error_projector 也进入计算图（梯度 0 而非 None）
        self.backbone.text_model.requires_grad_(False)
        scene = torch.randn(2, 16, 16)
        plan, _ = self.compiler(
            self.backbone, ["a", "b"], scene, torch.randn(2, 16), torch.randn(2, 16)
        )
        plan.square().mean().backward()
        for name, parameter in self.compiler.error_projector.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_history_in_dim_parameterized(self):
        compiler = SemanticCompiler(
            language_dim=32, vision_dim=16, hidden=24, history_in_dim=512
        )
        self.assertEqual(compiler.history_in_dim, 512)
        history = torch.randn(2, 512)
        plan, mask = compiler(
            self.backbone,
            ["a", "b"],
            torch.randn(2, 20, 16),
            history,
            torch.randn(2, 16),
        )
        self.assertEqual(tuple(plan.shape), (2, 8, 32))
        with self.assertRaisesRegex(ValueError, "semantic_history"):
            compiler(
                self.backbone, ["a"], torch.randn(1, 20, 16),
                torch.randn(1, 16), torch.randn(1, 16),
            )
        # 默认 None → vision_dim（向后兼容）
        default = SemanticCompiler(language_dim=32, vision_dim=16)
        self.assertEqual(default.history_in_dim, 16)
        self.assertEqual(default.error_in_dim, 16)
        self.assertEqual(default.n_err, 2)


class LoraSuffixTests(unittest.TestCase):
    def test_lora_suffixes_qo_subset_reduces_count(self):
        full = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1
        )
        self.assertEqual(full.lora_layer_count, 7)
        qo = QwenSemanticBackbone(
            make_backbone(num_layers=2),
            lora_rank=4,
            top_layers=1,
            lora_suffixes=("q_proj", "o_proj"),
        )
        self.assertEqual(qo.lora_layer_count, 2)
        self.assertLess(qo.lora_layer_count, full.lora_layer_count)
        self.assertEqual(qo.lora_suffixes, ("q_proj", "o_proj"))
        # 默认后缀 = 全 7 种
        self.assertEqual(len(full.lora_suffixes), 7)

    def test_build_e2e_policy_passes_suffixes(self):
        config = tiny_config()
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fake_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            e2e, _ = build_e2e_policy(
                config=config,
                device=torch.device("cpu"),
                semantic_adapter=True,
                semantic_top_layers=1,
                semantic_lora_suffixes=("q_proj", "o_proj"),
            )
        self.assertIsInstance(e2e.text_backbone, QwenSemanticBackbone)
        self.assertEqual(e2e.text_backbone.lora_layer_count, 2)


class TrainArgRound2Tests(unittest.TestCase):
    def test_role_query_tokens_must_be_positive(self):
        args = parse_args(
            ["--role-query", "--memory-split", "--role-query-tokens", "0"]
        )
        with self.assertRaisesRegex(ValueError, "role-query-tokens"):
            validate_args(args)

    def test_compile_n_readout_must_be_positive(self):
        args = parse_args(["--compile-n-readout", "0"])
        with self.assertRaisesRegex(ValueError, "compile-n-readout"):
            validate_args(args)

    def test_language_max_length_must_be_positive(self):
        args = parse_args(["--language-max-length", "0"])
        with self.assertRaisesRegex(ValueError, "language-max-length"):
            validate_args(args)

    def test_semantic_act_grad_scale_non_negative(self):
        args = parse_args(["--semantic-act-grad-scale", "-0.5"])
        with self.assertRaisesRegex(ValueError, "act-grad-scale"):
            validate_args(args)

    def test_semantic_lora_suffixes_non_empty(self):
        args = parse_args(["--semantic-lora-suffixes", ","])
        with self.assertRaisesRegex(ValueError, "lora-suffixes"):
            validate_args(args)

    def test_new_flags_have_structural_prerequisites(self):
        # P0-高优 fail-fast：flow-semantic 需要 compile-task + adaln；role-query
        # 需要 memory-split/action-query-cond 之一（旧版静默失效，改为报错）。
        with self.assertRaisesRegex(ValueError, "compile-task"):
            validate_args(parse_args(["--flow-semantic", "--single-task"]))
        with self.assertRaisesRegex(ValueError, "adaln"):
            validate_args(
                parse_args(["--flow-semantic", "--compile-task", "--e2e-data", "x.pt",
                            "--flow-cond", "entry", "--single-task"])
            )
        with self.assertRaisesRegex(ValueError, "memory-split"):
            validate_args(parse_args(["--role-query", "--single-task"]))
        # 满足前置条件后合法
        args = parse_args(
            [
                "--role-query", "--action-query-cond",
                "--flow-semantic", "--flow-cond", "adaln",
                "--compile-task", "--e2e-data", "x.pt", "--single-task",
            ]
        )
        validate_args(args)

    def test_dual_attention_with_sequential_warns_but_passes(self):
        args = parse_args(["--dual-attention", "--sequential-coupling", "2"])
        validate_args(args)  # 仅打印警告，不报错

    def test_dual_attention_with_coupling_one_rejected(self):
        # P0-高优：coupling=1 时每层都是 sequential，双注意力永不生效 → 报错
        args = parse_args(["--dual-attention", "--sequential-coupling", "1"])
        with self.assertRaisesRegex(ValueError, "dual-attention"):
            validate_args(args)

    def test_flow_semantic_with_adaln_allowed(self):
        args = parse_args(["--flow-semantic", "--flow-cond", "adaln", "--compile-task",
                           "--e2e-data", "x.pt"])
        validate_args(args)

    def test_round2_defaults(self):
        args = parse_args([])
        self.assertFalse(args.role_query)
        self.assertEqual(args.role_query_tokens, 16)
        self.assertFalse(args.dual_attention)
        self.assertFalse(args.flow_semantic)
        self.assertEqual(
            args.semantic_lora_suffixes,
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        self.assertEqual(args.language_max_length, 64)
        self.assertEqual(args.compile_n_readout, 16)
        self.assertEqual(args.semantic_act_grad_scale, 0.1)
        validate_args(args)


class EtaActScalingTests(unittest.TestCase):
    def test_lora_grad_scaled_others_untouched(self):
        adapter = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1
        )
        hidden, _ = adapter.encode_adapted(["a", "b"])
        hidden.square().mean().backward()
        before = {
            name: parameter.grad.clone()
            for name, parameter in adapter.named_parameters()
            if parameter.grad is not None
        }
        scale_semantic_lora_grads(adapter, 0.1)
        for name, parameter in adapter.named_parameters():
            if parameter.grad is None:
                continue
            if "lora_a" in name or "lora_b" in name:
                torch.testing.assert_close(parameter.grad, before[name] * 0.1, msg=name)
            else:
                torch.testing.assert_close(parameter.grad, before[name], msg=name)
        # 缩放后 lora_b 仍有非零梯度（lora_a 在初始化时经 lora_b=0 天然为 0 梯度，
        # 这是正确的数学行为——LoRA 初始输出为 0）
        lora_grads = [
            float(p.grad.abs().sum())
            for n, p in adapter.named_parameters()
            if p.grad is not None and ("lora_a" in n or "lora_b" in n)
        ]
        self.assertTrue(any(g > 0.0 for g in lora_grads))

    def test_scale_one_is_noop(self):
        adapter = QwenSemanticBackbone(
            make_backbone(num_layers=2), lora_rank=4, top_layers=1
        )
        hidden, _ = adapter.encode_adapted(["a"])
        hidden.square().mean().backward()
        before = {
            name: parameter.grad.clone()
            for name, parameter in adapter.named_parameters()
            if parameter.grad is not None
        }
        scale_semantic_lora_grads(adapter, 1.0)
        for name, parameter in adapter.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, before[name], msg=name)


class RolloutFlowSemanticTests(unittest.TestCase):
    def test_semantic_context_passed_when_compiling(self):
        e2e, config = make_compile_e2e(flow_semantic=True, flow_cond="adaln")
        inputs = rollout_inputs(4, config)
        seen = []
        original = e2e.policy.flow_velocity

        def spy(condition, noisy_actions, flow_time, semantic_context=None):
            seen.append(semantic_context)
            return original(condition, noisy_actions, flow_time, semantic_context=semantic_context)

        e2e.policy.flow_velocity = spy
        predicted, _, _ = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
            compile_every=2,
        )
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(ctx is not None for ctx in seen))
        self.assertEqual(tuple(seen[0].shape), (2, 8, 32))  # [B, n_readout, D]
        self.assertEqual(predicted.shape, (2, 4, 3, 4))

    def test_compile_every_zero_keeps_none(self):
        e2e, config = make_compile_e2e(flow_semantic=True, flow_cond="adaln")
        inputs = rollout_inputs(4, config)
        seen = []
        original = e2e.policy.flow_velocity

        def spy(condition, noisy_actions, flow_time, semantic_context=None):
            seen.append(semantic_context)
            return original(condition, noisy_actions, flow_time, semantic_context=semantic_context)

        e2e.policy.flow_velocity = spy
        e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
        )
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(ctx is None for ctx in seen))

    def test_rollout_flow_semantic_context_batch_dtype(self):
        # semantic_context 与 action_condition 的 dtype 可不同（flow head 内转换）
        e2e, config = make_compile_e2e(flow_semantic=True, flow_cond="adaln")
        inputs = rollout_inputs(4, config)
        predicted, _, _ = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
            compile_every=1,
        )
        self.assertEqual(predicted.shape, (2, 4, 3, 4))


class BuildE2ERound2Tests(unittest.TestCase):
    def test_compiler_wiring_history_dim_readout(self):
        # hidden_dim(8) != vision_dim(16)：history_in_dim 必须跟随 hidden_dim
        config = tiny_config(hidden_dim=8)
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
                compile_n_readout=12,
            )
        self.assertEqual(e2e.compiler.history_in_dim, 8)
        self.assertEqual(e2e.compiler.n_readout, 12)
        self.assertEqual(counts["compile_n_readout"], 12)

    def test_language_max_length_passthrough(self):
        config = tiny_config()
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained") as from_qwen,
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            from_qwen.return_value = fake_qwen
            build_e2e_policy(
                config=config, device=torch.device("cpu"), language_max_length=32
            )
        self.assertEqual(from_qwen.call_args.kwargs["max_length"], 32)

    def test_rollout_task_history_dim_matches_history_in_dim(self):
        # 真实配置链：memory.task 均值 [B, hidden_dim=8] 直连 compiler
        # history_in_dim=8（遗留修复：不再是 vision_dim=16）
        config = tiny_config(hidden_dim=8, memory_split=True)
        fake_qwen = make_backbone(num_layers=2)
        fake_vjepa = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        with (
            mock.patch.object(QwenTextBackbone, "from_pretrained", return_value=fake_qwen),
            mock.patch.object(VJEPA21Backbone, "from_pretrained", return_value=fake_vjepa),
        ):
            e2e, _ = build_e2e_policy(
                config=config,
                device=torch.device("cpu"),
                compile_task=True,
                compile_every=1,
            )
        self.assertEqual(e2e.compiler.history_in_dim, 8)
        inputs = rollout_inputs(4, config)
        predicted, _, _ = e2e.rollout(
            inputs["frames"], inputs["instructions"], inputs["proprio"],
            inputs["previous_action"], inputs["noisy_actions"], inputs["flow_time"],
            compile_every=1,
        )
        self.assertEqual(predicted.shape, (2, 4, 3, 4))


if __name__ == "__main__":
    unittest.main()
