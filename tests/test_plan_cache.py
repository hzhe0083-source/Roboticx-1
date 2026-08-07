"""Plan-Cache tests (2026-08-07): PlanResampler (方案 B), Qwen scene-teacher
(方案 A) and the closed-loop plan-refresh cache-rebuild logic.
"""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from va_compound.backbones import QwenTextBackbone, SceneTeacher
from va_compound.model import VACompoundConfig, VACompoundPolicy
from eval_metaworld import build_plan_language_cache, plan_refresh_due
from train import rollout_policy, sample_flow_matching_inputs, synthetic_sequence


def tiny_config(**overrides) -> VACompoundConfig:
    values = dict(
        language_dim=24,
        vision_dim=20,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        action_horizon=5,
        action_dim=6,
        proprio_dim=9,
    )
    values.update(overrides)
    return VACompoundConfig(**values)


class SceneFakeTokenizer:
    def __call__(self, texts, **_kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.arange(5).repeat(batch, 1),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]).repeat(batch, 1),
        }


class SceneFakeTextModel(nn.Module):
    """Frozen fake supporting inputs_embeds/position_ids.

    Applies a causal-ish cumulative transform so readout positions depend on
    everything before them (instructions + scene pseudo tokens).
    """

    def __init__(self, dim: int = 12):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, dim)

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **_kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class PlanCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    # ---- 方案 B：PlanResampler ----

    def test_plan_resampler_shapes_and_plan_cache(self) -> None:
        config = tiny_config(plan_resampler=True)
        model = VACompoundPolicy(config)
        scene = torch.randn(2, config.vision_dim)
        language = torch.randn(2, 7, config.language_dim)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        plan = model.plan_resampler(scene, language, mask)
        self.assertEqual(plan.shape, (2, 8, config.language_dim))
        self.assertTrue(plan.requires_grad)

        cache = model.build_plan_cache(scene, language, mask)
        self.assertEqual(cache.attention_mask.shape, (2, 15))  # 7 language + 8 plan
        # 原语言 mask 保留（padding 仍被屏蔽），plan tokens 全可见。
        torch.testing.assert_close(cache.attention_mask[:, :7], mask)
        self.assertTrue(cache.attention_mask[:, 7:].all())
        self.assertEqual(cache.layers[0].key.shape[2], 15)

    def test_plan_resampler_full_forward_backward(self) -> None:
        config = tiny_config(plan_resampler=True)
        model = VACompoundPolicy(config)
        scene = torch.randn(2, config.vision_dim)
        language = torch.randn(2, 7, config.language_dim)
        mask = torch.ones(2, 7, dtype=torch.bool)
        cache = model.build_plan_cache(scene, language, mask)
        vision = torch.randn(2, 11, config.vision_dim)
        noisy = torch.randn(2, config.action_horizon, config.action_dim)
        flow_time = torch.rand(2)
        predicted = model(
            vision,
            torch.randn(2, config.proprio_dim),
            torch.randn(2, config.action_dim),
            noisy,
            flow_time,
            language_cache=cache,
        )
        self.assertEqual(predicted.shape, (2, config.action_horizon, config.action_dim))
        predicted.square().mean().backward()
        self.assertTrue(torch.isfinite(model.plan_resampler.plan_queries.grad).all())
        self.assertTrue(torch.isfinite(model.plan_resampler.scene_proj.weight.grad).all())

    def test_plan_resampler_conditions_on_scene(self) -> None:
        config = tiny_config(plan_resampler=True)
        model = VACompoundPolicy(config).eval()
        language = torch.randn(1, 7, config.language_dim)
        mask = torch.ones(1, 7, dtype=torch.bool)
        scene_a = torch.randn(1, config.vision_dim)
        scene_b = torch.randn(1, config.vision_dim)
        with torch.no_grad():
            plan_a = model.plan_resampler(scene_a, language, mask)
            plan_a_repeat = model.plan_resampler(scene_a, language, mask)
            plan_b = model.plan_resampler(scene_b, language, mask)
        torch.testing.assert_close(plan_a, plan_a_repeat)
        self.assertFalse(torch.allclose(plan_a, plan_b, atol=1e-5))

    def test_plan_resampler_default_off(self) -> None:
        model = VACompoundPolicy(tiny_config())
        self.assertFalse(hasattr(model, "plan_resampler"))
        with self.assertRaisesRegex(ValueError, "plan_resampler is disabled"):
            model.build_plan_cache(torch.randn(1, 20), torch.randn(1, 7, 24))

    # ---- 方案 A：encode_with_scene ----

    def test_encode_with_scene_shapes_mask_and_gradients(self) -> None:
        backbone = QwenTextBackbone(SceneFakeTokenizer(), SceneFakeTextModel())
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(2, 16)
        plan, full_mask = backbone.encode_with_scene(
            ["pick red cup", "push blue cup"], scene, teacher.scene_projector, teacher.readout_tokens
        )
        self.assertEqual(plan.shape, (2, 8, 12))
        self.assertEqual(full_mask.shape, (2, 5 + 8 + 8))  # 5 tokens + 8 scene + 8 readout
        self.assertEqual(full_mask.dtype, torch.bool)
        self.assertTrue(plan.requires_grad)
        plan.square().mean().backward()
        # projector 与 readout 参数必须拿到梯度（冻结 Qwen 带梯度前向）。
        self.assertTrue(teacher.scene_projector[0].weight.grad is not None)
        self.assertTrue(torch.isfinite(teacher.scene_projector[0].weight.grad).all())
        self.assertTrue(torch.isfinite(teacher.readout_tokens.grad).all())
        self.assertGreater(float(teacher.readout_tokens.grad.abs().sum()), 0.0)

    def test_encode_with_scene_conditions_on_scene(self) -> None:
        backbone = QwenTextBackbone(SceneFakeTokenizer(), SceneFakeTextModel())
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene_a = torch.randn(1, 16)
        scene_b = torch.randn(1, 16)
        with torch.no_grad():
            plan_a, _ = backbone.encode_with_scene(
                ["pick red cup"], scene_a, teacher.scene_projector, teacher.readout_tokens
            )
            plan_b, _ = backbone.encode_with_scene(
                ["pick red cup"], scene_b, teacher.scene_projector, teacher.readout_tokens
            )
        self.assertFalse(torch.allclose(plan_a, plan_b, atol=1e-5))

    def test_scene_teacher_forward_wires_encode_with_scene(self) -> None:
        backbone = QwenTextBackbone(SceneFakeTokenizer(), SceneFakeTextModel())
        teacher = SceneTeacher(language_dim=12, vision_dim=16, n_scene=8, n_readout=8)
        scene = torch.randn(2, 16)
        plan, full_mask = teacher(backbone, ["a", "b"], scene)
        self.assertEqual(plan.shape, (2, 8, 12))
        self.assertEqual(full_mask.shape, (2, 21))

    # ---- train.py 集成 ----

    def test_rollout_policy_plan_resampler_branch(self) -> None:
        config = tiny_config(plan_resampler=True)
        model = VACompoundPolicy(config)
        batch = synthetic_sequence(config, 2, 4, torch.device("cpu"))
        noisy, flow_time, _ = sample_flow_matching_inputs(batch["actions"])
        velocities, conditions = rollout_policy(model, batch, noisy, flow_time)
        self.assertEqual(
            velocities.shape, (2, 4, config.action_horizon, config.action_dim)
        )
        velocities.square().mean().backward()
        self.assertTrue(torch.isfinite(model.plan_resampler.plan_queries.grad).all())

    def test_rollout_policy_scene_teacher_branch(self) -> None:
        config = tiny_config(scene_teacher=True)
        model = VACompoundPolicy(config)
        backbone = QwenTextBackbone(SceneFakeTokenizer(), SceneFakeTextModel(dim=24))
        teacher = SceneTeacher(language_dim=24, vision_dim=20, n_scene=8, n_readout=8)
        batch = synthetic_sequence(config, 2, 4, torch.device("cpu"))
        batch["language_hidden"] = batch["language_hidden"][:, :5]  # fake: 5 tokens
        batch["language_mask"] = batch["language_mask"][:, :5]
        noisy, flow_time, _ = sample_flow_matching_inputs(batch["actions"])
        velocities, conditions = rollout_policy(
            model, batch, noisy, flow_time,
            text_backbone=backbone, scene_teacher=teacher, tasks=["t0", "t1"],
        )
        self.assertEqual(
            velocities.shape, (2, 4, config.action_horizon, config.action_dim)
        )
        velocities.square().mean().backward()
        # readout 路径的梯度必须回传到 SceneTeacher 参数。
        self.assertTrue(teacher.readout_tokens.grad is not None)
        self.assertGreater(float(teacher.readout_tokens.grad.abs().sum()), 0.0)

    # ---- eval_metaworld.py：plan-refresh 缓存重建 ----

    def test_plan_refresh_due_first_decision_always_builds(self) -> None:
        self.assertTrue(plan_refresh_due(1, 0))
        self.assertTrue(plan_refresh_due(1, 3))

    def test_plan_refresh_due_cadence(self) -> None:
        self.assertFalse(plan_refresh_due(2, 0))
        self.assertFalse(plan_refresh_due(2, 3))
        self.assertTrue(plan_refresh_due(4, 3))  # 每 3 个决策：1, 4, 7, ...
        self.assertFalse(plan_refresh_due(5, 3))
        self.assertTrue(plan_refresh_due(7, 3))

    def test_build_plan_language_cache_plan_resampler(self) -> None:
        config = tiny_config(plan_resampler=True)
        model = VACompoundPolicy(config).eval()
        hidden = torch.randn(1, 7, config.language_dim)
        mask = torch.ones(1, 7, dtype=torch.bool)
        scene = torch.randn(1, config.vision_dim)
        cache = build_plan_language_cache(model, hidden, mask, scene)
        self.assertEqual(cache.attention_mask.shape, (1, 15))
        self.assertTrue(cache.attention_mask[:, 7:].all())

    def test_build_plan_language_cache_scene_teacher(self) -> None:
        config = tiny_config(scene_teacher=True)
        model = VACompoundPolicy(config).eval()
        backbone = QwenTextBackbone(SceneFakeTokenizer(), SceneFakeTextModel(dim=24))
        teacher = SceneTeacher(language_dim=24, vision_dim=20, n_scene=8, n_readout=8)
        hidden = torch.randn(1, 5, config.language_dim)
        mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
        scene = torch.randn(1, config.vision_dim)
        cache = build_plan_language_cache(
            model, hidden, mask, scene,
            instruction="pick red cup", text_backbone=backbone, scene_teacher=teacher,
        )
        self.assertEqual(cache.attention_mask.shape, (1, 13))  # 5 + 8 plan
        torch.testing.assert_close(cache.attention_mask[:, :5], mask)
        self.assertTrue(cache.attention_mask[:, 5:].all())

    def test_build_plan_language_cache_plain_passthrough(self) -> None:
        config = tiny_config()
        model = VACompoundPolicy(config).eval()
        hidden = torch.randn(1, 7, config.language_dim)
        mask = torch.ones(1, 7, dtype=torch.bool)
        cache = build_plan_language_cache(model, hidden, mask, torch.randn(1, 20))
        self.assertEqual(cache.attention_mask.shape, (1, 7))
        self.assertFalse(hasattr(model, "plan_resampler"))


if __name__ == "__main__":
    unittest.main()
