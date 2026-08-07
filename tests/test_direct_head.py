"""C²-VA Stage A tests (2026-08-07): Direct Head + v5 executed-action labels.

Covered:
1. DirectActionHead: 输出形状 / tanh 值域 / 梯度回流；
2. VACompoundPolicy.decode_actions：direct 路径确定性、flow 路径与
   sample_actions 完全一致（现有路径不变）；
3. v5 标签管线纯函数（import /tmp/make_v5_executed_actions.py）：
   denorm/clip/renorm、去别名（同一 executed 动作 → 唯一标签）、roundtrip；
4. train.py direct-head 训练路径形状级验证（rollout_policy + smooth_l1）。
"""
import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from va_compound.model import DirectActionHead, VACompoundConfig, VACompoundPolicy

# v5 标签管线脚本已移入仓库（2026-08-07 Codex 修正 9）；/tmp 旧路径仅作回退。
V5_SCRIPT_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "scripts" / "make_v5_executed_actions.py",
    Path("/tmp/make_v5_executed_actions.py"),
)
V5_SCRIPT = next((path for path in V5_SCRIPT_CANDIDATES if path.exists()), None)


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


def load_v5_script():
    """Load the label-pipeline pure functions from scripts/ (fallback /tmp); None when missing."""
    if V5_SCRIPT is None or not V5_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("make_v5_executed_actions", V5_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_direct_head_shapes_range_and_gradient(self) -> None:
        head = DirectActionHead(hidden_dim=32, action_dim=4)
        condition = torch.randn(3, 8, 32)
        output = head(condition)
        self.assertEqual(output.shape, (3, 8, 4))
        self.assertTrue((output.abs() < 1.0).all())  # tanh 严格 (-1, 1)
        output.square().mean().backward()
        for parameter in head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_direct_head_rejects_wrong_ndim(self) -> None:
        head = DirectActionHead(hidden_dim=32, action_dim=4)
        with self.assertRaisesRegex(ValueError, "action_condition"):
            head(torch.randn(2, 4, 8, 32))

    def test_decode_actions_direct_path_is_deterministic(self) -> None:
        config = tiny_config(direct_head=True)
        model = VACompoundPolicy(config).eval()
        condition = torch.randn(2, config.action_horizon, config.hidden_dim)
        with torch.no_grad():
            first = model.decode_actions(condition, steps=32)
            second = model.decode_actions(condition, steps=32)
        self.assertEqual(first.shape, (2, config.action_horizon, config.action_dim))
        self.assertTrue((first.abs() < 1.0).all())
        torch.testing.assert_close(first, second)  # 无采样噪声

    def test_decode_actions_flow_path_matches_sample_actions(self) -> None:
        config = tiny_config()  # direct_head=False：现有 flow 路径
        model = VACompoundPolicy(config).eval()
        condition = torch.randn(2, config.action_horizon, config.hidden_dim)
        noise = torch.randn(2, config.action_horizon, config.action_dim)
        with torch.no_grad():
            decoded = model.decode_actions(condition, steps=4, noise=noise)
            sampled = model.sample_actions(condition, steps=4, noise=noise)
        torch.testing.assert_close(decoded, sampled)

    def test_direct_head_default_off(self) -> None:
        self.assertFalse(VACompoundConfig().direct_head)
        model = VACompoundPolicy(tiny_config())
        self.assertFalse(hasattr(model, "direct_head"))

    def test_config_backward_compat_without_direct_head_key(self) -> None:
        # 旧 checkpoint 的 config dict 没有 direct_head 键 → 默认 False 加载。
        old = tiny_config().__dict__.copy()
        old.pop("direct_head")
        restored = VACompoundConfig(**old)
        self.assertFalse(restored.direct_head)

    def test_config_checkpoint_roundtrip_preserves_direct_head(self) -> None:
        config = tiny_config(direct_head=True)
        restored = VACompoundConfig(**config.__dict__)  # save_checkpoint/eval 路径
        self.assertTrue(restored.direct_head)


class DirectHeadTrainingPathTests(unittest.TestCase):
    """train.py direct-head 分支：rollout_policy 解码 + smooth_l1（形状级验证）。"""

    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_rollout_policy_direct_branch_shapes_and_backward(self) -> None:
        from train import rollout_policy, sample_flow_matching_inputs, synthetic_sequence

        config = tiny_config(direct_head=True)
        model = VACompoundPolicy(config)
        batch = synthetic_sequence(config, 2, 4, torch.device("cpu"))
        noisy, flow_time, _ = sample_flow_matching_inputs(batch["actions"])
        predictions, conditions = rollout_policy(model, batch, noisy, flow_time)
        self.assertEqual(
            predictions.shape, (2, 4, config.action_horizon, config.action_dim)
        )
        self.assertEqual(
            conditions.shape, (2, 4, config.action_horizon, config.hidden_dim)
        )
        self.assertTrue((predictions.abs() < 1.0).all())
        loss = F.smooth_l1_loss(predictions, batch["actions"])  # 训练 loss 同款
        loss.backward()
        self.assertTrue(torch.isfinite(model.direct_head.net[0].weight.grad).all())
        self.assertTrue(torch.isfinite(model.action_queries.grad).all())

    def test_rollout_policy_direct_branch_with_future_predict(self) -> None:
        from train import rollout_policy, sample_flow_matching_inputs, synthetic_sequence

        # future_predict 依赖 memory_split 提供 evidence/task（train.py 同款用法）。
        config = tiny_config(direct_head=True, future_predict=True, memory_split=True)
        model = VACompoundPolicy(config)
        batch = synthetic_sequence(config, 2, 4, torch.device("cpu"))
        noisy, flow_time, _ = sample_flow_matching_inputs(batch["actions"])
        predictions, conditions, memories = rollout_policy(
            model, batch, noisy, flow_time
        )
        self.assertEqual(
            predictions.shape, (2, 4, config.action_horizon, config.action_dim)
        )
        self.assertEqual(len(memories), 4)
        # future loss 不依赖 flow，direct 模式下照常可用
        pred_future = model.future_predictor(
            conditions[:, 0], memories[0].evidence, memories[0].task
        )
        target_future = batch["vision_tokens"][:, 1].mean(dim=1)
        future_loss = model.future_predictor.future_loss(pred_future, target_future)
        self.assertTrue(torch.isfinite(future_loss))
        self.assertGreaterEqual(future_loss.detach().item(), 0.0)


class ExecutedLabelPipelineTests(unittest.TestCase):
    """v5 标签管线纯函数（import /tmp/make_v5_executed_actions.py）。"""

    def setUp(self) -> None:
        self.m = load_v5_script()
        if self.m is None:
            self.skipTest(f"missing {V5_SCRIPT}")
        torch.manual_seed(7)

    def test_denorm_matches_eval_contract(self) -> None:
        q01 = torch.tensor([-3.96, -3.39, -12.74, -1.0])
        q99 = torch.tensor([8.86, 7.92, 10.0, 1.0])
        norm = torch.tensor([0.0, 0.5, -1.0, 1.0])
        torch.testing.assert_close(
            self.m.denorm(norm, q01, q99),
            norm * (q99 - q01) / 2 + (q99 + q01) / 2,
        )
        # eval 契约：denorm(±1) 必须恰好映射到 q99/q01。
        torch.testing.assert_close(self.m.denorm(torch.ones(4), q01, q99), q99)
        torch.testing.assert_close(self.m.denorm(-torch.ones(4), q01, q99), q01)

    def test_executed_clips_raw_to_unit_box(self) -> None:
        q01 = torch.tensor([-5.0, -5.0, -5.0, -1.0])
        q99 = torch.tensor([10.0, 10.0, 10.0, 1.0])
        norm = torch.tensor([[1.0, -1.0, 0.2, 0.5]])  # raw=[10, -5, 1, 0.5]
        executed = self.m.executed_from_actions(norm, q01, q99)
        self.assertTrue((executed.abs() <= 1.0).all())
        torch.testing.assert_close(executed, torch.tensor([[1.0, -1.0, 1.0, 0.5]]))
        # 越界原始动作全部收敛到边界原子 ±1（环境 clip 行为）。

    def test_pollution_fix_same_executed_same_label(self) -> None:
        # 核心修复：raw=1.0 与 raw=7.3 执行动作相同（都 clip 到 1.0），但 v4
        # 标签不同（-0.23 vs 0.76）——标签污染；v5 中两者必须共享同一标签。
        q01 = torch.tensor([-3.9621773, -3.3909476, -12.741096, -1.0])
        q99 = torch.tensor([8.8622, 7.924748, 10.0, 1.0])
        raw = torch.tensor([[1.0, 7.3, 0.5, 0.9]])
        v4_style = (raw - (q01 + q99) / 2) / ((q99 - q01) / 2)  # 未 clip 的 v4 标签
        self.assertNotEqual(v4_style[0, 0].item(), v4_style[0, 1].item())  # 旧标签不同
        labels, new_q01, new_q99 = self.m.build_executed_labels(v4_style, q01, q99)
        executed = self.m.executed_from_actions(v4_style, q01, q99)
        torch.testing.assert_close(executed[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(executed[0, 1], torch.tensor(1.0))
        self.assertEqual(labels[0, 0].item(), labels[0, 1].item())  # 新标签一致
        self.assertTrue((labels.abs() <= 1.0).all())

    def test_quantiles_are_1_and_99_percent_of_executed(self) -> None:
        executed = torch.clamp(3.0 * torch.randn(2000, 4), -1.0, 1.0)
        q01, q99 = self.m.quantiles_from_executed(executed, quantile=0.01)
        expected_q01, expected_q99 = torch.quantile(
            executed.reshape(-1, 4),
            torch.tensor([0.01, 0.99], dtype=executed.dtype),
            dim=0,
        )
        torch.testing.assert_close(q01, expected_q01)
        torch.testing.assert_close(q99, expected_q99)

    def test_denorm_roundtrip_recovers_executed(self) -> None:
        # 真实数据情形：executed 分布有 ±1 边界原子 → 新分位数恰为 ±1 →
        # renorm 恒等 → denorm(新标签) 精确还原 executed（验证脚本的输出 #4）。
        executed = torch.clamp(3.0 * torch.randn(5000, 4), -1.0, 1.0)
        q01, q99 = self.m.quantiles_from_executed(executed)
        torch.testing.assert_close(q01, -torch.ones(4))  # 边界原子 → 分位数到边
        torch.testing.assert_close(q99, torch.ones(4))
        labels = self.m.renorm_executed(executed, q01, q99)
        torch.testing.assert_close(labels, executed)  # renorm 恒等
        roundtrip = self.m.denorm(labels, q01, q99)
        torch.testing.assert_close(roundtrip, executed, atol=1e-6, rtol=1e-6)

    def test_renorm_clips_tails_when_quantiles_are_interior(self) -> None:
        # 分位数在内部（无边界原子）时，尾值映射后越过 ±1 → 保留 clip。
        q01 = torch.tensor([-0.5, -0.5, -0.5, -0.5])
        q99 = torch.tensor([0.5, 0.5, 0.5, 0.5])
        executed = torch.tensor([[-1.0, -0.25, 0.25, 1.0]])
        labels = self.m.renorm_executed(executed, q01, q99)
        self.assertTrue((labels.abs() <= 1.0).all())  # -1.0 → -1, 1.0 → +1（clip）
        self.assertAlmostEqual(float(labels[0, 0]), -1.0)
        self.assertAlmostEqual(float(labels[0, 3]), 1.0)
        # 界内值线性映射不变：-0.25 → (-0.25 - 0)/0.5 = -0.5
        self.assertAlmostEqual(float(labels[0, 1]), -0.5)

    def test_bucket_consistency_detects_violations(self) -> None:
        executed = torch.tensor([[0.5], [0.5], [0.3], [-1.0], [-1.0]])
        consistent = torch.tensor([[0.1], [0.1], [0.2], [-0.9], [-0.9]])
        buckets, violations = self.m.check_bucket_consistency(executed, consistent)
        self.assertEqual(violations, 0)
        self.assertEqual(buckets, 3)  # {0.5, 0.3, -1.0}

        inconsistent = torch.tensor([[0.1], [0.2], [0.2], [-0.9], [-0.9]])
        _, violations = self.m.check_bucket_consistency(executed, inconsistent)
        self.assertEqual(violations, 1)  # executed=0.5 桶内标签不一致

    def test_process_previous_action_keeps_t0_zero(self) -> None:
        q01 = torch.tensor([-3.96, -3.39, -12.74, -1.0])
        q99 = torch.tensor([8.86, 7.92, 10.0, 1.0])
        prev = torch.randn(6, 3, 4).clamp(-1.0, 1.0)
        prev[:, 0] = 0.0  # t=0 契约标记
        labels, new_q01, new_q99 = self.m.build_executed_labels(
            torch.randn(6, 3, 8, 4), q01, q99
        )
        processed = self.m.process_previous_action(prev, q01, q99, new_q01, new_q99)
        self.assertTrue((processed[:, 0] == 0.0).all())  # t=0 保持全零
        self.assertTrue((processed.abs() <= 1.0).all())
        # t>0 与 actions 共享同一新归一化空间（同一确定性管线）。
        torch.testing.assert_close(
            processed[:, 1:],
            self.m.renorm_executed(
                self.m.executed_from_actions(prev[:, 1:], q01, q99), new_q01, new_q99
            ),
        )


if __name__ == "__main__":
    unittest.main()
