"""C²-VA Stage B tests（2026-08-07，Codex 评审版）：收缩控制型 VA。

Covered:
1. C2ActionHead/ControllerParams 形状（ū/c̄/K）、Δc_0 硬置 0、K 零初始化；
2. K=0 时严格等价 Stage A Direct Head（含 checkpoint 兼容语义）；
3. decode_actions c2 分支数学（手动构造 e 验证 a = clip(ū − K·(c − c̄))）与
   oracle_reference（e≡0）；
4. ControllableProjection P：形状 / set_pca 冻结 / 权重形状校验；
5. train.py c2 训练路径：rollout_policy 返回 references、compute_c2_loss 反向、
   恢复残差 mask（e≈0 样本排除）、RecoveryDataset 加载；
6. 收缩指标 compute_contract_metrics：d_i、M_c1/M_c6、ρ 时间换算
   （ρ₁ = 0.8^(1/6) ≈ 0.9635，ρ₆ = 0.8）；
7. eval_metaworld.py c2_schedule 部署节奏（plan/feedback stride 解耦）；
8. prepare_mw_recovery.py：phase-stratified anchors、窗口帧、snapshot/restore
   determinism（metaworld 可用时）。
"""
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from va_compound.model import (
    C2ActionHead,
    ControllableProjection,
    ControllerParams,
    VACompoundConfig,
    VACompoundPolicy,
)


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


def make_model(**overrides) -> VACompoundPolicy:
    return VACompoundPolicy(tiny_config(**overrides))


class ControllerHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_c2_head_shapes_delta0_zero_and_gain_zero_init(self) -> None:
        head = C2ActionHead(hidden_dim=32, action_dim=6, control_dim=16)
        condition = torch.randn(3, 5, 32)
        delta, gain = head(condition)
        self.assertEqual(delta.shape, (3, 5, 16))
        self.assertEqual(gain.shape, (3, 5, 6, 16))
        torch.testing.assert_close(delta[:, 0], torch.zeros(3, 16))  # Δc_0 ≡ 0 硬设
        torch.testing.assert_close(gain, torch.zeros_like(gain))  # K 零初始化

    def test_controller_params_layout_and_anchor(self) -> None:
        config = tiny_config(direct_head=True, c2_controller=True)
        model = make_model(direct_head=True, c2_controller=True)
        condition = torch.randn(2, config.action_horizon, config.hidden_dim)
        c_current = model.control_projector(torch.randn(2, 64, config.vision_dim))
        params = model.controller_params(condition, c_current)
        self.assertIsInstance(params, ControllerParams)
        self.assertEqual(params.nominal.shape, (2, config.action_horizon, config.action_dim))
        self.assertEqual(params.reference.shape, (2, config.action_horizon, 16))
        self.assertEqual(params.gain.shape, (2, config.action_horizon, config.action_dim, 16))
        # c̄_0 = sg(c_anchor)（Δc_0=0）→ 与 c_current 一致
        torch.testing.assert_close(params.reference[:, 0], c_current.detach())

    def test_controller_params_rejects_bad_c_current(self) -> None:
        model = make_model(direct_head=True, c2_controller=True)
        condition = torch.randn(2, 5, 32)
        with self.assertRaisesRegex(ValueError, "c_current must have shape"):
            model.controller_params(condition, torch.randn(2, 7))
        with self.assertRaisesRegex(ValueError, "batch size"):
            model.controller_params(condition, torch.randn(3, 16))

    def test_apply_controller_symbol_and_clip(self) -> None:
        # 手动构造 K/e 验证 a = clip(ū − K·e)：K[:, :, :6, :6] = 0.5I
        batch, horizon, action_dim = 2, 5, 6
        nominal = torch.randn(batch, horizon, action_dim)
        reference = torch.randn(batch, horizon, 16)
        gain = torch.zeros(batch, horizon, action_dim, 16)
        gain[:, :, :6, :6] = 0.5 * torch.eye(6)
        params = ControllerParams(nominal=nominal, reference=reference, gain=gain)
        c_current = torch.randn(batch, 16)
        decoded = params.apply_controller(c_current)
        error = c_current[:, None, :] - reference
        manual = torch.clamp(nominal - 0.5 * error[:, :, :6], -1.0, 1.0)
        torch.testing.assert_close(decoded, manual, atol=1e-6, rtol=1e-6)
        self.assertTrue((decoded.abs() <= 1.0).all())

    def test_apply_controller_oracle_reference_zeroes_error(self) -> None:
        nominal = torch.tanh(torch.randn(2, 5, 6))  # Direct Head 输出域（tanh）
        params = ControllerParams(
            nominal=nominal,
            reference=torch.randn(2, 5, 16),
            gain=0.5 * torch.randn(2, 5, 6, 16),
        )
        c_current = torch.randn(2, 16)
        torch.testing.assert_close(
            params.apply_controller(c_current, oracle_reference=True),
            nominal,
        )

    def test_decode_actions_c2_requires_c_current(self) -> None:
        model = make_model(direct_head=True, c2_controller=True)
        condition = torch.randn(2, 5, 32)
        with self.assertRaisesRegex(ValueError, "c_current"):
            model.decode_actions(condition)
        c_current = model.control_projector(torch.randn(2, 64, 20))
        decoded = model.decode_actions(condition, c_current=c_current)
        self.assertEqual(decoded.shape, (2, 5, 6))
        self.assertTrue((decoded.abs() <= 1.0).all())

    def test_control_projection_shape_and_set_pca_freeze(self) -> None:
        projector = ControllableProjection(vision_dim=20, control_dim=16)
        vision = torch.randn(3, 64, 20)
        c = projector(vision)
        self.assertEqual(c.shape, (3, 16))
        with self.assertRaisesRegex(ValueError, "vision_tokens"):
            projector(torch.randn(3, 64, 20, 1))
        projector.set_pca(torch.randn(16, 20), torch.randn(16))
        self.assertTrue(all(not p.requires_grad for p in projector.parameters()))
        with self.assertRaisesRegex(ValueError, "pca weight"):
            projector.set_pca(torch.randn(16, 21), torch.randn(16))
        with self.assertRaisesRegex(ValueError, "pca bias"):
            projector.set_pca(torch.randn(16, 20), torch.randn(17))

    def test_k_zero_equals_stage_a_direct_head(self) -> None:
        # 同一 config 下：c2 模型初始（K≡0）解码 ≡ 无 c2 的 direct 模型解码。
        torch.manual_seed(3)
        config = tiny_config(direct_head=True, c2_controller=True)
        model_c2 = make_model(direct_head=True, c2_controller=True)
        model_a = make_model(direct_head=True)
        # 共享 Stage A 权重（c2 迁移语义：direct_head 权重直接复用）
        model_c2.direct_head.load_state_dict(model_a.direct_head.state_dict())
        condition = torch.randn(4, config.action_horizon, config.hidden_dim)
        c_current = model_c2.control_projector(torch.randn(4, 64, config.vision_dim))
        decoded_c2 = model_c2.decode_actions(condition, c_current=c_current)
        decoded_a = model_a.decode_actions(condition)
        torch.testing.assert_close(decoded_c2, decoded_a)

    def test_config_default_off_and_validation(self) -> None:
        self.assertFalse(VACompoundConfig().c2_controller)
        model = make_model()
        self.assertFalse(hasattr(model, "c2_head"))
        with self.assertRaisesRegex(ValueError, "c2_controller requires direct_head"):
            tiny_config(c2_controller=True)
        # 旧 checkpoint config dict 无 c2 键 → 默认 False 兼容
        old = tiny_config(direct_head=True).__dict__.copy()
        old.pop("c2_controller")
        restored = VACompoundConfig(**old)
        self.assertFalse(restored.c2_controller)
        self.assertTrue(restored.direct_head)
        # c2_control_dim 默认 16 且可 checkpoint 往返
        restored2 = VACompoundConfig(**tiny_config(direct_head=True, c2_controller=True).__dict__)
        self.assertTrue(restored2.c2_controller)
        self.assertEqual(restored2.c2_control_dim, 16)


class C2TrainingPathTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            c2_lambda_f=0.1, c2_lambda_r=1.0, c2_lambda_c=0.0,
        )

    def test_rollout_policy_c2_returns_references(self) -> None:
        from train import rollout_policy, synthetic_sequence

        config = tiny_config(direct_head=True, c2_controller=True)
        model = make_model(direct_head=True, c2_controller=True)
        batch = synthetic_sequence(config, 2, 4, torch.device("cpu"))
        batch["step_targets"] = torch.randn(2, 4, 6, 16)
        predictions, conditions, references = rollout_policy(model, batch, None, None)
        self.assertEqual(predictions.shape, (2, 4, config.action_horizon, config.action_dim))
        self.assertEqual(conditions.shape, (2, 4, config.action_horizon, config.hidden_dim))
        self.assertEqual(references.shape, (2, 4, config.action_horizon, 16))
        self.assertTrue((predictions.abs() <= 1.0).all())
        # 初始（K≡0）c2 解码 ≡ direct 解码
        direct = model.direct_head(conditions[:, 0])
        torch.testing.assert_close(predictions[:, 0], direct)

    def test_compute_c2_loss_backward_and_components(self) -> None:
        from train import compute_c2_loss, synthetic_sequence

        config = tiny_config(direct_head=True, c2_controller=True)
        model = make_model(direct_head=True, c2_controller=True)
        clean = synthetic_sequence(config, 4, 4, torch.device("cpu"))
        clean["step_targets"] = torch.randn(4, 4, 6, 16)
        # 假 recovery batch（与 RecoveryDataset collate 后的形状一致；
        # tiny config action_horizon=5 → step_index ∈ 0..4）
        n = 6
        rec = {
            "vision_tokens": torch.randn(n, 1, 64, config.vision_dim),
            "language_hidden": torch.randn(n, 8, config.language_dim),
            "language_mask": torch.ones(n, 8, dtype=torch.bool),
            "proprio": torch.randn(n, config.proprio_dim),
            "prev_action": torch.randn(n, config.action_dim).clamp(-1, 1),
            "expert_action": torch.randn(n, config.action_dim).clamp(-1, 1),
            "step_index": torch.tensor([0, 1, 2, 3, 4, 4]),
            "branch_id": torch.zeros(n, dtype=torch.long),
            "c_perturbed": torch.randn(n, 16),
            "c_nominal": torch.randn(n, 16),
        }
        rec["c_perturbed"][0] = rec["c_nominal"][0].clone()  # e=0（应被 mask）
        loss, logs = compute_c2_loss(model, clean, rec, self._args())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        # 冻结外的新参数有梯度；direct_head 不在训练集合（Stage A 冻结语义由
        # main() 实现，这里验证 loss 只依赖 c2 头可反传）。
        self.assertTrue(torch.isfinite(model.c2_head.gain_head[0].weight.grad).all())
        self.assertTrue(torch.isfinite(model.c2_head.reference_head[0].weight.grad).all())
        self.assertAlmostEqual(logs["rec_used"], 5.0)  # e≈0 的第 0 步被 mask

    def test_compute_c2_loss_recursion_and_gradient_flow(self) -> None:
        from train import compute_c2_loss, synthetic_sequence

        config = tiny_config(
            direct_head=True, c2_controller=True, memory_split=True,
        )
        model = make_model(
            direct_head=True, c2_controller=True, memory_split=True,
        )
        clean = synthetic_sequence(config, 4, 4, torch.device("cpu"))
        clean["step_targets"] = torch.randn(4, 4, 6, 16)
        rec = {
            "vision_tokens": torch.randn(2, 1, 64, config.vision_dim),
            "language_hidden": torch.randn(2, 8, config.language_dim),
            "language_mask": torch.ones(2, 8, dtype=torch.bool),
            "proprio": torch.randn(2, config.proprio_dim),
            "prev_action": torch.randn(2, config.action_dim).clamp(-1, 1),
            "expert_action": torch.randn(2, config.action_dim).clamp(-1, 1),
            "step_index": torch.tensor([1, 4]),
            "branch_id": torch.zeros(2, dtype=torch.long),
            "c_perturbed": torch.randn(2, 16),
            "c_nominal": torch.randn(2, 16),
        }
        loss, logs = compute_c2_loss(model, clean, rec, self._args())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_recovery_dataset_loads_v6b_payload(self) -> None:
        from train import RecoveryDataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v6b.pt"
            n = 8
            payload = {
                "vision_tokens_t": torch.randn(n, 1, 64, 768, dtype=torch.float16),
                "proprio": torch.randn(n, 4),
                "prev_action": torch.randn(n, 4),
                "expert_action": torch.randn(n, 4),
                "step_index": torch.arange(n) % 6,
                "branch_id": torch.arange(n) // 6,
                "c_perturbed": torch.randn(n, 16),
                "c_nominal": torch.randn(n, 16),
                "language_hidden": torch.randn(1, 13, 2048, dtype=torch.float16),
                "language_mask": torch.ones(1, 13, dtype=torch.bool),
            }
            torch.save(payload, path)
            dataset = RecoveryDataset(path)
            self.assertEqual(len(dataset), n)
            item = dataset[3]
            self.assertEqual(item["vision_tokens"].shape, (1, 64, 768))
            self.assertEqual(item["proprio"].shape, (4,))
            self.assertEqual(item["step_index"].item(), 3 % 6)
        # 不存在文件 → FileNotFoundError（自然失败路径）
        with self.assertRaises(FileNotFoundError):
            RecoveryDataset(Path("/nonexistent.pt"))
        # 存在但缺键 → ValueError
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.pt"
            torch.save({"expert_action": torch.zeros(2, 4)}, bad)
            with self.assertRaisesRegex(ValueError, "missing tensors"):
                RecoveryDataset(bad)


class ContractMetricsTests(unittest.TestCase):
    def test_rho_time_conversion(self) -> None:
        from train import compute_contract_metrics

        # 空数据：不崩，rho 换算正确（ρ₁ = ρ₆^(1/6)）
        empty = {
            "c_perturbed": torch.zeros(0, 16),
            "c_nominal": torch.zeros(0, 16),
            "branch_id": torch.zeros(0, dtype=torch.long),
            "step_index": torch.zeros(0, dtype=torch.long),
        }
        metrics = compute_contract_metrics(empty, rho6=0.8)
        self.assertAlmostEqual(metrics["rho6"], 0.8)
        self.assertAlmostEqual(metrics["rho1"], 0.8 ** (1 / 6), places=5)
        self.assertAlmostEqual(metrics["rho1"], 0.9635, places=4)

    def test_contract_metrics_aggregation(self) -> None:
        from train import compute_contract_metrics

        # 两个 branch、6 步：d = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]（理想收缩）
        # 与 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]（不收缩）
        d0 = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        d1 = np.ones(6)
        c_pert = torch.zeros(12, 16)
        c_nom = torch.zeros(12, 16)
        for index, d in enumerate(np.concatenate((d0, d1))):
            c_pert[index, 0] = d  # ‖c_δ − c_0‖ = d
        payload = {
            "c_perturbed": c_pert,
            "c_nominal": c_nom,
            "branch_id": torch.tensor([0] * 6 + [1] * 6),
            "step_index": torch.tensor(list(range(6)) + list(range(6))),
        }
        metrics = compute_contract_metrics(payload, rho6=0.8)
        # d_i = ‖c_δ − c_0‖/√16：branch0 步 i 的 d = raw_i/4 = 0.25, 0.225, 0.2, ...
        # M_c1 每步 max(0, d_{i+1} − 0.9635·d_i)：
        #   branch0 全 <0（理想收缩）；branch1 不收缩：0.25 − 0.9635·0.25 = 0.009125/步
        self.assertAlmostEqual(metrics["M_c1"], (5 * 0.009125) / 10, places=5)
        # M_c6: branch0: 0.5/4 − 0.8·1.0/4 < 0 → 0；branch1: 1/4 − 0.8·1/4 = 0.05
        self.assertAlmostEqual(metrics["M_c6"], 0.05 / 2, places=5)
        self.assertAlmostEqual(metrics["d0"], 1.0 / 4, places=5)
        self.assertAlmostEqual(metrics["d5"], (0.5 / 4 + 1.0 / 4) / 2, places=5)

    def test_contract_metrics_mask(self) -> None:
        from train import compute_contract_metrics

        payload = {
            "c_perturbed": torch.zeros(4, 16),
            "c_nominal": torch.zeros(4, 16),
            "branch_id": torch.tensor([0, 0, 1, 1]),
            "step_index": torch.tensor([0, 5, 0, 5]),
        }
        payload["c_perturbed"][0, 0] = 1.0
        payload["c_perturbed"][2, 0] = 1.0
        mask = torch.tensor([True, True, False, False])  # 只看 branch 0
        metrics = compute_contract_metrics(payload, rho6=0.8, mask=mask)
        self.assertAlmostEqual(metrics["d0"], 1.0 / 4)
        self.assertAlmostEqual(metrics["M_c6"], max(0.0, 0.0 - 0.8 * 0.25))


class C2ScheduleTests(unittest.TestCase):
    def test_plan_feedback_decoupled_schedule(self) -> None:
        from eval_metaworld import c2_schedule

        # plan_stride=6, feedback=1：token 0..5 顺序消费，第 6 步重规划
        events = []
        plan_step = None
        for step in range(12):
            due, fb_due, token = c2_schedule(step, plan_step, 6, 1, 8)
            if due:
                plan_step = step
                due, fb_due, token = True, True, 0
            events.append((step, due, fb_due, token))
        expected = [
            (0, True, True, 0), (1, False, True, 1), (2, False, True, 2),
            (3, False, True, 3), (4, False, True, 4), (5, False, True, 5),
            (6, True, True, 0), (7, False, True, 1), (8, False, True, 2),
            (9, False, True, 3), (10, False, True, 4), (11, False, True, 5),
        ]
        self.assertEqual(events, expected)

    def test_feedback_stride_keeps_action(self) -> None:
        from eval_metaworld import c2_schedule

        plan_step = 0
        # feedback=2：step 1 不刷新（保持动作），step 2 刷新 token 1
        self.assertEqual(c2_schedule(1, plan_step, 6, 2, 8), (False, False, 0))
        self.assertEqual(c2_schedule(2, plan_step, 6, 2, 8), (False, True, 1))
        # token 用尽（plan_stride 超 horizon）强制重规划
        due, fb_due, token = c2_schedule(9, plan_step, 20, 1, 8)
        self.assertTrue(due and fb_due and token == 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            c2_schedule(0, None, 0, 1, 8)


class RecoveryCollectionTests(unittest.TestCase):
    def test_phase_stratified_anchors(self) -> None:
        from prepare_mw_recovery import phase_stratified_anchors

        self.assertEqual(phase_stratified_anchors(60, 5), [1, 15, 30, 44, 59])
        self.assertEqual(phase_stratified_anchors(3, 5), [])  # 轨迹过短
        self.assertEqual(phase_stratified_anchors(30, 4), [1, 10, 19, 29])

    def test_window_frames_stride2(self) -> None:
        from prepare_mw_recovery import window_frames

        frame_log = {i: np.full((2, 2, 3), float(i)) for i in range(10, 18)}
        window = window_frames(frame_log, 16)
        self.assertEqual([int(frame[0, 0, 0]) for frame in window], [10, 12, 14, 16])

    def test_perturb_kind_config(self) -> None:
        from prepare_mw_recovery import (
            PERTURB_KIND_ORDER,
            DEFAULT_PERTURB_MIX,
            resolve_perturb_mix,
        )

        kinds, weights = resolve_perturb_mix(None, has_object_joint=True)
        self.assertEqual(kinds, PERTURB_KIND_ORDER)
        self.assertEqual(
            tuple(float(p) for p in DEFAULT_PERTURB_MIX.split(",")),
            (0.5, 0.3, 0.2),
        )
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_snapshot_restore_determinism(self) -> None:
        metaworld = None
        try:
            import metaworld  # noqa: F401
        except ImportError:
            self.skipTest("metaworld not installed")
        from prepare_mw_recovery import DEFAULT_ENV, make_env, restore_env, snapshot_env

        env = make_env(seed=0, env_name=DEFAULT_ENV)
        first = snapshot_env(env)
        env.data.qpos[:] = 0.0
        env.data.qvel[:] = 1.0
        restore_env(env, first)
        second = snapshot_env(env)
        for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "prev_obs", "target_pos"):
            self.assertTrue(np.array_equal(first[key], second[key]), key)
        self.assertEqual(first["time"], second["time"])
        # 再次恢复（重复 restore 同样确定）
        env.data.qpos[:] = 99.0
        restore_env(env, first)
        third = snapshot_env(env)
        for key in ("qpos", "qvel", "act"):
            self.assertTrue(np.array_equal(first[key], third[key]), key)
        env.close()

    def test_snapshot_torch_roundtrip(self) -> None:
        """数据文件里的 torch 张量快照经 restore_env 恢复（eval 恢复评估路径）。"""
        metaworld = None
        try:
            import metaworld  # noqa: F401
        except ImportError:
            self.skipTest("metaworld not installed")
        from prepare_mw_recovery import (
            DEFAULT_ENV,
            make_env,
            restore_env,
            snapshot_env,
            snapshot_to_tensors,
        )

        env = make_env(seed=0, env_name=DEFAULT_ENV)
        reference = snapshot_env(env)  # numpy 快照
        snapshot = snapshot_to_tensors(reference)  # 数据文件存储形态（torch）
        env.data.qpos[:] = 0.0
        restore_env(env, snapshot)
        restored = snapshot_env(env)
        for key in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "prev_obs", "target_pos"):
            self.assertTrue(np.array_equal(reference[key], restored[key]), key)
        self.assertEqual(reference["time"], restored["time"])
        env.close()


if __name__ == "__main__":
    unittest.main()
