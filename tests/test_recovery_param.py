"""参数化恢复数据链路的新增测试（不 import metaworld）。

覆盖：
- prepare_mw_recovery.resolve_perturb_mix：默认混合解析 / 非 object-joint
  任务自动降级 / 显式 object_joint 报错 / 显式零权重允许 / 权重归一化 /
  非法输入；
- make_quick_pilot：sample_rows 确定性+不相交、v5 子集形状/重编号/契约
  校验器、v6b 按 branch 抽样结构、v6a 行抽取。
"""
import collections
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp")

from make_quick_pilot import (  # noqa: E402
    sample_rows,
    subsample_v5,
    subsample_v6a,
    subsample_v6b,
    validate_v5_pilot,
)
from prepare_mw_recovery import (  # noqa: E402
    DEFAULT_PERTURB_MIX,
    resolve_perturb_mix,
)

TASKS = ["Press a button", "Insert a peg sideways"]


class PerturbMixTests(unittest.TestCase):
    def test_default_mix_parses_for_object_joint_env(self):
        kinds, weights = resolve_perturb_mix(None, has_object_joint=True)
        self.assertEqual(kinds, ("action", "tcp", "object_joint"))
        np.testing.assert_allclose(weights, (0.5, 0.3, 0.2))

    def test_default_mix_degrades_without_object_joint(self):
        kinds, weights = resolve_perturb_mix(None, has_object_joint=False)
        self.assertEqual(kinds, ("action", "tcp"))
        self.assertEqual(weights, (0.5, 0.5))

    def test_explicit_object_joint_without_env_joint_raises(self):
        with self.assertRaises(ValueError):
            resolve_perturb_mix("0.5,0.3,0.2", has_object_joint=False)

    def test_explicit_zero_object_joint_allowed_without_env_joint(self):
        kinds, weights = resolve_perturb_mix("0.5,0.5,0", has_object_joint=False)
        self.assertEqual(kinds, ("action", "tcp", "object_joint"))
        np.testing.assert_allclose(weights, (0.5, 0.5, 0.0))

    def test_explicit_mix_normalized(self):
        _, weights = resolve_perturb_mix("1,1,1", has_object_joint=True)
        np.testing.assert_allclose(weights, (1 / 3, 1 / 3, 1 / 3), atol=1e-6)

    def test_malformed_mix_rejected(self):
        for bad in ("0.5,0.3", "a,b,c", "0.5,0.3,-0.2", "0,0,0"):
            with self.assertRaises(ValueError, msg=bad):
                resolve_perturb_mix(bad, has_object_joint=True)

    def test_default_constant_used_as_fallback(self):
        kinds, weights = resolve_perturb_mix(None, has_object_joint=True)
        self.assertEqual(kinds, ("action", "tcp", "object_joint"))
        np.testing.assert_allclose(
            weights,
            [float(p) for p in DEFAULT_PERTURB_MIX.split(",")],
        )


def synthetic_v5(n_per_task=10, tasks=TASKS):
    """满足 v5 契约的合成 payload（prev[t=0] 全零、prev[t>0]=actions[t-1,5]）。"""
    n = n_per_task * len(tasks)
    actions = torch.randn(n, 4, 8, 4)
    previous = torch.zeros(n, 4, 4)
    previous[:, 1:] = actions[:, :-1, 5]
    return {
        "vision_tokens": torch.randn(n, 4, 3, 8, dtype=torch.float16),
        "language_hidden": torch.randn(n, 3, 16),
        "language_mask": torch.ones(n, 3, dtype=torch.bool),
        "proprio": torch.randn(n, 4, 4),
        "previous_action": previous,
        "actions": actions,
        "pair_id": torch.arange(n),
        "episode_id": torch.arange(n),
        "instruction_id": torch.tensor(
            [i for i in range(len(tasks)) for _ in range(n_per_task)], dtype=torch.long
        ),
        "normalization": {
            "action_q01": torch.zeros(4),
            "action_q99": torch.ones(4),
            "state_q01": torch.zeros(4),
            "state_q99": torch.ones(4),
        },
        "metadata": {"contract": "language_conditioned_mt50", "tasks": list(tasks), "fps": 80},
    }


class QuickPilotV5Tests(unittest.TestCase):
    def test_sample_rows_deterministic_and_disjoint(self):
        train1, val1 = sample_rows(200, 24, 16, seed=1234)
        train2, val2 = sample_rows(200, 24, 16, seed=1234)
        self.assertTrue(np.array_equal(train1, train2))
        self.assertTrue(np.array_equal(val1, val2))
        self.assertEqual(len(train1), 24)
        self.assertEqual(len(val1), 16)
        self.assertEqual(len(set(train1.tolist()) & set(val1.tolist())), 0)

    def test_sample_rows_insufficient(self):
        with self.assertRaises(ValueError):
            sample_rows(10, 24, 16)

    def test_subsample_v5_shapes_and_renumbering(self):
        payload = synthetic_v5(n_per_task=10)
        train_p, val_p, row_map = subsample_v5(
            payload, TASKS, n_train=4, n_val=2, seed=1234
        )
        for p in (train_p, val_p):
            for key in (
                "vision_tokens", "language_hidden", "language_mask", "proprio",
                "previous_action", "actions", "pair_id", "episode_id",
                "instruction_id", "normalization", "metadata",
            ):
                self.assertIn(key, p)
        self.assertEqual(tuple(train_p["vision_tokens"].shape), (8, 4, 3, 8))
        self.assertEqual(tuple(val_p["vision_tokens"].shape), (4, 4, 3, 8))
        self.assertEqual(
            collections.Counter(train_p["instruction_id"].tolist()), {0: 4, 1: 4}
        )
        self.assertEqual(
            collections.Counter(val_p["instruction_id"].tolist()), {0: 2, 1: 2}
        )
        self.assertEqual(train_p["metadata"]["tasks"], TASKS)
        self.assertEqual(val_p["metadata"]["tasks"], TASKS)
        for task, (tr, vl) in row_map.items():
            self.assertEqual(len(tr), 4)
            self.assertEqual(len(vl), 2)

    def test_validate_v5_pilot_passes_and_catches(self):
        train_p, _, _ = subsample_v5(
            synthetic_v5(n_per_task=10), TASKS, n_train=4, n_val=2, seed=1234
        )
        self.assertEqual(validate_v5_pilot(train_p), [])
        broken = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in train_p.items()}
        broken["previous_action"][1, 1, :] += 0.5
        self.assertNotEqual(validate_v5_pilot(broken), [])
        broken2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in train_p.items()}
        broken2["instruction_id"] = torch.tensor([0, 1, 2, 3, 0, 1, 0, 1])
        self.assertNotEqual(validate_v5_pilot(broken2), [])

    def test_subsample_v5_unknown_task(self):
        with self.assertRaises(ValueError):
            subsample_v5(synthetic_v5(), ["no such task", "Press a button"], 4, 2)


def synthetic_v6b(n_train_branches=4, n_heldout_branches=2, steps=3):
    n_branches = n_train_branches + n_heldout_branches
    n_rows = n_branches * steps
    split_bools = [False] * (n_train_branches * steps) + [True] * (n_heldout_branches * steps)
    return {
        "vision_tokens_t": torch.randn(n_rows, 1, 3, 8, dtype=torch.float16),
        "proprio": torch.randn(n_rows, 4),
        "prev_action": torch.randn(n_rows, 4),
        "expert_action": torch.randn(n_rows, 4),
        "step_index": torch.tensor([i % steps for i in range(n_rows)]),
        "branch_id": torch.arange(n_branches).repeat_interleave(steps),
        "seed": torch.zeros(n_rows, dtype=torch.long),
        "split": torch.tensor(split_bools, dtype=torch.bool),
        "c_perturbed": torch.randn(n_rows, 16),
        "c_nominal": torch.randn(n_rows, 16),
        "language_hidden": torch.randn(1, 3, 16),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
        "pca": {"weight": torch.randn(16, 768), "bias": torch.randn(16)},
        "recovery_start": [
            {"split": "train" if i < n_train_branches else "heldout"}
            for i in range(n_branches)
        ],
        "metadata": {"task": "Press a button", "recovery_steps": steps},
    }


class QuickPilotV6bTests(unittest.TestCase):
    def test_subsample_v6b_branch_structure(self):
        payload = synthetic_v6b(n_train_branches=4, n_heldout_branches=2, steps=3)
        out = subsample_v6b(payload, n_train_branches=3, n_heldout_branches=1, seed=1234)
        self.assertEqual(tuple(out["vision_tokens_t"].shape), (12, 1, 3, 8))
        self.assertEqual(len(out["recovery_start"]), 4)
        self.assertEqual(len(out["branch_id"]), 12)
        self.assertEqual(int(out["split"].sum()), 3)  # 1 heldout branch × 3 条
        self.assertEqual(set(out["branch_id"].tolist()), {0, 1, 2, 3})
        counts = collections.Counter(out["branch_id"].tolist())
        self.assertEqual(sorted(counts.values()), [3, 3, 3, 3])

    def test_subsample_v6b_insufficient_branches(self):
        payload = synthetic_v6b(n_train_branches=4, n_heldout_branches=2, steps=3)
        with self.assertRaises(ValueError):
            subsample_v6b(payload, n_train_branches=5, n_heldout_branches=1)

    def test_subsample_v6a_rows(self):
        step_targets = torch.randn(200, 4, 6, 16, dtype=torch.float16)
        subset, info = subsample_v6a(step_targets, [1, 3, 5], [2, 4])
        self.assertEqual(tuple(subset.shape), (5, 4, 6, 16))
        self.assertEqual(info, {"train": [1, 3, 5], "val": [2, 4]})
        self.assertTrue(torch.equal(subset[0], step_targets[1]))
        self.assertTrue(torch.equal(subset[-1], step_targets[4]))

    def test_subsample_v6a_invalid_rows(self):
        with self.assertRaises(ValueError):
            subsample_v6a(torch.randn(10, 4, 6, 16), [10], [0])
        with self.assertRaises(ValueError):
            subsample_v6a(torch.randn(10, 4, 6, 16), [0], [0])


if __name__ == "__main__":
    unittest.main()
