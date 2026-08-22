"""参数化恢复数据链路的新增测试（不 import metaworld）。

覆盖 prepare_mw_recovery.resolve_perturb_mix：默认混合解析 / 非 object-joint
任务自动降级 / 显式 object_joint 报错 / 显式零权重允许 / 权重归一化 / 非法输入。

原先本文件还测 make_quick_pilot 的 sample_rows/subsample_v5/v6a/v6b。该模块是
一次性试点脚手架（从未进仓库，靠 ``sys.path.insert(0, "/tmp")`` 导入），已随
休眠实验线一并清理，对应的 9 个测试同期移除。
"""
import unittest

import numpy as np

from prepare_mw_recovery import DEFAULT_PERTURB_MIX, resolve_perturb_mix


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


if __name__ == "__main__":
    unittest.main()
