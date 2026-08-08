"""相位完整采样（build_phase_starts）与采样协议一致性校验的单测。"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from prepare_metaworld import build_phase_starts


class BuildPhaseStartsTests(unittest.TestCase):
    def setUp(self) -> None:
        # 与 prepare_metaworld 常量一致：SEQUENCE_LENGTH=4, CONTROL_STRIDE=6,
        # ACTION_HORIZON=8 → required_span = 3*6 + 7 = 25。
        self.required_span = 25

    def test_last_start_pinned_for_n_ge_2(self) -> None:
        for length in (80, 150, 300, 600):
            for n in (2, 4, 6, 8):
                starts = build_phase_starts(length, self.required_span, n, seed=0)
                last = length - 1 - self.required_span
                self.assertEqual(starts[-1], last, f"L={length} n={n}")

    def test_n1_pins_last_start(self) -> None:
        self.assertEqual(build_phase_starts(80, self.required_span, 1, seed=0), [54])

    def test_too_short_episode_returns_empty(self) -> None:
        self.assertEqual(build_phase_starts(20, self.required_span, 6, seed=0), [])
        self.assertEqual(build_phase_starts(25, self.required_span, 6, seed=0), [])

    def test_deterministic_same_seed(self) -> None:
        a = build_phase_starts(300, self.required_span, 8, seed=0)
        b = build_phase_starts(300, self.required_span, 8, seed=0)
        self.assertEqual(a, b)

    def test_different_seed_changes_starts(self) -> None:
        a = build_phase_starts(300, self.required_span, 8, seed=0)
        b = build_phase_starts(300, self.required_span, 8, seed=7)
        self.assertNotEqual(a, b)

    def test_starts_within_bounds_and_sorted_unique(self) -> None:
        length = 300
        last = length - 1 - self.required_span
        for n in (2, 6, 8, 16):
            starts = build_phase_starts(length, self.required_span, n, seed=3)
            self.assertEqual(starts, sorted(set(starts)))
            self.assertTrue(all(0 <= s <= last for s in starts))

    def test_dedup_when_n_exceeds_available(self) -> None:
        # 极短但可用的 episode：合法起点少于 n_windows → 去重后不超界
        starts = build_phase_starts(40, self.required_span, 20, seed=0)
        self.assertEqual(starts, sorted(set(starts)))
        self.assertTrue(all(0 <= s <= 14 for s in starts))


class SamplingConsistencyTests(unittest.TestCase):
    """live 路径：payload metadata 中的采样协议必须与调用参数一致（Grok P0）。"""

    def test_mismatched_sampling_params_raise(self) -> None:
        from va_compound.live_vjepa import build_mw_plans

        payload = {
            "vision_tokens": torch.empty(9927, 4),
            "metadata": {
                "tasks": [f"task_{i}" for i in range(49)],
                "sampling": {
                    "mode": "phase",
                    "phase_bins": 6,
                    "phase_seed": 0,
                    "sequences_per_episode": 4,
                },
            },
        }
        # 数据提取用 phase_bins=6，live 误用 phase_bins=0（spe=4）→ 必须报错
        with self.assertRaises(ValueError) as ctx:
            build_mw_plans(payload, "/nonexistent/root", spe=4, phase_bins=0)
        self.assertIn("采样参数与 payload metadata 不一致", str(ctx.exception))

    def test_legacy_payload_without_sampling_still_works(self) -> None:
        from va_compound.live_vjepa import build_mw_plans

        root = "/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
        if not __import__("os").path.exists(root):
            self.skipTest("raw metaworld root not mounted")
        payload = {
            "vision_tokens": torch.empty(1, 4),
            "metadata": {"tasks": ["task_0"]},
        }
        # 无 sampling 字段的旧 payload：不校验，仅行数对齐（应报行数错而非参数错）
        with self.assertRaises(ValueError) as ctx:
            build_mw_plans(payload, root, spe=4, phase_bins=0)
        self.assertIn("live plans", str(ctx.exception))

    def test_mode_aware_check_ignores_irrelevant_spe(self) -> None:
        """Codex P1-1：phase 模式下 SPE 不生效，spe 不一致不应误拒。"""
        from prepare_metaworld import pq
        from va_compound.live_vjepa import build_mw_plans

        root = "/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
        if not __import__("os").path.exists(root):
            self.skipTest("raw metaworld root not mounted")
        tasks = pq.read_table(root + "/meta/tasks.parquet").to_pylist()
        task_texts = [t["__index_level_0__"] for t in tasks][:49]
        n = 14867  # phase_bins=6 的行数（与 prepare 同参数）
        payload = {
            "vision_tokens": torch.empty(n, 4),
            "metadata": {
                "tasks": task_texts,
                "control_stride": 6,
                "sampling": {
                    "mode": "phase",
                    "phase_bins": 6,
                    "phase_seed": 0,
                    "sequences_per_episode": 99,  # phase 模式无关，故意不一致
                    "success_only": False,
                },
            },
        }
        plans = build_mw_plans(payload, root, spe=4, phase_bins=6, phase_seed=0)
        self.assertEqual(len(plans), n)

    def test_control_stride_mismatch_raises(self) -> None:
        """Codex P0-2：control_stride 与 payload 不一致必须报错。"""
        from va_compound.live_vjepa import build_mw_plans

        payload = {
            "vision_tokens": torch.empty(1, 4),
            "metadata": {"tasks": ["task_0"], "control_stride": 6},
        }
        with self.assertRaises(ValueError) as ctx:
            build_mw_plans(payload, "/nonexistent/root", control_stride=2, spe=4)
        self.assertIn("control_stride", str(ctx.exception))


class FrameAugmentTests(unittest.TestCase):
    """π0.5 式帧增强：形状/类型保持，随机性生效。"""

    def test_shape_dtype_preserved(self) -> None:
        from va_compound.live_vjepa import augment_frames

        frames = np.zeros((2, 4, 384, 384, 3), dtype=np.uint8)
        frames[..., 0] = 128  # 非纯黑，避免增强后仍为 0
        out = augment_frames(frames)
        self.assertEqual(out.shape, frames.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_augmentation_changes_pixels(self) -> None:
        from va_compound.live_vjepa import augment_frames

        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (1, 1, 384, 384, 3), dtype=np.uint8)
        out = augment_frames(frames)
        self.assertFalse(np.array_equal(out, frames))

    def test_augmentation_keeps_clip_consistency(self) -> None:
        """Codex P1-2：同一 V-JEPA 时间窗的 4 帧共享一组增强参数——
        输入相同的帧增强后必须仍然相同（无假相机运动/闪烁）。"""
        from va_compound.live_vjepa import augment_frames

        rng = np.random.default_rng(1)
        clip = rng.integers(0, 256, (1, 4, 384, 384, 3), dtype=np.uint8)
        clip[0, 1] = clip[0, 0]  # 帧 0 与帧 1 完全一致
        out = augment_frames(clip)
        np.testing.assert_array_equal(out[0, 0], out[0, 1])


if __name__ == "__main__":
    unittest.main()
