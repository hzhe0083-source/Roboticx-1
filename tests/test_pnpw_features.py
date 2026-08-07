import unittest

import numpy as np

from prepare_pnpw_features import (
    build_sample_plans,
    clip_frame_indices,
    robust_normalize,
)


class PNPWFeatureTests(unittest.TestCase):
    def test_sample_plans_remain_inside_episode(self) -> None:
        episodes = [
            {
                "episode_index": 7,
                "length": 30,
                "tasks": ["pick white cube into the basket"],
            }
        ]
        plans = build_sample_plans(
            episodes,
            sequence_length=4,
            control_stride=2,
            sequence_stride=5,
            action_horizon=5,
            action_stride=1,
        )
        self.assertEqual([plan.decision_frames[0] for plan in plans], [0, 5, 10, 15])
        for plan in plans:
            last_target = plan.decision_frames[-1] + 4
            self.assertLess(last_target, 30)

    def test_clip_indices_are_causal_and_clamped_to_episode_start(self) -> None:
        indices = clip_frame_indices(
            2,
            video_start_frame=100,
            window=4,
            stride=2,
        )
        self.assertEqual(indices, (100, 100, 100, 102))

    def test_robust_normalization_clips_to_unit_interval(self) -> None:
        values = np.array([[-2.0, 5.0], [0.0, 10.0], [2.0, 15.0]], dtype=np.float32)
        low = np.array([-1.0, 5.0], dtype=np.float32)
        high = np.array([1.0, 15.0], dtype=np.float32)
        normalized = robust_normalize(values, low, high)
        np.testing.assert_allclose(
            normalized,
            np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
