"""扰动数据管道（scripts/prepare_mw_perturbations.py）纯函数测试（不 import metaworld）。

覆盖：
- resolve_perturb_mix：默认混合 / 归一化 / 非法输入；
- sample_perturb_delta：幅度 2–8mm、方向约束（横向/高度/物体/peg-hole）、确定性；
- contact_candidates：gripper / distance / any 判据、决策点上下界、非法 mode；
- window_frame_rows：与 clip_frame_indices 同序（最老帧在前）+ 行号引用语义；
- robust_normalize / norm_state / norm_action_clip：v5 空间与 executed-clip 契约；
- validate_payload：v5 同构键 + 扰动标注键 + prev 契约 + 幅度范围 + metadata。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # prepare_mw_perturbations 位于 scripts/

from prepare_mw_perturbations import (  # noqa: E402
    ACTION_HORIZON,
    CONTROL_STRIDE,
    DEFAULT_PERTURB_MIX,
    MAG_MAX,
    MAG_MIN,
    PERTURB_KIND_ORDER,
    SEQUENCE_LENGTH,
    VISION_STRIDE,
    VISION_WINDOW,
    contact_candidates,
    norm_action_clip,
    norm_state,
    resolve_perturb_mix,
    robust_normalize,
    sample_perturb_delta,
    validate_payload,
    window_frame_rows,
)

SQ01 = np.array([-0.4, 0.4, 0.0, 0.3])
SQ99 = np.array([0.4, 0.9, 0.4, 1.0])
AQ01 = np.full(4, -1.0)
AQ99 = np.full(4, 1.0)


class PerturbMixTests(unittest.TestCase):
    def test_default_mix_parses(self):
        kinds, weights = resolve_perturb_mix(None)
        self.assertEqual(kinds, PERTURB_KIND_ORDER)
        expected = [float(p) for p in DEFAULT_PERTURB_MIX.split(",")]
        np.testing.assert_allclose(weights, tuple(w / sum(expected) for w in expected))

    def test_explicit_mix_normalized(self):
        kinds, weights = resolve_perturb_mix("1,1,1,1")
        self.assertEqual(kinds, PERTURB_KIND_ORDER)
        np.testing.assert_allclose(weights, (0.25, 0.25, 0.25, 0.25), atol=1e-9)

    def test_zero_kind_weight_allowed(self):
        kinds, weights = resolve_perturb_mix("1,0,0,0")
        self.assertEqual(kinds, PERTURB_KIND_ORDER)
        np.testing.assert_allclose(weights, (1.0, 0.0, 0.0, 0.0))

    def test_malformed_mix_rejected(self):
        for bad in ("0.5,0.3", "a,b,c,d", "1,1,1,-1", "0,0,0,0", "1 2 3 4"):
            with self.assertRaises(ValueError, msg=bad):
                resolve_perturb_mix(bad)


class PerturbDeltaTests(unittest.TestCase):
    def test_magnitude_in_2_8mm_and_direction_constraints(self):
        rng = np.random.default_rng(0)
        for kind in PERTURB_KIND_ORDER:
            for _ in range(20):
                mag = float(rng.uniform(MAG_MIN, MAG_MAX))
                delta = sample_perturb_delta(kind, mag, rng)
                self.assertAlmostEqual(np.linalg.norm(delta), mag, places=6)
                if kind in ("eef_lateral", "peg_hole_relative"):
                    self.assertAlmostEqual(delta[2], 0.0, places=12)  # 水平面
                if kind == "eef_height":
                    self.assertAlmostEqual(delta[0], 0.0, places=12)
                    self.assertAlmostEqual(delta[1], 0.0, places=12)

    def test_deterministic_with_same_rng_state(self):
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        for kind in PERTURB_KIND_ORDER:
            np.testing.assert_allclose(
                sample_perturb_delta(kind, 0.005, rng1),
                sample_perturb_delta(kind, 0.005, rng2),
            )

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            sample_perturb_delta("nope", 0.005, np.random.default_rng(0))


def synthetic_obs(n_frames=80, seed=0):
    """合成 [L,4] obs.state + [L,39] environment_state：gripper 前段闭合、
    后段张开；手→obj1 距离中段近。保证三种 contact 判据都有命中且互有差异。"""
    rng = np.random.default_rng(seed)
    state = np.zeros((n_frames, 4))
    state[:, 3] = np.linspace(0.3, 0.9, n_frames)  # 前段 < 0.5（闭合），后段 > 0.5
    env_obs = np.zeros((n_frames, 39))
    env_obs[:, 0:3] = 0.0
    env_obs[:, 4:7] = np.linspace(0.15, 0.03, n_frames)[:, None]  # 距离 0.15 → 0.03
    env_obs[:, 7:39] = rng.normal(size=(n_frames, 32))
    return state, env_obs


class ContactCandidatesTests(unittest.TestCase):
    LENGTH = 80
    LAST = LENGTH - 1 - (SEQUENCE_LENGTH - 1) * CONTROL_STRIDE - (ACTION_HORIZON - 1)  # 54

    def test_gripper_mode_bounds_and_criterion(self):
        state, env_obs = synthetic_obs(self.LENGTH)
        cands = contact_candidates(
            state, env_obs, mode="gripper", gripper_threshold=0.5,
            distance_threshold=0.08,
        )
        self.assertTrue(np.all(cands >= 8))
        self.assertTrue(np.all(cands <= self.LAST))
        self.assertTrue(np.all(state[cands, 3] < 0.5))

    def test_distance_mode_criterion(self):
        state, env_obs = synthetic_obs(self.LENGTH)
        cands = contact_candidates(
            state, env_obs, mode="distance", gripper_threshold=0.5,
            distance_threshold=0.08,
        )
        dists = np.linalg.norm(env_obs[cands, 0:3] - env_obs[cands, 4:7], axis=1)
        self.assertTrue(np.all(dists < 0.08))

    def test_any_mode_is_union(self):
        state, env_obs = synthetic_obs(self.LENGTH)
        gripper = contact_candidates(
            state, env_obs, mode="gripper", gripper_threshold=0.5, distance_threshold=0.08,
        )
        distance = contact_candidates(
            state, env_obs, mode="distance", gripper_threshold=0.5, distance_threshold=0.08,
        )
        any_c = contact_candidates(
            state, env_obs, mode="any", gripper_threshold=0.5, distance_threshold=0.08,
        )
        self.assertEqual(
            sorted(set(gripper.tolist()) | set(distance.tolist())), any_c.tolist()
        )

    def test_no_room_returns_empty(self):
        state, env_obs = synthetic_obs(30)  # 30 帧 < 8 + 26
        cands = contact_candidates(
            state, env_obs, mode="any", gripper_threshold=0.5, distance_threshold=0.08,
        )
        self.assertEqual(len(cands), 0)

    def test_invalid_mode_raises(self):
        state, env_obs = synthetic_obs(self.LENGTH)
        with self.assertRaises(ValueError):
            contact_candidates(state, env_obs, mode="bad", gripper_threshold=0.5,
                               distance_threshold=0.08)


class WindowFrameRowsTests(unittest.TestCase):
    def test_order_and_clamp(self):
        rows = window_frame_rows(episode_start=1000, decision=50)
        self.assertEqual(rows.tolist(), [1044, 1046, 1048, 1050])  # 最老帧在前，stride 2
        early = window_frame_rows(episode_start=1000, decision=3)
        self.assertEqual(early.tolist(), [1000, 1000, 1001, 1003])  # 越界帧 clamp 到 episode 首行

    def test_decision_grid_consistency(self):
        # v5 决策网格：s + 6t 的窗口行号与 prev/actions 契约不冲突
        for t in range(SEQUENCE_LENGTH):
            rows = window_frame_rows(1000, 20 + t * CONTROL_STRIDE)
            self.assertEqual(len(rows), VISION_WINDOW)
            self.assertEqual(rows[-1], 1000 + 20 + t * CONTROL_STRIDE)


class NormalizationTests(unittest.TestCase):
    def test_norm_action_clip_identity_for_v5(self):
        # v5 action_q01/q99 = ±1 → executed-clip 归一化 = clip 本身
        raw = np.array([[-1.5, 0.2, 0.9, 7.0]])
        out = norm_action_clip(raw, AQ01, AQ99)
        np.testing.assert_allclose(out, np.clip(raw, -1.0, 1.0))

    def test_norm_state_maps_quantiles_to_bounds(self):
        raw = np.array([[-0.4, 0.9, 0.0, 1.0]])
        out = norm_state(raw, SQ01, SQ99)
        np.testing.assert_allclose(out[0], [-1.0, 1.0, -1.0, 1.0], atol=1e-5)

    def test_robust_normalize_zero_scale_guard(self):
        out = robust_normalize(np.array([0.5]), np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(out, [0.0])


def synthetic_payload(n=3, frame_size=96, mags=None):
    """满足契约的合成 payload（prev[t>0] = actions[t-1][5]，与 v5 一致）。"""
    rng = np.random.default_rng(0)
    actions = np.clip(rng.normal(0.0, 0.5, size=(n, SEQUENCE_LENGTH, ACTION_HORIZON, 4)), -1, 1)
    previous = np.zeros((n, SEQUENCE_LENGTH, 4))
    previous[:, 1:] = actions[:, :-1, 5]
    magnitudes = np.asarray(mags if mags is not None else rng.uniform(0.002, 0.008, size=n))
    return {
        "vision_tokens": torch.zeros(n, SEQUENCE_LENGTH, 64, 768, dtype=torch.float16),
        "language_hidden": torch.randn(n, 13, 2048, dtype=torch.float16),
        "language_mask": torch.ones(n, 13, dtype=torch.bool),
        "proprio": torch.zeros(n, SEQUENCE_LENGTH, 4),
        "previous_action": torch.from_numpy(previous.astype(np.float32)),
        "actions": torch.from_numpy(actions.astype(np.float32)),
        "pair_id": torch.zeros(n, dtype=torch.long),
        "instruction_id": torch.zeros(n, dtype=torch.long),
        "episode_id": torch.arange(n, dtype=torch.long),
        "frames": torch.zeros(n, SEQUENCE_LENGTH, VISION_WINDOW, frame_size, frame_size, 3,
                              dtype=torch.uint8),
        "frame_rows": torch.zeros(n, SEQUENCE_LENGTH, VISION_WINDOW, dtype=torch.long),
        "perturb_type": ["object"] * n,
        "perturb_magnitude": torch.from_numpy(magnitudes.astype(np.float32)),
        "aligned_v5_row": torch.full((n,), -1, dtype=torch.long),
        "source_episode": torch.zeros(n, dtype=torch.long),
        "decision_frame": torch.zeros(n, dtype=torch.long),
        "normalization": {
            "action_q01": torch.full((4,), -1.0),
            "action_q99": torch.full((4,), 1.0),
            "state_q01": torch.from_numpy(SQ01.astype(np.float32)),
            "state_q99": torch.from_numpy(SQ99.astype(np.float32)),
        },
        "metadata": {
            "contract": "perturbation_recovery_mt50",
            "tasks": ["Pick up a nut and place it onto a peg"],
            "fps": 80,
            "control_stride": CONTROL_STRIDE,
            "action_horizon": ACTION_HORIZON,
            "previous_action_contract": "v5_prevfix_20260807",
            "action_contract": "executed-clip-v5",
            "sampling": {"mode": "near-contact-perturbation"},
            "perturbation": {"types": list(PERTURB_KIND_ORDER)},
            "alignment": {"contract": "v5-row-key"},
            "vision": {"mode": "skeleton-zero", "frames_stored": True, "frame_size": frame_size},
        },
    }


class ValidatePayloadTests(unittest.TestCase):
    def test_valid_payload_passes(self):
        self.assertEqual(validate_payload(synthetic_payload()), [])

    def test_no_store_frames_passes(self):
        payload = synthetic_payload()
        payload["frames"] = torch.zeros(0, dtype=torch.uint8)
        payload["metadata"]["vision"]["frames_stored"] = False
        self.assertEqual(validate_payload(payload), [])

    def test_missing_key_caught(self):
        payload = synthetic_payload()
        del payload["perturb_type"]
        self.assertIn("missing keys", validate_payload(payload)[0])

    def test_prev_contract_violation_caught(self):
        payload = synthetic_payload()
        payload["previous_action"] = payload["previous_action"].clone()
        payload["previous_action"][1, 1] += 0.5
        self.assertIn("previous_action contract", validate_payload(payload)[0])

    def test_magnitude_out_of_range_caught(self):
        payload = synthetic_payload(mags=[0.001, 0.005, 0.009])
        self.assertIn("perturb_magnitude out of range", validate_payload(payload)[0])

    def test_bad_shape_caught(self):
        payload = synthetic_payload()
        payload["actions"] = torch.zeros(3, 4, 8)  # 缺动作维
        self.assertTrue(validate_payload(payload))

    def test_unknown_perturb_type_caught(self):
        payload = synthetic_payload()
        payload["perturb_type"] = ["teleport"] * 3
        self.assertIn("unknown kinds", validate_payload(payload)[0])

    def test_missing_metadata_field_caught(self):
        payload = synthetic_payload()
        del payload["metadata"]["sampling"]
        self.assertIn("metadata.sampling missing", validate_payload(payload)[0])


if __name__ == "__main__":
    unittest.main()
