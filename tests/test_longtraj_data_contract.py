from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from scripts import build_longtraj_features as build
from scripts import collect_long_trajectories as collect


class _Policy:
    def get_action(self, obs):
        return np.full(4, 0.25, dtype=np.float32)


class _Data:
    def __init__(self):
        self.mocap_pos = np.zeros((1, 3), dtype=np.float64)


class _Env:
    def __init__(self, success_after: int):
        self.success_after = success_after
        self.executed = 0
        self.data = _Data()
        self._target_pos = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)

    def _get_obs(self):
        obs = np.zeros(10, dtype=np.float32)
        obs[:4] = [self.executed, 0.1, 0.2, 1.0]
        obs[4:7] = [0.2, 0.3, self.executed / 1000.0]
        return obs

    def reset(self, seed=None):
        self.executed = 0
        return self._get_obs(), {}

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action):
        self.executed += 1
        success = self.executed > self.success_after
        return self._get_obs(), 0.0, False, False, {"success": success}


class _AlwaysPerturbRng:
    def integers(self, low, high=None):
        return 80 if low == collect.HOLD_FRAMES[0] else 1

    def random(self):
        return 0.0

    def uniform(self, low, high):
        return (low + high) / 2

    def choice(self, values):
        return "eef_height"


class LongTrajCollectorContractTest(unittest.TestCase):
    def test_no_perturb_has_explicit_timeline_and_door_metric(self):
        ep = collect._collect_episode_inner(
            _Env(success_after=5),
            _Policy(),
            "door-lock-v3",
            np.random.default_rng(7),
            perturb=False,
        )
        self.assertIsNotNone(ep)
        assert ep is not None
        n = len(ep["actions"])
        self.assertEqual(ep["first_success"], 5)
        self.assertEqual(ep["success_frame"], 5)
        self.assertIsNone(ep["perturb_start"])
        self.assertIsNone(ep["perturb_end"])
        self.assertIsNone(ep["perturb_kind"])
        self.assertIsNone(ep["perturb_magnitude"])
        self.assertFalse(ep["perturbed"])
        self.assertEqual(ep["metric_state"].shape, (n, 6))
        self.assertTrue(ep["metric_state_valid"].all())
        self.assertTrue(ep["action_executed"].all())
        self.assertTrue(ep["frame_valid"].all())
        self.assertTrue(ep["action_supervision_valid"][:6].all())
        self.assertFalse(ep["action_supervision_valid"][6:].any())

    def test_perturb_indices_include_settle_in_stored_timeline(self):
        ep = collect._collect_episode_inner(
            # First success lands inside the 12 stored settle actions.
            _Env(success_after=25),
            _Policy(),
            "door-lock-v3",
            _AlwaysPerturbRng(),
            perturb=True,
        )
        self.assertIsNotNone(ep)
        assert ep is not None
        self.assertEqual(ep["perturb_start"], 22)
        self.assertEqual(ep["perturb_end"], 34)
        self.assertEqual(ep["first_success"], 25)
        self.assertEqual(int(ep["settle_mask"].sum()), collect.PERTURB_SETTLE_STEPS)
        self.assertFalse(ep["action_supervision_valid"][22:34].any())
        self.assertTrue(ep["recovery_mask"][22:26].all())
        self.assertFalse(ep["recovery_mask"][26:].any())
        self.assertFalse(ep["recovery_mask"][:22].any())
        self.assertEqual(ep["perturb_event"]["kind"], "eef_height")
        self.assertAlmostEqual(float(ep["perturb_event"]["delta"][2]), 0.005)


class LongTrajBuilderContractTest(unittest.TestCase):
    @staticmethod
    def _normalization():
        return {
            "action_q01": torch.full((4,), -1.0),
            "action_q99": torch.full((4,), 1.0),
            "state_q01": torch.full((4,), -1.0),
            "state_q99": torch.full((4,), 1.0),
        }

    def test_targeted_build_emits_trainable_masks_language_and_source_ref(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_path = root / "ref.pt"
            source = root / "metaworld_longtraj_door-lock-v3_clean_v2_seed0.pt"
            output = root / "door_clean_windows.pt"
            task_text = build.ENV_TO_TASK["door-lock-v3"]
            torch.save({
                "normalization": self._normalization(),
                "metadata": {"tasks": [task_text]},
                "instruction_id": torch.tensor([0]),
                "language_hidden": torch.arange(6, dtype=torch.float16).reshape(1, 2, 3),
                "language_mask": torch.tensor([[True, True]]),
            }, ref_path)

            n = 50
            jpeg = collect.compress_frames(
                np.zeros((1, 2, 2, 3), dtype=np.uint8)
            )[0]
            settle = np.zeros(n, dtype=bool)
            settle[10:13] = True
            recovery = np.zeros(n, dtype=bool)
            recovery[10:29] = True
            metric = np.arange(n * 6, dtype=np.float32).reshape(n, 6)
            ep = {
                "frames": [jpeg] * n,
                "actions": np.zeros((n, 4), dtype=np.float32),
                "states": np.zeros((n, 4), dtype=np.float32),
                "first_success": 28,
                "frame_valid": np.ones(n, dtype=bool),
                "action_executed": np.ones(n, dtype=bool),
                # Builder must still enforce settle/post-success exclusions.
                "action_supervision_valid": np.ones(n, dtype=bool),
                "settle_mask": settle,
                "recovery_mask": recovery,
                "metric_state": metric,
                "metric_state_valid": np.ones(n, dtype=bool),
                "perturbed": True,
                "perturb_start": 10,
                "perturb_end": 13,
            }
            torch.save({
                "task": "door-lock-v3",
                "episodes": [ep],
                "normalization": self._normalization(),
                "metadata": {"contract_version": 2},
            }, source)

            result = build.phase1(
                8,
                task="door-lock-v3",
                input_paths=[source],
                output_path=output,
                ref_path=ref_path,
                legacy_policy="error",
            )
            self.assertEqual(result, output)
            payload = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(tuple(payload["action_valid_mask"].shape), (5, 4, 8))
            # Row 0, decision t=1 is action 6. Its future settle and recovery
            # branch are not valid before the random perturb is observable.
            self.assertFalse(bool(payload["action_valid_mask"][0, 1, 4]))  # action 10 settle
            self.assertFalse(bool(payload["action_valid_mask"][0, 1, 7]))  # action 13 recovery
            # Decision action 12 is post-perturb, so action 13 recovery is valid.
            self.assertTrue(bool(payload["action_valid_mask"][0, 2, 1]))
            # row 2 starts at 12; its final decision begins after first_success.
            self.assertFalse(bool(payload["action_valid_mask"][2, 3].any()))
            self.assertTrue(bool(payload["recovery_mask"].any()))
            self.assertEqual(tuple(payload["door_metric_state"].shape), (5, 4, 6))
            self.assertEqual(tuple(payload["language_hidden"].shape), (5, 2, 3))
            self.assertEqual(
                payload["frame_refs"][0][0], "door-lock-v3_clean_v2_seed0"
            )
            self.assertEqual(payload["metadata"]["contract_version"], 2)

            # Existing loader resolves the source key to the clean filename;
            # no canonical-file overwrite or loader patch is required.
            from va_compound.longtraj_frames import LongTrajFramesDataset
            dataset = LongTrajFramesDataset(output, longtraj_dir=root)
            item = dataset[0]
            self.assertEqual(item["frames"].shape, (4, 4, 2, 2, 3))

            with self.assertRaises(FileExistsError):
                build.phase1(
                    4,
                    task="door-lock-v3",
                    input_paths=[source],
                    output_path=output,
                    ref_path=ref_path,
                    legacy_policy="error",
                )

    def test_legacy_contract_warns_or_fails_explicitly(self):
        n = 30
        ep = {
            "frames": [b"x"] * n,
            "actions": np.zeros((n, 4), dtype=np.float32),
            "states": np.zeros((n, 4), dtype=np.float32),
            "success_frame": 20,
            "perturbed": True,
        }
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            semantics = build.resolve_episode_semantics(ep, "legacy", "warn")
        self.assertGreaterEqual(len(seen), 2)
        self.assertFalse(semantics["valid"][21:].any())
        with self.assertRaisesRegex(ValueError, "legacy episode"):
            build.resolve_episode_semantics(ep, "legacy", "error")

    def test_perturb_interval_must_have_an_end(self):
        n = 8
        ep = {
            "frames": [b"x"] * n,
            "actions": np.zeros((n, 4), dtype=np.float32),
            "states": np.zeros((n, 4), dtype=np.float32),
            "first_success": 7,
            "action_executed": np.ones(n, dtype=bool),
            "action_supervision_valid": np.ones(n, dtype=bool),
            "perturb_start": 3,
            "perturb_end": None,
        }
        with self.assertRaisesRegex(ValueError, "perturb_end is missing"):
            build.resolve_episode_semantics(ep, "broken", "error")


if __name__ == "__main__":
    unittest.main()
