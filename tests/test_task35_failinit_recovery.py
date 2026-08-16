from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts import collect_long_trajectories as collect
from scripts.collect_task35_failinit_recovery import load_fail13, p0_seeds
from tests.test_longtraj_data_contract import _AlwaysPerturbRng, _Env, _Policy


FAIL13 = Path(__file__).resolve().parents[1] / "artifacts" / "task35_15k_fail13.json"


class FailinitSeedPolicyTest(unittest.TestCase):
    def test_local_eval50_tuple_matches_eval_module(self):
        self.assertEqual(collect.TASK35_EVAL50_SEEDS, tuple(range(35000, 35050)))

    def test_p0_union_is_chronic_or_never_approach(self):
        payload = load_fail13(FAIL13)
        seeds = p0_seeds(payload, all_fail13=False)
        self.assertEqual(
            seeds,
            [35002, 35004, 35007, 35009, 35014, 35021, 35027, 35028, 35033, 35039],
        )
        self.assertNotIn(35036, seeds)
        self.assertIn(35027, seeds)
        self.assertEqual(p0_seeds(payload, all_fail13=True), payload["fail_seeds"])

    def test_forbid_eval_seeds_by_default(self):
        with self.assertRaisesRegex(ValueError, "overlap eval50"):
            collect.check_eval_seed_policy([35002, 99], allow_eval_seeds=False)
        collect.check_eval_seed_policy([35002], allow_eval_seeds=True)
        collect.check_eval_seed_policy([99], allow_eval_seeds=False)

    def test_planned_seeds_repeat_variants(self):
        self.assertEqual(
            collect.planned_episode_seeds([35002, 35004], 2),
            [35002, 35002, 35004, 35004],
        )


class FailinitCollectorContractTest(unittest.TestCase):
    def test_pinned_seed_is_not_drawn_from_rng(self):
        ep = collect._collect_episode_inner(
            _Env(success_after=5),
            _Policy(),
            "door-lock-v3",
            np.random.default_rng(7),
            perturb=False,
            episode_seed=35002,
        )
        self.assertIsNotNone(ep)
        assert ep is not None
        self.assertEqual(ep["episode_seed"], 35002)
        self.assertFalse(ep["perturbed"])

    def test_force_perturb_ignores_five_percent_gate(self):
        class NeverRandom(_AlwaysPerturbRng):
            def random(self):
                return 0.99

        ep = collect._collect_episode_inner(
            _Env(success_after=25),
            _Policy(),
            "door-lock-v3",
            NeverRandom(),
            perturb=True,
            episode_seed=35007,
            force_perturb=True,
            perturb_kinds=("eef_height",),
        )
        self.assertIsNotNone(ep)
        assert ep is not None
        self.assertEqual(ep["episode_seed"], 35007)
        self.assertTrue(ep["perturbed"])
        self.assertEqual(ep["n_perturb_events"], 1)
        self.assertEqual(ep["perturb_event"]["kind"], "eef_height")
        self.assertTrue(ep["recovery_mask"].any())

    def test_without_force_high_random_stays_nominal(self):
        class NeverRandom(_AlwaysPerturbRng):
            def random(self):
                return 0.99

        ep = collect._collect_episode_inner(
            _Env(success_after=5),
            _Policy(),
            "door-lock-v3",
            NeverRandom(),
            perturb=True,
            episode_seed=35007,
            force_perturb=False,
        )
        self.assertIsNotNone(ep)
        assert ep is not None
        self.assertFalse(ep["perturbed"])
        self.assertEqual(ep["n_perturb_events"], 0)

    def test_parser_requires_allow_flag_for_eval_inits(self):
        parser = collect.build_parser()
        args = parser.parse_args(
            [
                "--task", "peg-insert-side-v3",
                "--episode-seeds", "35002", "35004",
                "--force-perturb",
                "--allow-eval-seeds",
            ]
        )
        self.assertEqual(args.episode_seeds, [35002, 35004])
        self.assertTrue(args.force_perturb)
        self.assertTrue(args.allow_eval_seeds)


if __name__ == "__main__":
    unittest.main()
