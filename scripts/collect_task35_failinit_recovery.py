#!/usr/bin/env python
"""P0: collect approach/grasp recovery on the 15k fail inits.

This does not overwrite the elected 1807-window payload. It writes a new
recovery longtraj file. Training/eval on the new payload is a later C−A step.

Default seed set is handover P0: chronic 7 ∪ never-approach 6 (10 unique).
Near-insert 35036 is P1-only and is excluded unless --all-fail13.

CPU / EGL only. Do not start this while a trainer owns the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.collect_long_trajectories import (  # noqa: E402
    TASK35_EVAL50_SEEDS,
    get_policy,
    make_env,
    main as collect_main,
    reset_eval_init,
)

FAIL13_PATH = ROOT / "artifacts" / "task35_15k_fail13.json"
DEFAULT_COLLECTOR_SEED = 360
DEFAULT_OUTPUT = (
    ROOT / "data" / "metaworld_longtraj_peg-insert-side-v3_recovery_failinit_v2_seed360.pt"
)
# Scripted SawyerPegInsertionSideV3Policy grasps 35027 (grasp_ok=1) but never
# inserts before MetaWorld max_path_length=500 (min_d≈0.178). The longtraj
# collector drops unsuccessful episodes, so this init cannot enter a recovery
# file until a longer-horizon or different expert exists.
SCRIPTED_UNSOLVED_AT_H500 = (35027,)


def load_fail13(path: Path = FAIL13_PATH) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("contract") != "task35_15k_fail13_v1":
        raise ValueError(f"unexpected fail13 contract in {path}")
    return payload


def p0_seeds(payload: dict, *, all_fail13: bool) -> list[int]:
    if all_fail13:
        return [int(seed) for seed in payload["fail_seeds"]]
    chronic = {int(seed) for seed in payload["chronic_fails_all_four_ckpts"]}
    never = {int(seed) for seed in payload["buckets"]["never-approach"]}
    return sorted(chronic | never)


def probe_scripted(seeds: list[int], max_steps: int = 500) -> list[dict]:
    env = make_env("peg-insert-side-v3")
    policy = get_policy("peg-insert-side-v3")
    rows = []
    try:
        for seed in seeds:
            obs, _ = reset_eval_init(env, seed)
            first_success = None
            min_d = None
            max_grasp = 0.0
            for step in range(max_steps):
                action = policy.get_action(obs)
                obs, _reward, term, trunc, info = env.step(action)
                dist = info.get("obj_to_target")
                if dist is not None:
                    dist = float(dist)
                    min_d = dist if min_d is None else min(min_d, dist)
                max_grasp = max(max_grasp, float(info.get("grasp_reward", 0.0)))
                if first_success is None and info.get("success"):
                    first_success = step
                    break
                if term or trunc:
                    break
            row = {
                "episode_seed": int(seed),
                "scripted_success": first_success is not None,
                "first_success": first_success,
                "min_obj_to_target": min_d,
                "grasp_reward": max_grasp,
            }
            rows.append(row)
            print(
                f"  seed={seed} success={row['scripted_success']} "
                f"first_success={first_success} min_d={min_d} grasp_r={max_grasp:.3f}"
            )
    finally:
        env.close()
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail13", type=Path, default=FAIL13_PATH)
    parser.add_argument("--all-fail13", action="store_true")
    parser.add_argument("--variants-per-seed", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_COLLECTOR_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Only test that the scripted expert can solve the pinned inits",
    )
    parser.add_argument(
        "--perturb-kinds",
        nargs="+",
        default=["eef_lateral", "eef_height"],
        help="Default approach kinds. Grasp kinds can be passed explicitly.",
    )
    parser.add_argument(
        "--skip-seeds",
        type=int,
        nargs="*",
        default=list(SCRIPTED_UNSOLVED_AT_H500),
        help="Drop pinned inits the scripted expert cannot finish in 500 steps",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = load_fail13(args.fail13)
    seeds = p0_seeds(payload, all_fail13=args.all_fail13)
    skipped = [seed for seed in seeds if seed in set(args.skip_seeds)]
    seeds = [seed for seed in seeds if seed not in set(args.skip_seeds)]
    unknown = [seed for seed in seeds if seed not in TASK35_EVAL50_SEEDS]
    if unknown:
        raise SystemExit(f"P0 seeds are not eval50 inits: {unknown}")
    if not seeds:
        raise SystemExit("no P0 seeds left after --skip-seeds")
    print(
        f"p0_seeds={seeds} n={len(seeds)} skipped={skipped} "
        f"variants={args.variants_per_seed}"
    )
    if args.probe:
        rows = probe_scripted(seeds)
        failed = [row["episode_seed"] for row in rows if not row["scripted_success"]]
        if failed:
            raise SystemExit(f"scripted expert failed on {failed}")
        print(f"[ok] scripted expert solved {len(rows)}/{len(rows)} pinned inits")
        return
    collect_argv = [
        "--task", "peg-insert-side-v3",
        "--episode-seeds", *[str(seed) for seed in seeds],
        "--variants-per-seed", str(args.variants_per_seed),
        "--force-perturb",
        "--allow-eval-seeds",
        "--perturb-kinds", *args.perturb_kinds,
        "--seed", str(args.seed),
        "--output", str(args.output),
    ]
    if args.overwrite:
        collect_argv.append("--overwrite")
    collect_main(collect_argv)


if __name__ == "__main__":
    main()
