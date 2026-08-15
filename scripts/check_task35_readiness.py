#!/usr/bin/env python
"""CPU-only readiness check for the remaining FM 15k + 50-seed path.

This never starts eval and never claims closed-loop success.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metaworld import (
    load_metaworld_description_to_env,
    require_task35_peg_insert_side,
    select_eval_tasks,
)
from scripts.plan_task35_eval_suite import REQUIRED_MILESTONES, SKIP_CLOSED_LOOP_STEPS
from scripts.summarize_task35_fm_train import ARCHIVE_MILESTONES
from scripts.task35_proc import is_inspector, trainer_processes

FEATURES = ROOT / "data" / "metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt"
CACHE = ROOT / "data" / "dino35_h6_clean60_recovery30_cache_v1"
ROI = ROOT / "checkpoints" / "dino_metric_roi_task35_v2_native480_seed777_1k.pt"
STEM = ROOT / "checkpoints" / "task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1"
MIN_FREE_GIB = 20.0
WAITER_NEEDLES = (
    "archive_task35_fm_milestones.sh",
    "wait_validate_task35_fm_milestone.sh 6000",
    "wait_validate_task35_fm_milestone.sh 9000",
    "wait_validate_task35_fm_milestone.sh 12000",
    "wait_task35_fm_finished_eval.sh",
    "tail -n 0 -F",
)


def _cmdlines() -> list[str]:
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except OSError:
            continue
        if cmd and not is_inspector(cmd):
            rows.append(cmd)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "task35_readiness.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmds = _cmdlines()
    trainers = trainer_processes()
    checks = {
        "trainer exclusive": len(trainers) == 1 and "direct-head" not in trainers[0]["cmd"],
        "save-every 1000": "--save-every 1000" in (trainers[0]["cmd"] if trainers else ""),
        "steps 15000": "--steps 15000" in (trainers[0]["cmd"] if trainers else ""),
        "15000 is archive milestone": 15000 in ARCHIVE_MILESTONES,
        "acceptance set is 3k+": REQUIRED_MILESTONES == (3000, 6000, 9000, 12000, 15000),
        "1k/2k skipped": SKIP_CLOSED_LOOP_STEPS == (1000, 2000),
        "features exist": FEATURES.is_file(),
        "ROI exist": ROI.is_file(),
        "cache exist": (CACHE / "block11.npy").is_file() and (CACHE / "block23.npy").is_file(),
        "disk >= 20 GiB": shutil.disk_usage(ROOT).free / (1024**3) >= MIN_FREE_GIB,
        "no Direct trainer": not any(
            "train.py" in cmd and "direct-head" in cmd and "python" in cmd for cmd in cmds
        ),
        "no eval job": not any("eval_metaworld.py" in cmd for cmd in cmds),
    }
    for needle in WAITER_NEEDLES:
        checks[f"waiter {needle}"] = any(needle in cmd for cmd in cmds)
    import torch

    features = torch.load(FEATURES, map_location="cpu", weights_only=True)
    selected = select_eval_tasks(features["metadata"]["tasks"], "35", 49)
    env_name = require_task35_peg_insert_side(
        selected, load_metaworld_description_to_env()
    )
    checks["task35 maps to peg-insert-side-v3"] = env_name == "peg-insert-side-v3"
    archived = {
        step: (Path(f"{STEM}_step{step}.pt").is_file() and Path(f"{STEM}_step{step}.pt.sha256").is_file())
        for step in ARCHIVE_MILESTONES
    }
    payload = {
        "contract": "task35_readiness_v1",
        "ok": all(checks.values()),
        "env_name": env_name,
        "task_text": selected[0][1],
        "free_gib": shutil.disk_usage(ROOT).free / (1024**3),
        "trainer_pid": None if not trainers else trainers[0]["pid"],
        "archived": archived,
        "checks": checks,
        "note": "readiness only; closed-loop insertion remains planned",
    }
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
