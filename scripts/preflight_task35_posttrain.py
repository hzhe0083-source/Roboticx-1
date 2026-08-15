#!/usr/bin/env python
"""CPU-only preflight before the 50-seed closed-loop suite.

Checks artifacts, disk, task35→peg-insert-side-v3 mapping, and the FM
checkpoint contract. This never starts eval or touches the trainer GPU.
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
from scripts.validate_task35_fm_checkpoint import validate_task35_fm_checkpoint_path

FEATURES = ROOT / "data" / "metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt"
CACHE = ROOT / "data" / "dino35_h6_clean60_recovery30_cache_v1"
ROI = ROOT / "checkpoints" / "dino_metric_roi_task35_v2_native480_seed777_1k.pt"
DINO = Path(
    "/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/"
    "snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors"
)
MIN_FREE_GIB = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-module-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        str(path)
        for path in (
            args.checkpoint,
            FEATURES,
            ROI,
            DINO,
            CACHE / "meta.json",
            CACHE / "block11.npy",
            CACHE / "block23.npy",
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("preflight missing: " + ", ".join(missing))
    free_gib = shutil.disk_usage(ROOT).free / (1024**3)
    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(f"only {free_gib:.1f} GiB free; need >= {MIN_FREE_GIB}")

    import torch

    features = torch.load(FEATURES, map_location="cpu", weights_only=True)
    selected = select_eval_tasks(features["metadata"]["tasks"], "35", 49)
    env_name = require_task35_peg_insert_side(
        selected, load_metaworld_description_to_env()
    )
    report = validate_task35_fm_checkpoint_path(
        args.checkpoint,
        expected_step=args.expected_step,
        load_modules=not args.skip_module_load,
    )
    payload = {
        "contract": "task35_posttrain_preflight_v1",
        "ok": True,
        "env_name": env_name,
        "task_text": selected[0][1],
        "free_gib": free_gib,
        "checkpoint": report,
    }
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
