#!/usr/bin/env python3
"""Re-encode a task language reference with a truncated Qwen3.5 text tower."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from va_compound.backbones import QwenTextBackbone

QWEN_FUSION_LAYERS = list(range(10, 15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--keep-layers", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    if args.keep_layers != 15:
        raise ValueError("this branch requires --keep-layers 15 (Qwen layers 0-14)")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    metadata = dict(source.get("metadata") or {})
    tasks = list(metadata.get("tasks") or source.get("tasks") or [])
    if not tasks:
        raise ValueError("source language reference has no task descriptions")
    normalization = source.get("normalization")
    if normalization is not None and not isinstance(normalization, dict):
        raise ValueError("source normalization must be a dictionary when present")

    backbone = QwenTextBackbone.from_pretrained(
        model_id=str(args.qwen_model.expanduser().resolve(strict=True)),
        device=args.device,
        dtype=args.dtype,
        max_length=64,
        keep_layers=args.keep_layers,
        local_files_only=True,
    )
    hierarchy, mask = backbone.encode(tasks, output_layers=QWEN_FUSION_LAYERS)
    hidden = backbone.mean_output_layers(hierarchy, QWEN_FUSION_LAYERS)
    payload = {
        "language_hidden": hidden.cpu().to(torch.float16),
        "language_mask": mask.cpu().bool(),
        "instruction_id": torch.arange(len(tasks), dtype=torch.long),
        "metadata": {
            **metadata,
            "tasks": tasks,
            "language_contract": "qwen35_truncated_multilevel_text_v1",
            "qwen_model_path": str(args.qwen_model.expanduser().resolve(strict=True)),
            "qwen_original_layers": backbone.original_num_layers,
            "qwen_keep_layers": backbone.keep_layers,
            "qwen_output_layer": backbone.keep_layers - 1,
            "qwen_fusion_layers": QWEN_FUSION_LAYERS,
            "qwen_layer_reduce": "mean_then_final_norm",
        },
    }
    if normalization is not None:
        payload["normalization"] = dict(normalization)
    if "tasks" in source:
        payload["tasks"] = tasks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(
        f"saved {args.output}: tasks={len(tasks)} "
        f"Qwen layers=0-{backbone.keep_layers - 1}, "
        f"mean(10-14) hidden={tuple(hidden.shape)}"
    )


if __name__ == "__main__":
    main()
