from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
import torch

from va_compound.backbones import (
    VJEPA21_CHECKPOINT_BYTES,
    VJEPA21_CHECKPOINT_NAME,
    VJEPA21_CHECKPOINT_URL,
    VJEPA21_ENTRYPOINT,
    VJEPA21_REPO,
    VJEPA21_REPO_REF,
)


QWEN_ID = "Qwen/Qwen3.5-2B"
QWEN_PATTERNS = ["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model"]
QWEN_REQUIRED = [
    "config.json",
    "model.safetensors-00001-of-00001.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def prepare_qwen(check_only: bool) -> bool:
    if not check_only:
        path = snapshot_download(QWEN_ID, allow_patterns=QWEN_PATTERNS, max_workers=4)
        print(f"qwen: {path}")
        return True

    resolved = []
    missing = []
    for filename in QWEN_REQUIRED:
        try:
            resolved.append(hf_hub_download(QWEN_ID, filename, local_files_only=True))
        except LocalEntryNotFoundError:
            missing.append(filename)
    if missing:
        print(f"qwen: missing {', '.join(missing)}")
        return False
    print(f"qwen: {Path(resolved[0]).parent}")
    return True


def prepare_vjepa(check_only: bool) -> bool:
    hub_dir = Path(torch.hub.get_dir())
    repo_dir = hub_dir / f"facebookresearch_vjepa2_{VJEPA21_REPO_REF}"
    checkpoint_path = hub_dir / "checkpoints" / VJEPA21_CHECKPOINT_NAME
    checkpoint_ready = (
        checkpoint_path.is_file()
        and checkpoint_path.stat().st_size == VJEPA21_CHECKPOINT_BYTES
    )

    if check_only:
        missing = []
        if not repo_dir.is_dir():
            missing.append("pinned source")
        if not checkpoint_ready:
            missing.append("checkpoint")
        if missing:
            print(f"vjepa: missing {', '.join(missing)}")
            return False
        print(f"vjepa: {checkpoint_path}")
        return True

    model, predictor = torch.hub.load(
        VJEPA21_REPO,
        VJEPA21_ENTRYPOINT,
        pretrained=False,
        trust_repo=True,
        skip_validation=True,
    )
    del model, predictor
    if not checkpoint_ready:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        torch.hub.download_url_to_file(
            VJEPA21_CHECKPOINT_URL,
            checkpoint_path,
            progress=True,
        )
    print(f"vjepa: {checkpoint_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare only the two backbone checkpoints used by VA")
    parser.add_argument("--model", choices=("qwen", "vjepa", "all"), default="all")
    parser.add_argument("--check", action="store_true", help="verify local cache without downloading")
    args = parser.parse_args()

    ready = []
    if args.model in ("qwen", "all"):
        ready.append(prepare_qwen(args.check))
    if args.model in ("vjepa", "all"):
        ready.append(prepare_vjepa(args.check))
    if not all(ready):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
