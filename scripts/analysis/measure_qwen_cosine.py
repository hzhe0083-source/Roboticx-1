"""Qwen instruction-embedding cosine verdict (reproducible, 2026-08-06).

Reproduces the §11.9 measurement: load a checkpoint's Qwen state (+LoRA when
present), encode the LIBERO task instructions with it, and report the mean
pairwise cosine of the final-layer hidden states (last token).  Also reports a
centered (per-token mean-subtracted) cosine as a robustness check against the
shared-mean-component caveat of §10.5.

Baselines printed for reference:
  - original Qwen (no adapter):  ~0.7647 on the 12-task 3-scene set
  - random embeddings:           ~0.08
  - B40k LoRA-collapsed:         ~0.9992

Usage:
  python scripts/analysis/measure_qwen_cosine.py --checkpoint checkpoints/libero_e2e_C2_40k.pt \
      --data data/libero_video_v2.pt
  python scripts/analysis/measure_qwen_cosine.py --none  # original Qwen only
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse

import torch

from prepare_pnpw_features import QwenTextBackbone


def pairwise_cosine(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[float, float]:
    """Mean pairwise cosine of per-instruction vectors.

    hidden: [N, seq, D] float.  Vector = last *valid* token hidden state,
    gathered by the attention mask (right-padding safe; 2026-08-06 Codex P0).
    """
    if mask is not None:
        lengths = mask.sum(-1).clamp_min(1)  # [N]
        vec = hidden[torch.arange(hidden.shape[0]), lengths - 1, :]  # [N, D]
    else:
        vec = hidden[:, -1, :]  # [N, D]
    vn = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = vn @ vn.T
    n = sim.shape[0]
    off = (sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))

    c = vec - vec.mean(dim=0, keepdim=True)  # centered per-dimension
    cn = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    csim = cn @ cn.T
    coff = (csim.sum() - torch.diagonal(csim).sum()) / (n * (n - 1))
    return float(off), float(coff)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen embedding cosine verdict")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="checkpoint with qwen_state_dict / lora (omit for original Qwen)")
    p.add_argument("--data", type=Path, default=Path("data/libero_video_v2.pt"),
                   help="payload whose metadata['tasks'] are encoded")
    p.add_argument("--device", default="cuda")
    p.add_argument("--none", action="store_true",
                   help="only original Qwen + random baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    tasks = list(payload["metadata"]["tasks"])
    print(f"instructions: {len(tasks)}")
    for t in tasks:
        print(f"  - {t}")

    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )

    with torch.inference_mode():
        hidden0, mask0 = text_backbone.encode(tasks)
    off0, coff0 = pairwise_cosine(hidden0.float(), mask0)
    print(f"\n[original Qwen]      pairwise_cosine={off0:.4f}  centered={coff0:.4f}")

    rng = torch.Generator(device=hidden0.device).manual_seed(0)
    rand = torch.randn_like(hidden0.float(), generator=rng)
    offr, coffr = pairwise_cosine(rand, mask0)
    print(f"[random baseline]    pairwise_cosine={offr:.4f}  centered={coffr:.4f}")

    if args.none:
        return

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    name = args.checkpoint.stem
    if ck.get("qwen_state_dict"):
        qwen_state = {
            k.removeprefix("text_model."): v
            for k, v in ck["qwen_state_dict"].items()
        }
        missing, unexpected = text_backbone.text_model.load_state_dict(
            qwen_state, strict=False
        )
        print(f"[{name}] qwen loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if ck.get("lora"):
        from va_compound.backbones import apply_lora

        rank = int(ck.get("training_contract", {}).get("lora_rank", 32))
        apply_lora(text_backbone.text_model, rank=rank)
        own = dict(text_backbone.text_model.named_parameters())
        n_loaded = 0
        for k, v in ck["lora"].items():
            clean = k.removeprefix("text_model.")
            if clean in own:
                own[clean].data.copy_(v)
                n_loaded += 1
        print(f"[{name}] lora rank={rank} weights loaded={n_loaded}")
    text_backbone.text_model.eval()

    with torch.inference_mode():
        hidden1, mask1 = text_backbone.encode(tasks)
    off1, coff1 = pairwise_cosine(hidden1.float(), mask1)
    print(f"[{name}] pairwise_cosine={off1:.4f}  centered={coff1:.4f}")
    print(
        f"\nVERDICT {name}: {'COLLAPSED' if off1 > 0.95 else 'PRESERVED' if off1 < 0.85 else 'AMBIGUOUS'} "
        f"(original {off0:.4f}, random {offr:.4f})"
    )


if __name__ == "__main__":
    main()
