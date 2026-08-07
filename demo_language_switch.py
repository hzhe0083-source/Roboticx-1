"""Language-switch demo for the VA Compound paper.

Core selling point: with a FIXED first observation (same vision, same
proprio, same previous action), switching only the language instruction must
switch the emitted action chunk to the correct one for that instruction.

Two quantitative results:
  1. instruction readability: from the emitted chunk, which instruction was
     the model conditioned on? (nearest-expert accuracy)
  2. branching: how far apart are the emitted chunks for different
     instructions, relative to the expert-chunk spread?
Plus one qualitative figure: action-chunk curves for two instructions
under the shared initial observation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**checkpoint["config"])
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(checkpoint["model"])

    vis = payload["vision_tokens"]          # [N, T, 64, 768]
    prop = payload["proprio"]               # [N, T, 9]
    prev = payload["previous_action"]       # [N, T, 7]
    act = payload["actions"]                # [N, T, 8, 7]
    lang = payload["language_hidden"]       # [N, Nl, 2048]
    lang_mask = payload["language_mask"]    # [N, Nl]
    instr = payload["instruction_id"]       # [N]
    tasks: list[str] = list(payload["metadata"]["tasks"])

    n_tasks = int(instr.max()) + 1
    print(f"tasks ({n_tasks}):")
    for i, t in enumerate(tasks):
        print(f"  [{i}] {t}")

    # Per-instruction language caches (one forward per instruction).
    caches = {}
    for k in range(n_tasks):
        idx = (instr == k).nonzero().flatten()[0].item()
        h = lang[idx : idx + 1].to(device)
        m = lang_mask[idx : idx + 1].to(device)
        caches[k] = model.build_language_cache(h, m)

    def emit(obs_idx: int, instr_k: int) -> np.ndarray:
        """Action chunk [8, 7] for obs i under instruction k."""
        with torch.inference_mode():
            cond = model.encode_condition(
                vis[obs_idx, 0:1].to(device),
                prop[obs_idx, 0:1].to(device),
                prev[obs_idx, 0:1].to(device),
                language_cache=caches[instr_k],
            )
            return model.sample_actions(cond, steps=8)[0].cpu().numpy()

    # Expert chunk for instruction k at the SAME initial state as obs i.
    # All samples in this contract share the initial observation, so any
    # sample of instruction k provides the expert chunk.
    expert = {
        k: act[(instr == k).nonzero().flatten()[0], 0].numpy() for k in range(n_tasks)
    }

    # ---- full evaluation over all samples ----
    n = len(vis)
    correct = np.zeros(n_tasks, dtype=int)
    total = np.zeros(n_tasks, dtype=int)
    branch_dists = []
    expert_spread = []
    for i in range(n):
        out = np.stack([emit(i, k) for k in range(n_tasks)])   # [K, 8, 7]
        for k in range(n_tasks):
            dists = [float(np.mean(np.abs(out[k] - expert[j]))) for j in range(n_tasks)]
            if int(np.argmin(dists)) == k:
                correct[k] += 1
            total[k] += 1
        # branching: mean pairwise distance of the K emitted chunks
        pair = 0.0
        for a in range(n_tasks):
            for b in range(a + 1, n_tasks):
                pair += float(np.mean(np.abs(out[a] - out[b])))
        branch_dists.append(pair / (n_tasks * (n_tasks - 1) / 2))
        exp_pair = 0.0
        for a in range(n_tasks):
            for b in range(a + 1, n_tasks):
                exp_pair += float(np.mean(np.abs(expert[a] - expert[b])))
        expert_spread.append(exp_pair / (n_tasks * (n_tasks - 1) / 2))

    acc = correct.sum() / total.sum()
    print(f"\ninstruction readability: {correct.sum()}/{total.sum()} = {acc*100:.1f}%")
    for k in range(n_tasks):
        print(f"  instr [{k}] {tasks[k][:48]}: {correct[k]}/{total[k]}")
    print(f"emitted-chunk spread: {np.mean(branch_dists):.4f}")
    print(f"expert-chunk spread:  {np.mean(expert_spread):.4f}")
    print(f"branching ratio:      {np.mean(branch_dists)/np.mean(expert_spread):.2f}")

    # ---- qualitative figure: one shared observation, two instructions ----
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        i0 = int(args.obs_idx)
        a = int(args.instr_a)
        b = int(args.instr_b)
        out_a = emit(i0, a)
        out_b = emit(i0, b)
        exp_a = expert[a]
        exp_b = expert[b]

        fig, axes = plt.subplots(7, 1, figsize=(9, 11), sharex=True)
        for d in range(7):
            ax = axes[d]
            t = np.arange(8)
            ax.plot(t, out_a[:, d], "-o", color="#C0392B", lw=2, ms=4, label=f"instr A")
            ax.plot(t, out_b[:, d], "-s", color="#2980B9", lw=2, ms=4, label=f"instr B")
            ax.plot(t, exp_a[:, d], "--", color="#E67E22", lw=1.4, label="expert A")
            ax.plot(t, exp_b[:, d], ":", color="#27AE60", lw=1.4, label="expert B")
            ax.set_ylabel(f"dim {d}")
            ax.legend(loc="upper right", fontsize=7)
        axes[0].set_title(
            "Same first observation, different instruction -> different action chunk\n"
            f"instr A: {tasks[a][:40]}\ninstr B: {tasks[b][:40]}"
        )
        axes[-1].set_xlabel("step within chunk")
        fig.tight_layout()
        out_png = Path(args.figure)
        fig.savefig(out_png, dpi=150)
        print(f"\nfigure saved: {out_png}")


def _parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--obs-idx", type=int, default=0)
    p.add_argument("--instr-a", type=int, default=0)
    p.add_argument("--instr-b", type=int, default=1)
    p.add_argument("--figure", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
