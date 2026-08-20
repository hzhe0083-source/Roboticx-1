"""L_m language-grounding verdict: same-scene dual-objective counterfactual closed loop.

For each scene, pick two executable objectives (g1, g2) sharing the same visual
scene layout, with instructions l1, l2.  On every matched initial state we roll
out four conditions (same physics, same init, only the language cache differs):

    D:  env(g1)+l1 -> r1,  env(g2)+l2 -> r2      (matched instruction)
    O:  env(g1)+l2 -> r3,  env(g2)+l1 -> r4      (swapped instruction)

Block contribution:  b = 1/2 * [(r1-r3) + (r2-r4)]
L_m = mean over matched blocks = D - O, with block bootstrap 95% CI.

Verdicts:
  L_m >> 0  -> obedience (policy follows the language swap)
  L_m ~= 0, D & O high -> language-selectivity missing (visual shortcut)
  L_m ~= 0, D & O low  -> OOD fragility (fails under both instructions)
  L_m < 0  -> inverse effect

Usage:
  python scripts/eval/eval_libero_language.py --checkpoint checkpoints/libero_uni_a_va8_20k.pt \
      --data data/libero_video.pt --device cuda --max-pairs 2 --trials-per-task 5
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch

from prepare_pnpw_features import QwenTextBackbone
from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy
from eval_libero_closedloop import rollout_task, preprocess  # reuse the core loop

# Data task order in data/libero_video.pt (0-based), grouped by scene.
# 0-3 LIVING_ROOM_SCENE2, 4-7 KITCHEN_SCENE2, 8-11 STUDY_SCENE1.
# The benchmark is pinned so both objectives come from the SAME scene layout.
PAIRS = [
    ("study_back_front", (8, 9), "libero_90"),
    ("study_left_right", (10, 11), "libero_90"),
    ("kitchen_back_front", (5, 6), "libero_90"),
    ("living_soup_butter", (0, 1), "libero_object"),
    ("living_milk_juice", (2, 3), "libero_object"),
]


def build_env_mapping(benchmark_names=("libero_90", "libero_spatial", "libero_object", "libero_goal")):
    from libero.libero import benchmark as bm
    import libero.libero.utils as lu

    lu.set_libero_path(custom_location=os.path.dirname(os.path.dirname(lu.__file__)))
    from libero.libero.envs import OffScreenRenderEnv

    by_bench = {}
    for name in benchmark_names:
        try:
            bench = bm.get_benchmark(name)()
        except KeyError:
            continue
        mapping = {}
        for task_id in range(bench.get_num_tasks()):
            task = bench.get_task(task_id)
            language = task.language.strip()
            mapping.setdefault(language, (bench, task, task_id))
        by_bench[name] = mapping
    return by_bench, OffScreenRenderEnv


# Exact instructions in the dataset that differ in wording from the local
# LIBERO benchmark tasks; mapped to the benchmark's phrasing.
ALIASES = {
    "put the black bowl in the middle on the plate": "put the middle black bowl on the plate",
    "pick up the alphabet soup and put it in the basket": "pick up the alphabet soup and place it in the basket",
    "pick up the butter and put it in the basket": "pick up the butter and place it in the basket",
    "pick up the milk and put it in the basket": "pick up the milk and place it in the basket",
    "pick up the orange juice and put it in the basket": "pick up the orange juice and place it in the basket",
}


def match_task(lang: str, mapping: dict):
    """Strip the scene prefix, exact match first, then a unique
    word-boundary substring match; None when ambiguous."""
    key = lang.strip()
    body = key.split(":", 1)[-1].strip()
    body = ALIASES.get(body, body)
    if body in mapping:
        return mapping[body]
    padded = " " + body + " "
    candidates = [
        v for k, v in mapping.items() if padded in (" " + k.strip() + " ")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def make_env(env_cls, bench, task_id):
    return env_cls(
        bddl_file_name=bench.get_task_bddl_file_path(task_id),
        robots=["Panda"],
        camera_heights=128,
        camera_widths=128,
        camera_names="agentview",
    )


def run_pair(
    model,
    vision_backbone,
    caches,
    env_cls,
    mapping,
    pair_name,
    pair,
    tasks,
    device,
    *,
    image_size=384,
    horizon_steps=400,
    trials_per_task=5,
    state_q01,
    state_q99,
):
    """Matched-block four-condition rollout for one objective pair."""
    i1, i2, bench_name = pair
    lang1, lang2 = tasks[i1], tasks[i2]
    mapping = mapping[bench_name]
    m1, m2 = match_task(lang1, mapping), match_task(lang2, mapping)
    if m1 is None or m2 is None:
        print(f"[{pair_name}] SKIP: env not matched "
              f"(g1={'ok' if m1 else lang1[:40]}, g2={'ok' if m2 else lang2[:40]})")
        return None
    bench1, task1, tid1 = m1
    bench2, task2, tid2 = m2
    env1 = make_env(env_cls, bench1, tid1)
    env2 = make_env(env_cls, bench2, tid2)

    # Same-scene sanity: init states of the two objectives must be close.
    s1 = bench1.get_task_init_states(tid1)
    s2 = bench2.get_task_init_states(tid2)
    n = min(trials_per_task, len(s1), len(s2))
    diffs = [np.abs(a - b).max() for a, b in zip(s1[:n], s2[:n])]
    print(f"[{pair_name}] init-state max diff over {n} matched states: "
          f"{max(diffs) if diffs else float('nan'):.2e} (target < 1e-3 for same scene)")

    cache1, cache2 = caches[i1], caches[i2]
    d_blocks, o_blocks = [], []
    for s_idx in range(n):
        # D conditions (matched instruction)
        r = rollout_task(model, vision_backbone, cache1, env1, [s1[s_idx]], device,
                         image_size=image_size, horizon_steps=horizon_steps,
                         state_q01=state_q01, state_q99=state_q99)
        r1 = r[0]
        r = rollout_task(model, vision_backbone, cache2, env2, [s2[s_idx]], device,
                         image_size=image_size, horizon_steps=horizon_steps,
                         state_q01=state_q01, state_q99=state_q99)
        r2 = r[0]
        # O conditions (swapped instruction)
        r = rollout_task(model, vision_backbone, cache2, env1, [s1[s_idx]], device,
                         image_size=image_size, horizon_steps=horizon_steps,
                         state_q01=state_q01, state_q99=state_q99)
        r3 = r[0]
        r = rollout_task(model, vision_backbone, cache1, env2, [s2[s_idx]], device,
                         image_size=image_size, horizon_steps=horizon_steps,
                         state_q01=state_q01, state_q99=state_q99)
        r4 = r[0]
        d_blocks.append(0.5 * (r1 + r2))
        o_blocks.append(0.5 * (r3 + r4))
        print(f"[{pair_name}] block {s_idx}: D=({r1},{r2}) O=({r3},{r4}) "
              f"D-O={d_blocks[-1] - o_blocks[-1]:+.2f}", flush=True)

    env1.close()
    env2.close()
    return np.array(d_blocks), np.array(o_blocks)


def bootstrap_ci(blocks: np.ndarray, n_iter=1000, seed=1234):
    rng = np.random.default_rng(seed)
    n = len(blocks)
    means = np.empty(n_iter)
    for k in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[k] = blocks[idx].mean()
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser(description="L_m same-scene dual-objective verdict")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    state_q01 = payload["normalization"]["state_q01"].numpy()
    state_q99 = payload["normalization"]["state_q99"].numpy()
    tasks = payload["metadata"]["tasks"]
    assert len(tasks) == 12, f"expected the 12-task dataset, got {len(tasks)}"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(ckpt["model"])

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", max_tokens=64, local_files_only=True
    )
    if ckpt.get("vjepa_state_dict"):
        # e2e 类 checkpoint（B40k/C1/C2）：V-JEPA 被微调过，必须加载训练后权重
        vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
        print("vision: loaded vjepa_state_dict from checkpoint")
    vision_backbone.freeze_all()

    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    if ckpt.get("qwen_state_dict"):
        qwen_state = {
            k.removeprefix("text_model."): v for k, v in ckpt["qwen_state_dict"].items()
        }
        missing, unexpected = text_backbone.text_model.load_state_dict(
            qwen_state, strict=False
        )
        print(f"qwen loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if ckpt.get("lora"):
        from va_compound.backbones import apply_lora

        rank = int(ckpt.get("training_contract", {}).get("lora_rank", 32))
        apply_lora(text_backbone.text_model, rank=rank)
        own = dict(text_backbone.text_model.named_parameters())
        for name, value in ckpt["lora"].items():
            clean = name.removeprefix("text_model.")
            if clean in own:
                own[clean].data.copy_(value)
    text_backbone.text_model.eval()
    hidden, mask = text_backbone.encode(tasks)
    del text_backbone
    caches = [
        model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(tasks))
    ]

    mapping, env_cls = build_env_mapping()
    print("env benchmarks:", {k: len(v) for k, v in mapping.items()})

    summary = []
    for name, (i1, i2), bench_name in PAIRS[: args.max_pairs]:
        out = run_pair(
            model, vision_backbone, caches, env_cls, mapping, name, (i1, i2, bench_name),
            tasks, device, image_size=384, horizon_steps=args.horizon,
            trials_per_task=args.trials_per_task,
            state_q01=state_q01, state_q99=state_q99,
        )
        if out is None:
            continue
        d_blocks, o_blocks = out
        diff = d_blocks - o_blocks
        lo, hi = bootstrap_ci(diff, seed=args.seed)
        lm = diff.mean()
        D = d_blocks.mean()
        O = o_blocks.mean()
        print(f"\n[{name}] D={D:.3f} O={O:.3f} L_m = {lm:+.4f}  "
              f"CI95 [{lo:+.4f}, {hi:+.4f}]  n_blocks={len(diff)}")
        summary.append((name, lm, lo, hi))

    print("\n===== L_m SUMMARY =====")
    for name, lm, lo, hi in summary:
        print(f"{name:24s} L_m={lm:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
