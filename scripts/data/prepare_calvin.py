"""Prepare CALVIN debug dataset features for VA training.

Builds the same feature-cache contract as scripts/data/prepare_libero.py / prepare_metaworld.py:
  vision_tokens [N, T=4, 64, 768]   (V-JEPA features, window stride 2)
  language_hidden [N, 1, 2048] + language_mask
  proprio [N, T, 9]                 (tcp pos/ori euler + gripper width + gripper action + 0)
  previous_action [N, T, 7]
  actions [N, T, H=8, 7]            (raw actions normalized to [-1,1] via act bounds)
  pair_id / instruction_id / normalization

NOTE: the debug split contains a single episode with 8 annotated language
segments (~300 decision points) — enough for an in-distribution smoke train,
not for benchmark-level numbers.  Full CALVIN (task_D_D, 166 GB) is the
benchmark split.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import os

import numpy as np
import torch

from prepare_pnpw_features import QwenTextBackbone
from va_compound.backbones import VJEPA21Backbone

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

VISION_WINDOW = 4
VISION_STRIDE = 2
SEQUENCE_LENGTH = 4
CONTROL_STRIDE = 3  # 30 Hz control, decide every 3 frames (10 Hz)
ACTION_HORIZON = 8
IMAGE_SIZE = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CALVIN debug features")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/media/ryan/robot-data/datasets/benchmark_data/raw/calvin/calvin_debug_dataset/validation"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/calvin_debug_features.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.model_dtype)
    data_dir: Path = args.data_dir

    # ---- load frames ----
    ep_ids = np.load(data_dir / "ep_start_end_ids.npy")
    start_id, end_id = int(ep_ids[0][0]), int(ep_ids[0][1])
    n_frames = end_id - start_id + 1
    print(f"episode frames {start_id}..{end_id} = {n_frames}")

    frame_ids = list(range(start_id, end_id + 1))
    raw = {k: np.zeros((n_frames, k_len), dtype=np.float32) for k, k_len in
           (("robot_obs", 15), ("scene_obs", 24), ("rel_actions", 7))}
    rgb = np.zeros((n_frames, 200, 200, 3), dtype=np.uint8)
    for i, fid in enumerate(frame_ids):
        f = np.load(data_dir / f"episode_{fid:07d}.npz", allow_pickle=True)
        raw["robot_obs"][i] = f["robot_obs"]
        raw["scene_obs"][i] = f["scene_obs"]
        raw["rel_actions"][i] = f["rel_actions"]
        rgb[i] = f["rgb_static"]
    print(f"loaded {n_frames} frames")

    # ---- normalization ----
    import yaml
    stats = yaml.safe_load(open(data_dir / "statistics.yaml"))
    r_mean = np.asarray(stats["robot_obs"]["mean"], dtype=np.float32)
    r_std = np.asarray(stats["robot_obs"]["std"], dtype=np.float32)
    act_min = np.asarray(stats["act_min_bound"], dtype=np.float32)
    act_max = np.asarray(stats["act_max_bound"], dtype=np.float32)

    def norm_state(robot_obs: np.ndarray) -> np.ndarray:
        z = (robot_obs - r_mean) / r_std
        p9 = np.concatenate([z[:7], z[14:15], np.zeros(1)]).astype(np.float32)
        return np.clip(p9, -3.0, 3.0) / 3.0

    def norm_action(a: np.ndarray) -> np.ndarray:
        scale = np.where(act_max - act_min < 1e-6, 1.0, act_max - act_min)
        n = 2.0 * (a - act_min) / scale - 1.0
        return np.clip(n, -1.0, 1.0).astype(np.float32)

    # ---- language segments ----
    lang = np.load(data_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    anns = [str(a) for a in lang["language"]["ann"]]
    indx = lang["info"]["indx"]
    seg_of_frame = np.full(n_frames, -1, dtype=np.int64)
    for seg_i, (s, e) in enumerate(indx):
        lo, hi = max(s, start_id), min(e, end_id)
        seg_of_frame[lo - start_id : hi - start_id + 1] = seg_i
    print(f"annotated frames: {(seg_of_frame >= 0).sum()}/{n_frames}")

    # ---- vision features (V-JEPA) ----
    from prepare_metaworld import preprocess_batch

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=args.model_dtype, max_tokens=64, local_files_only=True
    )
    vision_tokens = np.zeros((n_frames, 64, 768), dtype=np.float16)
    # decision-point windows: [d-6, d-4, d-2, d] for every possible decision d
    # (encode each frame once via a sliding window of 4-frame clips)
    clips = [
        [rgb[max(0, d - (VISION_WINDOW - 1 - k) * VISION_STRIDE)] for k in range(VISION_WINDOW)]
        for d in range(n_frames)
    ]
    with torch.inference_mode():
        for i in range(0, n_frames, 8):
            inputs = preprocess_batch(clips[i : i + 8], IMAGE_SIZE).to(device)
            flat, _ = vision_backbone.forward_variants(inputs)
            vision_tokens[i : i + 8] = flat.cpu().numpy().astype(np.float16)
    del vision_backbone
    print(f"vision features done {vision_tokens.shape}")

    # ---- language features (Qwen) ----
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype=args.model_dtype, local_files_only=True
    )
    hidden, mask = text_backbone.encode(anns)
    hidden = hidden.cpu().numpy()
    mask = mask.cpu().numpy()
    del text_backbone

    # ---- decision points ----
    plans = []
    for seg_i in range(len(anns)):
        valid = np.where(seg_of_frame == seg_i)[0]
        if len(valid) == 0:
            continue
        lo, hi = int(valid.min()), int(valid.max())
        # decision points: need window before + horizon after
        for d in range(lo + (VISION_WINDOW - 1) * VISION_STRIDE, hi - ACTION_HORIZON + 1):
            plans.append((seg_i, d))
    print(f"decision points: {len(plans)}")

    vt, pr, pa, ac, lh, lm, iid = [], [], [], [], [], [], []
    for seg_i, d in plans:
        # vision window: d-6, d-4, d-2, d
        idx = [d - (VISION_WINDOW - 1 - k) * VISION_STRIDE for k in range(VISION_WINDOW)]
        vt.append(vision_tokens[idx])
        pr.append(np.stack([norm_state(raw["robot_obs"][j]) for j in idx]))
        prev = norm_action(raw["rel_actions"][d])
        pa.append(np.stack([prev] * SEQUENCE_LENGTH))
        acts = np.stack([norm_action(raw["rel_actions"][d + 1 + k]) for k in range(ACTION_HORIZON)])
        ac.append(np.stack([acts] * SEQUENCE_LENGTH))
        lh.append(hidden[seg_i])
        lm.append(mask[seg_i])
        iid.append(seg_i)

    payload = {
        "vision_tokens": torch.from_numpy(np.asarray(vt, dtype=np.float16)),
        "language_hidden": torch.from_numpy(np.asarray(lh, dtype=np.float32)),
        "language_mask": torch.from_numpy(np.asarray(lm, dtype=np.int64)),
        "proprio": torch.from_numpy(np.asarray(pr, dtype=np.float32)),
        "previous_action": torch.from_numpy(np.asarray(pa, dtype=np.float32)),
        "actions": torch.from_numpy(np.asarray(ac, dtype=np.float32)),
        "pair_id": torch.arange(len(plans), dtype=torch.long),
        "instruction_id": torch.tensor(iid, dtype=torch.long),
        "metadata": {"tasks": anns, "source": "calvin_debug"},
        "normalization": {
            "action_q01": torch.from_numpy(act_min),
            "action_q99": torch.from_numpy(act_max),
            "state_q01": torch.from_numpy(r_mean - 3 * r_std),
            "state_q99": torch.from_numpy(r_mean + 3 * r_std),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved={args.output} shape={payload['vision_tokens'].shape}")


if __name__ == "__main__":
    main()
