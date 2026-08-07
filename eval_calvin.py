"""Closed-loop evaluation on CALVIN debug dataset (language-conditioned chain).

Protocol (CALVIN ABC->D style, debug subset):
  - Load the single validation episode (553567..555241), restore its initial
    scene/robot state in the calvin_env simulation.
  - Feed the policy the 8 task-sentence annotations from auto_lang_ann one at a
    time; a task counts as solved when Tasks.get_task_info(start_info, info)
    reports it (state-based success, independent of vision).
  - Report L1 (fraction of the 8 segments solved) and the longest chain.

The policy is the ORA0 VA compound (V-JEPA 2.1 + Qwen + 4-layer VA head).
This eval is a zero-shot cross-embodiment check with the LIBERO-trained B40k
checkpoint (7-D actions), or an in-distribution eval with a CALVIN-trained
checkpoint when one is provided.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("PYBULLET_EGL", "1")

import numpy as np
import torch
from torch.nn import functional as F

from prepare_pnpw_features import QwenTextBackbone
from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

VISION_WINDOW = 4
DECISION_STRIDE = 3  # CALVIN control at 30 Hz; decide every 3 steps (10 Hz)
ACTION_HORIZON = 8
MAX_EPISODE_STEPS = 600  # 60 s of closed-loop control per language segment

CALVIN_ENV_ROOT = "/tmp/calvin_env_tmp"  # calvin_env repo root (scene assets + hydra conf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop CALVIN debug eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="path to calvin_debug_dataset/validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-segments", type=int, default=8)
    return parser.parse_args()


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)[None]
    if tensor.shape[-1] != image_size:
        tensor = F.interpolate(
            tensor, size=(image_size, image_size), mode="bicubic",
            align_corners=False, antialias=True,
        )
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    assert config.action_dim == 7, f"CALVIN eval needs 7-D actions, got {config.action_dim}"
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(ckpt["model"])

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", max_tokens=64, local_files_only=True
    )
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )

    # ---- language: the annotations of the single debug episode ----
    ann_path = args.data_dir / "lang_annotations" / "auto_lang_ann.npy"
    lang = np.load(ann_path, allow_pickle=True).item()
    anns = [str(a) for a in lang["language"]["ann"][: args.max_segments]]
    print(f"language segments: {len(anns)}")
    hidden, mask = text_backbone.encode(anns)
    caches = [
        model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(anns))
    ]
    del text_backbone

    # ---- CALVIN environment (headless EGL, static camera only) ----
    from hydra import compose, initialize
    from hydra.utils import instantiate as hydra_instantiate

    os.chdir(CALVIN_ENV_ROOT)  # hydra config_path must be relative to cwd
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(
            config_name="config_data_collection",
            overrides=[
                "use_vr=false",
                "cameras=static_and_gripper",
                "record=false",
                "save_dir=/tmp/calvin_eval",
                "env.use_scene_info=true",
                "env.use_egl=true",
            ],
        )
    env = hydra_instantiate(cfg.env)
    tasks = hydra_instantiate(cfg.tasks)

    # ---- episode frames ----
    ep_ids = np.load(args.data_dir / "ep_start_end_ids.npy")  # [[start, end]]
    start_id, end_id = int(ep_ids[0][0]), int(ep_ids[0][1])
    print(f"episode frames: {start_id}..{end_id} ({end_id - start_id + 1} frames)")

    def load_frame(frame_id: int) -> dict:
        return np.load(args.data_dir / f"episode_{frame_id:07d}.npz", allow_pickle=True)

    # ---- action de-normalization (CALVIN min/max bounds) ----
    import yaml
    stats = yaml.safe_load(open(args.data_dir / "statistics.yaml"))
    act_min = np.asarray(stats["act_min_bound"], dtype=np.float32)
    act_max = np.asarray(stats["act_max_bound"], dtype=np.float32)

    # ---- proprio: [tcp_pos(3), tcp_orn_euler(3), gripper_width(1), gripper_action(1), 0]
    #       -> z-score with the dataset statistics, then tanh-clip to [-1, 1] ----
    r_mean = np.asarray(stats["robot_obs"]["mean"], dtype=np.float32)
    r_std = np.asarray(stats["robot_obs"]["std"], dtype=np.float32)

    def proprio9(robot_obs: np.ndarray) -> np.ndarray:
        z = (robot_obs - r_mean) / r_std  # 15-D z-score
        p9 = np.concatenate([z[:7], z[14:15], [0.0]]).astype(np.float32)  # pos3+orn3+width+gripper+0
        return np.clip(p9, -3.0, 3.0) / 3.0  # squash to roughly [-1,1]

    # ---- chain evaluation ----
    frame0 = load_frame(start_id)
    start_info = None
    results = []
    for seg_idx in range(len(anns)):
        obs = env.reset(
            robot_obs=frame0["robot_obs"].astype(np.float32),
            scene_obs=frame0["scene_obs"].astype(np.float32),
        )
        start_info = env.get_info()
        frame_buffer: list[np.ndarray] = []
        last_norm = np.zeros(7, dtype=np.float32)
        chunk = np.zeros((ACTION_HORIZON, 7), dtype=np.float32)
        memory = None
        solved_any = False
        for step in range(MAX_EPISODE_STEPS):
            img = obs["rgb_obs"]["rgb_static"]
            frame_buffer.append(img)
            if step == 0:
                # 2026-08-06 评估缺陷修复（与 eval_metaworld.py 同源）：首决策前
                # 执行 chunk 初始零值会反归一化为动作区间中点（非零），提前移动
                # 机械手。用首帧重复填充窗口使 step 0 立即推理。
                while len(frame_buffer) < (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                    frame_buffer.insert(0, img)
            if len(frame_buffer) > (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                frame_buffer.pop(0)
            if step % DECISION_STRIDE == 0 and len(frame_buffer) >= VISION_WINDOW:
                indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
                frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
                clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
                with torch.inference_mode():
                    tokens = vision_backbone(clip.unsqueeze(0), pooling="flat")
                proprio = torch.tensor(
                    proprio9(obs["robot_obs"]), dtype=torch.float32, device=device
                )[None, None]
                previous = torch.tensor(last_norm, dtype=torch.float32, device=device)[None, None]
                with torch.inference_mode():
                    cond, memory = model.encode_condition(
                        tokens.to(device),
                        proprio[0],
                        previous[0],
                        language_cache=caches[seg_idx],
                        visual_memory=memory,
                        return_visual_memory=True,
                    )
                    chunk = model.sample_actions(cond, steps=8)[0].cpu().numpy()
            # execute the planned action for this step (re-plan every DECISION_STRIDE)
            # 与训练标签一致裁剪模型输出到 [-1,1]（prepare_calvin.py 存盘即 clip），
            # 再反归一化；prev 反馈（last_norm）同样用裁剪值，避免分布外输入
            a_norm = np.clip(chunk[step % DECISION_STRIDE], -1.0, 1.0).astype(np.float32)
            last_norm = a_norm
            # de-normalize [-1,1] -> CALVIN raw action (gripper must be +/-1)
            action = act_min + (a_norm + 1.0) * 0.5 * (act_max - act_min)
            action[-1] = 1.0 if a_norm[-1] > 0.0 else -1.0
            obs, _, _, info = env.step(action)
            achieved = tasks.get_task_info(start_info, info)
            if achieved:
                solved_any = True
                print(f"  seg {seg_idx} [{anns[seg_idx][:40]}] solved at step {step}: {achieved}")
                break
        results.append(solved_any)
        print(f"segment {seg_idx}: {'SOLVED' if solved_any else 'FAILED'} ({anns[seg_idx][:50]})")
        # advance start frame: move to the next segment boundary (from lang info)
        # debug set: segments are consecutive; jump ~1/8 of the episode
        seg_len = (end_id - start_id + 1) // len(anns)
        frame0 = load_frame(min(start_id + (seg_idx + 1) * seg_len, end_id))

    env.close()
    n = len(results)
    print("=" * 50)
    print(f"CALVIN debug chain: solved {sum(results)}/{n} segments, L1 = {sum(results) / n:.1%}")
    longest = 0
    cur = 0
    for r in results:
        cur = cur + 1 if r else 0
        longest = max(longest, cur)
    print(f"longest consecutive chain: {longest}/{n}")


if __name__ == "__main__":
    main()
