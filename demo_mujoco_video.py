"""MuJoCo 可视化 demo：当前 MW checkpoint 在 MetaWorld 环境中的策略 rollout 视频。

- 推理：默认 CPU（不干扰正在 GPU 上跑的闭环评估）；--device cuda 可用 GPU（text backbone 自动 bf16 压显存）
- 渲染：EGL（metaworld rgb_array, corner2 相机，与训练数据同款）
- 输出：MP4（drawer-close-v3 / reach-v3 / push-v3）

用法: python demo_mujoco_video.py [--tasks ...] [--out demo_mw_rollout.mp4] [--max-steps 240] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch

from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy
from prepare_pnpw_features import QwenTextBackbone
from eval_metaworld import preprocess

IMAGE_SIZE = 384
VISION_WINDOW = 4
DECISION_STRIDE = 6
ACTION_HORIZON = 8
CHECKPOINT = "checkpoints/metaworld_va8_40k_full.pt"
FEATURES = "data/metaworld_features_v2_full.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default="drawer-close-v3,reach-v3,push-v3")
    p.add_argument("--out", type=str, default="demo_mw_rollout.mp4")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def write_mp4(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"saved {path} ({len(frames)} frames, {w}x{h})")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = "bfloat16" if device.type == "cuda" else "float32"

    import json
    import metaworld

    cfg = json.load(open(
        "/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json"
    ))
    desc = cfg["TASK_DESCRIPTIONS"]

    print(f"loading checkpoint ({device})...", flush=True)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(ckpt["model"])

    features = torch.load(FEATURES, map_location="cpu", weights_only=True)
    aq01 = features["normalization"]["action_q01"].numpy()
    aq99 = features["normalization"]["action_q99"].numpy()
    sq01 = features["normalization"]["state_q01"].numpy()
    sq99 = features["normalization"]["state_q99"].numpy()
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)

    print(f"loading vision/language backbones ({device}, {dtype})...", flush=True)
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=dtype, max_tokens=64, local_files_only=True
    ).eval()
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype=dtype, local_files_only=True
    )

    env_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    hidden, mask = text_backbone.encode([desc.get(n, n) for n in env_names])
    caches = [
        model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(env_names))
    ]

    all_frames: list[np.ndarray] = []
    for env_i, name in enumerate(env_names):
        print(f"rolling out {name} ...", flush=True)
        mt1 = metaworld.MT1(name, seed=42)
        env = mt1.train_classes[name](render_mode="rgb_array", camera_name="corner2")
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env._freeze_rand_vec = False
        obs, _ = env.reset(seed=1000 + env_i)

        frame_buffer: list[np.ndarray] = []
        memory = None
        last_norm = np.zeros(4)
        chunk = np.zeros((ACTION_HORIZON, 4))
        success = False
        for step in range(args.max_steps):
            img = np.asarray(env.render())
            all_frames.append(img)
            frame_buffer.append(img)
            if len(frame_buffer) > (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                frame_buffer.pop(0)
            if step % DECISION_STRIDE == 0 and len(frame_buffer) >= VISION_WINDOW:
                indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
                frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
                clip = torch.cat([preprocess(f, IMAGE_SIZE) for f in frames], dim=0).unsqueeze(0).to(device)
                with torch.inference_mode():
                    tokens = vision_backbone(clip, pooling="flat")  # [1, 64, D]
                    state = np.clip(
                        2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0
                    ).astype(np.float32)
                    cond, memory = model.encode_condition(
                        tokens,
                        torch.tensor(state, dtype=torch.float32)[None].to(device),
                        torch.tensor(last_norm, dtype=torch.float32)[None].to(device),
                        language_cache=caches[env_i],
                        visual_memory=memory,
                        return_visual_memory=True,
                    )
                    chunk = model.sample_actions(cond, steps=32)[0].cpu().numpy()
            # 与训练标签一致裁剪模型输出到 [-1,1]（robust_normalize 存盘即 clip），
            # 再反归一化；prev 反馈（last_norm）同样用裁剪值
            norm_action = np.clip(chunk[step % DECISION_STRIDE], -1.0, 1.0)
            action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
            obs, r_env, term, trunc, info = env.step(action)
            last_norm = norm_action
            if info.get("success"):
                success = True
                img = np.asarray(env.render())
                all_frames.append(img)
                break
            if term or trunc:
                break
        print(f"  {name}: success={success}, steps={len(all_frames)}", flush=True)
        env.close()

    write_mp4(all_frames, args.out)


if __name__ == "__main__":
    main()
