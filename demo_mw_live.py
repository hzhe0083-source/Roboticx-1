"""MetaWorld 闭环 rollout 实时演示视频（与 eval_metaworld.py 协议逐行一致）。

在评估正在 GPU 上跑的同时，用 CPU 对指定任务做闭环 rollout 并渲染 corner2
相机视频（叠加任务名 / 步数 / 成功横幅），用于直观验证策略的闭环行为。

协议对齐点（2026-08-07 与 eval_metaworld.py 同步）：
- 视觉窗口 [d-6, d-4, d-2, d]（时间升序），决策节奏 --execute-steps 步；
- step 0 用首帧重复填充窗口 → 立即推理（避免旧缺陷：先执行 chunk 的零值）；
- chunk 采样 32 步，按 chunk_start_step 相位取模 ACTION_HORIZON=8 执行；
- previous_action 用模型自身上一归一化动作（自激）；state=obs[:4] 分位数归一化。

用法: python demo_mw_live.py [--checkpoint ...] [--features ...] [--tasks ...] [--out ...]
默认: mw_va2_v4_40k.pt / metaworld_features_v4.pt / CPU
"""
from __future__ import annotations

import argparse
import json
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
MW_CONFIG = "/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json"

DEFAULT_CHECKPOINT = "checkpoints/mw_va2_v4_40k.pt"
DEFAULT_FEATURES = "data/metaworld_features_v4.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument(
        "--tasks",
        type=str,
        default="button-press-v3,coffee-button-v3,assembly-v3",
        help="逗号分隔的 metaworld env 名",
    )
    p.add_argument("--trials", type=int, default=1, help="每任务 rollout 次数（不同 seed）")
    p.add_argument("--out", type=str, default="demo_mw_v4_live.mp4")
    p.add_argument("--max-steps", type=int, default=500, help="与评估一致的 horizon")
    p.add_argument("--execute-steps", type=int, default=DECISION_STRIDE)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def write_mp4(frames: list[np.ndarray], path: str, fps: int) -> None:
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
    print(f"saved {path} ({len(frames)} frames, {w}x{h}, {fps} fps)", flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)  # 与评估一致：固定 flow 采样噪声
    device = torch.device(args.device)
    dtype = "bfloat16" if device.type == "cuda" else "float32"

    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(candidate):
            font = ImageFont.truetype(candidate, 22)
            break

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(ckpt["model"])
    assert config.proprio_dim == 4 and config.action_dim == 4, "expect 4D MetaWorld config"

    features = torch.load(args.features, map_location="cpu", weights_only=True)
    sq01 = features["normalization"]["state_q01"].numpy()
    sq99 = features["normalization"]["state_q99"].numpy()
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    aq01 = features["normalization"]["action_q01"].numpy()
    aq99 = features["normalization"]["action_q99"].numpy()

    print(f"loading backbones ({device}, {dtype})...", flush=True)
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=dtype, max_tokens=64, local_files_only=True
    )
    if ckpt.get("vjepa_state_dict"):
        vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
    vision_backbone.freeze_all()
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype=dtype, local_files_only=True
    )
    if ckpt.get("qwen_state_dict"):
        qwen_state = {k.removeprefix("text_model."): v for k, v in ckpt["qwen_state_dict"].items()}
        text_backbone.text_model.load_state_dict(qwen_state, strict=False)
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

    mw_config = json.load(open(MW_CONFIG))
    desc = mw_config["TASK_DESCRIPTIONS"]
    env_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    hidden, mask = text_backbone.encode([desc.get(n, n) for n in env_names])
    caches = [
        model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(env_names))
    ]
    del text_backbone

    import metaworld

    all_frames: list[np.ndarray] = []
    outcomes: list[tuple[str, bool, int]] = []
    for env_i, name in enumerate(env_names):
        mt1 = metaworld.MT1(name, seed=42)
        env = mt1.train_classes[name](render_mode="rgb_array", camera_name="corner2")
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env._freeze_rand_vec = False
        label = desc.get(name, name)
        for trial in range(args.trials):
            obs, _ = env.reset(seed=1000 * env_i + trial)
            frame_buffer: list[np.ndarray] = []
            memory = None
            last_norm = np.zeros(4)
            chunk = np.zeros((ACTION_HORIZON, 4))
            chunk_start_step = 0
            success = False
            episode_frames: list[np.ndarray] = []
            print(f"[{name} trial {trial}] rollout ...", flush=True)
            for step in range(args.max_steps):
                img = np.asarray(env.render())
                frame_buffer.append(img)
                if step == 0:
                    while len(frame_buffer) < (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                        frame_buffer.insert(0, img)
                if len(frame_buffer) > (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                    frame_buffer.pop(0)
                if step % args.execute_steps == 0 and len(frame_buffer) >= VISION_WINDOW:
                    indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
                    frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
                    clip = torch.cat([preprocess(f, IMAGE_SIZE) for f in frames], dim=0).to(device)
                    with torch.inference_mode():
                        tokens = vision_backbone(clip.unsqueeze(0), pooling="flat")
                    state = np.clip(
                        2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0
                    ).astype(np.float32)
                    with torch.inference_mode():
                        cond, memory = model.encode_condition(
                            tokens,
                            torch.tensor(state, device=device)[None, None][0],
                            torch.tensor(last_norm, dtype=torch.float32, device=device)[None, None][0],
                            language_cache=caches[env_i],
                            visual_memory=memory,
                            return_visual_memory=True,
                        )
                        chunk = model.sample_actions(cond, steps=32)[0].cpu().numpy()
                        chunk_start_step = step
                # 与训练标签一致裁剪模型输出到 [-1,1]（robust_normalize 存盘即 clip），
                # 再反归一化；prev 反馈（last_norm）同样用裁剪值
                norm_action = np.clip(
                    chunk[(step - chunk_start_step) % ACTION_HORIZON], -1.0, 1.0
                )
                action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
                obs, reward, terminated, truncated, info = env.step(action)
                last_norm = norm_action
                # 叠加字幕
                canvas = Image.fromarray(img).convert("RGB")
                draw = ImageDraw.Draw(canvas)
                if font:
                    draw.text((10, 8), f"{label}", fill="white", stroke_width=2, stroke_fill="black", font=font)
                    draw.text((10, 36), f"step {step}/{args.max_steps}", fill="white", stroke_width=2, stroke_fill="black", font=font)
                success = bool(info.get("success"))
                if success:
                    episode_frames.append(np.asarray(canvas))
                    break
                if terminated or truncated:
                    break
                episode_frames.append(np.asarray(canvas))
            # 结尾横幅（成功绿 / 失败红）
            h, w = episode_frames[-1].shape[:2]
            banner = Image.new("RGB", (w, 44), (34, 139, 34) if success else (178, 34, 34))
            banner_draw = ImageDraw.Draw(banner)
            if font:
                banner_draw.text(
                    (12, 8),
                    f"{name}: SUCCESS at step {step}" if success else f"{name}: FAILED after {args.max_steps} steps",
                    fill="white", stroke_width=2, stroke_fill="black", font=font,
                )
            episode_frames.append(np.asarray(banner))
            all_frames.extend(episode_frames)
            outcomes.append((name, success, step))
            print(f"  {name}: success={success} steps={step}", flush=True)
        env.close()

    write_mp4(all_frames, args.out, args.fps)
    for name, success, steps in outcomes:
        print(f"{name}: {'SUCCESS' if success else 'FAILED'} ({steps} steps)")


if __name__ == "__main__":
    main()
