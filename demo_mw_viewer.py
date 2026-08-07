"""MetaWorld 闭环 rollout MuJoCo 实时窗口观看（--device cuda 时亦不干扰正在跑的评估）。

与 eval_metaworld.py 协议逐行一致：
- 视觉窗口 [d-6, d-4, d-2, d] 时间升序，决策节奏 --execute-steps 步；
- step 0 首帧重复填充 → 立即推理（规避旧缺陷：先执行零值 chunk）；
- chunk 采样 32 步，按 chunk_start_step 相位取模 ACTION_HORIZON=8 执行；
- previous_action 自激；state=obs[:4] 分位数归一化；
- 实时节奏：每步 sleep 至 env.dt（真实时间），决策推理时间外显。

显存策略：Qwen（1.88B）只在启动时于 CPU 编码一次任务文本并构建语言 cache，
随后释放，GPU 上仅常驻 V-JEPA(fp16) + VA(fp32) ≈ 0.7GB —— 与两个闭环评估共存。

窗口：MuJoCo viewer（corner2 固定相机，与训练/评估同款视角）；关窗即停止。
用法: python demo_mw_viewer.py [--tasks ...] [--trials 1] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 窗口化渲染必须用 glfw 后端；必须在任何 mujoco/gym 导入之前设置
os.environ["MUJOCO_GL"] = "glfw"
os.environ.setdefault("DISPLAY", ":0")

import numpy as np
import torch
from torch.nn import functional as F

from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy
from prepare_pnpw_features import QwenTextBackbone

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

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
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=500, help="与评估一致的 horizon")
    p.add_argument("--execute-steps", type=int, default=DECISION_STRIDE)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--hold-success", type=float, default=6.0, help="成功后保持窗口秒数")
    p.add_argument("--hold-fail", type=float, default=3.0, help="失败后保持窗口秒数")
    return p.parse_args()


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)[None]
    if tensor.shape[-1] != image_size:
        tensor = F.interpolate(
            tensor, size=(image_size, image_size), mode="bicubic",
            align_corners=False, antialias=True,
        )
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def set_viewer_camera(handle, model, camera_name: str) -> None:
    import mujoco

    cam_id = model.camera_name2id(camera_name)
    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    handle.cam.fixedcamid = cam_id if cam_id >= 0 else -1
    handle.cam.distance = 1.0


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = "float16" if device.type == "cuda" else "float32"

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

    print(f"loading V-JEPA ({device}, {dtype})...", flush=True)
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=dtype, max_tokens=64, local_files_only=True
    )
    if ckpt.get("vjepa_state_dict"):
        vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
    vision_backbone.freeze_all()

    # Qwen 只在 CPU 上编码一次任务文本（1.88B，约几十秒），随后释放
    mw_config = json.load(open(MW_CONFIG))
    desc = mw_config["TASK_DESCRIPTIONS"]
    env_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"encoding {len(env_names)} task texts with Qwen on CPU (one-time)...", flush=True)
    text_backbone = QwenTextBackbone.from_pretrained(
        device="cpu", dtype="float32", local_files_only=True
    )
    hidden, mask = text_backbone.encode([desc.get(n, n) for n in env_names])
    del text_backbone
    caches = [
        model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(env_names))
    ]
    del hidden, mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    import mujoco
    import mujoco.viewer  # 3.3.0 需显式导入子模块
    import metaworld

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
            print(f"[{name} trial {trial}] reset; opening viewer window ...", flush=True)
            handle = mujoco.viewer.launch_passive(
                env.model, env.data, show_left_ui=False, show_right_ui=False
            )
            set_viewer_camera(handle, env.model, "corner2")
            title = f"VA closed-loop: {label}"
            try:
                mujoco.glfw.glfw.set_window_title(handle._simulate.window, title)
            except Exception:
                pass

            frame_buffer: list[np.ndarray] = []
            memory = None
            last_norm = np.zeros(4)
            chunk = np.zeros((ACTION_HORIZON, 4))
            chunk_start_step = 0
            success = False
            sim_dt = float(getattr(env, "dt", env.model.opt.timestep * env.frame_skip))
            try:
                for step in range(args.max_steps):
                    t_loop = time.perf_counter()
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
                        proprio = torch.tensor(state, device=device)[None, None]
                        previous = torch.tensor(last_norm, dtype=torch.float32, device=device)[None, None]
                        t_inf = time.perf_counter()
                        with torch.inference_mode():
                            cond, memory = model.encode_condition(
                                tokens,
                                proprio[0],
                                previous[0],
                                language_cache=caches[env_i],
                                visual_memory=memory,
                                return_visual_memory=True,
                            )
                            chunk = model.sample_actions(cond, steps=32)[0].cpu().numpy()
                            chunk_start_step = step
                        print(
                            f"  decision @step {step:3d}: {1000 * (time.perf_counter() - t_inf):6.0f} ms",
                            flush=True,
                        )
                    # 与训练标签一致裁剪模型输出到 [-1,1]（robust_normalize 存盘即
                    # clip），再反归一化；prev 反馈（last_norm）同样用裁剪值
                    norm_action = np.clip(
                        chunk[(step - chunk_start_step) % ACTION_HORIZON], -1.0, 1.0
                    )
                    action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
                    obs, reward, terminated, truncated, info = env.step(action)
                    last_norm = norm_action
                    handle.sync()
                    if not handle.is_running():
                        print("viewer window closed; stopping.", flush=True)
                        handle.close()
                        env.close()
                        print("OUTCOMES:", outcomes)
                        return
                    success = bool(info.get("success"))
                    if success or terminated or truncated:
                        break
                    # 实时节奏：sleep 至 env.dt
                    time.sleep(max(0.0, sim_dt - (time.perf_counter() - t_loop)))
            except KeyboardInterrupt:
                handle.close()
                env.close()
                print("interrupted.", flush=True)
                return
            hold = args.hold_success if success else args.hold_fail
            print(f"[{name} trial {trial}] {'SUCCESS' if success else 'FAILED'} @step {step} "
                  f"(hold {hold:.0f}s)", flush=True)
            for _ in range(int(hold / 0.05)):
                handle.sync()
                if not handle.is_running():
                    break
                time.sleep(0.05)
            handle.close()
            outcomes.append((name, success, step))
        env.close()

    print("\nOUTCOMES:")
    for name, success, steps in outcomes:
        print(f"  {name}: {'SUCCESS' if success else 'FAILED'} ({steps} steps)")


if __name__ == "__main__":
    main()
