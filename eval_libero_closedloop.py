"""Closed-loop evaluation of the VA policy in the LIBERO simulation.

Each task is rolled out from several initial states; at every decision point
the policy receives the causal 4-frame agentview window (resized to 384),
encodes it with V-JEPA, conditions the VA policy with the task language cache,
and samples an 8-step action chunk executed with receding horizon.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
from torch.nn import functional as F

from prepare_pnpw_features import QwenTextBackbone
from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406))
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225))

# Decision cadence: 4-frame window sampled with stride 2, decide every 2 steps.
DECISION_STRIDE = 2
WINDOW = 4
ACTION_STEPS_PER_CHUNK = 2  # receding horizon


def build_task_envs(benchmark_names=("libero_90", "libero_spatial", "libero_object", "libero_goal")):
    from libero.libero import benchmark as bm
    import libero.libero.utils as lu

    lu.set_libero_path(custom_location=os.path.dirname(os.path.dirname(lu.__file__)))
    from libero.libero.envs import OffScreenRenderEnv

    mapping = {}
    for name in benchmark_names:
        try:
            bench = bm.get_benchmark(name)()
        except KeyError:
            continue
        for task_id in range(bench.get_num_tasks()):
            task = bench.get_task(task_id)
            language = task.language.strip()
            mapping.setdefault(language, (bench, task, task_id))
    return mapping, OffScreenRenderEnv


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    """uint8 [H,W,3] -> normalized [1,3,S,S]."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)[None]
    if tensor.shape[-1] != image_size or tensor.shape[-2] != image_size:
        tensor = F.interpolate(
            tensor, size=(image_size, image_size), mode="bicubic", align_corners=False
        )
    mean = IMAGE_MEAN.view(1, 3, 1, 1)
    std = IMAGE_STD.view(1, 3, 1, 1)
    return (tensor - mean) / std


def rollout_task(
    model,
    vision_backbone,
    language_cache,
    env,
    init_states,
    device,
    *,
    image_size,
    horizon_steps,
    state_q01,
    state_q99,
):
    """Roll out one task over its init states; return success flags."""
    successes = []
    for init_state in init_states:
        env.reset()
        obs = env.set_init_state(init_state)
        frame_buffer: list[np.ndarray] = []
        memory = None
        action_chunk = None
        chunk_start_step = 0
        last_action = np.zeros(7)
        success = False
        step_count = 0
        while step_count < horizon_steps:
            frame_buffer.append(obs["agentview_image"])
            if len(frame_buffer) > (WINDOW - 1) * DECISION_STRIDE + 1:
                frame_buffer.pop(0)
            if (len(frame_buffer) - 1) % DECISION_STRIDE == 0 and len(frame_buffer) >= WINDOW:
                # Build causal window [t-6, t-4, t-2, t] and encode.
                indices = list(range(0, WINDOW * DECISION_STRIDE, DECISION_STRIDE))
                frames = [frame_buffer[len(frame_buffer) - 1 - i] for i in reversed(indices)]
                clip = torch.cat(
                    [preprocess(f, image_size) for f in frames], dim=0
                ).to(device).unsqueeze(0)
                with torch.inference_mode():
                    tokens = vision_backbone(clip, pooling="flat")
                # Training proprio is [7 joint angles, 2 gripper] normalized
                # with the dataset state quantiles; apply the same transform.
                state = np.concatenate(
                    [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
                )
                scale = state_q99 - state_q01
                scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
                state = np.clip(2.0 * (state - state_q01) / scale - 1.0, -1.0, 1.0)
                proprio = torch.tensor(state, dtype=torch.float32, device=device)[None, None]
                previous = torch.tensor(
                    last_action, dtype=torch.float32, device=device
                )[None, None]
                with torch.inference_mode():
                    condition, memory = model.encode_condition(
                        tokens,
                        proprio[0],
                        previous[0],
                        language_cache=language_cache,
                        visual_memory=memory,
                        return_visual_memory=True,
                    )
                    action_chunk = model.sample_actions(condition, steps=8)[0]
                    action_chunk = action_chunk.cpu().numpy()
                    chunk_start_step = step_count
            # Execute receding-horizon steps of the latest chunk. 模型输出即归一化动作：
            # 与训练标签一致裁剪到 [-1,1]（scripts/data/prepare_libero.py robust_normalize 存盘即
            # clip），prev 反馈（last_action）同样用裁剪值，避免分布外输入。
            if action_chunk is not None:
                action = np.clip(
                    action_chunk[(step_count - chunk_start_step) % 8], -1.0, 1.0
                )
            else:
                action = np.zeros(7)  # warm-up frames before the first decision
            obs, reward, done, info = env.step(action)
            last_action = action
            step_count += 1
            if env.check_success():
                success = True
                break
            if done:
                break
        successes.append(success)
    return successes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="feature .pt for language cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400, help="max control steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    state_q01 = payload["normalization"]["state_q01"].numpy()
    state_q99 = payload["normalization"]["state_q99"].numpy()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**checkpoint["config"])
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(checkpoint["model"])

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", max_tokens=64, local_files_only=True
    )
    # Language cache from the dataset (precomputed Qwen features).
    tasks = payload["metadata"]["tasks"]
    # Re-encode with Qwen for the simulator task language strings.
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    hidden, mask = text_backbone.encode(tasks)
    del text_backbone
    # Per-task caches (batch=1) so each rollout conditions on one instruction.
    task_caches = [
        model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
        for i in range(len(tasks))
    ]

    mapping, env_cls = build_task_envs()
    print(f"matched env tasks: {len(mapping)}")

    results = {}
    for task_index, task_text in enumerate(tasks):
        key = task_text.strip()
        language_cache = task_caches[task_index]
        if key not in mapping:
            # Try matching the instruction after the scene prefix.
            candidate = key.split(":", 1)[-1].strip()
            found = None
            for language, value in mapping.items():
                if candidate in language or language in key:
                    found = value
                    break
            if found is None:
                print(f"NO ENV for: {key[:60]}")
                continue
            bench, task, task_id = found
        else:
            bench, task, task_id = mapping[key]
        env = env_cls(
            bddl_file_name=bench.get_task_bddl_file_path(task_id),
            robots=["Panda"],
            camera_heights=128,
            camera_widths=128,
            camera_names="agentview",
        )
        init_states = bench.get_task_init_states(task_id)[: args.trials_per_task]
        successes = rollout_task(
            model,
            vision_backbone,
            language_cache,
            env,
            init_states,
            device,
            image_size=384,
            horizon_steps=args.horizon,
            state_q01=state_q01,
            state_q99=state_q99,
        )
        env.close()
        results[task_text[:50]] = successes
        print(f"task {task_text[:50]}: success {sum(successes)}/{len(successes)}")

    total = sum(len(v) for v in results.values())
    won = sum(sum(v) for v in results.values())
    print(f"\nCLOSED-LOOP SUCCESS: {won}/{total} = {won / total:.1%}")


if __name__ == "__main__":
    main()
