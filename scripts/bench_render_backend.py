"""定位闭环评测的串行瓶颈：MuJoCo 渲染 vs 仿真步进。

评测期实测 GPU 利用率 0~6%、进程只占 88 核里的 1.8 核，说明卡在单线程 CPU 环节。
渲染契约是 ``true_simulator_render_480_to_dino224_v1``（渲 480x480 再缩到 224），
而启动脚本用 ``MUJOCO_GL=osmesa``——纯软件 OpenGL，在 CPU 上单线程光栅化。
服务器有 libEGL_nvidia.so + 10_nvidia.json，可以走 GPU 硬件渲染。

后端由 ``MUJOCO_GL`` 在导入时决定，因此每个后端要单独一个进程：
    MUJOCO_GL=osmesa python scripts/bench_render_backend.py
    MUJOCO_GL=egl    python scripts/bench_render_backend.py
"""
from __future__ import annotations

import os
import time

import numpy as np

BACKEND = os.environ.get("MUJOCO_GL", "<unset>")
ENV_NAME = os.environ.get("BENCH_ENV", "door-unlock-v3")
N_WARMUP = 5
N_STEPS = 60


def main() -> None:
    import metaworld

    mt1 = metaworld.MT1(ENV_NAME, seed=42)
    env = mt1.train_classes[ENV_NAME](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env._freeze_rand_vec = False

    np.random.seed(0)
    env.reset(seed=0)

    for _ in range(N_WARMUP):
        env.step(env.action_space.sample())
        env.render()

    step_total = 0.0
    render_total = 0.0
    shape = None
    for _ in range(N_STEPS):
        action = env.action_space.sample()

        mark = time.perf_counter()
        env.step(action)
        step_total += time.perf_counter() - mark

        mark = time.perf_counter()
        image = env.render()
        render_total += time.perf_counter() - mark
        shape = None if image is None else image.shape

    step_ms = 1000.0 * step_total / N_STEPS
    render_ms = 1000.0 * render_total / N_STEPS
    print(f"backend           : MUJOCO_GL={BACKEND}")
    print(f"env               : {ENV_NAME}")
    print(f"render output     : {shape}")
    print(f"env.step()        : {step_ms:8.3f} ms/call")
    print(f"env.render()      : {render_ms:8.3f} ms/call")
    print(f"render share      : {100.0 * render_total / (step_total + render_total):5.1f}%")
    # 闭环一集：horizon=500 步仿真；execution_horizon=2 => 250 个决策点各渲 1 帧。
    per_episode = (500 * step_ms + 250 * render_ms) / 1000.0
    print(f"per-episode sim+render (500 step, 250 render): {per_episode:6.1f} s")
    print(f"20 trials         : {20 * per_episode / 60.0:6.1f} min")


if __name__ == "__main__":
    main()
