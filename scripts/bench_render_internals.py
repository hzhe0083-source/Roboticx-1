"""拆开 env.render() 的 105ms：是每次调用的固定开销，还是光栅化本身？

osmesa 与 egl 实测几乎相同（107.6 vs 104.7 ms），说明瓶颈不在 GL 后端。
本脚本分离三件事：
  1. 连续 render 不 step —— 若仍是 ~105ms，则与场景变化无关
  2. renderer 对象是否跨调用复用 —— 若每次新建，开销来自上下文/缓冲重建
  3. 直接用 mujoco.Renderer 在 480 与 224 两个尺寸渲染 —— 看是否随像素数缩放
"""
from __future__ import annotations

import os
import time

import numpy as np

BACKEND = os.environ.get("MUJOCO_GL", "<unset>")
ENV_NAME = os.environ.get("BENCH_ENV", "door-unlock-v3")
N = 30


def bench(label: str, fn, n: int = N) -> float:
    for _ in range(3):
        fn()
    mark = time.perf_counter()
    for _ in range(n):
        fn()
    per_call = 1000.0 * (time.perf_counter() - mark) / n
    print(f"{label:<46} {per_call:8.3f} ms/call")
    return per_call


def main() -> None:
    import mujoco
    import metaworld

    print(f"MUJOCO_GL={BACKEND}  mujoco={mujoco.__version__}")
    mt1 = metaworld.MT1(ENV_NAME, seed=42)
    env = mt1.train_classes[ENV_NAME](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env._freeze_rand_vec = False
    np.random.seed(0)
    env.reset(seed=0)

    bench("env.render() back-to-back (no step)", env.render)

    # renderer 是否复用？MujocoRenderer 把实例缓存在 _viewers / viewer 上。
    renderer = getattr(env, "mujoco_renderer", None)
    if renderer is not None:
        viewers = getattr(renderer, "_viewers", None)
        print(f"mujoco_renderer            : {type(renderer).__name__}")
        print(f"  _viewers cache           : {list(viewers) if viewers else viewers}")
        identities = set()
        for _ in range(3):
            env.render()
            viewers = getattr(renderer, "_viewers", {}) or {}
            identities.update(id(v) for v in viewers.values())
        print(f"  distinct viewer ids over 3 renders: {len(identities)}")

    # 直接用 mujoco.Renderer，绕开 gymnasium 封装。
    for size in (480, 224):
        native = mujoco.Renderer(env.model, height=size, width=size)
        camera = env.model.camera("corner2").id if hasattr(env.model, "camera") else -1

        def render_native() -> None:
            native.update_scene(env.data, camera=camera)
            native.render()

        bench(f"mujoco.Renderer {size}x{size} update+render", render_native)

        def render_only() -> None:
            native.render()

        bench(f"mujoco.Renderer {size}x{size} render only", render_only)
        native.close()


if __name__ == "__main__":
    main()
