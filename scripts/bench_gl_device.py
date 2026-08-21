"""确认 MuJoCo 的 GL 上下文到底落在哪个设备上，并在渲染时观察 GPU 占用。

渲染耗时不随像素数变化（480x480 与 224x224 同为 ~108ms），且 osmesa 与 egl
几乎一致，指向「EGL 静默退回软件光栅化」。GL_VENDOR / GL_RENDERER 字符串可以
直接定论：NVIDIA 硬件路径会报 "NVIDIA Corporation"，Mesa 软件路径报
"llvmpipe" 或 "softpipe"。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time

BACKEND = os.environ.get("MUJOCO_GL", "<unset>")
ENV_NAME = os.environ.get("BENCH_ENV", "door-unlock-v3")

GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02


def gl_strings() -> dict[str, str]:
    """上下文已由 MuJoCo 建好，这里只查询当前 current context 的标识。"""
    out: dict[str, str] = {}
    for name in ("libGL.so.1", "libGLESv2.so.2", "libOSMesa.so.8"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            lib.glGetString.restype = ctypes.c_char_p
            for label, code in (
                ("GL_VENDOR", GL_VENDOR),
                ("GL_RENDERER", GL_RENDERER),
                ("GL_VERSION", GL_VERSION),
            ):
                value = lib.glGetString(code)
                if value:
                    out[label] = value.decode(errors="replace")
            if out:
                out["_via"] = name
                return out
        except AttributeError:
            continue
    return out


def sample_gpu(stop: threading.Event, samples: list[str]) -> None:
    while not stop.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--format=csv,noheader", "--query-gpu=utilization.gpu"],
                capture_output=True, text=True, timeout=5,
            )
            samples.append(result.stdout.strip())
        except Exception:
            samples.append("err")
        time.sleep(0.25)


def main() -> None:
    import metaworld
    import mujoco

    print(f"MUJOCO_GL={BACKEND}  mujoco={mujoco.__version__}")
    for key in ("MUJOCO_EGL_DEVICE_ID", "EGL_PLATFORM", "__EGL_VENDOR_LIBRARY_FILENAMES",
                "LIBGL_ALWAYS_SOFTWARE", "DISPLAY"):
        print(f"  {key}={os.environ.get(key, '<unset>')}")

    mt1 = metaworld.MT1(ENV_NAME, seed=42)
    env = mt1.train_classes[ENV_NAME](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env._freeze_rand_vec = False
    env.reset(seed=0)
    env.render()

    print()
    info = gl_strings()
    if info:
        for key, value in info.items():
            print(f"{key:<14}= {value}")
    else:
        print("could not query GL strings (no loadable GL entry point)")

    print()
    print(f"model geoms={env.model.ngeom} lights={env.model.nlight} "
          f"textures={env.model.ntex} meshes={env.model.nmesh}")

    stop = threading.Event()
    samples: list[str] = []
    watcher = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    watcher.start()
    mark = time.perf_counter()
    for _ in range(40):
        env.render()
    elapsed = time.perf_counter() - mark
    stop.set()
    watcher.join(timeout=2)
    print(f"\n40 renders in {elapsed:.2f}s ({1000 * elapsed / 40:.1f} ms/call)")
    print(f"GPU util samples during render: {samples}")


if __name__ == "__main__":
    main()
