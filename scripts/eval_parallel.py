"""把闭环评测的 trial 拆到多进程并行，然后合并成单份结果。

为什么需要并行（实测，非推测）：
  - 评测期 GPU 利用率 0~6%，进程只占 88 核里的 1.8 核
  - env.step() 1.1ms，env.render() 108ms —— 渲染是仿真的 94 倍，占 99%
  - 渲染耗时不随像素数变化（480x480 与 224x224 同为 ~108ms），osmesa 与 egl
    也几乎一致；GL_RENDERER 查出来是 "llvmpipe"，即 CPU 软件光栅化
  - 强制 NVIDIA EGL 报 "EGL driver does not support PLATFORM_DEVICE"，且
    /dev/dri 不存在 —— 容器只给了 compute 能力，硬件渲染在容器内无法启用
  - LP_NUM_THREADS=8 能把渲染从 108ms 降到 80ms（1.35x），是唯一的单进程收益

因此并行是主要手段。每个 trial 开头都用 evaluation_episode_seed 重新播种
numpy/env/torch，跨 trial 无 RNG 依赖，所以分片结果与串行**逐 trial 完全一致**。

用法：
    python scripts/eval_parallel.py <ckpt> <features> [--shards 5] [--trials-per-task 10]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DINO = (
    "/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2."
    "lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors"
)
# 实测最优：1 线程 188ms / 8 线程 80ms / 32 线程 120ms（超订反而变慢）。
LLVMPIPE_THREADS = 8


def shard_bounds(total: int, shards: int) -> list[tuple[int, int]]:
    """把 [0, total) 切成 shards 段，长度差不超过 1。"""
    if not 1 <= shards <= total:
        raise ValueError(f"shards must be in [1, {total}], got {shards}")
    base, extra = divmod(total, shards)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        stop = start + base + (1 if index < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("features")
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--task-ids", default="0,16")
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dino", default=os.environ.get("DINO", DEFAULT_DINO))
    parser.add_argument("--python", default=os.environ.get("PY", "/opt/conda/bin/python"))
    parser.add_argument(
        "--tag",
        default="parallel",
        help="输出文件名标签，用于区分同一 checkpoint 的多次评测",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    name = Path(args.checkpoint).stem
    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    bounds = shard_bounds(args.trials_per_task, args.shards)

    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="osmesa",
        LP_NUM_THREADS=str(LLVMPIPE_THREADS),
        # 每个分片自己只用少量 BLAS 线程，避免 N 个进程互相超订 88 核。
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )

    procs: list[tuple[subprocess.Popen, Path, Path, tuple[int, int]]] = []
    started = time.time()
    for start, stop in bounds:
        json_path = out_dir / f"{name}_{args.tag}_t{start}-{stop}.json"
        log_path = out_dir / f"{name}_{args.tag}_t{start}-{stop}.log"
        command = [
            args.python, "-u", "-B", "eval_metaworld.py",
            "--checkpoint", args.checkpoint,
            "--features", args.features,
            "--main-vision-checkpoint", args.dino,
            "--task-ids", args.task_ids,
            "--trials-per-task", str(args.trials_per_task),
            "--trial-range", f"{start}:{stop}",
            "--execution-horizon", str(args.execution_horizon),
            "--horizon", str(args.horizon),
            "--direct-head", "auto",
            "--flow-samples", "1",
            "--device", "cuda",
            "--output-json", str(json_path),
        ]
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        procs.append((proc, json_path, log_path, (start, stop)))
        print(f"launched shard trials [{start},{stop}) pid={proc.pid} -> {log_path.name}",
              flush=True)

    failed: list[tuple[int, int]] = []
    for proc, _, log_path, span in procs:
        code = proc.wait()
        if code != 0:
            failed.append(span)
            print(f"FAIL shard {span} exit={code}; see {log_path}", file=sys.stderr)
    elapsed = time.time() - started
    if failed:
        print(f"aborting merge: {len(failed)} shard(s) failed: {failed}", file=sys.stderr)
        return 1

    records: list[dict] = []
    templates: list[dict] = []
    for _, json_path, _, _ in procs:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        records.extend(payload["trials"])
        templates.append(payload)

    seen = {(r["task_id"], r["trial"]) for r in records}
    if len(seen) != len(records):
        print("FAIL: duplicate (task_id, trial) across shards", file=sys.stderr)
        return 1

    per_task: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        per_task[record["task_id"]].append(record)

    merged = dict(templates[0])
    merged["trials"] = sorted(records, key=lambda r: (r["task_id"], r["trial"]))
    merged["completed_trials"] = len(records)
    merged["successes"] = sum(1 for r in records if r["success"])
    merged["success_rate"] = merged["successes"] / max(len(records), 1)
    merged["shards"] = [{"trial_range": f"{a}:{b}"} for _, _, _, (a, b) in procs]
    merged.pop("ci", None)  # 合并后不重算 bootstrap CI，避免伪造未计算的统计量

    merged_path = out_dir / f"{name}_{args.tag}_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    print()
    for task_id in sorted(per_task):
        rows = per_task[task_id]
        wins = sum(1 for r in rows if r["success"])
        print(f"task {task_id} ({rows[0]['task'][:40]}): {wins}/{len(rows)}")
    rate = 100.0 * merged["success_rate"]
    macro = 100.0 * sum(
        sum(1 for r in rows if r["success"]) / len(rows) for rows in per_task.values()
    ) / max(len(per_task), 1)
    print(f"CLOSED-LOOP SUCCESS: {merged['successes']}/{len(records)} = {rate:.1f}%")
    print(f"macro (per-task avg): {macro:.1f}% (n_tasks={len(per_task)})")
    print(f"merged results: {merged_path}")
    print(f"wall clock: {elapsed / 60.0:.1f} min across {args.shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
