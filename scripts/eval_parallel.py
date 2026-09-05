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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mt50_difficulty import summarize_mt50_benchmark_trials
from va_compound.statistics import binomial_wilson_ci, macro_bootstrap_ci

DEFAULT_DINO = "/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors"
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
    parser.add_argument("--language-features", default=None)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--shard-axis", choices=("trials", "tasks"), default="trials")
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--episode-seed-base", type=int, default=4042)
    parser.add_argument("--execution-horizon", type=int, default=15)
    parser.add_argument("--allow-execution-horizon-ablation", action="store_true")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--mt50-benchmark", action="store_true")
    parser.add_argument("--align-init", action="store_true")
    parser.add_argument("--peer-world-off", action="store_true")
    parser.add_argument("--dagger-output-dir", default=None)
    parser.add_argument("--dagger-takeover-min", type=int, default=45)
    parser.add_argument("--dagger-takeover-max", type=int, default=120)
    parser.add_argument("--dagger-prefix-keep", type=int, default=45)
    parser.add_argument("--gpus", default=os.environ.get("EVAL_GPUS", "0,1"))
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
    if args.shard_axis == "tasks":
        task_ids = (
            [int(token) for token in args.task_ids.split(",")]
            if args.task_ids is not None
            else list(range(50))
        )
        bounds = shard_bounds(len(task_ids), args.shards)
        jobs = [
            (0, args.trials_per_task, task_ids[start:stop], f"k{start}-{stop}")
            for start, stop in bounds
        ]
    else:
        bounds = shard_bounds(args.trials_per_task, args.shards)
        selected = (
            None
            if args.task_ids is None
            else [int(token) for token in args.task_ids.split(",")]
        )
        jobs = [(start, stop, selected, f"t{start}-{stop}") for start, stop in bounds]
    gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one CUDA device index")

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

    procs: list[tuple[subprocess.Popen, Path, Path, dict]] = []
    started = time.time()
    for shard_index, (start, stop, selected, suffix) in enumerate(jobs):
        json_path = out_dir / f"{name}_{args.tag}_{suffix}.json"
        log_path = out_dir / f"{name}_{args.tag}_{suffix}.log"
        command = [
            args.python, "-u", "-B", "eval_metaworld.py",
            "--checkpoint", args.checkpoint,
            "--features", args.features,
            "--main-vision-checkpoint", args.dino,
            "--trials-per-task", str(args.trials_per_task),
            "--episode-seed-base", str(args.episode_seed_base),
            "--trial-range", f"{start}:{stop}",
            "--execution-horizon", str(args.execution_horizon),
            "--horizon", str(args.horizon),
            "--direct-head", "auto",
            "--flow-samples", "1",
            "--device", "cuda",
            "--output-json", str(json_path),
        ]
        if args.language_features is not None:
            command.extend(("--language-features", args.language_features))
        if args.allow_execution_horizon_ablation:
            command.append("--allow-execution-horizon-ablation")
        if selected is not None:
            command.extend(("--task-ids", ",".join(map(str, selected))))
        if args.mt50_benchmark and args.shard_axis == "trials":
            command.append("--mt50-benchmark")
        if args.align_init:
            command.append("--align-init")
        if args.peer_world_off:
            command.append("--peer-world-off")
        if args.dagger_output_dir is not None:
            command.extend(
                (
                    "--dagger-output-dir", args.dagger_output_dir,
                    "--dagger-takeover-min", str(args.dagger_takeover_min),
                    "--dagger-takeover-max", str(args.dagger_takeover_max),
                    "--dagger-prefix-keep", str(args.dagger_prefix_keep),
                )
            )
        shard_env = dict(env)
        shard_env["CUDA_VISIBLE_DEVICES"] = gpus[shard_index % len(gpus)]
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=shard_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        shard = {"trial_range": f"{start}:{stop}"}
        if selected is not None:
            shard["task_ids"] = selected
        procs.append((proc, json_path, log_path, shard))
        print(f"launched shard {shard} pid={proc.pid} -> {log_path.name}", flush=True)

    failed: list[dict] = []
    for proc, _, log_path, shard in procs:
        code = proc.wait()
        if code != 0:
            failed.append(shard)
            print(f"FAIL shard {shard} exit={code}; see {log_path}", file=sys.stderr)
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
    expected_task_ids = {
        int(task_id) for template in templates for task_id in template["task_ids"]
    }
    expected = {
        (int(task_id), trial)
        for task_id in expected_task_ids
        for trial in range(args.trials_per_task)
    }
    if seen != expected:
        print(
            f"FAIL: incomplete trial grid missing={len(expected - seen)} "
            f"extra={len(seen - expected)}",
            file=sys.stderr,
        )
        return 1

    per_task: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        per_task[record["task_id"]].append(record)

    merged = dict(templates[0])
    merged["task_ids"] = sorted(expected_task_ids)
    merged["trials"] = sorted(records, key=lambda r: (r["task_id"], r["trial"]))
    merged["completed_trials"] = len(records)
    merged["successes"] = sum(1 for r in records if r["success"])
    merged["success_rate"] = merged["successes"] / max(len(records), 1)
    merged["shards"] = [shard for _, _, _, shard in procs]
    successes = [float(bool(row["success"])) for row in records]
    task_ids = [int(row["task_id"]) for row in records]
    if len(per_task) == 1:
        estimate, low, high = binomial_wilson_ci(
            merged["successes"], len(records)
        )
        ci_kind = "wilson"
    else:
        estimate, low, high = macro_bootstrap_ci(
            successes, task_ids, n_boot=2000, seed=0
        )
        ci_kind = "task_bootstrap"
    merged["ci"] = {
        "kind": ci_kind,
        "estimate": estimate,
        "low_95": low,
        "high_95": high,
    }
    merged["mt50_benchmark"] = summarize_mt50_benchmark_trials(records)
    if args.mt50_benchmark and not merged["mt50_benchmark"]["complete_mt50"]:
        print("FAIL: merged result is not complete MT50", file=sys.stderr)
        return 1

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
    benchmark = merged["mt50_benchmark"]
    if benchmark["complete_mt50"]:
        for group, values in benchmark["groups"].items():
            print(f"{group}: {100.0 * values['success_rate']:.1f}%")
        print(f"EVOMIND FOUR-TIER AVERAGE: {100.0 * benchmark['bucket_average']:.1f}%")
    print(f"merged results: {merged_path}")
    print(f"wall clock: {elapsed / 60.0:.1f} min across {args.shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
