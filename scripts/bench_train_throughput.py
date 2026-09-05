"""测量训练 step 时间与 GPU 利用率，对比 microbatch / batch-size 配置。

背景（实测）：
  - 原始 run 1500->3000 步用了 3h40m = 409 步/小时 = 8.8 s/step
  - 梯度诊断时 GPU 显存只用 4.6GB / 33GB 预算（卡是 46GB L20）
  - 启动脚本用 --main-vision-encode-batch 8，而 train.py 默认是 16，
    帮助文本写明「ViT-L 16-GiB GPU 安全默认 16」——8 是按本地 8GB 笔记本定的
  - 该参数不进 exact_run_contract，改动不破坏续训契约
  - 数据侧不是瓶颈：LongTrajFramesDataset 做任务级预解码缓存，JPEG 只在任务
    切换时解一次，--task-locality-block-batches 64 让切换很稀疏，
    所以 --num-workers 0 是刻意选择（避免多份 5.3GB 缓存拷贝）

冻结塔在 no_grad 下按 microbatch 分块前向，改块大小只改分块方式，不改优化语义。

用法：
    python scripts/bench_train_throughput.py            # 跑全部配置
    python scripts/bench_train_throughput.py 18 64      # 只跑 batch=18 encode=64
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DINO = os.environ.get(
    "BENCH_DINO",
    "/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2."
    "lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors",
)
# Dataset family under test; defaults keep the original v1 measurement.
VA_DATA = os.environ.get("BENCH_VA_DATA", "data/hard2_peer_h6_p2_va_train_v1.pt")
WORLD_DATA = os.environ.get(
    "BENCH_WORLD_DATA", "data/hard2_peer_h6_p2_world_train_v1.pt"
)
SPLIT_MANIFEST = os.environ.get(
    "BENCH_SPLIT_MANIFEST", "data/hard2_peer_h6_p2_world_split_v1.json"
)
LONGTRAJ_DIR = os.environ.get("BENCH_LONGTRAJ_DIR", "")
DECODE_CACHE_TASKS = os.environ.get("BENCH_DECODE_CACHE_TASKS", "")
PY = os.environ.get("PY", "/opt/conda/bin/python")
STEPS = int(os.environ.get("BENCH_STEPS", "24"))
# 前若干步含 CUDA 图/autotune 预热与任务解码，不计入吞吐。
WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))

CONFIGS: list[tuple[int, int]] = [(18, 8), (18, 64), (36, 64)]
if len(sys.argv) > 2:
    CONFIGS = [(int(sys.argv[1]), int(sys.argv[2]))]

STEP_RE = re.compile(r"(?:^|\s)step=(\d+)\s")
CUDA_RE = re.compile(r"cuda=([0-9.]+)/([0-9.]+)MiB")


def sample_gpu(stop: threading.Event, out: list[int], used_mib: list[int]) -> None:
    # train.py's own ``cuda=X/YMiB`` line reports a budget subcategory, not the
    # true peak: it read 4606/32664 at batch 18 while batch 28 died at
    # 44.52/44.53 GiB.  Driver-level memory.used is what actually predicts OOM.
    while not stop.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--format=csv,noheader,nounits",
                 "--query-gpu=utilization.gpu,memory.used"],
                capture_output=True, text=True, timeout=5,
            )
            util, memory = result.stdout.strip().split(",")
            out.append(int(util))
            used_mib.append(int(memory))
        except Exception:
            pass
        time.sleep(0.5)


def run(batch: int, encode_batch: int) -> dict:
    command = [
        PY, "-u", "-B", "train.py",
        "--va-data", VA_DATA,
        "--world-data", WORLD_DATA,
        "--visual-world-supervision",
        "--world-split-manifest", SPLIT_MANIFEST,
        "--va-world-mode", "peer_sync_h6",
        "--planning-stride", "2", "--control-stride", "2",
        "--wam4va", "--wmrm-inject", "all", "--wmrm-target", "dino",
        "--wmrm-adep-weight", "0", "--wmrm-cycle-steps", "2",
        "--wmrm-world-weight", "1.0",
        "--dino-main-vision", "--dino-dense-metric",
        "--main-vision-checkpoint", DINO,
        "--main-vision-grid", "16", "--main-vision-frames", "4",
        "--main-vision-temporal", "--main-vision-temporal-scale", "1.0",
        "--main-vision-encode-batch", str(encode_batch),
        "--metric-geometry-inject", "--wmrm-map-size", "16",
        "--wmrm-map-channels", "1024", "--wmrm-world-grid", "16",
        "--wmrm-predictor", "st_blocks", "--wmrm-predictor-depth", "6",
        "--wmrm-predictor-width", "384", "--wmrm-predictor-heads", "12",
        "--single-task", "--task-sampling", "balanced",
        "--task-locality-block-batches", "64", "--batch-size", str(batch),
        "--sequence-length", "4", "--min-sequence-length", "4",
        "--num-workers", "0", "--lr", "0.0001", "--seed", "0", "--device", "cuda",
        "--feature-autocast-bf16", "--va-layers", "8",
        "--va-attention-backend", "auto", "--flow-cond", "adaln",
        "--flow-layers", "6", "--flow-steps", "8",
        "--flow-prefix-steps", "2", "--flow-prefix-weight", "1.0",
        "--flow-tail-weight", "0.036",
        "--mtvj-train-metric-head", "--lr-mtvj-metric-head", "0.0003",
        "--mtvj-train-relation", "--lr-mtvj-relation", "0.00002",
        "--mtvj-visual-aux-every", "10", "--mtvj-visual-aux-batch", "8",
        "--steps", str(STEPS), "--save-every", "0",
        "--save", f"/tmp/bench_b{batch}_e{encode_batch}.pt",
    ]
    if LONGTRAJ_DIR:
        command += ["--longtraj-dir", LONGTRAJ_DIR]
    if DECODE_CACHE_TASKS:
        command += ["--longtraj-decode-cache-tasks", DECODE_CACHE_TASKS]
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        MUJOCO_GL="osmesa", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )

    stop = threading.Event()
    utils: list[int] = []
    used: list[int] = []
    watcher = threading.Thread(target=sample_gpu, args=(stop, utils, used), daemon=True)

    stamps: dict[int, float] = {}
    peak_mib = 0.0
    total_mib = 0.0
    oom = False
    raw_log = ROOT / "logs" / f"bench_train_b{batch}_e{encode_batch}.log"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    watcher.start()
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    with raw_log.open("w", encoding="utf-8") as sink:
        for line in proc.stdout:
            sink.write(line)
            if "out of memory" in line.lower():
                oom = True
            match = STEP_RE.search(line)
            if match:
                stamps[int(match.group(1))] = time.perf_counter()
                found = CUDA_RE.search(line)
                if found:
                    peak_mib = max(peak_mib, float(found.group(1)))
                    total_mib = float(found.group(2))
    code = proc.wait()
    stop.set()
    watcher.join(timeout=2)

    steps = sorted(stamps)
    measured = [s for s in steps if s > WARMUP]
    if len(measured) >= 2:
        span = stamps[measured[-1]] - stamps[measured[0]]
        per_step = span / (len(measured) - 1)
    else:
        per_step = float("nan")
    busy = [u for u in utils if u > 0]
    return {
        "batch": batch,
        "encode_batch": encode_batch,
        "exit_code": code,
        "oom": oom,
        "steps_seen": len(steps),
        "s_per_step": per_step,
        "steps_per_hour": (3600.0 / per_step) if per_step == per_step else None,
        "gpu_util_mean": (sum(utils) / len(utils)) if utils else None,
        "gpu_util_p90": (sorted(utils)[int(0.9 * (len(utils) - 1))] if utils else None),
        "gpu_util_nonzero_frac": (len(busy) / len(utils)) if utils else None,
        "cuda_peak_mib": peak_mib,
        "cuda_budget_mib": total_mib,
        "driver_peak_used_mib": max(used) if used else None,
        "raw_log": str(raw_log),
    }


def main() -> int:
    results = []
    for batch, encode_batch in CONFIGS:
        print(f"\n{'=' * 62}\nbatch={batch} encode_batch={encode_batch}\n{'=' * 62}",
              flush=True)
        record = run(batch, encode_batch)
        results.append(record)
        for key, value in record.items():
            print(f"  {key:<24}= {value}", flush=True)

    print(f"\n{'=' * 62}\nsummary\n{'=' * 62}")
    print(f"{'batch':>6} {'encode':>7} {'s/step':>9} {'steps/h':>9} "
          f"{'gpu%':>6} {'cuda MiB':>10} {'drv MiB':>9}")
    baseline = None
    for record in results:
        if record["s_per_step"] != record["s_per_step"]:
            print(f"{record['batch']:>6} {record['encode_batch']:>7}   FAILED "
                  f"(exit={record['exit_code']} oom={record['oom']})")
            continue
        if baseline is None:
            baseline = record["s_per_step"] / record["batch"]
        per_sample = record["s_per_step"] / record["batch"]
        print(f"{record['batch']:>6} {record['encode_batch']:>7} "
              f"{record['s_per_step']:>9.2f} {record['steps_per_hour']:>9.0f} "
              f"{record['gpu_util_mean']:>6.1f} {record['cuda_peak_mib']:>10.0f} "
              f"{record['driver_peak_used_mib'] or 0:>9.0f}"
              f"   per-sample speedup vs first: {baseline / per_sample:.2f}x")

    out = ROOT / "logs" / "bench_train_throughput.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
