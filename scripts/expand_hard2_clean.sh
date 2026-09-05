#!/usr/bin/env bash
# hard2 干净示范扩产（--no-perturb），与 recovery 扩产并行。
#
# 为什么单独跑：recovery 模式要求每集至少一次扰动事件，而扰动有 5% 触发门，实测
# 约 4/5 的采集被 "skip nominal episode" 丢弃——被丢掉的正是完整的成功专家轨迹。
# 干净模式接受率接近 100%，同样的 CPU 时间能多拿约 5 倍示范。
#
# 两份数据用途不同也互补：recovery 教"被扰动后怎么恢复"，clean 教"标准执行"。
# build_longtraj_features.py 支持多个 --input，可以一起切窗。
#
# seed 段与 recovery 扩产（40000+）和 eval50（35000-35049）都不重叠。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
NORM_REF=${NORM_REF:-data/longtraj_normalization_ref.pt}
SHARDS=${SHARDS:-12}
PER_SHARD=${PER_SHARD:-10}
SEED_BASE=${SEED_BASE:-50000}
OUTDIR=data/expand_clean
TASKS=(assembly-v3 door-unlock-v3)

[[ -f "$NORM_REF" ]] || { echo "missing normalization ref: $NORM_REF" >&2; exit 1; }
mkdir -p "$OUTDIR" logs/expand_clean

echo "=== hard2 干净示范扩产开始 $(date '+%F %T') ==="
echo "每任务 ${SHARDS} 分片 x ${PER_SHARD} seed = $(( SHARDS * PER_SHARD )) 条"
for task in "${TASKS[@]}"; do
  for ((i = 0; i < SHARDS; i++)); do
    start=$(( SEED_BASE + i * PER_SHARD ))
    seeds=()
    for ((s = start; s < start + PER_SHARD; s++)); do seeds+=("$s"); done
    out="$OUTDIR/metaworld_longtraj_${task}_clean_v2_shard${i}.pt"
    [[ -f "$out" ]] && { echo "skip existing $out"; continue; }
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
      MUJOCO_GL=osmesa LP_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -u -B scripts/collect_long_trajectories.py \
      --task "$task" --no-perturb \
      --episode-seeds "${seeds[@]}" \
      --normalization-ref "$NORM_REF" \
      --output "$out" \
      > "logs/expand_clean/${task}_shard${i}.log" 2>&1 &
  done
done
echo "已启动 $(jobs -rp | wc -l) 个干净采集进程，等待..."
wait
echo "=== 干净扩产结束 $(date '+%F %T') ==="
"$PY" - "$OUTDIR" <<'PY'
import sys, pathlib, torch
outdir = pathlib.Path(sys.argv[1])
for task in ("assembly-v3", "door-unlock-v3"):
    shards = sorted(outdir.glob(f"metaworld_longtraj_{task}_clean_v2_shard*.pt"))
    total = sum(
        len(torch.load(p, map_location="cpu", weights_only=False)["episodes"])
        for p in shards
    )
    print(f"{task}: {len(shards)} 个分片, {total} 条干净示范")
PY
