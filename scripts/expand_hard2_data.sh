#!/usr/bin/env bash
# hard2 数据扩产：每任务 +120 条脚本专家示范（30 -> 150），按 pinned seed 分片并行。
#
# 为什么可以并行：采集是纯 CPU（MuJoCo 仿真 + OSMesa 软件渲染），不碰 GPU。
# 实测 LP_NUM_THREADS=1 时 env.render()=188ms，=8 时 80ms。24 个分片各锁 1
# 线程只会吃掉 24/88 核。每进程 4 个 llvmpipe 线程 ≈ 96 个渲染线程，贴着 88 核。
#
# 为什么用 pinned seed：--episode-seeds 固定 MetaWorld init，分片间 seed 不重叠即
# 保证无重复 init。避开 35000-35049（eval50 的 init；采集器默认也会拒绝）。
# door-unlock 现有 30 条的 seed 全部 >1.4e8，assembly 的 v1 数据根本没记 seed，
# 所以 40000+ 这段既可读又与两者都不冲突。
#
# 归一化：q01/q99 必须继承、禁止单独算。默认来源是 9.8GB 的 fullframe 文件（服务器
# 上没有），这里改为指向从现有 hard2 数据抽出的小参考文件——已验证两个现有 raw
# 文件的 normalization 逐位一致，所以继承来源等价。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
NORM_REF=${NORM_REF:-data/longtraj_normalization_ref.pt}
SHARDS=${SHARDS:-12}
PER_SHARD=${PER_SHARD:-10}
SEED_BASE=${SEED_BASE:-40000}
OUTDIR=data/expand
TASKS=(assembly-v3 door-unlock-v3)

[[ -f "$NORM_REF" ]] || { echo "missing normalization ref: $NORM_REF" >&2; exit 1; }
mkdir -p "$OUTDIR" logs/expand

launch_shard(){
  local task=$1 shard=$2
  local start=$(( SEED_BASE + shard * PER_SHARD ))
  local seeds=()
  for ((s = start; s < start + PER_SHARD; s++)); do seeds+=("$s"); done
  local out="$OUTDIR/metaworld_longtraj_${task}_recovery_v2_shard${shard}.pt"
  local log="logs/expand/${task}_shard${shard}.log"
  if [[ -f "$out" ]]; then echo "skip existing $out"; return 0; fi
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa LP_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -u -B scripts/collect_long_trajectories.py \
    --task "$task" \
    --episode-seeds "${seeds[@]}" \
    --normalization-ref "$NORM_REF" \
    --output "$out" > "$log" 2>&1 &
}

echo "=== hard2 扩产开始 $(date '+%F %T') ==="
echo "每任务 ${SHARDS} 分片 x ${PER_SHARD} seed = $(( SHARDS * PER_SHARD )) 条新示范"
for task in "${TASKS[@]}"; do
  for ((i = 0; i < SHARDS; i++)); do launch_shard "$task" "$i"; done
done
echo "已启动 $(jobs -rp | wc -l) 个采集进程，等待全部完成..."
wait
echo "=== 全部分片结束 $(date '+%F %T') ==="

for task in "${TASKS[@]}"; do
  ok=0; missing=0; total=0
  for ((i = 0; i < SHARDS; i++)); do
    out="$OUTDIR/metaworld_longtraj_${task}_recovery_v2_shard${i}.pt"
    if [[ -f "$out" ]]; then ok=$((ok + 1)); else missing=$((missing + 1)); fi
  done
  total=$("$PY" - "$OUTDIR" "$task" <<'PY'
import sys, pathlib, torch
outdir, task = pathlib.Path(sys.argv[1]), sys.argv[2]
n = 0
for p in sorted(outdir.glob(f"metaworld_longtraj_{task}_recovery_v2_shard*.pt")):
    n += len(torch.load(p, map_location="cpu", weights_only=False)["episodes"])
print(n)
PY
)
  echo "$task: 分片成功 ${ok}/${SHARDS}（缺 ${missing}），新增 episode 合计 ${total}"
done
