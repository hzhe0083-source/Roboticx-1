#!/usr/bin/env bash
# Replace four unrecoverable pinned-init shards with deterministic successful-init sampling.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
ROOT=${OUT_ROOT:-/root/ora0_all49_expand60_v1}
NORM_REF=${NORM_REF:-/root/private_data/ORA0/data/longtraj_normalization_ref.pt}
CPUSET=${CPUSET:-0-31}
LD_PRELOAD_PATH=${LD_PRELOAD_PATH:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}

tasks=(disassemble-v3 faucet-open-v3 peg-insert-side-v3 stick-pull-v3)
shards=(2 1 0 0)
rng_seeds=(1900122 1900201 1900280 1900430)
pids=()

for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  shard=${shards[$i]}
  rng_seed=${rng_seeds[$i]}
  task_dir=$ROOT/shards/$task
  log_dir=$ROOT/logs/$task
  out=$task_dir/metaworld_longtraj_${task}_recovery_v2_shard${shard}.pt
  tmp=$task_dir/.$(basename "$out").tmp
  [[ ! -e "$tmp" ]] || { printf 'ERROR: stale temporary %s\n' "$tmp" >&2; exit 1; }
  if [[ -f "$out" ]]; then
    printf '[skip] %s already exists\n' "$out"
    continue
  fi
  mkdir -p "$task_dir" "$log_dir"
  log=$log_dir/shard${shard}_fallback1.log
  nice -n 10 ionice -c 3 taskset -c "$CPUSET" env \
    CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa LP_NUM_THREADS=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    LD_PRELOAD="$LD_PRELOAD_PATH" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -u -B scripts/collect_long_trajectories.py \
    --task "$task" --episodes 10 --seed "$rng_seed" --force-perturb \
    --normalization-ref "$NORM_REF" --output "$out" >"$log" 2>&1 &
  pids+=("$!")
  printf '[launch] task=%s shard=%s pid=%s log=%s\n' \
    "$task" "$shard" "${pids[-1]}" "$log"
done

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done
(( failures == 0 )) || { printf 'ERROR: %d fallback collectors failed\n' "$failures" >&2; exit 1; }
printf '[ok] four fallback shards completed\n'
