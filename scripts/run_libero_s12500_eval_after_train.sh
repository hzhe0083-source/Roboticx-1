#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${LIBERO_TRAIN_PID:-46363}"
ROOT="/root/private_data/ORA0_next"
PYTHON="/root/libero_spatial_ora0_v1/venv/bin/python"
CHECKPOINT="/root/ora0_ckpts/libero_spatial_from_s3224_h15p15_va_wm_pcgrad_slotidfix_e50_b32_v2.pt"
DATA="/root/libero_spatial_ora0_v1/libero_spatial_h15p15_t4_v1.pt"
DINO="/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors"
OUT="/root/evo1_eval/libero_h15_slotidfix_s12500_fixed3"

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 20; done
mkdir -p "$OUT"
cd "$ROOT"

run_shard() {
  local gpu="$1" tasks="$2" name="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/root/libero_official_ora0:$ROOT" \
    LIBERO_CONFIG_PATH=/root/libero_spatial_ora0_v1/config \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    "$PYTHON" -u eval_libero_closedloop.py \
    --checkpoint "$CHECKPOINT" --data "$DATA" \
    --main-vision-checkpoint "$DINO" --task-ids "$tasks" \
    --trials-per-task 3 --horizon 350 --flow-steps 8 \
    --settle-steps 10 --memory-reset-every 4 --seed 1000 \
    --device cuda --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
}

run_shard 0 0,1,2,3,4 shard0 &
pid0=$!
run_shard 1 5,6,7,8,9 shard1 &
pid1=$!
wait "$pid0" "$pid1"
