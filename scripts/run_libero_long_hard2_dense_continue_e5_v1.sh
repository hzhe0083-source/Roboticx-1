#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/libero_hard2_ora0_v1/code
PY=/root/libero_4suite_ora0_v1/venv/bin/python
DATA_ROOT=/root/libero_spatial_ora0_v1
DATA="$DATA_ROOT/libero_10_hard2_t3_t4_dualview5_h50p15_worldh15_va1024_qwen08_last6_denseall_v7.pt"
SOURCE=/root/ora0_ckpts/libero_10_hard2_t3_t4_dualview5_worldh15_va1024_qwen08_last6_scratch_h50p15_e50_b16_v7.pt
SAVE=/root/ora0_ckpts/libero_10_hard2_denseall_worldh15_qwen08_last6_continue_e5_b16_v8.pt
LOG="$DATA_ROOT/logs/libero_10_hard2_denseall_worldh15_qwen08_last6_continue_e5_b16_v8.log"

mkdir -p "$DATA_ROOT/logs" /root/ora0_ckpts
cd "$ROOT"
export PYTHONPATH="/root/libero_official_ora0:$ROOT"
export LIBERO_CONFIG_PATH="$DATA_ROOT/config"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

common=(
  --data "$DATA" --longtraj "$DATA_ROOT/longtraj"
  --hdf5-dir "$DATA_ROOT/datasets" --suites libero_10 --local-task-ids 3,4
  --qwen /root/models/Qwen3.5-0.8B
  --dino /root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors
  --save "$SAVE" --dense-windows
  --epochs 5 --stage1-steps 0
  --batch-size 16 --mixed-tasks 2 --anchor-fraction 0
  --lr 0.00001 --lr-new 0.00003 --lr-qwen 0.000001 --lr-dino 0.000001
  --prev-dropout 1 --encode-batch 16 --save-every 1000 --gpus 2
  --va-last3-cross-attn --dino-qwen-cross-modal-bridge
)

[[ -f "$DATA" ]] || "$PY" libero_train.py prepare "${common[@]}"
resume=(--resume-weights "$SOURCE")
[[ -f "$SAVE" ]] && resume=(--resume "$SAVE")
"$PY" libero_train.py preflight "${common[@]}" "${resume[@]}"
exec "$PY" -u libero_train.py train "${common[@]}" "${resume[@]}" 2>&1 | tee -a "$LOG"
