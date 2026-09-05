#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/libero_4suite_ora0_v1/code
PY=/root/libero_4suite_ora0_v1/venv/bin/python
DATA_ROOT=/root/libero_spatial_ora0_v1
QWEN=/root/models/Qwen3.5-2B
DATA="$DATA_ROOT/libero_4suite_h50p15_t4_dualview5_maskedtail_v2.pt"
SAVE=/root/ora0_ckpts/libero_4suite_dualview5_scratch_h50p15_twostage_e50_b32_grouped_v4.pt
LOG="$DATA_ROOT/logs/libero_4suite_dualview5_scratch_h50p15_twostage_e50_b32_grouped_v4.log"
ENCODE_BATCH=${ENCODE_BATCH:-32}

mkdir -p "$DATA_ROOT/logs" /root/ora0_ckpts
cd "$ROOT"
export PYTHONPATH="/root/libero_official_ora0:$ROOT"
export LIBERO_CONFIG_PATH=/root/libero_spatial_ora0_v1/config
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

common=(
  --data "$DATA"
  --longtraj "$DATA_ROOT/longtraj"
  --hdf5-dir "$DATA_ROOT/datasets"
  --suites libero_spatial,libero_object,libero_goal,libero_10
  --qwen "$QWEN"
  --dino /root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors
  --save "$SAVE"
  --epochs 50 --stage1-steps 8000
  --batch-size 32 --mixed-tasks 4 --anchor-fraction 0.25
  --lr 0.00001 --lr-new 0.00003 --lr-qwen 0.000001 --lr-dino 0.000001
  --qwen-lora-rank 16 --prev-dropout 1 --encode-batch "$ENCODE_BATCH"
  --save-every 500 --gpus 2
  --va-last3-cross-attn --dino-qwen-cross-modal-bridge
)

if [[ ! -f "$DATA" ]]; then
  "$PY" libero_train.py prepare "${common[@]}"
fi
"$PY" libero_train.py preflight "${common[@]}"

resume=()
[[ -f "$SAVE" ]] && resume=(--resume "$SAVE")
exec "$PY" -u libero_train.py train "${common[@]}" "${resume[@]}" 2>&1 | tee -a "$LOG"
