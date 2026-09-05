#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/private_data/ORA0_next
PY=/opt/conda/bin/python
PREP_PY=/root/libero_spatial_ora0_v1/venv/bin/python
DATA=/root/libero_spatial_ora0_v1/libero_spatial_h15p4_t4_v1.pt
LANG=/root/libero_spatial_ora0_v1/libero_spatial_qwen35_2b_l0_14_mean10_14_v1.pt
SAVE=/root/ora0_ckpts/libero_spatial_from_s3224_h15p4_allfix_e50_b32_v1.pt
LOG="$ROOT/logs/libero_spatial_from_s3224_h15p4_allfix_e50_b32_v1.log"

cd "$ROOT"
if [[ ! -f "$DATA" ]]; then
  "$PREP_PY" libero_train.py prepare --data "$DATA" --language-reference "$LANG"
fi
"$PY" libero_train.py preflight --data "$DATA"
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u libero_train.py train \
  --data "$DATA" \
  --save "$SAVE" \
  --epochs 50 \
  --batch-size 32 \
  --gpus 2 \
  --prev-dropout 1 \
  --lr 1e-5 \
  --lr-new 3e-5 \
  --encode-batch 16 \
  2>&1 | tee "$LOG"
