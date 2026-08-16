#!/usr/bin/env bash
# DINO-metric 中间 checkpoint 评测冒烟（1 trial，验证 eval 新路径端到端）。
# 用法：bash scripts/run_dino_densemetric_eval_smoke.sh [CKPT]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:-checkpoints/e7_dino_main_p35_densemetric_15k.pt}
FEATURES=data/metaworld_longtraj_windows_h48_dino35_clean.pt
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --task-ids 35 --trials-per-task 1 \
  --main-vision-checkpoint "$DINO" \
  --device cuda \
  2>&1 | tee logs/e7_dino_main_p35_densemetric_eval_smoke.log
