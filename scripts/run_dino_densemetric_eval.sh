#!/usr/bin/env bash
# DINO-metric 15k 闭环评测（2026-08-15：DINO-main + dense + metric 全栈）。
# metric head/relation encoder 从主 checkpoint 严格重建（无外部 metric 文件）。
# 用法：训练完成后 bash scripts/run_dino_densemetric_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=checkpoints/e7_dino_main_p35_densemetric_15k.pt
FEATURES=data/metaworld_longtraj_windows_h48_dino35_clean.pt
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --task-ids 35 --trials-per-task 10 \
  --main-vision-checkpoint "$DINO" \
  --device cuda \
  2>&1 | tee logs/e7_dino_main_p35_densemetric_15k_closedloop_1x10.log
