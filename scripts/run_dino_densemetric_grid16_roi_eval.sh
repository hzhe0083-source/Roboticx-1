#!/usr/bin/env bash
# DINO-metric grid16 + ROI 精修闭环评测（2026-08-16）：粗 metric 头 +
# refine_metric_roi_positions_dino 有界残差融合，10 次 trial 对照。
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=checkpoints/e7_dino_main_p35_dm_grid16_15k.pt
ROI=checkpoints/dino_metric_roi_head_p35_1k.pt
FEATURES=data/metaworld_longtraj_windows_h48_dino35_clean.pt
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --task-ids 35 --trials-per-task 10 \
  --main-vision-checkpoint "$DINO" \
  --dino-roi-checkpoint "$ROI" \
  --dino-roi-alpha 1.0 \
  --device cuda \
  2>&1 | tee logs/e7_dino_main_p35_dm_grid16_15k_roi_closedloop_1x10.log
