#!/usr/bin/env bash
# DINO-metric grid16 + 视觉辅助训练（2026-08-16，P0 探针 + 用户决策：真正训练 MT-VJ）。
# - 主视觉全 16×16 patch（grid=16，1024 token/决策，无池化）；
# - dense+metric 保留；
# - --mtvj-visual-aux-every 50：仿真真值视觉辅助 loss（_dino_visual_aux_loss，
#   hinge+pos+offset+BCE(vis)，只反传 metric head）——MT-VJ 高清头的真正训练信号。
set -euo pipefail
cd "$(dirname "$0")/.."

DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B train.py \
  --dino-main-vision --dino-dense-metric \
  --main-vision-checkpoint "$DINO" \
  --main-vision-encode-batch 32 \
  --main-vision-grid 16 \
  --dino-feature-cache data/dino35_feature_cache \
  --data data/metaworld_longtraj_windows_h48_dino35_clean.pt \
  --single-task --task-sampling weighted \
  --batch-size 16 --sequence-length 4 --min-sequence-length 4 \
  --lr 0.0001 --seed 0 --device cuda \
  --flow-steps 8 --flow-cond adaln --flow-layers 6 \
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
  --va-layers 8 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 3e-4 \
  --mtvj-train-relation --lr-mtvj-relation 2e-5 \
  --mtvj-visual-aux-every 50 --mtvj-visual-aux-batch 8 \
  --mtvj-visual-aux-loc-lambda 1.0 --mtvj-visual-aux-vis-lambda 0.5 \
  --steps 15000 \
  --save checkpoints/e7_dino_main_p35_dm_grid16_15k.pt \
  --save-every 2000 \
  2>&1 | tee logs/e7_dino_main_p35_dm_grid16_15k.log
