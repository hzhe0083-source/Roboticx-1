#!/usr/bin/env bash
# DINO-metric grid16 训练（2026-08-16，P0 探针证据驱动：DINO patch 几何线性
# 可读 5.9-9.1px，8-13px 门通过；根因 = 主视觉 8×8 池化碾掉定位信息）。
# 变更：--main-vision-grid 16（全 16×16 patch，1024 token/决策）替代 8×8 池化；
# dense+metric 不变；从零训练 15000 步（缓存已含全 patch 特征，无需重编码）。
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
  --single-task --task-sampling balanced \
  --batch-size 16 --sequence-length 4 --min-sequence-length 4 \
  --lr 0.0001 --seed 0 --device cuda \
  --flow-steps 8 --flow-cond adaln --flow-layers 6 \
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
  --va-layers 8 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 1e-6 \
  --mtvj-train-relation --lr-mtvj-relation 2e-5 \
  --steps 15000 \
  --save checkpoints/e7_dino_main_p35_dm_grid16_15k.pt \
  --save-every 2000 \
  2>&1 | tee logs/e7_dino_main_p35_dm_grid16_15k.log
