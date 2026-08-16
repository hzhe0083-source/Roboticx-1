#!/usr/bin/env bash
# DINO-metric grid16 闭环评测（2026-08-16：主视觉全 16×16 patch，1024 token）。
# eval 从 ckpt config 读 main_vision_grid=16，在线编码同构（identity 池化）。
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=checkpoints/e7_dino_main_p35_dm_grid16_15k.pt
FEATURES=data/metaworld_longtraj_windows_h48_dino35_clean.pt
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --task-ids 35 --trials-per-task 10 \
  --main-vision-checkpoint "$DINO" \
  --device cuda \
  2>&1 | tee logs/e7_dino_main_p35_dm_grid16_15k_closedloop_1x10.log
