#!/usr/bin/env bash
# DINO ROI 精修头训练（2026-08-16）：GT+抖动粗定位 → 动态裁剪 → 冻结 DINO →
# grid-16 LanguageMetricField，hinge+pos+offset+BCE @224px。~20 分钟。
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B scripts/train_metric_roi_dino.py \
  --task peg-insert-side-v3 \
  --steps 1000 \
  --batch 16 \
  --lr 3e-4 \
  --jitter-px 10.0 \
  --save checkpoints/dino_metric_roi_head_p35_1k.pt \
  2>&1 | tee logs/dino_metric_roi_head_p35_1k.log
