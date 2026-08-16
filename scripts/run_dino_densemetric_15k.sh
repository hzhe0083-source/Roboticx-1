#!/usr/bin/env bash
# DINO-metric 15k 训练（2026-08-15 用户决策：DINO-main 接回 dense+metric）。
# dense evidence = DINO block11(g)/block23(d) 两帧 [d-2,d] patch（512 token，
# 1024 维）+ Δt；LanguageMetricField h_dim=1024、grid=16 从零训练；
# metric/relation 由动作 loss 联合微调（同 V-JEPA 联合协议 lr 1e-6 / 2e-5）。
# resume e7_dino_main_p35_2k.pt（global_step=2000）→ --steps 13000 ⇒ 共 15000 步。
# 2026-08-15 步时优化：--dino-feature-cache 预计算特征（在线编码占步时 84%，
# 缓存读把 ~9.4h 降到 ~2.5h；位级一致由 build_dino_feature_cache.py 验证）。
set -euo pipefail
cd "$(dirname "$0")/.."

DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ryan/.venvs/pytorch-gpu/bin/python -u -B train.py \
  --dino-main-vision --dino-dense-metric \
  --main-vision-checkpoint "$DINO" \
  --main-vision-encode-batch 32 \
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
  --resume checkpoints/e7_dino_main_p35_2k.pt \
  --steps 13000 \
  --save checkpoints/e7_dino_main_p35_densemetric_15k.pt \
  --save-every 2000 \
  2>&1 | tee logs/e7_dino_main_p35_densemetric_15k.log
