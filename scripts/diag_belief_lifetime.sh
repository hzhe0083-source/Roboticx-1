#!/usr/bin/env bash
# belief 寿命判定实验（零训练代价，同一 checkpoint、同一组 seed）。
#
# 基线 A（已知，重复两次均为 3/20 = 15%）：--world-reset-every 4
#   每 4 个决策点清整个 WAMState，belief 和 world_map 一起没了。一集 250 个决策点
#   即每集清 62 次，belief 的有效寿命是 4 个决策点（占一集 1.6%）。
#
# 实验 B：--world-reset-every 0 --world-map-reset-every 1
#   belief 跨整集存活；world_map 每个决策点重锚回真实 DINO 帧，开环深度从 2000 次
#   propose 截断到 8 次（单决策点的 8 个 stage）。
#
# 三种可能结论：
#   B > 15%  belief 记忆有价值，之前被 --world-reset-every 连带擦掉了
#   B ≈ 15%  belief 即使保留也没学到东西 → 必须先修 stage_embed 累积与 stage 监督权重
#   B 发散   belief 自身无界（实测 stage_embed 纯累积项达初始幅度 504 倍）→ 同上，
#            且证明 --world-reset-every 是在给架构缺口打补丁
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
DINO=${DINO:-/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors}
CKPT=${1:-checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18_s3000.pt}
FEATURES=${2:-data/hard2_peer_h6_p2_eval_v1.pt}
TAG=${3:-belief_lifetime}

[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$FEATURES" ]] || { echo "missing features: $FEATURES" >&2; exit 1; }
[[ -f "$DINO" ]] || { echo "missing DINO: $DINO" >&2; exit 1; }

mkdir -p logs
LOG=logs/${TAG}.log
JSON=logs/${TAG}.json

echo "=== arm B: belief 跨整集存活 + map 每决策点重锚  START $(date '+%F %T') ==="
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  MUJOCO_GL=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --main-vision-checkpoint "$DINO" \
  --task-ids 0,16 \
  --trials-per-task 10 \
  --execution-horizon 2 \
  --horizon 500 \
  --direct-head auto \
  --flow-samples 1 \
  --device cuda \
  --world-reset-every 0 \
  --world-map-reset-every 1 \
  --output-json "$JSON" \
  2>&1 | tee "$LOG"
echo "=== arm B DONE $(date '+%F %T') ==="
