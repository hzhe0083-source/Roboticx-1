#!/usr/bin/env bash
# 在训练过的权重上复测 VA/World 梯度分解，用于区分「world→action 敏感度本来是 0」
# 与「只是初始化时小、训练后会长大」。
set -euo pipefail
cd /root/private_data/ORA0

CKPT=${1:-checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18_s3000.pt}
NSTEP=${2:-8}

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  MUJOCO_GL=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /opt/conda/bin/python -u -B scripts/diag_peer_gradient_balance.py "$NSTEP" "$CKPT"
