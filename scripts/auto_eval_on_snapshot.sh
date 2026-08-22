#!/usr/bin/env bash
set -euo pipefail
cd /root/private_data/ORA0
source /root/private_data/.ai_user_info/ai_proxy 2>/dev/null || true

TARGET_STEP=${1:-1500}
CKPT="checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18_s${TARGET_STEP}.pt"
LOG="logs/auto_eval_s${TARGET_STEP}.log"

export PY=/opt/conda/bin/python
export DINO=/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for checkpoint ${CKPT} to appear..."
while [[ ! -f "$CKPT" ]]; do
    sleep 20
done

# Wait an extra 5s to ensure torch.save has fully finished flushing to disk
sleep 5
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint found! Starting parallel closed-loop evaluation (10 trials/task)..."

bash scripts/eval_mw_hard2_wam4va.sh "$CKPT" data/hard2_peer_h6_p2_eval_v1.pt > "$LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation finished! Results saved to logs/"
