#!/usr/bin/env bash
# auto_transition_to_all49.sh
# 1. Monitors current Hard-2 run until step 3000 is completed (~40 epochs).
# 2. Runs closed-loop evaluation on the 3000-step checkpoint.
# 3. Gracefully stops Hard-2 training and launches All-49 MetaWorld training for 25 Epochs (30,715 steps at batch=18).
set -euo pipefail

cd /root/private_data/ORA0
source /root/private_data/.ai_user_info/ai_proxy 2>/dev/null || true

export PY=/opt/conda/bin/python
export VERIFY_PY=/opt/conda/bin/python
export DINO=/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

HARD2_CKPT_3000="checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18_s3000.pt"
HARD2_LOG="logs/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18.log"
ALL49_DATA="data/metaworld_longtraj_windows_h48_all49_repaired_v2_clean.pt"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Monitor] Monitoring Hard-2 training towards step 3000..."

while true; do
    if [[ -f "$HARD2_CKPT_3000" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 3000] Checkpoint $HARD2_CKPT_3000 found!"
        break
    fi
    CURRENT_STEP=$(grep -E "step=[0-9]+" "$HARD2_LOG" 2>/dev/null | tail -n 1 | grep -o "step=[0-9]*" | cut -d'=' -f2 || echo "0")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Progress] Hard-2 Current Step: $CURRENT_STEP / 3000"
    sleep 30
done

# Wait 5s for disk flush
sleep 5

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Eval] Running closed-loop evaluation on step 3000 checkpoint..."
bash scripts/eval_mw_hard2_wam4va.sh "$HARD2_CKPT_3000" data/hard2_peer_h6_p2_eval_v1.pt > logs/auto_eval_s3000.log 2>&1 || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Transition] Stopping Hard-2 training process..."
pkill -f "train.py.*scratch.s30000_b18" 2>/dev/null || true
sleep 5

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [All-49] Launching Full MetaWorld (49 tasks) 25-Epoch training (batch=18, steps=30715)..."
ALL49_RUN_ID="mw_all49_dino_flow_matching_25ep_b18_v1"
ALL49_SAVE="checkpoints/${ALL49_RUN_ID}.pt"
ALL49_LOG="logs/${ALL49_RUN_ID}.log"

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  MUJOCO_GL=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$PY" -u -B train.py \
    --data "$ALL49_DATA" \
    --dino-main-vision --dino-dense-metric \
    --main-vision-checkpoint "$DINO" \
    --main-vision-grid 16 --main-vision-frames 4 --main-vision-temporal \
    --main-vision-temporal-scale 1.0 --main-vision-encode-batch 8 \
    --metric-geometry-inject \
    --single-task --task-sampling balanced \
    --task-locality-block-batches 16 --batch-size 18 \
    --sequence-length 4 --min-sequence-length 4 --num-workers 0 --lr 0.0001 \
    --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 \
    --va-attention-backend auto --flow-cond adaln --flow-layers 6 --flow-steps 8 \
    --flow-prefix-steps 2 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
    --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
    --mtvj-train-relation --lr-mtvj-relation 0.00002 \
    --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 \
    --steps 30715 --save-every 1500 --save-step-copies --save "$ALL49_SAVE" \
    > "$ALL49_LOG" 2>&1 &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Done] Full MetaWorld 49-task 25-Epoch training launched successfully (PID=$!)!"
