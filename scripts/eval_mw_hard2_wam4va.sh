#!/usr/bin/env bash
# Normal MetaWorld closed-loop eval for the two hard tasks (10 trials/task).
# WAM4VA is loaded from checkpoint config (wmrm=True). --wam off only disables
# the old JointWorldActionFlow residual, not WAM4VA.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
CKPT=${1:-checkpoints/mw_hard2_wam4va_h48_15k.pt}
FEATURES=${2:-data/metaworld_longtraj_windows_h48_asm_doorunlock.pt}
NAME=$(basename "$CKPT" .pt)
LOG=logs/${NAME}_eval10.log
JSON=logs/${NAME}_eval10.json

[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$FEATURES" ]] || { echo "missing features: $FEATURES" >&2; exit 1; }
[[ -f "$DINO" ]] || { echo "missing DINO: $DINO" >&2; exit 1; }

mkdir -p logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --main-vision-checkpoint "$DINO" \
  --task-ids 0,16 \
  --trials-per-task 10 \
  --execute-steps 6 \
  --horizon 500 \
  --wam off \
  --direct-head auto \
  --flow-samples 1 \
  --device cuda \
  --output-json "$JSON" \
  2>&1 | tee "$LOG"
