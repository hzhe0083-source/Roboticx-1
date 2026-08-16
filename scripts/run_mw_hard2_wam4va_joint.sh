#!/usr/bin/env bash
# Stage 2: unfreeze VA+FM+WAM. Resume the JEPA world checkpoint.
#   bash scripts/run_mw_hard2_wam4va_joint.sh
#   bash scripts/run_mw_hard2_wam4va_joint.sh 10000 6
# Requires: checkpoints/mw_hard2_wam4va_world.pt
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock.pt
WORLD=checkpoints/mw_hard2_wam4va_world.pt
SAVE=checkpoints/mw_hard2_wam4va_h48_15k.pt
LOG=logs/mw_hard2_wam4va_h48_15k.log
STEPS=${1:-15000}
BATCH=${2:-6}

[[ -f "$DINO" ]] || { echo "missing DINO weights: $DINO" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "missing $DATA; run world stage first" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "missing world ckpt $WORLD; run scripts/run_mw_hard2_wam4va_world.sh" >&2; exit 1; }
[[ ! -e "$SAVE" ]] || { echo "refusing to overwrite $SAVE" >&2; exit 1; }

mkdir -p checkpoints logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  --data "$DATA" \
  --dino-main-vision --dino-dense-metric \
  --main-vision-checkpoint "$DINO" \
  --main-vision-grid 16 --main-vision-frames 4 \
  --main-vision-temporal --main-vision-temporal-scale 1.0 \
  --main-vision-encode-batch 8 \
  --metric-geometry-inject \
  --wam4va --wmrm-inject last --wmrm-target dino \
  --wmrm-cycle-steps 6 \
  --single-task --task-sampling weighted --task-locality-block-batches 16 \
  --batch-size "$BATCH" --sequence-length 4 --min-sequence-length 4 \
  --num-workers 0 \
  --lr 0.0001 --seed 0 --device cuda \
  --va-layers 8 --va-attention-backend auto \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 \
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
  --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 \
  --resume "$WORLD" \
  --steps "$STEPS" --save-every 1000 \
  --save "$SAVE" \
  2>&1 | tee "$LOG"
