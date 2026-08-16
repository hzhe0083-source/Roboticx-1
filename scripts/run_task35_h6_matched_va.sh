#!/usr/bin/env bash
# Matched task35 precision experiment. The only arm difference is --direct-head.
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${1:?usage: $0 fm [steps] [batch] [run_tag]}
STEPS=${2:-20000}
BATCH=${3:-6}
RUN_TAG=${4:-${STEPS}}
case "$ARM" in
  fm) DECODER=();;
  direct)
    if [[ "${TASK35_ALLOW_DIRECT:-0}" != "1" ]]; then
      echo "Direct training is disabled for this task35 run. Set TASK35_ALLOW_DIRECT=1 to override." >&2
      exit 2
    fi
    DECODER=(--direct-head)
    ;;
  *) echo "ARM must be fm (Direct is disabled unless TASK35_ALLOW_DIRECT=1)" >&2; exit 2;;
esac

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt
CACHE=data/dino35_h6_clean60_recovery30_cache_v1
ROI=checkpoints/dino_metric_roi_task35_v2_native480_seed777_1k.pt
SAVE=checkpoints/task35_h6_dino_mtvj_${ARM}_${RUN_TAG}.pt
LOG=logs/task35_h6_dino_mtvj_${ARM}_${RUN_TAG}.log

for path in "$DINO" "$DATA" "$ROI" "$CACHE/meta.json" "$CACHE/index.pkl" \
  "$CACHE/block11.npy" "$CACHE/block23.npy" "$CACHE/raw_frames.npy"; do
  [[ -e "$path" ]] || { echo "missing required artifact: $path" >&2; exit 1; }
done
[[ ! -e "$SAVE" ]] || { echo "refusing to overwrite $SAVE" >&2; exit 1; }

COMMON=(
  --task35-precision-contract
  --data "$DATA" --dino-feature-cache "$CACHE"
  --dino-main-vision --dino-dense-metric
  --main-vision-checkpoint "$DINO"
  --main-vision-grid 16 --main-vision-frames 4
  --main-vision-temporal --main-vision-temporal-scale 1.0
  --metric-geometry-inject
  --dino-roi-checkpoint "$ROI" --dino-roi-alpha 1.0
  --single-task --task-sampling weighted --task-locality-block-batches 16
  --batch-size "$BATCH" --sequence-length 4 --min-sequence-length 4
  --num-workers 0
  --lr 0.0001 --seed 0 --device cuda
  --va-layers 8 --va-attention-backend auto
  --flow-cond adaln --flow-layers 6 --flow-steps 8
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 1.0
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003
  --mtvj-train-relation --lr-mtvj-relation 0.00002
  --mtvj-visual-aux-every "${MTVJ_VISUAL_AUX_EVERY:-10}"
  --mtvj-visual-aux-batch "${MTVJ_VISUAL_AUX_BATCH:-8}"
  --steps "$STEPS" --save-every 1000 --main-vision-encode-batch 16
  --save "$SAVE"
)

mkdir -p checkpoints logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py "${COMMON[@]}" "${DECODER[@]}" 2>&1 | tee "$LOG"
