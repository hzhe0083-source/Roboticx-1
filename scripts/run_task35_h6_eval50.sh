#!/usr/bin/env bash
# Fixed paired 50-trial task35 precision evaluation (seeds 35000..35049).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: $0 checkpoint.pt [log-name]}
NAME=${2:-$(basename "$CKPT" .pt)}
PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
FEATURES=data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt
CACHE=data/dino35_h6_clean60_recovery30_cache_v1
ROI=checkpoints/dino_metric_roi_task35_v2_native480_seed777_1k.pt
LOG=logs/${NAME}_eval50.log
for path in "$CKPT" "$DINO" "$FEATURES" "$ROI" "$CACHE/meta.json" \
  "$CACHE/block11.npy" "$CACHE/block23.npy"; do
  [[ -f "$path" ]] || { echo "missing $path" >&2; exit 1; }
done
mkdir -p logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -u -B eval_metaworld.py \
  --task35-precision-contract \
  --checkpoint "$CKPT" --features "$FEATURES" \
  --dino-feature-cache "$CACHE" \
  --main-vision-checkpoint "$DINO" \
  --dino-roi-checkpoint "$ROI" --dino-roi-alpha 1.0 \
  --task-ids 35 --trials-per-task 50 --execute-steps 6 --horizon 500 \
  --wam off --direct-head auto --debug-stage-metrics --flow-samples 1 \
  --device cuda 2>&1 | tee "$LOG"
