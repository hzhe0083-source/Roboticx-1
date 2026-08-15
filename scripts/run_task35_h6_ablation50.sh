#!/usr/bin/env bash
# Fixed-seed task35 FM causal diagnostic. This is not precision acceptance.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: $0 checkpoint.pt temporal-reverse|geometry-zero|geometry-shuffle|roi-off|dense-zero [log-name]}
ABLATION=${2:?missing ablation}
NAME=${3:-$(basename "$CKPT" .pt)_${ABLATION}}
case "$ABLATION" in
  temporal-reverse|geometry-zero|geometry-shuffle|roi-off|dense-zero) ;;
  *) echo "unsupported task35 ablation: $ABLATION" >&2; exit 2;;
esac
PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
FEATURES=data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt
ROI=checkpoints/dino_metric_roi_task35_v2_native480_seed777_1k.pt
LOG=logs/${NAME}.log
for path in "$CKPT" "$DINO" "$FEATURES" "$ROI"; do
  [[ -f "$path" ]] || { echo "missing $path" >&2; exit 1; }
done
mkdir -p logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -u -B eval_metaworld.py \
  --checkpoint "$CKPT" --features "$FEATURES" \
  --main-vision-checkpoint "$DINO" \
  --dino-roi-checkpoint "$ROI" --dino-roi-alpha 1.0 \
  --task35-causal-ablation "$ABLATION" \
  --task-ids 35 --trials-per-task 50 --execute-steps 6 --horizon 500 \
  --wam off --direct-head auto --debug-stage-metrics --flow-samples 1 \
  --device cuda 2>&1 | tee "$LOG"
