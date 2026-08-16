#!/usr/bin/env bash
# Short FM finetune from the elected 15k weights on the P0 C payload.
# New stem only. Does not overwrite 15k or the 1807-window run.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${1:-2000}
BATCH=${2:-6}
RUN_TAG=${3:-failinitC_from15k_b6_sdpa_aux10b8_v1}

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_failinit_v2.pt
CACHE=data/dino35_h6_clean60_recovery30_failinit_cache_v1
ROI=checkpoints/dino_metric_roi_task35_v2_native480_seed777_1k.pt
CKPT=checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000.pt
SAVE=checkpoints/task35_h6_dino_mtvj_fm_${RUN_TAG}.pt
LOG=logs/task35_h6_dino_mtvj_fm_${RUN_TAG}.log

if "$PY" -B scripts/task35_proc.py --check trainer; then
  echo "trainer already owns the GPU" >&2
  exit 3
fi

for path in "$DINO" "$DATA" "$ROI" "$CKPT" "$CACHE/meta.json" "$CACHE/index.pkl" \
  "$CACHE/block11.npy" "$CACHE/block23.npy" "$CACHE/raw_frames.npy"; do
  [[ -e "$path" ]] || { echo "missing required artifact: $path" >&2; exit 1; }
done
[[ ! -e "$SAVE" ]] || { echo "refusing to overwrite $SAVE" >&2; exit 1; }

EXPECTED_DATA=2d3dcde1b7aeac39bee88b0cbfcf054aa93cf5f61185d18fbbb0b68966f6591b
CACHE_DATA=$("$PY" -B -c "import json; print(json.load(open('$CACHE/meta.json'))['dataset_sha256'])")
[[ "$CACHE_DATA" == "$EXPECTED_DATA" ]] || {
  echo "C cache/dataset SHA mismatch: $CACHE_DATA" >&2
  exit 1
}

mkdir -p checkpoints logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  --task35-precision-contract \
  --resume-weights "$CKPT" \
  --data "$DATA" --dino-feature-cache "$CACHE" \
  --dino-main-vision --dino-dense-metric \
  --main-vision-checkpoint "$DINO" \
  --main-vision-grid 16 --main-vision-frames 4 \
  --main-vision-temporal --main-vision-temporal-scale 1.0 \
  --metric-geometry-inject \
  --dino-roi-checkpoint "$ROI" --dino-roi-alpha 1.0 \
  --single-task --task-sampling weighted --task-locality-block-batches 16 \
  --batch-size "$BATCH" --sequence-length 4 --min-sequence-length 4 \
  --num-workers 0 \
  --lr 0.0001 --seed 0 --device cuda \
  --va-layers 8 --va-attention-backend auto \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 \
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 1.0 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
  --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --mtvj-visual-aux-every "${MTVJ_VISUAL_AUX_EVERY:-10}" \
  --mtvj-visual-aux-batch "${MTVJ_VISUAL_AUX_BATCH:-8}" \
  --steps "$STEPS" --save-every 1000 --main-vision-encode-batch 16 \
  --save "$SAVE" \
  2>&1 | tee "$LOG"
