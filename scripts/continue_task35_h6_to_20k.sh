#!/usr/bin/env bash
# After the original 15k trainer exits, exact-resume 5000 more FM updates to 20k.
# Same save path and log so the archiver can copy 18000/20000. Never starts Direct.
set -euo pipefail
cd "$(dirname "$0")/.."

STEM=${1:-checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1}
LIVE=${STEM}.pt
ARCH15=${STEM}_step15000.pt
LOG=logs/$(basename "$STEM").log
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt
CACHE=data/dino35_h6_clean60_recovery30_cache_v1
ROI=checkpoints/dino_metric_roi_task35_v2_native480_seed777_1k.pt

echo "waiting for 15k trainer exit before exact-resume to 20k" >&2
while "$PY" -B scripts/task35_proc.py --check trainer >/dev/null; do
  sleep 30
done

# Do not resume from the live file until 15k is archived. Starting a new
# trainer on the same path would race the 15k copy.
SRC=""
for _ in $(seq 1 20); do
  if [[ -s "$ARCH15" && -s "${ARCH15}.sha256" ]]; then
    found=$("$PY" -B scripts/peek_task35_checkpoint_step.py "$ARCH15")
    expected=$(awk '{print $1}' "${ARCH15}.sha256")
    actual=$(sha256sum "$ARCH15" | awk '{print $1}')
    if [[ "$found" == "15000" && -n "$expected" && "$expected" == "$actual" ]]; then
      SRC=$ARCH15
      break
    fi
  fi
  sleep 15
done
if [[ -z "$SRC" ]]; then
  echo "no archived global_step=15000 checkpoint to exact-resume from: $ARCH15" >&2
  exit 4
fi
if [[ -s "${STEM}_step20000.pt" && -s "${STEM}_step20000.pt.sha256" ]]; then
  echo "20k archive already exists; not starting another trainer" >&2
  exit 0
fi
if "$PY" -B scripts/task35_proc.py --check trainer >/dev/null; then
  echo "a trainer appeared before 20k resume; refusing to start a second one" >&2
  exit 3
fi

echo "exact-resuming $SRC for 5000 more steps (15000 -> 20000)" >&2
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  --task35-precision-contract \
  --resume-exact "$SRC" \
  --data "$DATA" --dino-feature-cache "$CACHE" \
  --dino-main-vision --dino-dense-metric \
  --main-vision-checkpoint "$DINO" \
  --main-vision-grid 16 --main-vision-frames 4 \
  --main-vision-temporal --main-vision-temporal-scale 1.0 \
  --metric-geometry-inject \
  --dino-roi-checkpoint "$ROI" --dino-roi-alpha 1.0 \
  --single-task --task-sampling weighted --task-locality-block-batches 16 \
  --batch-size 6 --sequence-length 4 --min-sequence-length 4 \
  --num-workers 0 \
  --lr 0.0001 --seed 0 --device cuda \
  --va-layers 8 --va-attention-backend auto \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 \
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 1.0 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
  --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 \
  --steps 5000 --save-every 1000 --main-vision-encode-batch 16 \
  --save "$LIVE" 2>&1 | tee -a "$LOG"
