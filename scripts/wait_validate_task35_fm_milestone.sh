#!/usr/bin/env bash
# Wait for one archived FM milestone, then CPU-validate and slice-diagnose it.
set -euo pipefail
cd "$(dirname "$0")/.."

STEP=${1:?usage: $0 step [stem]}
STEM=${2:-checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1}
DEST=${STEM}_step${STEP}.pt
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
NAME=$(basename "${DEST%.pt}")

echo "waiting for $DEST" >&2
while [[ ! -s "$DEST" || ! -s "${DEST}.sha256" ]]; do
  sleep 15
done
echo "found $DEST" >&2
VALIDATE=logs/${NAME}_validate.json
SLICES=logs/${NAME}_clean_recovery_slices.json
# Prefer the archiver's CPU validator; only run it here if that report is absent.
if [[ ! -s "$VALIDATE" ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -B scripts/validate_task35_fm_checkpoint.py \
    "$DEST" --expected-step "$STEP" --output "$VALIDATE"
fi
if [[ ! -s "$SLICES" ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -B scripts/diag_task35_clean_recovery_slices.py \
    --checkpoint "$DEST" --batch 16 --output "$SLICES"
fi
echo "validated and sliced $DEST" >&2
