#!/usr/bin/env bash
# After the 15k trainer exits and the final archive exists, run post-train eval.
# Never starts while the trainer still owns the GPU. A crash without step15000
# is a hard failure, not an infinite wait.
set -euo pipefail
cd "$(dirname "$0")/.."

STEM=${1:-checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1}
DEST=${STEM}_step15000.pt
LIVE=${STEM}.pt
LOG=logs/$(basename "$STEM").log
NAME=$(basename "${DEST%.pt}")
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}

echo "waiting for trainer exit and $DEST" >&2
while "$PY" -B scripts/task35_proc.py --check trainer >/dev/null; do
  sleep 30
done

# Give the archiver a short window after the final save event.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ -s "$DEST" && -s "${DEST}.sha256" ]] && break
  sleep 15
done

if [[ ! -s "$DEST" || ! -s "${DEST}.sha256" ]]; then
  if [[ -s "$LIVE" ]]; then
    echo "no 15000 archive; checking live checkpoint after trainer exit" >&2
    if CUDA_VISIBLE_DEVICES= "$PY" -B scripts/validate_task35_fm_checkpoint.py \
      "$LIVE" --expected-step 15000 --skip-module-load \
      --output "logs/$(basename "$STEM")_live15000_validate.json"; then
      cp --reflink=auto --sparse=always "$LIVE" "${DEST}.tmp"
      mv "${DEST}.tmp" "$DEST"
      sha256sum "$DEST" > "${DEST}.sha256"
      echo "promoted live 15000 checkpoint to $DEST" >&2
    fi
  fi
fi

if [[ ! -s "$DEST" || ! -s "${DEST}.sha256" ]]; then
  echo "trainer exited without a valid 15000 archive: $DEST" >&2
  if [[ -f "$LOG" ]] && grep -E 'OutOfMemoryError|CUDA out of memory|Traceback \(most recent call last\)' "$LOG" >/dev/null; then
    echo "training log contains a crash traceback or OOM" >&2
  fi
  exit 4
fi

echo "trainer gone; archived $DEST" >&2
# 1k/2k stay mechanism-only; do not reload them before the 50-seed suite.
for step in 3000 6000 9000 12000 15000; do
  scripts/wait_validate_task35_fm_milestone.sh "$step" "$STEM"
done
CUDA_VISIBLE_DEVICES= "$PY" -B scripts/preflight_task35_posttrain.py \
  --checkpoint "$DEST" --expected-step 15000 \
  --output "logs/${NAME}_preflight.json"
scripts/run_task35_h6_eval_suite.sh "$STEM"
echo "finished post-train eval suite; best ledger is logs/task35_best_fm.json" >&2
