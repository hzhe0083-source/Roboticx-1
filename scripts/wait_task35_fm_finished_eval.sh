#!/usr/bin/env bash
# After the 15k trainer exits and the final archive exists, run post-train eval.
# Never starts while the trainer still owns the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

STEM=${1:-checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1}
DEST=${STEM}_step15000.pt
TRAINER_NEEDLE='train.py --task35-precision-contract'
NAME=$(basename "${DEST%.pt}")

echo "waiting for trainer exit and $DEST" >&2
while pgrep -f "$TRAINER_NEEDLE" >/dev/null; do
  sleep 30
done
while [[ ! -s "$DEST" || ! -s "${DEST}.sha256" ]]; do
  sleep 15
done
echo "trainer gone; archived $DEST" >&2
scripts/wait_validate_task35_fm_milestone.sh 15000 "$STEM"
scripts/run_task35_h6_posttrain_eval.sh "$DEST" "$NAME"
echo "finished post-train eval for $DEST" >&2
