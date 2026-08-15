#!/usr/bin/env bash
# After FM training finishes: validate, held-out metric, then 50-seed closed loop.
# Refuses to start while the 15k trainer still owns the GPU unless --force.
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=()
if [[ "${1:-}" == "--force" ]]; then
  FORCE=(--force)
  shift
fi
CKPT=${1:?usage: $0 [--force] checkpoint.pt [name]}
NAME=${2:-$(basename "$CKPT" .pt)}
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
FEATURES=data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt
TRAINER_NEEDLE='train.py --task35-precision-contract'

if [[ ${#FORCE[@]} -eq 0 ]] && pgrep -f "$TRAINER_NEEDLE" >/dev/null; then
  echo "FM trainer still running; refusing to take the GPU. Pass --force to override." >&2
  exit 3
fi

"$PY" -B scripts/validate_task35_fm_checkpoint.py "$CKPT" \
  --output "logs/${NAME}_validate.json"

"$PY" -B scripts/diag_task35_clean_recovery_slices.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --output "logs/${NAME}_clean_recovery_slices.json"

"$PY" -u -B scripts/eval_dino_metric_policy_task35.py \
  --checkpoint "$CKPT" \
  --language-data "$FEATURES" \
  --dino-checkpoint "$DINO" \
  --samples 100 --batch 8 --seed 2777 \
  --output "logs/${NAME}_metric_holdout2777.json"

scripts/run_task35_h6_eval50.sh "${FORCE[@]}" "$CKPT" "$NAME"
if [[ "${TASK35_SKIP_CAUSAL:-0}" != "1" ]]; then
  scripts/run_task35_h6_causal_suite.sh "${FORCE[@]}" "$CKPT" "$NAME"
fi
"$PY" -B scripts/select_task35_best_fm.py "logs/${NAME}_eval50.json" \
  --output "logs/${NAME}_best.json"
echo "post-train eval finished for $CKPT" >&2
