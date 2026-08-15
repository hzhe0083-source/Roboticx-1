#!/usr/bin/env bash
# After a precision 50-trial JSON exists: run the five causal diagnostics and compare.
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
if [[ ${#FORCE[@]} -eq 0 ]] && "$PY" -B scripts/task35_proc.py --check trainer >/dev/null; then
  echo "FM trainer still running; refusing to take the GPU. Pass --force to override." >&2
  exit 3
fi
BASE=logs/${NAME}_eval50.json
[[ -f "$BASE" ]] || { echo "missing baseline $BASE; run eval50 first" >&2; exit 1; }

ABLATIONS=(temporal-reverse geometry-zero geometry-shuffle roi-off dense-zero)
JSONS=()
for ablation in "${ABLATIONS[@]}"; do
  scripts/run_task35_h6_ablation50.sh "${FORCE[@]}" "$CKPT" "$ablation" "${NAME}_${ablation}"
  JSONS+=("logs/${NAME}_${ablation}.json")
done

/home/ryan/.venvs/pytorch-gpu/bin/python -B scripts/compare_task35_paired_eval.py \
  "$BASE" "${JSONS[@]}" \
  --output "logs/${NAME}_causal_compare.json"
echo "causal suite finished for $CKPT" >&2
