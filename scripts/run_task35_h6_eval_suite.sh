#!/usr/bin/env bash
# After FM training: 50-seed eval every validated milestone, then elect the best.
# Causal diagnostics run only on the winner. Refuses while the trainer owns GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=()
if [[ "${1:-}" == "--force" ]]; then
  FORCE=(--force)
  shift
fi
STEM=${1:-checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1}
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}

if [[ ${#FORCE[@]} -eq 0 ]] && "$PY" -B scripts/task35_proc.py --check trainer >/dev/null; then
  echo "FM trainer still running; refusing to take the GPU. Pass --force to override." >&2
  exit 3
fi

"$PY" -B scripts/list_task35_fm_candidates.py >/dev/null
"$PY" -B scripts/plan_task35_eval_suite.py --require-all \
  --output logs/task35_eval_suite_plan.json

mapfile -t TO_EVAL < <(
  "$PY" -B - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("logs/task35_eval_suite_plan.json").read_text())
for row in plan["to_eval"]:
    print(row["path"])
PY
)

for ckpt in "${TO_EVAL[@]+"${TO_EVAL[@]}"}"; do
  name=$(basename "${ckpt%.pt}")
  TASK35_SKIP_CAUSAL=1 scripts/run_task35_h6_posttrain_eval.sh "${FORCE[@]}" "$ckpt" "$name"
done

mapfile -t EVAL50 < <(
  "$PY" -B - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("logs/task35_eval_suite_plan.json").read_text())
for path in plan["eval50_paths"]:
    print(path)
PY
)
if (( ${#EVAL50[@]} == 0 )); then
  echo "eval suite planned no eval50 paths" >&2
  exit 1
fi
"$PY" -B scripts/select_task35_best_fm.py "${EVAL50[@]}" \
  --output logs/task35_best_fm.json

WINNER=$("$PY" -B - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("logs/task35_best_fm.json").read_text())["selected"]["path"])
PY
)
WINNER_NAME=$(basename "${WINNER%.pt}")
if [[ "${TASK35_SKIP_CAUSAL:-0}" != "1" ]]; then
  scripts/run_task35_h6_causal_suite.sh "${FORCE[@]}" "$WINNER" "$WINNER_NAME"
fi
"$PY" -B scripts/report_task35_fm_status.py >/dev/null
echo "eval suite finished; best=$WINNER" >&2
