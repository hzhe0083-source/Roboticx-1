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
  if ! "$PY" -B scripts/task35_proc.py --check pipeline >/dev/null; then
    echo "pipeline gone before milestone $STEP appeared: $DEST" >&2
    exit 4
  fi
  sleep 15
done
echo "found $DEST" >&2
expected_sha=$(awk '{print $1}' "${DEST}.sha256")
actual_sha=$(sha256sum "$DEST" | awk '{print $1}')
if [[ -z "$expected_sha" || "$expected_sha" != "$actual_sha" ]]; then
  echo "milestone SHA mismatch for $DEST expected=$expected_sha actual=$actual_sha" >&2
  exit 4
fi
found_step=$("$PY" -B scripts/peek_task35_checkpoint_step.py "$DEST")
if [[ "$found_step" != "$STEP" ]]; then
  echo "archive global_step=$found_step != expected $STEP for $DEST" >&2
  exit 4
fi
VALIDATE=logs/${NAME}_validate.json
SLICES=logs/${NAME}_clean_recovery_slices.json
LOCK=logs/${NAME}.lock
mkdir -p logs
exec 9>"$LOCK"
flock 9

wait_for_cpu_ram() {
  # Validate/slice load a 1.6 GiB checkpoint on CPU. Do not start if the
  # live trainer plus this job would thrash; wait instead of OOM-killing.
  local avail_kb needed_kb=4194304
  local waited=0
  while true; do
    avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [[ -n "$avail_kb" && "$avail_kb" -ge "$needed_kb" ]]; then
      return 0
    fi
    echo "low MemAvailable=${avail_kb:-?} kB; waiting before CPU validate/slice" >&2
    sleep 15
    waited=$((waited + 15))
    if (( waited >= 600 )); then
      echo "MemAvailable stayed below 4 GiB for 10 min; refusing CPU validate" >&2
      return 1
    fi
  done
}

run_cpu() {
  if command -v ionice >/dev/null 2>&1; then
    ionice -c 3 nice -n 10 "$@"
  else
    nice -n 10 "$@"
  fi
}

slice_matches_archive() {
  "$PY" -B - "$SLICES" "$actual_sha" "$STEP" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
sha = sys.argv[2]
step = int(sys.argv[3])
ok = (
    payload.get("contract") == "task35_clean_recovery_slice_v1"
    and payload.get("sha256") == sha
    and int(payload.get("global_step", -1)) == step
)
raise SystemExit(0 if ok else 1)
PY
}

report_matches_archive() {
  "$PY" -B - "$VALIDATE" "$actual_sha" "$STEP" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
sha = sys.argv[2]
step = int(sys.argv[3])
items = payload.get("reports", [payload])
ok = any(
    item.get("ok")
    and item.get("loaded_modules")
    and int(item.get("global_step", -1)) == step
    and item.get("sha256") == sha
    for item in items
)
raise SystemExit(0 if ok else 1)
PY
}

# Archiver only copies. This waiter owns full CPU validate/slice.
need_validate=1
if [[ -s "$VALIDATE" ]] && report_matches_archive; then
  need_validate=0
fi
need_slices=1
if [[ -s "$SLICES" ]] && slice_matches_archive; then
  need_slices=0
fi
if [[ "$need_validate" -eq 1 || "$need_slices" -eq 1 ]]; then
  wait_for_cpu_ram
fi
if [[ "$need_validate" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES= run_cpu "$PY" -B scripts/validate_task35_fm_checkpoint.py \
    "$DEST" --expected-step "$STEP" --output "$VALIDATE"
fi
if [[ "$need_slices" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES= run_cpu "$PY" -B scripts/diag_task35_clean_recovery_slices.py \
    --checkpoint "$DEST" --expected-step "$STEP" --batch 16 --output "$SLICES"
fi
# Compare against the previous archived slice if it exists.
PREV=""
case "$STEP" in
  2000) PREV=${STEM}_step1000 ;;
  3000) PREV=${STEM}_step2000 ;;
  6000) PREV=${STEM}_step3000 ;;
  9000) PREV=${STEM}_step6000 ;;
  12000) PREV=${STEM}_step9000 ;;
  15000) PREV=${STEM}_step12000 ;;
  18000) PREV=${STEM}_step15000 ;;
  20000) PREV=${STEM}_step18000 ;;
esac
if [[ -n "$PREV" && -s "logs/$(basename "$PREV")_clean_recovery_slices.json" && -s "$SLICES" ]]; then
  "$PY" -B scripts/compare_task35_slice_reports.py \
    "logs/$(basename "$PREV")_clean_recovery_slices.json" \
    "$SLICES" \
    --output "logs/${NAME}_vs_$(basename "$PREV")_slices.json"
fi
"$PY" -B scripts/list_task35_fm_candidates.py >/dev/null
"$PY" -B scripts/report_task35_fm_status.py >/dev/null
echo "validated and sliced $DEST" >&2
