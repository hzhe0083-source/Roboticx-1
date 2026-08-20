#!/usr/bin/env bash
# Sequential fixed-checkpoint WMRM ablation suite. Never starts training.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
CHECKPOINT=${CHECKPOINT:-checkpoints/mw_hard2_wam4va_visualmotion_actionrankcap02_v1.formal_12330_to_20000_s15000.pt}
FEATURES=${FEATURES:-data/metaworld_longtraj_windows_h48_asm_doorunlock.pt}
DINO=${DINO:-/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors}
# MetaWorld closed-loop validity: use the same episode budget as the formal
# H=500 protocol. H=60 and n=3 produced the 0/6 floor effect in v1/v2 because
# e.g. task16 seed 16003 only succeeds after step 60, and the first three
# task16 seeds are known zero-success under this checkpoint.
TRIALS=${TRIALS:-10}
HORIZON=${HORIZON:-500}
EXECUTION_HORIZON=${EXECUTION_HORIZON:-6}
TASK_IDS=${TASK_IDS:-0,16}
DEVICE=${DEVICE:-cuda}
OUT_PREFIX=${OUT_PREFIX:-logs/wmrm_ablation_step15000_h500_v3}
DRY_RUN=${DRY_RUN:-0}
LOCK=${LOCK:-/tmp/ora0_wam4va_visualmotion_train.lock}
MODES=(normal action-off vision-off both-off proposal-only)

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# -eq 0 ]] || fail "usage: bash $0 (configure with environment variables)"
command -v flock >/dev/null || fail "flock is required"
[[ -x "$PY" ]] || fail "missing Python: $PY"
for path in "$CHECKPOINT" "$FEATURES" "$DINO" eval_metaworld.py; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done
[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] || fail "TRIALS must be a positive integer"
[[ "$HORIZON" =~ ^[1-9][0-9]*$ ]] || fail "HORIZON must be a positive integer"
[[ "$HORIZON" -ge 500 ]] || fail "HORIZON must be >= 500 for this formal MetaWorld ablation (H=60 truncated known-success seeds)"
[[ "$EXECUTION_HORIZON" =~ ^(1|2|3|6)$ ]] || fail "EXECUTION_HORIZON must be one of 1,2,3,6"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1"

"$PY" -B - "$CHECKPOINT" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
config = payload.get("config") or {}
step = payload.get("global_step")
if step != 15000:
    raise SystemExit(f"checkpoint must be fixed step15000, got global_step={step!r}")
if not config.get("wmrm"):
    raise SystemExit("checkpoint config does not enable WMRM")
if int(config.get("wmrm_cycle_steps", -1)) != 6:
    raise SystemExit("checkpoint wmrm_cycle_steps/world_horizon must remain 6")
print("checkpoint contract: step15000, WMRM enabled, world_horizon=6", flush=True)
PY

exec 9>"$LOCK"
flock -n 9 || fail "another ORA0 trainer/evaluator owns the global lock"

for mode in "${MODES[@]}"; do
  [[ ! -e "${OUT_PREFIX}_${mode}.log" && ! -e "${OUT_PREFIX}_${mode}.json" ]] || \
    fail "refusing to overwrite output for mode=$mode"
done
[[ ! -e "${OUT_PREFIX}_summary.json" ]] || fail "refusing to overwrite summary output"

require_idle() {
  "$PY" -B - <<'PY'
from pathlib import Path
import os
matches = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        argv = [x.decode('utf-8', 'replace') for x in (entry / 'cmdline').read_bytes().split(b'\0') if x]
    except OSError:
        continue
    if any(Path(arg).name in {'train.py', 'eval_metaworld.py'} for arg in argv):
        matches.append((entry.name, ' '.join(argv)))
if matches:
    raise SystemExit('; '.join(f'active evaluator/trainer pid={pid}: {cmd[:240]}' for pid, cmd in matches))
print('process check: no train.py/eval_metaworld.py process', flush=True)
PY
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required for GPU safety check"
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || fail "cannot query GPU compute processes"
  [[ -z "${apps//[[:space:]]/}" ]] || fail "GPU has active compute processes: ${apps//$'\n'/; }"
  "$PY" -B - "$DEVICE" <<'PY'
import sys, torch
if sys.argv[1] == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
    raise SystemExit('exactly one CUDA device is required')
if sys.argv[1] == 'cuda':
    free, total = torch.cuda.mem_get_info()
    if free < int(total * 0.85):
        raise SystemExit(f'GPU lacks headroom: free={free} total={total}')
print('GPU check: idle with required headroom', flush=True)
PY
}

for mode in "${MODES[@]}"; do
  log="${OUT_PREFIX}_${mode}.log"
  json="${OUT_PREFIX}_${mode}.json"
  require_idle
  case "$mode" in
    normal) evaluator_mode=normal ;;
    action-off) evaluator_mode=action-write-off ;;
    vision-off) evaluator_mode=vision-write-off ;;
    both-off) evaluator_mode=both-write-off ;;
    proposal-only) evaluator_mode=proposal-only ;;
  esac
  mkdir -p "$(dirname "$log")"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'dry-run mode=%s evaluator-mode=%s log=%s json=%s\n' "$mode" "$evaluator_mode" "$log" "$json"
    continue
  fi
  printf 'starting mode=%s trials=%s tasks=%s horizon=%s execute=%s\n' "$mode" "$TRIALS" "$TASK_IDS" "$HORIZON" "$EXECUTION_HORIZON" | tee "$log"
  set -o pipefail
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B eval_metaworld.py \
      --checkpoint "$CHECKPOINT" --features "$FEATURES" --main-vision-checkpoint "$DINO" \
      --task-ids "$TASK_IDS" --trials-per-task "$TRIALS" --execution-horizon "$EXECUTION_HORIZON" \
      --horizon "$HORIZON" --flow-samples 1 --wam off --record-action-chunks \
      --wmrm-ablation-mode "$evaluator_mode" --device "$DEVICE" \
      --output-json "$json" 2>&1 | tee -a "$log"
done

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'dry-run complete; no evaluator was launched\n'
  exit 0
fi
jsons=()
for mode in "${MODES[@]}"; do jsons+=("${OUT_PREFIX}_${mode}.json"); done
"$PY" -B scripts/analyze_wmrm_ablation.py "${jsons[@]}" --output "${OUT_PREFIX}_summary.json"
printf 'completed all WMRM modes sequentially; summary=%s\n' "${OUT_PREFIX}_summary.json"
