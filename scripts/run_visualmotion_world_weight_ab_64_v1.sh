#!/usr/bin/env bash
# Sequential 64-update exact-source A/B: World weight 1.0 replay vs 0.5 migration.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_stable_detach_v1.candidate_diag.pt
EXPECTED_SOURCE_REALPATH=/home/ryan/Documents/robot/ORA0/checkpoints/mw_hard2_wam4va_visualmotion_stable_detach_v1.candidate_diag.pt
EXPECTED_SOURCE_SHA256=f580caa4c1588b2a9807f9b0ab746ac54259eaaa482cea16ce5001c30a382f11
EXPECTED_SOURCE_STEP=12010
TARGET_STEP=12074
UPDATES=64
MIGRATION_ID=wmrm_world_weight_1_to_0_5_v1
A_FAMILY=mw_hard2_wam4va_visualmotion_worldweight1_replay64_v1
B_FAMILY=mw_hard2_wam4va_visualmotion_worldweight0p5_migration64_v1
A_SAVE=checkpoints/${A_FAMILY}.pt
B_SAVE=checkpoints/${B_FAMILY}.pt
A_LOG=logs/${A_FAMILY}.train.log
B_LOG=logs/${B_FAMILY}.train.log
A_REPORT=diagnostics/${A_FAMILY}.json
B_REPORT=diagnostics/${B_FAMILY}.json
PAIR_REPORT=diagnostics/${A_FAMILY}_vs_${B_FAMILY}.json
ANALYZER=scripts/analyze_visualmotion_world_weight_ab.py
LOCK=/tmp/ora0_wam4va_visualmotion_train.lock
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
SPLIT=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha256_file() { sha256sum -- "$1" | cut -d' ' -f1; }

[[ $# -eq 0 ]] || { printf 'usage: bash %s\n' "$0" >&2; exit 2; }
command -v flock >/dev/null || fail "flock is required"
exec 9>"$LOCK"
flock -n 9 || fail "another visual-motion trainer owns the exclusive trainer lock"

for path in "$PY" "$SOURCE" "$DINO" "$DATA" "$SPLIT" "$ANALYZER" train.py; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done
mkdir -p checkpoints logs diagnostics
for path in "$A_SAVE" "$B_SAVE" "$A_LOG" "$B_LOG" "$A_REPORT" "$B_REPORT" "$PAIR_REPORT"; do
  [[ ! -e "$path" ]] || fail "refusing to overwrite paired output: $path"
done

available_kib=$(df --output=avail -k checkpoints | tail -n 1 | tr -d '[:space:]')
[[ "$available_kib" =~ ^[0-9]+$ ]] || fail "cannot determine checkpoint filesystem free space"
((available_kib >= 14 * 1024 * 1024)) || fail "at least 14 GiB free is required"
[[ "$A_SAVE" != "$SOURCE" && "$B_SAVE" != "$SOURCE" && "$A_SAVE" != "$B_SAVE" ]] || \
  fail "source and paired destinations must all differ"

source_sha_before=$(sha256_file "$SOURCE")
verify_source_unchanged_on_exit() {
  local status=$?
  trap - EXIT
  if [[ "$(sha256_file "$SOURCE")" != "$source_sha_before" ]]; then
    printf 'ERROR: immutable source checkpoint was modified\n' >&2
    exit 1
  fi
  exit "$status"
}
trap verify_source_unchanged_on_exit EXIT

require_no_active_train() {
  "$PY" -B - <<'PY'
from pathlib import Path
import os

matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        argv = [part.decode("utf-8", "replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
    except OSError:
        continue
    if any(Path(arg).name == "train.py" for arg in argv[1:]):
        matches.append((entry.name, " ".join(argv)))
if matches:
    for pid, command in matches:
        print(f"active train.py pid={pid}: {command[:300]}")
    raise SystemExit("an active train.py process exists")
print("train.py process check: idle", flush=True)
PY
}

require_idle_gpu() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required for compute-idle check"
  local compute_apps
  compute_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || \
    fail "cannot query GPU compute processes"
  [[ -z "${compute_apps//[[:space:]]/}" ]] || \
    fail "GPU has active compute processes: ${compute_apps//$'\n'/; }"
  printf 'GPU compute check: idle (display processes allowed)\n'
}

verify_source() {
  local actual_realpath actual_sha
  actual_realpath=$(realpath -e -- "$SOURCE") || fail "cannot resolve source checkpoint"
  [[ "$actual_realpath" == "$EXPECTED_SOURCE_REALPATH" ]] || fail "source realpath mismatch: $actual_realpath"
  actual_sha=$(sha256_file "$SOURCE")
  [[ "$actual_sha" == "$EXPECTED_SOURCE_SHA256" ]] || fail "source SHA-256 mismatch: $actual_sha"
  "$PY" -B - "$SOURCE" "$EXPECTED_SOURCE_STEP" "$ANALYZER" <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1]).resolve(strict=True)
expected_step = int(sys.argv[2])
spec = spec_from_file_location("world_weight_ab_analyzer", sys.argv[3])
if spec is None or spec.loader is None:
    raise SystemExit("cannot load paired analyzer")
module = module_from_spec(spec)
spec.loader.exec_module(module)
payload = torch.load(path, map_location="cpu", weights_only=True)
module.validate_checkpoint_payload(
    payload, expected_step=expected_step, expected_weight=1.0, label="source"
)
print(f"source verified: {path} global_step={expected_step} world_weight=1.0", flush=True)
PY
}

COMMON=(
  --data "$DATA"
  --world-split-manifest "$SPLIT"
  --visual-world-supervision
  --world-action-rank-stage cycle
  --dino-main-vision --dino-dense-metric
  --main-vision-checkpoint "$DINO"
  --main-vision-grid 16 --main-vision-frames 4
  --main-vision-temporal --main-vision-temporal-scale 1.0
  --main-vision-encode-batch 8
  --metric-geometry-inject
  --wam4va --wmrm-inject all --wmrm-target dino
  --wmrm-cycle-steps 6
  --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16
  --wmrm-predictor st_blocks --wmrm-predictor-depth 6
  --wmrm-predictor-width 384 --wmrm-predictor-heads 12
  --wmrm-detach-proposal-stage-state
  --single-task --task-sampling balanced --task-locality-block-batches 4
  --batch-size 3 --sequence-length 4 --min-sequence-length 4
  --num-workers 0 --lr 0.0001 --seed 0 --device cuda
  --feature-autocast-bf16
  --va-layers 8 --va-attention-backend auto
  --flow-cond adaln --flow-layers 6 --flow-steps 8
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003
  --mtvj-train-relation --lr-mtvj-relation 0.00002
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8
)

run_arm() {
  local label=$1 weight=$2 save=$3 log=$4
  shift 4
  local -a migration_args=("$@") status
  printf 'arm %s: source step %s -> %s (%s updates), World weight=%s\n' \
    "$label" "$EXPECTED_SOURCE_STEP" "$TARGET_STEP" "$UPDATES" "$weight"
  [[ "$(sha256_file "$SOURCE")" == "$source_sha_before" ]] || fail "source hash trap fired before arm $label"
  set +e
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B train.py "${COMMON[@]}" \
    --wmrm-world-weight "$weight" \
    --steps "$UPDATES" --save-every 0 \
    --save "$save" --resume-exact "$SOURCE" \
    "${migration_args[@]}" 2>&1 | tee "$log"
  status=("${PIPESTATUS[@]}")
  set -e
  [[ "${status[0]}" -eq 0 ]] || fail "trainer failed for arm $label with status ${status[0]}"
  [[ "${status[1]}" -eq 0 ]] || fail "tee failed for arm $label; log is incomplete"
  [[ "$(sha256_file "$SOURCE")" == "$source_sha_before" ]] || fail "source hash trap fired after arm $label"
}

verify_final() {
  local label=$1 path=$2 expected_weight=$3 migration=$4
  "$PY" -B - "$label" "$path" "$TARGET_STEP" "$expected_weight" "$SOURCE" "$source_sha_before" "$migration" "$ANALYZER" <<'PY'
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import torch

label, checkpoint_arg, expected_step_arg, expected_weight_arg, source_arg, source_sha, migration, analyzer_arg = sys.argv[1:]
checkpoint = Path(checkpoint_arg).resolve(strict=True)
source = Path(source_arg).resolve(strict=True)
digest = sha256()
with source.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 << 20), b""):
        digest.update(chunk)
if digest.hexdigest() != source_sha:
    raise SystemExit("source checkpoint changed during paired run")
spec = spec_from_file_location("world_weight_ab_analyzer", analyzer_arg)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load paired analyzer")
module = module_from_spec(spec)
spec.loader.exec_module(module)
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
module.validate_checkpoint_payload(
    payload,
    expected_step=int(expected_step_arg),
    expected_weight=float(expected_weight_arg),
    label=f"arm {label}",
)
print(f"arm {label} verified: {checkpoint} weight={expected_weight_arg} migration={migration}", flush=True)
PY
}

verify_source
require_no_active_train
require_idle_gpu
run_arm A 1.0 "$A_SAVE" "$A_LOG"
verify_final A "$A_SAVE" 1.0 none
require_no_active_train
require_idle_gpu
run_arm B 0.5 "$B_SAVE" "$B_LOG" \
  --resume-exact-contract-migration "$MIGRATION_ID"
verify_final B "$B_SAVE" 0.5 "$MIGRATION_ID"

set +e
"$PY" -B "$ANALYZER" \
  --a-log "$A_LOG" --b-log "$B_LOG" \
  --a-report "$A_REPORT" --b-report "$B_REPORT" --output "$PAIR_REPORT" \
  --start-step 12011
analysis_status=$?
set -e
case "$analysis_status" in
  0) printf 'paired protocol PASS: %s\n' "$PAIR_REPORT" ;;
  2) printf 'paired protocol NO-GO: %s\n' "$PAIR_REPORT" >&2 ;;
  *) fail "paired analyzer failed with status $analysis_status" ;;
esac
exit "$analysis_status"
