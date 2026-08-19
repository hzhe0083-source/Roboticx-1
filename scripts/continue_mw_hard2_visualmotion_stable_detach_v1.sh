#!/usr/bin/env bash
# Safe, short continuation protocol for the immutable joint_v1 step-12000 source.
#
# baseline-replay: exact-compatible replay without any new semantic flag.
# stabilized-candidate: adds --wmrm-detach-proposal-stage-state, but only through
# the explicitly recognized exact-contract migration interface below.
# Neither mode is a formal continuation: both stop at diagnostic step 12010.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE=${1:-}
PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_joint_v1.pt
EXPECTED_SOURCE_REALPATH=/home/ryan/Documents/robot/ORA0/checkpoints/mw_hard2_wam4va_visualmotion_joint_v1.pt
EXPECTED_SOURCE_SHA256=0b7438c0d4f681787043a1703fc754ba977b11891419a633cc018dfae6458113
EXPECTED_SOURCE_STEP=12000
DIAGNOSTIC_TARGET_STEP=12010
FAMILY=mw_hard2_wam4va_visualmotion_stable_detach_v1
BASELINE_RUN_ID=${FAMILY}.baseline_replay_diag
CANDIDATE_RUN_ID=${FAMILY}.candidate_diag
LOCK=/tmp/ora0_wam4va_visualmotion_train.lock
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
SPLIT=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json
MIGRATION_FLAG=--resume-exact-contract-migration
MIGRATION_ID=wmrm_detach_proposal_stage_state_v1

usage() {
  echo "usage: bash $0 {baseline-replay|stabilized-candidate}" >&2
  echo "Both modes stop at diagnostic global_step ${DIAGNOSTIC_TARGET_STEP}; neither starts formal continuation." >&2
  exit 2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage
case "$MODE" in
  baseline-replay|stabilized-candidate) ;;
  *) usage ;;
esac

command -v flock >/dev/null || die "flock is required"
exec 9>"$LOCK"
flock -n 9 || die "another visual-motion trainer owns the exclusive trainer lock"

for path in "$PY" "$SOURCE" "$DINO" "$DATA" "$SPLIT" train.py; do
  [[ -f "$path" ]] || die "missing required file: $path"
done

checkpoint_sha256() {
  local output
  output=$(sha256sum -- "$1") || die "cannot hash checkpoint: $1"
  printf '%s\n' "${output%% *}"
}

verify_source_checkpoint() {
  local actual_realpath actual_sha
  actual_realpath=$(realpath -e -- "$SOURCE") || die "cannot resolve source checkpoint"
  [[ "$actual_realpath" == "$EXPECTED_SOURCE_REALPATH" ]] || \
    die "source path mismatch: $actual_realpath"
  actual_sha=$(checkpoint_sha256 "$SOURCE")
  [[ "$actual_sha" == "$EXPECTED_SOURCE_SHA256" ]] || \
    die "source SHA-256 mismatch: $actual_sha"

  "$PY" -B - "$SOURCE" "$EXPECTED_SOURCE_STEP" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1]).resolve(strict=True)
expected_step = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=True)
required = {
    "model", "optimizer_state", "sampler_state", "rng_state",
    "exact_run_contract", "exact_resume_version", "global_step",
}
missing = sorted(required - payload.keys())
if missing:
    raise SystemExit(f"source lacks exact-resume state: {missing}")
if payload["exact_resume_version"] != 2:
    raise SystemExit("source exact_resume_version is not 2")
if payload["global_step"] != expected_step:
    raise SystemExit(
        f"source global_step={payload['global_step']!r} != {expected_step}"
    )
optimizer = payload["optimizer_state"]
if not isinstance(optimizer, dict) or optimizer.get("kind") != "adamw" or "state_dict" not in optimizer:
    raise SystemExit("source optimizer exact state is not AdamW")
sampler = payload["sampler_state"]
expected_sampler = {
    "sampler_contract_version": 3,
    "batch_size": 3,
    "block_batches": 4,
    "sampling_mode": "balanced",
    "seed": 0,
    "active_tasks": [0, 16],
}
for key, expected in expected_sampler.items():
    if sampler.get(key) != expected:
        raise SystemExit(f"source sampler {key}={sampler.get(key)!r} != {expected!r}")
for key in ("epoch", "batch_cursor", "dataset_fingerprint", "task_weights"):
    if key not in sampler:
        raise SystemExit(f"source sampler lacks exact field {key}")
rng = payload["rng_state"]
if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
    raise SystemExit("source RNG state is incomplete")
arguments = payload["exact_run_contract"].get("arguments") or {}
expected_arguments = {
    "batch_size": 3,
    "num_workers": 0,
    "seed": 0,
    "task_sampling": "balanced",
    "task_locality_block_batches": 4,
    "world_action_rank_stage": "cycle",
    "wmrm": True,
    "wmrm_inject": "all",
    "wmrm_target": "dino",
}
for key, expected in expected_arguments.items():
    if arguments.get(key) != expected:
        raise SystemExit(f"source exact contract {key}={arguments.get(key)!r} != {expected!r}")
print(f"source verified: {path} global_step={expected_step}", flush=True)
PY
}

refuse_output_family() {
  local run_id=$1 path
  local -a existing=()
  shopt -s nullglob
  for path in checkpoints/"${run_id}"* logs/"${run_id}"* diagnostics/"${run_id}"*; do
    existing+=("$path")
  done
  shopt -u nullglob
  ((${#existing[@]} == 0)) || {
    printf 'refusing to overwrite immutable diagnostic family %s:\n' "$run_id" >&2
    printf '  %s\n' "${existing[@]}" >&2
    exit 1
  }
}

require_no_trainer() {
  "$PY" -B - <<'PY'
from pathlib import Path
import os

matches = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        argv = [part.decode("utf-8", "replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
    except OSError:
        continue
    if any(Path(arg).name == "train.py" for arg in argv[1:]):
        matches.append((proc.name, " ".join(argv)))
if matches:
    for pid, command in matches:
        print(f"active trainer pid={pid}: {command[:300]}")
    raise SystemExit("trainer must be idle")
print("trainer process check: idle", flush=True)
PY
}

require_idle_gpu() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required to prove GPU idleness"
  local compute_apps
  compute_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || \
    die "cannot query GPU compute processes"
  [[ -z "${compute_apps//[[:space:]]/}" ]] || \
    die "GPU has active compute processes: ${compute_apps//$'\n'/; }"
  "$PY" -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one idle CUDA device is required")
free_bytes, total_bytes = torch.cuda.mem_get_info()
if free_bytes < int(total_bytes * 0.85):
    raise SystemExit(
        f"GPU lacks the required diagnostic headroom: free={free_bytes / 2**30:.2f}GiB "
        f"total={total_bytes / 2**30:.2f}GiB"
    )
print("GPU check: no compute processes and at least 85% memory free", flush=True)
PY
}

candidate_migration_args() {
  grep -Fq -- '"--wmrm-detach-proposal-stage-state"' train.py || \
    die "train.py does not support intended --wmrm-detach-proposal-stage-state; candidate refused"
  grep -Fq -- "\"$MIGRATION_FLAG\"" train.py || \
    die "train.py exposes no controlled $MIGRATION_FLAG mechanism; candidate refused"
  grep -Fq -- "\"$MIGRATION_ID\"" train.py || \
    die "train.py does not recognize controlled migration $MIGRATION_ID; candidate refused"
  extra_args=(--wmrm-detach-proposal-stage-state "$MIGRATION_FLAG" "$MIGRATION_ID")
}

verify_diagnostic_checkpoint() {
  local checkpoint=$1 expected_step=$2 expected_detach=$3
  "$PY" -B - "$checkpoint" "$expected_step" "$expected_detach" "$SOURCE" "$source_sha_before" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1]).resolve(strict=True)
expected = int(sys.argv[2])
expected_detach = sys.argv[3] == "true"
source = Path(sys.argv[4]).resolve(strict=True)
expected_source_sha = sys.argv[5]
h = sha256()
with source.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
if h.hexdigest() != expected_source_sha:
    raise SystemExit("source SHA changed before diagnostic contract verification")
payload = torch.load(path, map_location="cpu", weights_only=True)
required = ("optimizer_state", "sampler_state", "rng_state", "exact_run_contract")
if payload.get("global_step") != expected:
    raise SystemExit(f"diagnostic global_step={payload.get('global_step')!r} != {expected}")
if payload.get("exact_resume_version") != 2 or any(key not in payload for key in required):
    raise SystemExit("diagnostic checkpoint lacks exact continuation state")
contract = payload["exact_run_contract"]
arguments = contract.get("arguments") or {}
model_config = contract.get("model_config") or {}
for location, actual in (
    ("arguments", arguments.get("wmrm_detach_proposal_stage_state")),
    ("model_config", model_config.get("wmrm_detach_proposal_stage_state")),
):
    if actual is not expected_detach:
        raise SystemExit(
            f"diagnostic {location} detach contract={actual!r} != {expected_detach!r}"
        )
if arguments.get("max_gradient_norm") is not None:
    raise SystemExit("diagnostic max_gradient_norm contract is not the known default None")
if "resume_exact_contract_migration" in arguments:
    raise SystemExit("operational migration ID leaked into semantic run contract")
print(f"diagnostic verified: {path} global_step={expected}", flush=True)
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
  --wmrm-world-weight 1.0 --wmrm-cycle-steps 6
  --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16
  --wmrm-predictor st_blocks --wmrm-predictor-depth 6
  --wmrm-predictor-width 384 --wmrm-predictor-heads 12
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

verify_source_checkpoint
source_sha_before=$(checkpoint_sha256 "$SOURCE")
verify_source_unchanged() {
  local status=$?
  trap - EXIT
  [[ "$(checkpoint_sha256 "$SOURCE")" == "$source_sha_before" ]] || \
    die "source checkpoint was modified"
  exit "$status"
}
trap verify_source_unchanged EXIT
extra_args=()
case "$MODE" in
  baseline-replay)
    run_id=$BASELINE_RUN_ID
    expected_detach=false
    ;;
  stabilized-candidate)
    run_id=$CANDIDATE_RUN_ID
    expected_detach=true
    candidate_migration_args
    ;;
esac

refuse_output_family "$run_id"
require_no_trainer
require_idle_gpu
mkdir -p checkpoints logs diagnostics
save=checkpoints/${run_id}.pt
log=logs/${run_id}.train_step12000_to_step12010.log
[[ "$save" != "$SOURCE" ]] || die "source and destination checkpoint must differ"

# Ten one-step immutable copies (_s12001.._s12010) give dense initial evidence.
echo "diagnostic only: mode=$MODE step12000 -> step12010; formal continuation is intentionally out of scope"
set +e
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py "${COMMON[@]}" "${extra_args[@]}" \
  --steps 10 --save-every 1 --save-step-copies \
  --save "$save" --resume-exact "$SOURCE" 2>&1 | tee "$log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
[[ "${pipeline_status[0]}" -eq 0 ]] || exit "${pipeline_status[0]}"
[[ "${pipeline_status[1]}" -eq 0 ]] || die "tee failed; diagnostic log is incomplete"
[[ "$(checkpoint_sha256 "$SOURCE")" == "$source_sha_before" ]] || \
  die "source checkpoint was modified"

for ((step = EXPECTED_SOURCE_STEP + 1; step <= DIAGNOSTIC_TARGET_STEP; step++)); do
  archive=checkpoints/${run_id}_s${step}.pt
  [[ -s "$archive" ]] || die "missing immutable diagnostic archive: $archive"
  verify_diagnostic_checkpoint "$archive" "$step" "$expected_detach"
done
verify_diagnostic_checkpoint "$save" "$DIAGNOSTIC_TARGET_STEP" "$expected_detach"
echo "diagnostic milestone reached at step ${DIAGNOSTIC_TARGET_STEP}; STOP before formal continuation"
