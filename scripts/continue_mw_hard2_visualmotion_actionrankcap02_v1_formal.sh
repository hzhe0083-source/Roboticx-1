#!/usr/bin/env bash
# Formal exact continuation: immutable static2/cap0.2 step 12330 -> step 20000.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_actionrankcap02_validation256_v1.pt
EXPECTED_SOURCE_REALPATH=/home/ryan/Documents/robot/ORA0/checkpoints/mw_hard2_wam4va_visualmotion_actionrankcap02_validation256_v1.pt
EXPECTED_SOURCE_SHA256=f75693d9f5ac449e5a6627a03f8ab57e15ff89aad30047719db5a31227b9f334
EXPECTED_EXACT_CONTRACT_SHA256=c1170d61398daf0687521a881340b85a5457e5020d0ce1a58d73406877ce0a52
EXPECTED_TRAINING_CONTRACT_SHA256=732ff34c104a29c3c985e35b75ec149887a8a740e6a58e6cb13f7d67ff7c2cc5
EXPECTED_DATASET_FINGERPRINT=41181fc115389d76abb00f054cbd8b318bd534204a27c08fd8163c928d662e45
EXPECTED_SOURCE_STEP=12330
TARGET_STEP=20000
ADDITIONAL_STEPS=7670
SAVE_EVERY=500
FIRST_SAVE_STEP=12500
EXPECTED_ARCHIVE_COUNT=16
DISK_RESERVE_BYTES=$((4 * 1024 * 1024 * 1024))
FAMILY=mw_hard2_wam4va_visualmotion_actionrankcap02_v1.formal_12330_to_20000
SAVE=checkpoints/${FAMILY}.pt
LOG=logs/${FAMILY}.train.log
LOCK=/tmp/ora0_wam4va_visualmotion_train.lock
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
SPLIT=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha(){ sha256sum -- "$1" | cut -d' ' -f1; }
[[ $# -eq 0 ]] || { printf 'usage: bash %s\n' "$0" >&2; exit 2; }
command -v flock >/dev/null || fail 'flock is required'
exec 9>"$LOCK"
flock -n 9 || fail 'another visual-motion trainer owns the exclusive global lock'
for path in "$PY" "$SOURCE" "$DINO" "$DATA" "$SPLIT" train.py; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done

source_sha_before=$(sha "$SOURCE")
verify_source_unchanged_on_exit(){
  local exit_status=$?
  trap - EXIT
  [[ "$(sha "$SOURCE")" == "$source_sha_before" ]] || {
    printf 'ERROR: immutable source checkpoint was modified\n' >&2
    exit 1
  }
  exit "$exit_status"
}
trap verify_source_unchanged_on_exit EXIT

verify_checkpoint(){
  local checkpoint=$1 expected_step=$2 label=$3
  "$PY" -B - "$checkpoint" "$expected_step" "$EXPECTED_EXACT_CONTRACT_SHA256" \
    "$EXPECTED_TRAINING_CONTRACT_SHA256" "$EXPECTED_DATASET_FINGERPRINT" "$label" <<'PY'
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1]).resolve(strict=True)
expected_step = int(sys.argv[2])
expected_exact_digest, expected_training_digest, expected_dataset, label = sys.argv[3:7]
payload = torch.load(path, map_location="cpu", weights_only=True)
required = {
    "model", "optimizer_state", "sampler_state", "rng_state",
    "exact_run_contract", "training_contract", "exact_resume_version", "global_step",
}
if not isinstance(payload, dict) or required - payload.keys():
    raise SystemExit(f"{label} lacks full exact-resume state: {sorted(required - payload.keys())}")
if payload["global_step"] != expected_step or payload["exact_resume_version"] != 2:
    raise SystemExit(f"{label} step/version mismatch")

def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
if digest(payload["exact_run_contract"]) != expected_exact_digest:
    raise SystemExit(f"{label} full exact run contract mismatch")
if digest(payload["training_contract"]) != expected_training_digest:
    raise SystemExit(f"{label} full training contract mismatch")

model = payload["model"]
if not isinstance(model, dict) or not model:
    raise SystemExit(f"{label} model state is absent")
for name, value in model.items():
    if not isinstance(name, str) or not isinstance(value, torch.Tensor):
        raise SystemExit(f"{label} model state entry is malformed: {name!r}")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise SystemExit(f"{label} model state contains non-finite values: {name}")

optimizer = payload["optimizer_state"]
if not isinstance(optimizer, dict) or optimizer.get("kind") != "adamw":
    raise SystemExit(f"{label} optimizer is not AdamW")
state_dict = optimizer.get("state_dict")
if not isinstance(state_dict, dict) or set(state_dict) != {"state", "param_groups"}:
    raise SystemExit(f"{label} AdamW state_dict is incomplete")
if not isinstance(state_dict["state"], dict) or not state_dict["state"]:
    raise SystemExit(f"{label} AdamW moment state is absent")
if not isinstance(state_dict["param_groups"], list) or not state_dict["param_groups"]:
    raise SystemExit(f"{label} AdamW parameter groups are absent")
for parameter_id, state in state_dict["state"].items():
    if not isinstance(state, dict):
        raise SystemExit(f"{label} AdamW state {parameter_id!r} is malformed")
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise SystemExit(f"{label} AdamW state is non-finite: {parameter_id!r}.{key}")
for group in state_dict["param_groups"]:
    if not isinstance(group, dict) or not isinstance(group.get("params"), list) or not group["params"]:
        raise SystemExit(f"{label} AdamW parameter group is malformed")
    for key in ("lr", "weight_decay", "eps"):
        if key not in group or not math.isfinite(float(group[key])):
            raise SystemExit(f"{label} AdamW group has invalid {key}")

sampler = payload["sampler_state"]
expected_sampler = {
    "sampler_contract_version": 3, "batch_size": 3, "block_batches": 4,
    "sampling_mode": "balanced", "seed": 0, "active_tasks": [0, 16],
    "dataset_fingerprint": expected_dataset, "task_weights": [1.0, 1.0],
}
if not isinstance(sampler, dict):
    raise SystemExit(f"{label} sampler state is absent")
for key, expected in expected_sampler.items():
    if sampler.get(key) != expected:
        raise SystemExit(f"{label} sampler {key}={sampler.get(key)!r} != {expected!r}")
for key in ("epoch", "batch_cursor"):
    if type(sampler.get(key)) is not int or sampler[key] < 0:
        raise SystemExit(f"{label} sampler {key} is invalid")

rng = payload["rng_state"]
if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
    raise SystemExit(f"{label} RNG state is incomplete")
if not isinstance(rng["torch_cpu"], torch.Tensor) or rng["torch_cpu"].numel() == 0:
    raise SystemExit(f"{label} CPU RNG state is invalid")
if not isinstance(rng["torch_cuda"], list) or not rng["torch_cuda"]:
    raise SystemExit(f"{label} CUDA RNG state is invalid")
if not all(isinstance(item, torch.Tensor) and item.numel() for item in rng["torch_cuda"]):
    raise SystemExit(f"{label} CUDA RNG entries are invalid")
if not isinstance(rng["python"], (list, tuple)) or not isinstance(rng["numpy"], dict):
    raise SystemExit(f"{label} Python/NumPy RNG state is invalid")

arguments = payload["exact_run_contract"].get("arguments") or {}
model_config = payload["exact_run_contract"].get("model_config") or {}
for key, expected in {
    "wmrm_action_rank_per_sample_cap": 0.2,
    "wmrm_static_constraint_weight": 2.0,
    "wmrm_world_weight": 1.0,
    "wmrm_detach_proposal_stage_state": True,
    "world_action_rank_stage": "cycle",
    "max_gradient_norm": None,
}.items():
    if arguments.get(key) != expected:
        raise SystemExit(f"{label} argument contract {key} mismatch")
if model_config.get("wmrm_detach_proposal_stage_state") is not True:
    raise SystemExit(f"{label} model detach contract mismatch")
ranking = payload["training_contract"].get("world_action_ranking") or {}
static = payload["training_contract"].get("world_static_copy_constraint") or {}
if ranking.get("per_sample_cap") != 0.2 or static.get("weight") != 2.0:
    raise SystemExit(f"{label} cap/static training contract mismatch")
print(f"{label} verified: {path} global_step={expected_step} full model/AdamW/sampler/RNG/contracts", flush=True)
PY
}

require_no_active_train(){ "$PY" -B - <<'PY'
from pathlib import Path
import os
matches = []
for process in Path('/proc').iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid():
        continue
    try:
        argv = [x.decode('utf-8', 'replace') for x in (process / 'cmdline').read_bytes().split(b'\0') if x]
    except OSError:
        continue
    if any(Path(arg).name == 'train.py' for arg in argv):
        matches.append((process.name, ' '.join(argv)))
if matches:
    raise SystemExit('; '.join(f'active train.py pid={pid}: {command[:300]}' for pid, command in matches))
print('exact train.py process scan: idle', flush=True)
PY
}

require_idle_gpu(){
  command -v nvidia-smi >/dev/null || fail 'nvidia-smi is required'
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || fail 'cannot query GPU compute processes'
  [[ -z "${apps//[[:space:]]/}" ]] || fail "GPU has active compute processes: ${apps//$'\n'/; }"
  "$PY" -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('exactly one CUDA device is required')
free, total = torch.cuda.mem_get_info()
if free < int(total * 0.85):
    raise SystemExit(f'GPU lacks headroom: free={free/2**30:.2f}GiB total={total/2**30:.2f}GiB')
print('GPU compute check: idle; display processes allowed; at least 85% memory free', flush=True)
PY
}

[[ "$(realpath -e -- "$SOURCE")" == "$EXPECTED_SOURCE_REALPATH" ]] || fail 'source realpath mismatch'
[[ "$source_sha_before" == "$EXPECTED_SOURCE_SHA256" ]] || fail "source SHA-256 mismatch: $source_sha_before"
verify_checkpoint "$SOURCE" "$EXPECTED_SOURCE_STEP" source

shopt -s nullglob
existing=(checkpoints/${FAMILY}* logs/${FAMILY}* diagnostics/${FAMILY}*)
shopt -u nullglob
((${#existing[@]} == 0)) || { printf 'ERROR: refusing to overwrite new output family %s\n' "$FAMILY" >&2; printf '  %s\n' "${existing[@]}" >&2; exit 1; }
require_no_active_train
require_idle_gpu
available_kib=$(df --output=avail -k checkpoints | tail -n 1 | tr -d '[:space:]')
[[ "$available_kib" =~ ^[0-9]+$ ]] || fail 'cannot determine checkpoint filesystem free space'
checkpoint_bytes=$(stat -c '%s' "$SOURCE")
expected_checkpoint_bytes=$((EXPECTED_ARCHIVE_COUNT * checkpoint_bytes))
# Each periodic save briefly needs the rolling file plus its immutable step copy;
# reserve headroom for that atomic write and unrelated filesystem activity.
required_free_bytes=$((expected_checkpoint_bytes + checkpoint_bytes + DISK_RESERVE_BYTES))
available_bytes=$((available_kib * 1024))
((available_bytes >= required_free_bytes)) || fail "insufficient checkpoint disk space for ${EXPECTED_ARCHIVE_COUNT} immutable archives plus rolling save: require ${required_free_bytes} bytes ($((required_free_bytes / 1024 / 1024)) MiB) free (checkpoint size=${checkpoint_bytes} bytes, reserve=${DISK_RESERVE_BYTES} bytes), have ${available_bytes} bytes ($((available_bytes / 1024 / 1024)) MiB)"
mkdir -p checkpoints logs
[[ "$SAVE" != "$SOURCE" ]] || fail 'source and destination must differ'
[[ "$(sha "$SOURCE")" == "$source_sha_before" ]] || fail 'source immutable trap fired before launch'

printf 'formal exact continuation: global_step %s -> %s (%s updates); static=2.0 cap=0.2 world=1.0 detach=true; rolling destination plus %s immutable checkpoint archives at global steps %s,%s,...,%s; no migration\n' \
  "$EXPECTED_SOURCE_STEP" "$TARGET_STEP" "$ADDITIONAL_STEPS" "$EXPECTED_ARCHIVE_COUNT" "$FIRST_SAVE_STEP" "$((FIRST_SAVE_STEP + SAVE_EVERY))" "$TARGET_STEP"
set +e
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  --data "$DATA" --world-split-manifest "$SPLIT" --visual-world-supervision --world-action-rank-stage cycle \
  --dino-main-vision --dino-dense-metric --main-vision-checkpoint "$DINO" --main-vision-grid 16 --main-vision-frames 4 \
  --main-vision-temporal --main-vision-temporal-scale 1.0 --main-vision-encode-batch 8 --metric-geometry-inject \
  --wam4va --wmrm-inject all --wmrm-target dino --wmrm-world-weight 1.0 --wmrm-static-constraint-weight 2.0 \
  --wmrm-action-rank-per-sample-cap 0.2 --wmrm-cycle-steps 6 --wmrm-map-size 16 --wmrm-map-channels 1024 \
  --wmrm-world-grid 16 --wmrm-predictor st_blocks --wmrm-predictor-depth 6 --wmrm-predictor-width 384 \
  --wmrm-predictor-heads 12 --wmrm-detach-proposal-stage-state --single-task --task-sampling balanced \
  --task-locality-block-batches 4 --batch-size 3 --sequence-length 4 --min-sequence-length 4 --num-workers 0 \
  --lr 0.0001 --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 --va-attention-backend auto \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 --steps "$ADDITIONAL_STEPS" --save-every "$SAVE_EVERY" \
  --save-step-copies --save "$SAVE" --resume-exact "$SOURCE" 2>&1 | tee "$LOG"
pipeline_status=("${PIPESTATUS[@]}")
set -e
((${#pipeline_status[@]} == 2)) || fail "incomplete pipeline status: ${pipeline_status[*]}"
[[ "${pipeline_status[0]}" -eq 0 ]] || fail "trainer failed with status ${pipeline_status[0]}"
[[ "${pipeline_status[1]}" -eq 0 ]] || fail "tee failed with status ${pipeline_status[1]}; log is incomplete"
[[ -f "$LOG" && -s "$LOG" ]] || fail 'tee log is missing or empty'
[[ "$(sha "$SOURCE")" == "$source_sha_before" ]] || fail 'source immutable trap fired after training'
[[ -f "$SAVE" ]] || fail 'trainer did not produce the rolling checkpoint'
verify_checkpoint "$SAVE" "$TARGET_STEP" final-rolling
archive_count=0
for ((step=FIRST_SAVE_STEP; step<=TARGET_STEP; step+=SAVE_EVERY)); do
  archive=${SAVE%.pt}_s${step}.pt
  [[ -f "$archive" ]] || fail "missing immutable checkpoint archive: $archive"
  verify_checkpoint "$archive" "$step" "archive-s${step}"
  archive_count=$((archive_count + 1))
done
((archive_count == EXPECTED_ARCHIVE_COUNT)) || fail "checkpoint archive count mismatch: expected $EXPECTED_ARCHIVE_COUNT, verified $archive_count"
[[ "$(sha "$SAVE")" == "$(sha "${SAVE%.pt}_s${TARGET_STEP}.pt")" ]] || fail 'final rolling checkpoint differs from immutable step-20000 archive'
printf 'formal continuation complete: global_step=%s; rolling_checkpoint=%s; immutable_archives=%s (%s..%s every %s); complete_tee_log=%s\n' \
  "$TARGET_STEP" "$SAVE" "$archive_count" "$FIRST_SAVE_STEP" "$TARGET_STEP" "$SAVE_EVERY" "$LOG"
