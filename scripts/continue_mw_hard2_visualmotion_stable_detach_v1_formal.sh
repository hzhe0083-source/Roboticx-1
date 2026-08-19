#!/usr/bin/env bash
# Formal stable-detach continuation: immutable candidate step 12010 -> step 20000.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_stable_detach_v1.candidate_diag.pt
EXPECTED_SOURCE_REALPATH=/home/ryan/Documents/robot/ORA0/checkpoints/mw_hard2_wam4va_visualmotion_stable_detach_v1.candidate_diag.pt
EXPECTED_SOURCE_SHA256=f580caa4c1588b2a9807f9b0ab746ac54259eaaa482cea16ce5001c30a382f11
EXPECTED_SOURCE_STEP=12010
TARGET_STEP=20000
ADDITIONAL_STEPS=7990
MIGRATION_ID=wmrm_static_constraint_weight_4_to_2_v1
STATIC_CONSTRAINT_WEIGHT=2.0
WORLD_WEIGHT=1.0
RUN_ID=mw_hard2_wam4va_visualmotion_stable_detach_static2_v1.formal_12010_to_20000
SAVE=checkpoints/${RUN_ID}.pt
LOG=logs/${RUN_ID}.train.log
LOCK=/tmp/ora0_wam4va_visualmotion_train.lock
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
SPLIT=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null || fail "flock is required"
exec 9>"$LOCK"
flock -n 9 || fail "another visual-motion trainer owns the exclusive trainer lock"

for path in "$PY" "$SOURCE" "$DINO" "$DATA" "$SPLIT" train.py; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done

sha256_file() { sha256sum -- "$1" | cut -d' ' -f1; }
source_sha_before=$(sha256_file "$SOURCE")
verify_source_unchanged_on_exit() {
  local status=$?
  trap - EXIT
  [[ "$(sha256_file "$SOURCE")" == "$source_sha_before" ]] || {
    printf 'ERROR: source checkpoint was modified\n' >&2
    exit 1
  }
  exit "$status"
}
trap verify_source_unchanged_on_exit EXIT

verify_source() {
  local realpath actual_sha
  realpath=$(realpath -e -- "$SOURCE") || fail "cannot resolve source checkpoint"
  [[ "$realpath" == "$EXPECTED_SOURCE_REALPATH" ]] || fail "source realpath mismatch: $realpath"
  actual_sha=$(sha256_file "$SOURCE")
  [[ "$actual_sha" == "$EXPECTED_SOURCE_SHA256" ]] || fail "source SHA-256 mismatch: $actual_sha"
  "$PY" -B - "$SOURCE" "$EXPECTED_SOURCE_STEP" <<'PY'
import sys
import torch
from pathlib import Path

path = Path(sys.argv[1]).resolve(strict=True)
expected_step = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=True)
required = {"model", "optimizer_state", "sampler_state", "rng_state", "exact_run_contract", "exact_resume_version", "global_step"}
missing = sorted(required - payload.keys())
if missing:
    raise SystemExit(f"source lacks exact-resume state: {missing}")
if payload["exact_resume_version"] != 2 or payload["global_step"] != expected_step:
    raise SystemExit("source exact-resume version or global step mismatch")
optimizer = payload["optimizer_state"]
if optimizer.get("kind") != "adamw" or "state_dict" not in optimizer:
    raise SystemExit("source optimizer is not exact AdamW state")
sampler = payload["sampler_state"]
for key, expected in {"sampler_contract_version": 3, "batch_size": 3, "block_batches": 4, "sampling_mode": "balanced", "seed": 0, "active_tasks": [0, 16]}.items():
    if sampler.get(key) != expected:
        raise SystemExit(f"source sampler {key}={sampler.get(key)!r} != {expected!r}")
for key in ("epoch", "batch_cursor", "dataset_fingerprint", "task_weights"):
    if key not in sampler:
        raise SystemExit(f"source sampler lacks {key}")
rng = payload["rng_state"]
if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
    raise SystemExit("source RNG state is incomplete")
contract = payload["exact_run_contract"]
args = contract.get("arguments") or {}
model = contract.get("model_config") or {}
if args.get("wmrm_detach_proposal_stage_state") is not True or model.get("wmrm_detach_proposal_stage_state") is not True:
    raise SystemExit("source detach contract is not exactly true")
if args.get("wmrm_world_weight") != 1.0:
    raise SystemExit("source World weight is not exactly 1.0")
if args.get("wmrm_static_constraint_weight", 4.0) != 4.0:
    raise SystemExit("source static constraint weight is not exactly 4.0")
static = (payload.get("training_contract") or {}).get("world_static_copy_constraint") or {}
if static.get("weight") != 4.0:
    raise SystemExit("source training contract static constraint weight is not exactly 4.0")
if args.get("max_gradient_norm") is not None:
    raise SystemExit("source max_gradient_norm is not exactly None")
for key, expected in {
    "main_vision_backbone": "dinov2_vitl14_reg4",
    "main_vision_dim": 1024,
    "main_vision_frames": 4,
    "main_vision_grid": 16,
    "wmrm": True,
    "wmrm_inject": "all",
    "wmrm_target": "dino",
    "wmrm_predictor": "st_blocks",
    "wmrm_predictor_depth": 6,
    "wmrm_predictor_width": 384,
    "wmrm_predictor_heads": 12,
}.items():
    if model.get(key) != expected:
        raise SystemExit(f"source model {key}={model.get(key)!r} != {expected!r}")
print(f"source verified: {path} global_step={expected_step} AdamW detach=true max_gradient_norm=None", flush=True)
PY
}

refuse_existing_outputs() {
  local -a existing=()
  shopt -s nullglob
  for path in checkpoints/${RUN_ID}* logs/${RUN_ID}* diagnostics/${RUN_ID}*; do existing+=("$path"); done
  shopt -u nullglob
  ((${#existing[@]} == 0)) || { printf 'ERROR: refusing to overwrite output family %s\n' "$RUN_ID" >&2; printf '  %s\n' "${existing[@]}" >&2; exit 1; }
}

require_no_trainer() {
  "$PY" -B - <<'PY'
from pathlib import Path
import os
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
    try: argv = [x.decode('utf-8', 'replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    except OSError: continue
    if any(Path(arg).name == 'train.py' for arg in argv[1:]):
        raise SystemExit(f"trainer must be idle: pid={proc.name} command={' '.join(argv)[:300]}")
print('trainer process check: idle', flush=True)
PY
}

require_idle_gpu() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required for GPU idleness check"
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || fail "cannot query GPU compute processes"
  [[ -z "${apps//[[:space:]]/}" ]] || fail "GPU has active compute processes: ${apps//$'\n'/; }"
  "$PY" -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('exactly one idle CUDA device is required')
free, total = torch.cuda.mem_get_info()
if free < int(total * 0.85):
    raise SystemExit(f'GPU lacks headroom: free={free/2**30:.2f}GiB total={total/2**30:.2f}GiB')
print('GPU check: no compute processes; display use is allowed; at least 85% memory free', flush=True)
PY
}

verify_final() {
  "$PY" -B - "$SAVE" "$TARGET_STEP" "$SOURCE" "$source_sha_before" <<'PY'
import sys
from hashlib import sha256
from pathlib import Path
import torch
path, expected, source, source_sha = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2]), Path(sys.argv[3]).resolve(strict=True), sys.argv[4]
h = sha256()
with source.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
if h.hexdigest() != source_sha: raise SystemExit('source checkpoint changed during continuation')
p = torch.load(path, map_location='cpu', weights_only=True)
if p.get('global_step') != expected or p.get('exact_resume_version') != 2: raise SystemExit('final step/version mismatch')
for key in ('model','optimizer_state','sampler_state','rng_state','exact_run_contract'):
    if key not in p: raise SystemExit(f'final checkpoint lacks {key}')
if p['optimizer_state'].get('kind') != 'adamw' or 'state_dict' not in p['optimizer_state']: raise SystemExit('final optimizer is not exact AdamW state')
s = p['sampler_state']
for key, expected_value in {'sampler_contract_version':3, 'batch_size':3, 'block_batches':4, 'sampling_mode':'balanced', 'seed':0, 'active_tasks':[0,16]}.items():
    if s.get(key) != expected_value: raise SystemExit(f'final sampler {key} mismatch')
for key in ('epoch','batch_cursor','dataset_fingerprint','task_weights'):
    if key not in s: raise SystemExit(f'final sampler lacks {key}')
a = (p['exact_run_contract'].get('arguments') or {}); m = (p['exact_run_contract'].get('model_config') or {})
if a.get('wmrm_detach_proposal_stage_state') is not True or m.get('wmrm_detach_proposal_stage_state') is not True: raise SystemExit('final detach contract is not true')
if a.get('wmrm_world_weight') != 1.0: raise SystemExit('final World weight is not 1.0')
if a.get('wmrm_static_constraint_weight') != 2.0: raise SystemExit('final static constraint weight is not 2.0')
static = (p.get('training_contract') or {}).get('world_static_copy_constraint') or {}
if static.get('weight') != 2.0: raise SystemExit('final training contract static constraint weight is not 2.0')
if a.get('max_gradient_norm') is not None: raise SystemExit('final max_gradient_norm is not None')
for key, expected_value in {'main_vision_backbone':'dinov2_vitl14_reg4','main_vision_dim':1024,'main_vision_frames':4,'main_vision_grid':16,'wmrm':True,'wmrm_inject':'all','wmrm_target':'dino','wmrm_predictor':'st_blocks','wmrm_predictor_depth':6,'wmrm_predictor_width':384,'wmrm_predictor_heads':12}.items():
    if m.get(key) != expected_value: raise SystemExit(f'final model {key} mismatch')
if set(p['rng_state']) != {'python','numpy','torch_cpu','torch_cuda'}: raise SystemExit('final RNG state incomplete')
print(f'final checkpoint verified: {path} global_step={expected} AdamW detach=true max_gradient_norm=None', flush=True)
PY
}

verify_source
refuse_existing_outputs
require_no_trainer
require_idle_gpu
available_kib=$(df --output=avail -k checkpoints | tail -n 1 | tr -d '[:space:]')
[[ "$available_kib" =~ ^[0-9]+$ ]] || fail "cannot determine checkpoint filesystem free space"
((available_kib >= 14 * 1024 * 1024)) || fail "at least 14 GiB free is required"
mkdir -p checkpoints logs
[[ "$SAVE" != "$SOURCE" ]] || fail 'source and destination must differ'
printf 'formal controlled migration: global_step %s -> %s (%s updates); static constraint 4.0 -> %s; World weight fixed at %s; rolling save every 1000 global updates; no evaluator\n' "$EXPECTED_SOURCE_STEP" "$TARGET_STEP" "$ADDITIONAL_STEPS" "$STATIC_CONSTRAINT_WEIGHT" "$WORLD_WEIGHT"
set +e
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  --data "$DATA" --world-split-manifest "$SPLIT" --visual-world-supervision --world-action-rank-stage cycle \
  --dino-main-vision --dino-dense-metric --main-vision-checkpoint "$DINO" --main-vision-grid 16 --main-vision-frames 4 \
  --main-vision-temporal --main-vision-temporal-scale 1.0 --main-vision-encode-batch 8 --metric-geometry-inject \
  --wam4va --wmrm-inject all --wmrm-target dino --wmrm-world-weight "$WORLD_WEIGHT" --wmrm-static-constraint-weight "$STATIC_CONSTRAINT_WEIGHT" --wmrm-cycle-steps 6 \
  --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 --wmrm-predictor st_blocks --wmrm-predictor-depth 6 \
  --wmrm-predictor-width 384 --wmrm-predictor-heads 12 --wmrm-detach-proposal-stage-state \
  --single-task --task-sampling balanced --task-locality-block-batches 4 --batch-size 3 --sequence-length 4 --min-sequence-length 4 \
  --num-workers 0 --lr 0.0001 --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 --va-attention-backend auto \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 --steps "$ADDITIONAL_STEPS" --save-every 1000 \
  --save "$SAVE" --resume-exact "$SOURCE" --resume-exact-contract-migration "$MIGRATION_ID" 2>&1 | tee "$LOG"
status=("${PIPESTATUS[@]}")
set -e
[[ "${status[0]}" -eq 0 ]] || exit "${status[0]}"
[[ "${status[1]}" -eq 0 ]] || fail 'tee failed; log is incomplete'
[[ "$(sha256_file "$SOURCE")" == "$source_sha_before" ]] || fail 'source checkpoint was modified'
verify_final
printf 'formal continuation complete: global_step=%s; log=%s; evaluator not run\n' "$TARGET_STEP" "$LOG"
