#!/usr/bin/env bash
# Safe short validation: immutable step-12010 candidate -> static constraint weight 2.0, World weight fixed at 1.0, 64 updates.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_stable_detach_v1.candidate_diag.pt
EXPECTED_SOURCE_REALPATH=/home/ryan/Documents/robot/ORA0/checkpoints/mw_hard2_wam4va_visualmotion_stable_detach_v1.candidate_diag.pt
EXPECTED_SOURCE_SHA256=f580caa4c1588b2a9807f9b0ab746ac54259eaaa482cea16ce5001c30a382f11
EXPECTED_SOURCE_STEP=12010; TARGET_STEP=12074; UPDATES=64
MIGRATION_ID=wmrm_static_constraint_weight_4_to_2_v1
FAMILY=mw_hard2_wam4va_visualmotion_staticweight2_validation64_v1
SAVE=checkpoints/${FAMILY}.pt; LOG=logs/${FAMILY}.train.log; REPORT=diagnostics/${FAMILY}.json
ANALYZER=scripts/analyze_visualmotion_static_weight_2_validation.py
LOCK=/tmp/ora0_wam4va_visualmotion_train.lock
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt; SPLIT=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json
fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }; sha(){ sha256sum -- "$1" | cut -d' ' -f1; }
[[ $# -eq 0 ]] || { printf 'usage: bash %s\n' "$0" >&2; exit 2; }; command -v flock >/dev/null || fail 'flock is required'; exec 9>"$LOCK"; flock -n 9 || fail 'another visual-motion trainer owns the exclusive trainer lock'
for path in "$PY" "$SOURCE" "$DINO" "$DATA" "$SPLIT" "$ANALYZER" train.py; do [[ -f "$path" ]] || fail "missing required file: $path"; done
mkdir -p checkpoints logs diagnostics; for path in "$SAVE" "$LOG" "$REPORT"; do [[ ! -e "$path" ]] || fail "refusing to overwrite immutable output: $path"; done
available_kib=$(df --output=avail -k checkpoints | tail -n 1 | tr -d '[:space:]'); [[ "$available_kib" =~ ^[0-9]+$ ]] || fail 'cannot determine checkpoint filesystem free space'; ((available_kib >= 8 * 1024 * 1024)) || fail 'at least 8 GiB free is required'
source_sha_before=$(sha "$SOURCE"); [[ "$(realpath -e -- "$SOURCE")" == "$EXPECTED_SOURCE_REALPATH" ]] || fail 'source realpath mismatch'; [[ "$source_sha_before" == "$EXPECTED_SOURCE_SHA256" ]] || fail 'source SHA-256 mismatch'
require_no_active_train(){ "$PY" -B - <<'PY'
from pathlib import Path
import os
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try: argv=[x.decode('utf-8','replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except OSError: continue
    if any(Path(x).name=='train.py' for x in argv[1:]): raise SystemExit(f'active train.py pid={p.name}')
print('trainer process check: idle')
PY
}
require_idle_gpu(){ command -v nvidia-smi >/dev/null || fail 'nvidia-smi is required'; local apps; apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || fail 'cannot query GPU'; [[ -z "${apps//[[:space:]]/}" ]] || fail "GPU has active compute processes: $apps"; printf '%s\n' 'GPU compute check: idle (display processes allowed)'; }
verify_source(){ "$PY" -B - "$SOURCE" "$EXPECTED_SOURCE_STEP" "$EXPECTED_SOURCE_SHA256" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys, torch
p=Path(sys.argv[1]).resolve(strict=True); h=sha256(p.read_bytes()).hexdigest()
if h != sys.argv[3]: raise SystemExit('source hash mismatch')
x=torch.load(p,map_location='cpu',weights_only=True)
required={'model','optimizer_state','sampler_state','rng_state','exact_run_contract','exact_resume_version','global_step'}
if not isinstance(x,dict) or required-x.keys(): raise SystemExit('source lacks exact-resume state')
if x['global_step'] != int(sys.argv[2]) or x['exact_resume_version'] != 2: raise SystemExit('source step/version mismatch')
if x['optimizer_state'].get('kind') != 'adamw': raise SystemExit('source optimizer is not AdamW')
args=x['exact_run_contract'].get('arguments',{})
if args.get('wmrm_world_weight') != 1.0 or args.get('wmrm_static_constraint_weight', 4.0) != 4.0: raise SystemExit('source endpoint weights are not world=1.0/static=4.0')
print('source verified')
PY
}
COMMON=(--data "$DATA" --world-split-manifest "$SPLIT" --visual-world-supervision --world-action-rank-stage cycle --dino-main-vision --dino-dense-metric --main-vision-checkpoint "$DINO" --main-vision-grid 16 --main-vision-frames 4 --main-vision-temporal --main-vision-temporal-scale 1.0 --main-vision-encode-batch 8 --metric-geometry-inject --wam4va --wmrm-inject all --wmrm-target dino --wmrm-cycle-steps 6 --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 --wmrm-predictor st_blocks --wmrm-predictor-depth 6 --wmrm-predictor-width 384 --wmrm-predictor-heads 12 --wmrm-detach-proposal-stage-state --single-task --task-sampling balanced --task-locality-block-batches 4 --batch-size 3 --sequence-length 4 --min-sequence-length 4 --num-workers 0 --lr 0.0001 --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 --va-attention-backend auto --flow-cond adaln --flow-layers 6 --flow-steps 8 --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 --mtvj-train-relation --lr-mtvj-relation 0.00002 --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8)
verify_source; require_no_active_train; require_idle_gpu
[[ "$(sha "$SOURCE")" == "$source_sha_before" ]] || fail 'source hash trap fired before arm'
set +e; PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u -B train.py "${COMMON[@]}" --wmrm-world-weight 1.0 --wmrm-static-constraint-weight 2.0 --steps "$UPDATES" --save-every 0 --save "$SAVE" --resume-exact "$SOURCE" --resume-exact-contract-migration "$MIGRATION_ID" 2>&1 | tee "$LOG"; status=(${PIPESTATUS[@]}); set -e
[[ "${status[0]}" -eq 0 ]] || fail "trainer failed with status ${status[0]}"; [[ "${status[1]}" -eq 0 ]] || fail 'tee failed'; [[ "$(sha "$SOURCE")" == "$source_sha_before" ]] || fail 'source hash trap fired after arm'
[[ -f "$SAVE" ]] || fail 'trainer did not produce final checkpoint'
set +e; "$PY" -B "$ANALYZER" --log "$LOG" --checkpoint "$SAVE" --report "$REPORT"; analysis_status=$?; set -e
[[ "$(sha "$SOURCE")" == "$source_sha_before" ]] || fail 'source hash trap fired after final validation'
exit "$analysis_status"
