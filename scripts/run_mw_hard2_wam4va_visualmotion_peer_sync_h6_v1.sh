#!/usr/bin/env bash
# Scratch-only hard2 visual-motion lifecycle for the peer-synchronous H6 topology.
# Historical H48 runners and artifacts are read-only inputs; this runner owns a
# separate H6 data, split, checkpoint, diagnostic, and log namespace.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
ASSEMBLY_RAW=data/metaworld_longtraj_assembly-v3.pt
DOOR_RAW=data/metaworld_longtraj_door-unlock-v3.pt
ALLTASK_H48_REF=data/metaworld_longtraj_windows_h48.pt
EXPECTED_ASSEMBLY_SHA256=c61f3b2102dea781c9db2a73109472e6e181f46e33536879a5eab181ee190ea0
EXPECTED_DOOR_SHA256=309726cd679753633bf9bb658635b890affcc666523cb530bab62db4d9699bf1
EXPECTED_ALLTASK_H48_REF_SHA256=5adc69fce88cdfc5a62b0fa4e9da536d2368a81e6ebb5c23543bca2810ab19a4
SOURCE=data/hard2_peer_h6_source_v1.pt
TRAIN_DATA=data/hard2_peer_h6_train_v1.pt
EVAL_DATA=data/hard2_peer_h6_eval_v1.pt
SPLIT_MANIFEST=data/hard2_peer_h6_split_v1.json
FAMILY=mw_hard2_wam4va_visualmotion_peer_sync_h6_v1
LOCK=/tmp/ora0_wam4va_visualmotion_train.lock
MODE=${1:-}
BATCH=${2:-3}
GATE_STEPS=(50 300 1000)

usage(){ printf 'usage: bash %s {prepare|preflight|smoke10|pilot300|20k} [batch-size]\n' "$0" >&2; exit 2; }
fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha(){ sha256sum -- "$1" | cut -d' ' -f1; }
[[ $# -le 2 ]] || usage
case "$MODE" in prepare|preflight|smoke10|pilot300|20k) ;; *) usage;; esac
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || fail 'batch-size must be a positive integer'

for path in "$PY" "$ASSEMBLY_RAW" "$DOOR_RAW" "$ALLTASK_H48_REF" train.py \
  scripts/build_longtraj_features.py scripts/split_wam4va_episode_holdout.py; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done
[[ "$(sha "$ASSEMBLY_RAW")" == "$EXPECTED_ASSEMBLY_SHA256" ]] || fail 'raw assembly SHA-256 mismatch'
[[ "$(sha "$DOOR_RAW")" == "$EXPECTED_DOOR_SHA256" ]] || fail 'raw door-unlock SHA-256 mismatch'
[[ "$(sha "$ALLTASK_H48_REF")" == "$EXPECTED_ALLTASK_H48_REF_SHA256" ]] || fail 'all-task H48 normalization/language reference SHA-256 mismatch'

prepare_h6_data(){
  for path in "$SOURCE" "$TRAIN_DATA" "$EVAL_DATA" "$SPLIT_MANIFEST"; do
    [[ ! -e "$path" ]] || fail "refusing to overwrite immutable H6 data family: $path"
  done
  local assembly_sha_before door_sha_before ref_sha_before
  assembly_sha_before=$(sha "$ASSEMBLY_RAW")
  door_sha_before=$(sha "$DOOR_RAW")
  ref_sha_before=$(sha "$ALLTASK_H48_REF")
  "$PY" -B scripts/build_longtraj_features.py \
    --phase 1 --horizon 6 --data-contract peer_sync_h6_world_windows_v1 --legacy-policy infer \
    --input "$ASSEMBLY_RAW" --input "$DOOR_RAW" --ref "$ALLTASK_H48_REF" --output "$SOURCE"
  [[ "$(sha "$ASSEMBLY_RAW")" == "$assembly_sha_before" ]] || fail 'raw assembly changed during H6 preparation'
  [[ "$(sha "$DOOR_RAW")" == "$door_sha_before" ]] || fail 'raw door-unlock changed during H6 preparation'
  [[ "$(sha "$ALLTASK_H48_REF")" == "$ref_sha_before" ]] || fail 'all-task H48 reference changed during H6 preparation'
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$SOURCE" --train-output "$TRAIN_DATA" --eval-output "$EVAL_DATA" \
    --manifest-output "$SPLIT_MANIFEST" --heldout-fraction 0.10 --seed 0
}

preflight_h6(){
  for path in "$SOURCE" "$TRAIN_DATA" "$EVAL_DATA" "$SPLIT_MANIFEST"; do
    [[ -f "$path" ]] || fail "missing prepared peer_sync_h6 artifact: $path (run prepare explicitly)"
  done
  "$PY" -B - "$ASSEMBLY_RAW" "$DOOR_RAW" "$ALLTASK_H48_REF" "$SOURCE" "$TRAIN_DATA" "$EVAL_DATA" "$SPLIT_MANIFEST" \
    "$EXPECTED_ASSEMBLY_SHA256" "$EXPECTED_DOOR_SHA256" "$EXPECTED_ALLTASK_H48_REF_SHA256" <<'PY'
from pathlib import Path
import hashlib, json, sys
import torch
from scripts.split_wam4va_episode_holdout import canonical_manifest_sha256, transition_mask
assembly, door, ref, source, train, evaluation, manifest_path = map(Path, sys.argv[1:8])
expected_assembly_sha, expected_door_sha, expected_ref_sha = sys.argv[8:11]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
for label, path, expected in (
    ('raw assembly', assembly, expected_assembly_sha),
    ('raw door-unlock', door, expected_door_sha),
    ('all-task H48 reference', ref, expected_ref_sha),
):
    if digest(path) != expected: raise SystemExit(f'{label} SHA mismatch')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if manifest.get('contract') != 'wam4va_episode_holdout_manifest_v1': raise SystemExit('split contract mismatch')
if (manifest.get('data_protocol') or {}).get('contract') != 'peer_sync_h6_world_windows_v1': raise SystemExit('H6 data protocol mismatch')
if (manifest.get('data_protocol') or {}).get('shape') != {'sequence_length': 4, 'action_horizon': 6, 'action_dim': 4}: raise SystemExit('H6 data protocol shape mismatch')
if (manifest.get('data_protocol') or {}).get('logged_action_chunk') != 'full_h6': raise SystemExit('H6 logged action chunk mismatch')
if manifest.get('manifest_sha256') != canonical_manifest_sha256(manifest): raise SystemExit('manifest canonical SHA mismatch')
if Path(manifest.get('manifest_path', '')).resolve() != manifest_path.resolve(): raise SystemExit('manifest path mismatch')
source_contract = manifest.get('source') or {}
if Path(source_contract.get('path', '')).resolve() != source.resolve(): raise SystemExit('manifest source path mismatch')
if source_contract.get('sha256') != digest(source): raise SystemExit('manifest source SHA mismatch')
if source_contract.get('n_windows') != 891: raise SystemExit('H6 source window count mismatch')
expected_counts = {'source': 891, 'train': 793, 'eval': 98}
expected_sources = [(assembly.resolve(), expected_assembly_sha), (door.resolve(), expected_door_sha)]
for name, path in (('source', source), ('train', train), ('eval', evaluation)):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    actions = payload.get('actions'); valid = payload.get('action_valid_mask'); recovery = payload.get('recovery_mask')
    if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (expected_counts[name], 4, 6, 4): raise SystemExit(f'{name} is not exact expected-count/T4/H6/A4')
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != actions.shape[:-1]: raise SystemExit(f'{name} action_valid_mask mismatch')
    if not isinstance(recovery, torch.Tensor) or recovery.dtype != torch.bool or recovery.shape != actions.shape[:-1]: raise SystemExit(f'{name} recovery_mask mismatch')
    metadata = payload.get('metadata') or {}
    if metadata.get('contract') != 'peer_sync_h6_world_windows_v1' or metadata.get('contract_version') != 1: raise SystemExit(f'{name} H6 metadata contract mismatch')
    if metadata.get('action_horizon') != 6 or metadata.get('logged_action_chunk') != 'full_h6': raise SystemExit(f'{name} metadata is not full H6')
    parent_identity = metadata.get('parent_identity') or {}
    reference_identity = parent_identity if name == 'source' else parent_identity.get('payload_parent_identity') or {}
    if Path(reference_identity.get('path', '')).resolve() != ref.resolve() or reference_identity.get('sha256') != expected_ref_sha: raise SystemExit(f'{name} normalization/language reference identity mismatch')
    source_identities = metadata.get('source_identities') or []
    actual_sources = [(Path(item.get('path', '')).resolve(), item.get('sha256')) for item in source_identities]
    if actual_sources != expected_sources: raise SystemExit(f'{name} raw source identities mismatch')
    if not bool(transition_mask(valid).any()): raise SystemExit(f'{name} has no valid H6 World transitions')
    if sorted(int(x) for x in torch.unique(payload['instruction_id']).tolist()) != [0, 16]: raise SystemExit(f'{name} task set mismatch')
for split_name, path in (('train', train), ('eval', evaluation)):
    split = (manifest.get('splits') or {}).get(split_name) or {}
    if split.get('windows') != expected_counts[split_name]: raise SystemExit(f'{split_name} manifest count mismatch')
    if Path(split.get('output_path', '')).resolve() != path.resolve(): raise SystemExit(f'{split_name} output path mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload.get('metadata') or {}
    if metadata.get('split_manifest_sha256') != manifest.get('manifest_sha256'): raise SystemExit(f'{split_name} manifest binding mismatch')
if set((manifest.get('validation') or {}).values()) != {True}: raise SystemExit('split validation is not fully true')
print(f'peer_sync_h6 data preflight: PASS source_sha256={digest(source)} manifest={manifest["manifest_sha256"]}')
PY
  [[ -f "$DINO" ]] || fail "missing optional training-only DINO checkpoint: $DINO"
  "$PY" -B - "$SPLIT_MANIFEST" "$DINO" <<'PY'
import sys
from train import parse_args, validate_args
args = parse_args(['--data', 'unused.pt', '--world-split-manifest', sys.argv[1], '--visual-world-supervision', '--wam4va', '--wmrm-target', 'dino', '--wmrm-cycle-steps', '6', '--wmrm-inject', 'all', '--wmrm-adep-weight', '0', '--va-world-mode', 'peer_sync_h6', '--va-layers', '8', '--wmrm-predictor', 'st_blocks', '--wmrm-predictor-depth', '6', '--wmrm-predictor-width', '384', '--wmrm-predictor-heads', '12', '--wmrm-map-size', '16', '--wmrm-map-channels', '1024', '--wmrm-world-grid', '16', '--dino-main-vision', '--main-vision-checkpoint', sys.argv[2], '--main-vision-grid', '16', '--main-vision-frames', '4', '--sequence-length', '4', '--min-sequence-length', '4', '--single-task', '--flow-prefix-steps', '6'])
validate_args(args)
if args.va_world_mode != 'peer_sync_h6' or args.wmrm_adep_weight != 0.0: raise SystemExit('peer_sync_h6 CLI preflight mismatch')
if args.resume is not None or args.resume_weights is not None or args.resume_exact_contract_migration is not None: raise SystemExit('scratch preflight enabled forbidden resume/migration state')
print('peer_sync_h6 CLI preflight: PASS')
PY
}

refuse_output_family(){
  local run_id=$1 path; local -a existing=()
  shopt -s nullglob
  for path in checkpoints/${run_id}* logs/${run_id}* diagnostics/${run_id}*; do existing+=("$path"); done
  shopt -u nullglob
  ((${#existing[@]} == 0)) || { printf 'ERROR: refusing to overwrite immutable output family %s\n' "$run_id" >&2; printf '  %s\n' "${existing[@]}" >&2; exit 1; }
}

require_no_active_train(){
  "$PY" -B - <<'PY'
from pathlib import Path
import os
for process in Path('/proc').iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid(): continue
    try: argv = [x.decode('utf-8', 'replace') for x in (process/'cmdline').read_bytes().split(b'\0') if x]
    except OSError: continue
    if any(Path(arg).name == 'train.py' for arg in argv[1:]): raise SystemExit(f'active train.py pid={process.name}')
print('trainer process check: idle')
PY
}

verify_checkpoint(){
  local checkpoint=$1 expected_step=$2
  "$PY" -B - "$checkpoint" "$expected_step" "$SPLIT_MANIFEST" <<'PY'
from pathlib import Path
import json, sys, torch
path = Path(sys.argv[1]).resolve(strict=True); expected_step = int(sys.argv[2])
manifest = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
payload = torch.load(path, map_location='cpu', weights_only=True)
required = {'model','optimizer_state','sampler_state','rng_state','exact_run_contract','exact_resume_version','global_step','training_contract','config'}
missing = sorted(required - payload.keys())
if missing: raise SystemExit(f'checkpoint lacks exact-resume state: {missing}')
if payload['exact_resume_version'] != 2 or payload['global_step'] != expected_step: raise SystemExit('checkpoint exact step/version mismatch')
config = payload['config']
if config.get('va_world_mode') != 'peer_sync_h6' or config.get('action_horizon') != 6: raise SystemExit('checkpoint is not peer_sync_h6/H6')
arguments = payload['exact_run_contract'].get('arguments') or {}
model = payload['exact_run_contract'].get('model_config') or {}
for contract in (arguments, model):
    if contract.get('va_world_mode') != 'peer_sync_h6': raise SystemExit('exact run contract lost peer_sync_h6')
if arguments.get('wmrm_adep_weight') != 0.0: raise SystemExit('peer_sync_h6 action-dependence weight must remain zero')
if arguments.get('num_workers') != 0: raise SystemExit('num_workers contract must remain zero')
if arguments.get('resume_exact_contract_migration') is not None: raise SystemExit('migrations are forbidden')
if (payload.get('training_contract') or {}).get('split_manifest_sha256') != manifest.get('manifest_sha256'): raise SystemExit('checkpoint split manifest binding mismatch')
print(f'checkpoint preflight: PASS {path} global_step={expected_step}')
PY
}

milestone(){ printf 'checkpoints/%s.step%s.pt\n' "$1" "$2"; }
train_log(){ printf 'logs/%s.train_to_step%s.log\n' "$1" "$2"; }

run_segment(){
  local run_id=$1 start=$2 target=$3 source_checkpoint=$4 save log updates source_sha_before='' source_sha_after=''
  [[ -f "$DINO" ]] || fail "missing optional training-only DINO checkpoint: $DINO"
  save=$(milestone "$run_id" "$target"); log=$(train_log "$run_id" "$target"); updates=$((target-start))
  ((target > start)) || fail 'segment must advance global_step'
  [[ ! -e "$save" && ! -e "$log" ]] || fail "refusing to overwrite immutable milestone: $save or $log"
  local -a resume_args=()
  if ((start == 0)); then
    [[ "$source_checkpoint" == scratch ]] || fail 'first segment must start from scratch'
  else
    [[ "$source_checkpoint" != scratch && "$source_checkpoint" != "$save" ]] || fail 'continuation requires a distinct exact-resume source'
    verify_checkpoint "$source_checkpoint" "$start"
    source_sha_before=$(sha "$source_checkpoint")
    resume_args=(--resume-exact "$source_checkpoint")
  fi
  require_no_active_train
  set +e
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B train.py \
    --data "$TRAIN_DATA" --world-split-manifest "$SPLIT_MANIFEST" --visual-world-supervision \
    --va-world-mode peer_sync_h6 --wam4va --wmrm-inject all --wmrm-target dino \
    --wmrm-adep-weight 0 --wmrm-cycle-steps 6 --wmrm-world-weight 1.0 \
    --dino-main-vision --dino-dense-metric --main-vision-checkpoint "$DINO" \
    --main-vision-grid 16 --main-vision-frames 4 --main-vision-temporal \
    --main-vision-temporal-scale 1.0 --main-vision-encode-batch 8 --metric-geometry-inject \
    --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
    --wmrm-predictor st_blocks --wmrm-predictor-depth 6 --wmrm-predictor-width 384 --wmrm-predictor-heads 12 \
    --single-task --task-sampling balanced --task-locality-block-batches 4 \
    --batch-size "$BATCH" --sequence-length 4 --min-sequence-length 4 --num-workers 0 \
    --lr 0.0001 --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 --va-attention-backend auto \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 --flow-prefix-steps 6 \
    --flow-prefix-weight 1.0 --flow-tail-weight 0.036 --mtvj-train-metric-head \
    --lr-mtvj-metric-head 0.0003 --mtvj-train-relation --lr-mtvj-relation 0.00002 \
    --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 --steps "$updates" --save-every 0 \
    --save "$save" "${resume_args[@]}" 2>&1 | tee "$log"
  local -a status=("${PIPESTATUS[@]}"); set -e
  ((${#status[@]} == 2)) || fail 'unexpected training pipeline status'
  [[ "${status[0]}" -eq 0 ]] || exit "${status[0]}"
  [[ "${status[1]}" -eq 0 && -s "$log" ]] || fail 'tee failed or training log is incomplete'
  if ((start > 0)); then source_sha_after=$(sha "$source_checkpoint"); [[ "$source_sha_after" == "$source_sha_before" ]] || fail 'exact-resume source checkpoint was modified'; fi
  verify_checkpoint "$save" "$target"
}

run_lineage(){
  local run_id=$1; shift
  local start=0 source_checkpoint=scratch target
  for target in "$@"; do
    run_segment "$run_id" "$start" "$target" "$source_checkpoint"
    source_checkpoint=$(milestone "$run_id" "$target"); start=$target
  done
}

command -v flock >/dev/null || fail 'flock is required'
exec 9>"$LOCK"
flock -n 9 || fail 'another visual-motion lifecycle owns the exclusive lock'

case "$MODE" in
  prepare) prepare_h6_data; preflight_h6;;
  preflight) preflight_h6;;
  smoke10)
    preflight_h6; run_id=${FAMILY}.smoke10; refuse_output_family "$run_id"
    printf 'scratch-only smoke: 0 -> 10; no automatic long continuation\n'
    run_lineage "$run_id" 10;;
  pilot300)
    preflight_h6; run_id=${FAMILY}.pilot300; refuse_output_family "$run_id"
    printf 'scratch-only pilot: 0 -> 50 -> 300; exact-resume continuation; STOP at 300\n'
    run_lineage "$run_id" 50 300;;
  20k)
    preflight_h6; run_id=${FAMILY}.long20k; refuse_output_family "$run_id"
    printf 'scratch-only formal lineage: 0 -> 50 -> 300 -> 1000 -> 20000\n'
    run_lineage "$run_id" "${GATE_STEPS[@]}" 20000;;
esac
