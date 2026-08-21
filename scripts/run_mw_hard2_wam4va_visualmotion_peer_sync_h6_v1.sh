#!/usr/bin/env bash
# P2/H6 joint layer-delayed VA↔World state exchange over disjoint data streams.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
VERIFY_PY=${VERIFY_PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
DINO=${DINO:-/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors}
# FRAMES_DIR holds one JPEG-frame file per task.  The peer_sync_h6 contract keys
# a window's frame pointer by env name + episode index within that file, so
# expansion episodes must be merged into these per-task files
# (scripts/merge_longtraj_expansion.py) rather than passed as extra inputs.
FRAMES_DIR=${FRAMES_DIR:-data}
ASSEMBLY_RAW=$FRAMES_DIR/metaworld_longtraj_assembly-v3.pt
DOOR_RAW=$FRAMES_DIR/metaworld_longtraj_door-unlock-v3.pt
# Phase 1 reads only normalization/tasks/language from the reference.  Any all-49
# build carrying identical values is an equivalent substitute; override when the
# canonical file is unavailable.
ALLTASK_H48_REF=${ALLTASK_H48_REF:-data/metaworld_longtraj_windows_h48.pt}
# DATA_TAG selects the immutable split family.  Defaults reproduce v1.
DATA_TAG=${DATA_TAG:-v1}
# Resident decoded-task budget; empty keeps train.py's own default.
DECODE_CACHE_TASKS=${DECODE_CACHE_TASKS:-}
# 1 = single process; >1 launches torchrun over that many local GPUs.  Global
# --batch-size is split across ranks (48 on 2 GPUs is 24 per card).
NGPUS=${NGPUS:-1}
SOURCE=data/hard2_peer_h6_p2_source_${DATA_TAG}.pt
WORLD_POOL=data/hard2_peer_h6_p2_world_pool_${DATA_TAG}.pt
VA_TRAIN_DATA=data/hard2_peer_h6_p2_va_train_${DATA_TAG}.pt
WORLD_TRAIN_DATA=data/hard2_peer_h6_p2_world_train_${DATA_TAG}.pt
EVAL_DATA=data/hard2_peer_h6_p2_eval_${DATA_TAG}.pt
PARTITION_MANIFEST=data/hard2_peer_h6_p2_va_world_partition_${DATA_TAG}.json
WORLD_SPLIT_MANIFEST=data/hard2_peer_h6_p2_world_split_${DATA_TAG}.json
FAMILY=mw_hard2_va_world_state_exchange_joint_h6_p2_${DATA_TAG}
LOCK=/tmp/ora0_va_world_state_exchange_joint_h6_p2_v1.lock

MODE=${1:-}
STEPS=${2:-}
BATCH=${3:-3}
RESUME_EXACT=${RESUME_EXACT:-}
RUN_ID=${RUN_ID:-}
SAVE_EVERY=${SAVE_EVERY:-1500}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}

usage(){
  printf 'usage: bash %s {prepare|preflight|joint} [steps] [batch-size]\n' "$0" >&2
  exit 2
}
fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -le 3 ]] || usage
case "$MODE" in prepare|preflight|joint) ;; *) usage;; esac
if [[ "$MODE" == joint ]]; then
  [[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail 'steps must be a positive integer'
fi
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || fail 'batch-size must be a positive integer'

for path in "$PY" "$VERIFY_PY" "$ASSEMBLY_RAW" "$DOOR_RAW" "$ALLTASK_H48_REF" \
  train.py scripts/build_longtraj_features.py scripts/split_wam4va_episode_holdout.py; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done

prepare_data(){
  local inputs=(--input "$ASSEMBLY_RAW" --input "$DOOR_RAW")
  printf 'phase 1 frame dir: %s\n' "$FRAMES_DIR"
  if [[ ! -f "$SOURCE" ]]; then
    "$PY" -B scripts/build_longtraj_features.py \
      --phase 1 --horizon 6 --planning-stride 2 \
      --data-contract peer_sync_h6_p2_world_windows_v1 \
      --legacy-policy infer "${inputs[@]}" \
      --ref "$ALLTASK_H48_REF" --output "$SOURCE"
  fi
  for path in "$WORLD_POOL" "$VA_TRAIN_DATA" "$WORLD_TRAIN_DATA" \
    "$EVAL_DATA" "$PARTITION_MANIFEST" "$WORLD_SPLIT_MANIFEST"; do
    [[ ! -e "$path" ]] || fail "refusing to overwrite immutable split: $path"
  done

  # First split: action/VA episodes versus the independent World+eval pool.
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$SOURCE" --train-output "$WORLD_POOL" \
    --eval-output "$VA_TRAIN_DATA" --manifest-output "$PARTITION_MANIFEST" \
    --heldout-fraction 0.50 --seed 101
  # Second split: World training episodes versus final held-out evaluation.
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$WORLD_POOL" --train-output "$WORLD_TRAIN_DATA" \
    --eval-output "$EVAL_DATA" --manifest-output "$WORLD_SPLIT_MANIFEST" \
    --heldout-fraction 0.20 --seed 202
}

preflight(){
  for path in "$SOURCE" "$WORLD_POOL" "$VA_TRAIN_DATA" "$WORLD_TRAIN_DATA" "$EVAL_DATA" \
    "$PARTITION_MANIFEST" "$WORLD_SPLIT_MANIFEST" "$DINO"; do
    [[ -f "$path" ]] || fail "missing prepared artifact: $path"
  done
  "$VERIFY_PY" -B - "$VA_TRAIN_DATA" "$WORLD_TRAIN_DATA" "$EVAL_DATA" \
    "$PARTITION_MANIFEST" "$WORLD_SPLIT_MANIFEST" <<'PY'
from pathlib import Path
import json, sys, torch

va_path, world_path, eval_path, partition_path, world_manifest_path = map(Path, sys.argv[1:])
payloads = {
    name: torch.load(path, map_location="cpu", weights_only=True)
    for name, path in (("va", va_path), ("world", world_path), ("eval", eval_path))
}
episodes = {}
for name, payload in payloads.items():
    actions = payload.get("actions")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 4 or tuple(actions.shape[1:]) != (4, 6, 4):
        raise SystemExit(f"{name} is not T4/H6/A4")
    metadata = payload.get("metadata") or {}
    cadence = {
        "contract": "peer_sync_h6_p2_world_windows_v1",
        "contract_version": 1,
        "fps": 80,
        "control_stride": 2,
        "planning_stride": 2,
        "sequence_length": 4,
        "decision_offsets": [0, 2, 4, 6],
        "action_horizon": 6,
        "action_label_offsets": [0, 1, 2, 3, 4, 5],
    }
    bad_cadence = {
        key: (metadata.get(key), expected)
        for key, expected in cadence.items()
        if metadata.get(key) != expected
    }
    if bad_cadence:
        raise SystemExit(f"{name} planning-frequency mismatch: {bad_cadence}")
    if sorted(int(x) for x in torch.unique(payload["instruction_id"]).tolist()) != [0, 16]:
        raise SystemExit(f"{name} task set mismatch")
    episodes[name] = {int(x) for x in payload["episode_id"].tolist()}
for left, right in (("va", "world"), ("va", "eval"), ("world", "eval")):
    if episodes[left] & episodes[right]:
        raise SystemExit(f"episode leakage: {left}/{right}")
partition = json.loads(partition_path.read_text(encoding="utf-8"))
world_manifest = json.loads(world_manifest_path.read_text(encoding="utf-8"))
for name, manifest in (("partition", partition), ("world", world_manifest)):
    if (manifest.get("data_protocol") or {}).get("contract") != "peer_sync_h6_p2_world_windows_v1":
        raise SystemExit(f"{name} manifest is not P2/H6")
    if (manifest.get("transition_rule") or {}).get("current_action_prefix_steps") != 2:
        raise SystemExit(f"{name} manifest World transition is not P2")
if Path(partition["splits"]["eval"]["output_path"]).name != va_path.name:
    raise SystemExit("VA split binding mismatch")
if Path(world_manifest["splits"]["train"]["output_path"]).name != world_path.name:
    raise SystemExit("World split binding mismatch")
if Path(world_manifest["splits"]["eval"]["output_path"]).name != eval_path.name:
    raise SystemExit("eval split binding mismatch")
print("disjoint VA/World/eval data preflight: PASS")
PY

  "$PY" -B - "$WORLD_SPLIT_MANIFEST" "$DINO" <<'PY'
import sys
from train import parse_args, validate_args

common = [
    "--va-data", "va-unused.pt", "--world-data", "world-unused.pt",
    "--visual-world-supervision", "--world-split-manifest", sys.argv[1],
    "--wam4va", "--va-world-mode", "peer_sync_h6",
    "--planning-stride", "2", "--control-stride", "2",
    "--wmrm-inject", "all", "--wmrm-target", "dino", "--wmrm-cycle-steps", "2",
    "--wmrm-adep-weight", "0", "--va-layers", "8", "--wmrm-predictor", "st_blocks",
    "--wmrm-predictor-depth", "6", "--wmrm-predictor-width", "384",
    "--wmrm-predictor-heads", "12", "--wmrm-map-size", "16",
    "--wmrm-map-channels", "1024", "--wmrm-world-grid", "16", "--dino-main-vision",
    "--main-vision-checkpoint", sys.argv[2], "--main-vision-grid", "16",
    "--main-vision-frames", "4", "--sequence-length", "4",
    "--min-sequence-length", "4", "--single-task", "--task-sampling", "balanced",
    "--flow-prefix-steps", "2",
]
validate_args(parse_args(common))
print("joint dual-stream peer training CLI preflight: PASS")
PY
}

checkpoint_contract(){
  "$VERIFY_PY" -B - "$1" "$VA_TRAIN_DATA" "$WORLD_TRAIN_DATA" <<'PY'
from pathlib import Path
import sys, torch

checkpoint, va_path, world_path = map(Path, sys.argv[1:])
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
contract = payload.get("training_contract") or {}
required = {
    "peer_training_mode": "joint_dual_stream",
    "peer_world_topology": "one_stage_delayed_bidirectional_state_kv_v1",
    "peer_gradient_boundary": "fully_differentiable_bidirectional_messages_v1",
    "peer_data_isolation": "separate_va_world_episode_datasets_per_step_v1",
    "peer_dual_stream_optimizer": "va_backward_then_world_backward_one_optimizer_step_v1",
}
bad = {key: (contract.get(key), value) for key, value in required.items() if contract.get(key) != value}
if bad: raise SystemExit(f"checkpoint contract mismatch: {bad}")
identities = {}
for key, expected_path in (
    ("peer_va_data_identity", va_path),
    ("peer_world_data_identity", world_path),
):
    identity = contract.get(key)
    if not isinstance(identity, dict) or not identity.get("full_file_sha256"):
        raise SystemExit(f"checkpoint lacks strong {key}")
    if Path(identity.get("resolved_path", "")).resolve() != expected_path.resolve():
        raise SystemExit(f"checkpoint {key} path mismatch")
    identities[key] = identity["full_file_sha256"]
if identities["peer_va_data_identity"] == identities["peer_world_data_identity"]:
    raise SystemExit("checkpoint VA/World data identities must differ")
summary = contract.get("peer_data_isolation_summary") or {}
if summary.get("contract") != required["peer_data_isolation"]:
    raise SystemExit("checkpoint lacks peer data-isolation summary")
if summary.get("task_ids") != [0, 16]:
    raise SystemExit("checkpoint peer task set mismatch")
config = payload.get("config") or {}
if config.get("va_world_mode") != "peer_sync_h6":
    raise SystemExit("checkpoint is not peer_sync_h6")
config_cadence = {
    "action_horizon": (config.get("action_horizon"), 6),
    "planning_stride": (config.get("planning_stride"), 2),
    "wmrm_cycle_steps": (config.get("wmrm_cycle_steps"), 2),
}
bad_config_cadence = {
    key: values for key, values in config_cadence.items()
    if values[0] != values[1]
}
if bad_config_cadence:
    raise SystemExit(f"checkpoint config is not 80Hz/P2/H6: {bad_config_cadence}")
for key in ("peer_va_data_identity", "peer_world_data_identity"):
    metadata = (
        (contract[key].get("payload_schema") or {})
        .get("non_tensors", {})
        .get("metadata", {})
    )
    expected_metadata = {
        "contract": "peer_sync_h6_p2_world_windows_v1",
        "contract_version": 1,
        "fps": 80,
        "control_stride": 2,
        "planning_stride": 2,
        "sequence_length": 4,
        "decision_offsets": [0, 2, 4, 6],
        "action_horizon": 6,
        "action_label_offsets": [0, 1, 2, 3, 4, 5],
    }
    bad_metadata = {
        name: (metadata.get(name), value)
        for name, value in expected_metadata.items()
        if metadata.get(name) != value
    }
    if bad_metadata:
        raise SystemExit(f"checkpoint {key} is not 80Hz/P2/H6: {bad_metadata}")
arguments = (payload.get("exact_run_contract") or {}).get("arguments") or {}
expected_arguments = {
    "control_stride": 2,
    "planning_stride": 2,
    "wmrm_cycle_steps": 2,
    "flow_prefix_steps": 2,
    "task_sampling": "balanced",
}
bad_arguments = {
    key: (arguments.get(key), value)
    for key, value in expected_arguments.items()
    if arguments.get(key) != value
}
if bad_arguments:
    raise SystemExit(f"checkpoint P2 run contract mismatch: {bad_arguments}")
print(f"checkpoint preflight: PASS {checkpoint} (joint_dual_stream)")
PY
}

require_no_active_train(){
  "$VERIFY_PY" -B - <<'PY'
from pathlib import Path
import os
for process in Path("/proc").iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid():
        continue
    try:
        argv = [item for item in (process / "cmdline").read_bytes().split(b"\0") if item]
        executable = (process / "exe").resolve().name
    except OSError:
        continue
    if executable.startswith("python") and any(
        Path(item.decode("utf-8", "replace")).name == "train.py"
        for item in argv[1:]
    ):
        raise SystemExit(f"active train.py pid={process.name}")
print("trainer process check: idle")
PY
}

run_joint(){
  local run_id=${RUN_ID:-${FAMILY}.scratch.s${STEPS}}
  local save=${CHECKPOINT_DIR}/${run_id}.pt log=logs/${run_id}.log
  local resume_args=()
  if [[ -n "$RESUME_EXACT" ]]; then
    [[ -f "$RESUME_EXACT" ]] || fail "missing exact-resume checkpoint: $RESUME_EXACT"
    resume_args=(--resume-exact "$RESUME_EXACT")
  fi
  local frame_args=(--longtraj-dir "$FRAMES_DIR")
  if [[ -n "$DECODE_CACHE_TASKS" ]]; then
    frame_args+=(--longtraj-decode-cache-tasks "$DECODE_CACHE_TASKS")
  fi
  [[ ! -e "$save" && ! -e "$log" ]] || fail "refusing to overwrite $run_id"
  mkdir -p "$CHECKPOINT_DIR" logs
  require_no_active_train
  local launcher
  if [[ "$NGPUS" -gt 1 ]]; then
    [[ "$BATCH" -eq $(( BATCH / NGPUS * NGPUS )) ]] \
      || fail "batch-size $BATCH must divide across $NGPUS GPUs"
    launcher=("$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" --max_restarts=0)
  else
    launcher=("$PY" -u -B)
  fi
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${launcher[@]}" train.py --va-data "$VA_TRAIN_DATA" --world-data "$WORLD_TRAIN_DATA" \
    --visual-world-supervision --world-split-manifest "$WORLD_SPLIT_MANIFEST" \
    --va-world-mode peer_sync_h6 --planning-stride 2 --control-stride 2 \
    --wam4va --wmrm-inject all --wmrm-target dino \
    --wmrm-adep-weight 0 --wmrm-cycle-steps 2 --wmrm-world-weight 1.0 \
    --world-action-rank-stage final \
    --dino-main-vision --dino-dense-metric --main-vision-checkpoint "$DINO" \
    --main-vision-grid 16 --main-vision-frames 4 --main-vision-temporal \
    --main-vision-temporal-scale 1.0 --main-vision-encode-batch 8 \
    --metric-geometry-inject --wmrm-map-size 16 --wmrm-map-channels 1024 \
    --wmrm-world-grid 16 --wmrm-predictor st_blocks --wmrm-predictor-depth 6 \
    --wmrm-predictor-width 384 --wmrm-predictor-heads 12 --single-task \
    --task-sampling balanced --task-locality-block-batches 64 --batch-size "$BATCH" \
    --sequence-length 4 --min-sequence-length 4 --num-workers 0 --lr 0.0001 \
    --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 \
    --va-attention-backend auto --flow-cond adaln --flow-layers 6 --flow-steps 8 \
    --flow-prefix-steps 2 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
    --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
    --mtvj-train-relation --lr-mtvj-relation 0.00002 \
    --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 \
    --steps "$STEPS" --save-every "$SAVE_EVERY" --save-step-copies --save "$save" \
    "${frame_args[@]}" "${resume_args[@]}" 2>&1 | tee "$log"
  checkpoint_contract "$save"
}

command -v flock >/dev/null || fail 'flock is required'
exec 9>"$LOCK"
flock -n 9 || fail 'another joint VA/World run owns the lock'

case "$MODE" in
  prepare) prepare_data; preflight;;
  preflight) preflight;;
  joint) preflight; run_joint;;
esac
