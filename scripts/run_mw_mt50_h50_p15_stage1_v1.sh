#!/usr/bin/env bash
# Evo-1-matched Stage1: predict H50, execute/replan at P15, train action SFT only.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-preflight}
PY=${PY:-/opt/conda/bin/python}
ROOT=${ORA0_DATA_ROOT:-/root/private_data/ORA0}
ONLINE_INDEX=${ONLINE_INDEX:-$ROOT/mt50_dagger_recovery_r1_r2/data_v2/mt50_full_episode_online_index_v1.json}
FRAMES_DIR=${FRAMES_DIR:-$ROOT/data/frames_v2}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/root/private_data/ORA0/checkpoints/mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5_s3224.pt}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/ora0_ckpts}
H50_STEPS=${H50_STEPS:-10000}
SAVE_EVERY=${SAVE_EVERY:-1000}
BATCH=${BATCH:-48}
NGPUS=${NGPUS:-2}
RUN_ID=${RUN_ID:-mw_mt50_h50_p15_stage1_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_steps${H50_STEPS}_v1}
SAVE=$CHECKPOINT_DIR/$RUN_ID.pt
LOG=logs/$RUN_ID.log
MIGRATION=peer_h15_to_h50_action_horizon_weights_only_v1
LOCK=/tmp/ora0_mt50_h50_p15_stage1.lock

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

preflight(){
  for path in "$ONLINE_INDEX" "$BASE_CHECKPOINT" "$DINO"; do
    [[ -f "$path" ]] || fail "missing $path"
  done
  (( H50_STEPS > 0 )) || fail "H50_STEPS must be positive"
  (( SAVE_EVERY > 0 )) || fail "SAVE_EVERY must be positive"
  [[ "$BATCH" == 48 ]] || fail "Stage1 global batch must remain 48"
  [[ "$NGPUS" == 2 ]] || fail "Stage1 requires two GPUs"
  (( BATCH % NGPUS == 0 )) || fail "batch must divide across GPUs"
  "$PY" -B - "$BASE_CHECKPOINT" "$ONLINE_INDEX" "$FRAMES_DIR" <<'PY'
import sys
import hashlib
import json
from pathlib import Path
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
config = checkpoint.get("config") or {}
required = {
    "action_horizon": 15,
    "planning_stride": 15,
    "deployment_execution_horizon": 15,
    "wmrm_cycle_steps": 15,
    "va_world_mode": "peer_sync_h6",
    "main_vision_backbone": "dinov2_vitl14_reg4",
}
bad = {key: (config.get(key), value) for key, value in required.items()
       if config.get(key) != value}
if bad:
    raise SystemExit(f"base checkpoint is not the s3224 H15/P15 contract: {bad}")
state = checkpoint.get("model") or {}
if "action_queries" not in state or tuple(state["action_queries"].shape[:1]) != (15,):
    raise SystemExit("base checkpoint lacks H15 action_queries")
if any(key.startswith("extension_flow_head.") for key in state):
    raise SystemExit("base checkpoint is already H50-expanded")
if int(checkpoint.get("global_step", -1)) != 3224:
    raise SystemExit(
        f"base checkpoint must be immutable s3224, got {checkpoint.get('global_step')}"
    )
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for block in iter(lambda: stream.read(16 << 20), b""):
        digest.update(block)
if digest.hexdigest() != "cf734545c0e6ec9eb355b777126983d4f2dfd2be9003003501c61851eedbc7df":
    raise SystemExit(f"unexpected s3224 SHA256: {digest.hexdigest()}")
index = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
paths = [Path(index["language_reference"]["path"])]
frames_dir = Path(sys.argv[3])
paths.extend(
    Path(item["source_path"])
    if Path(item["source_path"]).is_file()
    else frames_dir / f"metaworld_longtraj_{item['task']}.pt"
    for item in index["tasks"]
)
paths.extend(
    Path(item["source_path"])
    for item in index["episodes"]
    if item.get("source_path") is not None
)
paths.extend(Path(item["path"]) for item in index.get("additional_sources") or [])
missing = sorted({str(path) for path in paths if not path.is_file()})
if missing:
    raise SystemExit(f"online index has missing base/overlay sources: {missing[:8]}")
print(f"s3224 H15/P15 checkpoint: PASS global_step={checkpoint.get('global_step')}")
PY
  printf 'H50/P15 Stage1 preflight: PASS; target_steps=%s save_every=%s batch=%s GPUs=%s; World loss=off\n' \
    "$H50_STEPS" "$SAVE_EVERY" "$BATCH" "$NGPUS"
}

run_train(){
  local launch_mode=${1:-fresh}
  preflight
  local run_steps=$H50_STEPS
  local resume_args=(
    --resume-weights "$BASE_CHECKPOINT"
    --resume-weights-migration "$MIGRATION"
  )
  local tee_args=()
  if [[ "$launch_mode" == exact ]]; then
    [[ -f "$SAVE" && -f "$LOG" ]] || fail "exact resume requires $SAVE and $LOG"
    local completed
    completed=$("$PY" -B - "$SAVE" <<'PY'
import sys
import torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True).get("global_step", -1)))
PY
)
    (( completed > 0 && completed < H50_STEPS )) || \
      fail "checkpoint global_step=$completed is outside (0,$H50_STEPS)"
    run_steps=$((H50_STEPS - completed))
    resume_args=(--resume-exact "$SAVE")
    tee_args=(-a)
    printf 'exact resume: global_step=%s remaining=%s target=%s\n' \
      "$completed" "$run_steps" "$H50_STEPS"
  else
    [[ ! -e "$SAVE" && ! -e "$LOG" ]] || fail "refusing to overwrite $RUN_ID"
    ! compgen -G "${SAVE%.pt}_s*.pt" >/dev/null || \
      fail "refusing to overwrite existing step checkpoints for $RUN_ID"
  fi
  ! pgrep -af '[p]ython.*train.py' >/dev/null || fail 'another train.py is active'
  mkdir -p "$CHECKPOINT_DIR" logs
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" \
      --max_restarts=0 train.py \
      --data "$ONLINE_INDEX" --online-episode-sampling --online-action-horizon 50 \
      --online-episode-samples 6 --single-task --task-sampling mixed \
      --mixed-tasks-per-batch 4 --anchor-replay-fraction 0.25 --pcgrad \
      --task-locality-block-batches 1 --longtraj-decode-cache-tasks 50 \
      --va-world-mode peer_sync_h6 --va-only --planning-stride 15 \
      --control-stride 15 --deployment-execution-horizon 15 \
      --wam4va --wmrm-full-language-tokens --slot-free-policy \
      --wmrm-inject all --wmrm-target dino --wmrm-adep-weight 0 \
      --wmrm-cycle-steps 15 --wmrm-world-weight 1.0 \
      --dino-main-vision --main-vision-checkpoint "$DINO" \
      --main-vision-grid 16 --main-vision-frames 4 --main-vision-temporal \
      --main-vision-temporal-scale 1.0 --main-vision-encode-batch 16 \
      --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
      --wmrm-predictor st_blocks --wmrm-predictor-depth 6 \
      --wmrm-predictor-width 384 --wmrm-predictor-heads 12 \
      --wmrm-predictor-copies 1 --batch-size "$BATCH" --sequence-length 4 \
      --min-sequence-length 4 --num-workers 0 --disable-runtime-integrity-checks \
      --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 \
      --va-attention-backend auto --flow-cond adaln --flow-layers 6 \
      --flow-steps 8 --flow-prefix-steps 15 --flow-prefix-weight 1.0 \
      --flow-tail-weight 1.0 --lr 0.00001 --steps "$run_steps" \
      --save-every "$SAVE_EVERY" --save-step-copies --save "$SAVE" \
      --longtraj-dir "$FRAMES_DIR" "${resume_args[@]}" \
      2>&1 | tee "${tee_args[@]}" "$LOG"
}

train(){ run_train fresh; }
resume(){ run_train exact; }

evaluate(){
  [[ -f "$SAVE" ]] || fail "missing completed Stage1 checkpoint $SAVE"
  CKPT="$SAVE" TAG="${RUN_ID}_seed4042_h50p15" \
    exec scripts/run_mw_mt50_acceptance_v1.sh
}

command -v flock >/dev/null || fail "flock is required"
exec 9>"$LOCK"
flock -n 9 || fail "another H50 Stage1 launcher owns the lock"
case "$MODE" in
  preflight) preflight ;;
  train) train ;;
  resume) resume ;;
  eval) evaluate ;;
  *) fail "usage: $0 {preflight|train|resume|eval}" ;;
esac
