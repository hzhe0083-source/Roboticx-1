#!/usr/bin/env bash
# H50/P15 joint phase: unfreeze VA, World and all DINO; train three epochs.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-preflight}
PY=${PY:-/opt/conda/bin/python}
ONLINE_INDEX=${ONLINE_INDEX:-/root/evo1_metaworld_longtraj_v1/evo1_mt50_online_index.json}
FRAMES_DIR=${FRAMES_DIR:-/root/evo1_metaworld_longtraj_v1}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/root/ora0_ckpts/mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1_s313.pt}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/private_data/ORA0_next/checkpoints}
BATCH=${BATCH:-16}
NGPUS=${NGPUS:-2}
EPOCHS=3
SAMPLES_PER_EPISODE=6
MIXED_TASKS=4
MAIN_VISION_ENCODE_BATCH=${MAIN_VISION_ENCODE_BATCH:-4}
RUN_ID=${RUN_ID:-mw_mt50_h50_p15_joint_va_wm_dinoall_separatepcgrad_from_s313_e3_b16_v1}
SAVE=$CHECKPOINT_DIR/$RUN_ID.pt
LOG=logs/$RUN_ID.log
MIGRATION=peer_h50_action_only_to_joint_weights_only_v1
LOCK=/tmp/ora0_mt50_h50_p15_joint_full3.lock

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

read_steps(){
  "$PY" -B - "$ONLINE_INDEX" "$BATCH" "$SAMPLES_PER_EPISODE" "$EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
batch, samples_per_episode, epochs = map(int, sys.argv[2:])
episodes = len(index.get("episodes") or [])
if episodes != 2500:
    raise SystemExit(f"expected 2500 official Evo-1 episodes, got {episodes}")
samples = episodes * samples_per_episode
steps_per_epoch = (samples + batch - 1) // batch
print(episodes, samples, steps_per_epoch, epochs * steps_per_epoch)
PY
}

preflight(){
  for path in "$ONLINE_INDEX" "$BASE_CHECKPOINT" "$DINO"; do
    [[ -f "$path" ]] || fail "missing $path"
  done
  [[ "$BATCH" == 16 ]] || fail "joint full-DINO run requires global batch 16"
  [[ "$NGPUS" == 2 ]] || fail "joint full-DINO run requires two GPUs"
  (( BATCH % NGPUS == 0 )) || fail "batch must divide across GPUs"
  (( BATCH / NGPUS % MIXED_TASKS == 0 )) || \
    fail "each rank must contain all mixed tasks"
  "$PY" -B - "$BASE_CHECKPOINT" <<'PY'
import hashlib
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
config = checkpoint.get("config") or {}
contract = checkpoint.get("training_contract") or {}
required = {
    "global_step": (checkpoint.get("global_step"), 313),
    "action_horizon": (config.get("action_horizon"), 50),
    "planning_stride": (config.get("planning_stride"), 15),
    "deployment_execution_horizon": (
        config.get("deployment_execution_horizon"), 15
    ),
    "wmrm_cycle_steps": (config.get("wmrm_cycle_steps"), 15),
    "va_world_mode": (config.get("va_world_mode"), "peer_sync_h6"),
    "peer_training_mode": (contract.get("peer_training_mode"), "va_only"),
    "pcgrad_scope": (contract.get("pcgrad_scope"), "per_task_va_action_v1"),
}
bad = {key: value for key, value in required.items() if value[0] != value[1]}
if bad:
    raise SystemExit(f"s313 contract mismatch: {bad}")
digest = hashlib.sha256()
with open(path, "rb") as stream:
    for block in iter(lambda: stream.read(16 << 20), b""):
        digest.update(block)
expected = "29851a4b013d346b9dc9b2047beb2200236431c1e05d29aeafb77c57942b8cdb"
if digest.hexdigest() != expected:
    raise SystemExit(f"unexpected s313 SHA256: {digest.hexdigest()}")
print("immutable s313 H50/P15 action-only checkpoint: PASS")
PY
  read -r EPISODES SAMPLES STEPS_PER_EPOCH STEPS < <(read_steps)
  printf 'joint full-unfreeze preflight: PASS; baseline=43.6%% episodes=%s ' "$EPISODES"
  printf 'samples/epoch=%s steps/epoch=%s epochs=%s total_steps=%s batch=%s\n' \
    "$SAMPLES" "$STEPS_PER_EPOCH" "$EPOCHS" "$STEPS" "$BATCH"
}

run_train(){
  local launch_mode=${1:-fresh}
  preflight
  read -r _ _ STEPS_PER_EPOCH TOTAL_STEPS < <(read_steps)
  local run_steps=$TOTAL_STEPS
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
    (( completed > 0 && completed < TOTAL_STEPS )) || \
      fail "checkpoint global_step=$completed is outside (0,$TOTAL_STEPS)"
    run_steps=$((TOTAL_STEPS - completed))
    resume_args=(--resume-exact "$SAVE")
    tee_args=(-a)
    printf 'exact resume: phase_step=%s remaining=%s target=%s\n' \
      "$completed" "$run_steps" "$TOTAL_STEPS"
  else
    [[ ! -e "$SAVE" && ! -e "$LOG" ]] || fail "refusing to overwrite $RUN_ID"
  fi
  if [[ -n "${RUN_STEPS_OVERRIDE:-}" ]]; then
    (( RUN_STEPS_OVERRIDE > 0 )) || fail "RUN_STEPS_OVERRIDE must be positive"
    run_steps=$RUN_STEPS_OVERRIDE
  fi
  local save_args=(--save-every "$STEPS_PER_EPOCH" --save "$SAVE")
  if [[ "${NO_SAVE:-0}" == 1 ]]; then
    save_args=()
  fi
  ! pgrep -af '[p]ython.*train.py' >/dev/null || fail 'another train.py is active'
  mkdir -p "$CHECKPOINT_DIR" logs
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" \
      --max_restarts=0 train.py \
      --va-data "$ONLINE_INDEX" --world-data "$ONLINE_INDEX" \
      --online-episode-sampling --online-action-horizon 50 \
      --online-episode-samples "$SAMPLES_PER_EPISODE" \
      --peer-shared-full-data --visual-world-supervision \
      --world-split-manifest "$ONLINE_INDEX" \
      --single-task --task-sampling mixed --mixed-tasks-per-batch "$MIXED_TASKS" \
      --anchor-replay-fraction 0.25 --pcgrad --pcgrad-separate-world \
      --task-locality-block-batches 1 --longtraj-decode-cache-tasks 50 \
      --va-world-mode peer_sync_h6 --planning-stride 15 --control-stride 15 \
      --deployment-execution-horizon 15 --wam4va --wmrm-full-language-tokens \
      --slot-free-policy --wmrm-inject all --wmrm-target dino \
      --wmrm-adep-weight 0 --wmrm-cycle-steps 15 --wmrm-world-weight 1.0 \
      --world-action-rank-stage final --dino-main-vision \
      --main-vision-checkpoint "$DINO" --vision-unfreeze-all --lr-vision 0.000001 \
      --main-vision-grid 16 --main-vision-frames 4 --main-vision-temporal \
      --main-vision-temporal-scale 1.0 \
      --main-vision-encode-batch "$MAIN_VISION_ENCODE_BATCH" \
      --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
      --wmrm-predictor st_blocks --wmrm-predictor-depth 6 \
      --wmrm-predictor-width 384 --wmrm-predictor-heads 12 \
      --wmrm-predictor-copies 1 --batch-size "$BATCH" --sequence-length 4 \
      --min-sequence-length 4 --num-workers 0 --peer-batch-prefetch \
      --peer-batch-prefetch-depth 1 --disable-runtime-integrity-checks \
      --seed 0 --device cuda --feature-autocast-bf16 --va-layers 8 \
      --va-attention-backend auto --flow-cond adaln --flow-layers 6 \
      --flow-steps 8 --flow-prefix-steps 15 --flow-prefix-weight 1.0 \
      --flow-tail-weight 1.0 --lr 0.00001 --steps "$run_steps" \
      "${save_args[@]}" \
      --longtraj-dir "$FRAMES_DIR" "${resume_args[@]}" \
      2>&1 | tee "${tee_args[@]}" "$LOG"
}

evaluate(){
  [[ -f "$SAVE" ]] || fail "missing completed checkpoint $SAVE"
  CKPT="$SAVE" TAG="${RUN_ID}_seed4042_h50p15" \
    exec scripts/run_mw_mt50_acceptance_v1.sh
}

command -v flock >/dev/null || fail "flock is required"
exec 9>"$LOCK"
flock -n 9 || fail "another H50 joint launcher owns the lock"
case "$MODE" in
  preflight) preflight ;;
  train) run_train fresh ;;
  resume) run_train exact ;;
  eval) evaluate ;;
  *) fail "usage: $0 {preflight|train|resume|eval}" ;;
esac
