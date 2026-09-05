#!/usr/bin/env bash
# Continue the slot-free H15 policy from complete MT50 episodes.  Crop starts
# are generated online; this runner never consumes or builds a windows payload.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
ROOT=${EXPAND60_ROOT:-/root/ora0_all49_expand60_v1}
DATA_DIR=${EXPAND60_DATA_DIR:-$ROOT/data_v2}
FRAMES_DIR=${EXPAND60_FRAMES_DIR:-$ROOT/frames_v2}
BASE_RAW_MANIFEST=${BASE_RAW_MANIFEST:-$DATA_DIR/mt50_raw_canonical_identity_60ep_v2.json}
RAW_MANIFEST=${RAW_MANIFEST:-$DATA_DIR/mt50_raw_canonical_identity_60ep_online_repaired_v3.json}
ONLINE_INDEX=${ONLINE_INDEX:-$DATA_DIR/mt50_full_episode_online_index_v1.json}
REPLACEMENT_DIR=${REPLACEMENT_DIR:-$ROOT/online_replacements_v1}
REPAIRED_FRAMES_DIR=${REPAIRED_FRAMES_DIR:-$ROOT/frames_online_repaired_v1}
EXISTING_EVAL=${EXISTING_EVAL:-/root/ora0_all49_data/all49_peer_h15_p15_eval_v1.pt}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/root/ora0_ckpts/mw_all49_wam4va_h15_p15_full10722_e23_lang_slotfree_scratch_v5d16_s5152.pt}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
MODE=${1:-preflight}
BATCH=${BATCH:-48}
EPOCHS=${EPOCHS:-62}
if [[ "$MODE" == antiforget || "$MODE" == antiforget-resume ]]; then
  BASE_CHECKPOINT=${ANTIFORGET_BASE_CHECKPOINT:-/root/ora0_ckpts/mw_mt50_wam4va_h15_full_episode_online60_e62_s24986_lang_slotfree_sparse_v3_s21762.pt}
  EPOCHS=10
elif [[ "$MODE" == recovery ]]; then
  BASE_CHECKPOINT=/root/ora0_ckpts/mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5_s2015.pt
  EPOCHS=1
elif [[ "$MODE" == dagger ]]; then
  BASE_CHECKPOINT=${DAGGER_BASE_CHECKPOINT:-/root/ora0_ckpts/mw_mt50_recovery25_mixed4_anchor25_pcgrad_lr1e5_from_s2015_e1_v1.pt}
  ONLINE_INDEX=${DAGGER_ONLINE_INDEX:-$DATA_DIR/mt50_full_episode_online_dagger_v1.json}
  EPOCHS=1
elif [[ "$MODE" == capacity16 ]]; then
  BASE_CHECKPOINT=${CAPACITY_BASE_CHECKPOINT:-/root/private_data/ORA0/checkpoints/mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5_s3224.pt}
  EPOCHS=50
  BATCH=${CAPACITY_BATCH:-20}
fi
MIXED_TASKS_PER_BATCH=${MIXED_TASKS_PER_BATCH:-4}
[[ "$MODE" != capacity16 ]] || MIXED_TASKS_PER_BATCH=${CAPACITY_MIXED_TASKS_PER_BATCH:-5}
ONLINE_SAMPLES_PER_EPISODE=${ONLINE_SAMPLES_PER_EPISODE:-6}
EXPECTED_SOURCE_EPISODES=${EXPECTED_SOURCE_EPISODES:-3420}
EXPECTED_TRAIN_EPISODES=${EXPECTED_TRAIN_EPISODES:-3222}
EXPECTED_EVAL_EPISODES=${EXPECTED_EVAL_EPISODES:-198}
NGPUS=${NGPUS:-2}
INDEX_WORKERS=${INDEX_WORKERS:-6}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/ora0_ckpts}
MAIN_VISION_ENCODE_BATCH=${MAIN_VISION_ENCODE_BATCH:-16}
DAGGER_TASK_LOCALITY_BLOCK_BATCHES=${DAGGER_TASK_LOCALITY_BLOCK_BATCHES:-1}
DAGGER_LONGTRAJ_DECODE_CACHE_TASKS=${DAGGER_LONGTRAJ_DECODE_CACHE_TASKS:-220}
if [[ "$MODE" == antiforget || "$MODE" == antiforget-resume || "$MODE" == recovery || "$MODE" == dagger || "$MODE" == capacity16 ]]; then
  PEER_BATCH_PREFETCH_DEPTH=${PEER_BATCH_PREFETCH_DEPTH:-4}
else
  PEER_BATCH_PREFETCH_DEPTH=${PEER_BATCH_PREFETCH_DEPTH:-16}
fi
LOCK=/tmp/ora0_mt50_full_episode_online_h15.lock

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage(){ printf 'usage: %s {prepare|preflight|joint|resume|antiforget|antiforget-resume|recovery|dagger|capacity16}\n' "$0" >&2; exit 2; }

prepare(){
  for path in "$BASE_RAW_MANIFEST" "$EXISTING_EVAL"; do
    [[ -f "$path" ]] || fail "missing $path"
  done
  if [[ ! -f "$RAW_MANIFEST" ]]; then
    scripts/collect_mt50_online_repairs.sh
    "$PY" -u -B -m scripts.repair_mt50_zero_supervision \
      --raw-manifest "$BASE_RAW_MANIFEST" \
      --replacement-dir "$REPLACEMENT_DIR" \
      --output-dir "$REPAIRED_FRAMES_DIR" \
      --output-manifest "$RAW_MANIFEST"
  fi
  "$PY" -u -B -m scripts.build_online_episode_index \
    --raw-manifest "$RAW_MANIFEST" --existing-eval "$EXISTING_EVAL" \
    --output "$ONLINE_INDEX" --workers "$INDEX_WORKERS"
}

read_counts(){
  "$PY" -B - "$ONLINE_INDEX" "$ONLINE_SAMPLES_PER_EPISODE" "$BATCH" "$EPOCHS" "$MODE" \
    "$EXPECTED_SOURCE_EPISODES" "$EXPECTED_TRAIN_EPISODES" "$EXPECTED_EVAL_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
samples_per_episode, batch, epochs = map(int, sys.argv[2:5])
mode = sys.argv[5]
expected = tuple(map(int, sys.argv[6:9]))
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("contract") != "full_episode_online_random_h15_v1":
    raise SystemExit("wrong online episode contract")
if (data.get("sampling_protocol") or {}).get("offline_windows") is not False:
    raise SystemExit("offline window payload is forbidden")
counts = data.get("counts") or {}
triple = tuple(counts.get(key) for key in (
    "source_episodes", "train_episodes", "eval_episodes"
))
if mode == "dagger":
    if not data.get("dagger_augmentation") or triple[1] <= 3222 or triple[2] != 198:
        raise SystemExit(f"wrong DAgger episode counts: {triple}")
elif triple != expected:
    raise SystemExit(f"wrong MT50 episode counts: {triple}, expected {expected}")
samples = triple[1] * samples_per_episode
steps_per_epoch = (samples + batch - 1) // batch
print(triple[0], triple[1], triple[2], samples, steps_per_epoch, epochs * steps_per_epoch)
PY
}

preflight(){
  for path in "$ONLINE_INDEX" "$RAW_MANIFEST" "$EXISTING_EVAL" \
    "$BASE_CHECKPOINT" "$DINO"; do
    [[ -f "$path" ]] || fail "missing $path"
  done
  if [[ "$MODE" == capacity16 ]]; then
    [[ "$BATCH" == 20 ]] || fail "capacity run global batch must remain 20"
    [[ "$MIXED_TASKS_PER_BATCH" == 5 ]] || \
      fail "capacity run must mix exactly 5 tasks per batch"
  else
    [[ "$BATCH" == 48 ]] || fail "formal global batch must remain 48"
  fi
  if [[ "$MODE" == antiforget || "$MODE" == antiforget-resume ]]; then
    [[ "$EPOCHS" == 10 ]] || fail "antiforget run must run exactly 10 epochs"
  elif [[ "$MODE" == recovery || "$MODE" == dagger ]]; then
    [[ "$EPOCHS" == 1 ]] || fail "recovery run must run exactly 1 epoch"
  elif [[ "$MODE" == capacity16 ]]; then
    [[ "$EPOCHS" == 50 ]] || fail "capacity run must run exactly 50 epochs"
  else
    [[ "$EPOCHS" == 62 ]] || fail "formal continuation must remain 62 epochs"
  fi
  [[ "$ONLINE_SAMPLES_PER_EPISODE" == 6 ]] || \
    fail "formal exposure match requires 6 online samples per episode"
  (( BATCH % NGPUS == 0 )) || fail "batch must divide across GPUs"
  if [[ "$MODE" == dagger ]]; then
    "$PY" -B - "$ONLINE_INDEX" "$FRAMES_DIR" <<'PY'
import sys
from va_compound.longtraj_frames import OnlineLongTrajEpisodeDataset

dataset = OnlineLongTrajEpisodeDataset(
    sys.argv[1], longtraj_dir=sys.argv[2], samples_per_episode=6,
    recovery_samples_per_episode=3, decode_cache_tasks=100,
)
print(f"DAgger online dataset: PASS rows={len(dataset)}")
PY
  else
    "$PY" -B -m scripts.build_online_episode_index \
      --raw-manifest "$RAW_MANIFEST" --existing-eval "$EXISTING_EVAL" \
      --output "$ONLINE_INDEX" --validate-only
  fi
  read -r SOURCE_EPISODES TRAIN_EPISODES EVAL_EPISODES \
    ONLINE_SAMPLES STEPS_PER_EPOCH STEPS < <(read_counts)
  (( ONLINE_SAMPLES % NGPUS == 0 )) || fail "online epoch rows must divide across GPUs"
  printf 'MT50 full-episode online preflight: PASS; source=%s train=%s eval=%s; ' \
    "$SOURCE_EPISODES" "$TRAIN_EPISODES" "$EVAL_EPISODES"
  printf 'random_samples/epoch=%s steps/epoch=%s; %s epochs=%s steps; offline_windows=0\n' \
    "$ONLINE_SAMPLES" "$STEPS_PER_EPOCH" "$EPOCHS" "$STEPS"
}

run_joint(){
  local launch_mode=${1:-fresh}
  preflight
  read -r _ _ _ _ STEPS_PER_EPOCH STEPS < <(read_counts)
  local default_run_id=mw_mt50_wam4va_h15_full_episode_online60_e62_s${STEPS}_lang_slotfree_resume_v1
  if [[ "$launch_mode" == antiforget* ]]; then
    default_run_id=mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5
  elif [[ "$launch_mode" == recovery ]]; then
    default_run_id=mw_mt50_recovery25_mixed4_anchor25_pcgrad_lr1e5_from_s2015_e1_v1
  elif [[ "$launch_mode" == dagger ]]; then
    default_run_id=mw_mt50_dagger_mixed4_anchor25_pcgrad_lr1e5_from_recovery_e1_v1
  elif [[ "$launch_mode" == capacity16 ]]; then
    default_run_id=mw_mt50_capacity_va16_wm205m_world15_gate7_mixed5_b20_anchor25_pcgrad_lr1e5_from_s3224_e50_v1
  fi
  local run_id=${RUN_ID:-$default_run_id}
  local save=$CHECKPOINT_DIR/$run_id.pt
  local log=logs/$run_id.log
  local run_steps=$STEPS
  local resume_args=(--resume-weights "$BASE_CHECKPOINT")
  local va_layers=8
  local wmrm_predictor_depth=6
  local wmrm_predictor_copies=1
  local wmrm_feature_metric=mse
  local depth_args=()
  local strategy_args=(
    --single-task --task-sampling full --task-locality-block-batches 64
    --longtraj-decode-cache-tasks 2 --lr 0.0001
  )
  local save_every=$((3 * STEPS_PER_EPOCH))
  if [[ "$launch_mode" == antiforget* || "$launch_mode" == capacity16 ]]; then
    local capacity_lr=0.00001
    [[ "$launch_mode" != capacity16 ]] || capacity_lr=${CAPACITY_LR:-0.00001}
    strategy_args=(
      --single-task --task-sampling mixed --mixed-tasks-per-batch "$MIXED_TASKS_PER_BATCH"
      --anchor-replay-fraction 0.25 --pcgrad
      --task-locality-block-batches 1 --longtraj-decode-cache-tasks 50
      --lr "$capacity_lr"
    )
    save_every=$STEPS_PER_EPOCH
  elif [[ "$launch_mode" == recovery ]]; then
    strategy_args=(
      --single-task --task-sampling mixed --mixed-tasks-per-batch 4
      --anchor-replay-fraction 0.25 --pcgrad
      --task-locality-block-batches 1 --longtraj-decode-cache-tasks 50
      --online-recovery-samples-per-episode 3 --lr 0.00001
    )
    save_every=$STEPS_PER_EPOCH
  elif [[ "$launch_mode" == dagger ]]; then
    strategy_args=(
      --single-task --task-sampling mixed --mixed-tasks-per-batch 4
      --anchor-replay-fraction 0.25 --pcgrad
      --task-locality-block-batches "$DAGGER_TASK_LOCALITY_BLOCK_BATCHES"
      --longtraj-decode-cache-tasks "$DAGGER_LONGTRAJ_DECODE_CACHE_TASKS"
      --online-recovery-samples-per-episode 3 --lr 0.00001
    )
    save_every=$STEPS_PER_EPOCH
  fi
  if [[ "$launch_mode" == capacity16 ]]; then
    va_layers=16
    wmrm_predictor_depth=7
    wmrm_predictor_copies=11
    wmrm_feature_metric=cosine
    depth_args=(
      --wmrm-stage-gate-start 7
      --wmrm-progress-ordinal-weight "${CAPACITY_PROGRESS_ORDINAL_WEIGHT:-0}"
    )
    if [[ "${CAPACITY_RESUME_EXPANDED:-0}" != 1 ]]; then
      depth_args+=(
        --resume-weights-migration peer_va8_world7_to_va16_world15_gated_capacity_v1
      )
    fi
    if [[ "${CAPACITY_NEW_ONLY:-0}" == 1 ]]; then
      depth_args+=(--capacity-new-only)
    fi
    if [[ "${CAPACITY_PHASE2_GATES:-0}" == 1 ]]; then
      [[ "${CAPACITY_RESUME_EXPANDED:-0}" == 1 ]] || \
        fail "CAPACITY_PHASE2_GATES requires CAPACITY_RESUME_EXPANDED=1"
      [[ "${CAPACITY_NEW_ONLY:-0}" != 1 ]] || \
        fail "CAPACITY_PHASE2_GATES cannot run during CAPACITY_NEW_ONLY"
      depth_args+=(--capacity-phase2-gates)
    fi
  fi
  if [[ -n "${RUN_STEPS_OVERRIDE:-}" ]]; then
    (( RUN_STEPS_OVERRIDE > 0 )) || fail "RUN_STEPS_OVERRIDE must be positive"
    run_steps=$RUN_STEPS_OVERRIDE
  fi
  local save_args=(
    --save-every "$save_every" --save-step-copies --save "$save"
  )
  if [[ "${NO_SAVE:-0}" == 1 ]]; then
    save_args=()
  fi
  local tee_args=()
  if [[ "$launch_mode" == *exact ]]; then
    [[ -f "$save" ]] || fail "exact resume checkpoint is missing: $save"
    local completed
    completed=$("$PY" -B - "$save" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(checkpoint.get("global_step", -1)))
PY
)
    (( completed > 0 && completed < STEPS )) || \
      fail "checkpoint global_step=$completed is outside (0,$STEPS)"
    run_steps=$((STEPS - completed))
    resume_args=(--resume-exact "$save")
    tee_args=(-a)
    printf 'exact resume: global_step=%s remaining=%s target=%s\n' \
      "$completed" "$run_steps" "$STEPS"
  else
    [[ ! -e "$save" && ! -e "$log" ]] || fail "refusing to overwrite run $run_id"
  fi
  ! pgrep -af '[p]ython.*train.py' >/dev/null || fail 'another train.py is active'
  mkdir -p "$CHECKPOINT_DIR" logs
  local launcher=("$PY" -u -B)
  if (( NGPUS > 1 )); then
    launcher=("$PY" -m torch.distributed.run --standalone \
      --nproc_per_node="$NGPUS" --max_restarts=0)
  fi
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${launcher[@]}" train.py \
    --va-data "$ONLINE_INDEX" --world-data "$ONLINE_INDEX" \
    --online-episode-sampling --online-episode-samples "$ONLINE_SAMPLES_PER_EPISODE" \
    --peer-shared-full-data --visual-world-supervision \
    --world-split-manifest "$ONLINE_INDEX" \
    --va-world-mode peer_sync_h6 --planning-stride 15 --control-stride 15 \
    --deployment-execution-horizon 15 --wam4va --wmrm-full-language-tokens \
    --slot-free-policy --wmrm-inject all --wmrm-target dino \
    --wmrm-adep-weight 0 --wmrm-cycle-steps 15 --wmrm-world-weight 1.0 \
    --world-action-rank-stage final --dino-main-vision \
    --main-vision-checkpoint "$DINO" --main-vision-grid 16 --main-vision-frames 4 \
    --main-vision-temporal --main-vision-temporal-scale 1.0 \
    --main-vision-encode-batch "$MAIN_VISION_ENCODE_BATCH" \
    --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
    --wmrm-predictor st_blocks --wmrm-predictor-depth "$wmrm_predictor_depth" \
    --wmrm-predictor-width 384 --wmrm-predictor-heads 12 \
    --wmrm-predictor-copies "$wmrm_predictor_copies" "${strategy_args[@]}" \
    --wmrm-feature-metric "$wmrm_feature_metric" \
    --batch-size "$BATCH" --sequence-length 4 \
    --min-sequence-length 4 --num-workers 0 --peer-batch-prefetch \
    --peer-batch-prefetch-depth "$PEER_BATCH_PREFETCH_DEPTH" \
    --disable-runtime-integrity-checks \
    --seed 0 --device cuda --feature-autocast-bf16 \
    --va-layers "$va_layers" --va-attention-backend auto --flow-cond adaln \
    --flow-layers 6 --flow-steps 8 --flow-prefix-steps 15 \
    --flow-prefix-weight 1.0 --flow-tail-weight 1.0 --steps "$run_steps" \
    "${save_args[@]}" \
    --longtraj-dir "$FRAMES_DIR" "${resume_args[@]}" "${depth_args[@]}" \
    2>&1 | tee "${tee_args[@]}" "$log"
}

command -v flock >/dev/null || fail 'flock is required'
exec 9>"$LOCK"
flock -n 9 || fail 'another MT50 online run owns the lock'
case "$MODE" in
  prepare) prepare; preflight ;;
  preflight) preflight ;;
  joint) run_joint ;;
  resume) run_joint exact ;;
  antiforget) run_joint antiforget ;;
  antiforget-resume) run_joint antiforget-exact ;;
  recovery) run_joint recovery ;;
  dagger) run_joint dagger ;;
  capacity16) fail "capacity16 is retired; keep the s3224 VA8/World7 model" ;;
  *) usage ;;
esac
