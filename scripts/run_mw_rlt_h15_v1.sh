#!/usr/bin/env bash
# Dense-reward RLT: frozen s3224 VLA, H15 actor/critic, native MetaWorld reward.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${RLT_MUJOCO_GL:-osmesa}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-8}"
OSMESA_LIB=${OSMESA_LIB:-/root/private_data/ORA0/runtime_libs/osmesa_jammy/usr/lib/x86_64-linux-gnu}
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MODE=${1:-preflight}
PY=${PY:-/opt/conda/bin/python}
ROOT=${EXPAND60_ROOT:-/root/ora0_all49_expand60_v1}
DATA_DIR=${EXPAND60_DATA_DIR:-$ROOT/data_v2}
BASE=${BASE:-/root/private_data/ORA0/checkpoints/mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5_s3224.pt}
FEATURES=${FEATURES:-/root/private_data/ORA0/features/all49_peer_h15_p15_eval_v1.pt}
LANGUAGE_FEATURES=${LANGUAGE_FEATURES:-/root/private_data/ORA0/mt50_dagger_recovery_r1_r2/data_v2/mt50_language_normalization_ref_v2.pt}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
DEMO_INDEX=${DEMO_INDEX:-$DATA_DIR/mt50_full_episode_online_dagger_r2_v1.json}
TOKEN=${TOKEN:-/root/private_data/ORA0/checkpoints/mw_mt50_s3224_rl_token_mt50_shared_seed7_persistent_v3.pt}
RLT_SEED=${RLT_SEED:-7}
RLT=${RLT:-/root/private_data/ORA0/checkpoints/mw_mt50_s3224_rlt_dense_h15_pcgrad_seed${RLT_SEED}_v1.pt}
TASK_IDS=${TASK_IDS:-all}
CHUNK_LENGTH=${CHUNK_LENGTH:-15}
TOKEN_STEPS=${TOKEN_STEPS:-10000}
TOKEN_BATCH_SIZE=${TOKEN_BATCH_SIZE:-4}
TOKEN_DECODE_CACHE_TASKS=${TOKEN_DECODE_CACHE_TASKS:-160}
TOKEN_TASK_BLOCK_BATCHES=${TOKEN_TASK_BLOCK_BATCHES:-1}
WARMUP_PER_TASK=${WARMUP_PER_TASK:-2}
ONLINE_PER_TASK=${ONLINE_PER_TASK:-20}
EPISODE_HORIZON=${EPISODE_HORIZON:-400}
WORLD_RESET_EVERY=${WORLD_RESET_EVERY:-4}
REPLAY_STRIDE=${REPLAY_STRIDE:-15}
PREFILL_EPISODES_PER_TASK=${PREFILL_EPISODES_PER_TASK:-0}
REWARD_MODE=${REWARD_MODE:-dense}
REWARD_SCALE=${REWARD_SCALE:-0.01}
UTD=${UTD:-5}
POLICY_DELAY=${POLICY_DELAY:-10}
BETA=${BETA:-5}
SAVE_EVERY_EPISODES=${SAVE_EVERY_EPISODES:-50}
COLLECTORS_PER_GPU=${COLLECTORS_PER_GPU:-4}
COLLECTOR_DEVICES=${COLLECTOR_DEVICES:-}

for path in "$PY" "$BASE" "$FEATURES" "$LANGUAGE_FEATURES" "$DINO"; do
  [[ -f "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
[[ -d "$OSMESA_LIB" ]] || { echo "missing: $OSMESA_LIB" >&2; exit 1; }

active_tasks=$TASK_IDS
if [[ "$MODE" == smoke ]]; then
  active_tasks=${SMOKE_TASK_IDS:-0,35}
fi
COMMON=(
  --checkpoint "$BASE"
  --features "$FEATURES"
  --language-features "$LANGUAGE_FEATURES"
  --main-vision-checkpoint "$DINO"
  --task-ids "$active_tasks"
  --chunk-length "$CHUNK_LENGTH"
  --world-reset-every "$WORLD_RESET_EVERY"
  --reward-mode "$REWARD_MODE"
  --reward-scale "$REWARD_SCALE"
  --replay-prefill-episodes-per-task "$PREFILL_EPISODES_PER_TASK"
  --replay-stride "$REPLAY_STRIDE"
  --seed "$RLT_SEED"
  --device cuda
)
if [[ -n "$COLLECTOR_DEVICES" ]]; then
  COMMON+=(--collector-devices "$COLLECTOR_DEVICES")
fi

if [[ "$MODE" == preflight ]]; then
  exec "$PY" -u -B train_rlt_metaworld.py --mode preflight "${COMMON[@]}"
fi
if [[ "$MODE" == token ]]; then
  [[ -f "$DEMO_INDEX" ]] || { echo "missing: $DEMO_INDEX" >&2; exit 1; }
  mkdir -p logs
  LOG="logs/rlt_token_task${active_tasks//,/-}_seed${RLT_SEED}_$(date +%Y%m%d_%H%M%S).log"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B train_rlt_metaworld.py --mode token-train "${COMMON[@]}" \
      --demo-index "$DEMO_INDEX" --longtraj-dir "$DATA_DIR" \
      --token-steps "$TOKEN_STEPS" --token-batch-size "$TOKEN_BATCH_SIZE" \
      --token-decode-cache-tasks "$TOKEN_DECODE_CACHE_TASKS" \
      --token-task-block-batches "$TOKEN_TASK_BLOCK_BATCHES" \
      --token-dagger-only \
      --output "$TOKEN" 2>&1 | tee "$LOG"
  exit ${PIPESTATUS[0]}
fi
if [[ "$MODE" == eval ]]; then
  [[ -f "$RLT" ]] || { echo "missing: $RLT" >&2; exit 1; }
  EVAL_ARGS=(
    --rlt-checkpoint "$RLT"
    --eval-episodes "${EVAL_EPISODES:-10}"
    --episode-horizon "$EPISODE_HORIZON"
  )
  if [[ -n "${EVAL_EPISODE_SEED_BASE:-}" ]]; then
    EVAL_ARGS+=(--eval-episode-seed-base "$EVAL_EPISODE_SEED_BASE")
  fi
  exec "$PY" -u -B train_rlt_metaworld.py --mode eval "${COMMON[@]}" "${EVAL_ARGS[@]}"
fi
if [[ "$MODE" == all ]]; then
  "$0" token
  exec "$0" train
fi
if [[ "$MODE" != smoke && "$MODE" != train ]]; then
  echo "usage: bash $0 [preflight|token|smoke|train|eval|all]" >&2
  exit 2
fi

[[ -f "$TOKEN" ]] || { echo "missing: $TOKEN (run '$0 token' first)" >&2; exit 1; }
if (( PREFILL_EPISODES_PER_TASK > 0 )); then
  [[ -f "$DEMO_INDEX" ]] || { echo "missing: $DEMO_INDEX" >&2; exit 1; }
fi
if [[ "$MODE" == smoke ]]; then
  warmup=2
  online=2
  bootstrap=20
  eval_episodes=2
  output="${RLT%.pt}_smoke.pt"
  collectors=${SMOKE_COLLECTORS:-2}
else
  if [[ "$TASK_IDS" == all ]]; then
    task_count=$("$PY" -B - "$LANGUAGE_FEATURES" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(len(payload["metadata"]["tasks"]))
PY
)
  else
    IFS=',' read -r -a task_array <<< "$TASK_IDS"
    task_count=${#task_array[@]}
  fi
  warmup=$((WARMUP_PER_TASK * task_count))
  online=$((ONLINE_PER_TASK * task_count))
  bootstrap=${BOOTSTRAP_UPDATES:-1000}
  eval_episodes=${EVAL_EPISODES:-10}
  output=$RLT
  collectors=$COLLECTORS_PER_GPU
fi

[[ ! -e "$output" && ! -e "${output%.pt}_e0.pt" ]] || {
  echo "refusing to overwrite existing RLT checkpoint: $output" >&2
  exit 1
}
mkdir -p logs
LOG="logs/rlt_h15_${MODE}_task${active_tasks//,/-}_seed${RLT_SEED}_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train_rlt_metaworld.py --mode train "${COMMON[@]}" \
    --token-checkpoint "$TOKEN" --demo-index "$DEMO_INDEX" \
    --longtraj-dir "$DATA_DIR" \
    --warmup-episodes "$warmup" \
    --bootstrap-updates "$bootstrap" --online-episodes "$online" \
    --utd "$UTD" --policy-delay "$POLICY_DELAY" --beta "$BETA" \
    --save-every-episodes "$SAVE_EVERY_EPISODES" \
    --eval-episodes "$eval_episodes" --episode-horizon "$EPISODE_HORIZON" \
    --collectors "$collectors" \
    --output "$output" 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
