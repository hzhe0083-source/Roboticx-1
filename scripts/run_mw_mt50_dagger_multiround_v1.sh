#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=${EXPAND60_ROOT:-/root/ora0_all49_expand60_v1}
DATA_DIR=${EXPAND60_DATA_DIR:-$ROOT/data_v2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/ora0_ckpts}
ROUNDS=${ROUNDS:-3}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-20}
REPEAT=${REPEAT:-1}
SHARDS=${SHARDS:-5}
EVAL_GPUS=${EVAL_GPUS:-0,1}
CURRENT_CKPT=${CURRENT_CKPT:-$CHECKPOINT_DIR/mw_mt50_recovery25_mixed4_anchor25_pcgrad_lr1e5_from_s2015_e1_v1.pt}
CURRENT_INDEX=${CURRENT_INDEX:-$DATA_DIR/mt50_full_episode_online_index_v1.json}
SEED_BASES=(${DAGGER_SEED_BASES:-14042 24042 44042})

(( ROUNDS >= 1 && ROUNDS <= ${#SEED_BASES[@]} )) || {
  printf 'ERROR: ROUNDS must be between 1 and %s\n' "${#SEED_BASES[@]}" >&2
  exit 1
}

for ((round = 1; round <= ROUNDS; round++)); do
  seed=${SEED_BASES[round - 1]}
  dagger_dir=$ROOT/dagger_round${round}_s${seed}_v1
  dagger_index=$DATA_DIR/mt50_full_episode_online_dagger_r${round}_v1.json
  run_id=mw_mt50_dagger_r${round}_mixed4_anchor25_pcgrad_lr1e5_v1
  next_ckpt=$CHECKPOINT_DIR/$run_id.pt

  printf '[dagger-multiround] round=%s/%s collect checkpoint=%s seed=%s trials/task=%s\n' \
    "$round" "$ROUNDS" "$CURRENT_CKPT" "$seed" "$TRIALS_PER_TASK"
  CKPT="$CURRENT_CKPT" DAGGER_DIR="$dagger_dir" BASE_INDEX="$CURRENT_INDEX" \
    DAGGER_INDEX="$dagger_index" TRIALS_PER_TASK="$TRIALS_PER_TASK" \
    EPISODE_SEED_BASE="$seed" REPEAT="$REPEAT" SHARDS="$SHARDS" \
    EVAL_GPUS="$EVAL_GPUS" TAG="dagger_r${round}_seed${seed}_h15" \
    scripts/run_mw_mt50_dagger_collect_v1.sh

  printf '[dagger-multiround] round=%s/%s train index=%s\n' \
    "$round" "$ROUNDS" "$dagger_index"
  DAGGER_BASE_CHECKPOINT="$CURRENT_CKPT" DAGGER_ONLINE_INDEX="$dagger_index" \
    RUN_ID="$run_id" scripts/run_mw_mt50_wam4va_h15_60ep_v2.sh dagger
  [[ -f "$next_ckpt" ]] || { printf 'ERROR: missing %s\n' "$next_ckpt" >&2; exit 1; }
  CURRENT_CKPT=$next_ckpt
  CURRENT_INDEX=$dagger_index
done

printf '[dagger-multiround] formal acceptance checkpoint=%s\n' "$CURRENT_CKPT"
CKPT="$CURRENT_CKPT" TAG="dagger_multiround_r${ROUNDS}_seed4042_h15" \
  SHARDS="$SHARDS" EVAL_GPUS="$EVAL_GPUS" scripts/run_mw_mt50_acceptance_v1.sh
