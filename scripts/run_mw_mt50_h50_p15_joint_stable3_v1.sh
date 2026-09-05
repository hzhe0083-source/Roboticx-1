#!/usr/bin/env bash
# Stable H50/P15 joint run: recovery starts, frozen DINO target, cosine World loss.
set -euo pipefail

export ONLINE_INDEX=${ONLINE_INDEX:-/root/private_data/ORA0/mt50_dagger_recovery_r1_r2/data_v2/mt50_evo1_clean2500_dagger970_online_v2.json}
export FRAMES_DIR=${FRAMES_DIR:-/root/evo1_metaworld_longtraj_v1}
export EXPECTED_EPISODES=${EXPECTED_EPISODES:-3470}
export ONLINE_RECOVERY_SAMPLES=${ONLINE_RECOVERY_SAMPLES:-2}
export TRAIN_DINO=${TRAIN_DINO:-0}
export WMRM_FEATURE_METRIC=${WMRM_FEATURE_METRIC:-cosine}
export MODEL_LR=${MODEL_LR:-0.000003}
export WMRM_PREDICTOR_LR=${WMRM_PREDICTOR_LR:-0.00001}
export MAIN_VISION_ENCODE_BATCH=${MAIN_VISION_ENCODE_BATCH:-16}
export LONGTRAJ_DECODE_CACHE_TASKS=${LONGTRAJ_DECODE_CACHE_TASKS:-149}
export PEER_PREFETCH_DEPTH=${PEER_PREFETCH_DEPTH:-4}
export RUN_ID=${RUN_ID:-mw_mt50_h50_p15_joint_clean2500_recovery970_dinofrozen_cosine_async149x4_from_s313_e3_b32_v3}

exec "$(dirname "$0")/run_mw_mt50_h50_p15_joint_full3_v1.sh" "$@"
