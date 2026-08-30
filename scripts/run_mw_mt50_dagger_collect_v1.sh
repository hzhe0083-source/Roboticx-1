#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
CKPT=${CKPT:-/root/ora0_ckpts/mw_mt50_recovery25_mixed4_anchor25_pcgrad_lr1e5_from_s2015_e1_v1.pt}
FEATURES=${FEATURES:-/root/ora0_all49_data/all49_peer_h15_p15_eval_v1.pt}
LANGUAGE_FEATURES=${LANGUAGE_FEATURES:-/root/ora0_all49_expand60_v1/data_v2/mt50_language_normalization_ref_v2.pt}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
DAGGER_DIR=${DAGGER_DIR:-/root/ora0_all49_expand60_v1/dagger_s14042_v1}
BASE_INDEX=${BASE_INDEX:-/root/ora0_all49_expand60_v1/data_v2/mt50_full_episode_online_index_v1.json}
DAGGER_INDEX=${DAGGER_INDEX:-/root/ora0_all49_expand60_v1/data_v2/mt50_full_episode_online_dagger_v1.json}
SHARDS=${SHARDS:-5}
EVAL_GPUS=${EVAL_GPUS:-0,1}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-10}
EPISODE_SEED_BASE=${EPISODE_SEED_BASE:-14042}
REPEAT=${REPEAT:-4}
TAG=${TAG:-dagger_seed${EPISODE_SEED_BASE}_h15}

for path in "$CKPT" "$FEATURES" "$LANGUAGE_FEATURES" "$DINO" "$BASE_INDEX"; do
  [[ -f "$path" ]] || { printf 'ERROR: missing %s\n' "$path" >&2; exit 1; }
done
[[ ! -e "$DAGGER_INDEX" ]] || { printf 'ERROR: refusing to overwrite %s\n' "$DAGGER_INDEX" >&2; exit 1; }
mkdir -p "$DAGGER_DIR"

"$PY" -u -B scripts/eval_parallel.py "$CKPT" "$FEATURES" \
  --language-features "$LANGUAGE_FEATURES" --dino "$DINO" --python "$PY" \
  --gpus "$EVAL_GPUS" --shards "$SHARDS" --trials-per-task "$TRIALS_PER_TASK" \
  --episode-seed-base "$EPISODE_SEED_BASE" --execution-horizon 15 --horizon 400 \
  --dagger-output-dir "$DAGGER_DIR" --dagger-takeover-min 45 \
  --dagger-takeover-max 120 --dagger-prefix-keep 45 \
  --tag "$TAG"

"$PY" -u -B -m scripts.build_dagger_online_index \
  --base-index "$BASE_INDEX" --dagger-dir "$DAGGER_DIR" \
  --output "$DAGGER_INDEX" --repeat "$REPEAT"

printf 'DAgger collection/index ready: %s\n' "$DAGGER_INDEX"
