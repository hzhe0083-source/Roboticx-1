#!/usr/bin/env bash
# Formal EvoMind-compatible MT50 acceptance for the current H15 checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
CKPT=${CKPT:-/root/private_data/ORA0/checkpoints/mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5_s3224.pt}
FEATURES=${FEATURES:-/root/private_data/ORA0/features/all49_peer_h15_p15_eval_v1.pt}
LANGUAGE_FEATURES=${LANGUAGE_FEATURES:-/root/private_data/ORA0/mt50_dagger_recovery_r1_r2/data_v2/mt50_language_normalization_ref_v2.pt}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
OSMESA_LIB=${OSMESA_LIB:-/root/private_data/ORA0/runtime_libs/osmesa_jammy/usr/lib/x86_64-linux-gnu}
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SHARDS=${SHARDS:-5}
EVAL_GPUS=${EVAL_GPUS:-0,1}
TAG=${TAG:-mt50_evomind_seed4042_h15}

for path in "$CKPT" "$FEATURES" "$LANGUAGE_FEATURES" "$DINO"; do
  [[ -f "$path" ]] || { printf 'ERROR: missing %s\n' "$path" >&2; exit 1; }
done
[[ -d "$OSMESA_LIB" ]] || { printf 'ERROR: missing %s\n' "$OSMESA_LIB" >&2; exit 1; }
if pgrep -f "train.py.*$(basename "$CKPT")" >/dev/null; then
  printf 'ERROR: training is still writing %s\n' "$CKPT" >&2
  exit 1
fi

exec "$PY" -u -B scripts/eval_parallel.py "$CKPT" "$FEATURES" \
  --language-features "$LANGUAGE_FEATURES" \
  --dino "$DINO" --python "$PY" --gpus "$EVAL_GPUS" --shards "$SHARDS" \
  --trials-per-task 10 --episode-seed-base 4042 \
  --execution-horizon 15 --horizon 400 --mt50-benchmark --tag "$TAG"
