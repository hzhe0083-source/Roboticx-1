#!/usr/bin/env bash
# Add 10 recovery episodes to the 50 push-back demos already in LeRobot MT50.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
OUT_ROOT=${OUT_ROOT:-/root/ora0_all49_expand60_v1/push_back}
SHARD_ROOT=$OUT_ROOT/shards
LOG_ROOT=$OUT_ROOT/logs
NORM_REF=${NORM_REF:-/root/private_data/ORA0/data/longtraj_normalization_ref.pt}
CPUSET=${CPUSET:-0-31}
COLLECTOR=${COLLECTOR:-scripts/collect_long_trajectories.py}
LD_PRELOAD_PATH=${LD_PRELOAD_PATH:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -f "$COLLECTOR" ]] || fail "missing collector: $COLLECTOR"
[[ -f "$NORM_REF" ]] || fail "missing normalization reference: $NORM_REF"
mkdir -p "$SHARD_ROOT" "$LOG_ROOT"

validate_shard(){
  local path=$1
  "$PY" -B - "$path" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
payload = torch.load(path, map_location="cpu", weights_only=False)
if payload.get("task") != "push-back-v3" or len(payload.get("episodes") or []) != 10:
    raise SystemExit(f"{path}: expected push-back-v3 with exactly 10 episodes")
episodes = payload["episodes"]
expected = list(range(649030, 649040))
actual = [episode.get("episode_seed") for episode in episodes]
if actual != expected:
    raise SystemExit(f"{path}: seed mismatch {actual} != {expected}")
events = [int(episode.get("n_perturb_events", 0)) for episode in episodes]
perturbed = [bool(episode.get("perturbed", False)) for episode in episodes]
if any(value < 1 for value in events) or not all(perturbed):
    raise SystemExit(f"{path}: recovery shard lacks a perturbation")
print(f"[skip valid] {path}: recovery episodes=10")
PY
}

out=$SHARD_ROOT/metaworld_longtraj_push-back-v3_recovery_v1_shard0.pt
tmp=$SHARD_ROOT/.$(basename "$out").tmp
log=$LOG_ROOT/recovery_shard0.log
[[ ! -e "$tmp" ]] || fail "stale collector temporary: $tmp"
if [[ -e "$out" ]]; then
  [[ -f "$out" ]] || fail "existing output is not a file: $out"
  validate_shard "$out"
else
  seeds=({649030..649039})
  printf '[launch] recovery seeds=649030..649039\n'
  nice -n 10 ionice -c 3 taskset -c "$CPUSET" env \
    CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa LP_NUM_THREADS=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    LD_PRELOAD="$LD_PRELOAD_PATH" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -u -B "$COLLECTOR" --task push-back-v3 --seed 900493 \
    --episode-seeds "${seeds[@]}" --force-perturb \
    --normalization-ref "$NORM_REF" --output "$out" >"$log" 2>&1
  validate_shard "$out"
fi
printf '[ok] push-back-v3 total contract: 50 recovered LeRobot + 10 recovery = 60\n'
