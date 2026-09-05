#!/usr/bin/env bash
# Replace nine reset-already-successful demonstrations with real recovery runs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
ROOT=${OUT_ROOT:-/root/ora0_all49_expand60_v1}
OUT=$ROOT/online_replacements_v1
NORM_REF=${NORM_REF:-/root/private_data/ORA0/data/longtraj_normalization_ref.pt}
CPUSET=${CPUSET:-0-31}
mkdir -p "$OUT" "$OUT/logs"

tasks=(coffee-button-v3 faucet-open-v3 faucet-close-v3)
counts=(2 3 4)
rng_seeds=(2608101 2608102 2608103)
pids=()

for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  count=${counts[$i]}
  out=$OUT/metaworld_longtraj_${task}_online_repair_v1.pt
  log=$OUT/logs/${task}.log
  if [[ -f "$out" ]]; then
    printf '[skip] %s\n' "$out"
    continue
  fi
  nice -n 10 ionice -c 3 taskset -c "$CPUSET" env \
    CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa LP_NUM_THREADS=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -u -B -m scripts.collect_long_trajectories \
    --task "$task" --episodes "$count" --seed "${rng_seeds[$i]}" \
    --force-perturb --normalization-ref "$NORM_REF" --output "$out" \
    >"$log" 2>&1 &
  pids+=("$!")
  printf '[launch] task=%s count=%s pid=%s\n' "$task" "$count" "${pids[-1]}"
done

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done
(( failures == 0 )) || { printf 'ERROR: %s repair collectors failed\n' "$failures" >&2; exit 1; }

"$PY" -B - "$OUT" <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch

root = Path(sys.argv[1])
expected = {"coffee-button-v3": 2, "faucet-open-v3": 3, "faucet-close-v3": 4}
seeds = set()
for task, count in expected.items():
    path = root / f"metaworld_longtraj_{task}_online_repair_v1.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes") or []
    if payload.get("task") != task or len(episodes) != count:
        raise SystemExit(f"{path}: expected {count} {task} episodes")
    for episode in episodes:
        valid = np.asarray(episode.get("action_supervision_valid", []), dtype=bool)
        if not valid.any() or int(episode.get("n_perturb_events", 0)) < 1:
            raise SystemExit(f"{path}: replacement has no recovery supervision")
        seed = int(episode["episode_seed"])
        if seed in seeds or 35000 <= seed < 35050:
            raise SystemExit(f"duplicate/eval replacement seed: {seed}")
        seeds.add(seed)
print(f"[ok] online repairs: episodes={sum(expected.values())} unique_seeds={len(seeds)}")
PY
