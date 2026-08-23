#!/usr/bin/env bash
# Add three deterministic 10-episode recovery shards to every ordinary MT50 task.
# assembly-v3 and door-unlock-v3 already have 270 episodes and are left untouched.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
RAW_IDENTITY_MANIFEST=${RAW_IDENTITY_MANIFEST:-/root/ora0_all49_data/all49_raw_canonical_identity_v1.json}
NORM_REF=${NORM_REF:-/root/private_data/ORA0/data/longtraj_normalization_ref.pt}
OUT_ROOT=${OUT_ROOT:-/root/ora0_all49_expand60_v1}
SHARD_ROOT=$OUT_ROOT/shards
LOG_ROOT=$OUT_ROOT/logs
MAX_JOBS=${MAX_JOBS:-4}
CPUSET=${CPUSET:-0-31}
DRY_RUN=${DRY_RUN:-0}
COLLECTOR=${COLLECTOR:-scripts/collect_long_trajectories.py}
COLLECT_LD_PRELOAD=${COLLECT_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}

EPISODE_SEED_BASE=600000
COLLECTOR_RNG_BASE=900000
SHARDS_PER_TASK=3
EPISODES_PER_SHARD=10

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1"
command -v "$PY" >/dev/null 2>&1 || fail "missing Python interpreter: $PY"
[[ -f "$RAW_IDENTITY_MANIFEST" ]] || fail "missing raw identity manifest: $RAW_IDENTITY_MANIFEST"
[[ -f "$NORM_REF" ]] || fail "missing normalization reference: $NORM_REF"
[[ -f "$COLLECTOR" ]] || fail "missing collector: $COLLECTOR"

# The manifest order is the canonical global task-id order used by evaluation.
task_output=$("$PY" -B - "$RAW_IDENTITY_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("contract") != "all49_canonical_raw_sources_v1":
    raise SystemExit(f"unexpected raw identity contract: {payload.get('contract')!r}")
sources = list(payload.get("sources") or [])
tasks = [str(item.get("task") or "") for item in sources]
if len(tasks) != 49 or len(set(tasks)) != 49 or any(not task for task in tasks):
    raise SystemExit("raw identity manifest must contain 49 unique non-empty tasks")
excluded = {"assembly-v3", "door-unlock-v3"}
if not excluded.issubset(tasks):
    raise SystemExit(f"raw identity manifest is missing excluded hard tasks: {excluded - set(tasks)}")
for task_id, task in enumerate(tasks):
    if task not in excluded:
        print(f"{task_id}\t{task}")
PY
) || fail "could not read canonical task order"
mapfile -t TASK_ROWS <<<"$task_output"
[[ ${#TASK_ROWS[@]} -eq 47 ]] || fail "expected 47 ordinary tasks, got ${#TASK_ROWS[@]}"

validate_existing_shard() {
  local path=$1 task=$2 seed_start=$3
  local expected=()
  local offset
  for ((offset = 0; offset < EPISODES_PER_SHARD; offset++)); do
    expected+=("$((seed_start + offset))")
  done
  "$PY" -B - "$path" "$task" "${expected[@]}" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected_task = sys.argv[2]
expected_seeds = [int(value) for value in sys.argv[3:]]
payload = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise SystemExit(f"{path}: payload is not a dict")
if payload.get("task") != expected_task:
    raise SystemExit(
        f"{path}: task={payload.get('task')!r}, expected {expected_task!r}"
    )
episodes = payload.get("episodes")
if not isinstance(episodes, (list, tuple)) or len(episodes) != 10:
    raise SystemExit(f"{path}: expected exactly 10 episodes")
if payload.get("n_episodes") != 10:
    raise SystemExit(f"{path}: n_episodes must equal 10")
actual_seeds = [episode.get("episode_seed") for episode in episodes]
if actual_seeds != expected_seeds:
    raise SystemExit(
        f"{path}: episode seeds mismatch: actual={actual_seeds}, "
        f"expected={expected_seeds}"
    )
bad_events = [
    index
    for index, episode in enumerate(episodes)
    if not isinstance(episode, dict)
    or episode.get("n_perturb_events") is None
    or int(episode["n_perturb_events"]) < 1
]
if bad_events:
    raise SystemExit(f"{path}: episodes without a recovery perturbation: {bad_events}")
print(f"[skip valid] {path}: task={expected_task} episodes=10")
PY
}

declare -a PLAN_TASK_IDS=()
declare -a PLAN_TASKS=()
declare -a PLAN_SHARDS=()
declare -a PLAN_SEED_STARTS=()

# Validate every existing artifact before starting any new process. Invalid or
# stale artifacts stop the run; this script never deletes or overwrites data.
for row in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r task_id task <<<"$row"
  for ((shard = 0; shard < SHARDS_PER_TASK; shard++)); do
    seed_start=$((EPISODE_SEED_BASE + task_id * 1000 + shard * EPISODES_PER_SHARD))
    task_dir=$SHARD_ROOT/$task
    out=$task_dir/metaworld_longtraj_${task}_recovery_v2_shard${shard}.pt
    tmp=$task_dir/metaworld_longtraj_${task}_recovery_v2_shard${shard}.pt.tmp
    [[ ! -e "$tmp" && ! -L "$tmp" ]] || fail "stale collector temporary exists: $tmp"
    if [[ -e "$out" || -L "$out" ]]; then
      [[ -f "$out" ]] || fail "existing shard is not a regular file: $out"
      validate_existing_shard "$out" "$task" "$seed_start"
      continue
    fi
    PLAN_TASK_IDS+=("$task_id")
    PLAN_TASKS+=("$task")
    PLAN_SHARDS+=("$shard")
    PLAN_SEED_STARTS+=("$seed_start")
  done
done

running=0
failures=0

reap_one() {
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  running=$((running - 1))
}

for index in "${!PLAN_TASKS[@]}"; do
  task_id=${PLAN_TASK_IDS[$index]}
  task=${PLAN_TASKS[$index]}
  shard=${PLAN_SHARDS[$index]}
  seed_start=${PLAN_SEED_STARTS[$index]}
  collector_seed=$((COLLECTOR_RNG_BASE + task_id * 10 + shard))
  task_dir=$SHARD_ROOT/$task
  log_dir=$LOG_ROOT/$task
  out=$task_dir/metaworld_longtraj_${task}_recovery_v2_shard${shard}.pt
  log=$log_dir/shard${shard}.log
  seeds=()
  for ((offset = 0; offset < EPISODES_PER_SHARD; offset++)); do
    seeds+=("$((seed_start + offset))")
  done
  cmd=(
    nice -n 10 ionice -c 3 taskset -c "$CPUSET"
    env
    "CUDA_VISIBLE_DEVICES="
    "MUJOCO_GL=osmesa"
    "LP_NUM_THREADS=2"
    "OMP_NUM_THREADS=1"
    "MKL_NUM_THREADS=1"
    "OPENBLAS_NUM_THREADS=1"
    "LD_PRELOAD=$COLLECT_LD_PRELOAD"
    "PYTHONDONTWRITEBYTECODE=1"
    "$PY" -u -B "$COLLECTOR"
    --task "$task"
    --seed "$collector_seed"
    --episode-seeds "${seeds[@]}"
    --force-perturb
    --normalization-ref "$NORM_REF"
    --output "$out"
  )

  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf ' > %q 2>&1\n' "$log"
    continue
  fi

  mkdir -p "$task_dir" "$log_dir"
  while ((running >= MAX_JOBS)); do
    reap_one
  done
  printf '[launch] task=%s shard=%d seeds=%d..%d log=%s\n' \
    "$task" "$shard" "$seed_start" "$((seed_start + EPISODES_PER_SHARD - 1))" "$log"
  "${cmd[@]}" >"$log" 2>&1 &
  running=$((running + 1))
done

if [[ "$DRY_RUN" == 1 ]]; then
  printf '[dry-run] planned=%d existing_valid=%d total=%d\n' \
    "${#PLAN_TASKS[@]}" "$((47 * SHARDS_PER_TASK - ${#PLAN_TASKS[@]}))" \
    "$((47 * SHARDS_PER_TASK))"
  exit 0
fi

while ((running > 0)); do
  reap_one
done
if ((failures > 0)); then
  fail "$failures collection job(s) failed; inspect logs and rerun missing shards"
fi

# Re-open the newly written atomic artifacts before declaring the expansion
# complete.  A collector exit code alone does not prove the seed/perturbation
# contract if a future collector version changes its payload.
for index in "${!PLAN_TASKS[@]}"; do
  task=${PLAN_TASKS[$index]}
  shard=${PLAN_SHARDS[$index]}
  seed_start=${PLAN_SEED_STARTS[$index]}
  out=$SHARD_ROOT/$task/metaworld_longtraj_${task}_recovery_v2_shard${shard}.pt
  [[ -f "$out" ]] || fail "collector did not create shard: $out"
  validate_existing_shard "$out" "$task" "$seed_start"
done
printf '[ok] all 47 ordinary tasks have three validated-or-collected 10-episode shards\n'
