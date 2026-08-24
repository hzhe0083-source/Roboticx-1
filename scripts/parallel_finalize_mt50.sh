#!/usr/bin/env bash
# Finish the remaining per-task raw merges with bounded process parallelism,
# then resume the verified MT50 source/train/eval build.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
WORKERS=${MT50_MERGE_WORKERS:-6}
BASE_MANIFEST=${BASE_MANIFEST:-/root/ora0_all49_data/all49_raw_canonical_identity_v1.json}
SHARD_ROOT=${SHARD_ROOT:-/root/ora0_all49_expand60_v1/shards}
FRAMES_DIR=${FRAMES_DIR:-/root/ora0_all49_expand60_v1/frames_v2}
LOG_ROOT=${LOG_ROOT:-/root/ora0_all49_expand60_v1/merge_logs_v2}

[[ "$WORKERS" =~ ^[0-9]+$ ]] && (( WORKERS >= 4 && WORKERS <= 8 )) || {
  printf 'ERROR: MT50_MERGE_WORKERS must be in [4,8]\n' >&2
  exit 1
}
exec 9>/tmp/ora0_parallel_finalize_mt50.lock
flock -n 9 || { printf '[skip] parallel MT50 finalizer already running\n'; exit 0; }
mkdir -p "$FRAMES_DIR" "$LOG_ROOT"

mapfile -t jobs < <("$PY" -B - "$BASE_MANIFEST" "$SHARD_ROOT" "$FRAMES_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
shard_root = Path(sys.argv[2])
frames = Path(sys.argv[3])
for item in manifest["sources"]:
    task = str(item["task"])
    if task in {"assembly-v3", "door-unlock-v3"}:
        continue
    output = frames / f"metaworld_longtraj_{task}.pt"
    if not output.exists():
        print(f"{task}\t{item['source_path']}\t{shard_root / task}")
PY
)

running=0
failures=0
reap_one() {
  wait -n || failures=$((failures + 1))
  running=$((running - 1))
}

printf '[parallel-merge] missing=%d workers=%d\n' "${#jobs[@]}" "$WORKERS"
for row in "${jobs[@]}"; do
  IFS=$'\t' read -r task base shard_dir <<<"$row"
  while (( running >= WORKERS )); do reap_one; done
  "$PY" -u -B -m scripts.merge_longtraj_expansion \
    --base "$base" --shard-dir "$shard_dir" --out-dir "$FRAMES_DIR" \
    >"$LOG_ROOT/$task.log" 2>&1 &
  running=$((running + 1))
  printf '[launch] %s pid=%d\n' "$task" "$!"
done
while (( running > 0 )); do reap_one; done
(( failures == 0 )) || {
  printf 'ERROR: %d parallel merge process(es) failed\n' "$failures" >&2
  exit 1
}

printf '[parallel-merge] complete; resuming MT50 finalizer\n'
exec "$PY" -u -B -m scripts.finalize_all49_to60
