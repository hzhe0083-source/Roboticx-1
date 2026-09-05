#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

exec 9>/tmp/ora0_wait_finalize_mt50.lock
flock -n 9 || { printf '[skip] MT50 finalizer already queued\n'; exit 0; }

SHARD_ROOT=${SHARD_ROOT:-/root/ora0_all49_expand60_v1/shards}
while :; do
  count=$(find "$SHARD_ROOT" -type f -name '*recovery_v2_shard*.pt' | wc -l)
  (( count == 141 )) && break
  if ! pgrep -f '[r]etry_missing_expand60.sh' >/dev/null; then
    printf 'ERROR: collection stopped with %d/141 shards\n' "$count" >&2
    exit 1
  fi
  sleep 20
done

printf '[%s] starting MT50 finalization\n' "$(date '+%F %T')"
exec nice -n 10 ionice -c 3 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /opt/conda/bin/python -u -B -m scripts.finalize_all49_to60
