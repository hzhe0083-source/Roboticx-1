#!/usr/bin/env bash
# Archive selected atomic task35 FM checkpoints without touching the trainer.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=${1:?usage: $0 training.log checkpoint.pt [step ...]}
CKPT=${2:?missing checkpoint path}
shift 2
if (( $# > 0 )); then
  STEPS=("$@")
else
  STEPS=(1000 2000 3000 6000 9000 12000 15000 18000 20000)
fi
[[ -f "$LOG" ]] || { echo "missing log: $LOG" >&2; exit 1; }
mkdir -p "$(dirname "$CKPT")" logs
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}

wanted() {
  local candidate=$1
  local step
  for step in "${STEPS[@]}"; do
    [[ "$candidate" == "$step" ]] && return 0
  done
  return 1
}

# Copy + SHA only. Full CPU validate/slice stays in the milestone waiter so a
# validate OOM or contract failure cannot abort the remaining 9k/12k/15k copies.
verify_copy_sha() {
  local source=$1
  local destination=$2
  local source_sha destination_sha
  source_sha=$(sha256sum "$source" | awk '{print $1}')
  destination_sha=$(sha256sum "$destination" | awk '{print $1}')
  if [[ -z "$source_sha" || "$source_sha" != "$destination_sha" ]]; then
    echo "archive SHA mismatch src=$source_sha dst=$destination_sha" >&2
    return 1
  fi
  printf '%s  %s\n' "$destination_sha" "$destination"
}

archive_complete() {
  local destination=$1
  [[ -s "$destination" && -s "${destination}.sha256" ]]
}

peek_step() {
  "$PY" -B scripts/peek_task35_checkpoint_step.py "$1"
}

hash_existing() {
  local destination=$1
  local digest
  digest=$(sha256sum "$destination" | awk '{print $1}')
  if [[ -z "$digest" ]]; then
    echo "failed to hash existing archive $destination" >&2
    return 1
  fi
  printf '%s  %s\n' "$digest" "$destination" > "${destination}.sha256"
  echo "wrote missing sha256 for $destination" >&2
}

archive_step() {
  local step=$1
  local stem=${CKPT%.pt}
  local destination="${stem}_step${step}.pt"
  local temporary="${destination}.tmp"
  local sidecar attempt found
  if archive_complete "$destination"; then
    echo "milestone already exists: $destination" >&2
    return 0
  fi
  if [[ -s "$destination" ]]; then
    # Never replace a same-name file with a later live checkpoint.
    if found=$(peek_step "$destination") && [[ "$found" == "$step" ]]; then
      hash_existing "$destination"
      return 0
    fi
    echo "refuse to recopy live over $destination (found_step=${found:-?}; want=$step)" >&2
    return 1
  fi
  [[ -f "$CKPT" ]] || {
    echo "checkpoint missing after save event: $CKPT" >&2
    return 1
  }
  if found=$(peek_step "$CKPT") && [[ "$found" != "$step" ]]; then
    echo "refuse to copy live global_step=$found as step=$step" >&2
    return 1
  fi
  for attempt in 1 2; do
    rm -f "$temporary"
    cp --reflink=auto --sparse=always "$CKPT" "$temporary"
    if sidecar=$(verify_copy_sha "$CKPT" "$temporary"); then
      if found=$(peek_step "$temporary") && [[ "$found" != "$step" ]]; then
        echo "copied file is global_step=$found, not $step; discarding" >&2
        rm -f "$temporary"
        return 1
      fi
      mv "$temporary" "$destination"
      printf '%s\n' "$sidecar" > "${destination}.sha256"
      echo "archived global_step=$step to $destination" >&2
      return 0
    fi
    echo "archive copy retry $attempt failed for step=$step" >&2
  done
  rm -f "$temporary"
  echo "failed to archive an intact copy of step=$step" >&2
  return 1
}

backfill_existing() {
  local line step
  while IFS= read -r line; do
    if [[ "$line" =~ global_step=([0-9]+).*periodic\ checkpoint\ saved ]]; then
      step=${BASH_REMATCH[1]}
      if wanted "$step"; then
        local destination="${CKPT%.pt}_step${step}.pt"
        if archive_complete "$destination"; then
          echo "backfill skip existing $destination" >&2
        else
          echo "backfill cannot copy historical step=$step without that exact file" >&2
        fi
      fi
    fi
  done < "$LOG"
}

all_done() {
  local step destination
  for step in "${STEPS[@]}"; do
    destination="${CKPT%.pt}_step${step}.pt"
    archive_complete "$destination" || return 1
  done
  return 0
}

backfill_existing
if all_done; then
  echo "all requested milestones already archived" >&2
  exit 0
fi

# The trainer writes checkpoint.tmp, atomically replaces CKPT, and only then
# prints this event. tail -F is therefore an event stream, not a file poller.
# A 30s read timeout lets us notice a dead trainer without dropping save events.
exec 3< <(tail -n 0 -F "$LOG")
TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true; exec 3<&-' EXIT
while true; do
  if IFS= read -r -t 30 -u 3 line; then
    if [[ "$line" =~ global_step=([0-9]+).*periodic\ checkpoint\ saved ]]; then
      step=${BASH_REMATCH[1]}
      if wanted "$step"; then
        # A failed 6k copy must not abort the 9k/12k/15k listener.
        if ! archive_step "$step"; then
          echo "archive_step failed for step=$step; keep listening for later milestones" >&2
        fi
        all_done && exit 0
      fi
    fi
  elif ! kill -0 "$TAIL_PID" 2>/dev/null; then
    echo "log follower died before all milestones were archived" >&2
    exit 4
  fi
  if ! "$PY" -B scripts/task35_proc.py --check pipeline >/dev/null; then
    if all_done; then
      echo "pipeline gone; all requested milestones already archived" >&2
      exit 0
    fi
    echo "pipeline gone before all milestones were archived" >&2
    exit 4
  fi
done
