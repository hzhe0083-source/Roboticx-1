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
  STEPS=(1000 2000 3000 6000 9000 12000 15000)
fi
[[ -f "$LOG" ]] || { echo "missing log: $LOG" >&2; exit 1; }
mkdir -p "$(dirname "$CKPT")" logs
PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
VALIDATE=scripts/validate_task35_fm_checkpoint.py

wanted() {
  local candidate=$1
  local step
  for step in "${STEPS[@]}"; do
    [[ "$candidate" == "$step" ]] && return 0
  done
  return 1
}

validate_destination() {
  local destination=$1
  local step=$2
  local name
  name=$(basename "${destination%.pt}")
  local report=logs/${name}_validate.json
  local lock=logs/${name}.lock
  [[ -f "$VALIDATE" ]] || return 0
  mkdir -p logs
  (
    flock 8
    if [[ ! -s "$report" ]]; then
      CUDA_VISIBLE_DEVICES= "$PY" -B "$VALIDATE" \
        "$destination" --expected-step "$step" --output "$report"
    fi
  ) 8>"$lock"
}

archive_step() {
  local step=$1
  local stem=${CKPT%.pt}
  local destination="${stem}_step${step}.pt"
  local temporary="${destination}.tmp"
  if [[ -e "$destination" ]]; then
    echo "milestone already exists: $destination" >&2
    return 0
  fi
  [[ -f "$CKPT" ]] || {
    echo "checkpoint missing after save event: $CKPT" >&2
    return 1
  }
  rm -f "$temporary"
  cp --reflink=auto --sparse=always "$CKPT" "$temporary"
  mv "$temporary" "$destination"
  sha256sum "$destination" > "${destination}.sha256"
  echo "archived global_step=$step to $destination" >&2
  validate_destination "$destination" "$step"
}

backfill_existing() {
  local line step
  while IFS= read -r line; do
    if [[ "$line" =~ global_step=([0-9]+).*periodic\ checkpoint\ saved ]]; then
      step=${BASH_REMATCH[1]}
      if wanted "$step"; then
        local destination="${CKPT%.pt}_step${step}.pt"
        if [[ -e "$destination" ]]; then
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
    [[ -e "$destination" ]] || return 1
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
        archive_step "$step"
        all_done && exit 0
      fi
    fi
  elif ! kill -0 "$TAIL_PID" 2>/dev/null; then
    echo "log follower died before all milestones were archived" >&2
    exit 4
  fi
  if ! "$PY" -B scripts/task35_proc.py --check trainer >/dev/null; then
    if all_done; then
      echo "trainer gone; all requested milestones already archived" >&2
      exit 0
    fi
    echo "trainer gone before all milestones were archived" >&2
    exit 4
  fi
done
