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
  STEPS=(1000 3000 6000 9000 12000 15000)
fi
[[ -f "$LOG" ]] || { echo "missing log: $LOG" >&2; exit 1; }
mkdir -p "$(dirname "$CKPT")"

wanted() {
  local candidate=$1
  local step
  for step in "${STEPS[@]}"; do
    [[ "$candidate" == "$step" ]] && return 0
  done
  return 1
}

archive_step() {
  local step=$1
  local stem=${CKPT%.pt}
  local destination="${stem}_step${step}.pt"
  local temporary="${destination}.tmp"
  [[ -f "$CKPT" ]] || {
    echo "checkpoint missing after save event: $CKPT" >&2
    return 1
  }
  [[ ! -e "$destination" ]] || {
    echo "milestone already exists: $destination" >&2
    return 1
  }
  rm -f "$temporary"
  cp --reflink=auto --sparse=always "$CKPT" "$temporary"
  mv "$temporary" "$destination"
  sha256sum "$destination" > "${destination}.sha256"
  echo "archived global_step=$step to $destination" >&2
}

# The trainer writes checkpoint.tmp, atomically replaces CKPT, and only then
# prints this event. tail -F is therefore an event stream, not a file poller.
while IFS= read -r line; do
  if [[ "$line" =~ global_step=([0-9]+).*periodic\ checkpoint\ saved ]]; then
    step=${BASH_REMATCH[1]}
    if wanted "$step"; then
      archive_step "$step"
      [[ "$step" == "${STEPS[-1]}" ]] && exit 0
    fi
  fi
done < <(tail -n 0 -F "$LOG")
