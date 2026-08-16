#!/usr/bin/env bash
# Status for the 10k+20k experiment. Does not start a second trainer.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/mw_hard2_wam4va_10k20k.log
STATUS=artifacts/EXPERIMENT_STATUS.md
WORLD=checkpoints/mw_hard2_wam4va_world_10k.pt
JOINT=checkpoints/mw_hard2_wam4va_joint.pt
mkdir -p artifacts
{
  echo "# WAM4VA 10k+20k status"
  echo
  echo "updated: $(date -Iseconds)"
  echo
  if pgrep -f "python -u -B train.py" >/dev/null; then
    echo "- trainer: RUNNING"
    pgrep -af "python -u -B train.py" | head -2 | sed 's/^/  - /'
  else
    echo "- trainer: STOPPED"
  fi
  if [[ -f "$LOG" ]]; then
    echo "- last log lines:"
    rg "^step=|^=====|done |Error|Traceback" "$LOG" | tail -8 | sed 's/^/  - /'
  else
    echo "- log missing: $LOG"
  fi
  echo "- world_10k: $([[ -f $WORLD ]] && echo YES || echo NO)"
  echo "- joint latest: $([[ -f $JOINT ]] && echo YES || echo NO)"
  echo "- joint step copies: $(ls checkpoints/mw_hard2_wam4va_joint_s*.pt 2>/dev/null | wc -l)"
  if [[ -f "$LOG" ]] && rg -q "^done " "$LOG"; then
    echo "- experiment train: DONE"
  fi
} | tee "$STATUS"
