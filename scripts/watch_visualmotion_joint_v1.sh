#!/usr/bin/env bash
# Live status for the joint_v1 300->1000->20000 continuation.
# Usage:
#   bash scripts/watch_visualmotion_joint_v1.sh
#   watch -n 5 bash /home/ryan/Documents/robot/ORA0/scripts/watch_visualmotion_joint_v1.sh
set -euo pipefail
ROOT=/home/ryan/Documents/robot/ORA0
cd "$ROOT"

LOG1000=logs/mw_hard2_wam4va_visualmotion_joint_v1.train300to1000.log
LOG20000=logs/mw_hard2_wam4va_visualmotion_joint_v1.train1000to20000.log
if [[ -f "$LOG20000" ]]; then
  LOG=$LOG20000
  TARGET=20000
else
  LOG=$LOG1000
  TARGET=1000
fi

printf 'time   %s\n' "$(date '+%F %T %z')"
if p=$(pgrep -f '/home/ryan/.venvs/openvla/bin/python -u -B train.py .*mw_hard2_wam4va_visualmotion_joint_v1' | head -n 1); then
  ps -o pid,stat,etime,%cpu,%mem,rss= -p "$p" | sed 's/^/proc   /'
else
  echo 'proc   STOPPED'
fi

if [[ -f "$LOG" ]]; then
  step=$(rg -o 'step=[0-9]+' "$LOG" | tail -n 1 | cut -d= -f2 || true)
  line=$(rg '^step=' "$LOG" | tail -n 1 || true)
  loss=$(printf '%s\n' "$line" | rg -o 'loss=[0-9.]+' | tail -n 1 || true)
  flow=$(printf '%s\n' "$line" | rg -o 'flow=[0-9.]+' | tail -n 1 || true)
  world=$(printf '%s\n' "$line" | rg -o 'world=[0-9.]+' | tail -n 1 || true)
  task=$(printf '%s\n' "$line" | rg -o 'task=[^ ]+' | tail -n 1 || true)
  printf 'log    %s\n' "$LOG"
  printf 'step   %s / %s  %s %s %s %s\n' "${step:-?}" "$TARGET" "${task:-}" "${loss:-}" "${flow:-}" "${world:-}"
  if [[ -n "${step:-}" && "$step" -lt "$TARGET" ]]; then
    remain=$((TARGET - step))
    # ~5.4 s/step from the current 300->1000 segment
    eta_min=$((remain * 54 / 10 / 60))
    printf 'eta    ~%s min to %s\n' "$eta_min" "$TARGET"
  fi
  echo 'last   '"$line"
else
  echo "log    missing $LOG"
fi
