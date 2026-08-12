#!/bin/bash
# 66k→100k 管道监视器：检测 pipeline 退出/阶段 checkpoint/异常文本。
set -u
cd /home/ryan/Documents/robot/ORA0 || exit 1
LOG=logs/monitor/to100k_watch.log
mkdir -p logs/monitor
LOG_B=logs/e7_mtvj_all49_stageB_71k.log
LOG_SMOKE=logs/e7_mtvj_all49_stageC_smoke16.log
LOG_C=logs/e7_mtvj_all49_stageC_100k.log
echo "TO100K_WATCH_START $(date '+%F %T')" > "$LOG"
while true; do
  if ! pgrep -f '^bash scripts/run_e7_all49_to100k\.sh' >/dev/null 2>&1; then
    { echo "PIPELINE_EXITED $(date '+%F %T')"; echo "--- pipeline log tail ---"; tail -n 5 logs/pipeline_to100k.log; ls -lh checkpoints/e7_mtvj_all49_stageB_71k.pt checkpoints/e7_mtvj_all49_stageC_100k.pt 2>&1; } >> "$LOG"
    break
  fi
  for F in "$LOG_B" "$LOG_SMOKE" "$LOG_C"; do
    if [ -f "$F" ]; then
      if tail -n 400 "$F" | rg -qi 'traceback|cuda out of memory|no space left|\bnan\b|\binf\b|AssertionError'; then
        { echo "ANOMALY $(date '+%F %T') in $F"; tail -n 15 "$F"; } >> "$LOG"
        break 2
      fi
    fi
  done
  DF_KB=$(df -P /home | awk 'NR==2{print $4}')
  if [ "$DF_KB" -lt 3145728 ]; then
    { echo "ANOMALY_DISK $(date '+%F %T')"; df -h /home; } >> "$LOG"
    break
  fi
  sleep 180
done
echo "TO100K_WATCH_END $(date '+%F %T')" >> "$LOG"
