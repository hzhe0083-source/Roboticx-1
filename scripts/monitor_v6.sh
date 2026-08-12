#!/bin/bash
# V6 视觉迁移监视器（交接 2026-08-12 19:59）：进程消失、NaN/Inf/OOM、磁盘、
# 连续两个 250-step 窗口 RMSE 明显恶化 → 记录现场并退出（触发通知）。
set -u
cd /home/ryan/Documents/robot/ORA0 || exit 1
LOG=logs/monitor/v6_ft5k_watch.log
mkdir -p logs/monitor
TRAIN_LOG=logs/train_metric_v6_all49_contractfix_ft5k.log
echo "V6_WATCH_START $(date '+%F %T')" > "$LOG"
LAST_RMSE=""
STALL=0
while true; do
  if ! pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train_metric_visual' >/dev/null 2>&1; then
    { echo "V6_PROCESS_EXITED $(date '+%F %T')"; tail -n 8 "$TRAIN_LOG"; ls -lh checkpoints/metric_field_v6_all49_contractfix_init10k_ft5k.pt 2>&1; } >> "$LOG"
    break
  fi
  if tail -n 300 "$TRAIN_LOG" | rg -qi 'traceback|cuda out of memory|no space left|xid|nan|inf'; then
    { echo "V6_ANOMALY_TEXT $(date '+%F %T')"; tail -n 20 "$TRAIN_LOG"; } >> "$LOG"
    break
  fi
  DF_KB=$(df -P /home | awk 'NR==2{print $4}')
  if [ "$DF_KB" -lt 3145728 ]; then
    { echo "V6_ANOMALY_DISK $(date '+%F %T')"; df -h /home; } >> "$LOG"
    break
  fi
  # 连续两个 250-step 窗口 RMSE 明显恶化（>+20%）
  RMSE=$(rg 'train RMSE' "$TRAIN_LOG" | tail -1 | rg -o 'RMSE [0-9.]+' | awk '{print $2}')
  if [ -n "$RMSE" ]; then
    if [ -n "$LAST_RMSE" ]; then
      WORSE=$(python3 -c "print(1 if float('$RMSE') > 1.2*float('$LAST_RMSE') else 0)")
      if [ "$WORSE" = "1" ]; then STALL=$((STALL+1)); else STALL=0; fi
      if [ "$STALL" -ge 2 ]; then
        { echo "V6_RMSE_DEGRADED $(date '+%F %T') last=$LAST_RMSE now=$RMSE"; tail -n 10 "$TRAIN_LOG"; } >> "$LOG"
        break
      fi
    fi
    LAST_RMSE=$RMSE
  fi
  sleep 120
done
echo "V6_WATCH_END $(date '+%F %T')" >> "$LOG"
