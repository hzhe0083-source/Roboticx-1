#!/bin/bash
# E7 joint80k 长程监视器：2 分钟轮询，任何崩坏模式即记录现场并退出（触发通知）
# 退出条件：进程退出 / NaN/Inf/OOM/Traceback/Xid / step 停滞 12 分钟 / 磁盘 < 3G
set -u
cd /home/ryan/Documents/robot/ORA0 || exit 1
LOG=logs/monitor/joint80k_watch.log
mkdir -p logs/monitor

SAVES=$(rg -c 'periodic checkpoint saved' logs/e7_mtvj_joint80k.log 2>/dev/null || echo 0)
LAST_STEP=$(rg -o 'step=[0-9]+' logs/e7_mtvj_joint80k.log | tail -1 | cut -d= -f2)
LAST_T=$(date +%s)
WARNED_DISK=0

echo "WATCH_START $(date '+%F %T') saves=$SAVES step=$LAST_STEP" > "$LOG"
while true; do
  # 1. 进程退出 → 训练结束或崩溃（^ 锚定命令行开头，避免匹配 bash wrapper/cron 进程）
  if ! pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train\.py' >/dev/null 2>&1; then
    { echo "PROCESS_EXITED $(date '+%F %T')"; echo "--- log tail ---"; tail -n 8 logs/e7_mtvj_joint80k.log; echo "--- final ckpt ---"; ls -lh checkpoints/e7_mtvj_joint80k.pt 2>&1; } >> "$LOG"
    break
  fi

  # 2. 落盘记录
  NOW=$(rg -c 'periodic checkpoint saved' logs/e7_mtvj_joint80k.log 2>/dev/null || echo 0)
  if [ "$NOW" -gt "$SAVES" ]; then
    echo "SAVE#$NOW $(date '+%F %T') last: $(tail -1 logs/e7_mtvj_joint80k.log)" >> "$LOG"
    SAVES=$NOW
  fi

  # 3. 日志文本异常（最近 300 行）
  if tail -n 300 logs/e7_mtvj_joint80k.log | rg -qi 'traceback|cuda out of memory|no space left|Xid|nan|inf'; then
    { echo "ANOMALY_TEXT $(date '+%F %T')"; echo "--- last 25 lines ---"; tail -n 25 logs/e7_mtvj_joint80k.log; } >> "$LOG"
    break
  fi

  # 4. step 停滞 > 12 分钟
  CUR=$(rg -o 'step=[0-9]+' logs/e7_mtvj_joint80k.log | tail -1 | cut -d= -f2)
  NOW_T=$(date +%s)
  if [ "$CUR" = "$LAST_STEP" ]; then
    if [ $((NOW_T - LAST_T)) -ge 720 ]; then
      { echo "ANOMALY_STALLED step=$CUR 自 $(date -d @"$LAST_T" '+%F %T') 起 12 分钟无进展，当前 $(date '+%F %T')"; tail -n 10 logs/e7_mtvj_joint80k.log; } >> "$LOG"
      break
    fi
  else
    LAST_STEP=$CUR
    LAST_T=$NOW_T
  fi

  # 5. 磁盘 < 3G → 异常退出；< 8G → 预警一次
  DF_KB=$(df -P /home | awk 'NR==2{print $4}')
  if [ "$DF_KB" -lt 3145728 ]; then
    { echo "ANOMALY_DISK $(date '+%F %T')"; df -h /home; } >> "$LOG"
    break
  fi
  if [ "$DF_KB" -lt 8388608 ] && [ "$WARNED_DISK" -eq 0 ]; then
    echo "DISK_WARN $(date '+%F %T') 剩余 $(df -h /home | tail -1 | awk '{print $4}')" >> "$LOG"
    WARNED_DISK=1
  fi

  sleep 120
done
echo "WATCH_END $(date '+%F %T')" >> "$LOG"
