#!/bin/bash
# E7 joint80k 恢复脚本（OOM 恢复用）：resume joint2000 checkpoint，补 28000 步 → 有效 80k
set -u
cd /home/ryan/Documents/robot/ORA0 || exit 1
LOG=logs/e7_mtvj_joint80k.log
if pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train\.py' >/dev/null 2>&1; then
  echo "已有训练进程在运行，拒绝启动（避免双训练）"; exit 1
fi
setsid nohup /home/ryan/.venvs/pytorch-gpu/bin/python -u train.py \
  --data data/metaworld_longtraj_windows_h48.pt \
  --dense-readout-mtvj \
  --metric-visual-checkpoint checkpoints/metric_field_v4.pt \
  --mtvj-train-relation --lr-mtvj-relation 2e-5 \
  --single-task --va-layers 8 \
  --lr 5e-6 --batch-size 16 --steps 28000 --seed 0 \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 \
  --task-sampling weighted --prev-dropout 0.1 \
  --resume checkpoints/e7_mtvj_joint80k.pt \
  --save checkpoints/e7_mtvj_joint80k.pt --save-every 1000 \
  >> "$LOG" 2>&1 < /dev/null &
echo "resumed, pid=$!"
sleep 30
tail -n 15 "$LOG" | rg '冻结|trainable|relation|resumed|step=' | tail -8
