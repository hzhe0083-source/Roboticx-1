#!/bin/bash
# 阶段 B/C 自动单线（门控流水线，2026-08-12）：
#   Gate A（视觉门，49×50 正式评测）→ 阶段 B（66k→71k 冻结适配）
#   → Gate B（16 步真实联合 smoke，验证第 8/16 步 aux loss/梯度）
#   → 阶段 C（71k→100k 双数据流联合）
# 任何门失败即停，保留最后完好 PT。用法：
#   bash scripts/pipeline_stageBC.sh [--threshold RMSE_PX] [--stage b|c|all]
set -u
cd /home/ryan/Documents/robot/ORA0 || exit 1
PY=/home/ryan/.venvs/pytorch-gpu/bin/python
THRESHOLD=${THRESHOLD:-15.0}
STAGE=${1:-all}
EVAL_JSON=${EVAL_JSON:-artifacts/metric_v5_all49_10k_heldout49x50.json}
CKPT_66K=checkpoints/e7_mtvj_joint66k_step14k.pt
V5_HEAD=checkpoints/metric_field_v5_all49.pt
CKPT_71K=checkpoints/e7_mtvj_stageB_71k.pt
CKPT_100K=checkpoints/e7_mtvj_stageC_100k.pt
LOG_B=logs/e7_mtvj_stageB_71k.log
LOG_SMOKE=logs/e7_mtvj_stageB_smoke16.log
LOG_C=logs/e7_mtvj_stageC_100k.log

fail() { echo "PIPELINE FAIL: $1"; exit 1; }
trap 'echo "PIPELINE INTERRUPTED（保留现场）"; exit 130' INT TERM

# ---------- Gate A：视觉门 ----------
gate_a() {
  echo "[Gate A] 视觉门：$EVAL_JSON"
  [ -f "$EVAL_JSON" ] || fail "评测 JSON 不存在：$EVAL_JSON"
  RMSE=$($PY -c "
import json
r = json.load(open('$EVAL_JSON'))
g = r['metrics']['gated_state']['aggregate']['rmse_px']
print(g)
")
  echo "[Gate A] gated_state RMSE = ${RMSE}px（阈值 ${THRESHOLD}px）"
  python3 -c "exit(0 if float('$RMSE') <= $THRESHOLD else 1)" || \
    fail "视觉门未过：${RMSE}px > ${THRESHOLD}px（保留 v5 head，不启动阶段 B）"
  echo "[Gate A] PASS"
}

# ---------- 阶段 B：66k→71k（冻结视觉头，迁移 v5 head） ----------
stage_b() {
  echo "[阶段 B] 66k→71k：迁移 v5 head + 冻结视觉头，5k 步"
  pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train\.py' >/dev/null && \
    fail "已有 train.py 在跑，拒绝双训练"
  setsid nohup $PY -u train.py \
    --data data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt \
    --dense-readout-mtvj \
    --metric-visual-checkpoint "$V5_HEAD" \
    --replace-mtvj-metric-head-from-external \
    --mtvj-train-relation --lr-mtvj-relation 2e-5 \
    --single-task --va-layers 8 \
    --lr 5e-6 --batch-size 16 --steps 5000 --seed 0 \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 \
    --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
    --task-sampling weighted --task-locality-block-batches 16 \
    --prev-dropout 0.1 \
    --resume "$CKPT_66K" \
    --save "$CKPT_71K" --save-every 1000 \
    >> "$LOG_B" 2>&1 < /dev/null &
  echo "[阶段 B] 已启动 pid=$!（日志 $LOG_B）"
  while pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train\.py' >/dev/null; do
    if rg -q 'Traceback|CUDA out of memory|No space left' "$LOG_B" 2>/dev/null; then
      fail "阶段 B 日志异常（tail）"
    fi
    sleep 60
  done
  [ -f "$CKPT_71K" ] || fail "阶段 B 未产出 checkpoint"
  tail -2 "$LOG_B" | cut -c1-140
  echo "[阶段 B] DONE: $CKPT_71K"
}

# ---------- Gate B：16 步真实联合 smoke ----------
gate_b() {
  echo "[Gate B] 16 步联合 smoke（--mtvj-train-metric-head --mtvj-visual-aux-every 8）"
  setsid nohup $PY -u train.py \
    --data data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt \
    --dense-readout-mtvj \
    --metric-visual-checkpoint "$V5_HEAD" \
    --mtvj-train-relation --lr-mtvj-relation 2e-5 \
    --mtvj-train-metric-head --lr-mtvj-metric-head 1e-6 \
    --mtvj-visual-aux-every 8 --mtvj-visual-aux-loc-lambda 1.0 \
    --mtvj-visual-aux-vis-lambda 0.5 --mtvj-visual-aux-batch 8 \
    --single-task --va-layers 8 \
    --lr 5e-6 --batch-size 16 --steps 16 --seed 0 \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 \
    --task-sampling weighted --task-locality-block-batches 16 \
    --prev-dropout 0.1 \
    --resume "$CKPT_71K" \
    --save /tmp/stageB_smoke16.pt \
    >> "$LOG_SMOKE" 2>&1 < /dev/null &
  while pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train\.py' >/dev/null; do sleep 10; done
  rg -q 'Traceback|CUDA out of memory|No space left' "$LOG_SMOKE" && \
    fail "smoke 日志异常"
  for S in 8 16; do
    LINE=$(rg "step=$S " "$LOG_SMOKE" | tail -1)
    echo "$LINE" | rg -q 'aux_hinge=|aux_rmse=' || fail "step=$S 缺 aux 字段"
    echo "$LINE" | rg -q 'metric_grad=0\.000000' && fail "step=$S metric_grad 为 0（视觉头无梯度）"
    echo "[Gate B] step=$S aux 字段 OK：$(echo "$LINE" | rg -o 'aux_hinge=[0-9.]+ aux_pos=[0-9.]+ aux_offset=[0-9.]+ aux_vis=[0-9.]+ aux_rmse=[0-9.]+px')"
  done
  echo "[Gate B] PASS（第 8/16 步 aux loss 与梯度正常）"
}

# ---------- 阶段 C：71k→100k（双数据流联合，29k 步） ----------
stage_c() {
  echo "[阶段 C] 71k→100k：解冻视觉头 + 双数据流联合，29k 步"
  pgrep -f '^/home/ryan/.venvs/pytorch-gpu/bin/python -u train\.py' >/dev/null && \
    fail "已有 train.py 在跑，拒绝双训练"
  setsid nohup $PY -u train.py \
    --data data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt \
    --dense-readout-mtvj \
    --metric-visual-checkpoint "$V5_HEAD" \
    --mtvj-train-relation --lr-mtvj-relation 2e-5 \
    --mtvj-train-metric-head --lr-mtvj-metric-head 1e-6 \
    --mtvj-visual-aux-every 8 --mtvj-visual-aux-loc-lambda 1.0 \
    --mtvj-visual-aux-vis-lambda 0.5 --mtvj-visual-aux-batch 8 \
    --single-task --va-layers 8 \
    --lr 5e-6 --batch-size 16 --steps 29000 --seed 0 \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 \
    --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
    --task-sampling weighted --task-locality-block-batches 16 \
    --prev-dropout 0.1 \
    --resume "$CKPT_71K" \
    --save "$CKPT_100K" --save-every 1000 \
    >> "$LOG_C" 2>&1 < /dev/null &
  echo "[阶段 C] 已启动 pid=$!（日志 $LOG_C）"
}

case "$STAGE" in
  all) gate_a && stage_b && gate_b && stage_c ;;
  b)   gate_a && stage_b ;;
  c)   [ -f "$CKPT_71K" ] || fail "缺少 $CKPT_71K（先跑阶段 B）"
       gate_b && stage_c ;;
  *)   echo "usage: $0 [all|b|c]"; exit 2 ;;
esac
echo "PIPELINE OK"
