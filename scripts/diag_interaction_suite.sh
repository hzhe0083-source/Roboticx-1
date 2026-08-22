#!/usr/bin/env bash
# 双模型交互判定套件（顺序两轮，各约 25 分钟）。
#
# 轮 1 —— 还原验证：撤销收缩后重跑基线。忠实的还原应当精确复现 13:11 那次
#          未收缩的 3/20=15%（同一 checkpoint、同一组 seed）。这是后续消融
#          结论成立的前提。
# 轮 2 —— 消息消融：World 状态递推全部照常，只把交给 VA 的 world_message 置零。
#          ≈15% => VA 没在用这份消息，交互在部署时空转
#          显著下降 => 消息确实在贡献
set -euo pipefail
cd /root/private_data/ORA0

export PY=/opt/conda/bin/python
export DINO=/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

CKPT=checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18_s3000.pt
FEATURES=data/hard2_peer_h6_p2_eval_v1.pt
NAME=$(basename "$CKPT" .pt)
BASE=logs/${NAME}_eval10

echo "=== round 1: revert verification START $(date '+%F %T') ==="
"$PY" scripts/revert_contraction.py va_compound/wmrm.py
grep -n "return base + delta\|belief = belief + belief_update" va_compound/wmrm.py

bash scripts/eval_mw_hard2_wam4va.sh "$CKPT" "$FEATURES"
for ext in log json; do
  [[ -f "${BASE}.${ext}" ]] && cp -p "${BASE}.${ext}" "${BASE}.reverted.${ext}"
done
echo "=== round 1 DONE $(date '+%F %T') ==="
tail -3 "${BASE}.reverted.log"

echo "=== round 2: world_message zero-ablation START $(date '+%F %T') ==="
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  MUJOCO_GL=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B scripts/diag_world_message_ablation.py "$CKPT" "$FEATURES" zero \
  2>&1 | tee logs/world_message_ablation_zero.log
echo "=== round 2 DONE $(date '+%F %T') ==="

echo
echo "--- baseline (uncontracted, 13:11) ---"
tail -3 "${BASE}.uncontracted.log"
echo "--- round 1 (reverted, expect identical to baseline) ---"
tail -3 "${BASE}.reverted.log"
echo "--- round 2 (world_message zeroed) ---"
grep -E "CLOSED-LOOP|macro|ABLATE" logs/world_message_ablation_zero.log | tail -5
