#!/usr/bin/env bash
# 判定实验：前向收缩对已训练模型闭环成功率的因果影响。
#
# 原始 checkpoint s30000_b18_s3000.pt 在「未收缩前向 + --world-reset-every 4」下
# 测得 15%（3/20）。本脚本用当前（已收缩）的 wmrm.py 重测同一个 checkpoint，
# 其余配置逐项相同。评测无梯度，因此本结果同时代表 straight-through 版本的部署行为
# （两者前向数值相同，仅梯度不同）。
#
#   结果 ~15% => 收缩前向无害，fixv1 的 0% 来自训练期梯度衰减 => straight-through 值得试
#   结果 ~0%  => 收缩前向本身是病因 => straight-through 也救不回来，应回退收缩
set -euo pipefail
cd /root/private_data/ORA0

export PY=/opt/conda/bin/python
export DINO=/root/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors

CKPT=checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s30000_b18_s3000.pt
NAME=$(basename "$CKPT" .pt)
BASE=logs/${NAME}_eval10

# 保住 13:11 那次「未收缩」的 15% 记录，评测脚本会按 checkpoint 名覆盖同名文件。
for ext in log json; do
  if [[ -f "${BASE}.${ext}" && ! -f "${BASE}.uncontracted.${ext}" ]]; then
    cp -p "${BASE}.${ext}" "${BASE}.uncontracted.${ext}"
    echo "baseline preserved: ${BASE}.uncontracted.${ext}"
  fi
done

echo "=== contracted-forward re-eval START $(date '+%F %T') ==="
grep -n "_MAP_RETENTION \*\|_BELIEF_RETENTION \*\|_st_scale_residual(" va_compound/wmrm.py | head
bash scripts/eval_mw_hard2_wam4va.sh "$CKPT" data/hard2_peer_h6_p2_eval_v1.pt

# 归档本次收缩版结果，并把未收缩基线复位回标准名，避免后续混淆。
for ext in log json; do
  [[ -f "${BASE}.${ext}" ]] && cp -p "${BASE}.${ext}" "${BASE}.contracted.${ext}"
done
echo "=== contracted-forward re-eval DONE $(date '+%F %T') ==="
echo "--- baseline (uncontracted, 13:11) ---"
tail -3 "${BASE}.uncontracted.log"
echo "--- this run (contracted) ---"
tail -3 "${BASE}.contracted.log"
