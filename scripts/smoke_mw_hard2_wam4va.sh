#!/usr/bin/env bash
# Two-stage WAM4VA smoke on assembly-v3 only.
# door-unlock-v3_fixed frames do not match the H48 window index contract.
#
#   bash scripts/smoke_mw_hard2_wam4va.sh           # 30+30 steps
#   bash scripts/smoke_mw_hard2_wam4va.sh 10        # shorter
#
# Stage 1: --wmrm-only, handshake off, 16x16 avg-pool DINO map
# Stage 2: --resume, handshake on, VA+FM+WAM
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
SRC=data/metaworld_longtraj_windows_h48_asm_doorunlock.pt
DATA=data/metaworld_longtraj_windows_h48_asm_only_smoke.pt
WORLD=checkpoints/smoke_wam4va_world.pt
JOINT=checkpoints/smoke_wam4va_joint.pt
LOG=logs/smoke_wam4va.log
STEPS=${1:-30}
WORLD_BATCH=${2:-12}
JOINT_BATCH=${3:-6}

[[ -f "$DINO" ]] || { echo "missing DINO weights: $DINO" >&2; exit 1; }
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
[[ -f data/metaworld_longtraj_assembly-v3.pt ]] || {
  echo "missing data/metaworld_longtraj_assembly-v3.pt" >&2
  exit 1
}

if [[ ! -f "$DATA" ]]; then
  echo "building assembly-only smoke subset → $DATA"
  "$PY" -B scripts/build_task_subset_windows.py --input "$SRC" --tasks 0 --output "$DATA"
fi

mkdir -p checkpoints logs
rm -f "$WORLD" "$JOINT"
: > "$LOG"

COMMON=(
  --data "$DATA"
  --dino-main-vision --dino-dense-metric
  --main-vision-checkpoint "$DINO"
  --main-vision-grid 16 --main-vision-frames 4
  --main-vision-temporal --main-vision-temporal-scale 1.0
  --main-vision-encode-batch 8
  --metric-geometry-inject
  --wam4va --wmrm-inject last --wmrm-target dino
  --wmrm-cycle-steps 6 --wmrm-map-size 16 --wmrm-map-channels 32
  --single-task --task-sampling weighted --task-locality-block-batches 16
  --sequence-length 4 --min-sequence-length 4 --num-workers 0
  --lr 0.0001 --seed 0 --device cuda
  --va-layers 8 --va-attention-backend auto
  --flow-cond adaln --flow-layers 6 --flow-steps 8
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036
  --steps "$STEPS" --save-every "$STEPS"
)

echo "===== stage 1 wmrm-only batch=$WORLD_BATCH steps=$STEPS =====" | tee -a "$LOG"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  "${COMMON[@]}" \
  --wmrm-only --batch-size "$WORLD_BATCH" \
  --save "$WORLD" \
  2>&1 | tee -a "$LOG"

echo "===== stage 2 joint batch=$JOINT_BATCH steps=$STEPS =====" | tee -a "$LOG"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  "${COMMON[@]}" \
  --batch-size "$JOINT_BATCH" \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
  --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --resume "$WORLD" \
  --save "$JOINT" \
  2>&1 | tee -a "$LOG"

echo "===== smoke weight check =====" | tee -a "$LOG"
"$PY" - "$WORLD" "$JOINT" <<'PY' | tee -a "$LOG"
import sys
import torch

world, joint = sys.argv[1], sys.argv[2]
w = torch.load(world, map_location="cpu", weights_only=True)
j = torch.load(joint, map_location="cpu", weights_only=True)
ws, js = w["model"], j["model"]
print("world handshake", w["config"].get("wmrm_handshake"), "map", w["config"].get("wmrm_map_size"))
print("joint handshake", j["config"].get("wmrm_handshake"), "map", j["config"].get("wmrm_map_size"))
print("world gate", float(ws["wmrm.gate_proj.weight"].abs().max()))
print("joint gate", float(js["wmrm.gate_proj.weight"].abs().max()))
same_layers = all(torch.allclose(ws[k], js[k]) for k in ws if k.startswith("layers."))
same_flow = all(torch.allclose(ws[k], js[k]) for k in ws if k.startswith("flow_head."))
same_wmrm = all(torch.allclose(ws[k], js[k]) for k in ws if k.startswith("wmrm."))
print("layers identical after joint", same_layers)
print("flow identical after joint", same_flow)
print("wmrm identical after joint", same_wmrm)
if w["config"].get("wmrm_handshake") is not False:
    raise SystemExit("stage-1 ckpt handshake should be False")
if j["config"].get("wmrm_handshake") is not True:
    raise SystemExit("stage-2 ckpt handshake should be True")
if j["config"].get("wmrm_map_size") != 16:
    raise SystemExit("expected wmrm_map_size=16")
if same_layers:
    raise SystemExit("joint should have updated VA layers")
if same_flow:
    raise SystemExit("joint should have updated flow_head")
if same_wmrm:
    raise SystemExit("joint should have updated wmrm")
print("smoke checks passed")
PY
