#!/usr/bin/env bash
# Two-task long-horizon WAM4VA: 10k world-only, then 20k joint.
#   0  assembly-v3
#   16 door-unlock-v3
# World keeps only the 10k file. Joint writes every 1k step copy.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
SRC=data/metaworld_longtraj_windows_h48_asm_doorunlock.pt
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_fitted.pt
WORLD=checkpoints/mw_hard2_wam4va_world_10k.pt
JOINT=checkpoints/mw_hard2_wam4va_joint.pt
LOG=logs/mw_hard2_wam4va_10k20k.log
WORLD_STEPS=${1:-10000}
JOINT_STEPS=${2:-20000}
WORLD_BATCH=${3:-12}
JOINT_BATCH=${4:-6}

[[ -f "$DINO" ]] || { echo "missing DINO: $DINO" >&2; exit 1; }
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
[[ -f data/metaworld_longtraj_assembly-v3.pt ]] || { echo "missing assembly frames" >&2; exit 1; }
[[ -e data/metaworld_longtraj_door-unlock-v3.pt ]] || { echo "missing door-unlock frames" >&2; exit 1; }
[[ ! -e "$WORLD" ]] || { echo "refusing to overwrite $WORLD" >&2; exit 1; }
[[ ! -e "$JOINT" ]] || { echo "refusing to overwrite $JOINT" >&2; exit 1; }

if [[ ! -f "$DATA" ]]; then
  echo "filtering windows to local frame files → $DATA"
  "$PY" -B scripts/filter_windows_to_available_frames.py --input "$SRC" --output "$DATA"
fi

"$PY" - "$DATA" <<'PY'
import sys
import torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
md = p.get("metadata") or {}
ids = {int(x) for x in p["instruction_id"].tolist()}
errors = []
if md.get("control_stride") != 6:
    errors.append(f"control_stride={md.get('control_stride')!r}")
if md.get("action_horizon") != 48 or int(p["actions"].shape[-2]) != 48:
    errors.append("action_horizon/chunk is not 48")
if int(p["actions"].shape[1]) != 4:
    errors.append(f"T={int(p['actions'].shape[1])} != 4")
if ids != {0, 16}:
    errors.append(f"task ids {sorted(ids)} != [0, 16]")
if errors:
    raise SystemExit("data contract failed: " + "; ".join(errors))
from collections import Counter
print("fitted windows", len(p["instruction_id"]), dict(Counter(p["instruction_id"].tolist())))
print("dropped", md.get("n_windows_dropped_frame_fit"))
PY

mkdir -p checkpoints logs
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
)

echo "===== stage 1 world-only ${WORLD_STEPS} steps batch=${WORLD_BATCH} =====" | tee "$LOG"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  "${COMMON[@]}" \
  --wmrm-only --batch-size "$WORLD_BATCH" \
  --steps "$WORLD_STEPS" --save-every "$WORLD_STEPS" \
  --save "$WORLD" \
  2>&1 | tee -a "$LOG"

echo "===== stage 2 joint ${JOINT_STEPS} steps batch=${JOINT_BATCH} =====" | tee -a "$LOG"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  "${COMMON[@]}" \
  --batch-size "$JOINT_BATCH" \
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
  --mtvj-train-relation --lr-mtvj-relation 0.00002 \
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 \
  --resume "$WORLD" \
  --steps "$JOINT_STEPS" --save-every 1000 --save-step-copies \
  --save "$JOINT" \
  2>&1 | tee -a "$LOG"

echo "done $WORLD then $JOINT" | tee -a "$LOG"
