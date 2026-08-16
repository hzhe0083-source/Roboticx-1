#!/usr/bin/env bash
# Stage 1: JEPA-style world only. Handshake off. Predictor does not read VA's A.
#   bash scripts/run_mw_hard2_wam4va_world.sh          # 5k, batch 6
#   bash scripts/run_mw_hard2_wam4va_world.sh 1000 6   # smoke
# Then: bash scripts/run_mw_hard2_wam4va_joint.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/ryan/.venvs/pytorch-gpu/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
SRC=data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock.pt
SAVE=checkpoints/mw_hard2_wam4va_world.pt
LOG=logs/mw_hard2_wam4va_world.log
STEPS=${1:-5000}
BATCH=${2:-6}

[[ -f "$DINO" ]] || { echo "missing DINO weights: $DINO" >&2; exit 1; }
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
if [[ ! -f "$DATA" ]]; then
  echo "building two-task subset → $DATA"
  "$PY" -B scripts/build_task_subset_windows.py --input "$SRC" --tasks 0,16 --output "$DATA"
fi
[[ ! -e "$SAVE" ]] || { echo "refusing to overwrite $SAVE" >&2; exit 1; }

if [[ -f "$DATA" ]]; then
  "$PY" - "$DATA" <<'PY'
import sys
import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu", weights_only=True)
md = payload.get("metadata") or {}
errors = []
if md.get("control_stride") != 6:
    errors.append(f"metadata.control_stride={md.get('control_stride')!r} != 6")
if md.get("action_horizon") != 48:
    errors.append(f"metadata.action_horizon={md.get('action_horizon')!r} != 48")
actions = payload["actions"]
if int(actions.shape[-2]) != 48:
    errors.append(f"actions chunk dim={actions.shape[-2]} != 48 (shape={tuple(actions.shape)})")
if int(actions.shape[1]) != 4:
    errors.append(f"actions time dim={int(actions.shape[1])} != 4")
got = {int(x) for x in (md.get("subset_task_ids") or [])}
if got != {0, 16}:
    errors.append(f"metadata.subset_task_ids={md.get('subset_task_ids')!r} != {{0, 16}}")
if errors:
    raise SystemExit("WAM4VA data contract failed:\n  " + "\n  ".join(errors))
print("data contract ok")
PY
fi

mkdir -p checkpoints logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B train.py \
  --data "$DATA" \
  --dino-main-vision --dino-dense-metric \
  --main-vision-checkpoint "$DINO" \
  --main-vision-grid 16 --main-vision-frames 4 \
  --main-vision-temporal --main-vision-temporal-scale 1.0 \
  --main-vision-encode-batch 8 \
  --metric-geometry-inject \
  --wam4va --wmrm-only --wmrm-inject last --wmrm-target dino \
  --wmrm-cycle-steps 6 \
  --single-task --task-sampling weighted --task-locality-block-batches 16 \
  --batch-size "$BATCH" --sequence-length 4 --min-sequence-length 4 \
  --num-workers 0 \
  --lr 0.0001 --seed 0 --device cuda \
  --va-layers 8 --va-attention-backend auto \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 \
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
  --steps "$STEPS" --save-every 1000 \
  --save "$SAVE" \
  2>&1 | tee "$LOG"
