#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-preflight}
PY=${PY:-/opt/conda/bin/python}
DATA=${DATA:-/root/libero_spatial_ora0_v1/libero_spatial_h8p2_t4_v1.pt}
LONGTRAJ=${LONGTRAJ:-/root/libero_spatial_ora0_v1/longtraj}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
RUN_ID=${RUN_ID:-libero_spatial_dino_va8_flow6_entry_h8p2_e23_b32_ddp2_seed0_v2}
SAVE=${SAVE:-/root/ora0_ckpts/$RUN_ID.pt}
STEPS=5750

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

preflight(){
  [[ -x "$PY" ]] || fail "missing Python: $PY"
  [[ -f "$DATA" ]] || fail "missing data: $DATA"
  [[ -d "$LONGTRAJ" ]] || fail "missing longtraj: $LONGTRAJ"
  [[ -f "$DINO" ]] || fail "missing DINO: $DINO"
  "$PY" -B - "$DATA" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
assert payload["metadata"]["contract"] == "libero_spatial_official_h8p2_t4_v1"
assert tuple(payload["actions"].shape) == (8000, 4, 8, 7)
assert tuple(payload["proprio"].shape) == (8000, 4, 9)
print("LIBERO Spatial data contract: PASS")
PY
  printf 'preflight: flow=entry H8/P2 steps=%s global_batch=32 GPUs=2 seed=0\n' "$STEPS"
}

preflight
[[ "$MODE" == preflight ]] && exit 0
[[ "$MODE" == run ]] || fail "usage: $0 [preflight|run]"
[[ ! -e "$SAVE" ]] || fail "output already exists: $SAVE"

exec env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --data "$DATA" --longtraj-dir "$LONGTRAJ" --longtraj-decode-cache-tasks 10 \
  --single-task --task-sampling full \
  --dino-main-vision --main-vision-checkpoint "$DINO" \
  --main-vision-grid 8 --main-vision-frames 4 --main-vision-temporal \
  --main-vision-temporal-scale 1 --main-vision-encode-batch 64 \
  --slot-free-policy --va-layers 8 --va-attention-backend auto \
  --flow-cond entry --flow-layers 6 --flow-steps 8 \
  --flow-prefix-steps 2 --flow-prefix-weight 1 --flow-tail-weight 1 \
  --planning-stride 2 --control-stride 2 --deployment-execution-horizon 2 \
  --prev-dropout 0.2 --feature-autocast-bf16 --disable-runtime-integrity-checks \
  --batch-size 32 --num-workers 0 --sequence-length 4 --min-sequence-length 4 \
  --lr 0.0001 --steps "$STEPS" --seed 0 --device cuda \
  --save-every 1250 --save-step-copies --save "$SAVE"
