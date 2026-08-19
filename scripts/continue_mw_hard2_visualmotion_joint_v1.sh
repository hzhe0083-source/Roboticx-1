#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=/home/ryan/.venvs/openvla/bin/python
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
SPLIT=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json
SOURCE=checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapcycle_v16.memfix_replay.step300.pt
FAMILY=mw_hard2_wam4va_visualmotion_joint_v1
STEP1000=checkpoints/${FAMILY}.step1000.pt
ROLLING=checkpoints/${FAMILY}.pt
LOG1000=logs/${FAMILY}.train300to1000.log
LOG20000=logs/${FAMILY}.train1000to20000.log

exec 9>/tmp/ora0_wam4va_visualmotion_train.lock
flock -n 9 || { echo "another visual-motion trainer owns the launch lock" >&2; exit 1; }

for path in "$PY" "$DINO" "$DATA" "$SPLIT" "$SOURCE"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done
for path in "$STEP1000" "$ROLLING" "$LOG1000" "$LOG20000"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite: $path" >&2; exit 1; }
done

COMMON=(
  --data "$DATA"
  --world-split-manifest "$SPLIT"
  --visual-world-supervision
  --world-action-rank-stage cycle
  --dino-main-vision --dino-dense-metric
  --main-vision-checkpoint "$DINO"
  --main-vision-grid 16 --main-vision-frames 4
  --main-vision-temporal --main-vision-temporal-scale 1.0
  --main-vision-encode-batch 8
  --metric-geometry-inject
  --wam4va --wmrm-inject all --wmrm-target dino
  --wmrm-world-weight 1.0 --wmrm-cycle-steps 6
  --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16
  --wmrm-predictor st_blocks --wmrm-predictor-depth 6
  --wmrm-predictor-width 384 --wmrm-predictor-heads 12
  --single-task --task-sampling balanced --task-locality-block-batches 4
  --batch-size 3 --sequence-length 4 --min-sequence-length 4
  --num-workers 0 --lr 0.0001 --seed 0 --device cuda
  --feature-autocast-bf16
  --va-layers 8 --va-attention-backend auto
  --flow-cond adaln --flow-layers 6 --flow-steps 8
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036
  --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003
  --mtvj-train-relation --lr-mtvj-relation 0.00002
  --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8
)

run_segment() {
  local source=$1
  local steps=$2
  local save_every=$3
  local save=$4
  local log=$5

  set +e
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B train.py "${COMMON[@]}" \
    --steps "$steps" --save-every "$save_every" \
    --save "$save" --resume-exact "$source" 2>&1 | tee "$log"
  local status=("${PIPESTATUS[@]}")
  set -e
  [[ "${status[0]}" -eq 0 ]] || exit "${status[0]}"
  [[ "${status[1]}" -eq 0 ]] || exit "${status[1]}"
}

verify_checkpoint() {
  local path=$1
  local expected_step=$2
  "$PY" -B - "$path" "$expected_step" <<'PY'
import sys
import torch

path, expected = sys.argv[1], int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=True)
contract = payload.get("exact_run_contract") or {}
arguments = contract.get("arguments") or {}
required = ("optimizer_state", "sampler_state", "rng_state", "exact_run_contract")
if payload.get("global_step") != expected:
    raise SystemExit(f"checkpoint step mismatch: {payload.get('global_step')} != {expected}")
if payload.get("exact_resume_version") != 2 or not all(k in payload for k in required):
    raise SystemExit("checkpoint lacks exact-resume state")
if arguments.get("batch_size") != 3 or arguments.get("world_action_rank_stage") != "cycle":
    raise SystemExit("checkpoint B contract mismatch")
print(f"checkpoint verified: {path} step={expected}", flush=True)
PY
}

echo "B formal continuation: step300 -> step1000"
run_segment "$SOURCE" 700 0 "$STEP1000" "$LOG1000"
verify_checkpoint "$STEP1000" 1000

echo "B formal long run: step1000 -> step20000; rolling save every 2000 steps"
run_segment "$STEP1000" 19000 2000 "$ROLLING" "$LOG20000"
verify_checkpoint "$ROLLING" 20000
