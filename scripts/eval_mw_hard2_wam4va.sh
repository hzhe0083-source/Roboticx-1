#!/usr/bin/env bash
# P2/H6 MetaWorld closed-loop eval for the two hard tasks (10 trials/task).
# The peer checkpoint fixes planning_stride=execution_horizon=wmrm_cycle_steps=2.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
DINO=${DINO:-/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors}
CKPT=${1:-checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s20000.pt}
FEATURES=${2:-data/hard2_peer_h6_p2_eval_v1.pt}
EXECUTION_HORIZON=2
NAME=$(basename "$CKPT" .pt)
LOG=logs/${NAME}_eval10.log
JSON=logs/${NAME}_eval10.json

[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$FEATURES" ]] || { echo "missing features: $FEATURES" >&2; exit 1; }
[[ -f "$DINO" ]] || { echo "missing DINO: $DINO" >&2; exit 1; }

"$PY" - "$CKPT" "$FEATURES" "$EXECUTION_HORIZON" <<'PY'
import sys
import torch

ckpt_path, features_path = sys.argv[1], sys.argv[2]
execution_horizon = int(sys.argv[3])
payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
cfg = payload.get("config") or {}
expected_config = {
    "va_world_mode": "peer_sync_h6",
    "wmrm": True,
    "action_horizon": 6,
    "planning_stride": 2,
    "wmrm_cycle_steps": 2,
}
bad_config = {
    key: (cfg.get(key), value)
    for key, value in expected_config.items()
    if cfg.get(key) != value
}
if bad_config:
    raise SystemExit(f"checkpoint is not peer 80Hz/P2/H6: {bad_config}")
if execution_horizon != cfg["planning_stride"]:
    raise SystemExit(
        "execution_horizon must equal checkpoint planning_stride: "
        f"{execution_horizon} != {cfg['planning_stride']}"
    )
contract = payload.get("training_contract") or {}
required_contract = {
    "peer_training_mode": "joint_dual_stream",
    "peer_world_topology": "one_stage_delayed_world_minus_one_last_va_consume_v1",
    "peer_gradient_boundary": "fully_differentiable_bidirectional_messages_v1",
    "peer_data_isolation": "separate_va_world_episode_datasets_per_step_v1",
    "peer_dual_stream_optimizer": "va_backward_then_world_backward_one_optimizer_step_v1",
}
bad_contract = {
    key: (contract.get(key), value)
    for key, value in required_contract.items()
    if contract.get(key) != value
}
for key in ("peer_va_data_identity", "peer_world_data_identity"):
    identity = contract.get(key)
    if not isinstance(identity, dict) or not identity.get("full_file_sha256"):
        bad_contract[key] = (identity, "strong identity")
arguments = (payload.get("exact_run_contract") or {}).get("arguments") or {}
expected_arguments = {
    "control_stride": 2,
    "planning_stride": 2,
    "wmrm_cycle_steps": 2,
    "flow_prefix_steps": 2,
}
for key, value in expected_arguments.items():
    if arguments.get(key) != value:
        bad_contract[f"exact_run_contract.arguments.{key}"] = (
            arguments.get(key), value
        )
if bad_contract:
    raise SystemExit(f"checkpoint is not joint dual-stream P2: {bad_contract}")
features = torch.load(features_path, map_location="cpu", weights_only=True)
metadata = features.get("metadata") or {}
actions = features.get("actions")
if not isinstance(actions, torch.Tensor) or tuple(actions.shape[1:]) != (4, 6, 4):
    raise SystemExit("features are not T4/H6/A4")
expected_metadata = {
    "contract": "peer_sync_h6_p2_world_windows_v1",
    "contract_version": 1,
    "fps": 80,
    "control_stride": 2,
    "planning_stride": 2,
    "sequence_length": 4,
    "decision_offsets": [0, 2, 4, 6],
    "action_horizon": 6,
    "action_label_offsets": [0, 1, 2, 3, 4, 5],
}
bad_metadata = {
    key: (metadata.get(key), value)
    for key, value in expected_metadata.items()
    if metadata.get(key) != value
}
if bad_metadata:
    raise SystemExit(f"features are not 80Hz/P2/H6: {bad_metadata}")
print("peer deployment preflight: PASS (80Hz/P2/H6, planning=execution=world cycle)")
PY

mkdir -p logs
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  MUJOCO_GL=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u -B eval_metaworld.py \
  --checkpoint "$CKPT" \
  --features "$FEATURES" \
  --main-vision-checkpoint "$DINO" \
  --task-ids 0,16 \
  --trials-per-task 10 \
  --execution-horizon "$EXECUTION_HORIZON" \
  --horizon 500 \
  --direct-head auto \
  --flow-samples 1 \
  --device cuda \
  --output-json "$JSON" \
  2>&1 | tee "$LOG"
