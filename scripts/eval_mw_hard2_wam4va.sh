#!/usr/bin/env bash
# H15/P15 MetaWorld closed-loop eval for the two hard tasks (10 trials/task).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/pytorch-gpu/bin/python}
DINO=${DINO:-/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors}
[[ $# -ge 1 ]] || { echo "usage: bash $0 CHECKPOINT [FEATURES]" >&2; exit 2; }
CKPT=$1
ALL49_DATA_DIR=${ALL49_DATA_DIR:-/root/ora0_all49_data}
FEATURES=${2:-$ALL49_DATA_DIR/all49_peer_h15_p15_eval_v1.pt}
EXECUTION_HORIZON=${EXECUTION_HORIZON:-15}
PEER_WORLD_OFF=${PEER_WORLD_OFF:-0}
WORLD_ARGS=()
WORLD_SUFFIX=
if [[ "$PEER_WORLD_OFF" == 1 ]]; then
  WORLD_ARGS=(--peer-world-off)
  WORLD_SUFFIX=_worldoff
elif [[ "$PEER_WORLD_OFF" != 0 ]]; then
  echo "PEER_WORLD_OFF must be 0 or 1" >&2
  exit 2
fi
NAME=$(basename "$CKPT" .pt)
LOG=logs/${NAME}_p${EXECUTION_HORIZON}${WORLD_SUFFIX}_eval10.log
JSON=logs/${NAME}_p${EXECUTION_HORIZON}${WORLD_SUFFIX}_eval10.json

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
action_horizon = int(cfg.get("action_horizon", 0))
world_horizon = int(cfg.get("wmrm_cycle_steps", 0))
deployment_horizon = int(
    cfg.get("deployment_execution_horizon", 0) or cfg.get("planning_stride", 0)
)
data_contract = "peer_sync_h15_p15_world_windows_v1"
if action_horizon != 15 or world_horizon != 15:
    raise SystemExit("evaluation requires H15 action and World horizons")
expected_config = {
    "va_world_mode": "peer_sync_h6",
    "wmrm": True,
    "action_horizon": action_horizon,
    "planning_stride": 15,
    "wmrm_cycle_steps": world_horizon,
}
bad_config = {
    key: (cfg.get(key), value)
    for key, value in expected_config.items()
    if cfg.get(key) != value
}
if deployment_horizon != execution_horizon:
    bad_config["deployment_execution_horizon"] = (
        deployment_horizon,
        execution_horizon,
    )
if bad_config:
    raise SystemExit(f"checkpoint is not true H15/P15: {bad_config}")
if execution_horizon != deployment_horizon:
    raise SystemExit(
        "execution_horizon must equal checkpoint deployment horizon: "
        f"{execution_horizon} != {deployment_horizon}"
    )
contract = payload.get("training_contract") or {}
required_contract = {
    "peer_training_mode": "joint_dual_stream",
    "peer_world_topology": "world_minus_one_same_endpoint_fixed_current_anchor_v2",
    "peer_gradient_boundary": "world_map_stopgrad_policy_projection_trainable_v1",
    "peer_dual_stream_optimizer": "va_backward_then_world_backward_one_optimizer_step_v1",
    "peer_flow_topology": "h6_prefix_h9_tail_one_way_detached_flow_v1",
}
bad_contract = {
    key: (contract.get(key), value)
    for key, value in required_contract.items()
    if contract.get(key) != value
}
allowed_data_contracts = {
    "separate_va_world_episode_datasets_per_step_v1",
    "shared_full_va_world_payload_independent_batches_per_step_v1",
}
if contract.get("peer_data_isolation") not in allowed_data_contracts:
    bad_contract["peer_data_isolation"] = (
        contract.get("peer_data_isolation"),
        sorted(allowed_data_contracts),
    )
for key in ("peer_va_data_identity", "peer_world_data_identity"):
    identity = contract.get(key)
    if not isinstance(identity, dict) or not identity.get("full_file_sha256"):
        bad_contract[key] = (identity, "strong identity")
arguments = (payload.get("exact_run_contract") or {}).get("arguments") or {}
expected_arguments = {
    "control_stride": 15,
    "planning_stride": 15,
    "wmrm_cycle_steps": world_horizon,
    "deployment_execution_horizon": execution_horizon,
    "flow_prefix_steps": 15,
}
for key, value in expected_arguments.items():
    if arguments.get(key) != value:
        bad_contract[f"exact_run_contract.arguments.{key}"] = (
            arguments.get(key), value
        )
if bad_contract:
    raise SystemExit(f"checkpoint is not joint dual-stream P15: {bad_contract}")
features = torch.load(features_path, map_location="cpu", weights_only=True)
metadata = features.get("metadata") or {}
actions = features.get("actions")
if not isinstance(actions, torch.Tensor) or tuple(actions.shape[1:]) != (4, action_horizon, 4):
    raise SystemExit(f"features are not T4/H{action_horizon}/A4")
expected_metadata = {
    "contract": data_contract,
    "contract_version": 1,
    "fps": 80,
    "control_stride": 15,
    "planning_stride": 15,
    "sequence_length": 4,
    "decision_offsets": [0, 15, 30, 45],
    "action_horizon": action_horizon,
    "action_label_offsets": list(range(action_horizon)),
}
expected_metadata.update(
    world_target_horizon=15,
    world_target_offsets=[15, 30, 45, 60],
)
bad_metadata = {
    key: (metadata.get(key), value)
    for key, value in expected_metadata.items()
    if metadata.get(key) != value
}
if bad_metadata:
    raise SystemExit(f"features are not true H15/P15: {bad_metadata}")
previous_action = features.get("previous_action")
if (
    not isinstance(previous_action, torch.Tensor)
    or tuple(previous_action.shape) != tuple(actions.shape[:2]) + (4,)
    or not torch.equal(previous_action[:, 1:], actions[:, :-1, 14])
):
    raise SystemExit(
        "features do not feed prior P15 segment token14 as next previous_action"
    )
print("peer deployment preflight: PASS (H15/P15 matched train/eval cadence)")
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
  "${WORLD_ARGS[@]}" \
  2>&1 | tee "$LOG"
