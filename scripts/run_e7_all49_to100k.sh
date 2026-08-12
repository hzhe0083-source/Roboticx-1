#!/bin/bash
# E7 all49 repaired coarse-head path: effective 66k -> 71k -> 100k.
# Stage C is started only after a real 16-step auxiliary smoke succeeds.
set -euo pipefail

cd /home/ryan/Documents/robot/ORA0

PYTHON=/home/ryan/.venvs/pytorch-gpu/bin/python
DATA=data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt
METRIC=checkpoints/metric_field_v6_all49_contractfix_init10k_ft5k.pt
BASE=checkpoints/e7_mtvj_joint66k_step14k.pt
STAGE_B=checkpoints/e7_mtvj_all49_stageB_71k.pt
SMOKE=checkpoints/e7_mtvj_all49_stageC_smoke16.pt
FINAL=checkpoints/e7_mtvj_all49_stageC_100k.pt
SIDECAR=checkpoints/e7_mtvj_all49_stageC_100k.meta.json
LOG_B=logs/e7_mtvj_all49_stageB_71k.log
LOG_SMOKE=logs/e7_mtvj_all49_stageC_smoke16.log
LOG_C=logs/e7_mtvj_all49_stageC_100k.log
LOCK=logs/e7_mtvj_all49_to100k.lock
EFFECTIVE_STEP_OFFSET=66000
STAGE_B_TARGET=5000
STAGE_C_TARGET=34000

mkdir -p logs checkpoints
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another e7 all49-to-100k pipeline owns $LOCK; refusing a duplicate run."
  exit 1
fi

checkpoint_step() {
  "$PYTHON" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
step = checkpoint.get("global_step")
if not isinstance(step, int) or step < 0:
    raise ValueError(f"{sys.argv[1]} has invalid global_step={step!r}")
print(step)
PY
}

require_idle_gpu() {
  local compute_processes
  if ! compute_processes=$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null); then
    echo "Cannot query GPU compute processes; refusing to launch training."
    exit 1
  fi
  if [ -n "${compute_processes//[[:space:]]/}" ]; then
    echo "GPU already has compute work; refusing a concurrent trainer:"
    echo "$compute_processes"
    exit 1
  fi
}

append_log_marker() {
  local log_path=$1
  local message=$2
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$message" >> "$log_path"
}

log_line_count() {
  local log_path=$1
  if [ -f "$log_path" ]; then
    wc -l < "$log_path"
  else
    echo 0
  fi
}

"$PYTHON" - "$METRIC" "$DATA" "$BASE" <<'PY'
from pathlib import Path
import sys

import torch

from scripts.build_longtraj_features import ENV_TO_TASK

metric_path, data_path, base_path = map(Path, sys.argv[1:])
metric = torch.load(metric_path, map_location="cpu", weights_only=True)
cfg = metric["config"]
assert metric.get("contract") == "mt_vj_metric_field_v1"
assert cfg.get("steps_done") == cfg.get("steps") == 5000
assert cfg.get("training_state_version", 0) >= 2
assert cfg.get("relation_encoder_trained") is True
assert cfg.get("loc_only") is False
assert cfg.get("language_cache_available") is True
assert list(cfg.get("tasks", ())) == list(ENV_TO_TASK)
# V6 provenance：以旧 V5 10k 为 init 源（SHA 绑定），全新 Adam/RNG/step；
# sample/visibility/loss 三份新契约必须就位（交接 2026-08-12 19:59）。
init_source = cfg.get("initialization_source")
assert init_source is not None and init_source.get("sha256") == \
    "556b334198851fd036cdec31be35da3eb11361e0c7a430b108e9ba1520577a2c"
assert cfg.get("sample_rng_contract") == "parent_seed_per_sample_v1"
assert cfg.get("metric_visibility_contract") == "entity_aware_object_interface_v1"
assert cfg.get("metric_loss_contract") == "hinge_pos_offset_geom_alias_vis_v1"

data = torch.load(data_path, map_location="cpu", weights_only=True)
descriptions = list(data["metadata"]["tasks"])
assert len(descriptions) == len(set(descriptions)) == 49
assert descriptions == list(ENV_TO_TASK.values())
assert data["actions"].shape[-2:] == (48, 4)
assert data["action_valid_mask"].shape == data["actions"].shape[:-1]
assert bool(data["action_valid_mask"].any())

base = torch.load(base_path, map_location="cpu", weights_only=True)
assert base.get("global_step") in (None, 0)
assert base["config"]["num_layers"] == 8
assert base["config"]["action_horizon"] == 48
assert base["config"]["flow_cond"] == "adaln"
assert base["config"]["flow_layers"] == 6
relation = base["mtvj_relation_encoder"]
assert relation["g_proj.weight"].shape == (512, 8)
assert relation["nu_proj.weight"].shape == (512, 8)
base_contract = base["training_contract"]
assert base_contract["metric_state_source"] == "p_flat"
assert base_contract["metric_contract_version"] == 2
assert base_contract["metric_relation_joint_trained"] is True
print("preflight contracts: PASS", flush=True)
PY

COMMON=(
  --data "$DATA"
  --dense-readout-mtvj
  --metric-visual-checkpoint "$METRIC"
  --mtvj-train-relation --lr-mtvj-relation 2e-5
  --single-task --va-layers 8
  --lr 5e-6 --batch-size 16 --seed 0
  --flow-cond adaln --flow-layers 6 --flow-steps 8
  --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036
  --task-sampling weighted --task-locality-block-batches 16
  --prev-dropout 0.1
)

if [ ! -f "$STAGE_B" ]; then
  require_idle_gpu
  echo "stage B: effective 66k -> 71k (ordinary resume + one-shot migration)"
  append_log_marker "$LOG_B" \
    "START ordinary migration: base=$BASE target_global=$STAGE_B_TARGET"
  "$PYTHON" -u train.py \
    "${COMMON[@]}" \
    --replace-mtvj-metric-head-from-external \
    --steps "$STAGE_B_TARGET" \
    --resume "$BASE" \
    --save "$STAGE_B" --save-every 1000 \
    >> "$LOG_B" 2>&1
else
  stage_b_step=$(checkpoint_step "$STAGE_B")
  if [ "$stage_b_step" -eq "$STAGE_B_TARGET" ]; then
    echo "stage B already complete at global_step=$stage_b_step; skipping."
  elif [ "$stage_b_step" -lt "$STAGE_B_TARGET" ]; then
    stage_b_remaining=$((STAGE_B_TARGET - stage_b_step))
    require_idle_gpu
    echo "stage B: exact-resume $stage_b_remaining updates ($stage_b_step -> $STAGE_B_TARGET)"
    append_log_marker "$LOG_B" \
      "RESUME exact: from_global=$stage_b_step target_global=$STAGE_B_TARGET"
    "$PYTHON" -u train.py \
      "${COMMON[@]}" \
      --steps "$stage_b_remaining" \
      --resume-exact "$STAGE_B" \
      --save "$STAGE_B" --save-every 1000 \
      >> "$LOG_B" 2>&1
  else
    echo "Stage B checkpoint global_step=$stage_b_step exceeds target $STAGE_B_TARGET."
    exit 1
  fi
fi

"$PYTHON" - "$STAGE_B" "$METRIC" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

import torch


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite(state: dict, label: str) -> None:
    bad = [
        name
        for name, value in state.items()
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
    ]
    assert not bad, f"{label} contains non-finite tensors: {bad[:8]}"


checkpoint_path, metric_path = map(Path, sys.argv[1:])
c = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
assert c.get("global_step") == 5000
assert c.get("exact_resume_version") == 2
assert all(key in c for key in ("optimizer_state", "sampler_state", "rng_state", "exact_run_contract"))
contract = c["training_contract"]
assert contract["metric_state_source"] == "p_times_visibility_flat"
assert contract["metric_contract_version"] == 3
assert contract["metric_head_checkpointed"] is True
assert contract["metric_head_joint_trained"] is False
assert contract["metric_relation_joint_trained"] is True
assert contract["task_sampling"] == "weighted"
assert contract["task_locality_block_batches"] == 16
assert contract["flow_prefix_steps"] == 6
assert contract["flow_prefix_weight"] == 1.0
assert contract["flow_tail_weight"] == 0.036
assert "mtvj_metric_head" in c and "mtvj_relation_encoder" in c
metric_identity = c["mtvj_metric_checkpoint_identity"]
assert metric_identity["sha256"] == file_sha256(metric_path)
assert metric_identity["size_bytes"] == metric_path.stat().st_size
assert_finite(c["mtvj_metric_head"], "metric head")
assert_finite(c["mtvj_relation_encoder"], "relation encoder")
print("stage B checkpoint: PASS", flush=True)
PY

STAGE_C_ARGS=(
  --mtvj-train-metric-head --lr-mtvj-metric-head 1e-6
  --mtvj-visual-aux-every 8
  --mtvj-visual-aux-loc-lambda 1.0
  --mtvj-visual-aux-vis-lambda 0.5
  --mtvj-visual-aux-batch 8
)

if [ -f "$FINAL" ]; then
  stage_c_step=$(checkpoint_step "$FINAL")
else
  stage_c_step=-1
fi

if [ "$stage_c_step" -eq -1 ]; then
  # Real auxiliary-path smoke. It writes a separate artifact and production
  # Stage C restarts from Stage B, so smoke updates never leak into production.
  smoke_complete=false
  if [ -f "$SMOKE" ]; then
    smoke_step=$(checkpoint_step "$SMOKE")
    if [ "$smoke_step" -eq 5016 ]; then
      smoke_complete=true
      echo "stage C smoke already complete at global_step=5016; validating."
    elif [ "$smoke_step" -lt 5000 ] || [ "$smoke_step" -gt 5016 ]; then
      echo "Smoke checkpoint has invalid global_step=$smoke_step; refusing to overwrite it."
      exit 1
    fi
  fi

  if [ "$smoke_complete" = false ]; then
    require_idle_gpu
    echo "stage C smoke: 16 real updates (auxiliary steps 8 and 16)"
    smoke_log_start=$(log_line_count "$LOG_SMOKE")
    append_log_marker "$LOG_SMOKE" \
      "START ordinary smoke from Stage B: expected_global=5008,5016"
    "$PYTHON" -u train.py \
      "${COMMON[@]}" \
      "${STAGE_C_ARGS[@]}" \
      --steps 16 \
      --resume "$STAGE_B" \
      --save "$SMOKE" --save-every 8 \
      >> "$LOG_SMOKE" 2>&1

    smoke_log_first_new=$((smoke_log_start + 1))
    smoke_aux_count=$(tail -n +"$smoke_log_first_new" "$LOG_SMOKE" | rg -c 'aux_total=' || true)
    if [ "$smoke_aux_count" -ne 2 ]; then
      echo "Stage C smoke failed: expected exactly two aux_total records."
      exit 1
    fi
    if ! tail -n +"$smoke_log_first_new" "$LOG_SMOKE" | rg -q '^step=8 .*aux_total='; then
      echo "Stage C smoke failed: missing auxiliary update at local step 8/global 5008."
      exit 1
    fi
    if ! tail -n +"$smoke_log_first_new" "$LOG_SMOKE" | rg -q '^step=16 .*aux_total='; then
      echo "Stage C smoke failed: missing auxiliary update at local step 16/global 5016."
      exit 1
    fi
    if tail -n +"$smoke_log_first_new" "$LOG_SMOKE" | \
      rg -qi 'traceback|cuda out of memory|no space left|\bnan\b|\binf\b'; then
      echo "Stage C smoke failed: anomaly text found in the new log segment."
      exit 1
    fi
  fi

  "$PYTHON" - "$SMOKE" <<'PY'
from pathlib import Path
import sys

import torch

c = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=True)
assert c.get("global_step") == 5016
assert c.get("exact_resume_version") == 2
contract = c["training_contract"]
assert contract["metric_state_source"] == "p_times_visibility_flat"
assert contract["metric_contract_version"] == 3
assert contract["metric_head_joint_trained"] is True
assert contract["metric_relation_joint_trained"] is True
arguments = c["exact_run_contract"]["arguments"]
assert arguments["mtvj_visual_aux_every"] == 8
assert arguments["mtvj_visual_aux_batch"] == 8
assert arguments["mtvj_visual_aux_loc_lambda"] == 1.0
assert arguments["mtvj_visual_aux_vis_lambda"] == 0.5
assert all(key in c for key in ("optimizer_state", "sampler_state", "rng_state"))
print("stage C smoke checkpoint: PASS", flush=True)
PY

  require_idle_gpu
  echo "stage C: effective 71k -> 100k (ordinary objective transition)"
  stage_c_log_start=$(log_line_count "$LOG_C")
  append_log_marker "$LOG_C" \
    "START ordinary Stage C: from_global=5000 target_global=$STAGE_C_TARGET"
  "$PYTHON" -u train.py \
    "${COMMON[@]}" \
    "${STAGE_C_ARGS[@]}" \
    --steps 29000 \
    --resume "$STAGE_B" \
    --save "$FINAL" --save-every 1000 \
    >> "$LOG_C" 2>&1
elif [ "$stage_c_step" -eq "$STAGE_C_TARGET" ]; then
  echo "stage C already complete at global_step=$stage_c_step; skipping smoke and training."
elif [ "$stage_c_step" -gt 5000 ] && [ "$stage_c_step" -lt "$STAGE_C_TARGET" ]; then
  stage_c_remaining=$((STAGE_C_TARGET - stage_c_step))
  require_idle_gpu
  echo "stage C: exact-resume $stage_c_remaining updates ($stage_c_step -> $STAGE_C_TARGET)"
  stage_c_log_start=$(log_line_count "$LOG_C")
  append_log_marker "$LOG_C" \
    "RESUME exact Stage C: from_global=$stage_c_step target_global=$STAGE_C_TARGET"
  "$PYTHON" -u train.py \
    "${COMMON[@]}" \
    "${STAGE_C_ARGS[@]}" \
    --steps "$stage_c_remaining" \
    --resume-exact "$FINAL" \
    --save "$FINAL" --save-every 1000 \
    >> "$LOG_C" 2>&1
else
  echo "Stage C checkpoint global_step=$stage_c_step is outside the valid (5000, $STAGE_C_TARGET] range."
  exit 1
fi

if [ -n "${stage_c_log_start+x}" ]; then
  stage_c_log_first_new=$((stage_c_log_start + 1))
  if tail -n +"$stage_c_log_first_new" "$LOG_C" | \
    rg -qi 'traceback|cuda out of memory|no space left|\bnan\b|\binf\b'; then
    echo "Stage C failed: anomaly text found in the new log segment."
    exit 1
  fi
fi

"$PYTHON" - "$FINAL" "$STAGE_B" "$BASE" "$METRIC" "$SIDECAR" \
  "$EFFECTIVE_STEP_OFFSET" <<'PY'
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite(state: dict, label: str) -> None:
    bad = [
        name
        for name, value in state.items()
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
    ]
    assert not bad, f"{label} contains non-finite tensors: {bad[:8]}"


final_path, stage_b_path, base_path, metric_path, sidecar_path = map(
    Path, sys.argv[1:6]
)
effective_step_offset = int(sys.argv[6])
c = torch.load(final_path, map_location="cpu", weights_only=True)
assert c.get("global_step") == 34000
assert c.get("exact_resume_version") == 2
assert all(key in c for key in ("optimizer_state", "sampler_state", "rng_state", "exact_run_contract"))
contract = c["training_contract"]
assert contract["metric_state_source"] == "p_times_visibility_flat"
assert contract["metric_contract_version"] == 3
assert contract["metric_head_checkpointed"] is True
assert contract["metric_head_joint_trained"] is True
assert contract["metric_relation_joint_trained"] is True
assert contract["task_sampling"] == "weighted"
assert contract["task_locality_block_batches"] == 16
assert contract.get("mtvj_roi_enabled") is not True
assert "mtvj_metric_head" in c and "mtvj_relation_encoder" in c
arguments = c["exact_run_contract"]["arguments"]
assert arguments["mtvj_visual_aux_every"] == 8
assert arguments["mtvj_visual_aux_batch"] == 8
assert arguments["mtvj_visual_aux_loc_lambda"] == 1.0
assert arguments["mtvj_visual_aux_vis_lambda"] == 0.5
metric_identity = c["mtvj_metric_checkpoint_identity"]
metric_sha = file_sha256(metric_path)
assert metric_identity["sha256"] == metric_sha
assert metric_identity["size_bytes"] == metric_path.stat().st_size
assert_finite(c["model"], "VA policy")
assert_finite(c["mtvj_metric_head"], "metric head")
assert_finite(c["mtvj_relation_encoder"], "relation encoder")

metadata = {
    "contract": "e7_mtvj_all49_effective_step_v1",
    "scope": "coarse_metric_head_only_no_roi",
    "checkpoint": str(final_path.resolve()),
    "checkpoint_sha256": file_sha256(final_path),
    "checkpoint_global_step": int(c["global_step"]),
    "effective_step_offset": effective_step_offset,
    "effective_step": effective_step_offset + int(c["global_step"]),
    "stage_b_checkpoint": str(stage_b_path.resolve()),
    "stage_b_global_step": 5000,
    "stage_b_effective_step": effective_step_offset + 5000,
    "base_checkpoint": str(base_path.resolve()),
    "base_checkpoint_sha256": file_sha256(base_path),
    "metric_checkpoint": str(metric_path.resolve()),
    "metric_checkpoint_sha256": metric_sha,
    "roi_enabled": False,
    "generated_at": datetime.now().astimezone().isoformat(),
}
assert metadata["effective_step"] == 100000
temporary = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
temporary.replace(sidecar_path)
print("final effective-100k checkpoint and sidecar: PASS", flush=True)
PY

echo "DONE: $FINAL"
echo "META: $SIDECAR (effective_step_offset=$EFFECTIVE_STEP_OFFSET)"
