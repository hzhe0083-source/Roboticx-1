#!/usr/bin/env bash
# Fixed-split two-task WAM4VA visual-motion training and held-out gates.
#
# Usage:
#   bash scripts/run_mw_hard2_wam4va_visualmotion_joint_v1.sh qualify50
#   bash scripts/run_mw_hard2_wam4va_visualmotion_joint_v1.sh smoke10
#   bash scripts/run_mw_hard2_wam4va_visualmotion_joint_v1.sh pilot300
#   bash scripts/run_mw_hard2_wam4va_visualmotion_joint_v1.sh qualify
#   bash scripts/run_mw_hard2_wam4va_visualmotion_joint_v1.sh 20k
#
# qualify50: isolated scratch-to-50 constrained-loss check. qualify: one scratch
# lineage, gated at global steps 50/300/1000, followed by an immutable
# step-1001 exact-resume probe.  20k: requires that qualification
# lineage, then starts a second lineage from scratch, gates its own 50/300/1000
# milestones, and only then continues to 20000 with immutable 1k archives.
set -euo pipefail

cd "$(dirname "$0")/.."

FAMILY=mw_hard2_wam4va_visualmotion_joint_v12
PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
SOURCE=data/metaworld_longtraj_windows_h48_asm_doorunlock_fitted.pt
TRAIN_DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
EVAL_DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_eval_v1.pt
SPLIT_MANIFEST=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json
EXPECTED_SOURCE_SHA=5933ee297b4f4fbdb5b9e0d249a92bbe8ecc2c302a331459b677515c377b8093
GATE_EVALUATOR=scripts/eval_wam4va_world_action.py
BATCH=${2:-${WAM4VA_BATCH_SIZE:-3}}
EVAL_BATCH=${WAM4VA_EVAL_BATCH_SIZE:-4}
MODE=${1:-}
QUAL_RUN_ID=${FAMILY}.qualification
QUAL50_RUN_ID=${FAMILY}.qualification50
SMOKE10_RUN_ID=${FAMILY}.smoke10
PILOT300_RUN_ID=${FAMILY}.pilot300
LONG_RUN_ID=${FAMILY}.long20k
GATE_STEPS=(50 300 1000)

usage() {
  echo "usage: bash $0 {smoke10|pilot300|qualify50|qualify|20k} [batch-size]" >&2
  exit 2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ $# -le 2 ]] || usage
case "$MODE" in
  smoke10|pilot300|qualify50|qualify|20k|20000) ;;
  *)
    usage
    ;;
esac
[[ "$MODE" == "20000" ]] && MODE=20k
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || die "batch-size must be a positive integer"
[[ "$EVAL_BATCH" =~ ^[1-9][0-9]*$ ]] || \
  die "WAM4VA_EVAL_BATCH_SIZE must be a positive integer"
((EVAL_BATCH >= 2)) || die "WAM4VA_EVAL_BATCH_SIZE must be >= 2"

QUAL_PROBE_SAVE=checkpoints/${QUAL_RUN_ID}.resume_exact_probe_step1001.pt
QUAL_PROBE_LOG=logs/${QUAL_RUN_ID}.resume_exact_probe_step1001.log
QUAL_PROBE_RECEIPT=diagnostics/${QUAL_RUN_ID}.resume_exact_probe_step1001.json

# One lock covers split validation, training, held-out evaluation, and the
# exact-resume probe. The stable lock file is coordination state, not output.
command -v flock >/dev/null || die "flock is required"
exec 9>"/tmp/ora0_wam4va_visualmotion_train.lock"
if ! flock -n 9; then
  die "another WAM4VA visual-motion lifecycle currently owns the launch lock"
fi

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

for path in "$PY" "$DINO" "$SOURCE" "$TRAIN_DATA" "$EVAL_DATA" \
  "$SPLIT_MANIFEST" "$GATE_EVALUATOR" train.py \
  scripts/split_wam4va_episode_holdout.py \
  data/metaworld_longtraj_assembly-v3.pt \
  data/metaworld_longtraj_door-unlock-v3.pt; do
  require_file "$path"
done

validate_fixed_split() {
  "$PY" -B - \
    "$SOURCE" "$TRAIN_DATA" "$EVAL_DATA" "$SPLIT_MANIFEST" \
    "$EXPECTED_SOURCE_SHA" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from scripts.split_wam4va_episode_holdout import (
    canonical_manifest_sha256,
    transition_mask,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


source, train_path, eval_path, manifest_path = map(
    lambda value: Path(value).resolve(), sys.argv[1:5]
)
expected_source_sha = sys.argv[5]
actual_source_sha = sha256_file(source)
if actual_source_sha != expected_source_sha:
    raise SystemExit(
        f"fixed source SHA mismatch: {actual_source_sha} != {expected_source_sha}"
    )

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("contract") != "wam4va_episode_holdout_manifest_v1":
    raise SystemExit(f"unexpected split contract: {manifest.get('contract')!r}")
if manifest.get("manifest_sha256") != canonical_manifest_sha256(manifest):
    raise SystemExit("split manifest canonical SHA mismatch")
if Path(manifest.get("manifest_path", "")).resolve() != manifest_path:
    raise SystemExit("split manifest path is not the fixed v1 path")

source_contract = manifest.get("source") or {}
if source_contract.get("sha256") != expected_source_sha:
    raise SystemExit("split manifest is not bound to the trusted source SHA")
if Path(source_contract.get("path", "")).resolve() != source:
    raise SystemExit("split manifest source path mismatch")
if source_contract.get("n_windows") != 1474:
    raise SystemExit("trusted source must contain exactly 1474 windows")

selection = manifest.get("selection") or {}
expected_selection = {
    "unit": "episode_id",
    "stratify_by": "instruction_id",
    "heldout_fraction": 0.10,
    "seed": 0,
}
for key, expected in expected_selection.items():
    if selection.get(key) != expected:
        raise SystemExit(f"split selection {key}={selection.get(key)!r} != {expected!r}")

rule = manifest.get("transition_rule") or {}
if (
    rule.get("contract") != "wam4va_world_transition_mask_v1"
    or rule.get("current_action_prefix_steps") != 6
    or rule.get("next_action_index") != 0
):
    raise SystemExit("split transition rule is not the fixed cycle-6 contract")
for key in ("episode_single_task", "episode_disjoint", "rows_disjoint", "rows_exhaustive"):
    if (manifest.get("validation") or {}).get(key) is not True:
        raise SystemExit(f"split validation.{key} is not true")

manifest_tasks = manifest.get("tasks") or []
if sorted(int(item["task_id"]) for item in manifest_tasks) != [0, 16]:
    raise SystemExit("split manifest must contain exactly tasks 0 and 16")
for item in manifest_tasks:
    if item["source"]["episodes"] != 30:
        raise SystemExit(f"task {item['task_id']} must have 30 source episodes")
    if len(item["train"]["episode_ids"]) != 27 or len(item["eval"]["episode_ids"]) != 3:
        raise SystemExit(f"task {item['task_id']} must use a 27/3 episode split")

splits = manifest.get("splits") or {}
expected_paths = {"train": train_path, "eval": eval_path}
episode_sets: dict[str, set[int]] = {}
for split_name, payload_path in expected_paths.items():
    contract = splits.get(split_name) or {}
    if Path(contract.get("output_path", "")).resolve() != payload_path:
        raise SystemExit(f"manifest {split_name} output path mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    actions = payload.get("actions")
    action_valid = payload.get("action_valid_mask")
    task_ids = payload.get("instruction_id")
    episode_ids = payload.get("episode_id")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 4:
        raise SystemExit(f"{split_name} actions must be [N,T,H,A]")
    if tuple(actions.shape[1:]) != (4, 48, 4):
        raise SystemExit(f"{split_name} actions shape {tuple(actions.shape)} is not T4/H48/A4")
    if (
        not isinstance(action_valid, torch.Tensor)
        or action_valid.dtype != torch.bool
        or action_valid.shape != actions.shape[:-1]
    ):
        raise SystemExit(f"{split_name} action_valid_mask contract mismatch")
    if sorted(int(value) for value in torch.unique(task_ids).tolist()) != [0, 16]:
        raise SystemExit(f"{split_name} does not contain exactly tasks 0 and 16")
    metadata = payload.get("metadata") or {}
    if metadata.get("split_name") != split_name:
        raise SystemExit(f"{split_name} metadata.split_name mismatch")
    if metadata.get("split_contract") != manifest:
        raise SystemExit(f"{split_name} embedded split contract differs from manifest")
    if metadata.get("split_manifest_id") != manifest.get("manifest_id"):
        raise SystemExit(f"{split_name} manifest id mismatch")
    if metadata.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
        raise SystemExit(f"{split_name} manifest SHA mismatch")
    if int(actions.shape[0]) != int(contract.get("windows", -1)):
        raise SystemExit(f"{split_name} window count differs from manifest")
    episodes = {int(value) for value in episode_ids.tolist()}
    if episodes != {int(value) for value in contract.get("episode_ids", [])}:
        raise SystemExit(f"{split_name} episode list differs from manifest")
    episode_sets[split_name] = episodes
    observed_transition = transition_mask(action_valid)
    declared_transition = (contract.get("mask_stats") or {}).get("transition") or {}
    if (
        int(observed_transition.sum()) != int(declared_transition.get("true", -1))
        or observed_transition.numel() != int(declared_transition.get("total", -1))
        or not bool(observed_transition.any())
    ):
        raise SystemExit(f"{split_name} transition-mask statistics mismatch")

if episode_sets["train"] & episode_sets["eval"]:
    raise SystemExit("train/eval episode leakage detected")
print(
    "fixed split: PASS "
    f"manifest={manifest['manifest_id']} "
    f"train={splits['train']['windows']} eval={splits['eval']['windows']}",
    flush=True,
)
PY
}

validate_train_cli() {
  "$PY" -B - "$SPLIT_MANIFEST" <<'PY'
import sys
from pathlib import Path

from train import (
    WORLD_ACTION_DONOR_CONTRACT,
    WORLD_ACTION_RANKING,
    WORLD_LOSS_COMPONENT_WEIGHTS,
    WORLD_NO_REGRESSION,
    WORLD_STATIC_COPY_CONSTRAINT,
    WORLD_STAGE_AUXILIARY_DECAY,
    WORLD_SUPERVISION_CONTRACT,
    parse_args,
)

args = parse_args(
    [
        "--world-split-manifest",
        sys.argv[1],
        "--visual-world-supervision",
        "--feature-autocast-bf16",
        "--wam4va",
        "--wmrm-target",
        "dino",
    ]
)
if (
    not args.visual_world_supervision
    or not args.feature_autocast_bf16
    or not args.wmrm
    or args.wmrm_target != "dino"
):
    raise SystemExit("train CLI did not activate WAM4VA/DINO World supervision")
if Path(args.world_split_manifest).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit("train CLI did not retain the fixed World split manifest")
if WORLD_SUPERVISION_CONTRACT != "visual_motion_constrained_v5":
    raise SystemExit(
        "train.py does not expose the constrained v5 World supervision graph"
    )
if WORLD_LOSS_COMPONENT_WEIGHTS != {
    "all": 0.25,
    "motion": 0.25,
    "top20": 0.50,
}:
    raise SystemExit("train.py visual World component weights changed")
if WORLD_STAGE_AUXILIARY_DECAY != 0.25:
    raise SystemExit("train.py visual World stage decay changed")
if WORLD_NO_REGRESSION != {
    "all_ratio": 1.0,
    "weight": 1.0,
    "components": ["all"],
}:
    raise SystemExit("train.py visual World no-regression contract changed")
if WORLD_STATIC_COPY_CONSTRAINT != {
    "static_ratio": 1.05,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "stage_chain_exact_hinge_v1",
    "reduction": "sum_stages_then_masked_transition_mean",
    "boundary": "copy_then_detached_min_previous_copy",
}:
    raise SystemExit("train.py visual World static-copy constraint changed")
if WORLD_ACTION_RANKING != {
    "stage": "full_8stage_counterfactual_final",
    "top10_min_relative_margin": 0.05,
    "top10_strong_relative_margin": 0.10,
    "weight": 1.0,
    "negatives": ["shuffle", "zero"],
    "schedule": "both_each_valid_transition",
    "mask": "per_negative_and_both_for_strong",
    "rng": "logged_branch_replay",
    "gradient": "wrong_actions_only_detached_real_margin_v1",
}:
    raise SystemExit("train.py visual World action-ranking contract changed")
if WORLD_ACTION_DONOR_CONTRACT != "train_split_task_cross_episode_proprio_nearest_v1":
    raise SystemExit("train.py visual World donor contract changed")
print("train visual-motion CLI: PASS", flush=True)
PY
}

refuse_output_family() {
  local run_id=$1
  local -a existing=()
  local path
  shopt -s nullglob
  for path in checkpoints/"${run_id}"* logs/"${run_id}"* diagnostics/"${run_id}"*; do
    existing+=("$path")
  done
  shopt -u nullglob
  if ((${#existing[@]})); then
    echo "refusing to overwrite existing ${run_id} output-family artifacts:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    exit 1
  fi
}

require_no_trainer() {
  "$PY" -B - <<'PY'
from pathlib import Path
import os

matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        args = [
            value.decode("utf-8", "replace")
            for value in (entry / "cmdline").read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        continue
    if not args:
        continue
    has_train_py = any(Path(value).name == "train.py" for value in args[1:])
    launcher = Path(args[0]).name.lower()
    python_launcher = (
        launcher.startswith("python")
        or launcher in {"torchrun", "accelerate", "uv"}
        or any(Path(value).name.lower().startswith("python") for value in args[1:3])
    )
    if has_train_py and python_launcher:
        matches.append((int(entry.name), " ".join(args)))

if matches:
    for pid, command in matches:
        print(f"active trainer pid={pid}: {command[:300]}")
    raise SystemExit(1)
print("/proc trainer check: idle", flush=True)
PY
}

require_idle_gpu() {
  # Trainer exclusivity is enforced by require_no_trainer plus the family lock.
  # NVML is unavailable in some launch sandboxes even when CUDA is usable.
  "$PY" -B - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA device must be available")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(
    "CUDA check: available "
    f"free={free_bytes / 2**30:.1f}GiB total={total_bytes / 2**30:.1f}GiB",
    flush=True,
)
PY
}

verify_checkpoint() {
  local checkpoint=$1
  local expected_step=$2
  "$PY" -B - "$checkpoint" "$expected_step" "$SPLIT_MANIFEST" <<'PY'
from pathlib import Path
import json
import sys

import torch

checkpoint_path = Path(sys.argv[1]).resolve(strict=True)
expected_step = int(sys.argv[2])
manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if payload.get("global_step") != expected_step:
    raise SystemExit(
        f"checkpoint global_step={payload.get('global_step')!r} != {expected_step}"
    )
for key in (
    "exact_resume_version",
    "optimizer_state",
    "sampler_state",
    "rng_state",
    "exact_run_contract",
):
    if key not in payload:
        raise SystemExit(f"checkpoint lacks exact-resume field {key}")
if payload.get("exact_resume_version") != 2:
    raise SystemExit("checkpoint exact_resume_version must be 2")
contract = payload.get("training_contract") or {}
expected_world_contract = {
    "world_supervision": "visual_motion_constrained_v5",
    "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
    "world_transition": "current_first6_and_next_first_v1",
    "world_stage_auxiliary_decay": 0.25,
    "world_logged_branch": "matched_context_full_forward_v1",
    "world_no_regression": {
        "all_ratio": 1.0,
        "weight": 1.0,
        "components": ["all"],
    },
    "world_static_copy_constraint": {
        "static_ratio": 1.05,
        "weight": 4.0,
        "region": "outside_top20",
        "penalty": "stage_chain_exact_hinge_v1",
        "reduction": "sum_stages_then_masked_transition_mean",
        "boundary": "copy_then_detached_min_previous_copy",
    },
    "world_action_ranking": {
        "stage": "full_8stage_counterfactual_final",
        "top10_min_relative_margin": 0.05,
        "top10_strong_relative_margin": 0.10,
        "weight": 1.0,
        "negatives": ["shuffle", "zero"],
        "schedule": "both_each_valid_transition",
        "mask": "per_negative_and_both_for_strong",
        "rng": "logged_branch_replay",
        "gradient": "wrong_actions_only_detached_real_margin_v1",
    },
    "world_action_donor_contract": "train_split_task_cross_episode_proprio_nearest_v1",
    "world_action_donor_sha256": "ccc80e054ffdd068d9fc6238863e89d5d7bf49f4d0ecfb3074eefee617a5ee25",
    "world_action_donor_transitions": 3297,
    "world_action_rank_transitions": 2931,
}
contract_mismatches = {
    key: (contract.get(key), expected)
    for key, expected in expected_world_contract.items()
    if contract.get(key) != expected
}
if contract_mismatches:
    raise SystemExit(
        f"checkpoint visual-motion loss contract mismatch: {contract_mismatches}"
    )
if contract.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
    raise SystemExit("checkpoint is not bound to the fixed split manifest")
config = payload.get("config") or {}
expected = {
    "num_layers": 8,
    "action_horizon": 48,
    "wmrm": True,
    "wmrm_inject": "all",
    "wmrm_target": "dino",
    "wmrm_cycle_steps": 6,
    "wmrm_handshake": True,
    "wmrm_map_size": 16,
    "wmrm_map_channels": 1024,
    "wmrm_world_grid": 16,
    "wmrm_predictor": "st_blocks",
    "wmrm_predictor_depth": 6,
    "wmrm_predictor_width": 384,
    "wmrm_predictor_heads": 12,
    "main_vision_grid": 16,
    "main_vision_frames": 4,
    "main_vision_dim": 1024,
}
bad = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if bad:
    raise SystemExit(f"checkpoint architecture contract mismatch: {bad}")
print(f"checkpoint: PASS {checkpoint_path} global_step={expected_step}", flush=True)
PY
}

verify_gate_report() {
  local checkpoint=$1
  local report=$2
  local expected_step=$3
  "$PY" -B - \
    "$EVAL_DATA" "$SPLIT_MANIFEST" \
    "$checkpoint" "$report" "$expected_step" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from scripts.eval_wam4va_world_action import evaluate_go_no_go


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


eval_path = Path(sys.argv[1]).resolve(strict=True)
manifest_path = Path(sys.argv[2]).resolve(strict=True)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_sha = manifest.get("manifest_sha256")
eval_sha = sha256_file(eval_path)
checkpoint_path = Path(sys.argv[3]).resolve(strict=True)
report_path = Path(sys.argv[4]).resolve(strict=True)
expected_step = int(sys.argv[5])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("contract") != "wam4va_world_action_heldout_v1":
    raise SystemExit(f"{report_path}: unexpected held-out report contract")
gate = report.get("gate") or {}
if gate.get("passed") is not True or gate.get("decision") != "GO":
    raise SystemExit(f"{report_path}: gate is not a machine-readable GO")
expected_thresholds = {
    "min_relative_gain_top10": 0.10,
    "max_static_copy_ratio": 1.05,
    "min_action_relative_degradation": 0.05,
    "one_action_min_relative_degradation": 0.10,
    "action_difference_ci_low_must_exceed": 0.0,
}
if gate.get("thresholds") != expected_thresholds:
    raise SystemExit(f"{report_path}: gate thresholds changed")
per_task = report.get("per_task") or {}
task_macro = report.get("task_macro") or {}
recomputed_gate = evaluate_go_no_go(
    per_task,
    {
        "relative_gain_top10": (task_macro.get("metrics") or {}).get(
            "relative_gain_top10", {}
        ),
        "n_tasks": task_macro.get("n_tasks"),
        "task_ids": task_macro.get("task_ids"),
        "bootstrap": task_macro.get("bootstrap"),
    },
    full_heldout_evaluation=True,
    checkpoint_world_supervision_valid=True,
    checkpoint_world_logged_branch_valid=True,
)
if recomputed_gate != gate:
    raise SystemExit(f"{report_path}: stored gate does not match report metrics")
gate_tasks = gate.get("per_task") or {}
if set(gate_tasks) != {"assembly-v3", "door-unlock-v3"}:
    raise SystemExit(f"{report_path}: gate task set is not assembly+door-unlock")
if not all(item.get("passed") is True for item in gate_tasks.values()):
    raise SystemExit(f"{report_path}: a task-local gate did not pass")

checkpoint_report = report.get("checkpoint") or {}
if Path(checkpoint_report.get("path", "")).resolve() != checkpoint_path:
    raise SystemExit(f"{report_path}: checkpoint path binding mismatch")
if checkpoint_report.get("global_step") != expected_step:
    raise SystemExit(f"{report_path}: checkpoint step binding mismatch")
if checkpoint_report.get("world_supervision") != "visual_motion_constrained_v5":
    raise SystemExit(f"{report_path}: old World loss graph is forbidden")
if checkpoint_report.get("world_logged_branch") != "matched_context_full_forward_v1":
    raise SystemExit(f"{report_path}: unmatched logged World branch is forbidden")
if checkpoint_report.get("world_contract_valid") is not True:
    raise SystemExit(f"{report_path}: checkpoint World contract is not valid")
if checkpoint_report.get("sha256") != sha256_file(checkpoint_path):
    raise SystemExit(f"{report_path}: checkpoint SHA binding mismatch")

eval_report = report.get("eval_dataset") or {}
if Path(eval_report.get("path", "")).resolve() != eval_path:
    raise SystemExit(f"{report_path}: eval split path binding mismatch")
if eval_report.get("sha256") != eval_sha:
    raise SystemExit(f"{report_path}: eval split SHA binding mismatch")
if eval_report.get("manifest_sha256") != manifest_sha:
    raise SystemExit(f"{report_path}: split manifest SHA binding mismatch")
if Path(eval_report.get("manifest_path", "")).resolve() != manifest_path:
    raise SystemExit(f"{report_path}: split manifest path binding mismatch")

protocol = report.get("protocol") or {}
if (
    protocol.get("cycle_steps") != 6
    or protocol.get("action_shape") != [6, 4]
    or protocol.get("transition_mask")
    != "current first 6 all valid and next first action valid"
    or protocol.get("shuffle")
    != "same_task_different_episode_proprio_nearest"
    or protocol.get("bootstrap") != "paired_within_task_equal_episode_weight"
    or protocol.get("bootstrap_resamples") != 4000
    or protocol.get("seed") != 0
    or protocol.get("world_stage_count") != 8
    or protocol.get("target_enters_predictor") is not False
    or protocol.get("full_heldout_evaluation") is not True
    or protocol.get("max_transitions_per_task") is not None
):
    raise SystemExit(f"{report_path}: held-out protocol contract mismatch")
if {int(item.get("task_id", -1)) for item in per_task.values()} != {0, 16}:
    raise SystemExit(f"{report_path}: report task IDs mismatch")
if not all(
    (item.get("target_permutation") or {}).get("passed") is True
    for item in per_task.values()
):
    raise SystemExit(f"{report_path}: target permutation did not pass")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
contract = checkpoint.get("training_contract") or {}
expected_world_contract = {
    "world_supervision": "visual_motion_constrained_v5",
    "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
    "world_transition": "current_first6_and_next_first_v1",
    "world_stage_auxiliary_decay": 0.25,
    "world_logged_branch": "matched_context_full_forward_v1",
    "world_no_regression": {
        "all_ratio": 1.0,
        "weight": 1.0,
        "components": ["all"],
    },
    "world_static_copy_constraint": {
        "static_ratio": 1.05,
        "weight": 4.0,
        "region": "outside_top20",
        "penalty": "stage_chain_exact_hinge_v1",
        "reduction": "sum_stages_then_masked_transition_mean",
        "boundary": "copy_then_detached_min_previous_copy",
    },
    "world_action_ranking": {
        "stage": "full_8stage_counterfactual_final",
        "top10_min_relative_margin": 0.05,
        "top10_strong_relative_margin": 0.10,
        "weight": 1.0,
        "negatives": ["shuffle", "zero"],
        "schedule": "both_each_valid_transition",
        "mask": "per_negative_and_both_for_strong",
        "rng": "logged_branch_replay",
        "gradient": "wrong_actions_only_detached_real_margin_v1",
    },
    "world_action_donor_contract": "train_split_task_cross_episode_proprio_nearest_v1",
    "world_action_donor_sha256": "ccc80e054ffdd068d9fc6238863e89d5d7bf49f4d0ecfb3074eefee617a5ee25",
    "world_action_donor_transitions": 3297,
    "world_action_rank_transitions": 2931,
}
if (
    checkpoint.get("global_step") != expected_step
    or any(contract.get(key) != expected for key, expected in expected_world_contract.items())
    or contract.get("split_manifest_sha256") != manifest_sha
):
    raise SystemExit(f"{checkpoint_path}: checkpoint contract changed after gate")
print(f"gate {expected_step}: PASS {report_path}", flush=True)
PY
}

milestone_checkpoint() {
  printf 'checkpoints/%s.step%s.pt\n' "$1" "$2"
}

milestone_train_log() {
  printf 'logs/%s.train_to_step%s.log\n' "$1" "$2"
}

milestone_gate_report() {
  printf 'diagnostics/%s.gate_step%s.json\n' "$1" "$2"
}

milestone_gate_log() {
  printf 'logs/%s.gate_step%s.log\n' "$1" "$2"
}

checkpoint_sha256() {
  local output
  output=$(sha256sum -- "$1") || die "cannot hash checkpoint: $1"
  printf '%s\n' "${output%% *}"
}

run_training_segment() {
  local start_step=$1
  local target_step=$2
  local source_checkpoint=$3
  local save=$4
  local log=$5
  local save_step_copies=${6:-0}
  local additional_steps
  local source_sha_before=""
  local source_sha_after=""
  local -a resume_args=()
  local -a save_copy_args=()
  local -a pipeline_status

  [[ "$start_step" =~ ^[0-9]+$ && "$target_step" =~ ^[1-9][0-9]*$ ]] || \
    die "training segment steps must be non-negative integers"
  ((target_step > start_step)) || die "training segment must advance global_step"
  [[ "$save_step_copies" == 0 || "$save_step_copies" == 1 ]] || \
    die "save_step_copies must be 0 or 1"
  additional_steps=$((target_step - start_step))
  [[ ! -e "$save" && ! -e "$log" ]] || \
    die "refusing to overwrite segment output: $save or $log"

  if ((start_step == 0)); then
    [[ "$source_checkpoint" == "scratch" ]] || \
      die "global_step 0 segment must start from scratch"
  else
    [[ "$source_checkpoint" != "scratch" ]] || \
      die "nonzero segment requires an exact-resume source"
    [[ "$source_checkpoint" != "$save" ]] || \
      die "exact-resume source and output checkpoint must be distinct"
    verify_checkpoint "$source_checkpoint" "$start_step"
    source_sha_before=$(checkpoint_sha256 "$source_checkpoint")
    resume_args=(--resume-exact "$source_checkpoint")
  fi
  if ((save_step_copies)); then
    save_copy_args=(--save-step-copies)
  fi

  echo "training segment: start=${start_step} updates=${additional_steps} target=${target_step} save=${save}"
  require_no_trainer
  require_idle_gpu
  set +e
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B train.py \
    --data "$TRAIN_DATA" \
    --world-split-manifest "$SPLIT_MANIFEST" \
    --visual-world-supervision \
    --dino-main-vision --dino-dense-metric \
    --main-vision-checkpoint "$DINO" \
    --main-vision-grid 16 --main-vision-frames 4 \
    --main-vision-temporal --main-vision-temporal-scale 1.0 \
    --main-vision-encode-batch 8 \
    --metric-geometry-inject \
    --wam4va --wmrm-inject all --wmrm-target dino \
    --wmrm-world-weight 1.0 \
    --wmrm-cycle-steps 6 \
    --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
    --wmrm-predictor st_blocks --wmrm-predictor-depth 6 \
    --wmrm-predictor-width 384 --wmrm-predictor-heads 12 \
    --single-task --task-sampling balanced --task-locality-block-batches 4 \
    --batch-size "$BATCH" --sequence-length 4 --min-sequence-length 4 \
    --num-workers 0 \
    --lr 0.0001 --seed 0 --device cuda \
    --feature-autocast-bf16 \
    --va-layers 8 --va-attention-backend auto \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 \
    --flow-prefix-steps 6 --flow-prefix-weight 1.0 --flow-tail-weight 0.036 \
    --mtvj-train-metric-head --lr-mtvj-metric-head 0.0003 \
    --mtvj-train-relation --lr-mtvj-relation 0.00002 \
    --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 \
    --steps "$additional_steps" --save-every 1000 \
    "${save_copy_args[@]}" \
    --save "$save" \
    "${resume_args[@]}" \
    2>&1 | tee "$log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    echo "trainer exited with status ${pipeline_status[0]}; preserving $log and all partial artifacts" >&2
    exit "${pipeline_status[0]}"
  fi
  if [[ "${pipeline_status[1]}" -ne 0 ]]; then
    die "tee exited with status ${pipeline_status[1]}; training log is incomplete"
  fi
  if ((start_step > 0)); then
    source_sha_after=$(checkpoint_sha256 "$source_checkpoint")
    [[ "$source_sha_before" == "$source_sha_after" ]] || \
      die "exact-resume segment modified its source checkpoint: $source_checkpoint"
  fi
  verify_checkpoint "$save" "$target_step"
}

run_heldout_gate() {
  local checkpoint=$1
  local report=$2
  local log=$3
  local expected_step=$4
  local -a pipeline_status

  verify_checkpoint "$checkpoint" "$expected_step"
  [[ ! -e "$report" && ! -e "$log" ]] || \
    die "refusing to overwrite held-out output: $report or $log"
  require_no_trainer
  require_idle_gpu
  set +e
  "$PY" -u -B "$GATE_EVALUATOR" \
    --checkpoint "$checkpoint" \
    --eval-data "$EVAL_DATA" \
    --output-json "$report" \
    --main-vision-checkpoint "$DINO" \
    --longtraj-dir data \
    --task-ids 0,16 \
    --device cuda --batch-size "$EVAL_BATCH" --encode-batch 8 \
    --bootstrap-resamples 4000 --seed 0 \
    2>&1 | tee "$log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  if [[ "${pipeline_status[1]}" -ne 0 ]]; then
    die "tee exited with status ${pipeline_status[1]}; held-out log is incomplete"
  fi
  if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    echo "held-out gate step ${expected_step} exited with status ${pipeline_status[0]}; preserving diagnostics" >&2
  fi
  return "${pipeline_status[0]}"
}

run_gated_lineage() {
  local run_id=$1
  local start_step=0
  local source_checkpoint=scratch
  local target_step checkpoint train_log gate_report gate_log gate_status

  for target_step in "${GATE_STEPS[@]}"; do
    checkpoint=$(milestone_checkpoint "$run_id" "$target_step")
    train_log=$(milestone_train_log "$run_id" "$target_step")
    gate_report=$(milestone_gate_report "$run_id" "$target_step")
    gate_log=$(milestone_gate_log "$run_id" "$target_step")
    run_training_segment \
      "$start_step" "$target_step" "$source_checkpoint" "$checkpoint" "$train_log"
    if run_heldout_gate "$checkpoint" "$gate_report" "$gate_log" "$target_step"; then
      verify_gate_report "$checkpoint" "$gate_report" "$target_step"
    else
      gate_status=$?
      echo "NO-GO at ${run_id} step ${target_step}; stopping before further training" >&2
      exit "$gate_status"
    fi
    source_checkpoint=$checkpoint
    start_step=$target_step
  done
}

run_qualification50() {
  local checkpoint train_log gate_report gate_log gate_status

  checkpoint=$(milestone_checkpoint "$QUAL50_RUN_ID" 50)
  train_log=$(milestone_train_log "$QUAL50_RUN_ID" 50)
  gate_report=$(milestone_gate_report "$QUAL50_RUN_ID" 50)
  gate_log=$(milestone_gate_log "$QUAL50_RUN_ID" 50)
  run_training_segment 0 50 scratch "$checkpoint" "$train_log"
  if run_heldout_gate "$checkpoint" "$gate_report" "$gate_log" 50; then
    verify_gate_report "$checkpoint" "$gate_report" 50
    echo "completed isolated 50-step qualification: ${QUAL50_RUN_ID}"
  else
    gate_status=$?
    echo "NO-GO at ${QUAL50_RUN_ID} step 50; preserving checkpoint and diagnostics" >&2
    return "$gate_status"
  fi
}

run_pilot300() {
  local start_step=0
  local source_checkpoint=scratch
  local target_step checkpoint train_log gate_report gate_log gate_status

  for target_step in 50 300; do
    checkpoint=$(milestone_checkpoint "$PILOT300_RUN_ID" "$target_step")
    train_log=$(milestone_train_log "$PILOT300_RUN_ID" "$target_step")
    gate_report=$(milestone_gate_report "$PILOT300_RUN_ID" "$target_step")
    gate_log=$(milestone_gate_log "$PILOT300_RUN_ID" "$target_step")
    run_training_segment \
      "$start_step" "$target_step" "$source_checkpoint" "$checkpoint" "$train_log"
    if run_heldout_gate "$checkpoint" "$gate_report" "$gate_log" "$target_step"; then
      verify_gate_report "$checkpoint" "$gate_report" "$target_step"
    else
      gate_status=$?
      echo "diagnostic gate NO-GO at ${PILOT300_RUN_ID} step ${target_step} (status ${gate_status}); continuing pilot" >&2
    fi
    source_checkpoint=$checkpoint
    start_step=$target_step
  done
}

write_probe_receipt() {
  local source=$1
  local source_sha=$2
  "$PY" -B - \
    "$source" "$QUAL_PROBE_SAVE" "$QUAL_PROBE_RECEIPT" \
    "$source_sha" "$SPLIT_MANIFEST" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


source = Path(sys.argv[1]).resolve(strict=True)
probe = Path(sys.argv[2]).resolve(strict=True)
receipt = Path(sys.argv[3]).resolve()
source_sha = sys.argv[4]
manifest = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
if receipt.exists():
    raise SystemExit(f"refusing to overwrite exact-resume receipt: {receipt}")
if sha256_file(source) != source_sha:
    raise SystemExit("exact-resume source SHA changed before receipt write")
payload = {
    "contract": "wam4va_visualmotion_resume_exact_probe_v1",
    "passed": True,
    "source_checkpoint": str(source),
    "source_global_step": 1000,
    "source_sha256_before": source_sha,
    "source_sha256_after": source_sha,
    "probe_checkpoint": str(probe),
    "probe_global_step": 1001,
    "probe_sha256": sha256_file(probe),
    "run_family": "mw_hard2_wam4va_visualmotion_joint_v12",
    "world_supervision": "visual_motion_constrained_v5",
    "world_logged_branch": "matched_context_full_forward_v1",
    "split_manifest_sha256": manifest["manifest_sha256"],
}
receipt.parent.mkdir(parents=True, exist_ok=True)
with receipt.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(f"exact-resume receipt: {receipt}", flush=True)
PY
}

verify_probe_receipt() {
  local source=$1
  "$PY" -B - \
    "$source" "$QUAL_PROBE_SAVE" "$QUAL_PROBE_RECEIPT" \
    "$SPLIT_MANIFEST" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


source_path = Path(sys.argv[1]).resolve(strict=True)
probe_path = Path(sys.argv[2]).resolve(strict=True)
receipt_path = Path(sys.argv[3]).resolve(strict=True)
manifest = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
source_sha = sha256_file(source_path)
probe_sha = sha256_file(probe_path)
if (
    receipt.get("contract") != "wam4va_visualmotion_resume_exact_probe_v1"
    or receipt.get("passed") is not True
    or receipt.get("source_global_step") != 1000
    or receipt.get("probe_global_step") != 1001
    or receipt.get("source_sha256_before") != source_sha
    or receipt.get("source_sha256_after") != source_sha
    or receipt.get("probe_sha256") != probe_sha
    or receipt.get("run_family") != "mw_hard2_wam4va_visualmotion_joint_v12"
    or receipt.get("world_supervision") != "visual_motion_constrained_v5"
    or receipt.get("world_logged_branch") != "matched_context_full_forward_v1"
    or receipt.get("split_manifest_sha256") != manifest.get("manifest_sha256")
    or Path(receipt.get("source_checkpoint", "")).resolve() != source_path
    or Path(receipt.get("probe_checkpoint", "")).resolve() != probe_path
):
    raise SystemExit(f"{receipt_path}: exact-resume receipt binding mismatch")
source = torch.load(source_path, map_location="cpu", weights_only=True)
probe = torch.load(probe_path, map_location="cpu", weights_only=True)
expected_world_contract = {
    "world_supervision": "visual_motion_constrained_v5",
    "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
    "world_transition": "current_first6_and_next_first_v1",
    "world_stage_auxiliary_decay": 0.25,
    "world_logged_branch": "matched_context_full_forward_v1",
    "world_no_regression": {
        "all_ratio": 1.0,
        "weight": 1.0,
        "components": ["all"],
    },
    "world_static_copy_constraint": {
        "static_ratio": 1.05,
        "weight": 4.0,
        "region": "outside_top20",
        "penalty": "stage_chain_exact_hinge_v1",
        "reduction": "sum_stages_then_masked_transition_mean",
        "boundary": "copy_then_detached_min_previous_copy",
    },
    "world_action_ranking": {
        "stage": "full_8stage_counterfactual_final",
        "top10_min_relative_margin": 0.05,
        "top10_strong_relative_margin": 0.10,
        "weight": 1.0,
        "negatives": ["shuffle", "zero"],
        "schedule": "both_each_valid_transition",
        "mask": "per_negative_and_both_for_strong",
        "rng": "logged_branch_replay",
        "gradient": "wrong_actions_only_detached_real_margin_v1",
    },
    "world_action_donor_contract": "train_split_task_cross_episode_proprio_nearest_v1",
    "world_action_donor_sha256": "ccc80e054ffdd068d9fc6238863e89d5d7bf49f4d0ecfb3074eefee617a5ee25",
    "world_action_donor_transitions": 3297,
    "world_action_rank_transitions": 2931,
}
for label, checkpoint, expected_step in (
    ("source", source, 1000),
    ("probe", probe, 1001),
):
    contract = checkpoint.get("training_contract") or {}
    if (
        checkpoint.get("global_step") != expected_step
        or checkpoint.get("exact_resume_version") != 2
        or any(contract.get(key) != value for key, value in expected_world_contract.items())
        or contract.get("split_manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise SystemExit(f"{label} checkpoint violates exact visual-motion contract")
if source.get("exact_run_contract") != probe.get("exact_run_contract"):
    raise SystemExit("exact-resume probe changed the semantic run contract")
print(f"exact-resume gate: PASS {receipt_path}", flush=True)
PY
}

run_qualification_probe() {
  local source
  local source_sha
  source=$(milestone_checkpoint "$QUAL_RUN_ID" 1000)
  source_sha=$(checkpoint_sha256 "$source")
  run_training_segment 1000 1001 "$source" "$QUAL_PROBE_SAVE" "$QUAL_PROBE_LOG"
  [[ "$(checkpoint_sha256 "$source")" == "$source_sha" ]] || \
    die "the exact-resume probe modified its source checkpoint"
  write_probe_receipt "$source" "$source_sha"
  verify_probe_receipt "$source"
}

verify_qualification() {
  local step checkpoint report
  for step in "${GATE_STEPS[@]}"; do
    checkpoint=$(milestone_checkpoint "$QUAL_RUN_ID" "$step")
    report=$(milestone_gate_report "$QUAL_RUN_ID" "$step")
    verify_checkpoint "$checkpoint" "$step"
    verify_gate_report "$checkpoint" "$report" "$step"
  done
  verify_checkpoint "$QUAL_PROBE_SAVE" 1001
  verify_probe_receipt "$(milestone_checkpoint "$QUAL_RUN_ID" 1000)"
  echo "qualification prerequisites: PASS"
}

verify_long_archives() {
  local step archive
  verify_checkpoint "$(milestone_checkpoint "$LONG_RUN_ID" 1000)" 1000
  for ((step = 2000; step <= 20000; step += 1000)); do
    archive="checkpoints/${LONG_RUN_ID}_s${step}.pt"
    [[ -s "$archive" ]] || die "missing or empty 1k checkpoint archive: $archive"
    [[ ! -e "${archive}.tmp" ]] || die "stale checkpoint archive tmp: ${archive}.tmp"
  done
  verify_checkpoint "checkpoints/${LONG_RUN_ID}.pt" 20000
  echo "formal 1k archive set: PASS step1000..step20000"
}

validate_fixed_split
validate_train_cli
mkdir -p checkpoints logs diagnostics

case "$MODE" in
  smoke10)
    refuse_output_family "$SMOKE10_RUN_ID"
    echo "smoke10: fresh scratch -> 10, batch=${BATCH}"
    run_training_segment \
      0 10 scratch \
      "$(milestone_checkpoint "$SMOKE10_RUN_ID" 10)" \
      "$(milestone_train_log "$SMOKE10_RUN_ID" 10)"
    ;;
  pilot300)
    refuse_output_family "$PILOT300_RUN_ID"
    echo "pilot300: fresh scratch -> 50 -> 300; diagnostic gates do not launch 20k"
    run_pilot300
    ;;
  qualify50)
    refuse_output_family "$QUAL50_RUN_ID"
    echo "qualification50: fresh scratch -> 50 (+50), batch=${BATCH}"
    run_qualification50
    ;;
  qualify)
    refuse_output_family "$QUAL_RUN_ID"
    echo "qualification: scratch -> 50 (+50) -> 300 (+250) -> 1000 (+700)"
    run_gated_lineage "$QUAL_RUN_ID"
    run_qualification_probe
    verify_qualification
    echo "completed qualification; 20k remains separately scratch-only"
    ;;
  20k)
    verify_qualification
    refuse_output_family "$LONG_RUN_ID"
    echo "formal: fresh scratch -> gated 50/300/1000; qualification weights are forbidden"
    run_gated_lineage "$LONG_RUN_ID"
    echo "formal gates GO; continuing exact formal lineage 1000 -> 20000 (+19000)"
    run_training_segment \
      1000 20000 "$(milestone_checkpoint "$LONG_RUN_ID" 1000)" \
      "checkpoints/${LONG_RUN_ID}.pt" \
      "logs/${LONG_RUN_ID}.train_step1000_to_step20000.log" \
      1
    verify_long_archives
    echo "completed formal ${LONG_RUN_ID}; checkpoint=checkpoints/${LONG_RUN_ID}.pt"
    ;;
esac
