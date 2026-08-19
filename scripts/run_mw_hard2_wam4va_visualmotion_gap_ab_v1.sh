#!/usr/bin/env bash
# Sequential A/B research runs for the v7 oracle-ST visual-motion loss.
#
# Usage:
#   bash scripts/run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh preflight
#   bash scripts/run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh smoke10
#   bash scripts/run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh pilot300
#   bash scripts/run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh 20k
#
# The 20k mode deliberately treats held-out GO/NO-GO as a diagnostic. It runs
# both research variants from scratch through 50, 300, 1000, and 20000. Trainer
# failures remain fatal. The 1000-to-20000 segment atomically overwrites one
# rolling checkpoint every 1000 steps, so it does not create 19 large archives.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-/home/ryan/.venvs/openvla/bin/python}
DINO=/home/ryan/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
SOURCE=data/metaworld_longtraj_windows_h48_asm_doorunlock_fitted.pt
TRAIN_DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
EVAL_DATA=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_eval_v1.pt
SPLIT_MANIFEST=data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json
EXPECTED_SOURCE_SHA=5933ee297b4f4fbdb5b9e0d249a92bbe8ecc2c302a331459b677515c377b8093
GATE_EVALUATOR=scripts/eval_wam4va_world_action.py

A_FAMILY=mw_hard2_wam4va_visualmotion_oraclestgapfinal_v16
B_FAMILY=mw_hard2_wam4va_visualmotion_oraclestgapcycle_v16
BATCH=3
EVAL_BATCH=${WAM4VA_EVAL_BATCH_SIZE:-4}
MODE=${1:-}
VARIANTS=(final cycle)

usage() {
  echo "usage: bash $0 {preflight|smoke10|pilot300|20k}" >&2
  exit 2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "${WAM4VA_RESEARCH_VARIANT:-both}" in
  both) ;;
  final) VARIANTS=(final) ;;
  cycle) VARIANTS=(cycle) ;;
  *) die "WAM4VA_RESEARCH_VARIANT must be both, final, or cycle" ;;
esac

[[ $# -eq 1 ]] || usage
case "$MODE" in
  preflight|smoke10|pilot300|20k|20000) ;;
  *) usage ;;
esac
[[ "$MODE" == "20000" ]] && MODE=20k
[[ "$EVAL_BATCH" =~ ^[2-9][0-9]*$ ]] || \
  die "WAM4VA_EVAL_BATCH_SIZE must be an integer >= 2"

# This stable lock is shared with the qualification runner. It is held across
# preflight, training, and held-out evaluation for both A/B variants.
command -v flock >/dev/null || die "flock is required"
exec 9>"/tmp/ora0_wam4va_visualmotion_train.lock"
if ! flock -n 9; then
  die "another WAM4VA visual-motion lifecycle currently owns the launch lock"
fi

family_for_variant() {
  case "$1" in
    final) printf '%s\n' "$A_FAMILY" ;;
    cycle) printf '%s\n' "$B_FAMILY" ;;
    *) die "unknown action-rank variant: $1" ;;
  esac
}

suffix_for_mode() {
  case "$1" in
    smoke10) printf 'smoke10\n' ;;
    pilot300) printf 'pilot300\n' ;;
    20k|preflight) printf 'research20k\n' ;;
    *) die "unknown runner mode: $1" ;;
  esac
}

run_id_for() {
  printf '%s.%s\n' "$(family_for_variant "$1")" "$(suffix_for_mode "$2")"
}

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

from scripts.eval_wam4va_world_action import _validate_fixed_eval_payload
from scripts.split_wam4va_episode_holdout import canonical_manifest_sha256
from train import validate_visual_world_training_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


source_path, train_path, eval_path, manifest_path = (
    Path(value).resolve(strict=True) for value in sys.argv[1:5]
)
expected_source_sha = sys.argv[5]
if sha256_file(source_path) != expected_source_sha:
    raise SystemExit("fixed source SHA mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("manifest_sha256") != canonical_manifest_sha256(manifest):
    raise SystemExit("fixed split manifest canonical SHA mismatch")
if (manifest.get("source") or {}).get("sha256") != expected_source_sha:
    raise SystemExit("split manifest is not bound to the trusted source SHA")

train_payload = torch.load(train_path, map_location="cpu", weights_only=True)
train_identity = validate_visual_world_training_split(
    train_payload, train_path, manifest_path
)
eval_payload = torch.load(eval_path, map_location="cpu", weights_only=True)
eval_identity = _validate_fixed_eval_payload(eval_payload, [0, 16])
if train_identity.get("manifest_sha256") != eval_identity.get("manifest_sha256"):
    raise SystemExit("train/eval split manifest identity mismatch")
if eval_identity.get("manifest_sha256") != manifest.get("manifest_sha256"):
    raise SystemExit("eval payload is not bound to the fixed manifest")
print(
    "fixed split: PASS "
    f"manifest={manifest['manifest_id']} "
    f"train={train_payload['actions'].shape[0]} "
    f"eval={eval_payload['actions'].shape[0]}",
    flush=True,
)
PY
}

validate_train_cli() {
  "$PY" -B - "$SPLIT_MANIFEST" <<'PY'
from pathlib import Path
import sys

from train import (
    WORLD_ACTION_DONOR_CONTRACT,
    WORLD_LOSS_COMPONENT_WEIGHTS,
    WORLD_NO_REGRESSION,
    WORLD_STAGE_AUXILIARY_DECAY,
    WORLD_STATIC_COPY_CONSTRAINT,
    WORLD_SUPERVISION_CONTRACT,
    parse_args,
)

expected_static = {
    "static_ratio": 1.0,
    "weight": 4.0,
    "region": "outside_top20",
    "penalty": "copy_budget_hinge_plus_always_copy_anchor_v1",
    "reduction": "stage_aux_weighted_masked_mean",
    "boundary": "1.00_detached_copy_each_stage",
}
if WORLD_SUPERVISION_CONTRACT != "visual_motion_oracle_stgap_v7":
    raise SystemExit("train.py does not expose visual_motion_oracle_stgap_v7")
if WORLD_LOSS_COMPONENT_WEIGHTS != {"all": 0.25, "motion": 0.25, "top20": 0.50}:
    raise SystemExit("visual World component weights changed")
if WORLD_STAGE_AUXILIARY_DECAY != 0.25:
    raise SystemExit("visual World stage decay changed")
if WORLD_NO_REGRESSION != {
    "all_ratio": 1.0,
    "weight": 1.0,
    "components": ["all"],
}:
    raise SystemExit("visual World all-region guard changed")
if WORLD_STATIC_COPY_CONSTRAINT != expected_static:
    raise SystemExit("visual World copy-budget constraint changed")
if WORLD_ACTION_DONOR_CONTRACT != "train_split_task_cross_episode_proprio_nearest_v1":
    raise SystemExit("visual World donor contract changed")

for stage_mode in ("final", "cycle"):
    args = parse_args(
        [
            "--world-split-manifest",
            sys.argv[1],
            "--visual-world-supervision",
            "--world-action-rank-stage",
            stage_mode,
            "--feature-autocast-bf16",
            "--wam4va",
            "--wmrm-target",
            "dino",
            "--batch-size",
            "3",
        ]
    )
    if args.world_action_rank_stage != stage_mode:
        raise SystemExit(f"train CLI did not retain action-rank stage {stage_mode}")
    if args.batch_size != 3:
        raise SystemExit("research runner must use batch size 3")
    if Path(args.world_split_manifest).resolve() != Path(sys.argv[1]).resolve():
        raise SystemExit("train CLI did not retain the fixed split manifest")
    if not args.visual_world_supervision or not args.wmrm or args.wmrm_target != "dino":
        raise SystemExit("train CLI did not activate WAM4VA/DINO World supervision")
print("train visual-motion A/B CLI: PASS", flush=True)
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

preflight_output_families() {
  local mode=$1
  local variant
  # Check both before starting A, so B cannot fail 28 hours later on a stale file.
  for variant in "${VARIANTS[@]}"; do
    refuse_output_family "$(run_id_for "$variant" "$mode")"
  done
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
    launcher = Path(args[0]).name.lower()
    has_train_py = any(Path(value).name == "train.py" for value in args[1:])
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

report_disk_budget() {
  "$PY" -B - <<'PY'
from pathlib import Path
import shutil

usage = shutil.disk_usage(Path("checkpoints").resolve())
print(
    "rolling-checkpoint disk: "
    f"free={usage.free / 2**30:.1f}GiB; "
    "A/B retain step50, step300, step1000, and one rolling/final checkpoint each",
    flush=True,
)
PY
}

checkpoint_sha256() {
  local output
  output=$(sha256sum -- "$1") || die "cannot hash checkpoint: $1"
  printf '%s\n' "${output%% *}"
}

verify_checkpoint() {
  local checkpoint=$1
  local expected_step=$2
  local stage_mode=$3
  "$PY" -B - "$checkpoint" "$expected_step" "$stage_mode" "$SPLIT_MANIFEST" <<'PY'
from pathlib import Path
import json
import sys

import torch

checkpoint_path = Path(sys.argv[1]).resolve(strict=True)
expected_step = int(sys.argv[2])
stage_mode = sys.argv[3]
manifest = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if payload.get("global_step") != expected_step:
    raise SystemExit(
        f"checkpoint global_step={payload.get('global_step')!r} != {expected_step}"
    )
for key in ("optimizer_state", "sampler_state", "rng_state", "exact_run_contract"):
    if key not in payload:
        raise SystemExit(f"checkpoint lacks exact-resume field {key}")
if payload.get("exact_resume_version") != 2:
    raise SystemExit("checkpoint exact_resume_version must be 2")

ranking = {
    "stage": (
        "final_direct_matched_context"
        if stage_mode == "final"
        else "rotating_8stage_direct_matched_context"
    ),
    "top10_min_relative_margin": 0.12,
    "weight": 1.0,
    "negatives": ["shuffle"],
    "diagnostic_negatives": ["zero"],
    "schedule": (
        "final_each_valid_transition"
        if stage_mode == "final"
        else "(global_step+time_index)%num_stages"
    ),
    "context": "logged_stage_detached_pair",
    "gradient": "oracle_motion_straight_through_exact_gap_v1",
}
expected_world = {
    "world_supervision": "visual_motion_oracle_stgap_v7",
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
        "static_ratio": 1.0,
        "weight": 4.0,
        "region": "outside_top20",
        "penalty": "copy_budget_hinge_plus_always_copy_anchor_v1",
        "reduction": "stage_aux_weighted_masked_mean",
        "boundary": "1.00_detached_copy_each_stage",
    },
    "world_action_ranking": ranking,
    "world_action_donor_contract": "train_split_task_cross_episode_proprio_nearest_v1",
    "world_action_donor_sha256": "ccc80e054ffdd068d9fc6238863e89d5d7bf49f4d0ecfb3074eefee617a5ee25",
    "world_action_donor_transitions": 3297,
    "world_action_rank_transitions": 2931,
}
contract = payload.get("training_contract") or {}
bad_world = {
    key: (contract.get(key), expected)
    for key, expected in expected_world.items()
    if contract.get(key) != expected
}
if bad_world:
    raise SystemExit(f"checkpoint visual-motion contract mismatch: {bad_world}")
if contract.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
    raise SystemExit("checkpoint is not bound to the fixed split manifest")

expected_config = {
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
config = payload.get("config") or {}
bad_config = {
    key: (config.get(key), expected)
    for key, expected in expected_config.items()
    if config.get(key) != expected
}
if bad_config:
    raise SystemExit(f"checkpoint architecture mismatch: {bad_config}")

sampler = payload.get("sampler_state") or {}
if sampler.get("batch_size") != 3:
    raise SystemExit("checkpoint sampler batch size is not 3")
arguments = (payload.get("exact_run_contract") or {}).get("arguments") or {}
if arguments.get("batch_size") != 3:
    raise SystemExit("exact run contract batch size is not 3")
if arguments.get("world_action_rank_stage") != stage_mode:
    raise SystemExit("exact run contract action-rank stage mismatch")
print(
    f"checkpoint: PASS {checkpoint_path} "
    f"global_step={expected_step} action_rank_stage={stage_mode}",
    flush=True,
)
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

run_training_segment() {
  local stage_mode=$1
  local start_step=$2
  local target_step=$3
  local source_checkpoint=$4
  local save=$5
  local log=$6
  local additional_steps=$((target_step - start_step))
  local source_sha_before=""
  local source_sha_after=""
  local -a resume_args=()
  local -a pipeline_status

  ((target_step > start_step)) || die "training segment must advance global_step"
  [[ ! -e "$save" && ! -e "$log" ]] || \
    die "refusing to overwrite segment output: $save or $log"
  if ((start_step == 0)); then
    [[ "$source_checkpoint" == scratch ]] || \
      die "global_step 0 segment must start from scratch"
  else
    [[ "$source_checkpoint" != scratch && "$source_checkpoint" != "$save" ]] || \
      die "nonzero segment requires a distinct exact-resume source"
    verify_checkpoint "$source_checkpoint" "$start_step" "$stage_mode"
    source_sha_before=$(checkpoint_sha256 "$source_checkpoint")
    resume_args=(--resume-exact "$source_checkpoint")
  fi

  echo "training ${stage_mode}: start=${start_step} updates=${additional_steps} target=${target_step} save=${save}"
  require_no_trainer
  require_idle_gpu
  set +e
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -B train.py \
    --data "$TRAIN_DATA" \
    --world-split-manifest "$SPLIT_MANIFEST" \
    --visual-world-supervision \
    --world-action-rank-stage "$stage_mode" \
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
    --save "$save" \
    "${resume_args[@]}" \
    2>&1 | tee "$log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    echo "trainer exited with status ${pipeline_status[0]}; preserving ${log}" >&2
    exit "${pipeline_status[0]}"
  fi
  [[ "${pipeline_status[1]}" -eq 0 ]] || \
    die "tee exited with status ${pipeline_status[1]}; training log is incomplete"
  if ((start_step > 0)); then
    source_sha_after=$(checkpoint_sha256 "$source_checkpoint")
    [[ "$source_sha_before" == "$source_sha_after" ]] || \
      die "exact-resume segment modified source checkpoint: $source_checkpoint"
  fi
  verify_checkpoint "$save" "$target_step" "$stage_mode"
}

verify_gate_report() {
  local checkpoint=$1
  local report=$2
  local expected_step=$3
  "$PY" -B - "$checkpoint" "$report" "$expected_step" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from scripts.eval_wam4va_world_action import evaluate_go_no_go


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


checkpoint_path = Path(sys.argv[1]).resolve(strict=True)
report_path = Path(sys.argv[2]).resolve(strict=True)
expected_step = int(sys.argv[3])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("contract") != "wam4va_world_action_heldout_v1":
    raise SystemExit("unexpected held-out report contract")
checkpoint = report.get("checkpoint") or {}
if (
    Path(checkpoint.get("path", "")).resolve() != checkpoint_path
    or checkpoint.get("sha256") != sha256_file(checkpoint_path)
    or checkpoint.get("global_step") != expected_step
    or checkpoint.get("world_supervision") != "visual_motion_oracle_stgap_v7"
    or checkpoint.get("world_logged_branch") != "matched_context_full_forward_v1"
    or checkpoint.get("world_contract_valid") is not True
):
    raise SystemExit("held-out report checkpoint binding mismatch")
protocol = report.get("protocol") or {}
if (
    protocol.get("cycle_steps") != 6
    or protocol.get("action_shape") != [6, 4]
    or protocol.get("full_heldout_evaluation") is not True
    or protocol.get("max_transitions_per_task") is not None
    or protocol.get("world_stage_count") != 8
    or protocol.get("target_enters_predictor") is not False
):
    raise SystemExit("held-out report protocol mismatch")
per_task = report.get("per_task") or {}
if set(per_task) != {"assembly-v3", "door-unlock-v3"}:
    raise SystemExit("held-out report task set mismatch")
if not all(
    (item.get("target_permutation") or {}).get("passed") is True
    for item in per_task.values()
):
    raise SystemExit("target permutation check failed")
task_macro = report.get("task_macro") or {}
recomputed = evaluate_go_no_go(
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
if recomputed != report.get("gate"):
    raise SystemExit("stored gate does not match held-out metrics")
print(
    f"diagnostic gate {expected_step}: {recomputed['decision']} {report_path}",
    flush=True,
)
PY
}

run_heldout_gate() {
  local checkpoint=$1
  local report=$2
  local log=$3
  local expected_step=$4
  local stage_mode=$5
  local -a pipeline_status

  verify_checkpoint "$checkpoint" "$expected_step" "$stage_mode"
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
  [[ "${pipeline_status[1]}" -eq 0 ]] || \
    die "tee exited with status ${pipeline_status[1]}; held-out log is incomplete"
  if [[ "${pipeline_status[0]}" -ne 0 && "${pipeline_status[0]}" -ne 2 ]]; then
    die "held-out evaluator failed with status ${pipeline_status[0]}"
  fi
  verify_gate_report "$checkpoint" "$report" "$expected_step"
  if [[ "${pipeline_status[0]}" -eq 2 ]]; then
    echo "research gate NO-GO at step ${expected_step}; diagnostics preserved, continuing" >&2
  fi
}

run_diagnostic_milestones() {
  local run_id=$1
  local stage_mode=$2
  shift 2
  local start_step=0
  local source_checkpoint=scratch
  local target_step checkpoint train_log gate_report gate_log

  for target_step in "$@"; do
    checkpoint=$(milestone_checkpoint "$run_id" "$target_step")
    train_log=$(milestone_train_log "$run_id" "$target_step")
    gate_report=$(milestone_gate_report "$run_id" "$target_step")
    gate_log=$(milestone_gate_log "$run_id" "$target_step")
    run_training_segment \
      "$stage_mode" "$start_step" "$target_step" "$source_checkpoint" \
      "$checkpoint" "$train_log"
    run_heldout_gate \
      "$checkpoint" "$gate_report" "$gate_log" "$target_step" "$stage_mode"
    source_checkpoint=$checkpoint
    start_step=$target_step
  done
}

run_smoke_variant() {
  local stage_mode=$1
  local run_id
  run_id=$(run_id_for "$stage_mode" smoke10)
  run_training_segment \
    "$stage_mode" 0 10 scratch \
    "$(milestone_checkpoint "$run_id" 10)" \
    "$(milestone_train_log "$run_id" 10)"
}

run_pilot_variant() {
  local stage_mode=$1
  local run_id
  run_id=$(run_id_for "$stage_mode" pilot300)
  run_diagnostic_milestones "$run_id" "$stage_mode" 50 300
}

run_20k_variant() {
  local stage_mode=$1
  local run_id source rolling final_log final_report final_gate_log
  run_id=$(run_id_for "$stage_mode" 20k)

  run_diagnostic_milestones "$run_id" "$stage_mode" 50 300 1000
  source=$(milestone_checkpoint "$run_id" 1000)
  rolling="checkpoints/${run_id}.pt"
  final_log="logs/${run_id}.train_step1000_to_step20000.log"
  echo "research ${stage_mode}: exact resume 1000 -> 20000 with one rolling checkpoint"
  run_training_segment \
    "$stage_mode" 1000 20000 "$source" "$rolling" "$final_log"

  final_report=$(milestone_gate_report "$run_id" 20000)
  final_gate_log=$(milestone_gate_log "$run_id" 20000)
  run_heldout_gate \
    "$rolling" "$final_report" "$final_gate_log" 20000 "$stage_mode"
  echo "completed research ${stage_mode}: ${rolling}"
}

validate_fixed_split
validate_train_cli
mkdir -p checkpoints logs diagnostics
report_disk_budget
preflight_output_families "$MODE"

case "$MODE" in
  preflight)
    require_no_trainer
    require_idle_gpu
    echo "A/B 20k preflight: PASS; no output artifacts created"
    ;;
  smoke10)
    for variant in "${VARIANTS[@]}"; do
      run_smoke_variant "$variant"
    done
    ;;
  pilot300)
    for variant in "${VARIANTS[@]}"; do
      run_pilot_variant "$variant"
    done
    ;;
  20k)
    echo "research A/B: scratch -> 50 -> 300 -> 1000 -> 20000, batch=${BATCH}"
    for variant in "${VARIANTS[@]}"; do
      run_20k_variant "$variant"
    done
    ;;
esac
