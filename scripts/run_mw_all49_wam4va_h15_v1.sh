#!/usr/bin/env bash
# All-49 H15/P15 dual-stream VA<->World training on scripted-expert episodes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
VERIFY_PY=${VERIFY_PY:-$PY}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
ALL49_DATA_DIR=${ALL49_DATA_DIR:-/root/ora0_all49_data}
ALL49_FRAMES_DIR=${ALL49_FRAMES_DIR:-/root/ora0_all49_raw}
HARD2_FRAMES_DIR=${HARD2_FRAMES_DIR:-data/frames_v2}
ALLTASK_H48_REF=${ALLTASK_H48_REF:-data/metaworld_longtraj_windows_h48_all49_repaired_v2_clean.pt}
RAW_IDENTITY_MANIFEST=$ALL49_DATA_DIR/all49_raw_canonical_identity_v1.json

SOURCE=$ALL49_DATA_DIR/all49_peer_h15_p2_source_v1.pt
WORLD_POOL=$ALL49_DATA_DIR/all49_peer_h15_p2_world_pool_v1.pt
VA_DATA=$ALL49_DATA_DIR/all49_peer_h15_p2_va_train_v1.pt
WORLD_DATA=$ALL49_DATA_DIR/all49_peer_h15_p2_world_train_v1.pt
EVAL_DATA=$ALL49_DATA_DIR/all49_peer_h15_p2_eval_v1.pt
PARTITION_MANIFEST=$ALL49_DATA_DIR/all49_peer_h15_p2_va_world_partition_v1.json
WORLD_MANIFEST=$ALL49_DATA_DIR/all49_peer_h15_p2_world_split_v1.json

MODE=${1:-}
STEPS=${2:-}
BATCH=${3:-48}
EPOCHS=${EPOCHS:-25}
NGPUS=${NGPUS:-2}
RUN_ID=${RUN_ID:-mw_all49_wam4va_h15_p15_e25_v1}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/ora0_ckpts}
SAVE_EVERY=${SAVE_EVERY:-670}
FORMAL_CHECKPOINT=${FORMAL_CHECKPOINT:-/root/ora0_ckpts/mw_hard2_l20_h15_p15_prefix_tail_from_s1752.to_equiv_s5037.pt}
RESUME_WEIGHTS=${RESUME_WEIGHTS:-$FORMAL_CHECKPOINT}
MAIN_VISION_ENCODE_BATCH=${MAIN_VISION_ENCODE_BATCH:-8}
LOCK=/tmp/ora0_all49_wam4va_h15_v1.lock

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage(){ printf 'usage: %s prepare | %s {preflight|joint} STEPS [global-batch] (EPOCHS=25)\n' "$0" "$0" >&2; exit 2; }

prepare_raw_view(){
  mkdir -p "$ALL49_DATA_DIR" "$ALL49_FRAMES_DIR"
  "$PY" -B - "$ALLTASK_H48_REF" "$HARD2_FRAMES_DIR" \
    "$ALL49_FRAMES_DIR" "$RAW_IDENTITY_MANIFEST" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys
import torch

ref_path, hard2_dir, frames_dir, manifest_path = map(Path, sys.argv[1:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ref = torch.load(ref_path, map_location="cpu", weights_only=True)
declared = list((ref.get("metadata") or {}).get("source_files", []))
if len(declared) != 49:
    raise SystemExit(f"all49 reference must declare 49 sources, got {len(declared)}")

entries = []
seen_tasks = set()
for declared_source in declared:
    name = Path(declared_source).name
    if not name.startswith("metaworld_longtraj_") or not name.endswith(".pt"):
        raise SystemExit(f"invalid all49 source name: {name}")
    task = name[len("metaworld_longtraj_") : -len(".pt")]
    if task.endswith("_fixed"):
        task = task[: -len("_fixed")]
    if task in seen_tasks:
        raise SystemExit(f"duplicate all49 task source: {task}")
    seen_tasks.add(task)

    if task == "assembly-v3":
        source = hard2_dir / "metaworld_longtraj_assembly-v3.pt"
    elif task == "door-unlock-v3":
        source = hard2_dir / "metaworld_longtraj_door-unlock-v3.pt"
    else:
        source = frames_dir / name
    source = source.expanduser().resolve(strict=False)
    if not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(f"missing or empty all49 raw trajectory source: {source}")

    canonical = (
        frames_dir / f"metaworld_longtraj_{task}.pt"
    ).expanduser().absolute()
    source_size = int(source.stat().st_size)
    source_sha = sha256_file(source)
    if os.path.lexists(canonical):
        if not canonical.is_file():
            raise SystemExit(f"canonical frame target is not a file: {canonical}")
        target_size = int(canonical.stat().st_size)
        target_sha = source_sha if os.path.samefile(source, canonical) else sha256_file(canonical)
        if target_size != source_size or target_sha != source_sha:
            raise SystemExit(
                "refusing to overwrite non-source canonical frame file: "
                f"{canonical} source(size={source_size},sha={source_sha}) "
                f"target(size={target_size},sha={target_sha})"
            )
    else:
        canonical.symlink_to(source)
    if (
        not canonical.is_file()
        or int(canonical.stat().st_size) != source_size
        or (source_sha if os.path.samefile(source, canonical) else sha256_file(canonical))
        != source_sha
    ):
        raise SystemExit(f"canonical frame identity verification failed: {canonical}")
    entries.append(
        {
            "task": task,
            "source_path": str(source),
            "canonical_path": str(canonical),
            "size_bytes": source_size,
            "sha256": source_sha,
        }
    )

payload = {
    "contract": "all49_canonical_raw_sources_v1",
    "reference": {
        "path": str(ref_path.expanduser().resolve(strict=True)),
        "size_bytes": int(ref_path.stat().st_size),
        "sha256": sha256_file(ref_path),
    },
    "sources": entries,
}
encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
if manifest_path.exists():
    if manifest_path.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"raw identity manifest changed; refusing overwrite: {manifest_path}")
else:
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if temporary.exists():
        raise SystemExit(f"stale raw identity manifest temporary: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, manifest_path)
print(f"all49 canonical raw view: PASS; tasks={len(entries)} manifest={manifest_path}")
PY
}

prepare_data(){
  mkdir -p "$ALL49_DATA_DIR"
  for artifact in "$SOURCE" "$WORLD_POOL" "$VA_DATA" "$WORLD_DATA" "$EVAL_DATA" \
    "$PARTITION_MANIFEST" "$WORLD_MANIFEST"; do
    [[ ! -e "$artifact" ]] || fail "refusing to overwrite immutable artifact: $artifact"
  done
  prepare_raw_view
  "$PY" -B - "$RAW_IDENTITY_MANIFEST" "$SOURCE" <<'PY'
from pathlib import Path
import json
import sys
from scripts.build_longtraj_features import PEER_SYNC_H15_P2_CONTRACT, phase1

manifest_path, output = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
inputs = [Path(item["source_path"]) for item in manifest.get("sources", [])]
if len(inputs) != 49:
    raise SystemExit(f"raw identity manifest must declare 49 sources, got {len(inputs)}")
phase1(
    15,
    input_paths=inputs,
    output_path=output,
    ref_path=Path(manifest["reference"]["path"]),
    legacy_policy="infer",
    data_contract=PEER_SYNC_H15_P2_CONTRACT,
    planning_stride=2,
)
PY
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$SOURCE" --train-output "$WORLD_POOL" --eval-output "$VA_DATA" \
    --manifest-output "$PARTITION_MANIFEST" --heldout-fraction 0.50 --seed 101
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$WORLD_POOL" --train-output "$WORLD_DATA" --eval-output "$EVAL_DATA" \
    --manifest-output "$WORLD_MANIFEST" --heldout-fraction 0.20 --seed 202
}

preflight(){
  local enforce_steps=${1:-0}
  for artifact in "$SOURCE" "$WORLD_POOL" "$VA_DATA" "$WORLD_DATA" "$EVAL_DATA" \
    "$PARTITION_MANIFEST" "$WORLD_MANIFEST" "$RAW_IDENTITY_MANIFEST" "$DINO" \
    "$RESUME_WEIGHTS" "$FORMAL_CHECKPOINT"; do
    [[ -f "$artifact" ]] || fail "missing required artifact: $artifact"
  done
  "$VERIFY_PY" -B - "$SOURCE" "$WORLD_POOL" "$VA_DATA" "$WORLD_DATA" \
    "$EVAL_DATA" "$PARTITION_MANIFEST" "$WORLD_MANIFEST" \
    "$RAW_IDENTITY_MANIFEST" "$ALL49_FRAMES_DIR" "$BATCH" "$STEPS" \
    "$EPOCHS" "$enforce_steps" "$RESUME_WEIGHTS" "$FORMAL_CHECKPOINT" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys
import torch

(
    source_path,
    world_pool_path,
    va_path,
    world_path,
    eval_path,
    partition_path,
    world_manifest_path,
    raw_manifest_path,
    frames_dir,
) = map(
    Path, sys.argv[1:10]
)
global_batch = int(sys.argv[10])
requested_steps = sys.argv[11]
epochs = int(sys.argv[12])
enforce_steps = bool(int(sys.argv[13]))
resume_path = Path(sys.argv[14])
formal_path = Path(sys.argv[15])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_sha256(manifest: dict) -> str:
    canonical = dict(manifest)
    canonical.pop("manifest_id", None)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(label, path, source, train, eval_output):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "wam4va_episode_holdout_manifest_v1" or int(
        manifest.get("contract_version", -1)
    ) != 1:
        raise SystemExit(f"{label}: unknown manifest contract")
    if canonical_manifest_sha256(manifest) != manifest.get("manifest_sha256"):
        raise SystemExit(f"{label}: canonical manifest SHA mismatch")
    source_contract = manifest.get("source") or {}
    if Path(str(source_contract.get("path", ""))).name != source.name:
        raise SystemExit(f"{label}: source path binding mismatch")
    if sha256_file(source) != source_contract.get("sha256"):
        raise SystemExit(f"{label}: source SHA mismatch")
    splits = manifest.get("splits") or {}
    for split_name, expected in (("train", train), ("eval", eval_output)):
        actual = Path(str((splits.get(split_name) or {}).get("output_path", ""))).name
        if actual != expected.name:
            raise SystemExit(f"{label}: {split_name} output binding mismatch")
    return manifest


partition = validate_manifest(
    "partition", partition_path, source_path, world_pool_path, va_path
)
world_manifest = validate_manifest(
    "world", world_manifest_path, world_pool_path, world_path, eval_path
)
if (
    (world_manifest.get("source") or {}).get("payload_output_identity")
    != ((partition.get("splits") or {}).get("train") or {}).get("output_identity")
):
    raise SystemExit("World manifest is not bound to the partition World pool")
if (world_manifest.get("source") or {}).get("payload_parent_identity") != partition.get(
    "source"
):
    raise SystemExit("World manifest parent is not bound to the partition source")

raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
if raw_manifest.get("contract") != "all49_canonical_raw_sources_v1":
    raise SystemExit("unknown all49 raw identity manifest contract")
reference = raw_manifest.get("reference") or {}
reference_path = Path(str(reference.get("path", "")))
if (
    not reference_path.is_file()
    or int(reference_path.stat().st_size) != int(reference.get("size_bytes", -1))
    or sha256_file(reference_path) != reference.get("sha256")
):
    raise SystemExit("all49 H48 reference identity mismatch")
raw_entries = list(raw_manifest.get("sources") or [])
if len(raw_entries) != 49 or len({item.get("task") for item in raw_entries}) != 49:
    raise SystemExit("raw identity manifest must contain 49 unique tasks")
declared_raw = (partition.get("source") or {}).get("payload_source_identities") or []
manifest_raw = [
    {
        "path": str(Path(item["source_path"]).resolve(strict=True)),
        "sha256": item["sha256"],
        "size_bytes": int(item["size_bytes"]),
    }
    for item in raw_entries
]
if manifest_raw != declared_raw:
    raise SystemExit("raw identity list is not bound to the H15 source payload")
if (world_manifest.get("source") or {}).get("payload_source_identities") != declared_raw:
    raise SystemExit("World manifest raw identities differ from the partition source")
expected_tasks = {str(item["task"]) for item in raw_entries}
for item in raw_entries:
    source = Path(item["source_path"])
    canonical = Path(item["canonical_path"])
    expected_size = int(item["size_bytes"])
    expected_sha = str(item["sha256"])
    if not source.is_file() or not canonical.is_file():
        raise SystemExit(f"missing raw/canonical frame source for {item['task']}")
    if source.stat().st_size != expected_size or canonical.stat().st_size != expected_size:
        raise SystemExit(f"raw frame size mismatch for {item['task']}")
    source_sha = sha256_file(source)
    canonical_sha = source_sha if os.path.samefile(source, canonical) else sha256_file(canonical)
    if source_sha != expected_sha or canonical_sha != expected_sha:
        raise SystemExit(f"raw frame SHA mismatch for {item['task']}")
    expected_canonical = frames_dir / f"metaworld_longtraj_{item['task']}.pt"
    if canonical.absolute() != expected_canonical.expanduser().absolute():
        raise SystemExit(f"canonical frame path mismatch for {item['task']}")

payloads = {
    name: torch.load(path, map_location="cpu", weights_only=True)
    for name, path in (("va", va_path), ("world", world_path), ("eval", eval_path))
}
payload_bindings = {
    "va": (partition, "eval"),
    "world": (world_manifest, "train"),
    "eval": (world_manifest, "eval"),
}
episode_sets = {}
expected_ids = list(range(49))
for name, payload in payloads.items():
    actions = payload.get("actions")
    metadata = payload.get("metadata") or {}
    if not isinstance(actions, torch.Tensor) or tuple(actions.shape[1:]) != (4, 15, 4):
        raise SystemExit(f"{name}: expected T4/H15/A4, got {getattr(actions, 'shape', None)}")
    required = {
        "contract": "peer_sync_h15_p2_world_windows_v1",
        "contract_version": 1,
        "fps": 80,
        "control_stride": 2,
        "planning_stride": 2,
        "action_horizon": 15,
        "world_target_horizon": 15,
        "world_target_offsets": [15, 17, 19, 21],
    }
    bad = {key: (metadata.get(key), value) for key, value in required.items()
           if metadata.get(key) != value}
    if bad:
        raise SystemExit(f"{name}: H15 cadence mismatch: {bad}")
    task_ids = sorted(int(value) for value in torch.unique(payload["instruction_id"]))
    if task_ids != expected_ids:
        raise SystemExit(f"{name}: expected instruction ids 0..48, got {task_ids}")
    target_valid = payload.get("world_target_valid_mask")
    target_refs = payload.get("world_target_frame_refs")
    if not isinstance(target_valid, torch.Tensor) or target_valid.shape != actions.shape[:2]:
        raise SystemExit(f"{name}: invalid world_target_valid_mask")
    if target_valid.dtype != torch.bool:
        raise SystemExit(f"{name}: world_target_valid_mask must be bool")
    if not isinstance(target_refs, (list, tuple)) or len(target_refs) != len(actions):
        raise SystemExit(f"{name}: invalid world_target_frame_refs")
    manifest, split_name = payload_bindings[name]
    split = (manifest.get("splits") or {}).get(split_name) or {}
    if metadata.get("split_manifest_sha256") != manifest.get("manifest_sha256"):
        raise SystemExit(f"{name}: split manifest SHA binding mismatch")
    if metadata.get("split_name") != split_name:
        raise SystemExit(f"{name}: split name binding mismatch")
    if metadata.get("output_identity") != split.get("output_identity"):
        raise SystemExit(f"{name}: split output identity mismatch")
    if metadata.get("parent_identity") != manifest.get("source"):
        raise SystemExit(f"{name}: split parent identity mismatch")
    if metadata.get("source_identities") != declared_raw:
        raise SystemExit(f"{name}: raw source identity binding mismatch")
    episode_sets[name] = set(zip(
        payload["instruction_id"].tolist(), payload["episode_id"].tolist(), strict=True
    ))
    frame_tasks = {str(ref[0]) for ref in payload["frame_refs"]}
    target_tasks = {str(ref[0]) for ref in target_refs}
    if frame_tasks != expected_tasks or target_tasks != expected_tasks:
        raise SystemExit(f"{name}: frame sources do not cover the canonical 49 tasks")
    for task_file in frame_tasks:
        raw = frames_dir / f"metaworld_longtraj_{task_file}.pt"
        if not raw.is_file():
            raise SystemExit(f"{name}: missing frame source {raw}")
for left, right in (("va", "world"), ("va", "eval"), ("world", "eval")):
    if episode_sets[left] & episode_sets[right]:
        raise SystemExit(f"episode leakage: {left}/{right}")
if global_batch <= 0:
    raise SystemExit("global batch must be positive")
steps_per_epoch = len(payloads["va"]["actions"]) // global_batch
if steps_per_epoch <= 0:
    raise SystemExit("global batch must produce at least one VA step per epoch")
expected_steps = epochs * steps_per_epoch
if epochs != 25:
    raise SystemExit(f"formal all49 run requires EPOCHS=25, got {epochs}")
if enforce_steps and requested_steps != str(expected_steps):
    raise SystemExit(
        f"formal all49 run requires STEPS={expected_steps} "
        f"(25 * {steps_per_epoch}), got {requested_steps!r}"
    )


def checkpoint_signature(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or {}
    contract = checkpoint.get("training_contract") or {}
    state = checkpoint.get("model") or {}
    expected_config = {
        "num_layers": 8,
        "hidden_dim": 512,
        "action_horizon": 15,
        "action_dim": 4,
        "proprio_dim": 4,
        "mode": "bidir_va",
        "va_world_mode": "peer_sync_h6",
        "planning_stride": 2,
        "deployment_execution_horizon": 15,
        "wmrm_cycle_steps": 15,
        "wmrm_predictor": "st_blocks",
        "wmrm_predictor_depth": 6,
        "wmrm_predictor_width": 384,
        "wmrm_predictor_heads": 12,
        "metric_geometry_inject": True,
        "dino_dense_metric": True,
        "main_vision_temporal": True,
    }
    bad = {
        key: (config.get(key), expected)
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if bad:
        raise SystemExit(f"resume checkpoint config mismatch: {bad}")
    expected_contract = {
        "peer_training_mode": "joint_dual_stream",
        "peer_world_topology": "world_minus_one_same_endpoint_fixed_current_anchor_v2",
        "peer_gradient_boundary": "world_map_stopgrad_policy_projection_trainable_v1",
        "peer_data_isolation": "separate_va_world_episode_datasets_per_step_v1",
        "peer_dual_stream_optimizer": "va_backward_then_world_backward_one_optimizer_step_v1",
        "planning_stride": 2,
        "planning_hz": 40.0,
        "peer_high_frequency_contract": {
            "action_prediction": "full_action_chunk_each_decision_v2",
            "world_transition": "logged_world_horizon_action_chunk_v2",
            "world_target": "explicit_endpoint_at_world_horizon_v2",
            "readout_auxiliary": "full_logged_action_chunk_v2",
        },
        "peer_flow_topology": "h6_prefix_h9_tail_one_way_detached_flow_v1",
    }
    bad = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if bad:
        raise SystemExit(f"resume checkpoint training contract mismatch: {bad}")
    required_shapes = {
        "action_queries": (15, 512),
        "wmrm.stage_embed.weight": (7, 512),
        "tail_flow_head.velocity_head.weight": (4, 512),
    }
    actual_shapes = {
        key: tuple(state[key].shape) if key in state else None for key in required_shapes
    }
    if actual_shapes != required_shapes:
        raise SystemExit(
            f"resume checkpoint H15/World7/tail-Flow shapes mismatch: {actual_shapes}"
        )
    if (
        not checkpoint.get("mtvj_metric_head")
        or not checkpoint.get("mtvj_relation_encoder")
        or not checkpoint.get("mtvj_metric_head_config")
    ):
        raise SystemExit("resume checkpoint lacks metric head/relation weights or config")
    return {"config": expected_config, "contract": expected_contract, "shapes": actual_shapes}


resume_signature = checkpoint_signature(resume_path)
formal_signature = (
    resume_signature
    if resume_path.resolve(strict=True) == formal_path.resolve(strict=True)
    else checkpoint_signature(formal_path)
)
if resume_signature != formal_signature:
    raise SystemExit("resume checkpoint is not architecture-compatible with formal H15 checkpoint")
print(
    f"all49 H15 peer preflight: PASS; VA={len(payloads['va']['actions'])}, "
    f"World={len(payloads['world']['actions'])}, eval={len(payloads['eval']['actions'])}, "
    f"steps/epoch={steps_per_epoch}, 25 epochs={expected_steps} steps, "
    f"resume={resume_path}"
)
PY
}

require_idle(){
  ! pgrep -af '[p]ython.*train.py' >/dev/null || fail 'another train.py is active'
}

run_joint(){
  [[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail 'joint requires a positive step count'
  [[ "$EPOCHS" == 25 ]] || fail 'formal all49 joint run requires EPOCHS=25'
  [[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || fail 'global batch must be positive'
  [[ "$NGPUS" =~ ^[1-9][0-9]*$ ]] || fail 'NGPUS must be positive'
  (( BATCH % NGPUS == 0 )) || fail "global batch $BATCH must divide across $NGPUS GPUs"
  [[ -n "$RESUME_WEIGHTS" && -f "$RESUME_WEIGHTS" ]] \
    || fail 'RESUME_WEIGHTS must name the compatible H15 checkpoint'
  local save=$CHECKPOINT_DIR/$RUN_ID.pt
  local log=logs/$RUN_ID.log
  [[ ! -e "$save" && ! -e "$log" ]] || fail "refusing to overwrite run $RUN_ID"
  mkdir -p "$CHECKPOINT_DIR" logs
  require_idle
  local launcher=("$PY" -u -B)
  if (( NGPUS > 1 )); then
    launcher=("$PY" -m torch.distributed.run --standalone \
      --nproc_per_node="$NGPUS" --max_restarts=0)
  fi
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${launcher[@]}" train.py --va-data "$VA_DATA" --world-data "$WORLD_DATA" \
    --visual-world-supervision --world-split-manifest "$WORLD_MANIFEST" \
    --va-world-mode peer_sync_h6 --planning-stride 2 --control-stride 2 \
    --deployment-execution-horizon 15 --wam4va --wmrm-inject all --wmrm-target dino \
    --wmrm-adep-weight 0 --wmrm-cycle-steps 15 --wmrm-world-weight 1.0 \
    --world-action-rank-stage final --dino-main-vision --dino-dense-metric \
    --main-vision-checkpoint "$DINO" --main-vision-grid 16 --main-vision-frames 4 \
    --main-vision-temporal --main-vision-temporal-scale 1.0 \
    --main-vision-encode-batch "$MAIN_VISION_ENCODE_BATCH" --metric-geometry-inject \
    --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
    --wmrm-predictor st_blocks --wmrm-predictor-depth 6 --wmrm-predictor-width 384 \
    --wmrm-predictor-heads 12 --single-task --task-sampling balanced \
    --task-locality-block-batches 64 --batch-size "$BATCH" --sequence-length 4 \
    --min-sequence-length 4 --num-workers 0 --lr 0.0001 --seed 0 --device cuda \
    --feature-autocast-bf16 --va-layers 8 --va-attention-backend auto \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 --flow-prefix-steps 2 \
    --flow-prefix-weight 1.0 --flow-tail-weight 1.0 --mtvj-train-metric-head \
    --lr-mtvj-metric-head 0.0003 --mtvj-train-relation --lr-mtvj-relation 0.00002 \
    --mtvj-visual-aux-every 10 --mtvj-visual-aux-batch 8 --steps "$STEPS" \
    --save-every "$SAVE_EVERY" --save-step-copies --save "$save" \
    --longtraj-dir "$ALL49_FRAMES_DIR" --resume-weights "$RESUME_WEIGHTS" \
    2>&1 | tee "$log"
}

command -v flock >/dev/null || fail 'flock is required'
exec 9>"$LOCK"
flock -n 9 || fail 'another all49 H15 run owns the lock'
case "$MODE" in
  prepare) prepare_data; preflight 0 ;;
  preflight) preflight 1 ;;
  joint) preflight 1; run_joint ;;
  *) usage ;;
esac
