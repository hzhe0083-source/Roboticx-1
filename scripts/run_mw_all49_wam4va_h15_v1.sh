#!/usr/bin/env bash
# All-49 H15/P15 dual-stream VA<->World training on true P15 recurrent windows.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/conda/bin/python}
VERIFY_PY=${VERIFY_PY:-$PY}
DINO=${DINO:-/root/private_data/newhost_env/models/dinov2_vitl14_reg4.safetensors}
ALL49_DATA_DIR=${ALL49_DATA_DIR:-/root/ora0_all49_data}
ALL49_FRAMES_DIR=${ALL49_FRAMES_DIR:-/root/ora0_all49_raw}
HARD2_FRAMES_DIR=${HARD2_FRAMES_DIR:-data/frames_v2}
ALLTASK_H48_REF=${ALLTASK_H48_REF:-data/metaworld_longtraj_windows_h48_all49_repaired_v2_clean.pt}
RAW_IDENTITY_MANIFEST=${RAW_IDENTITY_MANIFEST:-$ALL49_DATA_DIR/all49_raw_canonical_identity_v1.json}

SOURCE=${SOURCE:-$ALL49_DATA_DIR/all49_peer_h15_p15_source_v1.pt}
WORLD_POOL=$ALL49_DATA_DIR/all49_peer_h15_p15_world_pool_v1.pt
VA_DATA=$ALL49_DATA_DIR/all49_peer_h15_p15_va_train_v1.pt
WORLD_DATA=$ALL49_DATA_DIR/all49_peer_h15_p15_world_train_v1.pt
EVAL_DATA=${EVAL_DATA:-$ALL49_DATA_DIR/all49_peer_h15_p15_eval_v1.pt}
PARTITION_MANIFEST=$ALL49_DATA_DIR/all49_peer_h15_p15_va_world_partition_v1.json
WORLD_MANIFEST=$ALL49_DATA_DIR/all49_peer_h15_p15_world_split_v1.json
FULL_TRAIN=${FULL_TRAIN:-$ALL49_DATA_DIR/all49_peer_h15_p15_full_train_v1.pt}
FULL_MANIFEST=${FULL_MANIFEST:-$ALL49_DATA_DIR/all49_peer_h15_p15_full_split_v1.json}

MODE=${1:-}
STEPS=${2:-}
BATCH=${3:-48}
EPOCHS=${EPOCHS:-23}
EXPECTED_EPOCHS=${EXPECTED_EPOCHS:-23}
NGPUS=${NGPUS:-2}
RUN_ID=${RUN_ID:-mw_all49_wam4va_h15_p15_full10722_e23_lang_slotfree_scratch_v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/ora0_ckpts}
SAVE_EVERY=${SAVE_EVERY:-670}
MAIN_VISION_ENCODE_BATCH=${MAIN_VISION_ENCODE_BATCH:-16}
# Ordinary-task JPEG decode is 4-10s; assembly is 40-70s. One prefetch
# thread plus 2-4 batches/task means depth 8 (~64s) still leaves a hole
# when the slower DDP rank decodes. Depth 16 covers ~140s.
PEER_BATCH_PREFETCH_DEPTH=${PEER_BATCH_PREFETCH_DEPTH:-16}
EXPECTED_SOURCE_WINDOWS=${EXPECTED_SOURCE_WINDOWS:-11903}
EXPECTED_TRAIN_WINDOWS=${EXPECTED_TRAIN_WINDOWS:-10722}
EXPECTED_EVAL_WINDOWS=${EXPECTED_EVAL_WINDOWS:-1181}
EXPECTED_TASKS=${EXPECTED_TASKS:-49}
EXPECTED_RAW_CONTRACT=${EXPECTED_RAW_CONTRACT:-all49_canonical_raw_sources_v1}
RESUME_WEIGHTS=${RESUME_WEIGHTS:-}
LOCK=/tmp/ora0_all49_wam4va_h15_p15_v1.lock

fail(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage(){ printf 'usage: %s {prepare|prepare-full} | %s {preflight|joint} STEPS [global-batch]\n' "$0" "$0" >&2; exit 2; }

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
from scripts.build_longtraj_features import PEER_SYNC_H15_P15_CONTRACT, phase1

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
    data_contract=PEER_SYNC_H15_P15_CONTRACT,
    planning_stride=15,
)
PY
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$SOURCE" --train-output "$WORLD_POOL" --eval-output "$VA_DATA" \
    --manifest-output "$PARTITION_MANIFEST" --heldout-fraction 0.50 --seed 101
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$WORLD_POOL" --train-output "$WORLD_DATA" --eval-output "$EVAL_DATA" \
    --manifest-output "$WORLD_MANIFEST" --heldout-fraction 0.20 --seed 202
}

prepare_full_data(){
  for artifact in "$SOURCE" "$EVAL_DATA" "$WORLD_MANIFEST"; do
    [[ -f "$artifact" ]] || fail "prepare-full requires existing artifact: $artifact"
  done
  [[ ! -e "$FULL_TRAIN" ]] || fail "refusing to overwrite immutable artifact: $FULL_TRAIN"
  [[ ! -e "$FULL_MANIFEST" ]] || fail "refusing to overwrite immutable artifact: $FULL_MANIFEST"
  "$PY" -B scripts/split_wam4va_episode_holdout.py \
    --input "$SOURCE" --train-output "$FULL_TRAIN" --eval-output "$EVAL_DATA" \
    --manifest-output "$FULL_MANIFEST" --reuse-existing-eval
}

preflight(){
  local enforce_steps=${1:-0}
  for artifact in "$SOURCE" "$FULL_TRAIN" "$EVAL_DATA" "$FULL_MANIFEST" \
    "$RAW_IDENTITY_MANIFEST" "$DINO"; do
    [[ -f "$artifact" ]] || fail "missing required artifact: $artifact"
  done
  [[ -z "$RESUME_WEIGHTS" || -f "$RESUME_WEIGHTS" ]] || \
    fail "missing resume-weights checkpoint: $RESUME_WEIGHTS"
  "$VERIFY_PY" -B - "$SOURCE" "$FULL_TRAIN" "$EVAL_DATA" "$FULL_MANIFEST" \
    "$RAW_IDENTITY_MANIFEST" "$ALL49_FRAMES_DIR" "$BATCH" "$STEPS" \
    "$EPOCHS" "$enforce_steps" "$EXPECTED_SOURCE_WINDOWS" \
    "$EXPECTED_TRAIN_WINDOWS" "$EXPECTED_EVAL_WINDOWS" \
    "$RESUME_WEIGHTS" "$EXPECTED_TASKS" "$EXPECTED_RAW_CONTRACT" \
    "$EXPECTED_EPOCHS" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys
import torch

(
    source_path,
    train_path,
    eval_path,
    full_manifest_path,
    raw_manifest_path,
    frames_dir,
) = map(Path, sys.argv[1:7])
global_batch = int(sys.argv[7])
requested_steps = sys.argv[8]
epochs = int(sys.argv[9])
enforce_steps = bool(int(sys.argv[10]))
expected_counts = {
    "source": int(sys.argv[11]),
    "train": int(sys.argv[12]),
    "eval": int(sys.argv[13]),
}
resume_weights = sys.argv[14]
expected_task_count = int(sys.argv[15])
expected_raw_contract = sys.argv[16]
expected_epochs = int(sys.argv[17])
if expected_task_count <= 0:
    raise SystemExit("expected task count must be positive")


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


full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
if full_manifest.get("contract") != "wam4va_episode_holdout_manifest_v1" or int(
    full_manifest.get("contract_version", -1)
) != 1:
    raise SystemExit("full: unknown manifest contract")
if canonical_manifest_sha256(full_manifest) != full_manifest.get("manifest_sha256"):
    raise SystemExit("full: canonical manifest SHA mismatch")
source_contract = full_manifest.get("source") or {}
if Path(str(source_contract.get("path", ""))).name != source_path.name:
    raise SystemExit("full: source path binding mismatch")
if sha256_file(source_path) != source_contract.get("sha256"):
    raise SystemExit("full: source SHA mismatch")
splits = full_manifest.get("splits") or {}
for split_name, expected in (("train", train_path), ("eval", eval_path)):
    actual = Path(str((splits.get(split_name) or {}).get("output_path", ""))).name
    if actual != expected.name:
        raise SystemExit(f"full: {split_name} output binding mismatch")
selection = full_manifest.get("selection") or {}
if selection.get("rule") != "exact_existing_eval_episode_complement_v1":
    raise SystemExit("full: expected exact existing-eval complement selection")
if sha256_file(eval_path) != selection.get("existing_eval_sha256"):
    raise SystemExit("full: elected eval SHA mismatch")

raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
if raw_manifest.get("contract") != expected_raw_contract:
    raise SystemExit(
        f"raw identity contract mismatch: {raw_manifest.get('contract')!r} "
        f"!= {expected_raw_contract!r}"
    )
reference = raw_manifest.get("reference") or {}
reference_path = Path(str(reference.get("path", "")))
if (
    not reference_path.is_file()
    or int(reference_path.stat().st_size) != int(reference.get("size_bytes", -1))
    or sha256_file(reference_path) != reference.get("sha256")
):
    raise SystemExit("all49 H48 reference identity mismatch")
raw_entries = list(raw_manifest.get("sources") or [])
if (
    len(raw_entries) != expected_task_count
    or len({item.get("task") for item in raw_entries}) != expected_task_count
):
    raise SystemExit(
        f"raw identity manifest must contain {expected_task_count} unique tasks"
    )
declared_raw = source_contract.get("payload_source_identities") or []
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
    for name, path in (
        ("source", source_path),
        ("train", train_path),
        ("eval", eval_path),
    )
}
counts = {name: len(payload["actions"]) for name, payload in payloads.items()}
if counts != expected_counts:
    raise SystemExit(f"formal full-data counts mismatch: {counts}")
episode_sets = {}
row_sets = {}
expected_ids = list(range(expected_task_count))
for name, payload in payloads.items():
    actions = payload.get("actions")
    metadata = payload.get("metadata") or {}
    if not isinstance(actions, torch.Tensor) or tuple(actions.shape[1:]) != (4, 15, 4):
        raise SystemExit(f"{name}: expected T4/H15/A4, got {getattr(actions, 'shape', None)}")
    required = {
        "contract": "peer_sync_h15_p15_world_windows_v1",
        "contract_version": 1,
        "fps": 80,
        "control_stride": 15,
        "planning_stride": 15,
        "sequence_length": 4,
        "decision_offsets": [0, 15, 30, 45],
        "action_horizon": 15,
        "action_label_offsets": list(range(15)),
        "world_target_horizon": 15,
        "world_target_offsets": [15, 30, 45, 60],
    }
    bad = {key: (metadata.get(key), value) for key, value in required.items()
           if metadata.get(key) != value}
    if bad:
        raise SystemExit(f"{name}: H15 cadence mismatch: {bad}")
    task_ids = sorted(int(value) for value in torch.unique(payload["instruction_id"]))
    if task_ids != expected_ids:
        raise SystemExit(
            f"{name}: expected instruction ids 0..{expected_task_count - 1}, "
            f"got {task_ids}"
        )
    target_valid = payload.get("world_target_valid_mask")
    target_refs = payload.get("world_target_frame_refs")
    frame_refs = payload.get("frame_refs")
    previous_action = payload.get("previous_action")
    if not isinstance(target_valid, torch.Tensor) or target_valid.shape != actions.shape[:2]:
        raise SystemExit(f"{name}: invalid world_target_valid_mask")
    if target_valid.dtype != torch.bool:
        raise SystemExit(f"{name}: world_target_valid_mask must be bool")
    if not isinstance(target_refs, (list, tuple)) or len(target_refs) != len(actions):
        raise SystemExit(f"{name}: invalid world_target_frame_refs")
    if not isinstance(frame_refs, (list, tuple)) or len(frame_refs) != len(actions):
        raise SystemExit(f"{name}: invalid frame_refs")
    if (
        not isinstance(previous_action, torch.Tensor)
        or tuple(previous_action.shape) != tuple(actions.shape[:2]) + (4,)
        or not torch.equal(previous_action[:, 1:], actions[:, :-1, 14])
    ):
        raise SystemExit(
            f"{name}: next previous_action is not prior P15 segment token14"
        )
    for row, (current_ref, target_ref) in enumerate(
        zip(frame_refs, target_refs, strict=True)
    ):
        current_indices = current_ref[2]
        target_indices = target_ref[2]
        if len(current_indices) != 4 or len(target_indices) != 4:
            raise SystemExit(f"{name}: row {row} requires four decision/target refs")
        current_decisions = [int(indices[-1]) for indices in current_indices]
        world_targets = [int(indices[0]) for indices in target_indices]
        if world_targets != [decision + 15 for decision in current_decisions]:
            raise SystemExit(
                f"{name}: row {row} World refs are not d+15 endpoints"
            )
    if name == "train":
        split = splits.get("train") or {}
        if metadata.get("split_manifest_sha256") != full_manifest.get("manifest_sha256"):
            raise SystemExit("train: split manifest SHA binding mismatch")
        if metadata.get("split_name") != "train":
            raise SystemExit("train: split name binding mismatch")
        if metadata.get("output_identity") != split.get("output_identity"):
            raise SystemExit("train: split output identity mismatch")
        if metadata.get("parent_identity") != source_contract:
            raise SystemExit("train: split parent identity mismatch")
    # The elected eval artifact is intentionally frozen across dataset growth,
    # so its historical raw identities differ. Its immutable SHA and exact row
    # membership are verified above/below; source and train must bind new raw.
    if name != "eval" and metadata.get("source_identities") != declared_raw:
        raise SystemExit(f"{name}: raw source identity binding mismatch")
    episode_sets[name] = set(zip(
        payload["instruction_id"].tolist(), payload["episode_id"].tolist(), strict=True
    ))
    row_keys = {
        (
            int(task_id),
            int(episode_id),
            json.dumps(frame_ref, separators=(",", ":")),
        )
        for task_id, episode_id, frame_ref in zip(
            payload["instruction_id"].tolist(),
            payload["episode_id"].tolist(),
            frame_refs,
            strict=True,
        )
    }
    if len(row_keys) != len(actions):
        raise SystemExit(f"{name}: duplicate row identities")
    row_sets[name] = row_keys
    frame_tasks = {str(ref[0]) for ref in payload["frame_refs"]}
    target_tasks = {str(ref[0]) for ref in target_refs}
    if frame_tasks != expected_tasks or target_tasks != expected_tasks:
        raise SystemExit(
            f"{name}: frame sources do not cover the canonical "
            f"{expected_task_count} tasks"
        )
    for task_file in frame_tasks:
        raw = frames_dir / f"metaworld_longtraj_{task_file}.pt"
        if not raw.is_file():
            raise SystemExit(f"{name}: missing frame source {raw}")
if episode_sets["train"] & episode_sets["eval"]:
    raise SystemExit("episode leakage: train/eval")
if episode_sets["train"] | episode_sets["eval"] != episode_sets["source"]:
    raise SystemExit("train/eval episodes are not the exhaustive source partition")
if row_sets["train"] & row_sets["eval"]:
    raise SystemExit("row leakage: train/eval")
if row_sets["train"] | row_sets["eval"] != row_sets["source"]:
    raise SystemExit("full train is not exactly SOURCE - EVAL")
for split_name in ("train", "eval"):
    split = splits.get(split_name) or {}
    if int(split.get("windows", -1)) != counts[split_name]:
        raise SystemExit(f"full manifest {split_name} window count mismatch")
    expected_episodes = {int(value) for value in split.get("episode_ids", [])}
    actual_episodes = {episode for _, episode in episode_sets[split_name]}
    if expected_episodes != actual_episodes:
        raise SystemExit(f"full manifest {split_name} episode mismatch")
if global_batch <= 0:
    raise SystemExit("global batch must be positive")
if global_batch != 48:
    raise SystemExit(f"formal all49 full run requires global batch=48, got {global_batch}")
steps_per_epoch = (counts["train"] + global_batch - 1) // global_batch
last_batch = counts["train"] % global_batch
expected_steps = epochs * steps_per_epoch
if epochs != expected_epochs:
    raise SystemExit(
        f"formal run requires EPOCHS={expected_epochs}, got {epochs}"
    )
if enforce_steps and requested_steps != str(expected_steps):
    raise SystemExit(
        f"formal run requires STEPS={expected_steps} "
        f"({expected_epochs} * {steps_per_epoch}), got {requested_steps!r}"
    )


print(
    f"MT{expected_task_count} H15 full preflight: PASS; "
    f"shared VA+World train={counts['train']}, "
    f"eval={counts['eval']}, steps/epoch={steps_per_epoch} (last={last_batch}), "
    f"{expected_epochs} epochs={expected_steps} steps, "
    f"initialization={'weights:' + Path(resume_weights).name if resume_weights else 'scratch'}, "
    "language=full-token VA+World, policy=slot-free"
)
PY
}

require_idle(){
  ! pgrep -af '[p]ython.*train.py' >/dev/null || fail 'another train.py is active'
}

run_joint(){
  [[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail 'joint requires a positive step count'
  [[ "$EPOCHS" == "$EXPECTED_EPOCHS" ]] || \
    fail "formal joint run requires EPOCHS=$EXPECTED_EPOCHS"
  [[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || fail 'global batch must be positive'
  [[ "$NGPUS" =~ ^[1-9][0-9]*$ ]] || fail 'NGPUS must be positive'
  (( BATCH % NGPUS == 0 )) || fail "global batch $BATCH must divide across $NGPUS GPUs"
  local save=$CHECKPOINT_DIR/$RUN_ID.pt
  local log=logs/$RUN_ID.log
  [[ ! -e "$save" && ! -e "$log" ]] || fail "refusing to overwrite run $RUN_ID"
  mkdir -p "$CHECKPOINT_DIR" logs
  require_idle
  local launcher=("$PY" -u -B)
  local resume_args=()
  if [[ -n "$RESUME_WEIGHTS" ]]; then
    [[ -f "$RESUME_WEIGHTS" ]] || fail "missing resume-weights checkpoint: $RESUME_WEIGHTS"
    resume_args=(--resume-weights "$RESUME_WEIGHTS")
  fi
  if (( NGPUS > 1 )); then
    launcher=("$PY" -m torch.distributed.run --standalone \
      --nproc_per_node="$NGPUS" --max_restarts=0)
  fi
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${launcher[@]}" train.py --va-data "$FULL_TRAIN" --world-data "$FULL_TRAIN" \
    --peer-shared-full-data --visual-world-supervision \
    --world-split-manifest "$FULL_MANIFEST" \
    --va-world-mode peer_sync_h6 --planning-stride 15 --control-stride 15 \
    --deployment-execution-horizon 15 --wam4va --wmrm-full-language-tokens \
    --slot-free-policy --wmrm-inject all --wmrm-target dino \
    --wmrm-adep-weight 0 --wmrm-cycle-steps 15 --wmrm-world-weight 1.0 \
    --world-action-rank-stage final --dino-main-vision \
    --main-vision-checkpoint "$DINO" --main-vision-grid 16 --main-vision-frames 4 \
    --main-vision-temporal --main-vision-temporal-scale 1.0 \
    --main-vision-encode-batch "$MAIN_VISION_ENCODE_BATCH" \
    --wmrm-map-size 16 --wmrm-map-channels 1024 --wmrm-world-grid 16 \
    --wmrm-predictor st_blocks --wmrm-predictor-depth 6 --wmrm-predictor-width 384 \
    --wmrm-predictor-heads 12 --single-task --task-sampling full \
    --task-locality-block-batches 64 --batch-size "$BATCH" --sequence-length 4 \
    --min-sequence-length 4 --num-workers 0 --peer-batch-prefetch \
    --peer-batch-prefetch-depth "$PEER_BATCH_PREFETCH_DEPTH" \
    --longtraj-decode-cache-tasks 2 --disable-runtime-integrity-checks \
    --lr 0.0001 --seed 0 --device cuda \
    --feature-autocast-bf16 --va-layers 8 --va-attention-backend auto \
    --flow-cond adaln --flow-layers 6 --flow-steps 8 --flow-prefix-steps 15 \
    --flow-prefix-weight 1.0 --flow-tail-weight 1.0 --steps "$STEPS" \
    --save-every "$SAVE_EVERY" --save-step-copies --save "$save" \
    --longtraj-dir "$ALL49_FRAMES_DIR" "${resume_args[@]}" \
    2>&1 | tee "$log"
}

command -v flock >/dev/null || fail 'flock is required'
exec 9>"$LOCK"
flock -n 9 || fail 'another all49 H15 run owns the lock'
case "$MODE" in
  prepare) prepare_data; prepare_full_data; preflight 0 ;;
  prepare-full) prepare_full_data; preflight 0 ;;
  preflight) preflight 1 ;;
  joint) preflight 1; run_joint ;;
  *) usage ;;
esac
