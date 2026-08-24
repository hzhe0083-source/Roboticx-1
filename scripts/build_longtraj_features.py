#!/usr/bin/env python
"""长轨迹 → 预计算特征训练数据（E7 路径，2026-08-09）。

两阶段：
  Phase 1（轻量，无 GPU）：遍历 data/metaworld_longtraj_*.pt（JPEG 压缩帧），
    滑动窗口切片 → 动作/状态/prev（executed-clip + 全局 q01/q99 继承）+
    帧索引 (task, ep, start)。输出 data/metaworld_longtraj_windows.pt。
  Phase 2（GPU ~30-40 分钟）：按帧索引从 per-task 文件解压窗口帧 →
    冻结原始 V-JEPA 2.1 编码（spatiotemporal 288，与 E7 视觉路径一致）→
    memmap 到 /media/ryan/robot-data/longtraj_st288.npy + meta.pt。

训练命令（E7）：train.py --data data/metaworld_longtraj_windows.pt
  --local-slots-data /media/ryan/robot-data/longtraj_st288_meta.pt ...

用法：
  python scripts/build_longtraj_features.py [--device cuda]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 80
CONTROL_STRIDE = 6
SEQUENCE_LENGTH = 4
ACTION_HORIZON = 8
VISION_WINDOW = 4
VISION_STRIDE = 2

REF = ROOT / "data" / "metaworld_fullframe_executed.pt"
ST_NPY_DIR = Path("/media/ryan/robot-data")
LEGACY_PERTURB_SETTLE_STEPS = 12
PEER_SYNC_H6_CONTRACT = "peer_sync_h6_world_windows_v1"
PEER_SYNC_H6_P2_CONTRACT = "peer_sync_h6_p2_world_windows_v1"
PEER_SYNC_H15_P2_CONTRACT = "peer_sync_h15_p2_world_windows_v1"
PEER_SYNC_H15_P15_CONTRACT = "peer_sync_h15_p15_world_windows_v1"
PEER_SYNC_H6_VERSION = 1
PEER_SYNC_H6_P2_VERSION = 1
PEER_SYNC_H6_P2_STRIDE = 2
PEER_SYNC_H15_P15_STRIDE = 15
PEER_SYNC_H15_CONTRACTS = frozenset({
    PEER_SYNC_H15_P2_CONTRACT,
    PEER_SYNC_H15_P15_CONTRACT,
})
PEER_SYNC_H6_CONTRACTS = frozenset({
    PEER_SYNC_H6_CONTRACT,
    PEER_SYNC_H6_P2_CONTRACT,
    *PEER_SYNC_H15_CONTRACTS,
})


def win_out(horizon: int, task: str | None = None) -> Path:
    suffix = "" if task is None else f"_{task}"
    return ROOT / "data" / f"metaworld_longtraj_windows_h{horizon}{suffix}.pt"


def st_paths(horizon: int, task: str | None = None) -> tuple[Path, Path]:
    suffix = "" if task is None else f"_{task}"
    return (ST_NPY_DIR / f"longtraj_st288_h{horizon}{suffix}.npy",
            ST_NPY_DIR / f"longtraj_st288_h{horizon}{suffix}_meta.pt")


def _sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(stat.st_size),
    }


def _frame_ref_key(path: Path) -> str:
    """Key accepted by LongTrajFramesDataset's existing filename resolver."""
    prefix = "metaworld_longtraj_"
    if not path.name.startswith(prefix) or path.suffix != ".pt":
        raise ValueError(
            f"longtraj source must be named {prefix}<source>.pt, got {path.name}"
        )
    return path.stem[len(prefix):]


def _save_new(payload: dict, path: Path, *, overwrite: bool = False) -> None:
    """Atomic save; refuse accidental replacement unless explicitly requested."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing dataset: {path}; choose --output or pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"stale temporary output exists: {tmp}")
    torch.save(payload, tmp)
    tmp.replace(path)


_WARNED_LEGACY: set[tuple[str, str]] = set()


def _legacy_issue(message: str, policy: str, category: str) -> None:
    # ``infer`` is a repair mode, so anything it cannot infer uniquely is an
    # error.  Only the compatibility ``warn`` mode is allowed to continue.
    if policy in {"error", "infer"}:
        raise ValueError(message)
    source_file = message.split(":episode[", 1)[0]
    key = (source_file, category)
    if key in _WARNED_LEGACY:
        return
    _WARNED_LEGACY.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal ``True`` intervals as half-open ``[start, end)`` runs."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def _infer_legacy_v1_semantics(ep: dict, source: str) -> dict[str, object]:
    """Recover the task-agnostic timeline contract of the original collector.

    The v1 collector inserted one *exact* 12-action all-zero settle block after
    a random pre-success perturbation, but ``success_frame`` remained on the
    outer policy-step timeline.  Consequently every stored action after that
    block is shifted by 12.  Inference deliberately fails if the stored episode
    does not identify that event uniquely; silently guessing would corrupt H48
    supervision again.
    """
    forbidden = [
        key for key in (
            "first_success", "settle_mask", "recovery_mask",
            "perturb_start", "perturb_end", "perturb_event",
        )
        if key in ep
    ]
    if forbidden:
        raise ValueError(
            f"{source}: --legacy-policy infer expects the original v1 schema, "
            f"but partial timeline annotations are present: {forbidden}"
        )
    if "success_frame" not in ep:
        raise ValueError(
            f"{source}: --legacy-policy infer requires legacy success_frame"
        )
    raw_success = ep["success_frame"]
    if isinstance(raw_success, (bool, np.bool_)) or not isinstance(
        raw_success, (int, np.integer)
    ):
        raise ValueError(
            f"{source}: legacy success_frame must be an integer, got {raw_success!r}"
        )

    actions = np.asarray(ep["actions"])
    n = len(actions)
    if actions.shape != (n, 4):
        raise ValueError(
            f"{source}: legacy actions must have shape ({n},4), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"{source}: legacy actions contain NaN/Inf")
    raw_success = int(raw_success)
    if not 0 <= raw_success < n:
        raise ValueError(
            f"{source}: legacy success_frame={raw_success} outside [0,{n})"
        )
    perturbed_value = ep.get("perturbed")
    if not isinstance(perturbed_value, (bool, np.bool_)):
        raise ValueError(
            f"{source}: --legacy-policy infer requires scalar bool perturbed"
        )
    perturbed = bool(perturbed_value)

    zero_runs = _true_runs(np.equal(actions, 0).all(axis=1))
    candidates = [
        (start, end)
        for start, end in zero_runs
        if end - start == LEGACY_PERTURB_SETTLE_STEPS
        and 0 < start <= raw_success
        and end < n
    ]
    if not perturbed:
        if candidates:
            raise ValueError(
                f"{source}: perturbed=False conflicts with pre-success exact "
                f"{LEGACY_PERTURB_SETTLE_STEPS}-zero-action block(s) {candidates}"
            )
        return {
            "first_success": raw_success,
            "settle": np.zeros(n, dtype=bool),
            "recovery": np.zeros(n, dtype=bool),
            "perturb_start": None,
            "perturb_end": None,
        }

    if len(candidates) != 1:
        run_summary = [(start, end, end - start) for start, end in zero_runs]
        raise ValueError(
            f"{source}: perturbed legacy episode must contain exactly one maximal "
            f"pre-success {LEGACY_PERTURB_SETTLE_STEPS}-action all-zero settle "
            f"block; candidates={candidates}, all_zero_runs={run_summary}"
        )
    perturb_start, perturb_end = candidates[0]
    first_success = raw_success + LEGACY_PERTURB_SETTLE_STEPS
    if not perturb_end <= first_success < n:
        raise ValueError(
            f"{source}: inferred first_success={first_success} is inconsistent "
            f"with perturb=[{perturb_start},{perturb_end}) and length={n}"
        )
    settle = np.zeros(n, dtype=bool)
    settle[perturb_start:perturb_end] = True
    recovery = np.zeros(n, dtype=bool)
    recovery[perturb_start:first_success + 1] = True
    return {
        "first_success": first_success,
        "settle": settle,
        "recovery": recovery,
        "perturb_start": perturb_start,
        "perturb_end": perturb_end,
    }


def resolve_episode_semantics(ep: dict, source: str,
                              legacy_policy: str = "warn") -> dict[str, object]:
    """Resolve v2 masks and metric state, with explicit v1 fallback.

    Returned masks use the stored pre-observation/action timeline. ``valid`` is
    suitable for direct horizon supervision: actions must have been executed,
    have a valid aligned frame, not be settle actions, and occur no later than
    the first successful action.
    """
    n = len(ep["actions"])
    lengths = {"frames": len(ep["frames"]), "states": len(ep["states"])}
    if any(length != n for length in lengths.values()):
        raise ValueError(f"{source}: timeline length mismatch actions={n}, {lengths}")

    if legacy_policy not in {"warn", "error", "infer"}:
        raise ValueError(f"unknown legacy_policy={legacy_policy!r}")
    is_v2 = "action_supervision_valid" in ep and "action_executed" in ep
    inferred: dict[str, object] | None = None
    if not is_v2:
        if legacy_policy == "infer":
            inferred = _infer_legacy_v1_semantics(ep, source)
        else:
            _legacy_issue(
                f"{source}: legacy episode has no v2 execution/validity contract; "
                "post-success actions can be masked, but historical perturb-settle "
                "positions are unknowable",
                legacy_policy,
                "contract",
            )

    def bool_array(key: str, default: np.ndarray) -> np.ndarray:
        value = np.asarray(ep.get(key, default), dtype=bool).copy()
        if value.shape != (n,):
            raise ValueError(f"{source}: {key} must have shape ({n},), got {value.shape}")
        return value

    frame_valid = bool_array("frame_valid", np.ones(n, dtype=bool))
    executed = bool_array("action_executed", np.ones(n, dtype=bool))
    inferred_settle = (
        np.asarray(inferred["settle"], dtype=bool)
        if inferred is not None else np.zeros(n, dtype=bool)
    )
    settle = bool_array("settle_mask", inferred_settle)
    if (inferred is None and "settle_mask" not in ep
            and bool(ep.get("perturbed", False))):
        _legacy_issue(
            f"{source}: perturbed legacy episode lacks perturb_start/end; "
            "settle targets cannot be identified safely",
            legacy_policy,
            "settle",
        )
    supplied_valid = ep.get("action_supervision_valid", ep.get("action_valid"))
    valid = bool_array(
        "action_supervision_valid",
        np.ones(n, dtype=bool) if supplied_valid is None else np.asarray(supplied_valid),
    )
    first_success = (
        inferred["first_success"]
        if inferred is not None
        else ep.get("first_success", ep.get("success_frame"))
    )
    if first_success is None:
        _legacy_issue(
            f"{source}: no first_success/success_frame; post-success masking is impossible",
            legacy_policy,
            "success",
        )
    else:
        first_success = int(first_success)
        if not 0 <= first_success < n:
            raise ValueError(f"{source}: first_success={first_success} outside [0,{n})")
        valid[first_success + 1:] = False
    valid &= frame_valid & executed & ~settle

    inferred_recovery = (
        np.asarray(inferred["recovery"], dtype=bool)
        if inferred is not None else np.zeros(n, dtype=bool)
    )
    recovery = bool_array("recovery_mask", inferred_recovery)
    event = ep.get("perturb_event") or {}
    perturb_start = (
        inferred["perturb_start"]
        if inferred is not None else ep.get("perturb_start", event.get("start"))
    )
    perturb_end = (
        inferred["perturb_end"]
        if inferred is not None else ep.get("perturb_end", event.get("end"))
    )
    if "recovery_mask" not in ep and perturb_start is not None:
        start = int(perturb_start)
        end = first_success + 1 if first_success is not None else n
        if not 0 <= start <= end <= n:
            raise ValueError(f"{source}: invalid inferred recovery interval [{start},{end})")
        recovery[start:end] = True
    if perturb_start is not None:
        perturb_start = int(perturb_start)
        if perturb_end is None:
            raise ValueError(
                f"{source}: perturb_start is present but perturb_end is missing"
            )
        perturb_end = int(perturb_end)
        if not 0 <= perturb_start <= perturb_end <= n:
            raise ValueError(
                f"{source}: invalid perturb interval [{perturb_start},{perturb_end})"
            )

    metric = ep.get("metric_state")
    if metric is None and "lock_pos" in ep and "lock_target" in ep:
        metric = np.concatenate(
            [np.asarray(ep["lock_pos"]), np.asarray(ep["lock_target"])], axis=-1
        )
    if metric is None:
        metric_state = np.zeros((n, 6), dtype=np.float32)
        metric_valid = np.zeros(n, dtype=bool)
    else:
        metric_state = np.asarray(metric, dtype=np.float32).copy()
        if metric_state.shape != (n, 6):
            raise ValueError(
                f"{source}: metric_state must have shape ({n},6), got {metric_state.shape}"
            )
        metric_valid = bool_array(
            "metric_state_valid", np.isfinite(metric_state).all(axis=-1)
        )
        metric_valid &= np.isfinite(metric_state).all(axis=-1)
        metric_state = np.nan_to_num(metric_state, copy=False)

    return {
        "valid": valid,
        "recovery": recovery,
        "frame_valid": frame_valid,
        "metric_state": metric_state,
        "metric_valid": metric_valid,
        "first_success": first_success,
        "perturb_start": perturb_start,
        "perturb_end": perturb_end,
        "legacy_inferred": inferred is not None,
    }

# MT1 环境名 → lerobot 任务文本（与 REF.metadata.tasks 对齐，2026-08-09 全量核对）
ENV_TO_TASK = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "basketball-v3": "Dunk the basketball into the basket",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
    "box-close-v3": "Grasp the cover and close the box with it",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
    "button-press-v3": "Press a button",
    "button-press-wall-v3": "Bypass a wall and press a button",
    "coffee-button-v3": "Push a button on the coffee machine",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "coffee-push-v3": "Push a mug under a coffee machine",
    "dial-turn-v3": "Rotate a dial 180 degrees",
    "disassemble-v3": "Pick a nut out of a peg",
    "door-close-v3": "Close a door with a revolving joint",
    "door-lock-v3": "Lock the door by rotating the lock clockwise",
    "door-open-v3": "Open a door with a revolving joint",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise",
    "hand-insert-v3": "Insert the gripper into a hole",
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "faucet-open-v3": "Rotate the faucet counter-clockwise",
    "faucet-close-v3": "Rotate the faucet clockwise",
    "hammer-v3": "Hammer a screw on the wall",
    "handle-press-side-v3": "Press a handle down sideways",
    "handle-press-v3": "Press a handle down",
    "handle-pull-side-v3": "Pull a handle up sideways",
    "handle-pull-v3": "Pull a handle up",
    "lever-pull-v3": "Pull a lever down 90 degrees",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
    "pick-out-of-hole-v3": "Pick up a puck from a hole",
    "pick-place-v3": "Pick and place a puck to a goal",
    "plate-slide-v3": "Slide a plate into a cabinet",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways",
    "plate-slide-back-v3": "Get a plate from the cabinet",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways",
    "peg-insert-side-v3": "Insert a peg sideways",
    "peg-unplug-side-v3": "Unplug a peg sideways",
    "soccer-v3": "Kick a soccer into the goal",
    "stick-push-v3": "Grasp a stick and push a box using the stick",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick",
    "push-v3": "Push the puck to a goal",
    "push-back-v3": "Pull a puck to a goal",
    "push-wall-v3": "Bypass a wall and push a puck to a goal",
    "reach-v3": "Reach a goal position",
    "reach-wall-v3": "Bypass a wall and reach a goal",
    "shelf-place-v3": "Pick and place a puck onto a shelf",
    "sweep-into-v3": "Sweep a puck into a hole",
    "sweep-v3": "Sweep a puck off the table",
    "window-open-v3": "Push and open a window",
    "window-close-v3": "Push and close a window",
}


def clip_frame_indices(decision: int, video_start_frame: int = 0,
                       window: int = VISION_WINDOW, stride: int = VISION_STRIDE):
    """与 canonical（prepare_pnpw_features.clip_frame_indices / live_vjepa）
    完全一致的历史帧窗：决策点 d 用 [d-(window-1)*stride, ..., d-2*stride, d-stride, d]
    （clamp 到轨迹起点）。Codex P0-3（2026-08-09）：此前误写为 [d, d+2, d+4, d+6]
    未来帧——特征含目标动作结果，训练泄漏且与 live/eval 契约相反。"""
    off = np.arange(window) - (window - 1)   # [-(w-1), ..., -1, 0]
    return np.clip(decision + off * stride, video_start_frame, None)


def phase1(horizon: int, *, task: str | None = None,
           input_paths: list[Path] | None = None,
           output_path: Path | None = None,
           ref_path: Path = REF,
           legacy_policy: str = "warn",
           data_contract: str | None = None,
           planning_stride: int = CONTROL_STRIDE,
           overwrite: bool = False) -> Path:
    """窗口切片（动作/状态/prev/帧索引），无 GPU。horizon=action chunk 长度。

    Passing ``task`` selects only that task and defaults to a task-suffixed new
    output, so a door-lock repair cannot replace the all-task H48 dataset.
    ``input_paths`` can select a clean collector file explicitly.
    """
    if data_contract not in {None, *PEER_SYNC_H6_CONTRACTS}:
        raise ValueError(f"unknown data_contract={data_contract!r}")
    expected_peer_horizon = 15 if data_contract in PEER_SYNC_H15_CONTRACTS else 6
    if data_contract in PEER_SYNC_H6_CONTRACTS and horizon != expected_peer_horizon:
        raise ValueError(
            f"{data_contract} requires exact action horizon H{expected_peer_horizon}, "
            f"got H{horizon}"
        )
    if (
        isinstance(planning_stride, (bool, np.bool_))
        or not isinstance(planning_stride, (int, np.integer))
        or planning_stride <= 0
    ):
        raise ValueError(f"planning_stride must be a positive integer, got {planning_stride!r}")
    planning_stride = int(planning_stride)
    if data_contract in {PEER_SYNC_H6_P2_CONTRACT, PEER_SYNC_H15_P2_CONTRACT}:
        if planning_stride != PEER_SYNC_H6_P2_STRIDE:
            raise ValueError(
                f"{data_contract} requires planning_stride="
                f"{PEER_SYNC_H6_P2_STRIDE}, got {planning_stride}"
            )
    elif data_contract == PEER_SYNC_H15_P15_CONTRACT:
        if planning_stride != PEER_SYNC_H15_P15_STRIDE:
            raise ValueError(
                f"{PEER_SYNC_H15_P15_CONTRACT} requires planning_stride="
                f"{PEER_SYNC_H15_P15_STRIDE}, got {planning_stride}"
            )
    elif planning_stride != CONTROL_STRIDE:
        raise ValueError(
            f"planning_stride={planning_stride} requires data_contract="
            f"{PEER_SYNC_H6_P2_CONTRACT}; legacy contracts remain stride="
            f"{CONTROL_STRIDE}"
        )

    ref_path = Path(ref_path).expanduser().resolve(strict=True)
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    aq01, aq99 = ref["normalization"]["action_q01"], ref["normalization"]["action_q99"]
    sq01, sq99 = ref["normalization"]["state_q01"], ref["normalization"]["state_q99"]
    norm = dict(ref["normalization"])
    out_path = Path(output_path) if output_path is not None else win_out(horizon, task)

    if "language_hidden" not in ref or "language_mask" not in ref:
        raise ValueError(
            f"{ref_path}: missing language_hidden/language_mask; output would not be trainable"
        )
    n_tasks = len(ref["metadata"]["tasks"])
    task_language: list[torch.Tensor] = []
    task_language_mask: list[torch.Tensor] = []
    for tid in range(n_tasks):
        rows = (ref["instruction_id"] == tid).nonzero(as_tuple=False)
        if not len(rows):
            raise ValueError(f"{ref_path}: no language cache row for instruction_id={tid}")
        row = int(rows[0, 0])
        task_language.append(ref["language_hidden"][row])
        task_language_mask.append(ref["language_mask"][row])
    task_language_t = torch.stack(task_language)
    task_language_mask_t = torch.stack(task_language_mask)

    def robust(x, lo, hi):
        lo_n, hi_n = lo.numpy(), hi.numpy()
        return np.clip(2 * (x - lo_n) / (hi_n - lo_n) - 1, -1, 1)

    if input_paths:
        files = [Path(path) for path in input_paths]
    elif task is not None:
        files = [ROOT / "data" / f"metaworld_longtraj_{task}.pt"]
    else:
        # Canonical all-task build only. Variant clean/recovery sources must be
        # selected explicitly with --input so they cannot silently duplicate a task.
        files = sorted(
            path for name in ENV_TO_TASK
            if (path := ROOT / "data" / f"metaworld_longtraj_{name}.pt").is_file()
        )
    files = [path.expanduser().resolve(strict=False) for path in files]
    missing_files = [str(path) for path in files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"longtraj input files not found: {missing_files}")
    print(f"phase1(h={horizon}): {len(files)} task files")
    W = []
    episodes_seen = 0
    dropped_empty = 0
    legacy_episodes_inferred = 0
    legacy_perturb_events_inferred = 0
    for fi, path in enumerate(files):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "task" not in data or "episodes" not in data:
            raise ValueError(f"{path}: not a longtraj task payload")
        if task is not None and data["task"] != task:
            raise ValueError(
                f"{path}: contains task={data['task']!r}, requested task={task!r}"
            )
        task_text = ENV_TO_TASK.get(data["task"])
        if task_text is None:
            raise ValueError(f"{path}: unknown task {data['task']!r}")
        try:
            tid = ref["metadata"]["tasks"].index(task_text)
        except ValueError as exc:
            raise ValueError(f"{path}: task text absent from reference: {task_text!r}") from exc
        source_key = (
            data["task"]
            if data_contract in PEER_SYNC_H6_CONTRACTS
            else _frame_ref_key(path)
        )
        for ei, ep in enumerate(data["episodes"]):
            episodes_seen += 1
            frames_jpeg = ep["frames"]      # list[bytes]
            actions = ep["actions"]         # [T,4]
            states = ep["states"]           # [T,4]
            T = len(frames_jpeg)
            semantics = resolve_episode_semantics(
                ep, f"{path.name}:episode[{ei}]", legacy_policy
            )
            if semantics["legacy_inferred"]:
                legacy_episodes_inferred += 1
                legacy_perturb_events_inferred += int(
                    semantics["perturb_start"] is not None
                )
            explicit_world_target = data_contract in PEER_SYNC_H15_CONTRACTS
            last_start = T - 1 - (
                (SEQUENCE_LENGTH - 1) * planning_stride
                + (horizon if explicit_world_target else horizon - 1)
            )
            if last_start < 0:
                continue
            for s in range(0, last_start + 1, planning_stride):
                target_idx = np.asarray([
                    [s + t * planning_stride + h for h in range(horizon)]
                    for t in range(SEQUENCE_LENGTH)
                ], dtype=np.int64)
                action_valid_mask = semantics["valid"][target_idx]
                decision_idx = s + np.arange(SEQUENCE_LENGTH) * planning_stride
                endpoint_idx = decision_idx + horizon
                # A pre-perturb observation cannot predict which random
                # perturb/recovery branch will occur later in its H-step target.
                # Keep recovery supervision once the perturb is observable.
                if semantics["perturb_start"] is not None:
                    unseen_recovery = (
                        semantics["recovery"][target_idx]
                        & (decision_idx[:, None] < semantics["perturb_start"])
                    )
                    action_valid_mask &= ~unseen_recovery
                if not bool(action_valid_mask.any()):
                    dropped_empty += 1
                    continue
                acts = np.asarray(actions)[target_idx]
                prev = np.stack([
                    np.zeros(4, dtype=np.float32)
                    if s + t * planning_stride == 0
                    else actions[s + t * planning_stride - 1]
                    for t in range(SEQUENCE_LENGTH)
                ])
                proprio = np.stack([
                    states[s + t * planning_stride] for t in range(SEQUENCE_LENGTH)
                ])
                # 帧索引：每个决策点的 4 帧窗口（编码阶段取帧）。
                # 存纯 list（Codex P0-4：numpy 使 weights_only=True 加载失败）。
                frame_idx = np.stack([
                    clip_frame_indices(s + t * planning_stride)
                    for t in range(SEQUENCE_LENGTH)
                ])  # [T, W]
                world_target_frame_idx = (
                    endpoint_idx[:, None].tolist()
                    if explicit_world_target else None
                )
                world_target_valid = (
                    action_valid_mask.all(axis=1)
                    & semantics["frame_valid"][endpoint_idx]
                    if explicit_world_target else None
                )
                W.append({
                    "actions": robust(acts, aq01, aq99).astype(np.float32),
                    "prev": robust(prev, aq01, aq99).astype(np.float32),
                    "proprio": robust(proprio, sq01, sq99).astype(np.float32),
                    "task_id": tid,
                    "ep_id": fi * 10000 + ei,
                    # Existing LongTrajFramesDataset resolves this as
                    # data/metaworld_longtraj_{source_key}.pt. A clean source
                    # therefore remains distinct without a loader change.
                    "task_file": source_key,
                    "ep_idx": ei,
                    "frame_idx": frame_idx.tolist(),
                    "world_target_frame_idx": world_target_frame_idx,
                    "world_target_valid": world_target_valid,
                    "action_valid_mask": action_valid_mask,
                    "recovery_mask": semantics["recovery"][target_idx],
                    "decision_recovery": semantics["recovery"][decision_idx],
                    "metric_state": semantics["metric_state"][decision_idx],
                    "metric_state_valid": semantics["metric_valid"][decision_idx],
                    "first_success": semantics["first_success"],
                })
    n = len(W)
    if n == 0:
        raise ValueError("phase1 produced zero windows with valid action supervision")
    print(f"phase1(h={horizon}): {n} windows, tasks={len(set(w['task_id'] for w in W))}")
    instruction_id = torch.tensor([w["task_id"] for w in W], dtype=torch.long)
    output_identity = {
        "contract": data_contract or "language_conditioned_mt50_longtraj_v2",
        "path": str(out_path.expanduser().resolve(strict=False)),
        "shape": {"windows": n, "sequence_length": SEQUENCE_LENGTH,
                  "action_horizon": horizon, "action_dim": 4},
    }
    parent_identity = _file_identity(ref_path)
    source_identities = [_file_identity(path) for path in files]
    payload = {
        "actions": torch.from_numpy(np.stack([w["actions"] for w in W])),
        "previous_action": torch.from_numpy(np.stack([w["prev"] for w in W])),
        "proprio": torch.from_numpy(np.stack([w["proprio"] for w in W])),
        "instruction_id": instruction_id,
        "episode_id": torch.tensor([w["ep_id"] for w in W], dtype=torch.long),
        "pair_id": torch.arange(n, dtype=torch.long),
        "frame_refs": [(w["task_file"], w["ep_idx"], w["frame_idx"]) for w in W],
        "action_valid_mask": torch.from_numpy(
            np.stack([w["action_valid_mask"] for w in W])
        ),
        "recovery_mask": torch.from_numpy(np.stack([w["recovery_mask"] for w in W])),
        "decision_recovery": torch.from_numpy(
            np.stack([w["decision_recovery"] for w in W])
        ),
        "door_metric_state": torch.from_numpy(
            np.stack([w["metric_state"] for w in W]).astype(np.float32)
        ),
        "door_metric_state_valid": torch.from_numpy(
            np.stack([w["metric_state_valid"] for w in W])
        ),
        "first_success": torch.tensor(
            [-1 if w["first_success"] is None else w["first_success"] for w in W],
            dtype=torch.long,
        ),
        # Broadcast the frozen per-task Qwen cache now; no follow-up mutation by
        # add_language_cache_to_longtraj.py is required.
        "language_hidden": task_language_t[instruction_id],
        "language_mask": task_language_mask_t[instruction_id],
        "normalization": norm,
        "metadata": {
            "contract": data_contract or "language_conditioned_mt50_longtraj_v2",
            "contract_version": (
                (
                    PEER_SYNC_H6_P2_VERSION
                    if data_contract == PEER_SYNC_H6_P2_CONTRACT
                    else PEER_SYNC_H6_VERSION
                )
                if data_contract in PEER_SYNC_H6_CONTRACTS
                else 2
            ),
            "tasks": ref["metadata"]["tasks"],
            "fps": FPS,
            "planning_stride": planning_stride,
            "control_stride": planning_stride,
            "sequence_length": SEQUENCE_LENGTH,
            "decision_offsets": [
                t * planning_stride for t in range(SEQUENCE_LENGTH)
            ],
            "action_horizon": horizon,
            "action_label_offsets": list(range(horizon)),
            "action_dim": 4,
            "action_contract": "executed-clip-fullframe",
            "logged_action_chunk": (
                f"full_h{horizon}"
                if data_contract in PEER_SYNC_H6_CONTRACTS else "full_horizon"
            ),
            "parent_identity": parent_identity,
            "source_identities": source_identities,
            "output_identity": output_identity,
            "observation_action_alignment": "frame/state[i] is pre-action; action[i] executed once",
            "action_valid_mask": (
                "[N,T,H], excludes settle, actions after first_success, and recovery "
                "targets not yet observable at the decision"
            ),
            "recovery_mask": "[N,T,H], perturb_start through first_success inclusive",
            "frame_ref_contract": "data/metaworld_longtraj_{frame_refs[i][0]}.pt",
            "source_files": [str(path.resolve()) for path in files],
            "n_source_files": len(files),
            "n_trajectories": episodes_seen,
            "dropped_all_invalid_windows": dropped_empty,
            "legacy_policy": legacy_policy,
            "legacy_episodes_inferred": legacy_episodes_inferred,
            "legacy_perturb_events_inferred": legacy_perturb_events_inferred,
        },
    }
    if data_contract in PEER_SYNC_H15_CONTRACTS:
        payload["world_target_frame_refs"] = [
            (w["task_file"], w["ep_idx"], w["world_target_frame_idx"])
            for w in W
        ]
        payload["world_target_valid_mask"] = torch.from_numpy(
            np.stack([w["world_target_valid"] for w in W])
        )
        payload["metadata"].update(
            {
                "world_target_horizon": horizon,
                "world_target_offsets": [
                    t * planning_stride + horizon
                    for t in range(SEQUENCE_LENGTH)
                ],
                "world_target_frame_ref_contract": (
                    "single_endpoint_frame_after_full_action_chunk_v1"
                ),
            }
        )
    _save_new(payload, out_path, overwrite=overwrite)
    print(f"[out] {out_path}: {n} windows")
    return out_path


def phase2(device: str, horizon: int, *, task: str | None = None,
           windows_path: Path | None = None,
           st_npy_path: Path | None = None,
           st_meta_path: Path | None = None,
           overwrite: bool = False) -> None:
    """按帧索引解压窗口帧 → 冻结原始 V-JEPA 编码 → ST288 memmap。"""
    from prepare_pnpw_features import VJEPA21Backbone
    from va_compound.live_vjepa import _slot_coords

    win_path = Path(windows_path) if windows_path is not None else win_out(horizon, task)
    default_npy, default_meta = st_paths(horizon, task)
    st_npy = Path(st_npy_path) if st_npy_path is not None else default_npy
    st_meta = Path(st_meta_path) if st_meta_path is not None else default_meta
    existing = [str(path) for path in (st_npy, st_meta) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite phase2 outputs: {existing}; choose new paths or pass --overwrite"
        )
    win = torch.load(win_path, map_location="cpu", weights_only=False)
    refs = win["frame_refs"]
    n = len(refs)
    print(f"phase2(h={horizon}): {n} windows, backbone=原始 V-JEPA 2.1（冻结）")

    # 窗口按 (任务, episode) 分组（phase1 按 (file, ep, start) 顺序 append，
    # 同一 (task, ep) 的窗口连续；dict 保持插入序 → 同一任务的组连续）。
    # 注意：必须按任务流式解码→编码→释放，一次性全量解码会吃满内存
    # （50 任务 ≈ 45GB 原图，本机只有 31GB，直接 swap 地狱）；且单任务整
    # 体解码（~5GB）+ memmap 脏页 + 模型也会超 31GB（实测 33GB swap 崩溃），
    # 故按 episode 懒解码：单 ep ~150MB，峰值内存 ~10GB（2026-08-09）。
    groups: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = {}
    for r, (tf, ei, fidx) in enumerate(refs):
        groups.setdefault((tf, ei), []).append((r, fidx))
    print(f"  grouped: {len(groups)} (task,ep) 组, "
          f"max {max(len(v) for v in groups.values())} windows/组", flush=True)

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=torch.float16, max_tokens=144, local_files_only=True
    )
    vision_backbone.eval()
    coords = _slot_coords()

    ST_NPY, ST_META = st_npy, st_meta
    ST_NPY.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(ST_NPY, dtype=np.float16, mode="w+", shape=(n, SEQUENCE_LENGTH, 288, 768))
    print(f"memmap: {ST_NPY} shape={mm.shape}")

    t0 = time.time()
    B = 16
    done = 0
    cur_tf = None
    task_data = None
    for (tf, ei), items in groups.items():
        if tf != cur_tf:
            if cur_tf is not None:
                mm.flush()  # memmap 脏页写回，释放页缓存（防 31GB 内存爆）
                torch.cuda.empty_cache()
            task_data = torch.load(ROOT / "data" / f"metaworld_longtraj_{tf}.pt",
                                   map_location="cpu", weights_only=False)
            nf = sum(len(ep["frames"]) for ep in task_data["episodes"])
            print(f"  loaded {tf}: {len(task_data['episodes'])} eps, {nf} frames",
                  flush=True)
            cur_tf = tf
        # episode 级懒解码（单 ep ~150MB；整任务解码 ~5GB 会超 31GB 内存）
        ep_frames = [
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
            for b in task_data["episodes"][ei]["frames"]
        ]
        for start in range(0, len(items), B):
            rows = items[start:start + B]
            clips = []  # 每窗口 [T, W, 384, 384, 3]
            for r, fidx in rows:
                fidx = np.asarray(fidx)  # frame_refs 可能是纯 list（weights_only 兼容转换后）
                T, W = fidx.shape
                clip = np.stack([
                    np.stack([ep_frames[int(fidx[t, w])] for w in range(W)])
                    for t in range(T)
                ])  # [T, W, 384, 384, 3]
                clips.append(clip)
            frames_batch = np.stack(clips)  # [B, T, W, 384, 384, 3]
            with torch.inference_mode():
                # 不用 encode_live_frames：其 preprocess_batch 在 CPU 跑
                # bicubic+antialias 且 list() 逐元素转换（1.13 亿 Python 对象/批，
                # GPU 长期空闲，实测单批 >20s）。帧已 384×384（解码时 resize），
                # 直接在 GPU 归一化 → V-JEPA 前向（2026-08-09 卡死修复）。
                b, t, w, hh, ww, _ = frames_batch.shape
                frames = np.ascontiguousarray(
                    frames_batch.reshape(b * t * w, hh, ww, 3)
                )
                video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                video = video.div_(255.0).to(torch.device(device))
                if video.shape[-1] != 384 or video.shape[-2] != 384:
                    video = F.interpolate(
                        video, size=(384, 384), mode="bicubic",
                        align_corners=False, antialias=True,
                    )
                mean = torch.tensor(
                    (0.485, 0.456, 0.406), device=video.device
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    (0.229, 0.224, 0.225), device=video.device
                ).view(1, 3, 1, 1)
                inputs = ((video - mean) / std).reshape(b * t, w, 3, 384, 384)
                raw = vision_backbone._encode(inputs)  # [B*T, grid_tokens, D]
                st = vision_backbone._pool(raw, "spatiotemporal")  # [B*T, 288, D]
                st = st.reshape(b, t, 288, -1)  # [B, T, 288, 768]
            for k, (r, _) in enumerate(rows):
                mm[r] = st[k].cpu().numpy()
            done += len(rows)
            if (start // B) % 50 == 0:
                el = time.time() - t0
                print(f"  {tf}:{ei} {start}/{len(items)} "
                      f"(total {done}/{n}, {el:.0f}s)", flush=True)
        del ep_frames
        gc.collect()
    mm.flush()
    # meta.pt（与 mw_local288.pt / extract_st288_finetuned 同契约：coords 必须
    # 是 torch Tensor（weights_only=True 加载），load_st288_memmap 裸数据分支
    # 按 metadata.rows/tokens_per_decision 推断 shape）
    meta = {
        "vision_tokens_st_npy": str(ST_NPY),
        "coords": torch.from_numpy(coords),  # torch Tensor [288,3]，与 mw_local288 一致
        "metadata": {
            "source": f"longtraj-scripted-expert:h{horizon}",
            "rows": n,
            "tokens_per_decision": 288,
            "grid": [24, 24],
            "slot_grid": 12,
            "pooling": "spatiotemporal",
            "action_horizon": horizon,
        },
    }
    torch.save(meta, ST_META)
    print(f"[out] {ST_META}: rows={n} tok=288（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", choices=("1", "2"), default="1")
    ap.add_argument("--horizon", type=int, default=8,
                    help="action chunk 长度（E7 用 48；文件名/特征路径按 horizon 区分）")
    ap.add_argument("--task", choices=tuple(ENV_TO_TASK),
                    help="仅构建一个任务；默认输出任务后缀文件，不覆盖全任务文件")
    ap.add_argument("--input", type=Path, action="append",
                    help="phase1 精确源文件，可重复；clean door-lock 应显式指定")
    ap.add_argument(
        "--ref",
        type=Path,
        default=REF,
        help="phase1 的 normalization + per-task language cache 来源；可直接使用"
        "现有 windows_h48.pt，避免再次加载更大的 fullframe reference",
    )
    ap.add_argument("--output", type=Path, help="phase1 输出 windows 文件")
    ap.add_argument("--windows", type=Path, help="phase2 输入 windows 文件")
    ap.add_argument("--st-npy", type=Path, help="phase2 特征 memmap 输出")
    ap.add_argument("--st-meta", type=Path, help="phase2 metadata 输出")
    ap.add_argument(
        "--data-contract", choices=tuple(sorted(PEER_SYNC_H6_CONTRACTS)),
        help=("emit an explicit peer protocol; cadence-specific contracts require "
              "their matching --planning-stride"),
    )
    ap.add_argument(
        "--planning-stride", type=int, default=CONTROL_STRIDE,
        help=("decision/control cadence in source frames; default 6 preserves old "
              "datasets; P2/P15 contracts require stride 2/15 respectively"),
    )
    ap.add_argument(
        "--legacy-policy", choices=("warn", "error", "infer"), default="warn",
        help=("旧数据策略：warn 保持兼容性警告；error 拒绝；infer 严格识别旧采集器"
              "唯一的12步零动作扰动块并修正 success/mask（歧义时失败）"),
    )
    ap.add_argument("--overwrite", action="store_true",
                    help="显式允许覆盖输出（默认拒绝）")
    args = ap.parse_args()
    if args.phase == "1":
        phase1(
            args.horizon,
            task=args.task,
            input_paths=args.input,
            output_path=args.output,
            ref_path=args.ref,
            legacy_policy=args.legacy_policy,
            data_contract=args.data_contract,
            planning_stride=args.planning_stride,
            overwrite=args.overwrite,
        )
    else:
        phase2(
            args.device,
            args.horizon,
            task=args.task,
            windows_path=args.windows,
            st_npy_path=args.st_npy,
            st_meta_path=args.st_meta,
            overwrite=args.overwrite,
        )
