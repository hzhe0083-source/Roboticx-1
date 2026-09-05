"""LIBERO data contracts, constants, profiles, and dataset helpers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from va_compound.longtraj_frames import LongTrajFramesDataset


ACTION_HORIZON = 50
EXECUTION_HORIZON = 15
WORLD_HORIZON = EXECUTION_HORIZON
SEQUENCE = 8
DECISION_OFFSETS = np.arange(SEQUENCE, dtype=np.int64) * EXECUTION_HORIZON
VISION_OFFSETS = np.array((-6, -4, -2, 0), dtype=np.int64)
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SOURCE_DATA_CONTRACT = "libero_4suite_h50p15_t4_dualview5_worldh15_va1024_qwen08_last6_denseall_v7"
SOURCE_INITIALIZATION = "dense_all_windows_continue_from_s5000_v1"
DATA_CONTRACT = "libero_4suite_h50p15_t8_dualview5_worldh15_va1024_qwen08_last6_denseall_v8"
JOINT_DATA_CONTRACT = "libero_joint_episode_t8_p15_state_delta_v10"
H15_DATA_CONTRACT = "libero_joint_episode_t8_h15_state_delta_v11"
ALL_STARTS_DATA_CONTRACT = "libero_joint_all_starts_t8_h15_state_delta_v12"
FRESH_INITIALIZATION = "t8_dense_continue_from_t4_s1000_v1"
FUSION_LAYERS = list(range(18, 24))
CROSS_MODAL_VA_LAYERS = list(range(len(FUSION_LAYERS)))


RUN_SCHEDULE_PROFILES = {
    int(task_count): profile
    for task_count, profile in json.loads(
        (Path(__file__).resolve().parents[2] / "configs/libero/run_schedules.json").read_text()
    ).items()
}


class _StaticAnchorDataset:
    """Resolve the mixed sampler's negative anchor ids for static payloads."""

    def __init__(self, dataset: LongTrajFramesDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return self.dataset[-index - 1 if index < 0 else index]


def _normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scale = np.where(np.abs(high - low) < 1e-6, 1.0, high - low)
    return np.clip(2.0 * (values - low) / scale - 1.0, -1.0, 1.0).astype(np.float32)


def _suite_names(raw: str) -> tuple[str, ...]:
    suites = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not suites or any(suite not in LIBERO_SUITES for suite in suites):
        raise ValueError(f"--suites must be drawn from {LIBERO_SUITES}")
    if len(set(suites)) != len(suites):
        raise ValueError("--suites contains duplicates")
    return suites


def _local_task_ids(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    task_ids = tuple(sorted(int(part.strip()) for part in raw.split(",") if part.strip()))
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(i not in range(10) for i in task_ids):
        raise ValueError("--local-task-ids must contain unique ids in 0..9")
    return task_ids


def _official_task_specs(
    suites: tuple[str, ...], local_task_ids: tuple[int, ...] | None = None
) -> list[dict]:
    from libero.libero import benchmark

    if local_task_ids is not None and len(suites) != 1:
        raise ValueError("--local-task-ids requires exactly one suite")
    registry = benchmark.get_benchmark_dict()
    specs = []
    for suite in suites:
        task_suite = registry[suite]()
        if task_suite.n_tasks != 10:
            raise ValueError(f"{suite} has {task_suite.n_tasks} tasks, expected 10")
        selected = local_task_ids if local_task_ids is not None else range(task_suite.n_tasks)
        for local_task_id in selected:
            task = task_suite.get_task(local_task_id)
            specs.append(
                {
                    "task_id": len(specs),
                    "suite": suite,
                    "local_task_id": local_task_id,
                    "description": task.language.strip(),
                }
            )
    if len({spec["description"] for spec in specs}) != len(specs):
        raise ValueError("LIBERO task instructions must be unique")
    return specs


def _attach_dense_world_action_donors(payload: dict) -> None:
    """Use the next episode at matching relative progress as an O(N) donor."""
    actions = payload["actions"]
    tasks = payload["instruction_id"]
    episodes = payload["episode_id"]
    starts = payload["crop_start"]
    table = actions.new_zeros((*actions.shape[:2], WORLD_HORIZON, actions.shape[-1]))
    mask = torch.zeros(actions.shape[:2], dtype=torch.bool)
    for task in torch.unique(tasks, sorted=True):
        task_rows = torch.where(tasks == task)[0]
        task_episodes = torch.unique(episodes[task_rows], sorted=True)
        for position, episode in enumerate(task_episodes):
            own = torch.where(episodes == episode)[0]
            donor_episode = task_episodes[(position + 1) % len(task_episodes)]
            donor = torch.where(episodes == donor_episode)[0]
            own_starts, donor_starts = starts[own], starts[donor]
            if not torch.equal(own_starts, torch.arange(len(own))) or not torch.equal(
                donor_starts, torch.arange(len(donor))
            ):
                raise ValueError("dense window rows must contain every ordered start")
            donor_positions = torch.round(
                own_starts.float()
                * max(len(donor) - 1, 0)
                / max(len(own) - 1, 1)
            ).long()
            candidate = actions[donor[donor_positions], :, :WORLD_HORIZON]
            table[own] = candidate
            mask[own] = candidate.ne(actions[own, :, :WORLD_HORIZON]).any(dim=(-1, -2))
    payload["world_rank_shuffle_action"] = table
    payload["world_rank_shuffle_mask"] = mask
    payload["metadata"]["world_action_donor_contract"] = (
        "task_next_episode_relative_progress_v1"
    )


def _attach_joint_world_action_donors(payload: dict) -> None:
    """Use same task next episode at relative decision progress, valid donors only."""
    actions = payload["actions"]
    tasks = payload["instruction_id"]
    episodes = payload["episode_id"]
    decision_counts = payload["decision_count"]

    table = actions.new_zeros((*actions.shape[:2], WORLD_HORIZON, actions.shape[-1]))
    mask = torch.zeros(actions.shape[:2], dtype=torch.bool)

    for task in torch.unique(tasks, sorted=True):
        task_rows = torch.where(tasks == task)[0]
        task_episodes = torch.unique(episodes[task_rows], sorted=True)
        for position, episode in enumerate(task_episodes):
            own_rows = torch.where(episodes == episode)[0]
            donor_episode = task_episodes[(position + 1) % len(task_episodes)]
            donor_rows = torch.where(episodes == donor_episode)[0]

            donor_decisions: list[tuple[int, int]] = []
            for d_row in donor_rows.tolist():
                d_count = int(decision_counts[d_row].item())
                for dt in range(d_count):
                    donor_decisions.append((d_row, dt))

            d_donor = len(donor_decisions)
            d_own = int(decision_counts[own_rows].sum().item())

            k = 0
            for row in own_rows.tolist():
                count = int(decision_counts[row].item())
                for t in range(count):
                    donor_idx = int(round(k * max(d_donor - 1, 0) / max(d_own - 1, 1)))
                    donor_idx = min(max(donor_idx, 0), d_donor - 1)
                    donor_row, donor_t = donor_decisions[donor_idx]
                    candidate = actions[donor_row, donor_t, :WORLD_HORIZON]
                    table[row, t] = candidate
                    mask[row, t] = candidate.ne(actions[row, t, :WORLD_HORIZON]).any()
                    k += 1

    payload["world_rank_shuffle_action"] = table
    payload["world_rank_shuffle_mask"] = mask
    payload["metadata"]["world_action_donor_contract"] = (
        "task_next_episode_relative_progress_v1"
    )


def _attach_all_starts_world_action_donors(payload: dict) -> None:
    """Attach next-episode relative progress donors for compact all-starts payload [N, 1, 15, 7]."""
    actions = payload["actions"]
    tasks = payload["instruction_id"]
    episodes = payload["episode_id"]

    table = actions.new_zeros(actions.shape)
    mask = torch.zeros(actions.shape[:2], dtype=torch.bool)

    for task in torch.unique(tasks, sorted=True):
        task_rows = torch.where(tasks == task)[0]
        task_episodes = torch.unique(episodes[task_rows], sorted=True)
        for position, episode in enumerate(task_episodes):
            own_rows = torch.where(episodes == episode)[0]
            donor_episode = task_episodes[(position + 1) % len(task_episodes)]
            donor_rows = torch.where(episodes == donor_episode)[0]

            d_own = len(own_rows)
            d_donor = len(donor_rows)
            for t in range(d_own):
                donor_idx = int(round(t * max(d_donor - 1, 0) / max(d_own - 1, 1)))
                donor_idx = min(max(donor_idx, 0), d_donor - 1)
                candidate = actions[donor_rows[donor_idx]]
                table[own_rows[t]] = candidate
                mask[own_rows[t]] = candidate.ne(actions[own_rows[t]]).any(dim=-1).any(dim=-1)

    payload["world_rank_shuffle_action"] = table
    payload["world_rank_shuffle_mask"] = mask
    payload["metadata"]["world_action_donor_contract"] = (
        "task_next_episode_relative_progress_v1"
    )


def _validate_all_starts_data(payload: dict, path: Path) -> dict:
    metadata = payload.get("metadata", {})
    if metadata.get("contract") != ALL_STARTS_DATA_CONTRACT:
        raise ValueError(f"unexpected data contract in {path}: expected {ALL_STARTS_DATA_CONTRACT}")
    tasks = list(metadata.get("tasks") or [])
    specs = list(metadata.get("task_specs") or [])
    n_tasks = int(metadata.get("n_tasks", 0))
    n_demos = int(metadata.get("n_demos", 0))
    if (
        n_tasks < 1
        or len(tasks) != n_tasks
        or len(specs) != n_tasks
        or [spec.get("task_id") for spec in specs] != list(range(n_tasks))
    ):
        raise ValueError("LIBERO task metadata and task specs are inconsistent")

    episode_lengths = metadata.get("episode_lengths")
    if (
        not isinstance(episode_lengths, list)
        or len(episode_lengths) != n_demos
        or any(not isinstance(L, int) or L <= EXECUTION_HORIZON for L in episode_lengths)
    ):
        raise ValueError("metadata episode_lengths must be a list of demo lengths > 15")

    expected_metadata = {
        "contract": ALL_STARTS_DATA_CONTRACT,
        "sampling_contract": "all_starts_random_tbptt8_v1",
        "window_sampling": "all_starts_random_tbptt8_v1",
        "storage_sequence_length": 1,
        "sequence_length": SEQUENCE,
        "action_horizon": EXECUTION_HORIZON,
        "planning_stride": EXECUTION_HORIZON,
        "control_stride": EXECUTION_HORIZON,
        "decision_offsets": [0],
        "vision_offsets": VISION_OFFSETS.tolist(),
        "vision_input": "agentview_history4_plus_current_wrist_v2",
        "world_target_view": "eye_in_hand_rgb",
        "world_target_horizon": WORLD_HORIZON,
        "world_target_offsets": [WORLD_HORIZON],
        "world_target_alignment": f"obs[d+{WORLD_HORIZON}]",
        "target_alignment": f"obs[d]_to_actions[d+1:d+{EXECUTION_HORIZON + 1}]",
        "logged_action_chunk": "real_p15",
        "language_source": "online_qwen35_0_8b_last6_full_v1",
        "language_dim": 1024,
        "state_delta_contract": "joint7_gripper2_unclipped_q01q99_delta_h15_v1",
        "memory_contract": "offset_replay_tbptt8_v1",
        "world_action_donor_contract": "task_next_episode_relative_progress_v1",
    }
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise ValueError(f"LIBERO all-starts metadata mismatch: {mismatch}")

    actions = payload.get("actions")
    if not isinstance(actions, Tensor) or tuple(actions.shape[1:]) != (1, EXECUTION_HORIZON, 7):
        raise ValueError("LIBERO compact data actions must be [N, 1, 15, 7]")
    if not torch.isfinite(actions).all():
        raise ValueError("LIBERO actions must be finite")
    N = len(actions)

    instruction_ids = payload.get("instruction_id")
    if (
        not isinstance(instruction_ids, Tensor)
        or tuple(instruction_ids.shape) != (N,)
        or instruction_ids.dtype != torch.long
        or sorted(torch.unique(instruction_ids).tolist()) != list(range(n_tasks))
    ):
        raise ValueError("LIBERO task metadata and instruction ids are inconsistent")

    proprio = payload.get("proprio")
    if not isinstance(proprio, Tensor) or tuple(proprio.shape) != (N, 1, 9):
        raise ValueError("LIBERO compact data proprio must be [N, 1, 9]")
    if not torch.isfinite(proprio).all():
        raise ValueError("LIBERO proprio must be finite")

    previous = payload.get("previous_action")
    if not isinstance(previous, Tensor) or tuple(previous.shape) != (N, 1, 7):
        raise ValueError("LIBERO compact data previous_action must be [N, 1, 7]")
    if not torch.isfinite(previous).all():
        raise ValueError("LIBERO previous_action must be finite")

    delta = payload.get("world_state_delta")
    if not isinstance(delta, Tensor) or tuple(delta.shape) != (N, 1, 9) or delta.dtype != torch.float32:
        raise ValueError("LIBERO compact data world_state_delta must be [N, 1, 9] float32")
    if not torch.isfinite(delta).all():
        raise ValueError("LIBERO world_state_delta must be finite")

    valid = payload.get("action_valid_mask")
    if not isinstance(valid, Tensor) or tuple(valid.shape) != (N, 1, EXECUTION_HORIZON) or valid.dtype != torch.bool:
        raise ValueError("LIBERO compact data action_valid_mask must be [N, 1, 15] bool")
    if not bool(valid.all()):
        raise ValueError("all compact decisions must have full action_valid_mask")

    world_valid = payload.get("world_target_valid_mask")
    if not isinstance(world_valid, Tensor) or tuple(world_valid.shape) != (N, 1) or world_valid.dtype != torch.bool:
        raise ValueError("LIBERO compact data world_target_valid_mask must be [N, 1] bool")
    if not bool(world_valid.all()):
        raise ValueError("all compact decisions must have full world_target_valid_mask")

    donor_actions = payload.get("world_rank_shuffle_action")
    if not isinstance(donor_actions, Tensor) or tuple(donor_actions.shape) != (N, 1, EXECUTION_HORIZON, 7):
        raise ValueError("LIBERO compact data world_rank_shuffle_action must be [N, 1, 15, 7]")
    if not torch.isfinite(donor_actions).all():
        raise ValueError("LIBERO donor actions must be finite")

    donor_mask = payload.get("world_rank_shuffle_mask")
    if not isinstance(donor_mask, Tensor) or tuple(donor_mask.shape) != (N, 1) or donor_mask.dtype != torch.bool:
        raise ValueError("LIBERO compact data world_rank_shuffle_mask must be [N, 1] bool")

    crop_start = payload.get("crop_start")
    if not isinstance(crop_start, Tensor) or tuple(crop_start.shape) != (N,) or crop_start.dtype != torch.long:
        raise ValueError("LIBERO compact data crop_start must be [N] int64")

    episode_ids = payload.get("episode_id")
    if not isinstance(episode_ids, Tensor) or tuple(episode_ids.shape) != (N,) or episode_ids.dtype != torch.long:
        raise ValueError("LIBERO compact data episode_id must be [N] int64")

    pair_id = payload.get("pair_id")
    if not isinstance(pair_id, Tensor) or tuple(pair_id.shape) != (N,) or pair_id.dtype != torch.long:
        raise ValueError("LIBERO compact data pair_id must be [N] int64")

    anchor_eligible = payload.get("anchor_eligible")
    if not isinstance(anchor_eligible, Tensor) or tuple(anchor_eligible.shape) != (N,) or anchor_eligible.dtype != torch.bool:
        raise ValueError("LIBERO compact data anchor_eligible must be [N] bool")

    language_hidden = payload.get("language_hidden")
    if not isinstance(language_hidden, Tensor) or tuple(language_hidden.shape) != (N, 1, 1) or language_hidden.dtype != torch.float16:
        raise ValueError("LIBERO compact data language_hidden must be [N, 1, 1] float16")

    language_mask = payload.get("language_mask")
    if not isinstance(language_mask, Tensor) or tuple(language_mask.shape) != (N, 1) or language_mask.dtype != torch.bool:
        raise ValueError("LIBERO compact data language_mask must be [N, 1] bool")

    normalization = payload.get("normalization", {})
    state_q01 = normalization.get("state_q01")
    state_q99 = normalization.get("state_q99")
    if (
        not isinstance(state_q01, Tensor)
        or tuple(state_q01.shape) != (9,)
        or not torch.isfinite(state_q01).all()
        or not isinstance(state_q99, Tensor)
        or tuple(state_q99.shape) != (9,)
        or not torch.isfinite(state_q99).all()
    ):
        raise ValueError("normalization must include finite state_q01 and state_q99 of shape (9,)")

    scale = normalization.get("state_delta_scale")
    if not isinstance(scale, Tensor) or tuple(scale.shape) != (9,) or not torch.isfinite(scale).all() or not bool((scale > 0).all()):
        raise ValueError("normalization must include positive state_delta_scale of shape (9,)")
    scale_expected = state_q99 - state_q01
    scale_expected = torch.where(scale_expected.abs() < 1e-6, torch.ones_like(scale_expected), scale_expected)
    if not torch.equal(scale, scale_expected):
        raise ValueError("state_delta_scale differs from state quantiles")

    frame_refs = payload.get("frame_refs")
    target_refs = payload.get("world_target_frame_refs")
    if not isinstance(frame_refs, list) or len(frame_refs) != N:
        raise ValueError("LIBERO compact data frame_refs must be list of length N")
    if not isinstance(target_refs, list) or len(target_refs) != N:
        raise ValueError("LIBERO compact data world_target_frame_refs must be list of length N")

    expected_total = sum(L - EXECUTION_HORIZON for L in episode_lengths)
    if N != expected_total:
        raise ValueError(f"total rows {N} does not match expected decisions {expected_total} from episode_lengths")

    unique_episodes = torch.unique(episode_ids, sorted=True).tolist()
    if unique_episodes != list(range(n_demos)):
        raise ValueError(f"episode_ids must cover 0..{n_demos - 1} exactly")

    for ep in range(n_demos):
        ep_rows = torch.where(episode_ids == ep)[0]
        if not torch.equal(ep_rows, torch.arange(ep_rows[0].item(), ep_rows[0].item() + len(ep_rows))):
            raise ValueError(f"episode {ep} rows are not contiguous")
        if len(torch.unique(instruction_ids[ep_rows])) != 1:
            raise ValueError(f"episode {ep} crosses tasks")
        L = episode_lengths[ep]
        expected_d = L - EXECUTION_HORIZON
        if len(ep_rows) != expected_d:
            raise ValueError(f"episode {ep} row count {len(ep_rows)} does not match L-15={expected_d}")
        if not torch.equal(crop_start[ep_rows], torch.arange(expected_d, dtype=torch.long)):
            raise ValueError(f"episode {ep} crop_starts must be contiguous 0..{expected_d - 1}")

        if len(ep_rows) > 1:
            if not torch.equal(previous[ep_rows[1:], 0], actions[ep_rows[:-1], 0, 0]):
                raise ValueError(f"episode {ep} previous_action continuity broken")
            if not torch.equal(actions[ep_rows[:-1], 0, 1:], actions[ep_rows[1:], 0, :-1]):
                raise ValueError(f"episode {ep} inner action target continuity broken")

        ep_task_key, ep_demo_idx, _ = frame_refs[int(ep_rows[0])]
        for row in ep_rows.tolist():
            c_key, c_demo, c_frames = frame_refs[row]
            t_key, t_demo, t_frames = target_refs[row]
            if (c_key, c_demo) != (ep_task_key, ep_demo_idx) or (t_key, t_demo) != (ep_task_key, ep_demo_idx):
                raise ValueError(f"episode {ep} frame references cross demos")
            if len(c_frames) != 1 or len(c_frames[0]) != 5 or len(t_frames) != 1 or len(t_frames[0]) != 1:
                raise ValueError("compact frame_refs must have shape [1, 5] and target [1, 1]")
            d = int(crop_start[row].item())
            if c_frames[0][:4] != np.maximum(d + VISION_OFFSETS, 0).tolist():
                raise ValueError(f"episode {ep} agentview frame refs mismatch at d={d}")
            if c_frames[0][4] - d != L:
                raise ValueError(f"episode {ep} wrist base does not match demo length {L} at d={d}")
            if t_frames[0] != [L + d + WORLD_HORIZON]:
                raise ValueError(f"episode {ep} future wrist target mismatch at d={d}")

    return payload


def _validate_data(
    path: Path,
    *,
    architecture_version: str = "legacy",
    window_sampling: str | None = None,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    metadata = payload.get("metadata", {})
    if window_sampling is not None and metadata.get("window_sampling") != window_sampling:
        raise ValueError(
            f"window_sampling mismatch in {path}: expected {window_sampling}, got {metadata.get('window_sampling')}"
        )
    contract = metadata.get("contract")
    if contract == ALL_STARTS_DATA_CONTRACT:
        if architecture_version != "dual_tower_h15_v1":
            raise ValueError(f"unexpected data contract in {path}")
        return _validate_all_starts_data(payload, path)

    is_h15 = architecture_version == "dual_tower_h15_v1"
    joint = architecture_version in ("dual_tower_expert_v1", "dual_tower_h15_v1")
    expected_contract = (
        H15_DATA_CONTRACT
        if is_h15
        else (JOINT_DATA_CONTRACT if joint else DATA_CONTRACT)
    )
    if contract != expected_contract:
        raise ValueError(f"unexpected data contract in {path}")
    tasks = list(metadata.get("tasks") or [])
    specs = list(metadata.get("task_specs") or [])
    n_tasks = int(metadata.get("n_tasks", 0))
    instruction_ids = payload.get("instruction_id")
    if (
        n_tasks < 1
        or len(tasks) != n_tasks
        or len(specs) != n_tasks
        or not isinstance(instruction_ids, Tensor)
        or sorted(torch.unique(instruction_ids).tolist()) != list(range(n_tasks))
    ):
        raise ValueError("LIBERO task metadata and instruction ids are inconsistent")
    actions = payload.get("actions")
    local_action_horizon = EXECUTION_HORIZON if is_h15 else ACTION_HORIZON
    if not isinstance(actions, Tensor) or tuple(actions.shape[1:]) != (SEQUENCE, local_action_horizon, 7):
        raise ValueError(f"LIBERO training data must be T8/H{local_action_horizon}/A7")
    if not torch.isfinite(actions).all():
        raise ValueError("LIBERO actions must be finite")
    proprio = payload.get("proprio")
    if not isinstance(proprio, Tensor) or tuple(proprio.shape[1:]) != (SEQUENCE, 9):
        raise ValueError("LIBERO training data must contain T8/D9 proprio")
    if not torch.isfinite(proprio).all():
        raise ValueError("LIBERO proprio must be finite")
    previous = payload.get("previous_action")
    if not isinstance(previous, Tensor) or tuple(previous.shape[1:]) != (SEQUENCE, 7):
        raise ValueError("LIBERO training data must contain T8/A7 previous_action")
    if not torch.isfinite(previous).all():
        raise ValueError("LIBERO previous_action must be finite")

    expected_metadata = {
        "window_sampling": "episode_contiguous_p15_v1" if joint else "all_legal_starts_v1",
        "planning_stride": EXECUTION_HORIZON,
        "control_stride": EXECUTION_HORIZON,
        "decision_offsets": DECISION_OFFSETS.tolist(),
        "world_target_horizon": WORLD_HORIZON,
        "world_target_offsets": (
            DECISION_OFFSETS + WORLD_HORIZON
        ).tolist(),
        "world_target_alignment": f"obs[d+{WORLD_HORIZON}]",
        "language_source": "online_qwen35_0_8b_last6_full_v1",
        "language_dim": 1024,
        "previous_action_model_input": "zero_v1",
        "vision_input": "agentview_history4_plus_current_wrist_v2",
        "world_target_view": "eye_in_hand_rgb",
        "short_horizon_padding": "episode_storage_only_v1" if is_h15 else "repeat_last_masked_v1",
        "minimum_real_action_prefix": EXECUTION_HORIZON,
        "action_horizon": local_action_horizon,
        "logged_action_chunk": "real_p15" if is_h15 else "masked_h50_real_p15_prefix",
        "target_alignment": "obs[d]_to_actions[d+1:d+16]" if is_h15 else "obs[d]_to_actions[d+1:d+51]_masked_tail",
    }
    if joint:
        expected_metadata["window_bound"] = "complete_p15_v1" if is_h15 else "complete_p15_masked_h50_v1"
        expected_metadata["state_delta_contract"] = "joint7_gripper2_unclipped_q01q99_delta_h15_v1"
        expected_metadata["memory_contract"] = "episode_tbptt8_v1"
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise ValueError(f"LIBERO H{local_action_horizon}/P15 metadata mismatch: {mismatch}")
    valid = payload.get("action_valid_mask")
    if not isinstance(valid, Tensor) or valid.shape != actions.shape[:-1]:
        raise ValueError("LIBERO H50 data requires an aligned action_valid_mask")
    world_valid = payload.get("world_target_valid_mask")
    if not isinstance(world_valid, Tensor) or world_valid.shape != valid.shape[:2]:
        raise ValueError("LIBERO H50 World targets require an aligned [N,T] mask")
    crop_start = payload.get("crop_start")
    if not isinstance(crop_start, Tensor) or crop_start.shape != instruction_ids.shape:
        raise ValueError("dense LIBERO data requires one crop_start per row")

    if joint:
        if valid.dtype != torch.bool or world_valid.dtype != torch.bool:
            raise ValueError("joint validity masks must be boolean")
        decision_valid = payload.get("decision_valid_mask")
        if not isinstance(decision_valid, Tensor) or decision_valid.shape != (len(actions), SEQUENCE):
            raise ValueError("joint data requires decision_valid_mask of shape [N, 8]")
        if decision_valid.dtype != torch.bool:
            raise ValueError("decision_valid_mask must be boolean")
        if bool((decision_valid[:, 1:] & ~decision_valid[:, :-1]).any()):
            raise ValueError("decision_valid_mask must be a true prefix mask")

        decision_count = payload.get("decision_count")
        if not isinstance(decision_count, Tensor) or decision_count.shape != (len(actions),):
            raise ValueError("joint data requires decision_count of shape [N]")
        if bool((decision_count < 1).any()) or bool((decision_count > SEQUENCE).any()):
            raise ValueError("decision_count must be in 1..8")
        if decision_count.dtype != torch.long or crop_start.dtype != torch.long:
            raise ValueError("joint decision counts and crop starts must be int64")
        if not torch.equal(decision_count, decision_valid.sum(dim=-1)):
            raise ValueError("decision_count must match decision_valid_mask count")

        episode_start = payload.get("episode_start")
        episode_end = payload.get("episode_end")
        if not isinstance(episode_start, Tensor) or episode_start.shape != (len(actions),) or episode_start.dtype != torch.bool:
            raise ValueError("joint data requires boolean episode_start [N]")
        if not isinstance(episode_end, Tensor) or episode_end.shape != (len(actions),) or episode_end.dtype != torch.bool:
            raise ValueError("joint data requires boolean episode_end [N]")

        # real P15 only valid decisions, padded action_valid_mask all false
        valid_act = valid[decision_valid]
        if bool((valid_act[:, 1:] & ~valid_act[:, :-1]).any()):
            raise ValueError("action_valid_mask must be a true prefix on valid decisions")
        if not bool(valid_act[:, :EXECUTION_HORIZON].all()):
            raise ValueError("every valid joint decision requires a real P15 prefix")
        if bool(valid[~decision_valid].any()):
            raise ValueError("padded decisions must have all-false action_valid_mask")

        # all world valid == decision_valid
        if not torch.equal(world_valid, decision_valid):
            raise ValueError("world_target_valid_mask must equal decision_valid_mask for joint data")

        # delta validation
        delta = payload.get("world_state_delta")
        if not isinstance(delta, Tensor) or tuple(delta.shape) != (len(actions), SEQUENCE, 9) or delta.dtype != torch.float32:
            raise ValueError("joint data requires world_state_delta of shape [N, 8, 9] float32")
        if not torch.isfinite(delta).all():
            raise ValueError("world_state_delta must be finite")
        if bool(delta[~decision_valid].any()):
            raise ValueError("padded decisions must have zero world_state_delta")

        # state_delta_scale validation
        normalization = payload.get("normalization", {})
        scale = normalization.get("state_delta_scale")
        if not isinstance(scale, Tensor) or tuple(scale.shape) != (9,) or not torch.isfinite(scale).all() or not bool((scale > 0).all()):
            raise ValueError("joint normalization must include positive state_delta_scale of shape (9,)")

        # episode continuity: next crop_start = prior crop_start + 15 * decision_count, start/end correct, no duplicate starts
        episode_ids = payload.get("episode_id")
        if not isinstance(episode_ids, Tensor) or episode_ids.shape != (len(actions),):
            raise ValueError("joint data requires episode_id of shape [N]")
        if episode_ids.dtype != torch.long:
            raise ValueError("joint episode ids must be int64")
        scale_expected = normalization["state_q99"] - normalization["state_q01"]
        scale_expected = torch.where(scale_expected.abs() < 1e-6, torch.ones_like(scale_expected), scale_expected)
        if not torch.equal(scale, scale_expected):
            raise ValueError("state_delta_scale differs from state quantiles")
        if any(len(payload[key]) != len(actions) for key in ("proprio", "previous_action", "instruction_id", "frame_refs", "world_target_frame_refs")):
            raise ValueError("joint payload rows must align")
        for ep in torch.unique(episode_ids, sorted=True):
            ep_rows = torch.where(episode_ids == ep)[0]
            if len(torch.unique(instruction_ids[ep_rows])) != 1:
                raise ValueError("episode cannot cross tasks")
            if not bool((decision_count[ep_rows[:-1]] == SEQUENCE).all()):
                raise ValueError("nonfinal episode windows must contain T8 decisions")
            if not torch.equal(previous[ep_rows[1:], 0], actions[ep_rows[:-1], SEQUENCE - 1, EXECUTION_HORIZON - 1]):
                raise ValueError("previous_action must continue across episode windows")
            if not bool(episode_start[ep_rows[0]]) or bool(episode_start[ep_rows[1:]].any()):
                raise ValueError(f"episode {ep} must have exactly one start at first window")
            if not bool(episode_end[ep_rows[-1]]) or bool(episode_end[ep_rows[:-1]].any()):
                raise ValueError(f"episode {ep} must have exactly one end at last window")
            if int(crop_start[ep_rows[0]].item()) != 0:
                raise ValueError(f"episode {ep} must start at crop_start 0")
            ep_crops = crop_start[ep_rows]
            if len(torch.unique(ep_crops)) != len(ep_crops):
                raise ValueError(f"episode {ep} has duplicate crop_starts")
            expected_next = ep_crops[:-1] + EXECUTION_HORIZON * decision_count[ep_rows[:-1]]
            if not torch.equal(ep_crops[1:], expected_next):
                raise ValueError(f"episode {ep} crop_start continuity broken")
            identity = payload["frame_refs"][int(ep_rows[0])][:2]
            wrist_base = None
            for row in ep_rows.tolist():
                current_ref = payload["frame_refs"][row]
                future_ref = payload["world_target_frame_refs"][row]
                if current_ref[:2] != identity or future_ref[:2] != identity:
                    raise ValueError("episode frame references cross demonstrations")
                for t in range(int(decision_count[row])):
                    d = int(crop_start[row]) + EXECUTION_HORIZON * t
                    current = current_ref[2][t]
                    future = future_ref[2][t]
                    base = current[-1] - d
                    if wrist_base is None:
                        wrist_base = base
                    if current[:4] != np.maximum(d + VISION_OFFSETS, 0).tolist() or base != wrist_base or future != [base + d + WORLD_HORIZON]:
                        raise ValueError("episode frame references do not match decision endpoints")

        # previous_action alignment checked only consecutive valid positions
        valid_pairs = decision_valid[:, 1:] & decision_valid[:, :-1]
        if not torch.equal(
            previous[:, 1:][valid_pairs],
            actions[:, :-1, EXECUTION_HORIZON - 1][valid_pairs],
        ):
            raise ValueError("joint previous_action is not aligned on consecutive valid positions")
    else:
        if not bool(valid[:, :, :EXECUTION_HORIZON].all()):
            raise ValueError("every LIBERO H50 training decision requires a real P15 prefix")
        if n_tasks == 2 and (not bool(valid.all()) or not bool(world_valid.all())):
            raise ValueError("the two LIBERO-Long tasks require complete action and World targets")
        if not torch.equal(
            previous[:, 1:],
            actions[:, :-1, EXECUTION_HORIZON - 1],
        ):
            raise ValueError("LIBERO P15 previous_action is not aligned to action token14")

    if any(len(decision) != 5 for _, _, decisions in payload["frame_refs"] for decision in decisions):
        raise ValueError("LIBERO dual-view source requires four agent frames plus one wrist frame")
    if any(len(decision) != 1 for _, _, decisions in payload["world_target_frame_refs"] for decision in decisions):
        raise ValueError("LIBERO World target requires one future wrist frame")
    return payload


def _validate_run_schedule(payload: dict, args: argparse.Namespace) -> tuple[int, int, str]:
    metadata = payload["metadata"]
    n_tasks = int(metadata["n_tasks"])
    is_all_starts = (
        metadata.get("contract") == ALL_STARTS_DATA_CONTRACT
        or getattr(args, "window_sampling", None) == "all_starts_random_tbptt8_v1"
    )
    if is_all_starts:
        if metadata.get("contract") != ALL_STARTS_DATA_CONTRACT:
            raise ValueError(f"all_starts_random_tbptt8_v1 requires contract {ALL_STARTS_DATA_CONTRACT}")
        if getattr(args, "architecture_version", "legacy") != "dual_tower_h15_v1":
            raise ValueError("all-starts training requires architecture_version dual_tower_h15_v1")
        if args.anchor_fraction != 0.0:
            raise ValueError(f"all-starts training requires anchor_fraction == 0.0, got {args.anchor_fraction}")
        if args.batch_size <= 0 or args.gpus <= 0 or args.batch_size % args.gpus != 0:
            raise ValueError("all-starts training requires positive batch_size divisible by gpus")
        if args.epochs < 1:
            raise ValueError("all-starts training requires epochs >= 1")
        if args.stage1_steps < 0:
            raise ValueError("all-starts training requires stage1_steps >= 0")
        counts = metadata.get("task_counts", [])
        if len(counts) != n_tasks or any(c <= 0 for c in counts) or sum(counts) != len(payload["actions"]):
            raise ValueError("all-starts task_counts must match positive dataset rows")

        from va_compound.data.all_starts import AllStartsWindowBatchSampler
        sampler = AllStartsWindowBatchSampler(
            payload,
            args.batch_size,
            getattr(args, "seed", 0),
            args.mixed_tasks,
            rank=0,
            world_size=args.gpus,
            epoch_offset=getattr(args, "epoch_offset", 0),
        )
        epoch_lens = sampler.epoch_lengths(args.epochs)
        first_epoch_length = epoch_lens[0]
        total_steps = sum(epoch_lens)
        grouping = (
            str(RUN_SCHEDULE_PROFILES[n_tasks]["grouping"])
            if n_tasks in RUN_SCHEDULE_PROFILES
            else (
                "single_task_t8_local1_deferred_v1"
                if n_tasks == 1
                else "all_starts_random_tbptt8_v1"
            )
        )
        return first_epoch_length, total_steps, grouping

    is_joint = getattr(args, "architecture_version", "legacy") in ("dual_tower_expert_v1", "dual_tower_h15_v1")
    if n_tasks == 1:
        if not is_joint:
            raise ValueError("this trainer supports the 40-task run or the LIBERO-Long task3+4 probe")
        suites = list(metadata.get("suites", []))
        if len(suites) != 1 or suites[0] not in LIBERO_SUITES:
            raise ValueError(f"single-task joint training requires exactly one suite from {LIBERO_SUITES}")
        task_specs = metadata.get("task_specs", [])
        if len(task_specs) != 1:
            raise ValueError("single-task joint training requires exactly one task spec")
        local_task_id = int(task_specs[0].get("local_task_id", -1))
        if local_task_id not in range(10):
            raise ValueError(f"single-task joint training requires local_task_id in 0..9, got {local_task_id}")
        if args.mixed_tasks != 1:
            raise ValueError(f"single-task joint training requires mixed_tasks == 1, got {args.mixed_tasks}")
        if args.anchor_fraction != 0.0:
            raise ValueError(f"single-task joint training requires anchor_fraction == 0.0, got {args.anchor_fraction}")
        if args.batch_size <= 0 or args.gpus <= 0 or args.batch_size % args.gpus != 0:
            raise ValueError("single-task joint training requires positive batch_size divisible by gpus")
        if args.stage1_steps < 0 or args.epochs < 1:
            raise ValueError("joint training requires stage1_steps >= 0 and epochs >= 1")
        counts = metadata.get("task_counts", [])
        if len(counts) != 1 or any(c <= 0 for c in counts) or sum(counts) != len(payload["actions"]):
            raise ValueError("joint task_counts must match positive dataset rows")
        from va_compound.data.episode_stream import EpisodeWindowBatchSampler
        steps_per_epoch = len(EpisodeWindowBatchSampler(
            payload, args.batch_size, getattr(args, "seed", 0), args.mixed_tasks,
            rank=0, world_size=args.gpus,
        ))
        return steps_per_epoch, steps_per_epoch * args.epochs, "single_task_t8_local1_deferred_v1"

    if n_tasks not in RUN_SCHEDULE_PROFILES:
        raise ValueError("this trainer supports the 40-task run or the LIBERO-Long task3+4 probe")
    profile = RUN_SCHEDULE_PROFILES[n_tasks]
    local_task_ids = [int(spec["local_task_id"]) for spec in metadata["task_specs"]]
    expected = {
        "rows": profile["rows"],
        "batch_size": profile["batch_size"],
        "mixed_tasks": profile["mixed_tasks"],
        "stage1_steps": profile["stage1_steps"],
        "epochs": profile["epochs"],
        "gpus": 2,
        "anchor_fraction": 0.0 if n_tasks == 2 else 0.25,
        "max_steps": None,
        "suites": profile["suites"],
        "local_task_ids": profile["local_task_ids"],
    }
    actual = {
        "rows": len(payload["actions"]),
        "batch_size": args.batch_size,
        "mixed_tasks": args.mixed_tasks,
        "stage1_steps": args.stage1_steps,
        "epochs": args.epochs,
        "gpus": args.gpus,
        "anchor_fraction": args.anchor_fraction,
        "max_steps": args.max_steps,
        "suites": list(metadata["suites"]),
        "local_task_ids": None if n_tasks == 40 else local_task_ids,
    }
    if getattr(args, "architecture_version", "legacy") in ("dual_tower_expert_v1", "dual_tower_h15_v1"):
        for key in ("rows", "stage1_steps", "epochs", "max_steps"):
            expected.pop(key)
        expected["anchor_fraction"] = 0.0
        counts = metadata.get("task_counts", [])
        if len(counts) != n_tasks or any(c <= 0 for c in counts) or sum(counts) != actual["rows"]:
            raise ValueError("joint task_counts must match positive dataset rows")
        if args.stage1_steps < 0 or args.epochs < 1:
            raise ValueError("joint training requires stage1_steps >= 0 and epochs >= 1")
    mismatch = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if mismatch:
        raise ValueError(f"LIBERO run schedule mismatch: {mismatch}")
    if getattr(args, "architecture_version", "legacy") in ("dual_tower_expert_v1", "dual_tower_h15_v1"):
        from va_compound.data.episode_stream import EpisodeWindowBatchSampler
        steps_per_epoch = len(EpisodeWindowBatchSampler(
            payload, args.batch_size, getattr(args, "seed", 0), args.mixed_tasks,
            rank=0, world_size=args.gpus,
        ))
    else:
        steps_per_epoch = (actual["rows"] + args.batch_size - 1) // args.batch_size
    return steps_per_epoch, steps_per_epoch * args.epochs, str(profile["grouping"])


def _validate_dense_source(source: dict) -> None:
    contract = source.get("training_contract") or {}
    expected = {
        "global_step": 1_000,
        "initialization": SOURCE_INITIALIZATION,
        "data_contract": SOURCE_DATA_CONTRACT,
        "phase": "stage2_qwen_va_dino_joint",
        "source_global_step": 5_000,
        "total_steps": 4_955,
        "action_horizon": ACTION_HORIZON,
        "deployment_execution_horizon": EXECUTION_HORIZON,
        "wmrm_cycle_steps": WORLD_HORIZON,
        "qwen_training": "last6_full_layers18_23_v1",
        "main_vision_joint_trained": True,
    }
    actual = {"global_step": int(source.get("global_step", -1)), **contract}
    mismatch = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatch:
        raise ValueError(f"dense continuation source mismatch: {mismatch}")
    if not isinstance(source.get("qwen_trainable_state_dict"), dict) or not isinstance(
        source.get("main_vision_trainable_state_dict"), dict
    ):
        raise ValueError("dense continuation requires trained Qwen and DINO states")
    if not isinstance(source.get("optimizer"), dict):
        raise ValueError("dense continuation requires the source optimizer state")
