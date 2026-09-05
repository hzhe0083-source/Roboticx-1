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


def _validate_data(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    metadata = payload.get("metadata", {})
    if metadata.get("contract") != DATA_CONTRACT:
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
    if tuple(payload["actions"].shape[1:]) != (SEQUENCE, ACTION_HORIZON, 7):
        raise ValueError("LIBERO training data must be T8/H50/A7")
    if tuple(payload["proprio"].shape[1:]) != (SEQUENCE, 9):
        raise ValueError("LIBERO training data must contain T8/D9 proprio")
    expected_metadata = {
        "window_sampling": "all_legal_starts_v1",
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
        "short_horizon_padding": "repeat_last_masked_v1",
        "minimum_real_action_prefix": EXECUTION_HORIZON,
    }
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise ValueError(f"LIBERO H50/P15 metadata mismatch: {mismatch}")
    valid = payload.get("action_valid_mask")
    if not isinstance(valid, Tensor) or valid.shape != payload["actions"].shape[:-1]:
        raise ValueError("LIBERO H50 data requires an aligned action_valid_mask")
    if not bool(valid[:, :, :EXECUTION_HORIZON].all()):
        raise ValueError("every LIBERO H50 training decision requires a real P15 prefix")
    world_valid = payload.get("world_target_valid_mask")
    if not isinstance(world_valid, Tensor) or world_valid.shape != valid.shape[:2]:
        raise ValueError("LIBERO H50 World targets require an aligned [N,T] mask")
    if n_tasks == 2 and (not bool(valid.all()) or not bool(world_valid.all())):
        raise ValueError("the two LIBERO-Long tasks require complete action and World targets")
    crop_start = payload.get("crop_start")
    if not isinstance(crop_start, Tensor) or crop_start.shape != instruction_ids.shape:
        raise ValueError("dense LIBERO data requires one crop_start per row")
    if not torch.equal(
        payload["previous_action"][:, 1:],
        payload["actions"][:, :-1, EXECUTION_HORIZON - 1],
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
    mismatch = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if mismatch:
        raise ValueError(f"LIBERO run schedule mismatch: {mismatch}")
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
