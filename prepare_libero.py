#!/usr/bin/env python
"""HDF5 data preparation and preprocessing for LIBERO benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from va_compound.data.libero import (
    ACTION_HORIZON,
    DATA_CONTRACT,
    DECISION_OFFSETS,
    EXECUTION_HORIZON,
    FUSION_LAYERS,
    JOINT_DATA_CONTRACT,
    LIBERO_SUITES,
    SEQUENCE,
    SOURCE_DATA_CONTRACT,
    VISION_OFFSETS,
    WORLD_HORIZON,
    _attach_dense_world_action_donors,
    _local_task_ids,
    _normalize,
    _official_task_specs,
    _suite_names,
)
from va_compound.world_supervision import prepare_visual_world_action_ranking


def _save_new(payload: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _task_description(handle) -> str:
    return json.loads(handle["data"].attrs["problem_info"])["language_instruction"].strip()


def _ensure_longtraj(
    source: Path,
    output: Path,
    *,
    task_id: int,
    suite: str,
    local_task_id: int,
    description: str,
) -> None:
    if output.is_file():
        existing = torch.load(output, map_location="cpu", weights_only=False, mmap=True)
        metadata = existing.get("metadata") or {}
        if (
            metadata.get("source") != str(source)
            or metadata.get("task") != description
            or len(existing.get("episodes") or []) != 50
        ):
            raise ValueError(f"existing longtraj identity mismatch: {output}")
        return
    import h5py

    episodes = []
    with h5py.File(source, "r") as handle:
        names = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        if len(names) != 50:
            raise ValueError(f"{source} has {len(names)} demos, expected 50")
        for name in names:
            obs = handle["data"][name]["obs"]
            agent = obs["agentview_rgb"][()]
            wrist = obs["eye_in_hand_rgb"][()]
            action_count = len(handle["data"][name]["actions"])
            if agent.shape != wrist.shape or len(agent) != action_count:
                raise ValueError(f"camera/action streams differ in {source}:{name}")
            episodes.append(
                {
                    # One shared frame array keeps the existing frame-ref loader:
                    # [all agent frames, all wrist frames].
                    "frames": np.concatenate(
                        (np.flip(agent, axis=1), np.flip(wrist, axis=1)), axis=0
                    ).copy()
                }
            )
    _save_new(
        {
            "episodes": episodes,
            "metadata": {
                "contract": "libero_official_upright_agent4_wrist1_frames_v2",
                "task_id": task_id,
                "suite": suite,
                "local_task_id": local_task_id,
                "task": description,
                "source": str(source),
            },
        },
        output,
    )


def _window_max_start(length: int, *, joint_frontend: bool) -> int:
    p15_max_start = (
        length
        - 1
        - int(DECISION_OFFSETS[-1])
        - EXECUTION_HORIZON
    )
    if p15_max_start < 0:
        return -1
    if joint_frontend:
        return p15_max_start
    action_h50_max_start = (
        length
        - 1
        - int(DECISION_OFFSETS[-1])
        - ACTION_HORIZON
    )
    return (
        action_h50_max_start
        if action_h50_max_start >= 0
        else p15_max_start
    )


def prepare_data(args: argparse.Namespace) -> None:
    import h5py

    joint_frontend = getattr(args, "architecture_version", "legacy") == "dual_tower_expert_v1"
    if joint_frontend and not args.dense_windows:
        raise ValueError("dual_tower_expert_v1 requires --dense-windows")
    if args.windows_per_demo < 1:
        raise ValueError("--windows-per-demo must be positive")
    suites = _suite_names(args.suites)
    specs = _official_task_specs(suites, _local_task_ids(args.local_task_ids))
    by_description = {spec["description"]: spec for spec in specs}
    sources = [
        source
        for suite in suites
        for source in sorted((args.hdf5_dir / suite).glob("*.hdf5"))
    ]
    if len(sources) != 10 * len(suites):
        raise ValueError(
            f"expected {10 * len(suites)} HDF5 files, got {len(sources)}"
        )

    ordered: list[tuple[int, str, int, str, Path]] = []
    states = []
    for source in sources:
        with h5py.File(source, "r") as handle:
            description = _task_description(handle)
            spec = by_description.get(description)
            if spec is None:
                if args.local_task_ids is not None:
                    continue
                raise ValueError(f"unknown task in {source}: {description}")
            ordered.append(
                (
                    int(spec["task_id"]),
                    str(spec["suite"]),
                    int(spec["local_task_id"]),
                    description,
                    source,
                )
            )
            for name in handle["data"]:
                obs = handle["data"][name]["obs"]
                states.append(
                    np.concatenate((obs["joint_states"][()], obs["gripper_states"][()]), axis=1)
                )
    ordered.sort()
    if [item[0] for item in ordered] != list(range(len(specs))):
        raise ValueError("HDF5 files do not map one-to-one onto official LIBERO tasks")
    all_states = np.concatenate(states).astype(np.float32)
    state_low, state_high = np.quantile(all_states, (0.01, 0.99), axis=0).astype(np.float32)

    actions_out, previous_out, proprio_out = [], [], []
    valid_out, world_valid_out = [], []
    instruction_ids, episode_ids, crop_starts, frame_refs, target_refs = [], [], [], [], []
    task_counts = [0] * len(specs)
    global_episode = 0
    for task_id, suite, local_task_id, description, source in ordered:
        task_key = f"{suite}_t{local_task_id:02d}_agent4_wrist1"
        longtraj = args.longtraj / f"metaworld_longtraj_{task_key}.pt"
        _ensure_longtraj(
            source,
            longtraj,
            task_id=task_id,
            suite=suite,
            local_task_id=local_task_id,
            description=description,
        )
        with h5py.File(source, "r") as handle:
            names = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
            if len(names) != 50:
                raise ValueError(f"{source} has {len(names)} demos, expected 50")
            for demo_index, name in enumerate(names):
                demo = handle["data"][name]
                raw_actions = demo["actions"][()].astype(np.float32)
                if not np.isfinite(raw_actions).all() or np.abs(raw_actions).max() > 1.0001:
                    raise ValueError(f"raw OSC actions outside [-1,1]: {source}:{name}")
                obs = demo["obs"]
                raw_state = np.concatenate(
                    (obs["joint_states"][()], obs["gripper_states"][()]), axis=1
                ).astype(np.float32)
                max_start = _window_max_start(len(raw_actions), joint_frontend=joint_frontend)
                if max_start < 0:
                    raise ValueError(f"demo too short for T8/H50/P15: {source}:{name}")
                starts = (
                    np.arange(max_start + 1, dtype=np.int64)
                    if args.dense_windows
                    else np.linspace(
                        0,
                        max_start,
                        args.windows_per_demo,
                        dtype=np.int64,
                    )
                )
                for start in starts:
                    decisions = start + DECISION_OFFSETS
                    chunks, masks = [], []
                    for decision in decisions:
                        chunk = raw_actions[
                            decision + 1 : decision + 1 + ACTION_HORIZON
                        ]
                        valid = np.arange(ACTION_HORIZON) < len(chunk)
                        if len(chunk) < EXECUTION_HORIZON:
                            raise RuntimeError("every H50 row must have a real P15 prefix")
                        if len(chunk) < ACTION_HORIZON:
                            chunk = np.concatenate(
                                (
                                    chunk,
                                    np.repeat(
                                        chunk[-1:], ACTION_HORIZON - len(chunk), axis=0
                                    ),
                                ),
                                axis=0,
                            )
                        chunks.append(chunk)
                        masks.append(valid)
                    actions_out.append(np.stack(chunks))
                    valid_out.append(np.stack(masks))
                    previous_out.append(raw_actions[decisions])
                    proprio_out.append(_normalize(raw_state[decisions], state_low, state_high))
                    agent_history = np.maximum(
                        decisions[:, None] + VISION_OFFSETS[None], 0
                    )
                    wrist_current = (len(raw_actions) + decisions)[:, None]
                    current = np.concatenate((agent_history, wrist_current), axis=1)
                    # WAM predicts the future wrist map, which is the last view in
                    # the five-frame source layout and the precision-critical view.
                    future = decisions + WORLD_HORIZON
                    world_valid_out.append(future < len(raw_actions))
                    target = (
                        len(raw_actions) + np.minimum(future, len(raw_actions) - 1)
                    )[:, None]
                    frame_refs.append((task_key, demo_index, current.tolist()))
                    target_refs.append((task_key, demo_index, target.tolist()))
                    instruction_ids.append(task_id)
                    episode_ids.append(global_episode)
                    crop_starts.append(int(start))
                    task_counts[task_id] += 1
                global_episode += 1

    instruction_id = torch.tensor(instruction_ids, dtype=torch.long)
    actions = torch.from_numpy(np.stack(actions_out)).float()
    previous = torch.from_numpy(np.stack(previous_out)).float()
    proprio = torch.from_numpy(np.stack(proprio_out)).float()
    valid = torch.from_numpy(np.stack(valid_out)).bool()
    world_valid = torch.from_numpy(np.stack(world_valid_out)).bool()
    # Online Qwen replaces these tiny schema placeholders before every forward.
    language_hidden = torch.zeros((len(actions), 1, 1), dtype=torch.float16)
    language_mask = torch.ones((len(actions), 1), dtype=torch.bool)
    contract_str = (
        JOINT_DATA_CONTRACT
        if joint_frontend
        else (DATA_CONTRACT if args.dense_windows else SOURCE_DATA_CONTRACT)
    )
    metadata_payload = {
        "contract": contract_str,
        "tasks": [spec["description"] for spec in specs],
        "task_specs": specs,
        "suites": list(suites),
        "n_tasks": len(specs),
        "n_demos": global_episode,
        "windows_per_demo": None if args.dense_windows else args.windows_per_demo,
        "window_sampling": (
            "all_legal_starts_v1" if args.dense_windows else "fixed_linspace16_v1"
        ),
        "task_counts": task_counts,
        "sequence_length": SEQUENCE,
        "action_horizon": ACTION_HORIZON,
        "planning_stride": EXECUTION_HORIZON,
        "control_stride": EXECUTION_HORIZON,
        "decision_offsets": DECISION_OFFSETS.tolist(),
        "vision_offsets": VISION_OFFSETS.tolist(),
        "vision_input": "agentview_history4_plus_current_wrist_v2",
        "vision_frame_layout": [
            "agentview_d-6",
            "agentview_d-4",
            "agentview_d-2",
            "agentview_d",
            "eye_in_hand_d",
        ],
        "world_target_view": "eye_in_hand_rgb",
        "logged_action_chunk": "masked_h50_real_p15_prefix",
        "world_target_horizon": WORLD_HORIZON,
        "world_target_offsets": (
            DECISION_OFFSETS + WORLD_HORIZON
        ).tolist(),
        "world_target_alignment": f"obs[d+{WORLD_HORIZON}]",
        "target_alignment": "obs[d]_to_actions[d+1:d+51]_masked_tail",
        "previous_action_alignment": "actions[d]",
        "previous_action_model_input": "zero_v1",
        "orientation_contract": "vertical_flip_opengl_to_upright_once",
        "action_contract": "raw_libero_osc_pose_minus1_plus1",
        "proprio_contract": "q01q99(joint_states7+gripper_states2)",
        "language_source": "online_qwen35_0_8b_last6_full_v1",
        "language_dim": 1024,
        "qwen_keep_layers": 24,
        "qwen_fusion_layers": FUSION_LAYERS,
        "qwen_base_readout": "layer23_final_norm",
        "qwen_fusion_reduce": "none",
        "short_horizon_padding": "repeat_last_masked_v1",
        "minimum_real_action_prefix": EXECUTION_HORIZON,
    }
    if joint_frontend:
        metadata_payload["window_bound"] = "complete_p15_masked_h50_v1"
    payload = {
        "actions": actions,
        "previous_action": previous,
        "proprio": proprio,
        "language_hidden": language_hidden,
        "language_mask": language_mask,
        "instruction_id": instruction_id,
        "episode_id": torch.tensor(episode_ids, dtype=torch.long),
        "crop_start": torch.tensor(crop_starts, dtype=torch.long),
        "pair_id": torch.arange(len(actions), dtype=torch.long),
        "frame_refs": frame_refs,
        "world_target_frame_refs": target_refs,
        "action_valid_mask": valid,
        "recovery_mask": torch.zeros_like(valid),
        "world_target_valid_mask": world_valid,
        "anchor_eligible": torch.ones(len(actions), dtype=torch.bool),
        "normalization": {
            "action_q01": torch.full((7,), -1.0),
            "action_q99": torch.full((7,), 1.0),
            "state_q01": torch.from_numpy(state_low),
            "state_q99": torch.from_numpy(state_high),
        },
        "metadata": metadata_payload,
    }
    if args.dense_windows:
        _attach_dense_world_action_donors(payload)
    else:
        prepare_visual_world_action_ranking(
            payload, planning_stride=EXECUTION_HORIZON
        )
    expected = sum(task_counts)
    if joint_frontend:
        assert min(task_counts) > 0
    elif args.dense_windows:
        if len(specs) == 2:
            assert task_counts == [4684, 5159]
        else:
            assert min(task_counts) > 0
    else:
        assert expected == len(specs) * 50 * args.windows_per_demo
        assert min(task_counts) == max(task_counts) == 50 * args.windows_per_demo
    assert tuple(actions.shape) == (expected, SEQUENCE, ACTION_HORIZON, 7)
    assert tuple(proprio.shape) == (expected, SEQUENCE, 9)
    assert bool(valid[:, :, :EXECUTION_HORIZON].all())
    assert torch.equal(
        previous[:, 1:], actions[:, :-1, EXECUTION_HORIZON - 1]
    )
    _save_new(payload, args.data)
    print(f"PASS prepared {args.data}: {tuple(actions.shape)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/root/libero_spatial_ora0_v1/"
            "libero_4suite_h50p15_t4_dualview5_maskedtail_v2.pt"
        ),
    )
    parser.add_argument(
        "--longtraj",
        type=Path,
        default=Path("/root/libero_spatial_ora0_v1/longtraj"),
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=Path("/root/libero_spatial_ora0_v1/datasets"),
    )
    parser.add_argument("--suites", default=",".join(LIBERO_SUITES))
    parser.add_argument(
        "--local-task-ids",
        help="Comma-separated local task ids; requires exactly one --suites entry.",
    )
    parser.add_argument("--windows-per-demo", type=int, default=16)
    parser.add_argument("--dense-windows", action="store_true")
    parser.add_argument(
        "--architecture-version",
        choices=("legacy", "dual_tower_expert_v1"),
        default="legacy",
        help="Architecture version for LIBERO data preparation.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    prepare_data(args)


if __name__ == "__main__":
    main()
