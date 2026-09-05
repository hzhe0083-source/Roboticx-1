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
    ALL_STARTS_DATA_CONTRACT,
    DATA_CONTRACT,
    DECISION_OFFSETS,
    EXECUTION_HORIZON,
    FUSION_LAYERS,
    JOINT_DATA_CONTRACT,
    H15_DATA_CONTRACT,
    LIBERO_SUITES,
    SEQUENCE,
    SOURCE_DATA_CONTRACT,
    VISION_OFFSETS,
    WORLD_HORIZON,
    _attach_all_starts_world_action_donors,
    _attach_dense_world_action_donors,
    _attach_joint_world_action_donors,
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

    window_sampling = getattr(args, "window_sampling", "episode_contiguous_p15_v1")
    is_all_starts = window_sampling == "all_starts_random_tbptt8_v1"
    is_h15 = getattr(args, "architecture_version", "legacy") == "dual_tower_h15_v1"
    if is_all_starts:
        if not is_h15:
            raise ValueError("--window-sampling all_starts_random_tbptt8_v1 requires --architecture-version dual_tower_h15_v1")
        if not args.dense_windows:
            raise ValueError("--window-sampling all_starts_random_tbptt8_v1 requires --dense-windows")

    action_horizon = EXECUTION_HORIZON if is_h15 else ACTION_HORIZON
    joint_frontend = getattr(args, "architecture_version", "legacy") in ("dual_tower_expert_v1", "dual_tower_h15_v1")
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
    state_delta_scale = np.where(np.abs(state_high - state_low) < 1e-6, 1.0, state_high - state_low).astype(np.float32)

    actions_out, previous_out, proprio_out = [], [], []
    valid_out, world_valid_out = [], []
    instruction_ids, episode_ids, crop_starts, frame_refs, target_refs = [], [], [], [], []
    decision_valid_out, decision_counts_out = [], []
    episode_starts_out, episode_ends_out = [], []
    world_state_delta_out = []
    episode_lengths = []
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
                if is_all_starts:
                    if raw_state.shape != (len(raw_actions), 9) or not np.isfinite(raw_state).all():
                        raise ValueError(f"invalid joint/gripper state stream: {source}:{name}")
                    episode_lengths.append(len(raw_actions))
                    demo_decisions = np.arange(0, len(raw_actions) - EXECUTION_HORIZON, dtype=np.int64)
                    if len(demo_decisions) == 0:
                        raise ValueError(f"demo too short for all-starts decisions: {source}:{name}")
                    for d in demo_decisions:
                        d_val = int(d)
                        chunk = raw_actions[d_val + 1 : d_val + 1 + EXECUTION_HORIZON][None, ...]
                        valid = np.ones((1, EXECUTION_HORIZON), dtype=bool)
                        delta = 2.0 * (raw_state[d_val + 15 : d_val + 16] - raw_state[d_val : d_val + 1]) / state_delta_scale
                        actions_out.append(chunk)
                        valid_out.append(valid)
                        previous_out.append(raw_actions[d_val : d_val + 1])
                        proprio_out.append(_normalize(raw_state[d_val : d_val + 1], state_low, state_high))
                        world_state_delta_out.append(delta.astype(np.float32))

                        agent_history = np.maximum(d_val + VISION_OFFSETS, 0)
                        wrist_current = len(raw_actions) + d_val
                        current = np.concatenate((agent_history, [wrist_current]))[None, :]
                        target = np.array([[len(raw_actions) + d_val + WORLD_HORIZON]])

                        world_valid_out.append(np.ones(1, dtype=bool))
                        frame_refs.append((task_key, demo_index, current.tolist()))
                        target_refs.append((task_key, demo_index, target.tolist()))
                        instruction_ids.append(task_id)
                        episode_ids.append(global_episode)
                        crop_starts.append(d_val)
                        task_counts[task_id] += 1
                    global_episode += 1
                elif joint_frontend:
                    if raw_state.shape != (len(raw_actions), 9) or not np.isfinite(raw_state).all():
                        raise ValueError(f"invalid joint/gripper state stream: {source}:{name}")
                    demo_decisions = np.arange(0, len(raw_actions) - EXECUTION_HORIZON, EXECUTION_HORIZON, dtype=np.int64)
                    if len(demo_decisions) == 0:
                        raise ValueError(f"demo too short for joint decisions: {source}:{name}")
                    num_windows = (len(demo_decisions) + SEQUENCE - 1) // SEQUENCE
                    for w_idx in range(num_windows):
                        win_decisions = demo_decisions[w_idx * SEQUENCE : (w_idx + 1) * SEQUENCE]
                        valid_k = len(win_decisions)
                        crop_start_val = int(win_decisions[0])
                        is_ep_start = bool(w_idx == 0)
                        is_ep_end = bool(w_idx == num_windows - 1)

                        d_valid = np.zeros(SEQUENCE, dtype=bool)
                        d_valid[:valid_k] = True

                        chunks, masks = [], []
                        deltas = []
                        last_d = int(win_decisions[-1])

                        storage_decisions = np.empty(SEQUENCE, dtype=np.int64)
                        storage_decisions[:valid_k] = win_decisions
                        storage_decisions[valid_k:] = last_d

                        for i in range(SEQUENCE):
                            decision = int(storage_decisions[i])
                            chunk = raw_actions[decision + 1 : decision + 1 + action_horizon]
                            if i < valid_k:
                                valid = np.arange(action_horizon) < len(chunk)
                                if len(chunk) < EXECUTION_HORIZON:
                                    raise RuntimeError("every valid joint decision must have a real P15 prefix")
                                if len(chunk) < action_horizon:
                                    chunk = np.concatenate(
                                        (
                                            chunk,
                                            np.repeat(chunk[-1:], action_horizon - len(chunk), axis=0),
                                        ),
                                        axis=0,
                                    )
                                delta = 2.0 * (raw_state[decision + 15] - raw_state[decision]) / state_delta_scale
                            else:
                                valid = np.zeros(action_horizon, dtype=bool)
                                if len(chunk) < action_horizon:
                                    chunk = np.concatenate(
                                        (
                                            chunk,
                                            np.repeat(chunk[-1:], action_horizon - len(chunk), axis=0),
                                        ),
                                        axis=0,
                                    )
                                delta = np.zeros(9, dtype=np.float32)

                            chunks.append(chunk)
                            masks.append(valid)
                            deltas.append(delta)

                        actions_out.append(np.stack(chunks))
                        valid_out.append(np.stack(masks))
                        previous_out.append(raw_actions[storage_decisions])
                        proprio_out.append(_normalize(raw_state[storage_decisions], state_low, state_high))
                        world_state_delta_out.append(np.stack(deltas).astype(np.float32))

                        agent_history = np.maximum(storage_decisions[:, None] + VISION_OFFSETS[None], 0)
                        wrist_current = (len(raw_actions) + storage_decisions)[:, None]
                        current = np.concatenate((agent_history, wrist_current), axis=1)

                        future = storage_decisions + WORLD_HORIZON
                        w_valid = np.zeros(SEQUENCE, dtype=bool)
                        w_valid[:valid_k] = True
                        world_valid_out.append(w_valid)

                        target = (len(raw_actions) + np.minimum(future, len(raw_actions) - 1))[:, None]
                        frame_refs.append((task_key, demo_index, current.tolist()))
                        target_refs.append((task_key, demo_index, target.tolist()))
                        instruction_ids.append(task_id)
                        episode_ids.append(global_episode)
                        crop_starts.append(crop_start_val)
                        decision_valid_out.append(d_valid)
                        decision_counts_out.append(valid_k)
                        episode_starts_out.append(is_ep_start)
                        episode_ends_out.append(is_ep_end)
                        task_counts[task_id] += 1
                    global_episode += 1
                else:
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
                                decision + 1 : decision + 1 + action_horizon
                            ]
                            valid = np.arange(action_horizon) < len(chunk)
                            if len(chunk) < EXECUTION_HORIZON:
                                raise RuntimeError("every H50 row must have a real P15 prefix")
                            if len(chunk) < action_horizon:
                                chunk = np.concatenate(
                                    (
                                        chunk,
                                        np.repeat(
                                            chunk[-1:], action_horizon - len(chunk), axis=0
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
        ALL_STARTS_DATA_CONTRACT if is_all_starts else (
            H15_DATA_CONTRACT if is_h15 else JOINT_DATA_CONTRACT
            if joint_frontend
            else (DATA_CONTRACT if args.dense_windows else SOURCE_DATA_CONTRACT)
        )
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
            "all_starts_random_tbptt8_v1"
            if is_all_starts
            else (
                (
                    "episode_contiguous_p15_v1"
                    if joint_frontend
                    else "all_legal_starts_v1"
                )
                if args.dense_windows
                else "fixed_linspace16_v1"
            )
        ),
        "task_counts": task_counts,
        "sequence_length": SEQUENCE,
        "storage_sequence_length": 1 if is_all_starts else SEQUENCE,
        "action_horizon": action_horizon,
        "planning_stride": EXECUTION_HORIZON,
        "control_stride": EXECUTION_HORIZON,
        "decision_offsets": [0] if is_all_starts else DECISION_OFFSETS.tolist(),
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
        "logged_action_chunk": "real_p15" if is_h15 else "masked_h50_real_p15_prefix",
        "world_target_horizon": WORLD_HORIZON,
        "world_target_offsets": [WORLD_HORIZON] if is_all_starts else (
            DECISION_OFFSETS + WORLD_HORIZON
        ).tolist(),
        "world_target_alignment": f"obs[d+{WORLD_HORIZON}]",
        "target_alignment": "obs[d]_to_actions[d+1:d+16]" if is_h15 else "obs[d]_to_actions[d+1:d+51]_masked_tail",
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
        "short_horizon_padding": "episode_storage_only_v1" if is_h15 else "repeat_last_masked_v1",
        "minimum_real_action_prefix": EXECUTION_HORIZON,
    }
    if is_all_starts:
        metadata_payload["sampling_contract"] = "all_starts_random_tbptt8_v1"
        metadata_payload["state_delta_contract"] = "joint7_gripper2_unclipped_q01q99_delta_h15_v1"
        metadata_payload["memory_contract"] = "offset_replay_tbptt8_v1"
        metadata_payload["episode_lengths"] = episode_lengths
    elif joint_frontend:
        metadata_payload["window_bound"] = "complete_p15_v1" if is_h15 else "complete_p15_masked_h50_v1"
        metadata_payload["state_delta_contract"] = "joint7_gripper2_unclipped_q01q99_delta_h15_v1"
        metadata_payload["memory_contract"] = "episode_tbptt8_v1"
    normalization_payload = {
        "action_q01": torch.full((7,), -1.0),
        "action_q99": torch.full((7,), 1.0),
        "state_q01": torch.from_numpy(state_low),
        "state_q99": torch.from_numpy(state_high),
    }
    if is_all_starts or joint_frontend:
        normalization_payload["state_delta_scale"] = torch.from_numpy(state_delta_scale)
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
        "normalization": normalization_payload,
        "metadata": metadata_payload,
    }
    if is_all_starts:
        payload["world_state_delta"] = torch.from_numpy(np.stack(world_state_delta_out)).float()
        _attach_all_starts_world_action_donors(payload)
    elif joint_frontend:
        payload["decision_valid_mask"] = torch.from_numpy(np.stack(decision_valid_out))
        payload["decision_count"] = torch.tensor(decision_counts_out, dtype=torch.long)
        payload["episode_start"] = torch.tensor(episode_starts_out, dtype=torch.bool)
        payload["episode_end"] = torch.tensor(episode_ends_out, dtype=torch.bool)
        payload["world_state_delta"] = torch.from_numpy(np.stack(world_state_delta_out)).float()
        _attach_joint_world_action_donors(payload)
    elif args.dense_windows:
        _attach_dense_world_action_donors(payload)
    else:
        prepare_visual_world_action_ranking(
            payload, planning_stride=EXECUTION_HORIZON
        )
    expected = sum(task_counts)
    if is_all_starts:
        assert min(task_counts) > 0
        assert tuple(actions.shape) == (expected, 1, EXECUTION_HORIZON, 7)
        assert tuple(proprio.shape) == (expected, 1, 9)
        assert bool(valid.all())
        assert bool(world_valid.all())
        for ep in range(global_episode):
            ep_rows = torch.where(payload["episode_id"] == ep)[0]
            if len(ep_rows) > 1:
                assert torch.equal(previous[ep_rows[1:], 0], actions[ep_rows[:-1], 0, 0])
    elif joint_frontend:
        assert min(task_counts) > 0
        assert tuple(actions.shape) == (expected, SEQUENCE, action_horizon, 7)
        assert tuple(proprio.shape) == (expected, SEQUENCE, 9)
        dec_v = payload["decision_valid_mask"]
        assert bool(valid[dec_v][:, :EXECUTION_HORIZON].all())
        valid_pairs = dec_v[:, 1:] & dec_v[:, :-1]
        assert torch.equal(
            previous[:, 1:][valid_pairs], actions[:, :-1, EXECUTION_HORIZON - 1][valid_pairs]
        )
    else:
        assert expected == len(specs) * 50 * args.windows_per_demo
        assert min(task_counts) == max(task_counts) == 50 * args.windows_per_demo
        assert tuple(actions.shape) == (expected, SEQUENCE, action_horizon, 7)
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
        choices=("legacy", "dual_tower_expert_v1", "dual_tower_h15_v1"),
        default="legacy",
        help="Architecture version for LIBERO data preparation.",
    )
    parser.add_argument(
        "--window-sampling",
        choices=("episode_contiguous_p15_v1", "all_starts_random_tbptt8_v1"),
        default="episode_contiguous_p15_v1",
        help="Window sampling strategy for LIBERO data preparation.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    prepare_data(args)


if __name__ == "__main__":
    main()
