"""Tests for closed-loop evaluation contract and protocol with compact all-starts data."""
from __future__ import annotations

import copy
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

import eval_libero_closedloop
from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.data.libero import (
    ALL_STARTS_DATA_CONTRACT,
    H15_DATA_CONTRACT,
    JOINT_DATA_CONTRACT,
)


def test_eval_contract_all_starts_metadata_and_protocol(tmp_path, monkeypatch):
    data_path = tmp_path / "all_starts.pt"
    ckpt_path = tmp_path / "ckpt.pt"
    dino_path = tmp_path / "dino.pt"
    dino_path.touch()

    # Create dummy payload with ALL_STARTS_DATA_CONTRACT
    metadata = {
        "contract": ALL_STARTS_DATA_CONTRACT,
        "sampling_contract": "all_starts_random_tbptt8_v1",
        "window_sampling": "all_starts_random_tbptt8_v1",
        "storage_sequence_length": 1,
        "sequence_length": 8,
        "action_horizon": 15,
        "planning_stride": 15,
        "control_stride": 15,
        "decision_offsets": [0],
        "vision_offsets": [-6, -4, -2, 0],
        "vision_input": "agentview_history4_plus_current_wrist_v2",
        "world_target_view": "eye_in_hand_rgb",
        "world_target_horizon": 15,
        "world_target_offsets": [15],
        "world_target_alignment": "obs[d+15]",
        "target_alignment": "obs[d]_to_actions[d+1:d+16]",
        "logged_action_chunk": "real_p15",
        "language_source": "online_qwen35_0_8b_last6_full_v1",
        "language_dim": 1024,
        "state_delta_contract": "joint7_gripper2_unclipped_q01q99_delta_h15_v1",
        "memory_contract": "offset_replay_tbptt8_v1",
        "world_action_donor_contract": "task_next_episode_relative_progress_v1",
        "n_tasks": 2,
        "n_demos": 2,
        "tasks": ["libero_10 task 3", "libero_10 task 4"],
        "suites": ["libero_10"],
        "task_specs": [
            {"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "libero_10 task 3"},
            {"task_id": 1, "suite": "libero_10", "local_task_id": 4, "description": "libero_10 task 4"},
        ],
        "episode_lengths": [30, 30],
        "task_counts": [15, 15],
        "action_contract": "raw_libero_osc_pose_minus1_plus1",
    }
    normalization = {
        "action_q01": torch.full((7,), -1.0),
        "action_q99": torch.full((7,), 1.0),
        "state_q01": torch.zeros(9),
        "state_q99": torch.ones(9),
        "state_delta_scale": torch.ones(9),
    }
    vision_offsets = np.array([-6, -4, -2, 0])
    payload = {
        "actions": torch.zeros(30, 1, 15, 7),
        "previous_action": torch.zeros(30, 1, 7),
        "proprio": torch.zeros(30, 1, 9),
        "world_state_delta": torch.zeros(30, 1, 9),
        "action_valid_mask": torch.ones(30, 1, 15, dtype=torch.bool),
        "world_target_valid_mask": torch.ones(30, 1, dtype=torch.bool),
        "world_rank_shuffle_action": torch.zeros(30, 1, 15, 7),
        "world_rank_shuffle_mask": torch.zeros(30, 1, dtype=torch.bool),
        "crop_start": torch.cat([torch.arange(15), torch.arange(15)]),
        "episode_id": torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)]),
        "instruction_id": torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)]),
        "pair_id": torch.arange(30),
        "anchor_eligible": torch.ones(30, dtype=torch.bool),
        "language_hidden": torch.zeros(30, 1, 1, dtype=torch.float16),
        "language_mask": torch.ones(30, 1, dtype=torch.bool),
        "frame_refs": [("libero_10_t03_agent4_wrist1", 0, [np.maximum(d + vision_offsets, 0).tolist() + [30 + d]]) for d in range(15)] +
                      [("libero_10_t04_agent4_wrist1", 1, [np.maximum(d + vision_offsets, 0).tolist() + [30 + d]]) for d in range(15)],
        "world_target_frame_refs": [("libero_10_t03_agent4_wrist1", 0, [[30 + d + 15]]) for d in range(15)] +
                                   [("libero_10_t04_agent4_wrist1", 1, [[30 + d + 15]]) for d in range(15)],
        "normalization": normalization,
        "metadata": metadata,
    }
    torch.save(payload, data_path)

    from train_libero import _fresh_config
    config_obj = _fresh_config(architecture_version="dual_tower_h15_v1")
    config = dict(config_obj.__dict__)

    contract = {
        "action_decoder": "conditional_flow_matching",
        "flow_steps": 8,
        "sequence_length": 8,
        "memory_reset_every": 0,
        "initialization": "fresh_dual_tower_h15_v1",
        "data_contract": ALL_STARTS_DATA_CONTRACT,
        "suites": ["libero_10"],
        "n_tasks": 2,
        "task_specs": metadata["task_specs"],
        "action_horizon": 15,
        "planning_stride": 15,
        "deployment_execution_horizon": 15,
        "wmrm_cycle_steps": 15,
        "flow_slot_identity": "per_slot_action_condition_v1",
        "flow_prefix_steps": 15,
        "flow_prefix_weight": 1.0,
        "flow_tail_weight": 0.0,
        "qwen_joint_trained": True,
        "qwen_keep_layers": 24,
        "qwen_fusion_layers": list(range(18, 24)),
        "cross_modal_va_layers": list(range(6)),
        "qwen_base_readout": "layer23_final_norm",
        "qwen_fusion_reduce": "none",
        "qwen_training": "last6_full_layers18_23_v1",
        "qwen_trainable_layers": list(range(18, 24)),
        "qwen_final_norm_frozen": True,
        "qwen_hidden_dim": 1024,
        "main_vision_trainable_layers": list(range(18, 24)),
        "main_vision_frames": 5,
        "dino_fusion_layers": list(range(18, 24)),
        "dino_base_readout": "block23_norm",
        "dino_fusion_reduce": "none",
        "wmrm_feature_metric": "cosine",
        "wmrm_evidence": "post_va_vl_fused_tokens_v1",
        "previous_action_input": "zero_v1",
        "wmrm_world_dim": 1024,
        "wmrm_predictor_width": 1024,
        "flow_hidden_dim": 512,
        "vision_input": "agentview_history4_plus_current_wrist_v2",
        "world_target_view": "eye_in_hand_rgb",
        "fusion_initialization": "dual_tower_zero_output_v1",
        "architecture_version": "dual_tower_h15_v1",
        "source_global_step": -1,
        "stage1_steps": 100,
        "total_steps": 200,
        "optimizer_initialization": "fresh_adamw_v1",
        "execution_gradient_contract": "h15_unified_live_va_v1",
        "qwen_world_cache": "per_observation_joint_live_v1",
        "stage1_world_current_vision_cache": "disabled_joint_frontend_v1",
        "memory_contract": "offset_replay_tbptt8_v1",
        "sampling_contract": "all_starts_random_tbptt8_v1",
        "window_sampling": "all_starts_random_tbptt8_v1",
        "state_delta_contract": "joint7_gripper2_unclipped_q01q99_delta_h15_v1",
        "world_state_loss_weight": 1.0,
        "main_vision_joint_trained": False,
        "main_vision_base_sha256": eval_libero_closedloop._sha256(dino_path),
        "wmrm_target_teacher": "shared_online_dino_block23_stopgrad_v1",
    }

    class TinyMockPolicy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.dummy = torch.nn.Linear(1, 1)

    monkeypatch.setattr(eval_libero_closedloop, "VACompoundPolicy", TinyMockPolicy)
    checkpoint = {
        "config": config,
        "model": TinyMockPolicy(config_obj).state_dict(),
        "training_contract": contract,
        "global_step": 50,
    }
    torch.save(checkpoint, ckpt_path)

    qwen_dir = tmp_path / "qwen"
    qwen_dir.mkdir()

    # Sentinel to stop evaluation right after all contract validation passes
    class ValidationPassedSentinel(Exception):
        pass

    def stop_at_backbone(*args, **kwargs):
        raise ValidationPassedSentinel("validation_passed")

    monkeypatch.setattr(
        eval_libero_closedloop.TimmActionVisionBackbone,
        "from_pretrained",
        stop_at_backbone,
    )
    monkeypatch.setattr(eval_libero_closedloop, "_language_caches", lambda *args, **kwargs: ({0: None, 1: None}, {}))

    args = [
        "--checkpoint", str(ckpt_path),
        "--data", str(data_path),
        "--main-vision-checkpoint", str(dino_path),
        "--qwen", str(qwen_dir),
        "--trials-per-task", "1",
        "--device", "cpu",
    ]
    monkeypatch.setattr("sys.argv", ["eval_libero_closedloop.py"] + args)
    with pytest.raises(ValidationPassedSentinel):
        eval_libero_closedloop.main()

    # Reject if memory_contract is old episode_tbptt8_v1 with ALL_STARTS data
    bad_ckpt = copy.deepcopy(checkpoint)
    bad_ckpt["training_contract"]["memory_contract"] = "episode_tbptt8_v1"
    bad_ckpt_path = tmp_path / "bad_ckpt.pt"
    torch.save(bad_ckpt, bad_ckpt_path)
    monkeypatch.setattr("sys.argv", ["eval_libero_closedloop.py", "--checkpoint", str(bad_ckpt_path), "--data", str(data_path), "--main-vision-checkpoint", str(dino_path), "--qwen", str(qwen_dir), "--trials-per-task", "1", "--device", "cpu"])
    with pytest.raises(ValueError, match="incomplete four-suite checkpoint contract"):
        eval_libero_closedloop.main()

    # Reject if sampling_contract is missing with ALL_STARTS data
    bad_ckpt2 = copy.deepcopy(checkpoint)
    del bad_ckpt2["training_contract"]["sampling_contract"]
    bad_ckpt2_path = tmp_path / "bad_ckpt2.pt"
    torch.save(bad_ckpt2, bad_ckpt2_path)
    monkeypatch.setattr("sys.argv", ["eval_libero_closedloop.py", "--checkpoint", str(bad_ckpt2_path), "--data", str(data_path), "--main-vision-checkpoint", str(dino_path), "--qwen", str(qwen_dir), "--trials-per-task", "1", "--device", "cpu"])
    with pytest.raises(ValueError, match="incomplete four-suite checkpoint contract"):
        eval_libero_closedloop.main()
