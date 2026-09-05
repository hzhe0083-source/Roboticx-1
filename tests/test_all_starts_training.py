import argparse
from pathlib import Path
import pytest
import torch

from train_libero import (
    _parser,
    _prepare_batch,
    _atomic_checkpoint,
    train,
    preflight,
)
from va_compound.data.all_starts import (
    AllStartsStreamDataset,
    AllStartsWindowBatchSampler,
    MEMORY_CONTRACT as ALL_STARTS_MEMORY_CONTRACT,
    SAMPLING_CONTRACT as ALL_STARTS_SAMPLING_CONTRACT,
)
from va_compound.data.libero import (
    ALL_STARTS_DATA_CONTRACT,
    H15_DATA_CONTRACT,
    EXECUTION_HORIZON,
    WORLD_HORIZON,
    SEQUENCE,
    VISION_OFFSETS,
    _validate_data,
    _validate_run_schedule,
)
from va_compound.training.episode_memory import EpisodeMemoryBank
from va_compound import VACompoundConfig, VACompoundPolicy


def _make_dummy_all_starts_payload():
    # 2 demos, length 25 each => decisions = 25 - 15 = 10 each => total 20 rows
    n_demos = 2
    episode_lengths = [25, 25]
    N = sum(L - EXECUTION_HORIZON for L in episode_lengths)
    
    raw_actions = torch.randn(2, 25, 7).clamp(-1, 1)
    actions = torch.stack([raw_actions[ep, d + 1:d + 16] for ep in range(2) for d in range(10)])[:, None]
    proprio = torch.randn(N, 1, 9)
    previous_action = torch.zeros(N, 1, 7)
    
    # ensure previous_action continuity: previous_action[d] == actions[d-1, 0, 0]
    # ep 0: rows 0..9
    for i in range(1, 10):
        previous_action[i, 0] = actions[i - 1, 0, 0]
    # ep 1: rows 10..19
    for i in range(11, 20):
        previous_action[i, 0] = actions[i - 1, 0, 0]
        
    world_state_delta = torch.randn(N, 1, 9)
    action_valid_mask = torch.ones(N, 1, 15, dtype=torch.bool)
    world_target_valid_mask = torch.ones(N, 1, dtype=torch.bool)
    world_rank_shuffle_action = torch.randn(N, 1, 15, 7)
    world_rank_shuffle_mask = torch.ones(N, 1, dtype=torch.bool)
    
    crop_start = torch.cat([torch.arange(10, dtype=torch.long), torch.arange(10, dtype=torch.long)])
    episode_id = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    pair_id = torch.arange(N, dtype=torch.long)
    anchor_eligible = torch.ones(N, dtype=torch.bool)
    language_hidden = torch.zeros(N, 1, 1, dtype=torch.float16)
    language_mask = torch.ones(N, 1, dtype=torch.bool)
    instruction_id = torch.zeros(N, dtype=torch.long)
    
    frame_refs = []
    world_target_frame_refs = []
    for ep in range(n_demos):
        for d in range(10):
            agent_refs = [max(0, d + int(off)) for off in VISION_OFFSETS]
            wrist_ref = 25 + d
            frame_refs.append(("task0", ep, [agent_refs + [wrist_ref]]))
            world_target_frame_refs.append(("task0", ep, [[wrist_ref + WORLD_HORIZON]]))
            
    payload = {
        "actions": actions,
        "proprio": proprio,
        "previous_action": previous_action,
        "world_state_delta": world_state_delta,
        "action_valid_mask": action_valid_mask,
        "world_target_valid_mask": world_target_valid_mask,
        "world_rank_shuffle_action": world_rank_shuffle_action,
        "world_rank_shuffle_mask": world_rank_shuffle_mask,
        "crop_start": crop_start,
        "episode_id": episode_id,
        "pair_id": pair_id,
        "anchor_eligible": anchor_eligible,
        "language_hidden": language_hidden,
        "language_mask": language_mask,
        "instruction_id": instruction_id,
        "frame_refs": frame_refs,
        "world_target_frame_refs": world_target_frame_refs,
        "normalization": {
            "state_q01": torch.zeros(9),
            "state_q99": torch.ones(9),
            "state_delta_scale": torch.ones(9),
        },
        "metadata": {
            "contract": ALL_STARTS_DATA_CONTRACT,
            "sampling_contract": ALL_STARTS_SAMPLING_CONTRACT,
            "window_sampling": ALL_STARTS_SAMPLING_CONTRACT,
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
            "memory_contract": ALL_STARTS_MEMORY_CONTRACT,
            "world_action_donor_contract": "task_next_episode_relative_progress_v1",
            "n_tasks": 1,
            "tasks": ["task 0"],
            "task_specs": [{"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "task 0"}],
            "n_demos": 2,
            "episode_lengths": episode_lengths,
            "task_counts": [N],
            "suites": ["libero_10"],
        }
    }
    return payload


def test_prepare_batch_preserves_replay_id_and_offset():
    """Verify that _prepare_batch retains replay_id and replay_offset."""
    raw = {
        "frames": torch.zeros(1, 8, 5, 224, 224, 3, dtype=torch.uint8),
        "vision_tokens": torch.zeros(1, 8, 10, 1024),
        "language_hidden": torch.zeros(1, 8, 10, 1024),
        "language_mask": torch.ones(1, 8, 10, dtype=torch.bool),
        "proprio": torch.zeros(1, 8, 9),
        "previous_action": torch.zeros(1, 8, 7),
        "actions": torch.zeros(1, 8, 15, 7),
        "action_valid_mask": torch.ones(1, 8, 15, dtype=torch.bool),
        "instruction_id": torch.tensor([0]),
        "replay_id": torch.tensor([42]),
        "replay_offset": torch.tensor([7]),
        "crop_start": torch.tensor([7]),
        "stream_id": torch.tensor([0]),
        "stream_active": torch.tensor([True]),
        "episode_id": torch.tensor([2]),
        "decision_count": torch.tensor([8]),
        "episode_start": torch.tensor([True]),
        "episode_end": torch.tensor([False]),
    }
    batch = _prepare_batch(
        raw,
        vision=None,
        device=torch.device("cpu"),
        encode_batch=1,
        world=False,
        prev_dropout=1.0,
        layerwise_cross_modal=False,
        cached_vision=raw["vision_tokens"],
    )
    assert "replay_id" in batch
    assert "replay_offset" in batch
    assert batch["replay_id"].item() == 42
    assert batch["replay_offset"].item() == 7


def test_cli_parser_window_sampling_choices():
    parser = _parser()
    args = parser.parse_args(["train", "--window-sampling", "all_starts_random_tbptt8_v1"])
    assert args.window_sampling == "all_starts_random_tbptt8_v1"
    
    args_default = parser.parse_args(["train"])
    assert args_default.window_sampling == "episode_contiguous_p15_v1"


def test_reject_all_starts_payload_with_old_cli_default(tmp_path):
    payload = _make_dummy_all_starts_payload()
    data_path = tmp_path / "all_starts.pt"
    torch.save(payload, data_path)

    # Preflight with default window_sampling
    args = _parser().parse_args([
        "preflight",
        "--longtraj", str(tmp_path), "--qwen", str(tmp_path), "--dino", str(data_path),
        "--data", str(data_path),
        "--architecture-version", "dual_tower_h15_v1",
        "--window-sampling", "episode_contiguous_p15_v1",
    ])
    with pytest.raises(ValueError, match="all-starts data contract requires --window-sampling all_starts_random_tbptt8_v1"):
        preflight(args)


def test_validate_run_schedule_all_starts(tmp_path):
    payload = _make_dummy_all_starts_payload()
    data_path = tmp_path / "all_starts.pt"
    torch.save(payload, data_path)

    args = _parser().parse_args([
        "train",
        "--data", str(data_path),
        "--architecture-version", "dual_tower_h15_v1",
        "--window-sampling", "all_starts_random_tbptt8_v1",
        "--batch-size", "2",
        "--gpus", "1",
        "--mixed-tasks", "1",
        "--anchor-fraction", "0.0",
        "--epochs", "3", "--stage1-steps", "0",
    ])
    step_e1, total_steps, grouping = _validate_run_schedule(payload, args)
    
    sampler = AllStartsWindowBatchSampler(
        payload,
        batch_size=2,
        seed=0,
        mixed_tasks_per_batch=1,
        rank=0,
        world_size=1,
    )
    expected_lens = sampler.epoch_lengths(3)
    assert step_e1 == expected_lens[0]
    assert total_steps == sum(expected_lens)
    assert grouping == "single_task_t8_local1_deferred_v1"


def test_atomic_checkpoint_all_starts_contract(tmp_path):
    payload = _make_dummy_all_starts_payload()
    from tests.test_unified_h15_policy import make_model
    model = make_model()
    config = model.config
    
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
    
    dummy_text = DummyModule()
    dummy_vision = DummyModule()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": parameters[:1]}, {"params": parameters[1:]},
        {"params": [dummy_text.p]}, {"params": [dummy_vision.p]},
    ], lr=1e-5)
    
    sampler = AllStartsWindowBatchSampler(payload, 2, 0, 1, 0, 1)
    epoch_lengths = sampler.epoch_lengths(2)
    save_path = tmp_path / "ckpt.pt"
    
    _atomic_checkpoint(
        save_path,
        config=config,
        model=model,
        text_backbone=dummy_text,
        vision=dummy_vision,
        optimizer=optimizer,
        step=epoch_lengths[0],
        total_steps=sum(epoch_lengths),
        stage1_steps=sum(epoch_lengths),
        encode_batch=1,
        qwen_sha256="dummy",
        dino_sha256="dummy",
        data_sha256="dummy",
        run_metadata=payload["metadata"],
        pcgrad_forward_grouping="single_task_t8_local1_deferred_v1",
        source_checkpoint="fresh_dual_tower_h15_v1",
        source_global_step=-1,
        sampler=sampler,
        world_sampler=sampler,
        episode_runtime_states=[{"action": {}, "world": {}}],
        window_sampling=ALL_STARTS_SAMPLING_CONTRACT,
        epoch_lengths=epoch_lengths,
    )
    
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    contract = ckpt["training_contract"]
    assert contract["data_contract"] == ALL_STARTS_DATA_CONTRACT
    assert contract["window_sampling"] == ALL_STARTS_SAMPLING_CONTRACT
    assert contract["sampling_contract"] == ALL_STARTS_SAMPLING_CONTRACT
    assert contract["memory_contract"] == ALL_STARTS_MEMORY_CONTRACT
    assert contract["epoch_lengths"] == epoch_lengths
