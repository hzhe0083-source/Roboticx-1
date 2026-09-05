import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from train_libero import _parser, _prepare_batch, _atomic_checkpoint, _fresh_config
from va_compound import VACompoundConfig


def test_cli_execution_flags_defaults():
    parser = _parser()
    args = parser.parse_args(["train"])
    assert args.episode_microbatch == 1
    assert args.joint_observation_chunk == 0
    assert args.epoch_offset == 0


def test_cli_execution_flags_custom():
    parser = _parser()
    args = parser.parse_args([
        "train",
        "--episode-microbatch", "4",
        "--joint-observation-chunk", "16",
        "--epoch-offset", "2",
    ])
    assert args.episode_microbatch == 4
    assert args.joint_observation_chunk == 16
    assert args.epoch_offset == 2


def test_prepare_batch_forwards_joint_observation_chunk(monkeypatch):
    called_kwargs = {}

    def mock_encode_dual_tower_batch(frames, instructions, vision, text, fusion, device, **kwargs):
        called_kwargs.update(kwargs)
        batch = len(instructions)
        seq = frames.shape[1] if hasattr(frames, "shape") else len(frames[0])
        return (
            torch.zeros(batch, seq, 16 * 16, 128),
            torch.zeros(batch, seq, 10, 128),
            torch.ones(batch, seq, 10, dtype=torch.bool),
        )

    monkeypatch.setattr("train_libero.encode_dual_tower_batch", mock_encode_dual_tower_batch)

    raw = {
        "frames": torch.zeros(1, 8, 5, 224, 224, 3),
        "instruction_id": torch.tensor([0]),
        "actions": torch.zeros(1, 8, 15, 7),
        "previous_action": torch.zeros(1, 8, 7),
    }
    joint_fusion = MagicMock()
    joint_text = MagicMock()
    joint_tasks = ["task 0"]
    vision = MagicMock()

    # Zero chunk (default) -> observation_chunk_size not passed
    called_kwargs.clear()
    _prepare_batch(
        raw.copy(),
        vision=vision,
        device=torch.device("cpu"),
        encode_batch=16,
        world=False,
        prev_dropout=1.0,
        layerwise_cross_modal=False,
        joint_text=joint_text,
        joint_tasks=joint_tasks,
        joint_fusion=joint_fusion,
        joint_observation_chunk=0,
    )
    assert called_kwargs == {"grid": 16}
    assert "observation_chunk_size" not in called_kwargs

    # Non-zero chunk -> observation_chunk_size passed
    called_kwargs.clear()
    _prepare_batch(
        raw.copy(),
        vision=vision,
        device=torch.device("cpu"),
        encode_batch=16,
        world=False,
        prev_dropout=1.0,
        layerwise_cross_modal=False,
        joint_text=joint_text,
        joint_tasks=joint_tasks,
        joint_fusion=joint_fusion,
        joint_observation_chunk=8,
    )
    assert called_kwargs == {"grid": 16, "observation_chunk_size": 8}


def test_atomic_checkpoint_execution_config(tmp_path):
    config = _fresh_config(
        architecture_version="dual_tower_h15_v1",
    )
    model = MagicMock()
    model.state_dict.return_value = {}
    text_backbone = MagicMock()
    text_backbone.named_parameters.return_value = []
    vision = MagicMock()
    vision.named_parameters.return_value = []
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {}
    optimizer.param_groups = [{}, {}, {"lr": 1e-6}, {"lr": 1e-6}]
    sampler = MagicMock()
    sampler.state_dict.return_value = {}
    world_sampler = MagicMock()
    world_sampler.state_dict.return_value = {}

    ckpt_path = tmp_path / "test.pt"
    _atomic_checkpoint(
        ckpt_path,
        config=config,
        model=model,
        text_backbone=text_backbone,
        vision=vision,
        optimizer=optimizer,
        step=1,
        total_steps=10,
        stage1_steps=0,
        encode_batch=16,
        qwen_sha256="qsha",
        dino_sha256="dsha",
        data_sha256="datasha",
        run_metadata={"suites": ["libero_10"], "n_tasks": 1, "task_specs": [{"local_task_id": 0}]},
        pcgrad_forward_grouping="grouping",
        source_checkpoint="source",
        source_global_step=0,
        sampler=sampler,
        world_sampler=world_sampler,
        execution_config={"episode_microbatch": 2, "joint_observation_chunk": 4},
    )

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "execution_config" in saved["training_contract"]
    assert saved["training_contract"]["execution_config"] == {
        "episode_microbatch": 2,
        "joint_observation_chunk": 4,
    }
