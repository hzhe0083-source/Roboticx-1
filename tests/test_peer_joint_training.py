"""Focused contracts for peer VA/World joint training."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from train import (
    backward_peer_joint_losses,
    main,
    next_peer_joint_batches,
    parse_args,
    prepare_visual_world_action_ranking,
    rollout_policy,
    validate_args,
    validate_peer_data_isolation,
)


def _payload(episodes: list[int]) -> dict[str, torch.Tensor]:
    rows = len(episodes)
    return {
        "episode_id": torch.tensor(episodes, dtype=torch.long),
        "instruction_id": torch.tensor([0, 0, 16, 16], dtype=torch.long),
        "actions": torch.zeros(rows, 4, 6, 4),
        "proprio": torch.zeros(rows, 4, 3),
        "language_hidden": torch.zeros(rows, 5, 12),
    }


def _peer_args(tmp_path: Path, *extra: str) -> list[str]:
    va_data = tmp_path / "va.pt"
    world_data = tmp_path / "world.pt"
    checkpoint = tmp_path / "dino.pt"
    manifest = tmp_path / "world_split.json"
    for path in (va_data, world_data, checkpoint, manifest):
        path.write_bytes(b"contract fixture")
    return [
        "--va-data",
        str(va_data),
        "--world-data",
        str(world_data),
        "--visual-world-supervision",
        "--world-split-manifest",
        str(manifest),
        "--wam4va",
        "--va-world-mode",
        "peer_sync_h6",
        "--wmrm-inject",
        "all",
        "--wmrm-target",
        "dino",
        "--wmrm-cycle-steps",
        "6",
        "--wmrm-adep-weight",
        "0",
        "--va-layers",
        "8",
        "--wmrm-predictor",
        "st_blocks",
        "--wmrm-predictor-depth",
        "6",
        "--wmrm-predictor-width",
        "384",
        "--wmrm-predictor-heads",
        "12",
        "--wmrm-map-size",
        "16",
        "--wmrm-map-channels",
        "1024",
        "--wmrm-world-grid",
        "16",
        "--dino-main-vision",
        "--main-vision-checkpoint",
        str(checkpoint),
        "--main-vision-grid",
        "16",
        "--main-vision-frames",
        "4",
        "--sequence-length",
        "4",
        "--min-sequence-length",
        "4",
        "--single-task",
        "--task-sampling",
        "balanced",
        "--flow-prefix-steps",
        "6",
        *extra,
    ]


def test_peer_datasets_are_episode_disjoint_but_task_aligned() -> None:
    va_payload = _payload([10, 10, 11, 11])
    world_payload = _payload([20, 20, 21, 21])

    identity = validate_peer_data_isolation(va_payload, world_payload)

    assert identity["va_episode_count"] == 2
    assert identity["world_episode_count"] == 2
    assert identity["task_ids"] == [0, 16]
    with pytest.raises(ValueError, match="episode-disjoint"):
        validate_peer_data_isolation(va_payload, _payload([11, 11, 21, 21]))


def test_peer_cli_requires_joint_streams_and_rejects_separated_phases(
    tmp_path: Path,
) -> None:
    args = parse_args(_peer_args(tmp_path))
    validate_args(args)
    assert args.data is None
    assert args.va_data != args.world_data

    for phase_flag in ("--va-only", "--world-only"):
        with pytest.raises(ValueError, match="jointly optimized"):
            validate_args(parse_args(_peer_args(tmp_path, phase_flag)))
    with pytest.raises(ValueError, match="same-snapshot action-dependence"):
        validate_args(
            parse_args(_peer_args(tmp_path, "--wmrm-adep-weight", "0.1"))
        )


def test_peer_p2_requires_one_consistent_planning_stride(tmp_path: Path) -> None:
    p2 = _peer_args(
        tmp_path,
        "--planning-stride",
        "2",
        "--control-stride",
        "2",
        "--wmrm-cycle-steps",
        "2",
        "--flow-prefix-steps",
        "2",
    )
    args = parse_args(p2)
    validate_args(args)
    assert args.planning_stride == 2
    assert args.control_stride == args.wmrm_cycle_steps == args.flow_prefix_steps == 2

    with pytest.raises(ValueError, match="flow-prefix-steps"):
        validate_args(parse_args([*p2, "--flow-prefix-steps", "6"]))
    with pytest.raises(ValueError, match="control-stride"):
        validate_args(parse_args([*p2, "--control-stride", "3"]))


def test_p2_world_ranking_uses_transition_prefix_but_keeps_h6_labels() -> None:
    payload = _payload([10, 11, 20, 21])
    payload["actions"] = torch.arange(
        4 * 4 * 6 * 4, dtype=torch.float32
    ).reshape(4, 4, 6, 4)
    payload["action_valid_mask"] = torch.ones(4, 4, 6, dtype=torch.bool)

    prepare_visual_world_action_ranking(payload, planning_stride=2)

    assert payload["actions"].shape[-2:] == (6, 4)
    assert payload["world_rank_shuffle_action"].shape == (4, 3, 2, 4)
    source = inspect.getsource(rollout_policy)
    assert ": model.config.action_horizon" in source
    assert "logged_chunk.to(dtype=final_readout.dtype)" in source


def test_h15_world_ranking_uses_full_endpoint_action_chunk(tmp_path: Path) -> None:
    args = parse_args(
        _peer_args(
            tmp_path,
            "--planning-stride",
            "2",
            "--control-stride",
            "2",
            "--wmrm-cycle-steps",
            "15",
            "--flow-prefix-steps",
            "2",
        )
    )
    validate_args(args)
    payload = _payload([10, 11, 20, 21])
    payload["actions"] = torch.arange(
        4 * 4 * 15 * 4, dtype=torch.float32
    ).reshape(4, 4, 15, 4)
    payload["action_valid_mask"] = torch.ones(4, 4, 15, dtype=torch.bool)
    payload["world_target_valid_mask"] = torch.ones(4, 4, dtype=torch.bool)
    payload["metadata"] = {"world_target_horizon": 15}

    prepare_visual_world_action_ranking(payload, planning_stride=2)

    assert payload["world_rank_shuffle_action"].shape == (4, 4, 15, 4)
    assert payload["world_rank_shuffle_mask"].shape == (4, 4)


def test_joint_streams_fetch_independent_batches_and_restart_independently() -> None:
    va_loader = [{"stream": "va", "value": torch.tensor(2.0)}]
    world_loader = [{"stream": "world", "value": torch.tensor(3.0)}]
    va_iterator = iter(va_loader)
    world_iterator = iter(world_loader)

    va_batch, world_batch, va_iterator, world_iterator = next_peer_joint_batches(
        va_iterator, va_loader, world_iterator, world_loader
    )
    assert va_batch is not world_batch
    assert va_batch["stream"] == "va"
    assert world_batch["stream"] == "world"

    restarted_va, restarted_world, _, _ = next_peer_joint_batches(
        va_iterator, va_loader, world_iterator, world_loader
    )
    assert restarted_va["stream"] == "va"
    assert restarted_world["stream"] == "world"


def test_joint_losses_accumulate_both_stream_gradients_before_one_step() -> None:
    shared = torch.nn.Parameter(torch.tensor(1.0))
    va_private = torch.nn.Parameter(torch.tensor(1.0))
    world_private = torch.nn.Parameter(torch.tensor(1.0))

    class CountingSGD(torch.optim.SGD):
        def __init__(self, parameters) -> None:
            super().__init__(parameters, lr=0.1)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    optimizer = CountingSGD([shared, va_private, world_private])
    va_loss = 2.0 * shared + 5.0 * va_private

    def world_forward() -> torch.Tensor:
        # World builds its independent graph only after the VA graph has been
        # consumed, while keeping the accumulated VA gradients for this step.
        torch.testing.assert_close(shared.grad, torch.tensor(2.0))
        return 3.0 * shared + 7.0 * world_private

    optimizer.zero_grad(set_to_none=True)
    backward_peer_joint_losses(va_loss, world_forward)

    torch.testing.assert_close(shared.grad, torch.tensor(5.0))
    torch.testing.assert_close(va_private.grad, torch.tensor(5.0))
    torch.testing.assert_close(world_private.grad, torch.tensor(7.0))
    optimizer.step()
    assert optimizer.step_calls == 1
    torch.testing.assert_close(shared, torch.tensor(0.5))
    torch.testing.assert_close(va_private, torch.tensor(0.5))
    torch.testing.assert_close(world_private, torch.tensor(0.3))


def test_training_loop_wires_separate_va_and_world_objectives_into_joint_backward() -> None:
    source = inspect.getsource(main)
    assert "next_peer_joint_batches(" in source
    assert 'objective="va" if dual_peer_data else "joint"' in source
    assert 'objective="world"' in source
    assert "backward_peer_joint_losses(" in source
