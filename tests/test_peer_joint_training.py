"""Focused contracts for peer VA/World joint training."""

from __future__ import annotations

import inspect
from pathlib import Path
import copy

import pytest
import torch

from va_compound.world_contract import PEER_SHARED_FULL_DATA_CONTRACT

from train import (
    PeerJointBatchPrefetcher,
    TaskLocalityWeightedSampler,
    backward_peer_joint_losses,
    main,
    next_peer_joint_batches,
    parse_args,
    peer_prefetch_fill_limit,
    peer_prefetch_must_wait_for_commit,
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


def test_peer_shared_full_data_requires_identical_payloads() -> None:
    payload = _payload([10, 10, 20, 20])

    identity = validate_peer_data_isolation(
        payload,
        copy.deepcopy(payload),
        shared_full_data=True,
    )

    assert identity["contract"] == PEER_SHARED_FULL_DATA_CONTRACT
    assert identity["shared_full_data"] is True
    assert identity["shared_windows"] == 4

    changed = copy.deepcopy(payload)
    changed["actions"][0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="identical VA/World payload identity"):
        validate_peer_data_isolation(payload, changed, shared_full_data=True)


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


def test_peer_shared_full_data_explicitly_allows_one_dataset_path(
    tmp_path: Path,
) -> None:
    cli = _peer_args(tmp_path)
    va_path = cli[cli.index("--va-data") + 1]
    cli[cli.index("--world-data") + 1] = va_path

    with pytest.raises(ValueError, match="must be different files"):
        validate_args(parse_args(cli))

    args = parse_args([*cli, "--peer-shared-full-data"])
    validate_args(args)
    assert args.peer_shared_full_data is True


def test_peer_batch_prefetch_is_opt_in_and_requires_dual_streams(
    tmp_path: Path,
) -> None:
    args = parse_args(
        _peer_args(
            tmp_path,
            "--peer-batch-prefetch",
            "--peer-batch-prefetch-depth",
            "4",
        )
    )
    validate_args(args)
    assert args.peer_batch_prefetch is True
    assert args.peer_batch_prefetch_depth == 4
    with pytest.raises(ValueError, match="requires peer_sync_h6"):
        validate_args(parse_args(["--single-task", "--peer-batch-prefetch"]))
    with pytest.raises(ValueError, match="depth must be positive"):
        validate_args(
            parse_args(["--single-task", "--peer-batch-prefetch-depth", "0"])
        )


def test_runtime_integrity_checks_are_default_on_and_explicitly_disabled(
    tmp_path: Path,
) -> None:
    assert parse_args(_peer_args(tmp_path)).runtime_integrity_checks is True
    disabled = parse_args(
        _peer_args(tmp_path, "--disable-runtime-integrity-checks")
    )
    validate_args(disabled)
    assert disabled.runtime_integrity_checks is False
    assert (
        inspect.getsource(main).count(
            "runtime_integrity_checks=args.runtime_integrity_checks"
        )
        == 3
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


def test_peer_h15_p15_requires_one_consistent_replanning_cadence(
    tmp_path: Path,
) -> None:
    p15 = _peer_args(
        tmp_path,
        "--planning-stride",
        "15",
        "--control-stride",
        "15",
        "--deployment-execution-horizon",
        "15",
        "--wmrm-cycle-steps",
        "15",
        "--flow-prefix-steps",
        "15",
    )
    args = parse_args(p15)
    validate_args(args)
    assert args.planning_stride == 15
    assert (
        args.control_stride
        == args.deployment_execution_horizon
        == args.wmrm_cycle_steps
        == args.flow_prefix_steps
        == 15
    )

    with pytest.raises(ValueError, match="flow-prefix-steps"):
        validate_args(parse_args([*p15, "--flow-prefix-steps", "2"]))


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
    assert "logged_chunk.to(dtype=stage_readouts[0].dtype)" in source


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


def test_h15_p15_world_ranking_uses_full_endpoint_action_chunk() -> None:
    payload = _payload([10, 11, 20, 21])
    payload["actions"] = torch.arange(
        4 * 4 * 15 * 4, dtype=torch.float32
    ).reshape(4, 4, 15, 4)
    payload["action_valid_mask"] = torch.ones(4, 4, 15, dtype=torch.bool)
    payload["world_target_valid_mask"] = torch.ones(4, 4, dtype=torch.bool)
    payload["metadata"] = {"world_target_horizon": 15}

    prepare_visual_world_action_ranking(payload, planning_stride=15)

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


def _full_index_stream(
    seed: int, n_rows: int = 18
) -> tuple[TaskLocalityWeightedSampler, object]:
    sampler = TaskLocalityWeightedSampler(
        instruction_id=torch.arange(n_rows) % 2,
        episode_id=torch.arange(n_rows) // 2,
        task_weights=torch.ones(2),
        batch_size=4,
        seed=seed,
        sampling_mode="full",
    )
    loader = torch.utils.data.DataLoader(list(range(n_rows)), batch_sampler=sampler)
    return sampler, loader


def _collect_peer_epochs(
    prefetch_depth: int,
) -> tuple[list[list[int]], list[list[int]]]:
    va_sampler, va_loader = _full_index_stream(7)
    world_sampler, world_loader = _full_index_stream(8)
    va_iterator = iter(va_loader)
    world_iterator = iter(world_loader)
    prefetcher = None
    total_steps = 2 * len(va_sampler)
    if prefetch_depth:
        prefetcher = PeerJointBatchPrefetcher(
            va_iterator,
            va_loader,
            world_iterator,
            world_loader,
            depth=prefetch_depth,
        )
        prefetcher.fill(
            peer_prefetch_fill_limit(
                total_steps, va_sampler, world_sampler
            )
        )

    va_seen: list[list[int]] = []
    world_seen: list[list[int]] = []
    for step in range(total_steps):
        if prefetcher is None:
            va_batch, world_batch, va_iterator, world_iterator = (
                next_peer_joint_batches(
                    va_iterator, va_loader, world_iterator, world_loader
                )
            )
        else:
            va_batch, world_batch, va_iterator, world_iterator = prefetcher.result()
        va_seen.append(va_batch.tolist())
        world_seen.append(world_batch.tolist())

        remaining_steps = total_steps - step - 1
        defer = False
        if prefetcher is not None and remaining_steps:
            fill_limit = peer_prefetch_fill_limit(
                remaining_steps,
                va_sampler,
                world_sampler,
                current_batch_consumed=True,
            )
            prefetcher.fill(fill_limit)
            defer = fill_limit == 0 and peer_prefetch_must_wait_for_commit(
                va_sampler, world_sampler
            )
        va_sampler.advance()
        world_sampler.advance()
        if prefetcher is not None and defer:
            prefetcher.fill(
                peer_prefetch_fill_limit(
                    remaining_steps, va_sampler, world_sampler
                )
            )

    if prefetcher is not None:
        prefetcher.close()
    return va_seen, world_seen


def test_peer_depth4_prefetch_preserves_order_across_two_epochs() -> None:
    synchronous = _collect_peer_epochs(prefetch_depth=0)
    prefetched = _collect_peer_epochs(prefetch_depth=4)

    assert prefetched == synchronous
    for stream in prefetched:
        for epoch in range(2):
            batches = stream[epoch * 5 : (epoch + 1) * 5]
            assert sorted(index for batch in batches for index in batch) == list(range(18))
            assert [len(batch) for batch in batches] == [4, 4, 4, 4, 2]


def test_peer_prefetch_queue_reaches_configured_depth_four() -> None:
    va_sampler, va_loader = _full_index_stream(9, n_rows=20)
    world_sampler, world_loader = _full_index_stream(10, n_rows=20)
    prefetcher = PeerJointBatchPrefetcher(
        iter(va_loader),
        va_loader,
        iter(world_loader),
        world_loader,
        depth=4,
    )
    assert prefetcher.fill(
        peer_prefetch_fill_limit(5, va_sampler, world_sampler)
    ) == 4
    assert prefetcher.queued_batches == 4
    for _ in range(4):
        prefetcher.result()
    prefetcher.close()


def test_uncommitted_prefetched_batch_repeats_after_exact_resume() -> None:
    va_sampler, va_loader = _full_index_stream(11, n_rows=20)
    world_sampler, world_loader = _full_index_stream(12, n_rows=20)
    prefetcher = PeerJointBatchPrefetcher(
        iter(va_loader),
        va_loader,
        iter(world_loader),
        world_loader,
        depth=4,
    )
    prefetcher.fill(4)
    prefetcher.result()
    va_sampler.advance()
    world_sampler.advance()
    va_state = va_sampler.state_dict()
    world_state = world_sampler.state_dict()
    expected_va, expected_world, _, _ = prefetcher.result()
    prefetcher.close()

    resumed_va, resumed_va_loader = _full_index_stream(11, n_rows=20)
    resumed_world, resumed_world_loader = _full_index_stream(12, n_rows=20)
    resumed_va.load_state_dict(va_state)
    resumed_world.load_state_dict(world_state)
    actual_va, actual_world, _, _ = next_peer_joint_batches(
        iter(resumed_va_loader),
        resumed_va_loader,
        iter(resumed_world_loader),
        resumed_world_loader,
    )
    assert actual_va.tolist() == expected_va.tolist()
    assert actual_world.tolist() == expected_world.tolist()


def test_short_run_fill_limit_leaves_no_extra_fetch_for_close() -> None:
    class CountingLoader:
        def __init__(self, stream: str) -> None:
            self.stream = stream
            self.fetches = 0

        def __iter__(self):
            index = 0
            while True:
                self.fetches += 1
                yield {"stream": self.stream, "index": index}
                index += 1

    va_loader = CountingLoader("va")
    world_loader = CountingLoader("world")
    prefetcher = PeerJointBatchPrefetcher(
        iter(va_loader),
        va_loader,
        iter(world_loader),
        world_loader,
        depth=4,
    )
    prefetcher.fill(peer_prefetch_fill_limit(2))
    assert prefetcher.queued_batches == 2
    prefetcher.result()
    prefetcher.fill(peer_prefetch_fill_limit(1))
    prefetcher.result()
    prefetcher.fill(peer_prefetch_fill_limit(0))
    assert prefetcher.queued_batches == 0
    prefetcher.close()
    assert va_loader.fetches == world_loader.fetches == 2


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
    assert "peer joint World stream requires DINO-main dense metric encoding" not in source
    assert "if config.dino_dense_metric:" in source
    assert "metric_tokens = None" in source
    assert "metric_g = None" in source
    assert source.count("task_order_seed=(") == 3
    assert 'getattr(args, "peer_shared_full_data", False)' in source


def test_training_hot_path_uses_clip_preclip_norm_without_full_tensor_scans() -> None:
    source = inspect.getsource(main)
    assert "raw_gradient_norm = validate_preclip_gradient_norms(" in source
    assert "validate_update_gradients(" not in source
    # One full optimizer/parameter/state scan remains before training starts.
    assert source.count("validate_optimizer_update_state(optimizer)") == 1
    assert source.count(
        "validate_optimizer_update_state(optimizer, validate_values=False)"
    ) >= 3
