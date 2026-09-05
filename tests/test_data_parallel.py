from __future__ import annotations

import torch
from torch import nn

from va_compound.data_parallel import (
    DATA_PARALLEL_CONTRACT,
    WorldTopology,
    reduce_update_gradients,
    resolve_world_topology,
)


def test_single_process_topology_is_the_default(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    topology = resolve_world_topology()
    assert topology == WorldTopology(rank=0, world_size=1, local_rank=0)
    assert topology.is_primary
    assert not topology.is_distributed


def test_torchrun_env_resolves_rank_layout(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    topology = resolve_world_topology()
    assert topology.rank == 1
    assert topology.world_size == 2
    assert topology.local_rank == 1
    assert not topology.is_primary
    assert topology.is_distributed


def test_reduce_is_a_noop_without_a_process_group() -> None:
    module = nn.Linear(4, 4)
    loss = module(torch.ones(2, 4)).sum()
    loss.backward()
    before = module.weight.grad.detach().clone()
    reduce_update_gradients(
        list(module.named_parameters()), WorldTopology()
    )
    assert torch.equal(module.weight.grad, before)


def test_contract_name_is_stable() -> None:
    assert DATA_PARALLEL_CONTRACT == "manual_post_backward_grad_allreduce_v1"
