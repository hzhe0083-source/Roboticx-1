"""Single-host multi-GPU data parallelism for the peer-synchronous trainer.

Why not ``DistributedDataParallel``: a peer step runs two sequential backwards
(VA, then World) into one optimizer update.  DDP starts its gradient allreduce
when the first backward's autograd graph completes, so any parameter that only
receives gradient from the VA backward keeps a rank-local gradient forever and
the replicas silently diverge -- no error, just two different models.  This
module instead reduces every optimizer-visible gradient exactly once, after all
backwards have run.  That ordering also keeps the existing global-norm clipping
honest: ``clip_update_gradients`` then sees the true global gradient instead of
a rank-local one, so the logged ``grad=`` norm stays comparable to single-card
runs.

Why not ``DataParallel``: a step here is an orchestration of many module calls
exchanging ``WAMState`` dataclasses, not one ``forward``, so its scatter/gather
has nothing to wrap.

Exactness caveat, deliberately not hidden: per-rank losses are means over each
rank's own valid transitions, and averaging two such means is not identical to
one mean over the union when ``action_valid`` masks differ between the halves.
Reducing gradients rather than losses is the standard trade (identical to how
DDP behaves) but it means a 2x24 run is not bit-equivalent to a 1x48 run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

# Gradient buckets are capped instead of fusing everything into one flat buffer:
# batch 24 already sits at 42.8 GiB of a 46.1 GiB L20, so a full-size flat copy
# of the gradients would not fit.  128 MiB still amortizes collective latency.
BUCKET_BYTES = 128 << 20

DATA_PARALLEL_CONTRACT = "manual_post_backward_grad_allreduce_v1"


@dataclass(frozen=True)
class WorldTopology:
    """Resolved process-group layout for one training process."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_primary(self) -> bool:
        """Only the primary rank may write logs, checkpoints, or eval output."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def resolve_world_topology(environ: dict[str, str] | None = None) -> WorldTopology:
    """Read the torchrun-provided layout, defaulting to single-process."""
    env = os.environ if environ is None else environ
    world_size = int(env.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    rank = int(env.get("RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK {rank} outside WORLD_SIZE {world_size}")
    local_rank = int(env.get("LOCAL_RANK", str(rank)))
    return WorldTopology(rank=rank, world_size=world_size, local_rank=local_rank)


def initialize(topology: WorldTopology, device: torch.device) -> None:
    """Join the NCCL process group; a no-op for single-process runs."""
    if not topology.is_distributed:
        return
    if device.type != "cuda":
        raise ValueError("multi-process data parallelism requires CUDA devices")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this build")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.get_world_size() != topology.world_size:
        raise RuntimeError(
            f"process group world size {dist.get_world_size()} does not match "
            f"WORLD_SIZE {topology.world_size}"
        )


def shutdown(topology: WorldTopology) -> None:
    if topology.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(topology: WorldTopology) -> None:
    if topology.is_distributed:
        dist.barrier()


def _assert_gradient_symmetry(
    named_parameters: list[tuple[str, Tensor]],
    topology: WorldTopology,
    device: torch.device,
) -> None:
    """Fail loudly when ranks disagree on which parameters carry gradient.

    A collective is only well defined if every rank issues the same sequence of
    them.  Ranks that disagree here would otherwise hang for hours rather than
    crash, which is the worse failure for a long run.
    """
    present = torch.tensor(
        [parameter.grad is not None for _, parameter in named_parameters],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(present)
    disagreement = (present != 0) & (present != topology.world_size)
    if bool(disagreement.any()):
        offenders = [
            named_parameters[index][0]
            for index in torch.nonzero(disagreement).flatten().tolist()
        ][:8]
        raise RuntimeError(
            "ranks disagree on which parameters received gradient; "
            f"first offenders: {offenders}"
        )


def reduce_update_gradients(
    named_parameters: list[tuple[str, Tensor]],
    topology: WorldTopology,
) -> None:
    """Average optimizer-visible gradients across ranks, in place.

    Call this after every backward for the step and before gradient validation
    and clipping, so the norms that get logged and clipped are global.
    """
    if not topology.is_distributed:
        return
    gradients = [
        parameter.grad
        for _, parameter in named_parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return
    device = gradients[0].device
    _assert_gradient_symmetry(named_parameters, topology, device)
    scale = 1.0 / float(topology.world_size)
    bucket: list[Tensor] = []
    bucket_bytes = 0

    def flush(entries: list[Tensor]) -> None:
        if not entries:
            return
        flat = torch.cat([tensor.reshape(-1) for tensor in entries])
        dist.all_reduce(flat)
        flat.mul_(scale)
        offset = 0
        for tensor in entries:
            count = tensor.numel()
            tensor.copy_(flat[offset : offset + count].view_as(tensor))
            offset += count

    for gradient in gradients:
        span = gradient.numel() * gradient.element_size()
        if bucket and bucket_bytes + span > BUCKET_BYTES:
            flush(bucket)
            bucket, bucket_bytes = [], 0
        bucket.append(gradient)
        bucket_bytes += span
    flush(bucket)


def all_ranks_failed(topology: WorldTopology, failed: bool, device: torch.device) -> bool:
    """Agree on aborting so one rank's NaN cannot hang the other rank."""
    if not topology.is_distributed:
        return failed
    flag = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag)
    return bool(flag.item() > 0)


def reduce_scalar_mean(value: float, topology: WorldTopology, device: torch.device) -> float:
    """Average a logged scalar so rank-0 logs describe the global batch."""
    if not topology.is_distributed:
        return value
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return float(tensor.item()) / float(topology.world_size)


def broadcast_parameters(parameters: list[Tensor], topology: WorldTopology) -> None:
    """Copy rank-0 weights to every replica before the first step."""
    if not topology.is_distributed:
        return
    for parameter in parameters:
        dist.broadcast(parameter.data, src=0)
