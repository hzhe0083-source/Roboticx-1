import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace

from va_compound.training.gradients import backward_pcgrad


def _worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        marker = torch.nn.Parameter(torch.tensor(0.0))
        topology = SimpleNamespace(is_distributed=True, world_size=2)

        def forward():
            first = parameter.square() * 2 if rank == 0 else marker * 0
            second = parameter.square() * 2 if rank == 1 else marker * 0
            return [first, second]

        backward_pcgrad([forward], [("weight", parameter), ("marker", marker)],
                        topology=topology, allow_inactive_ranks=True)
        torch.save(parameter.grad, f"{output}/{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_pcgrad_handles_exhausted_episode_rank(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path}/rendezvous", str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        torch.testing.assert_close(torch.load(tmp_path / f"{rank}.pt", weights_only=True), torch.tensor(4.0))
