"""Stream complete-task microbatches with one global denominator."""
import torch


def task_microbatch_forward(forward, raw, task_ids, limit, topology, device):
    if len(task_ids) != 1:
        raise ValueError('gradient microbatch currently requires single-task training')
    size = len(raw['instruction_id'])
    count = raw['decision_count'].sum().to(device=device, dtype=torch.float32)
    if topology.is_distributed:
        import torch.distributed as dist
        dist.all_reduce(count)
    def generate():
        for start in range(0, size, limit):
            chunk = {k: v[start:start + limit] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == size else v
                     for k, v in raw.items()}
            local = chunk['decision_count'].sum().to(device=device, dtype=torch.float32)
            losses = forward(task_raw=chunk, group_ids=task_ids)
            if len(losses) != 1:
                raise ValueError('single-task microbatch produced multiple objectives')
            yield losses[0] * (local * topology.world_size / count.clamp_min(1))
            del losses, chunk
    return generate
