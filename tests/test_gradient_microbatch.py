from types import SimpleNamespace

import numpy as np
import torch

from va_compound.training.gradients import backward_pcgrad
from va_compound.training.gradient_microbatch import task_microbatch_forward


def test_streamed_microbatch_matches_weighted_full_task_gradient():
    p = torch.nn.Parameter(torch.tensor(2.0))
    raw = {'instruction_id': torch.zeros(4, dtype=torch.long), 'decision_count': torch.tensor([8, 3, 5, 0]), 'x': torch.tensor([1., 3., 5., 7.])}
    def forward(task_raw, group_ids):
        weights = task_raw['decision_count'].float()
        return [((p * task_raw['x']).square() * weights).sum() / weights.sum().clamp_min(1)]
    expected = torch.autograd.grad(forward(raw, [0])[0], p)[0]
    topology = SimpleNamespace(is_distributed=False, world_size=1)
    generator = task_microbatch_forward(forward, raw, [0], 2, topology, torch.device('cpu'))
    backward_pcgrad([generator], [('p',p)], allow_single_task=True)
    torch.testing.assert_close(p.grad, expected)


def test_microbatch_slices_numpy_frames_with_tensor_metadata():
    raw = {'instruction_id': torch.zeros(4, dtype=torch.long),
           'decision_count': torch.ones(4),
           'frames': np.arange(24, dtype=np.uint8).reshape(4, 2, 3)}
    seen = []
    def forward(task_raw, group_ids):
        assert len(task_raw['frames']) == len(task_raw['instruction_id'])
        seen.append(task_raw['frames'])
        return [torch.tensor(1.)]
    topology = SimpleNamespace(is_distributed=False, world_size=1)
    losses = list(task_microbatch_forward(forward, raw, [0], 2, topology, torch.device('cpu'))())
    np.testing.assert_array_equal(np.concatenate(seen), raw['frames'])
    torch.testing.assert_close(sum(losses), torch.tensor(1.))


def test_distributed_denominator_handles_inactive_microbatch(monkeypatch):
    counts = torch.tensor([[8., 3., 0., 0.], [0., 0., 5., 0.]])
    x = torch.tensor([[1., 3., 5., 7.], [2., 4., 6., 8.]])
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda value: value.fill_(counts.sum()))
    gradients = []
    for rank in range(2):
        p = torch.nn.Parameter(torch.tensor(2.))
        raw = {'instruction_id': torch.zeros(4, dtype=torch.long),
               'decision_count': counts[rank], 'x': x[rank]}
        def forward(task_raw, group_ids):
            weights = task_raw['decision_count']
            return [((p * task_raw['x']).square() * weights).sum() / weights.sum().clamp_min(1)]
        topology = SimpleNamespace(is_distributed=True, world_size=2)
        generate = task_microbatch_forward(forward, raw, [0], 2, topology, torch.device('cpu'))
        backward_pcgrad([generate], [('p', p)], allow_single_task=True)
        gradients.append(p.grad)
    expected = (4 * x.square() * counts).sum() / counts.sum()
    torch.testing.assert_close(torch.stack(gradients).mean(), expected)
