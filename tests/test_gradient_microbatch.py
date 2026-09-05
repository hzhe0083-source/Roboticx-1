from types import SimpleNamespace

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
