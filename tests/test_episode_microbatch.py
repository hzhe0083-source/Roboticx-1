import copy

import torch

from tests.test_episode_world_training import batch, run
from tests.test_unified_h15_policy import make_model
from va_compound.training.episode_memory import EpisodeMemoryBank


def test_grouped_windows_preserve_loss_gradients_and_memory():
    torch.manual_seed(88)
    serial = make_model()
    grouped = copy.deepcopy(serial)
    grouped.episode_microbatch = 4
    banks = [EpisodeMemoryBank(), EpisodeMemoryBank()]
    for start, end in ((0, False), (30, True)):
        rows = [batch(count=2, start=start, end=end) for _ in range(4)]
        data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
        data['stream_id'] = torch.arange(4)
        data['episode_id'] = torch.arange(4)
        data['actions'] = data['actions'][:, :, :15]
        data['action_valid_mask'] = data['action_valid_mask'][:, :, :15]
        outputs, losses = [], []
        for model, bank in zip((serial, grouped), banks):
            model.zero_grad(set_to_none=True)
            output, condition = run(model, data, bank)
            loss = output.square().mean() + model.last_wmrm_loss
            loss.backward()
            bank.commit()
            outputs.append(output)
            losses.append(loss)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-4, atol=3e-6)
        torch.testing.assert_close(losses[0], losses[1], rtol=3e-4, atol=3e-6)
        for (name, p), (_, q) in zip(serial.named_parameters(), grouped.named_parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=2e-5, msg=name)
        assert banks[0].entries.keys() == banks[1].entries.keys()
        for stream in banks[0].entries:
            for left, right in zip(banks[0].entries[stream][2].layers, banks[1].entries[stream][2].layers):
                torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-6)
