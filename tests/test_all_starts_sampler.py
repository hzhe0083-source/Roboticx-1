from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from va_compound.data.all_starts import AllStartsStreamDataset, AllStartsWindowBatchSampler
from va_compound.training.episode_memory import EpisodeMemoryBank


def payload():
    lengths = [137, 171, 16, 29, 288]
    episodes = torch.cat([torch.full((n - 15,), i) for i, n in enumerate(lengths)])
    starts = torch.cat([torch.arange(n - 15) for n in lengths])
    n = len(starts)
    return dict(episode_id=episodes, crop_start=starts,
                instruction_id=torch.zeros(n, dtype=torch.long),
                actions=torch.arange(n).float()[:, None, None, None].expand(n, 1, 15, 7),
                action_valid_mask=torch.ones(n, 1, 15, dtype=torch.bool),
                world_state_delta=torch.ones(n, 1, 9))


def test_full_coverage_random_boundaries_and_exact_sampler_resume():
    p = payload()
    schedules = []
    for epoch in range(3):
        visits = []
        windows = []
        for rank in range(2):
            s = AllStartsWindowBatchSampler(p, 4, 42, 1, rank, 2)
            s.epoch = epoch
            schedule = list(s)
            for b in schedule:
                for rows, stream, active in b:
                    if active:
                        visits.extend(rows)
                        windows.append(rows)
                        d = p['crop_start'][list(rows)]
                        assert (d[1:] - d[:-1] == 15).all()
                        assert p['episode_id'][list(rows)].unique().numel() == 1
            s.advance(2)
            state = s.state_dict()
            restored = AllStartsWindowBatchSampler(p, 4, 42, 1, rank, 2)
            restored.load_state_dict(state)
            assert list(s) == list(restored)
            before = s.state_dict()
            assert len(s.epoch_lengths(4)) == 4
            assert before == s.state_dict()
        assert Counter(visits) == Counter(range(len(p['episode_id'])))
        schedules.append(sorted(windows))
    assert schedules[0] != schedules[1]


class FakeDataset:
    def __init__(self):
        self.payload = payload()
    def __len__(self):
        return len(self.payload['episode_id'])
    def __getitem__(self, row):
        return {key: value[row] for key, value in self.payload.items()}


def test_window_materialization_padding_and_replay_identity():
    d = AllStartsStreamDataset(FakeDataset())
    item = d[((2, 17, 32), 1, True)]
    assert item['actions'].shape == (8, 15, 7)
    assert item['actions'][:3, 0, 0].tolist() == [2, 17, 32]
    assert item['replay_id'] == 2 and item['replay_offset'] == 2
    assert item['episode_start'] and not item['episode_end']
    assert item['decision_valid_mask'].sum() == 3
    assert not item['action_valid_mask'][3:].any()
    assert not item['world_state_delta'][3:].any()
    inactive = d[((2, 17, 32), 1, False)]
    assert not inactive['action_valid_mask'].any()
    assert inactive['decision_count'] == 0
    with pytest.raises(ValueError, match='boundary'):
        d[((2, 18), 1, True)]


def test_offset_memory_isolation_and_legacy_rejection():
    bank = EpisodeMemoryBank(replay_offsets=True)
    args = dict(device='cpu', dtype=torch.float32)
    assert bank.begin(0, 32, 2, True, **args) is None
    memory = SimpleNamespace(detach=lambda: torch.ones(1))
    bank.finish(0, 32, 32, False, memory)
    bank.commit()
    assert torch.equal(bank.begin(0, 32, 32, False, **args), torch.ones(1))
    with pytest.raises(ValueError, match='discontinuous'):
        bank.begin(0, 33, 32, False, **args)
    with pytest.raises(ValueError, match='overwrite'):
        bank.begin(0, 33, 3, True, **args)
    bank.finish(0, 32, 47, True, memory)
    bank.commit()
    assert bank.begin(0, 33, 3, True, **args) is None
    with pytest.raises(ValueError, match='overwrite'):
        EpisodeMemoryBank().begin(0, 32, 2, True, **args)
    with pytest.raises(ValueError, match='contract'):
        EpisodeMemoryBank().load_state_dict(bank.state_dict())


def test_offset_rollout_next_update_resume():
    import copy
    from tests.test_episode_world_training import batch, run
    from tests.test_unified_h15_policy import make_model

    torch.manual_seed(73)
    model = make_model()
    bank = EpisodeMemoryBank(replay_offsets=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def data(start, end):
        value = batch(count=1, start=start, end=end)
        value['actions'] = value['actions'][:, :, :15]
        value['action_valid_mask'] = value['action_valid_mask'][:, :, :15]
        value['replay_id'] = torch.tensor([152])
        value['replay_offset'] = torch.tensor([2])
        value['episode_start'] = torch.tensor([start == 2])
        return value

    def update(m, optimizer, memory, value):
        optimizer.zero_grad(set_to_none=True)
        velocity, _ = run(m, value, memory)
        (velocity.square().mean() + m.last_wmrm_loss).backward()
        optimizer.step()
        memory.commit()
        return velocity.detach()

    update(model, opt, bank, data(2, False))
    restored = make_model()
    restored.load_state_dict(model.state_dict())
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored_bank = EpisodeMemoryBank(replay_offsets=True)
    restored_bank.load_state_dict(copy.deepcopy(bank.state_dict()))
    next_data = data(17, True)
    first = update(model, opt, bank, next_data)
    second = update(restored, restored_opt, restored_bank, next_data)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    assert bank.entries == restored_bank.entries == {}
