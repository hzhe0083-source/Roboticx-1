import copy

import pytest
import torch

from tests.test_all_starts_sampler import payload
from va_compound.data.all_starts import AllStartsWindowBatchSampler
from va_compound.data.libero import ALL_STARTS_DATA_CONTRACT
from va_compound.training.libero_transfer import validate_transfer_source


def test_epoch_offset_matches_uninterrupted_sampling():
    p = payload()
    old = AllStartsWindowBatchSampler(p, 8, 42, 1, 0, 2)
    old.epoch = 3
    new = AllStartsWindowBatchSampler(p, 8, 42, 1, 0, 2, epoch_offset=3)
    assert list(old) == list(new)
    restored = AllStartsWindowBatchSampler(p, 8, 42, 1, 0, 2, epoch_offset=3)
    new.advance(1)
    restored.load_state_dict(new.state_dict())
    assert list(new) == list(restored)
    with pytest.raises(ValueError, match='epoch_offset'):
        restored.load_state_dict(AllStartsWindowBatchSampler(p, 8, 42, 1, 0, 2).state_dict())


def test_rebatch_only_at_empty_completed_epoch():
    contract = dict(architecture_version='dual_tower_h15_v1', data_contract=ALL_STARTS_DATA_CONTRACT,
                    action_horizon=15, memory_contract='offset_replay_tbptt8_v1',
                    execution_gradient_contract='h15_unified_live_va_v1', main_vision_joint_trained=True,
                    flow_prefix_weight=1., flow_tail_weight=0., epoch_lengths=[5, 6])
    state = dict(epoch=1, batch_cursor=0)
    source = dict(config=dict(architecture_version='dual_tower_h15_v1', action_horizon=15),
                  training_contract=contract, global_step=5, sampler_state=state,
                  world_sampler_state=copy.deepcopy(state),
                  episode_runtime_states=[dict(action=dict(entries={}), world=dict(entries={}))])
    for key in ('model','optimizer','qwen_trainable_state_dict','main_vision_trainable_state_dict'):
        source[key] = {'x': torch.ones(1)}
    validate_transfer_source(source)
    bad = copy.deepcopy(source)
    bad['sampler_state']['batch_cursor'] = 1
    with pytest.raises(ValueError, match='boundary'):
        validate_transfer_source(bad)
    bad = copy.deepcopy(source)
    bad['episode_runtime_states'][0]['action']['entries'][0] = 'live'
    with pytest.raises(ValueError, match='empty'):
        validate_transfer_source(bad)
