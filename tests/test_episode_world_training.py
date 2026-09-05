import copy

import pytest
import torch

from tests.test_layerwise_expert_policy import config
from va_compound import VACompoundPolicy
from va_compound.data.episode_stream import EpisodeStreamDataset, EpisodeWindowBatchSampler
from va_compound.training.episode_memory import EpisodeMemoryBank
from va_compound.training.rollout import rollout_policy


def batch(count=2, start=0, end=False):
    length = 3
    valid = torch.arange(length)[None] < count
    return {
        "vision_tokens": torch.randn(1, length, 4, 8),
        "language_hidden": torch.randn(1, length, 4, 12),
        "language_mask": torch.ones(1, length, 4, dtype=torch.bool),
        "proprio": torch.randn(1, length, 5),
        "previous_action": torch.zeros(1, length, 4),
        "actions": torch.randn(1, length, 50, 4).clamp(-1, 1),
        "action_valid_mask": valid[:, :, None].expand(1, length, 50).clone(),
        "world_target_valid_mask": valid.clone(),
        "world_target_map": torch.randn(1, length, 8, 2, 2),
        "world_rank_shuffle_action": torch.randn(1, length, 15, 4).clamp(-1, 1),
        "world_rank_shuffle_mask": valid.clone(),
        "world_state_delta": torch.randn(1, length, 5),
        "decision_valid_mask": valid,
        "decision_count": torch.tensor([count]),
        "stream_active": torch.tensor([True]), "stream_id": torch.tensor([0]),
        "episode_id": torch.tensor([10]), "instruction_id": torch.tensor([0]),
        "crop_start": torch.tensor([start]), "episode_start": torch.tensor([start == 0]),
        "episode_end": torch.tensor([end]),
    }


def run(model, data, bank, world=True):
    return rollout_policy(model, data, torch.zeros_like(data["actions"]),
                          torch.full(data["actions"].shape[:2], .5),
                          episode_memory=bank, visual_world_supervision=world,
                          train_world_model=world, summarize_visual_world_metrics=False)


def test_state_supervision_updates_shared_world_and_not_labels():
    torch.manual_seed(42)
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    data = batch()
    data["world_state_delta"].requires_grad_()
    bank = EpisodeMemoryBank()
    velocity, condition = run(model, data, bank)
    assert velocity.shape == (1, 3, 50, 4)
    assert condition.shape == (1, 3, 3, 50, 16)
    assert torch.count_nonzero(velocity[:, 2:]) == 0
    loss = model.last_world_state_delta_loss
    loss.backward()
    assert data["world_state_delta"].grad is None
    assert model.wmrm.state_delta_head[-1].weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for n, p in model.wmrm.named_parameters()
               if not n.startswith("state_delta_head"))
    with pytest.raises(ValueError, match="committed"):
        bank.state_dict()
    bank.commit()
    memory = bank.entries[0][2]
    assert all(not value.requires_grad for value in memory.layers)
    assert not memory.world_state.world_map.requires_grad
    assert bank.entries[0][:2] == (10, 30)


def test_cross_window_resume_and_episode_end():
    torch.manual_seed(10)
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    bank = EpisodeMemoryBank()
    run(model, batch(), bank, world=False)
    bank.commit()
    state = copy.deepcopy(bank.state_dict())
    restored = EpisodeMemoryBank()
    restored.load_state_dict(state)
    data = batch(count=1, start=30, end=True)
    first = run(model, data, bank, world=False)[0]
    second = run(model, data, restored, world=False)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.sum().backward()
    bank.commit()
    restored.commit()
    assert bank.entries == restored.entries == {}
    with pytest.raises(ValueError, match="missing"):
        run(model, data, bank, world=False)


def test_targets_and_padded_frames_do_not_change_forward():
    torch.manual_seed(9)
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    data = batch(count=1, end=True)
    original = run(model, data, EpisodeMemoryBank())[0]
    changed = copy.deepcopy(data)
    changed["world_state_delta"] += 100
    changed["vision_tokens"][:, 1:] += 500
    actual = run(model, changed, EpisodeMemoryBank())[0]
    torch.testing.assert_close(original, actual, rtol=0, atol=0)


def test_batch_streams_reset_independently_and_reject_time_jumps():
    torch.manual_seed(5)
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    first = batch(count=2, end=False)
    second = batch(count=1, end=True)
    combined = {key: torch.cat((value, second[key]), dim=0) for key, value in first.items()}
    combined["stream_id"] = torch.tensor([0, 1])
    combined["episode_id"] = torch.tensor([10, 11])
    bank = EpisodeMemoryBank()
    run(model, combined, bank, world=False)
    bank.commit()
    assert set(bank.entries) == {0}
    jump = batch(count=1, start=45, end=True)
    with pytest.raises(ValueError, match="discontinuous"):
        run(model, jump, bank, world=False)
    next_window = batch(count=1, start=30, end=True)
    next_window["vision_tokens"].requires_grad_()
    velocity = run(model, next_window, bank, world=False)[0]
    velocity.sum().backward()
    assert next_window["vision_tokens"].grad is not None
    assert all(not x.requires_grad for x in bank.entries[0][2].layers)


def test_resume_preserves_next_optimizer_update(tmp_path):
    torch.manual_seed(12)
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bank = EpisodeMemoryBank()

    def update(policy, optim, memory, data):
        optim.zero_grad(set_to_none=True)
        run(policy, data, memory)
        policy.last_wmrm_loss.backward()
        optim.step()
        memory.commit()

    update(model, optimizer, bank, batch())
    path = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "memory": bank.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored_model = VACompoundPolicy(model.config).eval()
    restored_model.load_state_dict(saved["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_bank = EpisodeMemoryBank()
    restored_bank.load_state_dict(saved["memory"])
    data = batch(count=1, start=30, end=True)
    update(model, optimizer, bank, data)
    update(restored_model, restored_optimizer, restored_bank, data)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored_model.state_dict()[name], rtol=0, atol=0)


def test_inactive_storage_slot_does_not_mutate_dataset_or_memory():
    data = batch(count=2)
    item = {key: value[0] for key, value in data.items()}
    wrapped = EpisodeStreamDataset([item])
    inactive = wrapped[(0, 0, False)]
    assert not inactive["decision_valid_mask"].any()
    assert item["decision_valid_mask"].sum() == 2
    inactive_batch = {key: value.unsqueeze(0) for key, value in inactive.items()}
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    bank = EpisodeMemoryBank()
    velocity, _ = run(model, inactive_batch, bank)
    assert not bank.pending and not bank.entries
    assert velocity.requires_grad and torch.count_nonzero(velocity) == 0
    model.last_wmrm_task_losses[0].backward()


def test_episode_sampler_distributed_coverage_and_resume():
    rows = [(task, task * 10 + episode, start * 120)
            for task in range(2) for episode in range(5) for start in range(episode % 3 + 1)]
    payload = {key: torch.tensor([row[i] for row in rows])
               for i, key in enumerate(("instruction_id", "episode_id", "crop_start"))}
    samplers = [EpisodeWindowBatchSampler(payload, 8, 3, 2, rank, 2) for rank in range(2)]
    assert len(samplers[0]) == len(samplers[1])
    seen = []
    for batches in zip(*map(iter, samplers), strict=True):
        for entries in batches:
            assert {int(payload["instruction_id"][row]) for row, _, _ in entries} == {0, 1}
            seen.extend(row for row, _, active in entries if active)
    assert sorted(seen) == list(range(len(rows)))
    samplers[0].advance()
    state = samplers[0].state_dict()
    restored = EpisodeWindowBatchSampler(payload, 8, 3, 2, 0, 2)
    restored.load_state_dict(state)
    assert list(restored) == list(samplers[0])
    old_length = len(restored)
    restored.advance(old_length - restored.batch_cursor)
    assert restored.epoch == 1 and restored.batch_cursor == 0
    assert len(restored) == old_length
