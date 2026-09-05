"""Run only real decisions, with independently owned memory for each stream."""
from __future__ import annotations

import torch
import torch.nn.functional as F


_METRICS = (
    "last_wmrm_loss", "last_wmrm_base_loss", "last_world_no_regression_loss",
    "last_world_static_constraint_loss", "last_world_action_rank_loss",
    "last_world_action_shuffle_loss", "last_world_action_zero_loss",
    "last_world_action_strong_loss", "last_world_action_readout_loss",
    "last_world_action_readout_rmse", "last_world_late_stage_anchor_loss",
    "last_world_state_delta_loss",
)


def rollout_episode_windows(rollout, options):
    options = dict(options)
    model, batch = options["model"], options["batch"]
    bank = options.pop("episode_memory")
    options.pop("initial_visual_memory", None)
    if model.config.architecture_version not in ("dual_tower_expert_v1", "dual_tower_h15_v1"):
        raise ValueError("episode rollout requires joint architecture")
    size, length, horizon, action_dim = batch["actions"].shape
    zero = next(model.parameters()).sum() * 0.0
    velocities, conditions, records = [None] * size, [None] * size, []
    groups = {}
    from va_compound.training.memory_batch import memory_signature, stack_memories
    max_group = int(getattr(model, "episode_microbatch", 1))
    if max_group < 1:
        raise ValueError("episode_microbatch must be positive")
    for row in range(size):
        active = bool(batch["stream_active"][row])
        count = int(batch["decision_count"][row]) if active else 0
        mask = batch["decision_valid_mask"][row]
        expected = torch.arange(length, device=mask.device) < count
        if count < 0 or count > length or not torch.equal(mask, expected):
            raise ValueError("episode window requires a contiguous real decision prefix")
        if not active:
            velocities[row] = batch["actions"].new_zeros((1, length, horizon, action_dim)) + zero
            conditions[row] = batch["actions"].new_zeros((1, length, 3, horizon, model.config.hidden_dim)) + zero
            continue
        if count == 0:
            raise ValueError("active episode window is empty")
        stream = int(batch["stream_id"][row])
        episode = int(batch["episode_id"][row])
        start = int(batch["crop_start"][row])
        if bank.replay_offsets:
            offset = int(batch["replay_offset"][row])
            replay = int(batch["replay_id"][row])
            if not 0 <= offset < 15 or replay != episode * 15 + offset or start % 15 != offset:
                raise ValueError("invalid offset replay identity")
            episode = replay
        elif "replay_id" in batch:
            raise ValueError("offset replay requires its own memory contract")
        device = batch["vision_tokens"].device
        memory_dtype = (torch.get_autocast_dtype(device.type)
                        if torch.is_autocast_enabled(device.type)
                        else model.vision_projection.weight.dtype)
        memory = bank.begin(stream, episode, start, bool(batch["episode_start"][row]),
                            device=device, dtype=memory_dtype)
        masks = tuple((key, tuple(batch[key][row, :count].flatten().tolist()))
                      for key in ("world_target_valid_mask", "world_rank_shuffle_mask") if key in batch)
        key = (int(batch["instruction_id"][row]), count, memory_signature(memory), masks)
        if float(options.get("wmrm_late_stage_anchor_weight", 0)) > 0:
            key = (*key, row)
        groups.setdefault(key, []).append((row, stream, episode, start, memory))
    for entries in groups.values():
        for begin in range(0, len(entries), max_group):
            members = entries[begin:begin + max_group]
            rows = [entry[0] for entry in members]
            count = int(batch["decision_count"][rows[0]])
            single = {}
            temporal = {"actions", "proprio", "previous_action", "vision_tokens", "dino_last4",
                        "action_valid_mask", "recovery_mask", "world_target_valid_mask",
                        "world_rank_shuffle_action", "world_rank_shuffle_mask", "world_target_map",
                        "world_state_delta", "decision_valid_mask"}
            if batch["language_hidden"].ndim == 4:
                temporal.update(("language_hidden", "language_mask"))
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
                    value = value[rows]
                    if key in temporal:
                        value = value[:, :count]
                single[key] = value
            call = dict(options, batch=single,
                        noisy_actions=options["noisy_actions"][rows, :count],
                        flow_time=options["flow_time"][rows, :count],
                        initial_visual_memory=stack_memories([entry[4] for entry in members]))
            velocity, condition = rollout(**call)
            for position, (row, stream, episode, start, _) in enumerate(members):
                memory = model.last_rollout_visual_memory.index_select(torch.tensor([position], device=velocity.device))
                bank.finish(stream, episode, start + count * model.config.planning_stride,
                            bool(batch["episode_end"][row]), memory)
                velocities[row] = F.pad(velocity[position:position + 1], (0, 0, 0, 0, 0, length - count))
                conditions[row] = F.pad(condition[position:position + 1], (0, 0, 0, 0, 0, 0, 0, length - count))
            records.append((int(batch["instruction_id"][rows[0]]), count * len(rows),
                            {name: getattr(model, name, None) for name in _METRICS},
                            dict(model.last_wmrm_task_losses)))
    def reduce_metric(name, task=None):
        values = [(count, metrics[name])
                  for task_id, count, metrics, _ in records
                  if (task is None or task_id == task) and metrics[name] is not None]
        return sum((count * value for count, value in values), zero) / max(sum(count for count, _ in values), 1)
    for name in _METRICS:
        setattr(model, name, reduce_metric(name))
    model.last_wmrm_task_losses = {}
    for task in torch.unique(batch["instruction_id"]).tolist():
        values = [(count, losses[task]) for task_id, count, _, losses in records
                  if task_id == task and task in losses]
        model.last_wmrm_task_losses[task] = sum((count * value for count, value in values), zero) / max(sum(count for count, _ in values), 1)
    model.last_visual_world_metrics = {}
    return torch.cat(velocities), torch.cat(conditions)
