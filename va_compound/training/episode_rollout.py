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
    if model.config.architecture_version != "dual_tower_expert_v1":
        raise ValueError("episode rollout requires joint architecture")
    size, length, horizon, action_dim = batch["actions"].shape
    zero = next(model.parameters()).sum() * 0.0
    velocities, conditions, records = [], [], []
    for row in range(size):
        active = bool(batch["stream_active"][row])
        count = int(batch["decision_count"][row]) if active else 0
        mask = batch["decision_valid_mask"][row]
        expected = torch.arange(length, device=mask.device) < count
        if count < 0 or count > length or not torch.equal(mask, expected):
            raise ValueError("episode window requires a contiguous real decision prefix")
        if not active:
            velocities.append(batch["actions"].new_zeros((1, length, horizon, action_dim)) + zero)
            conditions.append(batch["actions"].new_zeros((1, length, 3, horizon, model.config.hidden_dim)) + zero)
            continue
        if count == 0:
            raise ValueError("active episode window is empty")
        stream = int(batch["stream_id"][row])
        episode = int(batch["episode_id"][row])
        start = int(batch["crop_start"][row])
        device = batch["vision_tokens"].device
        memory_dtype = (torch.get_autocast_dtype(device.type)
                        if torch.is_autocast_enabled(device.type)
                        else model.vision_projection.weight.dtype)
        memory = bank.begin(stream, episode, start, bool(batch["episode_start"][row]),
                            device=device, dtype=memory_dtype)
        single = {}
        temporal = {"actions", "proprio", "previous_action", "vision_tokens", "dino_last4",
                    "action_valid_mask", "recovery_mask", "world_target_valid_mask",
                    "world_rank_shuffle_action", "world_rank_shuffle_mask", "world_target_map",
                    "world_state_delta", "decision_valid_mask"}
        if batch["language_hidden"].ndim == 4:
            temporal.update(("language_hidden", "language_mask"))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
                value = value[row:row + 1]
                if key in temporal:
                    value = value[:, :count]
            single[key] = value
        call = dict(options, batch=single,
                    noisy_actions=options["noisy_actions"][row:row + 1, :count],
                    flow_time=options["flow_time"][row:row + 1, :count],
                    initial_visual_memory=memory)
        velocity, condition = rollout(**call)
        bank.finish(stream, episode, start + count * model.config.planning_stride,
                    bool(batch["episode_end"][row]), model.last_rollout_visual_memory)
        velocities.append(F.pad(velocity, (0, 0, 0, 0, 0, length - count)))
        conditions.append(F.pad(condition, (0, 0, 0, 0, 0, 0, 0, length - count)))
        records.append((int(batch["instruction_id"][row]), count,
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
