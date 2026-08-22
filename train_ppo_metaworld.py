"""PPO fine-tuning of the VA policy action head on MetaWorld (ReinFlow-lite).

Starting from an imitation-learned checkpoint, we freeze the vision backbone
(V-JEPA) and the language encoder (Qwen); only the VA composite, the flow
head, the flow-noise schedule (32 params) and a value head are trained.  The
policy is the *augmented Markov flow policy*: each Euler transition adds
Gaussian noise with per-step per-dimension scale sigma (FlowNoiseSchedule),
and PPO uses the exact joint denoising-path log-probability.  At evaluation
the transition noise is dropped, recovering the deterministic Euler policy.

Reward: sparse success within the executed macro action (6 primitive steps).
Success and termination end the episode without bootstrapping; time
truncation bootstraps the critic.

Usage:
  python train_ppo_metaworld.py --il-checkpoint checkpoints/metaworld_va8_40k_full.pt \
      --features data/metaworld_features_v2_full.pt --tasks drawer-close-v3,reach-v3 \
      --steps 1000 --device cuda --save checkpoints/mw_ppo_rl.pt
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from va_compound.backbones import VJEPA21Backbone
from va_compound.model import VACompoundConfig, VACompoundPolicy, VisualMemory
from prepare_pnpw_features import QwenTextBackbone
from eval_metaworld import preprocess

IMAGE_SIZE = 384
VISION_WINDOW = 4
DECISION_STRIDE = 6  # 80 FPS; execute 6 primitives per macro action
ACTION_HORIZON = 8
FLOW_STEPS = 32  # 与评估协议一致（§9：Euler 32 步；RL 优化与部署采样器必须同口径）
EXECUTE_STEPS = 6  # execute the first 6 of the 8-step chunk (Codex protocol)

GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.1
VALUE_COEF = 0.5
PPO_EPOCHS = 4
MINIBATCH = 128
ACTOR_LR = 3e-6
CRITIC_LR = 1e-4
GRAD_CLIP = 0.5


class FlowNoiseSchedule(nn.Module):
    """Per-step per-dim transition noise: sigma = 0.02 + 0.06 sigmoid(alpha)."""

    def __init__(self, steps: int = FLOW_STEPS, action_dim: int = 4) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(steps, action_dim))

    def forward(self) -> torch.Tensor:
        return 0.02 + 0.06 * torch.sigmoid(self.alpha)


class ValueHead(nn.Module):
    """Critic over the action condition (detached): horizon-mean -> MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.net[-1].weight)  # zero-init last layer
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, action_condition: torch.Tensor) -> torch.Tensor:
        return self.net(action_condition.mean(dim=1))  # [B, D] -> [B, 1]


def _validate_ppo_checkpoint_config(config: VACompoundConfig) -> None:
    """Reject checkpoint modes whose rollout semantics PPO cannot reproduce."""
    if getattr(config, "va_world_mode", "legacy") == "peer_sync_h6":
        raise ValueError(
            "PPO does not support peer_sync_h6 checkpoints: peer World state "
            "advances from the readout before the exploratory Flow action is known, "
            "so rollout and recomputed-policy semantics cannot be aligned. Use a "
            "legacy PPO checkpoint/config instead."
        )


def _stack_visual_memories(memories: list[VisualMemory]) -> VisualMemory:
    """Batch recurrent snapshots without dropping optional or World state."""
    if not memories:
        raise ValueError("cannot stack an empty visual-memory list")

    def stack_optional(name: str):
        values = [getattr(memory, name) for memory in memories]
        present = [value is not None for value in values]
        if any(present) and not all(present):
            raise ValueError(f"visual-memory field {name} is only partially present")
        return None if not any(present) else torch.cat(values, dim=0)

    layer_count = len(memories[0].layers)
    if any(len(memory.layers) != layer_count for memory in memories):
        raise ValueError("visual memories have inconsistent layer counts")
    world_states = [memory.world_state for memory in memories]
    world_present = [state is not None for state in world_states]
    if any(world_present) and not all(world_present):
        raise ValueError("visual-memory world_state is only partially present")
    world_state = None
    if all(world_present):
        state_type = type(world_states[0])
        if any(type(state) is not state_type for state in world_states):
            raise ValueError("visual memories have inconsistent world_state types")

        def stack_world_field(name: str):
            values = [getattr(state, name) for state in world_states]
            present = [value is not None for value in values]
            if any(present) and not all(present):
                raise ValueError(f"world-state field {name} is only partially present")
            return None if not any(present) else torch.cat(values, dim=0)

        world_state = state_type(
            belief=stack_world_field("belief"),
            innovation=stack_world_field("innovation"),
            world_map=stack_world_field("world_map"),
        )

    gates = [memory.gate for memory in memories]
    # ``gate`` is scalar diagnostics rather than recurrent tensor state. Keep it
    # when homogeneous; otherwise mark the batched diagnostic unavailable.
    gate = gates[0] if all(value == gates[0] for value in gates) else None
    return VisualMemory(
        layers=tuple(
            torch.cat([memory.layers[index] for memory in memories], dim=0)
            for index in range(layer_count)
        ),
        evidence=stack_optional("evidence"),
        task=stack_optional("task"),
        task_spec=stack_optional("task_spec"),
        pending_future=stack_optional("pending_future"),
        gate=gate,
        world_state=world_state,
    )


class RolloutBuffer:
    """Per-macro-transition storage for PPO (TBPTT=1)."""

    def __init__(self) -> None:
        self.frames: list = []  # (vision_tokens fp16, proprio, previous_action, memory|None)
        self.lang_hidden: list = []
        self.lang_mask: list = []
        self.paths: list = []  # list of K+1 [H, A] CPU fp32 tensors
        self.old_logp: list = []
        self.rewards: list = []
        self.dones: list = []  # True = no bootstrap (success/terminated)
        self.values: list = []
        self.returns: list = []
        self.advantages: list = []

    def __len__(self) -> int:
        return len(self.rewards)


def make_envs(env_names: list[str], seed: int):
    import metaworld

    envs = []
    for name in env_names:
        mt1 = metaworld.MT1(name, seed=seed)
        env = mt1.train_classes[name](render_mode="rgb_array", camera_name="corner2")
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]  # corner2 (lerobot capture)
        env._freeze_rand_vec = False
        envs.append(env)
    return envs


def macro_rollout(
    env,
    model,
    noise_schedule,
    value_head,
    vision_backbone,
    language_cache,
    sq01,
    sq99,
    aq01,
    aq99,
    macro_steps,
    device,
    seed_base,
):
    """Roll out one env for macro_steps decisions; returns list of transitions.

    Rhythm matches eval_metaworld.py exactly: every primitive step renders a
    frame into a sliding 19-frame window; at decision points (step % 6 == 0
    with a full window) a chunk is sampled with flow noise and the next
    DECISION_STRIDE primitives are executed from it.  Reward is sparse success
    over the executed block; success/termination stop the episode without
    critic bootstrapping (time truncation bootstraps).
    """
    transitions = []
    obs, _ = env.reset(seed=seed_base)
    frame_buffer = []
    last_norm = np.zeros(4)
    memory = None
    success = False
    terminated = False
    chunk = np.zeros((ACTION_HORIZON, 4))
    step_count = 0
    decisions = 0
    pending = None  # fields of the current decision; reward/done filled on execution
    while not success and not terminated:
        img = env.render()
        frame_buffer.append(img)
        if step_count == 0:
            # 2026-08-06 评估缺陷修复（与 eval_metaworld.py 同源）：首决策前
            # 不得执行 chunk 初始零值（归一化零反归一化后 = (aq99+aq01)/2，
            # 裁剪后 [1,1,-1,0]，把机械手提前移动 ~4.3cm）。用首帧重复填充
            # 窗口使 step 0 立即推理，与训练首决策窗口（重复帧）同分布。
            while len(frame_buffer) < (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                frame_buffer.insert(0, img)
        if len(frame_buffer) > (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
            frame_buffer.pop(0)
        if step_count % DECISION_STRIDE == 0 and len(frame_buffer) >= VISION_WINDOW:
            if pending is not None:
                transitions.append(pending)
            if decisions >= macro_steps:
                pending = None  # stop cleanly: no unexecuted block counted
                break
            decisions += 1
            indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
            frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
            clip = torch.cat([preprocess(f, IMAGE_SIZE) for f in frames], dim=0).to(device)
            with torch.no_grad():  # no_grad (not inference): stored tensors must
                # remain usable as autograd inputs in ppo_update (inference
                # tensors cannot be saved for backward)
                tokens = vision_backbone(clip.unsqueeze(0), pooling="flat")  # [1, T, D]
                state = np.clip(
                    2.0 * (obs[:4] - sq01) / (sq99 - sq01) - 1.0, -1.0, 1.0
                ).astype(np.float32)
                proprio = torch.tensor(state, device=device, dtype=torch.float32)[None, None]
                previous = torch.tensor(last_norm, device=device, dtype=torch.float32)[None, None]
                cond, next_memory = model.encode_condition(
                    tokens,
                    proprio[0],
                    previous[0],
                    language_cache=language_cache,
                    visual_memory=memory,  # input memory of this decision (TBPTT=1)
                    return_visual_memory=True,
                )
                cond = cond.float()
                value = value_head(cond).item()
                sigma = noise_schedule()
                path = model.sample_flow_trajectory(cond, steps=FLOW_STEPS, sigma=sigma)
                old_logp = model.flow_trajectory_log_prob(path, cond, sigma)
                chunk = np.asarray(path[-1][0].detach().cpu().numpy(), dtype=np.float32)
            pending = {
                "tokens": tokens.detach().cpu().half(),
                "proprio": proprio[0].cpu(),
                "previous": previous[0].cpu(),
                "memory": memory,  # the memory THIS decision consumed (pre-encode)
                "path": [p.detach().cpu().float() for p in path],
                "old_logp": old_logp.detach().cpu().item(),
                "reward": 0.0,
                "done": False,
                "value": value,
            }
            memory = next_memory  # advance recurrent state for the next decision
        # ---- execute one primitive from the current chunk ----
        # phase matches eval_metaworld.py: chunk[0..5] over the 6 executed steps
        # 与训练标签一致裁剪模型输出到 [-1,1]（robust_normalize 存盘即 clip），
        # 再反归一化；prev 反馈（last_norm）同样用裁剪值，避免分布外输入
        norm_action = np.clip(chunk[step_count % DECISION_STRIDE], -1.0, 1.0)
        action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
        obs, r_env, term, trunc, info = env.step(action)
        last_norm = norm_action
        step_count += 1
        if info.get("success"):
            success = True
            if pending is not None:
                pending["reward"] = 1.0
                pending["done"] = True
        elif term:
            terminated = True
            if pending is not None:
                pending["done"] = True
        elif trunc:
            # time truncation: keep the block, bootstrap the critic (done stays False)
            break
    if pending is not None and all(pending is not t for t in transitions):
        transitions.append(pending)
    return transitions, success


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    """GAE; dones=True marks success/termination (no bootstrap)."""
    n = len(rewards)
    returns = np.zeros(n)
    advantages = np.zeros(n)
    gae = 0.0
    next_value = 0.0
    for i in reversed(range(n)):
        if dones[i]:
            next_value = 0.0
            gae = 0.0
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages[i] = gae
        returns[i] = advantages[i] + values[i]
        next_value = values[i]
    return returns, advantages


def ppo_update(buffer, model, noise_schedule, value_head, actor_opt, critic_opt, device):
    """Multi-epoch PPO over the buffer; conditions rebuilt from stored
    frozen-language hidden with TBPTT=1 memory (same graph as rollout).

    Batched update (2026-08-06): the minibatch is stacked into one
    encode_condition + one flow_trajectory_log_prob call (both batch-aware)
    instead of per-sample loops. Transitions are grouped by whether consumed
    memory is present, then their scores are restored to the original shuffled
    minibatch order before PPO targets are paired. The log-prob math is
    unchanged (test_flow_ppo.py keeps the equivalence guarantee)."""
    n = len(buffer)
    idx = list(range(n))
    for _ in range(PPO_EPOCHS):
        random.shuffle(idx)
        for start in range(0, n, MINIBATCH):
            mb = idx[start : start + MINIBATCH]
            logps_by_index: dict[int, torch.Tensor] = {}
            vpreds_by_index: dict[int, torch.Tensor] = {}

            def score_group(group: list[int]) -> None:
                if not group:
                    return
                tokens = torch.cat([buffer.frames[i][0] for i in group]).to(device)
                proprio = torch.cat([buffer.frames[i][1] for i in group]).to(device)
                previous = torch.cat([buffer.frames[i][2] for i in group]).to(device)
                memories = [buffer.frames[i][3] for i in group]
                memory = (
                    None
                    if memories[0] is None
                    else _stack_visual_memories(memories)
                )
                cond = model.encode_condition(
                    tokens,
                    proprio,
                    previous,
                    language_hidden=torch.stack(
                        [buffer.lang_hidden[i] for i in group]
                    ).to(device),
                    language_mask=torch.stack(
                        [buffer.lang_mask[i] for i in group]
                    ).to(device),
                    visual_memory=memory,
                ).float()
                sigma = noise_schedule()
                path = [torch.stack([buffer.paths[i][k][0] for i in group]).to(device)
                        for k in range(len(buffer.paths[group[0]]))]
                logp = model.flow_trajectory_log_prob(path, cond, sigma)
                vpred = value_head(cond.detach()).squeeze(-1)
                for group_offset, buffer_index in enumerate(group):
                    logps_by_index[buffer_index] = logp[group_offset]
                    vpreds_by_index[buffer_index] = vpred[group_offset]

            score_group([i for i in mb if buffer.frames[i][3] is not None])
            score_group([i for i in mb if buffer.frames[i][3] is None])
            # Regrouping changes execution order, not PPO sample order. Restore
            # the exact shuffled minibatch order before pairing with old/adv/ret.
            logps = torch.stack([logps_by_index[i] for i in mb])
            vpreds = torch.stack([vpreds_by_index[i] for i in mb])
            old = torch.tensor([buffer.old_logp[i] for i in mb], device=device, dtype=torch.float32)
            ratio = torch.exp(logps - old)
            adv = torch.tensor(
                [buffer.advantages[i] for i in mb], device=device, dtype=torch.float32
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = torch.tensor(
                [buffer.returns[i] for i in mb], device=device, dtype=torch.float32
            )
            clip_adv = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv
            actor_loss = -torch.min(ratio * adv, clip_adv).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(noise_schedule.parameters()),
                GRAD_CLIP,
            )
            actor_opt.step()
            critic_opt.zero_grad()
            value_loss = VALUE_COEF * F.mse_loss(vpreds, ret)
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_head.parameters(), GRAD_CLIP)
            critic_opt.step()
    return actor_loss.item(), value_loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--il-checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--tasks", type=str, required=True, help="comma-separated MT env names")
    parser.add_argument("--steps", type=int, default=1000, help="PPO update iterations")
    parser.add_argument("--macro-steps", type=int, default=64, help="decisions per env per iter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=Path("checkpoints/mw_ppo_rl.pt"))
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.il_checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    _validate_ppo_checkpoint_config(config)
    model = VACompoundPolicy(config).eval().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"IL checkpoint: {args.il_checkpoint.name} (VA {config.num_layers} layers)")

    features = torch.load(args.features, map_location="cpu", weights_only=True)
    sq01 = features["normalization"]["state_q01"].numpy()
    sq99 = features["normalization"]["state_q99"].numpy()
    aq01 = features["normalization"]["action_q01"].numpy()
    aq99 = features["normalization"]["action_q99"].numpy()

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", max_tokens=64, local_files_only=True
    ).eval()
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True
    )
    env_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # 语言条件与 IL 数据/评估口径一致：编码任务描述（TASK_DESCRIPTIONS），
    # 而非环境名（drawer-close-v3）。找不到映射时退回环境名。
    import json

    _cfg_path = Path(
        "/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json"
    )
    if _cfg_path.exists():
        _descriptions = json.load(open(_cfg_path))["TASK_DESCRIPTIONS"]
        language_texts = [_descriptions.get(name, name) for name in env_names]
    else:
        language_texts = env_names
    hidden, mask = text_backbone.encode(language_texts)
    del text_backbone

    noise_schedule = FlowNoiseSchedule(FLOW_STEPS, config.action_dim).to(device)
    value_head = ValueHead(config.hidden_dim).to(device)
    # sigma (flow-noise schedule) is a policy parameter: it enters the
    # log-prob, so it must be optimized by the actor optimizer.
    actor_opt = torch.optim.AdamW(
        list(model.parameters()) + list(noise_schedule.parameters()), lr=ACTOR_LR
    )
    critic_opt = torch.optim.AdamW(list(value_head.parameters()), lr=CRITIC_LR)

    envs = make_envs(env_names, seed=42)
    buffer = RolloutBuffer()
    for step in range(1, args.steps + 1):
        # ---- rollout ----
        buffer.__init__()
        per_env_success = []
        env_edges = []  # (start, end) of each env's transitions in the buffer
        for env_i, (env, name) in enumerate(zip(envs, env_names, strict=True)):
            cache = model.build_language_cache(
                hidden[env_i : env_i + 1].to(device), mask[env_i : env_i + 1].to(device)
            )
            transitions, ok = macro_rollout(
                env, model, noise_schedule, value_head, vision_backbone, cache,
                sq01, sq99, aq01, aq99, args.macro_steps, device,
                seed_base=1000 * args.seed + env_i,
            )
            per_env_success.append(ok)
            start = len(buffer)
            for tr in transitions:
                buffer.frames.append((tr["tokens"], tr["proprio"], tr["previous"], tr["memory"]))
                buffer.lang_hidden.append(hidden[env_i].cpu())
                buffer.lang_mask.append(mask[env_i].cpu())
                buffer.paths.append(tr["path"])
                buffer.old_logp.append(tr["old_logp"])
                buffer.rewards.append(tr["reward"])
                buffer.dones.append(tr["done"])
                buffer.values.append(tr["value"])
            env_edges.append((start, len(buffer)))
        # ---- GAE (per env to avoid cross-boundary bootstrapping) ----
        buffer.returns = np.zeros(len(buffer))
        buffer.advantages = np.zeros(len(buffer))
        for start, end in env_edges:
            if end > start:
                ret, adv = compute_gae(
                    np.asarray(buffer.rewards[start:end]),
                    np.asarray(buffer.values[start:end]),
                    np.asarray(buffer.dones[start:end]),
                )
                buffer.returns[start:end] = ret
                buffer.advantages[start:end] = adv
        # ---- PPO ----
        a_loss, v_loss = ppo_update(
            buffer, model, noise_schedule, value_head, actor_opt, critic_opt, device
        )
        sr = sum(per_env_success) / len(per_env_success)
        print(
            f"[{step}/{args.steps}] ep_success={int(sum(per_env_success))} "
            f"reward_sum={np.sum(buffer.rewards):.0f} sr={sr:.2f} "
            f"actor={a_loss:.4f} value={v_loss:.4f} "
            f"sigma_mean={noise_schedule().mean().item():.4f}",
            flush=True,
        )
        if args.save and step % args.save_every == 0:
            torch.save(
                {
                    "config": config.__dict__,
                    "model": model.state_dict(),
                    "noise_schedule": noise_schedule.state_dict(),
                    "value_head": value_head.state_dict(),
                    "step": step,
                },
                args.save,
            )

    torch.save(
        {
            "config": config.__dict__,
            "model": model.state_dict(),
            "noise_schedule": noise_schedule.state_dict(),
            "value_head": value_head.state_dict(),
            "step": args.steps,
        },
        args.save,
    )
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
