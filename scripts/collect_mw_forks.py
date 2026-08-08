#!/usr/bin/env python
"""严格 fork 采集器：同种子同状态双任务（同帧不同指令不同专家动作）。

基于 2026-08-08 实测验证的构造机制：
- 同场景任务对（如 drawer-close-v3 / drawer-open-v3）共用同一 XML；
- 同 seed reset 后 qpos 仅任务定义关节不同（close: 抽屉拉出 -0.15；open: 0.0），
  基准位 body("drawer").pos 可能差一个常量偏移；
- 对齐 = 拷贝 body/site/geom 位置 + `env._set_obj_xyz(同值)`（不要手动
  mj_forward，实测会使渲染全图变暗；也不要用 set_env_state——参数是
  (mocap_pos, mocap_quat) 而非 (qpos, qvel)，不对称）；
- goal site 在 corner2 视角不可见（own-reset 像素差异全在抽屉带）。

输出（原始值，未归一化、未编码）：
  data/mw_fork_raw_<pair>.pt:
    frames     [2N, 4, 4, 384, 384, 3] uint8   （N 对 × 2 分支 × 4 决策点）
    states     [2N, 4, 39] float32             （env obs[:39]，含 goal）
    raw_actions[2N, 4, 8, 4] float32           （发送给 env 的原始动作）
    env_names  [2] str, instruction_ids [2N] int（A/B 分支标记）
    pair_ids   [2N] int（2 行共享一个 pair_id）
    seeds      [2N] int, drawer_pos [2N] float

后处理（scripts/assemble_fork_dataset.py，下阶段）：
  帧 → V-JEPA 特征；states[:4] → v5 归一化 proprio；动作走 v5 executed-clip
  标签管线；语言行从 v5 按 instruction_id 复制。

用法（EGL 渲染需要无头 GL；训练结束后 GPU 空闲时跑）：
  MUJOCO_GL=egl EGL_PLATFORM=surfaceless python scripts/collect_mw_forks.py \
      --pair drawer-close-v3 drawer-open-v3 \
      --positions 0.0 -0.075 -0.15 --seeds 24
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import metaworld  # noqa: F401  (环境注册)

PAIRS = {
    # 2026-08-08 实测验证（同 XML / 同种子 qpos 差异仅任务定义关节或为 0）
    ("drawer-close-v3", "drawer-open-v3"): {
        "joint": 9, "positions": [0.0, -0.075, -0.15],
    },
    ("faucet-close-v3", "faucet-open-v3"): {
        # 同种子 reset 后 qpos 全等（maxdiff 0.0）；仅 body/site 位置差 0.07/0.05
        "joint": None, "positions": [0.0],
    },
    ("window-close-v3", "window-open-v3"): {
        "joint": 9, "positions": [0.0],
    },
    ("door-close-v3", "door-open-v3"): {
        # 手臂关节也随机化不同（0-6 差 1.57），align 全量拷贝 qpos 即可
        "joint": 9, "positions": [0.0],
    },
}
CAMERA = "corner2"
VISION_WINDOW = 4
VISION_STRIDE = 2
CONTROL_STRIDE = 6
SEQUENCE_LENGTH = 4
ACTION_HORIZON = 8
IMAGE_SIZE = 384


def make_env(env_name: str) -> object:
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name](render_mode="rgb_array", camera_name=CAMERA)
    env.set_task(ml1.train_tasks[0])
    env._freeze_rand_vec = False
    return env


def align_b_to_a(env_a, env_b, joint_idx: int | None, joint_value: float | None) -> None:
    """把 env_b 的场景对齐到 env_a（qpos 任务关节 + 全部模型位置字段）。"""
    env_b.data.qpos[:] = env_a.data.qpos
    env_b.data.qvel[:] = env_a.data.qvel
    env_b.data.mocap_pos[:] = env_a.data.mocap_pos
    env_b.data.mocap_quat[:] = env_a.data.mocap_quat
    for field in ("body_pos", "body_quat", "site_pos", "site_quat", "geom_pos", "geom_quat"):
        src = getattr(env_a.model, field)
        getattr(env_b.model, field)[:] = src
    if joint_idx is not None and joint_value is not None:
        env_b.data.qpos[joint_idx] = joint_value
        env_a.data.qpos[joint_idx] = joint_value


def render(env, size: int = IMAGE_SIZE) -> np.ndarray:
    img = np.asarray(env.render())
    if img.shape[:2] != (size, size):
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((size, size)))
    return img


def collect_pair(
    env_a_name: str,
    env_b_name: str,
    expert_a,
    expert_b,
    seed: int,
    joint_idx: int | None,
    joint_value: float | None,
) -> tuple[dict, dict]:
    """采集一对 fork 样本（两个分支各 4 决策点）。返回 (sample_a, sample_b)。"""
    env_a, env_b = make_env(env_a_name), make_env(env_b_name)
    env_a.seed(seed)
    env_b.seed(seed)
    env_a.reset()
    env_b.reset()
    for e in (env_a, env_b):
        e.model.cam_pos[2] = [0.75, 0.075, 0.7]  # lerobot 采集同款 corner2
    align_b_to_a(env_a, env_b, joint_idx, joint_value)

    def rollout(env, expert) -> dict:
        frames, states, actions = [], [], []
    def rollout(env, expert) -> dict:
        frames, states, actions = [], [], []
        frame_buffer: list[np.ndarray] = []

        def current_obs() -> np.ndarray:
            # parquet/lerobot 契约：state 4 维 (x,y,z,gripper)；env 内部 obs 39 维，
            # 取前 4 维与训练数据一致（v5 proprio 也是 obs[:4] 归一化）
            if hasattr(env, "_get_obs"):
                obs = np.asarray(env._get_obs(), dtype=np.float32)
            else:
                obs = np.asarray(env._get_curr_obs_combined_no_goal(), dtype=np.float32)
            return obs[:4].copy()

        for _ in range(SEQUENCE_LENGTH):
            # 决策点：缓冲补足 4 帧窗口 [decision-6, decision-4, decision-2, decision]
            while len(frame_buffer) < (VISION_WINDOW - 1) * VISION_STRIDE + 1:
                frame_buffer.append(render(env))
            window = [frame_buffer[-7], frame_buffer[-5], frame_buffer[-3], frame_buffer[-1]]
            frames.append(np.stack(window))
            states.append(current_obs())
            chunk = []
            with torch.inference_mode():
                act = expert(torch.from_numpy(current_obs())[None]).squeeze(0).numpy()
            for step_i in range(ACTION_HORIZON):
                act = np.clip(act, -1.0, 1.0).astype(np.float32)
                env.step(act)
                chunk.append(act.copy())
                if step_i % VISION_STRIDE == 1:
                    frame_buffer.append(render(env))
                    if len(frame_buffer) > 16:
                        frame_buffer = frame_buffer[-8:]
            actions.append(chunk)
        return {
            "frames": np.stack(frames),           # [4, 4, 384, 384, 3]
            "states": np.stack(states),           # [4, 4]（lerobot state 4 维）
            "raw_actions": np.stack(actions),     # [4, 8, 4]
        }

    # 分支 A：在 env_a 上执行专家 A；分支 B：对齐后的 env_b 上执行专家 B。
    # 注意：两分支 t=0 的帧在渲染上严格一致（对齐验证），后续决策点各自演化。
    sample_a = rollout(env_a, expert_a)
    sample_b = rollout(env_b, expert_b)

    # t=0 像素一致性验证（严格 fork 契约）
    d = np.abs(sample_a["frames"][0].astype(int) - sample_b["frames"][0].astype(int))
    print(f"  seed={seed} pos={joint_value} t0 meandiff={d.mean():.3f} maxdiff={d.max()}")
    if d.mean() > 2.0:
        raise RuntimeError(f"fork t0 帧不一致 meandiff={d.mean():.3f}（对齐失败）")
    return sample_a, sample_b


def load_expert(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_fork_experts", Path(__file__).resolve().parent / "train_fork_experts.py"
    )
    te = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te)

    model = te.BCExpert()
    model.load_state_dict(ckpt["model"])
    model.eval()
    sm, ss = ckpt["state_mean"], ckpt["state_std"]
    am, astd = ckpt["action_mean"], ckpt["action_std"]

    def policy(obs: torch.Tensor) -> np.ndarray:
        x = (obs - sm) / ss
        y = model(x[None]).detach().numpy()[0] * astd + am
        return y

    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True, metavar=("ENV_A", "ENV_B"))
    ap.add_argument("--positions", nargs="+", type=float, default=None,
                    help="任务关节取值（缺省用 PAIRS 配置）")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(24)))
    ap.add_argument("--experts-dir", type=Path,
                    default=Path("/media/ryan/robot-data/fork_experts"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    key = tuple(args.pair)
    if key not in PAIRS:
        raise ValueError(f"未登记任务对 {key}（PAIRS）——需先验证同 XML/同机制")
    cfg = PAIRS[key]
    positions = args.positions if args.positions is not None else cfg["positions"]

    expert_a = load_expert(args.experts_dir / f"{args.pair[0]}.pt")
    expert_b = load_expert(args.experts_dir / f"{args.pair[1]}.pt")

    frames_all, states_all, actions_all, pair_ids, inst_ids, seeds_all, poss = (
        [], [], [], [], [], [], []
    )
    pid = 0
    for seed in args.seeds:
        for pos in positions:
            sa, sb = collect_pair(
                args.pair[0], args.pair[1], expert_a, expert_b,
                seed, cfg.get("joint"), pos,
            )
            for sample, inst in ((sa, 0), (sb, 1)):
                frames_all.append(sample["frames"])
                states_all.append(sample["states"])
                actions_all.append(sample["raw_actions"])
                pair_ids.append(pid)
                inst_ids.append(inst)
                seeds_all.append(seed)
                poss.append(pos)
            pid += 1

    out = args.out / f"mw_fork_raw_{args.pair[0].split('-')[0]}.pt"
    torch.save(
        {
            "frames": torch.from_numpy(np.stack(frames_all)).to(torch.uint8),
            "states": torch.from_numpy(np.stack(states_all)).to(torch.float32),
            "raw_actions": torch.from_numpy(np.stack(actions_all)).to(torch.float32),
            "pair_ids": torch.tensor(pair_ids),
            "inst_ids": torch.tensor(inst_ids),
            "seeds": torch.tensor(seeds_all),
            "positions": torch.tensor(poss),
            "env_names": list(args.pair),
        },
        out,
    )
    print(f"saved {out}（{len(pair_ids)//2} 对 fork）")


if __name__ == "__main__":
    main()
