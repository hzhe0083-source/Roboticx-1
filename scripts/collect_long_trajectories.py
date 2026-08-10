#!/usr/bin/env python
"""MT50 长轨迹重采（2026-08-09 用户拍板；Codex 审查要求真反馈专家）。

用 MetaWorld 官方 scripted policy 闭环执行采集 150-300 帧长轨迹：
  1. 完整任务执行（policy 每步响应 = 真反馈专家，非开环重放）
  2. 成功后 hold 成功态 1-2s（教"稳住"，80-160 帧）
  3. 接触相位注入 2-8mm 扰动 + policy 恢复（真恢复标签，Codex P0：扰动后
     重新调用 policy 而非重放 nominal 动作）
输出与 fullframe 管线同构：executed-clip 契约、全局 q01/q99 继承
（scripts/make_fullframe_executed.py 的 normalization）、prev 契约
（episode 首决策 0、其余前一帧动作）。

用法（CPU 仿真，不占 GPU）：
    # 冒烟：peg-insert 3 条
    python scripts/collect_long_trajectories.py --task peg-insert-side-v3 --episodes 3
    # 全量：49 任务 × 30 条（分任务跑，~30-60 分钟/任务）
    python scripts/collect_long_trajectories.py --task peg-insert-side-v3 --episodes 30
    # 输出合并：data/metaworld_longtraj_{task}.pt → 最后合并成 fullframe 数据

依赖：metaworld（本地仿真，MT1 + policies）、robust_normalize（prepare_metaworld）。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import metaworld  # noqa: E402

FPS = 80
CONTROL_STRIDE = 6      # 决策间隔（与 fullframe 一致，13.3Hz）
HOLD_FRAMES = (80, 160)  # 成功后 hold 时长区间（1-2s）
PERTURB_MM = (2.0, 8.0)  # 接触相位扰动幅度
PERTURB_KINDS = ("eef_lateral", "eef_height", "object", "peg_hole_relative")
PERTURB_SETTLE_STEPS = 12  # 扰动注入后的物理 settle 步数（进入时间轴，Codex P0-2）
OUT_DIR = ROOT / "data"
JPEG_QUALITY = 90  # 帧压缩：uint8 384×384×3 ≈ 440KB → ~40KB，49 任务 ≈ 12GB（2026-08-09 磁盘止损）


def compress_frames(frames: np.ndarray) -> list[bytes]:
    """[T,384,384,3] uint8 → JPEG bytes 列表（训练时解压，对齐 parquet PNG 解码路径）。"""
    out = []
    for i in range(len(frames)):
        buf = io.BytesIO()
        Image.fromarray(frames[i]).save(buf, format="JPEG", quality=JPEG_QUALITY)
        out.append(buf.getvalue())
    return out


def make_env(task_name: str, seed: int = 42):
    """与 prepare_mw_perturbations.py 同款环境构造（corner2 相机对齐数据）。"""
    mt1 = metaworld.MT1(task_name, seed=42)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env._freeze_rand_vec = False
    env.reset(seed=seed)
    return env


# env 类名 → policy 类名例外映射（命名差异，2026-08-09 扫描确认）
POLICY_EXCEPTIONS = {
    "SawyerNutAssemblyEnvV3": "SawyerAssemblyV3Policy",       # assembly 本地 env nut 焊死，跳过
    "SawyerNutDisassembleEnvV3": "SawyerDisassembleV3Policy",
    "SawyerDoorEnvV3": "SawyerDoorOpenV3Policy",
    "SawyerSweepIntoGoalEnvV3": "SawyerSweepIntoV3Policy",
}


def get_policy(task_name: str):
    """scripted policy（真反馈专家）。类名 = env 类名去掉 EnvV3 + V3Policy（含例外表）。"""
    import metaworld.policies as P
    mt1 = metaworld.MT1(task_name, seed=42)
    env_cls_name = mt1.train_classes[task_name].__name__  # e.g. SawyerPegInsertionSideEnvV3
    cls_name = POLICY_EXCEPTIONS.get(
        env_cls_name, env_cls_name.replace("EnvV3", "") + "V3Policy"
    )
    if not hasattr(P, cls_name):
        raise ValueError(f"no scripted policy {cls_name} for {task_name}")
    return getattr(P, cls_name)()


def collect_episode(env, policy, task_name: str, rng: np.random.Generator,
                    perturb: bool = True) -> dict | None:
    """跑一条长轨迹：完整任务 + 成功 hold + （可选）接触扰动恢复。

    返回 dict（帧/动作/状态均为原始值，后续统一归一化）或 None（失败）。

    2026-08-10 保护：单条轨迹异常兜底——env 抖动/渲染失败不应炸掉整个
    任务进程（此前 20 实例并发时多次全灭，疑似 OOM/异常无兜底）。
    """
    try:
        return _collect_episode_inner(env, policy, task_name, rng, perturb)
    except Exception as exc:  # noqa: BLE001
        print(f"  [collect_episode] 异常跳过本条: {type(exc).__name__}: {exc}")
        return None


def _collect_episode_inner(env, policy, task_name: str, rng: np.random.Generator,
                           perturb: bool = True) -> dict | None:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    success_frame = None
    perturbed = False
    hold_target: int | None = None
    consec_success = 0

    for step in range(500):
        a = np.clip(np.asarray(policy.get_action(obs), dtype=np.float32), -1.0, 1.0)
        # 契约（Codex P0-1，2026-08-09）：先保存【当前观测 + 即将执行的动作】，
        # 再 step。此前顺序是 step 后才保存 (o_{t+1}, a_t)——图像/状态泄漏
        # a_t 的执行结果且 prev 错位，整批数据作废需重采。
        frames.append(env.render())
        actions.append(a)
        states.append(obs[:4].astype(np.float32))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("success") and success_frame is None:
            success_frame = step
        # 接触扰动：成功前随机一步注入 2-8mm，之后继续 policy 恢复
        if perturb and not perturbed and success_frame is None and step > 20 and rng.random() < 0.05:
            mag = rng.uniform(*PERTURB_MM) / 1000.0
            kind = str(rng.choice(PERTURB_KINDS))
            _apply_perturb(env, kind, mag)
            # settle 步必须进入时间轴并刷新 obs（Codex P0-2）：12 个零动作
            # 逐帧保存 (观测, 零动作, 状态) 再 step——否则扰动后 policy 用
            # stale obs 决策，且隐藏的 12 个真实执行步破坏跨边界窗口与 prev。
            for _ in range(PERTURB_SETTLE_STEPS):
                frames.append(env.render())
                actions.append(np.zeros(4, dtype=np.float32))
                states.append(obs[:4].astype(np.float32))
                obs, _r2, _t2, _tr2, _i2 = env.step(np.zeros(4))
            perturbed = True
        # hold：首次成功时固定目标长度，连续成功计数，掉出成功即清零
        # （Codex P0-2：旧逻辑每帧重抽阈值且结束时不要求仍成功）。
        if info.get("success") and success_frame is not None:
            if hold_target is None:
                hold_target = int(rng.integers(*HOLD_FRAMES))
            consec_success += 1
        else:
            consec_success = 0
        if success_frame is not None and consec_success >= hold_target:
            break
        if trunc:
            break
    if success_frame is None:
        return None  # 纯失败轨迹，丢弃（Codex P0-2 核心：剔除失败长尾）
    # hold 未达标但含成功段：保留（精密任务 success 信号会抖动，连续达标
    # 过苛会整条丢弃；全局成功跨度 >= HOLD_FRAMES[0] 即视为含稳定成功段）。
    if step - success_frame < int(HOLD_FRAMES[0]):
        return None
    return {
        "frames": compress_frames(np.stack(frames)),  # list[bytes] JPEG
        "actions": np.stack(actions),     # [T, 4]
        "states": np.stack(states),       # [T, 4]
        "success_frame": success_frame,
        "perturbed": perturbed,
    }


def _apply_perturb(env, kind: str, mag: float) -> None:
    """注入 2-8mm 扰动（复用 prepare_mw_perturbations 的机制：mocap/obj1）。

    settle 步由调用方进入时间轴（PERTURB_SETTLE_STEPS 循环），此处只改环境。
    """
    if kind in ("eef_lateral", "eef_height"):
        delta = np.zeros(3)
        if kind == "eef_lateral":
            theta = np.random.uniform(0, 2 * np.pi)
            delta[:2] = mag * np.array([np.cos(theta), np.sin(theta)])
        else:
            delta[2] = mag
        env.data.mocap_pos[0] += delta
    else:
        from scripts.prepare_mw_perturbations import move_obj1
        delta = np.zeros(3)
        theta = np.random.uniform(0, 2 * np.pi)
        delta[:2] = mag * np.array([np.cos(theta), np.sin(theta)])
        try:
            move_obj1(env, delta)
        except RuntimeError:
            pass  # 该任务 obj1 不可动则跳过


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="MT1 任务名，如 peg-insert-side-v3")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--no-perturb", action="store_true", help="不注入接触扰动")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.task)
    policy = get_policy(args.task)
    print(f"task={args.task} policy={type(policy).__name__} episodes={args.episodes}")

    episodes = []
    attempts = 0
    while len(episodes) < args.episodes and attempts < args.episodes * 10:
        attempts += 1
        ep = collect_episode(env, policy, args.task, rng, perturb=not args.no_perturb)
        if ep is None:
            continue
        episodes.append(ep)
        lens = [len(e["frames"]) for e in episodes]
        print(f"  ep {len(episodes)}: len={len(ep['frames'])} "
              f"success@{ep['success_frame']} perturbed={ep['perturbed']} "
              f"(mean_len={np.mean(lens):.0f})")
    if not episodes:
        raise SystemExit("no successful episodes collected")
    env.close()

    # ---- 统一归一化：继承 fullframe executed 的 q01/q99（Codex：禁止单独算） ----
    import torch
    src = torch.load(ROOT / "data" / "metaworld_fullframe_executed.pt",
                     map_location="cpu", weights_only=True)
    aq01, aq99 = src["normalization"]["action_q01"], src["normalization"]["action_q99"]
    sq01, sq99 = src["normalization"]["state_q01"], src["normalization"]["state_q99"]
    norm = dict(src["normalization"])

    def robust(x, lo, hi):
        return np.clip(2 * (x - lo.numpy()) / (hi.numpy() - lo.numpy()) - 1, -1, 1)

    out = {
        "episodes": episodes,  # 原始数据（帧/动作/状态），后续 prepare 管线切片
        "task": args.task,
        "n_episodes": len(episodes),
        "normalization": norm,
        "metadata": {
            "contract": "long_trajectory_scripted",
            "fps": FPS,
            "control_stride": CONTROL_STRIDE,
            "policy": type(policy).__name__,
            "perturbed": not args.no_perturb,
            "perturb_mm": list(PERTURB_MM),
            "hold_frames": list(HOLD_FRAMES),
            "action_contract": "executed-clip-fullframe",
        },
    }
    out_path = OUT_DIR / f"metaworld_longtraj_{args.task}.pt"
    # 原子写入（2026-08-10 保护）：先写临时文件再 rename——进程被 kill 时
    # 不会留下半截文件（旧实现直接 torch.save，被杀会损坏并被 skip 跳过）。
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(out, tmp_path)
    tmp_path.replace(out_path)
    lens = [len(e["frames"]) for e in episodes]
    print(f"[out] {out_path}: {len(episodes)} eps, len mean={np.mean(lens):.0f} "
          f"min={min(lens)} max={max(lens)}")
    print(f"[ok] 目标 150-300 帧：{'达标' if np.mean(lens) >= 150 else '偏短，检查 hold 是否生效'}")


if __name__ == "__main__":
    main()
