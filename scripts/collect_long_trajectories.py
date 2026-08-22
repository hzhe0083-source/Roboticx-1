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
JPEG_QUALITY = 90  # 帧压缩：当前 env.render() 原生 480×480 RGB → JPEG
# Same contract as eval_metaworld.TASK35_EVAL50_SEEDS. Kept local so this
# CPU collector does not import the GPU eval module.
TASK35_EVAL50_SEEDS = tuple(range(35000, 35050))
DEFAULT_PINNED_ATTEMPTS = 10


def compress_frames(frames: np.ndarray) -> list[bytes]:
    """[T,H,W,3] uint8 render frames → JPEG bytes（保持原始像素尺寸）。"""
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


def reset_eval_init(env, episode_seed: int):
    """Recreate the eval50 init for ``episode_seed``.

    MetaWorld v3 ``reset(seed=)`` is ignored by ``SawyerXYZEnv.reset``;
    layout is drawn from the global NumPy RNG. Both lines are required.
    """
    seed = int(episode_seed)
    np.random.seed(seed)
    return env.reset(seed=seed)


def resolve_perturb_kinds(kinds: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    resolved = tuple(PERTURB_KINDS if kinds is None else kinds)
    unknown = [kind for kind in resolved if kind not in PERTURB_KINDS]
    if unknown:
        raise ValueError(f"unknown perturb kinds {unknown}; allowed={list(PERTURB_KINDS)}")
    if not resolved:
        raise ValueError("perturb kinds must be non-empty")
    return resolved


def blocked_eval50_seeds(seeds: list[int] | tuple[int, ...]) -> list[int]:
    allowed = set(TASK35_EVAL50_SEEDS)
    return [int(seed) for seed in seeds if int(seed) in allowed]


def check_eval_seed_policy(seeds: list[int] | tuple[int, ...], allow_eval_seeds: bool) -> None:
    blocked = blocked_eval50_seeds(seeds)
    if blocked and not allow_eval_seeds:
        raise ValueError(
            "episode seeds overlap eval50 "
            f"{TASK35_EVAL50_SEEDS[0]}-{TASK35_EVAL50_SEEDS[-1]}: {blocked}. "
            "Pass --allow-eval-seeds to train on those inits, or pick other seeds."
        )


def planned_episode_seeds(seeds: list[int], variants_per_seed: int) -> list[int]:
    if variants_per_seed < 1:
        raise ValueError("variants-per-seed must be >= 1")
    return [int(seed) for seed in seeds for _ in range(int(variants_per_seed))]


def collect_episode(
    env,
    policy,
    task_name: str,
    rng: np.random.Generator,
    perturb: bool = True,
    *,
    episode_seed: int | None = None,
    force_perturb: bool = False,
    perturb_kinds: list[str] | tuple[str, ...] | None = None,
) -> dict | None:
    """跑一条长轨迹：完整任务 + 成功 hold + （可选）接触扰动恢复。

    返回 dict（帧/动作/状态均为原始值，后续统一归一化）或 None（失败）。

    2026-08-10 保护：单条轨迹异常兜底——env 抖动/渲染失败不应炸掉整个
    任务进程（此前 20 实例并发时多次全灭，疑似 OOM/异常无兜底）。
    """
    try:
        return _collect_episode_inner(
            env,
            policy,
            task_name,
            rng,
            perturb,
            episode_seed=episode_seed,
            force_perturb=force_perturb,
            perturb_kinds=perturb_kinds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [collect_episode] 异常跳过本条: {type(exc).__name__}: {exc}")
        return None


def _collect_episode_inner(
    env,
    policy,
    task_name: str,
    rng: np.random.Generator,
    perturb: bool = True,
    *,
    episode_seed: int | None = None,
    force_perturb: bool = False,
    perturb_kinds: list[str] | tuple[str, ...] | None = None,
) -> dict | None:
    # MetaWorld v3 的 reset_model 仍从全局 NumPy RNG 取布局；仅传
    # env.reset(seed=...) 在部分版本不会控制它。两边同时设种子，保证同一个
    # collector seed 真正可复现，并把 episode_seed 写进样本供审计/去重。
    if episode_seed is None:
        episode_seed = int(rng.integers(0, 2**31))
    else:
        episode_seed = int(episode_seed)
    kinds = resolve_perturb_kinds(perturb_kinds)
    obs, _ = reset_eval_init(env, episode_seed)
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    action_success: list[bool] = []
    action_sources: list[str] = []
    lock_positions: list[np.ndarray] = []
    lock_targets: list[np.ndarray] = []
    lock_metric_valid: list[bool] = []
    first_success: int | None = None
    perturb_event: dict | None = None
    hold_target: int | None = None
    consec_success = 0

    def _record_lock_metric(current_obs: np.ndarray) -> None:
        """Record the pre-action door state on the same index as the action."""
        if task_name != "door-lock-v3":
            return
        pos = np.asarray(current_obs[4:7], dtype=np.float32)
        target = np.asarray(
            getattr(env, "_target_pos", np.full(3, np.nan)), dtype=np.float32
        ).reshape(-1)[:3]
        valid = pos.shape == (3,) and target.shape == (3,)
        if not valid:
            pos = np.full(3, np.nan, dtype=np.float32)
            target = np.full(3, np.nan, dtype=np.float32)
        lock_positions.append(pos)
        lock_targets.append(target)
        lock_metric_valid.append(bool(valid and np.isfinite(pos).all() and np.isfinite(target).all()))

    def _execute(action: np.ndarray, source: str):
        """Store (pre-action observation, action), then execute exactly once."""
        nonlocal obs, first_success, hold_target, consec_success
        index = len(actions)
        frames.append(env.render())
        actions.append(np.asarray(action, dtype=np.float32))
        states.append(np.asarray(obs[:4], dtype=np.float32))
        action_sources.append(source)
        _record_lock_metric(obs)
        obs, reward, term, trunc, info = env.step(action)
        succeeded = bool(info.get("success"))
        action_success.append(succeeded)
        if succeeded and first_success is None:
            # Index in the stored frames/actions timeline, including settle actions.
            first_success = index
        if succeeded:
            if hold_target is None:
                hold_target = int(rng.integers(*HOLD_FRAMES))
            consec_success += 1
        else:
            consec_success = 0
        return reward, term, trunc, info

    for policy_step in range(500):
        a = np.clip(np.asarray(policy.get_action(obs), dtype=np.float32), -1.0, 1.0)
        # 契约（Codex P0-1，2026-08-09）：先保存【当前观测 + 即将执行的动作】，
        # 再 step。此前顺序是 step 后才保存 (o_{t+1}, a_t)——图像/状态泄漏
        # a_t 的执行结果且 prev 错位，整批数据作废需重采。
        _r, term, trunc, info = _execute(a, "policy")
        # 接触扰动：成功前随机一步注入 2-8mm，之后继续 policy 恢复
        if (perturb and perturb_event is None and first_success is None
                and not term and not trunc and policy_step > 20
                and (force_perturb or rng.random() < 0.05)):
            mag = rng.uniform(*PERTURB_MM) / 1000.0
            kind = str(rng.choice(kinds))
            applied = _apply_perturb(env, kind, mag, rng)
            if applied["applied"]:
                # MuJoCo state changed outside env.step; refresh the proprio/object
                # observation before recording the first settle action.
                get_obs = getattr(env, "_get_obs", None)
                if callable(get_obs):
                    obs = get_obs()
                start = len(actions)
                settle_truncated = False
                for _ in range(PERTURB_SETTLE_STEPS):
                    _r2, _t2, _tr2, _i2 = _execute(
                        np.zeros(4, dtype=np.float32), "perturb_settle"
                    )
                    if _t2 or _tr2:
                        settle_truncated = True
                        break
                perturb_event = {
                    "start": start,
                    "end": len(actions),  # exclusive
                    "kind": kind,
                    "magnitude_m": float(mag),
                    "magnitude_mm": float(mag * 1000.0),
                    "delta": applied["delta"],
                    "applied": True,
                }
                if settle_truncated:
                    term, trunc = True, True
        # hold：所有真实执行步（包括 settle）都参与成功状态计数。
        if first_success is not None and consec_success >= hold_target:
            break
        if term or trunc:
            break
    if first_success is None:
        return None  # 纯失败轨迹，丢弃（Codex P0-2 核心：剔除失败长尾）
    # hold 未达标但含成功段：保留（精密任务 success 信号会抖动，连续达标
    # 过苛会整条丢弃；全局成功跨度 >= HOLD_FRAMES[0] 即视为含稳定成功段）。
    completed_hold = hold_target is not None and consec_success >= hold_target
    if (not completed_hold
            and len(actions) - 1 - first_success < int(HOLD_FRAMES[0])):
        return None

    n = len(actions)
    settle_mask = np.asarray(
        [source == "perturb_settle" for source in action_sources], dtype=bool
    )
    frame_valid = np.ones(n, dtype=bool)
    action_executed = np.ones(n, dtype=bool)
    action_supervision_valid = ~settle_mask
    action_supervision_valid[first_success + 1:] = False
    recovery_mask = np.zeros(n, dtype=bool)
    if perturb_event is not None:
        recovery_mask[perturb_event["start"]:first_success + 1] = True

    result = {
        "episode_seed": episode_seed,
        "frames": compress_frames(np.stack(frames)),  # list[bytes] JPEG
        "actions": np.stack(actions),     # [T, 4]
        "states": np.stack(states),       # [T, 4]
        # success_frame is retained as a precise compatibility alias. Unlike the
        # v1 collector it now uses the stored action timeline, including settle.
        "first_success": first_success,
        "success_frame": first_success,
        "action_success": np.asarray(action_success, dtype=bool),
        "frame_valid": frame_valid,
        "action_executed": action_executed,
        "action_source": action_sources,
        "settle_mask": settle_mask,
        "action_valid": action_supervision_valid,
        "action_supervision_valid": action_supervision_valid,
        "recovery_mask": recovery_mask,
        "perturbed": perturb_event is not None,
        "n_perturb_events": int(perturb_event is not None),
        "perturb_start": None if perturb_event is None else perturb_event["start"],
        "perturb_end": None if perturb_event is None else perturb_event["end"],
        "perturb_kind": None if perturb_event is None else perturb_event["kind"],
        "perturb_magnitude": (
            None if perturb_event is None else perturb_event["magnitude_m"]
        ),
        "perturb_magnitude_mm": (
            None if perturb_event is None else perturb_event["magnitude_mm"]
        ),
        "perturb_event": perturb_event,
    }
    if task_name == "door-lock-v3":
        result.update({
            "lock_pos": np.stack(lock_positions),
            "lock_target": np.stack(lock_targets),
            "metric_state": np.concatenate(
                [np.stack(lock_positions), np.stack(lock_targets)], axis=-1
            ),
            "metric_state_valid": np.asarray(lock_metric_valid, dtype=bool),
        })
    return result


def _apply_perturb(env, kind: str, mag: float,
                   rng: np.random.Generator) -> dict:
    """注入 2-8mm 扰动（复用 prepare_mw_perturbations 的机制：mocap/obj1）。

    settle 步由调用方进入时间轴（PERTURB_SETTLE_STEPS 循环），此处只改环境。
    """
    if kind in ("eef_lateral", "eef_height"):
        delta = np.zeros(3)
        if kind == "eef_lateral":
            theta = rng.uniform(0, 2 * np.pi)
            delta[:2] = mag * np.array([np.cos(theta), np.sin(theta)])
        else:
            delta[2] = mag
        env.data.mocap_pos[0] += delta
        return {"applied": True, "delta": delta.astype(np.float32)}
    else:
        from mw_expert_replay import move_body

        delta = np.zeros(3)
        theta = rng.uniform(0, 2 * np.pi)
        delta[:2] = mag * np.array([np.cos(theta), np.sin(theta)])

        def _is_robot_body(env, bid: int) -> bool:
            while bid > 0:
                name = env.model.body(bid).name or ""
                if name.startswith(("right", "left")) or "claw" in name or name == "hand":
                    return True
                bid = int(env.model.body_parentid[bid])
            return False

        cur = env._get_obs()[4:7].copy()
        moved = False
        for i in range(env.model.nsite):
            if _is_robot_body(env, int(env.model.site_bodyid[i])):
                continue
            if np.allclose(env.data.site_xpos[i], cur, atol=0.02):
                move_body(env, int(env.model.site_bodyid[i]), delta)
                moved = True
                break
        if not moved:
            for bid in range(env.model.nbody):
                if _is_robot_body(env, bid):
                    continue
                name = env.model.body(bid).name
                if not name:
                    continue
                if np.allclose(env.data.body(bid).xpos, cur, atol=0.02):
                    move_body(env, bid, delta)
                    moved = True
                    break
        if not moved:
            return {"applied": False, "delta": np.zeros(3, dtype=np.float32)}
        return {"applied": True, "delta": delta.astype(np.float32)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="MT1 任务名，如 peg-insert-side-v3")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--no-perturb", action="store_true", help="不注入接触扰动")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-seeds",
        type=int,
        nargs="+",
        help="Pin MetaWorld inits (np.random.seed + env.reset). Not the same as --seed.",
    )
    parser.add_argument(
        "--variants-per-seed",
        type=int,
        default=1,
        help="Accepted recovery variants per pinned episode seed (default 1)",
    )
    parser.add_argument(
        "--force-perturb",
        action="store_true",
        help="Inject one perturb on the first eligible pre-success step (skip the 5% gate)",
    )
    parser.add_argument(
        "--perturb-kinds",
        nargs="+",
        choices=list(PERTURB_KINDS),
        help="Restrict perturb kinds; default is all four kinds",
    )
    parser.add_argument(
        "--allow-eval-seeds",
        action="store_true",
        help="Allow episode seeds in 35000-35049 (eval50 inits). Default is forbid.",
    )
    parser.add_argument(
        "--normalization-ref", type=Path, default=None,
        help="继承 action/state q01/q99 的来源文件；缺省是 9.8GB 的 "
        "data/metaworld_fullframe_executed.pt。只读取其 ``normalization`` 字段，"
        "所以任何已继承同一份归一化的文件都等价（可用小参考文件避免搬 9.8GB）。",
    )
    parser.add_argument(
        "--output", type=Path,
        help="新输出路径；默认使用带 clean/recovery、v2、seed 后缀的新文件",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="显式允许覆盖 --output（默认拒绝覆盖任何现有数据）",
    )
    return parser


def _accept_episode(ep: dict | None, *, recovery: bool) -> tuple[dict | None, str | None]:
    if ep is None:
        return None, "failed-or-no-success"
    if recovery and not ep["perturbed"]:
        return None, "nominal"
    return ep, None


def collect_requested_episodes(
    env,
    policy,
    task_name: str,
    rng: np.random.Generator,
    *,
    perturb: bool,
    force_perturb: bool,
    perturb_kinds: list[str] | tuple[str, ...] | None,
    episode_seeds: list[int] | None,
    n_episodes: int,
    allow_eval_seeds: bool,
) -> list[dict]:
    recovery = perturb
    if episode_seeds:
        planned = planned_episode_seeds(episode_seeds, 1)
        # variants are expanded by the caller so retries stay on one init.
        check_eval_seed_policy(planned, allow_eval_seeds)
        target = planned
    else:
        target = [None] * n_episodes

    episodes: list[dict] = []
    attempts = 0
    max_attempts = max(len(target) * DEFAULT_PINNED_ATTEMPTS, n_episodes * 10)
    seed_index = 0
    while seed_index < len(target) and attempts < max_attempts:
        attempts += 1
        pinned = target[seed_index]
        if pinned is None and not allow_eval_seeds:
            # Peek the next random seed without consuming perturb RNG: draw, reject
            # eval50 collisions, then collect with that exact seed.
            candidate = int(rng.integers(0, 2**31))
            if blocked_eval50_seeds([candidate]):
                print(f"  skip eval50 seed {candidate}: pass --allow-eval-seeds to keep it")
                continue
            pinned = candidate
        ep = collect_episode(
            env,
            policy,
            task_name,
            rng,
            perturb=perturb,
            episode_seed=pinned,
            force_perturb=force_perturb,
            perturb_kinds=perturb_kinds,
        )
        accepted, reason = _accept_episode(ep, recovery=recovery)
        if accepted is None:
            if reason == "nominal":
                print("  skip nominal episode: recovery collection requires >=1 perturb event")
            continue
        if not allow_eval_seeds and blocked_eval50_seeds([accepted["episode_seed"]]):
            print(
                f"  skip eval50 seed {accepted['episode_seed']}: "
                "pass --allow-eval-seeds to keep it"
            )
            continue
        episodes.append(accepted)
        seed_index += 1
        lens = [len(e["frames"]) for e in episodes]
        print(
            f"  ep {len(episodes)}: seed={accepted['episode_seed']} "
            f"len={len(accepted['frames'])} success@{accepted['success_frame']} "
            f"perturbed={accepted['perturbed']} (mean_len={np.mean(lens):.0f})"
        )
    if not episodes:
        raise SystemExit("no successful episodes collected")
    if len(episodes) != len(target):
        raise SystemExit(
            f"collected only {len(episodes)}/{len(target)} contract-valid episodes "
            f"after {attempts} attempts"
        )
    return episodes


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.variants_per_seed < 1:
        raise SystemExit("--variants-per-seed must be >= 1")
    if args.no_perturb and args.force_perturb:
        raise SystemExit("--force-perturb is incompatible with --no-perturb")
    if args.episode_seeds:
        pinned = planned_episode_seeds(args.episode_seeds, args.variants_per_seed)
        check_eval_seed_policy(pinned, args.allow_eval_seeds)
        n_episodes = len(pinned)
    else:
        pinned = None
        n_episodes = args.episodes

    rng = np.random.default_rng(args.seed)
    env = make_env(args.task)
    policy = get_policy(args.task)
    print(
        f"task={args.task} policy={type(policy).__name__} episodes={n_episodes} "
        f"force_perturb={args.force_perturb} allow_eval_seeds={args.allow_eval_seeds}"
    )

    if pinned is None:
        episodes = collect_requested_episodes(
            env,
            policy,
            args.task,
            rng,
            perturb=not args.no_perturb,
            force_perturb=args.force_perturb,
            perturb_kinds=args.perturb_kinds,
            episode_seeds=None,
            n_episodes=n_episodes,
            allow_eval_seeds=args.allow_eval_seeds,
        )
    else:
        episodes = []
        for seed in pinned:
            batch = collect_requested_episodes(
                env,
                policy,
                args.task,
                rng,
                perturb=not args.no_perturb,
                force_perturb=args.force_perturb,
                perturb_kinds=args.perturb_kinds,
                episode_seeds=[seed],
                n_episodes=1,
                allow_eval_seeds=args.allow_eval_seeds,
            )
            episodes.extend(batch)
    if args.no_perturb:
        if any(ep["perturbed"] or ep["n_perturb_events"] for ep in episodes):
            raise RuntimeError("clean collection contains a perturbation event")
    elif any(not ep["perturbed"] or ep["n_perturb_events"] < 1 for ep in episodes):
        raise RuntimeError("recovery collection contains a nominal episode")
    env.close()

    # ---- 统一归一化：继承 fullframe executed 的 q01/q99（Codex：禁止单独算） ----
    import torch
    norm_ref = args.normalization_ref or (
        ROOT / "data" / "metaworld_fullframe_executed.pt"
    )
    if not Path(norm_ref).is_file():
        raise SystemExit(
            f"missing normalization reference: {norm_ref}\n"
            "继承 q01/q99 是硬契约（禁止单独算）。若默认的 9.8GB fullframe 文件不在，"
            "用 --normalization-ref 指向任一已继承同一份归一化的文件。"
        )
    src = torch.load(norm_ref, map_location="cpu", weights_only=True)
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
            "contract": "long_trajectory_scripted_v2",
            "contract_version": 2,
            "fps": FPS,
            "control_stride": CONTROL_STRIDE,
            "policy": type(policy).__name__,
            # Keep the legacy field, but make configured-vs-observed semantics
            # explicit. Episode `perturbed` records whether an event was applied.
            "perturbed": not args.no_perturb,
            "perturbation_enabled": not args.no_perturb,
            "perturbation_mode": (
                "disabled_by_no_perturb" if args.no_perturb
                else "forced_single_pre_success" if args.force_perturb
                else "single_random_pre_success"
            ),
            "perturbation_data_present": any(ep["perturbed"] for ep in episodes),
            "eval_seed_policy": (
                "allow-eval-seeds" if args.allow_eval_seeds else "forbid"
            ),
            "pinned_episode_seeds": (
                None if args.episode_seeds is None else [int(s) for s in args.episode_seeds]
            ),
            "variants_per_seed": int(args.variants_per_seed),
            "perturb_kinds": list(resolve_perturb_kinds(args.perturb_kinds)),
            "episode_perturbation_contract": (
                "clean: n_perturb_events=0 for every episode; "
                "recovery: n_perturb_events>=1 for every episode"
            ),
            "perturb_mm": list(PERTURB_MM),
            "perturb_settle_steps": PERTURB_SETTLE_STEPS,
            "hold_frames": list(HOLD_FRAMES),
            "action_contract": "executed-clip-fullframe",
            "observation_action_alignment": (
                "frames/states/metric_state[i] are pre-action observations; "
                "actions[i] was executed exactly once; action_success[i] is its post-step result"
            ),
            "index_contract": (
                "first_success, perturb_start and perturb_end index the stored action timeline; "
                "perturb_end is exclusive"
            ),
            "supervision_contract": (
                "action_supervision_valid excludes perturb settle and all actions after first_success"
            ),
            "door_metric_state": "[lock_pos_xyz=obs[4:7], lock_target_xyz=env._target_pos]",
        },
    }
    mode = "clean" if args.no_perturb else "recovery"
    out_path = args.output or (
        OUT_DIR / f"metaworld_longtraj_{args.task}_{mode}_v2_seed{args.seed}.pt"
    )
    protected = {
        (OUT_DIR / "metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt").resolve(),
        (OUT_DIR / "metaworld_longtraj_peg-insert-side-v3_clean_v2_seed350.pt").resolve(),
        (OUT_DIR / "metaworld_longtraj_peg-insert-side-v3_recovery_v2_seed351.pt").resolve(),
    }
    if out_path.resolve() in protected:
        raise FileExistsError(f"refusing to write elected/source dataset: {out_path}")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing data: {out_path}; choose --output or pass --overwrite"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写入（2026-08-10 保护）：先写临时文件再 rename——进程被 kill 时
    # 不会留下半截文件（旧实现直接 torch.save，被杀会损坏并被 skip 跳过）。
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    if tmp_path.exists():
        raise FileExistsError(f"stale temporary output exists: {tmp_path}")
    torch.save(out, tmp_path)
    tmp_path.replace(out_path)
    lens = [len(e["frames"]) for e in episodes]
    print(f"[out] {out_path}: {len(episodes)} eps, len mean={np.mean(lens):.0f} "
          f"min={min(lens)} max={max(lens)}")
    print(f"[ok] 目标 150-300 帧：{'达标' if np.mean(lens) >= 150 else '偏短，检查 hold 是否生效'}")


if __name__ == "__main__":
    main()
