#!/usr/bin/env python
"""C²-VA Stage B 恢复数据收集（v6b，Codex 评审 2026-08-07 版）。

从 MetaWorld 任务（默认 button-press-v3；--env / --task-text / --policy-name
可切换，如 peg-insert-side-v3）本地脚本专家成功轨迹的 phase-stratified anchors
分叉 nominal/perturbed 配对分支，收集"扰动恢复"transitions 与冻结 PCA 控制
投影 P 的权重：

- 50 个成功 expert seeds × 5 个 phase-stratified anchors × 4 个扰动分支
  = 1000 paired branches（seed 80/20 split，约 6000 recovery transitions）；
- 扰动类型（一次只施一种，Codex 修正 4；混合权重 --perturb-mix 可配，
  默认 0.5,0.3,0.2 = action,tcp,object_joint）：
  * action 动作注入：ε_xyz ~ N(0, 0.5²)，clip 后真实执行（gripper 保持专家值）；
  * tcp：手 base 平移 σ=1cm、逐轴 cap 2cm；
  * object_joint（button 等带 object joint 任务专用）：仅向"未按下"方向
    回弹 σ=5mm、cap 1cm；非 object-joint 任务未显式传 --perturb-mix 时
    自动降级为 action/tcp 各半，显式传含 object_joint 的 mix 则报错；
- 每 branch 跑 6 恢复步（专家恢复动作），每步存扰动观测窗口帧（单独 render，
  绝不复用 clean 特征）、归一化 proprio/prev、专家恢复动作（executed 空间）；
- snapshot/restore 覆盖 qpos/qvel/mocap/act/time/_prev_obs/target；
- 全部收集后从 train split 的差向量（z_perturbed − z_nominal 的 LN 空间）
  做 top-16 PCA + whitening → P 权重（Linear(768,16)），投影出每步的
  c_perturbed / c_nominal（16 维），与 vision_tokens_t 一起写入输出文件。

输出 data/mw_buttonpress_v6b.pt（键见模块底部 build_payload）：
- 训练（train.py --c2-controller）：vision_tokens_t / proprio / prev_action /
  expert_action / step_index / branch_id / c_perturbed / c_nominal /
  language_hidden / language_mask / pca{weight,bias}；
- 评估（eval_metaworld.py --c2-recovery-eval）：recovery_start（每 branch 的
  完整 snapshot + reset_seed + prev_action + split）；
- 指标（train.py compute_contract_metrics）：split / branch_id / step_index /
  c_perturbed / c_nominal。

用法（GPU 任务，GPU 空闲后运行；CPU 仿真不占 GPU）：
    python prepare_mw_recovery.py [--seeds 50 --anchors 5 --branches 4 \
        --output data/mw_buttonpress_v6b.pt --device cuda]
    # 第二任务（peg-insert-side，无 object joint → 默认混合自动降级 action/tcp）：
    python prepare_mw_recovery.py --env peg-insert-side-v3 \
        --task-text "Insert a peg sideways" --policy-name peg-insert-side \
        --output data/mw_peginsert_v6b.pt

依赖：metaworld（本地脚本专家；--policy-name 映射 button-press →
SawyerButtonPressV3Policy、peg-insert-side → SawyerPegInsertionSideV3Policy）、
V-JEPA 2.1 本地权重（va_compound/backbones.VJEPA21Backbone.from_pretrained
local_files_only=True）、data/metaworld_features_v5.pt（归一化 + 语言条件）。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/home/ryan/Documents/robot/ORA0")
MW_CONFIG = (
    Path("/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json")
)
FEATURES_PATH = ROOT / "data" / "metaworld_features_v5.pt"
DEFAULT_TASK = "Press a button"
DEFAULT_ENV = "button-press-v3"
DEFAULT_POLICY = "button-press"
DEFAULT_PERTURB_MIX = "0.5,0.3,0.2"  # 逗号分隔 3 权重，顺序固定 action,tcp,object_joint
VISION_WINDOW = 4
VISION_STRIDE = 2
CONTROL_STRIDE = 6
CONTROL_DIM = 16

# 扰动配置（Codex 修正 4）
PERTURB_KIND_ORDER = ("action", "tcp", "object_joint")
# 脚本专家策略映射（--policy-name → metaworld.policies 类名）。
# 注：metaworld.policies 中 peg-insert-side 的实际类名是
# SawyerPegInsertionSideV3Policy（SawyerPegInsertSideV3Policy 不存在）。
POLICY_MAP = {
    "button-press": "SawyerButtonPressV3Policy",
    "peg-insert-side": "SawyerPegInsertionSideV3Policy",
}
ACTION_EPS_STD = 0.5  # 动作注入：N(0, 0.5²)（手臂 xyz；gripper 保持专家值）
TCP_STD = 0.01  # TCP：σ=1cm
TCP_CAP = 0.02  # TCP：逐轴 cap 2cm
BUTTON_STD = 0.005  # object_joint：σ=5mm
BUTTON_CAP = 0.01  # object_joint：cap 1cm
RECOVERY_STEPS = 6
ANCHOR_MARGIN = 1  # anchor 至少离轨迹首尾 1 步


def resolve_perturb_mix(
    mix_arg: str | None,
    has_object_joint: bool,
    default: str = DEFAULT_PERTURB_MIX,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """解析/校验扰动混合 → (kinds, weights)（权重已归一化，顺序 action,tcp,object_joint）。

    - 未显式传 mix 且环境无 object joint → 自动降级 ('action','tcp') / (0.5, 0.5)；
    - 显式传 mix 且 object_joint 权重 > 0 但环境无 object joint → ValueError；
    - object_joint 权重 = 0 时等价纯 action/tcp 混合（非 object-joint 任务可用）。
    """
    explicit = mix_arg is not None
    text = mix_arg if explicit else default
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"--perturb-mix 需要恰好 3 个逗号分隔权重（顺序 action,tcp,object_joint），"
            f"got {text!r}"
        )
    try:
        weights = [float(part) for part in parts]
    except ValueError:
        raise ValueError(
            f"--perturb-mix 权重必须为数字（顺序 action,tcp,object_joint），got {text!r}"
        ) from None
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"--perturb-mix 权重必须非负且和 > 0，got {text!r}")
    if weights[2] > 0 and not has_object_joint:
        if not explicit:
            return ("action", "tcp"), (0.5, 0.5)
        raise ValueError(
            "--perturb-mix 含 object_joint（object joint qpos 回弹）扰动，但当前任务环境"
            "没有 object joint（如 button 类）；非 object-joint 任务请用纯 action,tcp "
            "混合（例如 0.5,0.5,0）"
        )
    total = sum(weights)
    return PERTURB_KIND_ORDER, tuple(w / total for w in weights)


def resolve_policy_class(policy_name: str, policy_class: str | None):
    """--policy-name 映射或 --policy-class 类名 → metaworld.policies 策略类。"""
    import metaworld.policies as mp

    if policy_class is not None:
        cls = getattr(mp, policy_class, None)
        if cls is None:
            raise SystemExit(f"unknown policy class {policy_class!r} in metaworld.policies")
        return cls
    cls = getattr(mp, POLICY_MAP.get(policy_name, policy_name), None)
    if cls is None:
        raise SystemExit(
            f"unknown policy {policy_name!r} — 可用映射: {sorted(POLICY_MAP)}，"
            "或 --policy-class 直接传 metaworld.policies 类名"
        )
    return cls


def _joint_name(model, jnt: int) -> str:
    """跨版本安全地取 joint 名字。"""
    import mujoco

    try:
        return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt))
    except Exception:
        return str(model.joint(jnt).name)


def find_button_qpos_index(env) -> int:
    """按钮自由 joint 在 qpos 中的起始索引（button-press 的 slider）。"""
    model = env.model
    for jnt in range(model.njnt):
        name = _joint_name(model, jnt)
        if name and "button" in name.lower():
            return int(model.jnt_qposadr[jnt])
    for jnt in range(model.njnt):
        body = model.body(model.jnt_bodyid[jnt])
        if "button" in str(body.name).lower():
            return int(model.jnt_qposadr[jnt])
    raise ValueError("button joint not found in env model")


def snapshot_env(env) -> dict:
    """完整状态快照（Codex 修正 4：qpos/qvel/mocap/act/time/_prev_obs/target）。"""
    data = env.data
    return {
        "qpos": np.array(data.qpos, dtype=np.float64).copy(),
        "qvel": np.array(data.qvel, dtype=np.float64).copy(),
        "act": (
            np.array(data.act, dtype=np.float64).copy()
            if data.act is not None and len(data.act)
            else np.zeros(0, dtype=np.float64)
        ),
        "time": float(data.time),
        "mocap_pos": (
            np.array(data.mocap_pos, dtype=np.float64).copy()
            if env.model.nmocap
            else np.zeros((0, 3), dtype=np.float64)
        ),
        "mocap_quat": (
            np.array(data.mocap_quat, dtype=np.float64).copy()
            if env.model.nmocap
            else np.zeros((0, 4), dtype=np.float64)
        ),
        "prev_obs": np.array(getattr(env, "_prev_obs", np.zeros(18)), dtype=np.float64).copy(),
        "target_pos": np.array(getattr(env, "_target_pos", np.zeros(3)), dtype=np.float64).copy(),
    }


def _as_numpy(value) -> np.ndarray:
    """快照值 → numpy（兼容 torch 张量存储的数据文件与 numpy 内存快照）。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def restore_env(env, snapshot: dict) -> None:
    """从快照恢复环境（确定性：同一 snapshot 两次恢复状态一致）。"""
    import mujoco

    data = env.data
    data.qpos[:] = _as_numpy(snapshot["qpos"])
    data.qvel[:] = _as_numpy(snapshot["qvel"])
    if len(_as_numpy(snapshot["act"])) == len(data.act):
        data.act[:] = _as_numpy(snapshot["act"])
    data.time = float(snapshot["time"])
    if env.model.nmocap and len(_as_numpy(snapshot["mocap_pos"])) == env.model.nmocap:
        data.mocap_pos[:] = _as_numpy(snapshot["mocap_pos"])
        data.mocap_quat[:] = _as_numpy(snapshot["mocap_quat"])
    if hasattr(env, "_prev_obs"):
        env._prev_obs = _as_numpy(snapshot["prev_obs"]).copy()
    if hasattr(env, "_target_pos"):
        env._target_pos = _as_numpy(snapshot["target_pos"]).copy()
    mujoco.mj_forward(env.model, env.data)


def snapshot_to_tensors(snapshot: dict) -> dict:
    """快照转 torch 张量存储（weights_only=True 可加载的数据文件）。"""
    converted = {}
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray):
            converted[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            converted[key] = value
    return converted


def perturb_state(env, kind: str, rng: np.random.Generator) -> str:
    """对当前状态施加一次物理扰动（只改状态，不改动作输入）。

    action：无状态扰动（扰动作用在第 0 步执行动作上，见 run_branch）；
    tcp：手 base 平移 δ ~ N(0, 0.01²)，逐轴 cap 0.02；
    object_joint：object joint（如 button-press 的按钮）向未按下方向回弹
    |δ|（σ=0.005，cap 0.01）；仅限带 object joint 的任务
    （resolve_perturb_mix 已校验）。
    """
    import mujoco

    if kind == "action":
        return "action-injection (applied to step-0 executed action)"
    if kind == "tcp":
        delta = np.clip(rng.normal(0.0, TCP_STD, size=3), -TCP_CAP, TCP_CAP)
        env.data.qpos[:3] += delta
        mujoco.mj_forward(env.model, env.data)
        return f"tcp hand base moved by {np.round(delta, 4)}"
    if kind == "object_joint":
        index = find_button_qpos_index(env)
        delta = min(abs(float(rng.normal(0.0, BUTTON_STD))), BUTTON_CAP)
        # 实测按下 = joint 正向（0 → +），回弹 = 负向（"未按下"方向；--debug 验证）
        env.data.qpos[index] -= delta
        mujoco.mj_forward(env.model, env.data)
        return f"object joint rebounded by -{delta:.4f} (qpos={env.data.qpos[index]:.4f})"
    raise ValueError(f"unknown perturbation kind: {kind}")


def phase_stratified_anchors(success_step: int, n_anchors: int) -> list[int]:
    """沿成功轨迹均匀取阶段锚点（不含首尾 ANCHOR_MARGIN 步）。

    可插入锚点的区间不足 n_anchors 时返回 []（轨迹过短，无阶段可分层）。
    """
    if success_step <= 2 * ANCHOR_MARGIN:
        return []
    low = ANCHOR_MARGIN
    high = max(low + 1, success_step - ANCHOR_MARGIN)
    if high - low + 1 < n_anchors:
        return []
    anchors = np.linspace(low, high, n_anchors, dtype=int).tolist()
    return sorted(set(a for a in anchors if low <= a < success_step))


def make_env(seed: int, env_name: str):
    """与 eval_metaworld.py 完全相同的采集同款环境构造。"""
    import metaworld

    mt1 = metaworld.MT1(env_name, seed=42)
    env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]  # corner2 位置（lerobot 采集同款）
    env._freeze_rand_vec = False
    env.reset(seed=seed)
    return env


def window_frames(frame_log: dict[int, np.ndarray], step: int) -> list[np.ndarray]:
    """4 帧 stride-2 窗口 [s-6, s-4, s-2, s]（与 prepare_metaworld.py 的
    clip_frame_indices 一致；越界帧用最老可用帧重复，episode 首窗口同款）。"""
    indices = [max(0, step - offset * VISION_STRIDE) for offset in range(3, -1, -1)]
    frames = []
    for index in indices:
        if index in frame_log:
            frames.append(frame_log[index])
        else:
            # 缓存淘汰/越界：用缓存中最老帧（首窗口语义：重复帧）
            oldest = min(frame_log, default=index)
            frames.append(frame_log[oldest])
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "mw_buttonpress_v6b.pt")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--env", type=str, default=DEFAULT_ENV,
                        help="MetaWorld env 名（如 button-press-v3 / peg-insert-side-v3）")
    parser.add_argument("--task-text", type=str, default=DEFAULT_TASK,
                        help="v5 features metadata.tasks 中的任务文本")
    parser.add_argument("--policy-name", type=str, default=DEFAULT_POLICY,
                        help="脚本专家策略：POLICY_MAP 键（button-press / peg-insert-side）"
                             "或 metaworld.policies 类名")
    parser.add_argument("--policy-class", type=str, default=None,
                        help="直接指定 metaworld.policies 类名（覆盖 --policy-name）")
    parser.add_argument("--perturb-mix", type=str, default=None,
                        help=f"扰动混合权重，逗号分隔 3 个（顺序 action,tcp,object_joint；"
                             f"默认 {DEFAULT_PERTURB_MIX!r}；非 object-joint 任务未显式传时"
                             "自动降级为 0.5,0.5）")
    parser.add_argument("--seeds", type=int, default=50, help="成功专家轨迹种子数")
    parser.add_argument("--anchors", type=int, default=5, help="每轨迹 phase-stratified 锚点数")
    parser.add_argument("--branches", type=int, default=4, help="每锚点扰动分支数")
    parser.add_argument("--recovery-steps", type=int, default=RECOVERY_STEPS)
    parser.add_argument("--max-steps", type=int, default=500, help="主轨迹最大步数")
    parser.add_argument("--split-every", type=int, default=5, help="seed % N == 0 → held-out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16, help="V-JEPA 编码批大小")
    parser.add_argument("--dry-run", action="store_true", help="1 seed × 1 anchor × 1 branch")
    parser.add_argument("--debug", action="store_true", help="打印 object joint 范围与扰动细节")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1 or args.anchors < 1 or args.branches < 1 or args.recovery_steps < 1:
        raise SystemExit("seeds/anchors/branches/recovery-steps must be positive")
    if not (0.0 < 1.0 / args.split_every < 1.0):
        raise SystemExit("--split-every must be >= 2 (held-out 占比 = 1/split_every)")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # 环境探测：确认 env 可用、是否有 object joint（决定扰动混合合法性）
    probe = make_env(0, args.env)
    try:
        find_button_qpos_index(probe)
        has_object_joint = True
    except ValueError:
        has_object_joint = False
    probe.close()
    kinds, perturb_weights = resolve_perturb_mix(
        args.perturb_mix, has_object_joint=has_object_joint
    )
    if args.perturb_mix is None and not has_object_joint:
        print(
            f"WARNING: env {args.env} 无 object joint —— 默认扰动混合 "
            f"{DEFAULT_PERTURB_MIX!r} 自动降级为 action/tcp (0.5, 0.5)"
        )

    features = torch.load(args.features, map_location="cpu", weights_only=True)
    sq01 = features["normalization"]["state_q01"].numpy()
    sq99 = features["normalization"]["state_q99"].numpy()
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    aq01 = features["normalization"]["action_q01"].numpy()
    aq99 = features["normalization"]["action_q99"].numpy()
    tasks = features["metadata"]["tasks"]
    if args.task_text not in tasks:
        raise SystemExit(f"task {args.task_text!r} not found in {args.features}")
    task_index = tasks.index(args.task_text)
    row = int((features["instruction_id"] == task_index).nonzero()[0][0])
    # clone：切片是 v5 大张量的视图（.to(fp16) 同 dtype 返回自身共享 storage，
    # torch.save 会序列化整个 v5 语言张量——必须断开）。
    language_hidden = features["language_hidden"][row : row + 1].to(torch.float16).clone()
    language_mask = features["language_mask"][row : row + 1].clone()

    def norm_state(raw: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (raw - sq01) / scale_s - 1.0, -1.0, 1.0).astype(np.float32)

    def norm_action(raw: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.clip(raw, -1.0, 1.0) - (aq01 + aq99) / 2) / np.where(
                np.abs(aq99 - aq01) < 1e-6, 1.0, (aq99 - aq01) / 2
            ),
            -1.0,
            1.0,
        ).astype(np.float32)

    from prepare_metaworld import preprocess_batch
    from va_compound.backbones import VJEPA21Backbone

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=args.device, dtype="float16", max_tokens=64, local_files_only=True
    )
    vision_backbone.freeze_all()

    def encode_windows(windows: list[list[np.ndarray]]) -> list[torch.Tensor]:
        """窗口批编码 → 64 token fp16 特征（每窗口一次 V-JEPA 前向）。"""
        out: list[torch.Tensor] = []
        for start in range(0, len(windows), args.batch_size):
            chunk = windows[start : start + args.batch_size]
            inputs = preprocess_batch(chunk, args.image_size).to(args.device)
            with torch.inference_mode():
                flat, _ = vision_backbone.forward_variants(inputs)
            out.extend(t.to(device="cpu", dtype=torch.float16).contiguous() for t in flat)
        return out

    expert_policy = resolve_policy_class(args.policy_name, args.policy_class)()

    # ---- 收集 ----
    transitions: list[dict] = []  # 每项：窗口帧、proprio、prev、expert_action、step/branch/seed/split
    recovery_start: list[dict] = []  # 每 branch 的初始扰动 snapshot
    lz_samples: list[dict] = []  # 每 transition 的 (lz_delta, lz_nominal) 768D（PCA 用）
    branch_counter = 0
    success_seeds = 0
    n_seeds = 1 if args.dry_run else args.seeds
    n_anchors = 1 if args.dry_run else args.anchors
    n_branches = 1 if args.dry_run else args.branches
    for seed in range(n_seeds):
        # ---- 第一遍：专家跑到 success（无渲染，快），确定 anchors ----
        env = make_env(seed, args.env)
        step = 0
        success_step = -1
        while step < args.max_steps:
            obs = env._get_obs()
            action = expert_policy.get_action(obs)
            obs, _reward, terminated, truncated, info = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            if info.get("success"):
                success_step = step
                break
            if terminated or truncated:
                break
        if success_step < 0:
            env.close()
            print(f"seed={seed}: expert FAIL (no success in {step} steps), skip")
            continue
        anchors = phase_stratified_anchors(success_step, n_anchors)
        if not anchors:
            env.close()
            print(f"seed={seed}: trajectory too short (success at {success_step}), skip")
            continue
        success_seeds += 1
        env.close()
        # ---- 第二遍：确定性重放（同 seed → 同轨迹），渲染帧，anchor 处 snapshot ----
        env = make_env(seed, args.env)
        frame_log: dict[int, np.ndarray] = {}
        executed_log: dict[int, np.ndarray] = {}
        snapshots: dict[int, dict] = {}
        anchor_set = set(anchors)
        step = 0
        replay_stop = max(anchors)
        while step <= replay_stop and step < args.max_steps:
            obs = env._get_obs()
            action = expert_policy.get_action(obs)
            executed = np.clip(action, -1.0, 1.0)
            frame_log[step] = env.render()
            for stale in [s for s in frame_log if s <= step - 8]:
                del frame_log[stale]
            if step in anchor_set:
                snapshots[step] = snapshot_env(env)
                snapshots[step]["frames"] = {
                    key: value for key, value in frame_log.items()
                }
            obs, _reward, terminated, truncated, info = env.step(executed)
            executed_log[step] = norm_action(executed)
            step += 1
            if terminated or truncated:
                break
        for anchor in anchors:
            if anchor not in snapshots:
                print(f"seed={seed}: anchor {anchor} missing after replay, skip")
                continue
            anchor_snapshot = snapshots[anchor]
            anchor_prev = np.asarray(
                executed_log.get(anchor - 1, np.zeros(4)), dtype=np.float32
            )
            # ---- nominal 分支（每 anchor 一次，供 c_i^0 与 L_f 目标） ----
            branch_frames: dict[int, np.ndarray] = {
                key: value for key, value in anchor_snapshot["frames"].items()
            }
            restore_env(env, anchor_snapshot)
            nominal_lz: list[np.ndarray] = []
            nominal_windows: list[list[np.ndarray]] = []
            for i in range(args.recovery_steps):
                s = anchor + i
                obs = env._get_obs()
                action = np.clip(expert_policy.get_action(obs), -1.0, 1.0)
                branch_frames[s] = env.render()
                for stale in [k for k in branch_frames if k <= s - 8]:
                    del branch_frames[stale]
                nominal_windows.append(window_frames(branch_frames, s))
                obs, _r, _t, _tr, _info = env.step(action)
            nominal_tokens = encode_windows(nominal_windows)
            nominal_lz = [
                F.layer_norm(tokens.float().mean(dim=0), (768,)).numpy()
                for tokens in nominal_tokens
            ]
            # ---- 扰动分支（每 anchor n_branches 个，50/30/20 类型采样） ----
            for _b in range(n_branches):
                kind = str(rng.choice(kinds, p=perturb_weights))
                branch_frames = {
                    key: value for key, value in anchor_snapshot["frames"].items()
                }
                restore_env(env, anchor_snapshot)
                delta_desc = perturb_state(env, kind, rng)
                prev = np.asarray(anchor_prev, dtype=np.float32)
                branch_windows: list[list[np.ndarray]] = []
                branch_actions: list[np.ndarray] = []
                branch_prev: list[np.ndarray] = []
                branch_proprio: list[np.ndarray] = []
                epsilon = None
                if kind == "action":
                    epsilon = rng.normal(0.0, ACTION_EPS_STD, size=3)
                for i in range(args.recovery_steps):
                    s = anchor + i
                    obs = env._get_obs()
                    expert_action = np.clip(expert_policy.get_action(obs), -1.0, 1.0)
                    if kind == "action" and i == 0 and epsilon is not None:
                        executed = np.clip(expert_action + np.append(epsilon, 0.0), -1.0, 1.0)
                    else:
                        executed = expert_action
                    branch_frames[s] = env.render()
                    for stale in [k for k in branch_frames if k <= s - 8]:
                        del branch_frames[stale]
                    branch_windows.append(window_frames(branch_frames, s))
                    branch_actions.append(norm_action(executed))
                    branch_prev.append(prev)
                    branch_proprio.append(norm_state(obs[:4]))
                    env.step(executed)
                    prev = norm_action(executed)
                branch_tokens = encode_windows(branch_windows)
                is_heldout = seed % args.split_every == 0
                recovery_start.append(
                    {
                        "seed": seed,
                        "kind": kind,
                        "split": "heldout" if is_heldout else "train",
                        "reset_seed": int(seed),
                        # 扰动后、第 0 恢复步前的完整状态（torch 存储，
                        # eval --c2-recovery-eval 用 restore_env 恢复）。
                        "snapshot": snapshot_to_tensors(snapshot_env(env)),
                        "prev_action": torch.from_numpy(
                            np.asarray(branch_prev[0], dtype=np.float32)
                        ),
                        "anchor_step": anchor,
                    }
                )
                for i in range(args.recovery_steps):
                    z = branch_tokens[i].float().mean(dim=0)
                    lz_delta = F.layer_norm(z, (768,)).numpy()
                    transitions.append(
                        {
                            "tokens": branch_tokens[i],
                            "proprio": branch_proprio[i],
                            "prev_action": branch_prev[i],
                            "expert_action": branch_actions[i],
                            "step_index": i,
                            "branch_id": branch_counter,
                            "seed": seed,
                            "split": is_heldout,
                        }
                    )
                    lz_samples.append(
                        {"lz_delta": lz_delta, "lz_nominal": nominal_lz[i]}
                    )
                branch_counter += 1
                if args.debug:
                    print(
                        f"  seed={seed} anchor={anchor} kind={kind} "
                        f"{delta_desc} branch={branch_counter - 1} "
                        f"split={'heldout' if is_heldout else 'train'}"
                    )
        env.close()
        print(f"seed={seed}: expert success at step {success_step}, anchors={anchors}")

    if not transitions:
        raise SystemExit("no transitions collected — 检查专家策略与环境")
    print(
        f"collected: seeds_success={success_seeds} branches={branch_counter} "
        f"transitions={len(transitions)}"
    )
    vision_backbone.to(device="cpu")
    del vision_backbone
    import gc

    gc.collect()

    # ---- PCA（train split 差向量，Codex 修正 2）----
    pairs = [
        (
            np.asarray(sample["lz_delta"], dtype=np.float32)
            - np.asarray(sample["lz_nominal"], dtype=np.float32),
            np.asarray(sample["lz_delta"], dtype=np.float32),
        )
        for sample, transition in zip(lz_samples, transitions, strict=True)
        if not transition["split"]
    ]
    if not pairs:
        # 极端情况（如 dry-run 的 seed 0 恰为 held-out）：回退全部样本。
        print("WARNING: train split empty — PCA fallback to all samples")
        pairs = [
            (
                np.asarray(sample["lz_delta"], dtype=np.float32)
                - np.asarray(sample["lz_nominal"], dtype=np.float32),
                np.asarray(sample["lz_delta"], dtype=np.float32),
            )
            for sample in lz_samples
        ]
    delta = np.stack([pair[0] for pair in pairs])
    train_lz = np.stack([pair[1] for pair in pairs])
    mean_delta = delta.mean(axis=0, keepdims=True)
    centered = delta - mean_delta
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    top = min(CONTROL_DIM, s.shape[0])
    if top < CONTROL_DIM:
        print(f"WARNING: only {top} singular values from {len(delta)} train samples")
    explained = s[:top] ** 2 / (s**2).sum()
    weight = (vh[:top] / np.sqrt(s[:top, None] ** 2 + 1e-6)).astype(np.float32)
    if weight.shape[0] < CONTROL_DIM:
        pad = np.zeros((CONTROL_DIM - top, weight.shape[1]), dtype=np.float32)
        weight = np.concatenate((weight, pad), axis=0)
    mean_lz = train_lz.mean(axis=0).astype(np.float32)
    bias = (-weight @ mean_lz).astype(np.float32)

    def project(lz: np.ndarray) -> np.ndarray:
        return (weight @ lz + bias).astype(np.float32)

    # ---- 组装 payload ----
    vision_tokens_t = torch.stack(
        [t["tokens"].unsqueeze(0) for t in transitions]
    )  # [N, 1, 64, 768]
    payload = {
        "vision_tokens_t": vision_tokens_t,
        "proprio": torch.from_numpy(np.stack([t["proprio"] for t in transitions])),
        "prev_action": torch.from_numpy(np.stack([t["prev_action"] for t in transitions])),
        "expert_action": torch.from_numpy(np.stack([t["expert_action"] for t in transitions])),
        "step_index": torch.tensor([t["step_index"] for t in transitions], dtype=torch.long),
        "branch_id": torch.tensor([t["branch_id"] for t in transitions], dtype=torch.long),
        "seed": torch.tensor([t["seed"] for t in transitions], dtype=torch.long),
        "split": torch.tensor([t["split"] for t in transitions], dtype=torch.bool),
        "c_perturbed": torch.from_numpy(
            np.stack([project(sample["lz_delta"]) for sample in lz_samples])
        ),
        "c_nominal": torch.from_numpy(
            np.stack([project(sample["lz_nominal"]) for sample in lz_samples])
        ),
        "language_hidden": language_hidden,
        "language_mask": language_mask,
        "pca": {
            "weight": torch.from_numpy(weight),
            "bias": torch.from_numpy(bias),
            "singular_values": torch.from_numpy(s[:top].astype(np.float32)),
            "explained_ratio": torch.from_numpy(explained.astype(np.float32)),
            "mean_lz": torch.from_numpy(mean_lz),
        },
        "recovery_start": recovery_start,
        "metadata": {
            "task": args.task_text,
            "env": args.env,
            "policy": args.policy_name,
            "n_seeds": n_seeds,
            "n_anchors": n_anchors,
            "n_branches": n_branches,
            "recovery_steps": args.recovery_steps,
            "perturb_weights": dict(zip(kinds, perturb_weights, strict=True)),
            "split_every": args.split_every,
            "control_dim": CONTROL_DIM,
            "action_contract": "executed-clip-v5",
            "vision_pooling": "flat",
            "fps": 80,
            "control_stride": CONTROL_STRIDE,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    n_heldout = int(payload["split"].sum().item())
    print(
        f"saved={args.output.resolve()} N={len(transitions)} "
        f"train={len(transitions) - n_heldout} heldout={n_heldout} "
        f"branches={branch_counter} size={args.output.stat().st_size / 2**30:.2f}GiB"
    )
    print(
        f"pca: explained(top{top})={float(explained.sum()):.3f} "
        f"c_norm_mean={float(payload['c_perturbed'].float().norm(dim=1).mean()):.3f}"
    )


if __name__ == "__main__":
    main()
