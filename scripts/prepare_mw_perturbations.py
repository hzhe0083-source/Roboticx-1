#!/usr/bin/env python
"""C²-IRF v2 扰动恢复数据管道（设计文档 §六：微扰恢复数据，不新增恢复 loss）。

从 MetaWorld 示范轨迹（lerobot parquet，80FPS）选定 N 个 near-contact 决策点
（gripper 闭合 / 手→物体距离 < 阈值，可配），注入 2–8mm 精细扰动
（EEF 横向 / EEF 高度 / 物体位置 / peg-hole 相对偏移，--perturb-mix 概率可配），
随后用环境 step 回放数据中的后续专家动作（recovery rollout），记录扰动后观测
（渲染帧小样本 + 数据帧行号引用）、归一化 state（v5 空间）、专家恢复动作
（executed-clip-v5 空间），输出与 v5 payload 同构的 data/metaworld_perturbations.pt
（FeatureDataset 可直接加载，--single-task）。

回放机制复用 mw_expert_replay.py：同款 MT1 环境构造（corner2 相机、cam_pos、
_freeze_rand_vec=False）+ align_objects 数据对齐；对齐质量在决策点做 sanity
检查（手 ≤3cm、obj1 ≤2cm），不达标则跳过该决策点。

扰动机制（实测校准，2026-08-09）：
- EEF：MetaWorld 手臂由 mocap 目标经 7 关节 IK 跟踪，qpos[:3] 是旋转关节
  （right_j0）而非手位置——不能直接改 qpos。做法：mocap_pos += 命令向量 →
  12 步零动作 settle，实测 = 干净/扰动两分支（同快照）settle 后手位置之差
  （追赶/振荡在差值中抵消，实测 ≈ 0.97–1.00×命令），按 target/measured
  比例修正命令（≤3 次迭代，确定性收敛），记录实测幅度；
- 物体：obj1（obs[4:7] 匹配 site/body，跳过机械臂自身）经 free joint 平移，
  实测 = 精确；
- peg-hole 相对偏移：obj1 水平面平移（保持高度、仅横向错位）。

用法（CPU 仿真，不占 GPU；默认骨架模式不加载 V-JEPA）：
    # 冒烟（格式校验）：默认任务 peg-insert-side-v3 × 3 样本
    python scripts/prepare_mw_perturbations.py
    # 全量：peg-insert-side（peg 是自由体，obj1 语义与数据一致）100 样本
    python scripts/prepare_mw_perturbations.py --samples 100
    # 真实 V-JEPA flat-64 特征（GPU ~8GB；训练占用 4.2/16GB 时余量足够）
    python scripts/prepare_mw_perturbations.py --no-skeleton --device cuda

已知问题（2026-08-09 实测）：
- assembly-v3 不可用：本地 metaworld 的 SawyerNutAssemblyEnvV3 中 nut 与 peg
  是焊死一体（reset 时同位置），_get_pos_objects 返回 peg 上的固定 site
  "RoundNut-8"（obs[4:7] 恒定），与 lerobot 数据的"移动 nut"语义不符——数据
  回放对不上、obj1 扰动测不到位移。peg-insert-side 等 obj1 为自由体的任务正常；
- 手臂回放存在 mocap 追赶滞后（数据动作每步移动目标，本地手臂逐帧追赶）：
  决策点 sanity 按手 ≤--max-hand-err、obj1 ≤--max-obj-err 过滤，不达标跳过；
  EEF 扰动实测用干净/扰动两分支同源 settle 的位移差，追赶自动抵消。

依赖：metaworld（本地仿真）、v5 特征（data/metaworld_features_v5.pt，取
normalization + language 缓存）、lerobot parquet 数据集。
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 根模块（mw_expert_replay 等）导入兼容

# 回放机制复用（模块顶层只 import numpy/Path，metaworld 在 main 内惰性导入）
from mw_expert_replay import (  # noqa: E402
    align_objects,
    load_episode_rows,
    load_episodes,
    move_body,
)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

DATASET_ROOT = Path(
    "/home/ryan/Documents/robot/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
)
MW_CONFIG = Path(
    "/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json"
)
FEATURES_PATH = ROOT / "data" / "metaworld_features_v5.pt"
OUTPUT_PATH = ROOT / "data" / "metaworld_perturbations.pt"

# ---- 与 v5 一致的数据契约（prepare_metaworld.py 同款常量） ----
VISION_WINDOW = 4
VISION_STRIDE = 2
CONTROL_STRIDE = 6  # 80 FPS 决策间隔
SEQUENCE_LENGTH = 4  # 每样本决策点数（v5 同构 T=4）
ACTION_HORIZON = 8  # 每决策点动作步数
# 恢复回放跨度：最后一个决策 s+18 需要动作到 s+25（26 步）
RECOVERY_SPAN = (SEQUENCE_LENGTH - 1) * CONTROL_STRIDE + ACTION_HORIZON

# ---- 扰动配置 ----
PERTURB_KIND_ORDER = ("eef_lateral", "eef_height", "object", "peg_hole_relative")
DEFAULT_PERTURB_MIX = "0.3,0.2,0.3,0.2"  # 顺序 eef_lateral,eef_height,object,peg_hole_relative
MAG_MIN = 0.002  # 2mm
MAG_MAX = 0.008  # 8mm
MAG_TOL = 2e-4  # EEF 修正循环幅度容差（0.2mm）
VALIDATION_MAG_TOL = 5e-4  # 校验时幅度带宽容差（0.5mm）
MIN_MARGIN = 8  # 决策点最小步数（覆盖窗口回看 s-6 与 prev 行 s-1，避开 reset 瞬态）
SETTLE_STEPS = 12  # EEF 扰动后零动作 settle 步数（实测 step 10 已收敛）
SANITY_HAND_ERR = 0.03  # 决策点手对齐 sanity（米）
SANITY_OBJ_ERR = 0.02  # 决策点 obj1 对齐 sanity（米）


def robust_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """v5 归一化（与 prepare_pnpw_features.robust_normalize / eval_metaworld.py
    同款：2·(x−q01)/(q99−q01)−1，尾值 clip；v5 action q01/q99=±1 时退化为
    clip(x,−1,1)，即 executed-clip 恒等映射）。"""
    scale = np.where(np.abs(q99 - q01) < 1e-6, 1.0, q99 - q01)
    return np.clip(2.0 * (x - q01) / scale - 1.0, -1.0, 1.0).astype(np.float32)


def norm_state(raw: np.ndarray, sq01: np.ndarray, sq99: np.ndarray) -> np.ndarray:
    """obs[:4]（手 xyz + gripper）→ v5 state 空间（float32）。"""
    return robust_normalize(np.asarray(raw, dtype=float), sq01, sq99).astype(np.float32)


def norm_action_clip(raw: np.ndarray, aq01: np.ndarray, aq99: np.ndarray) -> np.ndarray:
    """executed-clip-v5：clip(raw,-1,1)（环境实际执行）→ v5 action 空间。"""
    return robust_normalize(
        np.clip(np.asarray(raw, dtype=float), -1.0, 1.0), aq01, aq99
    ).astype(np.float32)


def resolve_perturb_mix(
    mix_arg: str | None,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """解析/校验扰动混合 → (kinds, weights)（已归一化，顺序 PERTURB_KIND_ORDER）。

    4 个逗号分隔权重（eef_lateral,eef_height,object,peg_hole_relative），
    必须非负且和 > 0；未传时用 DEFAULT_PERTURB_MIX。
    """
    text = mix_arg if mix_arg is not None else DEFAULT_PERTURB_MIX
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != len(PERTURB_KIND_ORDER):
        raise ValueError(
            f"--perturb-mix 需要恰好 {len(PERTURB_KIND_ORDER)} 个逗号分隔权重"
            f"（顺序 {PERTURB_KIND_ORDER}），got {text!r}"
        )
    try:
        weights = [float(part) for part in parts]
    except ValueError:
        raise ValueError(
            f"--perturb-mix 权重必须为数字（顺序 {PERTURB_KIND_ORDER}），got {text!r}"
        ) from None
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"--perturb-mix 权重必须非负且和 > 0，got {text!r}")
    total = sum(weights)
    return PERTURB_KIND_ORDER, tuple(w / total for w in weights)


def sample_perturb_delta(
    kind: str, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """扰动类型 → 3D 命令向量（模长 = magnitude，单位米）。

    - eef_lateral / peg_hole_relative：水平面随机方向（z=0）；
    - eef_height：竖直 ±；
    - object：3D 随机单位方向。
    """
    if kind == "eef_lateral":
        theta = rng.uniform(0.0, 2.0 * np.pi)
        return magnitude * np.array([np.cos(theta), np.sin(theta), 0.0])
    if kind == "eef_height":
        sign = float(rng.choice((-1.0, 1.0)))
        return magnitude * np.array([0.0, 0.0, sign])
    if kind == "object":
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        return magnitude * vector / max(norm, 1e-12)
    if kind == "peg_hole_relative":
        theta = rng.uniform(0.0, 2.0 * np.pi)
        return magnitude * np.array([np.cos(theta), np.sin(theta), 0.0])
    raise ValueError(f"unknown perturbation kind: {kind}")


def contact_candidates(
    obs_state: np.ndarray,
    obs_env: np.ndarray,
    *,
    mode: str,
    gripper_threshold: float,
    distance_threshold: float,
    control_stride: int = CONTROL_STRIDE,
    sequence_length: int = SEQUENCE_LENGTH,
    action_horizon: int = ACTION_HORIZON,
    min_margin: int = MIN_MARGIN,
) -> np.ndarray:
    """从数据 obs 选定 near-contact 决策点 s（升序，int64 数组）。

    - 接触判据（数据行直接判定）：mode="gripper" → obs.state[3] < 阈值；
      "distance" → |hand − obj1| < 阈值（environment_state[0:3]/[4:7]）；
      "any" → 两者取或；
    - 决策点必须留出完整恢复跨度：s ≤ L−1−(T−1)·cs−(H−1)，且 s ≥ min_margin。
    """
    length = int(obs_state.shape[0])
    if mode not in ("any", "gripper", "distance"):
        raise ValueError(f"unknown contact mode: {mode!r}")
    last = length - 1 - (sequence_length - 1) * control_stride - (action_horizon - 1)
    s = np.arange(min_margin, last + 1, dtype=np.int64)
    if s.size == 0:
        return s
    gripper = obs_state[s, 3] < gripper_threshold
    dist = np.linalg.norm(obs_env[s, 0:3] - obs_env[s, 4:7], axis=1)
    near = dist < distance_threshold
    if mode == "gripper":
        keep = gripper
    elif mode == "distance":
        keep = near
    else:
        keep = gripper | near
    return s[keep]


def window_frame_rows(episode_start: int, decision: int) -> np.ndarray:
    """决策点 4 帧窗口的全局 parquet 行号（与 clip_frame_indices 同序：最老帧在前）。

    行号 = episode_start + max(0, decision − offset·VISION_STRIDE)，offset 升序
    （3,2,1,0）→ [d−6, d−4, d−2, d]。扰动后窗口的帧为仿真渲染（不在 parquet），
    行号作为"对应数据帧"引用，供 clean/perturbed 配对。
    """
    return np.asarray(
        [
            episode_start + max(0, decision - offset * VISION_STRIDE)
            for offset in range(VISION_WINDOW - 1, -1, -1)
        ],
        dtype=np.int64,
    )


def make_frames_small(windows: list[list[np.ndarray]], size: int) -> np.ndarray:
    """4 窗口 × 4 帧（480×480 uint8）→ [4, 4, size, size, 3] uint8。

    与 prepare_metaworld.decode_bytes 同款 PIL BICUBIC 缩采样。
    """
    from PIL import Image

    out = np.empty((SEQUENCE_LENGTH, VISION_WINDOW, size, size, 3), dtype=np.uint8)
    for t in range(SEQUENCE_LENGTH):
        for i in range(VISION_WINDOW):
            img = Image.fromarray(windows[t][i]).resize((size, size), Image.BICUBIC)
            out[t, i] = np.asarray(img)
    return out


def validate_payload(
    payload: dict,
    *,
    magnitude_range: tuple[float, float] = (MAG_MIN, MAG_MAX),
    magnitude_tol: float = VALIDATION_MAG_TOL,
) -> list[str]:
    """格式契约校验：v5 同构键 + 扰动标注键 + prev 契约 + 幅度范围 + metadata。

    返回错误列表（空 = 通过）。纯函数，tests/test_mw_perturbations.py 直接
    import；脚本保存后也会加载输出文件跑一遍（格式校验冒烟）。
    """
    errors: list[str] = []
    required = (
        "vision_tokens", "language_hidden", "language_mask", "proprio",
        "previous_action", "actions", "pair_id", "instruction_id", "episode_id",
        "normalization", "metadata", "frames", "frame_rows", "perturb_type",
        "perturb_magnitude", "aligned_v5_row", "source_episode", "decision_frame",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        return [f"missing keys: {missing}"]
    try:
        n = int(payload["actions"].shape[0])
    except Exception as exc:  # noqa: BLE001
        return [f"actions unusable: {exc}"]
    if n == 0:
        errors.append("empty dataset (n=0)")
    md = payload["metadata"]

    shapes = {
        "vision_tokens": (n, SEQUENCE_LENGTH, 64, 768),
        "language_hidden": (n, 13, 2048),
        "language_mask": (n, 13),
        "proprio": (n, SEQUENCE_LENGTH, 4),
        "previous_action": (n, SEQUENCE_LENGTH, 4),
        "actions": (n, SEQUENCE_LENGTH, ACTION_HORIZON, 4),
        "pair_id": (n,),
        "instruction_id": (n,),
        "episode_id": (n,),
        "perturb_magnitude": (n,),
        "aligned_v5_row": (n,),
        "source_episode": (n,),
        "decision_frame": (n,),
        "frame_rows": (n, SEQUENCE_LENGTH, VISION_WINDOW),
    }
    for key, shape in shapes.items():
        value = payload[key]
        if tuple(value.shape) != shape:
            errors.append(f"{key} shape {tuple(value.shape)} != {shape}")
    if payload["vision_tokens"].dtype != torch.float16:
        errors.append(f"vision_tokens dtype {payload['vision_tokens'].dtype} != float16")
    for key in ("proprio", "previous_action", "actions", "perturb_magnitude"):
        if payload[key].dtype != torch.float32:
            errors.append(f"{key} dtype {payload[key].dtype} != float32")
    for key in ("pair_id", "instruction_id", "episode_id", "aligned_v5_row",
                "source_episode", "decision_frame", "frame_rows"):
        if payload[key].dtype != torch.int64:
            errors.append(f"{key} dtype {payload[key].dtype} != int64")
    if payload["language_mask"].dtype != torch.bool:
        errors.append(f"language_mask dtype {payload['language_mask'].dtype} != bool")

    frames = payload["frames"]
    if frames.dtype != torch.uint8:
        errors.append(f"frames dtype {frames.dtype} != uint8")
    frame_size = int(md.get("vision", {}).get("frame_size", 96)) if isinstance(md, dict) else 96
    if frames.ndim == 1 and frames.shape == (0,):
        pass  # --no-store-frames：无帧小样本
    elif tuple(frames.shape) != (n, SEQUENCE_LENGTH, VISION_WINDOW, frame_size, frame_size, 3):
        errors.append(
            f"frames shape {tuple(frames.shape)} != "
            f"[n,{SEQUENCE_LENGTH},{VISION_WINDOW},{frame_size},{frame_size},3] 或 [0]"
        )

    # 扰动类型标注
    ptypes = payload["perturb_type"]
    if not isinstance(ptypes, list) or len(ptypes) != n:
        errors.append("perturb_type must be a list of str with length n")
    elif any(kind not in PERTURB_KIND_ORDER for kind in ptypes):
        errors.append(f"perturb_type contains unknown kinds: {set(ptypes) - set(PERTURB_KIND_ORDER)}")

    # 扰动幅度实测分布（2–8mm）
    mags = payload["perturb_magnitude"].numpy()
    lo, hi = magnitude_range
    if mags.size and (float(mags.min()) < lo - magnitude_tol or float(mags.max()) > hi + magnitude_tol):
        errors.append(
            f"perturb_magnitude out of range: "
            f"[{float(mags.min()) * 1000:.2f}, {float(mags.max()) * 1000:.2f}] mm "
            f"(expected [{lo * 1000:.1f}, {hi * 1000:.1f}])"
        )

    # v5 prev 契约：previous_action[t] == actions[t-1][5]（t>0）
    # （形状已被上面的校验覆盖；畸形张量在此跳过，避免二次崩溃）
    if (
        payload["previous_action"].ndim == 3
        and payload["actions"].ndim == 4
        and payload["previous_action"].shape[0] == payload["actions"].shape[0]
    ):
        prev_err = float(
            (payload["previous_action"][:, 1:] - payload["actions"][:, :-1, 5])
            .abs()
            .max()
            .item()
        )
        if prev_err > 1e-6:
            errors.append(
                f"previous_action contract broken (max |prev[t]-actions[t-1][5]| = {prev_err:.3g})"
            )

    # metadata 自描述
    for key in ("contract", "tasks", "fps", "control_stride", "action_horizon",
                "previous_action_contract", "action_contract", "sampling",
                "perturbation", "alignment", "vision"):
        if key not in md:
            errors.append(f"metadata.{key} missing")
    if md.get("fps") != 80:
        errors.append(f"metadata.fps {md.get('fps')} != 80")
    if md.get("control_stride") != CONTROL_STRIDE:
        errors.append(f"metadata.control_stride {md.get('control_stride')} != {CONTROL_STRIDE}")
    if md.get("action_horizon") != ACTION_HORIZON:
        errors.append(f"metadata.action_horizon {md.get('action_horizon')} != {ACTION_HORIZON}")
    if md.get("action_contract") != "executed-clip-v5":
        errors.append(f"metadata.action_contract {md.get('action_contract')!r} != executed-clip-v5")
    if md.get("previous_action_contract") != "v5_prevfix_20260807":
        errors.append(
            f"metadata.previous_action_contract {md.get('previous_action_contract')!r} "
            "!= v5_prevfix_20260807"
        )
    if not isinstance(md.get("sampling"), dict) or "mode" not in md["sampling"]:
        errors.append("metadata.sampling must be a self-describing dict with mode")
    if not isinstance(md.get("perturbation"), dict) or not md["perturbation"].get("types"):
        errors.append("metadata.perturbation.types missing")
    if not isinstance(md.get("alignment"), dict):
        errors.append("metadata.alignment missing")

    norm = payload["normalization"]
    for key in ("action_q01", "action_q99", "state_q01", "state_q99"):
        if key not in norm:
            errors.append(f"normalization.{key} missing")
    return errors


def scan_success_episodes(root: Path, episodes: list[dict]) -> set[int]:
    """扫描 raw parquet 的 next.success 列 → 至少成功过一次的 episode 起始行集合。

    与 prepare_metaworld.scan_episode_success 同款（列投影，不读图像）。
    """
    starts = sorted(int(ep["dataset_from_index"]) for ep in episodes)
    intervals = [
        (int(ep["dataset_from_index"]), int(ep["dataset_from_index"]) + int(ep["length"]))
        for ep in episodes
    ]
    ok: set[int] = set()
    for path in sorted(glob.glob(str(root / "data/chunk-000/*.parquet"))):
        table = pq.read_table(path, columns=["index", "next.success"])
        index_col = table.column("index").to_pylist()
        success_col = table.column("next.success").to_pylist()
        for row, flag in zip(index_col, success_col):
            if not flag:
                continue
            i = bisect.bisect_right(starts, row) - 1
            if i >= 0:
                ep_start, ep_end = intervals[i]
                if ep_start <= row < ep_end:
                    ok.add(ep_start)
    return ok


def make_env(env_name: str, seed: int = 42):
    """与 mw_expert_replay.py / eval_metaworld.py 完全相同的采集同款环境构造。"""
    import metaworld

    mt1 = metaworld.MT1(env_name, seed=42)
    env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]  # corner2 位置（lerobot 采集同款）
    env._freeze_rand_vec = False
    env.reset(seed=seed)
    return env


def _is_robot_body(env, bid: int) -> bool:
    """body（或祖先链）是否属于机械臂：right/left 前缀、claw、hand。

    近接触时刻手在抓取点，endEffector site 与 obj1 位置重合——site/body 匹配
    必须跳过机械臂自身，否则会匹配到手指/腕部 site 而测不到 obj1 位移
    （实测 0 位移，2026-08-09）。注意不能用"祖先含 hinge joint"判据：
    door/window/drawer 等任务的 obj1 本身带 hinge joint。
    """
    while bid > 0:
        name = env.model.body(bid).name or ""
        if name.startswith(("right", "left")) or "claw" in name or name == "hand":
            return True
        bid = int(env.model.body_parentid[bid])
    return False


def move_obj1(env, delta: np.ndarray) -> str:
    """平移 obj1（obs[4:7] 匹配 site 或 body）→ free joint 精确位移。

    匹配逻辑与 mw_expert_replay.align_objects 一致（site → body → 报错），
    但匹配必须跳过机械臂自身（_is_robot_body）。
    """
    cur = env._get_obs()[4:7].copy()
    for i in range(env.model.nsite):
        if _is_robot_body(env, int(env.model.site_bodyid[i])):
            continue
        if np.allclose(env.data.site_xpos[i], cur, atol=0.02):
            return move_body(env, int(env.model.site_bodyid[i]), delta)
    for bid in range(env.model.nbody):
        if _is_robot_body(env, bid):
            continue
        name = env.model.body(bid).name
        if not name:
            continue
        if np.allclose(env.data.body(bid).xpos, cur, atol=0.02):
            return move_body(env, bid, delta)
    raise RuntimeError(
        f"obj1 (obs[4:7]={np.round(cur, 3)}) not matched to any site/body"
    )


def apply_eef_perturb(
    env, snapshot: dict, delta_cmd: np.ndarray, target_mag: float
) -> float:
    """EEF 扰动：mocap 目标平移 + settle，实测 = 扰动分支与干净分支的手位移差。

    返回实测手位移（米）。关键（实测校准 2026-08-09）：数据动作每步都在移动
    mocap 目标，快照时手臂存在 cm 级"追赶滞后"——单独看扰动分支的 settle
    位移会把它混进实测（实测 23.7mm vs 命令 4.35mm）。正确做法是让干净分支
    同款 settle 12 步，取两分支手位置之差的模长（追赶在差值中抵消，实测 ≈
    0.99×命令）；再按 target/measured 比例修正命令（≤3 次迭代，确定性收敛）。
    """
    from prepare_mw_recovery import restore_env

    # 干净分支：同款 settle，记录追赶完成后的手位置
    restore_env(env, snapshot)
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(4))
    hand_clean = env._get_obs()[0:3].copy()

    measured = 0.0
    for _attempt in range(3):
        restore_env(env, snapshot)
        env.data.mocap_pos[0] += delta_cmd
        for _ in range(SETTLE_STEPS):
            env.step(np.zeros(4))
        hand_pert = env._get_obs()[0:3].copy()
        measured = float(np.linalg.norm(hand_pert - hand_clean))
        if abs(measured - target_mag) <= MAG_TOL:
            return measured
        if measured > 1e-6:
            delta_cmd = delta_cmd * (target_mag / measured)
    return measured  # 最后一次尝试的实测值（幅度带内校验由 validate_payload 把关）


def collect_perturbed_row(
    env,
    data_actions: np.ndarray,
    episode_start: int,
    s: int,
    *,
    kind: str,
    magnitude: float,
    rng: np.random.Generator,
    snapshot: dict,
    frames: dict[int, np.ndarray],
    sq01: np.ndarray,
    sq99: np.ndarray,
    aq01: np.ndarray,
    aq99: np.ndarray,
) -> dict:
    """在快照处施加扰动 → 回放数据中的后续专家动作（26 步）→ 记录一行样本。

    返回 dict：windows（4×4 帧 480 尺寸）、proprio/previous/actions（v5 空间）、
    frame_rows、kind、magnitude（实测，米）、decision、source_episode。
    """
    from prepare_mw_recovery import restore_env, window_frames

    if kind in ("eef_lateral", "eef_height"):
        delta = sample_perturb_delta(kind, magnitude, rng)
        measured = apply_eef_perturb(env, snapshot, delta, magnitude)
    else:
        restore_env(env, snapshot)
        delta = sample_perturb_delta(kind, magnitude, rng)
        before = env._get_obs()[4:7].copy()
        move_obj1(env, delta)
        measured = float(np.linalg.norm(env._get_obs()[4:7] - before))
        if measured < 1e-4:
            raise RuntimeError(f"obj1 move produced ~0 displacement ({measured:.2e} m)")

    # 恢复回放：帧缓存从快照帧（s-7..s）起步，逐数据动作 step
    rollout_frames = {j - s: frame for j, frame in frames.items()}
    windows: list[list[np.ndarray]] = []
    proprio: list[np.ndarray] = []
    for i in range(RECOVERY_SPAN):
        obs = env._get_obs()
        rollout_frames[i] = env.render()
        for stale in [key for key in rollout_frames if key <= i - 8]:
            del rollout_frames[stale]
        if i % CONTROL_STRIDE == 0 and i // CONTROL_STRIDE < SEQUENCE_LENGTH:
            proprio.append(norm_state(obs[:4], sq01, sq99))
            windows.append(window_frames(rollout_frames, i))
        env.step(data_actions[s + i])

    actions = np.stack(
        [
            norm_action_clip(data_actions[s + t * CONTROL_STRIDE + step], aq01, aq99)
            for t in range(SEQUENCE_LENGTH)
            for step in range(ACTION_HORIZON)
        ]
    ).reshape(SEQUENCE_LENGTH, ACTION_HORIZON, 4)
    previous = np.stack(
        [
            norm_action_clip(data_actions[s + t * CONTROL_STRIDE - 1], aq01, aq99)
            for t in range(SEQUENCE_LENGTH)
        ]
    )
    frame_rows = np.stack(
        [window_frame_rows(episode_start, s + t * CONTROL_STRIDE) for t in range(SEQUENCE_LENGTH)]
    )
    return {
        "windows": windows,
        "proprio": np.stack(proprio),
        "previous": previous,
        "actions": actions,
        "frame_rows": frame_rows,
        "kind": kind,
        "magnitude": measured,
        "decision": s,
        "source_episode": episode_start,
    }


def encode_windows_batch(samples: list[dict], args: argparse.Namespace, vision_backbone) -> torch.Tensor:
    """每样本 4 窗口 → [M, 4, 64, 768] fp16（flat-64，与 v5 同域，原始预训练权重）。"""
    from prepare_metaworld import preprocess_batch

    n = len(samples)
    out = torch.empty(n, SEQUENCE_LENGTH, 64, 768, dtype=torch.float16)
    clips: list[list[np.ndarray]] = []
    for sample in samples:
        clips.extend(list(window) for window in sample["windows"])
    for start in range(0, len(clips), args.batch_size):
        chunk = clips[start : start + args.batch_size]
        inputs = preprocess_batch(chunk, args.image_size).to(args.device)
        with torch.inference_mode():
            flat, _ = vision_backbone.forward_variants(inputs)
        flat = flat.to(device="cpu", dtype=torch.float16)
        rows = len(chunk) // SEQUENCE_LENGTH
        out[start // SEQUENCE_LENGTH : start // SEQUENCE_LENGTH + rows] = flat.reshape(
            rows, SEQUENCE_LENGTH, 64, 768
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH,
                        help="v5 特征（取 normalization + language 缓存，须含任务文本）")
    parser.add_argument("--task-index", type=int, default=28,
                        help="metaworld_config TASK_DESCRIPTIONS 顺序索引（默认 28="
                             "peg-insert-side-v3；0=assembly-v3 在本地 metaworld 版本"
                             "（assembly_peg_v3）中 nut 与 peg 焊死、obs[4:7] 是 peg 上的"
                             "固定 site，与 lerobot 数据语义不符，回放不可信，勿用）")
    parser.add_argument("--task-text", type=str, default=None,
                        help="直接给任务文本（覆盖 --task-index；须在 v5 features metadata.tasks 中）")
    parser.add_argument("--samples", type=int, default=3,
                        help="目标样本数（默认 3 = 冒烟规模；全量如 100）")
    parser.add_argument("--max-episodes", type=int, default=20,
                        help="最多尝试的（成功过滤后）episode 数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contact-mode", choices=("any", "gripper", "distance"), default="any",
                        help="near-contact 判据：gripper 闭合 / 距离 < 阈值 / 两者取或")
    parser.add_argument("--gripper-threshold", type=float, default=0.5,
                        help="gripper 闭合判据（数据 obs.state[3] < 阈值；实测闭合 ~0.30–0.44）")
    parser.add_argument("--distance-threshold", type=float, default=0.08,
                        help="手→obj1 近距判据（米）")
    parser.add_argument("--perturb-mix", type=str, default=None,
                        help=f"扰动混合权重，逗号分隔 4 个（顺序 {PERTURB_KIND_ORDER}；"
                             f"默认 {DEFAULT_PERTURB_MIX!r}）")
    parser.add_argument("--magnitude-min", type=float, default=MAG_MIN,
                        help="扰动幅度下界（米，默认 0.002）")
    parser.add_argument("--magnitude-max", type=float, default=MAG_MAX,
                        help="扰动幅度上界（米，默认 0.008）")
    parser.add_argument("--max-hand-err", type=float, default=SANITY_HAND_ERR,
                        help="决策点回放手对齐 sanity（米）")
    parser.add_argument("--max-obj-err", type=float, default=SANITY_OBJ_ERR,
                        help="决策点回放 obj1 对齐 sanity（米）")
    parser.add_argument("--no-skeleton", dest="skeleton", action="store_false",
                        help="计算真实 V-JEPA flat-64 特征（GPU ~8GB；默认骨架零占位）")
    parser.add_argument("--no-store-frames", dest="store_frames", action="store_false",
                        help="不存渲染帧小样本（默认存，扰动后观测的唯一视觉证据）")
    parser.add_argument("--frame-size", type=int, default=96,
                        help="帧小样本存储尺寸（PIL BICUBIC 缩采样）")
    parser.add_argument("--no-success-filter", dest="success_only", action="store_false",
                        help="不过滤（按 next.success 列）只保留成功 episode")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16, help="V-JEPA 编码批大小（非骨架模式）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(skeleton=True, store_frames=True, success_only=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.max_episodes < 1:
        raise SystemExit("--samples / --max-episodes must be positive")
    if not (0.0 < args.magnitude_min < args.magnitude_max):
        raise SystemExit("--magnitude-min/max must satisfy 0 < min < max")
    from prepare_mw_recovery import snapshot_env  # 快照/恢复复用 v6b 机制

    kinds, perturb_weights = resolve_perturb_mix(args.perturb_mix)
    rng = np.random.default_rng(args.seed)

    # ---- 任务解析：metaworld_config（env 名）↔ v5 features（任务文本） ----
    if not args.features.exists():
        raise SystemExit(f"missing {args.features}")
    if not MW_CONFIG.exists():
        raise SystemExit(f"missing {MW_CONFIG}")
    descriptions = json.load(open(MW_CONFIG))["TASK_DESCRIPTIONS"]
    env_names = list(descriptions.keys())
    if args.task_text is not None:
        env_name = next((name for name, text in descriptions.items() if text == args.task_text), None)
        if env_name is None:
            raise SystemExit(f"--task-text {args.task_text!r} not in metaworld config")
        task_text = args.task_text
    else:
        if not (0 <= args.task_index < len(env_names)):
            raise SystemExit(f"--task-index {args.task_index} out of range 0..{len(env_names)-1}")
        env_name = env_names[args.task_index]
        task_text = descriptions[env_name]

    v5 = torch.load(args.features, map_location="cpu", weights_only=True)
    tasks = v5["metadata"]["tasks"]
    if task_text not in tasks:
        raise SystemExit(f"task {task_text!r} not found in {args.features}")
    inst = int(tasks.index(task_text))
    v5_inst_ids = v5["instruction_id"]
    lang_hidden = v5["language_hidden"][v5_inst_ids == inst][0].to(torch.float16)
    lang_mask = v5["language_mask"][v5_inst_ids == inst][0]
    norm = v5["normalization"]
    sq01 = norm["state_q01"].numpy()
    sq99 = norm["state_q99"].numpy()
    aq01 = norm["action_q01"].numpy()
    aq99 = norm["action_q99"].numpy()
    print(
        f"task: {env_name} | {task_text} | mix={dict(zip(kinds, perturb_weights, strict=True))} "
        f"| samples={args.samples}"
    )

    # ---- episode 准备（成功过滤 + 确定性打乱） ----
    episodes = load_episodes()
    task_episodes = [ep for ep in episodes if task_text in str(ep.get("tasks"))]
    if args.success_only:
        ok = scan_success_episodes(args.dataset, episodes)
        before = len(task_episodes)
        task_episodes = [ep for ep in task_episodes if int(ep["dataset_from_index"]) in ok]
        print(f"success filter: {len(task_episodes)}/{before} episodes kept")
    if not task_episodes:
        raise SystemExit(f"no episodes for task {task_text!r}")
    order = rng.permutation(len(task_episodes))

    samples: list[dict] = []
    env = None
    for position in order:
        if len(samples) >= args.samples:
            break
        episode = task_episodes[int(position)]
        rows = load_episode_rows(episode)
        length = len(rows)
        obs_state = np.asarray([r["observation.state"] for r in rows], dtype=float)
        obs_env = np.asarray([r["observation.environment_state"] for r in rows], dtype=float)
        raw_actions = np.asarray([r["action"] for r in rows], dtype=float)
        data_actions = np.clip(raw_actions, -1.0, 1.0)  # 环境实际执行的动作

        candidates = contact_candidates(
            obs_state, obs_env,
            mode=args.contact_mode,
            gripper_threshold=args.gripper_threshold,
            distance_threshold=args.distance_threshold,
        )
        if candidates.size == 0:
            continue
        # 干净回放至最远候选点，决策点处做对齐 sanity + 快照（帧缓存 s-7..s）
        if env is not None:
            env.close()
        env = make_env(env_name, seed=42)
        align_objects(env, obs_env[0].copy(), env._get_obs())
        frame_log: dict[int, np.ndarray] = {}
        snapshots: dict[int, dict] = {}
        s_max = int(candidates.max())
        cand_set = set(int(c) for c in candidates)
        for step in range(s_max + 1):
            frame_log[step] = env.render()
            for stale in [key for key in frame_log if key <= step - 8]:
                del frame_log[stale]
            if step in cand_set:
                obs = env._get_obs()
                hand_err = float(np.linalg.norm(obs[:3] - obs_env[step][:3]))
                obj_err = float(np.linalg.norm(obs[4:7] - obs_env[step][4:7]))
                if hand_err <= args.max_hand_err and obj_err <= args.max_obj_err:
                    snapshots[step] = {"snapshot": snapshot_env(env), "frames": dict(frame_log)}
            if step < s_max:
                env.step(data_actions[step])
        valid = [s for s in candidates if int(s) in snapshots]
        if not valid:
            print(f"episode {episode['dataset_from_index']}: no valid decision (sanity), skip")
            continue
        quota = min(args.samples - len(samples), len(valid))
        chosen = rng.choice(np.asarray(valid), size=quota, replace=False)
        for s in chosen:
            s = int(s)
            kind = str(rng.choice(kinds, p=perturb_weights))
            magnitude = float(rng.uniform(args.magnitude_min, args.magnitude_max))
            try:
                record = collect_perturbed_row(
                    env, data_actions, int(episode["dataset_from_index"]), s,
                    kind=kind, magnitude=magnitude, rng=rng,
                    snapshot=snapshots[s]["snapshot"], frames=snapshots[s]["frames"],
                    sq01=sq01, sq99=sq99, aq01=aq01, aq99=aq99,
                )
            except (RuntimeError, ValueError) as exc:
                print(f"  sample at s={s} kind={kind} failed: {exc}, skip")
                continue
            if record["magnitude"] < args.magnitude_min * 0.5:
                # 实测幅度远低于目标（如 eef_height settle 后手未动 → 0.00mm）：
                # 幅度无效样本直接丢弃，否则整包校验失败（2026-08-09 E6 前置修复）
                print(
                    f"  sample at s={s} kind={kind} measured="
                    f"{record['magnitude'] * 1000:.2f}mm too small, skip"
                )
                continue
            samples.append(record)
            print(
                f"  sample {len(samples)}: s={s} kind={kind} "
                f"cmd={magnitude * 1000:.2f}mm measured={record['magnitude'] * 1000:.2f}mm"
            )
        print(
            f"episode {episode['dataset_from_index']} (len={length}): "
            f"candidates={len(candidates)} valid={len(valid)} chosen={len(chosen)}"
        )
    if env is not None:
        env.close()
    if not samples:
        raise SystemExit("no samples collected — 检查任务/接触阈值/回放 sanity")
    print(f"collected: {len(samples)} samples (target {args.samples})")

    # ---- vision：骨架零占位 或 真实 flat-64（与 v5 同域，原始预训练权重） ----
    vision_backbone = None
    if args.skeleton:
        vision = torch.zeros(len(samples), SEQUENCE_LENGTH, 64, 768, dtype=torch.float16)
        vision_mode = "skeleton-zero"
    else:
        from va_compound.backbones import VJEPA21Backbone

        vision_backbone = VJEPA21Backbone.from_pretrained(
            device=args.device, dtype="float16", max_tokens=64, local_files_only=True
        )
        vision_backbone.freeze_all()
        vision = encode_windows_batch(samples, args, vision_backbone)
        vision_mode = "vjepa-flat64"
        vision_backbone.to(device="cpu")
        del vision_backbone
        import gc

        gc.collect()

    # ---- 组装 payload（v5 同构 + 扰动标注键） ----
    n = len(samples)
    frames_arr = (
        np.stack([make_frames_small(sample["windows"], args.frame_size) for sample in samples])
        if args.store_frames
        else np.zeros(0, dtype=np.uint8)
    )
    magnitudes = np.asarray([s["magnitude"] for s in samples], dtype=np.float32)
    kinds_list = [s["kind"] for s in samples]
    payload = {
        "vision_tokens": vision,
        "language_hidden": lang_hidden.unsqueeze(0).expand(n, -1, -1).contiguous(),
        "language_mask": lang_mask.unsqueeze(0).expand(n, -1).contiguous(),
        "proprio": torch.from_numpy(np.stack([s["proprio"] for s in samples])),
        "previous_action": torch.from_numpy(np.stack([s["previous"] for s in samples])),
        "actions": torch.from_numpy(np.stack([s["actions"] for s in samples])),
        "pair_id": torch.zeros(n, dtype=torch.long),
        "instruction_id": torch.zeros(n, dtype=torch.long),
        "episode_id": torch.arange(n, dtype=torch.long),
        "frames": torch.from_numpy(frames_arr),
        "frame_rows": torch.from_numpy(np.stack([s["frame_rows"] for s in samples])),
        "perturb_type": kinds_list,
        "perturb_magnitude": torch.from_numpy(magnitudes),
        "aligned_v5_row": torch.full((n,), -1, dtype=torch.long),
        "source_episode": torch.tensor(
            [s["source_episode"] for s in samples], dtype=torch.long
        ),
        "decision_frame": torch.tensor([s["decision"] for s in samples], dtype=torch.long),
        "normalization": dict(norm),
        "metadata": {
            "contract": "perturbation_recovery_mt50",
            "tasks": [task_text],
            "fps": 80,
            "control_stride": CONTROL_STRIDE,
            "action_horizon": ACTION_HORIZON,
            "previous_action_contract": "v5_prevfix_20260807",
            "action_contract": "executed-clip-v5",
            "sampling": {
                "mode": "near-contact-perturbation",
                "seed": args.seed,
                "samples": n,
                "max_episodes": args.max_episodes,
                "success_only": args.success_only,
                "contact": {
                    "mode": args.contact_mode,
                    "gripper_threshold": args.gripper_threshold,
                    "distance_threshold": args.distance_threshold,
                },
                "control_stride": CONTROL_STRIDE,
                "action_horizon": ACTION_HORIZON,
                "sequence_length": SEQUENCE_LENGTH,
                "vision_window": VISION_WINDOW,
                "vision_stride": VISION_STRIDE,
                "replay_sanity": {
                    "max_hand_err": args.max_hand_err,
                    "max_obj_err": args.max_obj_err,
                },
            },
            "perturbation": {
                "types": list(PERTURB_KIND_ORDER),
                "mix": dict(zip(kinds, perturb_weights, strict=True)),
                "magnitude_range_mm": [args.magnitude_min * 1000, args.magnitude_max * 1000],
                "magnitude_measured_mm": {
                    "min": float(magnitudes.min()) * 1000,
                    "mean": float(magnitudes.mean()) * 1000,
                    "max": float(magnitudes.max()) * 1000,
                },
                "counts": {kind: kinds_list.count(kind) for kind in PERTURB_KIND_ORDER},
            },
            "alignment": {
                "contract": "v5-row-key",
                "features": args.features.name,
                "key_fields": ["instruction_id", "source_episode", "decision_frame"],
                "aligned_v5_row": (
                    "unresolved（v5.pt metadata 无 sampling 配置，行号不可反推；"
                    "行键字段已保留，供 clean/perturbed 配对）"
                ),
                "normalization": "inherited from v5 (state_q01/q99, action_q01/q99)",
            },
            "vision": {
                "mode": vision_mode,
                "frames_stored": args.store_frames,
                "frame_size": args.frame_size,
                "image_size": args.image_size,
                "parquet_root": str(args.dataset.resolve()),
            },
        },
    }
    errors = validate_payload(payload)
    if errors:
        raise SystemExit(f"payload validation failed:\n  " + "\n  ".join(errors))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    # 格式校验冒烟：重新加载（weights_only=True）再校验
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    reload_errors = validate_payload(reloaded)
    if reload_errors:
        raise SystemExit(f"reloaded payload validation failed:\n  " + "\n  ".join(reload_errors))
    mags_mm = 1000.0 * reloaded["perturb_magnitude"].numpy()
    counts = reloaded["metadata"]["perturbation"]["counts"]
    print(
        f"VALIDATION PASS: {args.output.resolve()} n={n} "
        f"magnitude(mm) min={mags_mm.min():.2f} mean={mags_mm.mean():.2f} "
        f"max={mags_mm.max():.2f} (target {args.magnitude_min * 1000:.1f}–"
        f"{args.magnitude_max * 1000:.1f}mm) counts={counts} "
        f"size={args.output.stat().st_size / 2**20:.1f}MiB"
    )


if __name__ == "__main__":
    main()
