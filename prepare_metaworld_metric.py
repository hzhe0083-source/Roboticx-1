#!/usr/bin/env python
"""MT-VJ 阶段 V 数据生成器：仿真器随机生成 metric 视觉预训练真值。

契约：artifacts/mt_vj_contract.md §3（make_metric_batch 接口）；
设计：artifacts/mt_vj_design.md（四角色度量场 / 阶段 V 预训练）。

每样本：全新 MT1 环境（随机 reset seed + 全局 np.random.seed 复现）、随机物体位置
（free joint 平移，同 scripts/prepare_mw_perturbations.py::move_obj1 手法）、
随机臂位（mocap 目标 + IK settle + 随机小动作）、随机视角（cam_pos ±3cm jitter）、
随机物体颜色（对象 geom rgba 亮度/色相扰动）。

关键点真值（世界坐标 → 图像坐标 pinhole 投影，0-1 归一化 y,x 序）：
    tool      = tcp_center（MetaWorld 官方 EEF）
    object    = env._get_pos_objects()[0:3]（主操作实体）
    target    = env._target_pos（任务成功目标）
    interface = 第二实体（若存在），否则 object（progress anchor）
覆盖本项目全部 49 个 MetaWorld 任务。reach 操作族直接控制 TCP，因此该族的
object/interface 均取 tool；这是操作族语义，不是逐任务关键点映射。

Task35 ``peg-insert-side-v3`` 使用任务对齐覆盖：固定四槽仍保持
``[tool, object, target, interface]``，但实体为
``[tcp, pegGrasp, hole, pegHead]``。因此第二关系对是
``pegHead - hole``，且目标锚点是画面中真实可见的 ``hole`` site，而不是位于
方块内部、视觉不可观测的 ``_target_pos``。

投影（2026-08-09 实测验证，误差 <2px）：
    p_cam = R^T (p − cam_pos)，R = mju_quat2Mat(cam_quat) 列 = 相机轴
    x = W/2 + f·(x̂·v)/(−ẑ·v)，y = H/2 − f·(ŷ·v)/(−ẑ·v)，f = (H/2)/tan(fovy/2)
验证方法：mujoco.Renderer offscreen（与 env.render() 逐像素一致，MAE≈0）+ RGB
site marker / seg geom 质心对照（右/左 EEF 1.0/2.0px、hole 0.6px、assembly peg
0.6px、pegTop 0.6px、hand-insert goal 2.9px）。注意：mujoco 3.3 seg 渲染对
「模型 pos 被改写过的 world 系 site」（如 peg-insert 的 goal）给出陈旧/错误位置，
故可见度与验证一律用深度（renderer depth）判遮挡、RGB marker 验 site。

relation 语义（[n,6]，与 metric head 输出同空间）：
    rel[:,0:2] = p_tool − p_object；rel[:,2:4] = p_progress − p_target
    rel[:,4] = axis_cos；rel[:,5] = depth_m(z_progress−z_target)
relation_aux（额外键，[n,4]）：[axis_alignment(cos∈[-1,1]),
    depth_m(z_progress−z_target), |tool−object|_m, |progress−target|_m]。
contact：|eef − obj| < 3cm → 1.0。

用法（CPU 仿真，无 GPU）：
    python prepare_metaworld_metric.py            # 冒烟：2 样本 + 投影验证 + /tmp/metric_sample.png
    python prepare_metaworld_metric.py --task assembly-v3 --n 8 --out-png /tmp/m.png
    python prepare_metaworld_metric.py --task any --n 4        # 混合任务
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 仓库根绝对 import 兼容（契约「开发约定」）

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from scripts.build_longtraj_features import ENV_TO_TASK  # noqa: E402

ROLE_NAMES = ("tool", "object", "target", "interface")
SUPPORTED_TASKS = tuple(ENV_TO_TASK)
# Reach 的成功量是 TCP→target；场景中的 dummy object 不参与任务。
DIRECT_TOOL_TARGET_TASKS = frozenset(("reach-v3", "reach-wall-v3"))
# Task35 的视觉成功几何不能用 generic achieved/target API：
# _get_pos_objects() 只给 pegGrasp，而 _target_pos 位于孔块内部、不可见。
TASK_ALIGNED_ROLE_SOURCES = {
    "peg-insert-side-v3": ("tcp_center", "pegGrasp", "hole", "pegHead"),
}
CAMERA_NAME = "corner2"
CAM_POS_DEFAULT = np.array([0.75, 0.075, 0.7])  # lerobot 采集同款 corner2
RENDER_SIZE = 480  # metaworld env 渲染分辨率（square）
IMAGE_SIZE = 384   # 输出帧分辨率
VISION_STRIDE = 2  # 历史帧步距（eval 契约 [d-6, d-4, d-2, d]）
SETTLE_STEPS = 10  # 随机臂位 IK settle 步数
OCCLUSION_TOL_M = 0.05  # 深度判遮挡容差（关键点表面深度 vs 像素深度）
ENTITY_RAY_TOL_M = 0.03  # 内部实体锚点：首个同实体命中可略晚于语义点
ENTITY_NEIGHBOR_RADIUS_PX = 24.0  # 只接纳语义点附近的同 body 可见表面
CONTACT_DIST_M = 0.03   # 接触判据：|eef − obj|
CAM_JITTER_M = 0.03     # 视角随机：cam_pos ±3cm
OBJ_JITTER_M = 0.03     # 物体位置随机：水平 ±3cm
_MT1_CACHE: dict[str, object] = {}
SAMPLE_RNG_CONTRACT = "parent_seed_per_sample_v1"


# 仅供人工投影校准使用；训练标签不读取此表。
PROJECTION_REFERENCE_TABLE: dict[str, dict[str, tuple]] = {
    "peg-insert-side-v3": {
        "object": ("body", "peg"),          # 被插的 peg（自由体）
        "target": ("site", "hole"),         # 插入目标：孔（可见；goal site 在方块内部不可见）
        "interface": ("site", "pegHead"),   # 精插接口：真正进入 hole 的杆尖
    },
    "assembly-v3": {
        "object": ("body", "RoundNut"),     # 螺母（obs[4:7] 即 RoundNut-8 site，同体）
        "target": ("env", "_target_pos"),   # 螺母落点 = peg 顶（可见；pegTop site 因 metaworld
                                            #   把世界坐标写进 body 相对 site pos 而错位，不能用）
        "interface": ("body", "peg"),       # 插杆（红柱，接触界面）
    },
    "hand-insert-v3": {
        "object": ("body", "obj"),          # 带孔方块（objGeom）
        "target": ("site", "goal"),         # 孔中心（桌面以下 2cm，孔口可见）
        "interface": ("point", "hole_mouth"),  # 孔口 = goal + [0,0,0.02]
    },
}


def _import_mujoco_metaworld():
    """惰性导入（必须在 MUJOCO_GL/EGL_PLATFORM 设定之后）。"""
    import mujoco  # noqa: F401
    import metaworld  # noqa: F401
    return mujoco, metaworld


def make_env(task: str, seed: int = 42):
    """与 prepare_mw_perturbations.make_env 同款环境构造（corner2 + cam_pos 对齐数据）。"""
    import metaworld
    mujoco = _import_mujoco_metaworld()[0]
    if task not in _MT1_CACHE:
        _MT1_CACHE[task] = metaworld.MT1(task, seed=42)
    mt1 = _MT1_CACHE[task]
    env = mt1.train_classes[task](render_mode="rgb_array", camera_name=CAMERA_NAME)
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = CAM_POS_DEFAULT.copy()
    env._freeze_rand_vec = False
    env.reset(seed=seed)  # seed 被 metaworld 3.0 忽略（reset_model 用全局 np.random）
    mujoco.mj_forward(env.model, env.data)  # 刷新 site_xpos（reset 后 data 是陈旧的）
    return env


# --------------------------------------------------------------------------
# 相机 pinhole 投影（已验证 <2px，见模块 docstring）
# --------------------------------------------------------------------------
def camera_params(env) -> tuple[np.ndarray, np.ndarray, float]:
    """→ (cam_pos[3], R[3,3]（列 = 相机 x/y/z 轴世界坐标）, f[px])。"""
    mujoco = _import_mujoco_metaworld()[0]
    m = env.model
    cam_pos = m.cam_pos[2].copy()
    q = m.cam_quat[2]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    R = mat.reshape(3, 3)
    f = (RENDER_SIZE / 2) / np.tan(np.radians(float(m.cam_fovy[2])) / 2)
    return cam_pos, R, f


def project_points(env, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界坐标 [N,3] → 图像像素 [N,2]（x,y 序）+ 深度 [N]（视图方向距离，>0 可见）。

    x = W/2 + f·(x̂·v)/(−ẑ·v)；y = H/2 − f·(ŷ·v)/(−ẑ·v)。返回前先 mj_forward
    保证 body/site xpos 新鲜。
    """
    mujoco = _import_mujoco_metaworld()[0]
    mujoco.mj_forward(env.model, env.data)
    cam_pos, R, f = camera_params(env)
    xhat, yhat, zhat = R[:, 0], R[:, 1], R[:, 2]
    v = np.asarray(points, dtype=float) - cam_pos[None, :]
    depth = -(v @ zhat)
    px = RENDER_SIZE / 2 + f * (v @ xhat) / depth
    py = RENDER_SIZE / 2 - f * (v @ yhat) / depth
    return np.stack([px, py], axis=1), depth


def _resolve_projection_reference(env, spec: tuple) -> np.ndarray | None:
    """人工投影校准表条目 → 世界坐标；不参与训练标签生成。"""
    kind, name = spec
    try:
        if kind == "body":
            return env.data.body(name).xpos.copy()
        if kind == "site":
            return env.data.site(name).xpos.copy()
        if kind == "env":
            return np.asarray(getattr(env, name), dtype=float).copy()
        if kind == "point":
            if name == "hole_mouth":
                goal = env.data.site("goal").xpos
                return goal + np.array([0.0, 0.0, 0.02])  # goal 在桌面下 2cm → 孔口
    except (KeyError, ValueError, AttributeError):
        return None
    raise ValueError(f"unknown keypoint spec {spec!r}")


def _site_world_position(env, task: str, name: str) -> np.ndarray:
    """Return a fresh MuJoCo site position with a strict 3-D contract."""
    mujoco = _import_mujoco_metaworld()[0]
    mujoco.mj_forward(env.model, env.data)
    try:
        point = np.asarray(env.data.site(name).xpos, dtype=float).reshape(-1)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{task}: required metric site {name!r} is missing") from exc
    if point.shape != (3,):
        raise ValueError(f"{task}: site {name!r} must have shape (3,), got {point.shape}")
    return point.copy()


def keypoint_world_positions(env, task: str) -> np.ndarray | None:
    """统一 MetaWorld 四角色世界坐标 ``[tool, object, target, interface]``。

    ``_get_pos_objects`` 是 MetaWorld observation 的官方 achieved-entity
    接口。双实体任务（hammer/stick）用第二实体表示任务进度；其余任务的
    interface/progress 与 object 相同。直接控制 TCP 的 reach 操作族不使用场景
    dummy object。

    ``peg-insert-side-v3`` 是经过实证的任务对齐例外：generic API 会生成
    ``[tcp, pegGrasp, _target_pos, pegGrasp]``，其中两个角色坍缩且内部 target
    在视觉上恒不可见。这里改为 ``[tcp, pegGrasp, hole, pegHead]``，保持固定
    slot 语义（target=hole，interface=pegHead），让第二关系直接表示插入误差。
    未知任务返回 ``None``，已声明支持的任务接口异常则立即报错。
    """
    if task not in SUPPORTED_TASKS:
        return None
    tool = np.asarray(env.tcp_center, dtype=float).reshape(-1)
    if tool.shape != (3,):
        raise ValueError(f"{task}: tcp_center must have shape (3,), got {tool.shape}")

    if task in TASK_ALIGNED_ROLE_SOURCES:
        world = np.stack(
            (
                tool,
                _site_world_position(env, task, "pegGrasp"),
                _site_world_position(env, task, "hole"),
                _site_world_position(env, task, "pegHead"),
            )
        )
    else:
        objects = np.asarray(env._get_pos_objects(), dtype=float).reshape(-1)
        target = np.asarray(getattr(env, "_target_pos", None), dtype=float).reshape(-1)
        if objects.size < 3 or objects.size % 3:
            raise ValueError(
                f"{task}: _get_pos_objects must contain one or more 3D entities, "
                f"got shape {objects.shape}"
            )
        if target.shape != (3,):
            raise ValueError(f"{task}: _target_pos must have shape (3,), got {target.shape}")
        entities = objects.reshape(-1, 3)
        if task in DIRECT_TOOL_TARGET_TASKS:
            obj = progress = tool
        else:
            obj = entities[0]
            progress = entities[1] if len(entities) > 1 else obj
        world = np.stack((tool, obj, target, progress))
    if not np.isfinite(world).all():
        raise ValueError(f"{task}: non-finite metric role coordinates")
    return world


def _relation_labels(
    keypoints_yx: np.ndarray,
    world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build relation/diagnostic labels from the generic four-role contract."""
    kp = np.asarray(keypoints_yx, dtype=float)
    xyz = np.asarray(world, dtype=float)
    if kp.shape != (4, 2) or xyz.shape != (4, 3):
        raise ValueError(f"expected keypoints (4,2) and world (4,3), got {kp.shape}, {xyz.shape}")
    tool, obj, target, progress = xyz
    tool_to_object = tool - obj
    progress_to_target = progress - target
    rel = np.zeros(6, dtype=np.float32)
    rel[0:2] = kp[0] - kp[1]
    rel[2:4] = kp[3] - kp[2]
    lhs = tool_to_object[:2]
    rhs = progress_to_target[:2]
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    rel[4] = float(np.dot(lhs, rhs) / denom) if denom > 1e-12 else 0.0
    rel[5] = float(progress[2] - target[2])
    aux = np.array(
        [
            rel[4],
            rel[5],
            np.linalg.norm(tool_to_object),
            np.linalg.norm(progress_to_target),
        ],
        dtype=np.float32,
    )
    return rel, aux


# --------------------------------------------------------------------------
# 随机化
# --------------------------------------------------------------------------
def _entity_anchor_body(
    env, point: np.ndarray, max_distance_m: float = 0.08
) -> int | None:
    """Find the nearest non-robot entity body for one semantic 3-D point."""
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"entity point must be finite (3,), got {point.shape}")
    candidates: list[tuple[float, int]] = []
    for site_id in range(env.model.nsite):
        body_id = int(env.model.site_bodyid[site_id])
        if body_id > 0 and not _is_robot_body(env, body_id):
            candidates.append(
                (float(np.linalg.norm(env.data.site_xpos[site_id] - point)), body_id)
            )
    for geom_id in range(env.model.ngeom):
        body_id = int(env.model.geom_bodyid[geom_id])
        if body_id > 0 and not _is_robot_body(env, body_id):
            candidates.append(
                (float(np.linalg.norm(env.data.geom_xpos[geom_id] - point)), body_id)
            )
    for body_id in range(1, env.model.nbody):
        if not _is_robot_body(env, body_id):
            candidates.append(
                (float(np.linalg.norm(env.data.body(body_id).xpos - point)), body_id)
            )
    if not candidates:
        return None
    distance, body_id = min(candidates)
    return body_id if distance <= max_distance_m else None


def _object_anchor_body(env, max_distance_m: float = 0.08) -> int | None:
    """Find the primary entity body through the generic MetaWorld object API."""
    obj = np.asarray(env._get_pos_objects(), dtype=float).reshape(-1)[:3]
    return _entity_anchor_body(env, obj, max_distance_m=max_distance_m)


def _move_object_random(env, rng: np.random.Generator) -> None:
    """Horizontally jitter a primary entity only when it has a free joint."""
    mujoco = _import_mujoco_metaworld()[0]
    bid = _object_anchor_body(env)
    if bid is None:
        return
    delta = np.zeros(3)
    delta[:2] = rng.uniform(-OBJ_JITTER_M, OBJ_JITTER_M, 2)
    # free joint 平移（沿父链找，同 mw_expert_replay.move_body）
    b = bid
    while True:
        moved = False
        for jnt in range(env.model.njnt):
            if int(env.model.jnt_bodyid[jnt]) == b:
                if env.model.jnt_type[jnt] == 0:  # free joint
                    adr = int(env.model.jnt_qposadr[jnt])
                    env.data.qpos[adr : adr + 3] += delta
                    moved = True
                break
        if moved:
            break
        parent = int(env.model.body_parentid[b])
        if parent <= 0 or parent == b:
            return  # 无 free joint：不改（物体可能焊死，如 assembly nut-peg）
        b = parent
    mujoco.mj_forward(env.model, env.data)


def _is_robot_body(env, bid: int) -> bool:
    """body（或祖先链）是否属于机械臂（同 prepare_mw_perturbations._is_robot_body）。"""
    while bid > 0:
        name = env.model.body(bid).name or ""
        if name.startswith(("right", "left")) or "claw" in name or name == "hand":
            return True
        bid = int(env.model.body_parentid[bid])
    return False


def _randomize_colors(env, rng: np.random.Generator) -> None:
    """Randomize primary-entity geoms found through the generic object API."""
    m = env.model
    body_id = _object_anchor_body(env)
    gids = (
        [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body_id]
        if body_id is not None
        else []
    )
    if not gids:
        return
    gray = np.ones(3) / np.sqrt(3)
    for g in gids:
        rgb = m.geom_rgba[g][:3].copy()
        if rng.random() < 0.5:
            # 绕灰轴随机旋转（保持亮度感、改变色相）
            theta = float(rng.uniform(0, 2 * np.pi))
            K = np.array(
                [
                    [0, -gray[2], gray[1]],
                    [gray[2], 0, -gray[0]],
                    [-gray[1], gray[0], 0],
                ]
            )
            rot = (
                np.eye(3) * np.cos(theta)
                + np.sin(theta) * K
                + (1 - np.cos(theta)) * np.outer(gray, gray)
            )
            rgb = rot @ rgb
        else:
            rgb = rgb * rng.uniform(0.5, 1.4, 3)  # 每通道亮度/饱和度缩放
        m.geom_rgba[g][:3] = np.clip(rgb, 0.05, 1.0)


def _jitter_camera(env, rng: np.random.Generator) -> None:
    """视角随机：corner2 cam_pos 各轴 ±CAM_JITTER_M（投影用同一 cam_pos，自洽）。"""
    cam = env.model.cam_pos[2]
    cam[:] = CAM_POS_DEFAULT + rng.uniform(-CAM_JITTER_M, CAM_JITTER_M, 3)
    if cam[2] < 0.55:  # 相机别掉到桌下
        cam[2] = 0.55


def _randomize_nonrobot_articulation(env, rng: np.random.Generator) -> int:
    """Sample every finite non-robot hinge/slide joint over its valid range.

    This single mechanism covers buttons, dials, doors, drawers, faucets,
    handles, levers, nails, plates, sticks and windows. Free joints remain under
    ``_move_object_random`` so rotations/translations are not made inconsistent.
    Returns the number of randomized scalar joints for contract tests.
    """
    mujoco = _import_mujoco_metaworld()[0]
    randomized = 0
    for joint_id in range(env.model.njnt):
        body_id = int(env.model.jnt_bodyid[joint_id])
        joint_type = int(env.model.jnt_type[joint_id])
        if (
            _is_robot_body(env, body_id)
            or joint_type not in (2, 3)  # slide / hinge
            or not bool(env.model.jnt_limited[joint_id])
        ):
            continue
        low, high = np.asarray(env.model.jnt_range[joint_id], dtype=float)
        if not np.isfinite((low, high)).all() or high - low <= 1e-8:
            continue
        qpos_addr = int(env.model.jnt_qposadr[joint_id])
        env.data.qpos[qpos_addr] = rng.uniform(low, high)
        dof_addr = int(env.model.jnt_dofadr[joint_id])
        env.data.qvel[dof_addr] = 0.0
        randomized += 1
    if randomized:
        mujoco.mj_forward(env.model, env.data)
    return randomized


# --------------------------------------------------------------------------
# 单样本采集
# --------------------------------------------------------------------------
def _chronological_capture_offsets(w: int) -> tuple[int, ...]:
    """Return oldest-to-newest frame offsets used by train and deployment."""
    if w < 1:
        raise ValueError("vision window must contain at least one frame")
    return tuple(VISION_STRIDE * k for k in range(w))


def _resize_chronological_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Resize without changing the oldest-to-newest temporal order."""
    if not frames:
        raise ValueError("cannot resize an empty vision window")
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC
                )
            )
            for frame in frames
        ]
    )


def _role_visibility(
    pixels_xy: np.ndarray,
    point_depths: np.ndarray,
    surface_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return honest visibility, surface visibility and in-frame diagnostics.

    All roles use the same depth-surface rule. In particular, an in-frame
    simulator target is not called visible when it is only a virtual coordinate;
    such an unobservable point must not contribute localization gradients.
    """
    pixels = np.asarray(pixels_xy, dtype=float)
    depths = np.asarray(point_depths, dtype=float)
    if pixels.shape != (4, 2) or depths.shape != (4,):
        raise ValueError(f"expected pixels (4,2) and depths (4,), got {pixels.shape}, {depths.shape}")
    in_frame = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < RENDER_SIZE)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < RENDER_SIZE)
        & (depths > 0)
    )
    surface_visible = np.zeros(4, dtype=np.float32)
    for role in range(4):
        if not in_frame[role]:
            continue
        px = int(round(float(pixels[role, 0])))
        py = int(round(float(pixels[role, 1])))
        px = min(px, RENDER_SIZE - 1)
        py = min(py, RENDER_SIZE - 1)
        if abs(float(surface_depth[py, px]) - float(depths[role])) < OCCLUSION_TOL_M:
            surface_visible[role] = 1.0
    visibility = surface_visible.copy()
    return visibility, surface_visible, in_frame.astype(np.float32)


def _first_ray_hit_body(env, point: np.ndarray) -> int | None:
    """Return the first body hit on the finite camera-to-point ray segment."""
    mujoco = _import_mujoco_metaworld()[0]
    camera, _, _ = camera_params(env)
    delta = np.asarray(point, dtype=np.float64) - camera
    point_distance = float(np.linalg.norm(delta))
    if not np.isfinite(point_distance) or point_distance <= 1e-9:
        return None
    geom_id = np.full(1, -1, dtype=np.int32)
    hit_distance = float(
        mujoco.mj_ray(
            env.model,
            env.data,
            camera.astype(np.float64),
            (delta / point_distance).astype(np.float64),
            None,
            1,
            -1,
            geom_id,
        )
    )
    if (
        hit_distance < 0.0
        or hit_distance > point_distance + ENTITY_RAY_TOL_M
        or int(geom_id[0]) < 0
    ):
        return None
    return int(env.model.geom_bodyid[int(geom_id[0])])


def _nearby_entity_surface_visible(
    env,
    anchor_point: np.ndarray,
    entity_body: int,
    *,
    max_radius_px: float = ENTITY_NEIGHBOR_RADIUS_PX,
) -> bool:
    """Probe visible same-body geom centers near an internal semantic anchor.

    The label coordinate stays at ``anchor_point``.  Geometry centers are only
    ray probes, and only when their image projection is within a bounded pixel
    neighborhood.  A different entity/robot hit remains an occlusion.
    """
    if max_radius_px < 0 or not np.isfinite(max_radius_px):
        raise ValueError("entity neighborhood radius must be finite and non-negative")
    anchor_px, anchor_depth = project_points(
        env, np.asarray(anchor_point, dtype=float).reshape(1, 3)
    )
    if anchor_depth[0] <= 0 or not np.isfinite(anchor_px).all():
        return False
    candidates: list[tuple[float, np.ndarray]] = []
    for geom_id in range(env.model.ngeom):
        if int(env.model.geom_bodyid[geom_id]) != int(entity_body):
            continue
        center = np.asarray(env.data.geom_xpos[geom_id], dtype=float).copy()
        center_px, center_depth = project_points(env, center.reshape(1, 3))
        if center_depth[0] <= 0 or not np.isfinite(center_px).all():
            continue
        pixel_distance = float(np.linalg.norm(center_px[0] - anchor_px[0]))
        if pixel_distance <= max_radius_px:
            candidates.append((pixel_distance, center))
    for _, center in sorted(candidates, key=lambda item: item[0]):
        if _first_ray_hit_body(env, center) == entity_body:
            return True
    return False


def _entity_aware_visibility(
    env,
    world: np.ndarray,
    surface_visible: np.ndarray,
    in_frame: np.ndarray,
) -> np.ndarray:
    """Augment only object/interface when the first ray hit is their own entity.

    This recovers generic internal anchors (drawers, pegs, handles) without
    treating a robot or another scene entity as evidence for the semantic point.
    Tool and target retain the strict depth-surface rule.  For task35 this means
    ``hole`` (target) is supervised only when the depth render honestly exposes
    it, while ``pegHead`` (interface) may use same-peg surface evidence.
    """
    points = np.asarray(world, dtype=float)
    strict = np.asarray(surface_visible, dtype=np.float32)
    framed = np.asarray(in_frame, dtype=np.float32)
    if points.shape != (4, 3) or strict.shape != (4,) or framed.shape != (4,):
        raise ValueError("entity-aware visibility expects world (4,3) and masks (4,)")
    entity_visible = strict.copy()
    for role in (1, 3):  # object, interface/progress
        if entity_visible[role] >= 0.5 or framed[role] < 0.5:
            continue
        entity_body = _entity_anchor_body(env, points[role])
        if entity_body is None or _is_robot_body(env, entity_body):
            continue
        hit_body = _first_ray_hit_body(env, points[role])
        same_entity_visible = hit_body == entity_body or (
            hit_body is None
            and _nearby_entity_surface_visible(env, points[role], entity_body)
        )
        if same_entity_visible:
            entity_visible[role] = 1.0
    return entity_visible


def _sample_one(
    env,
    task: str,
    rng: np.random.Generator,
    w: int,
    *,
    include_raw_frames: bool = False,
) -> dict:
    mujoco = _import_mujoco_metaworld()[0]
    # 随机化（颜色/相机/物体位置；reset 随机化已在 make_metric_batch 里做）
    _randomize_colors(env, rng)
    _jitter_camera(env, rng)
    _move_object_random(env, rng)
    _randomize_nonrobot_articulation(env, rng)

    # 随机臂位：mocap 目标 + IK settle + 窗口内随机小动作（帧间有运动）
    env.data.mocap_pos[0] = [
        rng.uniform(-0.2, 0.3),
        rng.uniform(0.45, 0.9),
        rng.uniform(0.08, 0.35),
    ]
    env.data.mocap_quat[0] = np.array([1.0, 0.0, 1.0, 0.0])

    # Capture in the same chronological contract as action train/eval:
    # [t-6, t-4, t-2, t].  The loop advances forward in simulator time, so the
    # ascending offsets are already oldest-to-newest and must not be reversed.
    offsets = _chronological_capture_offsets(w)  # [0,2,4,6]
    # Codex P0-2（2026-08-10）：total = SETTLE + offsets[-1] + 1——保证
    # i=SETTLE+offsets[-1] 的"最新帧"也被渲染（4 帧齐全）；该帧渲染后不再
    # step（i < total-1 才 step），标签（world 坐标）与最新帧严格同状态。
    total = SETTLE_STEPS + offsets[-1] + 1

    def _rand_action() -> np.ndarray:
        a = np.zeros(4, dtype=np.float32)
        a[:3] = rng.uniform(-0.02, 0.02, 3)
        a[3] = 1.0  # 夹爪张开
        return a

    renderer = _Renderer(env)  # 与 env.render() 同相机同画面（MAE≈0）
    frames: list[np.ndarray] = []
    for i in range(total):
        if i >= SETTLE_STEPS and (i - SETTLE_STEPS) in offsets:
            frames.append(renderer.render_rgb())
        if i < total - 1:  # 最新帧渲染后不 step（P0-2：标签与最新帧同状态）
            env.step(_rand_action())
    # 深度（遮挡判据）+ 关键点（渲染后 data 新鲜）
    depth = renderer.render_depth()
    renderer.close()

    world = keypoint_world_positions(env, task)
    supported = world is not None
    if not supported:
        world = np.zeros((4, 3))
    pixels, depths = project_points(env, world)
    kp = np.stack([pixels[:, 1] / RENDER_SIZE, pixels[:, 0] / RENDER_SIZE], axis=1)  # y,x 归一化
    kp = np.clip(kp, 0.0, 1.0)  # 帧外点 clamp 到边界（visibility=0 掩码）
    _, surface_visible, in_frame = _role_visibility(pixels, depths, depth)
    entity_visible = (
        _entity_aware_visibility(env, world, surface_visible, in_frame)
        if supported
        else surface_visible.copy()
    )
    vis = entity_visible.copy()
    if not supported:
        vis[:] = 0.0
        surface_visible[:] = 0.0
        entity_visible[:] = 0.0
        in_frame[:] = 0.0

    rel, aux = _relation_labels(kp, world) if supported else (
        np.zeros(6, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    contact = (
        float(np.linalg.norm(world[0] - world[1]) < CONTACT_DIST_M)
        if supported and task not in DIRECT_TOOL_TARGET_TASKS
        else 0.0
    )

    # 帧 → [w, 384, 384, 3] uint8（PIL BICUBIC，与 prepare_mw_perturbations 同款）。
    # 捕获循环本身按时间升序，frames 已是 [t-6,t-4,t-2,t]；不要反转。
    frames_small = _resize_chronological_frames(frames)
    record = {
        "frames": frames_small,
        "keypoints": kp.astype(np.float32),
        "visibility": vis,
        "surface_visible": surface_visible,
        "entity_visible": entity_visible,
        "in_frame": in_frame,
        "relation": rel,
        "relation_aux": aux,
        "contact": np.float32(contact),
        "world": world.astype(np.float32),
        "supported": bool(supported),
    }
    if include_raw_frames:
        record["raw_frames"] = np.stack(frames)
    return record


class _Renderer:
    """与 env.render() 逐像素一致（实测 MAE≈0.0004）的 offscreen 渲染器（RGB + 深度）。"""

    def __init__(self, env):
        mujoco = _import_mujoco_metaworld()[0]
        self._renderer = mujoco.Renderer(env.model, height=RENDER_SIZE, width=RENDER_SIZE)
        self._env = env

    def _update(self) -> None:
        self._renderer.update_scene(self._env.data, camera=CAMERA_NAME)

    def render_rgb(self) -> np.ndarray:
        self._update()
        return self._renderer.render()

    def render_depth(self) -> np.ndarray:
        self._renderer.enable_depth_rendering()
        self._update()
        dep = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return dep

    def close(self) -> None:
        self._renderer.close()


def _derive_sample_specs(
    task: str,
    rng: np.random.Generator,
    n: int,
) -> list[tuple[str, int]]:
    """Resolve task and RNG identity before any sample is generated.

    Each record consumes exactly two 31-bit words from the caller's generator:
    one task-selection word and one sample-seed word.  Task selection uses the
    first word only for ``task='any'``; consuming it for fixed tasks keeps the
    parent stream independent of that branch.  Subsequent sample generation is
    driven solely by its local generator, so env reuse and batch boundaries do
    not change records.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if task != "any" and task not in SUPPORTED_TASKS:
        raise ValueError(
            f"unknown task {task!r}; expected one of {len(SUPPORTED_TASKS)} "
            "project MetaWorld tasks or 'any'"
        )
    # Keep the legacy reset-seed domain [0, 2**31), preserving every individual
    # augmentation's marginal distribution while decoupling sample streams.
    words = rng.integers(0, 2**31, size=(n, 2), dtype=np.int64)
    specs: list[tuple[str, int]] = []
    for task_word, seed_word in words:
        # Let Generator perform rejection sampling rather than using modulo, so
        # ``any`` remains exactly uniform even though 2**31 is not divisible by
        # 49.  This local draw cannot perturb another sample's stream.
        resolved_task = task
        if task == "any":
            task_rng = np.random.default_rng(int(task_word))
            resolved_task = SUPPORTED_TASKS[
                int(task_rng.integers(0, len(SUPPORTED_TASKS)))
            ]
        specs.append((resolved_task, int(seed_word)))
    return specs


def _reset_env_for_sample(
    env,
    sample_seed: int,
    base_geom_rgba: np.ndarray | None,
) -> None:
    """Reset one reused env from an isolated per-sample seed."""
    # Our colour augmentation mutates the MuJoCo model, while env.reset only
    # resets simulator data.  Restore the construction-time colours so sample
    # i cannot leak into i+1 (and a new batch's fresh env is equivalent).
    if base_geom_rgba is not None:
        env.model.geom_rgba[:] = base_geom_rgba
    np.random.seed(sample_seed)
    env.reset(seed=sample_seed)


def make_metric_batch(
    task: str,
    rng: np.random.Generator,
    n: int,
    frames_per_sample: int = 4,
    *,
    include_raw_frames: bool = False,
) -> dict:
    """仿真器随机生成阶段 V 数据（无策略，任意观测）。

    契约（artifacts/mt_vj_contract.md §3）：
        frames:        [n, frames_per_sample, 384, 384, 3] uint8（历史→当前，步距 2）
        language_text: [n] str（scripts/build_longtraj_features.ENV_TO_TASK）
        keypoints:     [n, 4, 2] float32（图像坐标 0-1，y,x 序；tool/object/target/interface）
        visibility:    [n, 4] float32（tool/target 严格表面；object/interface
                       允许相机首个 ray hit 为所属非机器人实体 body）
        relation:      [n, 6] float32（[tool−object(2), progress−target(2), axis_cos, depth_m]）
        contact:       [n] float32（|eef−obj| < 3cm）
    额外键（自描述）：surface_visible/entity_visible/in_frame [n,4]、tasks [n] str、
        relation_aux [n,4]、world [n,4,3]、supported [n] bool、meta dict。
    task 支持 "any"（每样本从全部 49 任务均匀抽样）；未知任务 fail-fast。
    随机性契约 ``parent_seed_per_sample_v1``：调用开始即从父 rng 为每个样本
    派生独立 seed，因此同一父 rng 下 8 个样本一次生成或分成 4+4 完全一致。
    """
    if frames_per_sample < 1:
        raise ValueError(f"frames_per_sample must be >= 1, got {frames_per_sample}")
    sample_specs = _derive_sample_specs(task, rng, n)
    tasks = [sample_task for sample_task, _ in sample_specs]
    batch: list[dict] = []
    # env 复用（2026-08-10 性能修复）：每样本新建 env ~0.9s（8 样本 → 7s/batch，
    # 20k 步 40-59h 不可行）。batch 内共享 1 个 env：样本间 reset（metaworld
    # reset 的 _get_state_rand_vec 随机化布局）+ _sample_one 内的颜色/相机/
    # 物体位置/臂位随机化保证样本多样性。混合任务（task="any"）回退新建。
    if len(set(tasks)) > 1:
        for t, sample_seed in sample_specs:
            sample_rng = np.random.default_rng(sample_seed)
            np.random.seed(sample_seed)
            env = make_env(t, seed=sample_seed)
            try:
                rec = _sample_one(
                    env,
                    t,
                    sample_rng,
                    frames_per_sample,
                    include_raw_frames=include_raw_frames,
                )
            finally:
                env.close()
            batch.append(rec)
    else:
        first_seed = sample_specs[0][1]
        np.random.seed(first_seed)
        env = make_env(tasks[0], seed=first_seed)
        model = getattr(env, "model", None)
        geom_rgba = getattr(model, "geom_rgba", None)
        base_geom_rgba = None if geom_rgba is None else geom_rgba.copy()
        try:
            for i, (t, sample_seed) in enumerate(sample_specs):
                if i > 0:
                    _reset_env_for_sample(env, sample_seed, base_geom_rgba)
                sample_rng = np.random.default_rng(sample_seed)
                rec = _sample_one(
                    env,
                    t,
                    sample_rng,
                    frames_per_sample,
                    include_raw_frames=include_raw_frames,
                )
                batch.append(rec)
        finally:
            env.close()
    result = {
        "frames": np.stack([b["frames"] for b in batch]),
        "language_text": [ENV_TO_TASK.get(t, t) for t in tasks],
        "keypoints": np.stack([b["keypoints"] for b in batch]),
        "visibility": np.stack([b["visibility"] for b in batch]),
        "surface_visible": np.stack([b["surface_visible"] for b in batch]),
        "entity_visible": np.stack([b["entity_visible"] for b in batch]),
        "in_frame": np.stack([b["in_frame"] for b in batch]),
        "relation": np.stack([b["relation"] for b in batch]),
        "relation_aux": np.stack([b["relation_aux"] for b in batch]),
        "contact": np.stack([b["contact"] for b in batch]),
        "world": np.stack([b["world"] for b in batch]),
        "tasks": tasks,
        "supported": np.asarray([b["supported"] for b in batch], dtype=bool),
        "meta": {
            "contract": "mt_vj_metric_field_v1",
            "sample_rng_contract": SAMPLE_RNG_CONTRACT,
            "roles": list(ROLE_NAMES),
            "interface_semantics": (
                "progress anchor: second achieved entity when present, else object; "
                "task35 override uses pegHead while target uses visible hole"
            ),
            "keypoints_order": "y,x normalized 0-1",
            "relation_units": "normalized image coords (p_tool-p_object, p_progress-p_target)",
            "relation_aux_units": ["axis_alignment cos", "depth_m (z_progress-z_target)",
                                   "|tool-object|_m", "|progress-target|_m"],
            "contact_units": "|eef-obj| < 0.03m -> 1",
            "visibility": "tool/target strict depth-surface; object/interface strict OR first ray hit is same non-robot entity body",
            "surface_visible": "strict in-frame + depth-surface diagnostic for every role",
            "entity_visible": "training visibility after generic same-entity ray augmentation for object/interface only",
            "in_frame": "projection-only diagnostic; never used as localization supervision",
            "camera": CAMERA_NAME,
            "camera_jitter_m": CAM_JITTER_M,
            "frame_size": IMAGE_SIZE,
            "render_size": RENDER_SIZE,
            "vision_stride": VISION_STRIDE,
            "task_role_source": {
                "tool": "env.tcp_center",
                "object_and_progress": "env._get_pos_objects()",
                "target": "env._target_pos",
                "task_overrides": {
                    name: list(sources)
                    for name, sources in TASK_ALIGNED_ROLE_SOURCES.items()
                },
                "direct_tool_target_families": sorted(DIRECT_TOOL_TARGET_TASKS),
            },
            "language_text_source": "scripts/build_longtraj_features.ENV_TO_TASK",
            "randomization": {
                "reset_seed": "isolated per-sample seed derived from caller rng",
                "object_pos_jitter_m": OBJ_JITTER_M,
                "arm_pose": "random mocap target + IK settle + random small actions",
                "camera_jitter_m": CAM_JITTER_M,
                "object_color": "hue rotation / channel scale on object geoms",
                "articulation": "all finite non-robot hinge/slide joints",
            },
        },
    }
    if include_raw_frames:
        result["raw_frames"] = np.stack([b["raw_frames"] for b in batch])
        result["meta"]["raw_frame_size"] = RENDER_SIZE
    return result


# --------------------------------------------------------------------------
# 投影验证（验收：已知物体 body 投影 vs render 图，误差 <2px）
# --------------------------------------------------------------------------
def verify_projection(task: str, seed: int = 0, verbose: bool = True) -> float:
    """单任务投影验证：四角色真值点投影 vs 渲染图实测像素，返回最大误差（px）。

    方法（mujoco 3.3 实测校准，2026-08-09）：
    - body 真值点：seg 渲染按 geom objid 取 body 上最大可视 geom 的质心；
    - site 真值点：临时把 site rgba 改为品红 → RGB 找品红 blob 质心（绕开
      site 标记颜色与场景撞色，以及 mujoco 3.3 seg 对改写过的 site 的陈旧问题）；
    - ("env"/"point") 派生点：不单独验证（由其来源 site/body 覆盖）。
    """
    mujoco = _import_mujoco_metaworld()[0]
    env = make_env(task, seed=seed)
    try:
        m, d = env.model, env.data
        renderer = mujoco.Renderer(m, height=RENDER_SIZE, width=RENDER_SIZE)

        def _render_rgb():
            renderer.update_scene(d, camera=CAMERA_NAME)
            return renderer.render()

        def _seg():
            renderer.enable_segmentation_rendering()
            renderer.update_scene(d, camera=CAMERA_NAME)
            out = renderer.render()
            renderer.disable_segmentation_rendering()
            return out

        seg = _seg()
        rgb = _render_rgb()

        def seg_body_centroid(bid: int, min_n: int = 3):
            best = None
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) != bid:
                    continue
                ys, xs = np.where(seg[:, :, 0] == g)
                if len(xs) > (best[1] if best else min_n):
                    best = (np.array([xs.mean(), ys.mean()]), len(xs))
            return best

        def magenta_marker(pp, rad: int = 30):
            x0, x1 = max(0, int(pp[0]) - rad), min(RENDER_SIZE, int(pp[0]) + rad + 1)
            y0, y1 = max(0, int(pp[1]) - rad), min(RENDER_SIZE, int(pp[1]) + rad + 1)
            win = rgb[y0:y1, x0:x1].astype(int)
            mask = (win[:, :, 0] > 80) & (win[:, :, 2] > 80) & (win[:, :, 1] < 60)
            ys, xs = np.where(mask)
            if len(xs) < 3:
                return None
            return np.array([x0 + xs.mean(), y0 + ys.mean()]), len(xs)

        errors: list[tuple[str, float]] = []

        def check(label: str, world_pos, gt_px):
            pixels, _ = project_points(env, np.asarray(world_pos, dtype=float)[None, :])
            pp = pixels[0]
            if gt_px is None:
                if verbose:
                    print(f"  {label}: proj={np.round(pp, 1)} (无参考像素，跳过)")
                return
            err = float(np.linalg.norm(pp - gt_px))
            errors.append((label, err))
            if verbose:
                flag = "OK" if err < 2.0 else ("~" if err < 5.0 else "!!")
                print(f"  {label}: proj={np.round(pp, 1)} gt={np.round(gt_px, 1)} "
                      f"err={err:.2f}px [{flag}]")

        table = PROJECTION_REFERENCE_TABLE.get(task, {})
        # tool：tcp_center = 两指 marker 中点（右/左 EEF seg 各自 ~1-2px）
        mids = []
        for sname in ("rightEndEffector", "leftEndEffector"):
            sid = int(m.site(sname).id)
            ys, xs = np.where((seg[:, :, 0] == sid) & (seg[:, :, 1] == 6))
            if len(xs) >= 4:
                mids.append(np.array([xs.mean(), ys.mean()]))
        gt = np.mean(mids, axis=0) if mids else None
        check("tool tcp_center", env.tcp_center, gt)
        # object/target/interface
        for role in ("object", "target", "interface"):
            spec = table.get(role)
            if spec is None:
                continue
            kind, name = spec
            if kind == "body":
                bid = int(m.body(name).id)
                best = seg_body_centroid(bid)
                gt = best[0] if best is not None else None
                check(f"{role} {name}", d.body(name).xpos, gt)
            elif kind == "site":
                m.site(name).rgba = [1.0, 0.0, 1.0, 1.0]  # 品红隔离
                mujoco.mj_forward(m, d)
                rgb = _render_rgb()
                pixels, _ = project_points(env, d.site(name).xpos[None, :])
                mk = magenta_marker(pixels[0])
                m.site(name).rgba = [0.8, 0.0, 0.0, 1.0]
                mujoco.mj_forward(m, d)
                check(f"{role} {name}", d.site(name).xpos, mk[0] if mk else None)
            # ("env"/"point") 派生点：由其来源验证覆盖，不单独检查
        renderer.close()
        if errors:
            worst = max(errors, key=lambda e: e[1])
            if verbose:
                print(f"  -> max err={worst[1]:.2f}px ({worst[0]})")
            return worst[1]
        return float("inf")
    finally:
        env.close()


# --------------------------------------------------------------------------
# 可视化（验收图）
# --------------------------------------------------------------------------
def make_visualization(batch: dict, path: str, sample_ids: list[int] | None = None) -> None:
    """关键点叠在每样本最后（当前）帧上：tool=红 object=绿 target=蓝 interface=品红。

    不可见关键点画空心圆。输出横向并排 PNG。
    """
    from PIL import ImageDraw

    sample_ids = list(range(len(batch["frames"]))) if sample_ids is None else sample_ids
    colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255)]
    tiles = []
    for i in sample_ids:
        frame = batch["frames"][i, -1]  # 当前帧 [384,384,3]
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        kp = batch["keypoints"][i]  # [4,2] y,x
        vis = batch["visibility"][i]
        for r in range(4):
            y, x = kp[r]
            px, py = x * IMAGE_SIZE, y * IMAGE_SIZE
            if vis[r] > 0.5:
                draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=colors[r], width=3)
            else:
                draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(128, 128, 128), width=1)
            draw.text((px + 8, py - 4), ROLE_NAMES[r], fill=colors[r])
        tiles.append(img)
    out = Image.new("RGB", (len(tiles) * IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    for j, img in enumerate(tiles):
        out.paste(img, (j * IMAGE_SIZE, 0))
    out.save(path)
    print(f"[viz] {path} ({len(tiles)} samples side-by-side)")


# --------------------------------------------------------------------------
# 冒烟 / 验收
# --------------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="peg-insert-side-v3",
                    help="49 个项目任务之一，或 any（全任务均匀混合）")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-png", default="/tmp/metric_sample.png")
    ap.add_argument("--no-verify", action="store_true", help="跳过投影验证")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"make_metric_batch(task={args.task!r}, n={args.n}) ...")
    batch = make_metric_batch(args.task, rng, args.n)
    for key, shape in [
        ("frames", batch["frames"].shape), ("keypoints", batch["keypoints"].shape),
        ("visibility", batch["visibility"].shape), ("relation", batch["relation"].shape),
        ("contact", batch["contact"].shape),
    ]:
        print(f"  {key}: {shape}")
    print(f"  language_text: {batch['language_text']}")
    print(f"  tasks: {batch['tasks']} supported: {batch['supported'].tolist()}")
    print("  keypoints (y,x normalized 0-1)  [tool, object, target, interface]:")
    for i in range(len(batch["frames"])):
        print(f"    sample {i}: kp={np.round(batch['keypoints'][i], 3).tolist()}")
        print(f"              vis={np.round(batch['visibility'][i], 1).tolist()} "
              f"rel={np.round(batch['relation'][i], 3).tolist()} "
              f"contact={batch['contact'][i].item():.1f}")
    kp = batch["keypoints"]
    lo, hi = float(kp.min()), float(kp.max())
    print(f"  keypoint range: [{lo:.3f}, {hi:.3f}] -> in [0,1]: {0.0 <= lo and hi <= 1.0}")
    assert batch["frames"].shape == (args.n, 4, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert batch["keypoints"].shape == (args.n, 4, 2)
    assert batch["keypoints"].dtype == np.float32
    make_visualization(batch, args.out_png)

    if not args.no_verify:
        print("\n投影验证（世界坐标 → 渲染图，目标 <2px）:")
        max_errs = {}
        for t in PROJECTION_REFERENCE_TABLE:
            print(f"  [{t}]")
            max_errs[t] = verify_projection(t, seed=args.seed)
        print(f"\n[verify] max projection error per task: "
              f"{ {t: f'{e:.2f}px' for t, e in max_errs.items()} }")


if __name__ == "__main__":
    main()
