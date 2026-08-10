#!/usr/bin/env python
"""MT-VJ 阶段 V 数据生成器：仿真器随机生成 metric 视觉预训练真值。

契约：artifacts/mt_vj_contract.md §3（make_metric_batch 接口）；
设计：artifacts/mt_vj_design.md（四角色度量场 / 阶段 V 预训练）。

每样本：全新 MT1 环境（随机 reset seed + 全局 np.random.seed 复现）、随机物体位置
（free joint 平移，同 scripts/prepare_mw_perturbations.py::move_obj1 手法）、
随机臂位（mocap 目标 + IK settle + 随机小动作）、随机视角（cam_pos ±3cm jitter）、
随机物体颜色（对象 geom rgba 亮度/色相扰动）。

关键点真值（世界坐标 → 图像坐标 pinhole 投影，0-1 归一化 y,x 序）：
    tool      = tcp_center（rightEndEffector/leftEndEffector 中点，metaworld 官方 EEF）
    object    = 被操作物体 body xpos（per-task 映射表）
    target    = 目标 site/body（per-task 映射表）
    interface = 接触界面 site/body/计算点（per-task 映射表）
支持任务：peg-insert-side-v3 / assembly-v3 / hand-insert-v3；其余任务 fallback
返回形状一致的零值标签（meta["supported"]=False 标记）。

投影（2026-08-09 实测验证，误差 <2px）：
    p_cam = R^T (p − cam_pos)，R = mju_quat2Mat(cam_quat) 列 = 相机轴
    x = W/2 + f·(x̂·v)/(−ẑ·v)，y = H/2 − f·(ŷ·v)/(−ẑ·v)，f = (H/2)/tan(fovy/2)
验证方法：mujoco.Renderer offscreen（与 env.render() 逐像素一致，MAE≈0）+ RGB
site marker / seg geom 质心对照（右/左 EEF 1.0/2.0px、hole 0.6px、assembly peg
0.6px、pegTop 0.6px、hand-insert goal 2.9px）。注意：mujoco 3.3 seg 渲染对
「模型 pos 被改写过的 world 系 site」（如 peg-insert 的 goal）给出陈旧/错误位置，
故可见度与验证一律用深度（renderer depth）判遮挡、RGB marker 验 site。

relation 语义（[n,6]，与 metric head 输出同空间；拍板 2A 2026-08-10）：
    rel[:,0:2] = p_eef − p_obj；rel[:,2:4] = p_obj − p_target（归一化图像坐标 y,x）
    rel[:,4] = axis_cos；rel[:,5] = depth_m(z_obj−z_target)
relation_aux（额外键，[n,4]）：[axis_alignment(cos∈[-1,1]), depth_m(z_obj−z_target),
    |eef−obj|_m, |obj−target|_m]（世界距离供 mm 级评估用）。
contact：|eef − obj| < 3cm → 1.0。

用法（CPU 仿真，无 GPU）：
    python prepare_metaworld_metric.py            # 冒烟：2 样本 + 三任务投影验证 + /tmp/metric_sample.png
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
SUPPORTED_TASKS = ("peg-insert-side-v3", "assembly-v3", "hand-insert-v3")
CAMERA_NAME = "corner2"
CAM_POS_DEFAULT = np.array([0.75, 0.075, 0.7])  # lerobot 采集同款 corner2
RENDER_SIZE = 480  # metaworld env 渲染分辨率（square）
IMAGE_SIZE = 384   # 输出帧分辨率
VISION_STRIDE = 2  # 历史帧步距（eval 契约 [d-6, d-4, d-2, d]）
SETTLE_STEPS = 10  # 随机臂位 IK settle 步数
OCCLUSION_TOL_M = 0.05  # 深度判遮挡容差（关键点表面深度 vs 像素深度）
CONTACT_DIST_M = 0.03   # 接触判据：|eef − obj|
CAM_JITTER_M = 0.03     # 视角随机：cam_pos ±3cm
OBJ_JITTER_M = 0.03     # 物体位置随机：水平 ±3cm
_MT1_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------
# 每任务关键点映射表
#   kind ∈ {"body", "site", "point"}；point 为 (任务内计算函数名, ) 于运行时求值
# --------------------------------------------------------------------------
TASK_KEYPOINT_TABLE: dict[str, dict[str, tuple]] = {
    "peg-insert-side-v3": {
        "object": ("body", "peg"),          # 被插的 peg（自由体）
        "target": ("site", "hole"),         # 插入目标：孔（可见；goal site 在方块内部不可见）
        "interface": ("site", "pegGrasp"),  # 工具-物体接触界面：杆上的抓取点（投影在杆上，可见）
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

# 每任务「可动对象 body 名」（free joint 平移用；fallback 走 obs[4:7] 匹配）
TASK_OBJECT_BODY: dict[str, str] = {
    "peg-insert-side-v3": "peg",
    "assembly-v3": "asmbly_peg",
    "hand-insert-v3": "obj",
}

# 每任务对象 geom 名（颜色随机化用；None → 按 body 名找）
TASK_OBJECT_GEOM: dict[str, str | None] = {
    "peg-insert-side-v3": "peg",
    "assembly-v3": None,      # RoundNut body 上的全部 geom
    "hand-insert-v3": "objGeom",
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


def _resolve_keypoint_world(env, spec: tuple, task: str) -> np.ndarray | None:
    """映射表条目 → 世界坐标 [3]；找不到（如 task 无该 site）返回 None。"""
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


def keypoint_world_positions(env, task: str) -> np.ndarray | None:
    """四角色世界坐标 [4,3]（tool=tcp_center；其余 per-task 表）。不支持 → None。"""
    if task not in TASK_KEYPOINT_TABLE:
        return None
    pts = [env.tcp_center]
    for role in ("object", "target", "interface"):
        spec = TASK_KEYPOINT_TABLE[task][role]
        p = _resolve_keypoint_world(env, spec, task)
        if p is None:
            return None
        pts.append(p)
    return np.stack(pts)


# --------------------------------------------------------------------------
# 随机化
# --------------------------------------------------------------------------
def _move_object_random(env, task: str, rng: np.random.Generator) -> None:
    """物体 free joint 水平平移 ±OBJ_JITTER_M（同 move_obj1 手法：free joint 改 qpos）。"""
    import metaworld  # noqa: F401
    mujoco = _import_mujoco_metaworld()[0]
    bid = None
    body_name = TASK_OBJECT_BODY.get(task)
    if body_name is not None:
        try:
            bid = int(env.model.body(body_name).id)
        except KeyError:
            bid = None
    if bid is None:
        # fallback：obs[4:7] 匹配 site/body（跳过机械臂自身），与 move_obj1 一致
        cur = env._get_obs()[4:7].copy()
        for i in range(env.model.nsite):
            if _is_robot_body(env, int(env.model.site_bodyid[i])):
                continue
            if np.allclose(env.data.site_xpos[i], cur, atol=0.02):
                bid = int(env.model.site_bodyid[i])
                break
        if bid is None:
            for b in range(env.model.nbody):
                if _is_robot_body(env, b):
                    continue
                if np.allclose(env.data.body(b).xpos, cur, atol=0.02):
                    bid = b
                    break
    if bid is None:
        return  # 找不到可动物体：跳过（保持默认位置）
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


def _randomize_colors(env, task: str, rng: np.random.Generator) -> None:
    """对象 geom 颜色随机化（亮度/饱和度缩放 或 绕灰轴色相旋转），alpha 保持。"""
    m = env.model
    gids: list[int] = []
    gname = TASK_OBJECT_GEOM.get(task)
    if gname is not None:
        try:
            gids.append(int(m.geom(gname).id))
        except KeyError:
            pass
    else:
        bname = TASK_OBJECT_BODY.get(task)
        if bname is not None:
            try:
                bid = int(m.body(bname).id)
            except KeyError:
                bid = None
            if bid is not None:
                gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid]
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


# --------------------------------------------------------------------------
# 单样本采集
# --------------------------------------------------------------------------
def _sample_one(env, task: str, rng: np.random.Generator, w: int) -> dict:
    mujoco = _import_mujoco_metaworld()[0]
    # 随机化（颜色/相机/物体位置；reset 随机化已在 make_metric_batch 里做）
    _randomize_colors(env, task, rng)
    _jitter_camera(env, rng)
    _move_object_random(env, task, rng)

    # 随机臂位：mocap 目标 + IK settle + 窗口内随机小动作（帧间有运动）
    env.data.mocap_pos[0] = [
        rng.uniform(-0.2, 0.3),
        rng.uniform(0.45, 0.9),
        rng.uniform(0.08, 0.35),
    ]
    env.data.mocap_quat[0] = np.array([1.0, 0.0, 1.0, 0.0])

    offsets = [VISION_STRIDE * (w - 1 - k) for k in range(w)]  # [6,4,2,0]
    # Codex P0-2（2026-08-10）：total = SETTLE + offsets[0] + 1——保证
    # i=SETTLE+offsets[0] 的"最新帧"也被渲染（4 帧齐全）；该帧渲染后不再
    # step（i < total-1 才 step），标签（world 坐标）与最新帧严格同状态。
    total = SETTLE_STEPS + offsets[0] + 1

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
    # 可见度：帧内 + 深度一致（像素处表面深度 ≈ 关键点真实深度，容忍 OCCLUSION_TOL_M）
    vis = np.zeros(4, dtype=np.float32)
    for r in range(4):
        px, py = int(round(pixels[r, 0])), int(round(pixels[r, 1]))
        if 0 <= px < RENDER_SIZE and 0 <= py < RENDER_SIZE and depths[r] > 0:
            if abs(float(depth[py, px]) - float(depths[r])) < OCCLUSION_TOL_M:
                vis[r] = 1.0
    if not supported:
        vis[:] = 0.0

    # relation：6 维（拍板 2A，Codex P0-5）= [p_eef−p_obj(2), p_obj−p_target(2),
    # axis_cos, depth_m]（与 metric head 输出同空间；axis/depth 从世界坐标算）
    rel = np.zeros(6, dtype=np.float32)
    aux = np.zeros(4, dtype=np.float32)
    if supported:
        pe, po, pt, pi = world  # 世界坐标
        rel[0:2] = kp[0, :] - kp[1, :]   # p_eef − p_obj (y,x)
        rel[2:4] = kp[1, :] - kp[2, :]   # p_obj − p_target (y,x)
        deo = pe - po
        dot_ = po - pt
        d_xy = np.linalg.norm(dot_[:2])
        if d_xy > 1e-6:
            rel[4] = float(np.dot(deo[:2], dot_[:2]) / (np.linalg.norm(deo[:2]) * d_xy))
        else:
            rel[4] = 0.0  # 共线退化 → 对齐度 0（不可判）
        rel[5] = float(po[2] - pt[2])            # depth：物体在目标上方的高度差
        aux[2] = float(np.linalg.norm(deo))      # |eef − obj|（米）
        aux[3] = float(np.linalg.norm(dot_))     # |obj − target|（米）
    contact = float(np.linalg.norm(world[0] - world[1]) < CONTACT_DIST_M) if supported else 0.0

    # 帧 → [w, 384, 384, 3] uint8（PIL BICUBIC，与 prepare_mw_perturbations 同款）。
    # 捕获时序：循环按时间升序渲染，frames=[t, t-2, t-4, t-6] → 反转成历史在前。
    frames.reverse()
    frames_small = np.stack(
        [np.asarray(Image.fromarray(f).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)) for f in frames]
    )
    return {
        "frames": frames_small,
        "keypoints": kp.astype(np.float32),
        "visibility": vis,
        "relation": rel,
        "relation_aux": aux,
        "contact": np.float32(contact),
        "world": world.astype(np.float32),
        "supported": bool(supported),
    }


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


def _pick_tasks(task: str, rng: np.random.Generator, n: int) -> list[str]:
    if task == "any":
        return [str(rng.choice(SUPPORTED_TASKS)) for _ in range(n)]
    if task not in SUPPORTED_TASKS:
        return [task] * n  # 不支持任务：fallback（空标签），逐样本标记 supported=False
    return [task] * n


def make_metric_batch(task: str, rng: np.random.Generator, n: int,
                      frames_per_sample: int = 4) -> dict:
    """仿真器随机生成阶段 V 数据（无策略，任意观测）。

    契约（artifacts/mt_vj_contract.md §3）：
        frames:        [n, frames_per_sample, 384, 384, 3] uint8（当前帧+历史帧，步距 2）
        language_text: [n] str（scripts/build_longtraj_features.ENV_TO_TASK）
        keypoints:     [n, 4, 2] float32（图像坐标 0-1，y,x 序；tool/object/target/interface）
        visibility:    [n, 4] float32（帧内 + 深度遮挡判据）
        relation:      [n, 6] float32（[p_eef−p_obj(2), p_obj−p_target(2), axis_cos, depth_m]）
        contact:       [n] float32（|eef−obj| < 3cm）
    额外键（自描述）：tasks [n] str、relation_aux [n,4]、world [n,4,3]、
        supported [n] bool、meta dict（role/语义/相机说明）。
    task 支持 "any"（每样本随机抽支持任务）；不支持的任务 → 形状一致的零标签。
    """
    if frames_per_sample < 1:
        raise ValueError(f"frames_per_sample must be >= 1, got {frames_per_sample}")
    tasks = _pick_tasks(task, rng, n)
    batch: list[dict] = []
    # env 复用（2026-08-10 性能修复）：每样本新建 env ~0.9s（8 样本 → 7s/batch，
    # 20k 步 40-59h 不可行）。batch 内共享 1 个 env：样本间 reset（metaworld
    # reset 的 _get_state_rand_vec 随机化布局）+ _sample_one 内的颜色/相机/
    # 物体位置/臂位随机化保证样本多样性。混合任务（task="any"）回退新建。
    if len(set(tasks)) > 1:
        for i, t in enumerate(tasks):
            np.random.seed(int(rng.integers(0, 2**31)))
            env = make_env(t, seed=42)
            try:
                rec = _sample_one(env, t, rng, frames_per_sample)
            finally:
                env.close()
            batch.append(rec)
    else:
        np.random.seed(int(rng.integers(0, 2**31)))
        env = make_env(tasks[0], seed=42)
        try:
            for i, t in enumerate(tasks):
                if i > 0:
                    np.random.seed(int(rng.integers(0, 2**31)))
                    env.reset(seed=int(rng.integers(0, 2**31)))
                rec = _sample_one(env, t, rng, frames_per_sample)
                batch.append(rec)
        finally:
            env.close()
    return {
        "frames": np.stack([b["frames"] for b in batch]),
        "language_text": [ENV_TO_TASK.get(t, t) for t in tasks],
        "keypoints": np.stack([b["keypoints"] for b in batch]),
        "visibility": np.stack([b["visibility"] for b in batch]),
        "relation": np.stack([b["relation"] for b in batch]),
        "relation_aux": np.stack([b["relation_aux"] for b in batch]),
        "contact": np.stack([b["contact"] for b in batch]),
        "world": np.stack([b["world"] for b in batch]),
        "tasks": tasks,
        "supported": np.asarray([b["supported"] for b in batch], dtype=bool),
        "meta": {
            "contract": "mt_vj_metric_field_v1",
            "roles": list(ROLE_NAMES),
            "keypoints_order": "y,x normalized 0-1",
            "relation_units": "normalized image coords (p_eef-p_obj, p_obj-p_target)",
            "relation_aux_units": ["axis_alignment cos", "depth_m (z_obj-z_target)",
                                   "|eef-obj|_m", "|obj-target|_m"],
            "contact_units": "|eef-obj| < 0.03m -> 1",
            "visibility": "in-frame & depth-occlusion check (tol 0.05m)",
            "camera": CAMERA_NAME,
            "camera_jitter_m": CAM_JITTER_M,
            "frame_size": IMAGE_SIZE,
            "render_size": RENDER_SIZE,
            "vision_stride": VISION_STRIDE,
            "task_keypoint_table": {t: v for t, v in TASK_KEYPOINT_TABLE.items()},
            "language_text_source": "scripts/build_longtraj_features.ENV_TO_TASK",
            "randomization": {
                "reset_seed": "global np.random seeded per sample",
                "object_pos_jitter_m": OBJ_JITTER_M,
                "arm_pose": "random mocap target + IK settle + random small actions",
                "camera_jitter_m": CAM_JITTER_M,
                "object_color": "hue rotation / channel scale on object geoms",
            },
        },
    }


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

        table = TASK_KEYPOINT_TABLE.get(task, {})
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
                    help="任务名或 any（混合）；不支持任务 → 零标签 fallback")
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
        for t in SUPPORTED_TASKS:
            print(f"  [{t}]")
            max_errs[t] = verify_projection(t, seed=args.seed)
        print(f"\n[verify] max projection error per task: "
              f"{ {t: f'{e:.2f}px' for t, e in max_errs.items()} }")


if __name__ == "__main__":
    main()
