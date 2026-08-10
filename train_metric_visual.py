"""MT-VJ 阶段 V：控制度量视觉预训练（契约 §4，2026-08-10）。

冻结 V-JEPA 2.1（fp16）+ 冻结 Qwen（语言缓存只算一次），只训
LanguageMetricField（2.2M 参数，Adam lr=1e-3）。仿真器自动生成随机观测
（prepare_metaworld_metric.make_metric_batch），真值：角色像素位置 / 可见度 /
关系状态 / 接触。

loss = CE(heatmap, Gaussian 标签(σ=2px)) + Huber(p̂, p*) + 1.0·Huber(ĝ, g*)
       + BCE(visibility)
位置类损失按可见度掩码（不可见角色不监督位置，只监督可见度）；g* 关系用世界
坐标（米）[‖eef−obj‖, ‖obj−target‖, axis_alignment, depth]。

每 1000 步打印 train RMSE（px，可见角色，图像坐标 ×384）。checkpoint 契约：
{"config": {...}, "metric_head": state_dict, "relation_encoder": state_dict,
 "contract": "mt_vj_metric_field_v1"}（relation encoder 阶段 V 不训练，随机
初始化保存，供阶段 A train.py 集成加载）。

用法：
    python train_metric_visual.py --steps 20000 --batch-size 8
    python train_metric_visual.py --steps 5 --batch-size 2 --device cpu --verify
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_metaworld_metric import make_metric_batch
from scripts.build_longtraj_features import ENV_TO_TASK
from va_compound.backbones import QwenTextBackbone, VJEPA21Backbone
from va_compound.live_vjepa import _dense_coords
from va_compound.metric_visual_head import (
    D_PROJ,
    H_DIM,
    HEATMAP_GRID,
    N_ROLES,
    LanguageMetricField,
    RelationStateEncoder,
)

IMAGE_SIZE = 384
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
HEATMAP_SIGMA_PX = 2.0  # Gaussian 标签 σ（像素）
# 可见度深度带（米，相对 z_cam）：下界 -0.35 容忍体/孔中心等非表面点
# （如 peg-insert 的 hole site 在盒内，前表面近 ~0.3m）；上界 +0.25 容忍
# 表面法向偏移。带外 → 遮挡/出画 → 不可见。
VIS_DEPTH_BAND_M = (-0.35, 0.25)
RELATION_LAMBDA = 1.0
REL_RECON_LAMBDA = 0.1  # 拍板 3A（2026-08-10）：relation encoder z_g 重建辅助权重
DEFAULT_TASKS = "peg-insert-side-v3,assembly-v3,hand-insert-v3"
CONTRACT = "mt_vj_metric_field_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT-VJ 阶段 V：控制度量视觉预训练")
    parser.add_argument("--tasks", default=DEFAULT_TASKS,
                        help="逗号分隔的 metaworld v3 任务名")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="checkpoints/metric_field.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true",
                        help="训练前验证投影/标签（深度一致性 + 标注图）")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--allow-zero-language-smoke",
        action="store_true",
        help="仅冒烟：允许 Qwen 加载失败时回退随机 role query（正式训练必须 fail-fast，Codex P1-4）",
    )
    parser.add_argument(
        "--data-workers",
        type=int,
        default=4,
        help="阶段 V 数据生成并行 worker 数（make_metric_batch 每样本建 env+渲染，"
        "串行 ~4s/步不可行；4 worker 并行预取 → ~1s/步，2026-08-10）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 续训（自愈用：mujoco 渲染 GL 上下文长期运行失效 → "
        "看护检测卡死后 --resume 重启，2026-08-10）",
    )
    # ---- v2（2026-08-10，三方评审后）----
    parser.add_argument("--l2-norm", action="store_true",
                        help="query/d11 逐行 L2 归一化 → cosine 分数")
    parser.add_argument("--no-bias", action="store_true",
                        help="冻结 spatial_bias（防边缘分布捷径，评审一致要求）")
    parser.add_argument("--temp-init", type=float, default=10.0,
                        help="可学习温度初值（l2_norm 时分数 = temp·cos）")
    parser.add_argument("--sigma-px", type=float, default=2.0,
                        help="高斯标签 σ（像素）；v2 建议 ≥3-4（σ=2 在 patch 交叉点 "
                        "有 clamp 归一化伪影，target 只和 0.45）")
    parser.add_argument("--loc-only", action="store_true",
                        help="只训定位（CE+坐标），跳过 relation/vis/relation-encoder")
    parser.add_argument("--offset-supervision", action="store_true",
                        help="直接监督 GT patch 的 offset：δ* = p* − p_center（SmoothL1）")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="梯度累积步数（batch 4 太小时建议 ≥8）")
    parser.add_argument("--fixed-data", type=str, default=None,
                        help="tiny-set 模式：固定数据集 .pt（make_metric_batch 输出的 dict），"
                        "特征一次性预计算，循环内只训 head（过拟合门）")
    parser.add_argument("--mode-readout", action="store_true",
                        help="v3 模式读出：NMS 全局峰 + 局部 5×5 soft-argmax + 峰 offset "
                        "（探针实证：全网格期望读出在近乎全平的余弦面上 ≈ 均匀分布）")
    parser.add_argument("--hinge-loss", action="store_true",
                        help="v4 max-margin 目标替代 CE（探针实证：CE 在平坦余弦面上 "
                        "收敛到边缘分布，hinge 2000 步达 8.5px）")
    parser.add_argument("--hinge-margin", type=float, default=0.1)
    return parser.parse_args()


def preprocess_frames(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 [B, 4, 384, 384, 3] → 归一化 [B, 4, 3, 384, 384]（与 eval 一致）。"""
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 1, 4, 2, 3)
    tensor = tensor.float().div_(255.0)
    return (tensor - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)


def build_language_cache(
    text_backbone: QwenTextBackbone | None, texts: list[str]
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], bool]:
    """语言缓存只算一次：唯一任务文本 → (hidden [1, L, 2048] fp16, mask [1, L])。

    Qwen 不可用时回退：语言输入置零 + 打印警告（role query 保持随机初始化）。
    """
    unique = sorted(set(texts))
    cache = {}
    if text_backbone is None:
        print("WARNING: QwenTextBackbone 不可用——回退 role query 随机初始化，"
              "语言输入置零（契约 §4 允许的退化路径）")
        for text in unique:
            cache[text] = (torch.zeros(1, 1, 2048, dtype=torch.float16),
                           torch.ones(1, 1, dtype=torch.bool))
        return cache, False
    hidden, mask = text_backbone.encode(unique)  # [T, L, 2048] / [T, L]
    for i, text in enumerate(unique):
        cache[text] = (hidden[i : i + 1].cpu().to(torch.float16), mask[i : i + 1].cpu())
    return cache, True


def gather_language(
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.cat([cache[t][0] for t in texts], dim=0).to(device)
    mask = torch.cat([cache[t][1] for t in texts], dim=0).to(device)
    return hidden, mask


def _gen_batch_worker(task: str, seed: int, n: int) -> dict:
    """阶段 V 数据生成 worker（模块顶层：ProcessPoolExecutor 需可 pickle）。"""
    import numpy as np
    from prepare_metaworld_metric import make_metric_batch
    return make_metric_batch(task, np.random.default_rng(seed), n)


def gaussian_targets(
    keypoints: torch.Tensor, sigma_px: float = HEATMAP_SIGMA_PX,
    grid: int = HEATMAP_GRID, image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """keypoints [B, R, 2]（0-1, y,x）→ [B, R, grid, grid] 高斯标签（每图归一化）。

    Codex P1-2（2026-08-10）：用 patch 中心 (i+0.5)/grid 对齐，避免 keypoint=1
    时中心落在网格外（旧式 keypoint*grid 在边界处 target 总和 ~1.6e-22，CE 消失）。
    """
    sigma = sigma_px / (image_size / grid)  # 像素 → 网格单位
    yc = keypoints[..., 0:1] * grid - 0.5  # [B, R, 1]（patch 中心坐标系）
    xc = keypoints[..., 1:2] * grid - 0.5
    yy = torch.arange(grid, device=keypoints.device, dtype=keypoints.dtype)
    xx = torch.arange(grid, device=keypoints.device, dtype=keypoints.dtype)
    dist2 = (yy.view(1, 1, 1, grid) - yc.unsqueeze(-1)) ** 2 + (
        xx.view(1, 1, grid, 1) - xc.unsqueeze(-1)
    ) ** 2  # [B, R, grid, grid]
    target = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def compute_losses(out, keypoints, visibility, relation, sigma_px: float = HEATMAP_SIGMA_PX,
                   loc_only: bool = False, offset_supervision: bool = False,
                   hinge: bool = False, hinge_margin: float = 0.1) -> tuple[torch.Tensor, dict]:
    """v4（2026-08-10，探针实证）：``hinge`` 用 max-margin 目标替代 CE——
    max(s_GT) 必须超过 max(其余 patch) 至少 ``hinge_margin``。CE 在近乎全平的
    余弦面上梯度 ≈ mean(全图特征) − f_target，被静态背景主导 → 收敛到边缘
    分布（探针：三种读出 × 两种初始化全部 49-58px）；hinge 梯度 = −f_target +
    f_best_other 逐样本相干，2000 步即达 8.5px（scripts/diag_trained_linear_probe.py
    与 hinge 探针实证）。其余组件同前：CE/Huber 位置/offset 按可见度掩码。"""
    device = keypoints.device
    vis = visibility  # [B, R] float
    n_vis = vis.sum().clamp_min(1.0)
    grid = HEATMAP_GRID
    yi = torch.clamp(torch.floor(keypoints[..., 0] * grid).long(), 0, grid - 1)
    xi = torch.clamp(torch.floor(keypoints[..., 1] * grid).long(), 0, grid - 1)
    idx = yi * grid + xi  # [B, R]（片内位置；两片坐标相同）

    parts: dict[str, float] = {}
    if hinge:
        s = out.scores  # [B, R, 1152]
        s_gt = torch.maximum(
            s.gather(-1, idx.unsqueeze(-1)).squeeze(-1),
            s.gather(-1, (idx + grid * grid).unsqueeze(-1)).squeeze(-1),
        )  # [B, R]
        mask_excl = torch.ones_like(s, dtype=torch.bool)
        mask_excl.scatter_(-1, idx.unsqueeze(-1), False)
        mask_excl.scatter_(-1, (idx + grid * grid).unsqueeze(-1), False)
        s_other = s.masked_fill(~mask_excl, -1e9).max(dim=-1).values  # [B, R]
        h = F.relu(hinge_margin - (s_gt - s_other))
        loss_hinge = (h * vis).sum() / n_vis
        parts["hinge"] = loss_hinge.item()
        loss_cls = loss_hinge
    else:
        targets = gaussian_targets(keypoints, sigma_px=sigma_px)
        ce_per = -(targets * out.log_heatmap).sum(dim=(-2, -1))  # [B, R]
        loss_cls = (ce_per * vis).sum() / n_vis
        parts["ce"] = loss_cls.item()

    # Huber 位置（归一化图像坐标 → 像素乘 384 在 RMSE 中体现）
    pos_per = F.smooth_l1_loss(out.p, keypoints, reduction="none").sum(dim=-1)  # [B, R]
    loss_pos = (pos_per * vis).sum() / n_vis
    parts["pos"] = loss_pos.item()

    # v2：GT patch 的直接 offset 监督（δ* = p* − p_center，归一化坐标）
    loss_offset = torch.zeros((), device=device)
    if offset_supervision:
        gt_center = torch.stack(((yi + 0.5) / grid, (xi + 0.5) / grid), dim=-1)
        delta_star = keypoints - gt_center  # [B, R, 2]
        off = out.offset_full[:, :, : grid * grid]  # [B, R, 576, 2]（t=0 片）
        idx4 = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        off_gt = off.gather(dim=2, index=idx4).squeeze(2)  # [B, R, 2]
        loss_offset = (
            F.smooth_l1_loss(off_gt, delta_star, reduction="none").sum(dim=-1) * vis
        ).sum() / n_vis
        parts["offset"] = loss_offset.item()

    if not loc_only:
        # Huber 关系（g* 世界坐标，米）
        loss_rel = F.smooth_l1_loss(out.relation, relation, reduction="mean")
        # BCE 可见度
        loss_vis = F.binary_cross_entropy_with_logits(
            out.visibility_logits, vis, reduction="mean"
        )
        total = loss_cls + loss_pos + RELATION_LAMBDA * loss_rel + loss_vis
        parts.update({"rel": loss_rel.item(), "vis": loss_vis.item()})
    else:
        total = loss_cls + loss_pos + loss_offset
    return total, parts


def verify_labels(tasks: list[str], rng: np.random.Generator) -> None:
    """验证 §3 标签：投影深度一致性（几何自检，≈1-2px）+ 标注图。"""
    import mujoco
    from PIL import Image, ImageDraw

    from prepare_metaworld_metric import (
        CAMERA_NAME,
        ROLE_NAMES,
        VIS_DEPTH_BAND_M,
        _env_pool,
        _role_points,
        project_points,
    )

    colors = ["red", "lime", "blue", "yellow"]
    total_diff = []
    combos = []
    for task in tasks:
        env, renderer = _env_pool(task, 1)[0]
        m, d = env.model, env.data
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        env.reset(seed=int(rng.integers(0, 2**31)))
        renderer.update_scene(d, camera=cam_id)
        img = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera=cam_id)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        world = _role_points(env, task)
        pixels, z = project_points(np.stack([world[r] for r in ROLE_NAMES]), m, cam_id)
        diffs = []
        draw = ImageDraw.Draw(Image.fromarray(img.copy()))
        for k in range(len(ROLE_NAMES)):
            u, v = int(pixels[k, 0]), int(pixels[k, 1])
            if 0 <= u < IMAGE_SIZE and 0 <= v < IMAGE_SIZE:
                delta = float(depth[v, u]) - float(z[k])
                diffs.append(delta)
                draw.ellipse([u - 8, v - 8, u + 8, v + 8], outline=colors[k], width=3)
        total_diff.extend(diffs)
        combos.append(Image.fromarray(img))
        print(f"verify {task}: depth−z_cam per role = {[round(x, 3) for x in diffs]} m"
              f"（带 [{VIS_DEPTH_BAND_M[0]}, {VIS_DEPTH_BAND_M[1]}]，"
              f"带内 {sum(VIS_DEPTH_BAND_M[0] < x < VIS_DEPTH_BAND_M[1] for x in diffs)}/4 可见）")
    if total_diff:
        visible_frac = sum(
            VIS_DEPTH_BAND_M[0] < x < VIS_DEPTH_BAND_M[1] for x in total_diff
        ) / len(total_diff)
        print(f"verify: 深度带内比例 {visible_frac:.2f}（表面点 ~1-2px；"
              f"带外为体/孔中心或遮挡，标签可见度将置 0）")
    combo = Image.new("RGB", (IMAGE_SIZE * len(combos), IMAGE_SIZE))
    for i, im in enumerate(combos):
        combo.paste(im, (i * IMAGE_SIZE, 0))
    combo.save("/tmp/metric_batch_verify.png")
    print("verify: 标注图已保存 /tmp/metric_batch_verify.png")


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--tasks 不能为空")
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ---- V-JEPA 2.1（冻结，fp16） ----
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", local_files_only=True,
    )
    vision_backbone.freeze_all()
    coords = torch.from_numpy(_dense_coords()).to(device)  # [1152, 3]（[-1,1]，head 内部归一化）

    # ---- Qwen 语言缓存（冻结，只算一次；失败则退化） ----
    text_backbone = None
    if device.type == "cuda":
        lang_dtype, lang_device = "float16", device
    else:
        lang_dtype, lang_device = "float32", device
    try:
        text_backbone = QwenTextBackbone.from_pretrained(
            device=lang_device, dtype=lang_dtype, local_files_only=True,
        )
        print(f"train: QwenTextBackbone 就绪（{lang_dtype} on {lang_device}）")
    except Exception as exc:  # noqa: BLE001
        # Codex P1-4（2026-08-10）：正式训练 fail-fast——静默回退会训出
        # 无语言角色的 checkpoint（20k 步后才暴露）。仅显式 smoke 允许退化。
        if not args.allow_zero_language_smoke:
            raise RuntimeError(
                f"Qwen 加载失败且未指定 --allow-zero-language-smoke（正式训练 fail-fast）：{exc!r}"
            ) from exc
        print(f"WARNING: Qwen 加载失败（{exc!r}），回退 role query 随机初始化（仅 smoke）")
    language_cache, _ = build_language_cache(
        text_backbone, [ENV_TO_TASK[t] for t in tasks]
    )

    # ---- 模型（v2：loc_only 只训 metric head；relation encoder 恒创建以保契约） ----
    metric_head = LanguageMetricField(
        l2_norm=args.l2_norm,
        learnable_temp=args.l2_norm,
        temp_init=args.temp_init,
        freeze_bias=args.no_bias,
        mode_readout=args.mode_readout,
    ).to(device)
    # 拍板 3A（2026-08-10）：阶段 V 一起训练 RelationStateEncoder——metric tokens
    # 应为监督学习的关系编码（Codex P1-3），而非随机线性映射。loc_only 时仅
    # 随机初始化保存（checkpoint 契约 §2 要求该键），不进优化器。
    relation_encoder = RelationStateEncoder(state_dim=6).to(device)
    optimizer_params = list(metric_head.parameters())
    if args.loc_only:
        relation_encoder.eval()
        print("train: loc-only 模式——relation/vis 损失跳过，relation encoder 仅随机保存", flush=True)
    else:
        relation_encoder.train()
        optimizer_params += list(relation_encoder.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    n_params = sum(p.numel() for p in metric_head.parameters())
    print(f"train: device={device} tasks={tasks} steps={args.steps} "
          f"batch_size={args.batch_size} lr={args.lr} metric_head_params={n_params / 1e6:.2f}M")

    if args.verify:
        verify_labels(tasks, rng)

    # ---- 训练（2026-08-10 最终版：单进程串行） ----
    # 多进程数据生成被证实不可用：mujoco offscreen 渲染在 worker 进程
    # （fork 或 spawn）中崩溃/退化（BrokenProcessPool + GL 上下文问题），
    # 反复出现"卡住-变慢-消失"。回到单进程串行：每步 = 生成 ~7s + 训练 0.5s，
    # 因此步数由 20k 降到 5k（数据无限生成，metric head 2M 参数 4 万样本足够）。
    rmse_sum = 0.0
    rmse_count = 0
    start_step = 0
    if args.resume:
        # 2026-08-10 自愈：从 checkpoint 续训（mujoco 渲染 GL 上下文长期运行
        # 后失效 → fallback 软件渲染极慢 → 看护检测卡死自动 --resume 重启）
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        metric_head.load_state_dict(ck["metric_head"], strict=False)  # v2 新参数保持初始化
        if relation_encoder is not None:
            relation_encoder.load_state_dict(ck["relation_encoder"])
        start_step = int(ck.get("config", {}).get("steps_done", 0))
        print(f"train: resume from {args.resume}（steps_done={start_step}）", flush=True)

    # 2026-08-10 修复：checkpoint_payload 必须在训练循环前定义——周期保存
    # （循环内调用）此前因定义在循环后被 UnboundLocalError 崩掉（丢进度）。
    def checkpoint_payload() -> dict:
        return {
            "config": {
                "tasks": tasks,
                "steps": args.steps,
                "steps_done": step + 1,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "h_dim": H_DIM,
                "d_proj": D_PROJ,
                "n_roles": N_ROLES,
                "lang_dim": 2048,
                "image_size": IMAGE_SIZE,
                "heatmap_sigma_px": HEATMAP_SIGMA_PX,
                "relation_lambda": RELATION_LAMBDA,
                "relation_encoder_trained": True,  # 拍板 3A（2026-08-10）：阶段 V 联合训练
                "state_dim": 6,  # relation 状态维（拍板 2A；train.py 探针读取）
                "language_cache_available": text_backbone is not None,
                # v2（2026-08-10）：重建 ctor 用（train.py _load_mtvj_metric_checkpoint
                # 按签名过滤注入；缺省字段旧 checkpoint 加载不受影响）
                "l2_norm": args.l2_norm,
                "learnable_temp": args.l2_norm,
                "temp_init": args.temp_init,
                "freeze_bias": args.no_bias,
                "sigma_px": args.sigma_px,
                "mode_readout": args.mode_readout,
                "hinge_loss": args.hinge_loss,
                "hinge_margin": args.hinge_margin,
            },
            "metric_head": metric_head.state_dict(),
            "relation_encoder": relation_encoder.state_dict(),
            "contract": CONTRACT,
        }

    # ---- 数据源（v2）：tiny 固定集（特征一次性预计算，head-only）或仿真流 ----
    fixed = None
    if args.fixed_data:
        fixed = torch.load(args.fixed_data, map_location="cpu", weights_only=False)
        n_fixed = len(fixed["frames"])
        video = preprocess_frames(np.asarray(fixed["frames"]), device)
        with torch.no_grad():
            h5_f, h11_f = vision_backbone.encode_multi(video, out_layers=(5, 11))
        del video
        lang_cache_f, _ = build_language_cache(
            text_backbone, [str(t) for t in fixed["language_text"]]
        )
        kp_f = torch.from_numpy(np.asarray(fixed["keypoints"])).to(device)
        vis_f = torch.from_numpy(np.asarray(fixed["visibility"])).to(device)
        rel_f = torch.from_numpy(np.asarray(fixed["relation"])).to(device)
        print(f"train: tiny 固定集 {n_fixed} 样本，特征预计算完成（head-only）", flush=True)

    for step in range(start_step, args.steps):
        if fixed is not None:
            idx = torch.randperm(n_fixed, device=device)[: args.batch_size]
            h5, h11 = h5_f[idx], h11_f[idx]
            texts = [str(fixed["language_text"][int(i)]) for i in idx.cpu()]
            lang_hidden, lang_mask = gather_language(lang_cache_f, texts, device)
            keypoints, visibility, relation = kp_f[idx], vis_f[idx], rel_f[idx]
        else:
            task = tasks[int(rng.integers(0, len(tasks)))]
            batch = make_metric_batch(task, rng, args.batch_size)
            video = preprocess_frames(batch["frames"], device)  # [B,4,3,384,384]
            with torch.no_grad():
                h5, h11 = vision_backbone.encode_multi(video, out_layers=(5, 11))
                lang_hidden, lang_mask = gather_language(
                    language_cache, batch["language_text"], device
                )
            keypoints = torch.from_numpy(batch["keypoints"]).to(device)
            visibility = torch.from_numpy(batch["visibility"]).to(device)
            relation = torch.from_numpy(batch["relation"]).to(device)

        out = metric_head(h5, h11, lang_hidden, lang_mask, coords)
        loss, parts = compute_losses(
            out, keypoints, visibility, relation,
            sigma_px=args.sigma_px, loc_only=args.loc_only,
            offset_supervision=args.offset_supervision,
            hinge=args.hinge_loss, hinge_margin=args.hinge_margin,
        )
        if not args.loc_only and relation_encoder is not None:
            # 拍板 3A（2026-08-10）：relation encoder 重建监督——z_g 须保留 g_t 信息
            # （Codex P1-3；ν 分支无历史依赖，留阶段 A 监督）。
            g_true = relation
            nu_zero = torch.zeros_like(g_true)
            z_g, _ = relation_encoder(g_true, nu_zero)
            g_recon = relation_encoder.recon(z_g)
            loss = loss + REL_RECON_LAMBDA * F.mse_loss(g_recon, g_true)
        loss = loss / max(args.grad_accum, 1)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss 非有限值 @ step {step}: {loss.item()}")

        loss.backward()
        if (step + 1) % max(args.grad_accum, 1) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # train RMSE（px）：可见角色的位置误差 ×384
        err_px = (out.p.detach() - keypoints) * IMAGE_SIZE  # [B, R, 2]
        sq = (err_px ** 2).sum(dim=-1) * visibility  # [B, R]
        rmse_sum += float(sq.sum())
        rmse_count += int(visibility.sum())

        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            rmse = math.sqrt(rmse_sum / max(rmse_count, 1))
            parts_str = " ".join(f"{k} {v:.4f}" for k, v in parts.items())
            temp_str = ""
            if args.l2_norm and hasattr(metric_head, "temperature"):
                temp_str = f" temp={float(metric_head.temperature.detach()):.2f}"
            print(
                f"step {step + 1}/{args.steps} loss {loss.item():.4f} "
                f"({parts_str}){temp_str} "
                f"train RMSE {rmse:.2f} px  vis_mean {float(visibility.mean()):.2f}",
                flush=True,
            )
        # 周期保存（2026-08-10）：nohup 重定向下 print 会被块缓冲（进度不可见），
        # 且中途崩溃会丢全部训练——每 500 步原子落盘一次（自愈 resume 依赖）。
        if (step + 1) % 500 == 0:
            torch.save(checkpoint_payload(), Path(args.save).with_suffix(".pt.tmp"))
            Path(args.save).with_suffix(".pt.tmp").replace(args.save)
            print(f"  checkpoint @ step {step + 1} → {args.save}", flush=True)

    # ---- 保存 checkpoint（契约 §2/§4；checkpoint_payload 已定义在循环前） ----
    checkpoint = checkpoint_payload()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save(checkpoint, args.save)
    print(f"train: checkpoint saved -> {args.save} ({os.path.getsize(args.save) / 2**20:.1f} MiB)")

    # 验证 weights_only=True 可加载
    loaded = torch.load(args.save, map_location="cpu", weights_only=True)
    assert loaded["contract"] == CONTRACT
    assert set(loaded.keys()) == {"config", "metric_head", "relation_encoder", "contract"}
    print("train: checkpoint 验证通过（weights_only=True 可加载）")


if __name__ == "__main__":
    main()
