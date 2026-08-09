"""C²-IRF v2 Step 3：动态中央凹前缀（fovea，设计文档 §三）。

亚厘米分辨率的主机制：把活动交互关系周围的 ROI（边长 64–192px）放大到
384×384，等效把 patch 分辨率从 16px 提升到 4px（≈4.2mm/patch @96px
crop）。部署默认走完整 V-JEPA H11；冻结前缀保留为 adapter 训练后的可选路径。

- ``compute_roi``：双角色中心/协方差 → 像素空间 ROI [B, 3] (cy, cx, size)；
- ``apply_unified_crop``：对同一 batch 项的全部帧应用**同一个**仿射 crop
  （绝不允许逐帧重新居中——否则制造假的视觉运动，设计文档 §三.2）；
- ``FoveaPrefixEncoder``：共享官方 V-JEPA 2.1 模型实例的冻结前缀，输出
  [B, 2*24*24, D] = [B, 1152, D]，token 顺序与 ``live_vjepa._dense_coords()``
  的 (t, y, x) 网格一致（t 外层循环）。

设计纪律（§九）：1152 patch 只作只读 evidence；前缀永远冻结 + no_grad；
不新增任何辅助 loss。本模块不接训练循环（Wave 2 接入）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

IMAGE_SIZE = 384  # 与 live_vjepa.IMAGE_SIZE 一致
ROI_MIN_SIZE = 64  # 设计文档 §三.1：窄 crop
ROI_MAX_SIZE = 192  # 设计文档 §三.1：宽 crop


def compute_roi(
    mu: Tensor,
    cov: Tensor,
    k_d: float = 1.0,
    k_sigma: float = 2.0,
    image_size: int = IMAGE_SIZE,
    patch_size: int | None = None,
    min_size: int = ROI_MIN_SIZE,
    max_size: int = ROI_MAX_SIZE,
) -> Tensor:
    """双角色中心/协方差 → 像素空间 ROI [B, 3] (cy, cx, size_px)。

    设计文档 §三.1：ROI 中心 c_t = (μ_left + μ_right)/2；边长
    s_t = clip(k_d·|Δμ| + k_σ·√tr(Σ_t), 64, 192)。

    - ``mu`` [B, 2, 3]：两角色归一化中心 (t, y, x) ∈ [-1, 1]
      （LocalControlSlotReader centers 空间）。t 为时间片轴，与空间几何
      无关，距离/尺寸只取 (y, x)；
    - ``cov`` [B, 2, 3, 3]：两角色模式协方差，√trΣ 取两角色对角和的开方；
    - 中心像素用 patch 中心精确映射：py = (y_norm + 1)·half·patch + patch/2
      （half = (grid−1)/2，与 ``_dense_coords()`` 归一化约定一致）；
    - 尺寸像素 = (k_d·Δ + k_σ·√trΣ)·(image_size/2)，clip 到 [min_size, max_size]
      （默认 64–192）。

    返回 [B, 3] float32 (cy, cx, size_px)，供 ``apply_unified_crop`` 使用。
    """
    if mu.ndim != 3 or mu.shape[1] != 2 or mu.shape[2] != 3:
        raise ValueError(
            f"mu 必须为 [B, 2, 3]（两角色归一化 t/y/x 中心），got {tuple(mu.shape)}"
        )
    if cov.ndim != 4 or cov.shape[1] != 2 or cov.shape[2:] != (3, 3):
        raise ValueError(
            f"cov 必须为 [B, 2, 3, 3]（两角色 3×3 协方差），got {tuple(cov.shape)}"
        )
    if mu.shape[0] != cov.shape[0]:
        raise ValueError(f"mu/cov batch 不一致：{mu.shape[0]} vs {cov.shape[0]}")
    if not torch.isfinite(mu).all() or not torch.isfinite(cov).all():
        raise ValueError("mu/cov 必须全部有限（NaN/Inf 会让 crop 输出垃圾）")
    if min_size < 1 or max_size < min_size:
        raise ValueError(f"需要 1 ≤ min_size ≤ max_size，got {min_size}/{max_size}")
    if k_d < 0.0 or k_sigma < 0.0:
        raise ValueError("k_d/k_sigma 必须非负")
    # mu 来自固定的 24×24 dense V-JEPA 网格；渲染图可能是 480px，但在
    # 编码前会 resize 到 384px，因此默认 source patch 尺寸应随 image_size
    # 缩放（480/24=20），不能固定为 16。
    if patch_size is None:
        grid = 24
        patch_size = image_size / grid
    else:
        if patch_size < 1 or image_size % patch_size:
            raise ValueError(
                f"image_size 必须能被 patch_size 整除，got {image_size}/{patch_size}"
            )
        grid = image_size // patch_size
    half = (grid - 1) / 2
    mu = mu.float()
    cov = cov.float()

    center_norm = mu.mean(dim=1)  # [B, 3]（t 分量忽略）
    center_px = (center_norm[:, 1:] + 1.0) * half * patch_size + patch_size / 2.0
    delta = torch.linalg.vector_norm(mu[:, 0, 1:] - mu[:, 1, 1:], dim=-1)  # [B] (y,x) 距离
    tr_sum = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=(-1, -2))  # [B] trΣ0+trΣ1
    size_px = (k_d * delta + k_sigma * torch.sqrt(tr_sum.clamp_min(0.0))) * (
        image_size / 2.0
    )
    size_px = size_px.clamp(min_size, max_size)
    return torch.stack([center_px[:, 0], center_px[:, 1], size_px], dim=-1)


def full_to_crop_norm(
    mu: Tensor, roi: Tensor, image_size: int = IMAGE_SIZE
) -> Tensor:
    """全图归一化坐标 → crop 局部归一化坐标（审查 P0-3 修复）。

    - ``mu`` [..., 3]（t, y, x ∈ [-1, 1]，最后两维为空间坐标）；
    - ``roi`` [B, 3] (cy, cx, size_px)，沿 mu 的 batch 维广播；
    - 坐标沿用 24×24 patch-center 约定：n=±1 对应首/末 patch 中心，
      而不是图像边缘（仅 y/x 变换，t 不变）。
    """
    if mu.ndim < 2 or mu.shape[-1] != 3:
        raise ValueError(f"mu 必须为 [B, ..., 3]，got {tuple(mu.shape)}")
    roi = torch.as_tensor(roi, device=mu.device, dtype=mu.dtype)
    if roi.ndim != 2 or roi.shape != (mu.shape[0], 3):
        raise ValueError(f"roi 必须为 [{mu.shape[0]}, 3]，got {tuple(roi.shape)}")
    shape = (mu.shape[0],) + (1,) * (mu.ndim - 2)
    cy, cx, r = (roi[:, i].reshape(shape) for i in range(3))
    full_span = float(image_size) * 23.0 / 48.0
    crop_span = r * 23.0 / 48.0
    out = mu.clone()
    out[..., 1] = (float(image_size) / 2.0 + mu[..., 1] * full_span - cy) / crop_span
    out[..., 2] = (float(image_size) / 2.0 + mu[..., 2] * full_span - cx) / crop_span
    return out


def crop_to_full_norm(
    mu: Tensor, roi: Tensor, image_size: int = IMAGE_SIZE
) -> Tensor:
    """crop 局部归一化坐标 → 全图归一化坐标（审查 P0-3 修复，reader 输出逆变换）。

    逆变换与 ``full_to_crop_norm`` 使用同一 24×24 patch-center 约定。
    """
    if mu.ndim < 2 or mu.shape[-1] != 3:
        raise ValueError(f"mu 必须为 [B, ..., 3]，got {tuple(mu.shape)}")
    roi = torch.as_tensor(roi, device=mu.device, dtype=mu.dtype)
    if roi.ndim != 2 or roi.shape != (mu.shape[0], 3):
        raise ValueError(f"roi 必须为 [{mu.shape[0]}, 3]，got {tuple(roi.shape)}")
    shape = (mu.shape[0],) + (1,) * (mu.ndim - 2)
    cy, cx, r = (roi[:, i].reshape(shape) for i in range(3))
    full_span = float(image_size) * 23.0 / 48.0
    crop_span = r * 23.0 / 48.0
    out = mu.clone()
    out[..., 1] = (cy + mu[..., 1] * crop_span - float(image_size) / 2.0) / full_span
    out[..., 2] = (cx + mu[..., 2] * crop_span - float(image_size) / 2.0) / full_span
    return out


def crop_to_full_cov(
    cov: Tensor, roi: Tensor, image_size: int = IMAGE_SIZE
) -> Tensor:
    """crop 局部协方差 → 全图坐标协方差（审查 P0-3 修复）。

    线性缩放 y/x：Σ_full[yx] = (r/S)²·Σ_crop[yx]；t-y/x 交叉项缩放 r/S；t 不变。
    """
    if cov.ndim < 3 or cov.shape[-2:] != (3, 3):
        raise ValueError(f"cov 必须为 [B, ..., 3, 3]，got {tuple(cov.shape)}")
    roi = torch.as_tensor(roi, device=cov.device, dtype=cov.dtype)
    if roi.ndim != 2 or roi.shape != (cov.shape[0], 3):
        raise ValueError(f"roi 必须为 [{cov.shape[0]}, 3]，got {tuple(roi.shape)}")
    scale = roi[:, 2] / float(image_size)
    spatial_scale = scale.reshape((cov.shape[0],) + (1,) * (cov.ndim - 1))
    cross_scale = scale.reshape((cov.shape[0],) + (1,) * (cov.ndim - 2))
    out = cov.clone()
    out[..., 1:, 1:] = cov[..., 1:, 1:] * spatial_scale.square()
    out[..., 0, 1:] = cov[..., 0, 1:] * cross_scale
    out[..., 1:, 0] = cov[..., 1:, 0] * cross_scale
    return out


def apply_unified_crop(
    frames: np.ndarray,
    roi: Tensor | np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """同一仿射 crop 全部帧：[B, T, W, 384, 384, 3] uint8 → 同形状 uint8。

    设计文档 §三.2：同一个四帧窗口必须使用完全相同的 crop 仿射变换——ROI
    中心/边长只决定**一个**仿射，应用到该 batch 项的全部 T×W 帧。逐帧
    重新居中会把物体的真实位移抹掉（假静止）或伪造运动。

    - ``roi`` [B, 3] (cy, cx, size_px)：``compute_roi`` 的输出（像素空间）；
    - 仿射：输出像素 dst = scale·(src − center) + image_size/2，
      scale = image_size / size（无整数取整损失，子像素精确）；
    - 双线性采样 + reflect 边界填充（ROI 越界时镜像场景而非注入黑边）；
    - 输出 round/clip 回 uint8。

    返回与输入同形状的 numpy uint8 数组。
    """
    if not isinstance(frames, np.ndarray) or frames.ndim != 6 or frames.shape[-1] != 3:
        raise ValueError("frames 必须为 [B, T, W, H, W, 3] 的 numpy 数组")
    B, T, W, H, Wc, _ = frames.shape
    if H != Wc or H != image_size:
        raise ValueError(
            f"frames 必须为正方形 {image_size}×{image_size}，got {H}×{Wc}"
        )
    roi = torch.as_tensor(roi, dtype=torch.float32)
    if roi.ndim != 2 or roi.shape[1] != 3 or roi.shape[0] != B:
        raise ValueError(f"roi 必须为 [B, 3] (cy, cx, size)，got {tuple(roi.shape)}，B={B}")
    if not torch.isfinite(roi).all():
        raise ValueError("roi 必须全部有限")
    size = roi[:, 2]
    if (size <= 0.0).any() or (size > image_size).any():
        raise ValueError(f"roi size 必须在 (0, {image_size}]，got {size.tolist()}")

    # 注意：W 是窗口帧数（4），图像宽度是 Wc——reshape 必须用 Wc，否则
    # 会得到 (B*T*W, 3, H, W) 的错位形状。
    views = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 1, 2, 5, 3, 4)
    views = views.reshape(B * T * W, 3, H, Wc).float()  # [B*T*W, 3, 384, 384]
    scale = image_size / size  # [B] 输出像素 / 输入像素
    half = image_size / 2.0
    # 输出像素 i 采样输入连续坐标 (i − half)/scale + c：ROI 区域
    # [c − size/2, c + size/2] 放大到整幅输出。手工构造采样网格（不用
    # affine_grid——其 theta 是归一化坐标系的仿射，直接传像素平移会错位）。
    indices = torch.arange(image_size, dtype=torch.float32)
    grid_y = (indices[:, None] - half) / scale[None, :] + roi[:, 0][None, :]  # [H, B]
    grid_x = (indices[:, None] - half) / scale[None, :] + roi[:, 1][None, :]  # [W, B]
    # 连续坐标 → grid_sample 归一化坐标（align_corners=False 像素中心约定）
    grid_y = (grid_y + 0.5) * (2.0 / image_size) - 1.0
    grid_x = (grid_x + 0.5) * (2.0 / image_size) - 1.0
    grid = torch.stack(
        [
            grid_x.T[:, None, :].expand(B, H, Wc),  # x 采样位置随列变化
            grid_y.T[:, :, None].expand(B, H, Wc),  # y 采样位置随行变化
        ],
        dim=-1,
    )  # [B, H, Wc, 2] (x, y)——与 views 的 (b, t, w) 展平顺序一致
    grid = grid[:, None, None].expand(B, T, W, H, Wc, 2).reshape(
        B * T * W, H, Wc, 2
    )
    out = F.grid_sample(
        views, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )  # [B*T*W, 3, 384, 384]
    out = out.reshape(B, T, W, 3, H, Wc).permute(0, 1, 2, 4, 5, 3)
    return out.clamp(0, 255).round().to(torch.uint8).numpy()


class FoveaPrefixEncoder(nn.Module):
    """V-JEPA 2.1 冻结中央凹前缀（设计文档 §三.2/§九）。

    包装官方模型实例（不复制参数，与 ``VJEPA21Backbone`` 共享同一份权重）：
    只运行 patch_embed + blocks[:prefix_blocks] + 层级 norm，全部冻结 +
    no_grad——1152 token 只作只读 evidence，绝不进 VA 自注意力。

    前向复刻官方 video 分支（app/vjepa_2_1/models/vision_transformer.py，
    本仓库 2.1 模型 use_rope=True、无 pos_embed）：

    输入 [B, 4, 3, 384, 384] → permute [B, 3, 4, 384, 384] → patch_embed
    （PatchEmbed3D，t→h→w 展平）→（2.1 特有）video_mod_embed → blocks
    （官方签名 T/H_patches/W_patches/mode="video"；RoPE 位置 arange(T·H·W)
    为 t→h→w 线性序）→ norms_block[norm_index]。

    ``norm_index`` 默认 0：stage-1 层级 norm（官方 out_layers 语义中
    hierarchical_layers=[2,5,8,11]，第 0 个层级 norm 对应 layer 2 输出；
    设计文档的"hierarchical norm"）。输出 [B, 1152, D]，token 顺序与
    ``live_vjepa._dense_coords()`` 的 (t, y, x) 网格一致。

    仅支持官方 V-JEPA 2.1 模型结构（norms_block 必需）；视频分支假设
    4 帧窗口（偶数帧，不会触发 img_temporal_dim_size 的 img 分支）。
    """

    def __init__(
        self,
        model: nn.Module,
        prefix_blocks: int = 2,
        norm_index: int = 0,
    ) -> None:
        super().__init__()
        for attr in ("patch_embed", "blocks", "tubelet_size", "patch_size"):
            if not hasattr(model, attr):
                raise ValueError(f"FoveaPrefixEncoder 需要官方 V-JEPA 模型属性 {attr!r}")
        if not 1 <= prefix_blocks <= len(model.blocks):
            raise ValueError(
                f"prefix_blocks 必须在 [1, {len(model.blocks)}]，got {prefix_blocks}"
            )
        norms_block = getattr(model, "norms_block", None)
        if norms_block is None or not len(norms_block):
            raise ValueError("FoveaPrefixEncoder 需要官方 V-JEPA 2.1 的 norms_block")
        if not 0 <= norm_index < len(norms_block):
            raise ValueError(
                f"norm_index 必须在 [0, {len(norms_block) - 1}]，got {norm_index}"
            )
        self.model = model
        self.prefix_blocks = prefix_blocks
        self.norm = norms_block[norm_index]
        self.freeze_all()

    def freeze_all(self) -> None:
        """只冻结前缀涉及的子模块/参数（共享模型其余部分状态不受影响）。

        作用域刻意收窄：--vision-unfreeze-all 等其他路径解冻 blocks[2:]
        时，中央凹前缀仍保持冻结（设计文档 §九：foveal prefix 冻结）。
        """
        for module in (
            self.model.patch_embed,
            *self.model.blocks[: self.prefix_blocks],
            self.norm,
        ):
            module.requires_grad_(False).eval()
        mod_embed = getattr(self.model, "video_mod_embed", None)
        if mod_embed is not None:
            mod_embed.requires_grad_(False)

    @torch.no_grad()
    def forward(self, crops: Tensor) -> Tensor:
        """[B, frames, 3, H, W]（视频分支，frames 为偶数）→ [B, t·h·w, D]。

        H/W 为 384 时输出 [B, 2*24*24, D] = [B, 1152, D]，token 顺序
        (t, y, x)（t 外层）与 ``live_vjepa._dense_coords()`` 一致。
        """
        if crops.ndim != 5 or crops.shape[2] != 3:
            raise ValueError("crops 必须为 [batch, frames, 3, height, width]")
        frames = crops.shape[1]
        if frames < 2 or frames % 2:
            raise ValueError("V-JEPA 2.1 需要正偶数帧数")
        model = self.model
        temporal = getattr(model, "img_temporal_dim_size", None)
        if temporal is not None and temporal == frames:
            raise ValueError(
                f"img_temporal_dim_size={temporal} 会触发官方 img 分支（单图路径）；"
                "中央凹前缀只支持视频分支（4 帧窗口）"
            )
        parameter = next(model.parameters())
        x = crops.permute(0, 2, 1, 3, 4).to(device=parameter.device, dtype=parameter.dtype)
        if x.shape[3] % model.patch_size or x.shape[4] % model.patch_size:
            raise ValueError(
                f"H/W 必须能被 patch_size={model.patch_size} 整除，"
                f"got {x.shape[3]}x{x.shape[4]}"
            )
        t = x.shape[2] // model.tubelet_size
        h_patches = x.shape[3] // model.patch_size
        w_patches = x.shape[4] // model.patch_size
        x = model.patch_embed(x)
        if getattr(model, "modality_embedding", False):
            x = x + model.video_mod_embed
        for block in model.blocks[: self.prefix_blocks]:
            x, _ = block(
                x,
                mask=None,
                T=t,
                H_patches=h_patches,
                W_patches=w_patches,
                return_attn=False,
                mode="video",
            )
        return self.norm(x)
