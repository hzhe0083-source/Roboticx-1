"""C²-IRF v2 Step 3：中央凹 fovea 单元测试（va_compound/fovea.py）。

覆盖（全部 CPU 可跑；真实 V-JEPA 2.1 测试在本地 checkpoint 存在时启用）：

- ``compute_roi``：中心/尺寸公式、64–192 边界 clip、k_d/k_sigma 可配、
  输入校验；
- ``apply_unified_crop``：仿射一致性（同一窗口 4 帧同一变换——场景平移
  后相对几何不变；移动光斑的位移必须按 scale 保留，逐帧重新居中会抹掉
  位移）、384 恒等退化、形状/类型/ROI 校验；
- ``FoveaPrefixEncoder``：输出形状 [B, 1152, D]、只运行 blocks[:2]、
  作用域冻结（prefix 之外参数不受影响）、no_grad、输入校验；
- 真实模型：与官方 video 分支手工复刻逐位一致、patch_embed 网格顺序
  （t→y→x）与 ``_dense_coords()`` 对齐、显存/内存冒烟报告峰值。
"""
import math
import resource
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from va_compound.backbones import (
    VJEPA21Backbone,
    VJEPA21_CHECKPOINT_BYTES,
    VJEPA21_CHECKPOINT_NAME,
    VJEPA21_REPO_REF,
)
from va_compound.fovea import (
    FoveaPrefixEncoder,
    apply_unified_crop,
    compute_roi,
)
from va_compound.live_vjepa import _dense_coords

_HUB_DIR = Path(torch.hub.get_dir())
_HAS_VJEPA = (
    (_HUB_DIR / f"facebookresearch_vjepa2_{VJEPA21_REPO_REF}").is_dir()
    and (_HUB_DIR / "checkpoints" / VJEPA21_CHECKPOINT_NAME).is_file()
    and (_HUB_DIR / "checkpoints" / VJEPA21_CHECKPOINT_NAME).stat().st_size
    == VJEPA21_CHECKPOINT_BYTES
)


class _FakePatchEmbed(nn.Module):
    """模拟官方 PatchEmbed3D：Conv3d + (t·h·w) 展平。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _CountingBlock(nn.Module):
    """模拟 2.1 Block 的调用签名；记录被调用次数。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.calls = 0

    def forward(
        self,
        x,
        mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        return_attn=False,
        mode="video",
    ):
        self.calls += 1
        return self.fc(x), None


class FakePrefixModel(nn.Module):
    """模拟官方 V-JEPA 2.1 ViT 中央凹前缀所需的子模块结构。"""

    def __init__(self, dim: int = 16, n_blocks: int = 4) -> None:
        super().__init__()
        self.tubelet_size = 2
        self.patch_size = 16
        self.img_temporal_dim_size = 1
        self.modality_embedding = True
        self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embed = _FakePatchEmbed(dim)
        self.blocks = nn.ModuleList([_CountingBlock(dim) for _ in range(n_blocks)])
        self.norms_block = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])


class ComputeRoiTests(unittest.TestCase):
    def test_center_and_size_formula(self):
        # 两角色中心 (t=0, y=∓0.2, x=∓0.1) → 中心 (0, 0) → 像素 (192, 192)
        mu = torch.tensor([[[0.0, -0.2, -0.1], [0.0, 0.2, 0.1]]])
        cov = torch.zeros(1, 2, 3, 3)
        roi = compute_roi(mu, cov)
        self.assertAlmostEqual(float(roi[0, 0]), 192.0, places=4)
        self.assertAlmostEqual(float(roi[0, 1]), 192.0, places=4)
        # Δ = sqrt(0.16 + 0.04) = sqrt(0.2) → 尺寸 = Δ·192（clip 内 85.86）
        expected = math.sqrt(0.2) * 192.0
        self.assertAlmostEqual(float(roi[0, 2]), expected, places=3)

    def test_covariance_uncertainty_term(self):
        # trΣ = 3·(0.002 + 0.006) = 0.024；k_σ=2 → √trΣ·2·192 ≈ 59.5
        mu = torch.tensor([[[0.0, -0.2, -0.1], [0.0, 0.2, 0.1]]])
        cov = torch.zeros(1, 2, 3, 3)
        cov[0, 0].fill_diagonal_(0.002)
        cov[0, 1].fill_diagonal_(0.006)
        roi = compute_roi(mu, cov, k_d=1.0, k_sigma=2.0)
        tr_sum = 3.0 * (0.002 + 0.006)
        expected = (math.sqrt(0.2) + 2.0 * math.sqrt(tr_sum)) * 192.0
        self.assertAlmostEqual(float(roi[0, 2]), expected, places=3)
        self.assertLess(expected, 192.0)  # 本组数值不应触发 clip

    def test_k_weights_configurable(self):
        mu = torch.tensor([[[0.0, -0.2, -0.1], [0.0, 0.2, 0.1]]])
        cov = torch.zeros(1, 2, 3, 3)
        base = math.sqrt(0.2) * 192.0
        roi = compute_roi(mu, cov, k_d=2.0)
        self.assertAlmostEqual(float(roi[0, 2]), 2.0 * base, places=3)
        cov = torch.zeros(1, 2, 3, 3)
        cov[0, 0].fill_diagonal_(0.05)
        cov[0, 1].fill_diagonal_(0.05)
        tr_sum = 3.0 * 0.10
        roi = compute_roi(mu, cov, k_d=0.0, k_sigma=1.0)
        # 105.2 ∈ [64, 192]，不触发 clip
        self.assertAlmostEqual(float(roi[0, 2]), math.sqrt(tr_sum) * 192.0, places=3)

    def test_size_clipped_to_bounds(self):
        # 重合中心 + 零协方差 → 尺寸 0 → clip 到 min 64
        mu = torch.zeros(1, 2, 3)
        roi = compute_roi(mu, torch.zeros(1, 2, 3, 3))
        self.assertAlmostEqual(float(roi[0, 2]), 64.0, places=4)
        # 对角远端 → 543px → clip 到 max 192
        mu = torch.tensor([[[0.0, -1.0, -1.0], [0.0, 1.0, 1.0]]])
        roi = compute_roi(mu, torch.zeros(1, 2, 3, 3))
        self.assertAlmostEqual(float(roi[0, 2]), 192.0, places=4)
        # 自定义边界生效
        roi = compute_roi(mu, torch.zeros(1, 2, 3, 3), min_size=32, max_size=64)
        self.assertAlmostEqual(float(roi[0, 2]), 64.0, places=4)

    def test_exact_boundary_values_not_clipped(self):
        # Δ = 1/3 → 尺寸恰为 64；Δ = 1 → 恰为 192（clip 不应吞掉精确边界）
        mu = torch.tensor([[[0.0, -1.0 / 6.0, 0.0], [0.0, 1.0 / 6.0, 0.0]]])
        roi = compute_roi(mu, torch.zeros(1, 2, 3, 3))
        self.assertAlmostEqual(float(roi[0, 2]), 64.0, places=4)
        mu = torch.tensor([[[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]]])
        roi = compute_roi(mu, torch.zeros(1, 2, 3, 3))
        self.assertAlmostEqual(float(roi[0, 2]), 192.0, places=4)

    def test_batch_mapping(self):
        mu = torch.zeros(2, 2, 3)
        mu[1, 0, 1:] = torch.tensor([-0.5, 0.0])
        mu[1, 1, 1:] = torch.tensor([0.5, 0.0])
        roi = compute_roi(mu, torch.zeros(2, 2, 3, 3))
        self.assertEqual(roi.shape, (2, 3))
        self.assertAlmostEqual(float(roi[1, 0]), 192.0, places=4)  # y 中心
        self.assertAlmostEqual(float(roi[1, 2]), 192.0, places=4)  # Δ=1 → clip 上限

    def test_validation(self):
        mu = torch.zeros(1, 2, 3)
        cov = torch.zeros(1, 2, 3, 3)
        with self.assertRaises(ValueError):
            compute_roi(torch.zeros(1, 3, 3), cov)
        with self.assertRaises(ValueError):
            compute_roi(torch.zeros(1, 2, 2), cov)
        with self.assertRaises(ValueError):
            compute_roi(mu, torch.zeros(1, 2, 2, 2))
        with self.assertRaises(ValueError):
            compute_roi(torch.zeros(2, 2, 3), cov)
        with self.assertRaises(ValueError):
            compute_roi(torch.full((1, 2, 3), float("nan")), cov)
        with self.assertRaises(ValueError):
            compute_roi(mu, cov, min_size=0)
        with self.assertRaises(ValueError):
            compute_roi(mu, cov, min_size=200, max_size=192)


class ApplyUnifiedCropTests(unittest.TestCase):
    def test_identity_at_full_size(self):
        # size=384、中心=192 → 恒等仿射 → 逐位还原（warp 的往返保真度）。
        # 注意：size=384 但中心≠192 时 scale=1 仍有平移（非恒等）。
        frames = np.random.randint(0, 256, size=(2, 1, 4, 384, 384, 3), dtype=np.uint8)
        roi = torch.tensor([[192.0, 192.0, 384.0], [192.0, 192.0, 384.0]])
        out = apply_unified_crop(frames, roi)
        self.assertEqual(out.shape, frames.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.array_equal(out, frames))

    def test_shape_and_dtype(self):
        frames = np.random.randint(0, 256, size=(3, 2, 4, 384, 384, 3), dtype=np.uint8)
        roi = np.array([[200.0, 100.0, 128.0], [100.0, 300.0, 96.0], [50.0, 60.0, 64.0]])
        out = apply_unified_crop(frames, roi)
        self.assertEqual(out.shape, frames.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue((out <= 255).all())

    def test_same_affine_across_frames_preserves_motion(self):
        # 亮斑按 (8, 6)px/帧 移动；ROI 中心固定在起始位置，size=128（scale=3）。
        # 同一窗口 4 帧必须共用同一仿射：3 帧后总位移 = 3×(8,6)×3 = (72,54)。
        # 若逐帧重新居中，输出中位移会消失（假静止）——本测试专门抓这种假运动。
        scene = np.zeros((4, 384, 384, 3), dtype=np.uint8)
        for w in range(4):
            y0, x0 = 96 + w * 8, 96 + w * 6
            scene[w, y0 : y0 + 12, x0 : x0 + 12] = 255
        frames = scene[None, None]  # [1, 1, 4, 384, 384, 3]
        roi = torch.tensor([[102.0, 102.0, 128.0]])  # 点中心 (96+6, 96+6)，size=128
        out = apply_unified_crop(frames, roi)[0, 0]  # [4, 384, 384, 3]

        def centroid(frame):
            ys, xs = np.where(frame[..., 0] > 128)
            return np.array([ys.mean(), xs.mean()])

        c0 = centroid(out[0])
        c3 = centroid(out[3])
        np.testing.assert_allclose(c0, [192.0, 192.0], atol=2.0)
        # 总位移 = scale×(8,6)×3 帧 = (72, 54)
        np.testing.assert_allclose(c3, [192.0 + 72.0, 192.0 + 54.0], atol=2.5)
        np.testing.assert_allclose(c3 - c0, [72.0, 54.0], atol=3.0)

    def test_translation_invariance_via_compute_roi(self):
        # 两个光斑（R/G 通道分离，避免插值渐变污染质心阈值）的相对几何；
        # 整幅场景平移 (24, -16) 且 ROI 跟随平移后，crop 输出中的相对位置
        # 必须不变（设计 §三.2 平移不变性）。
        def draw(img, p, channel):
            y0, x0 = p.astype(int)
            img[y0 : y0 + 12, x0 : x0 + 12, channel] = 255

        def norm(px):
            # 像素 → 归一化（与 _dense_coords 同一约定：py = (n+1)·184 + 8）
            return (px - 8.0) / 184.0 - 1.0

        def make_pair(p1, p2):
            scene = np.zeros((384, 384, 3), dtype=np.uint8)
            draw(scene, p1, 0)
            draw(scene, p2, 1)
            mu = torch.tensor(
                [[[0.0, norm(p1[0]), norm(p1[1])], [0.0, norm(p2[0]), norm(p2[1])]]]
            )
            roi = compute_roi(mu, torch.zeros(1, 2, 3, 3))
            frames = np.broadcast_to(scene[None, :, :, :], (1, 2, 4, 384, 384, 3))
            return apply_unified_crop(frames, roi)[0, 0, 0]  # [384, 384, 3]

        def relative(frame):
            c1 = np.mean(np.argwhere(frame[..., 0] > 128), axis=0)
            c2 = np.mean(np.argwhere(frame[..., 1] > 128), axis=0)
            return c2 - c1

        p1, p2 = np.array([120.0, 140.0]), np.array([200.0, 220.0])
        shift = np.array([24.0, -16.0])
        out1 = make_pair(p1, p2)
        out2 = make_pair(p1 + shift, p2 + shift)
        rel1, rel2 = relative(out1), relative(out2)
        # 平移不变：相对几何一致
        np.testing.assert_allclose(rel1, rel2, atol=2.0)
        # 绝对一致性：相对位置 = (p2-p1) × (384/size)，size=Δ·192
        mu = torch.tensor(
            [[[0.0, norm(p1[0]), norm(p1[1])], [0.0, norm(p2[0]), norm(p2[1])]]]
        )
        size = float(compute_roi(mu, torch.zeros(1, 2, 3, 3))[0, 2])
        expected = (p2 - p1) * (384.0 / size)
        np.testing.assert_allclose(rel1, expected, atol=3.0)

    def test_validation(self):
        frames = np.zeros((1, 1, 4, 384, 384, 3), dtype=np.uint8)
        roi = torch.tensor([[192.0, 192.0, 96.0]])
        with self.assertRaises(ValueError):
            apply_unified_crop(np.zeros((1, 1, 4, 384, 384), dtype=np.uint8), roi)
        with self.assertRaises(ValueError):
            apply_unified_crop(np.zeros((1, 1, 4, 384, 384, 2), dtype=np.uint8), roi)
        with self.assertRaises(ValueError):
            apply_unified_crop(np.zeros((1, 1, 4, 384, 383, 3), dtype=np.uint8), roi)
        with self.assertRaises(ValueError):
            apply_unified_crop(frames, torch.tensor([[192.0, 192.0]]))
        with self.assertRaises(ValueError):
            apply_unified_crop(frames, torch.tensor([[192.0, 192.0, 0.0]]))
        with self.assertRaises(ValueError):
            apply_unified_crop(frames, torch.tensor([[192.0, 192.0, 385.0]]))
        with self.assertRaises(ValueError):
            apply_unified_crop(frames, torch.tensor([[float("nan"), 192.0, 96.0]]))
        # batch 不匹配
        with self.assertRaises(ValueError):
            apply_unified_crop(frames, torch.tensor([[192.0, 192.0, 96.0]] * 2))


class FoveaPrefixEncoderTests(unittest.TestCase):
    def test_shape_and_prefix_scope(self):
        model = FakePrefixModel(dim=16, n_blocks=4)
        fov = FoveaPrefixEncoder(model)
        crops = torch.randn(2, 4, 3, 384, 384)
        out = fov(crops)
        self.assertEqual(out.shape, (2, 2 * 24 * 24, 16))  # [B, 1152, D]
        # 只调用 blocks[:2]；blocks[2:] 不被调用
        self.assertEqual([b.calls for b in model.blocks[:2]], [1, 1])
        self.assertEqual([b.calls for b in model.blocks[2:]], [0, 0])
        # no_grad：输出不要求梯度
        self.assertFalse(out.requires_grad)

    def test_scoped_freeze(self):
        model = FakePrefixModel(dim=16, n_blocks=4)
        FoveaPrefixEncoder(model)
        # 前缀子模块全部冻结
        self.assertFalse(model.patch_embed.proj.weight.requires_grad)
        self.assertFalse(model.blocks[0].fc.weight.requires_grad)
        self.assertFalse(model.blocks[1].fc.weight.requires_grad)
        self.assertFalse(model.norms_block[0].weight.requires_grad)
        self.assertFalse(model.video_mod_embed.requires_grad)
        # 作用域收窄：prefix 之外不受影响（--vision-unfreeze-all 兼容）
        self.assertTrue(model.blocks[2].fc.weight.requires_grad)
        self.assertTrue(model.blocks[3].fc.weight.requires_grad)
        self.assertTrue(model.norms_block[1].weight.requires_grad)
        # eval 模式
        self.assertFalse(model.blocks[0].training)
        self.assertTrue(model.blocks[2].training)

    def test_prefix_blocks_configuration(self):
        model = FakePrefixModel(dim=16, n_blocks=4)
        fov = FoveaPrefixEncoder(model, prefix_blocks=1)
        fov(torch.randn(1, 4, 3, 384, 384))
        self.assertEqual(model.blocks[0].calls, 1)
        self.assertEqual(model.blocks[1].calls, 0)
        self.assertEqual(fov.norm, model.norms_block[0])

    def test_norm_index_configuration(self):
        model = FakePrefixModel(dim=16, n_blocks=4)
        fov = FoveaPrefixEncoder(model, norm_index=3)
        self.assertEqual(fov.norm, model.norms_block[3])
        with self.assertRaises(ValueError):
            FoveaPrefixEncoder(model, norm_index=4)
        with self.assertRaises(ValueError):
            FoveaPrefixEncoder(model, prefix_blocks=0)
        with self.assertRaises(ValueError):
            FoveaPrefixEncoder(model, prefix_blocks=13)

    def test_requires_official_model_structure(self):
        with self.assertRaises(ValueError):
            FoveaPrefixEncoder(nn.Linear(4, 4))
        class _NoHierNorms(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv3d(3, 4, (2, 16, 16), (2, 16, 16))
                self.blocks = nn.ModuleList([_CountingBlock(4)])
                self.tubelet_size = 2
                self.patch_size = 16

        with self.assertRaises(ValueError):
            FoveaPrefixEncoder(_NoHierNorms())

    def test_input_validation(self):
        model = FakePrefixModel(dim=16, n_blocks=4)
        fov = FoveaPrefixEncoder(model)
        with self.assertRaises(ValueError):
            fov(torch.randn(1, 4, 384, 384, 3))  # ndim 错误
        with self.assertRaises(ValueError):
            fov(torch.randn(1, 4, 4, 384, 384))  # 通道数错误
        with self.assertRaises(ValueError):
            fov(torch.randn(1, 5, 3, 384, 384))  # 奇数帧
        with self.assertRaises(ValueError):
            fov(torch.randn(1, 1, 3, 384, 384))  # 帧数过少


@unittest.skipUnless(_HAS_VJEPA, "本地 V-JEPA 2.1 checkpoint 不存在，跳过真实模型测试")
class RealVJEPAFoveaTests(unittest.TestCase):
    """真实 V-JEPA 2.1（CPU float32，本地 checkpoint）验证前缀契约。"""

    @classmethod
    def setUpClass(cls):
        cls.backbone = VJEPA21Backbone.from_pretrained(
            device="cpu", dtype="float32", local_files_only=True
        )

    def test_prefix_matches_official_video_branch(self):
        model = self.backbone.model
        fov = FoveaPrefixEncoder(model)
        crops = torch.randn(1, 4, 3, 384, 384)
        out = fov(crops)
        self.assertEqual(out.shape, (1, 1152, 768))
        # 官方 video 分支手工复刻：patch_embed → video_mod_embed →
        # blocks[:2]（官方签名）→ norms_block[0]；必须逐位一致。
        with torch.no_grad():
            x = crops.permute(0, 2, 1, 3, 4)
            t = x.shape[2] // model.tubelet_size
            h_patches = x.shape[3] // model.patch_size
            w_patches = x.shape[4] // model.patch_size
            ref = model.patch_embed(x)
            if getattr(model, "modality_embedding", False):
                ref = ref + model.video_mod_embed
            for block in model.blocks[:2]:
                ref, _ = block(
                    ref,
                    mask=None,
                    T=t,
                    H_patches=h_patches,
                    W_patches=w_patches,
                    return_attn=False,
                    mode="video",
                )
            ref = model.norms_block[0](ref)
        self.assertTrue(torch.equal(out, ref))

    def test_patch_embed_grid_order_matches_dense_coords(self):
        # patch_embed 是局部卷积（stride=patch）：t 片 0 的 (y=5, x=7) 单
        # patch 信号只影响展平 token 5*24+7=127 —— 证明 t→y→x 展平顺序
        # 与 _dense_coords() 的 (t, y, x) 网格一致。
        model = self.backbone.model
        FoveaPrefixEncoder(model)
        x = torch.zeros(1, 4, 3, 384, 384)
        x[:, 0:2, :, 5 * 16 : 6 * 16, 7 * 16 : 8 * 16] = 1.0  # t 片 0（帧 0-1）
        with torch.no_grad():
            pe = model.patch_embed(x.permute(0, 2, 1, 3, 4))
            base = model.patch_embed(torch.zeros(1, 3, 4, 384, 384))
        diff = (pe - base).norm(dim=-1)[0]
        idx = int(diff.argmax())
        self.assertEqual(idx, 5 * 24 + 7)
        self.assertEqual(int((diff != 0).sum()), 1)  # 只有该 token 受影响
        coords = _dense_coords()
        expected = np.array([-1.0, (5 - 11.5) / 11.5, (7 - 11.5) / 11.5])
        np.testing.assert_allclose(coords[idx], expected, atol=1e-6)

    def test_prefix_does_not_run_deeper_blocks(self):
        model = self.backbone.model
        fov = FoveaPrefixEncoder(model)
        original = model.blocks[2].forward

        def _boom(*_args, **_kwargs):
            raise AssertionError("blocks[2] 不应在前缀前向中运行")

        model.blocks[2].forward = _boom
        try:
            out = fov(torch.randn(1, 4, 3, 384, 384))
        finally:
            model.blocks[2].forward = original
        self.assertEqual(out.shape, (1, 1152, 768))

    def test_fovea_prefix_memory_smoke(self):
        """小 batch 显存/内存冒烟并报告峰值（不干扰正在运行的训练）。"""
        batch = 1
        peak_text = ""
        if torch.cuda.is_available():
            free_mib = torch.cuda.mem_get_info()[0] / 2**20
            if free_mib > 2048:  # 保守：训练占用 GPU，空闲 < 2GiB 时回退 CPU
                backbone = VJEPA21Backbone.from_pretrained(
                    device="cuda", dtype="bfloat16", local_files_only=True
                )
                fov = FoveaPrefixEncoder(backbone.model).cuda()
                torch.cuda.reset_peak_memory_stats()
                batch = 2
                try:
                    out = fov(torch.randn(batch, 4, 3, 384, 384, device="cuda"))
                finally:
                    del fov, backbone
                    torch.cuda.empty_cache()
                peak_mib = torch.cuda.max_memory_allocated() / 2**20
                peak_text = f"CUDA B={batch} 峰值显存 {peak_mib:.1f} MiB（空闲 {free_mib:.0f} MiB）"
            else:
                peak_text = f"GPU 空闲仅 {free_mib:.0f} MiB，回退 CPU 冒烟"
        if not peak_text:
            baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            out = FoveaPrefixEncoder(self.backbone.model)(
                torch.randn(batch, 4, 3, 384, 384)
            )
            peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline
            peak_text = f"CPU B={batch} 峰值 RSS {peak_kib} KiB"
        print(f"[fovea smoke] {peak_text}")
        self.assertEqual(out.shape, (batch, 1152, 768))


if __name__ == "__main__":
    unittest.main()


class CoordTransformTests(unittest.TestCase):
    """审查 P0-3 修复：full↔crop 归一化坐标变换与协方差缩放。"""

    def test_center_roi_roundtrip(self):
        """中心对齐 ROI（c=S/2, r=96, S=480）：crop 归一化 0 ↔ 全图 0。"""
        import torch
        from va_compound.fovea import crop_to_full_cov, crop_to_full_norm, full_to_crop_norm

        roi = torch.tensor([[240.0, 240.0, 96.0]])
        mu_full = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.3, -0.2]]])
        mu_crop = full_to_crop_norm(mu_full, roi, image_size=480)
        # 中心：n_crop = (n_full+1)*S/r − 2c/r = 0（c=S/2 时）
        torch.testing.assert_close(mu_crop[0, 0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-5, rtol=0)
        back = crop_to_full_norm(mu_crop, roi, image_size=480)
        torch.testing.assert_close(back, mu_full, atol=1e-5, rtol=0)

    def test_offset_roi_numeric(self):
        """手算核对：S=480, c=(240,240), r=96；n_full=(0.5,0.5) → n_crop=(2.5,2.5)。"""
        import torch
        from va_compound.fovea import full_to_crop_norm

        roi = torch.tensor([[240.0, 240.0, 96.0]])
        mu = torch.tensor([[[0.0, 0.5, 0.5]]])
        out = full_to_crop_norm(mu, roi, image_size=480)
        # (0.5+1)*480/96 − 2*240/96 = 7.5 − 5 = 2.5
        torch.testing.assert_close(out[0, 0, 1:], torch.tensor([2.5, 2.5]), atol=1e-5, rtol=0)

    def test_cov_scale(self):
        """协方差缩放：Σ_full[yx] = (r/S)²·Σ_crop；t 维不变；交叉项缩放 r/S。"""
        import torch
        from va_compound.fovea import crop_to_full_cov

        roi = torch.tensor([[240.0, 240.0, 96.0]])  # r/S = 0.2
        cov = torch.zeros(1, 2, 3, 3)
        cov[0, 0] = torch.eye(3)
        cov[0, 0, 0, 1] = cov[0, 0, 1, 0] = 0.5  # t-y 交叉项非零（验证缩放）
        out = crop_to_full_cov(cov, roi, image_size=480)
        self.assertAlmostEqual(float(out[0, 0, 1, 1]), 0.04, places=5)  # (0.2)²
        self.assertEqual(float(out[0, 0, 0, 0]), 1.0)  # t 不变
        self.assertAlmostEqual(float(out[0, 0, 0, 1]), 0.1, places=5)  # 交叉项 = 0.5×r/S

    def test_batch_roi(self):
        """批量 ROI（B=4）必须沿 batch 维广播，并与逐样本结果一致。"""
        import torch
        from va_compound.fovea import crop_to_full_cov, crop_to_full_norm, full_to_crop_norm

        roi = torch.tensor([
            [240.0, 240.0, 96.0],
            [120.0, 360.0, 192.0],
            [300.0, 160.0, 64.0],
            [80.0, 100.0, 128.0],
        ])
        mu = torch.randn(4, 6, 2, 3)  # [B, K, modes, 3]
        cov = torch.randn(4, 6, 2, 3, 3)
        mu_crop = full_to_crop_norm(mu, roi, image_size=480)
        back = crop_to_full_norm(mu_crop, roi, image_size=480)
        torch.testing.assert_close(back, mu, atol=1e-4, rtol=1e-4)
        per_sample = torch.cat([
            full_to_crop_norm(mu[b : b + 1], roi[b : b + 1], image_size=480)
            for b in range(4)
        ])
        torch.testing.assert_close(mu_crop, per_sample, atol=1e-5, rtol=1e-5)
        cov_full = crop_to_full_cov(cov, roi, image_size=480)
        cov_per_sample = torch.cat([
            crop_to_full_cov(cov[b : b + 1], roi[b : b + 1], image_size=480)
            for b in range(4)
        ])
        torch.testing.assert_close(cov_full, cov_per_sample, atol=1e-5, rtol=1e-5)

    def test_compute_roi_center_maps_to_crop_center(self):
        """生成 ROI 的同一 full-frame 中心必须映射到 crop 坐标 0。"""
        import torch
        from va_compound.fovea import compute_roi, full_to_crop_norm

        mu = torch.tensor([[[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]]])
        roi = compute_roi(
            mu, torch.zeros(1, 2, 3, 3), image_size=480,
            min_size=96, max_size=96,
        )
        local = full_to_crop_norm(mu, roi, image_size=480)
        torch.testing.assert_close(local[..., 1:], torch.zeros_like(local[..., 1:]), atol=1e-5, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_cpu_roi_with_cuda_readout(self):
        """eval 保持 ROI 在 CPU；坐标函数必须自动对齐 CUDA readout。"""
        from va_compound.fovea import crop_to_full_cov, crop_to_full_norm, full_to_crop_norm

        roi = torch.tensor([[240.0, 240.0, 96.0]])  # 与 apply_unified_crop 契约一致：CPU
        mu = torch.randn(1, 6, 2, 3, device="cuda")
        cov = torch.randn(1, 6, 2, 3, 3, device="cuda")
        local = full_to_crop_norm(mu, roi, image_size=480)
        back = crop_to_full_norm(local, roi, image_size=480)
        cov_full = crop_to_full_cov(cov, roi, image_size=480)
        self.assertEqual(local.device.type, "cuda")
        self.assertEqual(cov_full.device.type, "cuda")
        torch.testing.assert_close(back, mu, atol=1e-4, rtol=1e-4)
