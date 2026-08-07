import unittest
from types import SimpleNamespace

import torch
from torch import nn

from va_compound.backbones import (
    QwenTextBackbone,
    VJEPA21Backbone,
    pool_flat_tokens,
    pool_spatial_tokens,
)


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.arange(5).repeat(batch, 1),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]).repeat(batch, 1),
        }


class FakeTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 12)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeVideoModel(nn.Module):
    """Emulates the V-JEPA 2.1 ViT-B/16 patch grid (t=2, h=24, w=24)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.patch_size = 16
        self.tubelet_size = 2
        self.num_frames = 4
        self.img_height = 384
        self.img_width = 384

    def forward(self, videos):
        batch = videos.shape[0]
        tokens = torch.arange(2 * 24 * 24 * 16, dtype=torch.float32).view(1, 1152, 16)
        return tokens.repeat(batch, 1, 1) * self.scale


class BackboneTests(unittest.TestCase):
    def test_qwen_wrapper_returns_token_states_and_boolean_mask(self):
        backbone = QwenTextBackbone(FakeTokenizer(), FakeTextModel())
        hidden, mask = backbone.encode(["pick red cup", "push blue cup"])
        self.assertEqual(hidden.shape, (2, 5, 12))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(hidden.requires_grad)
        self.assertFalse(torch.is_inference(hidden))

    def test_vjepa_wrapper_flat_pooling_bounds_token_count(self):
        backbone = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        videos = torch.randn(2, 4, 3, 16, 16)
        tokens = backbone(videos)
        self.assertEqual(tokens.shape, (2, 64, 16))
        self.assertEqual(backbone.patch_grid(), (24, 24))

    def test_spatial_pooling_keeps_time_mean_and_2d_neighbourhoods(self):
        # Token layout is [t, h, w] with t slowest; build a [B,2,24,24,1] grid.
        grid_values = torch.zeros(1, 2, 24, 24, 1)
        grid_values[:, 0] = 1.0  # t=0 frame all ones, t=1 frame all zeros
        tokens = grid_values.reshape(1, 1152, 1)
        pooled = pool_spatial_tokens(tokens, (24, 24), max_tokens=64)
        self.assertEqual(pooled.shape, (1, 64, 1))
        self.assertAlmostEqual(float(pooled[0, 0, 0]), 0.5)  # time-mean of the two frames

        # Only the 3x3 neighbourhood (h=0..2, w=0..2) carries signal after
        # the time mean; it must land exactly in output bucket (0, 0).
        signal = torch.zeros(1, 2, 24, 24, 1)
        signal[:, :, 0:3, 0:3] = 1.0
        pooled = pool_spatial_tokens(signal.reshape(1, 1152, 1), (24, 24), max_tokens=64)
        self.assertAlmostEqual(float(pooled[0, 0, 0]), 1.0)  # 3x3/9 averaged over time
        self.assertEqual(float(pooled[0, 1, 0]), 0.0)  # next h-bucket is clean
        self.assertEqual(float(pooled[0, 8, 0]), 0.0)  # next w-bucket is clean

    def test_flat_pooling_bucket_straddles_image_rows(self):
        # The legacy 1D pool averages 18 consecutive [t,h,w] tokens per bucket;
        # the same 3x3 signal spreads over three buckets instead of one.
        signal = torch.zeros(1, 2, 24, 24, 1)
        signal[:, :, 0:3, 0:3] = 1.0
        pooled = pool_flat_tokens(signal.reshape(1, 1152, 1), max_tokens=64)
        self.assertEqual(pooled.shape, (1, 64, 1))
        self.assertAlmostEqual(float(pooled[0, 0, 0]), 3.0 / 18.0)
        self.assertGreater(float(pooled[0, 1, 0]), 0.0)  # row boundary straddled

    def test_spatiotemporal_pooling_keeps_the_time_axis(self):
        backbone = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        videos = torch.randn(2, 4, 3, 16, 16)
        tokens = backbone(videos, pooling="spatiotemporal")
        self.assertEqual(tokens.shape, (2, 128, 16))

    def test_forward_variants_produce_both_poolings_in_one_forward(self):
        backbone = VJEPA21Backbone(FakeVideoModel(), max_tokens=64)
        videos = torch.randn(2, 4, 3, 16, 16)
        flat, spatial = backbone.forward_variants(videos)
        self.assertEqual(flat.shape, (2, 64, 16))
        self.assertEqual(spatial.shape, (2, 64, 16))
        self.assertFalse(torch.equal(flat, spatial))

    def test_spatial_pooling_rejects_grid_mismatch(self):
        tokens = torch.randn(2, 64, 16)
        with self.assertRaises(ValueError):
            pool_spatial_tokens(tokens, (24, 24), max_tokens=64)


if __name__ == "__main__":
    unittest.main()
