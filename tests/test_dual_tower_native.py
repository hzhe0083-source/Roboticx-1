"""Native timm and hybrid Qwen3.5 blocks, without pretrained downloads."""
from types import SimpleNamespace

import torch
from timm.models.vision_transformer import VisionTransformer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from va_compound.vision.dual_tower import encode_dual_tower
from va_compound.vision.dual_tower_fusion import MultiLayerDualTowerFusion


def test_native_hybrid_qwen_and_dino_fusion_backward():
    torch.manual_seed(71)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [1, 1, 2]},
    )
    qwen = Qwen3_5TextModel(config).eval()
    dino = VisionTransformer(img_size=16, patch_size=8, embed_dim=24,
                             depth=3, num_heads=4, num_classes=0).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    text = SimpleNamespace(text_model=qwen, _tokenize_instructions=lambda _: (ids, mask))
    fusion = MultiLayerDualTowerFusion(24, 32, 32, 4, num_pairs=2)
    images = torch.randn(1, 2, 3, 16, 16)
    with torch.no_grad():
        expected_v = dino.forward_features(images.flatten(0, 1))[:, 1:].reshape(1, 8, 24)
        expected_l = qwen(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    visual, language, _ = encode_dual_tower(images, ["pick"], SimpleNamespace(model=dino), text, fusion)
    torch.testing.assert_close(visual, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(language, expected_l, rtol=0, atol=0)
    loss = (visual * torch.randn_like(visual)).sum() + (language * torch.randn_like(language)).sum()
    loss.backward()
    for pair in fusion.pairs:
        for projection in (pair.vision_out_proj, pair.language_out_proj):
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
            assert torch.count_nonzero(projection.weight.grad)
    assert all(not block._forward_hooks for block in qwen.layers)
