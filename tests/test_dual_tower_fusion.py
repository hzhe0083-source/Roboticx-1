"""Tests for DualTowerFusionPair and MultiLayerDualTowerFusion."""

import pytest
import torch
import torch.nn as nn
from va_compound.vision.dual_tower_fusion import DualTowerFusionPair, MultiLayerDualTowerFusion


def test_zero_init_identity_exactly():
    """Verify that at initialization, outputs are bit-for-bit / exactly equal to inputs."""
    torch.manual_seed(42)
    B, N, L = 2, 8, 5
    Dv, Dl, H, heads = 32, 48, 64, 4

    fusion = DualTowerFusionPair(
        vision_dim=Dv,
        language_dim=Dl,
        hidden_dim=H,
        num_heads=heads,
    )

    vision = torch.randn(B, N, Dv)
    language = torch.randn(B, L, Dl)
    language_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, False, False, False],
    ], dtype=torch.bool)

    new_vision, new_language = fusion(vision, language, language_mask)

    # Must be exact identity due to zero-init output projections
    assert torch.equal(new_vision, vision), "Vision output is not identical to input at initialization"
    assert torch.equal(new_language, language), "Language output is not identical to input at initialization"


def test_output_projections_get_gradient():
    """Verify gradients flow to zero-initialized output projections and other weights."""
    torch.manual_seed(42)
    B, N, L = 2, 6, 4
    Dv, Dl, H, heads = 16, 24, 32, 2

    fusion = DualTowerFusionPair(
        vision_dim=Dv,
        language_dim=Dl,
        hidden_dim=H,
        num_heads=heads,
    )

    vision = torch.randn(B, N, Dv, requires_grad=True)
    language = torch.randn(B, L, Dl, requires_grad=True)
    language_mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ], dtype=torch.bool)

    new_vision, new_language = fusion(vision, language, language_mask)
    loss = new_vision.sum() + new_language.sum()
    loss.backward()

    # Output projection weights and biases must have non-zero gradients
    assert fusion.vision_out_proj.weight.grad is not None
    assert fusion.vision_out_proj.weight.grad.abs().sum() > 0, "Vision out proj weight received zero grad"
    assert fusion.language_out_proj.weight.grad is not None
    assert fusion.language_out_proj.weight.grad.abs().sum() > 0, "Language out proj weight received zero grad"
    assert fusion.vision_out_proj.bias.grad is not None
    assert fusion.language_out_proj.bias.grad is not None

    # Input projections and attentions should also receive gradients
    # Note: Because output projection weights start at zero, dLoss/d(v_delta_h) = 0 initially,
    # so grad of inner weights like vision_proj starts at 0 if loss is only on final output.
    # But output projection itself receives delta_h^T * dLoss/d(output) != 0!
    # Let's verify that when output projections are perturbed, inner weights get gradients.
    with torch.no_grad():
        fusion.vision_out_proj.weight.fill_(0.01)
        fusion.language_out_proj.weight.fill_(0.01)
    fusion.zero_grad()
    new_v, new_l = fusion(vision, language, language_mask)
    (new_v.sum() + new_l.sum()).backward()
    assert fusion.vision_proj.weight.grad.abs().sum() > 0
    assert fusion.language_proj.weight.grad.abs().sum() > 0


def test_nonzero_projections_change_both_sides_simultaneous_snapshot():
    """Verify simultaneous snapshot property: both towers read the original input representation.

    If vision were updated first and language read updated vision (sequential),
    the language output would differ from simultaneous snapshot fusion.
    """
    torch.manual_seed(42)
    B, N, L = 2, 4, 3
    Dv, Dl, H, heads = 16, 16, 16, 2

    fusion = DualTowerFusionPair(
        vision_dim=Dv,
        language_dim=Dl,
        hidden_dim=H,
        num_heads=heads,
    )
    # Initialize output projections non-zero
    nn.init.normal_(fusion.vision_out_proj.weight, std=0.5)
    nn.init.normal_(fusion.language_out_proj.weight, std=0.5)

    vision = torch.randn(B, N, Dv)
    language = torch.randn(B, L, Dl)
    language_mask = torch.ones(B, L, dtype=torch.bool)

    new_vision, new_language = fusion(vision, language, language_mask)

    # Both sides must change
    assert not torch.equal(new_vision, vision)
    assert not torch.equal(new_language, language)

    # Compute manual reference of simultaneous snapshot:
    with torch.no_grad():
        v_h = fusion.vision_proj(fusion.vision_norm(vision))
        l_h = fusion.language_proj(fusion.language_norm(language))

        # Vision reading language
        v_delta_h, _ = fusion.v2l_attn(query=v_h, key=l_h, value=l_h, key_padding_mask=~language_mask, need_weights=False)
        ref_v = vision + fusion.vision_out_proj(v_delta_h)

        # Language reading vision (using original v_h, NOT updated vision)
        l_delta_h, _ = fusion.l2v_attn(query=l_h, key=v_h, value=v_h, need_weights=False)
        ref_l = language + fusion.language_out_proj(l_delta_h)

        assert torch.allclose(new_vision, ref_v, atol=1e-6)
        assert torch.allclose(new_language, ref_l, atol=1e-6)


def test_padding_safe_and_masked_positions_preserved():
    """Verify padded language tokens do not contaminate vision and masked positions are preserved."""
    torch.manual_seed(42)
    B, N, L = 1, 4, 4
    Dv, Dl, H, heads = 16, 16, 16, 2

    fusion = DualTowerFusionPair(
        vision_dim=Dv,
        language_dim=Dl,
        hidden_dim=H,
        num_heads=heads,
    )
    # Set non-zero output projections
    nn.init.normal_(fusion.vision_out_proj.weight, std=0.5)
    nn.init.normal_(fusion.language_out_proj.weight, std=0.5)

    vision = torch.randn(B, N, Dv)
    # Valid tokens at 0, 1; padded tokens at 2, 3
    language = torch.randn(B, L, Dl)
    language_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

    # Corrupt the padding positions with huge arbitrary values
    language_corrupted = language.clone()
    language_corrupted[:, 2:] = 9999.0

    new_v1, new_l1 = fusion(vision, language, language_mask)
    new_v2, new_l2 = fusion(vision, language_corrupted, language_mask)

    # 1. Vision should be completely unaffected by corrupted padding values
    assert torch.allclose(new_v1, new_v2, atol=1e-5), "Vision output changed when padding tokens changed"

    # 2. Masked language positions must strictly preserve their original input representation
    assert torch.equal(new_l1[:, 2:], language[:, 2:]), "Padding positions in language were modified"
    assert torch.equal(new_l2[:, 2:], language_corrupted[:, 2:]), "Padding positions in corrupted language were modified"

    # 3. Valid language positions should be updated
    assert not torch.equal(new_l1[:, :2], language[:, :2])


def test_all_padding_row_rejection():
    """All-padding language rows must raise a clear ValueError to avoid NaN."""
    fusion = DualTowerFusionPair(
        vision_dim=16,
        language_dim=16,
        hidden_dim=16,
        num_heads=2,
    )
    vision = torch.randn(2, 4, 16)
    language = torch.randn(2, 3, 16)
    # Row 1 has valid tokens, Row 0 is completely padded
    language_mask = torch.tensor([
        [False, False, False],
        [True, False, False],
    ], dtype=torch.bool)

    with pytest.raises(ValueError, match="All-padding language row"):
        fusion(vision, language, language_mask)


def test_shape_validation():
    """Verify input shape and type checks."""
    fusion = DualTowerFusionPair(
        vision_dim=16,
        language_dim=24,
        hidden_dim=32,
        num_heads=4,
    )

    # Mismatched hidden_dim and num_heads in constructor
    with pytest.raises(ValueError, match="divisible by num_heads"):
        DualTowerFusionPair(vision_dim=16, language_dim=24, hidden_dim=31, num_heads=4)

    # Mismatched vision_dim
    with pytest.raises(ValueError, match="vision feature dim mismatch"):
        fusion(torch.randn(2, 4, 12), torch.randn(2, 3, 24), torch.ones(2, 3, dtype=torch.bool))

    # Mismatched language_dim
    with pytest.raises(ValueError, match="language feature dim mismatch"):
        fusion(torch.randn(2, 4, 16), torch.randn(2, 3, 20), torch.ones(2, 3, dtype=torch.bool))

    # Batch dimension mismatch
    with pytest.raises(ValueError, match="Batch dimensions mismatch"):
        fusion(torch.randn(2, 4, 16), torch.randn(3, 3, 24), torch.ones(2, 3, dtype=torch.bool))

    # Sequence length mismatch in mask
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        fusion(torch.randn(2, 4, 16), torch.randn(2, 3, 24), torch.ones(2, 4, dtype=torch.bool))

    # Wrong mask dtype
    with pytest.raises(ValueError, match="torch.bool"):
        fusion(torch.randn(2, 4, 16), torch.randn(2, 3, 24), torch.ones(2, 3, dtype=torch.float32))


def test_multi_layer_dual_tower_fusion():
    """Verify MultiLayerDualTowerFusion owns ModuleList, num_pairs=6, bounds check."""
    multi_fusion = MultiLayerDualTowerFusion(
        vision_dim=16,
        language_dim=24,
        hidden_dim=32,
        num_heads=4,
        num_pairs=6,
    )
    assert len(multi_fusion) == 6
    assert isinstance(multi_fusion.pairs, nn.ModuleList)

    vision = torch.randn(2, 4, 16)
    language = torch.randn(2, 3, 24)
    language_mask = torch.ones(2, 3, dtype=torch.bool)

    # Valid indices
    for idx in range(6):
        nv, nl = multi_fusion.forward_pair(idx, vision, language, language_mask)
        assert torch.equal(nv, vision)
        assert torch.equal(nl, language)

    # Out of bounds indices
    with pytest.raises(IndexError, match="out of bounds"):
        multi_fusion.forward_pair(-1, vision, language, language_mask)

    with pytest.raises(IndexError, match="out of bounds"):
        multi_fusion.forward_pair(6, vision, language, language_mask)
