"""Tests for dual tower execution (encode_dual_tower)."""

from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
from va_compound.vision.dual_tower import encode_dual_tower
from va_compound.vision.dual_tower_fusion import DualTowerFusionPair, MultiLayerDualTowerFusion


class TinyDINOBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class TinyDINO(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 16, num_blocks: int = 4, num_prefix_tokens: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        # [B*V, 3, H, W] -> [B*V, N, embed_dim]
        self.patch_embed_conv = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.cls_token = nn.Parameter(torch.zeros(1, num_prefix_tokens, embed_dim))
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([TinyDINOBlock(embed_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.grad_checkpointing = False

    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*V, 3, H, W] -> [B*V, C, H', W'] -> [B*V, N, C]
        feat = self.patch_embed_conv(x)
        return feat.flatten(2).transpose(1, 2)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat((cls, x), dim=1)

    def patch_drop(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TinyQwenDecoderLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # Returns either tensor or tuple (hidden_states, ...)
        out = hidden_states + self.linear(hidden_states)
        return (out,)


class TinyQwenTextModel(nn.Module):
    def __init__(self, vocab_size: int = 50, embed_dim: int = 24, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([TinyQwenDecoderLayer(embed_dim) for _ in range(num_layers)])
        self.gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, use_cache: bool = False, return_dict: bool = True):
        x = self.embed(input_ids)
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        if return_dict:
            return SimpleNamespace(last_hidden_state=x)
        return x


class FakeVisionBackbone:
    def __init__(self, model: nn.Module):
        self.model = model


class FakeTextBackbone:
    def __init__(self, text_model: nn.Module):
        self.text_model = text_model

    def _tokenize_instructions(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # Return toy input_ids and mask: [B, L]
        # Deterministic mapping based on instruction string characters
        B = len(instructions)
        L = 5
        rows = []
        for inst in instructions:
            row = [(ord(c) % 35 + 1) for c in inst[:L]]
            if len(row) < L:
                row = row + [1] * (L - len(row))
            rows.append(row)
        ids = torch.tensor(rows, dtype=torch.long)
        # Valid mask with at least 1 token valid
        mask = torch.ones(B, L, dtype=torch.long)
        mask[:, -1] = 0  # Last token padding
        return ids, mask


def test_zero_fusion_equivalence_standalone():
    """Verify zero-init fusion outputs exactly equal standalone non-fused backbones."""
    torch.manual_seed(42)
    B, V, H, W = 2, 2, 8, 8
    Dv, Dl, H_dim, heads = 16, 24, 32, 2
    count = 2

    dino = TinyDINO(embed_dim=Dv, num_blocks=4)
    qwen = TinyQwenTextModel(embed_dim=Dl, num_layers=4)
    fusion = MultiLayerDualTowerFusion(vision_dim=Dv, language_dim=Dl, hidden_dim=H_dim, num_heads=heads, num_pairs=count)

    v_bb = FakeVisionBackbone(dino)
    t_bb = FakeTextBackbone(qwen)

    images = torch.randn(B, V, 3, H, W)
    instructions = ["pick cup", "place bowl"]

    # 1. Standalone execution of DINO
    flat = images.flatten(0, 1)
    visual_ref = dino.norm_pre(dino.patch_drop(dino._pos_embed(dino.patch_embed(flat))))
    for block in dino.blocks:
        visual_ref = block(visual_ref)
    prefix = dino.num_prefix_tokens
    patches_ref = dino.norm(visual_ref)[:, prefix:]
    expected_visual = patches_ref.reshape(B, V * patches_ref.shape[1], Dv)

    # 2. Standalone execution of Qwen
    input_ids, mask = t_bb._tokenize_instructions(instructions)
    qwen_out = qwen(input_ids=input_ids, attention_mask=mask, return_dict=True)
    expected_lang = qwen_out.last_hidden_state

    # 3. Dual tower execution with zero-initialized fusion
    fused_v, fused_l, out_mask = encode_dual_tower(images, instructions, v_bb, t_bb, fusion)

    assert torch.allclose(fused_v, expected_visual, atol=1e-6), "Zero fusion visual differs from standalone"
    assert torch.allclose(fused_l, expected_lang, atol=1e-6), "Zero fusion language differs from standalone"
    assert torch.equal(out_mask, mask.bool())


def test_nonzero_pair0_fusion_affects_next_real_blocks():
    """Verify that when pair 0 fusion is non-zero, it actually changes the inputs to subsequent blocks."""
    torch.manual_seed(42)
    B, V, H, W = 2, 2, 8, 8
    Dv, Dl, H_dim, heads = 16, 24, 32, 2
    count = 2

    dino = TinyDINO(embed_dim=Dv, num_blocks=4)
    qwen = TinyQwenTextModel(embed_dim=Dl, num_layers=4)
    fusion = MultiLayerDualTowerFusion(vision_dim=Dv, language_dim=Dl, hidden_dim=H_dim, num_heads=heads, num_pairs=count)

    # Make pair 0 non-zero
    nn.init.normal_(fusion.pairs[0].vision_out_proj.weight, std=0.5)
    nn.init.normal_(fusion.pairs[0].language_out_proj.weight, std=0.5)

    v_bb = FakeVisionBackbone(dino)
    t_bb = FakeTextBackbone(qwen)

    images = torch.randn(B, V, 3, H, W)
    instructions = ["pick cup", "place bowl"]

    # Run standalone non-fused to compare
    flat = images.flatten(0, 1)
    visual_ref = dino.norm_pre(dino.patch_drop(dino._pos_embed(dino.patch_embed(flat))))
    for block in dino.blocks:
        visual_ref = block(visual_ref)
    patches_ref = dino.norm(visual_ref)[:, dino.num_prefix_tokens:]
    standalone_v = patches_ref.reshape(B, V * patches_ref.shape[1], Dv)

    input_ids, mask = t_bb._tokenize_instructions(instructions)
    standalone_l = qwen(input_ids=input_ids, attention_mask=mask, return_dict=True).last_hidden_state

    # Run dual tower with active fusion
    fused_v, fused_l, _ = encode_dual_tower(images, instructions, v_bb, t_bb, fusion)

    # Outputs must differ from standalone because pair 0 changes intermediate states fed to block 1
    assert not torch.allclose(fused_v, standalone_v, atol=1e-4)
    assert not torch.allclose(fused_l, standalone_l, atol=1e-4)


def test_hooks_removed_on_exception():
    """Verify that hooks on qwen layers are cleaned up even if an exception occurs during forward."""
    B, V, H, W = 1, 1, 8, 8
    Dv, Dl, H_dim, heads = 16, 24, 32, 2
    count = 2

    dino = TinyDINO(embed_dim=Dv, num_blocks=4)
    qwen = TinyQwenTextModel(embed_dim=Dl, num_layers=4)
    fusion = MultiLayerDualTowerFusion(vision_dim=Dv, language_dim=Dl, hidden_dim=H_dim, num_heads=heads, num_pairs=count)

    v_bb = FakeVisionBackbone(dino)
    t_bb = FakeTextBackbone(qwen)

    images = torch.randn(B, V, 3, H, W)
    instructions = ["test"]

    # Monkey patch qwen forward to raise an error midway
    original_forward = qwen.forward
    def failing_forward(*args, **kwargs):
        raise RuntimeError("Simulated failure in qwen forward")
    qwen.forward = failing_forward

    with pytest.raises(RuntimeError, match="Simulated failure in qwen forward"):
        encode_dual_tower(images, instructions, v_bb, t_bb, fusion)

    qwen.forward = original_forward

    # Check that all hook handles were removed from qwen layers
    for layer in qwen.layers:
        assert len(layer._forward_hooks) == 0, "Forward hook remained on qwen layer after exception"


def test_gradients_reach_fusion_and_unfrozen_tail_with_frozen_backbones():
    """Gradients must reach fusion parameters and unfrozen tail blocks even when backbones are frozen."""
    torch.manual_seed(42)
    B, V, H, W = 2, 2, 8, 8
    Dv, Dl, H_dim, heads = 16, 24, 32, 2
    count = 2

    dino = TinyDINO(embed_dim=Dv, num_blocks=4)
    qwen = TinyQwenTextModel(embed_dim=Dl, num_layers=4)
    fusion = MultiLayerDualTowerFusion(vision_dim=Dv, language_dim=Dl, hidden_dim=H_dim, num_heads=heads, num_pairs=count)

    # Freeze base blocks
    for p in dino.patch_embed_conv.parameters():
        p.requires_grad = False
    for block in dino.blocks[:-count]:
        for p in block.parameters():
            p.requires_grad = False
    for p in qwen.embed.parameters():
        p.requires_grad = False
    for layer in qwen.layers[:-count]:
        for p in layer.parameters():
            p.requires_grad = False

    # Keep tail blocks and fusion unfrozen
    for block in dino.blocks[-count:]:
        for p in block.parameters():
            p.requires_grad = True
    for layer in qwen.layers[-count:]:
        for p in layer.parameters():
            p.requires_grad = True

    # Perturb fusion output projection so initial gradient flows into fusion submodules
    for pair in fusion.pairs:
        nn.init.normal_(pair.vision_out_proj.weight, std=0.1)
        nn.init.normal_(pair.language_out_proj.weight, std=0.1)

    v_bb = FakeVisionBackbone(dino)
    t_bb = FakeTextBackbone(qwen)

    images = torch.randn(B, V, 3, H, W)
    instructions = ["a", "b"]

    fused_v, fused_l, _ = encode_dual_tower(images, instructions, v_bb, t_bb, fusion)
    loss = fused_v.sum() + fused_l.sum()
    loss.backward()

    # Verify frozen base has no grad
    for block in dino.blocks[:-count]:
        for p in block.parameters():
            assert p.grad is None
    for layer in qwen.layers[:-count]:
        for p in layer.parameters():
            assert p.grad is None

    # Verify unfrozen tail blocks have grad
    for block in dino.blocks[-count:]:
        for p in block.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0
    for layer in qwen.layers[-count:]:
        for p in layer.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0

    # Verify fusion parameters have grad
    for pair in fusion.pairs:
        assert pair.vision_out_proj.weight.grad is not None and pair.vision_out_proj.weight.grad.abs().sum() > 0
        assert pair.language_out_proj.weight.grad is not None and pair.language_out_proj.weight.grad.abs().sum() > 0


def test_per_view_shape_validation():
    """Verify correct handling of various view counts [B, V, 3, H, W] and shape errors."""
    Dv, Dl, H_dim, heads = 16, 24, 32, 2
    count = 2

    dino = TinyDINO(embed_dim=Dv, num_blocks=4)
    qwen = TinyQwenTextModel(embed_dim=Dl, num_layers=4)
    fusion = MultiLayerDualTowerFusion(vision_dim=Dv, language_dim=Dl, hidden_dim=H_dim, num_heads=heads, num_pairs=count)
    v_bb = FakeVisionBackbone(dino)
    t_bb = FakeTextBackbone(qwen)

    # 1. Multi-view check: B=3, V=4
    B, V, H, W = 3, 4, 8, 8
    images = torch.randn(B, V, 3, H, W)
    instructions = ["one", "two", "three"]
    fused_v, fused_l, mask = encode_dual_tower(images, instructions, v_bb, t_bb, fusion)

    # Patch embed kernel 4x4 on 8x8 gives 2x2 = 4 patches per view
    patches_per_view = (H // 4) * (W // 4)
    expected_vision_tokens = V * patches_per_view
    assert fused_v.shape == (B, expected_vision_tokens, Dv)
    assert fused_l.shape == (B, 5, Dl)
    assert mask.shape == (B, 5)

    # 2. Invalid image dimensions (not 5D)
    with pytest.raises(ValueError, match="images must be"):
        encode_dual_tower(torch.randn(3, 3, 8, 8), instructions, v_bb, t_bb, fusion)

    # 3. Invalid channels
    with pytest.raises(ValueError, match="images must be"):
        encode_dual_tower(torch.randn(3, 4, 4, 8, 8), instructions, v_bb, t_bb, fusion)

    # 4. Instruction count mismatch with batch size
    with pytest.raises(ValueError, match="one instruction is required per observation"):
        encode_dual_tower(images, ["only one instruction"], v_bb, t_bb, fusion)
