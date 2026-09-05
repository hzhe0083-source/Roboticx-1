"""Tests for observation-chunked joint visual-language encoding in dual tower."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn as nn

from va_compound.vision.dual_tower_batch import encode_dual_tower_batch
from va_compound.vision.dual_tower_fusion import MultiLayerDualTowerFusion


class TinyDINOBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class TinyDINO(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 16,
        num_blocks: int = 2,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_embed_conv = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.cls_token = nn.Parameter(torch.zeros(1, num_prefix_tokens, embed_dim))
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([TinyDINOBlock(embed_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.grad_checkpointing = False

    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
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
        out = hidden_states + self.linear(hidden_states)
        return (out,)


class TinyQwenTextModel(nn.Module):
    def __init__(self, vocab_size: int = 50, embed_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([TinyQwenDecoderLayer(embed_dim) for _ in range(num_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        x = self.embed(input_ids)
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        if return_dict:
            return SimpleNamespace(last_hidden_state=x)
        return x


class FakeVisionBackbone:
    def __init__(self, model: nn.Module, image_size: int = 8):
        self.model = model
        self.image_size = image_size


class FakeTextBackbone:
    def __init__(self, text_model: nn.Module):
        self.text_model = text_model

    def _tokenize_instructions(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(instructions)
        L = 4
        rows = [[(ord(c) % 30 + 1) for c in inst[:L]] for inst in instructions]
        for r in rows:
            if len(r) < L:
                r.extend([1] * (L - len(r)))
        ids = torch.tensor(rows, dtype=torch.long)
        mask = torch.ones(B, L, dtype=torch.long)
        return ids, mask


class VariableLengthFakeTextBackbone:
    def __init__(self, text_model: nn.Module):
        self.text_model = text_model

    def _tokenize_instructions(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [len(inst.split()) for inst in instructions]
        max_l = max(lengths)
        B = len(instructions)
        rows = []
        masks = []
        for inst in instructions:
            words = inst.split()
            r = [(ord(w[0]) % 30 + 1) for w in words]
            m = [1] * len(r)
            if len(r) < max_l:
                pad_cnt = max_l - len(r)
                r.extend([0] * pad_cnt)
                m.extend([0] * pad_cnt)
            rows.append(r)
            masks.append(m)
        ids = torch.tensor(rows, dtype=torch.long)
        mask = torch.tensor(masks, dtype=torch.long)
        return ids, mask


def _build_model_harness(seed: int = 42, variable_length_text: bool = False):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    dino = TinyDINO(embed_dim=16, num_blocks=2)
    qwen = TinyQwenTextModel(embed_dim=16, num_layers=2)
    fusion = MultiLayerDualTowerFusion(
        vision_dim=16,
        language_dim=16,
        hidden_dim=16,
        num_heads=2,
        num_pairs=2,
    )
    # Give non-zero weights to fusion so bidirectional cross-attention affects outputs
    nn.init.normal_(fusion.pairs[0].vision_out_proj.weight, std=0.1)
    nn.init.normal_(fusion.pairs[0].language_out_proj.weight, std=0.1)
    vision = FakeVisionBackbone(dino, image_size=8)
    text = (
        VariableLengthFakeTextBackbone(qwen)
        if variable_length_text
        else FakeTextBackbone(qwen)
    )
    return dino, qwen, fusion, vision, text, device


def test_observation_chunk_size_validation():
    dino, qwen, fusion, vision, text, device = _build_model_harness()
    B, T, V, H, W = 2, 2, 2, 8, 8
    frames = np.zeros((B, T, V, H, W, 3), dtype=np.uint8)
    instructions = ["pick cup", "push block"]

    for invalid in [0, -1, -5, "8", 2.5, True, False]:
        with pytest.raises(ValueError, match="observation_chunk_size must be a positive integer"):
            encode_dual_tower_batch(
                frames,
                instructions,
                vision,
                text,
                fusion,
                device,
                grid=2,
                observation_chunk_size=invalid,
            )


def test_chunk_size_none_exact_default():
    """Verify chunk_size=None produces exact identical results as omitting the argument."""
    dino, qwen, fusion, vision, text, device = _build_model_harness()
    B, T, V, H, W = 2, 3, 2, 8, 8
    frames = np.random.randint(0, 256, (B, T, V, H, W, 3), dtype=np.uint8)
    instructions = ["pick cup", "push block"]

    v_default, l_default, m_default = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2
    )
    v_none, l_none, m_none = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2, observation_chunk_size=None
    )

    assert torch.equal(v_default, v_none)
    assert torch.equal(l_default, l_none)
    assert torch.equal(m_default, m_none)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 10, 100])
def test_chunk_sizes_match_default_and_gradients(chunk_size):
    """Verify chunk1, batch (2), partial (3), >batch (4), and capped (>total_obs 10, 100)

    produce identical visual tokens, language hidden states, masks, and gradients.
    """
    B, T, V, H, W = 2, 3, 2, 8, 8
    frames = np.random.randint(0, 256, (B, T, V, H, W, 3), dtype=np.uint8)
    instructions = ["pick cup", "push block"]

    # 1. Reference with chunk_size=None
    dino_ref, qwen_ref, fusion_ref, v_ref, t_ref, device = _build_model_harness(seed=123)
    v_ref_out, l_ref_out, m_ref_out = encode_dual_tower_batch(
        frames, instructions, v_ref, t_ref, fusion_ref, device, grid=2, observation_chunk_size=None
    )
    loss_ref = (v_ref_out.sum() + l_ref_out.sum())
    loss_ref.backward()
    grads_ref = {
        "dino_conv": dino_ref.patch_embed_conv.weight.grad.clone(),
        "dino_b0": dino_ref.blocks[0].linear.weight.grad.clone(),
        "qwen_l0": qwen_ref.layers[0].linear.weight.grad.clone(),
        "fusion_p0_v": fusion_ref.pairs[0].vision_out_proj.weight.grad.clone(),
    }

    # 2. Test with chunk_size
    dino_test, qwen_test, fusion_test, v_test, t_test, device = _build_model_harness(seed=123)
    v_chunk_out, l_chunk_out, m_chunk_out = encode_dual_tower_batch(
        frames, instructions, v_test, t_test, fusion_test, device, grid=2, observation_chunk_size=chunk_size
    )

    assert v_chunk_out.shape == (B, T, V * 4, 16)
    assert l_chunk_out.shape == (B, T, 4, 16)
    assert m_chunk_out.shape == (B, T, 4)

    assert torch.allclose(v_chunk_out, v_ref_out, atol=1e-5)
    assert torch.allclose(l_chunk_out, l_ref_out, atol=1e-5)
    assert torch.equal(m_chunk_out, m_ref_out)

    loss_test = (v_chunk_out.sum() + l_chunk_out.sum())
    loss_test.backward()
    for name, g_ref in grads_ref.items():
        if name == "dino_conv":
            g = dino_test.patch_embed_conv.weight.grad
        elif name == "dino_b0":
            g = dino_test.blocks[0].linear.weight.grad
        elif name == "qwen_l0":
            g = qwen_test.layers[0].linear.weight.grad
        elif name == "fusion_p0_v":
            g = fusion_test.pairs[0].vision_out_proj.weight.grad
        assert torch.allclose(g, g_ref, atol=1e-5), f"Gradient mismatch for {name} with chunk_size={chunk_size}"


def test_consistent_tokenizer_padding_across_chunks():
    """Verify that when instructions have different lengths across batch items,

    different chunk sizes (e.g. chunk_size=1 where instructions are split vs chunk_size=B)
    preserve globally aligned tokenizer padding and masks.
    """
    B, T, V, H, W = 2, 2, 2, 8, 8
    frames = np.random.randint(0, 256, (B, T, V, H, W, 3), dtype=np.uint8)
    # 2 words vs 5 words
    instructions = ["short inst", "this is a longer instruction"]

    dino, qwen, fusion, vision, text, device = _build_model_harness(variable_length_text=True)

    # chunk_size=1 processes each observation independently:
    # chunk 0: "short inst" (len 2)
    # chunk 1: "this is a longer instruction" (len 5)
    # global padding must pad chunk 0 to len 5
    v_c1, l_c1, m_c1 = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2, observation_chunk_size=1
    )

    assert v_c1.shape == (B, T, V * 4, 16)
    assert l_c1.shape == (B, T, 5, 16)
    assert m_c1.shape == (B, T, 5)

    # b=0 has 2 valid tokens, 3 padded tokens
    for t in range(T):
        assert m_c1[0, t].sum().item() == 2
        assert torch.all(m_c1[0, t, :2])
        assert not torch.any(m_c1[0, t, 2:])
        # The positions beyond the single-item chunk were padded with 0.0 by global padding
        assert torch.all(l_c1[0, t, 2:] == 0.0)

    # b=1 has 5 valid tokens
    for t in range(T):
        assert m_c1[1, t].sum().item() == 5
        assert torch.all(m_c1[1, t])

    # Compare chunk_size=3 (partial chunk)
    v_c3, l_c3, m_c3 = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2, observation_chunk_size=3
    )
    assert v_c3.shape == (B, T, V * 4, 16)
    assert l_c3.shape == (B, T, 5, 16)
    assert m_c3.shape == (B, T, 5)

    # Both chunk executions yield the same global mask
    assert torch.equal(m_c1, m_c3)
    # Valid tokens for both items match across chunk configurations
    assert torch.allclose(l_c1[0, :, :2], l_c3[0, :, :2], atol=1e-5)
    assert torch.allclose(l_c1[1, :, :5], l_c3[1, :, :5], atol=1e-5)


def test_chunk_size_capping_beyond_total_observations():
    """Verify chunk_size larger than total_obs runs safely in a single chunk and returns expected shape."""
    dino, qwen, fusion, vision, text, device = _build_model_harness()
    B, T, V, H, W = 1, 2, 2, 8, 8
    frames = np.random.randint(0, 256, (B, T, V, H, W, 3), dtype=np.uint8)
    instructions = ["pick cup"]

    v, l, m = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2, observation_chunk_size=999
    )
    assert v.shape == (1, 2, V * 4, 16)
    assert l.shape == (1, 2, 4, 16)
    assert m.shape == (1, 2, 4)


def test_time_major_order_matches_sequence_of_calls():
    """Verify that when chunk_size=batch, each chunk corresponds exactly to time index t=0, 1, ..."""
    B, T, V, H, W = 3, 2, 2, 8, 8
    frames = np.random.randint(0, 256, (B, T, V, H, W, 3), dtype=np.uint8)
    instructions = ["inst 0", "inst 1", "inst 2"]

    dino, qwen, fusion, vision, text, device = _build_model_harness()

    v_batch, l_batch, m_batch = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2, observation_chunk_size=B
    )
    v_def, l_def, m_def = encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, grid=2, observation_chunk_size=None
    )

    assert torch.allclose(v_batch, v_def)
    assert torch.allclose(l_batch, l_def)
    assert torch.equal(m_batch, m_def)
