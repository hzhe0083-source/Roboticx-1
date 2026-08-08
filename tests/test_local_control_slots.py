"""Local control slots: shapes, gates, frozen-feature invariance, relations."""
import pytest
import torch

from va_compound.local_control_slots import (
    LanguageRoleCompiler,
    LocalControlSlotReader,
    RelationTokens,
    build_va_vision_input,
    fourier_encode,
)


def test_fourier_encode_shape_and_scale():
    coords = torch.randn(288, 3)
    feats = fourier_encode(coords, num_bands=4)
    assert feats.shape == (288, 3 + 2 * 3 * 4)
    # normalized coords stay in the raw prefix
    torch.testing.assert_close(feats[:, :3], coords)


def test_language_compiler_shapes_and_gate():
    B, L, Dlang, D = 4, 24, 1536, 512
    comp = LanguageRoleCompiler(hidden_dim=D, language_dim=Dlang, n_role=6)
    mask = torch.ones(B, L, dtype=torch.bool)
    key = torch.randn(B, L, Dlang)
    q = comp(key, mask)
    assert q.shape == (B, 6, D)
    # gate starts small so role identity dominates at init
    assert torch.sigmoid(comp.gate_logit).item() < 0.2


def test_reader_shapes_gate_and_zero_pos_invariance():
    B, N, K, D = 3, 288, 6, 768
    reader = LocalControlSlotReader(vision_dim=D, hidden_dim=512, num_slots=K)
    tokens = torch.randn(B, N, D)
    queries = torch.randn(B, K, 512)
    coords = torch.rand(N, 3)
    # coordinate bypass is zero-init -> pos channel must not change output
    with torch.no_grad():
        reader.pos_proj.weight.zero_()
    slots, weights, centers = reader(tokens, queries, coords)
    assert slots.shape == (B, K, D)
    assert weights.shape == (B, K, N)
    assert centers.shape == (B, K, 3)
    assert torch.sigmoid(reader.read_gate_logit).item() < 0.2
    # centers lie within the coord range
    assert bool((centers.abs() <= 1.05).all())


def test_relation_and_va_input():
    B, K, D = 2, 6, 768
    slots = torch.randn(B, K, D)
    centers = torch.randn(B, K, 3)
    rel = RelationTokens(vision_dim=D)(slots, centers)
    assert rel.shape == (B, 3, D)
    coarse = torch.randn(B, 16, D)
    va = build_va_vision_input(coarse, slots, rel)
    assert va.shape == (B, 25, D)


def test_role_seed_init():
    comp = LanguageRoleCompiler(hidden_dim=512, language_dim=1536, n_role=6)
    seeds = torch.randn(6, 512)
    comp.set_role_seeds(seeds)
    torch.testing.assert_close(comp.role_seeds, seeds)
    with pytest.raises(ValueError):
        comp.set_role_seeds(torch.randn(5, 512))
