"""CPU tests for the E7 WAM JointWorldActionFlow module (Task 1 contract).

Covers the documented `va_compound.wam` interface: WAMConfig defaults,
the 55-65M parameter budget, zero-init action residual head, non-zero
scene heads, output shapes, the per-layer VA bypass mapping
`min(i * n_va // num_layers, n_va - 1)` for n_va=8 / 12 layers, and
bf16 finite forward.  All tests run on CPU (bf16 only on CUDA).

The module is implemented by a sibling agent; until it lands, every
test skips with "dependency not yet implemented".
"""

from __future__ import annotations

import pytest
import torch

try:
    from va_compound.wam import WAMConfig, JointWorldActionFlow, wam_config_from_state
except ImportError:
    WAMConfig = None  # type: ignore[assignment]
    JointWorldActionFlow = None  # type: ignore[assignment]
    wam_config_from_state = None  # type: ignore[assignment]

N_VA_LAYERS = 8
WAM_NUM_LAYERS = 12


def _need_wam():
    if JointWorldActionFlow is None or WAMConfig is None:
        pytest.skip("dependency not yet implemented: va_compound.wam")
    return JointWorldActionFlow, WAMConfig


@pytest.fixture(scope="module")
def wam_pair():
    """Default-config module built once; shared (read-mostly) by tests."""
    joint, cfg_cls = _need_wam()
    torch.manual_seed(0)
    cfg = cfg_cls()
    return joint(cfg), cfg


def _tiny_wam_config(cfg_cls) -> object:
    """12-layer WAM with tiny dims: keeps the VA-mapping test fast on CPU."""
    return cfg_cls(
        hidden_dim=32,
        num_layers=WAM_NUM_LAYERS,
        num_heads=4,
        ffn_hidden=64,
        cond_dim=32,
        vision_dim=32,
    )


def _forward(model, cfg, *, batch=2, va_layers=None, device="cpu"):
    """Feed the documented forward signature with synthetic tensors."""
    def rnd(*shape):
        return torch.randn(*shape, device=device)

    if va_layers is None:
        va_layers = tuple(
            rnd(batch, cfg.n_scene_tokens, cfg.hidden_dim) for _ in range(N_VA_LAYERS)
        )
    return model(
        action_condition=rnd(batch, cfg.action_horizon, cfg.hidden_dim),
        va_layers=va_layers,
        spatial_tokens=rnd(batch, cfg.n_scene_tokens, cfg.vision_dim),
        geo_tokens=rnd(batch, cfg.geo_dim),
        noisy_actions=rnd(batch, cfg.action_horizon, cfg.action_dim),
        noisy_scene_latents=rnd(batch, 3, cfg.n_scene_tokens, cfg.vision_dim),
        noisy_scene_geo=rnd(batch, 3, 2, cfg.geo_dim),
        flow_time=torch.rand(batch, device=device),
    )


def test_num_params_in_55_65M(wam_pair) -> None:
    model, _cfg = wam_pair
    num_params = (
        model.num_params() if callable(model.num_params) else model.num_params
    )
    assert 55_000_000 <= num_params <= 65_000_000
    assert num_params == sum(p.numel() for p in model.parameters())


def test_action_head_zero_init(wam_pair) -> None:
    model, cfg = wam_pair
    model.eval()
    with torch.no_grad():
        dv, _scene = _forward(model, cfg)
    assert dv.shape == (2, cfg.action_horizon, cfg.action_dim)
    assert torch.equal(dv, torch.zeros_like(dv))


def test_scene_heads_not_zero(wam_pair) -> None:
    model, cfg = wam_pair
    model.train()
    model.zero_grad()
    dv, scene = _forward(model, cfg)
    assert scene.latent.shape == (2, 3, cfg.n_scene_tokens, cfg.vision_dim)
    assert scene.geo.shape == (2, 3, 2, cfg.geo_dim)
    assert float(scene.latent.detach().abs().sum()) > 0.0
    assert float(scene.geo.detach().abs().sum()) > 0.0

    (dv.square().mean() + scene.latent.square().mean() + scene.geo.square().mean()).backward()

    # Per-horizon heads are the only Linears with out_features in
    # {vision_dim, geo_dim} and in_features == hidden_dim (action head is
    # [4, hidden] zero-init; token embedders have in_features 768/8/4).
    latent_heads = [
        p for p in model.parameters()
        if p.ndim == 2 and p.shape[0] == cfg.vision_dim and p.shape[1] == cfg.hidden_dim
    ]
    geo_heads = [
        p for p in model.parameters()
        if p.ndim == 2 and p.shape[0] == cfg.geo_dim and p.shape[1] == cfg.hidden_dim
    ]
    assert len(latent_heads) >= 1
    assert len(geo_heads) >= 1
    for parameter in latent_heads + geo_heads:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_forward_shapes(wam_pair) -> None:
    model, cfg = wam_pair
    model.eval()
    with torch.no_grad():
        dv, scene = _forward(model, cfg, batch=3)
    assert dv.shape == (3, cfg.action_horizon, cfg.action_dim)
    assert scene.latent.shape == (3, 3, cfg.n_scene_tokens, cfg.vision_dim)
    assert scene.geo.shape == (3, 3, 2, cfg.geo_dim)
    assert torch.isfinite(dv).all()
    assert torch.isfinite(scene.latent).all()
    assert torch.isfinite(scene.geo).all()


class _RecordingTuple(tuple):
    """Records integer reads so we can observe the per-layer VA index."""

    def __new__(cls, items):
        obj = super().__new__(cls, items)
        obj.reads = []
        return obj

    def __getitem__(self, key):
        if isinstance(key, int):
            self.reads.append(key)
        return super().__getitem__(key)


def test_va_layer_mapping() -> None:
    joint, cfg_cls = _need_wam()
    torch.manual_seed(0)
    cfg = _tiny_wam_config(cfg_cls)
    model = joint(cfg).eval()
    va = _RecordingTuple(torch.randn(1, cfg.n_scene_tokens, cfg.hidden_dim) for _ in range(N_VA_LAYERS))
    with torch.no_grad():
        _forward(model, cfg, batch=1, va_layers=va)

    reads = va.reads
    if reads:
        # Order must follow the documented bypass schedule; consecutive
        # duplicates (K and V reads, device checks) collapse away.
        collapsed = [reads[0]] + [v for prev, v in zip(reads, reads[1:]) if v != prev]
        assert collapsed == list(range(N_VA_LAYERS)), (
            f"per-layer VA reads {reads} do not follow min(i*{N_VA_LAYERS}//{WAM_NUM_LAYERS}, "
            f"{N_VA_LAYERS - 1}) mapping"
        )
        assert len(reads) >= WAM_NUM_LAYERS
    else:
        # The implementation copied/iterated the tuple before indexing, so
        # reads are unobservable.  Open the zero-init gates and check that
        # every va layer influences the output (surjectivity of the mapping).
        with torch.no_grad():
            for parameter in model.parameters():
                if float(parameter.data.abs().sum()) == 0.0:
                    parameter.copy_(torch.randn_like(parameter) * 0.02)
        va_tensors = [
            torch.randn(1, cfg.n_scene_tokens, cfg.hidden_dim, requires_grad=True)
            for _ in range(N_VA_LAYERS)
        ]
        _dv, scene = _forward(model, cfg, batch=1, va_layers=tuple(va_tensors))
        (scene.latent.square().mean() + scene.geo.square().mean()).backward()
        influenced = [
            k for k, tensor in enumerate(va_tensors)
            if tensor.grad is not None and float(tensor.grad.abs().sum()) > 0.0
        ]
        assert influenced == list(range(N_VA_LAYERS)), (
            f"va layers without any influence on the output: "
            f"{set(range(N_VA_LAYERS)) - set(influenced)}"
        )


def test_bf16_finite() -> None:
    joint, cfg_cls = _need_wam()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = cfg_cls()
    model = joint(cfg).to(device).eval()
    # On CPU this degrades to fp32 (bf16 matmuls need CUDA); per plan the
    # CPU path just proves fp32 finiteness of the same forward.
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        dv, scene = _forward(model, cfg, device=device)
    assert torch.isfinite(dv.float()).all()
    assert torch.isfinite(scene.latent.float()).all()
    assert torch.isfinite(scene.geo.float()).all()


def test_wam_config_from_state_restores_saved_fields() -> None:
    _need_wam()
    if wam_config_from_state is None:
        pytest.skip("dependency not yet implemented: wam_config_from_state")
    saved = {
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_hidden": 64,
        "horizons": [6, 24, 48],
        "action_horizon": 8,
        "unknown_future_key": "ignore-me",
    }
    cfg = wam_config_from_state(saved, hidden_dim=512, action_dim=4)
    assert cfg.hidden_dim == 32
    assert cfg.num_layers == 2
    assert cfg.action_horizon == 8
    assert cfg.horizons == (6, 24, 48)
    assert cfg.action_dim == 4
    fallback = wam_config_from_state(None, hidden_dim=256, action_dim=7)
    assert fallback.hidden_dim == 256
    assert fallback.action_dim == 7
    assert fallback.num_layers == 12
