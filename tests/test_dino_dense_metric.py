"""DINO-metric 路径测试（2026-08-15 用户决策：DINO-main 接回 MT-VJ dense+metric）。

覆盖：
1. _dino_main_online_encode(return_dense=True) 的 dense evidence 形状/帧序
   （前 256 = d-2、后 256 = d；block5→key5、block11→key11）；
2. 训练/评测两帧 dense evidence 输出一致；
3. LanguageMetricField(grid=16, h_dim=1024) 前向形状 + spatial_bias 网格；
4. 旧 V-JEPA metric head 构造契约（无 grid 键）默认 grid=24 逐字节兼容；
5. DenseEvidenceProjector(vision_dim=1024) 512-token evidence + Δt 方向；
6. VACompoundConfig dino_dense_metric 组合校验；
7. _main_vision_config_kwargs 新 flag → dense_readout_mtvj/dino_dense_metric。
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch

from train import _validate_dino_roi_resume_contract
from va_compound.model import (
    DenseEvidenceProjector,
    VACompoundConfig,
    VACompoundPolicy,
    dense_coords,
)


def test_dino_roi_resume_contract_rejects_missing_changed_or_unidentified() -> None:
    artifact = Path("roi-v2.pt")
    identity = {
        "sha256": "a" * 64,
        "size_bytes": 123,
        "contract": "dino_metric_roi_task35_v2",
    }
    checkpoint = {
        "training_contract": {"dino_roi_enabled": True, "dino_roi_alpha": 0.75},
        "dino_roi_checkpoint_identity": identity,
    }
    with pytest.raises(ValueError, match="requires --dino-roi-checkpoint"):
        _validate_dino_roi_resume_contract(
            checkpoint, runtime_checkpoint=None, runtime_alpha=0.75
        )
    with pytest.raises(ValueError, match="must exactly match"):
        _validate_dino_roi_resume_contract(
            checkpoint, runtime_checkpoint=artifact, runtime_alpha=0.5
        )
    assert _validate_dino_roi_resume_contract(
        checkpoint, runtime_checkpoint=artifact, runtime_alpha=0.75
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_dino_roi_resume_contract(
            checkpoint,
            runtime_checkpoint=artifact,
            runtime_alpha=0.75,
            runtime_identity={**identity, "sha256": "b" * 64},
        )
    missing_identity = {
        "training_contract": {"dino_roi_enabled": True, "dino_roi_alpha": 0.75}
    }
    with pytest.raises(ValueError, match="lacks its identity"):
        _validate_dino_roi_resume_contract(
            missing_identity,
            runtime_checkpoint=artifact,
            runtime_alpha=0.75,
            runtime_identity=identity,
        )


class FakeDinoBackbone:
    """假冻结塔：block 特征按 patch 位置编码（内容无关）。"""

    image_size = 224
    feature_dim = 1024

    def forward_hierarchical_dense(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        n = int(images.shape[0])
        tokens = torch.zeros(n, 256, self.feature_dim)
        tokens[..., 0] = torch.arange(256).view(1, 256)
        return {5: tokens + 1.0, 11: tokens + 11.0}


def _frames(b=1, t=1, w=4):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(b, t, w, 224, 224, 3), dtype=np.uint8)


def _dino_dense_config(**overrides) -> VACompoundConfig:
    base = dict(
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        main_vision_image_size=224,
        main_vision_dim=1024,
        main_vision_grid=8,
        main_vision_frames=4,
        main_vision_tokens=256,
        dense_readout_mtvj=True,
        dino_dense_metric=True,
    )
    base.update(overrides)
    return VACompoundConfig(**base)


def test_dino_dense_evidence_shapes_and_frame_order() -> None:
    from train import _dino_main_online_encode

    backbone = FakeDinoBackbone()
    frames = _frames(2, 2, 4)  # B=2, T=2, W=4
    tokens, dense = _dino_main_online_encode(
        frames, backbone, torch.device("cpu"),
        encode_batch=3, grid=8, window=4, return_dense=True,
    )
    assert tuple(tokens.shape) == (2, 2, 256, 1024)
    assert sorted(dense) == [5, 11]
    for layer in (5, 11):
        assert tuple(dense[layer].shape) == (2, 2, 512, 1024)
    # 位置编码通道：前 256 个 token 是帧 d-2（w=2）→ channel0 = 0..255；
    # 后 256 个是帧 d（w=3）→ 同样 0..255（块内重排为帧内索引）。
    ev = dense[11][0, 0]
    first_half = ev[:256, 0]
    second_half = ev[256:, 0]
    assert torch.equal(first_half, second_half)
    assert first_half[0].item() == pytest.approx(11.0)  # 第 0 个 patch + 11
    assert first_half[255].item() == pytest.approx(255.0 + 11.0)
    # 两层内容区分（key5 = block11 特征）。
    assert torch.equal(dense[5][0, 0] - 1.0, dense[11][0, 0] - 11.0)


def test_train_eval_dense_evidence_equivalence() -> None:
    from train import _dino_main_online_encode
    from eval_metaworld import _main_vision_encode_window

    backbone = FakeDinoBackbone()
    frames = _frames(1, 1, 4)
    _, train_dense = _dino_main_online_encode(
        frames, backbone, torch.device("cpu"),
        encode_batch=2, grid=8, window=4, return_dense=True,
    )
    frame_list = [frames[0, 0, i] for i in range(4)]
    _, eval_dense = _main_vision_encode_window(
        frame_list, backbone, torch.device("cpu"),
        grid=8, window=4, return_dense=True,
    )
    for layer in (5, 11):
        assert torch.equal(train_dense[layer][0, 0], eval_dense[layer][0])


def test_metric_head_grid16_forward_shapes() -> None:
    from va_compound.metric_visual_head import LanguageMetricField

    head = LanguageMetricField(lang_dim=32, h_dim=1024, d_proj=64, grid=16)
    assert head.dense_tokens == 512
    assert tuple(head.spatial_bias.shape) == (4, 2, 16, 16)
    b = 2
    h5 = torch.randn(b, 512, 1024)
    h11 = torch.randn(b, 512, 1024)
    lang = torch.randn(b, 5, 32)
    mask = torch.ones(b, 5, dtype=torch.bool)
    coords = dense_coords(512)
    out = head(h5, h11, lang, mask, coords)
    assert tuple(out.p.shape) == (b, 4, 2)
    assert tuple(out.heatmap.shape) == (b, 4, 16, 16)
    assert tuple(out.visibility.shape) == (b, 4)
    assert tuple(out.offset_full.shape) == (b, 4, 512, 2)
    # 零初始化空间偏置 → 初始 p̂ 接近网格中心（mode_readout 下 5×5 窗口内）。
    head_mode = LanguageMetricField(
        lang_dim=32, h_dim=1024, d_proj=64, grid=16, mode_readout=True
    )
    out_mode = head_mode(h5, h11, lang, mask, coords)
    assert torch.all((out_mode.p >= 0.0) & (out_mode.p <= 1.0))


def test_metric_head_grid16_wrong_tokens_rejected() -> None:
    from va_compound.metric_visual_head import LanguageMetricField

    head = LanguageMetricField(lang_dim=32, h_dim=1024, d_proj=64, grid=16)
    with pytest.raises(ValueError, match="512"):
        head(torch.randn(1, 1152, 1024), torch.randn(1, 1152, 1024),
             torch.randn(1, 5, 32), torch.ones(1, 5, dtype=torch.bool),
             dense_coords(1152))


def test_legacy_metric_contract_defaults_to_grid24() -> None:
    from train import (
        _canonical_mtvj_metric_head_config,
        _mtvj_metric_head_constructor_config,
    )
    from va_compound.metric_visual_head import LanguageMetricField

    # 旧 checkpoint config（无 grid 键）→ 默认 24，行为逐字节不变。
    legacy = {
        "lang_dim": 2048, "h_dim": 768, "d_proj": 192, "n_roles": 4,
        "l2_norm": False, "learnable_temp": False, "temp_init": 10.0,
        "freeze_bias": False, "mode_readout": False,
    }
    canonical = _canonical_mtvj_metric_head_config(legacy, require_complete=True)
    assert canonical["grid"] == 24
    head = LanguageMetricField(**canonical)
    # 构造语义可完整回读（含 grid），供 checkpoint 保存。
    roundtrip = _mtvj_metric_head_constructor_config(head)
    assert roundtrip["grid"] == 24
    assert tuple(head.spatial_bias.shape) == (4, 2, 24, 24)


def test_dense_projector_dino_dim_and_dt_direction() -> None:
    proj = DenseEvidenceProjector(vision_dim=1024, hidden_dim=512)
    b = 2
    h5 = torch.randn(b, 512, 1024)
    h11 = torch.randn(b, 512, 1024)
    metric_tokens = torch.randn(b, 2, 512)
    out = proj({5: h5, 11: h11}, metric_tokens)
    assert tuple(out.d.shape) == (b, 512, 192)
    assert tuple(out.g.shape) == (b, 512, 192)
    assert tuple(out.t.shape) == (b, 512, 192)
    assert tuple(out.coord_k.shape) == (512, 512)
    # Δt = 后 256（d 帧）− 前 256（d-2 帧），按片复制回 512。
    expected_t = proj.proj_t(h11[:, 256:] - h11[:, :256])
    assert torch.allclose(out.t[:, :256], expected_t, atol=1e-6)
    assert torch.allclose(out.t[:, 256:], expected_t, atol=1e-6)


def test_dense_projector_rejects_wrong_dim() -> None:
    proj = DenseEvidenceProjector(vision_dim=1024, hidden_dim=512)
    with pytest.raises(ValueError, match="vision_dim"):
        proj({5: torch.randn(1, 512, 768), 11: torch.randn(1, 512, 768)}, None)


def test_config_dino_dense_metric_validation() -> None:
    _dino_dense_config()  # 合法组合不抛
    with pytest.raises(ValueError, match="main_vision_backbone"):
        _dino_dense_config(main_vision_backbone="vjepa")
    with pytest.raises(ValueError, match="dense_readout_mtvj"):
        _dino_dense_config(dense_readout_mtvj=False)


def test_geometry_injection_is_zero_init_then_causally_effective() -> None:
    config = _dino_dense_config(
        language_dim=12,
        vision_dim=1024,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
        main_vision_grid=2,
        main_vision_tokens=16,
        metric_geometry_inject=True,
    )
    torch.manual_seed(13)
    model = VACompoundPolicy(config).eval()
    assert torch.count_nonzero(model.geometry_projection.weight) == 0
    assert torch.count_nonzero(model.geometry_projection.bias) == 0
    vision = torch.randn(2, 16, 1024)
    proprio = torch.randn(2, 5)
    previous = torch.randn(2, 4)
    language = torch.randn(2, 3, 12)
    mask = torch.ones(2, 3, dtype=torch.bool)
    cache = model.build_language_cache(language, mask)
    zero = torch.zeros(2, 8)
    one = torch.ones(2, 8)
    at_init_zero = model.encode_condition(
        vision, proprio, previous, language_cache=cache, metric_g=zero
    )
    at_init_one = model.encode_condition(
        vision, proprio, previous, language_cache=cache, metric_g=one
    )
    assert torch.equal(at_init_zero, at_init_one)
    # Zero-init must still be trainable from action loss.
    at_init_one.square().mean().backward()
    assert model.geometry_projection.weight.grad is not None
    assert torch.count_nonzero(model.geometry_projection.weight.grad) > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        # A uniform hidden shift is intentionally avoided: LayerNorm would cancel
        # that negative control exactly even though the route is connected.
        values = torch.linspace(
            -0.05, 0.05, model.geometry_projection.weight.numel()
        ).reshape_as(model.geometry_projection.weight)
        model.geometry_projection.weight.copy_(values)
    learned_one = model.encode_condition(
        vision, proprio, previous, language_cache=cache, metric_g=one
    )
    assert not torch.allclose(at_init_zero, learned_one)
    with pytest.raises(ValueError, match="requires metric_g"):
        model.encode_condition(vision, proprio, previous, language_cache=cache)


def test_geometry_injection_disabled_preserves_old_state_dict_contract() -> None:
    model = VACompoundPolicy(_dino_dense_config())
    assert model.geometry_projection is None
    assert not any(key.startswith("geometry_projection.") for key in model.state_dict())


def test_optional_temporal_geometry_modules_preserve_matched_common_initialization() -> None:
    common = dict(
        language_dim=12,
        vision_dim=1024,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
        main_vision_grid=2,
        main_vision_tokens=16,
    )
    torch.manual_seed(20260817)
    control = VACompoundPolicy(_dino_dense_config(**common))
    torch.manual_seed(20260817)
    treatment = VACompoundPolicy(
        _dino_dense_config(
            **common,
            main_vision_temporal=True,
            metric_geometry_inject=True,
        )
    )
    treatment_state = treatment.state_dict()
    optional_prefixes = (
        "main_vision_frame_embedding.",
        "geometry_projection.",
    )
    for key, value in control.state_dict().items():
        assert not key.startswith(optional_prefixes)
        torch.testing.assert_close(value, treatment_state[key], rtol=0, atol=0)


def test_direct_and_fm_share_bit_identical_common_initialization() -> None:
    common = dict(
        language_dim=12,
        vision_dim=1024,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=6,
        action_dim=4,
        proprio_dim=5,
        main_vision_grid=2,
        main_vision_tokens=16,
        main_vision_temporal=True,
        metric_geometry_inject=True,
        flow_layers=2,
    )
    torch.manual_seed(20260818)
    flow = VACompoundPolicy(_dino_dense_config(**common, direct_head=False))
    torch.manual_seed(20260818)
    direct = VACompoundPolicy(_dino_dense_config(**common, direct_head=True))
    flow_state = flow.state_dict()
    direct_state = direct.state_dict()
    for key, value in flow_state.items():
        assert key in direct_state
        torch.testing.assert_close(value, direct_state[key], rtol=0, atol=0)
    assert any(key.startswith("direct_head.") for key in direct_state if key not in flow_state)


def test_feature_cache_read_matches_online_encode(tmp_path) -> None:
    """缓存读（含 8×8 池化 + 两帧 dense 组装）与在线编码逐位一致。"""
    import json
    import pickle

    from train import (
        DinoFeatureCache,
        _dino_main_encode_from_cache,
        _dino_main_online_encode,
    )

    n_frames = 12
    base = torch.zeros(n_frames, 256, 1024)
    base[..., 0] = torch.arange(256).view(1, 256)  # patch 位置编码
    block11 = (base + 1.0).numpy().astype(np.float16)
    block23 = (base + 11.0).numpy().astype(np.float16)
    np.save(tmp_path / "block11.npy", block11)
    np.save(tmp_path / "block23.npy", block23)
    index = {("peg-insert-side-v3", 0, i): i for i in range(n_frames)}
    with (tmp_path / "index.pkl").open("wb") as fh:
        pickle.dump(index, fh)
    feature_sha = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("block11.npy", "block23.npy")
    }
    with (tmp_path / "meta.json").open("w") as fh:
        json.dump(
            {
                "frames": n_frames,
                "dataset_sha256": "x" * 64,
                "feature_sha256": feature_sha,
                "feature_identity_contract": "sha256_full_npy_v1",
                "model_id": "vit_large_patch14_reg4_dinov2.lvd142m",
                "image_size": 224,
                "chunk": 32,
                "grid": 8,
                "window": 4,
            },
            fh,
        )

    cache = DinoFeatureCache(tmp_path)
    rows = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int64)  # [1,1,4]
    tokens_c, dense_c = _dino_main_encode_from_cache(
        rows, cache, torch.device("cpu"), grid=8, window=4, return_dense=True
    )
    # 在线路径用同一假特征塔（FakeDinoBackbone 与缓存内容一致）。
    frames = _frames(1, 1, 4)
    tokens_o, dense_o = _dino_main_online_encode(
        frames, FakeDinoBackbone(), torch.device("cpu"),
        encode_batch=2, grid=8, window=4, return_dense=True,
    )
    assert torch.equal(tokens_c, tokens_o)
    for layer in (5, 11):
        assert torch.equal(dense_c[layer], dense_o[layer])


def test_feature_cache_rejects_content_corruption(tmp_path) -> None:
    import json
    import pickle

    from train import DinoFeatureCache

    n_frames = 2
    for name in ("block11.npy", "block23.npy"):
        np.save(tmp_path / name, np.zeros((n_frames, 256, 8), dtype=np.float16))
    with (tmp_path / "index.pkl").open("wb") as fh:
        pickle.dump({("task", 0, index): index for index in range(n_frames)}, fh)
    feature_sha = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("block11.npy", "block23.npy")
    }
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "feature_sha256": feature_sha,
                "feature_identity_contract": "sha256_full_npy_v1",
            }
        )
    )
    with (tmp_path / "block11.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        old = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match="block11.npy SHA-256 mismatch"):
        DinoFeatureCache(tmp_path)


def test_feature_cache_rows_contract(tmp_path) -> None:
    """LongTrajFramesDataset 缓存模式：无 frames 键、rows 形状 [T,W]。"""
    import json
    import pickle

    import numpy as np
    import torch as _torch

    from va_compound.longtraj_frames import LongTrajFramesDataset

    n_frames = 16
    np.save(tmp_path / "block11.npy", np.zeros((n_frames, 256, 1024), dtype=np.float16))
    np.save(tmp_path / "block23.npy", np.zeros((n_frames, 256, 1024), dtype=np.float16))
    index = {("peg-insert-side-v3", 0, i): i for i in range(n_frames)}
    with (tmp_path / "index.pkl").open("wb") as fh:
        pickle.dump(index, fh)
    with (tmp_path / "meta.json").open("w") as fh:
        json.dump({"frames": n_frames, "model_id": "m", "image_size": 224,
                   "chunk": 32, "grid": 8, "window": 4}, fh)

    payload = {
        "actions": _torch.zeros(2, 4, 48, 4),
        "previous_action": _torch.zeros(2, 4, 4),
        "proprio": _torch.zeros(2, 4, 9),
        "language_hidden": _torch.zeros(2, 5, 2048),
        "instruction_id": _torch.zeros(2, dtype=_torch.long),
        "pair_id": _torch.zeros(2, dtype=_torch.long),
        "language_mask": _torch.ones(2, 5, dtype=_torch.bool),
        "frame_refs": [
            ("peg-insert-side-v3", 0, [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]),
            ("peg-insert-side-v3", 0, [[2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, 11]]),
        ],
    }
    data_path = tmp_path / "windows.pt"
    _torch.save(payload, data_path)
    dataset = LongTrajFramesDataset(
        data_path,
        min_sequence_length=4,
        feature_cache=tmp_path,
        include_frames=False,
    )
    item = dataset[1]
    assert "frames" not in item
    assert tuple(item["frame_cache_rows"].shape) == (4, 4)
    assert item["frame_cache_rows"][0].tolist() == [2, 3, 4, 5]


def test_grid16_cache_online_eval_equivalence(tmp_path) -> None:
    """grid=16（全 patch，无池化损失）：缓存读 = 在线编码 = 评测编码 逐位一致。"""
    import json
    import pickle

    from train import (
        DinoFeatureCache,
        _dino_main_encode_from_cache,
        _dino_main_online_encode,
    )
    from eval_metaworld import _main_vision_encode_window

    n_frames = 12
    base = torch.zeros(n_frames, 256, 1024)
    base[..., 0] = torch.arange(256).view(1, 256)
    block11 = (base + 1.0).numpy().astype(np.float16)
    block23 = (base + 11.0).numpy().astype(np.float16)
    np.save(tmp_path / "block11.npy", block11)
    np.save(tmp_path / "block23.npy", block23)
    index = {("peg-insert-side-v3", 0, i): i for i in range(n_frames)}
    with (tmp_path / "index.pkl").open("wb") as fh:
        pickle.dump(index, fh)
    feature_sha = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("block11.npy", "block23.npy")
    }
    with (tmp_path / "meta.json").open("w") as fh:
        json.dump({"frames": n_frames, "model_id": "m", "image_size": 224,
                   "chunk": 32, "grid": 8, "window": 4,
                   "feature_sha256": feature_sha,
                   "feature_identity_contract": "sha256_full_npy_v1"}, fh)
    cache = DinoFeatureCache(tmp_path)
    rows = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int64)
    tok_c, dense_c = _dino_main_encode_from_cache(
        rows, cache, torch.device("cpu"), grid=16, window=4, return_dense=True
    )
    assert tuple(tok_c.shape) == (1, 1, 1024, 1024)
    frames = _frames(1, 1, 4)
    tok_o, dense_o = _dino_main_online_encode(
        frames, FakeDinoBackbone(), torch.device("cpu"),
        encode_batch=2, grid=16, window=4, return_dense=True,
    )
    assert torch.equal(tok_c, tok_o)
    for layer in (5, 11):
        assert torch.equal(dense_c[layer], dense_o[layer])
    frame_list = [frames[0, 0, i] for i in range(4)]
    tok_e, dense_e = _main_vision_encode_window(
        frame_list, FakeDinoBackbone(), torch.device("cpu"),
        grid=16, window=4, return_dense=True,
    )
    assert torch.equal(tok_c[0, 0], tok_e[0])
    for layer in (5, 11):
        assert torch.equal(dense_c[layer][0, 0], dense_e[layer][0])


def test_rollout_dino_dense_keeps_full_grid_base_vision(monkeypatch) -> None:
    """Regression for the old train/eval mismatch: DINO dense is additive K/V."""
    import train as train_module

    config = _dino_dense_config(
        language_dim=12,
        vision_dim=1024,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
        main_vision_grid=8,
        main_vision_tokens=256,
        metric_geometry_inject=True,
    )
    model = VACompoundPolicy(config)
    observed = []
    original = model.encode_condition

    def record(vision_tokens, *args, **kwargs):
        observed.append(
            {
                "vision": vision_tokens.detach().clone(),
                "dense": {
                    key: value.detach().clone()
                    for key, value in kwargs["dense_evidence"].items()
                },
                "metric_g": kwargs["metric_g"].detach().clone(),
            }
        )
        return original(vision_tokens, *args, **kwargs)

    monkeypatch.setattr(model, "encode_condition", record)
    batch = {
        "actions": torch.randn(1, 1, 3, 4),
        "vision_tokens": torch.randn(1, 1, 256, 1024),
        "proprio": torch.randn(1, 1, 5),
        "previous_action": torch.randn(1, 1, 4),
        "language_hidden": torch.randn(1, 3, 12),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    dense = {
        5: torch.randn(1, 1, 512, 1024),
        11: torch.randn(1, 1, 512, 1024),
    }
    metric_g = torch.randn(1, 1, 8)
    train_module.rollout_policy(
        model,
        batch,
        torch.randn(1, 1, 3, 4),
        torch.rand(1, 1),
        dense_evidence=dense,
        metric_g=metric_g,
    )
    assert len(observed) == 1
    assert torch.equal(observed[0]["vision"], batch["vision_tokens"][:, 0])
    for layer in (5, 11):
        assert torch.equal(observed[0]["dense"][layer], dense[layer][:, 0])
    assert observed[0]["metric_g"].shape == (1, 8)
    assert torch.equal(observed[0]["metric_g"], metric_g[:, 0])


def test_rollout_legacy_vjepa_dense_still_uses_pool16(monkeypatch) -> None:
    import train as train_module

    config = VACompoundConfig(
        language_dim=12,
        vision_dim=768,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        action_horizon=3,
        action_dim=4,
        proprio_dim=5,
        dense_readout_mtvj=True,
        dino_dense_metric=False,
    )
    model = VACompoundPolicy(config)
    observed = []
    original = model.encode_condition

    def record(vision_tokens, *args, **kwargs):
        observed.append(vision_tokens.detach().clone())
        return original(vision_tokens, *args, **kwargs)

    monkeypatch.setattr(model, "encode_condition", record)
    batch = {
        "actions": torch.randn(1, 1, 3, 4),
        "vision_tokens": torch.randn(1, 1, 288, 768),
        "proprio": torch.randn(1, 1, 5),
        "previous_action": torch.randn(1, 1, 4),
        "language_hidden": torch.randn(1, 3, 12),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    dense = {
        5: torch.randn(1, 1, 1152, 768),
        11: torch.randn(1, 1, 1152, 768),
    }
    train_module.rollout_policy(
        model,
        batch,
        torch.randn(1, 1, 3, 4),
        torch.rand(1, 1),
        dense_evidence=dense,
    )
    expected = train_module.pool_mtvj_coarse_tokens(dense[11][:, 0])
    assert len(observed) == 1
    assert observed[0].shape == (1, 16, 768)
    assert torch.equal(observed[0], expected)


def test_dino_visual_aux_uses_true_480px_raw_renders(monkeypatch) -> None:
    import train as train_module
    from va_compound.metric_visual_head import LanguageMetricField

    seen = {}

    def fake_batch(task, rng, batch, include_raw_frames=False):
        seen["include_raw_frames"] = include_raw_frames
        return {
            "raw_frames": np.zeros((batch, 4, 480, 480, 3), dtype=np.uint8),
            "frames": np.zeros((batch, 4, 384, 384, 3), dtype=np.uint8),
            "keypoints": np.full((batch, 4, 2), 0.5, dtype=np.float32),
            "visibility": np.ones((batch, 4), dtype=np.float32),
        }

    monkeypatch.setattr("prepare_metaworld_metric.make_metric_batch", fake_batch)
    monkeypatch.setattr(
        "scripts.build_longtraj_features.ENV_TO_TASK",
        {"peg-insert-side-v3": "Insert a peg sideways"},
    )
    backbone = FakeDinoBackbone()
    head = LanguageMetricField(
        lang_dim=8,
        h_dim=1024,
        d_proj=8,
        n_roles=4,
        l2_norm=True,
        learnable_temp=True,
        mode_readout=True,
        grid=16,
    )
    lang_cache = {
        "Insert a peg sideways": (
            torch.zeros(1, 3, 8),
            torch.ones(1, 3, dtype=torch.bool),
        )
    }
    loss, parts = train_module._dino_visual_aux_loss(
        backbone,
        head,
        "peg-insert-side-v3",
        np.random.default_rng(0),
        2,
        lang_cache,
        torch.device("cpu"),
        1.0,
        0.5,
    )
    assert seen["include_raw_frames"] is True
    assert torch.isfinite(loss)
    assert np.isfinite(parts["rmse_px"])


def test_task35_precision_contract_requires_complete_stack(tmp_path) -> None:
    from train import parse_args, validate_args

    dino = tmp_path / "dino.safetensors"
    roi = tmp_path / "roi.pt"
    cache = tmp_path / "cache"
    dino.write_bytes(b"dino")
    roi.write_bytes(b"roi")
    cache.mkdir()
    for name in ("meta.json", "index.pkl", "block23.npy", "block11.npy"):
        (cache / name).write_bytes(b"cache")
    common = [
        "--task35-precision-contract",
        "--data",
        str(tmp_path / "data.pt"),
        "--single-task",
        "--task-sampling",
        "weighted",
        "--dino-main-vision",
        "--dino-dense-metric",
        "--dino-feature-cache",
        str(tmp_path / "cache"),
        "--main-vision-checkpoint",
        str(dino),
        "--main-vision-grid",
        "16",
        "--main-vision-frames",
        "4",
        "--main-vision-temporal",
        "--metric-geometry-inject",
        "--dino-roi-checkpoint",
        str(roi),
        "--dino-roi-alpha",
        "1",
        "--mtvj-train-metric-head",
        "--mtvj-train-relation",
        "--mtvj-visual-aux-every",
        "10",
        "--mtvj-visual-aux-batch",
        "8",
        "--va-attention-backend",
        "auto",
    ]
    args = parse_args(common)
    validate_args(args)
    validate_args(parse_args(common + ["--resume-exact", str(tmp_path / "exact.pt")]))
    with pytest.raises(ValueError, match="no ordinary resume"):
        validate_args(parse_args(common + ["--resume", str(tmp_path / "legacy.pt")]))
    with pytest.raises(ValueError, match="main-vision-temporal"):
        validate_args(parse_args([arg for arg in common if arg != "--main-vision-temporal"]))
    wrong_grid = list(common)
    grid_index = wrong_grid.index("--main-vision-grid") + 1
    wrong_grid[grid_index] = "8"
    # The explicit precision contract rejects this before data/model construction.
    with pytest.raises(ValueError, match="grid 16"):
        validate_args(parse_args(wrong_grid))


def test_main_vision_config_kwargs_dino_metric() -> None:
    from train import _main_vision_config_kwargs

    class Args:
        dino_main_vision = True
        dino_dense_metric = True
        main_vision_grid = 8
        main_vision_frames = 4
        main_vision_temporal = True
        main_vision_temporal_scale = 1.0
        metric_geometry_inject = True

    kwargs = _main_vision_config_kwargs(Args())
    assert kwargs["dense_readout_mtvj"] is True
    assert kwargs["dino_dense_metric"] is True
    assert kwargs["main_vision_dim"] == 1024
    assert kwargs["main_vision_temporal"] is True
    assert kwargs["metric_geometry_inject"] is True

    class ArgsNoMetric:
        dino_main_vision = True
        dino_dense_metric = False
        main_vision_grid = 8
        main_vision_frames = 4

    kwargs = _main_vision_config_kwargs(ArgsNoMetric())
    assert "dense_readout_mtvj" not in kwargs
    assert "dino_dense_metric" not in kwargs
