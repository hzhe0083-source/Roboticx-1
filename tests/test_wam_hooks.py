"""CPU tests for the WAM sampling hooks and CLI mutex (Tasks 2 and 3).

- `wam_residual_fn=None` must be bit-identical to the pre-WAM behavior.
- A residual fn returning a constant c must shift the Euler trajectory by
  exactly c (velocity zeroed via monkeypatch, so the shift is exact).
- `--wam-joint` must be mutually exclusive with `--future-predict`/`--evsm`
  inside train.py's `validate_args`.

The policy hooks (Task 2) are already in `va_compound/model.py`; the
train.py mutex (Task 3) is implemented by a sibling agent and the test
skips until the flag exists.
"""

from __future__ import annotations

import pytest
import torch

try:
    from va_compound.model import VACompoundConfig, VACompoundPolicy
except ImportError:
    VACompoundConfig = None  # type: ignore[assignment]
    VACompoundPolicy = None  # type: ignore[assignment]

try:
    from train import parse_args, validate_args
except ImportError:
    parse_args = None  # type: ignore[assignment]
    validate_args = None  # type: ignore[assignment]


def _need_policy():
    if VACompoundConfig is None or VACompoundPolicy is None:
        pytest.skip("dependency not yet implemented: va_compound.model")
    return VACompoundConfig, VACompoundPolicy


def _tiny_config(config_cls):
    return config_cls(
        language_dim=24,
        vision_dim=20,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        action_horizon=5,
        action_dim=6,
        proprio_dim=9,
    )


def test_decode_actions_no_fn_unchanged() -> None:
    """wam_residual_fn=None (or omitted) must reproduce the legacy decode."""
    config_cls, policy_cls = _need_policy()
    torch.manual_seed(7)
    model = policy_cls(_tiny_config(config_cls)).eval()
    condition = torch.randn(2, model.config.action_horizon, model.config.hidden_dim)
    noise = torch.randn(2, model.config.action_horizon, model.config.action_dim)
    with torch.no_grad():
        legacy = model.decode_actions(condition, steps=4, noise=noise)
        explicit_none = model.decode_actions(
            condition, steps=4, noise=noise, wam_residual_fn=None
        )
        sampled_none = model.sample_actions(
            condition, steps=4, noise=noise, wam_residual_fn=None
        )
    torch.testing.assert_close(explicit_none, legacy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(sampled_none, legacy, rtol=0.0, atol=0.0)


def test_residual_fn_applied(monkeypatch) -> None:
    """A constant residual c shifts the Euler result by c.

    With `flow_velocity` stubbed to zero the base trajectory is x_0, so
    x_K = x_0 + K * (1/K) * c = x_0 + c — the hook contract
    `v = v + fn(action_condition, x_t, t_k)`.  Equality holds up to the
    fp32 rounding of four sequential `+=` accumulations.
    """
    config_cls, policy_cls = _need_policy()
    torch.manual_seed(11)
    config = _tiny_config(config_cls)
    model = policy_cls(config).eval()
    batch, horizon, dim = 2, config.action_horizon, config.action_dim
    condition = torch.randn(batch, horizon, config.hidden_dim)
    noise = torch.randn(batch, horizon, dim)
    residual = torch.randn(batch, horizon, dim)

    def zero_velocity(action_condition, _noisy_actions, _flow_time, semantic_context=None):
        del _noisy_actions, _flow_time, semantic_context
        return torch.zeros(action_condition.shape[0], horizon, dim)

    monkeypatch.setattr(model, "flow_velocity", zero_velocity)

    calls = []

    def residual_fn(action_condition, x_t, t_k):
        assert tuple(action_condition.shape) == (batch, horizon, config.hidden_dim)
        calls.append((x_t.clone(), t_k.clone()))
        return residual

    with torch.no_grad():
        plain = model.decode_actions(condition, steps=4, noise=noise)
        hooked = model.decode_actions(
            condition, steps=4, noise=noise, wam_residual_fn=residual_fn
        )

    torch.testing.assert_close(plain, noise, rtol=0.0, atol=0.0)
    torch.testing.assert_close(hooked, noise + residual, rtol=1e-6, atol=1e-6)

    assert len(calls) == 4
    for step, (_x_t, t_k) in enumerate(calls):
        assert t_k.shape == (batch,)
        torch.testing.assert_close(
            t_k, torch.full((batch,), step / 4.0), rtol=0.0, atol=0.0
        )


def test_wam_joint_mutex() -> None:
    if parse_args is None or validate_args is None:
        pytest.skip("dependency not yet implemented: train.py --wam-joint")
    args = parse_args([])
    if not hasattr(args, "wam_joint"):
        pytest.skip("dependency not yet implemented: train.py --wam-joint")

    # wam_joint alone must stay valid.
    args.wam_joint = True
    args.future_predict = False
    args.evsm = False
    validate_args(args)

    # wam_joint + future_predict must raise the documented mutex error.
    args.future_predict = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_args(args)


def test_pooling_implementations_equivalent() -> None:
    """eval 的 _wam_spatial16_from_h11（avg_pool2d）与 wam_cache 的
    wam_last_slice_pool（view+mean 块均值）语义必须一致：同一 [B,1152,768]
    输入的 6×6 块均值 → [B,16,768]（E7 审查：池化双实现等价）。"""
    try:
        from va_compound.wam_cache import wam_last_slice_pool
    except ImportError:
        pytest.skip("dependency not yet implemented: va_compound.wam_cache")
    try:
        from eval_metaworld import _wam_spatial16_from_h11
    except ImportError:
        pytest.skip("dependency not yet implemented: eval_metaworld")
    torch.manual_seed(20260812)
    h11 = torch.randn(2, 1152, 768)
    torch.testing.assert_close(
        _wam_spatial16_from_h11(h11),
        wam_last_slice_pool(h11),
        rtol=1e-5,
        atol=1e-6,
    )


def test_wam_forward_not_called_when_disabled(monkeypatch) -> None:
    """--wam off / --wam-alpha 0 时 eval 决议短路：wam=None → 两个 decode
    钩子不建 wam_residual_fn 闭包、wam.forward 永不被调（计数 stub 验证）。

    train 侧 rollout 钩子（train.py 的 rollout_policy）无 args 上下文且
    整体构造过重，不在本测试覆盖——train 侧由 fix-C agent 覆盖。
    """
    if VACompoundConfig is None:
        pytest.skip("dependency not yet implemented: va_compound.model")
    try:
        import eval_metaworld
        from eval_metaworld import _resolve_wam
    except ImportError:
        pytest.skip("dependency not yet implemented: eval_metaworld")
    from types import SimpleNamespace

    calls = {"n": 0}

    def residual_stub(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("wam 禁用时不得创建 wam_residual_fn 闭包")

    monkeypatch.setattr(eval_metaworld, "_make_wam_residual_fn", residual_stub)

    config = _tiny_config(VACompoundConfig)
    for wam_flag, alpha in (("off", 1.0), ("auto", 0.0), ("on", 0.0)):
        args = SimpleNamespace(wam=wam_flag, wam_alpha=alpha)
        wam = _resolve_wam({}, args, config, torch.device("cpu"))
        assert wam is None, f"--wam {wam_flag} / alpha {alpha} 必须短路为 None"
        # 决议 None 后两个 decode 钩子站点的契约：不建闭包、不传 wam_residual_fn。
        decode_kwargs = {}
        if wam is not None:
            decode_kwargs["wam_residual_fn"] = residual_stub()
        assert "wam_residual_fn" not in decode_kwargs

    # auto + 旧 checkpoint（无 wam_model 键）→ None，行为与旧版逐位一致。
    args = SimpleNamespace(wam="auto", wam_alpha=1.0)
    assert _resolve_wam({}, args, config, torch.device("cpu")) is None
    # on + 旧 checkpoint → 明确报错退出（拒绝静默回退）。
    args = SimpleNamespace(wam="on", wam_alpha=1.0)
    with pytest.raises(SystemExit):
        _resolve_wam({}, args, config, torch.device("cpu"))
    assert calls["n"] == 0


def test_resolve_wam_uses_saved_wam_config() -> None:
    """Checkpoint wam_config must win over VA hidden_dim / ACTION_HORIZON=8."""
    try:
        import dataclasses

        from eval_metaworld import _resolve_wam
        from va_compound.wam import JointWorldActionFlow, WAMConfig
    except ImportError:
        pytest.skip("dependency not yet implemented: eval/wam")
    from types import SimpleNamespace

    saved_cfg = WAMConfig(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_hidden=64,
        cond_dim=16,
        vision_dim=16,
        action_horizon=48,
        action_dim=4,
    )
    wam = JointWorldActionFlow(saved_cfg)
    ckpt = {
        "wam_model": wam.state_dict(),
        "wam_config": dataclasses.asdict(saved_cfg) | {"horizons": [6, 24, 48]},
    }
    va_config = SimpleNamespace(action_horizon=8, action_dim=4, hidden_dim=512)
    args = SimpleNamespace(wam="auto", wam_alpha=1.0)
    loaded = _resolve_wam(ckpt, args, va_config, torch.device("cpu"))
    assert loaded is not None
    assert loaded.config.num_layers == 2
    assert loaded.config.hidden_dim == 32
    assert loaded.config.action_horizon == 48
    assert loaded.config.horizons == (6, 24, 48)


def test_residual_fn_scene_path_is_not_zero(monkeypatch) -> None:
    """Scene tokens must follow x=(1-τ)ε, not zeros-every-step."""
    try:
        from types import SimpleNamespace

        from eval_metaworld import _make_wam_residual_fn
        from va_compound.wam import WAMSceneVelocity
    except ImportError:
        pytest.skip("dependency not yet implemented: eval/wam")

    captured = []

    class _StubWAM:
        def __call__(self, **kwargs):
            captured.append(
                {
                    "t": kwargs["flow_time"].detach().clone(),
                    "scene": kwargs["noisy_scene_latents"].detach().clone(),
                    "geo": kwargs["noisy_scene_geo"].detach().clone(),
                }
            )
            batch = kwargs["noisy_actions"].shape[0]
            dv = torch.zeros_like(kwargs["noisy_actions"])
            scene = WAMSceneVelocity(
                latent=torch.zeros(batch, 3, 16, 768),
                geo=torch.zeros(batch, 3, 2, 8),
            )
            return dv, scene

    memory = SimpleNamespace(layers=())
    spatial = torch.zeros(1, 16, 768)
    geo8 = torch.zeros(1, 8)
    fn = _make_wam_residual_fn(_StubWAM(), memory, spatial, geo8, 1.0)
    cond = torch.zeros(1, 8, 32)
    x_t = torch.zeros(1, 8, 4)
    torch.manual_seed(0)
    fn(cond, x_t, torch.tensor([0.0]))
    fn(cond, x_t, torch.tensor([0.5]))
    assert len(captured) == 2
    assert captured[0]["scene"].abs().sum() > 0
    assert captured[1]["scene"].abs().sum() > 0
    # Same ε: x(0.5) should be half of x(0)
    torch.testing.assert_close(captured[1]["scene"], 0.5 * captured[0]["scene"])
    torch.testing.assert_close(captured[1]["geo"], 0.5 * captured[0]["geo"])
