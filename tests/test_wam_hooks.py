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
