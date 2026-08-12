"""CPU regression tests for the MT-VJ train/eval runtime contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval_metaworld import (
    _load_mtvj_metric_checkpoint,
    _mtvj_visibility_gated_positions as _eval_visibility_gated_positions,
)
from scripts.build_longtraj_features import ENV_TO_TASK
from train import (
    _load_mtvj_metric_checkpoint as _load_train_mtvj_metric_checkpoint,
    _mtvj_metric_checkpoint_identity,
    _mtvj_metric_head_constructor_config,
    _mtvj_metric_head_optimizer_group,
    _mtvj_metric_deltas,
    _mtvj_online_encode,
    _mtvj_relation_optimizer_group,
    _mtvj_relation_tokens,
    _mtvj_visibility_gated_positions as _train_visibility_gated_positions,
    _restore_mtvj_policy_modules,
    parse_args,
    save_checkpoint,
    validate_args,
)
from va_compound.backbones import pool_mtvj_coarse_tokens
from va_compound.metric_visual_head import LanguageMetricField, RelationStateEncoder


def test_mtvj_h11_pool16_matches_historical_training_formula() -> None:
    h11 = torch.arange(2 * 1152 * 3, dtype=torch.float32).reshape(2, 1152, 3)
    expected = h11.reshape(2, 16, 72, 3).mean(dim=2)

    pooled = pool_mtvj_coarse_tokens(h11)

    torch.testing.assert_close(pooled, expected, rtol=0.0, atol=0.0)
    assert pooled.shape == (2, 16, 3)


def test_mtvj_h11_pool16_rejects_unequal_bins() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        pool_mtvj_coarse_tokens(torch.zeros(1, 1151, 4))


def test_mtvj_metric_delta_has_zero_first_decision() -> None:
    g = torch.tensor(
        [
            [[2.0, -1.0], [3.5, 4.0], [-2.0, 8.0]],
            [[7.0, 9.0], [6.0, 8.0], [6.5, 7.0]],
        ]
    )

    nu = _mtvj_metric_deltas(g)

    torch.testing.assert_close(nu[:, 0], torch.zeros_like(g[:, 0]))
    torch.testing.assert_close(nu[:, 1:], g[:, 1:] - g[:, :-1])


def test_mtvj_visibility_gate_nulls_invisible_coordinates_and_train_eval_match() -> None:
    p = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.4], [99.0, -99.0], [-7.0, 8.0]]]
    )
    out = SimpleNamespace(
        p=p, visibility=torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    )
    expected = torch.tensor([[0.1, 0.2, 0.15, 0.2, 0.0, 0.0, 0.0, 0.0]])
    train_g = _train_visibility_gated_positions(out)
    eval_g = _eval_visibility_gated_positions(out)
    torch.testing.assert_close(train_g, expected)
    torch.testing.assert_close(eval_g, expected)

    changed = SimpleNamespace(p=p.clone(), visibility=out.visibility)
    changed.p[:, 2:] = torch.tensor([[[1e6, -1e6], [-3e6, 4e6]]])
    torch.testing.assert_close(
        _train_visibility_gated_positions(changed), train_g, rtol=0.0, atol=0.0
    )


def test_mtvj_relation_tokens_backprop_only_through_action_path() -> None:
    relation = RelationStateEncoder(state_dim=8, d_model=16)
    relation.recon.requires_grad_(False)
    g = torch.randn(2, 4, 8)

    tokens = _mtvj_relation_tokens(g, relation)
    tokens.square().mean().backward()

    assert tokens.requires_grad
    assert relation.g_proj.weight.grad is not None
    assert relation.nu_proj.weight.grad is not None
    assert relation.norm.weight.grad is not None
    assert relation.recon.weight.grad is None
    assert g.grad is None


def test_mtvj_joint_step0_is_bitwise_identical_to_frozen_relation() -> None:
    torch.manual_seed(7)
    frozen = RelationStateEncoder(state_dim=8, d_model=16).eval()
    trainable = RelationStateEncoder(state_dim=8, d_model=16).train()
    trainable.load_state_dict(frozen.state_dict(), strict=True)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    trainable.recon.requires_grad_(False)
    g = torch.randn(3, 4, 8)

    with torch.no_grad():
        expected = _mtvj_relation_tokens(g, frozen)
    actual = _mtvj_relation_tokens(g, trainable)

    assert torch.equal(actual.detach(), expected)
    assert actual.requires_grad


def test_mtvj_online_encode_keeps_backbone_head_frozen_and_relation_in_graph() -> None:
    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward_hierarchical_dense(self, inputs, out_layers):
            values = self.scale * torch.ones(inputs.shape[0], 1152, 4)
            return {5: values, 11: values + 1.0}

    class FakeMetricHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.25))

        def forward(self, h5, h11, language_hidden, language_mask, coords):
            row = torch.arange(h5.shape[0], dtype=h5.dtype).view(-1, 1, 1)
            return SimpleNamespace(
                p=(self.anchor + 0.01 * row).expand(-1, 4, 2),
                visibility=torch.ones(h5.shape[0], 4),
            )

    backbone = FakeBackbone()
    metric_head = FakeMetricHead()
    relation = RelationStateEncoder(state_dim=8, d_model=16)
    relation.recon.requires_grad_(False)
    frames = torch.zeros(1, 2, 4, 32, 32, 3, dtype=torch.uint8)
    batch = {
        "language_hidden": torch.zeros(1, 3, 8),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    dense, tokens = _mtvj_online_encode(
        frames,
        backbone,
        metric_head,
        relation,
        batch,
        torch.device("cpu"),
    )
    assert tokens is not None and tokens.requires_grad
    weights = torch.linspace(-1.0, 1.0, tokens.shape[-1])
    (tokens * weights).sum().backward()

    assert dense[5].grad_fn is None
    assert backbone.scale.grad is None
    assert metric_head.anchor.grad is None
    assert relation.g_proj.weight.grad is not None
    assert relation.g_proj.weight.grad.abs().sum() > 0
    assert relation.nu_proj.weight.grad is not None
    assert relation.nu_proj.weight.grad.abs().sum() > 0


def test_mtvj_online_encode_can_backprop_into_metric_head_but_not_backbone() -> None:
    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward_hierarchical_dense(self, inputs, out_layers):
            values = self.scale * torch.ones(inputs.shape[0], 1152, 4)
            return {5: values, 11: values + 1.0}

    class FakeMetricHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.25))

        def forward(self, h5, h11, language_hidden, language_mask, coords):
            row = torch.arange(h5.shape[0], dtype=h5.dtype).view(-1, 1, 1)
            return SimpleNamespace(
                p=(self.anchor + 0.01 * row).expand(-1, 4, 2),
                visibility=torch.sigmoid(self.anchor).expand(h5.shape[0], 4),
            )

    backbone = FakeBackbone()
    metric_head = FakeMetricHead()
    relation = RelationStateEncoder(state_dim=8, d_model=16)
    for parameter in relation.parameters():
        parameter.requires_grad_(False)
    frames = torch.zeros(1, 2, 4, 32, 32, 3, dtype=torch.uint8)
    batch = {
        "language_hidden": torch.zeros(1, 3, 8),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    _, tokens = _mtvj_online_encode(
        frames,
        backbone,
        metric_head,
        relation,
        batch,
        torch.device("cpu"),
        train_metric_head=True,
    )
    assert tokens is not None
    weights = torch.linspace(-1.0, 1.0, tokens.shape[-1])
    (tokens * weights).sum().backward()

    assert metric_head.anchor.grad is not None
    assert metric_head.anchor.grad.abs() > 0
    assert backbone.scale.grad is None


def test_mtvj_joint_relation_argument_contract() -> None:
    with pytest.raises(ValueError, match="requires --dense-readout-mtvj"):
        validate_args(parse_args(["--mtvj-train-relation"]))

    missing_resume = parse_args(
        [
            "--dense-readout-mtvj",
            "--metric-visual-checkpoint",
            "metric.pt",
            "--mtvj-train-relation",
        ]
    )
    with pytest.raises(ValueError, match="requires --resume"):
        validate_args(missing_resume)

    valid = parse_args(
        [
            "--dense-readout-mtvj",
            "--metric-visual-checkpoint",
            "metric.pt",
            "--mtvj-train-relation",
            "--lr-mtvj-relation",
            "2e-5",
            "--resume",
            "policy.pt",
        ]
    )
    validate_args(valid)

    valid.sam_rho = 0.05
    with pytest.raises(ValueError, match="forbids --sam-rho"):
        validate_args(valid)


def test_mtvj_joint_metric_head_argument_contract() -> None:
    with pytest.raises(ValueError, match="requires --dense-readout-mtvj"):
        validate_args(parse_args(["--mtvj-train-metric-head"]))

    valid = parse_args(
        [
            "--dense-readout-mtvj",
            "--metric-visual-checkpoint",
            "metric.pt",
            "--mtvj-train-metric-head",
            "--lr-mtvj-metric-head",
            "1e-6",
            "--resume",
            "policy.pt",
        ]
    )
    validate_args(valid)

    valid.lr_mtvj_metric_head = 0.0
    with pytest.raises(ValueError, match="lr-mtvj-metric-head"):
        validate_args(valid)


def test_main_checkpoint_roundtrips_mtvj_relation_encoder(tmp_path) -> None:
    args = parse_args([])
    args.save = tmp_path / "policy.pt"
    config = SimpleNamespace(hidden_dim=16)
    model = nn.Linear(3, 2)
    relation = RelationStateEncoder(state_dim=8, d_model=16)

    save_checkpoint(
        args,
        config,
        model,
        None,
        relation_encoder=relation,
    )
    saved = torch.load(args.save, map_location="cpu", weights_only=True)

    assert saved["training_contract"]["metric_contract_version"] == 2
    assert saved["training_contract"]["metric_state_source"] == "p_flat"
    assert saved["training_contract"]["metric_relation_joint_trained"] is False
    for key, value in relation.state_dict().items():
        torch.testing.assert_close(
            saved["mtvj_relation_encoder"][key], value, rtol=0.0, atol=0.0
        )


def test_main_checkpoint_roundtrips_and_strictly_restores_metric_head(tmp_path) -> None:
    args = parse_args(
        ["--mtvj-train-metric-head", "--lr-mtvj-metric-head", "1e-6"]
    )
    args.save = tmp_path / "policy.pt"
    config = SimpleNamespace(hidden_dim=16)
    model = nn.Linear(3, 2)
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(
        metric_path,
        l2_norm=True,
        learnable_temp=True,
        temp_init=7.5,
        freeze_bias=True,
        mode_readout=True,
    )
    metric_head, relation = _load_train_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        config,
    )
    with torch.no_grad():
        for parameter in metric_head.parameters():
            parameter.fill_(0.125)

    save_checkpoint(
        args,
        config,
        model,
        None,
        relation_encoder=relation,
        metric_head=metric_head,
    )
    saved = torch.load(args.save, map_location="cpu", weights_only=True)
    assert saved["training_contract"]["metric_head_checkpointed"] is True
    assert saved["training_contract"]["metric_head_joint_trained"] is True
    assert set(saved["mtvj_metric_head_config"]) == {
        "lang_dim",
        "h_dim",
        "d_proj",
        "n_roles",
        "l2_norm",
        "learnable_temp",
        "temp_init",
        "freeze_bias",
        "mode_readout",
    }
    assert saved["mtvj_metric_checkpoint_identity"]["sha256"]
    assert saved["mtvj_metric_head_config"] == {
        "lang_dim": 8,
        "h_dim": 4,
        "d_proj": 2,
        "n_roles": 4,
        "l2_norm": True,
        "learnable_temp": True,
        "temp_init": 7.5,
        "freeze_bias": True,
        "mode_readout": True,
    }

    restored = LanguageMetricField(**saved["mtvj_metric_head_config"])
    _restore_mtvj_policy_modules(
        saved,
        relation_encoder=None,
        metric_head=restored,
        train_relation=False,
    )
    for key, value in metric_head.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0.0, atol=0.0)

    broken = dict(saved)
    broken["mtvj_metric_head"] = dict(saved["mtvj_metric_head"])
    broken["mtvj_metric_head"].pop(next(iter(broken["mtvj_metric_head"])))
    with pytest.raises(RuntimeError):
        _restore_mtvj_policy_modules(
            broken,
            relation_encoder=None,
            metric_head=restored,
            train_relation=False,
        )


def _write_external_metric_checkpoint(
    path,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
    **overrides,
) -> None:
    if tasks is None:
        tasks = tuple(ENV_TO_TASK)
    metric_config = {
        "lang_dim": 8,
        "h_dim": 4,
        "d_proj": 2,
        "n_roles": 4,
        "l2_norm": False,
        "learnable_temp": False,
        "temp_init": 10.0,
        "freeze_bias": False,
        "mode_readout": False,
    }
    metric_config.update(overrides)
    constructor_config = {
        key: metric_config[key]
        for key in (
            "lang_dim", "h_dim", "d_proj", "n_roles", "l2_norm",
            "learnable_temp", "temp_init", "freeze_bias", "mode_readout",
        )
    }
    metric_head = LanguageMetricField(**constructor_config)
    legacy_relation = RelationStateEncoder(state_dim=6, d_model=16)
    torch.save(
        {
            "contract": "mt_vj_metric_field_v1",
            "config": {
                **metric_config,
                "d_model": 16,
                "tasks": list(tasks),
                "loc_only": metric_config.get("loc_only", False),
                "relation_encoder_trained": metric_config.get(
                    "relation_encoder_trained", True
                ),
                "training_state_version": metric_config.get(
                    "training_state_version", 2
                ),
                "steps_done": metric_config.get("steps_done", 49),
            },
            "metric_head": metric_head.state_dict(),
            "relation_encoder": legacy_relation.state_dict(),
        },
        path,
    )


def test_train_loader_unfreezes_only_action_connected_relation_weights(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)

    metric_head, relation = _load_train_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        train_relation=True,
    )

    assert not metric_head.training
    assert all(not parameter.requires_grad for parameter in metric_head.parameters())
    assert relation.training
    trainable_names = {
        name for name, parameter in relation.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(not name.startswith("recon.") for name in trainable_names)

    args = parse_args(["--mtvj-train-relation", "--lr-mtvj-relation", "2e-5"])
    group = _mtvj_relation_optimizer_group(args, relation)
    assert group is not None
    assert group["lr"] == pytest.approx(2e-5)
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in relation.parameters() if parameter.requires_grad
    }


def test_train_loader_unfreezes_only_action_connected_metric_weights(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)

    metric_head, _ = _load_train_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        train_metric_head=True,
    )

    trainable_names = {
        name for name, parameter in metric_head.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert any(name.startswith("vis_mlp.") for name in trainable_names)
    assert not any(name.startswith("rel_mlp.") for name in trainable_names)
    assert metric_head.training

    args = parse_args(
        ["--mtvj-train-metric-head", "--lr-mtvj-metric-head", "1e-6"]
    )
    group = _mtvj_metric_head_optimizer_group(args, metric_head)
    assert group is not None
    assert group["lr"] == pytest.approx(1e-6)
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in metric_head.parameters() if parameter.requires_grad
    }


def _eval_metric_contract() -> dict:
    return {
        "metric_tokens_enabled": True,
        "metric_state_source": "p_times_visibility_flat",
        "metric_state_dim": 8,
        "metric_d_model": 16,
        "metric_contract_version": 3,
        "metric_head_checkpointed": True,
    }


def _policy_metric_metadata(path, metric_head) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return (
        _mtvj_metric_head_constructor_config(metric_head),
        _mtvj_metric_checkpoint_identity(path, checkpoint),
    )


def test_weights_only_resume_uses_main_metric_config_when_external_changes(
    tmp_path, capsys
) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    changed = torch.load(metric_path, map_location="cpu", weights_only=True)
    changed["config"]["mode_readout"] = True
    torch.save(changed, metric_path)

    loaded_metric, loaded_relation = _load_train_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        train_relation=True,
        policy_relation_state=policy_relation.state_dict(),
        policy_metric_state=policy_metric.state_dict(),
        policy_metric_config=policy_config,
        policy_metric_identity=policy_identity,
        policy_training_contract=_eval_metric_contract(),
    )

    assert loaded_metric.mode_readout is False
    assert loaded_metric._mtvj_external_checkpoint_identity == policy_identity
    for key, value in policy_metric.state_dict().items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    for key, value in policy_relation.state_dict().items():
        torch.testing.assert_close(
            loaded_relation.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    assert "WARNING: --resume" in capsys.readouterr().out


def test_explicit_metric_head_migration_replaces_head_preserves_relation_and_records_source(
    tmp_path, capsys
) -> None:
    metric_path = tmp_path / "all_task_metric.pt"
    _write_external_metric_checkpoint(metric_path, mode_readout=True)
    external = torch.load(metric_path, map_location="cpu", weights_only=True)
    external["metric_head"] = {
        key: torch.full_like(value, 0.75)
        for key, value in external["metric_head"].items()
    }
    torch.save(external, metric_path)

    policy_metric = LanguageMetricField(
        lang_dim=8, h_dim=4, d_proj=2, n_roles=4, mode_readout=False
    )
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    with torch.no_grad():
        for parameter in policy_metric.parameters():
            parameter.fill_(0.125)
        for parameter in policy_relation.parameters():
            parameter.fill_(0.25)
    policy_config = _mtvj_metric_head_constructor_config(policy_metric)
    policy_identity = {
        "path": "old-policy-source.pt",
        "sha256": "old-policy-sha",
        "size_bytes": 123,
        "contract": "mt_vj_metric_field_v1",
    }

    loaded_metric, loaded_relation = _load_train_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        train_relation=True,
        policy_relation_state=policy_relation.state_dict(),
        policy_metric_state=policy_metric.state_dict(),
        policy_metric_config=policy_config,
        policy_metric_identity=policy_identity,
        policy_training_contract=_eval_metric_contract(),
        replace_metric_head_from_external=True,
    )

    assert loaded_metric.mode_readout is True
    assert all(not parameter.requires_grad for parameter in loaded_metric.parameters())
    for key, value in external["metric_head"].items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    for key, value in policy_relation.state_dict().items():
        torch.testing.assert_close(
            loaded_relation.state_dict()[key], value, rtol=0.0, atol=0.0
        )

    # Regression for the second resume restore: it must restore relation but
    # must not overwrite the freshly migrated external head with the old one.
    external_metric_state = {
        key: value.clone() for key, value in loaded_metric.state_dict().items()
    }
    with torch.no_grad():
        for parameter in loaded_relation.parameters():
            parameter.zero_()
    _restore_mtvj_policy_modules(
        {
            "training_contract": _eval_metric_contract(),
            "mtvj_relation_encoder": policy_relation.state_dict(),
            "mtvj_metric_head": policy_metric.state_dict(),
            "mtvj_metric_head_config": policy_config,
            "mtvj_metric_checkpoint_identity": policy_identity,
        },
        relation_encoder=loaded_relation,
        metric_head=loaded_metric,
        train_relation=True,
        replace_metric_head_from_external=True,
    )
    for key, value in policy_relation.state_dict().items():
        torch.testing.assert_close(
            loaded_relation.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    for key, value in external_metric_state.items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    assert "显式跳过主 policy 的旧 metric head" in capsys.readouterr().out

    save_path = tmp_path / "migrated_policy.pt"
    args = parse_args(["--single-task", "--save", str(save_path)])
    save_checkpoint(
        args,
        SimpleNamespace(hidden_dim=16),
        nn.Linear(3, 2),
        None,
        relation_encoder=loaded_relation,
        metric_head=loaded_metric,
    )
    saved = torch.load(save_path, map_location="cpu", weights_only=True)
    assert "explicit all-task migration" in saved["training_contract"]["metric_head_source"]
    assert saved["mtvj_metric_head_migration"]["kind"] == (
        "replace_mtvj_metric_head_from_external"
    )
    assert set(saved["mtvj_metric_head_migration"]["required_tasks"]) == set(
        ENV_TO_TASK
    )
    assert len(saved["mtvj_metric_head_migration"]["required_tasks"]) == 49
    assert saved["training_contract"]["metric_head_joint_trained"] is False
    assert saved["mtvj_metric_checkpoint_identity"]["sha256"] == (
        _mtvj_metric_checkpoint_identity(metric_path, external)["sha256"]
    )


def test_explicit_metric_head_migration_rejects_exact_resume() -> None:
    args = parse_args(
        [
            "--dense-readout-mtvj",
            "--metric-visual-checkpoint",
            "metric.pt",
            "--resume-exact",
            "policy.pt",
            "--replace-mtvj-metric-head-from-external",
        ]
    )
    with pytest.raises(ValueError, match="禁止.*--resume-exact"):
        validate_args(args)


def test_explicit_metric_head_migration_requires_and_accepts_ordinary_resume() -> None:
    base = [
        "--dense-readout-mtvj",
        "--metric-visual-checkpoint",
        "metric.pt",
        "--replace-mtvj-metric-head-from-external",
    ]
    with pytest.raises(ValueError, match="requires ordinary --resume"):
        validate_args(parse_args(base))

    with pytest.raises(ValueError, match="--mtvj-train-relation"):
        validate_args(parse_args([*base, "--resume", "policy.pt"]))
    validate_args(
        parse_args([*base, "--resume", "policy.pt", "--mtvj-train-relation"])
    )


def test_explicit_metric_head_migration_rejects_untrained_visibility(tmp_path) -> None:
    metric_path = tmp_path / "loc_only_metric.pt"
    _write_external_metric_checkpoint(metric_path, loc_only=True)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    with pytest.raises(ValueError, match="visibility.*loc-only/random"):
        _load_train_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            train_relation=True,
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=policy_metric.state_dict(),
            policy_metric_config=_mtvj_metric_head_constructor_config(policy_metric),
            policy_metric_identity={"sha256": "old", "size_bytes": 1,
                                    "contract": "mt_vj_metric_field_v1"},
            policy_training_contract=_eval_metric_contract(),
            replace_metric_head_from_external=True,
        )


def test_explicit_metric_head_migration_rejects_checkpoint_missing_any_all_task(
    tmp_path,
) -> None:
    metric_path = tmp_path / "metric_missing_one_task.pt"
    missing_task = "peg-insert-side-v3"
    _write_external_metric_checkpoint(
        metric_path,
        tasks=[task for task in ENV_TO_TASK if task != missing_task],
    )
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_config = _mtvj_metric_head_constructor_config(policy_metric)

    with pytest.raises(ValueError, match=f"config.tasks.*{missing_task}"):
        _load_train_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            train_relation=True,
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=policy_metric.state_dict(),
            policy_metric_config=policy_config,
            policy_metric_identity={
                "sha256": "old",
                "size_bytes": 1,
                "contract": "mt_vj_metric_field_v1",
            },
            policy_training_contract=_eval_metric_contract(),
            replace_metric_head_from_external=True,
        )


def test_exact_resume_rejects_changed_metric_config_and_fingerprint(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    changed = torch.load(metric_path, map_location="cpu", weights_only=True)
    changed["config"]["mode_readout"] = True
    torch.save(changed, metric_path)

    with pytest.raises(ValueError, match="--resume-exact"):
        _load_train_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=policy_metric.state_dict(),
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
            exact_resume=True,
        )


def test_eval_rejects_external_metric_fingerprint_change(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    changed = torch.load(metric_path, map_location="cpu", weights_only=True)
    changed["revision"] = 2
    torch.save(changed, metric_path)

    with pytest.raises(ValueError, match="fingerprint"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=policy_metric.state_dict(),
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )


def test_main_metric_head_state_rejects_incomplete_constructor_config(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    _, policy_identity = _policy_metric_metadata(metric_path, policy_metric)

    with pytest.raises(ValueError, match="缺少完整 mtvj_metric_head_config"):
        _load_train_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=policy_metric.state_dict(),
            policy_metric_config={"lang_dim": 8},
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )


def test_eval_prefers_main_policy_mtvj_states_over_legacy_external(
    tmp_path, capsys
) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    constructor_only = torch.load(metric_path, map_location="cpu", weights_only=True)
    constructor_only.pop("metric_head")
    constructor_only.pop("relation_encoder")
    torch.save(constructor_only, metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    with torch.no_grad():
        for parameter in policy_relation.parameters():
            parameter.fill_(0.125)
        for parameter in policy_metric.parameters():
            parameter.fill_(0.25)

    loaded_metric, loaded_relation = _load_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        policy_relation_state=policy_relation.state_dict(),
        policy_metric_state=policy_metric.state_dict(),
        policy_metric_config=policy_config,
        policy_metric_identity=policy_identity,
        policy_training_contract=_eval_metric_contract(),
    )

    for key, value in policy_metric.state_dict().items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    for key, value in policy_relation.state_dict().items():
        torch.testing.assert_close(
            loaded_relation.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    output = capsys.readouterr().out
    assert "metric head from main policy checkpoint" in output
    assert "constructor config from main policy checkpoint" in output


def test_eval_fails_if_contract_declares_missing_policy_metric_head(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    external_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, external_metric
    )

    with pytest.raises(ValueError, match="mtvj_metric_head"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )


def test_eval_strictly_rejects_policy_metric_head_shape_mismatch(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)
    policy_metric = LanguageMetricField(lang_dim=8, h_dim=4, d_proj=2, n_roles=4)
    policy_config, policy_identity = _policy_metric_metadata(
        metric_path, policy_metric
    )
    broken_metric = dict(policy_metric.state_dict())
    first_key = next(iter(broken_metric))
    broken_metric[first_key] = broken_metric[first_key][:-1]

    with pytest.raises(ValueError, match="shape_mismatch"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_metric_state=broken_metric,
            policy_metric_config=policy_config,
            policy_metric_identity=policy_identity,
            policy_training_contract=_eval_metric_contract(),
        )


def test_eval_legacy_policy_uses_external_metric_head(tmp_path, capsys) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    external = torch.load(metric_path, map_location="cpu", weights_only=True)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)

    loaded_metric, _ = _load_mtvj_metric_checkpoint(
        metric_path,
        torch.device("cpu"),
        SimpleNamespace(hidden_dim=16),
        policy_relation_state=policy_relation.state_dict(),
        policy_training_contract={},
    )

    for key, value in external["metric_head"].items():
        torch.testing.assert_close(
            loaded_metric.state_dict()[key], value, rtol=0.0, atol=0.0
        )
    assert "external metric checkpoint (legacy migration)" in capsys.readouterr().out


def test_eval_rejects_incompatible_main_policy_metric_contract(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)
    policy_relation = RelationStateEncoder(state_dim=8, d_model=16)

    with pytest.raises(ValueError, match="metric 契约不兼容"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
            policy_relation_state=policy_relation.state_dict(),
            policy_training_contract={
                "metric_tokens_enabled": True,
                "metric_state_source": "relation",
                "metric_state_dim": 6,
                "metric_d_model": 16,
                "metric_contract_version": 1,
            },
        )


def test_eval_rejects_random_relation_rebuild_for_legacy_6d(tmp_path) -> None:
    metric_path = tmp_path / "metric.pt"
    _write_external_metric_checkpoint(metric_path)

    with pytest.raises(ValueError, match="dense-only"):
        _load_mtvj_metric_checkpoint(
            metric_path,
            torch.device("cpu"),
            SimpleNamespace(hidden_dim=16),
        )
