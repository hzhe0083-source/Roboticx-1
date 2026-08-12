"""CPU-only contract tests for metric-head checkpoint/resume semantics."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from prepare_metaworld_metric import SAMPLE_RNG_CONTRACT, SUPPORTED_TASKS
from train_metric_visual import (
    ALIAS_CONSISTENCY_LAMBDA,
    ALIAS_COORD_TOL,
    CONTRACT,
    DEFAULT_TASKS,
    GEOMETRY_CONSISTENCY_LAMBDA,
    METRIC_LOSS_CONTRACT,
    METRIC_VISIBILITY_CONTRACT,
    TRAINING_STATE_VERSION,
    balanced_task_for_step,
    build_checkpoint_config,
    build_training_state,
    checkpoint_file_identity,
    load_initialization_weights,
    load_task_weights_json,
    parse_args,
    release_text_backbone,
    resolve_training_args,
    restore_training_state,
    task_for_step,
)


def _versioned_config() -> dict:
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "tasks": ["assembly-v3"],
        "task_sampling": "weighted",
        "seed": 7,
        "steps": 100,
        "steps_done": 40,
        "batch_size": 8,
        "lr": 1e-3,
        "fixed_data": None,
        "relation_lambda": 1.0,
        "relation_recon_lambda": 0.1,
        "alias_consistency_lambda": ALIAS_CONSISTENCY_LAMBDA,
        "alias_coord_tolerance": ALIAS_COORD_TOL,
        "geometry_consistency_lambda": GEOMETRY_CONSISTENCY_LAMBDA,
        "sample_rng_contract": SAMPLE_RNG_CONTRACT,
        "metric_visibility_contract": METRIC_VISIBILITY_CONTRACT,
        "metric_loss_contract": METRIC_LOSS_CONTRACT,
        "relation_encoder_trained": False,
        "l2_norm": True,
        "learnable_temp": True,
        "temp_init": 10.0,
        "freeze_bias": True,
        "sigma_px": 4.0,
        "loc_only": True,
        "offset_supervision": True,
        "grad_accum": 4,
        "mode_readout": True,
        "hinge_loss": True,
        "hinge_margin": 0.1,
        "task_weights": None,
        "task_weights_sha256": None,
        "task_weights_source": None,
        "initialization_source": None,
    }


def _checkpoint(config: dict) -> dict:
    return {"contract": CONTRACT, "config": config}


def _init_checkpoint(config: dict) -> dict:
    return {
        "contract": CONTRACT,
        "config": config,
        "metric_head": {},
        "relation_encoder": {},
        # These must be ignored by weights-only initialization.
        "optimizer": {"stale": True},
        "numpy_rng_state": {"stale": True},
        "optimizer_steps_done": 999,
    }


def _write_all49_weights(tmp_path, *, name: str = "weights.json", first: int = 4):
    weights = {task: 1 for task in SUPPORTED_TASKS}
    weights[SUPPORTED_TASKS[0]] = first
    path = tmp_path / name
    path.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    return path, weights


def test_fresh_defaults_are_resolved_once() -> None:
    args = resolve_training_args(parse_args([]))
    assert args.steps == 20_000
    assert args.batch_size == 8
    assert args.lr == 1e-3
    assert args.seed == 0
    assert args.l2_norm is False
    assert args.learnable_temp is False
    assert args.grad_accum == 1
    assert args.tasks == DEFAULT_TASKS
    assert len(args.tasks.split(",")) == 49
    assert args.task_sampling == "weighted"


def test_release_text_backbone_empties_cuda_cache_only_for_cuda(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    assert release_text_backbone(None, torch.device("cpu")) is None
    assert calls == []
    assert release_text_backbone(None, torch.device("cuda")) is None
    assert calls == ["empty"]


def test_resume_and_init_checkpoint_are_cli_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--resume", "a.pt", "--init-checkpoint", "b.pt"])


def test_balanced_task_schedule_is_exact_and_resume_stable() -> None:
    tasks = ["a", "b", "c", "d"]
    uninterrupted = [balanced_task_for_step(tasks, step, seed=19) for step in range(20)]
    resumed = (
        [balanced_task_for_step(tasks, step, seed=19) for step in range(7)]
        + [balanced_task_for_step(tasks, step, seed=19) for step in range(7, 20)]
    )
    assert resumed == uninterrupted
    for start in range(0, 20, len(tasks)):
        assert sorted(uninterrupted[start : start + len(tasks)]) == sorted(tasks)
    assert uninterrupted != [balanced_task_for_step(tasks, step, seed=20) for step in range(20)]


def test_weighted_task_schedule_matches_difficulty_ratios_and_resume() -> None:
    tasks = ["reach-v3", "coffee-push-v3", "hammer-v3", "assembly-v3"]
    # 0.5/1/2/3 become one deterministic shuffled cycle with 1/2/4/6 slots.
    cycle = [task_for_step(tasks, step, seed=23) for step in range(13)]
    assert {task: cycle.count(task) for task in tasks} == {
        "reach-v3": 1,
        "coffee-push-v3": 2,
        "hammer-v3": 4,
        "assembly-v3": 6,
    }
    uninterrupted = [task_for_step(tasks, step, seed=23) for step in range(26)]
    resumed = [task_for_step(tasks, step, seed=23) for step in range(5)] + [
        task_for_step(tasks, step, seed=23) for step in range(5, 26)
    ]
    assert resumed == uninterrupted


def test_explicit_weighted_schedule_is_bounded_exact_and_resume_stable() -> None:
    tasks = ["a", "b", "c"]
    weights = {"a": 1, "b": 2, "c": 4}
    cycle = [
        task_for_step(tasks, step, seed=31, task_weights=weights)
        for step in range(7)
    ]
    assert {task: cycle.count(task) for task in tasks} == weights
    uninterrupted = [
        task_for_step(tasks, step, seed=31, task_weights=weights)
        for step in range(21)
    ]
    resumed = uninterrupted[:9] + [
        task_for_step(tasks, step, seed=31, task_weights=weights)
        for step in range(9, 21)
    ]
    assert resumed == uninterrupted
    with pytest.raises(ValueError, match="balanced.*explicit"):
        task_for_step(tasks, 0, seed=31, sampling="balanced", task_weights=weights)


def test_task_weights_json_is_canonical_bounded_and_hashed(tmp_path) -> None:
    path, expected = _write_all49_weights(tmp_path)
    weights, digest, source = load_task_weights_json(path)
    assert weights == expected
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source == str(path.resolve())

    missing = dict(expected)
    missing.pop(SUPPORTED_TASKS[-1])
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical all-49"):
        load_task_weights_json(missing_path)

    invalid = dict(expected)
    invalid[SUPPORTED_TASKS[0]] = True
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="integer in \\[1,4\\]"):
        load_task_weights_json(invalid_path)


def test_task_weights_are_checkpointed_and_semantically_immutable(tmp_path) -> None:
    path, expected = _write_all49_weights(tmp_path)
    args = resolve_training_args(
        parse_args(["--task-weights-json", str(path), "--steps", "8"])
    )
    config = build_checkpoint_config(
        args, list(SUPPORTED_TASKS), 8, language_cache_available=True
    )
    assert config["task_schedule"] == "step_derived_shuffled_explicit_weighted_cycles_v1"
    assert config["task_weights"] == expected
    assert config["task_weights_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    resumed = resolve_training_args(
        parse_args(["--resume", "unused.pt"]), _checkpoint(config)
    )
    assert resumed.task_weights == expected
    assert resumed.task_weights_sha256 == config["task_weights_sha256"]

    reformatted = tmp_path / "same-values-different-bytes.json"
    reformatted.write_text(json.dumps(expected, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic mismatch.*SHA256"):
        resolve_training_args(
            parse_args(
                ["--resume", "unused.pt", "--task-weights-json", str(reformatted)]
            ),
            _checkpoint(config),
        )


def test_versioned_resume_inherits_semantics_but_allows_operational_overrides() -> None:
    args = parse_args(
        [
            "--resume",
            "unused.pt",
            "--tasks",
            "door-lock-v3",
            "--steps",
            "120",
            "--batch-size",
            "4",
            "--lr",
            "0.0002",
        ]
    )
    args = resolve_training_args(args, _checkpoint(_versioned_config()))
    assert args.tasks == "door-lock-v3"
    assert (args.steps, args.batch_size, args.lr) == (120, 4, 2e-4)
    assert args.seed == 7
    assert args.loc_only and args.offset_supervision
    assert args.grad_accum == 4
    assert args.hinge_loss and args.mode_readout and args.no_bias


def test_init_checkpoint_inherits_model_semantics_but_starts_new_run() -> None:
    source = _init_checkpoint(_versioned_config())
    args = resolve_training_args(
        parse_args(
            [
                "--init-checkpoint",
                "unused.pt",
                "--steps",
                "8",
                "--batch-size",
                "2",
                "--lr",
                "0.0003",
                "--seed",
                "8",
            ]
        ),
        init_checkpoint=source,
    )
    assert (args.steps, args.batch_size, args.lr, args.seed) == (8, 2, 3e-4, 8)
    assert args.loc_only and args.offset_supervision and args.grad_accum == 4
    assert args.hinge_loss and args.mode_readout and args.no_bias
    assert args.initialization_source is None

    with pytest.raises(ValueError, match="explicit new --seed"):
        resolve_training_args(
            parse_args(["--init-checkpoint", "unused.pt", "--steps", "8"]),
            init_checkpoint=source,
        )
    with pytest.raises(ValueError, match="requires a new seed"):
        resolve_training_args(
            parse_args(
                [
                    "--init-checkpoint",
                    "unused.pt",
                    "--steps",
                    "8",
                    "--seed",
                    "7",
                ]
            ),
            init_checkpoint=source,
        )


def test_init_checkpoint_loads_only_strict_weights_and_keeps_adam_fresh() -> None:
    source_metric = torch.nn.Linear(2, 2)
    source_relation = torch.nn.Linear(2, 1)
    target_metric = torch.nn.Linear(2, 2)
    target_relation = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(
        list(target_metric.parameters()) + list(target_relation.parameters())
    )
    checkpoint = {
        "contract": CONTRACT,
        "config": {"steps_done": 10_000},
        "metric_head": source_metric.state_dict(),
        "relation_encoder": source_relation.state_dict(),
        "optimizer": {"must_not_load": True},
        "numpy_rng_state": {"must_not_load": True},
        "optimizer_steps_done": 2_500,
    }
    load_initialization_weights(target_metric, target_relation, checkpoint)
    for actual, expected in zip(target_metric.parameters(), source_metric.parameters()):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(target_relation.parameters(), source_relation.parameters()):
        torch.testing.assert_close(actual, expected)
    assert optimizer.state == {}

    broken = dict(checkpoint)
    broken["metric_head"] = {"wrong": torch.zeros(1)}
    with pytest.raises(RuntimeError):
        load_initialization_weights(target_metric, target_relation, broken)


def test_init_checkpoint_identity_records_exact_source_file(tmp_path) -> None:
    path = tmp_path / "source.pt"
    path.write_bytes(b"metric-source-bytes")
    identity = checkpoint_file_identity(
        path,
        {"contract": CONTRACT, "config": {"steps_done": 10_000}},
    )
    assert identity == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "contract": CONTRACT,
        "steps_done": 10_000,
        "sample_rng_contract": None,
        "metric_visibility_contract": None,
        "metric_loss_contract": None,
    }
    args = resolve_training_args(parse_args(["--steps", "4"]))
    args.initialization_source = identity
    config = build_checkpoint_config(
        args, list(SUPPORTED_TASKS), 4, language_cache_available=True
    )
    assert config["steps"] == 4
    assert config["steps_done"] == 4
    assert config["initialization_source"]["sha256"] == identity["sha256"]
    assert config["initialization_source"]["contract"] == CONTRACT
    assert config["initialization_source"]["steps_done"] == 10_000


def test_versioned_resume_rejects_constructor_or_loss_drift() -> None:
    args = parse_args(["--resume", "unused.pt", "--no-hinge-loss"])
    with pytest.raises(ValueError, match="semantic mismatch for hinge_loss"):
        resolve_training_args(args, _checkpoint(_versioned_config()))

    changed = _versioned_config()
    changed["relation_recon_lambda"] = 0.2
    with pytest.raises(ValueError, match="resume loss mismatch"):
        resolve_training_args(
            parse_args(["--resume", "unused.pt"]), _checkpoint(changed)
        )


def test_legacy_v4_requires_missing_flags_and_explicit_optimizer_reset() -> None:
    legacy = _versioned_config()
    legacy.pop("training_state_version")
    legacy.pop("loc_only")
    legacy.pop("offset_supervision")
    legacy.pop("grad_accum")

    with pytest.raises(ValueError, match="loc_only.*offset_supervision.*grad_accum"):
        resolve_training_args(
            parse_args(["--resume", "legacy.pt"]), _checkpoint(legacy)
        )

    explicit = [
        "--resume",
        "legacy.pt",
        "--loc-only",
        "--offset-supervision",
        "--grad-accum",
        "4",
    ]
    with pytest.raises(ValueError, match="--allow-legacy-optimizer-reset"):
        resolve_training_args(parse_args(explicit), _checkpoint(legacy))

    args = resolve_training_args(
        parse_args(explicit + ["--allow-legacy-optimizer-reset"]),
        _checkpoint(legacy),
    )
    assert args._legacy_resume is True
    assert args.loc_only and args.offset_supervision and args.grad_accum == 4


def test_steps_must_end_on_optimizer_boundary() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_training_args(parse_args(["--steps", "5", "--grad-accum", "4"]))


def test_checkpoint_config_records_complete_semantics() -> None:
    args = resolve_training_args(
        parse_args(
            [
                "--steps",
                "8",
                "--grad-accum",
                "4",
                "--l2-norm",
                "--loc-only",
                "--offset-supervision",
                "--mode-readout",
                "--hinge-loss",
            ]
        )
    )
    config = build_checkpoint_config(
        args, ["door-lock-v3"], 8, language_cache_available=True
    )
    assert config["optimizer_steps_done"] == 2
    assert config["relation_encoder_trained"] is False
    assert config["task_schedule"] == "step_derived_shuffled_difficulty_weighted_cycles_v1"
    assert config["task_sampling"] == "weighted"
    assert config["task_schedule_origin"] == 0
    assert config["task_weights"] is None
    assert config["task_weights_sha256"] is None
    assert config["initialization_source"] is None
    assert config["sample_rng_contract"] == SAMPLE_RNG_CONTRACT
    assert config["metric_visibility_contract"] == METRIC_VISIBILITY_CONTRACT
    assert config["metric_loss_contract"] == METRIC_LOSS_CONTRACT
    assert {
        "seed",
        "l2_norm",
        "learnable_temp",
        "temp_init",
        "freeze_bias",
        "sigma_px",
        "loc_only",
        "offset_supervision",
        "grad_accum",
        "mode_readout",
        "hinge_loss",
        "hinge_margin",
        "alias_consistency_lambda",
        "alias_coord_tolerance",
        "geometry_consistency_lambda",
        "metric_visibility_contract",
        "metric_loss_contract",
    } <= config.keys()


def test_training_state_weights_only_roundtrip_restores_optimizer_and_rng(tmp_path) -> None:
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rng = np.random.default_rng(13)

    state = build_training_state(optimizer, rng, completed_steps=4, grad_accum=4)
    path = tmp_path / "state.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    expected_np = int(rng.integers(0, 2**31))
    expected_torch = torch.rand(3)
    fresh_model = torch.nn.Linear(2, 1)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=9e-3)
    fresh_rng = np.random.default_rng(99)
    restore_training_state(
        loaded,
        fresh_optimizer,
        fresh_rng,
        completed_steps=4,
        grad_accum=4,
        requested_lr=2e-4,
    )

    assert int(fresh_rng.integers(0, 2**31)) == expected_np
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0.0, atol=0.0)
    assert fresh_optimizer.param_groups[0]["lr"] == 2e-4
    old_state = next(iter(optimizer.state.values()))
    new_state = next(iter(fresh_optimizer.state.values()))
    torch.testing.assert_close(new_state["exp_avg"], old_state["exp_avg"])


def test_training_state_refuses_partial_accumulation() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="gradient-accumulation window"):
        build_training_state(
            optimizer, np.random.default_rng(0), completed_steps=3, grad_accum=4
        )
    GEOMETRY_CONSISTENCY_LAMBDA,
    METRIC_LOSS_CONTRACT,
    METRIC_VISIBILITY_CONTRACT,
