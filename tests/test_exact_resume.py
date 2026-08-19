from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from train import (
    SAM,
    TaskLocalityWeightedSampler,
    TaskWeightedSampler,
    WORLD_ACTION_DONOR_CONTRACT,
    WORLD_ACTION_RANKING,
    WORLD_LOGGED_BRANCH_CONTRACT,
    WORLD_LOSS_COMPONENT_WEIGHTS,
    WORLD_NO_REGRESSION,
    WORLD_STAGE_AUXILIARY_DECAY,
    WORLD_STATIC_COPY_CONSTRAINT,
    WORLD_SUPERVISION_CONTRACT,
    WORLD_TRANSITION_CONTRACT,
    build_dataset_content_identity,
    build_exact_resume_state,
    build_exact_run_contract,
    clip_update_gradients,
    named_optimizer_parameters,
    validate_args,
    validate_finite_update_scalars,
    validate_optimizer_update_state,
    validate_update_gradients,
    final_checkpoint_save_due,
    parse_args,
    restore_exact_resume_state,
    save_checkpoint,
    validate_exact_run_contract,
    validate_visual_world_resume_contract,
    world_action_ranking_contract,
)


def _sampler() -> TaskLocalityWeightedSampler:
    return TaskLocalityWeightedSampler(
        instruction_id=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        episode_id=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        task_weights=torch.tensor([1.0, 1.0]),
        batch_size=2,
        seed=19,
        block_batches=1,
    )


def _model_and_optimizer() -> tuple[nn.Module, torch.optim.AdamW]:
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    return model, torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)


def _seed_training_rngs() -> None:
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)


def _contract(name: str = "baseline") -> dict:
    return {
        "contract_version": 1,
        "data_identity": {"full_file_sha256": name},
        "arguments": {"flow_tail_weight": 0.1},
        "model_config": {"num_layers": 8},
        "optimizer": {"kind": "adamw"},
        "mtvj": {},
    }


def _update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: TaskLocalityWeightedSampler,
) -> tuple[list[int], float, float, torch.Tensor]:
    indices = next(iter(sampler))
    python_draw = random.random()
    numpy_draw = float(np.random.random())
    torch_draw = torch.rand(())
    x = torch.tensor([[float(indices[0]), float(indices[1])]]) / 8.0
    target = torch.tensor([[python_draw + numpy_draw]]) + torch_draw
    optimizer.zero_grad(set_to_none=True)
    (model(x) - target).square().mean().backward()
    optimizer.step()
    sampler.advance()
    return indices, python_draw, numpy_draw, torch_draw.clone()


def test_two_updates_equal_one_save_resume_one_with_weights_only_load(tmp_path) -> None:
    baseline_model, baseline_optimizer = _model_and_optimizer()
    baseline_sampler = _sampler()
    _seed_training_rngs()
    _update(baseline_model, baseline_optimizer, baseline_sampler)
    baseline_second = _update(baseline_model, baseline_optimizer, baseline_sampler)

    interrupted_model, interrupted_optimizer = _model_and_optimizer()
    interrupted_sampler = _sampler()
    _seed_training_rngs()
    _update(interrupted_model, interrupted_optimizer, interrupted_sampler)
    checkpoint = {"model": interrupted_model.state_dict()}
    checkpoint.update(
        build_exact_resume_state(
            interrupted_optimizer, 1, interrupted_sampler, _contract()
        )
    )
    path = tmp_path / "exact.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    resumed_model, resumed_optimizer = _model_and_optimizer()
    resumed_sampler = _sampler()
    resumed_model.load_state_dict(loaded["model"], strict=True)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    global_step = restore_exact_resume_state(
        loaded,
        resumed_optimizer,
        resumed_sampler,
        runtime_exact_run_contract=_contract(),
    )
    resumed_second = _update(resumed_model, resumed_optimizer, resumed_sampler)

    assert global_step == 1
    assert resumed_second[:3] == baseline_second[:3]
    torch.testing.assert_close(resumed_second[3], baseline_second[3], rtol=0.0, atol=0.0)
    assert resumed_sampler.state_dict() == baseline_sampler.state_dict()
    for key, expected in baseline_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[key], expected, rtol=0.0, atol=0.0
        )


def test_restore_allows_null_sampler_for_dino_weighted_checkpoints() -> None:
    _, optimizer = _model_and_optimizer()
    payload = build_exact_resume_state(optimizer, 6000, None, _contract())
    assert payload["sampler_state"] is None
    step = restore_exact_resume_state(
        payload,
        optimizer,
        None,
        runtime_exact_run_contract=_contract(),
        restore_rng=False,
    )
    assert step == 6000
    weighted = TaskWeightedSampler(torch.tensor([1.0, 2.0, 3.0, 4.0]), batch_size=2, seed=0)
    restore_exact_resume_state(
        payload,
        optimizer,
        weighted,
        runtime_exact_run_contract=_contract(),
        restore_rng=False,
    )
    assert weighted.epoch == 0
    assert weighted.batch_cursor == 0
    with pytest.raises(ValueError, match="sampler_state=None"):
        restore_exact_resume_state(
            payload,
            optimizer,
            _sampler(),
            runtime_exact_run_contract=_contract(),
            restore_rng=False,
        )


def test_task_weighted_sampler_roundtrip_and_advance() -> None:
    weights = torch.tensor([0.5, 1.0, 2.0, 3.0, 0.5, 1.0])
    baseline = TaskWeightedSampler(weights, batch_size=2, seed=7)
    first = next(iter(baseline))
    baseline.advance()
    second = next(iter(baseline))
    resumed = TaskWeightedSampler(weights, batch_size=2, seed=7)
    resumed.load_state_dict(baseline.state_dict())
    assert next(iter(resumed)) == second
    resumed.advance()
    baseline.advance()
    assert resumed.state_dict() == baseline.state_dict()
    assert len(first) == 2
    with pytest.raises(ValueError, match="weights_sha256"):
        resumed.load_state_dict(
            {**baseline.state_dict(), "weights_sha256": "deadbeef"}
        )


def test_exact_resume_rejects_legacy_checkpoint() -> None:
    model, optimizer = _model_and_optimizer()
    del model
    with pytest.raises(ValueError, match="Use --resume for legacy"):
        restore_exact_resume_state({"model": {}}, optimizer, _sampler())


def test_main_checkpoint_contains_exact_training_state(tmp_path) -> None:
    path = tmp_path / "policy.pt"
    args = parse_args(["--single-task", "--save", str(path)])
    model, optimizer = _model_and_optimizer()
    sampler = _sampler()
    save_checkpoint(
        args,
        SimpleNamespace(),
        model,
        None,
        optimizer=optimizer,
        global_step=7,
        sampler=sampler,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["global_step"] == 7
    assert loaded["optimizer_state"]["kind"] == "adamw"
    assert loaded["sampler_state"] == sampler.state_dict()
    assert set(loaded["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert loaded["rng_state"]["numpy"]["state"].dtype == torch.int64
    assert loaded["exact_run_contract"]["contract_version"] == 1


def test_final_checkpoint_skips_same_step_periodic_save(tmp_path) -> None:
    path = tmp_path / "policy.pt"
    assert not final_checkpoint_save_due(path, 1000, 1000)


def test_final_checkpoint_kept_for_nonperiodic_final_step(tmp_path) -> None:
    path = tmp_path / "policy.pt"
    assert final_checkpoint_save_due(path, 1001, 1000)


def test_step_copy_collision_does_not_overwrite_archive(tmp_path) -> None:
    path = tmp_path / "policy.pt"
    args = parse_args(
        ["--single-task", "--save", str(path), "--save-step-copies"]
    )
    model, optimizer = _model_and_optimizer()
    save_checkpoint(
        args,
        SimpleNamespace(),
        model,
        None,
        optimizer=optimizer,
        global_step=7,
        sampler=_sampler(),
    )
    step_path = tmp_path / "policy_s7.pt"
    archived = step_path.read_bytes()

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_checkpoint(
            args,
            SimpleNamespace(),
            model,
            None,
            optimizer=optimizer,
            global_step=7,
            sampler=_sampler(),
        )

    assert step_path.read_bytes() == archived
    assert not (tmp_path / "policy_s7.pt.tmp").exists()


def test_update_guards_reject_nonfinite_without_optimizer_mutation() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    parameter_snapshot = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    optimizer_snapshot = optimizer.state_dict()
    loss = torch.tensor(float("nan"), requires_grad=True)
    with pytest.raises(FloatingPointError, match="non-finite update loss total"):
        validate_finite_update_scalars([("total", loss)])
    assert optimizer.state_dict() == optimizer_snapshot
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameter_snapshot[name])


def test_gradient_guard_names_first_bad_value_and_clip_is_finite() -> None:
    model = nn.Linear(2, 1)
    named = list(
        named_optimizer_parameters(
            torch.optim.AdamW(model.parameters()), ("model", model)
        )
    )
    model.weight.grad = torch.tensor([[1.0, float("inf")]])
    model.bias.grad = torch.tensor([1.0])
    with pytest.raises(FloatingPointError, match="non-finite gradient model.weight"):
        validate_update_gradients(named)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    norm = validate_update_gradients(named, max_norm=10.0)
    assert norm == pytest.approx(5.0990195)
    assert clip_update_gradients(named, max_norm=1.0) == pytest.approx(norm)
    assert torch.isfinite(model.weight.grad).all()


def test_gradient_threshold_applies_only_to_aggregate_norm() -> None:
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([2.0])
    assert validate_update_gradients([("parameter", parameter)], max_norm=3.0) == 2.0
    with pytest.raises(FloatingPointError, match="aggregate_norm"):
        validate_update_gradients([("parameter", parameter)], max_norm=1.0)


def test_optimizer_guard_rejects_nan_lr_and_nonfinite_state() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.param_groups[0]["lr"] = float("nan")
    with pytest.raises(ValueError, match=r"group\[0\] lr"):
        validate_optimizer_update_state(optimizer)

    optimizer.param_groups[0]["lr"] = 1e-2
    parameter = next(model.parameters())
    optimizer.state[parameter]["exp_avg"] = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError, match="optimizer state"):
        validate_optimizer_update_state(optimizer)


def test_validate_args_rejects_nan_learning_rate() -> None:
    args = parse_args(["--single-task", "--lr", "nan"])
    with pytest.raises(ValueError, match="--lr must be a positive finite value"):
        validate_args(args)


def test_parameter_guard_rejects_nonfinite_value_before_update() -> None:
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(FloatingPointError, match="non-finite parameter model.weight"):
        validate_update_gradients([("model.weight", model.weight)])


def test_sam_guard_can_restore_perturbation_without_base_step() -> None:
    model = nn.Linear(1, 1)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.1, lr=1e-2)
    original = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    model(torch.ones(1, 1)).backward()
    optimizer.first_step(zero_grad=True)
    model.weight.grad = torch.tensor([[float("nan")]])
    with pytest.raises(FloatingPointError, match="model.weight"):
        validate_update_gradients([("model.weight", model.weight)])
    optimizer.restore_step(zero_grad=True)
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name]), f"SAM restore changed bits for {name}"
    assert optimizer.base_optimizer.state_dict()["state"] == {}


def test_sam_roundtrip_uses_base_adamw_state() -> None:
    torch.manual_seed(7)
    model = nn.Linear(2, 1)
    optimizer = SAM(
        model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3, weight_decay=0.0
    )
    model(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()
    payload = build_exact_resume_state(optimizer, 1, _sampler(), _contract())
    assert payload["optimizer_state"]["kind"] == "sam_adamw"
    assert payload["optimizer_state"]["state_dict"]["state"]

    fresh = nn.Linear(2, 1)
    fresh_optimizer = SAM(
        fresh.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3, weight_decay=0.0
    )
    restore_exact_resume_state(
        payload,
        fresh_optimizer,
        _sampler(),
        runtime_exact_run_contract=_contract(),
        restore_rng=False,
    )
    assert fresh_optimizer.base_optimizer.state_dict()["state"]


def test_resume_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--resume", "legacy.pt", "--resume-exact", "exact.pt"])
    with pytest.raises(SystemExit):
        parse_args(["--resume-weights", "weights.pt", "--resume", "legacy.pt"])
    with pytest.raises(SystemExit):
        parse_args(["--resume-weights", "weights.pt", "--resume-exact", "exact.pt"])


@pytest.mark.parametrize("changed_key", ["actions", "action_valid_mask"])
def test_dataset_identity_rejects_changed_action_payload(tmp_path, changed_key: str) -> None:
    path = tmp_path / "windows.pt"
    payload = {
        "actions": torch.arange(24, dtype=torch.float32).reshape(2, 2, 6),
        "action_valid_mask": torch.ones(2, 2, 6, dtype=torch.bool),
        "instruction_id": torch.tensor([0, 0]),
        "episode_id": torch.tensor([0, 1]),
        "metadata": {"horizon": 6, "tasks": ["door-lock-v3"]},
    }
    torch.save(payload, path)
    baseline = build_dataset_content_identity(path, payload)

    changed = dict(payload)
    changed[changed_key] = payload[changed_key].clone()
    changed[changed_key].reshape(-1)[0] = (
        ~changed[changed_key].reshape(-1)[0]
        if changed[changed_key].dtype == torch.bool
        else changed[changed_key].reshape(-1)[0] + 1
    )
    torch.save(changed, path)
    current = build_dataset_content_identity(path, changed)

    assert baseline["full_file_sha256"] != current["full_file_sha256"]
    with pytest.raises(ValueError, match="data_identity.*full_file_sha256"):
        validate_exact_run_contract(
            {**_contract(), "data_identity": baseline},
            {**_contract(), "data_identity": current},
        )


def test_exact_contract_tracks_dino_roi_identity() -> None:
    _, optimizer = _model_and_optimizer()
    args = parse_args(["--single-task"])
    config = SimpleNamespace(num_layers=8, action_horizon=6)
    roi_a = SimpleNamespace(
        _dino_roi_identity={
            "sha256": "a" * 64,
            "size_bytes": 123,
            "contract": "dino_metric_roi_task35_v2",
            "path": "/ignored/a.pt",
        },
        _mtvj_roi_config={"canonical_image_size": 224},
    )
    roi_b = SimpleNamespace(
        _dino_roi_identity={
            "sha256": "b" * 64,
            "size_bytes": 123,
            "contract": "dino_metric_roi_task35_v2",
            "path": "/ignored/b.pt",
        },
        _mtvj_roi_config={"canonical_image_size": 224},
    )
    baseline = build_exact_run_contract(
        args, config, optimizer, _sampler(), roi_head=roi_a
    )
    changed = build_exact_run_contract(
        args, config, optimizer, _sampler(), roi_head=roi_b
    )
    assert baseline["mtvj"]["roi_checkpoint_identity"]["sha256"] == "a" * 64
    with pytest.raises(ValueError, match="roi_checkpoint_identity"):
        validate_exact_run_contract(baseline, changed)


@pytest.mark.parametrize(
    ("flag", "value", "field"),
    [
        ("--flow-tail-weight", "0.2", "flow_tail_weight"),
        ("--flow-prefix-weight", "0.8", "flow_prefix_weight"),
        ("--flow-prefix-steps", "5", "flow_prefix_steps"),
        ("--prev-dropout", "0.25", "prev_dropout"),
    ],
)
def test_exact_contract_rejects_changed_objective_args(
    flag: str, value: str, field: str
) -> None:
    model, optimizer = _model_and_optimizer()
    del model
    baseline_args = parse_args(["--single-task"])
    changed_args = parse_args(["--single-task", flag, value])
    config = SimpleNamespace(num_layers=8, action_horizon=48)
    baseline = build_exact_run_contract(
        baseline_args, config, optimizer, _sampler()
    )
    current = build_exact_run_contract(
        changed_args, config, optimizer, _sampler()
    )
    with pytest.raises(ValueError, match=field):
        validate_exact_run_contract(baseline, current)


def test_exact_contract_binds_wmrm_proposal_stage_detach() -> None:
    _, optimizer = _model_and_optimizer()
    baseline_args = parse_args(["--wam4va"])
    changed_args = parse_args(["--wam4va", "--wmrm-detach-proposal-stage-state"])
    config = SimpleNamespace(
        num_layers=8,
        action_horizon=48,
        wmrm=True,
        wmrm_detach_proposal_stage_state=False,
    )
    baseline = build_exact_run_contract(baseline_args, config, optimizer, _sampler())
    current = build_exact_run_contract(changed_args, config, optimizer, _sampler())
    with pytest.raises(ValueError, match="wmrm_detach_proposal_stage_state"):
        validate_exact_run_contract(baseline, current)


def test_controlled_detach_migration_allows_only_false_or_absent_to_true() -> None:
    _, optimizer = _model_and_optimizer()
    legacy_args = parse_args(["--wam4va"])
    candidate_args = parse_args(
        ["--wam4va", "--wmrm-detach-proposal-stage-state"]
    )
    legacy_config = SimpleNamespace(
        num_layers=8,
        action_horizon=48,
        wmrm=True,
        wmrm_detach_proposal_stage_state=False,
    )
    candidate_config = SimpleNamespace(
        num_layers=8,
        action_horizon=48,
        wmrm=True,
        wmrm_detach_proposal_stage_state=True,
    )
    saved = build_exact_run_contract(
        legacy_args, legacy_config, optimizer, _sampler()
    )
    current = build_exact_run_contract(
        candidate_args, candidate_config, optimizer, _sampler()
    )
    saved["arguments"].pop("wmrm_detach_proposal_stage_state")
    saved["arguments"].pop("max_gradient_norm")
    saved["model_config"].pop("wmrm_detach_proposal_stage_state")
    snapshot = copy.deepcopy(saved)

    validate_exact_run_contract(
        saved,
        current,
        migration_id="wmrm_detach_proposal_stage_state_v1",
    )
    assert saved == snapshot

    changed = copy.deepcopy(current)
    changed["arguments"]["flow_tail_weight"] = 0.2
    with pytest.raises(ValueError, match="controlled exact-resume migration.*flow_tail_weight"):
        validate_exact_run_contract(
            saved,
            changed,
            migration_id="wmrm_detach_proposal_stage_state_v1",
        )


def test_controlled_detach_migration_id_is_operational_and_not_a_bypass() -> None:
    _, optimizer = _model_and_optimizer()
    args = parse_args(
        [
            "--wam4va",
            "--wmrm-detach-proposal-stage-state",
            "--resume-exact-contract-migration",
            "wmrm_detach_proposal_stage_state_v1",
        ]
    )
    config = SimpleNamespace(
        num_layers=8,
        action_horizon=48,
        wmrm=True,
        wmrm_detach_proposal_stage_state=True,
    )
    contract = build_exact_run_contract(args, config, optimizer, _sampler())
    assert "resume_exact_contract_migration" not in contract["arguments"]
    with pytest.raises(ValueError, match="no coherent old-false-both"):
        validate_exact_run_contract(
            contract,
            contract,
            migration_id="wmrm_detach_proposal_stage_state_v1",
        )
    with pytest.raises(ValueError, match="unsupported"):
        validate_exact_run_contract(contract, contract, migration_id="wrong_v1")


def test_controlled_detach_migration_rejects_partial_representation_transition() -> None:
    _, optimizer = _model_and_optimizer()
    saved = _contract()
    current = copy.deepcopy(saved)
    saved["arguments"]["wmrm_detach_proposal_stage_state"] = False
    saved["model_config"]["wmrm_detach_proposal_stage_state"] = False
    current["arguments"]["wmrm_detach_proposal_stage_state"] = True
    current["model_config"]["wmrm_detach_proposal_stage_state"] = False
    with pytest.raises(ValueError, match="controlled exact-resume migration"):
        validate_exact_run_contract(
            saved,
            current,
            migration_id="wmrm_detach_proposal_stage_state_v1",
        )


def test_controlled_detach_migration_rejects_contradictory_saved_contract() -> None:
    saved = _contract()
    current = copy.deepcopy(saved)
    saved["arguments"]["wmrm_detach_proposal_stage_state"] = False
    saved["model_config"]["wmrm_detach_proposal_stage_state"] = True
    current["arguments"]["wmrm_detach_proposal_stage_state"] = True
    current["model_config"]["wmrm_detach_proposal_stage_state"] = True
    with pytest.raises(ValueError, match="controlled exact-resume migration"):
        validate_exact_run_contract(
            saved,
            current,
            migration_id="wmrm_detach_proposal_stage_state_v1",
        )


def test_migrated_restore_preserves_checkpoint_and_exact_training_state() -> None:
    model, optimizer = _model_and_optimizer()
    sampler = _sampler()
    _seed_training_rngs()
    _update(model, optimizer, sampler)
    saved_contract = _contract()
    saved_contract["arguments"].update(
        {"wmrm_detach_proposal_stage_state": False, "max_gradient_norm": None}
    )
    saved_contract["model_config"]["wmrm_detach_proposal_stage_state"] = False
    checkpoint = {"model": copy.deepcopy(model.state_dict())}
    checkpoint.update(build_exact_resume_state(optimizer, 1, sampler, saved_contract))
    checkpoint_snapshot = copy.deepcopy(checkpoint)

    restored_model, restored_optimizer = _model_and_optimizer()
    restored_sampler = _sampler()
    restored_model.load_state_dict(checkpoint["model"])
    current_contract = copy.deepcopy(saved_contract)
    current_contract["arguments"]["wmrm_detach_proposal_stage_state"] = True
    current_contract["model_config"]["wmrm_detach_proposal_stage_state"] = True
    step = restore_exact_resume_state(
        checkpoint,
        restored_optimizer,
        restored_sampler,
        runtime_exact_run_contract=current_contract,
        migration_id="wmrm_detach_proposal_stage_state_v1",
    )

    assert step == 1
    assert restored_optimizer.state_dict() == optimizer.state_dict()
    assert restored_sampler.state_dict() == sampler.state_dict()
    assert checkpoint.keys() == checkpoint_snapshot.keys()
    assert checkpoint["exact_run_contract"] == checkpoint_snapshot["exact_run_contract"]
    for name, value in checkpoint["model"].items():
        torch.testing.assert_close(value, checkpoint_snapshot["model"][name], rtol=0, atol=0)


def test_controlled_world_weight_migration_allows_only_1_to_0_5_without_mutation() -> None:
    saved = _contract()
    saved["arguments"]["wmrm_world_weight"] = 1.0
    current = copy.deepcopy(saved)
    current["arguments"]["wmrm_world_weight"] = 0.5
    saved_snapshot = copy.deepcopy(saved)
    current_snapshot = copy.deepcopy(current)

    validate_exact_run_contract(
        saved,
        current,
        migration_id="wmrm_world_weight_1_to_0_5_v1",
    )

    assert saved == saved_snapshot
    assert current == current_snapshot


@pytest.mark.parametrize(
    ("saved_weight", "current_weight", "extra_mismatch"),
    [
        (0.5, 1.0, False),
        (1.0, 1.0, False),
        (0.5, 0.5, False),
        (0.75, 0.5, False),
        (1.0, 0.25, False),
        (1, 0.5, False),
        (None, 0.5, False),
        (1.0, None, False),
        (1.0, 0.5, True),
    ],
)
def test_controlled_world_weight_migration_rejects_reverse_unnecessary_and_additional_mismatch(
    saved_weight, current_weight, extra_mismatch: bool
) -> None:
    saved = _contract()
    current = copy.deepcopy(saved)
    saved["arguments"]["wmrm_world_weight"] = saved_weight
    current["arguments"]["wmrm_world_weight"] = current_weight
    if extra_mismatch:
        current["arguments"]["flow_tail_weight"] = 0.2

    with pytest.raises(
        ValueError,
        match="controlled exact-resume migration.*wmrm_world_weight_1_to_0_5_v1.*refused",
    ):
        validate_exact_run_contract(
            saved,
            current,
            migration_id="wmrm_world_weight_1_to_0_5_v1",
        )


def test_static_constraint_weight_migration_is_strict_and_isolated() -> None:
    saved = _contract()
    saved["arguments"].update(
        wmrm_static_constraint_weight=4.0,
        wmrm_world_weight=1.0,
        wmrm_detach_proposal_stage_state=True,
    )
    current = copy.deepcopy(saved)
    current["arguments"]["wmrm_static_constraint_weight"] = 2.0

    validate_exact_run_contract(
        saved, current, migration_id="wmrm_static_constraint_weight_4_to_2_v1"
    )

    for key, value in (
        ("wmrm_world_weight", 0.5),
        ("wmrm_detach_proposal_stage_state", False),
        ("flow_tail_weight", 0.2),
    ):
        bad = copy.deepcopy(current)
        bad["arguments"][key] = value
        with pytest.raises(ValueError, match="controlled exact-resume migration"):
            validate_exact_run_contract(
                saved, bad, migration_id="wmrm_static_constraint_weight_4_to_2_v1"
            )


def test_static_constraint_weight_cli_is_semantic_and_migration_id_operational() -> None:
    _, optimizer = _model_and_optimizer()
    args = parse_args(
        [
            "--wam4va", "--visual-world-supervision",
            "--wmrm-world-weight", "1.0",
            "--wmrm-static-constraint-weight", "2.0",
            "--wmrm-detach-proposal-stage-state",
            "--resume-exact", "checkpoint.pt",
            "--resume-exact-contract-migration",
            "wmrm_static_constraint_weight_4_to_2_v1",
        ]
    )
    config = SimpleNamespace(
        num_layers=8, action_horizon=48, wmrm=True,
        wmrm_detach_proposal_stage_state=True,
    )
    contract = build_exact_run_contract(args, config, optimizer, _sampler())
    assert contract["arguments"]["wmrm_static_constraint_weight"] == 2.0
    assert contract["arguments"]["wmrm_world_weight"] == 1.0
    assert "resume_exact_contract_migration" not in contract["arguments"]


def test_controlled_world_weight_migration_id_is_operational_not_semantic() -> None:
    _, optimizer = _model_and_optimizer()
    args = parse_args(
        [
            "--wam4va",
            "--wmrm-world-weight",
            "0.5",
            "--resume-exact",
            "checkpoint.pt",
            "--resume-exact-contract-migration",
            "wmrm_world_weight_1_to_0_5_v1",
        ]
    )
    config = SimpleNamespace(num_layers=8, action_horizon=48, wmrm=True)
    contract = build_exact_run_contract(args, config, optimizer, _sampler())

    assert args.resume_exact_contract_migration == "wmrm_world_weight_1_to_0_5_v1"
    assert contract["arguments"]["wmrm_world_weight"] == 0.5
    assert "resume_exact_contract_migration" not in contract["arguments"]


def test_world_weight_migrated_restore_preserves_model_adamw_sampler_and_rng() -> None:
    model, optimizer = _model_and_optimizer()
    sampler = _sampler()
    _seed_training_rngs()
    _update(model, optimizer, sampler)
    saved_contract = _contract()
    saved_contract["arguments"]["wmrm_world_weight"] = 1.0
    checkpoint = {"model": copy.deepcopy(model.state_dict())}
    checkpoint.update(build_exact_resume_state(optimizer, 1, sampler, saved_contract))
    checkpoint_snapshot = copy.deepcopy(checkpoint)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(())

    restored_model, restored_optimizer = _model_and_optimizer()
    restored_sampler = _sampler()
    restored_model.load_state_dict(checkpoint["model"], strict=True)
    current_contract = copy.deepcopy(saved_contract)
    current_contract["arguments"]["wmrm_world_weight"] = 0.5
    step = restore_exact_resume_state(
        checkpoint,
        restored_optimizer,
        restored_sampler,
        runtime_exact_run_contract=current_contract,
        migration_id="wmrm_world_weight_1_to_0_5_v1",
    )

    assert step == 1
    assert restored_optimizer.state_dict() == optimizer.state_dict()
    assert restored_sampler.state_dict() == sampler.state_dict()
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(()), expected_torch, rtol=0.0, atol=0.0)
    assert checkpoint.keys() == checkpoint_snapshot.keys()
    assert checkpoint["exact_run_contract"] == checkpoint_snapshot["exact_run_contract"]
    for name, value in checkpoint_snapshot["model"].items():
        torch.testing.assert_close(checkpoint["model"][name], value, rtol=0.0, atol=0.0)
        torch.testing.assert_close(restored_model.state_dict()[name], value, rtol=0.0, atol=0.0)


def test_world_weight_migration_restore_refuses_before_mutating_training_state() -> None:
    model, optimizer = _model_and_optimizer()
    sampler = _sampler()
    _seed_training_rngs()
    _update(model, optimizer, sampler)
    saved_contract = _contract()
    saved_contract["arguments"]["wmrm_world_weight"] = 1.0
    checkpoint = build_exact_resume_state(optimizer, 1, sampler, saved_contract)

    fresh_model, fresh_optimizer = _model_and_optimizer()
    del fresh_model
    fresh_sampler = _sampler()
    optimizer_snapshot = copy.deepcopy(fresh_optimizer.state_dict())
    sampler_snapshot = copy.deepcopy(fresh_sampler.state_dict())
    python_rng_snapshot = random.getstate()
    numpy_rng_snapshot = np.random.get_state()
    torch_rng_snapshot = torch.get_rng_state().clone()
    current_contract = copy.deepcopy(saved_contract)
    current_contract["arguments"]["wmrm_world_weight"] = 0.5
    current_contract["arguments"]["flow_tail_weight"] = 0.2

    with pytest.raises(ValueError, match="flow_tail_weight"):
        restore_exact_resume_state(
            checkpoint,
            fresh_optimizer,
            fresh_sampler,
            runtime_exact_run_contract=current_contract,
            migration_id="wmrm_world_weight_1_to_0_5_v1",
        )

    assert fresh_optimizer.state_dict() == optimizer_snapshot
    assert fresh_sampler.state_dict() == sampler_snapshot
    assert random.getstate() == python_rng_snapshot
    current_numpy_rng = np.random.get_state()
    assert current_numpy_rng[0] == numpy_rng_snapshot[0]
    np.testing.assert_array_equal(current_numpy_rng[1], numpy_rng_snapshot[1])
    assert current_numpy_rng[2:] == numpy_rng_snapshot[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_snapshot)


def test_exact_contract_allows_checkpoint_archive_policy_changes() -> None:
    _, optimizer = _model_and_optimizer()
    baseline_args = parse_args(["--single-task"])
    changed_args = parse_args(["--single-task", "--save-step-copies"])
    config = SimpleNamespace(num_layers=8, action_horizon=48)

    baseline = build_exact_run_contract(
        baseline_args, config, optimizer, _sampler()
    )
    current = build_exact_run_contract(
        changed_args, config, optimizer, _sampler()
    )

    assert baseline == current


def test_visual_world_exact_resume_binds_fixed_action_donors() -> None:
    identity = {
        "manifest_sha256": "m" * 64,
        "source_sha256": "s" * 64,
        "world_action_donor_sha256": "d" * 64,
        "world_action_donor_transitions": 3297,
        "world_action_rank_transitions": 2931,
    }
    contract = {
        "world_supervision": WORLD_SUPERVISION_CONTRACT,
        "world_transition": WORLD_TRANSITION_CONTRACT,
        "world_loss_weights": WORLD_LOSS_COMPONENT_WEIGHTS,
        "world_stage_auxiliary_decay": WORLD_STAGE_AUXILIARY_DECAY,
        "world_no_regression": WORLD_NO_REGRESSION,
        "world_static_copy_constraint": WORLD_STATIC_COPY_CONSTRAINT,
        "world_action_ranking": WORLD_ACTION_RANKING,
        "world_action_donor_contract": WORLD_ACTION_DONOR_CONTRACT,
        "world_action_donor_sha256": identity["world_action_donor_sha256"],
        "world_action_donor_transitions": identity[
            "world_action_donor_transitions"
        ],
        "world_action_rank_transitions": identity[
            "world_action_rank_transitions"
        ],
        "world_logged_branch": WORLD_LOGGED_BRANCH_CONTRACT,
        "split_manifest_sha256": identity["manifest_sha256"],
        "split_source_sha256": identity["source_sha256"],
    }
    checkpoint = {"training_contract": contract}
    validate_visual_world_resume_contract(checkpoint, identity)

    final_ranking = world_action_ranking_contract("final")
    final_checkpoint = {
        "training_contract": {
            **contract,
            "world_action_ranking": final_ranking,
        }
    }
    validate_visual_world_resume_contract(
        final_checkpoint, identity, final_ranking
    )
    with pytest.raises(ValueError, match="world_action_ranking"):
        validate_visual_world_resume_contract(final_checkpoint, identity)

    changed = {**checkpoint, "training_contract": {**contract}}
    changed["training_contract"]["world_action_donor_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="world_action_donor_sha256"):
        validate_visual_world_resume_contract(changed, identity)
