from __future__ import annotations

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
