"""CPU contracts for the objective used inside the MetaWorld update loop."""
import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from va_compound.training import engine
from va_compound.training.config import parse_args


def _objective(monkeypatch, *, objective):
    tree = ast.parse(inspect.getsource(engine.run_metaworld))
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                   engine.__file__, "exec")
    args = parse_args([])
    args.wmrm_world_weight = 0.5
    model = SimpleNamespace(last_wmrm_loss=torch.tensor(6.0, requires_grad=True))
    velocity = torch.tensor(2.0, requires_grad=True)
    calls = []

    def rollout(*a, **kwargs):
        calls.append(kwargs)
        return velocity, None

    def flow(*a, **kwargs):
        if objective == "world":
            pytest.fail("World stream must not evaluate action-label FM loss")
        return velocity.square(), velocity.square(), velocity.square()

    namespace = dict(vars(engine))
    namespace.update(args=args, model=model, next_global_step=1,
                     text_backbone=None, scene_teacher=None, tasks=None,
                     servo=None, servo_stats=None,
                     rollout_policy=rollout, masked_flow_matching_loss=flow)
    exec(code, namespace)
    result = namespace["compute_loss"]({}, None, None, None, objective=objective)
    return result, model, velocity, calls


def test_world_objective_skips_fm_reduction_and_action_decode(monkeypatch):
    result, model, velocity, calls = _objective(monkeypatch, objective="world")
    assert result[0].item() == 3.0
    assert all(item.item() == 0 for item in result[2:5])
    assert calls[0]["compute_action_output"] is False
    result[0].backward()
    assert model.last_wmrm_loss.grad.item() == 0.5
    assert velocity.grad is None


def test_va_objective_has_no_world_loss_gradient(monkeypatch):
    result, model, velocity, calls = _objective(monkeypatch, objective="va")
    assert result[0].item() == 4.0
    assert calls[0]["train_world_model"] is False
    result[0].backward()
    assert velocity.grad.item() == 4.0
    assert model.last_wmrm_loss.grad is None
