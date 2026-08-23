from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from eval_metaworld import (
    _action_trace_metrics,
    _append_peer_trace_token,
    _assembly_trace_state,
    _condition_trace_summary,
    _peer_trace_stage_metrics,
    _peer_world_effect_metrics,
    _peer_world_trace_readout,
    _peer_world_trace_stages,
    _validate_peer_eval_trace,
)


def test_action_trace_separates_xyz_gripper_clip_and_saturation() -> None:
    world = np.asarray([[2.0, 0.0, -2.0, 0.5], [0.0, 0.0, 0.0, 2.0]])
    flow = np.asarray([[0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0, -2.0]])

    trace = _action_trace_metrics(world, flow)

    np.testing.assert_allclose(trace["distance"]["xyz_preclip_l2"], [2**0.5 * 2, 0])
    np.testing.assert_allclose(trace["distance"]["xyz_postclip_l2"], [2**0.5, 0])
    np.testing.assert_allclose(trace["distance"]["gripper_preclip_abs"], [1, 4])
    np.testing.assert_allclose(trace["distance"]["gripper_postclip_abs"], [1, 2])
    assert trace["saturation_disagreement"] == {
        "xyz": [True, False],
        "gripper": [False, False],
    }


class _AssemblyEnv:
    _target_pos = np.asarray([0.5, 0.5, 0.2])
    tcp_center = np.asarray([0.4, 0.4, 0.3])

    def __init__(self, nut: list[float]) -> None:
        self.nut = np.asarray(nut)

    def _get_site_pos(self, name: str) -> np.ndarray:
        assert name in {"RoundNut", "RoundNut-8"}
        return self.nut if name == "RoundNut" else self.nut + [0.0, 0.0, 0.01]

    def _get_pos_objects(self) -> np.ndarray:
        return self.nut + [0.0, 0.0, 0.01]


def test_assembly_trace_exact_geometry_threshold_sign_site_and_info_success() -> None:
    boundary = _assembly_trace_state(_AssemblyEnv([0.52, 0.5, 0.1]), {"success": 0})
    hooked = _assembly_trace_state(_AssemblyEnv([0.519, 0.5, 0.1]), {"success": 1})
    above_target = _assembly_trace_state(_AssemblyEnv([0.5, 0.5, 0.3]), {})

    np.testing.assert_allclose(boundary["xy_radius"], 0.02)
    np.testing.assert_allclose(boundary["z_gap"], 0.1)
    assert boundary["aligned"] is False
    assert boundary["hooked"] is True
    assert boundary["success"] is False
    assert hooked["aligned"] is True
    assert hooked["success"] is True
    assert above_target["hooked"] is False
    assert hooked["reward_z_condition"] is True
    np.testing.assert_allclose(hooked["metric_object"], [0.519, 0.5, 0.11])


def _decision() -> dict:
    return {
        "decision": 3,
        "flow_raw": [[2.0, -2.0, 0.25, 0.0], [0.1, 0.2, 0.3, 0.4]],
        "tokens": [],
        "executed_token_count": 0,
    }


def _append(decision: dict, token: int, normalized: np.ndarray) -> None:
    _append_peer_trace_token(
        decision,
        token=token,
        env_step=10 + token,
        normalized_command=normalized,
        denormalized_command=normalized * 2,
        pre_tcp=np.zeros(3),
        post_tcp=np.ones(3),
        reward=0.5,
        pre_assembly={"phase": "pre"},
        assembly={"success": False},
        terminated=token == 0,
        truncated=False,
    )


def test_trace_decision_token_env_step_alignment_clip_and_early_count() -> None:
    decision = _decision()
    _append(decision, 0, np.asarray([1.0, -1.0, 0.25, 0.0]))

    token = decision["tokens"][0]
    assert token["decision"] == decision["decision"] == 3
    assert token["token"] == 0
    assert token["env_step"] == 10
    assert token["normalized_command"] == [1.0, -1.0, 0.25, 0.0]
    assert token["terminal"] is True
    assert decision["executed_token_count"] == 1


def test_trace_rejects_wrong_or_mismatched_token() -> None:
    with pytest.raises(RuntimeError, match="next planned token"):
        _append(_decision(), 1, np.asarray([0.1, 0.2, 0.3, 0.4]))
    with pytest.raises(RuntimeError, match="does not match"):
        _append(_decision(), 0, np.zeros(4))


def test_peer_world_trace_readout_uses_final_pre_action_and_full_horizon() -> None:
    pre_actions = [torch.full((1, 6, 3), 1.0), torch.full((1, 6, 3), 2.0)]
    full_horizon = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    model = SimpleNamespace(
        _wmrm_inject_layers=lambda: [5, 2],
        last_wmrm_pre_actions=pre_actions,
        last_wmrm_auxes=[
            SimpleNamespace(env_action=full_horizon.clone()),
            SimpleNamespace(env_action=full_horizon.clone()),
        ],
        last_wmrm=SimpleNamespace(env_action=torch.full((1, 2, 4), -2.0)),
        world_action_readout=mock.Mock(return_value=full_horizon),
    )

    stages = _peer_world_trace_stages(model)
    stage, readout = _peer_world_trace_readout(model)

    assert model.world_action_readout.call_count == 4
    assert [row["stage"] for row in stages] == [2, 5]
    np.testing.assert_array_equal(stages[0]["operative"], full_horizon[0].numpy())
    assert stage == 5
    assert readout.shape == (6, 4)
    np.testing.assert_array_equal(readout, full_horizon[0].numpy())


def test_all_stage_and_same_noise_metrics_preserve_action_identity() -> None:
    flow = np.asarray([[0.0, 0.5, -0.5, 0.0], [1.5, 0.0, 0.0, 1.0]])
    stages = [
        {"stage": 0, "readout": flow.copy(), "operative": flow.copy()},
        {
            "stage": 1,
            "readout": flow + 0.25,
            "operative": flow + 0.5,
        },
    ]

    serialized = _peer_trace_stage_metrics(stages, flow)
    effect = _peer_world_effect_metrics(flow, flow + 0.25)

    assert serialized[0]["operative_matches_readout"] is True
    assert serialized[1]["operative_matches_readout"] is False
    assert serialized[0]["stage_ordinal"] == 0
    assert serialized[1]["stage_count"] == 2
    assert serialized[0]["operative_vs_flow"]["distance"]["xyz_postclip_l2"] == [0.0, 0.0]
    assert effect["world_on_flow_raw"] == flow.tolist()
    assert effect["world_off_flow_raw"] == (flow + 0.25).tolist()


def test_formal_h15_trace_has_seven_complete_operational_stages() -> None:
    flow = np.zeros((15, 4), dtype=np.float32)
    stages = [
        {
            "stage": index,
            "readout": np.full((15, 4), index / 10, dtype=np.float32),
            "operative": np.full((15, 4), index / 10, dtype=np.float32),
        }
        for index in range(7)
    ]

    serialized = _peer_trace_stage_metrics(stages, flow)

    assert len(serialized) == 7
    assert [row["stage_ordinal"] for row in serialized] == list(range(7))
    assert all(row["stage_count"] == 7 for row in serialized)
    assert all(np.asarray(row["operative_raw"]).shape == (15, 4) for row in serialized)


def test_condition_trace_summary_keeps_prefix_tail_and_token0() -> None:
    condition = torch.arange(1 * 15 * 3, dtype=torch.float32).reshape(1, 15, 3)

    trace = _condition_trace_summary(condition)

    assert trace["horizon"] == 15
    assert trace["hidden_dim"] == 3
    assert trace["token0"] == condition[0, 0].tolist()
    np.testing.assert_allclose(trace["prefix_mean"], condition[0, :6].mean(0))
    np.testing.assert_allclose(trace["tail_mean"], condition[0, 6:].mean(0))


def test_trace_world_off_incompatible_and_production_record_json_round_trip() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _validate_peer_eval_trace(
            output_json=Path("trace.json"),
            va_world_mode="peer_sync_h6",
            peer_world_off=True,
        )

    world = np.asarray([[0.9, -0.8, 0.7, -0.6], [0.5, -0.4, 0.3, -0.2]])
    flow = np.asarray([[1.2, -1.3, 0.25, 0.0], [0.1, 0.2, 0.3, 0.4]])
    decision = {
        "decision": 3,
        "env_step_start": 10,
        "world_stage": 5,
        **_action_trace_metrics(world, flow),
        "denormalized_command": (np.clip(flow, -1.0, 1.0) * 2).tolist(),
        "executed_token_count": 0,
        "tokens": [],
    }
    geometry = _assembly_trace_state(_AssemblyEnv([0.49, 0.5, 0.1]), {"success": 1})
    _append_peer_trace_token(
        decision,
        token=0,
        env_step=10,
        normalized_command=np.clip(flow[0], -1.0, 1.0),
        denormalized_command=np.clip(flow[0], -1.0, 1.0) * 2,
        pre_tcp=np.asarray([0.1, 0.2, 0.3]),
        post_tcp=np.asarray([0.2, 0.3, 0.4]),
        reward=0.5,
        pre_assembly=geometry,
        assembly=geometry,
        terminated=False,
        truncated=False,
    )
    record = {
        "task_id": 0,
        "task": "assemble nut",
        "trial": 1,
        "seed": 7,
        "success": True,
        "peer_eval_trace": [decision],
    }

    restored = json.loads(json.dumps(record, allow_nan=False))
    token = restored["peer_eval_trace"][0]["tokens"][0]
    assert restored == record
    assert set(record) == {"task_id", "task", "trial", "seed", "success", "peer_eval_trace"}
    assert set(geometry) <= set(token)
    assert token["terminal"] is True
    assert decision["executed_token_count"] == 1
    assert all(
        math.isfinite(value)
        for value in (
            token["reward"],
            token["xy_radius"],
            token["z_gap"],
            *token["target_pos"],
            *token["round_nut"],
            *token["pre_tcp"],
            *token["post_tcp"],
            *token["normalized_command"],
            *token["denormalized_command"],
        )
    )
