from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from scripts.eval_wam4va_world_action import (
    _checkpoint_world_contract,
    _index_visual_memory,
    _load_model_and_metric,
    _negate_ci,
    _proposal_pre_step_memory,
    _validate_fixed_eval_payload,
    _world_forward,
    check_target_permutation_invariance,
    evaluate_go_no_go,
    nearest_episode_shuffle,
    paired_episode_bootstrap,
    task_macro_paired_episode_bootstrap,
)
from va_compound.model import VisualMemory
from va_compound.wmrm import WAMState


class _TinyMemoryWorld:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            action_horizon=48, action_dim=4, va_world_mode="legacy"
        )
        self.current_memories: list[VisualMemory | None] = []
        self.decode_noises: list[torch.Tensor] = []
        self.last_wmrm_auxes: list[SimpleNamespace] = []

    def encode_condition(
        self,
        vision: torch.Tensor,
        _proprio: torch.Tensor,
        _previous_action: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, VisualMemory]:
        batch = vision.shape[0]
        condition = torch.zeros(batch, 48, 2, dtype=vision.dtype)
        memory = kwargs.get("visual_memory")
        return_memory = bool(kwargs.get("return_visual_memory", False))
        if return_memory:
            prior = (
                torch.zeros(batch, 1, 2, dtype=vision.dtype, requires_grad=True)
                if memory is None
                else memory.layers[0]
            )
            prior_world_state = (
                None if memory is None else memory.world_state
            )
            prior_belief = (
                torch.zeros(batch, 1, 2, dtype=vision.dtype, requires_grad=True)
                if prior_world_state is None or prior_world_state.belief is None
                else prior_world_state.belief
            )
            next_memory = VisualMemory(
                layers=(prior + vision[:, :1, :2] + 1.0,),
                world_state=WAMState(belief=prior_belief + 1.0),
            )
            return condition, next_memory

        self.current_memories.append(memory)
        env_action = kwargs["env_action"]
        prediction = env_action.mean(dim=(1, 2))[:, None, None, None]
        self.last_wmrm_auxes = [SimpleNamespace(z_tokens=prediction)]
        return condition

    def decode_actions(
        self,
        _condition: torch.Tensor,
        *,
        steps: int,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        assert steps == 8
        self.decode_noises.append(noise.clone())
        return noise


def test_t_greater_than_zero_conditions_share_proposal_pre_step_memory() -> None:
    model = _TinyMemoryWorld()
    batch = 2
    vision = torch.arange(batch * 3 * 2 * 2, dtype=torch.float32).reshape(
        batch, 3, 2, 2
    )
    proprio = torch.zeros(batch, 3, 4)
    previous_action = torch.zeros(batch, 3, 4)
    rows = torch.tensor([7, 19])

    memory = _proposal_pre_step_memory(
        model,
        vision,
        proprio,
        previous_action,
        object(),
        None,
        None,
        None,
        rows,
        2,
        seed=11,
    )

    assert memory is not None
    assert len(model.decode_noises) == 2
    assert not memory.layers[0].requires_grad
    forward_args = (
        model,
        vision[:, 2],
        proprio[:, 2],
        previous_action[:, 2],
        object(),
        None,
        None,
        None,
    )
    actions = (
        torch.ones(batch, 6, 4),
        torch.full((batch, 6, 4), 2.0),
        torch.zeros(batch, 6, 4),
    )
    for action in actions:
        _world_forward(*forward_args, action, visual_memory=memory)

    assert len(model.current_memories) == 3
    assert all(received is memory for received in model.current_memories)
    for received in model.current_memories:
        assert received is not None
        torch.testing.assert_close(received.layers[0], memory.layers[0], rtol=0, atol=0)

    target = torch.tensor([[[[1.0]]], [[[4.0]]]])

    def forward_with_target(loss_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maps = _world_forward(
            *forward_args, actions[0], visual_memory=memory
        )
        return maps, (maps[-1] - loss_target).square().flatten(1).mean(1)

    result = check_target_permutation_invariance(forward_with_target, target)
    assert result["passed"] is True
    assert model.current_memories[-2] is memory
    assert model.current_memories[-1] is memory


def test_index_visual_memory_selects_every_tensor_field() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    memory = VisualMemory(
        layers=(base, base + 10),
        evidence=base + 20,
        task=base + 30,
        task_spec=base + 40,
        pending_future=base[:, 0] + 50,
        gate=0.25,
        world_state=WAMState(
            belief=base + 60,
            innovation=base + 70,
            world_map=torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2),
        ),
    )

    selected = _index_visual_memory(memory, torch.tensor([2, 0]))

    assert selected is not None
    assert selected.gate == pytest.approx(0.25)
    torch.testing.assert_close(selected.layers[0], base[[2, 0]])
    torch.testing.assert_close(selected.layers[1], (base + 10)[[2, 0]])
    torch.testing.assert_close(selected.evidence, (base + 20)[[2, 0]])
    torch.testing.assert_close(selected.task, (base + 30)[[2, 0]])
    torch.testing.assert_close(selected.task_spec, (base + 40)[[2, 0]])
    torch.testing.assert_close(selected.pending_future, (base[:, 0] + 50)[[2, 0]])
    torch.testing.assert_close(
        selected.world_state.belief, (base + 60)[[2, 0]]
    )
    torch.testing.assert_close(
        selected.world_state.innovation, (base + 70)[[2, 0]]
    )
    torch.testing.assert_close(
        selected.world_state.world_map,
        torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)[[2, 0]],
    )


def test_peer_proposal_history_uses_one_encode_and_no_decode() -> None:
    model = _TinyMemoryWorld()
    model.config.va_world_mode = "peer_sync_h6"
    batch, history_steps = 2, 3
    sequence_steps = history_steps + 1
    vision = torch.ones(batch, sequence_steps, 2, 2)
    proprio = torch.zeros(batch, sequence_steps, 4)
    previous_action = torch.zeros(batch, sequence_steps, 4)

    memory = _proposal_pre_step_memory(
        model,
        vision,
        proprio,
        previous_action,
        object(),
        None,
        None,
        None,
        torch.tensor([3, 5]),
        history_steps,
        seed=17,
    )

    assert memory is not None
    assert len(model.decode_noises) == 0
    assert len(model.current_memories) == 0
    torch.testing.assert_close(
        memory.layers[0], torch.full((batch, 1, 2), 6.0)
    )
    assert memory.world_state is not None
    torch.testing.assert_close(
        memory.world_state.belief, torch.full((batch, 1, 2), 3.0)
    )
    assert not memory.world_state.belief.requires_grad


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("num_layers", 7),
        ("action_horizon", 47),
        ("wmrm_inject", "last"),
        ("wmrm_handshake", False),
        ("wmrm_predictor", "legacy"),
        ("wmrm_map_size", 8),
        ("wmrm_map_channels", 32),
        ("wmrm_world_grid", 8),
    ],
)
def test_model_loader_rejects_non_gate_architecture(
    field: str, invalid: object
) -> None:
    config = {
        "wmrm": True,
        "wmrm_target": "dino",
        "wmrm_cycle_steps": 6,
        "main_vision_grid": 16,
        "main_vision_frames": 4,
        "main_vision_dim": 1024,
        "num_layers": 8,
        "action_horizon": 48,
        "wmrm_inject": "all",
        "wmrm_handshake": True,
        "wmrm_predictor": "st_blocks",
        "wmrm_map_size": 16,
        "wmrm_map_channels": 1024,
        "wmrm_world_grid": 16,
    }
    config[field] = invalid

    with pytest.raises(ValueError, match=field):
        _load_model_and_metric(
            {"config": config, "model": {}}, torch.device("cpu")
        )


def _formal_model_config(*, va_world_mode: str = "legacy") -> dict:
    return {
        "wmrm": True,
        "wmrm_target": "dino",
        "wmrm_cycle_steps": 6,
        "main_vision_grid": 16,
        "main_vision_frames": 4,
        "main_vision_tokens": 1024,
        "main_vision_dim": 1024,
        "num_layers": 8,
        "action_horizon": 6 if va_world_mode == "peer_sync_h6" else 48,
        "action_dim": 4,
        "wmrm_inject": "all",
        "wmrm_handshake": True,
        "wmrm_predictor": "st_blocks",
        "wmrm_map_size": 16,
        "wmrm_map_channels": 1024,
        "wmrm_world_grid": 16,
        "va_world_mode": va_world_mode,
    }


def test_production_loader_accepts_formal_peer_h6_config_and_strict_state() -> None:
    loaded: dict[str, object] = {}

    class _LoadedPolicy:
        def __init__(self, config) -> None:
            loaded["config"] = config

        def to(self, device):
            loaded["device"] = device
            return self

        def eval(self):
            loaded["eval"] = True
            return self

        def load_state_dict(self, state, *, strict):
            loaded["state"] = state
            loaded["strict"] = strict

    state = {"world_action_readout.proj.weight": torch.ones(1)}
    with mock.patch(
        "scripts.eval_wam4va_world_action.VACompoundPolicy", _LoadedPolicy
    ):
        model, metric_head, relation_encoder, config = _load_model_and_metric(
            {
                "config": _formal_model_config(va_world_mode="peer_sync_h6"),
                "model": state,
            },
            torch.device("cpu"),
        )

    assert model is not None
    assert metric_head is None
    assert relation_encoder is None
    assert config.va_world_mode == "peer_sync_h6"
    assert config.action_horizon == 6
    assert loaded["state"] is state
    assert loaded["strict"] is True


def test_production_loader_preserves_strict_legacy_h48() -> None:
    config = _formal_model_config()
    config["action_horizon"] = 6
    with pytest.raises(
        ValueError,
        match="peer_sync_h6 requires action_horizon=6|action_horizon",
    ):
        _load_model_and_metric(
            {"config": config, "model": {}}, torch.device("cpu")
        )


def test_fixed_eval_action_shape_branches_by_va_world_mode() -> None:
    payload = {
        "actions": torch.zeros(1, 2, 6, 4),
        "previous_action": torch.zeros(1, 2, 4),
        "proprio": torch.zeros(1, 2, 3),
        "language_hidden": torch.zeros(1, 2, 1, 4),
        "instruction_id": torch.tensor([0]),
        "episode_id": torch.tensor([10]),
        "action_valid_mask": torch.ones(1, 2, 6, dtype=torch.bool),
        "recovery_mask": torch.zeros(1, 2, 6, dtype=torch.bool),
        "frame_refs": [["assembly-v3"]],
        "metadata": {},
    }

    with pytest.raises(ValueError, match="split_name"):
        _validate_fixed_eval_payload(
            payload, [0], va_world_mode="peer_sync_h6"
        )
    with pytest.raises(ValueError, match=r"\[N,T,48,4\]"):
        _validate_fixed_eval_payload(payload, [0], va_world_mode="legacy")


def test_checkpoint_world_contract_requires_peer_topology_and_action_source() -> None:
    checkpoint = {
        "config": {"va_world_mode": "peer_sync_h6"},
        "training_contract": {
            "world_supervision": "visual_motion_v1",
            "world_logged_branch": "matched_context_full_forward_v1",
            "va_world_mode": "peer_sync_h6",
            "peer_world_topology": "pre_stage_snapshot_parallel_va_world_v1",
            "peer_world_action_source": (
                "deterministic_readout_main_explicit_env_override_supervision_v1"
            ),
        },
    }
    assert _checkpoint_world_contract(checkpoint, allow_unmarked=False)[2] is True

    for field in ("peer_world_topology", "peer_world_action_source"):
        invalid = {
            **checkpoint,
            "training_contract": dict(checkpoint["training_contract"]),
        }
        invalid["training_contract"][field] = "legacy"
        with pytest.raises(ValueError, match=field):
            _checkpoint_world_contract(invalid, allow_unmarked=False)


def test_checkpoint_world_contract_requires_matched_context_logged_branch() -> None:
    valid = {
        "training_contract": {
            "world_supervision": "visual_motion_v1",
            "world_logged_branch": "matched_context_full_forward_v1",
        }
    }
    assert _checkpoint_world_contract(valid, allow_unmarked=False) == (
        "visual_motion_v1",
        "matched_context_full_forward_v1",
        True,
    )

    old = {
        "training_contract": {
            "world_supervision": "visual_motion_v1",
            "world_logged_branch": "independent_full_forward",
        }
    }
    with pytest.raises(ValueError, match="world_logged_branch"):
        _checkpoint_world_contract(old, allow_unmarked=False)
    assert _checkpoint_world_contract(old, allow_unmarked=True) == (
        "visual_motion_v1",
        "independent_full_forward",
        False,
    )

    constrained = {
        "training_contract": {
            "world_supervision": "visual_motion_constrained_v2",
            "world_logged_branch": "matched_context_full_forward_v1",
            "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
            "world_transition": "current_first6_and_next_first_v1",
            "world_stage_auxiliary_decay": 0.25,
            "world_no_regression": {
                "all_ratio": 1.0,
                "static_ratio": 1.05,
                "weight": 1.0,
            },
            "world_action_ranking": {
                "stage": "final_logged",
                "top10_min_relative_margin": 0.05,
                "top10_strong_relative_margin": 0.10,
                "weight": 0.25,
                "negatives": ["shuffle", "zero"],
            },
        }
    }
    assert _checkpoint_world_contract(constrained, allow_unmarked=False) == (
        "visual_motion_constrained_v2",
        "matched_context_full_forward_v1",
        True,
    )

    constrained_v3 = {
        "training_contract": {
            "world_supervision": "visual_motion_constrained_v3",
            "world_logged_branch": "matched_context_full_forward_v1",
            "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
            "world_transition": "current_first6_and_next_first_v1",
            "world_stage_auxiliary_decay": 0.25,
            "world_no_regression": {
                "all_ratio": 1.0,
                "static_ratio": 1.05,
                "weight": 1.0,
            },
            "world_static_copy_anchor": {
                "static_ratio": 1.05,
                "weight": 1.0,
                "region": "outside_top20",
                "gate": "per_sample",
            },
            "world_action_ranking": {
                "stage": "full_8stage_counterfactual_final",
                "top10_relative_margin": 0.10,
                "weight": 1.0,
                "negatives": ["shuffle", "zero"],
                "schedule": "alternating_global_step_plus_time",
                "rng": "logged_branch_replay",
                "gradient": "final_stage_recompute",
            },
        }
    }
    assert _checkpoint_world_contract(constrained_v3, allow_unmarked=False) == (
        "visual_motion_constrained_v3",
        "matched_context_full_forward_v1",
        True,
    )

    wrong_v3 = {
        "training_contract": {
            **constrained_v3["training_contract"],
            "world_action_ranking": constrained["training_contract"][
                "world_action_ranking"
            ],
        }
    }
    with pytest.raises(ValueError, match="world_action_ranking"):
        _checkpoint_world_contract(wrong_v3, allow_unmarked=False)

    constrained_v4 = {
        "training_contract": {
            "world_supervision": "visual_motion_constrained_v4",
            "world_logged_branch": "matched_context_full_forward_v1",
            "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
            "world_transition": "current_first6_and_next_first_v1",
            "world_stage_auxiliary_decay": 0.25,
            "world_no_regression": {
                "all_ratio": 1.0,
                "static_ratio": 1.05,
                "weight": 1.0,
            },
            "world_static_copy_constraint": {
                "static_ratio": 1.05,
                "weight": 4.0,
                "region": "outside_top20",
                "penalty": "relative_hinge_plus_half_normalized_square_v1",
                "eps": 1e-6,
            },
            "world_action_ranking": {
                "stage": "full_8stage_counterfactual_final",
                "top10_min_relative_margin": 0.05,
                "top10_strong_relative_margin": 0.10,
                "weight": 4.0,
                "negatives": ["shuffle", "zero"],
                "schedule": "both_each_valid_transition",
                "mask": "per_negative_and_both_for_strong",
                "rng": "logged_branch_replay",
                "gradient": "final_stage_recompute",
            },
        }
    }
    assert _checkpoint_world_contract(constrained_v4, allow_unmarked=False) == (
        "visual_motion_constrained_v4",
        "matched_context_full_forward_v1",
        True,
    )

    v4_with_v3_contract = {
        "training_contract": {
            **constrained_v4["training_contract"],
            "world_static_copy_constraint": None,
            "world_static_copy_anchor": constrained_v3["training_contract"][
                "world_static_copy_anchor"
            ],
            "world_action_ranking": constrained_v3["training_contract"][
                "world_action_ranking"
            ],
        }
    }
    with pytest.raises(ValueError, match="world_static_copy_constraint"):
        _checkpoint_world_contract(v4_with_v3_contract, allow_unmarked=False)

    constrained_v5 = {
        "training_contract": {
            "world_supervision": "visual_motion_constrained_v5",
            "world_logged_branch": "matched_context_full_forward_v1",
            "world_loss_weights": {"all": 0.25, "motion": 0.25, "top20": 0.50},
            "world_transition": "current_first6_and_next_first_v1",
            "world_stage_auxiliary_decay": 0.25,
            "world_no_regression": {
                "all_ratio": 1.0,
                "weight": 1.0,
                "components": ["all"],
            },
            "world_static_copy_constraint": {
                "static_ratio": 1.05,
                "weight": 4.0,
                "region": "outside_top20",
                "penalty": "stage_chain_exact_hinge_v1",
                "reduction": "sum_stages_then_masked_transition_mean",
                "boundary": "copy_then_detached_min_previous_copy",
            },
            "world_action_ranking": {
                "stage": "full_8stage_counterfactual_final",
                "top10_min_relative_margin": 0.05,
                "top10_strong_relative_margin": 0.10,
                "weight": 1.0,
                "negatives": ["shuffle", "zero"],
                "schedule": "both_each_valid_transition",
                "mask": "per_negative_and_both_for_strong",
                "rng": "logged_branch_replay",
                "gradient": "wrong_actions_only_detached_real_margin_v1",
            },
        }
    }
    assert _checkpoint_world_contract(constrained_v5, allow_unmarked=False) == (
        "visual_motion_constrained_v5",
        "matched_context_full_forward_v1",
        True,
    )

    v5_with_v4_contract = {
        "training_contract": {
            **constrained_v4["training_contract"],
            "world_supervision": "visual_motion_constrained_v5",
        }
    }
    with pytest.raises(ValueError, match="world_no_regression"):
        _checkpoint_world_contract(v5_with_v4_contract, allow_unmarked=False)
    assert _checkpoint_world_contract(v5_with_v4_contract, allow_unmarked=True) == (
        "visual_motion_constrained_v5",
        "matched_context_full_forward_v1",
        False,
    )

    gap_v6 = {
        "training_contract": {
            **constrained_v5["training_contract"],
            "world_supervision": "visual_motion_gap_v6",
            "world_static_copy_constraint": {
                "static_ratio": 1.05,
                "weight": 4.0,
                "region": "outside_top20",
                "penalty": "copy_budget_hinge_v1",
                "reduction": "stage_aux_weighted_masked_mean",
                "boundary": "1.05_detached_copy_each_stage",
            },
            "world_action_ranking": {
                "stage": "rotating_8stage_direct_matched_context",
                "top10_min_relative_margin": 0.05,
                "weight": 1.0,
                "negatives": ["shuffle"],
                "diagnostic_negatives": ["zero"],
                "context": "logged_stage_detached_pair",
                "gradient": "control_variate_real_minus_shuffle_v1",
                "schedule": "(global_step+time_index)%num_stages",
            },
        }
    }
    assert _checkpoint_world_contract(gap_v6, allow_unmarked=False) == (
        "visual_motion_gap_v6",
        "matched_context_full_forward_v1",
        True,
    )
    gap_v6["training_contract"]["world_action_ranking"]["negatives"] = [
        "shuffle", "zero"
    ]
    with pytest.raises(ValueError, match="world_action_ranking"):
        _checkpoint_world_contract(gap_v6, allow_unmarked=False)

    oracle_stgap_v7 = {
        "training_contract": {
            **constrained_v5["training_contract"],
            "world_supervision": "visual_motion_oracle_stgap_v7",
            "world_static_copy_constraint": {
                "static_ratio": 1.0,
                "weight": 4.0,
                "region": "outside_top20",
                "penalty": "copy_budget_hinge_plus_always_copy_anchor_v1",
                "reduction": "stage_aux_weighted_masked_mean",
                "boundary": "1.00_detached_copy_each_stage",
            },
            "world_action_ranking": {
                "stage": "final_direct_matched_context",
                "top10_min_relative_margin": 0.12,
                "weight": 1.0,
                "negatives": ["shuffle"],
                "diagnostic_negatives": ["zero"],
                "context": "logged_stage_detached_pair",
                "gradient": "oracle_motion_straight_through_exact_gap_v1",
                "schedule": "final_each_valid_transition",
            },
        }
    }
    assert _checkpoint_world_contract(
        oracle_stgap_v7, allow_unmarked=False
    ) == (
        "visual_motion_oracle_stgap_v7",
        "matched_context_full_forward_v1",
        True,
    )

    oracle_static2 = {
        "training_contract": {
            **oracle_stgap_v7["training_contract"],
            "world_static_copy_constraint": {
                **oracle_stgap_v7["training_contract"]["world_static_copy_constraint"],
                "weight": 2.0,
            },
        }
    }
    assert _checkpoint_world_contract(
        oracle_static2, allow_unmarked=False
    ) == (
        "visual_motion_oracle_stgap_v7",
        "matched_context_full_forward_v1",
        True,
    )
    oracle_static2_cap02 = {
        "training_contract": {
            **oracle_static2["training_contract"],
            "world_action_ranking": {
                **oracle_static2["training_contract"]["world_action_ranking"],
                "per_sample_cap": 0.2,
            },
        }
    }
    assert _checkpoint_world_contract(
        oracle_static2_cap02, allow_unmarked=False
    ) == (
        "visual_motion_oracle_stgap_v7",
        "matched_context_full_forward_v1",
        True,
    )
    oracle_static2_cap01 = {
        "training_contract": {
            **oracle_static2_cap02["training_contract"],
            "world_action_ranking": {
                **oracle_static2_cap02["training_contract"]["world_action_ranking"],
                "per_sample_cap": 0.1,
            },
        }
    }
    with pytest.raises(ValueError, match="world_action_ranking"):
        _checkpoint_world_contract(oracle_static2_cap01, allow_unmarked=False)

    oracle_static3 = {
        "training_contract": {
            **oracle_static2["training_contract"],
            "world_static_copy_constraint": {
                **oracle_static2["training_contract"]["world_static_copy_constraint"],
                "weight": 3.0,
            },
        }
    }
    with pytest.raises(ValueError, match="world_static_copy_constraint"):
        _checkpoint_world_contract(oracle_static3, allow_unmarked=False)

    missing_constraints = {
        "training_contract": {
            "world_supervision": "visual_motion_constrained_v2",
            "world_logged_branch": "matched_context_full_forward_v1",
        }
    }
    with pytest.raises(ValueError, match="world_no_regression"):
        _checkpoint_world_contract(missing_constraints, allow_unmarked=False)

    unknown = {
        "training_contract": {
            "world_supervision": "visual_motion_future_v3",
            "world_logged_branch": "matched_context_full_forward_v1",
        }
    }
    with pytest.raises(ValueError, match="world_supervision"):
        _checkpoint_world_contract(unknown, allow_unmarked=False)


def test_nearest_episode_shuffle_is_task_local_cross_episode_and_nearest() -> None:
    actions = torch.arange(6 * 6 * 4, dtype=torch.float32).reshape(6, 6, 4)
    proprio = torch.tensor(
        [
            [0.0, 0.0],   # task 0, ep 10 -> nearest cross-ep row 2
            [0.1, 0.0],   # task 0, ep 10 -> nearest cross-ep row 2
            [0.2, 0.0],   # task 0, ep 11 -> nearest row 1
            [100.0, 0.0], # task 1, ep 20 -> nearest row 4
            [100.1, 0.0], # task 1, ep 21 -> nearest row 3
            [101.0, 0.0], # task 1, ep 22 -> nearest row 4
        ]
    )
    tasks = torch.tensor([0, 0, 0, 1, 1, 1])
    episodes = torch.tensor([10, 10, 11, 20, 21, 22])

    shuffled, donors = nearest_episode_shuffle(actions, proprio, tasks, episodes)

    assert donors.tolist() == [2, 2, 1, 4, 3, 4]
    torch.testing.assert_close(shuffled, actions[donors])
    assert torch.equal(tasks[donors], tasks)
    assert bool((episodes[donors] != episodes).all())


def test_nearest_episode_shuffle_rejects_single_episode_task() -> None:
    with pytest.raises(ValueError, match="different-episode shuffle is impossible"):
        nearest_episode_shuffle(
            torch.zeros(2, 6, 4),
            torch.zeros(2, 3),
            torch.tensor([0, 0]),
            torch.tensor([7, 7]),
        )


def test_paired_episode_bootstrap_uses_episode_means_and_positive_ci() -> None:
    # Episode 1 has many windows but must receive the same weight as episodes 2/3.
    reference = torch.tensor([10.0] * 8 + [1.0, 1.0])
    candidate = reference + torch.tensor([1.0] * 8 + [2.0, 3.0])
    episodes = torch.tensor([1] * 8 + [2, 3])
    tasks = torch.zeros(10, dtype=torch.long)

    result = paired_episode_bootstrap(
        candidate,
        reference,
        episodes,
        task_ids=tasks,
        n_resamples=1000,
        seed=4,
    )

    assert result["difference"]["estimate"] == pytest.approx(2.0)
    assert result["difference"]["low"] > 0.0
    assert result["n_episodes"] == 3
    assert result["sampling_unit"] == "episode"
    assert result["episode_weighting"] == "equal"


def test_paired_episode_bootstrap_rejects_cross_task_pooling() -> None:
    with pytest.raises(ValueError, match="within exactly one task"):
        paired_episode_bootstrap(
            torch.tensor([2.0, 2.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([10, 20]),
            task_ids=torch.tensor([0, 1]),
            n_resamples=10,
        )


def test_task_macro_bootstrap_equal_weights_tasks_not_windows() -> None:
    # Task 0 contributes 100 windows at error delta 1; task 1 contributes one
    # episode at delta 3. Equal-task macro is 2, not the micro/window mean ~1.
    reference = torch.ones(101)
    candidate = reference + torch.tensor([1.0] * 100 + [3.0])
    episodes = torch.tensor([10] * 100 + [20])
    tasks = torch.tensor([0] * 100 + [1])

    result = task_macro_paired_episode_bootstrap(
        candidate,
        reference,
        episodes,
        tasks,
        n_resamples=50,
        seed=0,
    )
    window_micro = float((candidate - reference).mean())

    assert result["difference"]["estimate"] == pytest.approx(2.0)
    assert window_micro == pytest.approx(103.0 / 101.0)
    assert result["difference"]["estimate"] != pytest.approx(window_micro)
    assert result["relative_difference"]["estimate"] == pytest.approx(2.0)
    assert result["task_weighting"] == "equal"


def test_task_macro_relative_difference_is_macro_of_task_relatives() -> None:
    result = task_macro_paired_episode_bootstrap(
        torch.tensor([1.1, 1.1, 120.0, 120.0]),
        torch.tensor([1.0, 1.0, 100.0, 100.0]),
        torch.tensor([10, 11, 20, 21]),
        torch.tensor([0, 0, 1, 1]),
        n_resamples=100,
        seed=0,
    )

    assert result["relative_difference"]["estimate"] == pytest.approx(0.15)
    assert (
        result["relative_difference"]["aggregation"]
        == "equal_task_macro_of_task_relative_differences"
    )


def test_negated_macro_ci_preserves_aggregation_metadata() -> None:
    result = _negate_ci(
        {
            "estimate": -0.2,
            "low": -0.3,
            "high": -0.1,
            "confidence": 0.95,
            "aggregation": "equal_task_macro_of_task_relative_differences",
        }
    )

    assert result["estimate"] == pytest.approx(0.2)
    assert result["low"] == pytest.approx(0.1)
    assert result["high"] == pytest.approx(0.3)
    assert result["aggregation"] == "equal_task_macro_of_task_relative_differences"


def test_target_permutation_checker_detects_target_leakage() -> None:
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    fixed_prediction = torch.tensor([[5.0], [6.0]])

    invariant = check_target_permutation_invariance(
        lambda supplied: (
            fixed_prediction.clone(),
            (fixed_prediction - supplied[:, :1]).square(),
        ),
        target,
    )
    leaked = check_target_permutation_invariance(
        lambda supplied: (
            supplied[:, :1],
            (fixed_prediction - supplied[:, :1]).square(),
        ),
        target,
    )

    assert invariant["passed"] is True
    assert invariant["prediction_bitwise_equal"] is True
    assert leaked["passed"] is False
    assert leaked["max_abs_prediction_difference"] > 0.0


def test_target_permutation_requires_an_observed_loss_change() -> None:
    target = torch.tensor([[1.0], [4.0]])
    prediction = torch.tensor([[2.0], [3.0]])

    prediction_only = check_target_permutation_invariance(
        lambda _supplied: prediction, target
    )
    unchanged_loss = check_target_permutation_invariance(
        lambda _supplied: (prediction, torch.zeros(2)), target
    )

    assert prediction_only["passed"] is False
    assert prediction_only["loss_path_valid"] is False
    assert unchanged_loss["passed"] is False
    assert unchanged_loss["loss_changed"] is False

    nan_loss = check_target_permutation_invariance(
        lambda _supplied: (prediction, torch.full((2,), float("nan"))), target
    )
    assert nan_loss["passed"] is False
    assert nan_loss["loss_finite"] is False


def test_target_permutation_checker_confirms_only_loss_changes() -> None:
    target = torch.tensor([[1.0], [4.0]])
    prediction = torch.tensor([[2.0], [3.0]])

    result = check_target_permutation_invariance(
        lambda supplied: (prediction, (prediction - supplied).square()), target
    )

    assert result["passed"] is True
    assert result["prediction_bitwise_equal"] is True
    assert result["loss_changed"] is True


def _ci(value: float, *, low: float | None = None, high: float | None = None) -> dict:
    return {
        "estimate": value,
        "low": value if low is None else low,
        "high": value if high is None else high,
        "confidence": 0.95,
    }


def _gate_macro(value: float = 0.12, *, n_tasks: int = 2) -> dict:
    relative = _ci(value)
    relative["aggregation"] = "equal_task_macro_of_task_relative_differences"
    return {
        "relative_gain_top10": relative,
        "n_tasks": n_tasks,
        "task_ids": [0, 16] if n_tasks == 2 else [0],
        "bootstrap": {
            "sampling_unit": "episode_within_task",
            "task_weighting": "equal",
            "paired": True,
            "n_resamples": 4000,
        },
    }


def _passing_task() -> dict:
    return {
        "task_id": 0,
        "n_episodes": 3,
        "bootstrap": {
            "sampling_unit": "episode",
            "episode_weighting": "equal",
            "paired": True,
            "n_resamples": 4000,
        },
        "metrics": {
            "world_all": _ci(0.9),
            "copy_all": _ci(1.0),
            "relative_gain_top10": _ci(0.12),
            "world_static": _ci(1.02),
            "copy_static": _ci(1.0),
        },
        "action_dependency": {
            "shuffle_relative_degradation_top10": _ci(0.06),
            "zero_relative_degradation_top10": _ci(0.11),
            "shuffle_minus_real_top10": _ci(0.02, low=0.001),
            "zero_minus_real_top10": _ci(0.04, low=0.002),
        },
        "target_permutation": {
            "passed": True,
            "target_changed": True,
            "target_finite": True,
            "prediction_bitwise_equal": True,
            "prediction_finite": True,
            "max_abs_prediction_difference": 0.0,
            "loss_changed": True,
            "loss_finite": True,
            "loss_path_valid": True,
        },
    }


def test_gate_passes_only_when_every_task_passes() -> None:
    assembly = _passing_task()
    door = _passing_task()
    door["task_id"] = 16
    reports = {"assembly-v3": assembly, "door-unlock-v3": door}
    gate = evaluate_go_no_go(reports, _gate_macro())
    assert gate["decision"] == "GO"
    assert gate["passed"] is True

    # A high task-macro result cannot hide one task's missing action dependence.
    reports["door-unlock-v3"]["action_dependency"][
        "shuffle_relative_degradation_top10"
    ] = _ci(0.01)
    failed = evaluate_go_no_go(reports, _gate_macro(0.20))
    assert failed["decision"] == "NO-GO"
    assert failed["per_task"]["door-unlock-v3"][
        "shuffle_top10_ge_5pct_worse"
    ] is False


def test_gate_rejects_missing_task_or_non_95pct_action_ci() -> None:
    task = _passing_task()
    missing_task = evaluate_go_no_go(
        {"assembly-v3": task},
        _gate_macro(n_tasks=1),
    )
    assert missing_task["passed"] is False
    assert missing_task["task_count_matches"] is False

    task["action_dependency"]["shuffle_minus_real_top10"]["confidence"] = 0.50
    door = _passing_task()
    door["task_id"] = 16
    invalid_ci = evaluate_go_no_go(
        {"assembly-v3": task, "door-unlock-v3": door},
        _gate_macro(),
    )
    assert invalid_ci["passed"] is False
    assert invalid_ci["per_task"]["assembly-v3"][
        "action_difference_ci_is_95pct"
    ] is False


def test_gate_rejects_swapped_task_names_even_when_id_set_matches() -> None:
    assembly = _passing_task()
    door = _passing_task()
    door["task_id"] = 16

    gate = evaluate_go_no_go(
        {"door-unlock-v3": assembly, "assembly-v3": door},
        _gate_macro(),
    )

    assert gate["passed"] is False
    assert gate["task_count_matches"] is True
    assert gate["task_identity_matches"] is False


def test_gate_rejects_window_bootstrap_unpaired_ci_and_wrong_ci_direction() -> None:
    assembly = _passing_task()
    door = _passing_task()
    door["task_id"] = 16

    assembly["bootstrap"]["sampling_unit"] = "window"
    assembly["bootstrap"]["paired"] = False
    assembly["action_dependency"]["shuffle_minus_real_top10"]["low"] = -0.001
    gate = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door},
        _gate_macro(),
    )

    assert gate["passed"] is False
    assert gate["per_task"]["assembly-v3"][
        "episode_bootstrap_contract_valid"
    ] is False
    assert gate["per_task"]["assembly-v3"][
        "shuffle_ci_direction_positive"
    ] is False


def test_gate_rejects_non_episode_task_macro() -> None:
    assembly = _passing_task()
    door = _passing_task()
    door["task_id"] = 16
    macro = _gate_macro()
    macro["bootstrap"]["sampling_unit"] = "window"

    gate = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door}, macro
    )

    assert gate["passed"] is False
    assert gate["macro_bootstrap_contract_valid"] is False


def test_gate_rejects_truncated_smoke_as_formal_go() -> None:
    assembly = _passing_task()
    door = _passing_task()
    door["task_id"] = 16

    gate = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door},
        _gate_macro(),
        full_heldout_evaluation=False,
    )

    assert gate["decision"] == "NO-GO"
    assert gate["passed"] is False
    assert gate["full_heldout_evaluation"] is False


def test_gate_rejects_unmarked_checkpoint_diagnostic_as_formal_go() -> None:
    assembly = _passing_task()
    door = _passing_task()
    door["task_id"] = 16

    gate = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door},
        _gate_macro(),
        checkpoint_world_supervision_valid=False,
    )

    assert gate["decision"] == "NO-GO"
    assert gate["passed"] is False
    assert gate["checkpoint_world_supervision_valid"] is False

    old_branch = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door},
        _gate_macro(),
        checkpoint_world_logged_branch_valid=False,
    )
    assert old_branch["decision"] == "NO-GO"
    assert old_branch["passed"] is False
    assert old_branch["checkpoint_world_logged_branch_valid"] is False


def test_gate_rejects_degenerate_episode_ci_or_incomplete_target_check() -> None:
    assembly = _passing_task()
    assembly["n_episodes"] = 1
    door = _passing_task()
    door["task_id"] = 16
    degenerate = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door},
        _gate_macro(),
    )
    assert degenerate["passed"] is False
    assert degenerate["per_task"]["assembly-v3"][
        "episode_bootstrap_has_multiple_episodes"
    ] is False

    assembly = _passing_task()
    assembly["target_permutation"] = {"passed": True}
    incomplete = evaluate_go_no_go(
        {"assembly-v3": assembly, "door-unlock-v3": door},
        _gate_macro(),
    )
    assert incomplete["passed"] is False
    assert incomplete["per_task"]["assembly-v3"][
        "target_permutation_bitwise_invariant"
    ] is False

    wrong_macro = _gate_macro()
    wrong_macro["relative_gain_top10"].pop("aggregation")
    unlabelled = evaluate_go_no_go(
        {"assembly-v3": _passing_task(), "door-unlock-v3": door}, wrong_macro
    )
    assert unlabelled["passed"] is False
    assert unlabelled["macro_relative_gain_is_task_macro"] is False


def test_gate_requires_bitwise_target_invariance_per_task() -> None:
    task = _passing_task()
    task["target_permutation"] = {"passed": False}
    door = _passing_task()
    door["task_id"] = 16
    gate = evaluate_go_no_go(
        {"assembly-v3": task, "door-unlock-v3": door},
        _gate_macro(),
    )
    assert gate["passed"] is False
    assert gate["per_task"]["assembly-v3"][
        "target_permutation_bitwise_invariant"
    ] is False
