import pytest
import torch
from types import SimpleNamespace

from va_compound.data.libero import ACTION_HORIZON, DECISION_OFFSETS, EXECUTION_HORIZON, SEQUENCE, WORLD_HORIZON, _attach_dense_world_action_donors, _StaticAnchorDataset, _validate_run_schedule, _validate_dense_source
from train_libero import CROSS_MODAL_VA_LAYERS, _fresh_config, _initialize_scratch_fusion, _local_task_ids, _parser, _prepare_batch, _stage2_enabled, validate_cross_modal_language_contract
from va_compound import VACompoundConfig


def test_libero_fresh_config_uses_no_metaworld_action_shapes():
    assert not hasattr(_parser().parse_args(["preflight"]), "base")
    config = _fresh_config(
        va_last3_cross_attn=True,
        dino_qwen_cross_modal_bridge=True,
    )
    assert (config.action_horizon, config.action_dim, config.proprio_dim) == (50, 7, 9)
    assert (config.main_vision_frames, config.main_vision_tokens) == (5, 1280)
    assert (config.language_dim, config.hidden_dim, config.flow_hidden_dim) == (1024, 1024, 512)
    assert (config.wmrm_world_dim, config.wmrm_predictor_width) == (1024, 1024)
    assert config.num_layers == 8
    assert config.wmrm and config.wmrm_reads_fused_va_tokens
    assert config.wmrm_cycle_steps == 15
    assert config.va_last3_cross_attn and config.dino_qwen_cross_modal_bridge
    assert config.dino_qwen_cross_modal_layers == 6
    assert CROSS_MODAL_VA_LAYERS == [0, 1, 2, 3, 4, 5]


def test_scratch_fusion_starts_as_identity_without_dead_scalar_gates():
    from va_compound.policy.model import EvoCrossAttentionBlock

    readers = torch.nn.ModuleList([EvoCrossAttentionBlock(8, 2) for _ in range(3)])
    model = SimpleNamespace(
        dino_qwen_bridge=SimpleNamespace(
            vision_readers=readers[:1],
            language_readers=readers[1:2],
            gates=torch.nn.Parameter(torch.zeros(1, 2)),
        ),
        va_last3_readout=SimpleNamespace(
            readers=readers[2:],
            gates=torch.nn.Parameter(torch.zeros(1)),
        ),
    )
    _initialize_scratch_fusion(model)
    assert torch.equal(model.dino_qwen_bridge.gates, torch.ones(1, 2))
    assert torch.equal(model.va_last3_readout.gates, torch.ones(1))
    query, context = torch.randn(2, 4, 8), torch.randn(2, 5, 8)
    for reader in readers:
        assert torch.equal(reader(query, context), query)
    readers[0](query, context).square().mean().backward()
    assert readers[0].attn.out_proj.weight.grad.abs().sum() > 0
    assert readers[0].ff[-1].weight.grad.abs().sum() > 0


def test_peer_world_accepts_libero_action_dimension():
    config = VACompoundConfig(
        action_horizon=15,
        action_dim=7,
        proprio_dim=9,
        num_layers=8,
        planning_stride=15,
        deployment_execution_horizon=15,
        wmrm=True,
        wmrm_cycle_steps=15,
        wmrm_inject="all",
        va_world_mode="peer_sync_h6",
    )
    assert config.action_dim == 7


def test_libero_contract_is_h50_p15():
    config = VACompoundConfig(
        action_horizon=ACTION_HORIZON,
        action_dim=7,
        proprio_dim=9,
        num_layers=8,
        planning_stride=EXECUTION_HORIZON,
        deployment_execution_horizon=EXECUTION_HORIZON,
        wmrm=True,
        wmrm_cycle_steps=WORLD_HORIZON,
        wmrm_inject="all",
        va_world_mode="peer_sync_h6",
    )
    assert config.action_horizon == 50
    assert config.planning_stride == 15
    assert config.wmrm_cycle_steps == WORLD_HORIZON == 15
    assert SEQUENCE == 8
    assert DECISION_OFFSETS.tolist() == [0, 15, 30, 45, 60, 75, 90, 105]


def test_world_target_uses_the_same_dino_with_stop_gradient(monkeypatch):
    vision = object()
    calls = []

    def encode(frames, backbone, device, *, grid, window, **kwargs):
        calls.append(backbone)
        return torch.zeros(1, SEQUENCE, window * grid * grid, 1024, requires_grad=True)

    monkeypatch.setattr("train_libero._dino_main_online_encode", encode)
    batch = _prepare_batch(
        {
            "frames": torch.zeros(1, SEQUENCE, 5, 1, 1, 3),
            "world_target_frames": torch.zeros(1, SEQUENCE, 1, 1, 1, 3),
            "actions": torch.zeros(1, SEQUENCE, 50, 7),
            "previous_action": torch.ones(1, SEQUENCE, 7),
        },
        vision=vision,
        device=torch.device("cpu"),
        encode_batch=1,
        world=True,
        prev_dropout=1.0,
        layerwise_cross_modal=False,
    )
    assert calls == [vision, vision]
    assert batch["world_target_map"].shape == (1, SEQUENCE, 1024, 16, 16)
    assert not batch["world_target_map"].requires_grad


def test_static_anchor_dataset_decodes_negative_anchor_ids():
    wrapped = _StaticAnchorDataset(["zero", "one", "two"])
    assert wrapped[-1] == "zero"
    assert wrapped[-3] == "two"


def test_dense_world_donors_are_cross_episode_and_linear_layout():
    actions = torch.arange(4 * 2 * 15, dtype=torch.float32).reshape(4, 2, 15, 1)
    payload = {
        "actions": actions,
        "instruction_id": torch.zeros(4, dtype=torch.long),
        "episode_id": torch.tensor([0, 0, 1, 1]),
        "crop_start": torch.tensor([0, 1, 0, 1]),
        "metadata": {},
    }
    _attach_dense_world_action_donors(payload)
    assert torch.equal(
        payload["world_rank_shuffle_action"][:2], actions[2:, :, :WORLD_HORIZON]
    )
    assert torch.equal(
        payload["world_rank_shuffle_action"][2:], actions[:2, :, :WORLD_HORIZON]
    )
    assert bool(payload["world_rank_shuffle_mask"].all())


def test_cross_modal_language_contract_requires_full_qwen_last_six():
    validate_cross_modal_language_contract(
        {
            "qwen_keep_layers": 24,
            "qwen_fusion_layers": list(range(18, 24)),
            "qwen_base_readout": "layer23_final_norm",
            "qwen_fusion_reduce": "none",
            "language_dim": 1024,
            "language_source": "online_qwen35_0_8b_last6_full_v1",
        }
    )
    with pytest.raises(ValueError, match="full Qwen"):
        validate_cross_modal_language_contract(
            {
                "qwen_keep_layers": 24,
                "qwen_fusion_layers": list(range(10, 15)),
                "qwen_base_readout": "layer23_final_norm",
                "qwen_fusion_reduce": "none",
                "language_dim": 1024,
                "language_source": "online_qwen35_0_8b_last6_full_v1",
            }
        )


def test_two_stage_boundary_is_after_step_8000():
    assert not _stage2_enabled(8000, 8000)
    assert _stage2_enabled(8001, 8000)


def test_libero_long_hard2_t8_schedule_keeps_roughly_5000_updates():
    assert _local_task_ids("4,3") == (3, 4)
    payload = {
        "actions": [None] * 9_843,
        "metadata": {
            "n_tasks": 2,
            "suites": ["libero_10"],
            "task_specs": [
                {"local_task_id": 3},
                {"local_task_id": 4},
            ],
        },
    }
    args = SimpleNamespace(
        batch_size=8,
        mixed_tasks=2,
        stage1_steps=0,
        epochs=4,
        gpus=2,
        anchor_fraction=0.0,
        max_steps=None,
    )
    assert _validate_run_schedule(payload, args) == (
        1231,
        4924,
        "two_task_t8_dense_local2_deferred_v4",
    )


def test_dense_continuation_requires_completed_source():
    source = {
        "global_step": 1_000,
        "qwen_trainable_state_dict": {"q": torch.tensor(1)},
        "main_vision_trainable_state_dict": {"v": torch.tensor(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "training_contract": {
            "initialization": "dense_all_windows_continue_from_s5000_v1",
            "data_contract": "libero_4suite_h50p15_t4_dualview5_worldh15_va1024_qwen08_last6_denseall_v7",
            "source_global_step": 5_000,
            "total_steps": 4_955,
            "phase": "stage2_qwen_va_dino_joint",
            "action_horizon": 50,
            "deployment_execution_horizon": 15,
            "wmrm_cycle_steps": 15,
            "qwen_training": "last6_full_layers18_23_v1",
            "main_vision_joint_trained": True,
        },
    }
    _validate_dense_source(source)
    source["global_step"] = 800
    with pytest.raises(ValueError, match="source mismatch"):
        _validate_dense_source(source)
