from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_libero as trainer


def test_fresh_dual_configuration_and_parser():
    assert trainer._parser().parse_args(["train"]).architecture_version == "legacy"
    config = trainer._fresh_config(architecture_version="dual_tower_expert_v1")
    trainer._validate_model_contract(config)
    assert config.architecture_version == "dual_tower_expert_v1"
    assert config.flow_layers == 3
    assert not config.dino_qwen_cross_modal_bridge
    assert not config.va_last3_cross_attn


@pytest.mark.parametrize("architecture,contract", [
    ("legacy", trainer.DATA_CONTRACT),
    ("dual_tower_expert_v1", trainer.JOINT_DATA_CONTRACT),
])
def test_train_binds_validated_identity_to_both_samplers(monkeypatch, tmp_path, architecture, contract):
    args = trainer._parser().parse_args([
        "train", "--architecture-version", architecture,
        "--save", str(tmp_path / "checkpoint.pt"),
    ])
    if architecture == "legacy":
        args.resume_weights = tmp_path / "source.pt"
    payload = {
        "metadata": {"contract": contract, "tasks": ["pick", "place", "open", "close"]},
        "instruction_id": torch.arange(4).repeat_interleave(8),
        "episode_id": torch.arange(32),
        "crop_start": torch.zeros(32, dtype=torch.long),
        "anchor_eligible": torch.ones(32, dtype=torch.bool),
    }
    monkeypatch.setattr(trainer, "_validate_data", lambda *a, **kw: payload)
    monkeypatch.setattr(trainer, "_validate_run_schedule", lambda *a: (1, 1, "test"))
    topology = SimpleNamespace(local_rank=0, rank=0, world_size=1, is_distributed=False)
    monkeypatch.setattr(trainer, "resolve_world_topology", lambda: topology)
    monkeypatch.setattr(trainer, "initialize", lambda *a: None)
    monkeypatch.setattr(trainer, "shutdown", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "LongTrajFramesDataset", lambda *a, **kw: SimpleNamespace(payload=payload))
    monkeypatch.setattr(trainer, "_StaticAnchorDataset", lambda dataset: dataset)
    monkeypatch.setattr(trainer, "_sha256_file", lambda path: "test-data-sha")
    samplers = []
    sampler_name = "EpisodeWindowBatchSampler" if architecture == "dual_tower_expert_v1" else "TaskLocalityWeightedSampler"
    original_sampler = getattr(trainer, sampler_name)

    def capture_sampler(*a, **kw):
        sampler = original_sampler(*a, **kw)
        samplers.append(sampler)
        return sampler

    class ReachedLoader(Exception):
        pass

    def stop_at_loader(*a, **kw):
        raise ReachedLoader

    monkeypatch.setattr(trainer, sampler_name, capture_sampler)
    monkeypatch.setattr(trainer, "DataLoader", stop_at_loader)
    with pytest.raises(ReachedLoader):
        trainer.train(args)
    assert len(samplers) == 2
    for sampler in samplers:
        assert sampler.dataset_content_identity == {
            "contract": contract, "sha256": "test-data-sha",
        }


def test_joint_batch_preserves_language_graph_and_detaches_world_target(monkeypatch):
    fusion_weight = torch.nn.Parameter(torch.tensor(2.0))
    def joint(frames, instructions, vision, text, fusion, device, **kwargs):
        assert instructions == ["pick"]
        return (fusion_weight * torch.ones(1, 8, 1280, 4),
                fusion_weight * torch.ones(1, 8, 3, 6),
                torch.ones(1, 8, 3, dtype=torch.bool))
    def target(*args, **kwargs):
        assert not torch.is_grad_enabled()
        return torch.ones(1, 8, 256, 4)
    monkeypatch.setattr(trainer, "encode_dual_tower_batch", joint)
    monkeypatch.setattr(trainer, "_dino_main_online_encode", target)
    raw = dict(frames=np.zeros((1,8,5,2,2,3),dtype=np.uint8),
               world_target_frames=np.zeros((1,8,1,2,2,3),dtype=np.uint8),
               instruction_id=torch.tensor([0]), actions=torch.zeros(1,8,50,7),
               previous_action=torch.ones(1,8,7))
    batch = trainer._prepare_batch(raw, vision=None, device=torch.device("cpu"),
                                   encode_batch=5, world=True, prev_dropout=1,
                                   layerwise_cross_modal=False, joint_text=None,
                                   joint_tasks=["pick"], joint_fusion=object())
    assert batch["language_hidden"].requires_grad
    assert not batch["world_target_map"].requires_grad
    assert batch["world_target_map"].shape == (1,8,4,16,16)
    trainer._encode_language(batch, None, [], layerwise_cross_modal=False)
    batch["language_hidden"].sum().backward()
    assert fusion_weight.grad > 0
    assert torch.count_nonzero(batch["previous_action"]) == 0


def test_joint_checkpoint_restores_next_adamw_update(tmp_path):
    from tests.test_layerwise_expert_policy import config
    from va_compound import VACompoundConfig, VACompoundPolicy

    torch.manual_seed(13)
    model = VACompoundPolicy(config(world_state_supervision=True)).eval()
    text, vision = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": parameters[:1]}, {"params": parameters[1:]},
        {"params": text.parameters()}, {"params": vision.parameters()},
    ], lr=1e-4)
    condition = torch.randn(1, 3, 50, 16)
    noise, time = torch.randn(1, 50, 4), torch.rand(1)
    def update(policy, optim):
        optim.zero_grad(set_to_none=True)
        policy.flow_velocity(condition, noise, time).square().mean().backward()
        optim.step()
    update(model, optimizer)
    path = tmp_path / "joint.pt"
    sampler = SimpleNamespace(state_dict=lambda: {"epoch": 0, "batch_cursor": 1})
    trainer._atomic_checkpoint(
        path, config=model.config, model=model, text_backbone=text, vision=vision,
        optimizer=optimizer, step=1, total_steps=10, stage1_steps=0, encode_batch=5,
        qwen_sha256="test-qwen", dino_sha256="test-dino", data_sha256="test-data",
        run_metadata={"suites": ["libero_10"], "n_tasks": 2, "task_specs": []},
        pcgrad_forward_grouping="test", source_checkpoint="fresh_dual_tower_expert_v1",
        source_global_step=-1, sampler=sampler, world_sampler=sampler,
        episode_runtime_states=[{"action": trainer.EpisodeMemoryBank().state_dict(),
                                 "world": trainer.EpisodeMemoryBank().state_dict()}],
    )
    saved = torch.load(path, weights_only=True)
    assert saved["training_contract"]["memory_reset_every"] == 0
    assert saved["training_contract"]["memory_contract"] == "episode_tbptt8_v1"
    assert saved["training_contract"]["world_state_loss_weight"] == 1.0
    assert saved["episode_runtime_states"][0]["action"]["entries"] == {}
    assert saved["training_contract"]["execution_gradient_contract"] == "p15_live_h50_tail_detached_v1"
    assert saved["training_contract"]["data_contract"] == trainer.JOINT_DATA_CONTRACT
    assert saved["training_contract"]["optimizer_initialization"] == "fresh_adamw_v1"
    assert saved["training_contract"]["qwen_world_cache"] == "per_observation_joint_live_v1"
    assert saved["training_contract"]["stage1_world_current_vision_cache"] == "disabled_joint_frontend_v1"
    restored = VACompoundPolicy(VACompoundConfig(**saved["config"])).eval()
    restored.load_state_dict(saved["model"], strict=True)
    restored_parameters = list(restored.parameters())
    restored_text, restored_vision = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    trainer._load_selected_state(restored_text, saved["qwen_trainable_state_dict"])
    trainer._load_selected_state(restored_vision, saved["main_vision_trainable_state_dict"])
    restored_optimizer = torch.optim.AdamW([
        {"params": restored_parameters[:1]}, {"params": restored_parameters[1:]},
        {"params": restored_text.parameters()}, {"params": restored_vision.parameters()},
    ], lr=1e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    update(model, optimizer)
    update(restored, restored_optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_joint_schedule_allows_two_stage_probe_without_relaxing_legacy():
    import pytest
    from va_compound.data.libero import RUN_SCHEDULE_PROFILES, _validate_run_schedule
    profile = RUN_SCHEDULE_PROFILES[2]
    payload = {"actions": range(8), "metadata": {
        "n_tasks": 2, "suites": profile["suites"], "task_counts": [4, 4],
        "task_specs": [{"local_task_id": i} for i in profile["local_task_ids"]],
    }, "instruction_id": torch.arange(2).repeat_interleave(4),
       "episode_id": torch.arange(8), "crop_start": torch.zeros(8, dtype=torch.long)}
    args = SimpleNamespace(batch_size=profile["batch_size"], mixed_tasks=profile["mixed_tasks"],
                           stage1_steps=10, epochs=1, gpus=2, anchor_fraction=0,
                           max_steps=12, architecture_version="dual_tower_expert_v1")
    per_epoch, total, _ = _validate_run_schedule(payload, args)
    assert total == per_epoch
    args.architecture_version = "legacy"
    with pytest.raises(ValueError, match="schedule mismatch"):
        _validate_run_schedule(payload, args)
