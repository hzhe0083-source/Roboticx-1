from __future__ import annotations

from pathlib import Path

import torch

from va_compound.training.model_setup import migrate_peer_h15_to_h50_state
from va_compound.policy.model import VACompoundConfig, VACompoundPolicy


ROOT = Path(__file__).resolve().parents[1]


def _config(action_horizon: int) -> VACompoundConfig:
    return VACompoundConfig(
        language_dim=16,
        vision_dim=8,
        hidden_dim=16,
        num_layers=3,
        num_heads=4,
        action_horizon=action_horizon,
        planning_stride=15,
        deployment_execution_horizon=15,
        action_dim=4,
        proprio_dim=4,
        flow_layers=2,
        wmrm=True,
        wmrm_world_dim=8,
        wmrm_inject="all",
        wmrm_target="dino",
        wmrm_cycle_steps=15,
        wmrm_map_size=2,
        wmrm_map_channels=8,
        wmrm_world_grid=2,
        main_vision_frames=1,
        main_vision_grid=2,
        main_vision_tokens=4,
        va_world_mode="peer_sync_h6",
        tail_flow_condition_grad=action_horizon == 50,
    )


def test_h15_to_h50_migration_preserves_old_policy_prefix() -> None:
    torch.manual_seed(4)
    source = VACompoundPolicy(_config(15)).eval()
    checkpoint = {
        "config": source.config.__dict__,
        "model": source.state_dict(),
    }
    torch.manual_seed(9)
    target = VACompoundPolicy(_config(50)).eval()
    target.load_state_dict(
        migrate_peer_h15_to_h50_state(target, checkpoint), strict=True
    )

    assert target.layers[0].protected_action_prefixes == (6, 15)
    for key, value in target.extension_flow_head.state_dict().items():
        torch.testing.assert_close(
            value, source.tail_flow_head.state_dict()[key], rtol=0, atol=0
        )

    vision = torch.randn(2, 4, 8)
    proprio = torch.randn(2, 4)
    previous = torch.randn(2, 4)
    language = torch.randn(2, 3, 16)
    mask = torch.ones(2, 3, dtype=torch.bool)
    source_condition, source_memory = source.encode_condition(
        vision,
        proprio,
        previous,
        language_hidden=language,
        language_mask=mask,
        return_visual_memory=True,
    )
    source_env_actions = [aux.env_action.clone() for aux in source.last_wmrm_auxes]
    target_condition, target_memory = target.encode_condition(
        vision,
        proprio,
        previous,
        language_hidden=language,
        language_mask=mask,
        return_visual_memory=True,
    )
    torch.testing.assert_close(
        target_condition[:, :15], source_condition, rtol=1e-6, atol=1e-6
    )
    assert len(source_env_actions) == len(target.last_wmrm_auxes) == 2
    for source_action, target_aux in zip(
        source_env_actions, target.last_wmrm_auxes, strict=True
    ):
        assert source_action.shape == target_aux.env_action.shape == (2, 15, 4)
        torch.testing.assert_close(
            target_aux.env_action, source_action, rtol=1e-6, atol=1e-6
        )
    torch.testing.assert_close(
        target_memory.world_state.world_map,
        source_memory.world_state.world_map,
        rtol=1e-6,
        atol=1e-6,
    )

    noise15 = torch.randn(2, 15, 4)
    noise50 = torch.cat((noise15, torch.randn(2, 35, 4)), dim=1)
    flow_time = torch.tensor([0.2, 0.8])
    source_velocity = source.flow_velocity(source_condition, noise15, flow_time)
    target_velocity = target.flow_velocity(target_condition, noise50, flow_time)
    torch.testing.assert_close(
        target_velocity[:, :15], source_velocity, rtol=1e-6, atol=1e-6
    )


def test_h50_world_reads_only_the_executed_p15_prefix() -> None:
    model = VACompoundPolicy(_config(50))
    assert model.world_action_readout.horizon == model.wmrm.cycle_steps == 15
    assert model.config.action_horizon == 50


def test_stage1_launcher_is_action_only_and_defaults_to_evo_steps() -> None:
    text = (ROOT / "scripts/run_mw_mt50_h50_p15_stage1_v1.sh").read_text()
    assert "H50_STEPS=${H50_STEPS:-10000}" in text
    assert "SAVE_EVERY=${SAVE_EVERY:-1000}" in text
    assert '--data "$ONLINE_INDEX"' in text
    assert "--online-action-horizon 50" in text
    assert "--deployment-execution-horizon 15" in text
    assert "--wmrm-cycle-steps 15" in text
    assert "--va-only" in text
    assert "--pcgrad" in text
    assert "--visual-world-supervision" not in text
    assert "--world-data" not in text
    assert "--vision-unfreeze" not in text
    assert 'resume_args=(--resume-exact "$SAVE")' in text


def test_joint_full3_launcher_unfreezes_all_three_trainable_branches() -> None:
    text = (
        ROOT / "scripts/run_mw_mt50_h50_p15_joint_full3_v1.sh"
    ).read_text()
    assert "EPOCHS=3" in text
    assert "BATCH=${BATCH:-32}" in text
    assert "--va-data" in text and "--world-data" in text
    assert "--visual-world-supervision" in text
    assert "--pcgrad --pcgrad-separate-world" in text
    assert "--zero-redundancy-optimizer" in text
    assert "--vision-unfreeze-all --lr-vision 0.000001" in text
    assert "--va-only" not in text
    assert "peer_h50_action_only_to_joint_weights_only_v1" in text
    assert "--save-step-copies" not in text
    assert 'if [[ -n "${RUN_STEPS_OVERRIDE:-}" ]]; then' in text
    assert "RUN_STEPS_OVERRIDE > 0 && completed > 0" in text


def test_stable_joint_launcher_freezes_dino_and_samples_recovery() -> None:
    text = (
        ROOT / "scripts/run_mw_mt50_h50_p15_joint_stable3_v1.sh"
    ).read_text()
    assert "EXPECTED_EPISODES=${EXPECTED_EPISODES:-3470}" in text
    assert "ONLINE_RECOVERY_SAMPLES=${ONLINE_RECOVERY_SAMPLES:-2}" in text
    assert "TRAIN_DINO=${TRAIN_DINO:-0}" in text
    assert "WMRM_FEATURE_METRIC=${WMRM_FEATURE_METRIC:-cosine}" in text
    assert "MODEL_LR=${MODEL_LR:-0.000003}" in text
    assert "WMRM_PREDICTOR_LR=${WMRM_PREDICTOR_LR:-0.00001}" in text
    assert "MAIN_VISION_ENCODE_BATCH=${MAIN_VISION_ENCODE_BATCH:-16}" in text
    assert "LONGTRAJ_DECODE_CACHE_TASKS=${LONGTRAJ_DECODE_CACHE_TASKS:-149}" in text
    assert "PEER_PREFETCH_DEPTH=${PEER_PREFETCH_DEPTH:-4}" in text
