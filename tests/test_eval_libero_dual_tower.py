from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import torch

import eval_libero_closedloop
from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.policy.model import VisualMemory


def test_eval_libero_dual_tower_closedloop_decisions(monkeypatch):
    config = VACompoundConfig(
        language_dim=12,
        vision_dim=8,
        hidden_dim=16,
        num_layers=3,
        num_heads=4,
        action_dim=7,
        proprio_dim=9,
        action_horizon=6,
        planning_stride=6,
        deployment_execution_horizon=3,
        flow_layers=3,
        architecture_version="dual_tower_expert_v1",
        fusion_pair_count=2,
        wmrm=False,
        main_vision_grid=2,
        main_vision_tokens=20,
        main_vision_frames=5,
        main_vision_dim=8,
    )
    model = VACompoundPolicy(config).eval()

    class MockLiberoEnv:
        def __init__(self):
            self.step_count = 0

        def seed(self, seed: int):
            pass

        def reset(self):
            pass

        def set_init_state(self, init_state):
            return {
                "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((32, 32, 3), dtype=np.uint8),
                "robot0_joint_pos": np.zeros(7, dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }

        def step(self, action: np.ndarray):
            self.step_count += 1
            obs = self.set_init_state(None)
            done = False
            return obs, 0.0, done, {}

        def check_success(self) -> bool:
            return False

    frontend_calls = 0

    def mock_encode_dual_tower_batch(
        frames, instructions, vision, text, fusion, device, *, grid=16
    ):
        nonlocal frontend_calls
        frontend_calls += 1
        val = float(frontend_calls)
        tokens = torch.full((1, 1, 20, 8), val, dtype=torch.float32)
        language = torch.full((1, 1, 4, 12), val, dtype=torch.float32)
        mask = torch.ones((1, 1, 4), dtype=torch.bool)
        return tokens, language, mask

    monkeypatch.setattr(
        eval_libero_closedloop,
        "encode_dual_tower_batch",
        mock_encode_dual_tower_batch,
    )

    mock_normalized_state = MagicMock(return_value=np.zeros(9, dtype=np.float32))
    monkeypatch.setattr(
        eval_libero_closedloop,
        "_normalized_state",
        mock_normalized_state,
    )

    passed_visual_memories = []
    original_encode_condition = model.encode_condition

    def spy_encode_condition(*args, **kwargs):
        passed_visual_memories.append(kwargs.get("visual_memory"))
        return original_encode_condition(*args, **kwargs)

    model.encode_condition = spy_encode_condition

    built_caches = []
    original_build_language_cache = model.build_language_cache

    def spy_build_language_cache(*args, **kwargs):
        cache = original_build_language_cache(*args, **kwargs)
        built_caches.append(cache)
        return cache

    model.build_language_cache = spy_build_language_cache

    env = MockLiberoEnv()
    success, steps = eval_libero_closedloop.rollout_trial(
        model=model,
        vision=MagicMock(),
        language_cache=None,
        cross_modal_language_layers=None,
        env=env,
        init_state=np.zeros(10, dtype=np.float32),
        device=torch.device("cpu"),
        state_q01=np.zeros(9, dtype=np.float32),
        state_q99=np.ones(9, dtype=np.float32),
        horizon=6,
        flow_steps=3,
        settle_steps=0,
        memory_reset_every=0,
        policy_seed=42,
        previous_action_zero=False,
        dual_view=True,
        joint_text=MagicMock(),
        instruction="test instruction",
    )

    assert steps == 6
    assert not success

    # Verify frontend called once per decision (2 decisions total), not per flow integration step (2*3=6)
    assert frontend_calls == 2

    # Verify language cache was refreshed each decision with decision-dependent language
    assert len(built_caches) == 2
    assert built_caches[0].tokens.shape == built_caches[1].tokens.shape
    assert not torch.allclose(built_caches[0].tokens, built_caches[1].tokens)

    # Verify visual memory propagated across decisions
    assert len(passed_visual_memories) == 2
    assert passed_visual_memories[0] is None
    assert passed_visual_memories[1] is not None
    assert isinstance(passed_visual_memories[1], VisualMemory)
