import copy
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from eval_metaworld import OBSERVATION_STRIDE, VISION_WINDOW
from train_rlt_metaworld import (
    PILOT_ACTION_HORIZON,
    ChunkActor,
    MetaWorldRunner,
    RefineConfig,
    ReplayBuffer,
    TwinCritic,
    _assert_identity_residual,
    _collector_device,
    _validate_chunk_length,
    _validate_rlt_inputs,
    _token_task_local_batches,
    chunk_td_target,
    dino_main_frame_indices,
    overlapping_transition_rows,
    recovery_transition_rows,
    run_episode,
    squeeze_action_chunk,
    update_bc,
    update_multitask_rlt,
    update_rlt,
    vision_buffer_length,
)
from va_compound.rlt import RLTokenConfig, RLTokenModule


def _pilot_config(**overrides) -> SimpleNamespace:
    fields = dict(
        va_world_mode="peer_sync_h6",
        wmrm=True,
        slot_free_policy=True,
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_frames=4,
        main_vision_temporal=True,
        local_slots=False,
        dense_readout=False,
        dense_readout_mtvj=False,
        dino_dense_metric=False,
        action_vision_backbone="none",
        plan_resampler=False,
        scene_teacher=False,
        direct_head=False,
        c2_controller=False,
        flow_semantic=False,
        proprio_dim=4,
        action_dim=4,
        action_horizon=PILOT_ACTION_HORIZON,
        planning_stride=PILOT_ACTION_HORIZON,
        deployment_execution_horizon=PILOT_ACTION_HORIZON,
        wmrm_cycle_steps=PILOT_ACTION_HORIZON,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _pilot_features(horizon: int = PILOT_ACTION_HORIZON) -> dict:
    return {
        "metadata": {
            "action_horizon": horizon,
            "planning_stride": horizon,
            "control_stride": horizon,
            "sequence_length": 4,
        },
        "language_hidden": True,
        "language_mask": True,
        "instruction_id": True,
        "normalization": True,
    }


class RLTTest(unittest.TestCase):
    def test_collectors_round_robin_across_visible_cuda_devices(self) -> None:
        self.assertEqual(
            [str(_collector_device("cuda", "0,1", worker)) for worker in range(4)],
            ["cuda:0", "cuda:1", "cuda:0", "cuda:1"],
        )
        self.assertEqual(str(_collector_device("cuda:0", "", 3)), "cuda:0")
        with self.assertRaisesRegex(ValueError, "CUDA learner"):
            _collector_device("cpu", "0,1", 0)
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            _collector_device("cuda", "0,x", 0)

    def test_stacked_replay_round_trip(self) -> None:
        source = ReplayBuffer(4)
        for value in (1.0, 2.0):
            source.add(
                torch.full((3,), value),
                torch.full((2, 1), value),
                torch.full((2, 1), -value),
                torch.tensor([0.0, value]),
                torch.full((3,), value + 1),
                torch.full((2, 1), value + 1),
                value == 2,
                2,
            )
        restored = ReplayBuffer(4)
        restored.extend_stacked(source.stacked())
        self.assertEqual(len(restored), 2)
        for expected, actual in zip(source.stacked(), restored.stacked(), strict=True):
            torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, "9 fields"):
            restored.extend_stacked((torch.zeros(2),))

    def test_replay_balances_tasks_and_keeps_anchor_quota(self) -> None:
        replay = ReplayBuffer(8, [0, 1])
        for task_id in (0, 1):
            for value in range(4):
                replay.add(
                    torch.tensor([float(value)]),
                    torch.zeros(2, 1),
                    torch.zeros(2, 1),
                    torch.zeros(2),
                    torch.tensor([float(value + 1)]),
                    torch.zeros(2, 1),
                    False,
                    2,
                    task_id,
                )
            replay.add(
                torch.tensor([100.0 + task_id]),
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                torch.zeros(2),
                torch.tensor([101.0 + task_id]),
                torch.zeros(2, 1),
                True,
                2,
                task_id,
                anchor=True,
            )
        batch = replay.sample(8, torch.device("cpu"))
        self.assertEqual(torch.bincount(batch[-1], minlength=2).tolist(), [4, 4])
        task0 = replay.sample_task(0, 4, torch.device("cpu"))
        self.assertEqual(int((task0[0][:, 0] == 100).sum()), 1)

    def test_replay_protects_and_samples_whole_success_episodes(self) -> None:
        replay = ReplayBuffer(16, [0, 1])

        def add(value: float, task: int, reward: float, **kwargs) -> None:
            replay.add(
                torch.tensor([value]),
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                torch.tensor([0.0, reward]),
                torch.tensor([value + 1]),
                torch.zeros(2, 1),
                bool(reward),
                2,
                task,
                **kwargs,
            )

        for task in (0, 1):
            add(100 + task, task, 0.0, anchor=True)
        successful_episode = ReplayBuffer(8, [0])
        for index in range(3):
            successful_episode.add(
                torch.tensor([200.0 + index]),
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                torch.tensor([0.0, float(index == 2)]),
                torch.tensor([201.0 + index]),
                torch.zeros(2, 1),
                index == 2,
                2,
                0,
            )
        replay.extend_stacked(successful_episode.stacked(), successful=True)
        for value in range(20):
            add(float(value), 0, 0.0, successful=False)
            add(float(value), 1, 0.0, successful=False)

        task0 = replay.sample_task(0, 8, torch.device("cpu"))
        self.assertEqual(int((task0[0][:, 0] == 100).sum()), 2)
        self.assertEqual(int((task0[0][:, 0] >= 200).sum()), 3)
        self.assertEqual(
            sum(not bool(row[3].any()) for row in replay.online_success[0]), 2
        )
        task1 = replay.sample_task(1, 8, torch.device("cpu"))
        self.assertEqual(len(task1[0]), 8)
        batch = replay.sample(16, torch.device("cpu"))
        self.assertEqual(torch.bincount(batch[-1], minlength=2).tolist(), [8, 8])

    def test_replay_guarantees_positive_reward_per_task_batch(self) -> None:
        replay = ReplayBuffer(128, [0])
        for index in range(20):
            reward = float(index == 0)
            replay.add(
                torch.tensor([float(index)]),
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                torch.tensor([0.0, reward]),
                torch.tensor([float(index + 1)]),
                torch.zeros(2, 1),
                bool(reward),
                2,
                0,
                anchor=True,
            )
        for index in range(20):
            replay.add(
                torch.tensor([float(index)]),
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                torch.zeros(2),
                torch.tensor([float(index + 1)]),
                torch.zeros(2, 1),
                False,
                2,
                0,
                successful=False,
            )
        batch = replay.sample_task(0, 64, torch.device("cpu"))
        self.assertGreaterEqual(int(batch[3].gt(0).any(dim=1).sum()), 4)

    def test_utd_callback_runs_once_per_action_chunk_not_transition(self) -> None:
        class Runner:
            chunk_length = 6
            action_dim = 1
            primitive_steps = 0

            def reset(self, _seed):
                self.primitive_steps = 0
                return torch.tensor(0)

            def policy_state(self):
                return torch.tensor(self.primitive_steps)

            def execute(self, action, *, capture_stride=None):
                actual = 6 if self.primitive_steps == 0 else 4
                start = self.primitive_steps
                self.primitive_steps += actual
                intermediate = [
                    (offset, torch.tensor(start + offset))
                    for offset in range(capture_stride, actual, capture_stride)
                ]
                return (
                    torch.zeros(6),
                    self.primitive_steps == 10,
                    False,
                    actual,
                    action,
                    intermediate,
                )

        transitions = []
        chunks = []
        run_episode(
            Runner(),
            0,
            lambda _state: torch.zeros(6, 1),
            lambda *row: transitions.append(row),
            lambda: chunks.append(1),
            transition_stride=2,
        )
        self.assertEqual(len(transitions), 4)
        self.assertEqual(len(chunks), 2)

    def test_token_batches_cover_rows_once_without_cross_task_cache_thrash(self) -> None:
        tasks = [0, 0, 0, 1, 1, 2]
        batches = _token_task_local_batches(tasks, list(range(6)), 2, seed=3)
        self.assertEqual(
            sorted(index for batch in batches for index in batch), list(range(6))
        )
        self.assertTrue(
            all(len({tasks[index] for index in batch}) == 1 for batch in batches)
        )

    def test_vision_indices_match_eval_dino_window(self) -> None:
        self.assertEqual(VISION_WINDOW, 4)
        self.assertEqual(OBSERVATION_STRIDE, 2)
        self.assertEqual(dino_main_frame_indices(), [-7, -5, -3, -1])
        self.assertEqual(
            dino_main_frame_indices(),
            list(range(-2 * VISION_WINDOW + 1, 0, 2)),
        )
        self.assertEqual(vision_buffer_length(), 7)

    def test_stride2_sparse_frames_equal_dense_eval_window(self) -> None:
        dense = deque([0] * vision_buffer_length(), maxlen=vision_buffer_length())
        sparse = deque([0] * VISION_WINDOW, maxlen=VISION_WINDOW)
        for step in range(1, 13):
            dense.append(step)
            if step % OBSERVATION_STRIDE == 0:
                sparse.append(step)
                self.assertEqual(
                    [dense[index] for index in dino_main_frame_indices()],
                    list(sparse),
                )

    def test_runner_resets_world_state_every_four_committed_decisions(self) -> None:
        class Model:
            config = SimpleNamespace(
                main_vision_frames=4, main_vision_grid=1, action_dim=4
            )

            def __init__(self) -> None:
                self.memories = []

            def encode_condition(self, *_args, visual_memory=None, **_kwargs):
                self.memories.append(visual_memory)
                return torch.zeros(1, 8), object()

            def decode_actions(self, *_args, **_kwargs):
                return torch.zeros(1, 15, 4)

        model = Model()
        runner = MetaWorldRunner(
            model=model,
            vision=object(),
            language_cache=object(),
            env=object(),
            device=torch.device("cpu"),
            state_q01=torch.zeros(4).numpy(),
            state_scale=torch.ones(4).numpy(),
            action_q01=torch.zeros(4).numpy(),
            action_q99=torch.ones(4).numpy(),
            chunk_length=6,
            flow_steps=8,
            episode_horizon=400,
            world_reset_every=4,
            reward_mode="success",
            reward_scale=1.0,
        )
        runner.frames.extend([torch.zeros(1).numpy()] * 7)
        runner.obs = torch.zeros(4).numpy()
        reset_memory = object()
        with (
            patch(
                "train_rlt_metaworld._main_vision_encode_window",
                return_value=torch.zeros(1, 1, 1),
            ),
            patch(
                "train_rlt_metaworld._apply_local_vision",
                return_value=torch.zeros(1, 1, 1),
            ),
            patch(
                "train_rlt_metaworld._reset_world_state",
                return_value=reset_memory,
            ) as reset,
        ):
            for _ in range(4):
                runner.policy_state()
            runner.policy_state(commit_memory=False)
            self.assertEqual(runner.decision_count, 4)
            reset.assert_not_called()
            runner.policy_state()
        reset.assert_called_once()
        self.assertIs(model.memories[-1], reset_memory)
        self.assertEqual(runner.decision_count, 5)

    def test_squeeze_action_chunk_drops_batch(self) -> None:
        chunk = torch.zeros(1, 6, 4)
        chunk[0, 2, 1] = 0.5
        squeezed = squeeze_action_chunk(chunk, 6, 4)
        self.assertEqual(squeezed.shape, (6, 4))
        self.assertEqual(squeezed[2, 1], 0.5)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            squeeze_action_chunk(torch.zeros(2, 6, 4), 6, 4)

    def test_residual_actor_starts_as_vla_and_dropout_keeps_skip(self) -> None:
        torch.manual_seed(3)
        actor = ChunkActor(8, 6, 4, 32, fixed_std=0.0, residual=True)
        state = torch.randn(2, 8)
        reference = torch.rand(2, 6, 4) * 2 - 1
        torch.testing.assert_close(actor(state, reference), reference)
        torch.testing.assert_close(
            actor(state, reference, drop_reference=True, reference_dropout=1.0),
            reference,
        )
        replay = ReplayBuffer(4)
        replay.add(
            state[0],
            reference[0],
            reference[0],
            torch.zeros(6),
            state[0],
            reference[0],
            False,
            6,
        )
        loss = update_bc(
            replay,
            actor,
            torch.optim.Adam(actor.parameters(), lr=1e-3),
            batch_size=4,
            device=torch.device("cpu"),
            steps=3,
        )
        self.assertEqual(loss, 0.0)
        torch.testing.assert_close(actor(state, reference), reference)
        _assert_identity_residual(actor)
        actor.net[-1].bias.data[0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "changed before online"):
            _assert_identity_residual(actor)

        legacy = ChunkActor(8, 6, 4, 32, fixed_std=0.0)
        for parameter in legacy.parameters():
            parameter.data.zero_()
        torch.testing.assert_close(legacy(state, reference), torch.zeros_like(reference))

    def test_pilot_accepts_full_h15_chunk(self) -> None:
        horizon, stride, flow_steps = _validate_rlt_inputs(
            _pilot_config(), {"flow_steps": 8}, _pilot_features()
        )
        self.assertEqual((horizon, stride, flow_steps), (15, 15, 8))
        with self.assertRaisesRegex(ValueError, "H/P/execution/world"):
            _validate_rlt_inputs(
                _pilot_config(
                    action_horizon=6,
                    planning_stride=6,
                    deployment_execution_horizon=6,
                    wmrm_cycle_steps=6,
                ),
                {"flow_steps": 8},
                _pilot_features(6),
            )
        _validate_chunk_length(6, 15)
        _validate_chunk_length(15, 15)
        with self.assertRaisesRegex(ValueError, "positive"):
            _validate_chunk_length(0)
        with self.assertRaisesRegex(ValueError, "exceed"):
            _validate_chunk_length(16, 15)

    def test_runner_records_dense_reward_and_matches_h15_eval_frames(self) -> None:
        class Model:
            config = SimpleNamespace(
                main_vision_frames=4, main_vision_grid=1, action_dim=4
            )

            def encode_condition(self, *_args, **_kwargs):
                return torch.zeros(1, 8), None

            def decode_actions(self, *_args, **_kwargs):
                return torch.zeros(1, 15, 4)

        class Env:
            def __init__(self) -> None:
                self.steps = 0

            def step(self, _action):
                self.steps += 1
                return np.zeros(4), float(self.steps), False, False, {"success": 0}

            def render(self):
                return np.array([self.steps], dtype=np.float32)

        runner = MetaWorldRunner(
            model=Model(),
            vision=object(),
            language_cache=object(),
            env=Env(),
            device=torch.device("cpu"),
            state_q01=np.zeros(4),
            state_scale=np.ones(4),
            action_q01=np.zeros(4),
            action_q99=np.ones(4),
            chunk_length=15,
            flow_steps=8,
            episode_horizon=15,
            world_reset_every=4,
            reward_mode="dense",
            reward_scale=0.1,
        )
        runner.frames.extend([np.zeros(1, dtype=np.float32)] * 7)
        runner.obs = np.zeros(4)
        rewards, done, success, actual_steps, *_ = runner.execute(
            torch.zeros(15, 4)
        )
        torch.testing.assert_close(rewards, torch.arange(1, 16).float() / 10)
        self.assertTrue(done)
        self.assertFalse(success)
        self.assertEqual(actual_steps, 15)
        captured = []
        with (
            patch(
                "train_rlt_metaworld._main_vision_encode_window",
                side_effect=lambda frames, *_args, **_kwargs: (
                    captured.extend(float(frame[0]) for frame in frames)
                    or torch.zeros(1, 1, 1)
                ),
            ),
            patch(
                "train_rlt_metaworld._apply_local_vision",
                return_value=torch.zeros(1, 1, 1),
            ),
        ):
            runner.policy_state()
        self.assertEqual(captured, [9.0, 11.0, 13.0, 15.0])

    def test_chunk_target_masks_unexecuted_nonzero_rewards(self) -> None:
        target = chunk_td_target(
            rewards=torch.tensor([[0.0, 1.0, 9.0], [2.0, 9.0, 9.0]]),
            done=torch.tensor([1.0, 0.0]),
            actual_steps=torch.tensor([2, 1]),
            next_q=torch.tensor([5.0, 5.0]),
            gamma=0.9,
        )
        torch.testing.assert_close(target, torch.tensor([0.9, 6.5]))

    def test_critic_masks_unexecuted_terminal_action_tail(self) -> None:
        critic = TwinCritic(3, 4, 2, 8)
        state = torch.randn(2, 3)
        first = torch.randn(2, 4, 2)
        second = first.clone()
        second[0, 2:] = 100.0
        second[1, 1:] = -100.0
        steps = torch.tensor([2, 1])
        left = critic(state, first, steps)
        right = critic(state, second, steps)
        torch.testing.assert_close(left, right)

    def test_stride2_rows_cross_chunks_and_use_t_plus_c_endpoint(self) -> None:
        states = {step: torch.tensor(step) for step in (0, 2, 4, 6, 8)}
        planned = torch.arange(12, dtype=torch.float32)[:, None]
        rewards = torch.zeros(10)
        rewards[9] = 1
        first, next_start = overlapping_transition_rows(
            states,
            planned[:6],
            rewards[:6],
            next_start=0,
            executed_steps=6,
            done=False,
            chunk_length=6,
            stride=2,
        )
        second, next_start = overlapping_transition_rows(
            states,
            planned,
            rewards,
            next_start=next_start,
            executed_steps=10,
            done=True,
            chunk_length=6,
            stride=2,
        )
        rows = first + second
        self.assertEqual([row[0] for row in rows], [0, 2, 4, 6])
        self.assertEqual([int(row[4]) for row in rows[:2]], [6, 8])
        self.assertEqual([row[-2] for row in rows], [False, False, True, True])
        self.assertEqual([row[-1] for row in rows], [6, 6, 6, 4])
        for start, _, action, *_ in rows:
            torch.testing.assert_close(action[:, 0], torch.arange(start, start + 6.0))
        torch.testing.assert_close(rows[-1][3], torch.tensor([0, 0, 0, 1, 0, 0.0]))
        self.assertEqual(next_start, 8)  # start 8 has no sampled C6 action window

    def test_recovery_rows_use_intervention_action_as_reference(self) -> None:
        actions = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        state_vectors = {
            step: torch.tensor([float(step), 100 + float(step)])
            for step in range(0, 10, 2)
        }
        valid_recovery = torch.zeros(12, dtype=torch.bool)
        valid_recovery[4:9] = True
        rows = recovery_transition_rows(
            state_vectors,
            actions,
            valid_recovery,
            first_success=8,
            chunk_length=4,
            stride=2,
        )
        self.assertEqual([row[0] for row in rows], [4, 6, 8])
        start4, start6 = rows[:2]
        state, reference = start4[1]
        next_state, _ = start4[4]
        torch.testing.assert_close(state, state_vectors[4])
        torch.testing.assert_close(next_state, state_vectors[8])
        torch.testing.assert_close(reference, actions[4:8])
        torch.testing.assert_close(start4[2], actions[4:8])
        torch.testing.assert_close(start4[4][1], actions[8:12])
        self.assertTrue(start6[-2])
        self.assertEqual(start6[-1], 3)
        torch.testing.assert_close(start6[3], torch.tensor([0, 0, 1, 0.0]))

    def test_partial_bc_ignores_unexecuted_suffix(self) -> None:
        torch.manual_seed(3)
        state_dim, horizon, action_dim = 8, 3, 2
        actor_a = ChunkActor(state_dim, horizon, action_dim, 32, fixed_std=0.0)
        actor_b = copy.deepcopy(actor_a)
        replay_a = ReplayBuffer(8)
        replay_b = ReplayBuffer(8)
        state = torch.zeros(state_dim)
        reference = torch.ones(horizon, action_dim)
        action_a = reference.clone()
        action_b = reference.clone()
        action_b[-1] = -1
        for replay, action in ((replay_a, action_a), (replay_b, action_b)):
            replay.add(
                state,
                reference,
                action,
                torch.tensor([1.0, 0.0, 9.0]),
                state,
                reference,
                True,
                2,
            )
        update_bc(
            replay_a,
            actor_a,
            torch.optim.SGD(actor_a.parameters(), lr=0.1),
            batch_size=1,
            device=torch.device("cpu"),
            steps=1,
        )
        update_bc(
            replay_b,
            actor_b,
            torch.optim.SGD(actor_b.parameters(), lr=0.1),
            batch_size=1,
            device=torch.device("cpu"),
            steps=1,
        )
        for left, right in zip(actor_a.parameters(), actor_b.parameters(), strict=True):
            torch.testing.assert_close(left, right)

    def test_rl_token_reconstructs_with_stop_gradient_and_deploys_encoder(self) -> None:
        torch.manual_seed(3)
        config = RLTokenConfig(
            token_dim=16,
            num_tokens=1,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ff_dim=32,
        )
        module = RLTokenModule(config)
        source = torch.randn(3, 5, 16, requires_grad=True)
        loss = module.reconstruction_loss(source)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(source.grad)
        self.assertIsNotNone(module.rl_tokens.grad)
        deployed = RLTokenModule(config, with_decoder=False)
        deployed.load_encoder_state_dict(module.encoder_state_dict())
        module.eval()
        deployed.eval()
        torch.testing.assert_close(module.encode(source), deployed.encode(source))

    def test_chunk_target_bc_and_update(self) -> None:
        torch.manual_seed(3)
        state_dim, horizon, action_dim = 8, 3, 2
        actor = ChunkActor(state_dim, horizon, action_dim, 32, fixed_std=0.05)
        critic = TwinCritic(state_dim, horizon, action_dim, 32)
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        replay = ReplayBuffer(32)
        for _ in range(8):
            replay.add(
                torch.randn(state_dim),
                torch.randn(horizon, action_dim).clamp(-1, 1),
                torch.randn(horizon, action_dim).clamp(-1, 1),
                torch.zeros(horizon),
                torch.randn(state_dim),
                torch.randn(horizon, action_dim).clamp(-1, 1),
                False,
                horizon,
            )
        actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        bc = update_bc(
            replay, actor, actor_opt, batch_size=4, device=torch.device("cpu"), steps=1
        )
        self.assertTrue(torch.isfinite(torch.tensor(bc)))
        before = actor.net[-1].weight.detach().clone()
        step, metrics = update_rlt(
            replay,
            actor,
            critic,
            target_critic,
            actor_opt,
            torch.optim.Adam(critic.parameters(), lr=1e-3),
            RefineConfig(
                batch_size=4, utd=1, policy_delay=1, hidden_dim=32, chunk_length=3
            ),
            torch.device("cpu"),
            update_step=0,
            updates=1,
        )
        self.assertEqual(step, 1)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        )
        self.assertFalse(torch.equal(before, actor.net[-1].weight))

    def test_multitask_update_trains_each_critic_and_pcgrads_actor(self) -> None:
        torch.manual_seed(4)
        state_dim, horizon, action_dim = 8, 3, 2
        actor = ChunkActor(state_dim, horizon, action_dim, 32, fixed_std=0.0)
        critics = torch.nn.ModuleDict(
            {str(task): TwinCritic(state_dim, horizon, action_dim, 32) for task in (0, 1)}
        )
        targets = copy.deepcopy(critics).requires_grad_(False)
        replay = ReplayBuffer(32, [0, 1])
        for task in (0, 1):
            for _ in range(8):
                replay.add(
                    torch.randn(state_dim),
                    torch.randn(horizon, action_dim).clamp(-1, 1),
                    torch.randn(horizon, action_dim).clamp(-1, 1),
                    torch.zeros(horizon),
                    torch.randn(state_dim),
                    torch.randn(horizon, action_dim).clamp(-1, 1),
                    False,
                    horizon,
                    task,
                )
        actor_before = actor.net[-1].weight.detach().clone()
        critic_before = {
            key: critic.q1[0].weight.detach().clone() for key, critic in critics.items()
        }
        step, metrics = update_multitask_rlt(
            replay,
            actor,
            critics,
            targets,
            torch.optim.Adam(actor.parameters(), lr=1e-3),
            {
                key: torch.optim.Adam(critic.parameters(), lr=1e-3)
                for key, critic in critics.items()
            },
            RefineConfig(
                batch_size=8, utd=1, policy_delay=1, hidden_dim=32, chunk_length=3
            ),
            torch.device("cpu"),
            update_step=0,
            updates=1,
            update_actor=False,
        )
        self.assertEqual(step, 1)
        self.assertNotIn("actor_loss", metrics)
        self.assertTrue(torch.equal(actor_before, actor.net[-1].weight))
        for key, critic in critics.items():
            self.assertFalse(torch.equal(critic_before[key], critic.q1[0].weight))

        step, metrics = update_multitask_rlt(
            replay,
            actor,
            critics,
            targets,
            torch.optim.Adam(actor.parameters(), lr=1e-3),
            {
                key: torch.optim.Adam(critic.parameters(), lr=1e-3)
                for key, critic in critics.items()
            },
            RefineConfig(
                batch_size=8, utd=1, policy_delay=1, hidden_dim=32, chunk_length=3
            ),
            torch.device("cpu"),
            update_step=step,
            updates=1,
        )
        self.assertEqual(step, 2)
        self.assertEqual(metrics["pcgrad_comparisons"], 2)
        self.assertFalse(torch.equal(actor_before, actor.net[-1].weight))


if __name__ == "__main__":
    unittest.main()
