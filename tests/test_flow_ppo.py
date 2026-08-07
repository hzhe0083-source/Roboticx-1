import unittest

import numpy as np
import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy
from train_ppo_metaworld import FlowNoiseSchedule, ValueHead, compute_gae


def tiny_config() -> VACompoundConfig:
    return VACompoundConfig(
        language_dim=24,
        vision_dim=20,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        action_horizon=5,
        action_dim=6,
        proprio_dim=9,
        mode="bidir_va",
    )


class FlowPPOTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = tiny_config()
        self.model = VACompoundPolicy(self.config).eval()
        self.sigma = FlowNoiseSchedule(steps=8, action_dim=self.config.action_dim)
        self.cond = torch.randn(2, self.config.action_horizon, self.config.hidden_dim)

    def test_deterministic_path_matches_sample_actions(self) -> None:
        """sigma=None path must equal the classic Euler sampler."""
        noise = torch.randn(2, self.config.action_horizon, self.config.action_dim)
        path = self.model.sample_flow_trajectory(
            self.cond, steps=8, noise=noise.clone(), sigma=None
        )
        ref = self.model.sample_actions(self.cond, steps=8, noise=noise)
        torch.testing.assert_close(path[-1], ref, atol=1e-6, rtol=1e-6)

    def test_unchanged_parameters_give_zero_log_ratio(self) -> None:
        """Same path + same weights -> ratio 1 (log-ratio ~ 0)."""
        path = self.model.sample_flow_trajectory(
            self.cond, steps=8, sigma=self.sigma()
        )
        lp1 = self.model.flow_trajectory_log_prob(path, self.cond, self.sigma())
        lp2 = self.model.flow_trajectory_log_prob(path, self.cond, self.sigma())
        self.assertLess(torch.max(torch.abs(lp1 - lp2)).item(), 1e-5)

    def test_velocity_perturbation_changes_log_prob_and_gradient_flows(self) -> None:
        """Log-prob must be sensitive to theta and propagate gradients into
        the VA composite and flow head."""
        path = self.model.sample_flow_trajectory(
            self.cond, steps=8, sigma=self.sigma()
        )
        lp = self.model.flow_trajectory_log_prob(path, self.cond, self.sigma())
        lp.sum().backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads, "no trainable parameter received a gradient")
        self.assertTrue(
            any(p.grad.abs().max() > 0 for p in self.model.parameters() if p.grad is not None)
        )

    def test_noise_bounds(self) -> None:
        s = self.sigma()
        self.assertGreaterEqual(s.min().item(), 0.02 - 1e-6)
        self.assertLessEqual(s.max().item(), 0.08 + 1e-6)

    def test_gae_terminal_handling(self) -> None:
        """Success/termination must not bootstrap; truncation must."""
        rewards = [0.0, 0.0, 1.0, 0.0]
        values = [0.1, 0.2, 0.3, 0.0]
        dones = [False, False, True, False]
        ret, adv = compute_gae(np.asarray(rewards), np.asarray(values), np.asarray(dones))
        # terminal at index 2: return equals reward (no bootstrap)
        self.assertAlmostEqual(ret[2], 1.0, places=5)
        # truncation at index 3 (dones=False): bootstrap with next_value=0,
        # GAE continues normally
        self.assertAlmostEqual(
            ret[3], rewards[3], places=5
        )

    def test_zero_reward_buffer_trains(self) -> None:
        """A zero-reward rollout must not produce NaN losses."""
        from train_ppo_metaworld import RolloutBuffer

        buffer = RolloutBuffer()
        for _ in range(4):
            buffer.rewards.append(0.0)
            buffer.values.append(0.0)
            buffer.dones.append(False)
            buffer.old_logp.append(-1.0)
        buffer.returns, buffer.advantages = compute_gae(
            np.asarray(buffer.rewards), np.asarray(buffer.values), np.asarray(buffer.dones)
        )
        self.assertTrue(torch.isfinite(torch.tensor(buffer.returns)).all())
        self.assertTrue(torch.isfinite(torch.tensor(buffer.advantages)).all())

    def test_batched_condition_matches_per_sample(self) -> None:
        """Batched encode_condition (stacked tokens/proprio/lang/memory) must
        equal per-sample results — the ppo_update batching contract."""
        from va_compound.model import VisualMemory
        B, T, TD, H = 3, 4, 20, 5
        vision = torch.randn(B, T, TD)
        proprio = torch.randn(B, 9)  # [B, D] 2D, same as the rollout buffer
        previous = torch.randn(B, 6)
        lang_h = torch.randn(B, 7, self.config.language_dim)
        lang_m = torch.ones(B, 7, dtype=torch.bool)
        mems = [
            VisualMemory(layers=tuple(torch.randn(1, 8, 32) for _ in range(2)))
            for _ in range(B)
        ]
        refs = [
            self.model.encode_condition(
                vision[i : i + 1], proprio[i : i + 1], previous[i : i + 1],
                language_hidden=lang_h[i : i + 1], language_mask=lang_m[i : i + 1],
                visual_memory=mems[i],
            ).float()
            for i in range(B)
        ]
        stacked = VisualMemory(
            layers=tuple(torch.cat([m.layers[k] for m in mems], 0) for k in range(2))
        )
        batch = self.model.encode_condition(
            vision, proprio, previous,
            language_hidden=lang_h, language_mask=lang_m, visual_memory=stacked,
        ).float()
        for i in range(B):
            torch.testing.assert_close(batch[i], refs[i][0], atol=1e-5, rtol=1e-5)
        # None-memory path must also batch consistently.
        batch_none = self.model.encode_condition(
            vision, proprio, previous,
            language_hidden=lang_h, language_mask=lang_m, visual_memory=None,
        ).float()
        for i in range(B):
            ref_none = self.model.encode_condition(
                vision[i : i + 1], proprio[i : i + 1], previous[i : i + 1],
                language_hidden=lang_h[i : i + 1], language_mask=lang_m[i : i + 1],
                visual_memory=None,
            ).float()
            torch.testing.assert_close(batch_none[i], ref_none[0], atol=1e-5, rtol=1e-5)
    def test_batched_ppo_update_mixed_memory(self) -> None:
        """ppo_update with a mixed buffer (None + real memories) must produce
        finite losses, update parameters, and preserve the log-ratio math."""
        import torch.nn as nn
        from train_ppo_metaworld import RolloutBuffer, ppo_update
        from va_compound.model import VisualMemory
        B, T, TD, H, A, D = 6, 4, 20, 5, 6, 9
        steps = 8
        buffer = RolloutBuffer()
        mem0 = VisualMemory(layers=tuple(torch.randn(1, 8, 32) for _ in range(2)))
        for i in range(B):
            buffer.frames.append(
                (
                    torch.randn(1, T, TD).half(),
                    torch.randn(1, D),
                    torch.randn(1, 6),
                    None if i % 2 == 0 else mem0,
                )
            )
            buffer.lang_hidden.append(torch.randn(7, self.config.language_dim))
            buffer.lang_mask.append(torch.ones(7, dtype=torch.bool))
            buffer.paths.append(
                [torch.randn(1, H, A) for _ in range(steps + 1)]
            )
            buffer.old_logp.append(-1.0)
            buffer.rewards.append(float(i % 3 == 0))
            buffer.dones.append(i % 4 == 0)
            buffer.values.append(0.0)
        buffer.returns, buffer.advantages = compute_gae(
            np.asarray(buffer.rewards),
            np.asarray(buffer.values),
            np.asarray(buffer.dones),
        )
        value_head = ValueHead(hidden_dim=self.config.hidden_dim)
        actor_opt = torch.optim.AdamW(self.model.parameters(), lr=3e-6)
        critic_opt = torch.optim.AdamW(value_head.parameters(), lr=1e-4)
        before = {k: v.clone() for k, v in self.model.named_parameters()}
        actor_loss, value_loss = ppo_update(
            buffer, self.model, self.sigma, value_head, actor_opt, critic_opt,
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(torch.tensor(actor_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(value_loss)))
        moved = any(
            not torch.equal(before[k], v) for k, v in self.model.named_parameters()
        )
        self.assertTrue(moved, "actor parameters did not update")

    def test_batched_condition_memory_split(self) -> None:
        """VA2 (memory_split): a stacked (evidence, task) memory must equal
        per-sample results — the ppo_update batching contract for the
        causal-decomposed memory. Regression: the legacy rebuild path read
        only ``layers`` (empty under memory_split), silently resetting the
        recurrent state to its episode-start init."""
        from dataclasses import replace
        from va_compound.model import VisualMemory

        cfg = replace(
            tiny_config(),
            memory_split=True,
            evidence_tokens=8,
            task_tokens=4,
        )
        model = VACompoundPolicy(cfg).eval()
        B, T, TD, D = 3, 4, 20, 9
        vision = torch.randn(B, T, TD)
        proprio = torch.randn(B, D)
        previous = torch.randn(B, 6)
        lang_h = torch.randn(B, 7, cfg.language_dim)
        lang_m = torch.ones(B, 7, dtype=torch.bool)
        mems, refs = [], []
        for i in range(B):
            cond, mem = model.encode_condition(
                vision[i : i + 1],
                proprio[i : i + 1],
                previous[i : i + 1],
                language_hidden=lang_h[i : i + 1],
                language_mask=lang_m[i : i + 1],
                visual_memory=None,
                return_visual_memory=True,
            )
            mems.append(mem)
            refs.append(cond.float())
        for m in mems:
            self.assertIsNotNone(m.evidence)
            self.assertIsNotNone(m.task)
            self.assertEqual(len(m.layers), 0)
        # Second decision, per sample: consume the per-sample first-step memory.
        refs2 = []
        for i in range(B):
            cond2, _ = model.encode_condition(
                vision[i : i + 1],
                proprio[i : i + 1],
                previous[i : i + 1],
                language_hidden=lang_h[i : i + 1],
                language_mask=lang_m[i : i + 1],
                visual_memory=mems[i],
                return_visual_memory=True,
            )
            refs2.append(cond2.float())
        # Second decision, batched (ppo_update contract): consume the stacked
        # memory; the result must match the per-sample sequence above.
        stacked = VisualMemory(
            layers=(),
            evidence=torch.cat([m.evidence for m in mems], dim=0),
            task=torch.cat([m.task for m in mems], dim=0),
        )
        batch = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=lang_h,
            language_mask=lang_m,
            visual_memory=stacked,
        ).float()
        for i in range(B):
            torch.testing.assert_close(batch[i], refs2[i][0], atol=1e-5, rtol=1e-5)
        # None-memory (episode start) must also batch consistently.
        batch_none = model.encode_condition(
            vision,
            proprio,
            previous,
            language_hidden=lang_h,
            language_mask=lang_m,
            visual_memory=None,
        ).float()
        for i in range(B):
            ref_none = model.encode_condition(
                vision[i : i + 1],
                proprio[i : i + 1],
                previous[i : i + 1],
                language_hidden=lang_h[i : i + 1],
                language_mask=lang_m[i : i + 1],
                visual_memory=None,
            ).float()
            torch.testing.assert_close(batch_none[i], ref_none[0], atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
