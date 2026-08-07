import tempfile
import unittest
from pathlib import Path

import torch

from train import (
    FeatureDataset,
    PairedBatchSampler,
    paired_partner_indices,
    sample_flow_matching_inputs,
    sample_pair_intervention,
    semantic_pair_loss,
    synthetic_sequence,
)
from va_compound.model import VACompoundConfig


def paired_payload() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    samples, sequence, visual_tokens = 4, 4, 5
    vision = torch.randn(2, sequence, visual_tokens, 6).repeat_interleave(2, dim=0)
    proprio = torch.randn(2, sequence, 3).repeat_interleave(2, dim=0)
    previous = torch.randn(2, sequence, 2).repeat_interleave(2, dim=0)
    instruction_id = torch.tensor([0, 1, 0, 1])
    language_by_instruction = torch.randn(2, 3, 7)
    language = language_by_instruction[instruction_id]
    actions = torch.randn(samples, sequence, 3, 2)
    actions[1] = actions[0] + 1.0
    actions[3] = actions[2] - 1.0
    return {
        "vision_tokens": vision,
        "language_hidden": language,
        "language_mask": torch.ones(samples, 3, dtype=torch.bool),
        "proprio": proprio,
        "previous_action": previous,
        "actions": actions,
        "pair_id": torch.tensor([20, 20, 21, 21]),
        "instruction_id": instruction_id,
    }


class TrainingContractTests(unittest.TestCase):
    def write_payload(self, directory: str, payload: dict[str, torch.Tensor]) -> Path:
        path = Path(directory) / "features.pt"
        torch.save(payload, path)
        return path

    def test_feature_dataset_accepts_paired_continuous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = FeatureDataset(self.write_payload(directory, paired_payload()))
        self.assertEqual(len(dataset), 4)
        self.assertEqual(set(dataset.pair_groups), {20, 21})
        self.assertEqual(set(dataset.pair_groups[20]), {0, 1})

    def test_feature_dataset_accepts_explicit_single_task_contract(self) -> None:
        payload = paired_payload()
        payload["instruction_id"] = torch.zeros(4, dtype=torch.long)
        payload["pair_id"] = torch.arange(4)
        with tempfile.TemporaryDirectory() as directory:
            dataset = FeatureDataset(
                self.write_payload(directory, payload),
                require_pairs=False,
            )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.pair_groups, {})

    def test_feature_dataset_rejects_unshared_pair_start(self) -> None:
        payload = paired_payload()
        payload["vision_tokens"][1, 0, 0, 0] += 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_payload(directory, payload)
            with self.assertRaisesRegex(ValueError, "does not share its first vision_tokens"):
                FeatureDataset(path)

    def test_feature_dataset_rejects_action_indistinguishable_pair(self) -> None:
        payload = paired_payload()
        payload["actions"][1] = payload["actions"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_payload(directory, payload)
            with self.assertRaisesRegex(ValueError, "no identifiable action difference"):
                FeatureDataset(path)

    def test_paired_batch_sampler_keeps_two_goals_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = FeatureDataset(self.write_payload(directory, paired_payload()))
            batches = list(PairedBatchSampler(dataset, batch_size=4, seed=3))
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 4)
        for pair_id in (20, 21):
            indices = [
                index
                for index in batches[0]
                if int(dataset.payload["pair_id"][index]) == pair_id
            ]
            self.assertEqual(len(indices), 2)
            self.assertNotEqual(
                int(dataset.payload["instruction_id"][indices[0]]),
                int(dataset.payload["instruction_id"][indices[1]]),
            )

    def test_partner_indices_swap_rows_within_each_pair(self) -> None:
        partner = paired_partner_indices(
            torch.tensor([4, 4, 9, 9]),
            torch.tensor([0, 1, 1, 0]),
        )
        torch.testing.assert_close(partner, torch.tensor([1, 0, 3, 2]))

    def test_semantic_pair_loss_matches_goal_conditioned_action_delta(self) -> None:
        target = torch.randn(4, 2, 5)
        partner = torch.tensor([1, 0, 3, 2])
        exact, predicted_delta, target_delta = semantic_pair_loss(target, target, partner)
        self.assertEqual(exact.item(), 0.0)
        self.assertGreater(target_delta.item(), 0.0)
        self.assertEqual(predicted_delta.item(), target_delta.item())

        collapsed = target.clone()
        collapsed[1] = collapsed[0]
        collapsed[3] = collapsed[2]
        collapsed_loss, _, _ = semantic_pair_loss(collapsed, target, partner)
        self.assertGreater(collapsed_loss.item(), 0.0)

    def test_flow_inputs_follow_straight_probability_path(self) -> None:
        actions = torch.randn(4, 3, 2, 5)
        noisy, flow_time, target_velocity = sample_flow_matching_inputs(actions)
        recovered_noise = actions - target_velocity
        tau = flow_time[:, :, None, None]
        expected_noisy = (1.0 - tau) * recovered_noise + tau * actions
        torch.testing.assert_close(noisy, expected_noisy)

    def test_pair_intervention_is_identical_except_for_language_target(self) -> None:
        actions = torch.randn(4, 3, 2, 5)
        partner = torch.tensor([1, 0, 3, 2])

        # Shared-source CF (default): probe/tau shared within pairs, target
        # delta = (a_i - a_j) / (1 - tau).
        probe, flow_time, target_velocity = sample_pair_intervention(actions, partner)
        torch.testing.assert_close(probe[0], probe[1])
        torch.testing.assert_close(probe[2], probe[3])
        torch.testing.assert_close(flow_time[0], flow_time[1])
        torch.testing.assert_close(flow_time[2], flow_time[3])
        torch.testing.assert_close(
            target_velocity[0] - target_velocity[1],
            (actions[0, 0] - actions[1, 0]) / (1.0 - flow_time[0]),
        )
        # Midpoint probe: both partners must share the SAME source noise eps,
        # i.e. probe = (1-tau) eps + tau * mid with one common eps.
        mid = 0.5 * (actions[0, 0] + actions[1, 0])
        eps0 = (probe[0] - flow_time[0] * mid) / (1.0 - flow_time[0])
        eps1 = (probe[1] - flow_time[1] * mid) / (1.0 - flow_time[1])
        torch.testing.assert_close(eps0, eps1)
        # Consistency: probe == a0 - (1-tau) * tgt0.
        torch.testing.assert_close(probe[0], actions[0, 0] - (1.0 - flow_time[0]) * target_velocity[0])

        # Source-only point (probe_tau_max=0): probe == eps, tau == 0,
        # target = a - eps (legacy semantics preserved).
        probe0, tau0, tgt0 = sample_pair_intervention(actions, partner, probe_tau_max=0.0)
        recovered_noise = actions[:, 0] - tgt0
        torch.testing.assert_close(recovered_noise[0], recovered_noise[1])
        torch.testing.assert_close(recovered_noise[2], recovered_noise[3])
        torch.testing.assert_close(probe0, recovered_noise)
        torch.testing.assert_close(tau0, torch.zeros_like(tau0))
        torch.testing.assert_close(
            tgt0[0] - tgt0[1],
            actions[0, 0] - actions[1, 0],
        )

    def test_synthetic_sequence_is_paired_at_first_state(self) -> None:
        config = VACompoundConfig(
            language_dim=12,
            vision_dim=10,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            action_horizon=3,
            action_dim=4,
            proprio_dim=5,
        )
        batch = synthetic_sequence(config, 4, 4, torch.device("cpu"))
        for first, second in ((0, 1), (2, 3)):
            torch.testing.assert_close(
                batch["vision_tokens"][first, 0],
                batch["vision_tokens"][second, 0],
            )
            torch.testing.assert_close(
                batch["proprio"][first, 0],
                batch["proprio"][second, 0],
            )
            self.assertNotEqual(
                int(batch["instruction_id"][first]),
                int(batch["instruction_id"][second]),
            )
            self.assertGreater(
                float((batch["actions"][first] - batch["actions"][second]).abs().mean()),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
