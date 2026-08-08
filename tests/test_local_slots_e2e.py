"""PULSE-VA local-slot end-to-end smoke: tiny synthetic v5 + ST features."""
import tempfile
import unittest
from pathlib import Path

import torch

from va_compound.model import VACompoundConfig, VACompoundPolicy


def build_coords_smoke(n_tokens: int = 288) -> torch.Tensor:
    import torch

    rows = []
    grid = 12
    for t in range(2):
        for y in range(grid):
            for x in range(grid):
                half = (grid - 1) / 2
                rows.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
    return torch.tensor(rows[:n_tokens], dtype=torch.float32)


def make_fake_payload(samples: int = 4, n_st: int = 288) -> dict:
    torch.manual_seed(7)
    sequence = 4
    vision = torch.randn(samples, sequence, 64, 768)
    st = torch.randn(samples, sequence, n_st, 768)
    proprio = torch.randn(samples, sequence, 4)
    previous = torch.randn(samples, sequence, 4)
    instruction_id = torch.tensor([0, 1, 0, 1])
    language = torch.randn(2, 8, 1536)[instruction_id]
    actions = torch.randn(samples, sequence, 6, 4)
    return {
        "vision_tokens": vision,
        "vision_tokens_st": st,
        "coords": build_coords_smoke(n_st),
        "language_hidden": language,
        "language_mask": torch.ones(samples, 8, dtype=torch.bool),
        "proprio": proprio,
        "previous_action": previous,
        "actions": actions,
        "pair_id": torch.tensor([20, 20, 21, 21]),
        "instruction_id": instruction_id,
        "episode_id": torch.arange(samples),
    }


class LocalSlotE2ETests(unittest.TestCase):
    def test_local_vision_stream_and_forward(self) -> None:
        payload = make_fake_payload()
        cfg = VACompoundConfig(
            language_dim=1536,
            vision_dim=768,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            action_horizon=6,
            action_dim=4,
            proprio_dim=4,
            direct_head=True,
            local_slots=True,
        )
        model = VACompoundPolicy(cfg).eval()
        batch = {k: payload[k] for k in (
            "vision_tokens", "vision_tokens_st", "coords", "language_hidden",
            "language_mask", "proprio", "previous_action", "actions",
        )}
        cache = model.build_language_cache(batch["language_hidden"], batch["language_mask"])
        assert cache.role_queries is not None
        assert cache.role_queries.shape == (4, 6, 256)
        vision = model.build_local_vision(
            batch["vision_tokens_st"][:, 0],
            batch["coords"],
            cache.role_queries,
        )
        assert vision.shape == (4, 25, 768)
        with torch.inference_mode():
            cond, mem = model.encode_condition(
                vision,
                batch["proprio"][:, 0],
                batch["previous_action"][:, 0],
                language_cache=cache,
                return_visual_memory=True,
            )
            pred = model.decode_actions(cond)
        assert pred.shape == (4, 6, 4)

    def test_forward_gradients_flow_to_slot_modules(self) -> None:
        payload = make_fake_payload()
        cfg = VACompoundConfig(
            language_dim=1536, vision_dim=768, hidden_dim=128, num_layers=1,
            num_heads=4, action_horizon=4, action_dim=4, proprio_dim=4,
            direct_head=True, local_slots=True,
        )
        model = VACompoundPolicy(cfg).train()
        batch = {k: payload[k] for k in (
            "vision_tokens_st", "coords", "language_hidden", "language_mask",
            "proprio", "previous_action", "actions",
        )}
        cache = model.build_language_cache(batch["language_hidden"], batch["language_mask"])
        vision = model.build_local_vision(
            batch["vision_tokens_st"][:, 0], batch["coords"], cache.role_queries
        )
        cond, _ = model.encode_condition(
            vision, batch["proprio"][:, 0], batch["previous_action"][:, 0],
            language_cache=cache, return_visual_memory=True,
        )
        pred = model.decode_actions(cond)
        loss = (pred - batch["actions"][:, 0, : pred.shape[-2]]).pow(2).mean()
        loss.backward()
        for name, module in (
            ("role_compiler", model.role_compiler),
            ("slot_reader", model.slot_reader),
            ("relation_tokens", model.relation_tokens),
        ):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads, f"no gradients through {name}"
            assert all(bool(g.abs().sum() > 0) for g in grads), f"zero grad in {name}"


if __name__ == "__main__":
    unittest.main()
