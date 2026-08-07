"""Demo: autoregressive rollout of the trained flat policy vs expert actions.

Loads checkpoints/pnpw_flow.pt, rolls out 2 episodes with the deployment
interface (encode_condition + sample_actions, visual memory chain), samples
3 noises per decision point, and plots predicted vs expert trajectories.
"""
from __future__ import annotations

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from va_compound import VACompoundConfig, VACompoundPolicy


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("checkpoints/pnpw_flow.pt", map_location="cpu", weights_only=True)
    config = VACompoundConfig(**checkpoint["config"])
    model = VACompoundPolicy(config)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    payload = torch.load("data/pnpw_features.pt", map_location="cpu", weights_only=True)
    q01 = payload["normalization"]["action_q01"]
    q99 = payload["normalization"]["action_q99"]

    def denormalize(value: torch.Tensor, dimension: int) -> torch.Tensor:
        # 与训练标签一致裁剪模型输出到 [-1,1] 后再反归一化（robust_normalize 存盘即 clip）
        return (
            (value.clamp(-1.0, 1.0) + 1.0) / 2.0 * (q99[dimension] - q01[dimension])
            + q01[dimension]
        )

    episodes = sorted(payload["episode_id"].unique().tolist())
    selected = [episodes[0], episodes[1]]
    figures = []
    for episode in selected:
        index = int(
            (payload["episode_id"] == episode).nonzero()[0]
        )
        sample = {
            key: payload[key][index].to(device)
            for key in (
                "vision_tokens",
                "language_hidden",
                "language_mask",
                "proprio",
                "previous_action",
                "actions",
            )
        }
        language_cache = model.build_language_cache(
            sample["language_hidden"][None],
            sample["language_mask"][None],
        )
        memory = None
        predictions = []
        first_maes = []
        for time_index in range(sample["vision_tokens"].shape[0]):
            condition, memory = model.encode_condition(
                sample["vision_tokens"][time_index][None],
                sample["proprio"][time_index][None],
                sample["previous_action"][time_index][None],
                language_cache=language_cache,
                visual_memory=memory,
                return_visual_memory=True,
            )
            # three independent noises to show sampling diversity
            sampled = []
            for _ in range(3):
                sampled.append(model.sample_actions(condition, steps=8)[0])
            sampled = torch.stack(sampled)  # [3, H, A]
            predictions.append(sampled)
            target = sample["actions"][time_index]
            mae = float(
                (sampled.mean(dim=0)[0] - target[0]).abs().mean().cpu()
            )
            first_maes.append(mae)
        predictions = torch.stack(predictions)  # [T, 3, H, A]
        target = sample["actions"]  # [T, H, A]

        pred_flat = predictions.mean(dim=1).reshape(-1, config.action_dim)  # [T*H, A]
        pred_lo = predictions.reshape(-1, 3, config.action_dim).min(dim=1).values
        pred_hi = predictions.reshape(-1, 3, config.action_dim).max(dim=1).values
        expert = target.reshape(-1, config.action_dim)
        x = torch.arange(pred_flat.shape[0])

        figure, axes = plt.subplots(3, 2, figsize=(14, 10))
        for axis in axes.flat:
            axis.set_xlim(0, pred_flat.shape[0] - 1)
        for dimension in range(min(6, config.action_dim)):
            row, col = divmod(dimension, 2)
            axis = axes[row, col]
            axis.fill_between(
                x.cpu(),
                denormalize(pred_lo[:, dimension], dimension).cpu(),
                denormalize(pred_hi[:, dimension], dimension).cpu(),
                alpha=0.25,
                color="tab:blue",
                label="sampled range (3 noises)",
            )
            axis.plot(
                x.cpu(),
                denormalize(pred_flat[:, dimension], dimension).cpu(),
                color="tab:blue",
                linewidth=1.5,
                label="predicted (mean)",
            )
            axis.plot(
                x.cpu(),
                denormalize(expert[:, dimension], dimension).cpu(),
                color="tab:red",
                linewidth=1.5,
                linestyle="--",
                label="expert",
            )
            for boundary in range(1, sample["vision_tokens"].shape[0]):
                axis.axvline(boundary * config.action_horizon - 0.5, color="gray", linewidth=0.6)
            axis.set_title(f"action dim {dimension}")
            axis.set_xlabel("action step (4 decision points x 8 steps)")
            axis.grid(alpha=0.3)
            if dimension == 0:
                axis.legend(loc="upper right", fontsize=8)
        figure.suptitle(f"episode {episode} — autoregressive rollout (visual memory chain)", fontsize=13)
        figure.tight_layout()
        path = f"demo_rollout_ep{episode}.png"
        figure.savefig(path, dpi=110)
        figures.append((path, first_maes))
        print(
            f"episode={episode} first_step_mae per decision point: "
            + ", ".join(f"{value:.4f}" for value in first_maes)
        )
        print(f"saved={path}")
    return figures


if __name__ == "__main__":
    main()
