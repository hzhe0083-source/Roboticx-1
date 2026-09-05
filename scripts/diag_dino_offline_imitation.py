"""DINO 离线开环模仿诊断（2026-08-16）。

判别"没学会 vs 学会但闭环漂移"：用特征缓存构造训练/held-out 窗口的首决策
条件（vision + dense + metric），decode_actions（Euler 8，固定噪声）生成
48 步动作，逐位与专家动作比 L2 误差；并对 dense evidence 置零重解码，
度量 dense 分支推理端真实贡献（固定噪声消掉流采样随机性）。

实测（15k densemetric ckpt）：
- prefix 0-5 |err|≈0.22、overall≈0.42，专家动作幅值≈0.42 → 训练分布上也
  只拟合一半（欠拟合，非过拟合：heldout ≈ train）；
- 2k→15k 整体误差仅 0.433→0.416（4% 相对）——步数+dense+metric 增益微弱；
- dense 分支 Δpred≈0.11（专家幅值 25%），有/无 dense 误差 0.396 vs 0.413
  （小而正的离线增益，与 V-JEPA 消融方向一致）。
用法：
  python scripts/diag_dino_offline_imitation.py \
    --checkpoint checkpoints/e7_dino_main_p35_densemetric_15k.pt \
    [--no-metric] [--n-train 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from va_compound.vision.metric_runtime import (  # noqa: E402
    _build_dino_metric_stack,
    _dino_metric_tokens,
)
from va_compound.longtraj_frames import LongTrajFramesDataset  # noqa: E402
from va_compound.model import VACompoundConfig, VACompoundPolicy  # noqa: E402
from va_compound.vision.encoding import (  # noqa: E402
    DinoFeatureCache,
    _dino_main_encode_from_cache,
)

DATA = REPO / "data/metaworld_longtraj_windows_h48_dino35_clean.pt"
CACHE = REPO / "data/dino35_feature_cache"


def evaluate_split(indices: np.ndarray, split_name: str) -> None:
    model.eval()
    errs = []
    errs_zero_list = []
    dense_delta = []
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            idx = indices[start:start + 16]
            batch = {
                k: dataset.payload[k][idx].to(device)
                for k in ("actions", "proprio", "previous_action",
                          "language_hidden", "instruction_id")
            }
            batch["language_mask"] = dataset.payload["language_mask"][idx].to(device)
            rows = torch.as_tensor(dataset.cache_rows[idx, 0], device=device)  # [B, W]
            tokens_t, dense_t = _dino_main_encode_from_cache(
                rows[:, None, :], cache, device, grid=config.main_vision_grid,
                window=config.main_vision_frames, return_dense=True,
            )
            if metric_head is not None:
                metric_tokens_t, metric_g_t = _dino_metric_tokens(
                    metric_head, relation_encoder, dense_t, batch, device,
                    train_metric_head=False,
                )
                metric_tokens = metric_tokens_t[:, 0]
                metric_g = metric_g_t[:, 0]
            else:
                metric_tokens = None
                metric_g = None
            tokens = tokens_t[:, 0]
            dense = {k: v[:, 0] for k, v in dense_t.items()}
            cond = model.encode_condition(
                tokens,
                batch["proprio"][:, 0],
                batch["previous_action"][:, 0],
                language_hidden=batch["language_hidden"],
                language_mask=batch["language_mask"],
                dense_evidence=dense,
                metric_tokens=metric_tokens,
                metric_g=metric_g,
            )
            # 固定噪声：两次解码同一 ε，消掉流采样随机性。
            noise = torch.randn(cond.shape[0], 48, 4, device=device)
            pred = model.decode_actions(cond, steps=8, noise=noise)  # [B, 48, 4]
            expert = batch["actions"][:, 0]  # [B, 48, 4]
            errs.append((pred - expert).abs())
            dense_zero = {k: torch.zeros_like(v) for k, v in dense.items()}
            metric_zero = (
                torch.zeros_like(metric_tokens) if metric_tokens is not None else None
            )
            cond_zero = model.encode_condition(
                tokens,
                batch["proprio"][:, 0],
                batch["previous_action"][:, 0],
                language_hidden=batch["language_hidden"],
                language_mask=batch["language_mask"],
                dense_evidence=dense_zero,
                metric_tokens=metric_zero,
                metric_g=(
                    torch.zeros_like(metric_g) if metric_g is not None else None
                ),
            )
            pred_zero = model.decode_actions(cond_zero, steps=8, noise=noise)
            errs_zero_list.append((pred_zero - expert).abs())
            dense_delta.append((pred_zero - pred).abs())
    err = torch.cat(errs, dim=0)  # [N, 48, 4]
    err_zero = torch.cat(errs_zero_list, dim=0)
    ddelta = torch.cat(dense_delta, dim=0)
    pos = err.mean(dim=(0, 2))  # [48]
    print(f"\n[{split_name}] n={err.shape[0]} 离线开环 |误差|（每位置, 4 维均值）:")
    print(f"  prefix 0-5:  {pos[:6].mean().item():.4f}  (min {pos[:6].min().item():.4f})")
    print(f"  mid 6-23:    {pos[6:24].mean().item():.4f}")
    print(f"  tail 24-47:  {pos[24:].mean().item():.4f}")
    print(f"  overall:     {err.mean().item():.4f}")
    print(f"  前 6 位置:   " + " ".join(f"{v:.3f}" for v in pos[:6].tolist()))
    print(f"  dense Δpred: {ddelta.mean().item():.4f}")
    print(
        f"  有 dense |err|: {err.mean().item():.4f} | "
        f"无 dense |err|: {err_zero.mean().item():.4f} "
        f"(专家动作 |a|={batch['actions'][:, 0].abs().mean().item():.3f})"
    )


def main() -> None:
    global model, metric_head, relation_encoder, config, dataset, cache, device
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "checkpoints/e7_dino_main_p35_densemetric_15k.pt")
    parser.add_argument("--no-metric", action="store_true")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    model = VACompoundPolicy(config).to(device).eval()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected
    if args.no_metric:
        metric_head = relation_encoder = None
    else:
        metric_head, relation_encoder = _build_dino_metric_stack(
            device, config,
            train_metric_head=False, train_relation=False,
            saved_ctor_config=ckpt["mtvj_metric_head_config"],
        )
        metric_head.load_state_dict(ckpt["mtvj_metric_head"], strict=True)
        relation_encoder.load_state_dict(ckpt["mtvj_relation_encoder"], strict=True)
    cache = DinoFeatureCache(CACHE)
    dataset = LongTrajFramesDataset(
        DATA, min_sequence_length=4,
        feature_cache=CACHE, include_frames=False,
    )
    ep_ids = np.array([ep for _, ep, _ in dataset.refs])
    threshold = np.quantile(ep_ids, 0.8)
    heldout = np.where(ep_ids > threshold)[0]
    train_idx = np.where(ep_ids <= threshold)[0]
    rng = np.random.default_rng(0)
    train_idx = rng.choice(
        train_idx, size=min(args.n_train, len(train_idx)), replace=False
    )
    evaluate_split(train_idx, f"train-windows ({len(train_idx)})")
    evaluate_split(heldout, f"heldout-episodes ({len(heldout)})")


if __name__ == "__main__":
    main()
