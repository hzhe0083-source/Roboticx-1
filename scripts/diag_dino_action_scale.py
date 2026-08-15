"""DINO 动作幅度收缩验证（2026-08-16）。

离线开环诊断（diag_dino_offline_imitation.py）发现：模型预测动作幅度只有
专家动作的 ~11-20%（MSE 回归向均值收缩）。本脚本对 15k checkpoint 的
held-out 首决策解码结果做幅度重标定扫描 k∈[1,1.5,2,3,4]：
- |pred|/|expert|：输出收缩比；
- |k·pred − expert|：重标定后误差——若最优 k>1 显著降低误差，则证明
  幅度收缩是开环误差主因，闭环可直接用 k 倍率验证（零训练成本）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train import (  # noqa: E402
    DinoFeatureCache,
    _build_dino_metric_stack,
    _dino_main_encode_from_cache,
    _dino_metric_tokens,
)
from va_compound.longtraj_frames import LongTrajFramesDataset  # noqa: E402
from va_compound.model import VACompoundConfig, VACompoundPolicy  # noqa: E402

DATA = REPO / "data/metaworld_longtraj_windows_h48_dino35_clean.pt"
CACHE = REPO / "data/dino35_feature_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "checkpoints/e7_dino_main_p35_dm_grid16_15k.pt")
    parser.add_argument("--n-windows", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    model = VACompoundPolicy(config).to(device).eval()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected
    metric_head, relation_encoder = _build_dino_metric_stack(
        device, config, train_metric_head=False, train_relation=False,
        saved_ctor_config=ckpt["mtvj_metric_head_config"],
    )
    metric_head.load_state_dict(ckpt["mtvj_metric_head"], strict=True)
    relation_encoder.load_state_dict(ckpt["mtvj_relation_encoder"], strict=True)
    cache = DinoFeatureCache(CACHE)
    dataset = LongTrajFramesDataset(
        DATA, min_sequence_length=4, feature_cache=CACHE, include_frames=False,
    )
    # 与离线模仿诊断一致的 held-out 划分（episode 0.8 分位）
    payload = dataset.payload
    episodes = np.asarray(payload["episode_id"])
    uniq = np.unique(episodes)
    thr = np.quantile(np.arange(len(uniq)), 0.8)
    held_ep = uniq[int(thr):]
    held_idx = np.where(np.isin(episodes, held_ep))[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(held_idx, size=min(args.n_windows, len(held_idx)),
                     replace=False)

    preds = []
    experts = []
    with torch.no_grad():
        for start in range(0, len(idx), 16):
            i = idx[start:start + 16]
            batch = {
                k: payload[k][i].to(device)
                for k in ("actions", "proprio", "previous_action",
                          "language_hidden", "instruction_id")
            }
            batch["language_mask"] = payload["language_mask"][i].to(device)
            rows = torch.as_tensor(dataset.cache_rows[i, 0], device=device)
            tokens_t, dense_t = _dino_main_encode_from_cache(
                rows[:, None, :], cache, device, grid=config.main_vision_grid,
                window=config.main_vision_frames, return_dense=True,
            )
            metric_tokens_t = _dino_metric_tokens(
                metric_head, relation_encoder, dense_t, batch, device,
                train_metric_head=False,
            )
            cond = model.encode_condition(
                tokens_t[:, 0],
                batch["proprio"][:, 0],
                batch["previous_action"][:, 0],
                language_hidden=batch["language_hidden"],
                language_mask=batch["language_mask"],
                dense_evidence={k: v[:, 0] for k, v in dense_t.items()},
                metric_tokens=metric_tokens_t[:, 0],
            )
            noise = torch.randn(cond.shape[0], 48, 4, device=device)
            preds.append(model.decode_actions(cond, steps=8, noise=noise))
            experts.append(batch["actions"][:, 0])
    pred = torch.cat(preds, dim=0)   # [N,48,4]
    expert = torch.cat(experts, dim=0)
    pmag = pred.abs().mean(dim=(0, 2))   # [48] 每维幅度均值
    emag = expert.abs().mean(dim=(0, 2))
    print("每维幅度（4 维均值）: pred/expert:")
    for seg, sl in (("prefix0-5", slice(0, 6)), ("mid6-23", slice(6, 24)),
                    ("tail24-47", slice(24, 48))):
        print(f"  {seg}: pred={pmag[sl].mean():.4f} expert={emag[sl].mean():.4f} "
              f"ratio={pmag[sl].mean()/emag[sl].mean():.2%}")
    print("\n幅度重标定扫描（held-out 开环，固定噪声）:")
    base = (pred - expert).abs().mean(dim=(0, 2)).mean()
    print(f"  k=1.0: |err|={base:.4f}")
    for k in (1.5, 2.0, 3.0, 4.0):
        e = (k * pred - expert).abs().mean(dim=(0, 2)).mean()
        print(f"  k={k}: |err|={e:.4f}  ({(base-e)/base:+.1%})")
    # 尾部专项
    tail_err = (pred[:, 24:] - expert[:, 24:]).abs().mean()
    print(f"\ntail24-47: k=1 |err|={tail_err:.4f}")
    for k in (2.0, 3.0, 4.0):
        e = (k * pred[:, 24:] - expert[:, 24:]).abs().mean()
        print(f"  k={k}: tail |err|={e:.4f}")


if __name__ == "__main__":
    main()
