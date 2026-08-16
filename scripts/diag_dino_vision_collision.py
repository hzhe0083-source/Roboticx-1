"""训练集视觉表征碰撞诊断（2026-08-16）。

假设：冻结 OOD DINO 8×8 池化 token 在 732 窗口内不唯一——不同窗口（专家动作
不同）的视觉条件几乎相同 → 条件映射 (vision,proprio)→action 不可拟合 → 训练
loss 无法降到 0（连过拟合都做不到）。若碰撞常见且动作分歧大，则 15k 步不
过拟合的根本原因 = 表征分辨率不足，而非预算。

方法：从特征缓存重建每窗口首决策的池化 token（与训练完全同构），在 300 个
采样条件内做余弦最近邻，统计 (token 余弦, 专家动作 prefix-6 L2 距离) 联合
分布：高余弦 + 大动作距离 = 碰撞。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train import DinoFeatureCache, _dino_main_encode_from_cache  # noqa: E402
from va_compound.longtraj_frames import LongTrajFramesDataset  # noqa: E402

DATA = REPO / "data/metaworld_longtraj_windows_h48_dino35_clean.pt"
CACHE = REPO / "data/dino35_feature_cache"
device = torch.device("cuda")


def main() -> None:
    cache = DinoFeatureCache(CACHE)
    dataset = LongTrajFramesDataset(
        DATA, min_sequence_length=4, feature_cache=CACHE, include_frames=False
    )
    payload = dataset.payload
    n = len(dataset)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=300, replace=False)
    tokens = []
    actions = []
    proprio = []
    with torch.no_grad():
        for start in range(0, 300, 30):
            b = idx[start:start + 30]
            rows = torch.as_tensor(dataset.cache_rows[b, 0], device=device)
            tok = _dino_main_encode_from_cache(
                rows[:, None, :], cache, device, grid=8, window=4
            )[:, 0]  # [B, 256, 1024]
            tokens.append(tok)
            actions.append(payload["actions"][b, 0])  # [B, 48, 4]
            proprio.append(payload["proprio"][b, 0])  # [B, 9]
    tokens = torch.cat(tokens, dim=0)  # [300, 256, 1024]
    actions = torch.cat(actions, dim=0)
    proprio = torch.cat(proprio, dim=0)
    # 全局均值池 token → [300, 1024]（与训练"vision 均值摘要"同构）
    tokens = tokens.to(device)
    mean_tok = tokens.mean(dim=1)
    mean_tok = mean_tok / mean_tok.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    mean_tok = mean_tok.to(device)
    sim = mean_tok @ mean_tok.T  # [300, 300] 余弦
    sim.fill_diagonal_(-2.0)
    nn_sim, nn_idx = sim.max(dim=1)  # 每个条件的最近邻
    # 最近邻对的 prefix-6 动作距离（与模仿诊断同口径）
    act_nn = actions[nn_idx.cpu()]
    prefix_dist = (actions[:, :6] - act_nn[:, :6]).norm(dim=-1).mean(dim=-1).to(device)
    # proprio 距离（同状态但不同动作的情况由 proprio 区分吗？）
    pro_dist = (proprio - proprio[nn_idx.cpu()]).norm(dim=-1)
    print(f"n={len(idx)} 条件对的最近邻统计：")
    print(f"  token 余弦: mean={nn_sim.mean():.4f} min={nn_sim.min():.4f} "
          f"max={nn_sim.max():.4f}")
    print(f"  prefix6 动作 L2: mean={prefix_dist.mean():.4f} "
          f"max={prefix_dist.max():.4f}")
    print(f"  proprio L2: mean={pro_dist.mean():.4f}")
    # 碰撞计数：token 余弦 > 0.98 且动作距离 > 0.3（接近专家幅值 0.42）
    collide = (nn_sim > 0.98) & (prefix_dist > 0.3)
    print(f"  碰撞对 (cos>0.98 且 act>0.3): {int(collide.sum())}/{len(idx)}")
    near = (nn_sim > 0.95) & (prefix_dist > 0.2)
    print(f"  近碰撞 (cos>0.95 且 act>0.2): {int(near.sum())}/{len(idx)}")
    # 分桶：高余弦对的平均动作距离
    for th in (0.99, 0.98, 0.95, 0.90):
        m = nn_sim > th
        if m.any():
            print(f"  cos>{th}: {int(m.sum())} 对, 平均动作距离={prefix_dist[m].mean():.4f}")
    # 对照组：动作最近的邻居 vs token 余弦（若动作相似的邻居 token 也相似，
    # 则表征至少对"动作相似性"有序）。
    act_sim = torch.zeros_like(sim)
    for i in range(300):
        act_sim[i] = -(actions[i, :6] - actions[:, :6]).norm(dim=-1).mean(dim=-1)
    act_sim.fill_diagonal_(-1e9)
    _, act_nn = act_sim.max(dim=1)
    cos_for_act_nn = sim.gather(1, act_nn[:, None]).squeeze(1)
    print(f"  动作最近邻的 token 余弦: mean={cos_for_act_nn.mean():.4f} "
          f"(>0 说明同动作窗口视觉也相近)")
    # 与 16×16 不池化对比：dense 证据是否更可分？
    dense_tokens = []
    for start in range(0, 300, 30):
        b = idx[start:start + 30]
        rows = torch.as_tensor(dataset.cache_rows[b, 0], device=device)
        _, dense = _dino_main_encode_from_cache(
            rows[:, None, :], cache, device, grid=8, window=4, return_dense=True
        )
        dense_tokens.append(dense[11][:, 0, 256:].mean(dim=1))  # 帧 d 的 256 patch 均值
    dt = torch.cat(dense_tokens, dim=0)
    dt = dt / dt.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    dsim = dt @ dt.T
    dsim.fill_diagonal_(-2.0)
    dnn_sim, dnn_idx = dsim.max(dim=1)
    dact = (actions[:, :6] - actions[dnn_idx.cpu()][:, :6]).norm(dim=-1).mean(dim=-1).to(device)
    dcollide = (dnn_sim > 0.98) & (dact > 0.3)
    print(f"  [dense 帧d patch 均值] 碰撞对 (cos>0.98 且 act>0.3): "
          f"{int(dcollide.sum())}/{len(idx)}")


if __name__ == "__main__":
    main()
