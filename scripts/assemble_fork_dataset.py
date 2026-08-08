#!/usr/bin/env python
"""fork 原始采集 → 与 v5 同构的训练数据集（pair 生死门 A/C/D/E 用）。

关键约定（2026-08-08 设计）：
- pair 生死门在"稳定简单架构"上做（Codex Q4）→ 用 flat-64 特征 +
  FeatureDataset 路径，与 mw_v5_direct 主链同构；
- 特征域必须全数据集一致：v5 的 vision_tokens 来自**原始预训练 V-JEPA**，
  fork 行同样用 from_pretrained（不用 Stage B 微调 backbone）；
- 标签管线对齐 v5：state[:4] 用 v5 state_q01/q99 归一化；动作走 executed-clip
  （采集器已 clip，这里直接 robust_normalize）；prev t=0 = 0（归一化零）；
- language_hidden/mask 从 v5 按任务文本复制（Qwen 冻结缓存）；
- pair_id：v5 行保持唯一（无真实配对，FM-only）；fork 行 = FORK_PAIR_OFFSET+k
  （2 行共享）。D/E 训练时由 train.py 的 paired 路径只对 fork 对算 pair loss
  （partner<0 的行贡献零——需在 paired_partner_indices 加跳过逻辑，见备注）。

用法：
  python scripts/assemble_fork_dataset.py \
      --raw data/mw_fork_raw_drawer.pt \
      --v5 data/metaworld_features_v5.pt \
      --out data/mw_fork_drawer.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FORK_PAIR_OFFSET = 1 << 40  # 与 v5 pair_id（< 1e4）不冲突
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 根模块（prepare_pnpw_features 等）导入兼容


def robust_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    scale = (q99 - q01) / 2.0
    return np.clip(2.0 * (x - q01) / scale - 1.0, -1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/mw_fork_raw_drawer.pt")
    ap.add_argument("--v5", type=Path, default=ROOT / "data/metaworld_features_v5.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/mw_fork_drawer.pt")
    ap.add_argument("--merged", type=Path, default=None,
                    help="可选：v5+fork 合并数据集路径（C/D 组训练输入）")
    ap.add_argument("--shuffled", type=Path, default=None,
                    help="可选：E 组打乱配对数据集路径")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    raw = torch.load(args.raw, map_location="cpu", weights_only=True)
    v5 = torch.load(args.v5, map_location="cpu", weights_only=True)
    frames = raw["frames"]  # [2N, 4, 4, 384, 384, 3] uint8
    states = raw["states"].numpy()  # [2N, 4, 4]
    raw_actions = raw["raw_actions"].numpy()  # [2N, 4, 8, 4]（已 clip 执行）
    pair_ids = raw["pair_ids"].tolist()
    inst_ids = raw["inst_ids"].tolist()
    env_names = raw["env_names"]
    n_pairs = len(pair_ids) // 2

    # 1) 任务文本 → v5 的 instruction_id + language 行
    tasks = v5["metadata"]["tasks"]
    lang_hidden = v5["language_hidden"]
    lang_mask = v5["language_mask"]
    v5_inst_ids = v5["instruction_id"]
    task_to_v5_id = {t: i for i, t in enumerate(tasks)}
    # env 名 → 任务文本映射（与 train_fork_experts.ENV_TO_TASK 一致；
    # 2026-08-08 补全 4 对登记任务，文本已与 v5 metadata.tasks 逐一核对）
    env_to_task = {
        "drawer-close-v3": "Push and close a drawer",
        "drawer-open-v3": "Open a drawer",
        "faucet-close-v3": "Rotate the faucet clockwise",
        "faucet-open-v3": "Rotate the faucet counter-clockwise",
        "window-close-v3": "Push and close a window",
        "window-open-v3": "Push and open a window",
        "door-close-v3": "Close a door with a revolving joint",
        "door-open-v3": "Open a door with a revolving joint",
    }
    inst_a = task_to_v5_id[env_to_task[env_names[0]]]
    inst_b = task_to_v5_id[env_to_task[env_names[1]]]
    lang_a = lang_hidden[v5_inst_ids == inst_a][0]  # 该任务任一行即可（每任务同一文本）
    lang_b = lang_hidden[v5_inst_ids == inst_b][0]
    mask_a = lang_mask[v5_inst_ids == inst_a][0]
    mask_b = lang_mask[v5_inst_ids == inst_b][0]

    # 2) 帧 → flat-64 特征（原始预训练 backbone，与 v5 同域）
    from prepare_pnpw_features import VJEPA21Backbone
    from prepare_metaworld import preprocess_batch

    backbone = VJEPA21Backbone.from_pretrained(
        device=args.device, dtype=torch.float32, max_tokens=64, local_files_only=True
    )
    backbone.eval()
    n_rows = len(frames)
    vision = torch.empty(n_rows, 4, 64, 768, dtype=torch.float16)
    clips = [list(c) for c in frames.reshape(n_rows * 4, 4, 384, 384, 3)]
    with torch.inference_mode():
        for i in range(0, len(clips), 16):
            inputs = preprocess_batch(clips[i : i + 16], 384).to(args.device)
            flat, _ = backbone.forward_variants(inputs)
            vision[i // 4 : (i + 16) // 4] = (
                flat.reshape(-1, 4, 64, 768).cpu().half()
            )
    del backbone
    print(f"vision features: {tuple(vision.shape)}")

    # 3) 归一化（v5 契约）
    norm = v5["normalization"]
    q01, q99 = norm["action_q01"], norm["action_q99"]
    s_q01, s_q99 = norm["state_q01"], norm["state_q99"]
    # 动作：executed-clip（采集器已 clip）；归一化用 v5 分位数（同任务分布）
    actions = robust_normalize(raw_actions, q01, q99)
    actions = torch.from_numpy(actions).float()
    proprio = robust_normalize(states, s_q01, s_q99)
    proprio = torch.from_numpy(proprio).float()
    # prev：t=0 = 0（归一化零）；t>0 = 前一 chunk 末步标签
    previous = torch.zeros_like(actions[:, :, 0])
    previous[:, 1:] = actions[:, :-1, -1]

    # 4) pair/episode 元数据
    final_pair = torch.tensor(
        [FORK_PAIR_OFFSET + p for p in pair_ids], dtype=torch.int64
    )
    final_inst = torch.tensor(
        [inst_a if i == 0 else inst_b for i in inst_ids], dtype=torch.int64
    )
    final_ep = torch.arange(n_rows, dtype=torch.int64)
    lang = torch.stack([lang_a if i == 0 else lang_b for i in inst_ids])
    mask = torch.stack([mask_a if i == 0 else mask_b for i in inst_ids])

    out = {
        "vision_tokens": vision,
        "proprio": proprio,
        "previous_action": previous,
        "actions": actions,
        "language_hidden": lang,
        "language_mask": mask,
        "pair_id": final_pair,
        "instruction_id": final_inst,
        "episode_id": final_ep,
        "normalization": norm,
        "metadata": {
            "contract": "paired_fork",
            "tasks": tasks,
            "n_scenes": 1,
            "source": f"mw_fork:{env_names[0]}+{env_names[1]}",
            "n_pairs": n_pairs,
            "fork_pairs_offset": FORK_PAIR_OFFSET,
            "previous_action_contract": "v5_prevfix_20260807",
        },
    }
    torch.save(out, args.out)
    print(
        f"saved {args.out}: {n_rows} 行（{n_pairs} 对 fork，"
        f"inst {inst_a}/{inst_b}），动作/状态/特征与 v5 同构"
    )

    # 5) 合并版（C/D 组训练输入：v5 + fork 单数据集，pair_id 互不冲突）
    if args.merged is not None:
        merged = {k: v for k, v in v5.items() if k != "metadata"}
        for k in ("vision_tokens", "proprio", "previous_action", "actions",
                  "language_hidden", "language_mask", "pair_id",
                  "instruction_id", "episode_id"):
            merged[k] = torch.cat([v5[k], out[k]], dim=0)
        merged["metadata"] = {
            "contract": "merged_v5_fork",
            "tasks": tasks,
            "n_scenes": 1,
            "source": f"{v5['metadata'].get('contract','v5')}+{env_names[0]}/{env_names[1]}",
            "n_pairs": n_pairs,
            "fork_rows_start": len(v5["vision_tokens"]),
            "fork_pairs_offset": FORK_PAIR_OFFSET,
            "previous_action_contract": "v5_prevfix_20260807",
        }
        torch.save(merged, args.merged)
        print(f"saved merged {args.merged}: {len(merged['vision_tokens'])} 行")

    # 6) E 组：打乱配对（相同样本边际分布，pair 关系随机重排，2 行/组不同指令）
    if args.shuffled is not None:
        rng = np.random.default_rng(0)
        rows = torch.randperm(n_rows, generator=torch.Generator().manual_seed(0))
        new_pair = torch.zeros(n_rows, dtype=torch.int64)
        new_inst = torch.zeros(n_rows, dtype=torch.int64)
        new_ep = torch.arange(n_rows, dtype=torch.int64)
        for g in range(n_pairs):
            i, j = int(rows[2 * g]), int(rows[2 * g + 1])
            a, b = int(inst_ids[i]), int(inst_ids[j])
            if a == b:
                # 同指令撞组：交换 j 与下一组（或任意异指令行），保持 2 行/组
                for k in range(n_rows):
                    if int(inst_ids[k]) != a and k not in (i, j):
                        j = k
                        break
            new_pair[i] = new_pair[j] = FORK_PAIR_OFFSET + 1_000_000 + g
            new_inst[i] = inst_a if inst_ids[i] == 0 else inst_b
            new_inst[j] = inst_a if inst_ids[j] == 0 else inst_b
        out_shuffled = {k: v.clone() for k, v in out.items()}
        out_shuffled["pair_id"] = new_pair
        out_shuffled["instruction_id"] = new_inst
        out_shuffled["episode_id"] = new_ep
        out_shuffled["metadata"] = dict(out["metadata"])
        out_shuffled["metadata"]["contract"] = "paired_fork_shuffled"
        torch.save(out_shuffled, args.shuffled)
        print(f"saved shuffled {args.shuffled}（{n_pairs} 对，配对关系已打乱）")
    print("备注：D/E（λ=1）训练需 train.py paired 路径跳过非 fork 行（partner<0 零贡献）")


if __name__ == "__main__":
    main()
