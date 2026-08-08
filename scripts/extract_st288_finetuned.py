#!/usr/bin/env python
"""Stage B：用微调后的 V-JEPA（checkpoint 里的 vjepa_state_dict）重提取 ST288 特征。

动机：Stage B 训练是 live 在线编码（V-JEPA 全量解冻），评估时若用预计算
v5 特征（原始冻结 V-JEPA）会与微调权重不匹配；开环/语言消融必须用同一
backbone 重新编码。输出格式与 Stage A 的 mw_local288 一致（raw fp16 memmap
+ meta.pt），供 eval_mw_lang_ablation.py --local-slots-data 直接消费。

用法：
  python scripts/extract_st288_finetuned.py \
      --checkpoint checkpoints/stageB_langslot_40k.pt \
      --data data/metaworld_features_v5.pt \
      --output /media/ryan/robot-data/stageB_st288.npy \
      --meta /media/ryan/robot-data/stageB_st288_meta.pt \
      [--max-samples N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 根模块（prepare_pnpw_features 等）导入兼容


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/metaworld_features_v5.pt")
    p.add_argument("--output", type=Path, required=True, help="raw fp16 memmap npy")
    p.add_argument("--meta", type=Path, required=True, help="meta.pt（vision_tokens_st_npy 路径）")
    p.add_argument("--max-samples", type=int, default=0, help="0 = 全部（debug 用）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from prepare_pnpw_features import VJEPA21Backbone
    from va_compound.live_vjepa import (
        LiveVJEPADataset,
        _slot_coords,
        encode_live_frames,
    )

    root = Path(
        "/media/ryan/robot-data/datasets/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
    )
    t0 = time.time()
    dataset = LiveVJEPADataset(args.data, root, min_sequence_length=4)
    n_total = dataset.length
    print(f"dataset ready: {n_total} samples ({time.time()-t0:.0f}s)")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "vjepa_state_dict" not in ckpt:
        raise ValueError(f"{args.checkpoint} 没有 vjepa_state_dict 键")
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=torch.float32, max_tokens=144, local_files_only=True
    )
    vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
    vision_backbone.eval()
    print(f"backbone: loaded vjepa_state_dict from {args.checkpoint}")

    n = n_total if args.max_samples <= 0 else min(args.max_samples, n_total)
    D = 768
    seq = 4
    tok = 288
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(args.output, dtype=np.float16, mode="w+", shape=(n, seq, tok, D))
    coords = _slot_coords()

    batch_frames = []
    batch_indices = []
    for idx in range(n):
        item = dataset[idx]
        batch_frames.append(item["frames"])
        batch_indices.append(idx)
        if len(batch_frames) == args.batch_size or idx == n - 1:
            frames = np.stack(batch_frames)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                encoded = encode_live_frames(frames, vision_backbone, device)
            encoded = encoded.float().cpu().half().numpy()
            for local, global_idx in enumerate(batch_indices):
                mm[global_idx] = encoded[local]
            mm.flush()
            batch_frames = []
            batch_indices = []
            if (idx + 1) % 256 == 0 or idx == n - 1:
                print(f"  {idx+1}/{n} encoded ({time.time()-t0:.0f}s)")
    mm.flush()
    del mm

    meta = {
        "source": f"stageB_finetuned:{args.checkpoint.name}",
        "rows": n,
        "tokens_per_decision": tok,
        "grid": [24, 24],
        "slot_grid": 12,
        "pooling": "spatiotemporal",
    }
    torch.save(
        {
            "vision_tokens_st_npy": str(args.output),
            "coords": coords,  # np.ndarray（eval 侧 torch.from_numpy 消费，Stage A 同款契约）
            "metadata": meta,
        },
        args.meta,
    )
    print(f"done: {args.output} ({n}x{seq}x{tok}x{D} fp16) + {args.meta}")


if __name__ == "__main__":
    main()
