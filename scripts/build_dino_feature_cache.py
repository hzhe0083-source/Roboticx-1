"""Build DINO feature cache for DINO-main / DINO-metric training (2026-08-15).

训练步时 profile：在线 ViT-L 编码占 84%（2.97s/3.51s/步）。冻结塔确定性，
把全部唯一帧的 block11/block23 patch 特征离线预计算（fp16 memmap），训练
循环从缓存读。eval 仍在线编码（真实新帧），因此缓存必须与在线编码位级
一致——本脚本用与训练侧完全相同的预处理（bicubic 224 + ImageNet 归一化 +
同一 frozen tower），末尾抽 4 帧在线重算断言 torch.equal。

输出目录：
  <out>/block23.npy   [N, 256, 1024] fp16 memmap（行 i = index 第 i 个键）
  <out>/block11.npy   [N, 256, 1024] fp16 memmap
  <out>/index.pkl     {(task_file, ep_idx, frame_idx): row}
  <out>/meta.json     {frames, dataset_sha256, model_id, image_size, chunk}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from va_compound.longtraj_frames import LongTrajFramesDataset  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess_batch(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """与 train._dino_main_online_encode 完全相同的图像预处理。"""
    selected = np.ascontiguousarray(frames, dtype=np.uint8)
    images = torch.from_numpy(selected).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    if tuple(images.shape[-2:]) != (224, 224):
        images = F.interpolate(
            images, size=(224, 224), mode="bicubic",
            align_corners=False, antialias=True,
        )
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-vision-checkpoint", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from train import _build_dino_main_backbone
    from va_compound.model import VACompoundConfig

    dataset = LongTrajFramesDataset(args.data, min_sequence_length=4)
    # 唯一帧 → 行号（首见顺序，确定性）。
    index: OrderedDict = OrderedDict()
    for task_file, ep_idx, fidx in dataset.refs:
        for row in fidx:
            for f in row:
                key = (task_file, int(ep_idx), int(f))
                if key not in index:
                    index[key] = len(index)
    n_frames = len(index)
    keys = list(index.keys())
    print(f"unique frames: {n_frames}", flush=True)

    config = VACompoundConfig(
        main_vision_backbone="dinov2_vitl14_reg4",
        main_vision_model_id="vit_large_patch14_reg4_dinov2.lvd142m",
        main_vision_image_size=224,
        main_vision_dim=1024,
        main_vision_grid=8,
        main_vision_frames=4,
        main_vision_tokens=256,
    )
    import argparse as _argparse

    backbone = _build_dino_main_backbone(
        _argparse.Namespace(
            main_vision_checkpoint=args.main_vision_checkpoint.expanduser().absolute()
        ),
        config,
        torch.device(args.device),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    from numpy.lib.format import open_memmap

    fp23 = open_memmap(args.out / "block23.npy", mode="w+", dtype=np.float16,
                       shape=(n_frames, 256, 1024))
    fp11 = open_memmap(args.out / "block11.npy", mode="w+", dtype=np.float16,
                       shape=(n_frames, 256, 1024))

    # 按行号取帧（首见顺序），chunk 批编码（与训练 encode_batch 同大小）。
    decoded_tasks: dict[str, list] = {}
    t0 = __import__("time").time()
    for start in range(0, n_frames, args.chunk):
        chunk_keys = keys[start:start + args.chunk]
        frames = np.empty((len(chunk_keys), 480, 480, 3), dtype=np.uint8)
        for i, (task_file, ep_idx, frame_idx) in enumerate(chunk_keys):
            if task_file not in decoded_tasks:
                decoded_tasks[task_file] = dataset._decode_task(task_file)
            frames[i] = decoded_tasks[task_file][ep_idx][frame_idx]
        inputs = preprocess_batch(frames, torch.device(args.device))
        hierarchical = backbone.forward_hierarchical_dense(inputs.half())
        for i, row in enumerate(range(start, start + len(chunk_keys))):
            fp23[row] = hierarchical[11][i].cpu().numpy().astype(np.float16)
            fp11[row] = hierarchical[5][i].cpu().numpy().astype(np.float16)
        if (start + len(chunk_keys)) % 1000 < args.chunk or start + len(chunk_keys) == n_frames:
            print(f"  encoded {start+len(chunk_keys)}/{n_frames} "
                  f"({__import__('time').time()-t0:.0f}s)", flush=True)
    fp23.flush()
    fp11.flush()

    # 位级一致性验证：抽 4 帧，用完全相同的预处理 + 塔在线重算 block 特征。
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_frames, size=min(4, n_frames), replace=False)
    sample_keys = [keys[int(i)] for i in sample_idx]
    frames = np.stack([
        np.ascontiguousarray(decoded_tasks[t][e][f], dtype=np.uint8)
        for (t, e, f) in sample_keys
    ])
    inputs = preprocess_batch(frames, torch.device(args.device))
    hierarchical = backbone.forward_hierarchical_dense(inputs.half())
    for i, row in enumerate(sample_idx):
        ok23 = torch.equal(
            torch.from_numpy(fp23[int(row)]),
            hierarchical[11][i].cpu().half(),
        )
        ok11 = torch.equal(
            torch.from_numpy(fp11[int(row)]),
            hierarchical[5][i].cpu().half(),
        )
        print(f"  verify row {int(row)}: block23 bit-identical={ok23}, "
              f"block11 bit-identical={ok11}", flush=True)
        assert ok23 and ok11, "cache != online encode"

    with (args.out / "index.pkl").open("wb") as fh:
        pickle.dump(dict(index), fh)
    digest = hashlib.sha256()
    with args.data.expanduser().open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    meta = {
        "frames": n_frames,
        "dataset_sha256": digest.hexdigest(),
        "model_id": "vit_large_patch14_reg4_dinov2.lvd142m",
        "image_size": 224,
        "chunk": args.chunk,
        "grid": 8,
        "window": 4,
    }
    with (args.out / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"cache written: {args.out} ({n_frames} frames)", flush=True)


if __name__ == "__main__":
    main()
