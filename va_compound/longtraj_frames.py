"""LongTrajFramesDataset：MT-VJ 在线编码的帧数据源（2026-08-10）。

从 data/metaworld_longtraj_windows_h48.pt（预计算契约，含 frame_refs）+
data/metaworld_longtraj_{task}.pt（JPEG 压缩帧）按帧索引解码，输出与
LiveVJEPADataset 同契约：全部既有键 + ``frames [T, W, H, W, 3] uint8``
（480 原尺寸，resize 到 384 由训练侧 GPU 预处理完成）。

为什么需要：MT-VJ（--dense-readout-mtvj）在线编码需要原始帧，而 longtraj
数据是 JPEG 压缩（不在 lerobot 原始数据集里），LiveVJEPADataset 的 root
契约不适用。本类直接按 (task_file, ep_idx, frame_idx) 解码，训练/评估同构。
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def mtvj_collate(batch: list[dict]) -> dict:
    """MT-VJ 自定义 collate（2026-08-10，Codex P1-13 优化）：frames 是
    [T,W,480,480,3] uint8 大数组，default_collate 逐元素 numpy→tensor 转换
    极慢（176MB batch ~2s/批）；这里一次 np.stack，其余字段 torch 化。"""
    out: dict[str, object] = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if key == "frames":
            out[key] = np.stack(values)  # [B, T, W, H, W, 3] uint8
        else:
            out[key] = torch.stack([torch.as_tensor(v) for v in values])
    return out


class LongTrajFramesDataset:
    """windows payload + longtraj JPEG 帧 → 训练 batch（含 frames 键）。

    注意：与 FeatureDataset 的差异仅视觉侧——``vision_tokens`` 键不存在
    （MT-VJ 在线编码从 frames 现算 dense evidence）；actions/prev/proprio/
    language/instruction_id 语义与 windows 文件完全一致。
    """

    REQUIRED = (
        "actions",
        "previous_action",
        "proprio",
        "language_hidden",
        "instruction_id",
        "pair_id",
    )

    def __init__(self, path: str | Path, longtraj_dir: str | Path | None = None,
                 min_sequence_length: int = 4,
                 decode_cache_tasks: int = 1) -> None:
        self.path = Path(path)
        self.longtraj_dir = Path(longtraj_dir) if longtraj_dir else ROOT / "data"
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)
        missing = [k for k in self.REQUIRED if k not in self.payload]
        if missing:
            raise ValueError(f"missing tensors in dataset: {missing}")
        self.length = int(self.payload["actions"].shape[0])
        if self.length == 0:
            raise ValueError("training dataset is empty")
        self.refs = self.payload["frame_refs"]  # [(task_file, ep_idx, frame_idx[T,W])]
        if len(self.refs) != self.length:
            raise ValueError("frame_refs 长度与样本数不一致")
        # 帧窗契约校验：每个决策点的帧窗必须是历史帧（决策点 d 用 d-(W-1)*stride..d）
        self._task_cache: dict[str, dict] = {}
        # 任务级预解码缓存（Codex P1-13 优化，2026-08-10）：locality sampler 下
        # 同任务 batch 连续 → 解码帧驻留内存（1 任务 ≈ 5.3GB），JPEG 只在任务
        # 切换时解码一次（~60s/任务）；配合 num_workers=0 单 worker 防多份拷贝。
        self._decoded: dict[str, list[list[np.ndarray]]] = {}
        self.decode_cache_tasks = max(1, int(decode_cache_tasks))

    def _decode_task(self, task_file: str) -> list[list[np.ndarray]]:
        cached = self._decoded.get(task_file)
        if cached is not None:
            return cached
        data = self._load_task(task_file)
        # 多线程解码（PIL JPEG 解码在 C 层释放 GIL，8 线程 ~8 倍加速）；
        # 存 480 原尺寸（resize 交给 GPU 预处理，phase2 同款，避免 CPU bicubic）。
        t0 = time.time()
        decoded = []
        for ep in data["episodes"]:
            ep_frames = ep["frames"]
            with ThreadPoolExecutor(max_workers=8) as pool:
                frames = list(pool.map(
                    lambda b: np.asarray(
                        Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8
                    ),
                    ep_frames,
                ))
            decoded.append(frames)
        print(f"  [longtraj] 解码 {task_file}: {sum(len(e) for e in decoded)} 帧, "
              f"{time.time()-t0:.0f}s", flush=True)
        while len(self._decoded) >= self.decode_cache_tasks:
            oldest = next(iter(self._decoded))
            del self._decoded[oldest]
        self._decoded[task_file] = decoded
        return decoded

    def _load_task(self, task_file: str) -> dict:
        data = self._task_cache.get(task_file)
        if data is None:
            data = torch.load(
                self.longtraj_dir / f"metaworld_longtraj_{task_file}.pt",
                map_location="cpu",
                weights_only=False,
            )
            # 只保留最近 2 个任务文件（各 ~300MB 压缩态），防内存爆（oomd 教训）
            if len(self._task_cache) >= 2:
                oldest = next(iter(self._task_cache))
                del self._task_cache[oldest]
            self._task_cache[task_file] = data
        return data

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        item = {key: self.payload[key][index] for key in self.REQUIRED}
        if "language_mask" in self.payload:
            item["language_mask"] = self.payload["language_mask"][index]
        task_file, ep_idx, fidx = self.refs[index]
        ep_frames = self._decode_task(task_file)[ep_idx]  # 已解码 ndarray
        frames = np.stack([
            np.stack([ep_frames[int(f)] for f in row])
            for row in fidx
        ])  # [T, W, 384, 384, 3] uint8（零拷贝引用，与 phase2 契约一致）
        item["frames"] = frames
        return item
