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

import gc
import io
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
ACTION_MASK_KEYS = ("action_valid_mask", "horizon_mask")
PEER_SYNC_H6_CONTRACT = "peer_sync_h6_world_windows_v1"
PEER_SYNC_H15_P2_CONTRACT = "peer_sync_h15_p2_world_windows_v1"
# One decoded-task table per process. Peer training builds two
# LongTrajFramesDataset objects (VA + World); a per-dataset LRU of size 1
# still keeps two full 480px caches and re-decodes on every task switch.
# hard2 assembly-v3 is ~38 GiB decoded; two ranks × two copies plus a
# third in-flight decode exceeds the 240 GiB notebook cgroup.
_PROCESS_DECODED: OrderedDict[str, list[list[np.ndarray]]] = OrderedDict()
_PROCESS_DECODED_CAP = 1


def _decoded_cap(requested: int) -> int:
    global _PROCESS_DECODED_CAP
    _PROCESS_DECODED_CAP = max(_PROCESS_DECODED_CAP, max(1, int(requested)))
    return _PROCESS_DECODED_CAP


def _evict_decoded(keep: str | None = None) -> None:
    while len(_PROCESS_DECODED) >= _PROCESS_DECODED_CAP:
        oldest = next(iter(_PROCESS_DECODED))
        if oldest == keep:
            if len(_PROCESS_DECODED) == 1:
                return
            _PROCESS_DECODED.move_to_end(oldest)
            oldest = next(iter(_PROCESS_DECODED))
        del _PROCESS_DECODED[oldest]
        gc.collect()


def mtvj_collate(batch: list[dict]) -> dict:
    """MT-VJ 自定义 collate（2026-08-10，Codex P1-13 优化）：frames 是
    [T,W,480,480,3] uint8 大数组，default_collate 逐元素 numpy→tensor 转换
    极慢（176MB batch ~2s/批）；这里一次 np.stack，其余字段 torch 化。"""
    out: dict[str, object] = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if key in {"frames", "world_target_frames"}:
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
                 decode_cache_tasks: int = 1,
                 feature_cache: str | Path | None = None,
                 include_frames: bool = True,
                 include_world_target_frames: bool = False) -> None:
        self.path = Path(path)
        self.longtraj_dir = Path(longtraj_dir) if longtraj_dir else ROOT / "data"
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)
        missing = [k for k in self.REQUIRED if k not in self.payload]
        if missing:
            raise ValueError(f"missing tensors in dataset: {missing}")
        actions = self.payload["actions"]
        if not isinstance(actions, torch.Tensor) or actions.ndim != 4:
            raise ValueError("actions must have shape [N,T,H,A]")
        self.length = int(actions.shape[0])
        if self.length == 0:
            raise ValueError("training dataset is empty")
        metadata = self.payload.get("metadata") or {}
        if metadata.get("contract") == PEER_SYNC_H6_CONTRACT:
            if tuple(actions.shape[1:]) != (4, 6, 4):
                raise ValueError(
                    f"{PEER_SYNC_H6_CONTRACT} requires exact T4/H6/A4, "
                    f"got {tuple(actions.shape[1:])}"
                )
            if metadata.get("logged_action_chunk") != "full_h6":
                raise ValueError(f"{PEER_SYNC_H6_CONTRACT} requires full logged H6 chunk")
            for key in ("parent_identity", "source_identities", "output_identity"):
                if not metadata.get(key):
                    raise ValueError(f"{PEER_SYNC_H6_CONTRACT} requires metadata.{key}")
        if metadata.get("contract") == PEER_SYNC_H15_P2_CONTRACT:
            if tuple(actions.shape[1:]) != (4, 15, 4):
                raise ValueError(
                    f"{PEER_SYNC_H15_P2_CONTRACT} requires exact T4/H15/A4, "
                    f"got {tuple(actions.shape[1:])}"
                )
            if metadata.get("logged_action_chunk") != "full_h15":
                raise ValueError(
                    f"{PEER_SYNC_H15_P2_CONTRACT} requires full logged H15 chunk"
                )
        self.refs = self.payload["frame_refs"]  # [(task_file, ep_idx, frame_idx[T,W])]
        if len(self.refs) != self.length:
            raise ValueError("frame_refs 长度与样本数不一致")
        self.world_target_refs = self.payload.get("world_target_frame_refs")
        self.include_world_target_frames = bool(include_world_target_frames)
        if self.include_world_target_frames and (
            not isinstance(self.world_target_refs, (list, tuple))
            or len(self.world_target_refs) != self.length
        ):
            raise ValueError(
                "include_world_target_frames requires one world_target_frame_ref per sample"
            )
        for key in ACTION_MASK_KEYS:
            if key in self.payload:
                value = self.payload[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != self.payload["actions"].shape[:-1]
                ):
                    raise ValueError(
                        f"{key} must have shape {tuple(self.payload['actions'].shape[:-1])}, "
                        f"got {getattr(value, 'shape', None)}"
                    )
        # 帧窗契约校验：每个决策点的帧窗必须是历史帧（决策点 d 用 d-(W-1)*stride..d）
        self._task_cache: dict[str, dict] = {}
        # 任务级预解码缓存（Codex P1-13 优化，2026-08-10）：locality sampler 下
        # 同任务 batch 连续 → 解码帧驻留内存（1 任务 ≈ 5.3GB），JPEG 只在任务
        # 切换时解码一次（~60s/任务）；配合 num_workers=0 单 worker 防多份拷贝。
        self.decode_cache_tasks = _decoded_cap(decode_cache_tasks)
        # DINO 特征缓存（2026-08-15）：feature_cache 给定时每个样本返回其
        # 帧窗在缓存中的行号（frame_cache_rows [T, W] int64），不再解 JPEG
        # 帧（include_frames=False 时无 frames 键）——训练循环从预计算特征读，
        # 跳过在线 ViT-L 编码（占步时 84%）。
        self.feature_cache = Path(feature_cache) if feature_cache else None
        self.include_frames = bool(include_frames)
        self.cache_rows: np.ndarray | None = None
        self.cached_raw_frames: np.ndarray | None = None
        if self.feature_cache is not None:
            import pickle

            with (self.feature_cache / "index.pkl").open("rb") as fh:
                cache_index: dict = pickle.load(fh)
            rows = np.empty((self.length, len(self.refs[0][2]), len(self.refs[0][2][0])),
                            dtype=np.int64)
            for i, (task_file, ep_idx, fidx) in enumerate(self.refs):
                for t, row in enumerate(fidx):
                    for w, f in enumerate(row):
                        key = (task_file, int(ep_idx), int(f))
                        if key not in cache_index:
                            raise KeyError(
                                f"feature cache 缺少帧 {key}（样本 {i}）；"
                                "缓存与数据集不匹配"
                            )
                        rows[i, t, w] = cache_index[key]
            self.cache_rows = rows
            raw_path = self.feature_cache / "raw_frames.npy"
            if self.include_frames and raw_path.is_file():
                import json

                meta_path = self.feature_cache / "meta.json"
                meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
                if (
                    meta.get("raw_frame_contract")
                    != "exact_decoded_longtraj_jpeg_480_v1"
                    or meta.get("raw_frame_shape")
                    != [len(cache_index), 480, 480, 3]
                    or meta.get("raw_frame_dtype") != "uint8"
                    or not meta.get("raw_frames_sha256")
                ):
                    raise ValueError(
                        "cached raw frames lack exact 480px identity metadata"
                    )
                cached_raw = np.load(raw_path, mmap_mode="r")
                if cached_raw.shape != (len(cache_index), 480, 480, 3):
                    raise ValueError(
                        "cached raw frames must have shape "
                        f"[{len(cache_index)},480,480,3], got {cached_raw.shape}"
                    )
                if cached_raw.dtype != np.uint8:
                    raise ValueError("cached raw frames must be uint8")
                self.cached_raw_frames = cached_raw

    def _decode_task(self, task_file: str) -> list[list[np.ndarray]]:
        cached = _PROCESS_DECODED.get(task_file)
        if cached is not None:
            _PROCESS_DECODED.move_to_end(task_file)
            return cached
        # Evict before decoding so a process never holds LRU + in-flight frames.
        # The cache is process-wide: VA and World datasets share it.
        _evict_decoded()
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
        _evict_decoded(keep=task_file)
        _PROCESS_DECODED[task_file] = decoded
        _PROCESS_DECODED.move_to_end(task_file)
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
        for key in (
            "episode_id",
            "world_rank_shuffle_action",
            "world_rank_shuffle_mask",
            "world_target_valid_mask",
        ):
            if key in self.payload:
                item[key] = self.payload[key][index]
        for key in ACTION_MASK_KEYS:
            if key in self.payload:
                item[key] = self.payload[key][index]
        if "language_mask" in self.payload:
            item["language_mask"] = self.payload["language_mask"][index]
        if self.cache_rows is not None:
            # DINO 特征缓存模式：返回帧窗行号（缓存读取代在线编码）。
            item["frame_cache_rows"] = self.cache_rows[index]
            if not self.include_frames:
                return item
            if self.cached_raw_frames is not None:
                item["frames"] = np.asarray(
                    self.cached_raw_frames[self.cache_rows[index]], dtype=np.uint8
                )
                return item
        task_file, ep_idx, fidx = self.refs[index]
        ep_frames = self._decode_task(task_file)[ep_idx]  # 已解码 ndarray
        frames = np.stack([
            np.stack([ep_frames[int(f)] for f in row])
            for row in fidx
        ])  # [T, W, 480, 480, 3] uint8（零拷贝引用，与 phase2 契约一致）
        item["frames"] = frames
        if self.include_world_target_frames:
            target_file, target_ep, target_idx = self.world_target_refs[index]
            target_frames = self._decode_task(target_file)[int(target_ep)]
            item["world_target_frames"] = np.stack(
                [
                    np.stack([target_frames[int(frame)] for frame in row])
                    for row in target_idx
                ]
            )
        return item
