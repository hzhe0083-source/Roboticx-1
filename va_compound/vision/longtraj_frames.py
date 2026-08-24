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
import hashlib
import io
import json
import random
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
PEER_SYNC_H15_P15_CONTRACT = "peer_sync_h15_p15_world_windows_v1"
PEER_SYNC_H15_CONTRACTS = {
    PEER_SYNC_H15_P2_CONTRACT,
    PEER_SYNC_H15_P15_CONTRACT,
}
ONLINE_EPISODE_CONTRACT = "full_episode_online_random_h15_v1"
ONLINE_ACTION_DONOR_CONTRACT = "online_cross_episode_random_action_v1"
# One decoded-task table per process. Peer training builds two
# LongTrajFramesDataset objects (VA + World); a per-dataset LRU of size 1
# still keeps two full 480px caches and re-decodes on every task switch.
# hard2 assembly-v3 is ~38 GiB decoded; two ranks × two copies plus a
# third in-flight decode exceeds the 240 GiB notebook cgroup.
_PROCESS_DECODED: OrderedDict[str, list[list[np.ndarray]]] = OrderedDict()
_PROCESS_DECODED_CAP = 1
# Online sampling only needs the frames referenced by the current random crop.
# Keep a small process-wide LRU because VA and World datasets share the same
# raw episodes, and use one persistent pool instead of rebuilding a pool for
# every episode.
_PROCESS_FRAME_CACHE: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
_PROCESS_FRAME_CACHE_CAP = 2048
_PROCESS_RAW_TASKS: OrderedDict[str, dict] = OrderedDict()
_PROCESS_RAW_TASKS_CAP = 2
_PROCESS_JPEG_POOL = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="online-jpeg"
)


def _decode_jpeg_bytes(encoded: bytes) -> np.ndarray:
    return np.asarray(
        Image.open(io.BytesIO(encoded)).convert("RGB"), dtype=np.uint8
    )


def _decode_sparse_episode_frames(
    source_id: str,
    episode_index: int,
    encoded_frames: list[bytes],
    frame_indices: list[int],
) -> list[np.ndarray]:
    """Decode only unique frame indices referenced by one online crop."""

    requested = [int(index) for index in frame_indices]
    missing_keys: list[tuple[str, int, int]] = []
    missing_bytes: list[bytes] = []
    for frame_index in dict.fromkeys(requested):
        key = (source_id, int(episode_index), frame_index)
        cached = _PROCESS_FRAME_CACHE.get(key)
        if cached is not None:
            _PROCESS_FRAME_CACHE.move_to_end(key)
            continue
        missing_keys.append(key)
        missing_bytes.append(encoded_frames[frame_index])

    if missing_bytes:
        decoded = _PROCESS_JPEG_POOL.map(_decode_jpeg_bytes, missing_bytes)
        for key, frame in zip(missing_keys, decoded, strict=True):
            while len(_PROCESS_FRAME_CACHE) >= _PROCESS_FRAME_CACHE_CAP:
                _PROCESS_FRAME_CACHE.popitem(last=False)
            _PROCESS_FRAME_CACHE[key] = frame

    result = []
    for frame_index in requested:
        key = (source_id, int(episode_index), frame_index)
        frame = _PROCESS_FRAME_CACHE[key]
        _PROCESS_FRAME_CACHE.move_to_end(key)
        result.append(frame)
    return result


def _load_process_raw_task(source: Path) -> dict:
    """Share compressed episode data between the VA and World datasets."""

    key = str(source.resolve(strict=False))
    cached = _PROCESS_RAW_TASKS.get(key)
    if cached is not None:
        _PROCESS_RAW_TASKS.move_to_end(key)
        return cached
    data = torch.load(source, map_location="cpu", weights_only=False)
    while len(_PROCESS_RAW_TASKS) >= _PROCESS_RAW_TASKS_CAP:
        _PROCESS_RAW_TASKS.popitem(last=False)
    _PROCESS_RAW_TASKS[key] = data
    return data


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
        if metadata.get("contract") in PEER_SYNC_H15_CONTRACTS:
            contract = metadata["contract"]
            if tuple(actions.shape[1:]) != (4, 15, 4):
                raise ValueError(
                    f"{contract} requires exact T4/H15/A4, "
                    f"got {tuple(actions.shape[1:])}"
                )
            if metadata.get("logged_action_chunk") != "full_h15":
                raise ValueError(
                    f"{contract} requires full logged H15 chunk"
                )
            if contract == PEER_SYNC_H15_P15_CONTRACT:
                required_cadence = {
                    "planning_stride": 15,
                    "control_stride": 15,
                    "decision_offsets": [0, 15, 30, 45],
                    "world_target_horizon": 15,
                    "world_target_offsets": [15, 30, 45, 60],
                }
                mismatches = {
                    key: (metadata.get(key), expected)
                    for key, expected in required_cadence.items()
                    if metadata.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        f"{contract} cadence metadata mismatch: {mismatches}"
                    )
                previous_action = self.payload.get("previous_action")
                if (
                    not isinstance(previous_action, torch.Tensor)
                    or tuple(previous_action.shape) != tuple(actions.shape[:2]) + (4,)
                ):
                    raise ValueError(
                        f"{contract} requires previous_action [N,4,4]"
                    )
                if not torch.equal(previous_action[:, 1:], actions[:, :-1, 14]):
                    raise ValueError(
                        f"{contract} requires each next previous_action to equal "
                        "the prior P15 segment token14"
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


class OnlineLongTrajEpisodeDataset(LongTrajFramesDataset):
    """Full episodes -> random overlapping H15 training samples at access time.

    The JSON index contains episode membership and raw-file identities only.  It
    deliberately contains no action chunks, frame windows, or crop starts.  A
    sample start ``d`` is selected deterministically from ``(seed, epoch,
    episode, sample_slot)`` so exact resume is stable while every epoch sees new
    overlapping portions of the trajectory.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        longtraj_dir: str | Path | None = None,
        samples_per_episode: int = 6,
        sampling_seed: int = 0,
        decode_cache_tasks: int = 1,
        include_world_target_frames: bool = False,
    ) -> None:
        if split != "train":
            raise ValueError("online training dataset currently requires split='train'")
        if samples_per_episode < 1:
            raise ValueError("samples_per_episode must be positive")
        self.path = Path(path).expanduser().resolve(strict=True)
        index = json.loads(self.path.read_text(encoding="utf-8"))
        if index.get("contract") != ONLINE_EPISODE_CONTRACT:
            raise ValueError(
                f"unexpected online episode contract: {index.get('contract')!r}"
            )
        protocol = index.get("sampling_protocol") or {}
        required_protocol = {
            "sequence_length": 4,
            "action_horizon": 15,
            "decision_stride": 15,
            "crop_start_stride": 1,
            "world_target_horizon": 15,
        }
        bad = {
            key: (protocol.get(key), value)
            for key, value in required_protocol.items()
            if protocol.get(key) != value
        }
        if bad:
            raise ValueError(f"online episode sampling protocol mismatch: {bad}")

        tasks = list(index.get("tasks") or [])
        episodes = [
            dict(item) for item in index.get("episodes") or []
            if item.get("split") == split
        ]
        if not tasks or not episodes:
            raise ValueError("online episode index has no train episodes")
        task_ids = [int(item["task_id"]) for item in tasks]
        if task_ids != list(range(len(tasks))):
            raise ValueError("online episode task ids must be contiguous from zero")
        if any(int(item.get("valid_start_count", 0)) <= 0 for item in episodes):
            raise ValueError("every indexed train episode must have a valid online start")

        reference_path = Path(str(index["language_reference"]["path"]))
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        descriptions = list((reference.get("metadata") or {}).get("tasks") or [])
        if descriptions != [str(item["description"]) for item in tasks]:
            raise ValueError("online index task descriptions differ from language cache")
        language_hidden = reference.get("language_hidden")
        language_mask = reference.get("language_mask")
        normalization = reference.get("normalization") or {}
        if (
            not isinstance(language_hidden, torch.Tensor)
            or not isinstance(language_mask, torch.Tensor)
            or language_hidden.shape[0] != len(tasks)
            or language_mask.shape[0] != len(tasks)
        ):
            raise ValueError("online language reference must contain one row per task")
        for key in ("action_q01", "action_q99", "state_q01", "state_q99"):
            if not isinstance(normalization.get(key), torch.Tensor):
                raise ValueError(f"online language reference lacks normalization.{key}")

        self.index = index
        self.longtraj_dir = (
            Path(longtraj_dir).expanduser().resolve(strict=False)
            if longtraj_dir is not None else self.path.parent
        )
        self.samples_per_episode = int(samples_per_episode)
        self.sampling_seed = int(sampling_seed)
        self.epoch = 0
        self.include_world_target_frames = bool(include_world_target_frames)
        self.include_frames = True
        self.feature_cache = None
        self.cache_rows = None
        self.cached_raw_frames = None
        self.decode_cache_tasks = _decoded_cap(decode_cache_tasks)
        self._task_cache: dict[str, dict] = {}
        self._semantics_cache: dict[tuple[str, int], dict] = {}
        self._valid_starts_cache: dict[tuple[str, int], list[int]] = {}
        self._reference_hidden = language_hidden
        self._reference_mask = language_mask
        self._aq01 = normalization["action_q01"].cpu().numpy()
        self._aq99 = normalization["action_q99"].cpu().numpy()
        self._sq01 = normalization["state_q01"].cpu().numpy()
        self._sq99 = normalization["state_q99"].cpu().numpy()
        self.task_language_hidden = language_hidden
        self.task_language_mask = language_mask
        self.model_schema = {
            "language_dim": int(language_hidden.shape[-1]),
            "action_horizon": 15,
            "action_dim": int(self._aq01.shape[0]),
            "proprio_dim": int(self._sq01.shape[0]),
        }
        self._source_by_task = {
            str(item["task"]): Path(str(item["source_path"])) for item in tasks
        }
        self._episodes = episodes
        self._episodes_by_task: dict[int, list[dict]] = {}
        for entry in episodes:
            self._episodes_by_task.setdefault(int(entry["task_id"]), []).append(entry)
        self._rows = [
            (entry, slot)
            for entry in self._episodes
            for slot in range(self.samples_per_episode)
        ]
        self.length = len(self._rows)

        instruction = torch.tensor(
            [int(entry["task_id"]) for entry, _ in self._rows], dtype=torch.long
        )
        episode_id = torch.tensor(
            [int(entry["episode_id"]) for entry, _ in self._rows], dtype=torch.long
        )
        raw_sources = [
            {
                "path": str(item["source_path"]),
                "sha256": str(item["sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in tasks
        ]
        self.payload = {
            "instruction_id": instruction,
            "episode_id": episode_id,
            "pair_id": torch.arange(self.length, dtype=torch.long),
            # Zero-row schema tensors preserve exact-resume identity without
            # storing any offline crop or action label. Runtime rows are built
            # exclusively from the selected full episode in ``__getitem__``.
            "language_hidden": self._reference_hidden,
            "language_mask": self._reference_mask,
            "actions": torch.empty((0, 4, 15, int(self._aq01.shape[0]))),
            "proprio": torch.empty((0, 4, int(self._sq01.shape[0]))),
            "metadata": {
                "contract": ONLINE_EPISODE_CONTRACT,
                "tasks": descriptions,
                "split_name": split,
                "sequence_length": 4,
                "action_horizon": 15,
                "control_stride": 15,
                "planning_stride": 15,
                "crop_start_stride": 1,
                "world_target_horizon": 15,
                "samples_per_episode": self.samples_per_episode,
                "episode_count": len(self._episodes),
                "index_path": str(self.path),
                "index_sha256": _sha256_file(self.path),
                "raw_sources": raw_sources,
            },
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("online dataset epoch must be non-negative")
        self.epoch = int(epoch)

    def _task_source(self, task_file: str) -> Path:
        declared = self._source_by_task[task_file]
        canonical = self.longtraj_dir / f"metaworld_longtraj_{task_file}.pt"
        return declared if declared.is_file() else canonical

    def _load_task(self, task_file: str) -> dict:
        # The episode index is the authority.  ``longtraj_dir`` is only a
        # portability fallback; otherwise a repaired source could silently be
        # shadowed by the older canonical 60ep file.
        source = self._task_source(task_file)
        data = _load_process_raw_task(source)
        if data.get("task") != task_file:
            raise ValueError(f"{source}: expected task={task_file!r}")
        return data

    def _decode_episode_frames(
        self, task_file: str, episode_index: int, frame_indices: list[int]
    ) -> list[np.ndarray]:
        episode = self._load_task(task_file)["episodes"][int(episode_index)]
        return _decode_sparse_episode_frames(
            str(self._task_source(task_file).resolve(strict=False)),
            int(episode_index),
            episode["frames"],
            frame_indices,
        )

    @staticmethod
    def _file_sha_seed(*parts: object) -> int:
        encoded = ":".join(map(str, parts)).encode("utf-8")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")

    def _episode_semantics(self, task: str, episode_index: int) -> dict:
        key = (task, int(episode_index))
        cached = self._semantics_cache.get(key)
        if cached is not None:
            return cached
        from scripts.build_longtraj_features import resolve_episode_semantics

        episode = self._load_task(task)["episodes"][episode_index]
        semantics = resolve_episode_semantics(
            episode, f"{task}:episode[{episode_index}]", legacy_policy="infer"
        )
        self._semantics_cache[key] = semantics
        return semantics

    def _valid_starts(self, entry: dict) -> list[int]:
        task = str(entry["task"])
        episode_index = int(entry["episode_index"])
        key = (task, episode_index)
        cached = self._valid_starts_cache.get(key)
        if cached is not None:
            return cached
        episode = self._load_task(task)["episodes"][episode_index]
        semantics = self._episode_semantics(task, episode_index)
        length = len(episode["actions"])
        starts = []
        for start in range(max(0, length - 60)):
            decisions = start + np.arange(4) * 15
            targets = decisions[:, None] + np.arange(15)[None, :]
            valid = semantics["valid"][targets].copy()
            perturb_start = semantics["perturb_start"]
            if perturb_start is not None:
                unseen = (
                    semantics["recovery"][targets]
                    & (decisions[:, None] < int(perturb_start))
                )
                valid &= ~unseen
            if bool(valid.any()):
                starts.append(start)
        if len(starts) != int(entry["valid_start_count"]):
            raise ValueError(
                f"{task}:episode[{episode_index}] valid-start count changed: "
                f"{len(starts)} != {entry['valid_start_count']}"
            )
        self._valid_starts_cache[key] = starts
        return starts

    def _select_start(self, entry: dict, slot: int, *, donor: bool = False) -> int:
        starts = self._valid_starts(entry)
        seed = self._file_sha_seed(
            self.sampling_seed,
            self.epoch,
            int(entry["episode_id"]),
            "donor" if donor else "main",
        )
        rng = random.Random(seed)
        if len(starts) >= self.samples_per_episode:
            return rng.sample(starts, self.samples_per_episode)[slot]
        rng.shuffle(starts := list(starts))
        return starts[slot % len(starts)]

    @staticmethod
    def _normalize(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (values - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    def _crop(self, entry: dict, start: int) -> dict[str, np.ndarray | int | None]:
        task = str(entry["task"])
        episode_index = int(entry["episode_index"])
        episode = self._load_task(task)["episodes"][episode_index]
        semantics = self._episode_semantics(task, episode_index)
        decisions = start + np.arange(4) * 15
        targets = decisions[:, None] + np.arange(15)[None, :]
        endpoints = decisions + 15
        action_valid = semantics["valid"][targets].copy()
        perturb_start = semantics["perturb_start"]
        if perturb_start is not None:
            action_valid &= ~(
                semantics["recovery"][targets]
                & (decisions[:, None] < int(perturb_start))
            )
        actions = np.asarray(episode["actions"], dtype=np.float32)
        states = np.asarray(episode["states"], dtype=np.float32)
        previous = np.stack([
            np.zeros(4, dtype=np.float32) if decision == 0 else actions[decision - 1]
            for decision in decisions
        ])
        return {
            "actions": self._normalize(actions[targets], self._aq01, self._aq99).astype(np.float32),
            "previous_action": self._normalize(previous, self._aq01, self._aq99).astype(np.float32),
            "proprio": self._normalize(states[decisions], self._sq01, self._sq99).astype(np.float32),
            "action_valid_mask": action_valid,
            "recovery_mask": semantics["recovery"][targets],
            "decision_recovery": semantics["recovery"][decisions],
            "door_metric_state": semantics["metric_state"][decisions].astype(np.float32),
            "door_metric_state_valid": semantics["metric_valid"][decisions],
            "world_target_valid_mask": (
                action_valid.all(axis=1) & semantics["frame_valid"][endpoints]
            ),
            "first_success": -1 if semantics["first_success"] is None else int(semantics["first_success"]),
            "decisions": decisions,
            "endpoints": endpoints,
        }

    def _donor_entry(self, entry: dict, slot: int) -> dict:
        candidates = self._episodes_by_task[int(entry["task_id"])]
        if len(candidates) < 2:
            raise ValueError("online action ranking requires two train episodes per task")
        current = int(entry["episode_id"])
        others = [item for item in candidates if int(item["episode_id"]) != current]
        seed = self._file_sha_seed(self.sampling_seed, self.epoch, current, slot, "episode")
        return others[seed % len(others)]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        entry, slot = self._rows[index]
        start = self._select_start(entry, slot)
        crop = self._crop(entry, start)
        donor_entry = self._donor_entry(entry, slot)
        donor = self._crop(
            donor_entry, self._select_start(donor_entry, slot, donor=True)
        )
        donor_actions = np.asarray(donor["actions"])
        own_actions = np.asarray(crop["actions"])
        rank_mask = (
            np.asarray(crop["world_target_valid_mask"], dtype=bool)
            & np.asarray(donor["world_target_valid_mask"], dtype=bool)
            & np.any(donor_actions != own_actions, axis=(1, 2))
        )
        task_id = int(entry["task_id"])
        item = {
            key: crop[key]
            for key in (
                "actions", "previous_action", "proprio", "action_valid_mask",
                "recovery_mask", "decision_recovery", "door_metric_state",
                "door_metric_state_valid", "world_target_valid_mask", "first_success",
            )
        }
        item.update({
            "language_hidden": self._reference_hidden[task_id],
            "language_mask": self._reference_mask[task_id],
            "instruction_id": torch.tensor(task_id, dtype=torch.long),
            "episode_id": torch.tensor(int(entry["episode_id"]), dtype=torch.long),
            "pair_id": torch.tensor(index, dtype=torch.long),
            "world_rank_shuffle_action": donor_actions,
            "world_rank_shuffle_mask": rank_mask,
            "crop_start": torch.tensor(start, dtype=torch.long),
        })
        task = str(entry["task"])
        episode_index = int(entry["episode_index"])
        from scripts.build_longtraj_features import clip_frame_indices

        frame_rows = [
            [int(frame) for frame in clip_frame_indices(int(decision))]
            for decision in crop["decisions"]
        ]
        target_rows = (
            [[int(endpoint)] for endpoint in crop["endpoints"]]
            if self.include_world_target_frames else []
        )
        flat_current = [frame for row in frame_rows for frame in row]
        flat_targets = [frame for row in target_rows for frame in row]
        decoded = self._decode_episode_frames(
            task, episode_index, flat_current + flat_targets
        )
        current = np.stack(decoded[:len(flat_current)])
        item["frames"] = current.reshape(
            len(frame_rows), len(frame_rows[0]), *current.shape[1:]
        )
        if self.include_world_target_frames:
            targets = np.stack(decoded[len(flat_current):])
            item["world_target_frames"] = targets.reshape(
                len(target_rows), 1, *targets.shape[1:]
            )
        return item


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
