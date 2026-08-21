"""PULSE-VA Stage B：在线 V-JEPA 编码训练路径（--live-vjepa）。

预计算特征（v5/ST288）无法对 V-JEPA 反向传播；本模块提供：
- ``build_mw_plans``：按 v5 行重建 (episode, start) 采样计划（与
  prepare_mw_local_features.py 同一逻辑，逐行对齐 v5 索引）；
- ``LiveVJEPADataset``：FeatureDataset 的帧变体——除 vision_tokens 外
  返回全部既有键，外加原始解码帧 [T, W, H, W, 3] uint8；
- ``encode_live_frames``：帧 batch → V-JEPA（可训练）→ [B, T, 288, D]。

使用方式（train.py）：``--data data/metaworld_features_v5.pt --live-vjepa
--live-root <parquet 根> --vision-unfreeze-all --lr-vision 3e-6``。
Stage B 限定 --single-task（配对帧级契约留待数据侧；动作头 direct/flow
均可，full 架构用 flow + langslot 槽）。
"""
from __future__ import annotations

import collections
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

try:
    from prepare_metaworld import (
        VISION_STRIDE,
        VISION_WINDOW,
        build_phase_starts,
        decode_bytes,
        glob,
        preprocess_batch,
        pq,
        scan_episode_success,
    )
    from prepare_pnpw_features import clip_frame_indices
except Exception:  # pragma: no cover - 仅 IDE/文档场景
    VISION_STRIDE = 2
    VISION_WINDOW = 4
    build_phase_starts = None  # type: ignore
    decode_bytes = None  # type: ignore
    glob = None  # type: ignore
    preprocess_batch = None  # type: ignore
    pq = None  # type: ignore
    scan_episode_success = None  # type: ignore
    clip_frame_indices = None  # type: ignore

SEQUENCE_LENGTH = 4
CONTROL_STRIDE = 6
ACTION_HORIZON = 8
SPE = 4
SLOT_GRID = 12  # per time-slice pooled grid -> 12x12
N_TOKENS = 2 * SLOT_GRID * SLOT_GRID  # 288 (spatiotemporal 双时间片)
DENSE_GRID = 24  # Step 0 dense readout：384/16 patch 网格（不池化）
N_DENSE_TOKENS = 2 * DENSE_GRID * DENSE_GRID  # 1152（2 时间片 × 24×24 patch）
IMAGE_SIZE = 384
MAX_CACHE_FRAMES = 512  # Codex P0-4：512 帧 ≈ 0.22 GiB（有界，防 oomd）
MAX_PARQUET_TABLES = 2  # 解码表 LRU：最多持有 2 个 parquet 全列（防 43 GiB 全缓存）


def build_mw_plans(
    payload: dict,
    root: Path | str,
    *,
    control_stride: int = CONTROL_STRIDE,
    spe: int = SPE,
    phase_bins: int = 0,
    phase_seed: int = 0,
    success_only: bool = False,
    sliding: bool = False,
) -> list[tuple[dict, int]]:
    """重建 (episode, start) 计划并逐行对齐 payload 行索引（数量必须相等）。

    root 为 MetaWorld LeRobot 数据集的根目录
    （.../lerobot_metaworld_mt50），含 data/chunk-000/*.parquet 与
    meta/episodes/chunk-000/file-000.parquet。
    control_stride 为决策点间隔（80 FPS 帧）：6=13.3Hz，2=40Hz，1=80Hz。
    phase_bins>0 时改用相位完整采样（与 prepare_metaworld 同一实现），
    spe 仅用于 phase_bins=0 的旧协议。

    对齐硬化（Grok P0）：若 payload metadata 含 ``sampling`` 字段（新
    协议自描述），此处传入参数必须与之一致——否则行数可能碰巧相等而
    起点不同（静默错配帧与 action，毒化监督信号）。
    """
    # 对齐硬化（Grok P0 / Codex P0）：payload metadata 必须与调用参数一致。
    meta = payload.get("metadata", {})
    meta_cs = meta.get("control_stride")
    if meta_cs is not None and meta_cs != control_stride:
        raise ValueError(
            f"control_stride 与 payload metadata 不一致（{control_stride} vs "
            f"{meta_cs}）：live 训练必须与数据提取使用相同的 --control-stride"
        )
    sampling = meta.get("sampling")
    if sampling is not None:
        # 只比较当前 mode 实际生效的字段（Codex P1-1）：sliding/phase 下 SPE
        # 无关，比较它会造成语义一致的正常命令被误拒。
        if sliding:
            keys = ("success_only",)
        elif phase_bins > 0:
            keys = ("phase_bins", "phase_seed", "success_only")
        else:
            keys = ("sequences_per_episode", "success_only")
        expected = {"success_only": success_only, "sequences_per_episode": spe,
                    "phase_bins": phase_bins, "phase_seed": phase_seed}
        expected_mode = "sliding" if sliding else ("phase" if phase_bins > 0 else "uniform")
        mismatch = {}
        if sampling.get("mode") != expected_mode:
            mismatch["mode"] = (expected_mode, sampling.get("mode"))
        for k in keys:
            if expected[k] != sampling.get(k):
                mismatch[k] = (expected[k], sampling.get(k))
        if mismatch:
            raise ValueError(
                f"采样参数与 payload metadata 不一致（{mismatch}）：live 训练必须"
                f"与数据提取使用完全相同的采样参数"
            )
    if pq is None or glob is None:
        raise RuntimeError("prepare_metaworld 导入失败")
    root = Path(root)
    n, T = payload["vision_tokens"].shape[0], payload["vision_tokens"].shape[1]
    if T != SEQUENCE_LENGTH:
        raise ValueError(f"live-vjepa requires sequence length {SEQUENCE_LENGTH}, got {T}")

    data_files = sorted(glob.glob(str(root / "data/chunk-000/*.parquet")))
    if not data_files:
        raise FileNotFoundError(f"no parquet chunks under {root / 'data/chunk-000'}")
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    if success_only:
        ok_eps = scan_episode_success(root, episodes)
        episodes = [ep for ep in episodes if ep["dataset_from_index"] in ok_eps]
    by_task: dict[str, list] = collections.defaultdict(list)
    for ep in episodes:
        t = ep.get("tasks") or ep.get("task") or ""
        if isinstance(t, list):
            t = t[0] if t else ""
        by_task[str(t).strip()].append(ep)

    task_texts = payload["metadata"]["tasks"]
    plans: list[tuple[dict, int]] = []
    for task_text in task_texts:
        for ep in by_task.get(task_text, []):
            length = int(ep["length"])
            required_span = (SEQUENCE_LENGTH - 1) * control_stride + (ACTION_HORIZON - 1)
            last_start = length - 1 - required_span
            if last_start < 0:
                continue
            if sliding:
                starts = list(range(0, last_start + 1, control_stride))
            elif phase_bins > 0:
                starts = build_phase_starts(
                    length, required_span, phase_bins, seed=phase_seed
                )
            else:
                stride = max(1, last_start // max(spe, 1))
                starts = list(range(0, last_start + 1, stride))[:spe]
            for start in starts:
                plans.append((ep, start))
    if len(plans) != n:
        raise ValueError(f"live plans {len(plans)} != v5 rows {n}（数据版本不一致）")
    return plans


class FrameDecoder:
    """parquet 帧解码器：按全局行号解码 384×384 RGB（LRU 缓存）。"""

    def __init__(self, root: Path | str) -> None:
        root = Path(root)
        self.root = root
        self.file_meta: list[tuple[str, list]] = []
        for path in sorted(glob.glob(str(root / "data/chunk-000/*.parquet"))):
            table = pq.read_table(path, columns=["index"])
            self.file_meta.append((str(path), table.column("index").to_pylist()))
        # Codex P0-4：解码表 LRU（全列图像 43 GiB 总量，不能全量驻留）。
        self.decode_tables: "collections.OrderedDict[str, tuple[dict, list]]" = (
            collections.OrderedDict()
        )
        self.cache: "collections.OrderedDict[int, np.ndarray]" = collections.OrderedDict()

    def __call__(self, row: int) -> np.ndarray:
        hit = self.cache.get(row)
        if hit is not None:
            self.cache.move_to_end(row)
            return hit
        for path, meta in self.file_meta:
            if meta[0] <= row <= meta[-1]:
                if path not in self.decode_tables:
                    table = pq.read_table(path, columns=["index", "observation.image"])
                    pos = {g: local for local, g in enumerate(table.column("index").to_pylist())}
                    arr = table.column("observation.image").combine_chunks().to_pylist()
                    self.decode_tables[path] = (pos, arr)
                    self.decode_tables.move_to_end(path)
                    while len(self.decode_tables) > MAX_PARQUET_TABLES:
                        self.decode_tables.popitem(last=False)
                pos, arr = self.decode_tables[path]
                frame = decode_bytes(arr[pos[row]], IMAGE_SIZE)
                self.cache[row] = frame
                self.cache.move_to_end(row)
                while len(self.cache) > MAX_CACHE_FRAMES:
                    self.cache.popitem(last=False)
                return frame
        raise KeyError(f"row {row} 不在任何 parquet chunk 的 index 范围内")


def global_row(episode: dict, local_frame: int) -> int:
    return int(episode["dataset_from_index"]) + local_frame


class LiveVJEPADataset:
    """FeatureDataset 契约的帧变体：返回全部既有键 + ``frames``。

    - ``vision_tokens`` 键被移除（省 ~7.8 GiB RAM，对抗 systemd-oomd）；
    - ``frames``: [T, VISION_WINDOW, IMAGE_SIZE, IMAGE_SIZE, 3] uint8；
    - ``coords`` 常量 [288, 3]（与 ST288 相同的 slot 网格坐标）；
      ``dense_readout=True``（Step 0）时为 [1152, 3] 全量 patch 网格坐标。

    与 v5 的差异仅在视觉侧；language/proprio/actions/pair 语义不变。
    """

    def __init__(
        self,
        path: Path | str,
        root: Path | str,
        *,
        min_sequence_length: int = 4,
        vision_pooling: str = "spatiotemporal",
        control_stride: int = CONTROL_STRIDE,
        spe: int = SPE,
        phase_bins: int = 0,
        phase_seed: int = 0,
        success_only: bool = False,
        sliding: bool = False,
        frame_aug: bool = False,
        frame_aug_geometric: bool = True,
        dense_readout: bool = False,
    ) -> None:
        from train import FeatureDataset  # 延迟导入避免循环依赖

        self._inner = FeatureDataset(
            path,
            require_pairs=False,  # Stage B 限定 single-task
            min_sequence_length=min_sequence_length,
            vision_key=(
                "vision_tokens_spatial" if vision_pooling == "spatial" else "vision_tokens"
            ),
        )
        self.length = self._inner.length
        self.payload = self._inner.payload
        self.control_stride = control_stride
        self.frame_aug = frame_aug
        self.frame_aug_geometric = frame_aug_geometric
        # Codex P0-1：build_mw_plans 需要 vision_tokens 的形状 → 先建 plans 再释放。
        self.plans = build_mw_plans(
            self.payload,
            root,
            control_stride=control_stride,
            spe=spe,
            phase_bins=phase_bins,
            phase_seed=phase_seed,
            success_only=success_only,
            sliding=sliding,
        )
        # 释放 7.8 GiB 预计算视觉特征（live 模式下无用；__getitem__ 不再读取）。
        self.payload.pop("vision_tokens", None)
        self.payload.pop("vision_tokens_spatial", None)
        self.decoder = FrameDecoder(root)
        # Step 0 dense readout：coords 随读出模式切换（1152 全量 patch vs 288 池化槽）。
        self.coords = _dense_coords() if dense_readout else _slot_coords()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        item = {
            key: self.payload[key][index]
            for key in self._inner.REQUIRED
            if key != "vision_tokens"
        }
        if "language_mask" in self.payload:
            item["language_mask"] = self.payload["language_mask"][index]
        ep, start = self.plans[index]
        frames = []
        for off in range(SEQUENCE_LENGTH):
            decision = start + off * self.control_stride
            indices = clip_frame_indices(
                decision, video_start_frame=0, window=VISION_WINDOW, stride=VISION_STRIDE
            )
            frames.append([self.decoder(global_row(ep, idx)) for idx in indices])
        item["frames"] = np.stack(frames)  # [T, W, 384, 384, 3] uint8
        if self.frame_aug:
            item["frames"] = augment_frames(item["frames"], geometric=self.frame_aug_geometric)
        item["coords"] = self.coords
        return item


def _slot_coords(grid: int = SLOT_GRID) -> np.ndarray:
    """[2*grid*grid, 3]：与 ST288 提取（prepare_mw_local_features.build_coords）逐位一致
    ——(t∈{-1,1}, y, x)，t 外层循环 → y → x（Codex P0-5：曾为 (x,y,t) 顺序错位）。

    ``grid`` 默认 12（288 token，既有行为逐位不变）；``grid=24`` 生成 dense 读出
    （--dense-readout）的 [1152, 3] 全量 patch 网格坐标。
    """
    half = (grid - 1) / 2
    coords = []
    for t in range(2):
        for y in range(grid):
            for x in range(grid):
                coords.append((t * 2.0 - 1.0, (y - half) / half, (x - half) / half))
    return np.asarray(coords, dtype=np.float32)


def _dense_coords() -> np.ndarray:
    """[1152, 3]：Step 0 dense readout 的 2×24×24 patch 网格坐标（不池化）。"""
    return _slot_coords(grid=DENSE_GRID)


def augment_frames(frames: np.ndarray, geometric: bool = True) -> np.ndarray:
    """π0.5 式帧增强：RandomCrop(0.95) → Resize(384) → Rotate(±5°) → ColorJitter。

    输入 [T, W, 384, 384, 3] uint8，输出同形状；每 epoch 重新随机。
    每个 V-JEPA 时间窗（一个 t，W=4 帧）只采样一组增强参数并复用到全部
    4 帧（Codex P1-2 修复）：逐帧独立重采样会产生假相机运动/颜色闪烁，
    破坏 V-JEPA 时间窗（2 时间片）的一致性，episode 开头的 causal-clamp
    帧甚至会被增强成 4 个不同视图。
    注意：几何增强（crop/rotate）会扰动 slot 坐标网格（coords [288,3]）与
    场景点的对应关系——local_slots 开启时该通道变噪声，需自行权衡。
    ``geometric=False`` 时跳过 crop/rotate，只保留光度增强（ColorJitter）：
    精细定位任务（抓取/插入）下几何扰动 ≈ ±10px ≈ ±1cm 定位噪声，
    且 slot 坐标未同步变换（2026-08-09 审计 R4，E1 修复）。
    """
    import torch
    import torchvision.transforms as T
    from torchvision.transforms import functional as F
    from PIL import Image

    crop = round(IMAGE_SIZE * 0.95)
    out = np.empty_like(frames)
    for t_idx in range(frames.shape[0]):
        # 每组增强参数采样一次（该 clip 的 4 帧共享）。
        dummy = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        top, left, h, w = (
            T.RandomCrop.get_params(dummy, (crop, crop)) if geometric else (0, 0, IMAGE_SIZE, IMAGE_SIZE)
        )
        angle = float(torch.empty(1).uniform_(-5.0, 5.0)) if geometric else 0.0
        b, c, s, _hue = T.ColorJitter.get_params(
            brightness=(0.7, 1.3), contrast=(0.6, 1.4), saturation=(0.5, 1.5), hue=None
        )[-4:]
        for w_idx in range(frames.shape[1]):
            img = Image.fromarray(frames[t_idx, w_idx])
            if geometric:
                img = F.crop(img, top, left, h, w)
                img = F.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = F.rotate(img, angle)
            img = F.adjust_brightness(img, b)
            img = F.adjust_contrast(img, c)
            img = F.adjust_saturation(img, s)
            out[t_idx, w_idx] = np.asarray(img, dtype=np.uint8)
    return out


def encode_live_frames(
    frames_batch: np.ndarray,
    vision_backbone,
    device: torch.device,
    *,
    dense: bool = False,
    out_layers: Sequence[int] | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """[B, T, W, 384, 384, 3] uint8 → [B, T, N, D]（backbone dtype）。

    V-JEPA 前向 +（解冻时）反向；``dense=False`` 输出与 ST288 同构
    （288 token，既有行为不变）；``dense=True``（Step 0 dense readout）
    跳过池化，输出全量 2×24×24 = 1152 patch token [B, T, N_DENSE_TOKENS, D]，
    供角色查询 cross-attention 直接读出（Q≈6 × N=1152；1152 不进 VA 自注意力，
    设计文档 §九）。

    ``out_layers``（Step 4）非 None 时改走 ``encode_multi``：返回
    ``list[Tensor]``（顺序与 out_layers 一致），每层按同一 dense/池化规则
    折叠成 [B, T, N, D]（H⁵/H¹¹ 多层输出，只作只读 evidence）；默认 None
    保持既有行为（返回单个 Tensor）。
    """
    if preprocess_batch is None:
        raise RuntimeError("prepare_metaworld 导入失败")
    B, T, W, H, Wc, _ = frames_batch.shape
    clips = frames_batch.reshape(B * T, W, H, Wc, 3)
    clips_list = [list(clip) for clip in clips]
    inputs = preprocess_batch(clips_list, IMAGE_SIZE).to(device)  # [B*T, W, 3, 384, 384]

    def fold(raw: torch.Tensor) -> torch.Tensor:
        """单层 token 折叠：[B*T, N, D] → [B, T, N', D]（dense 不池化 / 其余 ST288）。"""
        if dense:
            # Step 0 dense readout：不池化。raw 必须是全量 patch（4 帧窗口 →
            # 2 时间片 × 24×24 = 1152；与 _dense_coords() 的 t→y→x 顺序一致）。
            if raw.ndim != 3 or raw.shape[1] != N_DENSE_TOKENS:
                raise ValueError(
                    f"dense readout 需要 raw tokens [., {N_DENSE_TOKENS}, D]"
                    f"（2×24×24 patch，4 帧窗口 → 2 时间片），got {tuple(raw.shape)}"
                )
            return raw.reshape(B, T, N_DENSE_TOKENS, -1)
        st = vision_backbone._pool(raw, "spatiotemporal")  # [B*T, N_TOKENS, D]
        return st.reshape(B, T, N_TOKENS, -1)

    if out_layers is None:
        return fold(vision_backbone._encode(inputs))  # [B*T, t_grid*h*w, D] 扁平（t→h→w）
    # Step 4：多层输出透传（encode_multi）——每层按同一 dense/池化规则折叠。
    return [fold(raw) for raw in vision_backbone.encode_multi(inputs, out_layers=out_layers)]


def load_st288_memmap(npy_path: str | Path, metadata: dict) -> torch.Tensor:
    """加载 ST288 大数组（mmap 零拷贝）。

    scripts/extract_st288_finetuned.py 用 np.memmap(mode="w+") 写裸数据（无 npy header），
    np.load 会误判为 pickle 而拒绝；此处先检查 magic，带 header 走 np.load，
    裸数据则从 metadata + 文件大小推断 shape。
    """
    npy_path = str(npy_path)
    with open(npy_path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        arr = np.load(npy_path, mmap_mode="r")
    else:
        import os

        n = metadata["rows"]
        seq = 4  # 提取脚本 SEQUENCE_LENGTH 约定
        tok = metadata["tokens_per_decision"]
        d = os.path.getsize(npy_path) // 2 // (n * seq * tok)
        if os.path.getsize(npy_path) != 2 * n * seq * tok * d:
            raise ValueError(f"npy 大小与 metadata 推断不符: {npy_path}")
        arr = np.memmap(npy_path, dtype=np.float16, mode="r", shape=(n, seq, tok, d))
    return torch.from_numpy(arr)
