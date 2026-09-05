"""Prepare MetaWorld MT50 (lerobot format) as frozen features for the VA pipeline.

49 tasks x 50 demos, 80 FPS, 480x480 PNG bytes inside parquet.  Task language
is the English task description (SmolVLA protocol); the dataset carries
next.success flags so per-sample success is available for diagnostics.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from prepare_pnpw_features import (
    QwenTextBackbone,
    VJEPA21Backbone,
    clip_frame_indices,
    robust_normalize,
)

# 80 FPS: 4 frames with stride 2 covers 0.05s (LIBERO 30FPS stride-2 covers 0.06s).
VISION_WINDOW = 4
VISION_STRIDE = 2
CONTROL_STRIDE = 6  # 80 FPS decision every 6 frames = 13.3 Hz cadence
SEQUENCE_LENGTH = 4
ACTION_HORIZON = 8


def scan_episode_success(root: Path | str, episodes: list[dict]) -> set[int]:
    """扫描 raw parquet 的 ``next.success`` 列，返回"至少成功过一次"的
    episode 的 ``dataset_from_index`` 集合（成功过滤用）。

    只做列投影读取（布尔列），chunk 文件不读图像，秒级完成。
    """
    files = sorted(glob.glob(str(Path(root) / "data/chunk-000/*.parquet")))
    # episode 起始行 → 排序数组，供二分定位行所属 episode
    starts = sorted(ep["dataset_from_index"] for ep in episodes)
    intervals = []
    for ep in episodes:
        intervals.append((ep["dataset_from_index"], ep["dataset_from_index"] + int(ep["length"])))
    ok: set[int] = set()
    for path in files:
        table = pq.read_table(path, columns=["index", "next.success"])
        index_col = table.column("index").to_pylist()
        success_col = table.column("next.success").to_pylist()
        pos = 0
        for row, flag in zip(index_col, success_col):
            if not flag:
                continue
            # 二分找包含 row 的 episode 区间
            i = bisect.bisect_right(starts, row) - 1
            if i >= 0:
                ep_start, ep_end = intervals[i]
                if ep_start <= row < ep_end:
                    ok.add(ep_start)
            pos += 1
    return ok


def build_phase_starts(
    length: int,
    required_span: int,
    n_windows: int,
    *,
    seed: int = 0,
) -> list[int]:
    """相位完整采样：按进度均匀分 n_windows 个 bin，起点取 bin 内确定性
    随机偏移（同 seed 可复现），并强制最后一个窗口覆盖 episode 末段
    （闭环失败带往往在轨迹后段）。

    - 与旧均匀协议（range(...)[:SPE]）的区别：① 起点在 bin 内随机而非
      固定等距；② 末窗口钉到 last_start。
    - n_windows=1 时返回 [last_start]（相位模式语义 = 覆盖末段）。
    """
    last_start = length - 1 - required_span
    if last_start < 0:
        return []
    if n_windows <= 1:
        # 相位模式语义 = 覆盖末段：n=1 也钉到 last_start（而非起点 0）。
        return [last_start]
    rng = random.Random(seed)
    starts = []
    for index in range(n_windows):
        lo = index * (last_start + 1) // n_windows
        hi = (index + 1) * (last_start + 1) // n_windows
        if hi - lo > 1:
            # bin 内确定性均匀随机偏移（同 seed 可复现）。
            offset = rng.randrange(0, hi - lo)
            start = min(lo + offset, last_start)
        else:
            start = lo
        starts.append(start)
    starts[-1] = last_start  # 强制覆盖末段（闭环失败带）
    return sorted(set(starts))


def decode_bytes(row: dict, image_size: int = 384) -> np.ndarray:
    raw = row.get("bytes")
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"missing image bytes at path={row.get('path')}")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    return np.asarray(image)


def preprocess_batch(clips: list[list[np.ndarray]], image_size: int) -> Tensor:
    """uint8 clips -> ImageNet-normalized [B,W,3,S,S]."""
    frames = [frame for clip in clips for frame in clip]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
    batch, channels, height, width = video.shape
    if height < width:
        resized_height = image_size
        resized_width = round(width * image_size / height)
    else:
        resized_width = image_size
        resized_height = round(height * image_size / width)
    flat = F.interpolate(
        video,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - image_size) // 2
    left = (resized_width - image_size) // 2
    flat = flat[:, :, top : top + image_size, left : left + image_size]
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    flat = (flat - mean) / std
    return flat.reshape(len(clips), VISION_WINDOW, channels, image_size, image_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MetaWorld MT50 frozen features")
    parser.add_argument("--dataset", type=Path, default=Path(
        "/home/ryan/Documents/robot/benchmark_data/raw/metaworld/lerobot_metaworld_mt50"
    ))
    parser.add_argument("--output", type=Path, default=Path("data/metaworld_features.pt"))
    parser.add_argument("--max-tasks", type=int, default=49)
    parser.add_argument("--task-start", type=int, default=0, help="first task index (for batch extraction)")
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Stage B：加载 checkpoint 里的 vjepa_state_dict（微调后的 V-JEPA）"
        "再提取特征——开环/语言消融评估必须与训练时同一 backbone，"
        "否则预计算特征与微调权重不匹配（镜像 eval_metaworld.py 的加载模式）",
    )
    parser.add_argument("--sequences-per-episode", type=int, default=1)
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=0,
        help="相位完整采样窗口数（0=关闭，用 --sequences-per-episode 均匀采样）："
        "6-8 时按进度分箱取窗口并强制覆盖 episode 末段（闭环失败带）。"
        "开启后覆盖 --sequences-per-episode。",
    )
    parser.add_argument(
        "--phase-seed",
        type=int,
        default=0,
        help="相位采样起点扰动种子（同参数同 seed → 同计划，可复现）",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="只保留至少成功过一次的 episode（按 raw 数据 next.success 列过滤）"
        "——全轨迹/相位采样必须开，否则失败 episode 的中后段错误示范会被学进去",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="全帧监督：窗口起点每 control-stride 帧滑动一个（S=C），保证每个"
        "决策点都至少出现在一个训练窗口里（π0.5 式全轨迹密集监督）。"
        "优先于 --phase-bins/--sequences-per-episode。",
    )
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="只产出骨架 payload（跳过 V-JEPA 特征提取，vision_tokens 用零占位）"
        "——配合 --live-vjepa 训练用：live 路径在线编码，离线特征会被丢弃，"
        "省 ~2-3h GPU 提取时间。",
    )
    parser.add_argument(
        "--control-stride",
        type=int,
        default=CONTROL_STRIDE,
        help="决策点间隔（80 FPS 帧）：6=13.3Hz（v5 默认），2=40Hz，1=80Hz",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--qwen-keep-layers",
        type=int,
        default=0,
        help="physically keep only the first N Qwen text layers; 0 keeps all 24",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="also write e2e video data (video_frames.npy uint8 memmap + meta.pt) "
        "to this directory (2026-08-06 user: V-JEPA full + Qwen top-layer e2e)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    cs = args.control_stride
    info = json.load(open(root / "meta/info.json"))
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    task_texts = [
        t["__index_level_0__"]
        for t in tasks[args.task_start : args.task_start + args.max_tasks]
    ]
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()

    # Group episodes by task text; sample plans.
    by_task: dict[str, list[dict]] = {}
    for episode in episodes:
        raw_task = episode.get("tasks") or episode.get("task") or ""
        if isinstance(raw_task, list):
            raw_task = raw_task[0] if raw_task else ""
        by_task.setdefault(str(raw_task).strip(), []).append(episode)

    if args.success_only:
        ok_eps = scan_episode_success(root, episodes)
        before = sum(len(v) for v in by_task.values())
        by_task = {
            task: [ep for ep in eps if ep["dataset_from_index"] in ok_eps]
            for task, eps in by_task.items()
        }
        after = sum(len(v) for v in by_task.values())
        print(f"success filter: {after}/{before} episodes kept")

    plans = []  # (episode, task_text, start)
    for task_text in task_texts:
        for episode in by_task.get(task_text, [])[:]:
            length = int(episode["length"])
            required_span = (SEQUENCE_LENGTH - 1) * cs + (ACTION_HORIZON - 1)
            last_start = length - 1 - required_span
            if last_start < 0:
                continue
            if args.sliding_window:
                # 全帧监督：起点每 cs 帧滑动一个（S=C）→ 每个决策点都被覆盖
                starts = list(range(0, last_start + 1, cs))
            elif args.phase_bins > 0:
                # 相位完整采样：进度分箱 + 强制覆盖末段（闭环失败带）。
                starts = build_phase_starts(
                    length, required_span, args.phase_bins, seed=args.phase_seed
                )
            else:
                stride = max(1, last_start // max(args.sequences_per_episode, 1))
                starts = list(range(0, last_start + 1, stride))[: args.sequences_per_episode]
            for start in starts:
                plans.append((episode, task_text, start))
    if not plans:
        raise ValueError("no plans produced")
    print(f"tasks={len(task_texts)} samples={len(plans)}")
    if args.dry_run:
        return

    data_files = sorted(glob.glob(str(root / "data/chunk-000/*.parquet")))
    file_meta = []
    for path in data_files:
        table = pq.read_table(path, columns=["index"])
        meta = table.column("index").to_pylist()
        file_meta.append((path, meta))

    def global_row(episode: dict, local_frame: int) -> int:
        return int(episode["dataset_from_index"]) + local_frame

    needed_rows: dict[str, set[int]] = {}
    for episode, _task, start in plans:
        for offset in range(SEQUENCE_LENGTH):
            decision = start + offset * cs
            indices = clip_frame_indices(
                decision, video_start_frame=0, window=VISION_WINDOW, stride=VISION_STRIDE
            )
            for frame in indices:
                row = global_row(episode, max(0, frame))
                for path, meta in file_meta:
                    if meta[0] <= row <= meta[-1]:
                        needed_rows.setdefault(path, set()).add(row)
                        break
            for step in range(ACTION_HORIZON):
                row = global_row(episode, decision + step)
                for path, meta in file_meta:
                    if meta[0] <= row <= meta[-1]:
                        needed_rows.setdefault(path, set()).add(row)
                        break
            if decision > 0:
                # P2（Codex 审查）：decision==0 时 global_row(ep, -1) 会读到
                # 上一 episode 的末行并污染归一化分位；prev 值由组装端置零
                # （P0-A 修复），此处分位集合不需要该行。
                previous_row = global_row(episode, decision - 1)
                for path, meta in file_meta:
                    if meta[0] <= previous_row <= meta[-1]:
                        needed_rows.setdefault(path, set()).add(previous_row)
                        break

    from collections import OrderedDict

    MAX_CACHE_FRAMES = 26000  # 有界帧缓存：~26k 帧 × 384×384×3 ≈ 11GB，防 v2 全量数据 OOM
    frame_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
    for path, rows in ([] if args.skeleton else needed_rows.items()):
        table = pq.read_table(path, columns=["index", "observation.image"])
        index_col = table.column("index").to_pylist()
        arr = table.column("observation.image").combine_chunks().to_pylist()
        position = {g: local for local, g in enumerate(index_col)}
        for row in rows:
            frame_cache[row] = decode_bytes(arr[position[row]], args.image_size)
            if len(frame_cache) > MAX_CACHE_FRAMES:
                frame_cache.popitem(last=False)
    print(f"frames decoded: {len(frame_cache)} (capped {MAX_CACHE_FRAMES})")

    # Language encoding (one pass per task).
    text_backbone = QwenTextBackbone.from_pretrained(
        device=args.device,
        dtype=args.model_dtype,
        keep_layers=args.qwen_keep_layers or None,
        local_files_only=True,
    )
    if args.qwen_keep_layers == 15:
        fusion_layers = list(range(10, 15))
        language_hierarchy, language_mask = text_backbone.encode(
            task_texts, output_layers=fusion_layers
        )
        language_hidden = text_backbone.mean_output_layers(
            language_hierarchy, fusion_layers
        )
    else:
        fusion_layers = None
        language_hidden, language_mask = text_backbone.encode(task_texts)
    qwen_keep_layers = text_backbone.keep_layers
    language_hidden = language_hidden.to(device="cpu", dtype=torch.float16)
    language_mask = language_mask.cpu()
    del text_backbone
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Vision features.
    vision_backbone = (
        None
        if args.skeleton
        else VJEPA21Backbone.from_pretrained(
            device=args.device,
            dtype=args.model_dtype,
            max_tokens=64,
            local_files_only=True,
        )
    )
    if args.backbone_checkpoint is not None and not args.skeleton:
        ckpt = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=True)
        if "vjepa_state_dict" not in ckpt:
            raise ValueError(
                f"{args.backbone_checkpoint} 没有 vjepa_state_dict 键"
                f"（keys={list(ckpt)[:8]}）"
            )
        vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
        print(f"backbone: loaded vjepa_state_dict from {args.backbone_checkpoint}")
        del ckpt
    flat_features: dict[int, Tensor] = {}
    batch_keys: list[int] = []
    batch_clips: list[list[np.ndarray]] = []

    def encode_pending() -> None:
        if not batch_keys:
            return
        inputs = preprocess_batch(batch_clips, args.image_size).to(args.device)
        with torch.inference_mode():
            flat, _ = vision_backbone.forward_variants(inputs)
        flat = flat.to(device="cpu", dtype=torch.float16)
        for key, token in zip(batch_keys, flat, strict=True):
            flat_features[key] = token.contiguous()
        batch_keys.clear()
        batch_clips.clear()

    seen = set()
    decode_tables: dict[str, tuple] = {}  # path -> (position, arr) 惰性重解码

    def get_frame(row: int) -> np.ndarray:
        """LRU 缓存读帧；被弹出时从 parquet 惰性重解码（按文件缓存解码表）。"""
        hit = frame_cache.get(row)
        if hit is not None:
            frame_cache.move_to_end(row)
            return hit
        for path, meta in file_meta:
            if meta[0] <= row <= meta[-1]:
                if path not in decode_tables:
                    table = pq.read_table(path, columns=["index", "observation.image"])
                    pos = {g: local for local, g in enumerate(table.column("index").to_pylist())}
                    arr = table.column("observation.image").combine_chunks().to_pylist()
                    decode_tables[path] = (pos, arr)
                    # P0（Codex 审查）：decode_tables 无界缓存会吃满内存
                    # （458 个 parquet 图像列 ≈ 39 GiB，机器 31 GiB 必 OOM）。
                    # LRU 上限 24 张（≈2.1 GiB）：逐渐用逐渐抛弃，总内存
                    # ~21 GiB，机器余量充足，不影响同机 GPU 训练。
                    if len(decode_tables) > 24:
                        decode_tables.pop(next(iter(decode_tables)))
                pos, arr = decode_tables[path]
                frame = decode_bytes(arr[pos[row]], args.image_size)
                frame_cache[row] = frame
                if len(frame_cache) > MAX_CACHE_FRAMES:
                    frame_cache.popitem(last=False)
                return frame
        raise KeyError(row)

    for episode, _task, start in ([] if args.skeleton else plans):
        for offset in range(SEQUENCE_LENGTH):
            decision = start + offset * cs
            key = global_row(episode, decision)
            if key in seen:
                continue
            seen.add(key)
            indices = clip_frame_indices(
                decision, video_start_frame=0, window=VISION_WINDOW, stride=VISION_STRIDE
            )
            batch_keys.append(key)
            batch_clips.append([get_frame(global_row(episode, max(0, idx))) for idx in indices])
            if len(batch_keys) == args.batch_size:
                encode_pending()
    encode_pending()
    del decode_tables
    del vision_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"vision features: {len(flat_features)}")

    # Actions/states with cross-task normalization.
    stat_actions, stat_states = [], []
    for path, _rows in needed_rows.items():
        table = pq.read_table(path, columns=["action", "observation.state"])
        stat_actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        stat_states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
    raw_actions = np.concatenate(stat_actions)
    raw_states = np.concatenate(stat_states)
    action_low, action_high = np.quantile(raw_actions, (0.01, 0.99), axis=0)
    state_low, state_high = np.quantile(raw_states, (0.01, 0.99), axis=0)

    norm_action: dict[int, np.ndarray] = {}
    norm_state: dict[int, np.ndarray] = {}
    for path, rows in needed_rows.items():
        table = pq.read_table(path, columns=["index", "action", "observation.state"])
        index_col = table.column("index").to_pylist()
        acts = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        position = {g: local for local, g in enumerate(index_col)}
        for row in rows:
            local = position[row]
            norm_action[row] = robust_normalize(acts[local][None], action_low, action_high)[0]
            norm_state[row] = robust_normalize(states[local][None], state_low, state_high)[0]

    task_to_id = {task: index for index, task in enumerate(task_texts)}
    # skeleton 模式下 flat_features 为空：零占位（live 训练加载后即 pop，内容无意义）。
    ZERO_TOK = torch.zeros(64, 768, dtype=torch.float16)
    vision_sequences = []
    proprio_sequences = []
    previous_sequences = []
    action_sequences = []
    language_sequences = []
    mask_sequences = []
    instruction_ids = []
    for episode, task, start in plans:
        task_id = task_to_id[task]
        vision_sequences.append(
            torch.stack(
                [flat_features.get(global_row(episode, start + offset * cs), ZERO_TOK) for offset in range(SEQUENCE_LENGTH)]
            )
        )
        proprio_sequences.append(
            torch.from_numpy(
                np.stack(
                    [norm_state[global_row(episode, start + offset * cs)] for offset in range(SEQUENCE_LENGTH)]
                )
            )
        )
        previous_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        # 修复 P0-A（2026-08-05 审查）：episode 首决策 prev 用 0（归一化中点，
                        # 与闭环评估 last_norm 初值一致），避免跨 episode 泄漏/自泄漏
                        np.zeros_like(norm_action[global_row(episode, start)])
                        if start + offset * cs == 0
                        else norm_action[global_row(episode, start + offset * cs) - 1]
                        for offset in range(SEQUENCE_LENGTH)
                    ]
                )
            )
        )
        action_sequences.append(
            torch.from_numpy(
                np.stack(
                    [
                        np.stack(
                            [
                                norm_action[global_row(episode, start + offset * cs) + step]
                                for step in range(ACTION_HORIZON)
                            ]
                        )
                        for offset in range(SEQUENCE_LENGTH)
                    ]
                )
            )
        )
        language_sequences.append(language_hidden[task_id])
        mask_sequences.append(language_mask[task_id])
        instruction_ids.append(task_id)

    payload = {
        "vision_tokens": torch.stack(vision_sequences),
        "language_hidden": torch.stack(language_sequences),
        "language_mask": torch.stack(mask_sequences),
        "proprio": torch.stack(proprio_sequences),
        "previous_action": torch.stack(previous_sequences),
        "actions": torch.stack(action_sequences),
        "pair_id": torch.arange(len(plans), dtype=torch.long),
        "instruction_id": torch.tensor(instruction_ids, dtype=torch.long),
        "episode_id": torch.arange(len(plans), dtype=torch.long),
        "normalization": {
            "action_q01": torch.from_numpy(action_low.astype(np.float32)),
            "action_q99": torch.from_numpy(action_high.astype(np.float32)),
            "state_q01": torch.from_numpy(state_low.astype(np.float32)),
            "state_q99": torch.from_numpy(state_high.astype(np.float32)),
        },
        "metadata": {
            "contract": "language_conditioned_mt50",
            "tasks": task_texts,
            "fps": int(info["fps"]),
            "control_stride": cs,
            "action_horizon": ACTION_HORIZON,
            "qwen_keep_layers": qwen_keep_layers,
            "qwen_output_layer": qwen_keep_layers - 1,
            "qwen_fusion_layers": fusion_layers,
            "qwen_layer_reduce": "mean_then_final_norm" if fusion_layers else None,
            # 采样协议自描述：live 训练必须用完全相同的参数重建计划，
            # 否则行数相同但起点不同的静默错配会毒化监督信号（Grok P0）。
            "sampling": {
                "mode": (
                    "sliding"
                    if args.sliding_window
                    else "phase" if args.phase_bins > 0 else "uniform"
                ),
                "phase_bins": args.phase_bins,
                "phase_seed": args.phase_seed,
                "sequences_per_episode": args.sequences_per_episode,
                "success_only": args.success_only,
                "sliding": args.sliding_window,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gib = args.output.stat().st_size / (1024**3)
    print(f"saved={args.output.resolve()} size={size_gib:.2f}GiB shape={payload['vision_tokens'].shape}")


if __name__ == "__main__":
    main()
