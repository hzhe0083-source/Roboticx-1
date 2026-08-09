#!/usr/bin/env python
"""长轨迹 → 预计算特征训练数据（E7 路径，2026-08-09）。

两阶段：
  Phase 1（轻量，无 GPU）：遍历 data/metaworld_longtraj_*.pt（JPEG 压缩帧），
    滑动窗口切片 → 动作/状态/prev（executed-clip + 全局 q01/q99 继承）+
    帧索引 (task, ep, start)。输出 data/metaworld_longtraj_windows.pt。
  Phase 2（GPU ~30-40 分钟）：按帧索引从 per-task 文件解压窗口帧 →
    冻结原始 V-JEPA 2.1 编码（spatiotemporal 288，与 E7 视觉路径一致）→
    memmap 到 /media/ryan/robot-data/longtraj_st288.npy + meta.pt。

训练命令（E7）：train.py --data data/metaworld_longtraj_windows.pt
  --local-slots-data /media/ryan/robot-data/longtraj_st288_meta.pt ...

用法：
  python scripts/build_longtraj_features.py [--device cuda]
"""
from __future__ import annotations

import argparse
import gc
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 80
CONTROL_STRIDE = 6
SEQUENCE_LENGTH = 4
ACTION_HORIZON = 8
VISION_WINDOW = 4
VISION_STRIDE = 2

REF = ROOT / "data" / "metaworld_fullframe_executed.pt"
ST_NPY_DIR = Path("/media/ryan/robot-data")


def win_out(horizon: int) -> Path:
    return ROOT / "data" / f"metaworld_longtraj_windows_h{horizon}.pt"


def st_paths(horizon: int) -> tuple[Path, Path]:
    return (ST_NPY_DIR / f"longtraj_st288_h{horizon}.npy",
            ST_NPY_DIR / f"longtraj_st288_h{horizon}_meta.pt")

# MT1 环境名 → lerobot 任务文本（与 REF.metadata.tasks 对齐，2026-08-09 全量核对）
ENV_TO_TASK = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "basketball-v3": "Dunk the basketball into the basket",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
    "box-close-v3": "Grasp the cover and close the box with it",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
    "button-press-v3": "Press a button",
    "button-press-wall-v3": "Bypass a wall and press a button",
    "coffee-button-v3": "Push a button on the coffee machine",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "coffee-push-v3": "Push a mug under a coffee machine",
    "dial-turn-v3": "Rotate a dial 180 degrees",
    "disassemble-v3": "Pick a nut out of a peg",
    "door-close-v3": "Close a door with a revolving joint",
    "door-lock-v3": "Lock the door by rotating the lock clockwise",
    "door-open-v3": "Open a door with a revolving joint",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise",
    "hand-insert-v3": "Insert the gripper into a hole",
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "faucet-open-v3": "Rotate the faucet counter-clockwise",
    "faucet-close-v3": "Rotate the faucet clockwise",
    "hammer-v3": "Hammer a screw on the wall",
    "handle-press-side-v3": "Press a handle down sideways",
    "handle-press-v3": "Press a handle down",
    "handle-pull-side-v3": "Pull a handle up sideways",
    "handle-pull-v3": "Pull a handle up",
    "lever-pull-v3": "Pull a lever down 90 degrees",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
    "pick-out-of-hole-v3": "Pick up a puck from a hole",
    "pick-place-v3": "Pick and place a puck to a goal",
    "plate-slide-v3": "Slide a plate into a cabinet",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways",
    "plate-slide-back-v3": "Get a plate from the cabinet",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways",
    "peg-insert-side-v3": "Insert a peg sideways",
    "peg-unplug-side-v3": "Unplug a peg sideways",
    "soccer-v3": "Kick a soccer into the goal",
    "stick-push-v3": "Grasp a stick and push a box using the stick",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick",
    "push-v3": "Push the puck to a goal",
    "push-wall-v3": "Bypass a wall and push a puck to a goal",
    "reach-v3": "Reach a goal position",
    "reach-wall-v3": "Bypass a wall and reach a goal",
    "shelf-place-v3": "Pick and place a puck onto a shelf",
    "sweep-into-v3": "Sweep a puck into a hole",
    "sweep-v3": "Sweep a puck off the table",
    "window-open-v3": "Push and open a window",
    "window-close-v3": "Push and close a window",
}


def clip_frame_indices(decision: int, video_start_frame: int = 0,
                       window: int = VISION_WINDOW, stride: int = VISION_STRIDE):
    """与 canonical（prepare_pnpw_features.clip_frame_indices / live_vjepa）
    完全一致的历史帧窗：决策点 d 用 [d-(window-1)*stride, ..., d-2*stride, d-stride, d]
    （clamp 到轨迹起点）。Codex P0-3（2026-08-09）：此前误写为 [d, d+2, d+4, d+6]
    未来帧——特征含目标动作结果，训练泄漏且与 live/eval 契约相反。"""
    off = np.arange(window) - (window - 1)   # [-(w-1), ..., -1, 0]
    return np.clip(decision + off * stride, video_start_frame, None)


def phase1(horizon: int) -> None:
    """窗口切片（动作/状态/prev/帧索引），无 GPU。horizon=action chunk 长度。"""
    ref = torch.load(REF, map_location="cpu", weights_only=True)
    aq01, aq99 = ref["normalization"]["action_q01"], ref["normalization"]["action_q99"]
    sq01, sq99 = ref["normalization"]["state_q01"], ref["normalization"]["state_q99"]
    norm = dict(ref["normalization"])
    out_path = win_out(horizon)

    def robust(x, lo, hi):
        lo_n, hi_n = lo.numpy(), hi.numpy()
        return np.clip(2 * (x - lo_n) / (hi_n - lo_n) - 1, -1, 1)

    files = sorted(
        p for p in ROOT.glob("data/metaworld_longtraj_*.pt")
        if not p.name.startswith("metaworld_longtraj_windows")
    )  # 排除 phase1 自身输出（windows 文件无 task 键）
    print(f"phase1(h={horizon}): {len(files)} task files")
    W = []
    for fi, path in enumerate(files):
        data = torch.load(path, map_location="cpu", weights_only=False)
        task_text = ENV_TO_TASK.get(data["task"])
        if task_text is None:
            continue
        try:
            tid = ref["metadata"]["tasks"].index(task_text)
        except ValueError:
            continue
        for ei, ep in enumerate(data["episodes"]):
            frames_jpeg = ep["frames"]      # list[bytes]
            actions = ep["actions"]         # [T,4]
            states = ep["states"]           # [T,4]
            T = len(frames_jpeg)
            last_start = T - 1 - ((SEQUENCE_LENGTH - 1) * CONTROL_STRIDE + (horizon - 1))
            if last_start < 0:
                continue
            for s in range(0, last_start + 1, CONTROL_STRIDE):
                acts = np.stack([
                    actions[s + t * CONTROL_STRIDE + h]
                    for t in range(SEQUENCE_LENGTH) for h in range(horizon)
                ]).reshape(SEQUENCE_LENGTH, horizon, 4)
                prev = np.stack([
                    np.zeros(4, dtype=np.float32)
                    if s + t * CONTROL_STRIDE == 0
                    else actions[s + t * CONTROL_STRIDE - 1]
                    for t in range(SEQUENCE_LENGTH)
                ])
                proprio = np.stack([
                    states[s + t * CONTROL_STRIDE] for t in range(SEQUENCE_LENGTH)
                ])
                # 帧索引：每个决策点的 4 帧窗口（编码阶段取帧）
                frame_idx = np.stack([
                    clip_frame_indices(s + t * CONTROL_STRIDE)
                    for t in range(SEQUENCE_LENGTH)
                ])  # [T, W]
                W.append({
                    "actions": robust(acts, aq01, aq99).astype(np.float32),
                    "prev": robust(prev, aq01, aq99).astype(np.float32),
                    "proprio": robust(proprio, sq01, sq99).astype(np.float32),
                    "task_id": tid,
                    "ep_id": fi * 10000 + ei,
                    "task_file": data["task"],
                    "ep_idx": ei,
                    "frame_idx": frame_idx,
                })
    n = len(W)
    print(f"phase1(h={horizon}): {n} windows, tasks={len(set(w['task_id'] for w in W))}")
    payload = {
        "actions": torch.from_numpy(np.stack([w["actions"] for w in W])),
        "previous_action": torch.from_numpy(np.stack([w["prev"] for w in W])),
        "proprio": torch.from_numpy(np.stack([w["proprio"] for w in W])),
        "instruction_id": torch.tensor([w["task_id"] for w in W], dtype=torch.long),
        "episode_id": torch.tensor([w["ep_id"] for w in W], dtype=torch.long),
        "pair_id": torch.arange(n, dtype=torch.long),
        "frame_refs": [(w["task_file"], w["ep_idx"], w["frame_idx"]) for w in W],
        "normalization": norm,
        "metadata": {
            "contract": "language_conditioned_mt50_longtraj",
            "tasks": ref["metadata"]["tasks"],
            "fps": FPS, "control_stride": CONTROL_STRIDE,
            "action_horizon": horizon,
            "action_contract": "executed-clip-fullframe",
            "n_trajectories": len(files),
        },
    }
    torch.save(payload, out_path)
    print(f"[out] {out_path}: {n} windows")


def phase2(device: str, horizon: int) -> None:
    """按帧索引解压窗口帧 → 冻结原始 V-JEPA 编码 → ST288 memmap。"""
    from prepare_pnpw_features import VJEPA21Backbone
    from va_compound.live_vjepa import _slot_coords

    win_path = win_out(horizon)
    st_npy, st_meta = st_paths(horizon)
    win = torch.load(win_path, map_location="cpu", weights_only=False)
    refs = win["frame_refs"]
    n = len(refs)
    print(f"phase2(h={horizon}): {n} windows, backbone=原始 V-JEPA 2.1（冻结）")

    # 窗口按 (任务, episode) 分组（phase1 按 (file, ep, start) 顺序 append，
    # 同一 (task, ep) 的窗口连续；dict 保持插入序 → 同一任务的组连续）。
    # 注意：必须按任务流式解码→编码→释放，一次性全量解码会吃满内存
    # （50 任务 ≈ 45GB 原图，本机只有 31GB，直接 swap 地狱）；且单任务整
    # 体解码（~5GB）+ memmap 脏页 + 模型也会超 31GB（实测 33GB swap 崩溃），
    # 故按 episode 懒解码：单 ep ~150MB，峰值内存 ~10GB（2026-08-09）。
    groups: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = {}
    for r, (tf, ei, fidx) in enumerate(refs):
        groups.setdefault((tf, ei), []).append((r, fidx))
    print(f"  grouped: {len(groups)} (task,ep) 组, "
          f"max {max(len(v) for v in groups.values())} windows/组", flush=True)

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device, dtype=torch.float16, max_tokens=144, local_files_only=True
    )
    vision_backbone.eval()
    coords = _slot_coords()

    ST_NPY, ST_META = st_npy, st_meta
    ST_NPY.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(ST_NPY, dtype=np.float16, mode="w+", shape=(n, SEQUENCE_LENGTH, 288, 768))
    print(f"memmap: {ST_NPY} shape={mm.shape}")

    t0 = time.time()
    B = 16
    done = 0
    cur_tf = None
    task_data = None
    for (tf, ei), items in groups.items():
        if tf != cur_tf:
            if cur_tf is not None:
                mm.flush()  # memmap 脏页写回，释放页缓存（防 31GB 内存爆）
                torch.cuda.empty_cache()
            task_data = torch.load(ROOT / "data" / f"metaworld_longtraj_{tf}.pt",
                                   map_location="cpu", weights_only=False)
            nf = sum(len(ep["frames"]) for ep in task_data["episodes"])
            print(f"  loaded {tf}: {len(task_data['episodes'])} eps, {nf} frames",
                  flush=True)
            cur_tf = tf
        # episode 级懒解码（单 ep ~150MB；整任务解码 ~5GB 会超 31GB 内存）
        ep_frames = [
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
            for b in task_data["episodes"][ei]["frames"]
        ]
        for start in range(0, len(items), B):
            rows = items[start:start + B]
            clips = []  # 每窗口 [T, W, 384, 384, 3]
            for r, fidx in rows:
                fidx = np.asarray(fidx)  # frame_refs 可能是纯 list（weights_only 兼容转换后）
                T, W = fidx.shape
                clip = np.stack([
                    np.stack([ep_frames[int(fidx[t, w])] for w in range(W)])
                    for t in range(T)
                ])  # [T, W, 384, 384, 3]
                clips.append(clip)
            frames_batch = np.stack(clips)  # [B, T, W, 384, 384, 3]
            with torch.inference_mode():
                # 不用 encode_live_frames：其 preprocess_batch 在 CPU 跑
                # bicubic+antialias 且 list() 逐元素转换（1.13 亿 Python 对象/批，
                # GPU 长期空闲，实测单批 >20s）。帧已 384×384（解码时 resize），
                # 直接在 GPU 归一化 → V-JEPA 前向（2026-08-09 卡死修复）。
                b, t, w, hh, ww, _ = frames_batch.shape
                frames = np.ascontiguousarray(
                    frames_batch.reshape(b * t * w, hh, ww, 3)
                )
                video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                video = video.div_(255.0).to(torch.device(device))
                if video.shape[-1] != 384 or video.shape[-2] != 384:
                    video = F.interpolate(
                        video, size=(384, 384), mode="bicubic",
                        align_corners=False, antialias=True,
                    )
                mean = torch.tensor(
                    (0.485, 0.456, 0.406), device=video.device
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    (0.229, 0.224, 0.225), device=video.device
                ).view(1, 3, 1, 1)
                inputs = ((video - mean) / std).reshape(b * t, w, 3, 384, 384)
                raw = vision_backbone._encode(inputs)  # [B*T, grid_tokens, D]
                st = vision_backbone._pool(raw, "spatiotemporal")  # [B*T, 288, D]
                st = st.reshape(b, t, 288, -1)  # [B, T, 288, 768]
            for k, (r, _) in enumerate(rows):
                mm[r] = st[k].cpu().numpy()
            done += len(rows)
            if (start // B) % 50 == 0:
                el = time.time() - t0
                print(f"  {tf}:{ei} {start}/{len(items)} "
                      f"(total {done}/{n}, {el:.0f}s)", flush=True)
        del ep_frames
        gc.collect()
    mm.flush()
    # meta.pt（与 mw_local288.pt / extract_st288_finetuned 同契约：coords 必须
    # 是 torch Tensor（weights_only=True 加载），load_st288_memmap 裸数据分支
    # 按 metadata.rows/tokens_per_decision 推断 shape）
    meta = {
        "vision_tokens_st_npy": str(ST_NPY),
        "coords": torch.from_numpy(coords),  # torch Tensor [288,3]，与 mw_local288 一致
        "metadata": {
            "source": f"longtraj-scripted-expert:h{horizon}",
            "rows": n,
            "tokens_per_decision": 288,
            "grid": [24, 24],
            "slot_grid": 12,
            "pooling": "spatiotemporal",
            "action_horizon": horizon,
        },
    }
    torch.save(meta, ST_META)
    print(f"[out] {ST_META}: rows={n} tok=288（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", choices=("1", "2"), default="1")
    ap.add_argument("--horizon", type=int, default=8,
                    help="action chunk 长度（E7 用 48；文件名/特征路径按 horizon 区分）")
    args = ap.parse_args()
    if args.phase == "1":
        phase1(args.horizon)
    else:
        phase2(args.device, args.horizon)
