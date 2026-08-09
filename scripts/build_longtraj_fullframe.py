#!/usr/bin/env python
"""长轨迹 → fullframe 训练数据（2026-08-09）。

读 data/metaworld_longtraj_*.pt（scripted 反馈专家长轨迹，150-300 帧/条），
按 sliding-window 密集采样产出与 metaworld_fullframe_executed.pt 同构的
payload（live-vjepa 训练路径用，vision_tokens 零占位将被 pop）：

  - 决策间隔 CONTROL_STRIDE=6（13.3Hz），窗口 SEQUENCE_LENGTH=4 决策点
  - 动作 executed-clip 契约（clip(raw,-1,1)）+ 全局 q01/q99 继承
    （normalization 直接来自 metaworld_fullframe_executed.pt，禁止单独重算）
  - prev 契约：episode 首决策 0、其余 = 前一帧动作（与 prepare_metaworld 同款）
  - action_horizon 参数化（默认 8 = 现管线；--horizon 48 供 E7 对齐 Evo-1 配方）

用法：
    python scripts/build_longtraj_fullframe.py [--horizon 8|48]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 80
CONTROL_STRIDE = 6
SEQUENCE_LENGTH = 4
ACTION_HORIZON = 8  # 默认与现管线一致；--horizon 48 对齐 Evo-1

REF = ROOT / "data" / "metaworld_fullframe_executed.pt"
OUT = ROOT / "data" / "metaworld_longtraj_fullframe.pt"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    args = parser.parse_args()
    H = args.horizon

    ref = torch.load(REF, map_location="cpu", weights_only=True)
    aq01, aq99 = ref["normalization"]["action_q01"], ref["normalization"]["action_q99"]
    sq01, sq99 = ref["normalization"]["state_q01"], ref["normalization"]["state_q99"]
    norm = dict(ref["normalization"])

    def robust(x: np.ndarray, lo: torch.Tensor, hi: torch.Tensor) -> np.ndarray:
        lo_n, hi_n = lo.numpy(), hi.numpy()
        return np.clip(2 * (x - lo_n) / (hi_n - lo_n) - 1, -1, 1)

    files = sorted(glob.glob(str(ROOT / "data" / "metaworld_longtraj_*.pt")))
    if not files:
        raise SystemExit("no metaworld_longtraj_*.pt files found")
    print(f"found {len(files)} task files")

    windows_actions: list[np.ndarray] = []
    windows_prev: list[np.ndarray] = []
    windows_proprio: list[np.ndarray] = []
    windows_vision: list[np.ndarray] = []
    task_ids: list[int] = []
    ep_ids: list[int] = []
    total_windows = 0

    for fi, path in enumerate(files):
        data = torch.load(path, map_location="cpu", weights_only=False)
        eps = data["episodes"]
        task = data["task"]
        task_text = ENV_TO_TASK.get(task)
        if task_text is None:
            print(f"  WARN: {task} has no ENV_TO_TASK mapping, skip")
            continue
        try:
            tid = ref["metadata"]["tasks"].index(task_text)
        except ValueError:
            print(f"  WARN: {task} -> {task_text!r} not in ref tasks, skip")
            continue
        for ei, ep in enumerate(eps):
            frames = ep["frames"]      # [T,384,384,3] uint8
            actions = ep["actions"]    # [T,4] raw
            states = ep["states"]      # [T,4]
            T = len(frames)
            # sliding-window：起点每 CONTROL_STRIDE 帧滑动
            last_start = T - 1 - ((SEQUENCE_LENGTH - 1) * CONTROL_STRIDE + (H - 1))
            if last_start < 0:
                continue
            for s in range(0, last_start + 1, CONTROL_STRIDE):
                # 4 决策点 × H 步动作块（executed-clip + 全局归一化）
                acts = np.stack([
                    actions[s + t * CONTROL_STRIDE + h]
                    for t in range(SEQUENCE_LENGTH) for h in range(H)
                ]).reshape(SEQUENCE_LENGTH, H, 4)
                # prev：episode 首决策 0，其余前一帧动作（prepare_metaworld 同款）
                prev = np.stack([
                    np.zeros(4, dtype=np.float32)
                    if s + t * CONTROL_STRIDE == 0
                    else actions[s + t * CONTROL_STRIDE - 1]
                    for t in range(SEQUENCE_LENGTH)
                ])
                proprio = np.stack([
                    states[s + t * CONTROL_STRIDE] for t in range(SEQUENCE_LENGTH)
                ])
                windows_actions.append(robust(acts, aq01, aq99))
                windows_prev.append(robust(prev, aq01, aq99))
                windows_proprio.append(robust(proprio, sq01, sq99))
                windows_vision.append(np.zeros((SEQUENCE_LENGTH, 64, 768), dtype=np.float16))
                task_ids.append(tid)
                ep_ids.append(fi * 1000 + ei)
                total_windows += 1
        print(f"  {task}: {len(eps)} eps → windows so far {total_windows}")

    if total_windows == 0:
        raise SystemExit("no windows collected")
    n = total_windows
    payload = {
        "vision_tokens": torch.from_numpy(np.stack(windows_vision)),
        "language_hidden": ref["language_hidden"],  # 任务文本同构（49 任务缓存，训练按 id 取）
        "language_mask": ref["language_mask"],
        "proprio": torch.from_numpy(np.stack(windows_proprio)),
        "previous_action": torch.from_numpy(np.stack(windows_prev)),
        "actions": torch.from_numpy(np.stack(windows_actions)),
        "pair_id": torch.arange(n, dtype=torch.long),
        "instruction_id": torch.tensor(task_ids, dtype=torch.long),
        "episode_id": torch.tensor(ep_ids, dtype=torch.long),
        "normalization": norm,
        "metadata": {
            "contract": "language_conditioned_mt50_longtraj",
            "tasks": ref["metadata"]["tasks"],
            "fps": FPS,
            "control_stride": CONTROL_STRIDE,
            "action_horizon": H,
            "sampling": {"mode": "sliding-longtraj", "success_only": True},
            "action_contract": "executed-clip-fullframe",
            "source": [Path(f).name for f in files],
        },
    }
    torch.save(payload, OUT)
    print(f"[out] {OUT}: {n} windows, horizon={H}, "
          f"tasks={len(set(task_ids))}, eps={len(set(ep_ids))}")


if __name__ == "__main__":
    main()
