"""MT50 任务难度分层权重（E7 分层采样，2026-08-09）。

权重表：easy 0.5 / medium 1.0 / hard 2.0 / very-hard 3.0
（sota_plan_v2.md 第 11 项，用户指示：困难任务多训练、简单任务少训练）。
key 必须与数据集 metadata.tasks（= ENV_TO_TASK 的 value）完全一致。
未收录任务 → 权重 1.0（medium），并在训练启动时打警告。
"""

TASK_WEIGHTS: dict[str, float] = {
    # ---- easy (0.5)：按钮/关门/推类，成功率高的任务 ----
    "Reach a goal position": 0.5,
    "Bypass a wall and reach a goal": 0.5,
    "Push the puck to a goal": 0.5,
    "Bypass a wall and push a puck to a goal": 0.5,
    "Press a button": 0.5,
    "Press a button from the top": 0.5,
    "Bypass a wall and press a button": 0.5,
    "Bypass a wall and press a button from the top": 0.5,
    "Close a door with a revolving joint": 0.5,
    "Open a door with a revolving joint": 0.5,
    "Push and close a drawer": 0.5,
    "Open a drawer": 0.5,
    "Slide a plate into a cabinet": 0.5,
    "Slide a plate into a cabinet sideways": 0.5,
    "Get a plate from the cabinet": 0.5,
    "Get a plate from the cabinet sideways": 0.5,
    "Press a handle down": 0.5,
    "Press a handle down sideways": 0.5,
    "Pull a handle up": 0.5,
    "Pull a handle up sideways": 0.5,
    "Pull a lever down 90 degrees": 0.5,
    "Push a button on the coffee machine": 0.5,
    "Push and close a window": 0.5,
    "Push and open a window": 0.5,
    "Rotate the faucet counter-clockwise": 0.5,
    "Rotate the faucet clockwise": 0.5,
    "Rotate a dial 180 degrees": 0.5,
    # ---- medium (1.0)：默认档 ----
    "Push a mug under a coffee machine": 1.0,
    "Pull a mug from a coffee machine": 1.0,
    "Kick a soccer into the goal": 1.0,
    "Sweep a puck off the table": 1.0,
    "Sweep a puck into a hole": 1.0,
    "Grasp a stick and pull a box with the stick": 1.0,
    "Lock the door by rotating the lock clockwise": 1.0,
    "Unlock the door by rotating the lock counter-clockwise": 1.0,
    "Grasp the cover and close the box with it": 1.0,
    "Dunk the basketball into the basket": 1.0,
    "Grasp the puck from one bin and place it into another bin": 1.0,
    # ---- hard (2.0)：抓取/放置/工具类 ----
    "Hammer a screw on the wall": 2.0,
    "Pick and place a puck onto a shelf": 2.0,
    "Pick and place a puck to a goal": 2.0,
    "Pick up a puck from a hole": 2.0,
    "Unplug a peg sideways": 2.0,
    "Pick a nut out of a peg": 2.0,
    "Grasp a stick and push a box using the stick": 2.0,
    "Pick a puck, bypass a wall and place the puck": 2.0,
    # ---- very-hard (3.0)：插入/装配类 ----
    "Insert the gripper into a hole": 3.0,
    "Insert a peg sideways": 3.0,
    "Pick up a nut and place it onto a peg": 3.0,
}

DEFAULT_WEIGHT = 1.0


def task_weights_for(tasks: list[str]) -> list[float]:
    """数据集 metadata.tasks（instruction_id 顺序）→ per-task 权重列表。"""
    unknown = [t for t in tasks if t not in TASK_WEIGHTS]
    if unknown:
        print(f"[task-sampling] WARNING: {len(unknown)} 任务无难度定义，按 1.0：{unknown[:5]}")
    return [TASK_WEIGHTS.get(t, DEFAULT_WEIGHT) for t in tasks]
