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
    "Pull a puck to a goal": 2.0,
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


# Evo-1 / FabriVLA / EvoMind 的 MT50 报告分桶。训练采样权重是本项目自己的
# curriculum，不能拿上面的权重反推 benchmark 难度。
MT50_BENCHMARK_GROUPS: dict[str, frozenset[str]] = {
    "easy": frozenset(
        {
            "button-press-topdown-v3",
            "button-press-topdown-wall-v3",
            "button-press-v3",
            "button-press-wall-v3",
            "coffee-button-v3",
            "dial-turn-v3",
            "door-close-v3",
            "door-lock-v3",
            "door-unlock-v3",
            "door-v3",
            "drawer-close-v3",
            "drawer-open-v3",
            "faucet-close-v3",
            "faucet-open-v3",
            "handle-press-side-v3",
            "handle-press-v3",
            "handle-pull-side-v3",
            "handle-pull-v3",
            "lever-pull-v3",
            "peg-unplug-side-v3",
            "plate-slide-back-side-v3",
            "plate-slide-back-v3",
            "plate-slide-side-v3",
            "plate-slide-v3",
            "reach-v3",
            "reach-wall-v3",
            "window-close-v3",
            "window-open-v3",
        }
    ),
    "medium": frozenset(
        {
            "basketball-v3",
            "bin-picking-v3",
            "box-close-v3",
            "coffee-pull-v3",
            "coffee-push-v3",
            "hammer-v3",
            "peg-insertion-side-v3",
            "push-wall-v3",
            "soccer-v3",
            "sweep-into-goal-v3",
            "sweep-v3",
        }
    ),
    "hard": frozenset(
        {
            "hand-insert-v3",
            "nut-assembly-v3",
            "pick-out-of-hole-v3",
            "pick-place-v3",
            "push-back-v3",
            "push-v3",
        }
    ),
    "very_hard": frozenset(
        {
            "nut-disassemble-v3",
            "pick-place-wall-v3",
            "shelf-place-v3",
            "stick-pull-v3",
            "stick-push-v3",
        }
    ),
}
MT50_BENCHMARK_TASK_TO_GROUP = {
    task: group
    for group, tasks in MT50_BENCHMARK_GROUPS.items()
    for task in tasks
}
if len(MT50_BENCHMARK_TASK_TO_GROUP) != 50:
    raise RuntimeError("official MT50 benchmark groups must contain 50 unique tasks")

# MetaWorld's native v3 registry and the Evo-1/FabriVLA Gym wrapper use five
# different slugs for the same environments. Keep rollout slugs native and
# canonicalize only for leaderboard grouping.
MT50_BENCHMARK_ENV_ALIASES = {
    "assembly-v3": "nut-assembly-v3",
    "disassemble-v3": "nut-disassemble-v3",
    "door-open-v3": "door-v3",
    "peg-insert-side-v3": "peg-insertion-side-v3",
    "sweep-into-v3": "sweep-into-goal-v3",
}


def canonical_mt50_benchmark_env(env_name: str) -> str:
    return MT50_BENCHMARK_ENV_ALIASES.get(env_name, env_name)


def task_weights_for(tasks: list[str]) -> list[float]:
    """数据集 metadata.tasks（instruction_id 顺序）→ per-task 权重列表。"""
    unknown = [t for t in tasks if t not in TASK_WEIGHTS]
    if unknown:
        print(f"[task-sampling] WARNING: {len(unknown)} 任务无难度定义，按 1.0：{unknown[:5]}")
    return [TASK_WEIGHTS.get(t, DEFAULT_WEIGHT) for t in tasks]


def summarize_mt50_benchmark_trials(trials: list[dict]) -> dict:
    """Reduce raw trials to the official four equally weighted difficulty tiers."""
    counts = {
        group: {"successes": 0, "trials": 0, "tasks": set()}
        for group in MT50_BENCHMARK_GROUPS
    }
    for row in trials:
        task = canonical_mt50_benchmark_env(str(row.get("env_name", "")))
        group = MT50_BENCHMARK_TASK_TO_GROUP.get(task)
        if group is None:
            raise ValueError(f"trial has unknown MT50 env_name: {task!r}")
        counts[group]["successes"] += int(bool(row.get("success")))
        counts[group]["trials"] += 1
        counts[group]["tasks"].add(task)

    seen = set().union(*(values["tasks"] for values in counts.values()))
    complete = seen == set(MT50_BENCHMARK_TASK_TO_GROUP)
    groups = {}
    for name, values in counts.items():
        n_trials = int(values["trials"])
        groups[name] = {
            "successes": int(values["successes"]),
            "trials": n_trials,
            "n_tasks": len(values["tasks"]),
            "success_rate": (
                None if n_trials == 0 else float(values["successes"] / n_trials)
            ),
        }
    rates = [groups[name]["success_rate"] for name in MT50_BENCHMARK_GROUPS]
    return {
        "contract": "evomind_mt50_four_tier_v1",
        "complete_mt50": complete,
        "n_tasks": len(seen),
        "groups": groups,
        "bucket_average": (
            float(sum(rates) / len(rates)) if complete else None
        ),
        "raw_episode_success": (
            None
            if not trials
            else float(sum(bool(row.get("success")) for row in trials) / len(trials))
        ),
    }
