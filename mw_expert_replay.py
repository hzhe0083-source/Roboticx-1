"""MetaWorld 专家回放冒烟（数据同源环境验证）。

目的：验证本地 metaworld 环境能否复现数据里的专家轨迹。这是模型闭环评估
公平性的前置条件（报告 §8.4 已知限制：此前 3.0.0/3.1.1 均 0/6）。

关键事实（实测确认）：
- 数据 80FPS；本地 metaworld env frame_skip=5 × timestep=0.0025 = 80Hz，一一对应。
- 数据 observation.environment_state 为 39 维 obs（非 qpos/qvel）：
  [hand(3), gripper(1), obj1_pos_quat(7), obj2(7)] × 2（帧堆叠）+ goal(3)。
- 本地与数据同模型（同 39 维布局、同 XML），唯一差异是随机 init 的物体位置。

对齐方式：reset 后把数据首帧的 nut 位姿写回 mujoco freejoint qpos，
手位置不动（数据首帧手位置与本地 reset 一致），再逐帧回放原始动作。

用法（CPU 仿真，不占 GPU）：
    python mw_expert_replay.py --task-index 0 --trials 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/ryan/Documents/robot/benchmark_data/raw/metaworld/lerobot_metaworld_mt50")
_CONFIG_CANDIDATES = (
    Path(__file__).resolve().parent / "metaworld_config.json",
    Path(
        "/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json"
    ),
)


def _load_task_descriptions() -> dict:
    for path in _CONFIG_CANDIDATES:
        if path.is_file():
            with path.open() as handle:
                return json.load(handle)["TASK_DESCRIPTIONS"]
    return {}


CONFIG: dict | None = None
TASK_DESCRIPTIONS = _load_task_descriptions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaWorld expert replay smoke")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


def load_episodes() -> list[dict]:
    import pyarrow.parquet as pq

    eps = pq.read_table(ROOT / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    return eps


def load_episode_rows(episode: dict) -> list[dict]:
    import pyarrow.parquet as pq

    start = int(episode["dataset_from_index"])
    length = int(episode["length"])
    # 用 episodes 元数据的 data/file_index 直接定位（此前遍历全部 492 个文件
    # 读取巨慢——末尾任务的 episode 需读几百个含图像的文件，卡数分钟）
    file_idx = int(episode["data/file_index"])
    path = ROOT / f"data/chunk-000/file-{file_idx:03d}.parquet"
    table = pq.read_table(path)
    first = int(table.column("index")[0].as_py())
    local = start - first
    return table.to_pylist()[local : local + length]


def find_nut_qpos_indices(env) -> tuple[int, int]:
    """返回 nut 在 qpos 中的 [起始, 结束) 索引（freejoint: xyz + quat）。"""
    model = env.model
    body_id = model.body("RoundNut").id
    # 找到该 body 的第一个 joint（free joint）
    for jnt in range(model.njnt):
        if model.jnt_bodyid[jnt] == body_id:
            adr = model.jnt_qposadr[jnt]
            nq = model.jnt_qposadr[jnt + 1] - adr if jnt + 1 < model.njnt else model.nq - adr
            return int(adr), int(adr + nq)
    raise ValueError("no joint for RoundNut")


def set_nut_pose(env, pos: np.ndarray, quat: np.ndarray) -> None:
    start, end = find_nut_qpos_indices(env)
    # 数据存的是 obs 里的 nut com 位置；qpos 是 body frame 原点，
    # 需减去 body 的 pos 偏移（实测 assembly 偏移 +0.13）
    offset = np.asarray(env.model.body("RoundNut").pos)
    env.data.qpos[start : start + 3] = pos - offset
    env.data.qpos[start + 3 : end] = quat


def move_body(env, bid: int, delta: np.ndarray) -> str:
    """平移 body：free joint 改 qpos；否则沿父链找 free joint；兜底改 body pos。

    box-close 等任务中 obj1 匹配到的可能是 weld 子体（如 top_link），其父链
    上是真正的自由体（boxbodytop）——沿父链平移才能带动整个物体。
    """
    name = env.model.body(bid).name
    import mujoco

    def try_free(b: int) -> bool:
        for jnt in range(env.model.njnt):
            if env.model.jnt_bodyid[jnt] == b:
                adr = int(env.model.jnt_qposadr[jnt])
                if env.model.jnt_type[jnt] == 0:  # free joint
                    env.data.qpos[adr : adr + 3] += delta
                    mujoco.mj_forward(env.model, env.data)
                    return True
                break
        return False

    if try_free(bid):
        return f"free body {name!r} moved by {np.round(delta, 3)}"
    parent = int(env.model.body_parentid[bid])
    # 注意：mujoco 中 world(body id 0) 的 parent 指向自身（parentid[0]=0），
    # 循环必须止于 world（parent > 0），否则死循环
    while parent > 0:
        if try_free(parent):
            return (
                f"ancestor free body {env.model.body(parent).name!r} moved by "
                f"{np.round(delta, 3)} (for {name!r})"
            )
        parent = int(env.model.body_parentid[parent])
    env.model.body(bid).pos = env.model.body(bid).pos + delta
    mujoco.mj_forward(env.model, env.data)
    return f"weld body {name!r} pos-moved by {np.round(delta, 3)}"


def align_objects(env, data_obs: np.ndarray, local_obs: np.ndarray) -> str:
    """把本地环境对齐到数据首帧：obj1（操作物体）平移 + target 位置。

    数据采集时物体与目标位置随机化生效（每 episode 不同），本地 reset 固定。
    实测（assembly/basketball/bin-picking/button-press 等）：对齐后专家回放
    success=True。obj1 参考点可能是 site 或 body com（per-env），故迭代对齐：
    每次按 obs[4:7] 差平移匹配 body（site 匹配 → body xpos 匹配 → 父链 free
    joint），直到 obs[4:7] 与数据一致（纯平移下 com/site 同步移动）。
    """
    target = data_obs[4:7].copy()
    target_quat = data_obs[7:11].copy()
    report = []
    import mujoco

    def obj1_body() -> int | None:
        cur = env._get_obs()[4:7]
        for i in range(env.model.nsite):
            if np.allclose(env.data.site_xpos[i], cur, atol=0.02):
                return int(env.model.site_bodyid[i])
        for bid in range(env.model.nbody):
            name = env.model.body(bid).name
            if not name or name.startswith(("right", "left")):
                continue
            if np.allclose(env.data.body(bid).xpos, cur, atol=0.02):
                return bid
        return None

    def free_joint_of(b: int) -> int | None:
        for jnt in range(env.model.njnt):
            if env.model.jnt_bodyid[jnt] == b:
                return jnt if env.model.jnt_type[jnt] == 0 else None
        return None

    # 1) quat 对齐：找 obj1 对应 free joint（含父链，止于 world id 0），写 qpos 四元数段
    bid = obj1_body()
    if bid is not None:
        jnt = free_joint_of(bid)
        if jnt is None:
            parent = int(env.model.body_parentid[bid])
            while parent > 0 and jnt is None:
                jnt = free_joint_of(parent)
                if jnt is None:
                    parent = int(env.model.body_parentid[parent])
        if jnt is not None:
            adr = int(env.model.jnt_qposadr[jnt])
            cur_quat = env.data.qpos[adr + 3 : adr + 7].copy()
            if np.linalg.norm(cur_quat - target_quat) > 1e-3:
                env.data.qpos[adr + 3 : adr + 7] = target_quat
                mujoco.mj_forward(env.model, env.data)
                report.append("quat aligned")

    # 2) 迭代对齐 pos（每次按 obs[4:7] 差平移，纯平移下 com/site 同步移动）
    for _round in range(6):
        cur = env._get_obs()[4:7].copy()
        delta = target - cur
        if np.linalg.norm(delta) < 1e-5:
            break
        matched = False
        for i in range(env.model.nsite):
            if np.allclose(env.data.site_xpos[i], cur, atol=0.02):
                report.append(move_body(env, int(env.model.site_bodyid[i]), delta))
                matched = True
                break
        if not matched:
            for body_id in range(env.model.nbody):
                name = env.model.body(body_id).name
                if not name or name.startswith(("right", "left")):
                    continue
                if np.allclose(env.data.body(body_id).xpos, cur, atol=0.02):
                    report.append(move_body(env, body_id, delta))
                    matched = True
                    break
        if not matched:
            report.append(f"round{_round}: obj1 not matched (delta {np.round(delta, 3)})")
            break
    # target 对齐（reward/success 判定 + peg 物理位置）
    env._target_pos = data_obs[36:39].copy()
    try:
        env.model.body("peg").pos = data_obs[36:39] - np.array([0.0, 0.0, 0.05])
        names = {env.model.site(i).name for i in range(env.model.nsite)}
        if "pegTop" in names:
            env.model.site("pegTop").pos = data_obs[36:39]
    except Exception:
        pass
    mujoco.mj_forward(env.model, env.data)
    final_err = float(np.linalg.norm(env._get_obs()[4:7] - target))
    report.append(f"final_obj1_err={final_err:.4f}")
    return "; ".join(report)


def main() -> None:
    args = parse_args()
    env_name = list(TASK_DESCRIPTIONS.keys())[args.task_index]
    description = TASK_DESCRIPTIONS[env_name]
    print(f"task[{args.task_index}] {env_name}: {description}")

    import metaworld

    episodes = load_episodes()
    task_episodes = [
        e for e in episodes if description in str(e.get("tasks"))
    ]
    print(f"episodes for task: {len(task_episodes)}")
    if not task_episodes:
        raise SystemExit(f"no episodes for {description!r}")

    wins = 0
    max_rewards = []
    for trial in range(args.trials):
        episode = task_episodes[trial % len(task_episodes)]
        rows = load_episode_rows(episode)
        first_obs = np.asarray(rows[0]["observation.environment_state"], dtype=float)
        actions = [np.asarray(r["action"], dtype=float) for r in rows]

        # lerobot 采集同款环境构造
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env._freeze_rand_vec = False
        env.reset(seed=42)

        # 关键对齐（实测结论，勿删）：
        # 1. 动作语义 = metaworld 标准环境动作（内部 clip(action,-1,1)×action_scale），
        #    数据 action 值域超 [-1,1] 但执行时被 clip——本地默认行为即正确，
        #    不需要 monkeypatch（unclipped 版手轨迹误差 0.045 vs clip 版 0.0002）。
        # 2. 操作物体（obj1）与目标位置（goal）随机化生效，需按数据对齐：
        #    obs[4:7] 可能是 site 或 body com（per-env），align_objects 通用匹配；
        #    obs[36:39] 写回 _target_pos（reward/success 判定用）。
        align_objects(env, first_obs, env._get_obs())

        success = False
        peak_reward = 0.0
        follow_errors = []
        hand_errors = []
        for step, action in enumerate(actions[: args.max_steps]):
            _, reward, terminated, truncated, info = env.step(action)
            peak_reward = max(peak_reward, float(reward))
            if info.get("success"):
                success = True
                break
            if step % 25 == 0 and step + 1 < len(rows):
                # 状态跟随：对比数据行里的 nut(obs[4:7]) 与手位置(obs[0:3])
                data_obs = np.asarray(rows[step + 1]["observation.environment_state"], dtype=float)
                cur_obs = env._get_obs()
                follow_errors.append(np.linalg.norm(cur_obs[4:7] - data_obs[4:7]))
                hand_errors.append(np.linalg.norm(cur_obs[:3] - data_obs[:3]))
        env.close()
        wins += int(success)
        max_rewards.append(peak_reward)
        tag = "SUCCESS" if success else "FAIL"
        err_str = (
            f" max_follow_err={max(follow_errors):.4f} max_hand_err={max(hand_errors):.4f}"
            if follow_errors
            else ""
        )
        print(f"  trial {trial}: {tag} peak_reward={peak_reward:.3f} steps={step+1}{err_str}")

    print(
        f"\nEXPERT REPLAY: {wins}/{args.trials} success; "
        f"peak_reward mean={np.mean(max_rewards):.3f}"
    )


if __name__ == "__main__":
    main()
