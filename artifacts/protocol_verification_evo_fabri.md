# 评估协议事实核验：Evo-1 与 FabriVLA（MetaWorld MT50）

> **任务**：只读 + 网络核验两条评估协议事实，供论文对比表与"RGB+EEF-only / RGB+state / state-only"三协议设计使用。
> **核验日期**：2026-08-08/09
> **证据版本**（git commit）：
> - `MINT-SJTU/Evo-1`：`1bf31c43`（2026-08-05）
> - `Youi-FabriX/FabriVLA`：`95609d27`（2026-07-24）
> - `Farama-Foundation/Metaworld`（gymnasium 版）：`ffb4f32d`（2026-08-07）
> - Evo-1 论文：arXiv:2511.04555；FabriVLA 论文：arXiv:2607.08575（v3，2026-08-02）
> - HuggingFace 发布物：`MINT-SJTU/Evo1_MetaWorld`（模型）、`MINT-SJTU/Evo1_MetaWorld_Dataset`（数据集）、`Youi-FabriX/FabriVLA`（模型）

---

## 0. 结论摘要

| # | 声明 | 证据 | 属实？ | 一句话结论 |
|---|---|---|---|---|
| E1 | Evo-1 官方评测客户端 `STATE_TAKE = 8` | 仓库代码行（见 §1.1） | ✅ 属实 | `mt50_evo1_client_prompt.py:55` |
| E2 | 输入 obs 前 8 维 = EEF xyz + gripper + object1 xyz + object1 quat_w（含物体坐标，**非纯 4D EEF-only**） | MetaWorld 源码 obs 布局（见 §1.2） | ✅ 属实（但见 E3） | 前 8 维 = `eef_xyz(3) + gripper(1) + obj1_pos(3) + obj1_quat_w(1)`，**泄漏了第一个物体的位置** |
| E3 | Evo-1 模型实际是按 **4 维 state（EEF xyz + gripper）** 训练的 | 官方 checkpoint `norm_stats.json` + 官方数据集首行（见 §1.3） | ✅ 属实（新发现，设计文档未掌握） | 24 维只是补零接口；评测时第 4–7 维（物体坐标）是训练分布外输入 → **Evo-1 80.6% 事实协议 ≈ RGB + 4D EEF-only** |
| F1 | FabriVLA 使用"24 维 state" | 仓库 config / 论文正文（见 §2） | ✅ 属实（但仅指 padded 接口宽度） | `state_dim=24` 是补零到 24 的模型接口，同 Evo-1 |
| F2 | 该 24 维 state 是否含 object positions | 官方评测 `--state-take 8` + checkpoint norm_stats（见 §2.3） | ❌ **训练 state 不含物体坐标**（4 维 = EEF+gripper，norm_stats 实证）；评测输入 obs 前 8 维含 obj1 xyz（分布外） | 与 Evo-1 完全同构：训练 4 维、接口补零 24、评测取 8 |

**对三协议设计的核心影响**：Evo-1 与 FabriVLA 的官方数字（80.6% / 90.0%）在**评估时**都不是严格 "RGB + 4D EEF-only"——它们多喂了 `obj1_pos(3) + obj1_quat_w(1)` 四个维度；但**训练时两家都有直接证据**（官方 checkpoint norm_stats，数值逐位相同）state 只有 4 维（EEF xyz + gripper），物体坐标维度在训练中是补零、评测时才是真实物体坐标（归一化后从 −1 跳到 ±1，属于分布外输入）。因此两家的数字应标为 "RGB + state（训练 4D EEF+gripper，接口 padded 24，评测取 obs 前 8 维）"，且与 "RGB+EEF-only" 协议实质等价。

---

## 1. Evo-1：`STATE_TAKE = 8` 与前 8 维语义

### 1.1 客户端代码证据

仓库：https://github.com/MINT-SJTU/Evo-1
文件：`MetaWorld_evaluation/mt50_evo1_client_prompt.py`

```python
# L55（User Config）
STATE_TAKE = 8                # Evo1 & rollout settings
HORIZON = 15
EPISODES = 10
EPISODE_HORIZON = 400
SEED = 4042

# L82-91
def obs_to_state(obs, take: int = STATE_TAKE) -> List[float]:
    if isinstance(obs, dict):
        if "observation" in obs:
            arr = np.asarray(obs["observation"], dtype=np.float32).ravel()
        else:
            parts = [np.asarray(v).ravel() for v in obs.values()]
            arr = np.concatenate(parts).astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr[:min(take, arr.shape[0])].tolist()
```

使用点（rollout 主循环）：

```python
# L353
obs, _ = sub.reset(seed=seed + ep)
...
# L383
state_vec = obs_to_state(obs)                       # 每决策步取 obs 前 8 维
# L386
actions = await evo1_infer(ws, img_bgr, state_vec, prompt=task_prompt)
# L388-391（chunk 执行：每 15 步 replan，动作只取前 4 维）
for i in range(HORIZON):
    a4 = np.asarray(actions[i][:4], dtype=np.float32)
    a4 = np.clip(a4, sub.action_space.low, sub.action_space.high)
    obs, _, terminated, truncated, info = sub.step(a4)
```

环境与图像协议（同一文件）：`gym.make_vec("Meta-World/MT50", vector_strategy="sync", seed=..., render_mode="rgb_array", camera_name="corner2")`（L298-304）；图像 448×448、旋转 180°、中心裁剪 2/3（L114-131）；每次推理发送 3 张图（1 真 + 2 零图，`image_mask=[1,0,0]`，L190-198）。

服务端（`Evo_1/scripts/Evo1_server.py`）：state 归一化后补零到 `target_dim=24`（L36, L160-173），再进模型（L244-266）：

```python
# L36
self.target_dim = 24
# L160-173
def normalize_state(self, state, arm_key, dataset_key):
    stats = self._get_stats_for(arm_key, dataset_key, "state")
    norm_state = self._normalize_tensor(state, stats, clamp=True)
    if norm_state.shape[-1] < self.target_dim:
        padding_size = self.target_dim - norm_state.shape[-1]
        ...norm_state = torch.cat([norm_state, pad_tensor], dim=-1)
    return norm_state
```

### 1.2 MetaWorld obs 布局（前 8 维具体是什么）

两个客户端都用 **gymnasium 新版 MetaWorld**（`gym.make_vec("Meta-World/MT50", ...)`，即 Farama-Foundation/Metaworld ≥2.5 的 API），其 39 维 state 布局与旧版 rlworkgroup 完全不同。源码：`metaworld/sawyer_xyz_env.py`：

```python
# L475-511（单帧组合，不含 goal；docstring: "flat observation array (18 elements)"）
def _get_curr_obs_combined_no_goal(self):
    pos_hand = self.get_endeff_pos()               # body("hand").xpos → 3 维（L67-69, L484）
    ...
    gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)  # 归一化开度 ∈[0,1]
    obs_obj_padded = np.zeros(self._obs_obj_max_len)   # L244: self._obs_obj_max_len = 14
    ...
    return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))  # 3+1+14 = 18

# L513-527（帧堆叠 + goal）
def _get_obs(self):   # docstring: "flat observation array (39 elements)"
    curr_obs = self._get_curr_obs_combined_no_goal()
    obs = np.hstack((curr_obs, self._prev_obs, pos_goal))   # 18 + 18 + 3 = 39
```

即 39 维布局为：

| 下标 | 内容 | | 下标 | 内容 |
|---|---|---|---|---|
| 0–2 | EEF 位置 xyz（`hand` body） | | 18–20 | 上一帧 EEF xyz |
| 3 | gripper 开度（两指间距/0.1，clip [0,1]） | | 21 | 上一帧 gripper 开度 |
| 4–6 | **object1 位置 xyz** | | 22–28 | 上一帧 object1 pos(3)+quat(4) |
| 7–10 | object1 四元数 | | 29–35 | 上一帧 object2 pos(3)+quat(4) |
| 11–13 | **object2 位置 xyz** | | 36–38 | goal 位置 xyz |
| 14–17 | object2 四元数 | | | |

（MT50 任务至多 2 个物体，`_obs_obj_max_len=14` 正好容纳 2×(3+4)。）

因此 **`obs[:8]` = `[eef_x, eef_y, eef_z, gripper, obj1_x, obj1_y, obj1_z, obj1_qw]`**——包含第一个物体的位置与四元数首分量。设计文档 §十二 的"非 4D EEF-only、需按 MetaWorld 版本锁定第 4–7 维语义"判断正确；**第 4–6 维 = object1 位置 xyz，第 7 维 = object1 四元数 w**。

### 1.3 训练侧真实 state 维度（关键新证据）

**官方发布的 MT50 checkpoint**（https://huggingface.co/MINT-SJTU/Evo1_MetaWorld）内 `norm_stats.json`：

```json
{
  "metaworld_robot": {
    "observation.state": {
      "min": [-0.499, 0.387, 0.044, 0.262],
      "max": [0.471, 0.892, 0.468, 1.0]
    },
    "action": {
      "min": [-10.851, -15.161, -17.021, -1.0],
      "max": [19.347, 12.408, 24.616, 1.0]
    }
  }
}
```

- state min/max **长度 = 4**：前 3 维数值范围 = sawyer 工作空间 EEF xyz；第 4 维 [0.262, 1.0] = gripper 开度。
- action min/max **长度 = 4**：前 3 维 = delta xyz（大范围来自示范数据），第 4 维 [−1,1] = gripper 指令。
- checkpoint `config.json`：`state_dim=24, per_action_dim=24, horizon=50`（模型接口宽 24，真实 4 维补零）。

**官方数据集**（https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset）首行（datasets-server `first-rows`）：

```
observation.state[0] = [0.0046, 0.6014, 0.1951, 1.0]     # eef xyz + gripper(全开=1.0)
action[0]          = [1.0542, -0.0139, -0.7520, 0.0]      # delta xyz + gripper
```

即**训练数据 `observation.state` 就是 4 维**。与论文附录 Table 5 的 "State dimension 24 (padded)" 完全自洽：24 是补零后的接口宽度（Evo-1 论文 `papers/evo1_2511_04555.txt` L589-592）。

**推论**：评测时喂入的 8 维里，第 4–7 维（obj1 pos + quat_w）是训练时从未见过的数值分布（训练时该位置恒为 0，归一化后恒为 −1；评测时是真实物体坐标，除以 (0−0) 后 clip 到 ±1）。**Evo-1 的 80.6% 事实上等价于 RGB + 4D state（EEF xyz + gripper）**，多喂的 4 维属于无效/干扰输入。

### 1.4 判定（Evo-1）

- `STATE_TAKE = 8`：**属实**（代码 L55，且是官方 README 指示的评测流程）。
- 前 8 维语义：**EEF xyz(3) + gripper(1) + object1 xyz(3) + object1 quat_w(1)**（新版 gymnasium MetaWorld 布局）；**不是**"4D EEF-only"，也**不是**旧版 MetaWorld 的 `pos+angle+vel` 排列。
- 但模型**训练时**只用前 4 维（EEF+gripper），第 4–7 维在训练中不存在（补零）。故把 Evo-1 协议写作 "RGB + state(4D EEF+gripper, padded 24)" 最准确；写作 "RGB+EEF-only" 也成立（额外 4 维未参与学习），但严格说评测输入含 obj1 坐标，论文应脚注。

---

## 2. FabriVLA：24 维 state 是否含 object positions

### 2.1 论文证据

论文：https://arxiv.org/abs/2607.08575（v3）。主文没有给出 `d_s` 的具体值，只说明 state 与 action 一样做补零：

> §3.2 Backbone and Encoders：the proprioceptive state $\mathbf{s}_k$, **zero padded like the actions**, is appended as one further token of width $e$（状态像动作一样补零后作为 1 个 token 拼接）
> §5.1 Model configuration：Meta-World actuates $d=4$ of the $D=24$ padded action dimensions.（24 是动作补零宽度，真实 4 维）

论文**从未声明** state 含 object positions；"24 维 state" 的说法只对应仓库 `ActionHeadConfig.state_dim=24`（补零接口，与 Evo-1 完全同构）。

### 2.2 仓库证据

仓库：https://github.com/Youi-FabriX/FabriVLA（`95609d27`）

评测脚本 `evaluations/metaworld/evaluate_mt50.py`（与 Evo-1 客户端同构；两家的 `tasks.jsonl` 与 `mt50_order.json` **逐字节相同**，diff 为空）：

```python
# L896
parser.add_argument("--state-take", type=int, default=8)   # 默认同样取 obs 前 8 维

# L334-348
def _prepare_state(self, obs):
    raw = as_observation_array(obs)          # L134-140，与 Evo-1 obs_to_state 等价
    take = min(int(self.args.state_take), raw.shape[0])
    if self.normalizer.enabled:
        take = min(take, self.normalizer.raw_state_dim)
    state = raw[:take].astype(np.float32)
    state = self.normalizer.normalize_state(state)     # 用 stats 前 take 维归一化
    state, state_mask = pad_1d(state, self.model_state_dim)   # 补零到 model_state_dim
    ...
```

模型接口（`fabri-vla/src/model/action_head.py`）：`ActionHeadConfig.state_dim=24, per_action_dim=24, horizon=50`（L18-20）；`state` 形状强制 `[B, 24]`（L250）。
训练配置（`fabri-vla/configs/train/1-exp_shallow_concat_proj_scratch_100k.yaml`）：`state_dim: 24, per_action_dim: 24, horizon: 50, image_size: 448`。
数据配置（`fabri-vla/dataset/config.yaml`）：`max_state_dim: 24, max_action_dim: 24, max_views: 3`；数据集为 LeRobot 格式 MetaWorld 数据集（README 要求 `meta/stats.json` 提供 state min/max）。

其余评测协议（`evaluate_mt50.py`）：10 episodes/task、400 步 horizon、`exec_horizon=5`（每 5 步 replan，L685-692）、成功判据 `info["success"]`（L703）、corner2 相机、448×448、rot180 + center crop 2/3、`N=50` 积分步。论文 §5.2：90.0% = Easy 95.0 / Med 88.2 / Hard 86.7 / V.Hard 90.0 的均分（四档平均，非逐任务微平均）。

### 2.3 官方 checkpoint norm_stats（下载分析）

`Youi-FabriX/FabriVLA/checkpoint_step_93000.pt`（1.76 GB）下载后解包提取 `config` 与 `norm_stats`：

```json
{
  "metaworld_sawyer": {
    "observation.state": {
      "min": [-0.499, 0.387, 0.044, 0.262],
      "max": [0.471, 0.892, 0.468, 1.0]
    },
    "action": {
      "min": [-10.851, -15.161, -17.021, -1.0],
      "max": [19.347, 12.408, 24.616, 1.0]
    }
  }
}
```

- **state min/max 长度 = 4**：前 3 维 = EEF xyz，第 4 维 = gripper 开度 [0.262, 1.0]。**训练 state 不含任何物体坐标。**
- action 长度 = 4（delta xyz + gripper）。
- `config`：`state_dim=24, per_action_dim=24, horizon=50, num_keep_layers=14, vlm=OpenGVLab/InternVL3_5-1B`。
- **与 Evo-1 的 norm_stats 数值逐位相同**（`metaworld_robot` vs `metaworld_sawyer` 仅是键名不同）——两家训练数据集统计分布一致（大概率同源，与 tasks.jsonl/mt50_order.json 逐字节相同互为印证）。

### 2.4 判定（FabriVLA）

- "24 维 state"：**属实但仅指 padded 接口宽度**（`state_dim=24`，同 Evo-1 的 24 padded；论文正文未声明 state 内容）。
- "含 end-effector、gripper、object positions"：**训练侧不成立**——官方 checkpoint 的 norm_stats 证明训练 state 只有 4 维（EEF xyz + gripper），**不含 object positions**；评测侧 `--state-take 8` 确实把 obj1 xyz + quat_w 喂进模型，但那是训练分布外的 4 个维度（训练时恒为 0）。**FabriVLA 90.0% 的事实协议 = RGB + 4D EEF-only（EEF xyz + gripper），与 Evo-1 完全同构。**

---

## 3. 对我们对比表的影响与三协议建议

### 3.1 两家协议的口径核对表

| 项目 | Evo-1（80.6%） | FabriVLA（90.0%） | 本项目现状（见 `eval_metaworld.py` / `baseline_table_literature.md` 脚注 6） |
|---|---|---|---|
| 任务集 | MT50 全 50 任务（`mt50_order.json` 四档分组） | 同左（文件逐字节相同） | 49 任务（**不能写 MT50**） |
| 每任务 episodes | 10（论文：10 trials × 5 runs 取平均） | 10 | 10 |
| horizon | 400 步 | 400 步 | — |
| 成功判据 | `info["success"]==1` | 同左 | — |
| replan 节奏 | 每 15 步（HORIZON=15） | 每 5 步（exec_horizon=5） | — |
| 图像 | corner2, 448×448, rot180, center-crop 2/3, 3 图（1 真+2 dummy） | corner2, 448×448, rot180, center-crop 2/3, 1 图 | — |
| state（评测输入） | obs 前 8 维 → 归一化 → 补零 24 | 同左（`--state-take 8` 默认） | — |
| state（训练输入） | **4 维 = EEF xyz + gripper**（norm_stats/数据集实证） | **4 维 = EEF xyz + gripper**（checkpoint norm_stats 实证，数值与 Evo-1 逐位相同） | 本项目自有 |
| 训练数据 | 50 demos/task（论文） | 同 Evo-1 数据集（tasks/order 相同，大概率同源） | 本项目自有 |

### 3.2 三协议建议（更新自 `c2irf_v2_vision_ablation.md` §十二）

| 协议 | 定义 | 用途 | 备注 |
|---|---|---|---|
| **RGB + EEF-only** | 图像 + `[eef_xyz(3), gripper(1)]`（4 维） | 主协议，证明视觉精密控制能力 | **与 Evo-1/FabriVLA 官方数字对齐**（他们的训练/事实协议即此） |
| **RGB + state** | 图像 + 完整 MetaWorld 39 维 state（或至少 8 维含 obj1 pos） | 公平 SOTA 对比 / 消融 state 信息量 | 注意：与官方评测的"前 8 维"对齐时，第 7 维（obj1 quat_w）是残废维度；建议要么复刻 8 维、要么明确写 24 padded 接口 |
| **state-only** | 只给 state（无图像） | 检查视觉是否真实贡献 | 建议直接做 4 维与 8 维两个变体 |

**具体建议**：

1. 对比表脚注统一改为：*Evo-1 与 FabriVLA 均以 RGB + state 在 MT50 上闭环评测：训练 state 为 4 维（EEF xyz + gripper），模型接口补零到 24 维；官方评估时取环境 obs 前 8 维（= EEF xyz + gripper + object1 xyz + object1 quat_w，后 4 维为训练分布外输入）。本项目对应协议为 RGB + 4D EEF-only。*（两家 norm_stats 数值逐位相同，可直接引用同一口径）
2. 不要复制"前 8 维"的残废协议（obj1 quat_w 第 8 维无意义、第 4–7 维对模型是分布外输入）；如果要做 "RGB + state"，做 **RGB + 完整 39 维**（信息量上界）或 **RGB + 8 维**（与官方评测输入逐位对齐）两个版本，结果放进消融表而非主表。
3. state-only 基线用 39 维（信息量上界）和 4 维（与官方对齐）各跑一遍，用于回答"视觉是否真实贡献"。
4. 主表对比口径：本项目 49 任务 × 10 episodes 闭环；Evo-1 论文口径为 50 demos/task × 10 trials × 5 runs；FabriVLA 为 10 episodes/task × H5。三者任务集/评估次数/重规划节奏均不同，**必须脚注**（沿用 `baseline_table_literature.md` 脚注 1/6）。
5. peg-insert 等双物体任务：两家官方 state 均不含第二个物体（hole）坐标——若我们的 "RGB+state" 协议给全 39 维，在双物体任务上会比官方多出 hole 信息，公平对比时应注明或单独给 "RGB+8D" 版本。

---

## 4. 需要更新的本地文档

- `artifacts/c2irf_v2_vision_ablation.md` §十二：第 2 条"Evo-1 官方评估客户端 STATE_TAKE=8，环境 observation 前 8 维送入模型（非 4D EEF-only）"——方向正确，但应补充：**训练 state 实为 4 维（EEF+gripper），8 维中的 4–7 维（obj1 pos + quat_w）是分布外输入**；第 1 条"FabriVLA 明确使用 24 维 state（含 end-effector、gripper、object positions）"——**24 仅是 padded 接口宽度，训练 state 为 4 维（EEF xyz + gripper），不含 object positions**（官方 checkpoint norm_stats 实证，见本报告 §2.3），90.0% 事实上是 RGB + 4D EEF-only。
- `artifacts/baseline_table_literature.md` 脚注 6：可补充两家输入通道细节（见 §3.2 建议 1）。

---

## 附：证据 URL 清单

- Evo-1 仓库：https://github.com/MINT-SJTU/Evo-1 （`MetaWorld_evaluation/mt50_evo1_client_prompt.py` L55/L82-91/L383；`Evo_1/scripts/Evo1_server.py` L36/L160-173；`Evo_1/config.py` L62）
- Evo-1 模型发布：https://huggingface.co/MINT-SJTU/Evo1_MetaWorld （`norm_stats.json`、`config.json`）
- Evo-1 数据集发布：https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset （`observation.state` 4 维，首行实证）
- Evo-1 论文：https://arxiv.org/abs/2511.04555 （附录 Table 5 "State dimension 24 (padded)"）
- MetaWorld（gymnasium 版）：https://github.com/Farama-Foundation/Metaworld （`metaworld/sawyer_xyz_env.py` L475-527/L244/L67-69）
- FabriVLA 仓库：https://github.com/Youi-FabriX/FabriVLA （`evaluations/metaworld/evaluate_mt50.py` L896/L334-348；`fabri-vla/src/model/action_head.py` L18-20；训练 config yaml；`fabri-vla/dataset/config.yaml`）
- FabriVLA 论文：https://arxiv.org/abs/2607.08575 （§3.2 "zero padded like the actions"、§5.1/§5.2）
- FabriVLA 模型发布：https://huggingface.co/Youi-FabriX/FabriVLA （`checkpoint_step_93000.pt`）
