# ORA0 — VA-World 视觉动作模型

MetaWorld 视觉-动作策略：**VA（Vision-Action）流 + World 流（WAM4VA）** 通过 peer-sync 双向交换，动作由 Flow Matching 解码。主视觉为冻结 DINOv2。

当前主线是 **hard2**（`assembly-v3` + `door-unlock-v3`），不是 50 任务全量 MetaWorld。语言表来自 49 任务参考构建，另外 47 个任务没有训练信号。

## 架构

- **VA 流**：双向 Transformer（`bidir_va`），语言条件 + 视觉记忆，8 层。
- **World 流（WAM4VA）**：`peer_sync_h6`，在 VA 的 8 层之间逐层交换。World 用 `belief` / `world_map` 两个循环状态预测未来 DINO 特征图，把 `world_message` 作为 K/V 注入下一层 VA。
- **动作解码**：条件 Flow Matching（8 步 Euler），前 2 步 weight 1.0、后 4 步 weight 0.036。
- **World 监督**：DINO map 回归 + 静态区约束 + 动作排序（`--world-action-rank-stage final`）。早期 stage 监督有下限 `floor=0.1`。

## 世界状态

两条流行为不同：

- **感知流 `world_map`**：每个决策点的 stage 0 重锚到**当前真实 DINO 最后一帧**，只在该决策点内部的 8 个 stage 上做残差细化。评测再加 `--world-reset-every 4`，与训练窗口对齐。
- **认知流 `belief`**：跨决策点持续。写入是门控凸组合（`_gate_fuse`）再加 RMSNorm，梯度打通，让「该记什么」可学。固定常数收缩（`retention=0.9`）实测有害，已回退。

## 目录结构

`va_compound/` 按功能域分子包，顶层保留向后兼容 shim（旧 `from va_compound.X import` 仍可用）：

```
va_compound/
├── policy/    model.py（VACompoundPolicy）, end_to_end.py
├── world/     wmrm.py（WAM4VA）, world_supervision.py, world_contract.py
├── vision/    backbones.py, live_vjepa.py, metric_roi.py,
│              metric_visual_head.py, fovea.py, longtraj_frames.py
├── control/   servo.py, local_control_slots.py
├── utils/     exact_resume.py, flow.py, statistics.py
└── data_parallel.py   双卡：两次 backward 之后手动 allreduce 梯度
```

## 数据

peer_sync_h6 窗口是 T4/H6/A4（`sequence_length=4`、`action_horizon=6`、`action_dim=4`）。VA 与 World 使用**不相交的 episode**。

帧指针按 `(任务名, 文件内 episode 下标)` 寻址。扩产分片必须先合成**每任务一个** longtraj 文件（`scripts/merge_longtraj_expansion.py`），不能把多个 shard 当作额外 `--input` 喂给 phase 1，否则下标会撞到基础集的同一条轨迹。

| 家族 | 含义 |
|------|------|
| `DATA_TAG=v1` | 原始 hard2 切分（每任务约 15 条 VA episode） |
| `DATA_TAG=v2` | 扩产合并后的切分（VA 270 / World 216 条 episode） |
| `FRAMES_DIR` | 每任务一个 `metaworld_longtraj_<env>.pt` 的目录；v2 用 `data/frames_v2` |

## 运行

用环境变量覆盖解释器和权重路径；不要写死本机 venv。

```bash
export PY=/opt/conda/bin/python
export VERIFY_PY=$PY
export DINO=/path/to/dinov2_vitl14_reg4.safetensors
```

### 单卡

L20（45 GiB）能稳住的是 **batch 24**。36 会 OOM。

```bash
DATA_TAG=v2 FRAMES_DIR=data/frames_v2 DECODE_CACHE_TASKS=2 \
  bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh joint 30000 24
```

### 双卡（全局 batch 48 = 每卡 24）

不要包 `DistributedDataParallel`：peer 一步是 VA backward 再 World backward，DDP 会在第一次 backward 结束时做 allreduce，VA 独有梯度永远不同步。`va_compound/data_parallel.py` 在两次 backward 都结束后手动平均梯度。

```bash
# 25 epoch ≈ 5450 步（VA 10471 窗口 / 48）
DATA_TAG=v2 FRAMES_DIR=data/frames_v2 DECODE_CACHE_TASKS=1 NGPUS=2 \
  SAVE_EVERY=436 CHECKPOINT_DIR=/path/to/local/ckpts \
  bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh joint 5450 48
```

`MODE`：`prepare`（数据准备）、`preflight`（契约校验）、`joint`（训练）。双卡时 `--batch-size` 是**全局** batch，必须能被 GPU 数整除。`num_workers` 必须为 0（解码帧常驻内存，fork 会翻倍拷贝）。

### 评测（闭环，10 trials/task）

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  bash scripts/eval_mw_hard2_wam4va.sh <checkpoint.pt> data/hard2_peer_h6_p2_eval_v2.pt
```

`eval_metaworld.py` 默认 `--world-reset-every 4`。无 `/dev/dri` 的容器用 OSMesa；`libOSMesa` 必须与主机 glibc 匹配（Ubuntu 22.04 不能加载 24.04 编出来的 `.so`）。

### 测试

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  "$PY" -m pytest tests/ -q --ignore=tests/test_recovery_param.py
```

## 当前主线

- 训练：`scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh`（v2 数据，双卡 batch 48）
- 评测：`eval_metaworld.py`（闭环，`--world-reset-every 4`）
- 数据：`data/hard2_peer_h6_p2_*_v2.pt` + `data/frames_v2/`
