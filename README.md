# ORA0 — VA-World 视觉动作模型

MetaWorld 视觉-动作策略：**VA（Vision-Action）流 + World 流（WAM4VA）** 通过 peer-sync 双向交换，动作由 Flow Matching 解码。主视觉为冻结 DINOv2（已取代 V-JEPA）。

## 架构

- **VA 流**：双向 Transformer（`bidir_va`），语言条件 + 视觉记忆，8 层。
- **World 流（WAM4VA）**：`peer_sync_h6` 模式，在 VA 的 8 层之间逐层交换。World 用 `belief`/`world_map` 两个循环状态预测未来 DINO 特征图，把 `world_message` 作为 K/V 注入下一层 VA 的注意力。
- **动作解码**：条件 Flow Matching（8 步 Euler），前 2 步 weight 1.0、后 4 步 weight 0.036。
- **World 监督**：DINO map 回归 + 静态区约束 + 动作排序（`--world-action-rank-stage final`）。

## 目录结构

`va_compound/` 按功能域分子包，顶层保留向后兼容 shim（旧 `from va_compound.X import` 仍可用）：

```
va_compound/
├── policy/    model.py（VACompoundPolicy）, end_to_end.py
├── world/     wmrm.py（WAM4VA）, world_supervision.py, world_contract.py
├── vision/    backbones.py, live_vjepa.py, metric_roi.py,
│              metric_visual_head.py, fovea.py, longtraj_frames.py
├── control/   servo.py, local_control_slots.py
└── utils/     exact_resume.py, flow.py, statistics.py
```

## 运行

### 环境

```bash
python=/home/ryan/Documents/robot/ORA/.venv/bin/python
```

### 训练（peer_sync_h6，hard2 双任务）

```bash
bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh joint 30000 18
```

`MODE` 可为 `prepare`（数据准备）、`preflight`（契约校验）、`joint`（训练）。训练脚本内部调用 `train.py`，关键超参：DINO 主视觉（grid 16 × 4 帧）、WMRM st_blocks predictor（6 层 × 384 宽）、batch 18、lr 1e-4。

### 评测（闭环，10 trials/task）

```bash
bash scripts/eval_mw_hard2_wam4va.sh <checkpoint.pt> data/hard2_peer_h6_p2_eval_v1.pt
```

自动编排脚本：`auto_eval_on_snapshot.sh`（快照评测）、`auto_transition_to_all49.sh`（hard2 → all49 切换）。

### 测试

```bash
/home/ryan/Documents/robot/ORA/.venv/bin/python -m pytest tests/ -q
```

部分测试需要 `av`、`metaworld`、`timm`、`einops`，均已加入本地 venv。

## 训练数据契约

peer_sync_h6 数据为 T4/H6/A4 窗口（`sequence_length=4`、`action_horizon=6`、`action_dim=4`），
VA 与 World 使用**不相交的 episode**（`validate_peer_data_isolation` 保证）。

## 循环状态稳定性

`wmrm.py` 的 `belief` 与 `world_map` 是跨 8 stage × T 决策点持久化的循环状态。

`belief` 的写入原为纯加法残差，展开是 `belief <- (I - KH) belief + K evidence`（Kalman 形式），稳定条件 `rho(I - KH) < 1` 从未被约束——训练只展开 4 个决策点，`1.331^4 = 3.14` 倍完全看不见。闭环实测（`scripts/diag_belief_growth.py`）`rho = 1.331`：每决策点恒定 ×1.331，决策点 145 时 `|belief| = 1.76e19`，146 溢出 float32，159 时 `world_message` 变 NaN。

修法是**门控融合**（`_gate_fuse`，对应 MemoryVLA 式 7-8）：`g = sigmoid(Linear([belief, update]))`、`belief <- g*update + (1-g)*belief`。凸组合让 `rho <= 1` 成为构造保证，且门看得见已积累的记忆，"该记什么"成为可学量。曾用的固定收缩（常数 `retention=0.9`）实测有害（同一 checkpoint 15% → 5%），因为常数门等比例衰减有用与有害的更新。**新增参数，旧 checkpoint 不能再 strict load。**

`world_map` 保持纯加法；它的开环深度由部署侧 `--world-map-reset-every 1` 截断到单个决策点的 8 个 stage（实测 159 个决策点触发 158 次重锚）。

## 当前主线

- 训练：`run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh`
- 评测：`eval_metaworld.py`（闭环，`--world-reset-every 4` 对齐训练窗口）
- 数据：`data/hard2_peer_h6_p2_*`（VA/World/eval 三份不相交切分）
