# P2 / H6 VA–WAM 联合训练交接文档（2026-08-21）

## 一句话状态

当前已完成 **40 Hz（80 Hz 控制、planning stride=2）** 的 VA–WAM 分层联合训练改造；训练在 step 200 前数值稳定，但尚未证明学会。闭环评估发现 WAM 的跨决策 belief 无界累积，约第 23 次规划发生 NaN。因此训练已暂停，下一位需要先完成 **有界 recurrent belief** 修复并重新训练/验证，不能直接宣称模型可部署。

## 当前架构契约

- VA 与 WAM 采用 one-stage-delayed peer exchange：每层从同一 pre-stage snapshot 并行读取；VA 输出和 WAM world message 在下一层相互可见；没有 `delta`、writeback 或 WAM 第二动作头。
- 最终物理动作唯一由 `LN(A_8) -> 6-layer AdaLN Flow Matching -> 8-step Euler -> H6×4` 产生。
- WAM 输出为 next-DINO latent map、world tokens、belief/innovation，不直接产生动作。
- 训练数据分离，但参数联合更新：每个 optimizer step 使用一个 VA batch 和一个 World batch，依次 backward，一次 optimizer step。VA/World episode 严格不重叠。
- 在线 VA↔WAM message 保持可微；为避免 recurrent Jacobian 爆炸，WAM 内部对“上一层的非线性读取”已有局部 stop-gradient，但 residual/message 数值和当前层参数训练仍保留。这不是 VA/WAM 二选一冻结。

核心代码：

- [va_compound/model.py](/home/ryan/Documents/robot/ORA0/va_compound/model.py)
- [va_compound/wmrm.py](/home/ryan/Documents/robot/ORA0/va_compound/wmrm.py)
- [train.py](/home/ryan/Documents/robot/ORA0/train.py)
- [eval_metaworld.py](/home/ryan/Documents/robot/ORA0/eval_metaworld.py)

## 为什么必须使用 P2 高规划频率

旧数据/部署实际是 `fps=80, control_stride=6`，即 **13.33 Hz**。H6 只缩短了预测动作尾部，并没有提高真实重新规划频率；仅把 eval 的 execution horizon 改成 1/2/3 会造成训练/部署时序失配。

当前 P2 统一为：

| 项目 | 当前值 |
| --- | --- |
| 控制频率 | 80 Hz |
| planning/control stride | 2 raw steps |
| 实际规划频率 | 40 Hz |
| 每次预测动作 | 完整 H6×4 |
| World transition | `d -> d+2`，只消费动作前缀 `H[:2]` |
| T4 decision offsets | `[0, 2, 4, 6]` |
| 闭环 execution horizon | 2 |

相关数据契约为 `peer_sync_h6_p2_world_windows_v1`。不能把 `planning_stride` 改成 2 却继续让 World predictor 使用完整 H6 预测 `d+2`，那会使用目标之后的动作，产生因果错误。

## 数据与训练入口

P2 数据已生成，三组 episode 无交集：

| 数据 | 窗口/episode |
| --- | --- |
| `data/hard2_peer_h6_p2_va_train_v1.pt` | 1350 / 30 |
| `data/hard2_peer_h6_p2_world_train_v1.pt` | 1034 / 24 |
| `data/hard2_peer_h6_p2_eval_v1.pt` | 207 / 6 |

对应 split manifest：

- `data/hard2_peer_h6_p2_va_world_partition_v1.json`
- `data/hard2_peer_h6_p2_world_split_v1.json`

预检命令：

```bash
cd /home/ryan/Documents/robot/ORA0
bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh preflight
```

正式 runner：

```bash
bash scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh joint <additional_steps> 3
```

在修改模型前向/损失契约后，**不要**把旧 checkpoint 当作 exact continuation；应新开 run ID，或明确更新训练契约和 resume 规则。

## 已完成的数值稳定化

### 1. Map recurrence

原来 world map 同时作为 ST predictor 的非线性输入和 residual base：

`m_next = m + R(clip, m, cond)`

跨 `T4 × 8 stages` 会重复乘 `I + J_R`。现在 predictor 读取 `previous_map.detach()`，但 residual base 保留原 `previous_map`。前向值不变；前层 map 仍通过恒等 residual 路径接收梯度；当前 predictor、VA 和 WAM 仍有梯度。

### 2. Innovation projection

`_project_out()` 原本直接计算平方和/点积；长闭环时有限的 `~1e18` innovation 会在 FP32 `square().sum()` 溢出成 `Inf`，之后 `Inf / Inf -> NaN`，而 `0 * NaN` 也会污染结果。

当前工作树已改为 scale-safe reduction + `torch.where()`，并且上一 innovation 仅作为投影方向的稳定 memory，不回传病态的逆范数梯度。最新新增的极大有限输入测试已通过；仍需在最终 belief 修复后跑完整 regression。

### 3. Belief recurrent reader

`evidence_from_belief`、`belief_write`、`belief_from_world` 对前一 belief 的非线性重复读取已改为 read-only context；当前 stage 的 update、world loss 与 residual state path 仍可训练。这将训练阶段的梯度尖峰从 `1e10` 量级消掉，但**不能阻止 belief 值本身在长闭环中持续累积**。

## 训练记录与检查点

### 干净 checkpoint

- step 100：
  `checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s2000_step100.pt`
  - `global_step=100`
  - P2/H6、joint dual-stream、双 sampler/RNG、optimizer state 完整
  - SHA256：`3798ba078e38eae2596c86ac66376ad2cab10e1478f5b98fc3be62227e38ab84`

- step 200（已冻结 hardlink）：
  `checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.recurrence_stable.from_s100.to_s2000_step200.pt`
  - `global_step=200`
  - 模型、metric、relation 和 AdamW state 均 finite
  - 参数没有权重爆炸；问题是运行时 recurrent state，不是 checkpoint corruption

### 已观察到的训练结果

`step 101–200` 使用 map/belief Jacobian 修复后：

- 无 `gradient_spike`、NaN、OOM；raw gradient 为 `5.99–18.79`。
- flow 前 10 步均值 `0.6808 -> 0.6345`，有小幅改善。
- World objective `0.4646 -> 0.6014`，后段含切换到 assembly，不能按此断言收敛。
- World `gain` 与 `mgain` 100 步内均未转正，尚未超过 static-copy baseline。
- VA/World 双流任务轮换独立正常：World 约 step 193 切至 assembly，VA 约 step 195 切至 assembly。

结论：**训练稳定，但没有“学会”的证据。**

日志：

- `logs/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.scratch.s2000.log`
- `logs/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.innovation_stable.from_s100.to_s2000.log`
- `logs/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.recurrence_stable.from_s100.to_s2000.launcher.log`

## 闭环失败：已定位根因

用 step 200 在 held-out task0/seed0 做 40 Hz eval：checkpoint 与 P2 契约均通过，但第 23 次规划的 WAM stage 1 报 `world_message contains NaN or Inf`。

故障链如下：

```text
training: 每 batch memory=None，只展开 T4 决策
deployment: memory 持续 500 environment steps
belief: 4.80 -> 689 -> 2.42e6 -> 7.28e15 -> 2.19e18
innovation energy overflow -> Inf/Inf -> NaN
-> belief_write -> belief_to_pred -> ST predictor -> z_map/world_message
```

故障前以下量均 finite，因而不是视觉输入/权重损坏：

- DINO token max `9.33`
- `metric_g` max `0.583`
- VA action/vision max `2.73 / 3.37`
- stage-0 map max `8.81`
- stage-1 evidence max `0.410`

训练 BF16 autocast、评估 policy FP32 确有差异，但 WAMState 和 `_project_out` 均强制 FP32；这不是 NaN 的第一原因。

诊断文件：

- `/tmp/ora0_eval_nan_optrace.log`
- `/tmp/ora0_eval_reset4_h500.log`
- `/tmp/p2_step200_finite_audit.json`

## 评估说明

诊断命令：

```bash
bash scripts/eval_mw_hard2_wam4va.sh \
  checkpoints/mw_hard2_va_world_state_exchange_joint_h6_p2_v1.recurrence_stable.from_s100.to_s2000_step200.pt \
  data/hard2_peer_h6_p2_eval_v1.pt
```

该 launcher 会强制 P2：`planning_stride=execution_horizon=wmrm_cycle_steps=2`，即 40 Hz；旧 P6/H48 checkpoint 会被拒绝。

`--memory-reset-every 4` 可作为“匹配当前 T4 训练深度”的诊断兜底：同一 checkpoint 已跑完 500 环境步且无 NaN，但 task0/seed0 结果为 `0/1`。它只能证明数值不崩，**不能**作为长期 world-memory 已学会的结论。

## 下一位的优先任务

1. 在 `WAM4VA._forward_from_snapshot()` 中实现无新参数的 **有界 belief state**（例如平滑 bounded/gated recurrence），使部署长期状态不会指数增长；保留 current-stage World loss 到 `belief_write` 的梯度，不要恢复 delta/writeback，不要冻结 VA 或 WAM。
2. 给 `_project_out()` 保留 scale-safe/`where` 实现，增加长时间状态测试：至少 250 次规划、每次 8 stages，断言 belief/innovation/map/message 全 finite。
3. 给训练增加更长的 recurrent rollout，或明确一个与训练长度一致的 deployment memory policy；当前 T4 训练、500-step 无限 memory 是未训练过的时域。
4. 由于 belief 前向会改变，创建新的 checkpoint lineage，先跑到 step 100/300。验收先看：无 spike、World gain/mgain 是否越过 0、long rollout finite。
5. 仅在无 reset 的 40 Hz held-out 闭环跑通后，执行 10 trials/task 正式评估；成功率、per-task 结果和 Wilson CI 才是“真正学会”的证据。

## 验收门槛

不要只用训练 loss 判断。至少同时满足：

1. 训练穿过双任务切换仍无 gradient spike/NaN。
2. World `gain` / `mgain` 在 held-out 或稳定验证上可超过 static baseline。
3. 无 `memory_reset_every` 的 500-step、40 Hz 闭环无非有限状态。
4. held-out MetaWorld 闭环成功率在 2 tasks × 10 trials 下报告，附 per-task 成功数和 95% Wilson CI。

## 已知工作树注意事项

- 工作树很脏，包含用户已有的删除/实验文件；不要使用 `git reset --hard` 或批量 checkout。
- 最新 `_project_out()` scale-safe patch 后仅跑过 innovation focused tests；在继续前跑：

```bash
/home/ryan/.venvs/pytorch-gpu/bin/python -m pytest -q \
  tests/test_wmrm.py tests/test_peer_sync_world.py tests/test_peer_joint_training.py \
  tests/test_visual_world_rollout.py tests/test_exact_resume.py \
  tests/test_p2_deployment_contract.py tests/test_peer_sync_h6_runner_protocol.py \
  tests/test_longtraj_data_contract.py tests/test_wam4va_episode_split.py \
  tests/test_wam4va_world_action_eval.py
```

- 在 latest safe-projection patch 之前，上述集合为 `207 passed`。
- 当前没有活跃 `train.py` 进程；step 200 checkpoint 已保存，可只读检查，但不应在未完成 belief 修复时继续把它当作部署模型。
