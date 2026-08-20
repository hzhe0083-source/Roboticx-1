# DINOv2 + MT-VJ + VA + FM 在 MetaWorld task35 失败审查

日期：2026-08-16  
对象：`checkpoints/e7_dino_main_p35_dm_grid16_15k.pt`  
任务：`peg-insert-side-v3`（全局 task 35）  
原始闭环结果：`1/10`；当前同 checkpoint 重跑：`2/10`。二者均未建立相对历史 `1/10` baseline 的可靠提升。

## 一句话判决

**不是“DINO 没有精细视觉能力”，而是模块间的信息契约断裂：task35 的 MT-VJ state 没有 hole 和 pegHead，随后又经一个对动作几乎无影响的 2-token 通道进入 VA；DINO 主视觉的 4 帧则没有 frame/time encoding，严格退化为无序帧-token 集。与此同时，名为 `clean` 的训练子集主要是 perturbed recovery 监督，当前 FM/递归策略是否适合这一条件分布仍未由 matched baseline 建立。**

---

## 1. 问题与最强 sanity baseline

| 项目 | 结果 | 证据等级 |
|---|---:|---|
| 原始 15k grid16 闭环 | `1/10` | supported，`logs/e7_dino_main_p35_dm_grid16_15k_closedloop_1x10.log` |
| 当前同 checkpoint 复跑 | `2/10` | supported，`logs/e7_dino_grid16_15k_repro_1x10.log` |
| 官方 scripted expert，同 seeds 35000–35009 | `10/10` | supported，本次审查实测 |
| 历史 V-JEPA/MT-VJ task35 量级 | 多为 `0/10–1/10` | supported，`logs/` 历史闭环日志 |
| checkpoint SHA256 | `44b0f94938ac0d05d3faeb49c21645724a046175f964f11da02705e1bb109afa` | supported |

统计限定：原日志的单任务 task-bootstrap 区间 `[10%,10%]` 是退化的伪窄区间。`1/10` 的 Wilson 95% CI 约为 `[1.8%,40.4%]`（Clopper–Pearson exact 为约 `[0.25%,44.5%]`）。因此 10 trials 足以说明系统不可依赖，但不足以区分小幅 ablation。

官方 expert 的 `10/10` 证明环境、seed 和 success 判据可解；它使用 privileged state，不能当作同观察预算的 learned-policy baseline。完整的 expert-through-student-chunker sanity test 仍应保留为下一轮 E0。

---

## 2. 根因排序

### RC1 — MT-VJ 的 task35 几何语义退化：没有 hole，也没有 success-critical pegHead

**等级：supported（结构与运行时实测）；它导致整个策略失败的因果份额为 partially supported。**

1. task35 的 `env._get_pos_objects()` 返回的是 `pegGrasp` site，而任务 success 使用 `pegHead`：
   - MetaWorld 源码：`.../site-packages/metaworld/envs/sawyer_peg_insertion_side_v3.py:130-135,164-177`；
   - success 是 `||([1,2,2] * (pegHead-target))|| <= 0.07`。
2. 当前 metric label generator 使用 generic roles：
   - `object = env._get_pos_objects()`；
   - 单实体任务 `interface/progress = object`；
   - `target = env._target_pos`；
   - 见 `prepare_metaworld_metric.py:175-206`。
3. task35 的 `_target_pos` 虽投影在画面内，但位于 box 内部；target 使用严格 depth-surface visibility，因此本次随机 128 样本中：
   - `target visibility = 0/128`；
   - `object/interface` 世界坐标完全相同，最大差为 `0`。
4. 真实 heldout 189 个窗口中，训练后 metric head 的 target predicted visibility 均值约 `1.02e-8`，即 `_mtvj_metric_positions = p × visibility` 将 target 坐标实际归零；tool/pegGrasp/interface visibility 均约 `0.99`。
5. `pegGrasp` 与 `pegHead` 初始世界距离固定约 `0.1304m`；corner2 的 480px render 中投影差为 `33.3–42.4px`（均值 `37.4px`），折算到 DINO 224px 输入约为 `15.5–19.8px`。

因此，task35 的 8-D metric state 实际近似是：

```text
[tool_yx, pegGrasp_yx, 0_target_yx, duplicate_pegGrasp_yx]
```

它没有 hole 位置、没有 pegHead，也没有 rod orientation。`aux_rmse≈11px` 只描述被判 visible 的角色，不证明学到了 task35 的插入几何。

额外契约问题：DINO visual aux 使用 `loc_only=True`（`train.py:3067-3079`）；策略侧 `_mtvj_metric_positions()` 只取 `p×visibility`（`train.py:94-109`），并不使用 `MetricFieldOutput.relation`。所以名义上的 axis/depth relation 不是当前 DINO policy 的有效输入。

### RC2 — 即使 metric 有信息，VA 也几乎不使用它

**等级：supported（checkpoint 级因果消融）。**

当前 dense readout 在每层把 `512` patch K/V 与 `2` metric K/V 拼入同一个 softmax，见 `va_compound/model.py:967-995`。没有 source balancing、metric bias 或单独 gate。

父会话在 heldout 16 窗口、固定 FM noise 上做 `full` vs `metric tokens=0`，保持 dense patches 不变：

| 消融 | `Δcondition / ||condition||` | `Δprediction / ||prediction||` | 动作 MAE 变化 |
|---|---:|---:|---:|
| metric-only zero | `0.0468%` | `0.0267%` | `3.32e-5` |

这直接说明当前 checkpoint 的 HD/metric channel 对动作近似无因果影响。metric head 并未 hard-detach：本 run `--mtvj-train-metric-head` 为真；但动作梯度必须穿过近零影响的读取通道，所以它主要由每 50 步一次的 visual aux 学定位，而不是由 policy loss 学“什么几何对控制有用”。

结论：继续训练 ROI 或把定位 RMSE 从 5px 降到 3px，仍会通过同一近死通道进入 policy。本次 ROI job 因此在 step 350（`rmse_full≈3.82px`）停止。

### RC3 — DINO main 的 4 帧没有时间编码，主路径严格不知道帧顺序

**等级：supported（形式化代码审查 + 实测置换消融）。**

缓存/在线 DINO 将 4 个单帧 patch grid 直接 reshape 为 `[B, 4×256, 1024]`，见 `train.py:2155-2192`。`encode_condition()` 仅做 per-token linear 和无位置偏置 attention，见 `va_compound/model.py:2083-2111`；没有 frame embedding、temporal positional embedding 或 temporal mask。

这不表示 DINO token 没有单帧空间信息；它表示 VA 对 4 个 frame block 的排列不敏感。实测：

| 操作 | `condition relative change` | `prediction relative change` | prediction MAE change |
|---|---:|---:|---:|
| 主 DINO 四帧完全反序，dense/metric 不变 | `1.1e-7` | `9e-8` | `2e-8` |
| dense 最近两帧反序，并重算 metric | `31.86%` | `58.14%` | `0.1477` |

证据日志：`logs/diag_dino_temporal_permutation_15k.log`。

因此，主 1024-token 路径只能读“4 帧中出现了什么”，不能读“哪帧更新、运动方向是什么”；有效时序几乎全部压在 dense 最近两帧的 `ΔtH11 + t-coordinate` 旁路上。把视频型 V-JEPA 换成逐帧 DINO 时，这是一处静默 temporal-contract regression。

### RC4 — `dino35_clean` 不是 task35 clean 数据；训练主分布是 recovery

**等级：数据事实 supported；它是否有害、造成多少失败为 partially supported。**

`data/metaworld_longtraj_windows_h48_dino35_clean.pt` 是从 all49 `fine2_dino_clean` 抽 task35 行得到的子集；metadata 的 `clean_resample` 列表不包含 task35。task35 仍来自 legacy：

- `data/metaworld_longtraj_peg-insert-side-v3.pt`；
- 30/30 episodes 的 `perturbed=True`；
- 仅 30 条成功轨迹，共 732 windows。

本次重算：

| 数据量 | recovery 占比 |
|---|---:|
| 所有 valid action targets | `78.43%` |
| 实际执行 prefix0–5 的 valid targets | `69.45%` |
| 按训练 prefix=1、tail=0.036 加权后的总 loss mass | `71.09%` |
| 含任意 recovery target 的窗口 | `720/732 = 98.36%` |

这不是少量 augmentation，而是训练主分布。它不自动等于“坏数据”：recovery 可能对闭环有益；现有实验没有 clean-nominal × recovery 的 matched crossing，因此只能说数据命名和实验解释错误，不能把 recovery 占比直接当作因果根因。

另有 information mismatch：scripted expert 按完整 39-D observation 的 `peg_pos` 与 `goal_pos` 做阈值状态机，而 student 的显式 proprio 只存 `obs[:4]`（EEF xyz + gripper），其余必须从 RGB/历史隐式恢复。该 mismatch 是 supported；是否形成不可约 ambiguity 仍需 oracle-state crossing 验证。

### RC5 — 当前 FM endpoint 分布较宽，但“动作幅度塌缩”已被证伪

**等级：FM variance partially supported；全局幅度收缩为 refuted。**

在 heldout 189 windows、valid prefix0–5、`K=12` 独立 FM samples 上：

| 指标 | 结果 |
|---|---:|
| expert `mean |a| / dim` | `0.5208` |
| prediction `mean |a| / dim` | `0.4883`，ratio `0.937` |
| single-sample MAE | `0.1849 ± 0.0059` |
| 12-sample mean action MAE | `0.1672` |
| predictive std / dim | `0.2247` |
| cosine | `0.7686` |
| oracle best-of-12 per-position MAE | `0.0287` |

证据日志：`logs/diag_dino_fm_multisample_15k.log`。

`best-of-12` 是使用 expert target 做选择的不可部署 oracle，且逐位置选择不保证 coherent chunk；它只说明 FM 分布中存在接近专家的样本。较大的 predictive std 说明单次 rollout 对 flow noise 敏感。

闭环 `K=8` sample mean 在 3 个探索性 trials 上没有改善，反而明显变差（`logs/e7_dino_grid16_15k_flowmean8_stage_diag_3x.log`）。这与多峰控制分布“均值可能不是有效动作”一致，但 n=3 且闭环对数值扰动敏感，不能升级为 FM 主因证明。

此前“prediction 只有 expert 11–20% 幅度”的结论来自忽略 `action_valid_mask` 和单 sample 误读，已撤销。masked 结果显示幅度约 `94–103%`，最优 scalar gain 约 `0.95`。

---

## 3. 闭环 mismatch：存在，但尚未证明是第一根因

| mismatch | 已建立事实 | 因果结论 |
|---|---|---|
| previous action | train 用 expert previous action；eval 用 model last action；本 15k run 未启用 `--prev-dropout` | mismatch supported；导致 1/10 only partially supported |
| recurrent visual memory | train 每样本只展开 `T=4` decisions；eval 持续递归整个 episode | mismatch supported；是否需 reset 取决于真实训练 state/reset 语义，因果 partially supported |
| replanning | 每 6 env steps 重规划，只执行 chunk 0–5 | contract supported；“6 步太慢”仍 speculative |
| FM solver | train/eval 均用 8 Euler steps | contract 同构 supported；8-step 数值误差为主因 speculative |

探索性 3-seed probes：

- `--prev-zero` 明显恶化，说明 previous action 当前是有用输入，不能在 eval 直接清零；这不排除 teacher-forcing exposure bias。
- `--memory-reset-every 4` 得到 `1/3`，原 telemetry run 为 `0/3`，有改善迹象但样本太小。
- read-only telemetry run 与原始 no-telemetry run 在 borderline seed 上不完全复现；这些 3-seed 结果只能用于定位，不应写成已证实的闭环因果。

阶段 telemetry 显示基础模型有两局将 success 使用的 pegHead weighted distance 降到约 `9.9–10.1cm`，但阈值为 `7cm`；另一路约 `30cm`。这说明部分 rollout 接近最后阶段，但不是稳定完成。

---

## 4. 已排除或必须降级的解释

| 旧解释 | 判决 | 原因 |
|---|---|---|
| “DINO 特征坏了” | refuted | frozen patch 对角色线性定位可达约 `5.9–9.1px`；grid16 wiring 正常 |
| “14px patch 天生达不到 task35 精度” | unsupported | success 是 3-D weighted radius `0.07m`，不是固定 5px 门槛 |
| “metric head 从 action loss detach” | refuted | 本 run `train_metric_head=True`，存在 action gradient path；问题是 policy 读取近零 |
| “FM 动作幅度只剩 11–20%” | refuted | 原诊断忽略 mask；masked amplitude ratio 约 `94–103%` |
| “tail42 就是轨迹后期插入阶段” | refuted | eval 每 6 步重规划，只执行 chunk 0–5 |
| “loss=0.203 本身导致失败” | unsupported | FM velocity loss 数值不能直接解释 closed-loop success |
| “ROI 再精修几像素即可修复” | refuted for current route | ROI 输出仍进入对动作影响约 `0.027%` 的 metric route |

---

## 5. 最小可证伪实验矩阵

### E0：完整执行栈 sanity gate（先做）

让 scripted expert action 经过 student eval 的同一 normalization、chunk buffer、clip、execute cadence 与 env wrapper。

- 若 `<8/10`：先修执行 plumbing，不再讨论 DINO/MT-VJ/FM。
- 若 `>=8/10`：gross action-interface bug 降级，进入 E1。

`8/10` 是 sanity gate，不是论文级差异显著性门槛；正式结论应扩大到固定 50 个 paired seeds。

### E1：Geometry content × fusion route 2×2

| 几何内容 / 路由 | 当前 2-token flat route | 直接注入 action/state conditioning |
|---|---:|---:|
| 当前退化 roles | A | B |
| task-aligned oracle roles：tool、pegGrasp、pegHead、hole | C | D |

推荐先用 simulator oracle geometry 做小模型/短训机制试验，不先训练更复杂 ROI。

- `D >> C`：fusion route 是核心瓶颈；
- `C >> A`：role semantics 是核心瓶颈；
- `D` 仍失败：优先查动作数据/decoder/closed-loop，不再加视觉模块。

直接注入可以是 `MLP(g)` 加到 state/action query；不要继续把 2 metric tokens 与 512 patch tokens放在同一个无平衡 softmax 中。

### E2：Data × decoder 2×2（必须 crossed）

| 训练数据 | deterministic direct prefix6 head | 当前 recurrent FM |
|---|---:|---:|
| 当前 30 条 perturbed/recovery-heavy 数据 | A | B（现系统） |
| clean nominal demos + 单独标注的 recovery demos | C | D |

控制变量：同 backbone、同 4-frame input、同训练 updates、同 prefix mask、同 paired eval seeds。

- `C-A` 估计数据侧效果；
- `B-A` 估计 FM/decoder 效果；
- `D` 与其余格比较交互。

不要先做 action gain 放大；幅度收缩已被证伪。deterministic head 是必要的简单 baseline，不是预设赢家。

### E3：只在 E1/E2 后测试 temporal/closed-loop 修复

用共同初态与共同随机数做：

1. DINO 4 个 frame blocks 加显式 frame embedding，matched retrain；
2. replan cadence `{1,3,6}`；
3. previous action `{expert clamp / self-generated training / normal}`；
4. memory `{matched T=4 reset / long-unroll matched training / continuous eval}`。

只有 paired success delta 的 95% CI 排除 0，才把相应 mismatch 升级为 causal root cause。若 T=4 只是 TBPTT detach 而非 state reset，不应把 eval memory 人为 reset 当成训练同构。

### E4：统计与 go/no-go

先在固定 50 个 paired seeds 上重建 current baseline；报告每个 seed 的 success、最小 `obj_to_target`、grasp/near 状态及 flow seed。单任务用 trial-level Wilson/exact binomial interval，不用 task bootstrap。

建议 go/no-go 规则：

- 机制试验必须在 paired success delta 上给出 95% CI；
- 同时报告连续指标 `min obj_to_target`，避免二元 7cm 阈值掩盖接近程度；
- 不能再用 `1/10 -> 2/10` 或 `3/10` 单独宣称模块有效。

---

## 6. 建议的最短修复路线

1. **先改数据/标签，不再加模块**：重采 task35 clean nominal；recovery 单独分层；roles 改为 `tool, pegGrasp, pegHead, hole`，并加入 contract test：target 不能 `0/N visible`，object/interface 不能意外完全同点。
2. **建立简单 baseline**：同数据训练 deterministic prefix6 head；确认现有视觉条件是否足以闭环。
3. **修融合契约**：task-aligned geometry 直接注入 state/action conditioning；必要时再比较 source-balanced metric attention。
4. **补 temporal contract**：给四个 DINO frame blocks 显式 frame embedding；不要把逐帧 DINO concat 当作视频编码器。
5. **最后处理闭环生成**：self-generated previous-action training、matched memory unroll、replan cadence；只有 direct baseline 建立后再决定是否保留 FM。

---

## 7. 本次审查产物

- 时间顺序与 metric causal probe：`logs/diag_dino_temporal_permutation_15k.log`
- FM 多样本 valid-prefix probe：`logs/diag_dino_fm_multisample_15k.log`
- 阶段 telemetry：`logs/e7_dino_grid16_15k_stage_diag_3x.log`
- previous-action probe：`logs/e7_dino_grid16_15k_prevzero_stage_diag_3x.log`
- memory probe：`logs/e7_dino_grid16_15k_mem4_stage_diag_3x.log`
- FM sample-mean probe：`logs/e7_dino_grid16_15k_flowmean8_stage_diag_3x.log`
- 当前 10-trial rerun：`logs/e7_dino_grid16_15k_repro_1x10.log`
- 新增 eval diagnostics：`eval_metaworld.py` 的 `--debug-stage-metrics`、`--flow-samples`
- 单任务 CI 修复：`va_compound/statistics.py`、`tests/test_stats_ci.py`

## 最终表述边界

可以说：

> DINO 的单帧几何信息存在，但当前 DINO→MT-VJ→VA 契约没有把 task35 的 success-critical geometry 与时间顺序可靠传给动作策略；同时训练数据并非 task35 clean，且当前 stochastic/recurrent decoder 尚未由 matched simple baseline 证明适合该数据分布。

不应说：

> DINO 失败了；14px patch 不够；loss 太高；FM 幅度塌缩；tail42 是插入阶段；ROI 精度再提高即可修复。
