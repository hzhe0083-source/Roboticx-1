# Stage B 主线结果登记（living document，数字以日志为准）

> 规则：每个数字必须来自对应日志文件（文件名+日期），禁止转抄猜测。
> 评估链：`logs/stageB_eval_chain.sh`；冒烟：`logs/stageB_smoke5000.sh`。

## 训练运行记录

| 运行 | 起止 | 配置 | 日志 | checkpoint | 状态 |
|---|---|---|---|---|---|
| v1 (nw0) | 2026-08-08 17:35–18:2x | 无 num_workers，25 steps/min | `stageB_langslot_40k_v1_nw0.log` | —（未到 5000） | 已停（重启加速） |
| v2 (nw4) | 2026-08-08 18:2x– | `--num-workers 4`，57 steps/min，ETA ~10.6h | `stageB_langslot_40k.log` | `checkpoints/stageB_langslot_40k.pt`（每 5000） | 训练中 |

**Loss 收敛趋势（风险项①缓解证据，2026-08-08 实测）**：
step 1: 0.244 → 100: 0.109 → 300: 0.083 → 400: 0.046 → 1000: 0.037 →
2000: 0.061（噪声峰）→ 近 400 步滑动均值 0.045。单调下行，未见发散。

## 5000 步迷你冒烟（`stageB_smoke5000.sh`，256 样本开环，2026-08-08 通过）

| 项 | 值 | 日志 |
|---|---|---|
| clean chunk0 / chunk_all | **0.1338 / 0.1043** | `stageB_smoke5000_lang.log` |
| wrong chunk0 / chunk_all | **0.4290 / 0.2746** | 同上 |
| wrong delta | **+220.7% / +163.4%**（语言流显著） | 同上 |
| 判据（宽松） | clean 有限且 <2.0 ✓；输出非退化 ✓ | — |
| 备注 | 修复链 5 处（sys.path/_dtype/max_tokens/coords/weights_only）后首次全通；训练 SIGSTOP 期间 Adam 零损失，CONT 后 step 5191→5831 连续 | `stageB_smoke5000_run6.log` |

## 完整评估链（40K 完成后，`stageB_eval_chain.sh`）

### 开环 + 语言消融（32 步口径，ST288 + 微调 backbone）

| 扰动 | chunk0 | chunk_all | delta chunk0 | 日志 |
|---|---|---|---|---|
| clean | 待填 | 待填 | — | `stageB_lang_ablation_wrong.log`（clean 每次重算） |
| wrong | 待填 | 待填 | 待填 | 同上 |
| blank | 待填 | 待填 | 待填 | `stageB_lang_ablation_blank.log` |
| swap | 待填 | 待填 | 待填 | `stageB_lang_ablation_swap.log` |

### 闭环 49×10（32 步口径）

| 指标 | 值 | 日志 |
|---|---|---|
| mean success | 待填 | `stageB_closedloop.log` |
| 95% CI | 待填 | 同上 |
| per-task 明细 | 待填 | 同上 |

### 与既有链对比

| 模型 | 开环 chunk_mae | 开环 success | 闭环 49×10 | 语言 wrong delta |
|---|---|---|---|---|
| mw_v5_direct_40k（Stage A 最强） | 0.0251 | 92.4% | 31.8% [22.6,41.4] | +1210% |
| **Stage B langslot（待填）** | — | — | — | — |

## 论文口径注意
- 闭环 49×10 与 Evo-1 的 10 trials/task 同口径但任务集不同（49 vs MT50 子集），
  对比表须脚注（见 `artifacts/baseline_table_literature.md`）。
- 闭环 95% CI 统一用 `scripts/closedloop_ci.py` 计算（macro ± 1.96·SE 口径，
  Wilson 为保守替代）——既有 v5 direct 的 [22.6, 41.4] 为手工值，
  脚本重算为 [22.2, 41.5]（同口径，±0.4pp 舍入差），论文一律用脚本值。

## 论文待回填清单（Stage B 数字落定后执行）
- [ ] `paper/ora0_paper.tex` MT50 节（Sec. 5.3.3 附近）：开环 chunk_mae/success、
      语言 vs task-id（wrong/taskid 消融）、闭环 49×10 + CI（closedloop_ci.py 口径）
      —— 当前文本是早期链旧数字（16.3% 等），须整节替换为 Stage B 数字
- [ ] `tab:baselines` 的 "ours" 两行：open-loop 与 closed-loop 数字更新 + 口径脚注
- [ ] LIBERO 节：等修复链（episode 分离重建 → C1/C2 → cosine 判决）数字
- [ ] VLA-RL 节：等 MT10 PPO 链数字
- [ ] 每个数字回填时注明日志文件名 + 日期（本文件规则）
