# Stage B 主线结果登记（living document，数字以日志为准）

> 规则：每个数字必须来自对应日志文件（文件名+日期），禁止转抄猜测。
> 评估链：`logs/stageB_eval_chain.sh`；冒烟：`logs/stageB_smoke5000.sh`。

## 训练运行记录

| 运行 | 起止 | 配置 | 日志 | checkpoint | 状态 |
|---|---|---|---|---|---|
| v1 (nw0) | 2026-08-08 17:35–18:2x | 无 num_workers，25 steps/min | `stageB_langslot_40k_v1_nw0.log` | —（未到 5000） | 已停（重启加速） |
| v2 (nw4) | 2026-08-08 18:2x– | `--num-workers 4`，57 steps/min，ETA ~10.6h | `stageB_langslot_40k.log` | `checkpoints/stageB_langslot_40k.pt`（每 5000） | 训练中（后被 fullframe 协议取代） |
| fullframe Stage 1 | 2026-08-08 23:0x–2026-08-09 02:04 | Evo-1 式：VA+槽+flow 联合训练，V-JEPA 冻结，10k 步 | `fullframe_stage1.log` | `checkpoints/fullframe_stage1.pt`（02:04） | ✅ 完成 |
| fullframe Stage 2 | 2026-08-09 02:1x–（运行中） | resume Stage1 + `--vision-unfreeze-all`（V-JEPA 解冻 lr 1e-6），40k 步，~59.5 步/分 | `fullframe_stage2.log` | `checkpoints/fullframe_stage2.pt`（每 5000，40k 终版覆盖） | 运行中（PID 2240269） |

**训练配置（fullframe 协议，Stage 1/2 共用）**：`--data data/metaworld_fullframe_skeleton.pt --live-vjepa --sliding-window --success-only --frame-aug --single-task --flow-cond adaln --flow-semantic --flow-layers 6 --flow-steps 10`（Stage 2 追加 `--vision-unfreeze-all --lr-vision 1e-6`）。全帧监督（窗口每 6 帧滑动）、π0.5 式帧增强；skeleton payload 只能配 `--live-vjepa` 用。评估链 checkpoint 指向 `checkpoints/fullframe_stage2.pt`。

**Loss 收敛趋势（风险项①缓解证据，2026-08-08 实测）**：
step 1: 0.244 → 100: 0.109 → 300: 0.083 → 400: 0.046 → 1000: 0.037 →
2000: 0.061（噪声峰）→ 近 400 步滑动均值 0.045。单调下行，未见发散。

**fullframe Stage 2 全程 loss（2026-08-09 实测，每 2000 步段均值）**：
0-2k: 0.0878 → 4-6k: 0.0825 → 8-10k: 0.0764 → 12-14k: 0.0761 →
16-18k: 0.0729 → 20-22k: 0.0702 → 24-26k: 0.0696 → 26-28k: 0.0683 →
28-30k: 0.0664 → 38-40k: 0.0646（完整段 n=2000）；
全程单调下行无发散（flow 速度场 MSE，与 direct 的 chunk_mae 不同度量不可直比）。

**✅ 训练完成（2026-08-09 13:38）**：step=40000，终版 checkpoint
`checkpoints/fullframe_stage2.pt`（721M，mtime 13:38）落盘并通过守望者校验；
守望者 13:39:47 启动评估链（阶段 1：ST288 重提取进行中）。
全程分段均值：0.0878 → 0.0825 → 0.0764 → 0.0761 → 0.0729 → 0.0702 →
0.0696 → 0.0683 → 0.0664 → 0.0646（0-2k 至 38-40k，单调下行，未见平台期）。

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
| mean success | **22.0% (108/490)** | `stageB_closedloop.log`（2026-08-09 16:5x，修复归一化后终版） |
| 95% CI | [14.5%, 30.6%] | 同上 |
| per-task 明细 | 见日志 | 同上 |

### 与既有链对比

| 模型 | 开环 chunk_mae | 开环 success | 闭环 49×10 | 语言 wrong delta |
|---|---|---|---|---|
| mw_v5_direct_40k（Stage A 最强） | 0.0251 | 92.4% | 31.8% [22.6,41.4] | +1210% |
| **Stage B langslot（待填）** | — | — | — | — |

## C_OL 反事实位移（2026-08-09 从既有日志提取，§5.6 回填素材）
日志：logs/col_libero_3scene_va8_20k.log / col_libero_e2e_B40k.log（2026-08-06 跑，
32 步口径，libero_video_v2 全量 360 样本）：

| 模型 | 首执行动作 C_OL | E_swap−E_clean（绝对） | 95% CI | 判读 |
|---|---|---|---|---|
| A（libero_3scene_va8_20k，冻结） | 0.04382 | **+0.01562** | [0.00819, 0.02344] | CI 不含 0 → 语言反事实效应显著 |
| B40k（e2e，坍塌） | 0.04434 | **−0.00034** | [−0.00737, 0.00721] | CI 含 0 → 语言不改变输出（坍塌行为证据） |

C_OL(exec)/持久性位移：A=0.160 / B40k=0.162（<0.1 才视为"语言几乎不改变输出"，
0.16 需结合 E_swap−E_clean 差异判读）。注：B40k 训练集=评估集（v2 全量），
绝对值待修复链 heldout 重测；A vs B40k 的**对比方向**不受泄漏影响。

## ⚠️ 归一化口径缺陷修复（2026-08-09 14:2x）

**现象**：`fullframe_stage2.pt` 手动闭环（13:52，`fullframe_stage2_closedloop.log`）
前 2 任务 0/10。

**根因**：stage1/2 训练用 `data/metaworld_fullframe_skeleton.pt` 的 q01/q99
（原始 parquet 动作分位数，如 action_q01=[-4.08,-3.84,-12.84,-1]、
action_q99=[9.37,8.18,10,1]；`robust_normalize = 2(x-low)/(high-low)-1 ∈ [-1,1]`，
与 `eval_metaworld.py:1002` 反归一化互为逆），而闭环评估链第 3 步错传了
`--features data/metaworld_features_v5.pt`（v5 是裁剪到 [-1,1] 的动作，
q01/q99=±1）→ 反归一化动作被错误缩放/平移（模型输出不再饱和到 ±1，
精细动作全部失真）→ 0/10。`eval_metaworld.py:1451` 注释中的零点
[2.45, 2.27, -1.37, 0] 本就是 skeleton 口径——v5 文件是错配。

**修复**：`logs/stageB_eval_chain.sh` 第 3 步 `--features` 改为
`data/metaworld_fullframe_skeleton.pt`（变量 `$SKELETON` + 注释）。
开环消融（第 2 步）的 --data 仍是 v5（与 ST288 9927 行对齐），
动作目标空间与训练不完全一致 → clean 绝对 MAE 不可直比，
wrong/blank/swap 的 delta（比值）不受影响。

**✅ 修复验证（2026-08-09 14:5x，`stageB_closedloop_sanity_fixed.log`，
6 任务 ×10，修复后 features）**：

| 任务 | 修复后 | 旧最强（mw_fix_full49, 13.9%） |
|---|---|---|
| 0 拿螺母放柱上 | 0/10 | 0/10 |
| 1 投篮 | 0/10 | 0/10 |
| 6 按按钮 | **8/10** | 5/10 |
| 13 关门 | **10/10** | 8/10 |
| 18 关抽屉 | **8/10** | 9/10 |
| 48 关窗 | **9/10** | 9/10 |
| 合计 | **35/60 = 58.3%** | — |

任务 0/1 为全系模型难点（旧模型亦 0/10），非判据。判据任务 6/13/18/48
全部非零且与旧最强持平/更优 → 修复有效，0/10 系评测归一化 bug 而非模型。
完整 49×10 已启动（PID 2551178，`stageB_closedloop.log`，预计 ~3-4h）。

## 论文口径注意
- 闭环 49×10 与 Evo-1 的 10 trials/task 同口径但任务集不同（49 vs MT50 子集），
  对比表须脚注（见 `artifacts/baseline_table_literature.md`）。
- 闭环 95% CI 统一用 `scripts/closedloop_ci.py` 计算（macro ± 1.96·SE 口径，
  Wilson 为保守替代）——既有 v5 direct 的 [22.6, 41.4] 为手工值，
  脚本重算为 [22.2, 41.5]（同口径，±0.4pp 舍入差），论文一律用脚本值。

## L_m 现状（2026-08-09 核查 logs/Lm_*.log，重要）
§5.6 的 `\todo{numbers for A and B40k}` 有原因：**A（va8_20k）与 B40k 的
L_m 全部为 0**（5 对 × 5 blocks，D=0.000 O=0.000，所有 rollout 失败）——
两模型在 LIBERO 模拟器上 0% 成功率（地板效应），L_m 无法测量（"低 D,O →
OOD fragility"读数）。另有第二问题：**matched-state 前提破坏**
（日志 "init-state max diff over 5 matched states: 2.35e-01 / 1.40e-01，
target < 1e-3"）——"同场景"匹配初态实际 diff 0.1-0.24，远大于目标。
→ L_m 修复依赖：修复链模型（heldout 泛化）先跑通闭环成功，再补 L_m；
matched-state 缺陷须在方法节如实说明或改用严格同初态采集。
§5.5 tab:2x2 未完成项：C2 blank sens 为 \todo{}（补自 cosine 日志，
C2 敏感度原日志在 /tmp 清理中丢失，需修复链重测）；cosine 精度统一用
0.9994/0.9984；A 行 +2381% 与 e2e 行敏感度是不同链/不同度量，表格须注明。
§5.8 Efficiency 口径修正：训练设备为单卡 16GB（RTX 3080 Laptop），
论文写 "24GB GPU" 需改；43.5M 是 4 层 VA 旧配置，Stage B（8 层 + 槽 +
flow6）参数数需重算。

## 论文待回填清单（Stage B 数字落定后执行）
- [ ] **§3 Method 结构缺口（2026-08-09 核查发现）**：§3.1 只有 4 组件（frozen
      Qwen / V-JEPA / VA / flow head），**未写 langslot 槽读出模块**（16 coarse +
      K 槽 + 3 relations）与 ST288 时空 token 表示（槽消融 direct288/fixedslot/
      langslot 是论文核心消融，方法节必须先补）；§3.2 需补双阶段协议（Stage 1
      V-JEPA 冻结 → Stage 2 --vision-unfreeze-all lr 1e-6 + sliding-window 全帧
      监督 + π0.5 式帧增强），并注明主线 FM-only（pair=0 是消融臂而非默认）；
      §3.4 部署速率 40.6Hz 为 Stage A direct 头数字，flow 32 步后需重测更新。
- [ ] `paper/ora0_paper.tex` MT50 节（Sec. 5.3.3 附近）：开环 chunk_mae/success、
      语言 vs task-id（wrong/taskid 消融）、闭环 49×10 + CI（closedloop_ci.py 口径）
      —— 当前文本是早期链旧数字（16.3% 等），须整节替换为 Stage B 数字
- [ ] `tab:baselines` 的 "ours" 两行：open-loop 与 closed-loop 数字更新 + 口径脚注
- [ ] LIBERO 节：等修复链（episode 分离重建 → C1/C2 → cosine 判决）数字
      —— 2026-08-09 核查：§5.2 的 tab:trio（blank +2381%/+13751% 等）来自旧特征链
      评估集（v4 泄漏口径），修复链 heldout 数字出来后整节 + 表格替换；
      1-scene/3-scene 双层结构保留（1-scene = 前 4 任务子集可从同一 heldout
      payload 导出）
- [ ] VLA-RL 节：等 MT10 PPO 链数字
- [ ] 每个数字回填时注明日志文件名 + 日期（本文件规则）
