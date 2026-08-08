# pair 契约生死门设计（Codex 评审结论 + fork 数据可行性初评）

日期：2026-08-08。触发：用户问消融顺序是否应把 pair 提前；Codex Q4 评审 + 本地数据核查。

## Codex 结论（Q4）：pair 必须前移为 P0 go/no-go

1. **顺序**：40K FM-only 主线完成后，pair 生死门是第一个 P0 实验（先于槽消融/门控 LoRA/C² PPO）。
   pair 决定论文核心叙事能否成立，证据门性质，不是普通组件消融。
2. **推荐对照设计（A/C/D/E）**：
   | 组 | 数据 | λ_pair | 归因 |
   |---|---|---|---|
   | A | 原始数据 | 0 | 原始 FM 基线 |
   | C | 原始 + fork | 0 | fork 数据本身的收益 |
   | D | 与 C 相同 | 1 | L_pair 增量（核心） |
   | E | 相同边际分布、配对关系打乱 | 1 | 排除普通正则化效应 |
   - D 在 held-out fork 上稳定优于 C 和 E、且标准任务成功率不实质下降 → 契约成立。
3. **平台选择**：LIBERO 现有近似 pair（跨 episode、视觉 diff 2.6）只能当附录/探索性证据；
   必须用**严格 fork**（所有策略可见输入一致，仅语言不同）。MW 先做低成本生死门，
   LIBERO 严格 fork 作论文级确认。
4. **pair 为负时的论文措辞**：未验证 ≠ 负结果。FM-only + 语言敏感度只够支撑
   "language-conditioned policy / language responsiveness"，不够支撑 "grounding"；
   若 λ=1 无增益，可写 "显式 pair loss 并非必要（严格配对评测揭示 FM 已具任务相关语言依赖）"。
5. **C² PPO 不提前**：RL 改变语言敏感度是必须做前后对照的理由。顺序：
   pre-RL pair gate → C² PPO → 同一 strict-fork 评测；必要时补 λ×PPO 2×2。

## fork 数据可行性初评（本地核查 2026-08-08）

MT50 49 任务中存在 ≥8 对天然同场景任务（共享环境 XML、仅目标相反）：

| A | B | 场景 |
|---|---|---|
| 18 Push and close a drawer | 19 Open a drawer | 抽屉 |
| 14 Lock door (clockwise) | 16 Unlock door (counter-clockwise) | 门锁 |
| 20 Faucet counter-clockwise | 21 Faucet clockwise | 水龙头 |
| 23 Handle down sideways | 25 Handle up sideways | 把手 |
| 24 Handle down | 26 Handle up | 把手 |
| 31 Plate into cabinet | 33 Plate from cabinet | 柜子 |
| 32 Plate into cabinet sideways | 34 Plate from cabinet sideways | 柜子 |
| 38 Stick push box | 39 Stick pull box | 箱子 |
| 47 Window open | 48 Window close | 窗户 |
| 0 Nut onto peg | 12 Nut out of peg | 螺母+钉子 |
| 13 Door close (revolving) | 15 Door open (revolving) | 门 |

**关键事实**：pair 契约只要求首观测一致（代码 `sample_pair_intervention` 只用
`actions[:, 0]`；`_validate_pair_contract` 校验的是 first obs/proprio/prev_action +
首动作差异）。因此严格 fork 只需在**种子对齐的同一初始状态**下取两个任务的
t=0 观测 + 各自专家首动作即可构造（t>0 状态随专家分叉，不要求一致）。

## 严格 fork 构造已验证可行（metaworld 3.0.0 实测，抽屉对）

**机制**（读源码 + 实测确认）：
- drawer-close-v3 与 drawer-open-v3 **共用同一 XML**（sawyer_xyz/sawyer_drawer.xml），
  模型结构完全一致；
- 同 seed reset：qpos 9/10 维相同，仅 drawer 关节不同——close 任务
  `_set_obj_xyz(-0.15)`（抽屉拉出），open 任务保持 0.0（抽屉关闭）；
- 抽屉基准位 `model.body("drawer").pos` 来自 `_get_state_rand_vec()`（同 seed 应同值，
  实测两 env 差 0.0437，疑随机空间范围不同，需核对）；
- goal site 位置不同（+0.2）但 **corner2 视角不可见**（own-reset 像素差异全在抽屉带）；
- 对齐操作（qpos + body_pos + site_pos + geom_pos 直接赋值，不调 mj_forward）后
  渲染 **meandiff 0.69**，残差仅抽屉带（渲染器运动学滞后）。

**实测踩坑记录**（fork 采集脚本必须遵守）：
1. **metaworld `set_env_state` 不对称**：`get_env_state` 返回 (qpos, qvel) 但
   `set_env_state` 按 (mocap_pos, mocap_quat) 解包——不能互配，直接用
   `env._set_obj_xyz(qpos[9])` 或 `data.qpos` 赋值；
2. **手动 `mj_forward`/`do_simulation` 会破坏渲染**（全图变暗 meandiff ~45）——
   只做 qpos/model 字段赋值，渲染器自身会刷新（或连续渲染两次）；
3. 现有 lerobot parquet 数据**未按任务对种子对齐**（首帧 thumb 最佳配对 0.47/255），
   严格 fork 必须用 metaworld 环境重采，不能从现有 parquet 拼。

**专家动作来源（待定项）**：fork 需要同一状态下两个任务的专家动作。
候选：① 用现有 50 demos/任务训两个小 BC 专家（简单可靠，推荐）；
② metaworld task demo 数据（需查 ML1.train_tasks[0].demo 是否存在）；
③ 从 parquet 找状态重合帧（已证不可行——无精确重合）。

**fork 采集流程（下一阶段实现）**：
1. 对每个同场景任务对 (A, B)：同 seed 构建两 env，重置后强制对齐
   （body_pos/site_pos/geom_pos + drawer 关节值取 A 的）；
2. 渲染验证像素一致（阈值：maxdiff 或特征余弦达标）后采样 t=0 决策点；
3. 用 BC 专家各出动作 → 组成 (obs, lang_A, act_A) / (obs, lang_B, act_B)；
4. 重复多 seed/多抽屉位置（0.0 / -0.15 / 中间值），构造 pair 数据集。

**待验证**（下阶段）：① 其余任务对（水龙头/窗/门）是否同 XML 同机制；
② 两 env 随机空间差异（body_pos 0.0437 来源）；③ BC 专家动作差异是否超
专家自身噪声（min_pair_action_delta 校验）。

## 执行状态
- [x] Codex Q4 评审（本文件）
- [x] metaworld 严格 fork 构造可行性实测（抽屉对，机制+踩坑已记录）
- [ ] 其余任务对验证 + fork 采集脚本（`scripts/collect_mw_forks.py`）
- [ ] A/C/D/E 对照脚本设计（train.py --pair-loss-weight 已支持 λ=0/1；E 组需打乱配对）

## 其余任务对验证（2026-08-08 补测，无渲染纯状态比对）

| 任务对 | 同 XML | 同种子 qpos 差 | 结论 |
|---|---|---|---|
| drawer-close/open | ✓ | 仅 joint 9（-0.15 vs 0） | 已验证（含渲染像素对齐） |
| **faucet-close/open** | ✓ | **0.0（全等）** | 最佳 fork 候选：任务差异纯在 reward/goal；body_pos 差 0.07 / site_pos 0.05，align 拷贝即可 |
| window-close/open | ✓ | 仅 joint 9（0.2） | 同抽屉机制 |
| door-close/open | ✓ | joint 0-6（1.57，手臂随机化不同）+ 9 | align 全量 qpos 拷贝可对齐 |
| peg-insert-side / unplug-side | **✗（不同 XML，nq 16 vs 10）** | — | **从登记剔除**，不是同场景对 |

collect_mw_forks.py 的 PAIRS 登记已按上表更新（peg 剔除、door 加入、faucet 免关节对齐）。

## A/C/D/E 实现评审（Codex Q5b，2026-08-08，二次评审完成）

**结论：方案有条件通过，但有三处必须修正，否则 D−C 无法归因 pair loss。**

1. **C 必须与 D 同构**：C 不能是"合并单 loader"。C/D/E 三组必须都用双 loader
   （v5 FM 批 + fork 批，同 k、同批组成、同 LR 进度），仅 `--pair-loss-weight`
   不同（C=0，D=1，E=1+打乱配对）→ D−C 才纯粹归因 pair loss。
   fork 批在 C 里也是 flow-only（无 pair 项）。
2. **k 取值**：按"每 v5 epoch 约遍历 fork 一次"对齐自然暴露：
   k = (9927/B_v)/(N_f/4)。留 12 对后 N_f=120 → k≈83（batch-mean 口径下）
   （我原拟 k=16 会把 fork 更新权重放大 4.1×，弃用）。
3. **E 组构造**：先切分 held-out；每真 pair 按指令分支稳定分 L/R；固定 seed
   置换 R 侧；拒绝与原 pair_id/instruction_id 相同的组合；新 pair_id 跨 epoch
   冻结；保存映射；断言全覆盖（每组 2 行、无真配偶、指令不同）。E 的解释
   限于"错误配对压力测试"（非纯语义对照）。
4. **统计功效（诚实报告）**：held-out 12 对 → 有效 n=12（非 24），配对 t 检验
   MDE≈0.89 SD（3 次比较校正后 1.08 SD）——只能检出大效应；二元成功率用
   exact McNemar。不显著≠无效；单 seed = 单 checkpoint 结论。
   缓解：多任务对（drawer+faucet+window）扩大 held-out 对数量。
5. **契约断言**：fork 批内 proprio/prev 必须 allclose 相等（D 用；E 不满足，
   解释受限）；pair 损失按有效 pair 数归一（semantic_pair_loss 已是 2 对均值，
   无需改）；训练日志记录 v5 批与 fork 批的梯度范数分离。
6. **架构选择**：pair loss 是 flow 专属（direct head 模式跳过）→ 生死门用
   flow head（非 --direct-head），flat-64 特征，与 VA2 同源。A 组也同架构。

**实现要点（train.py，pair 阶段执行）**：
- 新参数：`--fork-data <pt>`（真配对或 E 打乱版）+ `--fork-k`（默认 83）；
- 双 loader：v5 single-task loader + fork PairedBatchSampler loader，交替
  step % (k+1) == 0 取 fork 批；fork 批走 pair 计算路径（复用现有
  paired_partner_indices/sample_pair_intervention/semantic_pair_loss）；
- fork 契约断言：vision/proprio/prev 逐对 allclose；
- E 数据由 assemble_fork_dataset.py --shuffled 生成（已实现）。
