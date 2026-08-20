# LIBERO VA2 链异常诊断（2026-08-08 完成）

## 现象
`checkpoints/libero_va2_40k.pt`（bidir_va 8 层 + paired 契约，40K 步）开环评估：
- none: chunk_mae_norm **0.3597** / success 19.5%
- blank: 0.4001 (+11.2%) / success 3.1%
- swap: 0.3749 (+4.2%) / success 8.7%
- 对照 C1（全冻结适配器）：none 0.0839 / success 56.7%；blank +33.7%；swap +42.6%
→ VA2 开环远差、语言敏感度近失，曾疑与 paired_v4 契约不合格相关。

## 确诊：评估集泄漏 + 零泛化（两个独立问题）
1. **评估泄漏**：`libero_3scene_v4.pt`（360 ep）由 paired_v4 的 140 个训练 episode
   与 220 个新 episode **交错拼接**而成。逐对视觉距离（首决策 64×768 特征 L2）：
   - 偶数 episode 与训练集最近邻距离中位数 **0.0**（78% 完全相同的帧；任务 0-3 均值 0.03-0.05）
   - 奇数 episode 距离 ~40（全部未见）
2. **零泛化**：VA2 评估成败严格按奇偶交替——**未见 episode 180/180 全败（0%）**，
   "19.5% 成功" = 141/180 只发生在记忆的训练 episode 上。VA2 实际从未泛化到新 episode。

## 根因
- 训练数据仅 140 个 episode（每决策点 ~71 遍/40K 步）→ 深度过拟合记忆；
- paired 契约的 4 样本 pair 跨 4 个**不同 episode**（视觉 max diff 2.83，约特征范数 10%）：
  模型无法区分"动作差异来自指令还是来自视觉"，语言通道未被有效利用，
  敏感度（blank +11.2% / swap +4.2%）远低于 C1 且被未见即败掩盖。

## 修复方向（LIBERO 修复链，Stage B 之后执行）
1. **数据重建必须严格 train/val 分离**：v4 重建脚本（scripts/data/prepare_libero_paired.py / v4 链）
   把训练 episode 混入 eval 集是 bug；重建后评估集与训练集 episode 零重叠（验证脚本：
   parity_visdist 方法，见 /tmp/check_episodes.py、/tmp/parity_visdist.py 思路）。
2. VA2 重训走 single-task FM-only 协议（与 Stage B 同构）或修正 pair 契约后再跑；
   若走 pair 路径，pair 必须是同帧严格 fork（见 artifacts/pair_contract_go_no_go.md）。
3. 复评时以未见 episode 的 220/360 为准，且语言消融只在成功 episode 上统计敏感度
   （避免"失败即无差异"的假阴性）。

## 影响面
- 此链数字**不可入论文**（泄漏污染）。
- C1/B40k/C2 链（libero_e2e_*）用的是 e2e 视频数据（libero_video_v2），与 v4 特征
  数据不同源，需另行核验其评估集是否含训练 episode（待 LIBERO 修复链时一并检查）。

## 2026-08-09 核验结果：e2e 链与 LIBERO-100 链同样无 held-out
已读代码确认（resume_queue_v2.sh / next_phase.sh / evaluate.py / scripts/data/prepare_libero_video.py）：
- **C1/C2/B40k e2e 链**：训练与评估都用 `data/libero_video_v2.pt` 全量 360 样本
  （E2EDataset 无 split；next_phase.sh 的 eval_e2e/eval_libero_Lm 全部 --data 同一文件）。
  → 开环 MAE / 三件套 / C_OL 数字全部在训练数据上测得，不能作为泛化证据。
- **LIBERO-100 链**（resume_queue_v2.sh）：`scripts/data/prepare_libero.py --max-tasks 100` 后
  直接 `train.py --data libero_100_full.pt`，评估 `evaluate.py --val-episodes -1`
  （evaluate.py:241-244 语义 = 全部 episode，无 held-out 机制）→ 同样无 held-out。
- **不受影响的部分**：闭环评估（eval_libero_closedloop.py 在 LIBERO 模拟器上 rollout）
  用 benchmark 自带任务定义，与数据文件无关，协议不受泄漏影响。

## 修复工具（2026-08-09 已就绪）
`scripts/split_libero.py`（+ tests/test_split_libero.py，7 测试全绿）：
按 episode（真实 pair 存在时按 pair 组，组不撕裂）做 per-task 分层 seeded 切分，
输出 train/heldout 两个独立文件；metadata["split"] 记录协议；
train ∩ heldout episode 断言零重叠。下游训练/评估脚本零改动，一律用 --data 指向。
`--aligned-heldout`：所有任务留出同一组 episode ordinal → 每任务第 r 行 = 同一原始
episode，scripts/data/prepare_libero_paired.py 的"按行位置配对"在切分输出上保持有效
（2026-08-09 实测：非对齐版本配对 0 组存活，对齐后 29 组存活）。

**重要数据契约发现（2026-08-09）**：`data/libero_3scene.pt`（8月4日基准）的
previous_action 是旧契约（首决策为真实前一动作，pair 间 max diff 2.0），
配对门控 prev_atol=1e-3 必然全灭；`data/libero_3scene_v3.pt`
（`previous_action_contract: v3_prevfix_20260807`，首决策 prev 全零，P0-A 语义）
才是正确基准——paired_v3/v4 即由它产出（35 组/140 行，与 pair_contract 吻合）。
**修复链所有特征侧文件必须从 v3 派生**。

已生成（seed 0, 每任务留 8, aligned-heldout, v3 派生）：
- `data/libero_3scene_train.pt`（264 rows / 264 ep，prev[0]=0，单任务臂输入）
- `data/libero_3scene_heldout.pt`（96 rows / 96 ep，单任务臂评估集）
- `data/libero_3scene_train_paired.pt`（116 rows / 29 组，配对臂输入；
  FeatureDataset 校验：--pair-start-atol 0.2 --pair-start-cosine 0.99 通过）
- `/media/ryan/robot-data/libero_video_v2_train.pt` / `_heldout.pt`（264/96，
  e2e 视频臂输入/评估集，E2EDataset 校验通过）
- 原 VA2 训练命令已从 /tmp/libero_va2_chain.sh 恢复（40k 步 / paired_v4 /
  pair-loss 1.0 / pair-start-atol 0.2 / pair-start-cosine 0.99 /
  memory-split + evidence 16 + task 8 + seq-coupling 2 + future-predict 0.1）

**E2EDataset 阻塞已修（2026-08-09）**：旧版 libero_video/v2 的 pair_id 为平凡
（每组 1 行），E2EDataset 无条件 build_pair_groups 会抛
"needs at least two different instruction_id values"，导致当前代码无法
`--e2e-data` 训练单任务链（C1/C2 当年训练命令已随 /tmp 清理丢失）。
修复：仅当存在真 pair（id>=0 且 >1 行）才构建并严格校验，平凡 pair 走
单任务路径（与 FeatureDataset require_pairs=False 语义一致）；
tests/test_semantic_adapter.py +2 用例（trivial → pairs={}，真 pair → 构建）。

## 待办（2026-08-08 预检登记，LIBERO 修复链时执行）
1. **重建必须显式分离 train/eval episode**：`scripts/data/prepare_libero_paired.py` 目前无
   episode 级 split 参数（v4 泄漏根因）。重建时按 episode 先切 held-out（如每任务
   留 5-8 ep），再跑配对门控；评估集与训练集 episode 零重叠（用 episode_id 断言，
   与 Stage A v4 泄漏检查同款脚本）。
   → **已解决**：split_libero.py（aligned 模式）+ 已生成文件（见上）。
2. C1/B40k/C2 链的评估数据（libero_video_v2 源）需同样核验 episode 重叠。
   → **已核验**：训练=评估=全量 360，无 held-out（见上）；视频侧切分文件已生成。
3. 重建后走既定判决：Qwen cosine（paired 语言表征是否坍缩）+ 敏感度 + 开环 MAE。
   → 待 GPU 空闲后执行（VA2 单任务/配对两臂 + C1/C2 重训，评估一律用 *_heldout.pt）。
   执行脚本已就绪：`logs/libero_fix_chain.sh`（断点保护，5 训练臂 40k：
   VA2 st / VA2 pair / e2e C1(rank0,unfreeze0) / C2(rank32,unfreeze0) /
   B40k(rank32,unfreeze12)，然后 heldout 三件套 + cosine 判决 + C_OL；
   注意 --unfreeze-blocks 语义为 V-JEPA 块数，非 Qwen）。
