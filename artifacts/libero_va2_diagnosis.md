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
1. **数据重建必须严格 train/val 分离**：v4 重建脚本（prepare_libero_paired.py / v4 链）
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

## 待办（2026-08-08 预检登记，LIBERO 修复链时执行）
1. **重建必须显式分离 train/eval episode**：`prepare_libero_paired.py` 目前无
   episode 级 split 参数（v4 泄漏根因）。重建时按 episode 先切 held-out（如每任务
   留 5-8 ep），再跑配对门控；评估集与训练集 episode 零重叠（用 episode_id 断言，
   与 Stage A v4 泄漏检查同款脚本）。
2. C1/B40k/C2 链的评估数据（libero_video_v2 源）需同样核验 episode 重叠。
3. 重建后走既定判决：Qwen cosine（paired 语言表征是否坍缩）+ 敏感度 + 开环 MAE。
