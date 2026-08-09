# MW 多 start 重建设计（SOTA 追赶链前置，2026-08-09 预登记）

> 触发条件（Stage B 闭环数字出来后）：若粗任务保持高分而精任务（nut-peg 等）仍为 0
> → "数据覆盖是瓶颈"假设成立 → 多 start 重建；若精任务非零 → 数据覆盖假设已证伪，
> 无需此链（改走结构改进 SMC/SAM/LVK 或 VLA-RL）。
> 用户判据原文：粗任务保持高分 + 精任务从 0 变非零 = 数据覆盖假设验证。

## 问题
v5 数据 = lerobot/metaworld_mt50 固定 50 ep/任务（每任务 ~50 个初始布局）。
闭环失败模式常源于训练未覆盖的初始状态/布局（尤其精任务），SOTA 口径
（Evo-1 等）用更广 seed 覆盖。

## 设计（保守、可复用现有管线）
1. **专家来源**：MetaWorld 官方预训练 RL 策略（metaworld 环境自带 MT50 全部
   任务的 learned policies，`env.set_task` 后按官方 API 加载）——与
   prepare_mw_recovery.py 的"本地脚本专家"并列，覆盖 49 任务最全。
   备选：本项目已训模型（自我收集，DAgger 风格）——仅在官方策略不可用时用。
2. **采集**：每任务 50 → 150 ep；`env.reset(seed=s)` 覆盖 s ∈ 大种子集
   （metaworld 随机化物体初始布局），corner2 相机 80FPS 存帧 + state/action
   （executed-clip 同 v5 标签语义），success-only 过滤。
   复用 collect_mw_forks.py 的 EGL 离屏渲染 + 帧解码模式。
3. **特征化**：与 v5 同域——原始预训练 V-JEPA flat-64（离线提取，GPU 活，
   ~1h/50ep×49 任务量级）；或若 Stage B 链继续用 live 协议则存 parquet 帧
   + fullframe skeleton（与当前训练同构，代价是训练时在线编码）。
4. **重训**：直接 resume 或从 stage2 checkpoint 续训（数据量翻 3 倍，
   建议 60k 步或按 loss 收敛），→ 重闭环 49×10。
5. **重测口径不变**：32 步 Euler、macro CI、语言消融四件套同链。

## 成本预估（16GB 卡）
- 采集：CPU/EGL 并行 ~2-3h（100 ep×49 任务，每 ep ~30s）
- 特征化：~2-3h（GPU）
- 重训 60k：~8-10h
- 重闭环：~8h
→ 整链 ~1.5-2 天 GPU 串行，排在 fork 链（pair 消融）之后还是之前由数字决定：
  若闭环已显著优于 Stage A（>50%），fork 链优先（pair 主张是论文核心）；
  若仍 <40%，多 start 优先（数据杠杆更大）。

## 待办
- [x] 确认 metaworld 3.0.0 官方 RL 策略 API：`metaworld.policies.ENV_POLICY_MAP`
      + Sawyer*V3Policy（全部 MT50 任务）本地可用 ✓（2026-08-09 实测）
- [ ] 闭环数字后按触发条件决定是否执行
