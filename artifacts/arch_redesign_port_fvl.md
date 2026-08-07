# ORA0 架构大改设计定稿：Port-FVL（Feature-Space Foveal Verification Loop）

**日期**：2026-08-06/07 | **决策链**：用户需求 → Grok 事实查证（3 轮）→ Codex R1（5 个原创方案）→ Codex R2（定稿 + 实现规格）→ 用户审阅文档逐条核实 → 本文档

## 0. 结论摘要

- **主论文架构 = Port-FVL**（特征空间中央凹验证环）：冻结 Qwen 上两组层（orient 16:20 / verify 20:24）+ 冻结 V-JEPA 的 14×14 原生网格上做可微软路由，形成"定向→观察→验证→更新寄存器→生成动作"的闭环；两个骨干参数 bit-for-bit 不变（参数侵蚀免疫）。
- **必跑等算力对照 RB-EC**（repaired baseline, equal compute）：空间网格 + 64 broad tokens + 16 global tokens + 显式 state K/V + VA-8 + flow-4，无 fovea。FVL 只替换 16 个 global 摘要 token 为 16 个 foveal tokens，其余完全一致。
- **P0 数据 bug（实测确认）**：`data/metaworld_features_v2_full.pt` 的 `pair_id` 9927 个全唯一 → `L_pair` 在 MW 训练中**从未生效**。修复 = 采集真同状态分叉 pair（open↔close 类任务，2000 组）。
- **阶段链**：RB-EC 40k（3 seeds）→ Port-FVL 40k（3 seeds）→ 胜者 100k → PPO（冻结骨干 + 冻结 router）。机制验证门控（≥max(5pp, 2×SD) 才晋升；shuffle/中介测试不通过即 kill，不靠 100k/PPO 翻案）。
- **不新增 IL loss**：严格 `L_FM + L_pair`（用户约束）；未来 latent 预测等不做 loss，用事后干预审计替代。

## 1. 用户审阅文档逐条核实（Grok/代码实测）

| # | 审阅主张 | 核实结果 |
|---|---|---|
| 1 | L_pair 在随机 τ 下泄漏目标差 | **对当前代码不成立**：train.py:361 `sample_pair_intervention` 已是 τ=0 共享噪声；但仅监督首状态，可扩展共享中点 probe（τ∈[0,0.5]，不新增 loss） |
| 2 | pair 必须同决策状态分叉 | **成立且发现真 bug**：MW pair_id 全唯一 → L_pair 空转（实测 9927/9927 unique）→ 需真 fork pairs |
| 3 | M_t 是语言/动作污染的工作区（确认偏差） | 部分采纳：Port-FVL 保留逐层 VisualMemory + 新增 4 个语义寄存器（门控更新，episode 边界重置）；不做完整双记忆拆分（保持简单） |
| 4 | 同步双向 co-attention 非顺序修正 | 部分采纳：VA-8 保持（SMC-Attn 已有），改进放在 flow-4 + state K/V + fovea 环路；顺序 A→V/T→A 作为可选第二阶段 |
| 5 | Flow 头入口相加条件稀释 | **采纳**：flow 深度 2→4，8 Euler 步为主（32 步为对照），条件每层注入 |
| 6 | 64 token 平坦池化丢几何 | **采纳**：V-JEPA 原生 14×14 网格（196 tokens/frame），fovea 在其上做受限可微注意力 |
| 7 | L_future 未来 latent 预测 | **不采纳为 loss**（用户约束 + V-JEPA 2-AC/JEPA-VLA/VLA-JEPA 已做）；替代：干预审计（mask action→vision 通路） |
| 8 | CofactVLA 撞车 | **Grok 查实存在**（2608.04396，2026-08-05）：LIBERO 98.5%、OPG 正交投影 + CCR 协方差去混淆、无 MW、非共享噪声监督 → 论文必须引用并区分机制 |
| 9 | Qwen 4B/2B 口径 | 仓库实际 **Qwen3.5-2B**（审阅误写 4B） |
| 10 | V-JEPA 参数量 | **80M**（ViT-B/16@384，Grok 查实），论文修正 |

## 2. Port-FVL 规格（Codex R2 定稿）

### Qwen 执行
- Qwen3.5-2B：24 层 [3 linear-attn + 1 full-attn]×6；前缀缓存 0:16，orient 组 16:20，verify 组 20:24
- 每个语义 tick：orient 12 端口（4 每帧 broad + 1 proprio + 1 prev-action + 4 寄存器 + 2 locus 查询；有效序列 25 tokens），verify 20 端口（16 foveal = 2 loci × 2 scales × 4 frames + 4 readout；序列 45 tokens）
- 端口 = 学习投影进 Qwen 2048 维残差空间，RMS 匹配指令边界；训练时上组 Qwen **不包 no_grad**（参数零梯度但 Jacobian 传导 → "参数侵蚀免疫，非梯度路径免疫"）
- 若 top-8 延迟不达标 → 预声明 top-6（18:24），序列上限 40 tokens

### 路由（无辅助 loss 的可训练 fovea）
- 双 locus + 可微 soft-NMS；温度 τc 1.0→0.25（0-10k），固定 τa=0.5；固定半径 narrow 1.25 / broad 4.0（patch 单位）
- 时域偏移 δ=2tanh(...)，当前帧偏移固定 0；仅允许 10% token mask + ±1 patch 抖动增强；无 entropy/diversity loss
- 语言唯一进入 flow 的通路 = 4 个 active Qwen readout（防 bypass）

### 参数预算
| 组件 | 可训练 |
|---|---|
| VA-8 + flow-2（现 AQC） | 81.59M |
| +2 flow 层 | +6.30M |
| state K/V × 8 层 | +4.21M |
| router/ports/registers | +3.10M |
| **合计 ≈95.2M**（RB-EC 参数匹配 ±0.1M） | |

### 数据重建（数据盘，约 36-40 GiB）
- `mw_grid_v3/vision_grid.fp16`：[32223, 4, 196, 768]（strided-6 决策 bank，36.14 GiB，提取 ~1.5-3h）
- `mw_grid_v3/sequences.pt`：24759 条 4 决策序列（索引 + state/prev/actions/ids）
- `mw_grid_v3/vision_broad.fp16`：[32223, 64, 768]（从 bank 派生）
- `mw_forks_v3.pt`：2000 组真同状态 fork（door/drawer/window/faucet open↔close，每组建两个专家 chunk）
- LIBERO：`libero_video_v3_s1.pt`（stride-1 重建，~4.8GB）

### 训练与门控
- AdamW：VA/flow 1e-4，ports/router 2e-4，Qwen/V-JEPA LR=0；warmup 2k，40k 恒定，40k-100k cosine 至 10%；paired effective batch 4
- 时间：10k 筛选 45-75min；40k ×3 seeds 3-4.5h/seed；100k 7.5-10h
- 门控链：工程（缓存/梯度/延迟）→ smoke-6（peg-insert/hand-insert/sweep-into/stick-push/button-press/door-close，20 trials × 10k）→ MT10（3 seeds × 20 trials，≥max(5pp,2×SD) + bootstrap CI>0）→ 机制（dense/fixed/Qwen-bypass + address/content shuffle，各降 ≥max(5pp,2×SD) 且吃掉 ≥50% FVL 增益）→ MT49
- 延迟验收：端到端 p95 ≤60ms、p99 ≤70ms；Qwen 上组 p95 ≤15ms；fovea 增量 p95 ≤2ms（当前环境无 fused linear-attn kernel，须实测）

### 因果证据（语言服从性全套升级）
- 三件套 + C_OL + L_m + command-fork 全部保留
- **新中介测试**：a00=A(I,R(I))、a11=A(I',R(I'))、a10=A(I',R(I)) → route-mediated = a11−a10 须解释 ≥30% 的 matched-noise C_OL 位移（bootstrap CI 排除 0）
- 地址 shuffle / 内容 shuffle（RMS 匹配 donor 内容）

## 3. 论文定位段（Codex R2，§11 原文可用）

> 现有 VLA 要么把原生视觉/状态 token 放进单块 VLM–动作栈（π0、SmolVLA），要么把固定 final/intermediate VLM 上下文暴露给动作专家（Evo-1、GR00T N1、FabriVLA）；指令条件化视觉查询已出现在 Action QFormer，编码后互惠融合出现在 TurboVLA，双速率/异步执行由 GR00T N1 与 SmolVLA 确立——这些单独都不是我们的新颖点。我们提出 **Port-FVL**：特征空间中央凹验证环，学习出的视觉/状态残差端口激活两个冻结的 Qwen 上组层去定向、检查 V-JEPA 局部多尺度区域、更新持久验证寄存器、并条件化 flow 策略，全程不改变两个骨干的参数。等算力 dense/fixed 对照、消息 shuffle、指令→路由→动作中介测试使"闭环验证机制"因果可证伪。

## 4. 文件级实现清单（Codex R2 §7）

- `va_compound/backbones.py`：Qwen 前缀不可变缓存 + `forward_group(16:20/20:24)`（保输入梯度）；`VJEPA21Backbone.forward_grid()` → [B,4,196,768]
- `va_compound/model.py`：state K/V 源、语义寄存器、active-readout 条件化、flow 深度配置、干预钩子
- `va_compound/fovea.py`（新）：GridPortEncoder / SoftFoveaRouter / PortRegister / dense|fixed|fvl 模式
- `train.py`：mmap 索引数据集、fork-pair 加载器、对照模式、温度调度、优化器分组、严格 L=L_FM+L_pair
- `prepare_metaworld.py` / `end_to_end.py`：分片 grid bank 写入、真 pair 采集、LIBERO stride-1、checksum

## 5. 待执行队列（GPU 严格串行）

1. LIBERO 修复链收尾（L_m B40k + C_OL A，运行中）→ 台账/论文回填
2. VLA-RL MT10 正式链（现有 checkpoint，独立交付物，~2-3h）
3. Qwen group 延迟微基准（决定 top-8/top-6）→ RB-EC 数据重建（grid bank + forks，~2-4h）
4. RB-EC 10k 筛选 → 40k ×3 → Port-FVL 40k ×3 → 机制验证 → 100k → PPO
5. LIBERO-100 链（与 3-4 并行排队）
6. 论文定稿 + 图表包回填
