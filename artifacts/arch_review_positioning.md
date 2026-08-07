# 架构审计与学术定位备忘（2026-08-05）

来源：Codex 架构审计（gpt-5.6-sol，/tmp/codex-arch-review.md）+ Grok 定位 + 4 篇竞品原文精读（papers/remem_2607_12942 / memoryvla_2607_19236 / cogvla_2607_21046 / rbvla_2607_20659）。

## 1. 竞品图景（全部原文验证，arXiv ID 真实）

| 竞品 | 记忆机制 | 关键数字 | 与我们的重叠 |
|---|---|---|---|
| ReMem-VLA (2603.12942) | 双层 EMA 循环 query（帧级+块级）+ 过去图像重建 loss；BPTT truncation=1 + 固定无梯度循环路径 | MemoryBench 94.5% vs MemoryVLA 1.5% | 最接近：循环记忆 + 双向 connector（12 层） |
| MemoryVLA (2508.19236, ICLR'26) | 感知-认知记忆银行 + 检索 | LIBERO-5 96.5%、Bridge 71.9%、真机 84.0% | 记忆方向（检索式，不同路） |
| AVA-VLA (2511.18960) | recurrent belief state + 主动视觉注意力，T=4 BPTT 训练 | LIBERO/CALVIN SOTA | 训练方式与我们的 T=4 展开相同 |
| RB-VLA (2602.20659) | belief 状态 + world-model 自监督目标 | 比 π0 +52.5/+37.5pp，延迟 -5× | 固定大小记忆、无随历史增长 |
| CogVLA (2508.21046, NeurIPS'25) | V-L-A Coupled Attention（causal VL + 双向动作并行解码） | LIBERO 97.4%、真机 70.0%，训练 -2.5× 延迟 -2.8× | 双向 V↔A 耦合（我们的核心主张） |
| SmolVLA (2506.01844) | 轻量 flow，无记忆 | — | 数据同源（我们 MW 数据集与其相同） |
| LIBERO-Plus (2510.13626) | — | — | Blank 干扰协议的标准来源 |

## 2. Taste 判断

- **问题意识正确**（grounding + 固定内存 + 轻量），但架构四件套在 2026 每个单点都有已发表工作（双向耦合=CogVLA；递归记忆=ReMem/AVA/RB；轻量 flow=SmolVLA）。
- **最独特/最有 taste 的贡献 = 语言 grounding 的架构内因果证据**：LIBERO Blank +2381%/Swap +607% vs MW +0.1% 对比揭示"语言依赖是数据结构的函数，不是架构的"。7 篇竞品无一篇做此类机制分析。
- 最大风险：ReMem-VLA 消融证明"可学习循环 + 截断 BPTT 记忆失效"——我们的 M 更新正是可学习路径 + T=4。反证材料：M→A 阻断 +2.9% 说明短时程有效；我们只 claim 短时依赖。
- 故事线建议：收成 "Constant-memory recursive visual coupling for language-grounded control"，主打机制证据 + 两级损失简单性（ReMem 有 image loss、RB-VLA 有 world-model 目标，只有我们无额外 loss）。

## 3. Codex 结构弱点清单（优先级）

### P0-1 previous_action 闭环捷径
- 风险路径：错误动作 → previous_action → A→V → M → 下一动作（误差自激放大）
- 清零 +3792% 是 OOD 干预，不能单独证明过度依赖；但持久性基线强（模型仅好 10-30%）
- 最小改动：拆分 proprio/prev_action 投影 + 门控 + 训练 whole-vector dropout(10-30%) + availability bit；clean MAE 可略降，闭环必须升

### P0/P1-2 记忆是"单槽递归"而非"一步"
- M 理论上递归压缩全部历史；问题：无门控覆盖、T=4 只训练 3 次传递、M 混入动作/语言痕迹
- 方案（按序）：随机 burn-in(0-8 步) → 随机 T∈{2,4,8} → 两槽 FIFO+age embedding 对照 → 延迟线索测试（>4 帧遮挡）
- EMA 不建议第一版（逐 token EMA 会把不同实体平均）
- 无改善则收窄主张为"单步动作-视觉耦合状态"

### P1-3 共享 softmax 模态尺度竞争
- 语言 0.2% attention 不能说明语言弱（Blank 证明低 mass 高因果）；V/M/A/L 分布不同 + token 数悬殊
- 方案：每模态独立 pre-LN/RMSNorm（≠已失败的 QK-norm）+ 记录 ||Σα_sU_s|| 来源贡献
- ✅ **实现核对（2026-08-05）**：四流独立 pre-LN 已存在（model.py L129-132：norm_v_attn/norm_m_attn/norm_a_attn/norm_l，均在各自 Q/K/U 投影前应用）——Codex 建议的最小改动已满足；剩余可选改进仅为 per-source logit bias/value gate，低优先
- 已验证 FFN 有第二 residual ✓

### P2-4 Flow 接口
- 8→32 步改善是积分误差不是容量证据；条件仅入口注入会被冲淡
- 方案：每 block 重注入 + horizon offset embedding + 2→4 层对照；不要直接 8 层（32 步采样重复支付）

### P3-5 语言 K/V-only 合理（已验证）
- 可追加 1-4 个 gated pooled 指令摘要 token（不替换原 token）；低优先

## 4. 证据解读纠正（Codex 指出）

1. "M 只有一步历史"不准确——递归压缩全部历史，问题是保留稳定性
2. M→V +0.4% 不能证明记忆无用，但长期递归路径未被证明（4 帧窗口可能已覆盖所需动态）
3. 语言 0.2% attention 不能支持"语言投影有问题"
4. +3792% 是 OOD 干预；结合持久性基线才构成 P0 风险
5. "8 层深度有价值"未干净隔离（8 层@20k vs 4 层@10k 混淆深度与步数）——需 4 层@20k 对照
6. MW 旧 checkpoint 归一化错误不能决定结构；LIBERO 语言结果来自 FM（pair loss 恒 0）

## 5. 论文动作清单

- Related work 必引：ReMem-VLA、MemoryVLA、AVA-VLA、RB-VLA、CogVLA、SmolVLA、LIBERO-Plus
- 正面回应 ReMem-VLA TBPTT 批评（§记忆局限节）
- 闭环数字出来后决定结构实验：闭环好 → previous_action dropout 对照；闭环差 → 记忆 burn-in/两槽
- 统计口径已定：32 步、宏平均、95% CI、持久性基线、固定种子

## 6. 决定：结构实验全部等正式数据（B40k 终评 + MW 闭环）出来后再动
