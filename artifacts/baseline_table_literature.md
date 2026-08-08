# 基线对照表（文献数字，Grok 查证 2026-08-08）

> 状态：数字已逐条经 Grok Web 查证，来源 arXiv 号 + 出处位置在案。
> 用途：论文对比表素材。所有数字均为**文献原文/官方源**，本项目不自行训练基线（边界）。
> 口径差异一律脚注，不与本项目数字直接硬比。

## 汇总表

| 模型 | Params | 机器人预训练 | LIBERO Avg | MetaWorld | 评估 trials | 闭环 | 来源 |
|---|---|---:|---:|---:|---:|---|---|
| Evo-1 | 0.77B | 无 | **94.8%** | **80.6%** | 10 trials/task × 5 runs | 闭环+chunk | arXiv:2511.04555 Table 1 |
| SmolVLA 2.25B | 2.25B | 无 | **88.75%** | **68.24%** | 10 trials/task | 闭环每步 replan | arXiv:2506.01844 Table 2 |
| SmolVLA 0.45B | 0.45B | 无 | **87.3%** | **57.3%** | 10 trials/task | 同上 | arXiv:2506.01844 Table 2 |
| π0.5 (OpenPI FT) | ~3.4B | 有 | **96.9%** | —（原文未报） | 50 trials/task | 闭环 replan 5 | OpenPI README / TurboVLA Table 1 转引（π0.5 方法文 arXiv:2504.16054 主表无四 suite） |
| TurboVLA | 0.2B | 无 | **97.7%** | — | 50 trials/task（2000 total） | 闭环 12-step chunk | arXiv:2607.27205 Table 1 |
| π0 (FT) | 3.4B | 有 | **94.2%** | —（原文未报） | — | 闭环 | OFT arXiv:2502.19645 Table I（非 π0 原文；π0 = arXiv:2410.24164） |
| OpenVLA (FT) | 7B | 有 | **76.5±0.6** | —（原文未报） | 50 ep×10 tasks×3 seeds | 闭环 | arXiv:2406.09246 v2 App. E |

## LIBERO 子集明细

| 模型 | Spatial | Object | Goal | Long | Avg | 出处 |
|---|---:|---:|---:|---:|---:|---|
| Evo-1 | 92.7 | 97.7 | 96.3 | 92.3 | 94.8 | arXiv:2511.04555 T1 |
| SmolVLA 2.25B | 93.0 | 94.0 | 91.0 | 77.0 | 88.75 | arXiv:2506.01844 T2 |
| SmolVLA 0.45B | 90 | 96 | 92 | 71 | 87.3 | 同上 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | TurboVLA T1 转引 |
| TurboVLA | 99.2 | 99.8 | 97.4 | 94.2 | 97.7 | arXiv:2607.27205 T1 |
| π0 (OFT) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | arXiv:2502.19645 T1 |
| OpenVLA (FT) | 84.7±0.9 | 88.4±0.8 | 79.2±1.0 | 53.7±1.3 | 76.5±0.6 | arXiv:2406.09246 v2 App. E |

## MetaWorld 明细（Evo-1 难度分层）

| 模型 | Easy | Med | Hard | VeryHard | Avg | 出处 |
|---|---:|---:|---:|---:|---:|---|
| Evo-1 | 89.2 | 76.8 | 77.2 | 79.2 | 80.6 | arXiv:2511.04555 T1 |
| SmolVLA 2.25B | 87.14 | 51.82 | 70 | 64 | 68.24 | arXiv:2506.01844 T2 |
| SmolVLA 0.45B | 82.5 | 41.8 | 45.0 | 60.0 | 57.3 | 同上 |

## 口径脚注（论文必须写明）

1. **"10/50 trials" 含义**：Evo-1 的 10 = 每任务评估 10 trials；50 = MetaWorld 每任务 **50 个训练 demos**（不是 50 评估）。SmolVLA 同为 10 trials/task。π0.5/TurboVLA/OpenVLA 为 50 trials/task（VLA-Adapter 协议）。**trials 数不一致，CI 与方差不可比，必须脚注。**
2. Evo-1 LIBERO 94.8% = 标准 LIBERO 四子集（Spatial/Object/Goal/Long）40 任务的 5-run 平均；成功判据论文未展开（suite 默认 task completion）。闭合为闭环 + action chunking。
3. π0 的 94.2% 与 π0.5 的 96.9% 均为**二次源**（OFT Table I / TurboVLA Table 1 转引），π0 原论文（arXiv:2410.24164）不报标准 LIBERO 四 suite；π0.5 方法文（arXiv:2504.16054）主表也没有。论文中必须标注 secondary source。
4. OpenVLA 官方 LIBERO 为 50 trials × 3 seeds 的 suite-specific LoRA 微调，且输入为 img+lang only（无 wrist/state）；π0 输入含 img+wrist+proprio+lang —— 输入通道不同，直接对比有偏。
5. π0 / OpenVLA / π0.5 / TurboVLA **原论文均未报 MetaWorld**；MetaWorld 列填 "—"。TinyVLA 等在 MetaWorld 报的是自己的数字（非官方基线）。
6. 本项目 MetaWorld 协议为 49 任务 × 10 episodes 闭环（与 Evo-1 的 MT50 50 demos/10 trials 同为闭环但任务集/评估 episodes 数不同），对比时须与上表口径区分，必要时补 10/50 trials 对齐重测。

## 待办

- [ ] 论文定稿时按 LaTeX 三线表 + 脚注再排版
- [ ] 若审稿要求同协议复验：基线权重（smolvla/evo1）已下载留作可选复验（边界外，仅当用户批准）
