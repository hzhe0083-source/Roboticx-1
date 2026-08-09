# VLA-RL 设计对照（πRL 惯例查证，Grok 2026-08-09）

> 用途：VLA-RL 核心实验（完成判定 #5）设计依据；论文相关工作中 πRL/π0.5 引用
> 必须用下面纠正后的 arXiv 号。数字来源：πRL arXiv:2510.25889（Table 1/13 + RLinf
> 文档）、π0 arXiv:2410.24164、π0.5 arXiv:2504.16054、RL4VLA arXiv:2505.19789、
> SimpleVLA-RL arXiv:2509.09674、RECAP/π\*0.6 arXiv:2511.14759（非 PPO）、GR00T N1 arXiv:2503.14734（πRL 附录在其上
应用 PPO Flow-SDE）。

## 关键纠正（论文引用务必注意）
- **arXiv 2504.16054 = π0.5**（open-world VLA 论文），不是 PPO/RL 框架
- **开源 πRL PPO 工作 = arXiv 2510.25889**；π0 = 2410.24164
- Physical Intelligence 实机 RL 线 = RECAP/π\*0.6（2511.14759，advantage-conditioned，
  非 PPO）
- **π0/π0.5 原文不报 MetaWorld PPO**——MT50 数字全部出自 πRL

## 文献惯例（πRL 默认配方）
1. **冻结/训练**：RL 阶段冻结 VLM，只训 ~300M action expert + critic；
   VLM LoRA(r=32) 消融无明显收益，冻结更稳（论文原话"RL contributes more
   significantly to action generalization"）
2. **无 IL KL 项**：默认不惩罚离 SFT 策略；用标准 PPO clipped surrogate（ε=0.2），
   KL 仅作诊断（π0.5 LIBERO-Long 上 KL 升高用 cosine LR 退火稳定）；
   可选熵奖励 0.005（探索熵，非 IL KL）
3. **归一化**：state/action 用分位数 q01-q99（非 min-max）；RGB 不做 mean-std 归一
   （与 SFT 同预处理）；连续 flow 不用 256-bin 动作 token
4. **奖励**：MT50 上稀疏二值（success→1，否则 0）；chunk reward = chunk 内逐步
   reward 求和；无稠密几何 shaping（RL4VLA 用 0.1 grasp 中间阶段属轻 shaping 例外）
5. **规模**：MT50 多任务 450 outer epochs × batch 2048 chunk（+4 PPO epochs/批），
   并行 env 256+；单任务 ~0.1-1M env steps 起步；MT10 ~10^6
6. **超参（πRL Table 13 MT50）**：γ 0.99 / λ 0.95 / ε 0.2 / PPO epochs 4 /
   actor LR 1e-5（π0）/ 5e-6（π0.5）/ critic LR 1e-4 / chunk H=8 /
   RL 期间 denoise steps=4 / Flow-SDE noise 0.5

## 文献 IL→RL 提升（论文对比素材）
| 模型 | SFT | +PPO Flow-SDE | +PPO Flow-Noise | 最佳增益 |
|---|---|---:|---:|---:|---:|
| π0 (MT50) | 50.8% | 78.1% | **85.8%** | **+35.0pp** |
| π0.5 (MT50) | 43.8% | **70.7%** | 66.1% | +26.9pp |
| π0 (LIBERO) | 57.6% | — | **97.6%** | +40.0pp |
| π0.5 (LIBERO) | 77.1% | — | **98.3%** | +21.2pp |
| GR00T (LIBERO, 附录) | 52.5% | 89.9% | — | +37.4pp |
- OOD 警示（πRL ML45）：ID 增益不迁移到未见任务类型；RL 主要打磨动作级执行
- SmolVLA IL 基线 68.2%（πRL 引用）与本项目基线表 SmolVLA 68.24% 一致 ✓

## 本项目 PPO 脚本（train_ppo_metaworld.py）对照
| 项 | πRL 惯例 | 本项目 | 判定 |
|---|---|---|---|
| 冻结 VLM/视觉 | ✓ 冻结 | V-JEPA + Qwen 冻结 ✓ | 一致 |
| 训练范围 | action expert + critic | VA + flow head + noise sched + value head | 一致（VA=本项目 action expert 等价物）|
| KL vs IL | 无 | 无 | 一致（保持简单指令）|
| 稀疏奖励 | 0/1 success | 0/1（executed macro 内 success）✓ | 一致 |
| 归一化 | q01-q99 | v5 q01/q99 ✓ | 一致 |
| γ/λ/epochs | 0.99/0.95/4 | 0.99/0.95/4 ✓ | 一致 |
| clip ε | 0.2 | **0.1** | 更保守，可注脚或对齐 0.2 |
| actor LR | 1e-5/5e-6 | **3e-6** | 同数量级 ✓ |
| critic LR | 1e-4 | 1e-4 ✓ | 一致 |
| 噪声探索 | Flow-Noise（π0 最佳 +35pp）| FlowNoiseSchedule ✓ | 一致（π0 最佳变体）|
| RL 期 denoise | 4 | **32（=评估协议）** | 有意的差异：RL 直接优化部署采样器（项目口径：RL 与部署同采样器），论文注明 |
| 环境步数预算 | 单任务 0.1-1M；MT10 ~10^6 | MT10 正式链按 ~10^6 规划 | 对齐 |

## 待办
- [ ] MT10 正式链跑前：考虑 clip ε 对齐 0.2（或保留 0.1 并注脚更稳）
- [ ] 论文 VLA-RL 节引用 πRL=2510.25889（不可引 2504.16054 当 PPO）
- [ ] IL→RL 对比数字入表时与 πRL 增益（+27~35pp）对照；本项目预期 +15-35pp 有文献支撑
