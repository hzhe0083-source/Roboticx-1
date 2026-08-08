# V-JEPA 全量解冻 + 槽 + FM Head 评审（Codex Q1，2026-08-08）

触发：用户问 live 全量解冻 V-JEPA 组合的坑。Codex 深度评审。

## 核心判断
- **lr 1e-6 合理且偏保守**；global grad norm 0.4–1.8 不异常（clip 前 norm，无需改 LR）。
- **40K 步 ≈ 16 次数据遍历**（160K 样本 / 9927）——即使 1e-6，累计漂移不可忽略。
- **全量解冻不是 V-JEPA 官方机器人配方**：V-JEPA 2-AC 冻结编码器训 action predictor；
  V-JEPA 2.1 下游统一 frozen encoder；端到端 VidQA 解冻成功依赖 18M–88.5M 样本
  （arXiv:2506.09985, 2603.14482）——不能外推到 9927 样本。
- **主要风险不是 BF16 也不是"输出变常数"式坍塌**，而是：高层漂移、patch token
  同质化、FM-only 动作捷径。LayerNorm 掩盖范数异常 → 不能只看输出范数。
  （参考 Lossless Adaptation arXiv:2304.06600；ReVLA arXiv:2409.15250 微调后
  DINOv2 深度 probe 退化近常数。）
- **FM-only 梯度干扰先例**：π0.5 knowledge-insulation 发现新 action expert 的 FM
  梯度不利地干扰预训练 backbone，FM-only 收敛较慢；其 stop-grad 成立是因为
  backbone 另有离散动作/VLM loss——我们 FM-only 若 stop-grad 就等于冻结 V-JEPA。
- **BF16 + FP32 master 无需 GradScaler**（BF16 动态范围接近 FP32；GradScaler 也
  不能平衡 backbone/head 尺度；AdamW 按参数维护 moments）。gradient checkpoint
  只换显存，不解决遗忘/收敛；LLRD 不省显存。

## 推荐动作（按优先级）
1. **分组梯度日志**（每 100 步，unscale 后 clip 前）：V-JEPA 每 3-4 block、
   slot、VA、head 的 grad_rms = ||g||₂/√numel、update/weight、G²group/G²total、
   nonfinite/零梯度比例、clip 触发率。仅在持续 10× 跳升、nonfinite、或 head
   导致 >25% 步被全局裁剪时才干预。
   → 状态：**当前 run 未加**（需重启，代价不划算）；下一轮训练前加。
2. **表示漂移面板**：256 个按任务分层留出的 anchor clips，step-0 特征存档；
   记录 pre-final-LN token RMS、逐通道 std、centered covariance effective rank、
   与 step0 的 linear CKA、patch 平均 off-diagonal cosine。告警线（启发式）：
   norm/effective-rank 变化 >20%、中低层 CKA 前 5K 步 <0.9、token cosine 升 >0.1。
   → 状态：**step-0 anchor 未存档**（本轮重启已错过），下一轮训练前启用。
3. **slot/FM 独立监测**：slot 间 pairwise cosine、attention entropy；
   instruction shuffle / image shuffle 的 Δaction（模态被忽略检测）；
   FM 按 task 与 t 分 bin 的 target/pred norm、loss P50/P90。
4. **冻结策略最小消融（决定性实验，下一轮）**：从同一数据/seed fork 3-5K 步，
   三臂：全冻结 / 仅解冻最后 4 blocks + final norm / 全量 1e-6；
   比较 rollout success、FM loss、表示漂移。若 full FT 增益未超 episode/seed
   方差 → 保留 top-4 冻结方案。新训练可先冻结 backbone 1-2K 步稳定 slot/head，
   再 warmup 解冻顶层。
5. **gradient checkpoint 仅在 OOM 时开**（use_reentrant=False，FM t/noise 在
   checkpoint 区外采样）；86.8M 的 FP32 参数+梯度+Adam 2 moments ≈ 1.4GB，
   与激活相比是主要显存项，checkpoint 帮不上。

## 与当前实验的关系
- Stage A = 全冻结（预计算特征）→ 已有数字（92.4% 开环 / 31.8% 闭环，v5 direct）。
- Stage B = 全量解冻（在线）→ 训练中（step ~200 @57/min，ETA ~11h）。
- **top-4 解冻臂是补齐三臂对照的关键缺口**，排在槽消融之后。
- 表示漂移面板需在下次训练前固化（anchor 存档 + 每 5K 步指标）。

---

# 16GB 显存风险评审（Codex Q2，2026-08-08）

## 结论
bs4/T4 全解冻 BF16 无 checkpoint 可继续，前提：11.1GB 是含 backward + AdamW 的
完整峰值（重启后实测 11.5GB 稳定，GPU 93% 忙——健康）。阈值：max_reserved
≤13-13.5GB 继续无 checkpoint；超过则开 V-JEPA block checkpoint
（官方 vjepa2 已实现，use_reentrant=False），预期峰值 6-8.5GB，耗时 +25-35%。

## 配置核对（当前训练 vs 推荐）
| 项 | 推荐 | 当前 | 状态 |
|---|---|---|---|
| 参数/Adam FP32 + BF16 autocast | ✓ | ✓（训练命令行） | OK |
| 全局 grad clip 1.0（记录 pre-clip norm） | ✓ | ✓（train.py:2229；log grad= 即裁剪前 norm） | OK |
| V-JEPA block checkpoint | 显存>13.5GB 时开 | 未开（11.5GB） | OK |
| num_workers=4 persistent | ✓ | ✓（重启后 57/min，1.05s/step，符合 0.75-0.95 预测区间） | OK |
| pin_memory + non_blocking H2D | 锦上添花 | 未设 | 可选（下次训练前加） |
| encoder_lr = 0.05-0.1× head_lr | 1e-6 vs 1e-5 = 0.1× | ✓ | OK |
| flow loss 显式 FP32 | ✓ | 需核查（pred.float()/target.float()） | 待查 |

## 风险清单
1. 高：11.1GB 测量完整性 —— 已用重启后 nvidia-smi 11.45GB + 多步稳定验证 ✓
2. 高：全解冻优化不稳 —— clip 1.0 已有；持续大量步被裁剪则降 encoder LR
3. 中：偶发 OOM（Adam foreach / VA workspace）—— 超 13.5GB 再开 checkpoint
4. 中：num_workers 未真正并行 —— 已实测 1.05s/step（预期区间内）✓
5. 低：BF16 LN/GELU —— autocast 标准路径低风险；flow 自定义运算注意 FP32

## OOM 处理顺序（如需）
① V-JEPA block checkpoint → ② micro-batch 2 + 梯度累积 2（保持有效 bs4）→
③ 冻结 patch/pos/modality/register embedding + 前 2 层（连续前缀，optimizer 前）→
④ 最后才减 T（优先稀疏采样覆盖原时间跨度）。

## 下一步优化（下次训练前，不重启当前 run）
- pin_memory=True + move_batch non_blocking=True
- 预处理（preprocess_batch 0.16s）移进 worker/collate
- flow loss 显式 FP32 核查
