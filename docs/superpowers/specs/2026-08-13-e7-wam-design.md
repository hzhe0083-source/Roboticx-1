# E7 联合残差 WAM v1 — 设计文档

- 日期：2026-08-13
- 分支：`wam-e7`（base = `main@709d6d0`，工作树干净，无未提交补丁）
- 状态：设计定稿，待用户 review 后进入实施计划
- 决策记录：排期=双轨并行；L_consistency=Codex 方案 E（首轮占位）；分支=wam-e7；取舍=甲+甲+甲（独立模块+2处钩子 / 精简 cache / 4门里程碑）；WAM-VA 耦合=逐层旁路耦合（方案 C）；容量=60M 量级骨干（Grok 两轮调研后的结构定稿，见 §3.1）
- **动机（为什么预测世界）**：服务后续 harness 的自然语言服从性评测——VLA 只背「语言→动作」映射，对动作后果无概念；WAM 让「指令执行后的世界变化」可预测，动作残差朝该目标修正，语言服从从「背映射」升级为「因果对齐」。信息流不是串行链：指令经 VA 逐层加工进记忆快照，WAM 每层交叉注意力直接读对应 VA 层快照 + 当前 16 空间 token + 几何 token（action condition 仅初始化动作残差 token），语言/视觉信息逐层并行注入。

## 1. 目标与不变式

在 E7 精确基座（V6 视觉 → 有效 100k → ROI 验收）冻结后，训练独立
`JointWorldActionFlow`（WAM）：联合生成 48 步动作、3 个未来跨度的 V-JEPA
空间 latent（Δlatent）与 MT-VJ 几何关系，动作以残差形式叠加在原动作 Flow 上：

```
v_action = v_base + α · Δv_WAM        （α=0 或 wam=off → 纯旧路径）
```

三条不变式（最高优先级，任一破坏即停）：

1. **旧路径逐位一致**：`wam=off` 或 `α=0` 时，固定噪声下 condition、memory、
   动作与随机数状态必须与原版 `torch.equal`，且 WAM 代码不得执行。
2. **原权重只读**：原 100k checkpoint 及 VA+MT-VJ 默认路径、权重、评测方式
   全部保留，WAM 权重随新主 checkpoint 保存。
3. **原训练脚本不碰**：`run_e7_all49_to100k.sh` 契约硬编码（SHA/任务顺序断言），
   不做任何修改；WAM 使用独立脚本。

## 2. 基座现状与排期（双轨并行）

- 现状（2026-08-12 实测）：stage B 进行中（有效≈68k/100k），stage C 未开始，
  `e7_mtvj_all49_stageC_100k.pt` 与 sidecar 尚不存在；GPU（RTX 3080 16GB）被
  基座训练占用（~14.2GB）。
- **轨道 1（现在）**：WAM 模块、钩子、单测、cache 构建器、探针脚本 —— 全部
  不依赖 GPU、不碰训练路径。可用现有 `joint80k`/`joint66k` 做 smoke 级调试。
- **轨道 2（基座冻结后）**：正式 cache 构建（独占 GPU，约 4–8h）→ 动作依赖
  探针 → 64 样本过拟合 → 全量 20k 训练 → 评测验收。

## 3. 模型设计

### 3.1 新模块 `JointWorldActionFlow`（va_compound/wam.py，新文件）

骨干（~60M，目标 55–65M，实现后 `numel()` 实测写入 checkpoint 契约）：

- **L12 × d512 × 8 头 MHA**（与 VA 同宽，旁路 K/V 免对齐）；FFN **SwiGLU h=1408**
  （GELU 2048 留对照开关）；**RMSNorm pre-norm**；**qk-norm 开**（仓库已有
  苏式 per-head QK RMSNorm）。
- **时间条件：AdaLN-Zero，slim 版写死 `Linear(256, 6d)` 且零初始化**
  （审稿轮警告：抄仓库 flow 的 `Linear(2d,6d)` 会爆到 ~90M，破 60M 预算）。
- **残差纪律**：pre-norm + 出口 gate 零初始化（块初始恒等，苏剑林残差分析
  kexue.fm/archives/8994 + DiT AdaLN-Zero）；**旁路 CA 的 W_O 零初始化**
  （仓库 MT-VJ dense 同款纪律：训练起点 ≡ 无世界分支）；动作残差速度头
  零初始化；**场景头（Δlatent/几何速度）正常初始化**。
- **位置编码**：类型嵌入（动作/空间/几何）+ 组内位置嵌入（动作 index、
  4×4 空间位置、跨度 id）；异构 102 token 不用 RoPE。
- 层序：**SA → CA(读 VA 快照) → FFN**（惯例默认，留消融）。
- **VA 层映射**（12 vs 4 错配）：`va[min(i·n_va//n_wam, n_va−1)]`，K/V 复用，
  由探针验证。

联合去噪 102 个生成 token（动作与场景 token 共享同一 flow time τ，
与动作直线路径的 τ 一致）：
  - 48 个动作残差 token（对齐原动作 horizon）
  - 3 跨度（k∈{6,24,48}）× 16 个未来 V-JEPA Δlatent 空间 token
    （H11 最后时间片 24×24 网格 → 4×4 空间池化 = 16）
  - 3 × 2 个未来几何 token（每跨度：`g_future = p×visibility` 与
    `ν = g_future − g_current`）
- **逐层旁路耦合（方案 C）**：WAM 第 i 层与 VA 第 i 层对齐，读该层 VA 记忆
  快照（`VisualMemory.layers[i]`，只读 K/V）+ 当前 16 个空间 token + 当前
  几何 token；层内动作残差 token 与未来场景 token 双向注意力。VA 层不感知
  WAM 存在，WAM 不反向写入任何 VA 状态。
- 只读条件汇总：当前 VA action condition（[B,48,512]）、每层 VA 记忆快照、
  当前 16 个空间 token、当前几何 token。
- 输出：场景速度（V-JEPA/几何 token 自身去噪速度）+ 动作残差速度。
  **动作残差输出层零初始化，场景输出层正常初始化**。
  **每跨度（k=6/24/48）使用独立轻量预测头**（共享躯干，+1–3M 参数），
  避免多跨度目标在同一头上互相冲突（Grok 调研建议）。
- 容量预案：v1 即 60M 量级（L12×512，slim AdaLN）；若 G2 过拟合 64 样本
  无法下降 ≥50%，切换候选 B（L6×d768×12 头，隐维对齐 768 目标）重试，
  总参约束仍 ≤65M。不无限加肥。
- 想象结果不写入 VisualMemory，仅用于本次动作联合生成与诊断。

### 3.2 挂载点（只读小钩子，`wam_joint=False` 时零改动）

1. `VACompoundPolicy.encode_condition`：把每层 VA 记忆快照
   （`VisualMemory.layers[i]`）、当前空间/几何 token、action condition 传入
   WAM（N+1 处只读输入，N = VA 层数）。
2. `train.py rollout_policy` / 评测侧：速度合成
   `v = v_base + α·Δv_WAM`；训练时 WAM 场景 token 与动作残差 token 参与
   各自的 flow-matching 目标。
3. 推理延迟预算：每层旁路 K/V 投影为轻量线性层，总开销控制在原版 1.5× 内
   （M4 实测验证）。

### 3.3 旧模块保护

- `FutureLatentPredictor` 原样不动；`wam_joint=True` 与
  `future_predict/evsm` 在 `validate_args` 互斥报错。
- 旧 checkpoint 在 `--wam on` 下明确拒绝加载（无 WAM state）。

## 4. 损失

```
L = L_action + 0.5·L_VJ + 1.0·L_geo + 0.1·L_consistency
```

- `L_action`：联合速度匹配（v_base + Δv_WAM vs 直线路径目标速度），
  前 6 步权重 1、后 42 步权重 0.036（沿用 masked_flow_matching_loss）。
- `L_VJ`：3 跨度权重 1.0/0.5/0.25；目标 = `latent(d+k) − latent(d)`，
  避免静态背景虚假低 loss；GT 全部 stop-grad。
- `L_geo`：同 3 跨度权重；目标为几何残差 ν 与 g_future。
- `L_consistency`（0.1）：**首轮占位（权重 0）**，探针与过拟合门通过后启用
  Codex 方案 E（动态区重加权 + 几何恒等闭环）：
  - 从速度恢复 clean estimate：`x̂₁ = x_t + (1−τ)·v_θ`
  - 窗口 4 帧 GT latent 时空方差 → 动态区域权重
    `w = sg[clip(m/(mean_i m + ε), 0, 4)]`
  - `L_c = ½[ Σw·SmoothL1(Δẑ, sg(Δz*))/Σw + ⅓Σ_k SmoothL1(ν̂_k − ĝ_k + sg(g_d), 0) ]`
  - FP32 计算；GT 先 detach 再求差；监控归一化与截断。

## 5. 数据与 cache

### 5.1 来源

- `data/metaworld_longtraj_windows_h48_all49_repaired_v2.pt`（1.3G，23948 样本）
- 49 个 `data/metaworld_longtraj_<task>-v3.pt`（实测合计 **27G**，JPEG 在线解码；
  原方案写 26.4GB，以实测为准）
- **不用** `/media/ryan/robot-data/longtraj_st288_h48.npy`（65G，残次产物）
- 不重新采集数据。

### 5.2 anchor 与窗口

- 每 anchor 取 4 决策上下文的**最后一个决策点**，按原训练方式重建 VA 记忆。
- 未来窗口：`[d+k-6, d+k-4, d+k-2, d+k]`，k∈{6,24,48}；每跨度只取 d+k 时刻
  的 4×4=16 空间 token 为目标，窗口内其余时刻用于 mask 与动态区域权重。
- split：按任务内 `episode_id % 10` → 0–7 训练、8 验证、9 测试。**禁止随机
  窗口切分**。
- mask：屏蔽成功后、settle 阶段、无效动作，以及从正常阶段跨越外部随机扰动
  的未来目标；+48 缺失的任务仅作低权重辅助。

### 5.3 精简 cache 格式（预算 ~14–20GB，训练零重算）

每 anchor 一条记录：action condition [48,512]、当前 16 空间 token [16,768]、
当前几何 token、3 跨度未来 Δlatent 目标 [3,16,768] + 几何目标 [3,2,d_g]、
动作 [48,4]、mask、split/episode/task 标签。按 task 分片，memmap 读取。
sidecar 绑定：基座 checkpoint SHA、V6 视觉头身份、ROI 版本、数据 SHA、
latent 定义（H11/时间片/池化）、跨度、归一化统计、训练契约 —— 评测与训练
不得自行猜测默认值。

## 6. 训练

- 独立入口 `scripts/train_wam_e7.py` + 独立 sidecar 契约；不修改 train.py 主
  训练循环（WAM 钩子以可选参数接入）。
- 20k micro-step，batch 8，grad-accum 4；BF16；AdamW lr=1e-4；warmup 500；
  cosine decay；每 2k 保存，支持精确续训（扩展 exact_run_contract 字段）。
- 只更新 WAM；V-JEPA、MT-VJ、Qwen、VA、原动作 Flow 全部冻结。
- 训练时每 step 前向冻结基座 flow 得到 v_base（no_grad，成本低）。

## 7. 门（任一不过 → 保留研究 checkpoint，部署 wam=off）

| 门 | 内容 | 不过即停 |
|---|---|---|
| G1 动作依赖探针 | +6 几何误差比「场景不变」基线改善 ≥10%，且打乱动作后损失 ≥50% 增益 | 停 WAM 训练 |
| G2 过拟合 | 64 样本联合 loss 下降 ≥50%，动作与三种场景输出全 finite | 停 |
| G3 离线 | 优于 persistence；同任务打乱动作后性能明显下降 | 停 |
| G4 闭环 | 49×10 paired seeds：WAM ≥295/490；task-bootstrap 95% CI 下界 >0；hard/very-hard 宏平均不降；显存 ≤16GB；延迟 ≤1.5× | 部署 wam=off |

探针实现：线性回归器（动作 48×4 → Δg(+6)）对比「无动作常数基线」与
「shuffle 动作」两对照；GPU 轻量，在冻结基座的 cache 子集上运行。

## 8. 配置与接口

- 配置：`wam_joint=False`（默认）、`wam_horizons=(6,24,48)`、`wam_layers=4`、
  `wam_scene_tokens=16`。
- 运行：`--wam auto|on|off`、`--wam-alpha`；旧 checkpoint 在 auto/off 正常，
  在 on 明确拒绝。

## 9. 测试与验证清单

单测（M0 交付，不依赖 GPU）：

1. 旧 checkpoint 严格加载（wam 字段缺失 → 默认 off，行为不变）
2. WAM checkpoint 回环（save→load 逐位一致，含 sidecar SHA）
3. cache 时间对齐（窗口/决策点索引与 longtraj 帧索引对齐）
4. episode 无泄漏（split 按 episode_id，跨 split 无重叠 episode）
5. 所有 loss 项有梯度（action/VJ/geo 各自非零梯度；占位时 consistency=0）
6. exact resume 一致（续训首 step 与连续训练逐位一致）
7. **wam=off / α=0 位级一致**：固定噪声下 condition、memory、动作
   `torch.equal`，且断言 WAM forward 未执行（与「动作头零初始化」是两件
   独立的事，各自一条测试：零初始化只保证 α=1 训练起点 Δv≈0）
8. 动作残差头零初始化：训练第 0 步 v == v_base；CA 出口 W_O 零初始化：
   训练起点世界分支注入为零

## 10. 里程碑与时间估计

| 里程碑 | 内容 | 估计 | 依赖 |
|---|---|---|---|
| M0 | 分支 + WAM 模块 + 钩子 + 单测 + cache 构建器 + 探针脚本 | 1 天 | 无（现在做） |
| M1 | 正式 cache（4–8h GPU）+ 探针 G1 | 0.5–1 天 | 基座 100k 冻结 + GPU 空闲 |
| M2 | 64 样本过拟合 G2 | 2–4h | M1 |
| M3 | 20k 训练（约 1–1.5 天，3080）+ 离线 G3 | 2 天 | M2 |
| M4 | 三组闭环对照 + 全部验收门 G4 | 0.5–1 天 | M3 |

## 11. 固定假设与风险

- 第一版不用视频 VAE、RGB 解码或随机 latent；只有 V-JEPA 空间 latent 和
  MT-VJ 几何。
- 结果只表述为 49-task subset，不称官方 MT50。
- 实现时需从基座 checkpoint 钉死的未知量（**不得硬编码**）：E7 实际
  action_dim（从 ckpt config 读，勿假设 7）、metric state 精确维度
  （p_flat × visibility 的 8D relation 格式）、cache 实测规模（预算 14–20GB，
  超出则降池化档位）。
- H11 空间网格已由仓库契约钉死（实现时按契约验证）：`mt_vj_contract.md`
  写明 `DENSE_TOKENS=1152=2×24×24`（双时间片），取最后时间片 = 576 = 24×24
  → 4×4 空间池化 = 16 token。
- 基座训练期间 GPU 被占：M0 全部无 GPU；M1 起需基座训练结束或暂停窗口。
