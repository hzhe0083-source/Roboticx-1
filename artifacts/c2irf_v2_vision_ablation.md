# C²-IRF v2：双新息驱动的中央凹交互伺服（GPT Pro 设计评审）

> **定位**：视觉消融/读出头重构的完整设计文档（第三方深度评审，2026-08-09 存档）。
> **来源**：GPT Pro 对 Codex C²-IRF 方案的评审（对 `artifacts/` 中 Codex 讨论的修正版）。
> **状态**：设计定稿待实现；实施顺序见文末"最终实施优先级"。

---

# 总体判决

合并方向（Codex C²-IRF + GPT Pro 视觉头）是对的，但 C²-IRF 还不能直接定稿，有三个需要先修正的理论问题：

1. 把"任务误差"和"系统新息"混成了同一个量；
2. 直接使用 R† 不稳定，且 Codex 给出的符号和它定义的误差方向不一致；
3. "全程只有 L_FM(+L_pair)"与"预测参考几何、学习物理响应矩阵"之间存在可辨识性冲突。

最终版本定为：

# **C²-IRF v2：双新息驱动的中央凹交互伺服（Dual-Innovation Foveal Interaction Servo）**

核心不是"多层特征 + 关键点 + Jacobian"的堆叠，而是：

> **深层语义决定看哪个交互对象，局部残差与中央凹视觉决定实际相对误差；模型新息决定是否重新看，任务误差决定如何修正动作。**

---

# 一、先修正现有方案的四个问题

## 1. "新息"和"任务误差"必须分开

Codex 当前定义 `e_t = 计划参考几何 − 实际观测几何`——这其实是**任务跟踪误差**，不是控制论中的 innovation。

### 任务误差（负责产生动作修正）

```
r_t = g_t* − g_t
```

- g_t：当前观测到的交互几何
- g_t*：当前阶段期望达到的交互几何

### 模型新息（负责视觉重读/增益调节/响应更新）

```
ν_t = g_t − ĝ_{t|t−1},   ĝ_{t|t−1} = g_{t−1} + J_{t−1}·a_{t−1}
```

表示"上一步根据动作预测应该看到什么，与现在真正看到什么之间的差异"，负责：

- 判断视觉跟踪是否漂移
- 判断响应矩阵是否失效
- 触发中央凹重读
- 降低或关闭反馈增益
- 更新局部响应模型

**这两个量不能混用**——否则"目标没对准"和"视觉跟踪失败"会触发完全一样的行为。

## 2. 不要直接用伪逆

若定义 `J_t = ∂g_{t+1}/∂a_t`，动作修正应为 `Δa_t = α·J_t†·r_t`（正号）。

不建议显式计算 J†，因为：多个视觉维度可能高度相关；遮挡时 J 会退化；peg 接触前后局部动力学会突然改变；从少量轨迹得到的 J 容易病态。

改用**不确定性加权的阻尼最小二乘**：

```
Δa_t = α·(J_tᵀ W_t J_t + λI)⁻¹ J_tᵀ W_t r_t
W_t = diag(1/(σ₁²+ε), …, 1/(σ_d²+ε))    # 由视觉模式协方差和可见度决定
```

MetaWorld 动作只有 4 维 → 只需解 4×4 线性系统，代价可忽略。

最终动作：`a_t = clip(ā_t + β_t·Δa_t, −1, 1)`，β_t 由阶段、置信度和新息共同决定。

## 3. 现有 C² 不能原样接上去

当前仓库 `ControllableProjection` 先对所有视觉 token 求均值（`model.py:1182` `z = vision_tokens.mean(dim=1)`）——把保留的 peg/hole/局部边缘再次平均掉。更关键的是，`C2ActionHead` 强制 `Δc₀ = 0`（论文审计 `ora0_paper.md:726` h=0 Δc₀≡0），第一个动作 token 参考状态等于当前状态、第一步误差必然为零——对 40Hz 每步重规划策略，等于 C² 永远不修正动作。

不能只是 `dense readout → 现有 ControllableProjection → 现有 C²`，应替换为：

```
dense/foveal modes → RelationStateProjector → task error r_t → bounded servo correction
```

## 4. "只有 FM loss"与几何参考存在矛盾

当前仓库 C² 实际 loss（`train.py:754`）：`L = L_action + λf·L_future + λr·L_recovery`——参考 c̄ 和增益 K 靠 future/recovery 目标获得语义。

必须二选一：

- **路线 A（推荐）**：严格只有 L_FM+L_pair。不预测任意 latent reference；不声称网络输出物理 Jacobian；g_t 必须是显式定义的关系状态；g_t* 由任务关系直接构造；响应矩阵由在线观测更新（或只称 learned gain）。
- **路线 B**：保留预测未来参考几何的 C²，则必须保留 `L_reference = |g*_{t,h} − sg(g_{t+h})|`，否则参考头可任意退化、K 与 reference 尺度/符号不唯一。

---

# 二、最终推荐结构

## 1. V-JEPA 深层 dense evidence bank

完整保留 `H_t¹¹ ∈ R^{B×1152×768}`。不要把 1152 token 直接送入 VA 自注意力——它们只是只读 K/V evidence：

```
K_t = P_K(H_t¹¹) + φ(p)
V_t = P_V([H_t¹¹, R_t⁵, Δ_t H_t¹¹, φ(p)])
R_t⁵ = P_5(H_t⁵) − Up(Pool(P_5(H_t⁵)))
```

- 深层 feature：这是什么（语义身份）
- 中层残差：局部边缘和空间模式
- 时间差 Δ_tH：正在怎样运动
- 坐标 φ(p)：它在哪里

高频残差只是 value enrichment，**不再被包装成亚厘米定位机制**。

## 2. 深层语义 Key 也要加浅层空间修正

寻址 logit：

```
ℓ_{r,n} = q_{r,d}ᵀK¹¹_n + γ_r·q_{r,s}ᵀK⁵_n + b_coord(p_n) + b_track(p_n; μ̂_{r,t})
```

γ_r 零初始化或 0.01。不产生真正的亚像素信息，但提高 coarse mode 在 patch 网格上的稳定性。

## 3. 不再对整个 heatmap 做一次 soft-argmax

当前 `LocalControlSlotReader` 对整体注意力分布求加权平均坐标——两个候选物体存在时输出落在两者之间（假中点）。

改为**无辅助 loss 的模式提取**：

1. 每个角色产生 heatmap `L_r ∈ R^{2×24×24}`
2. 局部 NMS（max_pool2d）找 **top-2 峰** `m_{r,1}, m_{r,2}`
3. 每个峰只在局部 **5×5 邻域**做 soft-argmax（跨 patch 插值，不平均远距离候选）：

```
μ_{r,j} = Σ π_{r,j,n}·p_n,   Σ_{r,j} = Σ π_{r,j,n}(p_n−μ)(p_n−μ)ᵀ,   z_{r,j} = Σ π_{r,j,n}·V_n
```

4. 增加 **NULL 模式**：learned null key/value；物体被遮挡或不存在时查询选择 NULL，可见度 `v_r = 1 − P(∅)`，无需 visibility loss。

## 4. 不要先平均模式再控制

object/target 各有 2 模式 → 4 种配对假设，每个假设独立产生关系状态与动作修正，再按语言/阶段/可见度/一致性加权混合：

```
r_t^ij = g_t*^{ij} − g_t^{ij}
Δa_t^ij = Servo(J_t^ij, r_t^ij)
w_ij = softmax f(q_relation, z_o,i, z_t,j, μ_o,i − μ_t,j)
Δa_t = Σ w_ij Δa_t^ij
```

若假设熵 H(w) > H_max：降 β_t、暂停强反馈、触发中央凹重读。

---

# 三、真正解决亚厘米：动态中央凹前缀

## 1. 只为当前活动交互关系生成一个 ROI

ROI 中心 `c_t = (μ_left + μ_right)/2`；ROI 尺寸：

```
s_t = clip(k_d·|μ_left − μ_right| + k_σ·√tr(Σ_t), 64, 192)
```

- 角色相距远：不进入精密 servo
- 距离缩小但不确定性高：192px broad crop
- 接近接触且置信度高：96px 或 64px narrow crop

## 2. 对 crop 放大后只跑 V-JEPA 前缀

96×96 区域 resize 到 384×384 → 16px/patch（≈1.67cm）变为 **4px/patch（≈4.2mm）**——真正的亚厘米感知分辨率。

只跑 `patch_embed + block 0 + block 1 + hierarchical norm`（冻结前缀作为高分辨率局部编码器，比随机 CNN 合理）。

**必须对同一个四帧窗口使用完全相同的 crop 仿射变换**，不能每帧分别重新居中——否则制造假的视觉运动。

## 3. 全局与局部双速率

```
20Hz：完整全图 V-JEPA → 重新检测角色与模式
40Hz：foveal V-JEPA prefix → 更新局部关系与 servo
40Hz：proprio + bounded action correction
```

`|ν_t| > τ_ν` 或 `H(w) > τ_H` 或 `v_r < τ_v` 时立即提前全局刷新。（ResidualViT 是速度优化，非精密定位核心。）

---

# 四、局部控制状态定义

不要再用全局 PCA mean。建议 12–24 维显式关系状态：

```
g_t = [ μ_eef − μ_object
        μ_object − μ_target
        log(s_object/s_target)
        Δμ_object − Δμ_target
        P_z(z_object − z_target)
        q_gripper
        z_eef ]
```

期望关系初始化为零对齐 `g_t* = 0`，允许语言/phase 产生 bounded offset：

```
g_t* = δ_max·tanh(W_g[q_relation, q_phase])    # 最后一层零初始化
```

只有数据证明需要偏移时才打开——在只有动作 loss 时比任意 reference head 更容易辨识。

---

# 五、响应矩阵实现顺序

## MVP：直接预测有界低秩增益（先不要上 J†）

```
K_t = κ_t·U_t·V_tᵀ   (rank 2 或 4, U∈R^{4×r}, V∈R^{d×r})
Δa_t = K_t·r_t
κ_t = κ_max·tanh(ρ_t)
```

幅度上限 `|Δa_t|∞ ≤ c_t` 随阶段变化：coarse 0；approaching 5–10%；pre-contact 10–20%；contact correction 20–30%（需高置信度）；不确定/遮挡 0。

## 完整版：网络 prior + 在线 secant 更新

```
J_t⁰ = J_θ(z_t)                                    # 网络输出响应 prior
J_t = J_t⁰ + η·(y_t − J_t⁰s_t)s_tᵀ/(s_tᵀs_t + ε)  # Broyden/secant 更新
```

更新条件：角色可见、mode entropy 低、crop 未切换、phase 未变、动作幅度足够、无碰撞异常。

注意：在线 Jacobian/视觉伺服本身不是新贡献（kPAM 2.0、Neural Jacobian Fields 已覆盖）。差异必须落在：**语言角色定义交互关系；预测新息驱动视觉重读；任务误差驱动受限动作修正；dense evidence 从不被剪除。**

---

# 六、怎样坚持只有动作 loss

主训练目标仍可只用 `L = L_FM + λ_pair·L_pair`，但数据必须变：

1. **加微扰恢复数据（不是加恢复 loss）**：从 near-contact 状态创建 EEF 横向/高度偏移、object 偏移、peg/hole 相对偏移、grasp 后姿态偏移，专家给出恢复动作；与普通样本用完全相同的 L_FM。
2. **clean/perturbed pair 共享同一 (τ, ε)**：两条监督间的动作差异不被随机 flow noise 淹没。
3. **第一阶段冻结 base policy**：只训 dense reader、foveal adapter、relation projector、servo branch——否则 89% 拟合能力的 base path 会吸收恢复数据，servo 分支保持关闭。第二阶段再联合解冻。
4. **精细扰动 2–8mm（图像 2–8px）**：coarse path 看不出区别、dense/foveal path 看得到 → 动作 loss 强迫局部支路承担职责。

---

# 七、训练顺序（重排后）

```
Step 0  Dense H11（1152 readout）→ go/no-go：单步拟合 +5pp 或精密 chunk0 error −20%
Step 1  多模式 + NULL + 局部 5×5 soft-argmax
Step 2  低秩 servo gain（full-frame 24×24 坐标闭环）：zero-gain / gain-shuffle / wrong-role / open-loop 四消融
Step 3  中央凹 V-JEPA prefix（亚厘米主机制）：peg-insert / hand-insert / assembly / nut / sweep-into / stick-push
Step 4  H5/H11 residual head（降级为加分项——V-JEPA 2.1 最终层经 deep self-supervision 已较强）
Step 5  视觉 readout adapter（bottleneck 32，α=0.01，H detach）；无效才 last-1 block LoRA/解冻
```

**Step 0 目标**：回答唯一问题——停止池化能否突破 89% 单步拟合天花板？

---

# 八、关于零门控：不要制造梯度死区

`A = A_base + g·ΔA, g=0` 时 residual branch 内部参数梯度为 0，只有 gate 有梯度。

- 方案一：`g = σ(−4.6) ≈ 0.01`
- 方案二（推荐组合）：residual gate 初始化 0.01 + output projection 零初始化 + base policy 第一阶段冻结

FabriVLA 的零门控配全模型联合优化 + 大 batch；小数据、独立新支路不能机械复制。

---

# 九、8GB 与计算量

- Dense evidence：1152×768×2B = 1.69 MiB/sample；H5+H11 两层 ≈ 3.38 MiB；四层 6.75 MiB
- Role cross-attention：24 queries × 8 heads × 1152 keys → FP16 score 0.42 MiB（FP32 上限 0.84 MiB）
- **不要做**：1152 token 在 VA 内多层自注意力（FP32 单层 score ≈ 40.5 MiB，不含 QKV/FFN/BPTT）

推荐 8GB 配置：

```
Qwen：CPU/offline，只保留 role cache
V-JEPA：FP16/BF16，冻结 + no_grad
full-frame：1152 dense
VA：只接 16 global + 12 role modes + relation/state tokens
foveal prefix：冻结，只运行前 2 blocks
trainable：reader + servo + VA/flow
batch：1–2，gradient accumulation：8–16
```

---

# 十、当前全量解冻 run 怎么处理

- 当前 `--vision-unfreeze-all --lr-vision 3e-6` run 保留为**一次 ablation**，不再扩展 seed
- 主线改冻结 V-JEPA → readout adapter → last-1 block → last-2（有增益才做）
- 8GB 主配置不使用 `--vision-unfreeze-all`
- 理由：全量解冻混入视觉分辨率/多层读出/domain adaptation/数据暴露量四个变量，成功也说不清贡献

---

# 十一、原创性怎么写

"fovea"本身不能吹（ActFovea 已做 action-conditioned foveated regions；Compressor-VLA 已做语言引导 token 压缩）。claim 收敛到三个联动机制：

1. **Semantic–metric K/V factorization**：K=深层语义身份，V=局部残差+时间+度量证据
2. **Dual innovation**：ν_t（预测−观测偏差）控制视觉重读；r_t（期望−当前关系偏差）控制动作修正
3. **Evidence-preserving control**：dense patch 始终可访问、不做 token pruning，只压缩 VA 工作空间

定位表述（英文）：

> We preserve a dense V-JEPA evidence field while factorizing semantic addressing from metric readout. Prediction innovation adaptively reallocates foveal perception, whereas task-space relation error drives an uncertainty-weighted bounded servo correction.

---

# 十二、评估协议必须先修

- FabriVLA 明确使用 24 维 state（含 end-effector、gripper、object positions）——90.0% 不是严格 RGB+EEF-only
- Evo-1 官方评估客户端 `STATE_TAKE = 8`，环境 observation 前 8 维送入模型（非 4D EEF-only）——需按 MetaWorld 版本锁定第 4–7 维语义
- 最终报告三条协议：

| 协议 | 用途 |
|---|---|
| RGB + EEF/gripper only | 证明视觉精密控制能力 |
| RGB + 与 Fabri/Evo 匹配的 state | 公平 SOTA 对比 |
| state-only | 检查视觉是否真实贡献 |

- 本地评估默认 `max_tasks=49`，Stage B 台账 49×10——**不能写成 MT50**

---

# 最终实施优先级

```
Dense H11 → multi-mode roles → bounded direct servo → foveal prefix → H5 residual → readout adapter → last-block adaptation
```

最小完整算法：

```
H¹¹_t = VJEPA(I_t)
M_t = MultiModeRead(Q_L, H¹¹_t, p)
g_t = RelationState(M_t, s_t)
r_t = g_t* − g_t
ν_t = g_t − (g_{t−1} + J_{t−1}·a_{t−1})
Fovea refresh if |ν_t|, H(M_t), Σ_t high
a_t = ā_t + β_t·K_t·r_t
```

**1152 dense patch 解决信息丢失；foveal prefix 解决亚厘米分辨率；双新息和受限增益解决闭环稳态误差。多层 residual 和 V-JEPA adaptation 只是后续增量。**

---

## 参考

[1] V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning — https://arxiv.org/abs/2603.14482
[2] ResidualViT for Efficient Temporally Dense Video Encoding — https://arxiv.org/abs/2509.13255
[3] kPAM 2.0: Feedback Control for Category-Level Robotic Manipulation — https://arxiv.org/abs/2102.06279
[4] FabriVLA — https://arxiv.org/abs/2607.08575
[5] ActFovea: Runtime Safeguarding for VLA Policies — https://arxiv.org/abs/2607.29169

## 关联文档

- Codex 原始讨论（C²-IRF v1）：对话记录 /tmp/ora0_codex_brief.md + /tmp/codex-answer.md
- 论文相关章节：`paper/ora0_paper.md` §3.1/§5.3/§6.5
- 现有代码：`va_compound/local_control_slots.py`、`va_compound/live_vjepa.py`、`va_compound/backbones.py`、`train.py`
