## 结论

这个方向是对的，但必须同时改掉**两个视觉瓶颈**：

1. **名义策略瓶颈**：V-JEPA 原始 patch token 在进入 VA 之前已经被 flat pooling 压成 64 个 token。即使后面换成 role query，也无法恢复已经被平均掉的 peg-hole 局部几何。
2. **C² 控制瓶颈**：当前 `ControllableProjection` 又对 64 个 token 做全局均值，再用 button-recovery PCA 压到 16D。这个坐标系对 peg、nut、sweep 没有可迁移的局部控制意义。

因此，正确结构不是：

[
64\text{ tokens}\rightarrow \text{role query}\rightarrow \text{mean-PCA},
]

而是：

[
\boxed{
\text{V-JEPA dense spatiotemporal tokens}
\rightarrow
\text{language-programmed local slots}
\rightarrow
\begin{cases}
\text{VA nominal action}\
\text{role-relative C}^2\text{ control chart}
\end{cases}}
]

默认采用 **6 个槽、语言调制的固定角色、2×12×12=288 个视觉 token、显式坐标旁路、3 个关系 token**。第一轮不要解冻 V-JEPA，也不要扩大 VA。

我检查的是 GitHub `main`；它当前默认仍是 Qwen3.5-2B、4 层 VA，而你描述的本地 ORA0 是 4B、8 层，因此下面以接口位置为准，不假设本地行号完全一致。

---

# 1. 最小局部槽实现

## 1.1 槽数：默认 (K_s=6)

不要直接用现在的 `role_query_tokens=16`。动作监督下 16 个槽过多，容易出现多个槽读取同一高显著区域。

建议六个固定角色：

| 槽 | 语义角色           | 典型绑定                                      |
| - | -------------- | ----------------------------------------- |
| 0 | actor/tool     | gripper、末端执行器、扫杆                          |
| 1 | manipuland     | peg、nut、被推动物体                             |
| 2 | target         | hole、goal、按钮、容器                           |
| 3 | interface      | 接触面、插入界面、推动侧                              |
| 4 | constraint     | 障碍物、边界、支撑面、clearance                      |
| 5 | operator/phase | open/close、left/right、insert/retract、当前阶段 |

其中 `interface` 不是“第四个物体”，而是一个关系型视觉角色。它可以与 object 或 target 的注意区域重叠。

实验只需要做：

[
K_s\in{4,6,8},
]

主模型用 6。4 是极简消融，8 是容量上界，暂时不要跑 16。

---

## 1.2 采用两级查询，而不是纯固定 query 或纯语言 query

正确形式是：

[
Q_L
===

R+
g_L\odot
\operatorname{CrossAttn}
(R,H_L,H_L),
]

其中：

* (R\in\mathbb R^{6\times d})：固定角色种子，定义槽身份；
* (H_L)：Qwen 全部有效 token 的 hidden states；
* (Q_L)：当前指令实例化后的角色查询；
* (g_L) 初始设为 (\sigma(-2)\approx0.12)，防止训练开始时语言残差摧毁角色身份。

随后每个控制时刻：

[
S_t,A_t
=======

\operatorname{CrossAttn}
(Q_L,Z_t^{\text{dense}},Z_t^{\text{dense}}).
]

关键点是：

* Qwen 只运行一次；
* (Q_L) 也只生成一次并缓存；
* 每个控制步只运行很小的 visual cross-attention；
* 语言位于**视觉压缩之前**，直接决定保留哪一块局部视觉信息。

这不是“把 Qwen 当全局静态 encoder”。更准确地说，Qwen 在命令到来时编译出一个持续运行的局部视觉读出程序。

---

## 1.3 与现有 `RoleQueryResampler` 的关系

现有 `RoleQueryResampler` 已经能输出：

[
[B,K_s,d],
]

但它只读取语言，而且当前两个调用位置最终都执行了：

```python
role_out.mean(dim=1)
```

因此，角色身份再次被平均掉了。它现在是“更复杂的语言摘要器”，还不是视觉局部槽。

推荐做法：

* **复用**现有 `RoleQueryResampler`，作为一级 `LanguageRoleCompiler`；
* **新写**一个 `LocalControlSlotReader`，用一级输出读取视觉 patch；
* 不要把 visual cross-attention 硬塞进现有类，否则会破坏 `TaskResampler` 和 `action_query_cond` 的旧路径；
* 将未平均的 `role_queries` 放进 `LanguageCache`，避免每个控制步重新算语言 cross-attention。

建议扩展：

```python
@dataclass(frozen=True)
class LanguageCache:
    layers: tuple[LayerLanguageCache, ...]
    attention_mask: Tensor
    role_queries: Tensor | None = None  # [B, K_slot, hidden]
```

`build_language_cache()` 中完成：

```python
first_key = caches[0].key.transpose(1, 2).reshape(B, L, hidden_dim)
role_queries = self.role_resampler(first_key, language_mask)
```

---

## 1.4 可直接实现的视觉槽读取器

下面是核心接口，不需要新增训练目标：

```python
class LocalControlSlotReader(nn.Module):
    def __init__(
        self,
        vision_dim: int = 768,
        hidden_dim: int = 512,
        num_slots: int = 6,
        num_heads: int = 8,
        pos_dim: int = 27,  # xyz + 4-band sin/cos Fourier
    ) -> None:
        super().__init__()
        self.num_slots = num_slots

        self.vision_norm = nn.LayerNorm(vision_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Reader-side position channel. Zero-init preserves the initial
        # frozen V-JEPA feature behavior.
        self.pos_proj = nn.Linear(pos_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.pos_proj.weight)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )

        self.read_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        dense_tokens: Tensor,   # [B, N, 768]
        role_queries: Tensor,   # [B, K, 512]
        coords: Tensor,         # [N, 3], normalized t/y/x in [-1, 1]
    ) -> tuple[Tensor, Tensor, Tensor]:
        pos = fourier_encode(coords).to(
            device=dense_tokens.device,
            dtype=dense_tokens.dtype,
        )

        visual = self.vision_proj(self.vision_norm(dense_tokens))
        visual = visual + self.pos_proj(pos)[None]

        delta, weights = self.cross_attn(
            self.query_norm(role_queries),
            visual,
            visual,
            need_weights=True,
            average_attn_weights=False,
        )

        gate = torch.sigmoid(self.read_gate_logit)
        slots = role_queries + gate * delta
        slots = slots + self.ffn(self.output_norm(slots))

        # [B, heads, K, N] -> [B, K, N]
        weights = weights.float().mean(dim=1)
        centers = torch.einsum("bkn,nc->bkc", weights, coords.float())

        return slots, weights, centers
```

随后生成三个显式关系 token：

[
\begin{aligned}
R_t^{EO}&=f_{EO}(S^{actor},S^{object},
\mu^{actor}-\mu^{object}),\
R_t^{OT}&=f_{OT}(S^{object},S^{target},
\mu^{object}-\mu^{target}),\
R_t^{IC}&=f_{IC}(S^{interface},S^{target},
\mu^{interface}-\mu^{target}).
\end{aligned}
]

送入 VA 的视觉序列建议是：

[
V_t^{VA}
========

[\underbrace{C_t^{1:16}}*{\text{coarse context}};
\underbrace{S_t^{1:6}}*{\text{role slots}};
\underbrace{R_t^{EO},R_t^{OT},R_t^{IC}}_{\text{relations}}],
]

总共 25 个 token。这样既保留全局背景，也给精细控制提供显式局部关系。

**不要只把槽送进 C²。** Direct head 的 88.64% 已证明名义路径也受视觉接口限制。槽和关系 token 必须进入整个 VA nominal path。

---

## 1.5 初始化与“只靠 action loss 是否足够”

答案分两层：

* **作为性能结构，action loss 足够做第一轮判决。**
* **作为可解释语义槽，action loss 不足以证明每个槽确实对应你给它起的名字。**

STORM 为避免槽退化，专门使用视觉—语义预训练后再接策略训练；SlotVLA 则使用对象级标注和关系建模。这说明“纯动作监督自动产生稳定对象槽”不是可以默认成立的事实。([arXiv][1])

在不增加 IL loss 的约束下，用结构打破对称性：

1. 六个角色种子不能完全随机同分布。用冻结 Qwen 分别编码六条角色描述，投影到 VA hidden 空间后初始化 `role_queries`。
2. 固定角色顺序，关系头只读取指定槽对，例如 `object→target`，不能对槽做 permutation-invariant pooling。
3. 前 1,000–2,000 个 nominal 更新冻结角色种子，只训练语言残差和视觉 reader；随后再解冻。
4. 训练期间只监控而不优化 attention entropy、槽间重叠、slot knockout 和 simulator mask center error。

因此论文中应说“role-indexed latent control slots”，而不是未经验证地说“发现了 peg、hole、contact 对象”。

---

# 2. 空间信息怎么保留

## 2.1 V-JEPA 2.1 不是没有位置编码

官方 V-JEPA 2.1 hub 配置明确启用了：

* `use_rope=True`
* `interpolate_rope=True`
* patch size 16
* tubelet size 2

而 encoder 在每个 block 中传入 (T,H,W) 网格尺寸。启用 RoPE 时不再向输入显式相加绝对 position embedding，但注意力内部使用时空旋转位置编码。

所以：

> 不要在冻结 V-JEPA 输出上再粗暴相加一整套大幅 2D RoPE。

这会改变特征尺度，而且查询槽本身没有确定空间坐标，第一轮 cross-attention 也无法正确对查询应用 RoPE。

---

## 2.2 需要补的是 reader-side 坐标旁路

V-JEPA token 是 position-aware，不等于下游 reader 可以稳定恢复毫米级相对位置。特别是你的 `flat` pooling 会沿展平的 `[t,h,w]` 序列做 1D 自适应平均，甚至跨越图像行边界；仓库代码已经在注释中指出这一点。

推荐：

[
K_i=W_K\operatorname{LN}(z_i)+
\alpha W_P\phi(t_i,y_i,x_i),
]

[
U_i=W_U\operatorname{LN}(z_i)+
\beta W'_P\phi(t_i,y_i,x_i),
]

其中：

* (\phi) 为归一化坐标加 4 组 Fourier features；
* (W_P,W'_P) 零初始化；
* 只作用于新 reader，不改 V-JEPA；
* 从 attention map 直接计算槽中心、方差、面积和槽间距离。

这样不会破坏冻结特征语义，只是给下游增加一个可学习的度量坐标通道。

---

## 2.3 视觉 token 分辨率

4 帧输入、tubelet size 2、384/16 patch grid 对应：

[
2\times24\times24=1152
]

个原始 token。

16GB 单卡下，cross-attention 本身不是问题；问题主要是数据存储。按 39,708 个决策、FP16、768 维计算：

| 表示                 | 每决策 token |          数据约占 |
| ------------------ | --------: | ------------: |
| 当前 flat            |        64 |      3.64 GiB |
| 2×8×8              |       128 |      7.27 GiB |
| **推荐 MVP：2×12×12** |   **288** | **16.36 GiB** |
| 最后一时刻 24×24        |       576 |     32.72 GiB |
| 完整 raw             |      1152 |     65.44 GiB |

仓库已经有 `pool_spatiotemporal_tokens()`，但 README 和代码都说明它尚未接入训练管线。应该复用它，并将每个时间切片池化到 12×12，而不是重新写视觉编码器。

第一轮用 **288 token**。仍低于 95% 时再切 raw 1152 或 coarse-to-fine，不建议先用 spatial64 得出否定结论。

---

# 3. 语言如何真正定义槽

结论是：

[
\boxed{\text{固定角色种子}+\text{Qwen token-wise cross-attention}}
]

而不是二选一。

纯固定角色 token 的问题是，它只能学习“第 0 槽常常看机械手、第 1 槽常常看物体”，无法自然编码 open/close、push-left/push-right 等操作不对称。

纯语言生成 query 的问题是，槽身份容易随指令重排，C² 的控制坐标无法长期稳定。

对 `push-open` 与 `push-close`：

* actor/object/target 的视觉注意可能基本相同；
* 真正应该发生符号反转的是：

  * `operator/phase` 槽；
  * object-target relation token；
  * 未来参考状态 (\bar c_h(L))；
  * 最终动作差分。

不要强行要求 open 和 close 必须产生不同的 object attention map。正确验证是：

[
\Delta\hat a=
\hat a(I,s,L_{\text{open}})
---------------------------

\hat a(I,s,L_{\text{close}})
]

与专家差分

[
\Delta a^*=
a^*(I,s,L_{\text{open}})
------------------------

a^*(I,s,L_{\text{close}})
]

方向一致。

建议定义：

[
\operatorname{LCDA}
===================

\mathbf 1\left[
\cos(\Delta\hat a,\Delta a^*)>0
\right],
]

要求同观测反向指令对上的 direction accuracy 至少 90%，平均 cosine 至少 0.7。你的数据代码已经支持“首观测和状态相同、只有语言和目标动作不同”的 pair contract，因此可以直接做这个指标。

---

# 4. (\bar c/K) 收缩契约如何接入

## 4.1 不要给原始槽各自独立做 (\bar c/K)

三个选择中：

1. 继续旧全局 mean-PCA：错误；
2. 每个 raw slot 独立一套 (\bar c/K)：不稳定；
3. **在语言角色槽之上构建一个 role-relative control chart**：正确。

定义中间控制特征：

[
g_t=
[
S^{actor},
S^{object},
S^{target},
S^{interface},
R^{EO},
R^{OT},
R^{IC},
\Delta\mu^{EO},
\Delta\mu^{OT},
\text{overlap},
\text{spread},
\text{temporal delta}
].
]

再从 recovery train split 上拟合新的冻结投影：

[
c_t=P_{\text{slot}}\operatorname{LN}(g_t),
\qquad
c_t\in\mathbb R^{16}.
]

这里仍然保留 16D，所以现有 `C2ActionHead` 的：

* `reference_head`
* `gain_head`
* recovery residual loss
* held-out contract metric

都能继续复用。

但这 16D 已经从“全局按钮 PCA”变成了：

[
\boxed{
\text{actor-object}
\oplus
\text{object-target}
\oplus
\text{contact/interface}
\oplus
\text{constraint/phase}
}
]

这才可能救 peg-hole。

当前 `ControllableProjection` 明确执行 `vision_tokens.mean(dim=1)` 后再做 PCA，而 recovery 脚本也是在全局均值特征的扰动差空间上拟合 top-16 PCA。

因此，旧的：

* `pca.weight`
* `pca.bias`
* `c_perturbed`
* `c_nominal`
* v6a step targets

全部不能直接用于新结构。专家动作、恢复 snapshot 和 branch 信息可以复用，但 dense visual token 和 control targets 必须重新生成。

---

## 4.2 推荐训练顺序

**Stage A：局部槽名义策略**

[
\mathcal L=\mathcal L_{\text{action}}
]

关闭 C²，只训练 slot reader、VA 和 direct head。先确认 89% 天花板是否被突破。

**Stage B：冻结视觉控制坐标**

冻结 slot reader，使用 recovery 数据计算 (g_t)，拟合新的 (P_{\text{slot}})，产生新的 `c_perturbed/c_nominal`。

**Stage C：恢复 C²**

载入 Stage A，冻结或低学习率更新名义策略，只训练现有 reference/gain 分支。

**Stage D：PPO**

保持标准 PPO objective，不增加 RL loss。bounded residual 由架构限制，不由额外 PPO penalty 实现。

这满足“IL 不新增 loss，RL 使用标准 PPO”。

---

## 4.3 目前的“有界收缩”表述需要收紧

当前实现是：

[
a_h=
\operatorname{clip}
\left(
\bar u_h-K_h(c_t-\bar c_h),-1,1
\right).
]

这只保证**最终动作有界**，没有保证 correction 有界，也没有从数学上保证闭环 contraction；当前 `gain_head` 输出也没有显式范数约束。

最低成本改为：

[
r_h=K_h(c_t-\bar c_h),
]

[
\Delta a_h=
\delta_{\max}\odot
\tanh
\left(
r_h\oslash\delta_{\max}
\right),
]

[
a_h=
\operatorname{clip}
(\bar u_h-\Delta a_h,-1,1).
]

其中每个动作维度的 (\delta_{\max}) 用 recovery train split 中专家恢复残差绝对值的 95% 分位数确定。第一轮只修正 arm xyz，gripper correction 设为零，因为你的 gripper 已经达到 99%。

论文中应称：

> bounded contraction-oriented residual controller

或者：

> empirically contractive residual controller

不要直接称“certified contraction”。真正的 contraction certificate 需要学习或验证状态相关 contraction metric、闭环 Jacobian 或等价条件；ContractionPPO 就使用了专门的可微 contraction metric 层，而不是仅靠 PPO 和动作裁剪。([arXiv][2])

---

# 5. 判决门槛

## 5.1 离线单步

沿用你现在完全相同的 `<0.05` arm pass 定义：

| 指标                      |     停止 |   继续优化 |     结构通过 |    强结果 |
| ----------------------- | -----: | -----: | -------: | -----: |
| Overall arm pass        |   <92% | 92–95% | **≥95%** |   ≥97% |
| 95% bootstrap lower CI  |   <90% | 90–93% | **>93%** |   >95% |
| peg/nut/sweep/reach 子集  |   <88% | 88–92% | **≥92%** |   ≥95% |
| gripper                 | <98.5% |      — | **≥99%** | ≥99.5% |
| language-pair direction |   <80% | 80–90% | **≥90%** |   ≥95% |

95% 是最低“视觉接口假设成立”门槛，不是最终闭环充分条件。

同时报告：

* arm 三轴分别的 pass；
* arm L2 error 的 median、P90、P95；
* h=0 与 h=1…5；
* 精密任务和非精密任务分开；
* direct 与 C² 的差距。

改造后若 overall 95%，但 peg/nut 仍在 85%，说明模型只修复了普通任务，没有修复提出论文所依赖的局部几何问题。

---

## 5.2 闭环

从 18.2% 出发：

|           MT50 成功率 | 判决                                 |
| -----------------: | ---------------------------------- |
|               <25% | 改造基本失败                             |
|           **≥30%** | 工程上值得继续，尚不足以支撑论文                   |
|        **≥35–40%** | 核心架构假设通过                           |
|        **≥45–50%** | 有 CCF-A 论文价值，但仍需第二 benchmark 和泛化结果 |
| 接近或超过 matched SOTA | 完整竞争性结果                            |

精密任务的 10-episode smoke test 至少应做到：

* peg/nut/sweep/reach 不再全部 0/10；
* 至少三个精密任务达到 3/10；
* 随后用每任务 50 episodes、3 seeds 做正式评价。

在你给出的 68–80% 对照下，18.2→30% 是“继续”，不是论文终点。

---

# 6. 如果改完仍低于 95%，下一个杠杆是什么

顺序不能反。

### 第一优先级：确认 reader 真的看到了 dense token

若仍把 role query 放在 flat64 后面，这个实验无效。先做：

* flat64；
* spatial64；
* spatiotemporal288；
* raw1152；

四档视觉输入 probe。

### 第二优先级：显式相对几何

加入：

[
\mu^{object}-\mu^{target},\qquad
\mu^{actor}-\mu^{object},
]

以及 attention overlap、spread、时间差。

仅有 slot embedding 不代表动作头能稳定解码相对位置。

### 第三优先级：coarse-to-fine 二次读取

第一遍获得槽中心 (\mu_k)，第二遍只读取该位置周围的 raw 3×3 或 5×5 patch neighborhood。这个步骤对 peg-hole 比单纯增加槽数更有效。

### 第四优先级：时间与接触

保留两个 tubelet 的 token，不做 time mean；增加：

[
S_t^k-S_{t-1}^k
]

或短期 slot memory。contact 很多时候是视觉变化加 proprio 的联合事件，而不是单帧纹理。

### 第五优先级：V-JEPA 最后 1–2 block adapter

前四步完成后仍低于 95%，再给 V-JEPA 最后 1–2 block 加 rank-4/8 LoRA 或小 adapter。16GB 下不建议直接完整解冻。

### 最后才是 VA 容量

当前 action loss 已经收敛，而且 direct/C² 同时卡在相同天花板。更大的 VA 无法恢复前端已经平均掉的信息。只有当局部槽能线性读出相对几何、而 action pass 仍低时，才增加 VA 深度或 hidden width。

另一个应保留的 oracle 对照是：直接给相同 head 输入 MetaWorld 的真实 `hand-object-target` 相对坐标。若 oracle 也无法达到 98% 左右，就不能再把全部剩余误差归因于视觉。

---

# 7. 原创性边界

截至 **2026 年 8 月 8 日**，我没有找到名为 `PULSE-VA` 的机器人/VLA 近作，也没有找到公开论文完整组合了以下三项：

1. 语言实例化的固定角色局部槽；
2. 角色关系槽构成 C² 的控制误差坐标；
3. 高频有界 residual correction，并保持 action-only IL。

但这是检索结果，不是形式上的“无先例证明”。

已有工作碰撞如下：

| 组件                             | 已有工作                    | 碰撞程度                                |
| ------------------------------ | ----------------------- | ----------------------------------- |
| task/language-aware slots      | STORM                   | 高；冻结视觉模型、任务语义槽、分阶段训练                |
| object/relation slots          | SlotVLA                 | 高；对象槽、关系表示、LLM action decoder       |
| instruction-guided compression | RT-1、Compressor-VLA     | 高；语言调制视觉提取和 token 压缩                |
| 高频 chunk correction            | A2C2                    | 高；每控制步从新观测预测 action residual        |
| drift monitor/replanning       | VLA-Corrector           | 中；视觉 latent 偏差检测和纠偏重推理              |
| world-model verification       | CheckVLA                | 中；动作条件世界模型、风险触发、suffix rewrite      |
| clipped residual correction    | RTCF                    | 中；检索成功轨迹并注入受裁剪的低频 residual          |
| contraction in VLA/RL          | DiG-Flow、ContractionPPO | 中；分别涉及收敛式推理修正和正式 contraction metric |

STORM、SlotVLA、Compressor-VLA 已经使“我们首次提出语言感知槽”站不住脚。RT-1 更早就用语言 FiLM 调制视觉编码，再通过 TokenLearner 压缩视觉 token。([arXiv][1])

A2C2 是逐控制步 action-chunk residual；VLA-Corrector 是 latent drift monitor 加事件触发重推理；CheckVLA 是动作条件世界模型验证和 suffix rewrite。它们都不是“局部角色槽作为控制误差坐标”，但已经占据了 correction 叙事。([arXiv][3])

还需要特别注意 2026 年 8 月 5 日出现的 RTCF：它使用 coefficient-wise clipped 的低频动作 residual。它不使用语言槽，也不是 contraction chart，但会削弱“首次有界 residual correction”的表述。DiG-Flow 也已经声称其 guided inference refinement 具有 contraction 收敛性质。([arXiv][4])

因此最安全、也最有价值的贡献表述是：

> **PULSE-VA introduces a language-programmed role-aligned visual interface in which local object–target–contact relations form the error coordinates of a bounded high-frequency residual controller.**

不要写：

* first language-conditioned slots；
* first VLA correction；
* first contraction VLA；
* first instruction-guided visual compression。

你的真正新意是：

[
\boxed{
\text{Language}
\rightarrow
\text{typed local visual readout}
\rightarrow
\text{relative control chart}
\rightarrow
\text{bounded feedback action}
}
]

而不是其中任意一个单独模块。

---

# 8. ORA0 内的最小改动顺序

建议只增加或修改以下位置：

```text
/home/ryan/Documents/robot/ORA0/
├── va_compound/
│   ├── model.py                 # LanguageCache + slot routing + C² bounded residual
│   ├── backbones.py             # dense/spatiotemporal token API
│   └── local_control_slots.py   # 新增
├── prepare_mw_local_features.py # 新增，写入 /tmp
├── prepare_mw_recovery.py       # 新 slot-control targets
├── train.py                     # 新 flags，loss 不变
└── tests/
    └── test_local_control_slots.py
```

缓存只写：

```text
/tmp/pulse_va_mw/
```

GPU 严格串行时，执行顺序应是：

1. CPU 完成模块、shape test、cache detach/to test 和旧 checkpoint backward-compatibility test；
2. GPU 单独提取 `2×12×12` FP16 特征；
3. 关闭 C²，训练 slot nominal direct model；
4. 单步 arm 未达到 95%，不进入恢复训练和 PPO；
5. 达到 95% 后重新生成 slot recovery chart；
6. 训练 C²；
7. 先做闭环 smoke test，再做正式多 seed；
8. 最后才运行 PPO。

必须保留的消融链是：

[
\text{flat64}
\rightarrow
\text{spatiotemporal288}
\rightarrow
\text{fixed slots}
\rightarrow
\text{language slots}
\rightarrow
\text{+relative geometry}
\rightarrow
\text{+slot-C}^2.
]

其中 `spatiotemporal288` 无槽 baseline 很重要：它能判断收益究竟来自“单纯提高视觉分辨率”，还是确实来自语言定义的局部控制接口。

[1]: https://arxiv.org/abs/2601.20381 "https://arxiv.org/abs/2601.20381"
[2]: https://arxiv.org/abs/2603.19632 "https://arxiv.org/abs/2603.19632"
[3]: https://arxiv.org/abs/2509.23224 "https://arxiv.org/abs/2509.23224"
[4]: https://arxiv.org/abs/2608.04527 "https://arxiv.org/abs/2608.04527"

