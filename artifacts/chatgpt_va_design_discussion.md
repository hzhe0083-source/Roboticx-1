# ChatGPT 对话存档:Language VA 模型架构设计讨论

- **来源**:ChatGPT Pro 分享链接 `chatgpt.com/share/6a75ebcc-7288-83ec-9277-010e31509de6`
- **存档日期**:2026-08-07
- **主题**:70M V-JEPA + Qwen3.5 的 Language VA 模型架构设计;Xbot(ORA0)仓库语言利用问题诊断;MetaWorld 泛化性审查
- **用途**:存档备查 + 供 `chatgpt_va_design_audit.md` 逐条审查
- **注意**:对话中引用的仓库事实(数字、文件名)已核对,出入见审查文档"事实核查表";本存档保留对话原貌。

---

## 第 1 轮:原创架构设计(TR-VA:Task-Residual Vision-Action)

用户需求:70M V-JEPA + Qwen3.5-4B 做 Language VA 模型,提升语言理解能力(真正听懂任务而非做子任务)与推理频率。

### 核心思路

不是"Qwen 理解任务 → 输出子任务 → VA 执行",而是:

> Qwen 把完整指令编译成持续存在的"任务契约";VA 在每个控制周期重新计算当前视觉状态对整条任务契约的满足程度,选择最能减少任务残差的动作。

动机:现有 VLA 一条轨迹只有一条指令、数百个视觉—动作对,视觉—动作监督压倒语言—动作监督;场景足以猜出任务时 VLA 会忽略语言,甚至与场景矛盾的指令也会被忽略。

### 整体架构

```
Instruction → Qwen3.5-4B Task Compiler(仅指令变化/异常时运行)
    → Persistent Task Contract M_L(缓存,每动作周期被读取)
Video → V-JEPA-70M(时空 patch tokens)
    → Role-Grounded Belief Core(语言角色查询视觉:蓝杯在哪?条件满足否?遮挡/矛盾/停滞?)
    → Task Residuals + Belief State + Uncertainty
    → Fast Action Proposer(生成 K 个连续 action chunks)
    → Action-conditioned Semantic Predictor(预测每个候选之后 goal satisfaction/constraint/risk)
    → 选最能减少完整任务残差的动作 → Robot
```

### 模块 1:Language Task Compiler

- 不让 Qwen 输出子任务序列(会退化为技能调用),而是输出任务契约:
  - `[GLOBAL_GOAL] [ROLE_1..8] [CLAUSE_1..8] [CONSTRAINT_1..4] [TERMINATION] [ABSTAIN]`
- 特殊查询 token 追加在指令后,提取 Qwen 中间层(如 {16,24,32})与末层 hidden,经 Contract Resampler 压缩(层权重可学习),不只读 EOS、不做 mean pooling
- 示例指令"把蓝杯放进白碗,不碰红块,保持杯子直立"编译为:
  - Global goal: 蓝杯最终位于白碗内部
  - Roles: manipulated=蓝杯, destination=白碗, forbidden=红块
  - Clauses: inside(蓝杯,白碗), upright(蓝杯), not_contact(蓝杯,红块)
  - Termination: inside ∧ upright ∧ gripper_released
  - Abstain: 蓝杯/白碗不存在或绑定不可靠时不动
- 运行时主要是连续 latent token;可加可解释文本头(调试用),动作模型读 latent

### 模块 2:双速率 V-JEPA 感知(Anchor–Delta)

- V-JEPA 输出时空 patch tokens(不是全局 pooled embedding)
- Anchor 路径:每隔若干帧完整计算 `V_τ = E(x_{τ-K+1:τ})`
- Delta 路径:~10–20M 轻量更新器 `ΔV_t = D(x_t, x_{t-1}, s_t, a_{t-1})`,只更新 role slots,新 anchor 到达时纠正漂移
- 小物体/精细抓取:Delta 路径可用高分辨率腕部图像

### 模块 3:Role-Grounded Belief Core(关键模块)

- 语言角色 token 经 cross-attention 查询视觉 patch:`O_t = CrossAttn(M_L^role, V_t, V_t)`,输出"当前蓝杯状态""当前白碗状态"等角色槽
- 每个角色槽输出:绑定置信度、soft spatial mask、可见性、时序身份一致性、遮挡
- 角色槽经 recurrent memory 保持:`h_t = F_h(h_{t-1}, O_t, s_t, a_{t-1}, M_L)`
- 对每个语言 clause 输出三件套:`(q_j, ρ_j, u_j)` = 满足程度 / 需改变的 latent residual / 不确定性
- 任务能量:`E_t = Σ_j w_j (1-q_j) + λ_u u_j` —— 语言条件下的 latent task error,不是三维位置误差

### 模块 4:Predictive VA Core

- 候选动作:`A^(k) ~ π_θ(h_t, O_t, M_L, {ρ_j})`,K=4,H=8 或 16,flow steps 2–4,每块只执行前 1–4 步
- Task Contract 通过 gated cross-attention 注入动作头多层(不只输入端一次)
- 预测每个候选:`(ĥ_{t+H}, q̂_{t+H}, û_{t+H}, r̂_{t+H}) = F_ψ(h_t, A^(k), M_L)` —— 预测角色对象未来 latent、条件是否满足、风险、不确定性
- 评分 `J_k = Σ_j w_j (1-q̂) + λ_r r̂ + λ_u û + λ_s C_smooth`,选最小者
- 不是选择离散技能,而是在多个连续运动方案中选预计最能满足整条指令的

### 模块 5:事件驱动 Qwen 重编译

触发:`[u_t > τ_u] ∨ [E_{t-Δ} - E_t < ε] ∨ [‖h_t - ĥ_t‖ > τ_s] ∨ [指令变化]`,对应:角色绑定不可靠 / 无进展 / 预测与实际不一致 / 用户改指令。重编译接收原始指令 + 当前已满足/违反条件 + 不确定角色 + 预测失败信息。

### 频率设计

| 路径 | 目标频率 |
|---|---|
| Qwen Task Compiler | 指令变化或异常时 |
| V-JEPA Anchor | 5–15 Hz |
| Delta Perception | 20–50 Hz |
| Belief + Residual Core | 20–50 Hz |
| Action Expert | 20–50 Hz |
| Robot servo | 50–200 Hz |

注:异步 Fast–Slow VLA、30Hz 动作块、TurboVLA 32Hz(4090)已存在,"异步/高频"本身不是创新。

### 训练方法(四阶段)

- **A 对齐**:冻结 Qwen,训练 contract query tokens/layer mixer/resampler/slot gate/contract decoder;监督实体角色、目标、关系、否定、顺序、数量、完成条件、不可执行条件;之后只对 Qwen 顶部若干层 LoRA
- **B 绑定**:冻结 Qwen+V-JEPA,训练 Role-Grounded Belief Core;用 Qwen 多模态路径稀疏关键帧作为 teacher 蒸馏 `L_distill = 1 - cos(O_J, O_Qwen-MM)`,部署时不需要每帧跑 Qwen 视觉
- **C 行为克隆+未来预测**:`L_act = FlowMatching ||L1||`、`L_future = ‖Ô_{t+H} - sg(O_{t+H})‖`、`L_clause = BCE`、`L_done = BCE`;未来帧只作 target,不泄漏输入
- **D 反事实语言训练(必要,不是可选)**:
  - 同一场景不同指令(蓝杯→碗 / 红杯→碗 / 蓝杯→盘 / 不碰蓝杯 / 不存在的绿杯)
  - 对比损失 `L_cf = max[0, m + log p(A+|V,L-) - log p(A+|V,L+)]`
  - 同义改写一致性 `L_para = D_KL[π(A|V,L) ∥ π(A|V,Paraphrase(L))]`
  - 完整目标:`L = L_act + λ_f L_future + λ_c L_clause + λ_b L_bind + λ_cf L_cf + λ_p L_para + λ_d L_done`
  - 第一版不要在线 RL:先证明语言对动作有可测量因果影响

### 证明"不是子任务调用"的五类测试

1. 同场景不同指令 → 动作随语言改变
2. 同义语言一致性 → 改写不改执行
3. 否定与矛盾 → 目标不存在/自相矛盾时停止或澄清
4. 组合泛化 → 训练(蓝杯→碗,红杯→盘),测试(蓝杯→盘)
5. 完整目标验证 → 最终视觉关系确实满足语言条件

注:普通 benchmark 场景常暗示唯一动作,成功率无法判断语言是否参与决策。

### 最小可实现版本

Qwen3.5-4B → 24 个 Task Contract tokens(缓存)→ V-JEPA-70M patch tokens → 6 层 Role-Grounded Transformer → task belief + clause satisfaction → 8 层连续动作头 → 8 步 chunk。暂不做:K 候选 latent MPC、事件驱动重编译、长期记忆、在线 RL、全量微调 Qwen/V-JEPA。必须保留:角色查询视觉、Task Contract 每层注入、clause satisfaction head、abstain head、同场景反事实语言训练、执行后完整任务验证。

### 创新性自我审视

- V-JEPA+VLA 已有 VLA-JEPA、JEPA-VLA;异步 Fast–Slow 已有;角色 token + predicate verification 已有 POT-VLA(typed-subtask plan、RGB-D 记录、显式几何谓词)
- 真正的贡献候选:**把开放词汇指令编译为持续 Task Contract,每周期计算语言条件下任务残差,用 action-conditioned latent prediction 选择连续动作**——语言不降维成一次性子任务标签

---

## 第 2 轮:Xbot 仓库问题诊断与解决方式

### 结论先行

问题分两类:(1) Qwen 表征被训练塌陷(不同指令被 LoRA 推成几乎同一个向量);(2) 即使表征不塌陷,策略也可能不使用语言(视觉+上一动作+递归记忆已够拟合)。两者必须分别解决。换 4B、加深 VA、加 Memory、调 L_pair 都不能根治。

### 现状证据

- 代码把 Qwen 当昂贵文本 Encoder:加载完整多模态模型后只留 `full_model.model.language_model`,删掉视觉塔和 LM Head,只返回最后一层 last_hidden_state
- E2E 路径 Qwen LoRA 只接受动作 Flow Matching 梯度,pair_loss/future_loss 均设零
- 指令向量平均余弦:原始 0.7647 → LoRA 后 0.9992;冻结 V-JEPA 只训 Qwen LoRA 也得 0.9989 → 坍塌主要来源是 Qwen LoRA + 单一动作损失

### 一、先停止表征坍塌

1. **暂停 Qwen 动作损失 LoRA**:Qwen 全冻、V-JEPA 全冻、VA/Adapter/Flow Head 训练;用 `--lora-rank 0 --qwen-unfreeze-blocks 0 --unfreeze-blocks 0` 建干净基线。C1 实验证明冻结后指令空间维持原始 0.7647,Blank/Swap 敏感性远高于 LoRA 端到端模型。但冻结只是止血,接口重构才是核心
2. **不要全层 LoRA**(q/k/v/o/gate/up/down 全动范围过大);改用 Qwen 外部小 Adapter:`H' = LN(H_0 + α Δ_φ(H_0))`,α 初始 0,原始语言表征永远保留;后期需要 LoRA 只动顶部少量层 q/o 投影

### 二、Qwen 从"文本 Encoder"升级为"任务编译器"

- 保留完整 Qwen 与 LM Head(不再 `del full_model`),新增 `QwenTaskCompiler.compile(instruction, keyframes) -> TaskContract`
- 只在指令变化/新 episode/目标不确定/长时间无进展/预测与观察不一致时运行
- 输出目标状态契约(`<MANIPULAND> <TARGET> <GOAL> <KEEP> <AVOID> <DONE> <ABSTAIN>`),不输出子任务列表
- 同时保留显式 Contract(LM Head 生成,用于监督/调试/判断理解正确性)与 latent Contract(多 Qwen 层加权 `Σ softmax(α)_ℓ LN(H_ℓ)`,再 role-query cross-attention)
- role query 建议:`[GLOBAL_GOAL][MANIPULAND][TARGET][RELATION][CONSTRAINT][COMPLETION][UNCERTAINTY][ABSTAIN]`
- **不要再对语言 token 做简单均值**:当前 `action_query_cond` 和 `TaskResampler` 都是 mask-weighted mean → MLP,会压缩对象关系、否定作用域、条件例外、多对象角色、语序组合结构;`action_query_cond` 暂时关闭,后续换 role-query resampler

### 三、加入真正的动态 Semantic State

- 明确拆分:`C_L`(不可变任务契约)与 `S_t`(随执行变化的语义状态:角色已绑定、未抓住、upright 满足、红块距离下降、inside 未满足、done=false)
- 高频更新 `S_t = F_S(S_{t-1}, V_t, E_t, A_t, C_L, p_t)`
- 现有 memory_split/Evidence Memory/Task Memory 可作起点,但 task 要正式改成语义状态并加显式监督头:`progress_head / completion_head / constraint_head / uncertainty_head / role_visibility_head`,否则只是无语义的动作辅助 latent

### 四、语言不能与所有视觉 token 竞争一个 softmax

- 当前共享注意力把 Vision/Evidence/Memory/Action/Task/Language/State 放一次 softmax,SMC 只能缓解 token 数量偏置
- 建议动作流拆两条独立注意力:
  - `O_physical = Attn(Q_A, [V,E,A,S])`
  - `O_semantic = Attn(Q_A, [C_L, S_t])`
  - `A_{l+1} = A_l + O_physical + g_A O_semantic`,`g_A = σ(G(A_l, S_t))`
- Semantic State 用独立查询 `S_{l+1} = S_l + Attn(Q_S, [V,E,A,C_L])`
- 不要强制所有步语言 gate 都高:低层轨迹跟踪由视觉主导;语言应在目标选择、容器选择、关系判断、否定约束、松爪时机、完成判断、异常恢复处明显起作用

### 五、Flow Head 必须直接读取语义

- 现状:flow head 只从 action_condition 间接获得语言(entry 模式);任何环节丢失语言后无法恢复
- 改法:`semantic_context = cat(action_condition, semantic_state, task_contract.latent_tokens)`,每层 block 做 `cross_context` 注入
- **Language-Residual Flow**:`v_final = v_base + g_t Δv_L`,其中 `v_base = v_θ(x|V,E,p)`(运动可行性/平滑/碰撞/低层控制),`Δv_L = v_θ(x|V,E,p,C_L,S_t) - v_base`(目标/关系/约束/任务分叉)。**可直接测量 ‖Δv_L‖ 判断语言是否真正起作用**(在目标选择与分叉点若接近零说明语言被忽略)

### 六、L_pair 保留但必须接入 E2E 路径

- feature 路径 shared-source CF 设计正确(同一 probe/τ/噪声,只换语言,监督绝对场与差值)
- 问题:E2E 路径 pair_loss=0、future_loss=0 → Qwen LoRA 时完全没接受反事实语言约束
- 必须重写 E2EDataset 增加 pair_id/instruction_id/contract_labels/paraphrase_group/goal_state/constraint_labels,接入 PairedBatchSampler
- CF 损失加**动作分叉权重** `w_ij,t = min(1, ‖a_i,t - a_j,t‖/δ)`:不同任务开头可能都接近同一物体,不应强迫所有时刻动作都不同

### 七、Qwen 必须同时接受语义监督

`L = L_FM + λ_CF L_CF + λ_contract L_contract + λ_role L_role + λ_state L_state + λ_para L_para + λ_preserve L_preserve`

- L_contract:LM token cross-entropy 监督生成对象角色/目标/关系/否定约束/完成条件/不可执行条件
- L_role:role token 与 V-JEPA 正确对象区域对齐(仿真可直接提供 object ID/segmentation/3D pose)
- L_state:监督 grasped/inside/on/open/constraint violation/completion/uncertainty
- L_para:同义指令生成相同 Contract、Semantic State 与动作分布
- L_preserve:冻结教师 Q0,KL 保持 + 指令间 Gram 矩阵距离 `‖Z_θ Z_θᵀ - Z_0 Z_0ᵀ‖_F`(比监控平均余弦可靠)

### 八、使用 Qwen 多模态能力,不是只用文本塔

- 双速率:高频 V-JEPA→VA→动作;低频 Qwen(image+instruction+semantic summary)→ role grounding / goal-constraint-completion verification / 刷新 Contract
- 训练:Qwen 对稀疏关键帧出教师 `R^Q = QwenVL(I,L)`,V-JEPA 分支 `R^J = CrossAttn(Q_role, V, V)`,`L_ground = 1 - cos(R^J, sg(R^Q))`;部署只跑 V-JEPA,异常(绑定置信度低/遮挡/停滞/预测不一致)才重跑 Qwen
- SceneTeacher 只有 readout hidden、无 LM Head/显式 Contract/语义正确性监督,本质仍是 latent conditioner,可保留为蒸馏组件但不是最终方案

### 九、训练数据必须支持语言泛化

- PNPW 单指令不能验证语言切换;多任务不同场景也不够(可凭背景猜任务)
- 必须构造:同一精确状态不同可执行指令(最好相同 RGB/状态/历史,而不是特征余弦 0.99 近似);同义改写;组合留出(训练 蓝杯→碗、红杯→盘;测试 蓝杯→盘);否定与条件(用左边的碗,不是右边;保持直立;不碰红块);不可能任务(目标不存在/描述冲突 → abstain,不能选最相似对象)

### 十、代码修改顺序

| 优先级 | 文件 | 修改 |
|---|---|---|
| P0 | end_to_end.py | 禁止纯 L_FM 更新 Qwen;默认冻结 Qwen 和 V-JEPA |
| P0 | train.py | E2E 数据加 pair_id/instruction_id,接入 shared-source CF |
| P1 | backbones.py | 保留完整 Qwen+视觉+LM Head,增加 QwenTaskCompiler |
| P1 | model.py | TaskContractCache 替代 LanguageCache;删语言 mean pooling 改 role-query resampler;task 改有监督 Dynamic Semantic State |
| P1 | VACouplingLayer | 独立 semantic cross-attention,不与视觉共享 softmax |
| P1 | FlowMatchingHead | 每层直接读 Contract + Semantic State |
| P2 | train.py | 加 L_contract/L_role/L_state/L_para |
| P2 | backbones.py | Qwen 多模态关键帧教师 + V-JEPA role 蒸馏 |
| P3 | end_to_end.py | 事件驱动 Qwen semantic refresh |
| 最后 | backbones.py | 小范围顶部 LoRA + preserve loss |

### 十一、不要只用 Qwen cosine 判断语言能力

还要测:VA 实际消费的逐层 K_L/U_L 有效秩与奇异值谱;固定 V/Memory/State/噪声/τ 只换语言看流场方向;paraphrase 保持动作;swap 正确改变动作;不可能指令拒绝执行;屏蔽语言后目标选择下降;‖Δv_L‖ 在分叉点增大;闭环到达语言指定新目标。**关键区分:语言变化后"模型失败"≠"模型成功执行新任务"**——后者才算正确使用语言。

### 最小可行改造

Frozen Qwen full model → Task Contract text + 24 role tokens → V-JEPA role grounding → 8 Dynamic Semantic State tokens → 现有 memory_split VA → Language-Residual Flow Head;训练 L_FM + L_shared-CF + L_contract + L_role + L_done。暂不做:Qwen LoRA、PPO、全量 V-JEPA 解冻、C² Controller、PlanResampler、EVSM、多种 Memory 叠加。理由:语言问题在最上游(接口降级、动作损失破坏语义、E2E 无反事实监督),先改三点再讨论 2B 换 4B,否则 4B 只是更贵的 Encoder。

---

## 第 3 轮:泛化性质疑——在 MetaWorld 上并没有比 Evo-1 好

### 结论

之前的修改解决的是两个窄问题:防止表征塌陷、让语言能影响动作;**没有解决**新状态闭环控制、未见任务组合泛化、小物体视觉绑定、错误后恢复、跨任务共享运动规律。当前主要瓶颈不是 Language Encoder,而是视觉—语言落地、动作控制、数据覆盖、闭环恢复。

### 证据解读

- 能力表:防 embedding 压平(C1 保持、C2 塌陷)基本解决;语言影响动作(wrong instruction 增加开环误差)部分证明;**真正理解语言组合未证明**;MetaWorld 同分布闭环 17.8% 未解决;2-task pilot 的 3 个 held-out 全 0/10 → 明确没有未见任务泛化
- 闭环演化 7.1%→16.3%→17.8%:轨迹时间覆盖从 7.1→16.3(数据覆盖增益大),Qwen-conditioned action queries 只 16.3→17.8(语言接口增益小)
- 2-task pilot:Direct Head 训练任务 10/10,held-out 全 0 → 模型能学会训练任务但没有可迁移控制结构
- **语言敏感性 ≠ 语言泛化**:wrong instruction 只证明 ∂L/∂a ≠ 0;模型完全可能把 49 条 embedding 当 49 个连续 task ID(embedding 1→button policy...仍是任务索引)
- **task-ID 对照不干净**:scripts/eval/eval_mw_lang_ablation.py 评估时临时让 Qwen 编码 "task 0/1/2...",这些 hidden 从未在训练出现 → 结果变差只是输入分布变化。正确对照:从头用完整指令训练的模型 A vs 从头用可学习 task-ID embedding 训练的模型 B,数据/步数/参数/种子全同,再比同分布成功率、paraphrase、same-state fork、held-out 组合
- **MetaWorld 测的主要不是语言理解**:多数任务场景高度不同,视觉可直接判断任务;没有真正 same-state command forks,shared-CF loss 在 MetaWorld 没启用(只训 FM + future latent)
- 成功概率拆解:`P(success) ≈ P(understand)·P(ground)·P(control)·P(recover)`——目前只改善第一个因子

### Evo-1 为什么强

1. **原生多模态融合**:InternVL3 图像 patch 直接插入语言 token 序列,同一 Transformer 联合处理,用中间第 14 层融合表示,动作专家每层 cross-attn 读融合表示 → 每个决策周期拿到的是"这条指令下当前图像哪些区域重要"的联合表示。Xbot 是文本 Qwen + V-JEPA 两个独立骨干,所有视觉—语言绑定要由 ~43M VA Core 从几千个切片中重新学出
2. **两阶段训练**:先冻结 VLM 训 integration+action expert,再联合微调;每任务 50 条示范,10 trials × 5 runs → 80.6%,四个难度都强。Xbot 曝光量远少:单 start 2488 样本 → multi-start 9927 样本,每样本只 4 个决策点
3. **Evo-1 的 80.6% 也不是未见任务泛化**:同任务新初始化测试(状态泛化),不是 zero-shot unseen-task。但即使按这个较有限定义,Xbot 17.8% 仍说明基础控制能力远未达标

### VA2 结构上不保证语义泛化

- Task Workspace 没有语义监督(L_FM + λ_CF + λ_future 不要求 token 表示 manipuland/target/grasped/inside/open/contact/done)——"命名一个 latent 不会让它自动学会任务语义"
- Future Latent 太全局:预测 V-JEPA token 全局均值,画面大部分静态(桌面/背景/相机),模型可靠预测静态场景拿低 loss,不需要理解把手移动、peg 进孔、gripper 接触
- 语言与视觉没有预训练绑定:Qwen 只看文本仍不知道哪个 patch 是抽屉把手;4B 不会自动提高 MetaWorld 成功率,必须用其多模态路径(至少低频教师)

### 真正需要的架构:Grounded State-Transition VA

- **低频 Qwen3.5-VL 输出目标状态而非子任务**:`C = QwenVL(I_key, L)` → manipulated role / target role / desired relation / persistent constraints / completion predicate / uncertainty(如 object=drawer handle, desired transition=joint increases, termination=joint reaches target, constraint=gripper maintains contact)
- **高频 V-JEPA 跟踪语言角色**:`R_t = RoleGrounder(C_role, V_t, E_{t-1})`,角色 token 带 soft spatial mask/位置/可见度/运动/绑定置信度/与末端关系
- **预测当前任务谓词**:`p_t = h(R_t, S_t)`(contact(handle,gripper)、drawer_open_fraction、distance(peg,hole)、inside、grasp_confidence、completion);训练时仿真器直接提供这些状态和 success predicate,部署不需要
- **动作基于任务残差**:`r_t = p*(C) - p_t`;`a_t = π_motor(R,S) + π_task(R,S,r)`——π_motor 是所有任务共享的运动/接触/稳定控制,π_task 是语言目标导致的连续修正;共享 reach/contact maintenance/pose correction/gripper/recovery,但不离散成子任务 ID
- **预测对象级后果而非全局均值**:`R̂_{t+1} = F(R_t, a_t)`,`L_transition = Σ_k ‖R̂_{t+1}^k - sg(R_{t+1}^k)‖`(动作必须预测 drawer joint 增加、peg-hole distance 下降、grasp confidence 提高、contact 是否丢失)

### 先补齐闭环控制(改 Language 前)

1. **全量 executed-action 数据**:~21.4% 原始 expert actions 超环境范围被裁剪 → 旧标签多值对应同一执行动作;Direct Head 修复后 pilot 10/10 vs 旧 Flow 6/10;v5 executed 契约扩展到全 49 任务,再比 Direct Head / 2-step Consistency Flow / 32-step Flow
2. **阶段均衡采样**:pre-contact / first contact / object movement / constraint-critical / near completion / recovery;multi-start 7.1→16.3 证明时间覆盖是主要瓶颈之一,但不够
3. **Receding-Horizon 探针**:execute_steps = 6 / 2 / 1 零训练成本对比;若 1 显著提高说明是开环 chunk 问题而非语言
4. **DAgger / recovery collection**:专家 BC 不覆盖模型自己的失败状态;当前策略闭环 rollout → 找偏离 → scripted expert 接管 → 收集恢复 → 加回训练集,2–3 轮;C² pilot recovery 30%→50% 证明恢复数据有价值,但 reference predictor 损伤 clean 性能;简单 DAgger BC 可能比复杂反馈增益稳
5. **匹配基本训练曝光量**:40k steps、batch 1 或 4 远不够;task-balanced + phase-balanced sampler、batch 16–32、百万级决策展示、3 个 matched seeds、每任务独立统计;在基础 IL 达标前别上 49-task sparse PPO(17.8% 初始策略在困难任务几乎无正奖励,RL 只会强化已会任务,任务集中更严重)

### 泛化实验必须重新定义

- 报告:macro mean、median、zero-success task ratio、最低四分位、easy/medium/hard/very-hard 分组、>20%/50%/80% 任务数、**task concentration**(有效任务数 `N_eff = (Σ s_i)² / Σ s_i²`)
- 建立 MetaWorld-CF:同一物理初始状态、至少两个可执行目标、只换语言;加 paraphrase/否定/新目标位置/新对象—关系组合/impossible instruction/held-out task family(训练 door open/drawer close/window close → 测试 drawer open/window open)

### 最小判决实验(E0–E3)

| 版本 | 改动 | 回答的问题 |
|---|---|---|
| E0 | 官方 Evo-1 权重进同一本地 evaluator | 协议是否公平 |
| E1 | 全量 v5 executed + full-phase data + Direct Head + execute=1 | 基础控制上限 |
| E2 | E1 + 2 轮 DAgger recovery | 闭环 OOD 是否主因 |
| E3 | E2 + Grounded State-Transition Contract | 语言状态分解是否提高 held-out 泛化 |

每版报告 macro/median/zero-task ratio/N_eff/difficulty buckets/held-out composition/same-state command selection。go/no-go:MT10 median ≥60% 且无大量 0-success;MT50 宏平均 ~50% 再谈语言结构增益;Grounded Contract 的 held-out composition 必须明显超过从头训练的 task-ID baseline;paraphrase 下降小;swapped executable instruction 必须完成新目标。

### 最终判断

当前可以声称:冻结语言编码器防表征侵蚀;双向 VA 能用语言条件改变动作。**不能声称**:解决了 VLA 泛化或让模型真正理解未见任务。"闭环主要受数据覆盖限制"也未证伪(数据覆盖重要,但没解释与 Evo-1 相差 60+ 个百分点的原因)。

最合理路线:**先把共享闭环控制做强 → 再把语言绑定到共享状态转变空间 → 最后用 held-out 组合证明泛化**。继续只优化 Language Cache / Task Token / Pair Loss,大概率让模型"更语言敏感",但不会让 MetaWorld 从 17.8% 自动升到 Evo-1 水平。


---

## 第 4 轮:Dual-Path Adaptive Language VA(第三种方案)

回应"完全冻结 Qwen 只取一次最后层 hidden state 不是最终答案":冻结只防破坏、不能充分调用 LLM 推理/视觉语言绑定/状态更新;直接 LoRA 又让 L_FM 单独改写语言模型(C2 中指令表征余弦 0.857→0.998,语言敏感性消失)。核心结论:**不能让动作模仿损失无约束地微调 LLM**。需要第三种方案:

> **冻结语义先验 + 受约束的机器人语义适配 —— Dual-Path Adaptive Language VA**

### 一、区分"使用 LLM"与"微调 LLM"三个层次

1. 静态编码:instruction → final hidden → cache(当前 Xbot 主要方式,只做句→向量映射)
2. 主动推理:instruction + scene + task state + execution result → 更新 goal/role/constraint/completion state(参数冻结也能真实调用预训练推理能力)
3. 参数适配:让 Qwen 学机器人领域概念(可抓取区域、receptacle、"打开"的状态变化、接触/松爪/插入完成)——需要适配,但不能只用动作回归损失

真正的问题不是"冻不冻结",而是:Qwen 是否参与场景条件推理?适配梯度是否受语义约束?原始能力是否有不可破坏的保底路径?Action QFormer 分析支持:动作监督对 action-compatible representation 必要,但直接施加到继承的多模态通路会损害语言处理与对象绑定;更合理用 instruction-conditioned queries 在动作侧重组多模态表示。

### 二、双路径 Qwen(不是复制两个 4B)

同一 Qwen 两种运行模式(原始参数 θ0):

- **路径 A:Frozen Prior Path**——关所有 Adapter,`H_t^0 = sg[Q_θ0(X_t)]`,保留开放词汇/空间逻辑关系/否定/指代/组合/视觉语言知识,**永远不接受动作梯度**
- **路径 B:Adaptive Robotics Path**——顶部少量层 LoRA,`H_t^φ = Q_{θ0,φ}(X_t)`,学机器人语义修正
- 融合:`ΔH_t = P_Δ(H_t^φ − H_t^0)`,`H_t* = H_t^0 + g_t ⊙ ΔH_t`,`g_t = σ(G(H_t^0, ΔH_t, Z_t))`
- **g_t 零初始化或负偏置初始化**:训练起点完全等于原始 Qwen,只有训练证据充分时才启用残差
- 与全冻结不同:原始 Qwen 提供不可破坏先验;Adapter 学具身知识;Adapter 不能整体覆盖原始语义;动作损失最坏只改残差
- PriorVLA 类似原则:冻结 Prior Expert + Adaptation Expert + 查询机制融合,利于 OOD 与少样本适配

### 三、Qwen 不能只看指令

当前输入只有指令文本 → 只能文本理解,不知道哪个视觉对象是 cup、是否抓住、是否靠近 bowl、是否失败。新输入:

```
X_t = [L, Z_t^scene, S_{t-1}^semantic, ΔZ_t, e_{t-1}]
```

- L:原始指令;Z_t^scene:场景视觉 token;S_{t-1}:上次语义状态;ΔZ_t:最近视觉变化;e_{t-1}:预测与真实观察偏差
- 示例输入:指令 + "candidate blue cup: visible, confidence 0.91; bowl: visible; cup-bowl: outside; gripper-cup: contact uncertain; cup orientation: tilted 12°" + "attempted closing gripper; expected grasp; observed cup motion minimal"
- 输出:完整任务在当前世界中的语义解释(manipulated object/target/goal relation/persistent constraint/current status/next semantic focus/done),**不是离散子任务**,动作仍由 VA 连续生成

### 四、双频率:Qwen 低频主动运行(1–5 Hz),VA 高频控制(20–50 Hz)

Qwen 运行时机:episode 开始、指令改变、每 4–10 个 VA 决策、角色绑定置信度下降、任务进展停滞、未来状态预测失败、可能违反约束、completion 不确定。关键:**instruction-triggered + event-triggered**,不只是指令变化时运行一次。

### 五、输出 Task Contract,不输出子任务 ID

readout tokens:`[GLOBAL_GOAL][MANIPULAND][TARGET][RELATION][CONSTRAINT][COMPLETION][FAILURE][UNCERTAINTY]`,`C_t = CrossAttn(Q_contract, H_t*, H_t*)`。**废除语言均值池化**(`summary = (language_key * mask).sum() / mask.sum()`——均值抹掉 left/right、inside/beside、touch/do-not-touch、多实体角色、否定作用域)。输出分两部分:连续 latent Contract(直接进 VA 和 Flow Head)+ 可解释 Contract Head(LM Head 或分类头输出 JSON,用于训练/诊断,部署不必生成长文本)。

### 六、动作梯度进入 Qwen 的方式:梯度分工

不能让 L_FM 以原权重直接更新全部 Qwen LoRA。分工:

| 损失 | 更新 VA | 更新 Adapter | 更新 Qwen Base |
|---|---|---|---|
| 动作 L_act | 是 | 弱更新 | 否 |
| Contract L_contract | 可选 | 强更新 | 否 |
| Role Grounding L_role | 是 | 强更新 | 否 |
| Semantic State L_state | 是 | 强更新 | 否 |
| Counterfactual L_CF | 是 | 中等更新 | 否 |
| Paraphrase L_para | 是 | 强更新 | 否 |
| Anchor L_anchor | 否 | 是 | 教师路径冻结 |

`∇_φ = η_act ∇_φ L_act + ∇_φ L_semantic`,η_act ∈ [0.05, 0.2] 首轮搜索范围。Adapter 主要由"语义是否正确"训练,动作损失轻微调整;VA/Flow Head 承担主要动作拟合。

### 七、表示锚定

- `L_anchor = Σ_l ‖Norm(H_φ,l) − sg[Norm(H_0,l)]‖²`(选定的中高层)
- `L_geometry = ‖G_φ − G_0‖_F²`,`G = Z Zᵀ`(指令间相对几何,阻止 12 条指令挤成同一向量)
- 顺序:先 frozen anchor → 再受限 Adapter → 最后 SAM 稳定;SAM 不能替代语义监督(近期报告:平坦性保持优化显著缓解 instruction blindness)

### 八、只微调这些位置

第一版不碰 q/k/v/o/gate/up/down 全部。最保守:顶部 4–6 层、只 q_proj/o_proj、rank 8 或 16、Adapter 输出作 residual、base 永远冻结。稳定后逐步试:顶部 v_proj、多模态 connector、Contract readout、scene-token projector。暂时不动:全部 MLP、底层 token embedding、所有层 K/V、final norm。Evo-1 强在**两阶段**(先冻结建接口、再解冻联合细化),不是冻结到底。

### 九、Xbot 代码改法

- `QwenTextBackbone` → `QwenSemanticBackbone`:`encode_prior`(no_grad)/`encode_adapted`(顶部 LoRA)/`compile_task(instruction, scene_tokens, previous_semantic_state, execution_error)`;不删多模态视觉模块/LM Head/中间层
- `SemanticContract`:prior_tokens/adapted_tokens/fused_tokens + role/goal/constraint/completion/uncertainty tokens
- `build_language_cache` → `build_semantic_cache(contract)`(immutable prior K/V + adaptive residual K/V + dynamic semantic-state K/V)
- VACouplingLayer:物理与语义两条注意力,`action += physical_update + semantic_gate * semantic_update`
- FlowMatchingHead 每层直接读 action_condition + semantic_state + prior contract + adapted residual
- end_to_end.py:禁止 LoRA + L_FM only;E2E 必须接 paired counterfactual batch(当前 pair loss 直接设零是 LoRA 只拟合动作的重要原因)

### 十、三阶段训练

- **Stage A**:冻结 Qwen,输入 instruction + scene + semantic history;训练 scene projector/readout tokens/role grounder/semantic-state updater/VA/action head——与静态冻结不同,Qwen 已在进行场景条件推理
- **Stage B**:启用顶部 LoRA,`H* = H0 + gΔH`,训练 L_contract + L_role + L_state + L_anchor + 0.1·L_act
- **Stage C**:轻度联合细化;仅当指令 embedding 有效秩、paraphrase 一致性、same-state command-fork、held-out composition、MT50 median、zero-success 数量全部保持/提升才继续,否则退回 Stage B

### 十一、对 MetaWorld 泛化的边界

Dual-path 能解决:Qwen 不再是静态 Encoder、能学具身语义、先验不被覆盖、语言/场景/执行形成动态闭环、未见表达更可能泛化。**不能**弥补:轨迹阶段覆盖不足、无 DAgger/recovery 数据、64-token 全局池化丢接触细节、6 步开环执行、动作标签与执行不一致、训练曝光量远低于 Evo-1。完整路线 = Dual-path + role-grounded state transition + full-trajectory data + recovery data + receding-horizon control。机械臂可能"理解正确但执行失败"。

### 最终判断

完全冻结 + 只读一次最后层 hidden 确实无法充分使用 LLM;正确修正不是恢复全局 LoRA,而是:**冻结原始 Qwen 作为不可破坏先验 + 训练受约束的机器人语义残差 + 让 Qwen 低频读取场景与执行反馈**。一句话:**冻结的是原始能力,不是冻结推理过程;微调的是具身语义残差,不是让动作损失重写整个语言模型。**
