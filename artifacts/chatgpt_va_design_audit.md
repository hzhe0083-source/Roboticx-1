# 审查报告:ChatGPT 语言 VA 架构讨论 vs Xbot(ORA0)仓库现状

- **审查日期**:2026-08-07
- **审查对象**:`artifacts/chatgpt_va_design_discussion.md`(ChatGPT Pro 三轮对话存档)
- **对照代码**:`va_compound/model.py`、`va_compound/backbones.py`、`train.py`、`eval_metaworld.py`、`logs/*`、`VA_COMPOUND_REPORT.md`
- **审查方法**:对话断言逐一与仓库代码/日志核对;建议按"已实现 / 部分实现 / 未实现"分类;给出独立判断与优先级

---

## 1. 事实核查表(对话断言 vs 仓库证据)

| # | 对话断言 | 仓库证据 | 结论 |
|---|---|---|---|
| F1 | 70M V-JEPA | 代码用 V-JEPA 2.1 ViT-B/384(`backbones.py:170`) | ⚠️ 非 70M 档,是 ViT-B 级(约 90M+ 参数量级);对话与代码口径不一,需澄清目标 |
| F2 | Qwen3.5-4B | 代码默认/缓存均为 `Qwen/Qwen3.5-2B`(`backbones.py:215`) | ⚠️ **仓库是 2B 不是 4B**;若用户计划换 4B,对话并未在仓库中落地 |
| F3 | 约 43M VA Core | `VA_COMPOUND_REPORT.md:551`:43.5M trainable | ✓ |
| F4 | 单 start 2488 样本 → multi-start 9927 | 报告 208/231 行;v5 数据实测 9927×4 决策点 | ✓ |
| F5 | 闭环 7.1% → 16.3% → 17.8% | `mw_numbers_ledger.md`:7.1(35/490,已废弃) → 16.3(80/490,已废弃) → AQC 17.8(87/490) | ✓(注意:仓库最新重测口径为 13.9%,修复评估缺陷后 CI 重叠,17.8 是 AQC 口径) |
| F6 | Evo-1 80.6%,每任务 50 demo,10 trials×5 runs | 报告 267/433/533 行一致 | ✓ |
| F7 | 原始 Qwen 余弦 0.7647;LoRA 后 0.9992;只训 LoRA 0.9989 | `logs/cosine_*.log`:original **0.8573**;B40k(LoRA)**0.9994 COLLAPSED**;C1(冻结)=0.8573 不变 | ⚠️ 方向一致(坍塌确认),**原始值 0.7647 未在现有日志复现**(可能是更早版本或不同数据集的测量);0.9992 vs 0.9994 基本吻合 |
| F8 | 2-task pilot:训练任务 10/10,3 个 held-out 全 0/10 | `logs/mw_pilot_direct_cl.log`:button **10/10**,其余 **4** 个任务 0/10,总 20%;`mw_pilot_c2v2_*.log` 同构(6-10/10 button,其余 0) | ⚠️ 可见日志是 5 任务评估、4 个 held-out 全 0;"2-task 训练"构成未在训练日志中记录,待核 |
| F9 | E2E 路径 pair_loss=0、future_loss=0 | `train.py:1645-1648`(e2e 分支 pair_loss 置零) | ✓ |
| F10 | ~21.4% expert actions 超环境范围被裁剪 | `make_v5_executed_actions.py` 的背景契约(executed-clip-v5)一致;21.4% 具体数字未在日志中找到 | ⚠️ 方向一致,精确数字待核 |
| F11 | C1=冻结 Qwen 保语义,C2=LoRA 塌陷 | `cosine_libero_e2e_C1_40k.log`(=original)/`C2_40k` 存在 | ✓ |
| F12 | task-ID 对照:评估时临时编码 "task i" | `eval_mw_lang_ablation.py` 存在 | ✓(行为与描述一致,未深验) |
| F13 | 旧 Flow 6/10 vs Direct Head 10/10(button) | `mw_pilot_direct_cl.log` 10/10;旧 Flow 6/10 未在可见日志中直接复现 | ⚠️ 待核 |
| F14 | MetaWorld 无 same-state command forks,shared-CF 未用于 MW | v5 数据无 pair 结构(`--single-task` 训练),✓;报告亦如此记录 | ✓ |
| F15 | Evo-1 用 InternVL3 原生多模态、第 14 层中间特征 | 与 Evo-1 论文(2511.04555)一致 | ✓ |

**核查小结**:对话对仓库的宏观理解准确(F3/F4/F5/F9/F11/F14),坍塌问题方向正确(F7);**主要出入在模型规格(F1/F2:70M V-JEPA、4B Qwen 与仓库实际不符)和个别 pilot/cosine 数字(F8/F13:4 个 held-out、0.8573)**。

---

## 2. 建议 vs 现状映射

### 2.1 仓库已实现(对话要求保留/继续)

| 对话建议 | 现状 |
|---|---|
| shared-source CF 反事实损失(feature 路径) | ✓ `train.py:461-501`(`sample_pair_intervention`),配对采样器已有 |
| memory_split:受保护 evidence + task 工作区 | ✓ `model.py:216/286/310`,与对话"Semantic State 起点"吻合 |
| future latent 预测(未来特征) | ✓ `model.py:415`,但仅全局均值(对话批评成立,见 §3) |
| execute_steps / memory-reset / prev-zero 探针 | ✓ `eval_metaworld.py:81/88/96` |
| v5 executed-action 标签(修复裁剪二义性) | ✓ 49 任务全量(v5 数据实测) |
| 恢复数据收集(C² v6b,DAgger 雏形) | ✓ `prepare_mw_recovery.py` + v6b 数据(880/1000 分支) |
| SceneTeacher(Qwen 看场景的 latent conditioner) | ✓ `backbones.py:398`,对话也承认它只是蒸馏组件 |
| Evo-1 对照表 + 本地 Evoagent | ✓ 报告 433 行已登记 |

### 2.2 对话提出、仓库未实现(核心增量)

1. **QwenTaskCompiler**:保留完整 Qwen(视觉塔+LM Head),低频运行,输出显式+latent 双分支 Task Contract
2. **Role-Grounded Belief Core**:语言角色 token 跨注意力查询视觉 patch,带绑定置信度/遮挡/身份一致性的角色槽 + 循环记忆
3. **语义监督头**:progress/completion/constraint/uncertainty/role_visibility head(现在 task 无监督)
4. **独立 semantic cross-attention**:语言与视觉分开 softmax,带门控 `g_A`(现在共享一个 softmax)
5. **Language-Residual Flow Head**:`v_final = v_base + g_t·Δv_L`,可测量 ‖Δv_L‖
6. **E2E 反事实监督**:E2EDataset 加 pair_id/instruction_id,接入 CF 损失(现 E2E pair=0)
7. **MetaWorld-CF 数据**:同场景不同指令 / 同义改写 / 组合留出 / 不可能任务
8. **阶段均衡采样**:pre-contact/contact/object movement/constraint-critical/near-completion/recovery
9. **task-ID 从头训练对照**(模型 B 用可学习 embedding)
10. **泛化统计口径**:median / zero-success ratio / N_eff / difficulty buckets
11. **E0 判决实验**:官方 Evo-1 权重进入同一本地 evaluator
12. **DAgger 2–3 轮**(当前只有单轮 v6b 恢复数据)

### 2.3 对话明确要求"暂不做"的(与仓库现状一致)

Qwen 全量 LoRA、PPO(仓库有 `train_ppo_metaworld.py` 但对话建议 IL 达标前不上)、全量 V-JEPA 解冻、EVSM/PlanResampler/多 Memory 叠加——与仓库"开关多但语言问题在最上游"的判断一致。

---

## 3. 逐部分审查意见

### 3.1 诊断部分(第 2、3 轮)——整体正确,认同

- **"两问题必须分开解决"(表征塌陷 vs 策略不用语言)**:与仓库证据完全吻合(C1 冻结=0.8573 不塌,B40k=0.9994 塌;wrong-instruction 开环敏感但闭环无跨任务泛化)。这是对话最有价值的贡献。
- **"语言敏感性 ≠ 语言泛化"与 task-ID 对照批评**:正确。`eval_mw_lang_ablation.py` 的 task-ID 条件是 OOD 输入,不能证明"语言内容 > task identity"。仓库论文里相关结论需要降级表述。
- **"MetaWorld 主要测不到语言理解"**:正确且重要——49 任务场景差异大,视觉足以区分任务;同分布成功率提升主要来自控制/数据。这与项目目标("提升语言理解")存在张力:**MetaWorld 不是检验语言理解的基准**,应靠 LIBERO(仓库已有)和 MetaWorld-CF 检验。
- **"P(success)≈P(understand)·P(ground)·P(control)·P(recover)"**:框架合理,与 7.1→16.3(数据覆盖)→17.8(语言接口)的增量排序一致。

### 3.2 架构建议(第 1、2 轮)——方向合理,但注意三点

1. **Task Contract 的显式符号分支(LM token CE)**:有价值但工作量/数据标注成本高。最小版可先只做 latent role tokens + 监督头,显式文本分支后置——对话自己的"最小可行改造"也认可这一点。
2. **role-query resampler 取代 mean pooling**:正确诊断(mean pooling 压掉关系/否定作用域)。实现上 role query 的语义初始化依赖 tokenizer 词汇,需要验证 8 个 role token 能否稳定绑定语义。
3. **独立 semantic attention + 语言残差 flow**:设计中最好的一点是**可测量性**(‖Δv_L‖、gate g_A 的分布)——这能把"语言是否被使用"从推测变成消融指标。建议优先落地这一点,它是整个重构的"仪表盘"。

### 3.3 对话可能低估/未覆盖的点

- **记忆递归深度的部署缺口**:对话全程未提 `--memory-reset-every`(训练 4 步 vs 部署几十步的契约缺口)。仓库已有对照开关,应纳入 E1 探针。
- **动作归一化的特殊性**:q01/q99=±1,反归一化是恒等变换;对话未讨论归一化对多任务共享的影响(不同任务动作分布差异 vs 共享归一化空间)。
- **chunk 重叠**:训练 8 步 chunk 与 6 步决策间隔的重叠语义未讨论。
- **Evo-1 的 80.6% 复现前提**:对话建议 E0(官方权重本地评估)但未提协议差异(官方 10 trials×5 runs、50 demo/任务、corner2 相机是否一致)——仓库报告已登记 Evoagent 本地仓库,这一步应最先做。

### 3.4 与仓库记录的出入(需澄清)

- F1/F2:对话按"70M V-JEPA + Qwen3.5-4B"设计,仓库是 ViT-B + 2B。**如果 4B 是目标,应先把接口重构完成再换模型**——对话自己也说了(否则只是更贵的 Encoder)。
- F8:对话说 "2-task pilot 3 held-out",仓库可见日志是 5 任务评估 4 held-out 全 0。结论不变(无跨任务泛化),但引用口径要统一。

---

## 4. 独立判断与优先级

### 我的排序(结合仓库当前状态)

| 优先级 | 动作 | 成本 | 回答的问题 |
|---|---|---|---|
| P0 | 当前 `mw_v5_direct_40k` 完成后全 49 任务评估 + `--memory-reset-every 4` / `--prev-zero` / `execute_steps=1` 探针 | 零训练成本 | 基础控制上限、协议缺口 |
| P0 | E0:官方 Evo-1 权重进同一 evaluator | 已登记本地仓库 | 协议公平性(与 80.6% 的可比性) |
| P1 | 冻结 Qwen 干净基线(`--lora-rank 0`)+ 全量 v5 executed 49 任务训练 | 训练时间 | 止血后的真实基线 |
| P1 | E2EDataset 补 pair_id,接入 shared-CF(修复 pair=0) | 中等 | 语言反事实监督 |
| P1 | 阶段均衡采样(数据脚本) | 中等 | 数据覆盖瓶颈 |
| P2 | 语言残差 flow + ‖Δv_L‖ 仪表盘 | 大 | 语言是否被使用的可测证据 |
| P2 | role-query resampler + 语义监督头 | 大 | 语言语义落地 |
| P2 | task-ID 从头训练对照 + N_eff/median/zero-ratio 口径 | 小-中 | 泛化证据 |
| P3 | DAgger 2–3 轮 | 中-大 | 闭环恢复 |
| 暂缓 | Qwen 4B、PPO、全量解冻、EVSM/PlanResampler 叠加 | — | 避免在语言接口修复前堆开关 |

### 结论

对话的整体诊断(语言接口被降级 + 动作损失破坏语义 + E2E 无反事实监督)与仓库证据一致,路线图(先控制与数据 → 再语言绑定 → 最后 held-out 泛化证明)合理,其中 **‖Δv_L‖ 可测性设计**和 **task-ID 对照批评**是值得直接采纳的两点。

但需要注意:**对话本身没有引入新的原理**,它是对 Evo-1(两阶段+原生多模态)、POT-VLA(角色+谓词验证)、异步 Fast-Slow VLA 等已有思路的组合;其价值在于把仓库的问题定位准确,并给出了可执行的优先级。真正的风险是过度工程化——在 P0/P1 的基础控制问题解决之前,不应投入 Task Contract 重构(与对话自己的"最小判决实验 E0→E3"顺序一致)。

---

## 5. 落地进度追踪(2026-08-07 追加)

### 已完成(代码 + 测试全绿)

| 项 | 来源 | 实现 | 验证 |
|---|---|---|---|
| E2E 反事实监督(pair_id/instruction_id + shared-CF) | 第二轮诊断 P0 | `train.py` build_pair_groups/E2EDataset/PairedBatchSampler 泛化 + e2e 分支 pair loss | 测试全绿 |
| 默认冻结 Qwen(止血) | 第二轮 P0 | `--lora-rank` 默认 0;`build_e2e_policy` rank≤0 不 attach LoRA | 同上 |
| 双路径自适应(Dual-Path 核心) | 第 4 轮 | `QwenSemanticBackbone`:prior(no_grad)+ 顶部层 LoRA(7 投影全包,用户定稿)+ 门控融合(g 初始≈0)+ anchor_loss/geometry_loss | 33 测试 |
| 中间层特征 | 第 4 轮 | encode/encode_trainable/encode_with_scene 支持 output_layers | 同上 |
| apply_lora top_layers | 第 4 轮 | 只包装最后 N 层 | 同上 |
| compile_task(场景条件推理) | 第 4 轮三 | `SemanticCompiler`(scene/history/delta 投影 + readout tokens)+ `encode_with_scene(extra_embeds)` + rollout 低频重编译(compile_every) | 37 测试 |
| 三阶段训练 | 第 4 轮十 | `--training-stage a/b/c` + validate 规则 + checkpoint 阶段迁移(strict=False) | 同上 |
| E2E checkpoint 兼容 | — | semantic_gate/semantic_compiler/lora 键双前缀剥离 | 同上 |

### 第二轮完整版(进行中,子代理实现)

role-query resampler 取代均值池化(#3)、双注意力 physical+semantic(#4)、FlowMatchingHead 逐层读语义(#5)、执行误差输入 e_{t-1}(#2)、η_act 梯度缩放(#7)、LoRA 后缀可配(定稿默认全包)、token 放开(role 16/readout 16/max_length 可配)。

### 明确不做(依赖标注数据或已声明取舍)

- L_contract/L_role/L_state 监督与可解释 Contract 文本 head:依赖仿真标注(E2E 数据无 goal_state/constraint_labels 字段)
- Qwen 视觉塔直接读图:用 V-JEPA tokens 投影替代
- 事件触发重编译(停滞/置信度/预测失败):第一版只做周期触发
- eval_metaworld.py 闭环评估:compile_task 属 E2E 在线路径,feature 路径评估协议不变
