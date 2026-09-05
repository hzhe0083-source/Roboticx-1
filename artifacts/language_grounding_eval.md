# 语言 grounding 评测设计（Codex Q3 评审结论，2026-08-08）

触发：用户问单任务 FM-only 下语言 grounding 评测建议。Codex 深度评审。

## 核心结论
最小充分集 = ①开环 command-fork（动作按正确方向改变）②闭环 command-fork（改变所完成的目标）
③paraphrase + 最小语义翻转（排除 task-id/索引解释）。blank/swap/wrong/C_OL 降为辅助诊断。
**没有配对训练损失不影响做配对评测**——正结果反而证明模型获得了未显式监督的反事实控制能力。

## 三个主指标
1. **M_fork（开环主指标，command-fork routing margin）**：
   M = [d(â_A,a*_B)+d(â_B,a*_A) − d(â_A,a*_A)−d(â_B,a*_B)] / [2(d(a*_A,a*_B)+ε)]
   M>0 = 输出不仅变化，且分别更接近各自命令的正确专家分支。（需两套专家动作参考；
   只在 expert-divergence decision windows 评测，避免早期动作相同的假阴性。）
2. **C_OL（保留为辅助）**：只证明"语言变了动作也变了"。
3. **Δ_CL^cmd（闭环主指标）**：G_ij = P(实现目标 j | do(l_i))，
   Δ = ½[(G_AA−G_BA)+(G_BB−G_AB)]。同时评估两个目标，只看原目标 SR 下降不能证明执行了替代命令。

## 证据强度排序（评测矩阵）
| 强度 | 评测 | 主指标 | 能支持的结论 |
|---|---|---|---|
| S1 | 开环 command-fork（同观测/同 task-id/同 FM noise，双命令双专家） | M_fork | 指令产生方向正确的直接动作效应 |
| S2 | 闭环 command-fork（同 start state 克隆 + noise tape，双目标均可达） | G_ij, Δ_CL^cmd | 指令改变闭环目标选择（行为层总因果效应） |
| S3 | 语义选择性（未见 paraphrase 保持；单语义槽翻转） | D_semantic vs D_paraphrase | 支持语义而非表面形式/模板 ID |
| A1 | clean vs valid matched swap | C_OL, paired ΔMAE | 动作分布因果依赖 command identity（与 task-code lookup 兼容，措辞受限） |
| A2 | paired 闭环 clean vs swap | ΔSR=(n10−n01)/N | 原指令对原目标成功的必要性 |
| B1 | near-wrong（同场景同句法单槽错误） | 绝对 MAE/Δ/wrong-goal attraction | 细粒度内容敏感性 |
| B2 | blank/generic-null（如 "perform the task"） | ΔMAE, ΔSR | 文本通道依赖（mask/长度/范数均 OOD，只能写 text-channel dependence） |
| 判别 | language vs task-id/random code | SR、fork margin、paraphrase 泛化 | 自然语言超出离散条件码；决定性证据仍是 S1–S3 |

## 关键陷阱（针对本项目现状）
1. **swap 与 task-id 不可识别性**：正 swap 效应可能只是 Qwen 向量被当 49 类 codebook
   （语义假阳性）；观测泄漏 task identity 又可能使模型继续执行原任务（语言效应假阴性）。
   只有**保持 task/env identity 不变的同场景双目标 fork** 能同时绕开两者。
2. **+1210% 不能单独作为 grounding 证据**（随机错误任务主要测类别区分/OOD 崩坏）。
3. **31.8% 闭环下避免假阴性**（失败 episode 扰动无差异）：必须另加
   - 目标 grounding/progress（正确物体首次接触、目标距离下降、subgoal/progress-AUC）
   - clean-success disruption D_succ = n10/(n10+n11)（已展示能力上的语言必要性）
   - 关键分叉状态评测（按专家分支差异选点 reset，不按模型结果选）
4. **统计口径**：单位是 task/start 或 trajectory；task-macro 均值 + task→start 分层
   paired bootstrap 95% CI。10 ep/task 足以发现大效应，不宜做逐任务显著性。
5. **闭环 fork 实现**：相同序列化 start state + 环境参数 + 逐 replanning-step 预生成
   noise tape（仅同 seed 不够——两轨迹 RNG 调用次数不同）；轨迹分叉后观测不同是
   因果传播，不是配对失效。
6. **language vs task-id**：若主张"语义 grounding"需要反 task-code 证据；
   优先做同任务 command-fork + 未见 paraphrase，预算允许再加 matched task-id baseline。
7. **论文措辞边界**：主文核心句建议 "Holding the observation, task identity, and sampler
   noise fixed, changing between two feasible commands redirected the predicted action
   toward the corresponding expert branch. From cloned simulator states, the same
   intervention changed which goal was achieved."；边界句 "command-conditioned control
   on the tested language distribution, rather than unrestricted open-vocabulary
   language understanding."

## 与本项目现状的对接
- 现有资产：C_OL（col_* 脚本）、L_m、wrong(+1210%)、blank/swap（evaluate.py，仅
  flat/spatial）、task-id（scripts/eval/eval_mw_lang_ablation.py --taskid-lang）。
- 缺口：①M_fork 需两套专家动作（fork 数据，见 pair_contract_go_no_go.md 的
  同场景任务对）；②闭环 G_ij 需 sim state 克隆 + noise tape；③paraphrase 未见
  评测（Qwen 已冻结，只需文本改写重编码）。
- MT50 天然同场景对（10+ 对）是 S1/S2 的现成载体。
- Stage B 后执行顺序：先建 fork 子集评测（S1/S2），再补 S3；task-id 对照视主张而定。
