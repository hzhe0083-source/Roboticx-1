# MT50 SOTA 修改计划 v2（2026-08-09，三方审查整合：Codex 代码 + Claude 算法 + Grok 数据）

目标：MetaWorld MT50 单卡（RTX 3080 Laptop）冲击 70-85%（对标 SmolVLA 57.3% / Evo-1 80.6% / πRL 85.8%）。
原则：协议对齐吃满 BC（Evo-1 配方）→ flow-noise PPO 收尾（πRL 路径）；机制创新不进主线关键路径。

---

## 三方审查共识

| 主题 | Codex（代码） | Claude（算法） | Grok（数据） |
|---|---|---|---|
| 短轨迹 | — | **误诊，实为 chunk 太短（H=8）** | 数据集正常；BC 上限 55-70% |
| 长轨迹重采 | 需要真反馈专家（scripted 闭环） | **否，H=48 即可，重采是数天工程换不确定增益** | 方案 A 可选（CPU 数小时） |
| servo | 不可辨识（联合训练 base 吸收） | **实测死亡（三重零门）** | 微扰数据是伪恢复 |
| 恢复数据 | 伪恢复（开环重放非反馈专家） | 转评估集/RL 课程 | 扩 49 任务需重做（384 帧+统一归一化） |
| chunk | execute-6 正确 receding horizon | **H=48/execute-12 主解** | H=8→16/32 值得 |
| RL | PPO 必须重构（PolicyRuntime） | flow-noise PPO 三处改后即 πRL 对齐 | 85%+ 必经 RL |
| VA16 | — | **回 VA8（无依据+欠训）** | — |

## 已确认的真资产（E6 产出，全部保留带入 E7）

1. executed 标签（raw→clip→±1 恒等归一化，违规 0）
2. prev 契约（episode 首决策 0、其余真实前帧动作）
3. 无几何增广（--no-frame-aug-geometric）
4. 高曝光（1.2M 窗口 / ~54.6 epoch）
5. 微扰恢复管线（降级用途：评估集 / RL reset 课程）

## 必须砍掉（主线）

- servo 双新息（实测 |Δa|=0 / H=ln4 / flag=1，结构性死亡）
- fovea 双速率闭环 + 对应的 2×6-8h 评估
- multi-mode 槽 + dense-1152 读出（几何无监督，无探针证据）
- VA16 → 回 VA8（12 层随机 + lr 1e-5 欠训）
- 递归 VisualMemory（panel16 实测 mem4 −3.1pp 噪声内）
- MW 语言消融训练臂（视觉可判任务，语言≈任务码）
- 数据拼接/重采样方案（破坏动力学）

## 保留

- flow6-AdaLN + semantic cross-attn（π0 式逐层条件）
- 冻结 Qwen 语言缓存、冻结原始 V-JEPA 2.1
- prev-action 输入（−23.7pp 实证必要）+ 可选 10-20% prev dropout
- train_ppo_metaworld.py（修复后 = πRL 对齐）

---

## 阶段 0：E6 止损（已完成）

E6 已停（用户指示）。五个真资产确认保留。**不再跑 fovea/无 fovea 双评估**（servo 死亡 → fovea 评估注定同分，Claude 判决）。

## 阶段 1（E7）：协议对齐 BC —— 目标 50-65%，~4-5 天

### 数据改动
1. `scripts/make_fullframe_executed.py`：窗口 action 标签 **H=8 → 48**（0.6s ≈ 演示长度 2/3），越界位置 valid-mask（π0 同款 padding/mask 语义）
2. 归一化/prev/executed 契约保持不变（E6 真资产）
3. 微扰数据：E7 训练**不用**（降级为评估集），不再混入

### 训练改动（train.py / model.py）
4. FM loss 乘 valid-mask；`--sequence-length 4` 保留；帧窗 stride 2→4
5. `action_horizon=48`（config 项）；**memory 关闭**（砍 T=4 vs 83 步递归失配）
6. **VA 回 8 层**（继承 stage B/E1 视觉侧权重，action query 重新初始化）
7. 视觉路径回 **spatiotemporal 288 + coarse 池化**（槽/multi-mode/servo/dense 全不开）
8. **lr 1e-4**（对齐 22% 基线配方，不是 E6 的 1e-5）
9. 训练：batch 16 × 60k 步（~44 epoch），V-JEPA 冻结；flow 步数训练/部署统一 8-10（先 1h 验证 8 步无阶梯伪影）
10. 评估：`eval_metaworld.py` ACTION_HORIZON 从 checkpoint config 读；`--execute-steps 12`（flag 已有）
11. **任务分层采样（用户指示，2026-08-09）**：DataLoader 按 instruction_id 加权——hard/very-hard 任务
    （抓取/插入/工具类）窗口过采样 2-3 倍、easy 任务（按钮/关门类）降采样。理由：难任务
    失败多、样本天然少，不给更多曝光就是浪费预算。权重表按 MT50 四档难度
    （easy 0.5 / medium 1.0 / hard 2.0 / very-hard 3.0，可调）或按闭环失败率自适应。

### 代码修复（Codex P0，防崩溃）
11. perturb 尾批 drop_last（E7 不用微扰则自动规避；保留修复供后续）
12. servo reset（砍掉 servo 后无此项）
13. resume 契约严格化（架构迁移显式 flag + migration manifest）
14. 评估读 checkpoint contract（flow_steps 等不硬编码 32）

### 判据
- 49×10 闭环。若 ≥45% 且失败集中在初始布局 OOD → 触发多 start 重建（+2 天，条件项）
- 若 <45% → 跑 direct-head 对照臂（仓库现成，历史 31.8%）分离 flow 欠拟合因素

## 阶段 2：flow-noise PPO —— 目标 70-85%，~4-6 天

### 工程（1 天）
15. `train_ppo_metaworld.py`：抽出与 eval 共用的 PolicyRuntime（Codex P0-3：当前 PPO 跑的是另一套策略，必须携带 dense/flow-semantic/servo/prev/memory/normalization——E7 砍掉 dense/servo 后接口简化但仍需统一）
16. 向量化 8-16 env 批量前向（3080 上 RL 墙钟最大项，5-10×）
17. FLOW_STEPS 32→8（πRL RL 期 denoise=4；统一 8）
18. CLIP 0.1→0.2（πRL Table 13）
19. 修 PPO minibatch 顺序错配（logp/value scatter 回填）+ 真实 bootstrap + seed 含 episode 计数（Codex P0-4/5）
20. flow-noise 数值保护：finite guard、approx-KL、clip fraction、KL early-stop、unbiased=False（Codex P1）

### 训练
21. MT10 验证链（`logs/vla_rl_mt10.sh` 骨架）~1 天 → 全 49 任务（失败率加权采样），~2M env steps
22. 奖励稀疏 0/1 无 shaping；无 IL-KL、无 BC 混合；防坍塌三保险：sigma 下限 0.02、approx-KL 监控提前停、每 50 iter 快照 + 成功率回退自动 revert

## 阶段 3（并行/收尾，GPU-轻）

23. S1/S2 command-fork 评估（MT50 天然 ~10 对同场景任务：door/window/drawer/faucet open↔close）——论文核心主张唯一强证据，最终 checkpoint 跑一次
24. 四件套（blank/wrong/swap/taskid）最终模型辅助表（先修：共享噪声 bank、hidden/mask 成对替换、null token 防 NaN）
25. 微扰集作为鲁棒性评估表
26. 若冲 85%+：RL 失败 episode 采 DAgger 式数据回灌一轮 BC（对症且便宜）

---

## 时间线（单卡）

| 阶段 | 内容 | 时长 |
|---|---|---|
| 0 | E6 止损 | ✅ 已完成 |
| 1 | E7 协议对齐 BC（H=48/VA8/lr1e-4/60k） | ~4-5 天 |
| 2 | flow-noise PPO（MT10→MT50） | ~4-6 天 |
| 3 | fork 评估 + 四件套 + 回灌 | ~1-2 天（并行） |
| 总计 | | ~10-14 天到 70-85% |

## 待用户拍板

1. **长轨迹重采 vs H=48**：Claude 明确否决重采（Evo-1 同款短演示到 80.6%，chunk 才是差异）；重采是数天工程换不确定增益。推荐：**不重采，直接 H=48**。若坚持重采（可叠加，但推迟 E7 启动 1-2 天），用 scripted policy 闭环反馈专家方式（Codex 要求）。
2. **VA8 回退**：Claude 判决（无依据+欠训），推荐执行。
3. **E7 启动条件**：代码改动清单 1-14 就绪即启动。

## 工程规范（教训固化）

- kill 训练必须清进程组（`kill -- -PGID` 或 `pat="train.py --data"; pkill -f "$pat"`），kill 后 ps 验证 0 残留
- pkill -f 匹配 cmdline（fork worker cmdline = 父进程命令，pkill "pt_data_worker" 无效）
- 训练/评估归一化口径必须同文件同字典
- 评估脚本默认读 checkpoint contract，禁止硬编码
- 恢复数据必须有真反馈专家标签（开环重放 ≠ 恢复）

---

## 2026-08-09 深夜补充：Codex 数据审查（不通过→已修）+ Claude 对 C²-IRF v2 评审

### Codex 数据管道审查：5 个 P0（全部处理完毕）

1. **采集观测错拍（P0-1，最重）**：collect_long_trajectories.py 先 env.step 再保存 (o_{t+1}, a_t)——图像泄漏 a_t 执行结果、prev 错位。已改为先保存 (o_t, a_t) 再 step。**1470 条旧轨迹作废，重采已启动**（4 路并行，~2.5h）。
2. **扰动 settle 隐藏 + hold 松弛（P0-2）**：settle 12 步进入时间轴并刷新 obs；hold 改为固定目标+连续成功计数，剔除纯失败长尾（精密任务 success 抖动时放宽为"成功跨度≥80 帧即保留"）。
3. **帧窗未来帧（P0-3）**：phase1 误写 [d,d+2,d+4,d+6] → 改 canonical 历史帧 [d-6,d-4,d-2,d]（与 prepare_pnpw_features/live/eval 一致）。已编码 ST288 作废重跑。
4. **启动阻塞（P0-4）**：local_tokens 路径 config vision_dim 从 local_tokens.shape[-1] 读；frame_refs 纯 Python（已修）；正式文件移除 18.6GB 占位。
5. **resume/评估协议（P0-5）**：H8→H48 action_queries shape 迁移（不匹配键重新初始化+migration 打印）；eval flow Euler 步数从 checkpoint contract 读（训练 8 步，不再硬编码 32）。

P1 已修：executed=clip(a) 显式化（采集端）、任务级分层采样（task_w/row_count 消除窗口数偏置）。P1 待办：normalization 内嵌 checkpoint+hash、JPEG/CPU-GPU resize 一致性 hash。

### Claude 对 GPT Pro C²-IRF v2 评审（2026-08-09，Fable 5）

- 采纳：r/ν 分离、阻尼最小二乘 4×4、NMS+局部 soft-argmax（机制）、W_o=0 零初始化、clean/perturbed 共享噪声、**两阶段冻结=可辨识性真修复**、协议核查表（FabriVLA 24 维 state / Evo-1 STATE_TAKE=8 直接进论文）。
- 修正：微扰数据必须数千条级程序化生成（否则重演 100 条×2400 遍记忆化）；开销异议换成 500GB 存储/40Hz 延迟论据，判决不变。
- 拒绝（当前阶段）：中央凹前缀（block0/1 零探针证据+过度工程）；4 假设混合（H=ln4 死亡现场同款结构，需坍缩监测+kill 判据）。
- **点睛建议**：MetaWorld 仿真器白送 oracle 物体位姿——加 oracle keypoint 辅助头监督 g_t，槽几何/可见度/可辨识性三异议同时消解，成本≈0。
- **正面结论**：精度来源 = 闭环纠错带宽 > 漏斗策略 > 单帧视锐度。H=48 开环执行才是第一精度嫌疑人；receding horizon k∈{8,16,48} 是零成本 eval 消融。dense/foveal 必要性用 oracle keypoint 探针一次性证伪。

### E7 行动增量（按 Claude 建议）

- E7 原样跑：ST288+H48 BC 60k；eval 记录 per-task 成功 + near-miss（末尾 success 帧距离）+ `--execute-steps` 8/16/48 消融
- RL 阶段：flow-noise PPO + **特权 critic**（24 维 state 只进 critic）
- 并行臂（BC 出分后/空档）：精度任务子集（peg-insert-side/assembly/hand-insert ~15GB）H11 dense 探针，判据 oracle keypoint RMSE 对比容差带；阴性 → dense/foveal/C²-IRF 全线永久关闭，写成论文 finding
- C²-IRF v2 最小核（r_t+DLS+W_o=0+两阶段冻结+oracle 辅助）排最后，预注册 kill 判据（混合熵/κ 遥测/微扰 eval 集）

---

## 2026-08-10 凌晨补充：Codex 对 C²-IRF v2 评审（与 Claude 共振，附事实修正）

### 事实修正（Codex 核查官方代码）
- **FabriVLA 未用 24 维特权 state**：state_dim=24 是 padded 接口，官方 normalization 仅 4 raw 维，eval clamp 到 raw_state_dim=4 补零。90.0% 是 RGB+4D EEF。本地 protocol_verification_evo_fabri.md:181-205 自相矛盾（引用了 clamp 却写 8D）——需修正。
- **Evo-1 H=50 只执行 15 步重读**，不是长开环。
- **MT50 精密任务判定容差 2-7cm 量级**（hand-insert/peg-insert 源码）——"亚厘米"非 MT50 硬需求。
- **πRL Flow-Noise 不需要 √dt**（每步 covariance=σ²，非 SDE）——撤回旧审查该条。

### 总判决
C²-IRF v2 不并入 E7、不做 RL 前置。保留支线：Dense H11 Step 0（判据：held-out chunk0 error ≥20% 或同种子精度任务闭环 +5pp 且 CI 不跨 0）+ r/ν 分离。两阶段冻结仅解决 base/correction gauge，servo 内部 μ/g*/J/K 仍不可辨识（--servo-only 还在训 reader；perturb 包 H=8 不能与 E7 H=48 混批）。中央凹实测 21Hz 上限 + norm off-by-one，条件采纳。

### 归因 2×2（Codex）
| | ST288 | H11 dense |
|---|---|---|
| clean | A | B |
| 真反馈 perturb | C | D |
B−A=视觉粒度；C−A=数据收益；(D−C)−(B−A)=dense 对恢复的额外价值。reader 参数量/seed/步数/执行 cadence 必须匹配。

### E7 硬门（Codex）
- E7 ≥45% 才进 RL；<45% 先 direct-head 诊断
- RL 前 PPO 四硬门：E7 同构 PolicyRuntime、minibatch logp 顺序、真实 bootstrap/seed/500 步覆盖、KL/finite/memory/checkpoint 契约
- servo/fovea 严格串行：几何 probe → frozen-base servo → zero-gain/gain-shuffle/wrong-role/open-loop 消融 → fovea（H2 adapter 已训练 + p95<25ms + 异步刷新 p95<50ms + flag 不饱和）
