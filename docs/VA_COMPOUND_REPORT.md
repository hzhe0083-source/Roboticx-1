# 双向 VA 复合体

## 1. 核心结构

Qwen3.5-2B只在指令变化时运行一次，将全部语言token的最后层hidden states缓存为语言记忆$K_L,U_L$。V-JEPA 2.1将当前窗口编码为视觉token $V_t$；每层只保留上一时刻同层输出$M_{t-1}$作为目标视觉记忆。VA复合体使用共享Attention：

$$
Q=
\begin{bmatrix}V_tW_Q^V\\A_tW_Q^A\end{bmatrix},\qquad
K=
\begin{bmatrix}V_tW_K^V\\M_{t-1}W_K^M\\A_tW_K^A\\K_L\end{bmatrix},\qquad
U=
\begin{bmatrix}V_tW_U^V\\M_{t-1}W_U^M\\A_tW_U^A\\U_L\end{bmatrix}.
$$

$$
\boxed{
[V_t',A_t']=[V_t,A_t]+\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt d}\right)U
}
$$

每层令$M_t=V_t'$并覆盖旧记忆，因此历史长度不增长。当前Vision以Q读取上一目标视觉、Action和固定语言锚点，Action同时读取它们；命令变化时清空$M$。多层交互后得到一次性计算的语义动作条件$C_t=\operatorname{LN}(A_t^N)$。轻量Flow Head以$C_t$为条件，对完整动作块建立向量场$\dot a^\tau=v_\theta(a^\tau,\tau\mid C_t)$。部署时VA只运行一次，8步Euler积分只重复Flow Head。

## 2. 训练与验证

Qwen与V-JEPA初期保持冻结；用同一任务的连续4个视觉时刻展开VA，使动作损失通过$M_{t-1}$反传到先前视觉状态、语言投影、记忆K/V和Flow Head。对归一化专家动作$a$采样共享维度的高斯噪声$\epsilon$与$\tau\sim U(0,1)$：

$$
a^\tau=(1-\tau)\epsilon+\tau a,\qquad
\mathcal L_{FM}=\mathbb E\left\|v_\theta(a^\tau,\tau\mid C_t)-(a-\epsilon)\right\|_2^2.
$$

每个训练`pair_id`包含同一第0时刻视觉与机器人状态下的两个互斥指令。若在随机$\tau>0$比较，$a^\tau$本身会泄漏专家动作；因此辅助分支为二者使用同一噪声$\epsilon_p$并固定$\tau=0$，使Flow输入也完全相同，只保留语言条件不同：

$$
\mathcal L_{pair}=\operatorname{Huber}
\left[
v_\theta(\epsilon_p,0\mid C_i)-v_\theta(\epsilon_p,0\mid C_j),
a_i-a_j
\right].
$$

总损失为$\mathcal L=\mathcal L_{FM}+\lambda_{pair}\mathcal L_{pair}$。该项使语言目标成为解释动作差异的必要输入。训练序列必须连续且$T\ge4$；核心对照保持参数相同，但禁止Memory、Action和Language写入Vision。若递归双向结构不能提高长遮挡恢复、指令交换正确率和任务成功率，就停止结构主张。

## 3. P0 实验结论（PNPW 单任务）

### 3.1 评估协议修正（重要）

30 FPS 示范数据动作平滑，"复制上一动作"（persistence baseline）在首步阈值上天然达到 99.7%——**首步 success 指标被平凡基线主导**（见 Copycat/Past-Token Prediction 分析，arXiv:2505.09561）。所有指标必须同时报告：
- **持久性基线**（previous_action 直接复制）
- 模型相对基线的增益（chunk_mae 差值）
- chunk_mae（整块预测误差，不受平滑性主导）

### 3.2 池化变体（flat vs spatial）

| 指标 | flat (A) | spatial (B) |
|---|---|---|
| success（首步<0.05） | 90.9% | 84.5% |
| first_mae_raw | 1.228 | 1.542 |
| chunk_mae | 0.0497 | 0.0564 |
| vs 持久性基线 chunk | **-0.0055（真实增益 ~10%）** | -0.0014 |

结论：flat 保持主变体；spatial（时间均值+2D网格）损伤细粒度信息。注：以上为 4 层@10k 结果，深度探针见 3.4。

### 3.3 四流审计（gpt-5.6-sol 干预实验，32 样本）

对已训练 checkpoint 做推理时 mask 阻断，各路径误差变化：

| 阻断路径 | 误差变化 | 判定 |
|---|---|---|
| 动作→视觉 (A→V) | +7.7% | **起作用（主要反向路径）** |
| uni_a 等价（全部→视觉） | +8.2% | 与 A→V 近似 |
| 记忆→动作 (M→A) | +2.9% | **起作用（记忆走动作侧）** |
| 记忆→视觉 (M→V) | +0.4% | 几乎无 |
| 语言→任何 | ~0.0% | **单任务数据上语言是常数，无法验证** |
| previous_action 清零 | +3792% | 最重要输入 |

注意力分配（t>0）：视觉查询 V=51% M=30% A=18% L=0.2%；动作查询 V=66% M=18% A=17% L=0%。

实现审计结论：mask 逻辑、梯度链、归一化、记忆语义均无错误（35 单测通过）。V/A/M 三流真实被模型使用；语言流必须多指令数据验证。

### 3.4 深度探针（VA 层数，步数对齐关键）

| 配置 | success | first_mae_raw | chunk_mae | vs 基线 chunk |
|---|---|---|---|---|
| 4 层 @10k | 90.9% | 1.228 | 0.0497 | -0.0055 |
| 8 层 @10k | 85.7% | 1.406 | 0.0518 | -0.0034 |
| **8 层 @20k** | **99.1%** | **1.063** | **0.0345** | **-0.0207** |
| 16 层 @20k | 训练中 | — | — | — |

**关键教训：加深必须配足步数。** 8 层@10k 的"更差"是欠训练假象；8 层@20k 显著优于 4 层（chunk -31%），真实增益 4 倍于 4 层。深度有价值，方向为"加深 + 足步数"。

### 3.5 采样配置

Euler 8 步产生阶梯感（相邻步跳变 0.0045），32 步降为 0.0004（10 倍改善），同时块间重叠不一致减半。部署建议 `--flow-steps 32`。

## 4. 端到端微调管线（已就绪，待服务器额度）

- `prepare_pnpw_video.py`：原始视频帧 + 指令文本（data/pnpw_video.pt，9.85 GiB）
- `va_compound/backbones.py`：手写 LoRA（LoRALinear/apply_lora，96 层适配器）
- `va_compound/end_to_end.py`：Qwen3.5 (LoRA) + V-JEPA 2.1 (全量解冻) + VA/Flow 全量
- `train.py --e2e-data`：端到端训练模式（冒烟通过）
- 训练命令：`train.py --e2e-data data/pnpw_video.pt --steps 10000 --batch-size 8 --lora-rank 32 --unfreeze-blocks 24 --save checkpoints/pnpw_e2e.pt`

## 5. 多任务数据（本地已发现，语言流验证的关键）

本地 EvoStudio 数据 7 目录 3 任务（全部 12 维动作、30 FPS、同相机结构）：

| 任务 | episodes | 帧数 |
|---|---|---|
| pick white cube into the basket | 114 | 38,209 |
| take unguent into box | 64 | 25,623 |
| Hold the toothbrush cup... | 111 | 54,409 |
| 合计 | 289 | 118,241 |

`prepare_multi_task.py` 已就绪（合并特征提取 + 跨任务统一归一化，data/multi_task_features.pt，9602 样本）。多任务训练使语言不再是常数，语言流可学习（文献：LIBERO 2306.03310 每任务 50 条即可；RoboTwin 2506.18088 10 真机+合成）。

### 5.1 多任务实验结果（8 层@20k）

| 指标 | 值（旧口径 8 步） | 值（32 步口径复现，2026-08-05 阶段 A） |
|---|---|---|
| success（首步<0.05） | 97.7% | 94.9% [90.7%, 98.9%]（tasks=3 宏平均） |
| chunk_mae | 0.0420（持久性基线 0.0315，模型差 +33%） | 0.0484 [0.0435, 0.0531]（基线 0.0315，差 +53%） |

欠训练信号：9602 样本 20000 步 = 每样本仅 16.6 epochs（单任务为 107）。

**语言流验证（干预实验）**：
- 屏蔽语言全部路径：32 步口径复现 chunk 0.04763 vs clean 0.04763（**+0.0%**，语言冗余再确认）
- 指令交换测试（旧口径）：task 1（toothbrush）+ 错误指令 → +44%；task 0/2 → -12%/-23%
- 判定：语言流是"可计算的弱信号"——交换指令确实改变输出（路径存在），但模型主要靠视觉区分任务（3 任务场景完全不同，视觉 shortcut 主导）
- **严格语言 grounding 验证需要同场景不同指令数据（LIBERO Spatial/Object/Goal 式）**——这是数据设计限制，不是架构失败

## 6. 文献对照

- **Evo-1**（arXiv:2511.04555，CVPR 2026）：0.77B 轻量 VLA，两阶段训练（Stage 1 冻结 VLM 训动作头 = 本项目 P0 同构；Stage 2 全量微调）。两阶段 > 单阶段消融支持本项目分阶段路线
- **WAM4D**（arXiv:2606.14048）：LingBot-VA 骨干 + 空间寄存器蒸馏深度；双向寄存器消融 +1.4pp（支持 bidir_va 主张）；LingBot-VA 长历史漂移问题支持短记忆设计
- **Copycat/PTP**（arXiv:2505.09561）：动作持久性使开环指标虚高——本报告 3.1 的直接依据
- **LIBERO**（2306.03310）：语言 grounding 需多任务数据——本报告 5 的依据

**记忆 / 双向耦合类 VLA 竞品（7 篇，原文已精读落盘 papers/，完整图谱见 §10.1）**：

| 竞品 | 机制 | 关键数字 | 与本工作重叠 |
|---|---|---|---|
| ReMem-VLA (2603.12942) | 双层 EMA 循环 query + 过去图像重建 loss | MemoryBench 94.5% | 循环记忆 + 双向 connector（最接近） |
| MemoryVLA (2508.19236, ICLR'26) | 感知-认知记忆银行 + 检索 | LIBERO-5 96.5% | 记忆方向（检索式，非递归） |
| AVA-VLA (2511.18960) | recurrent belief + 主动视觉注意力，T=4 BPTT | LIBERO/CALVIN SOTA | 训练展开方式相同 |
| RB-VLA (2602.20659) | belief 状态 + world-model 自监督 | 比 π0 高 52.5/37.5pp | 固定大小记忆、每任务一次 VLM |
| CogVLA (2508.21046, NeurIPS'25) | V-L-A Coupled Attention 双向并行解码 | LIBERO 97.4%、真机 70.0% | 双向 V↔A 耦合 |
| SmolVLA (2506.01844) | 轻量 flow（数据同源） | — | metaworld_mt50 数据来源 |
| LIBERO-Plus (2510.13626) | robustness 分析 | — | Blank 干扰协议标准来源 |

定位：单点机制均有先例，本工作独特主张 = **架构内语言 grounding 因果证据（LIBERO Blank/Swap vs MW 跨环境）+ 两级损失简单性**（上述竞品均有额外监督：image loss / world-model 目标 / 检索模块——仅本工作为 L_FM + L_pair，无额外 loss）。

## 7. LIBERO 实验（语言 grounding 最终判决）

### 7.1 数据

本地 benchmark_data 发现 LIBERO-100（26GB，100 任务 × 50 episodes，LeRobot 格式，图像内嵌 parquet）与 MetaWorld MT50（43GB，49 任务，80FPS）。`prepare_libero.py` 支持 parquet 内嵌 PNG 解码 → V-JEPA 特征。

- 1 场景子集：LIVING_ROOM_SCENE2 下 4 任务（alphabet soup / butter / milk / orange juice），各 30 样本，共 120 样本（data/libero_features.pt）
- 3 场景子集：+ KITCHEN_SCENE2 + STUDY_SCENE1，12 任务 360 样本（data/libero_3scene.pt，统一 Qwen 编码）

### 7.2 语言流三件套（Clean / Blank / Swap）——2026-08-05 以可复现口径重测

**口径说明（重要）**：原表格数字（Clean mse 0.0319/0.0416、Blank +3805%/+1446%）的评估脚本未留存，无法复现；现以 `evaluate.py --perturb`（blank=语言流整体置零、swap=指令按 instruction_id 轮转 +1）重测，**flow_steps=32**（§3.5 部署口径），chunk_mse_norm = 归一化动作全块平方误差：

| 数据集 | Clean mse | Blank | Swap | 判定 |
|---|---|---|---|---|
| LIBERO 1 场景（8层@20k，libero_va8_20k.pt） | 0.00106 | **+13751%** | **+1518%** | **语言是必要条件** |
| LIBERO 3 场景（8层@20k，libero_3scene_va8_20k.pt） | 0.00254 | **+2381%** | **+607%** | **跨场景语言 grounding** |

（对照：8 步 Euler 口径下 3 场景 Blank +116%、Swap +5%——效应随采样精度大幅增强；32 步为部署标准。原报告 +3805%/+518% 等数字方向一致但精确值不可复现，已废弃。）

分任务明细（1 场景，旧记录）：blank +2726%~+5467%，swap +0%~+935%。

**这是语言流的最终判决**：之前 PNPW（常数指令）与 EvoStudio 3 任务（场景可分）的"语言弱信号"结论全部归因于数据设计限制；在"同场景不同指令"（LIBERO Spatial 式，Grok 论文 LIBERO-Plus 2510.13626 标准协议）下，语言流被证明必要且主导。**VA 复合体四流设计（含语言流）架构成立。**

### 7.3 闭环评估（LIBERO 仿真）

安装了 libero 0.1.1 + robosuite 1.4.0 + bddl 1.0.1（EGL 渲染用 mesa+surfaceless 配置），编写 eval_libero_closedloop.py。

**结果：模型 0/4 成功；专家动作回放也失败（164 步）**。

结论：训练数据（kevin_libero100）的初始物体位置/场景与 libero_90 官方环境 init states 不对齐，专家轨迹在环境中无效——**数据-环境不对齐导致闭环评估无效，不能据此判定模型能力**。正确路径：用与训练数据同源的 init states/场景重新对齐后评估（LIBERO 官方协议要求同套配置）。

**数据字段复核（2026-08-05，MW 对齐经验后重新检查）**：kevin_libero100 parquet 仅有 `observation.state`（机器人关节角）与 `observation.ee_state`，**无物体位姿字段**（碗/书本等场景物体状态缺失）——与 MetaWorld 数据（含 `environment_state` 39 维，可对齐物体+目标）不同。无法从数据恢复场景物体位置，官方环境 init 亦不匹配（§7.3 专家回放 0/4）——**LIBERO 闭环的 init 对齐在数据层面不可行**；可行路径仅剩官方 demos（`ORA/external/LIBERO` 为源码仓库，无本地 demos，需下载官方 libero_90 数据集并重做数据管线/重训）或自采数据（记录 mujoco init state）。**按停止规则：记录阻塞，LIBERO 以开环三件套为主证据**。

### 7.4 能力边界（对"保障≠能用"的回应）

| 声明 | 证据 | 状态 |
|---|---|---|
| 语言流被使用（必要） | Blank/Swap 三件套 | ✅ 证明 |
| 动作生成精度（开环） | chunk MAE 优于基线 84% | ✅ 证明 |
| 闭环任务完成（能用） | — | ⚠️ 数据-环境不对齐，无法公平评估 |

---

## 8. MetaWorld MT50 实验（语言条件，4D 动作/状态）

### 8.1 数据与协议
- 数据：lerobot/metaworld_mt50（SmolVLA 同款，49 任务 × 50 demos、80FPS、480×480 corner2）
- 特征管线：prepare_metaworld.py（parquet PNG 解码 → V-JEPA 2.1 特征，分批提取防 OOM）
- 协议（Grok 验证）：10 ep/task 闭环、四档难度（E28/M11/H6/VH5）、语言条件为 VLA 标准
- **⚠️ 数据管线 bug（2026-08-05 发现并修复）**：旧版 `metaworld_features.pt` 由 `/tmp/run_mw_batches.sh` 分 5 批提取（每批 10 任务）后合并，存在两个相互叠加的缺陷：
  1. **instruction_id 每批从 0 编号，合并未加偏移** → 全文件仅 10 个唯一 id（49 任务重叠），任务分组/消融语义损坏；
  2. **每批用各自任务子集的 1%/99% 分位数独立归一化**，合并时却只用批 0 的 q01/q99 参数 → 动作归一化空间错乱（修复版与 bug 版 actions MAE 0.236、q01/q99 差异达 3 倍）。
  - 修复：全量 49 任务单次提取（`prepare_metaworld.py --max-tasks 49`），instruction_id 0-48 唯一、统一归一化、语言嵌入一致（vision_tokens 与旧版一致，actions 差异完全来自归一化）。修复版数据 `data/metaworld_features.pt`（2488 样本/49 任务），旧文件备份为 `.buggy_instid`。
  - **影响**：旧 checkpoint（`metaworld_va8_40k.pt`）训练于 bug 版数据（每批独立归一化），在修复版数据上开环 chunk_mae 0.208、success 2.9%（见 8.2 勘误）——**必须用修复版数据重训**。

### 8.2 训练与开环结果（VA 8 层，40k 步）——勘误
| 指标 | 旧值（bug 版数据，**已证伪**） | 修复版数据实测（旧 checkpoint） |
|---|---|---|
| chunk_mae_norm（全序列） | ~~0.0436~~ | **0.2080** [0.1911, 0.2250]（49 任务宏平均±95% CI） |
| first_mae_norm | ~~0.0413~~ | 0.1779 |
| 首步 success（<0.05 阈值） | ~~83.0%~~ | **2.9%** [0.6%, 6.2%] |
| 持久性基线 chunk_mae | ~~0.1802~~ | 0.1466 |
| vs 基线 | ~~-76%~~ | **+0.059（劣于基线）** |
- **勘误说明**：旧数字（0.0436/83%）基于归一化错乱的数据——每批独立归一化使"归一化动作"语义漂移，模型学到的是批 0 参数空间下的分布；修复版数据（正确统一归一化）下旧 checkpoint 失效（宏平均口径，evaluate.py 现输出 95% CI）。
- **结论**：§8.2 的正式数字需待修复版数据重训 40k 后重评估（`/tmp/run_mw_retrain.sh`，B40k 完成后串行启动）。

**多 start 全覆盖数据重训正式结果（40k 步，32 步口径，数据源 logs/mw_full_openloop.log，2026-08-06）：**

| 指标 | 值（多 start 全覆盖） | 旧值（修复数据单 start 覆盖） | 持久性基线 | vs 基线 |
|---|---|---|---|---|
| chunk_mae_norm（49 任务宏平均） | **0.0806** [0.0709, 0.0921] | 0.0896 [0.0802, 0.0994] | 0.09304 | **-14.1%（chunk_mae -0.0132）** |
| 首步 success | **50.8%** [45.3%, 56.1%] | 31.4% | 88.5% | -37.6pp（首步阈值被持久性主导，§3.1 已知） |
| first_mae_norm | 0.07250 | 0.08548 | 0.04409 | 首步 MAE 口径受持久性主导 |

- **多 start 全覆盖数据效果**：每 episode 取 start ∈ {0, L/3, 2L/3, L}（sequences-per-episode=4），训练覆盖全程而非仅开头 0.33s → chunk_mae 0.0896→0.0806（-10%），首步 success 31.4%→50.8%（+19.4pp）。
- 9927 样本 × 4 决策点 = 39708 决策点，宏平均 + 任务级 bootstrap 95% CI（固定种子）。

### 8.3 语言流消融（错误指令）——待重训后重做
- 旧结果（~~+0.1%~~，基于 bug 版数据与损坏的 instruction_id）**作废**；重建脚本 `eval_mw_lang_ablation.py`（指令轮转 +1 mod 49 替换，决策点 0 与全序列双口径）已就绪，待重训后运行。

**多 start 全覆盖数据重训后语言消融（数据源 logs/mw_full_ablation.log，2026-08-06）：**

| 条件 | chunk0（决策点） | chunk_all（全序列） | vs clean |
|---|---|---|---|
| clean（正确指令） | 0.09118 | 0.07991 | — |
| wrong（指令轮转 +1） | 0.23066 | 0.16664 | **chunk0 +153.0% / chunk_all +108.5%** |
| taskid（task-id 替代语言） | 0.18553 | 0.14502 | **chunk0 +103.5% / chunk_all +81.5%** |
| 持久性基线 | — | 0.09304 | — |

**结论（多 start 数据下语言 grounding 保持）**：
1. wrong 显著劣于 clean（+108.5%），taskid 替代也显著劣化（+81.5%）——**wrong vs taskid 差 +27pp**：语言内容（超出 task-id 的语义细节）确实参与决策。
2. 与单 start 覆盖数据版（+182%/+148%）相比效应幅度略降——多 start 训练覆盖全程后模型对语言的依赖更均衡（clean 本身更好，0.0899→0.0799），语言仍是可用的因果条件通道。
3. clean chunk_all 0.07991 与开环 0.07990 一致（口径交叉验证 ✓）。

### 8.4 闭环评估：环境-数据一致性已确认（专家回放 18/18）
- **旧结论推翻**：此前"metaworld 3.1.1/3.0.0 均 0/6、物理模型差异"的判定是错的——真实原因是回放时未对齐数据的目标位置/物体位置（且动作语义理解有误）。
- **专家回放冒烟（`mw_expert_replay.py`，6 任务 × 3 trials）18/18 全成功**（`/tmp/mw_smoke_now.log`，2026-08-05：18 行 `EXPERT REPLAY: 1/1 success`、0 FAIL，peak_reward mean 10.0）——**闭环前置条件（≥16/18）正式满足**，本地 metaworld 3.0.0 + mujoco 3.3.0 与数据采集环境物理一致。
- 关键对齐事实（实测验证）：
  1. **动作语义 = metaworld 标准环境动作**：内部 `clip(action,-1,1)×action_scale`；数据 action 值域超 [-1,1]（±7.3）但执行时被 clip——本地默认行为即正确（clip 版手轨迹误差 0.0002 vs 无 clip 0.045）。
  2. **物体位置**：数据采集时物体位置固定（49 任务所有 episode 首帧 nut 位置相同），本地 reset 默认即一致；obs[4:7] 是 site/com 位置（非 body 位置），不能直接写回 qpos。
  3. **目标位置（peg 等）随机化生效**（每 episode 不同）：必须把数据首帧 goal（obs[36:39]）写回 `_target_pos` 并移动对应 body——缺失时 reward/success 判定必失败。
- **闭环评估公平性成立**：本地可公平跑 49 任务 × 10 ep 闭环（`eval_metaworld.py`，固定种子、MT1(env_name, seed=42) 采集同款构造、corner2 相机位置修正、不 flip——实测数据图像与本地渲染一致，flip 反而 MAE 55）。**待修复版数据重训完成后执行**。
- 闭环脚本与训练管线对齐清单（2026-08-05 逐项验证）：图像归一化同 mean/std ✓、resize 同 bicubic+antialias ✓、窗口帧时间降序 [d, d-2, d-4, d-6]（V-JEPA 3D patch 编码对帧序敏感）✓、决策节奏 4 帧窗口/6 帧步进 ✓、动作反归一化（模型输出 norm → 环境原始动作）✓、previous_action 用归一化空间 ✓、输出宏平均 + 任务级 bootstrap 95% CI ✓。

### 8.5 对比表（文献基准，2026-08-06 Grok 二次查证 + 口径脚注规范；基线数字全部引用原文，不自行训练）
| 模型 | MetaWorld MT50（闭环） | LIBERO avg | 说明 |
|---|---|---|---|
| FabriVLA (2607.08575) | 90.0% | — | imitation SOTA |
| LA4VLA (2606.27295) | 87.5% | — | |
| π0+ALAM (2605.10819) | 85.0% | — | |
| Evo-Depth (2605.14950) | 84.4% | — | |
| Evo-1 (2511.04555) | **80.6%** † | 94.8% † | 10 trials × 5 runs |
| SmolVLA (2506.01844) | **~68%** † | ~89% † | 10 trials/task，VLM-init only |
| π0.5 (2504.16054) | —（无官方 MW） | **~97%** ‡ | OpenPI/LeRobot 协议，50 trials |
| **TurboVLA (2607.27205)** | —（无 MW） | **97.7%** ‡（99.2/99.8/97.4/94.2） | 0.2B 去 LLM 主干；VLA-Adapter 协议，50 trials/task；代码权重公开 |
| π0 (2410.24164) | 47.9% § | 94.2% ‡ | MW 为第三方转引（SmolVLA） |

**口径脚注（Grok 规范模板）**：
† MT50：50 demos/task、10 trials/task（Seo 难度划分）；SmolVLA/Evo-1 为 VLM-init only（无机器人预训练）。
‡ LIBERO：OpenVLA 报 3 seeds × 50 ep/task 均值±stderr；SmolVLA/Evo-1 报 10 trials/task；OpenPI π0/π0.5 用 50 trials/task（seed 7）——不同协议不混比，单表内统一一族。
§ π0 MT50 数字为第三方转引（Shukor et al.），非原论文。
| **本工作（开环，32 步口径）** | **0.0896** [0.0802, 0.0994]（chunk_mae_norm，49 任务宏平均；持久性基线 0.1466，-39%；success 31.4% vs 基线 71.5%——首步阈值被持久性主导，见 §3.1） | 开环口径，与闭环文献数字不可直接比较 |
| **本工作（闭环 49×10）** | **7.1%** [2.7%, 12.7%]（35/490） | 固定种子 1000×task+trial；⚠️ 数据覆盖局限：训练样本仅含每 episode 开头 0.33s（4 决策点），闭环 500 步几乎全程 OOD——分布内任务（drawer 10/10）证明管线有效；多 start 数据重建后重测 |

### 8.6 LIBERO 端到端（配置 B）状态
- 10k 中间：chunk 0.123（端到端欠训，确认需 40k——与 PNPW 过拟合结论一致）
- 40k 续训：2026-08-05 完成（checkpoints/libero_e2e_B40k.pt，30000 步续训），终评四步自动执行

**终评四步（32 步口径，数据源 /tmp/pipeline_final_eval.log，2026-08-05 夜）：**

| 步骤 | first_mae | chunk_mae | success | 备注 |
|---|---|---|---|---|
| ① 正常（360 样本） | 0.05354 | **0.07591** | 59.0%（首步<0.05） | 持久性基线 first 0.09139 / chunk 0.14391；vs baseline chunk **-47%** |
| ② Blank（语言切除） | 0.05468 | 0.07702 | — | vs ① **+1.5%** |
| ③ Swap（指令轮转） | 0.05523 | 0.07796 | — | vs ① **+2.7%** |
| ④ MW 对照 | 0.08548 | **0.0896** [0.0802, 0.0994] | 31.4% [26.0%, 37.2%] | = 修复版重训 40k 开环（§8.2，pipeline Step 3 已执行） |

**⚠️ 与冻结基线（配置 A）对比——重要发现（2026-08-05）：**
- B40k 端到端微调后 chunk_mae 0.0759 **劣于**冻结 8 层@20k 的 0.0368（阶段 A 实测，32 步口径），success 59.0% vs 92.4%；
- 语言敏感性几乎消失：B40k Blank +1.5%/Swap +2.7% vs 冻结模型 Blank +2381%/Swap +607%（§7.2）；
- **确诊根因（2026-08-05 实测）**：把 B40k 的 LoRA 权重加载进原始 Qwen 后编码 12 个指令，平均余弦相似度 **0.9992**（原始 Qwen 0.7647、随机基线 0.0829）——**LoRA 微调把语言嵌入空间压平成近同一向量**，语言条件在推理时无区分度；
- **叙事修正（2026-08-05 Codex 审计）**：配置 A 的 contract 实测 `paired_multi_goal=False`——A 也是 FM-only 训练！"L_pair 保留语言 grounding"的说法不成立。A vs B 的差异实为**冻结特征路径（V-JEPA 缓存 + Qwen 嵌入缓存）vs e2e 微调路径（解冻 V-JEPA 12 block + Qwen LoRA）**：冻结路径保留语言 grounding（+2381%），e2e 微调路径失去（+1.5%）。V-JEPA 微调与 Qwen LoRA 各自贡献待 2×2 对照（§10.5）区分。
- **次要因素**：B40k VA 仅 4 层（冻结为 8 层）；previous_action 跨 episode 泄漏（§10.5 P0-A，待修）。
- **论文定位**：主结果以配置 A（冻结特征，FM-only）为语言 grounding 证据；B40k 作为 e2e 微调对照如实报告。

- **语言流机制结论（B40k 修正）**：40k 充分训练后，LIBERO 语言流作用从 10k 时的"同场景多指令必要"（Blank +2381% 旧口径）降为 **Blank +1.5% / Swap +2.7%**（32 步口径）——与 MetaWorld 结论（跨环境语言冗余 +0.1%）一致：**训练充分后视觉主导，语言流冗余**。论文叙事按此修正（语言流是"渐进冗余的通道"，而非"必要通道"）。
- **对比说明**：冻结基线 0.0367（阶段 A 实测，特征输入 12 任务同集）与 B40k 0.07591（视频输入 e2e）为**同一任务集两种模态**——B40k 是端到端全链路（含视觉塔误差），冻结是组件化（干净特征）；报告对比注明模态差异。vs 10k 中间值 0.123（同视频模态，阶段 B 32 步口径）待跑后填入。

---

## 9. 统计口径与可复现性（2026-08-05 审计）

**评估口径（论文统一声明）**：
- **flow_steps=32**（Euler 部署步数，§3.5 实测 8 步阶梯感 0.0045 vs 32 步 0.0004；32 步亦显著改善 chunk_mae，如 LIBERO 3 场景 0.196→0.037）
- 指标：chunk_mae_norm / chunk_mse_norm（归一化动作全块误差）、first_mae_norm、success（首步 <0.05 阈值）
- **宏平均**：任务（instruction_id）等权，任务内样本先平均（`va_compound.statistics.macro_bootstrap_ci`）
- **95% CI**：bootstrap 百分位（B=2000、固定 seed=0，重采样单元=任务）
- **持久性基线并报**（复制上一动作，§3.1 依据 arXiv:2505.09561）
- **固定种子**：闭环评估 1000×task_index + trial
- 语言扰动：`evaluate.py --perturb`（blank=语言流置零、swap=指令轮转 +1）、`eval_mw_lang_ablation.py`（MW 消融）

**可复现性审计结果（2026-08-05）**：
- ✅ §3.2/§3.4（PNPW flat/spatial/深度探针）：实测 0.0492/0.0552/0.0324 vs 报告 0.0497/0.0564/0.0345——一致（8 步口径）
- ✅ §7.2 语言三件套：新口径（32 步）重测，Blank +13751%（1 场景）/ +2381%（3 场景）——定性结论更强；原数字（+3805%/+1446%）脚本未留存，已废弃
- ⚠️ §8.2 MW：旧数字基于归一化错乱数据，已勘误；正式数字待重训
- ⚠️ §5.1 多任务：待复现（B40k 完成后 GPU 评估）
- 原则：**所有论文数字必须由现有脚本（evaluate.py / eval_e2e.py / eval_metaworld.py / eval_mw_lang_ablation.py）重新生成**，未留存脚本的数字一律废弃

---

## 10. 结构审计与竞品定位（2026-08-05，Codex gpt-5.6-sol 审计 + 7 篇竞品原文精读）

### 10.1 竞品图谱（arXiv 号经 API 验证，原文已落盘 papers/）

| 竞品 | 记忆机制 | 关键数字 | 与本工作重叠 |
|---|---|---|---|
| ReMem-VLA (2603.12942) | 双层 EMA 循环 query（帧级+块级）+ 过去图像重建 loss | MemoryBench 94.5% vs MemoryVLA 1.5% | **最接近**：循环记忆+双向 connector |
| MemoryVLA (2508.19236, ICLR'26) | 感知-认知记忆银行+检索 | LIBERO-5 96.5%、Bridge 71.9% | 记忆方向（检索式） |
| AVA-VLA (2511.18960) | recurrent belief + 主动视觉注意力，**T=4 BPTT** | LIBERO/CALVIN SOTA | 训练展开方式相同 |
| RB-VLA (2602.20659) | belief 状态 + world-model 自监督目标 | 比 π0 高 52.5/37.5pp | 固定大小记忆、VLM 每任务一次 |
| CogVLA (2508.21046, NeurIPS'25) | V-L-A Coupled Attention（双向动作并行解码） | LIBERO 97.4%、真机 70.0% | **双向 V↔A 耦合** |
| SmolVLA (2506.01844) | —（轻量 flow） | — | 数据同源（metaworld_mt50） |
| LIBERO-Plus (2510.13626) | —（robustness 分析） | — | Blank 干扰协议标准来源 |

**定位结论**：单点均有先例（双向耦合→CogVLA；递归记忆→ReMem/AVA/RB）；本工作独特主张 = **架构内语言 grounding 因果证据（LIBERO Blank +2381%/Swap +607% vs MW 跨环境 +0.1%）+ 两级损失简单性**（ReMem 有 image loss、RB-VLA 有 world-model 目标、MemoryVLA 有检索模块——仅本工作无额外监督）。

### 10.2 结构弱点清单（Codex 审计，按优先级）

**P0-1 previous_action 闭环自激**：`错误动作 → previous_action → A → V → M → 下一动作`。清零 +3792% 属 OOD 干预（不能单独证明过度依赖），但结合持久性基线（模型仅好 10-30%）构成闭环部署最高风险。最小改动：拆分 proprio/pa 投影 + 独立门控 + 训练时 10-30% 整体 dropout + availability bit。**不新增 loss。**
**P0-2 记忆是"单槽递归压缩状态"而非"一步记忆"**：理论感受野无限，但无门控覆盖、T=4 只训练 3 次传递、M=V' 混入动作/语言痕迹。M→V +0.4% 削弱长期递归主张但不否定双向（A→V +7.7% 仍在）。最小改动：随机 0-8 步 burn-in → 随机 T∈{2,4,8} → 两槽 FIFO+age embedding 对照 → "线索后 >4 帧遮挡"测试。
**P1 共享 softmax 模态竞争**：V/M/A/L 异分布同 softmax；语言 0.2% 注意力不说明弱（Blank/Swap 已证因果）。方案：模态独立 pre-LN/RMSNorm（≠已弃用的 QK-norm）+ 各来源实际输出贡献 ||Σα_sU_s|| 审计。
**P2 Flow 条件接口**：条件仅入口加一次 + 无 horizon offset embedding；8→32 步改善是积分误差非容量证据。方案：每 block 重注入条件 + offset embedding + 2→4 层对照 + NFE 曲线 8/16/32/64。
**P3 语言摘要 token**：AttnPool 全局 token 追加（**不替换**原始 token），α_L 从零初始化。

已核对：FFN 第二 residual 存在（x2=x1+FFN(LN(x1))，model.py L260-261）✓

### 10.3 证据表述纠正（Codex 指出，论文写作必须遵守）

1. M 不是"一步历史"——是单槽递归压缩状态；问题是稳定性而非感受野。
2. M→V +0.4% 不能证伪记忆，但"长期递归记忆"主张尚未被证明（4 帧窗口可能已覆盖动态）。
3. 语言 attention 0.2% 不支持"语言投影有问题"。
4. previous_action +3792% 是 OOD 干预，不能单独证明过度依赖。
5. **8 层@20k vs 4 层@10k 步数未对齐，"深度有价值"未被干净隔离**——需 4 层@20k 等计算量对照；LIBERO 语言证据来自 FM 条件本身（pair loss 恒为 0，无成对数据）。

### 10.4 论文动作

- 故事线重收：*Constant-memory recursive visual coupling for language-grounded control*（不并列四模块）。
- Related work 必须加：ReMem-VLA / MemoryVLA / AVA-VLA / RB-VLA / CogVLA / SmolVLA / LIBERO-Plus。
- 闭环数据后结构改进优先级（待用户决定是否实施）：previous_action 鲁棒化 > 记忆 burn-in/两槽对照。

### 10.5 B40k 语言坍塌：对照矩阵与判决实验设计（2026-08-05 用户审查 + Codex v1/v2 定稿）

**坍塌证据**：B40k 的 LoRA 加载进原始 Qwen 后 12 指令嵌入平均 cos 0.9992（原始 0.7647/随机 0.0829）——语言嵌入被压平，推理无区分度（Blank +1.5% vs A 的 +2381%）。

**叙事修正（Codex v1 实测）**：配置 A 的 contract `paired_multi_goal=False`——A 也是 FM-only！"L_pair 保留语言 grounding"不成立。真实差异 = **冻结特征路径 vs e2e 微调路径**。

**2×2 对照矩阵（Qwen × 训练协议，固定 V-JEPA 状态/VA 深度/数据/预算）**：
| 配置 | V-JEPA | Qwen | VA | 命令 | 回答 |
|---|---|---|---|---|---|
| A（已有） | 冻结（缓存） | 冻结（缓存） | 8 层 | 特征管线 | 冻结路径基准（+2381%） |
| C1（新） | 冻结 | **冻结（无 LoRA）** | 4 层 | `--unfreeze-blocks 0 --lora-rank 0` | Qwen 不参与训练时 e2e 是否仍保留语言 |
| C2（新） | 冻结 | LoRA r32 | 4 层 | `--unfreeze-blocks 0 --lora-rank 32` | LoRA 单独是否造成压平 |
| B（已有 B40k） | 后 12 block | LoRA r32 | 4 层 | `--unfreeze-blocks 12` | e2e 全解冻现状（压平） |
C1/C2/B 三格只差 Qwen/V-JEPA 解冻组合 → 定位坍塌来源。

**判决实验优先级**：
1. Open-loop 筛查（零训练成本）：NMAE + 输出位移 C_OL（`eval_e2e_col.py`，已实现）+ 持久性基线；每 perturb 重算 K/V、重置循环记忆；按 trajectory 聚合；重点看 episode 初期/动作分叉点（teacher-forcing 后半段掩盖语言效应）
2. 同场景可执行 swap 闭环（主判决）：预选 4-6 个 A 上 clean 高、swap 效应大的任务；L_m = ½[P(g1|l1)−P(g1|l2) + P(g2|l2)−P(g2|l1)]；换目标成功=服从，双失败=OOD 脆弱，同目标=语言选择性缺失
3. held-out command-fork（同视觉初态双可执行指令）
4. V-JEPA checkpoint 漂移（fixed frames，normalized cosine / linear CKA）
5. attention mass（最低）
0.9992 需用 VA 实际消费的逐层语言 K/V 中心化有效秩复核（Codex 建议：原始 hidden 未中心化余弦受公共分量影响）；真正判决 = 固定 V/M/A/噪声/时间后换可执行指令，流场差是否方向正确。

**口径陷阱**：相对增幅配绝对误差+CI；swap 轮转只是筛查；MW +0.1% 是负控制不是坍塌证据。

**P0-A previous_action 跨 episode 泄漏（Codex v1 + 审查 agent-12 确认）**：prepare 脚本在本地 decision=0 时先换全局行号再减一 → 除首条 episode 外，episode 首步读到上一条 episode 末动作；MW 侧（start 恒 0）则 previous_action[:,0]==actions[:,0,0] 自泄漏。修复：episode 首步 prev=0/BOS + availability bit（改 prepare 脚本 → 重建数据 → 重训 B/C）。

**修复候选（Codex v1 定稿）**：先修数据边界 → Qwen frozen+FM-only（C1）→ 真 e2e pair（需共享 V/M/A_prev 的配对数据，当前 libero_features_paired.pt 配对视觉不一致 max diff 1.95 不可用）→ LoRA 延迟开启/降 LR/仅收 L_pair 梯度（仍两级 loss）→ V-JEPA/VA8 最后做。正交正则明确不做（第三 loss 项 + 可能错误拉开同义指令）。

**架构级第一性原理（Codex）**：纯架构无法保证语言被使用（cross-attn/FiLM/AdaLN/MoE 均有"语言投影归零"常数解）——统计可识别性需要反事实数据；不新增 loss 的架构手段：语言独立 cross-attention（与 V/M/A 更新相加）、每 flow block 重注入 frozen-language FiLM/AdaLN。

**文献口径**：直接依据 Knowledge Insulation (2505.23705)；π0.5 (2504.16054) 语言评测设计；Evo-1 仅动机引用。空白：V-JEPA 冻结/后层/LoRA/全量 × LDS 对照表。

**最安全主张**（限定 V-JEPA 2.1 ViT-B / Qwen 2B / LIBERO）：允许视觉编码器/语言塔端到端更新可重复降低策略对反事实指令的行为选择性；表征保持/冻结可在相近 clean 成功率下缓解。

**Codex v2 定稿补充（2026-08-05 夜）**：
- **C 必须分段训练对齐 B**（B = 10k + resume 30k）：C1/C2 均 `10k → resume 30k`（脚本 /tmp/run_config_c1.sh、/tmp/run_config_c2.sh），不能从头连续 40k（resume 不恢复 AdamW/RNG，属 warm start，两段式是 B 的实际协议）
- **冻结 V-JEPA 保持 eval()**：train.py 已修（`e2e_model.train()` 后对无可训参数的子模块置 eval）
- **C_OL 定稿公式**（eval_e2e_col.py 已重写）：clean/swap 每决策点复用同一 flow 噪声 z；主指标 = 首执行动作位移 `C_exec = mean|a^clean_{t,0} - a^swap_{t,0}|`，次指标 = 完整 chunk；同噪声报 E_clean/E_swap/配对绝对差；任务级重采样配对 bootstrap CI；聚合 decision→trajectory→task 宏平均
- **L_m 定稿**：D=½[P(g1|l1)+P(g2|l2)]，O=½[P(g1|l2)+P(g2|l1)]，L_m=D−O；CI 重采样完整 matched trial block；每个语言 rollout 同时跑两个 goal scorer
- **锚定正则定稿**：锚 V-JEPA block 11 final post-norm 的 post-flat-pooling 64 tokens（VA 实际消费接口）；norm = per-token channel L2，loss = mean‖u_θ−sg(u_0)‖²；f_0 用 eval+no_grad 固定、由 prepare 缓存 fp16 target；λ pilot {0.01, 0.1, 1.0} 首选 0.1，以视觉参数上 anchor/FM 梯度范数比 0.1–0.5 校准；机制阶段保持 pair=0；**判决门：若 C2 的 Qwen cosine 仍 ~0.999 → 主因是 LoRA+FM-only，放弃视觉 anchor 转向语言主干隔离/L_pair**
- **论文表述边界**：现在只能写"B40k 出现命令表征区分度丧失（cos 0.9992），不证明端到端 VLA 普遍语义坍塌"；B/C matched 完成后才可写"V-JEPA 12 block 更新降低反事实行为选择性"；Anchor 赢过 drift-matched Low-LR 后才可写锚定有效性。对照表：Standard(12,1e-5,λ=0) / Low-LR(12,3e-6~3e-7,0) / Anchor(12,1e-5,λ*) / C(0,–,0)，正式报告 ≥3 matched seeds，统一分组梯度裁剪。

---

## 11. 轻量对比（2026-08-05 实测，bench_inference.py）

| 项 | 值 |
|---|---|
| 动作头参数量（4 层 VA + Flow） | **43.51M**（不含冻结 Qwen/Vision） |
| 部署时延（32 Euler 步，4 层 VA 单次前向） | **24.66 ms/决策 → 40.6 Hz** |
| 对比锚点 | SmolVLA（数据同源轻量 flow）：0.45B 总参；π0 系列 3B+ 需双 A100 部署 |

轻量主张证据链：VA 仅 8 层递归（参数量 ≪ VLM）、Flow Head 8 层、无扩散迭代采样（flow 一次性积分）、单卡 24GB 训练。

### 11.2 VLA-RL 核心实验设计（2026-08-05，Grok 查证 VLA-RL 惯例 + 用户确认）
- **定位：论文核心实验之一**（非可选对照）——用强化学习微调 VA 复合体 VLA 本身
- **方案（πRL/π0-RL 风格）**：从 MW fix checkpoint（IL）初始化，冻结 V-JEPA+Qwen 特征，仅微调 4 层 VA 动作头 + critic，MetaWorld sparse success reward，PPO/GRPO
- **预期量级（Grok 2026-08-06 二次查证修正）**：πRL arXiv:2510.25889 **MT50 数字在 Table 6**：π0 Flow-Noise 85.8%（91.1/81.8/78.3/92.0 难度分桶）、π0 Flow-SDE 78.1%、π0.5 Flow-SDE 70.7%、SmolVLA 68.2%、π0 SFT 50.8%（lerobot metaworld_mt50 2500 轨迹 SFT，稀疏成功奖励；**trials/task 原文未钉死**，对比须脚注）；πRL LIBERO 97.6% 是另一基准不可入 MT50 列；**ReinFlow 无 MT50 数字**（fig10 旧锚点已修正）；DPPO ICLR'25 1-57→86-97%
- **关键工程点**：flow matching 策略的 log-prob 计算（PPO 需要）——需 Codex 设计最小实现（flow-noise 近似/ReinFlow 可学习噪声/高斯简化三选一）
- **下界引用**：MT50 from-scratch SAC 38.5%/PPO 35.4%（Yu et al. 2020）
- **主张边界**：在线 RL 微调 flow VA 头（冻结特征上）提升 IL 成功率——与 πRL/ReinFlow 同协议
- 实施：MW 闭环完成后（IL checkpoint 就绪），~1-2h GPU，49 任务稀疏奖励

### 11.3 基线对照复现计划（2026-08-05，Grok 查证官方协议与权重）
| 基线 | MW 权重 | MW 官方数字 | 评估协议 | 24GB 可行性 |
|---|---|---|---|---|
| **Evo-1** (0.77B) | `MINT-SJTU/Evo1_MetaWorld`（HF）✓ 本地有 Evoagent 仓库 | **80.6%** | 10 trials × 5 seeds | 训练+推理 ✓ |
| **SmolVLA** (~0.45B) | `lerobot/smolvla_metaworld`（HF）✓ 数据同源 | **~68%** | 10 trials/task | 训练+推理 ✓ |
| **π0.5** (~3.3-4B) | MW 无官方权重/协议 | —（π0 ~48%） | LIBERO 50 trials（OpenPI） | 推理 ✓ / 全量训练 ✗（80GB） |

- **MW 对照**：SmolVLA + Evo-1（下载官方权重 → 按官方协议闭环评估 10 trials/task）；π0.5 在 MW 标注 unofficial 或省略
- **LIBERO 对照**：Evo-1（Evo1_LIBERO）+ π0.5（lerobot/pi05_libero_base）
- **口径警示（Grok）**：10-trial 与 50-trial 数字不能混表，需脚注；训练量（SmolVLA 100k×bs64 vs Evo-1 10k+65k×bs16 vs 我们 40k×bs1）也要注明
- 基线评估复用 eval_metaworld.py 的闭环协议（MT1/种子/相机），模型输入适配各自权重

### 11.4 业界对照与论文定位（2026-08-05，33 篇本地论文 2025-2026 + Grok 大厂旗舰综合）
**四大支柱共识**：① 数据——互联网先验 + 异构混合/合成反事实（CAST +27%、π0.5 97.6% 异构）；② 架构——VLM 表示 + flow/diffusion 动作头（π0.5/GR00T/RDT-1B），接口设计是泛化新赌注（Action QFormer 指令条件 query）；③ 训练——三阶段 + 保护预训练表示（SAM 平坦化微调治 instruction blindness 零改动 +217%，LIBERO-CF）；④ 推理——显式时序状态 + verifier 事件触发重规划（CheckVLA/Sentinel），慢思考低频 + 快动作高频共识。

**本工作定位（外部验证）**：
- ✅ 已踩中：flow matching 动作头、显式循环视觉记忆（≈μVLA/ReMem/RB-VLA 路线）
- ✅ 结构性优点：语言走 Qwen 外部静态 cache → 动作训练不破坏语义先验（业界治 instruction blindness 的问题在静态 cache 结构下应不存在）——**C1（Qwen 冻结）正是该结构性主张的直接验证**；B40k 的 LoRA 压平说明"破坏来自语言塔参与微调"而非 cache 结构
- ⚠️ 缺口：数据单一（LIBERO/MW）→ CAST 反事实增强（零采集）；指令-场景交互弱（静态 KV）→ 指令条件 query；训练稳定性 → SAM 平坦化微调；记忆仅视觉层 → 语义/工作记忆；无闭环验证 → verifier + 事件触发

**新增实验候选（简单优先，已备本地文献）**：
1. **LIBERO-CF 反事实基准评估**（papers/libero_cf.txt）：验证语言遵循 vs 视觉捷径——与"更听话"主线合并
2. **SAM 平坦化微调**（papers/flatness_2606_23641.txt）：零架构改动，LIBERO-CF +217%（OpenVLA-OFT/π0.5）——若 C2 确认 LoRA 压平，SAM 作为比锚定正则更简单的修复候选
3. **CAST 反事实指令+动作合成**（papers/cast_2508_13446）：零采集数据增强，多任务泛化最便宜一步
4. **μVLA 受控记忆实验**（papers/mu_vla.txt）：加 m 记忆 token 看上限
实施排序：C1/C2 判决后，SAM（若适用）→ LIBERO-CF 评估 → CAST 增强（LIBERO-100 训练时）

### 11.5 CAG 推理增强（2026-08-05，Grok 查证，与"更听话"主线合并）
- **CAG（LIBERO-CF 论文）** = 推理时双分支动作混合：π_CAG(a|o,l) = π_uncond(a|o,∅) + ω(π_cond(a|o,l) − π_uncond(a|o,∅))，ω≈1.5-3
- **零训练**：TF 变体直接用同一 VLA 推理时丢语言跑 uncond 分支；VA 变体可选训一个无语言 VA 先验（标准 demos）
- **与本架构天然对应**：我们的 `--perturb blank`（语言切除）就是 uncond 分支——CAG 可直接在我们的 checkpoint 上实现（推理侧，无架构改动）
- 实验：B40k/A 模型 + CAG(ω 扫描) 在 swap/错误指令下测"更听话"提升——推理增强对照，与结构方案（Codex 设计中）互补

### 11.6 服从性原创结构设计（2026-08-05，Codex 第一性原理推导，未做 prior-art 检索勿写"首次提出"）
**机制框架修正（Codex 批判）**：
- Qwen encoder 表示侵蚀：离线冻结路径 ✓ 结构性阻断（sg 防 Qwen 漂移）
- **语言读出/通信通道侵蚀：未解决**（norm_l/k_l/u_l 可坍缩、q_a 可正交、out_a 可忽略）——命名"语言接口坍缩"，B40k 压平即实证
- **e2e 路径主张不成立**：encode_trainable 无 detach，LoRA 收动作梯度
- 最安全表述：stop-gradient 防 Qwen 漂移，但不防语言读出退化，不保证因果语言使用

**方案 A SMC-Attn（源测度校正注意力，推荐先做）**：softmax 前对来源 s 减 log N_s → P_s ∝ 平均证据而非 token 数 × 平均证据；复制 V/M token r 次动作输出不变（可测）；0 参数 ~15 行；β∈{0,0.5,1} 消融
**方案 B LVK（语言否决核）**：语言方向 k̄_L 与 V/M key 的 cos 一致性作为乘积核加入 logits（源内中心化防隐式预算）；语言从弱竞争者变证据兼容度；对 ②c 更强
**方案 C CSH（竞争分片头）**：每 head 只读 owner 子集 V/M token；风险高，备选消融

**实验协议（Codex 定稿）**：
- flat ↔ SMC 严格对照（同数据/seed/步数）；SMC 提高 P_L 且换指令动作位移、clean 不降 → 主方法；只提 P_L 不动动作 → 上 LVK；最后 SMC+LVK
- Token-refinement test：复制 V/M 2×/4×，SMC eval 模式动作不变
- Command-fork：固定 V/M/状态/prev/noise/τ 只换可执行指令，测 ‖Δv‖ 与专家 Δv* 方向余弦
- 2×2：原始/CAST 数据 × flat/SMC 或 LVK
- 推理侧 certificate（零训练）：R = ‖a(l)−a(l⁻)‖/(ε+‖a(l)−a(l⁺)‖)，同噪声三推，反事实位移小 → 拒绝执行/请求澄清
- attention mass 仅诊断非主证据；clean success 必须同报

### 11.7 基线评估适配要点（2026-08-06，权重已下载）
- **SmolVLA**（lerobot 格式）：输入 3×camera(256²) + state(6d)——MW 本地仅 corner2 单相机，需复制 3 份或改输入管线；state 取 ee 相关 6 维；用 lerobot 库加载 model.safetensors + preprocessor/postprocessor
- **Evo-1**（Megatron mp_rank 格式）：Evoagent 本地仓库（/home/ryan/Documents/robot/Evoagent）加载；评估协议 10 trials × 5 seeds（官方）
- 统一评估：复用 eval_metaworld.py 的 MT1/种子/相机协议，模型输入各自适配；10-trial 口径与官方一致
- 排序：C1 → MW 全链 → C2 → 基线评估（GPU 串行）

### 11.8 TurboVLA 与 RoboTwin 2.0（2026-08-06，Grok 查证）
- **TurboVLA**（arXiv:2607.27205，H-EmbodVis）：去 LLM 主干的 0.2B 轻量 VLA，LIBERO 99.2/99.8/97.4/94.2 → **97.7 avg**（50 trials/task VLA-Adapter 协议）；无 MetaWorld 数字；无自创 benchmark
- **RoboTwin 2.0 clean50**（TurboVLA 主要评估的 bimanual benchmark）：双臂 50 任务、单多任务策略、100 clean rollouts/task、50 步 chunk 14-D 关节动作——**与单臂 VA 架构不直接兼容**（动作维度/双臂协调），列为候选：评估双臂适配成本（动作头 14-D + 双相机输入）后决定是否纳入
- 行动：TurboVLA 进对比表 ✓；RoboTwin 2.0 记录为候选 benchmark（单臂架构适配成本评估中）

### 11.9 C1 完成 + Qwen cosine 判决（2026-08-06 03:48）
- **C1（V-JEPA+Qwen 全冻结，e2e 40k 步，v2 数据）训练完成**：checkpoints/libero_e2e_C1_40k.pt（347MB），step 30000 loss 0.109 收尾
- **Qwen cosine 判决：C1 = 0.7647**（与原始 Qwen 完全一致，未训练）vs B40k LoRA = 0.9992（压平）
- **结论**：冻结路径下语言嵌入完全不受训练影响——"stop-gradient 防语言表示漂移"**结构性成立**（2×2 矩阵第 1 格 ✓）
- 下一步：C2（V-JEPA 冻结 + Qwen LoRA）→ 若 C2 cos 也 ~0.999 → LoRA 是压平主因，弃视觉锚定转向语言隔离/L_pair；若 C2 保持 0.765 → V-JEPA 微调是主因
- **C2 判决（2026-08-06 10:50）**：Qwen cosine = 0.9989（原始 0.7647 / B40k LoRA 0.9992）→ LoRA 是压平主因，弃视觉锚定，转语言隔离/L_pair 修复方向
- **修复方案落定（2026-08-06 13:30 梳理）**：**C1（Qwen 全冻结 + VA 全量 + V-JEPA 冻结）即"语言隔离"最简实现**——cosine 保持 0.7647（Qwen 本体未参与训练，嵌入空间零漂移）+ 行为敏感性保持（blank +33.0%/swap +42.2%，远强于 B40k 的 +1.5%/+2.7%）；`build_language_cache(detach=True)` 钩子已就位（model.py:419）供残留风险（VA 层语言投影漂移）备用；与 A/B40k 的完整对照（开环 chunk_mae + 三件套 + C_OL）已排入 v2 队列 0b 段，fig5/fig7 回填后即闭环

### 11.10 CPU 侧工程推进（2026-08-06 上午，C2 等待期间）
- **L_m 判决脚本完成（eval_libero_Lm.py）**：同场景双目标四条件 matched-block 判决；5/5 配对匹配验证（study_back_front/study_left_right/kitchen_back_front/living_soup_butter/living_milk_juice，benchmark 显式绑定 libero_90/libero_object 防跨布局误配）；D/O 对比同 env 同 init 仅语言不同 = 严格 matched；block bootstrap 95% CI
  - **修复**：eval_libero_closedloop.py 的 build_task_envs 缺 libero_object/libero_goal → LIVING 场景任务匹配不到（B40k 闭环评估漏任务隐患）
- **VLA-RL 实施完成（ReinFlow-lite PPO）**：model.py 新增 sample_flow_trajectory/flow_trajectory_log_prob（增广 Markov flow 策略，逐 Euler 转移注入可学习噪声 σ=0.02+0.06·sigmoid(α)，32 参数）；train_ppo_metaworld.py（FlowNoiseSchedule + ValueHead 零初始化 + GAE 成功/终止不 bootstrap 时间截断 bootstrap + TBPTT=1 detached memory + 每更新重建语言 cache）；tests/test_flow_ppo.py 6 测（确定性路径=经典采样器、零 log-ratio、梯度流、噪声界、GAE terminal、零奖励）；**pytest 50 全绿**
- **LIBERO-100 全量链就绪**：prepare_libero.py 新增 --scene ALL（默认路径更新到 /mnt/robot-data/datasets/benchmark_data）；dry-run 100 任务 × 20 ep × 2 seq = 4000 样本/8 万帧
- **自动队列 /tmp/c2_after_queue.sh**（nohup 已启动，PID 1270240）：等待 C2 → LoRA 判决（CPU）→ 记录 §11.9 → MW 多 start 全链（含 VLA-RL smoke 3 任务 120 iter）→ L_m A → L_m B40k → LIBERO-100 特征 → 训练 10k → 开环+三件套
- **论文初稿 paper/ora0_paper.md**：摘要/引言/方法（架构+两级 loss+机制段）/实验设置/结果 5.1-5.5/结论 + references.bib 42 条（arXiv API 抓取 + Grok 验证 V-JEPA2.1=2603.14482/ReinFlow=2505.22094/DPPO=2409.00588/RoboTwin2.0=2506.18088/GR00T=2503.14734/Qwen3=2505.09388）；待填：L_m、VLA-RL、MW 多 start、LIBERO-100
- **VLA-RL 审查修复（2026-08-06 上午，agent 审计 7 项全修）**：① ppo_update proprio/previous 多一维（硬崩溃）；② eval_libero_Lm.py main 3 元组解包（硬崩溃）；③ chunk 相位 `%8→%DECISION_STRIDE`（先执行计划尾部）；④ memory off-by-one（存输出→存输入，TBPTT=1 严格一致）；⑤ GAE 跨 env 污染（按 env 分算再拼接）；⑥ noise_schedule 梯度被 critic_opt.zero_grad 清零（σ 并入 actor_opt——σ 是策略参数）；⑦ model.py logp 归一化缺 H·A 常数（σ 可训练后 ratio 有偏，已修 `H*A*log2π + H*Σlogσ²`）；⑧ 附带：rollout 改 no_grad（inference tensor 不能进 autograd）、trunc 后 break、state 归一化、noise 形状校验。验证：σ-α 梯度连接 4.8e0 ✓、memory 首决策 None ✓、ppo_update 集成跑通 ✓、pytest 50 全绿
- **CALVIN 适配评估（Grok 查证）**：CALVIN 无需 Coppelia（PyBullet+EGL），debug 集 1.3G，完整集无 HF 官方镜像；RLBench 需 Coppelia+PyRep（一周+改造）→ 后置；轻量 VLA 主报 LIBERO（Turbo/Evo/Smol），CALVIN 锚点 FLOWER ~4.5。决策：CALVIN debug 集优先，RLBench 候选

### 11.11 CALVIN 完整链路决策记录（2026-08-06 13:10，Grok 二次查证）
- **官方数据**：`calvin.cs.uni-freiburg.de`，`task_D_D.zip` = 166GB（train+val 打包，无官方 validation-only 包）；SHA256 校验文件随包提供；社区 HF 仅部分子集镜像，官方完整集优先
- **数据规模**：每环境 ~6h teleop，总计 ~2.4M 步、34 项技能、~20k 标注序列/389 条指令表述（language 仅覆盖 ~1% 数据）
- **标准协议**：ABC→D（A/B/C 训练、D 测试）为 VLA 论文惯例设置；**只用环境 D 做零样本闭环评估不需要 166GB 训练数据**——只需 calvin_env 仿真（PyBullet）+ 官方 LH-MTLC eval（1000 chains × 5 任务）
- **文献口径警示**（Grok 强调）：GR00T/CogVLA/RT-2/π0/OpenVLA 原论文**无官方 CALVIN ABC→D 数字**；OpenVLA 的 CALVIN 数字来自第三方微调（RoboDual/DreamVLA 3.27）——对比表中若引用必须注明来源
- **决策**：① 数据侧用现有 debug 集（1 集 8 段语言标注）做 in-distribution smoke；② benchmark 数字路径 = 装 calvin_env + 官方 eval 脚本跑 D 环境零样本（不下载 166GB）；③ 排在 MW 链 + LIBERO-100 之后（GPU 串行）；④ RLBench（CoppeliaSim）维持候选不优先

### 11.12 CPU 侧验证与复现性修复（2026-08-06 13:20，MW 重训等待期）
- **v2 队列全接口验证**：resume_queue_v2.sh 的 C2 col / L_m A/B / LIBERO-100 prepare / train / evaluate 参数与脚本 argparse 全匹配；`py_compile` 全部通过
- **C2 col 冒烟 PASS**（eval_e2e_col.py，2 样本 CPU 116s）：加载/数据/rollout/C_OL 指标链全通——此前两次"545 字节即断"确认为 GPU 占用等环境问题而非代码问题（队列已有 non-fatal 保护）
- **复现性缺口修复**：C1 三件套日志随 /tmp 清理丢失（数字仅存 todo），C2 行为评估（开环+三件套）从未跑过 → 队列新增 0b 段：C1+C2 各 openloop/blank/swap 共 6 个评估，fig5/fig7 的 C1/C2 行数字来源落盘 logs/
- **VLA-RL 训练器审查**（train_ppo_metaworld.py）：rollout 用 language_cache 与 update 用 language_hidden 两条路径等价（encode_condition 内部统一走 build_language_cache，old_logp 不失真）；GAE/稀疏奖励/成功终止不 bootstrap/时间截断 bootstrap/TBPTT=1 memory 图一致全部确认；清理一处死代码（重复计算未使用的 value_loss）；编译验证通过
- **LIBERO-100 数据源验证**：kevin_libero100_lerobot 26G（LeRobot 格式 data/meta）就绪，数据盘 350G 可用
- **论文初稿方法章节核对**：§3.1-3.4 与代码一致（Qwen/Qwen3.5-2B ✓、Q/K/U 公式 ✓、L_pair=0 口径与日志 pair=0.000000 一致 ✓）；§5.6/5.7 协议描述与 eval_libero_Lm.py / train_ppo_metaworld.py 实现全匹配
- **MW 重训实测速度 ~464 步/分钟**（12:41 启动，非交接笔记的 130 步/分——该数字属 LIBERO B40k 口径）→ 预计 ~14:10 完成训练、~17:00 闭环出数字；loss 收敛趋势 0.39→0.25→0.21→0.19（2k 步区间均值，单调下降）

### 11.13 基线数字 Grok 二次查证定稿（2026-08-06 14:45，论文对比表最终口径）

**查证方法**：grok -p 单轮查证（arXiv 原文/官方文档优先，共 8 问），结果与 §8.5 旧表逐项核对，以下为定稿口径（**论文引用必须用此版**）：

| 模型 | MW MT50 | LIBERO | 安全引用写法 | 第三方？ |
|---|---|---|---|---|
| Evo-1 | **80.6%**（Table 1，10 trials×5 seeds） | **94.8%**（Table 1；long 92.3%） | Evo-1 Table 1，10 trials×5 seeds | 否 |
| SmolVLA 0.45B | **57.3%**（Table 2） | **87.3%**（90/96/92/71，Table 2） | SmolVLA Table 2，10 trials/task | 否 |
| SmolVLA 2.25B | **68.24%** | **88.75%**（93/94/91/77） | SmolVLA Table 2 | 否 |
| πRL MW 扩展 | π0 Flow-Noise **85.8%** / Flow-SDE **78.1%**、π0.5 SDE **70.7%**/Noise **66.1%**、SmolVLA **68.2%**（转引）、π0 SFT **50.8%**（πRL 自训） | — | πRL MetaWorld 扩展结果/RLinf（**v1 Table 6 是超参表，勿写 Table 6**）；SFT=2500 traj lerobot metaworld_mt50，sparse 0/1，total_num_envs=64；评测 trials/task 未在表注写死 → 脚注写 "per RLinf/πRL MetaWorld eval protocol" | SmolVLA 68.2% 是转引 |
| π0.5 | — | **96.9%**（98.8/98.2/98.0/92.4）**非 π0.5 原文**（原文无 LIBERO） | OpenPI/πRL Table 1 "Full Dataset SFT"，50 trials/task（10 subtasks×50 states=500/suite） | **是**（相对 π0.5 主文） |
| TurboVLA | — | **97.7%**（99.2/99.8/97.4/94.2，Table 1） | TurboVLA Table 1，50 rollouts/task 共 2000 trials，VLA-Adapter 协议 | 否 |
| π0 | **47.9%**（原文无 MW，SmolVLA Table 2 转引） | **94.2%**（96.8/98.8/95.8/85.2，原文无 LIBERO，OpenPI full SFT 第三方报告） | 两个数字均须标"第三方转引" | **是**（两者皆非 π0 原文） |
| OpenVLA | — | **76.5%**（84.7/88.4/79.2/53.7，**Appendix E v2**，3 seeds×500 rollouts=10×50） | OpenVLA GitHub/v2 Appendix E（主文无 LIBERO） | 作者附录（非主文） |
| OpenVLA CALVIN | — | 原文无；仅第三方微调（VLAS/RoboDual/AVA-VLA 等） | 引用必须注明第三方 | **是** |

**口径警示落实**：
1. §8.5 旧表 SmolVLA "~68%/~89%" 是 2.25B 行——论文改为双行（0.45B 57.3/87.3 + 2.25B 68.24/88.75），并与我们的 43.5M trainable 对齐说明（0.45B 行为最接近对比）。
2. π0.5 LIBERO ~97 必须写 "OpenPI/πRL full-dataset SFT 报告"，不能写 "π0.5 论文 Table"。
3. π0 的 MW 47.9% 与 LIBERO 94.2% 双标第三方（SmolVLA Table 2 / OpenPI）。
4. OpenVLA 76.5% 标 Appendix E (v2)。
5. CALVIN 若报告必须注第三方/或仅作背景。
6. VLA-RL 对照行：πRL MW 数字 + "per RLinf/πRL MetaWorld eval protocol" 脚注。

**SOTA 目标校准（2026-08-06 14:45）**：本工作 43.5M trainable 与 SmolVLA 0.45B 最可比 → 追赶目标 = SmolVLA 0.45B **57.3%**（MW）/ **87.3%**（LIBERO）；Evo-1 80.6% 为 0.77B 两阶段 + 5 seeds，作远期目标；TurboVLA 97.7% 为 50-trials 口径（协议不同，脚注区隔）。

### 11.14 MW 多 start 闭环正式结果 + SOTA 差距归因（2026-08-06 15:15）

- **CLOSED-LOOP SUCCESS: 80/490 = 16.3%，macro 16.3% [9.4%, 24.1%]**（logs/mw_full_closedloop.log，metaworld_va8_40k_full.pt，49×10，horizon 500，32 步口径；分任务表见 logs/mw_closedloop_summary.md）
- 对比：单 start 7.1% → 多 start 16.3%（×2.3）；**仍远低于 SmolVLA 0.45B 57.3% / Evo-1 80.6%**
- **差距归因（初步，2026-08-06）**：
  1. 短任务成功（button 8/10、coffee-button 7/10、door-close 8/10、drawer-close 8/10、window-close 9/10），长任务全 0-3/10 → 闭环误差累积 + 长程记忆/策略容量不足；
  2. previous_action 闭环自激（P0-1）：训练 teacher-forcing，闭环自我执行误差经 prev→A→V→M 路径累积；
  3. IL 开环增益有限（chunk_mae 0.080 vs 持久性 0.093，-14%）；SmolVLA/Evo-1 全轨迹数据 + 更强 flow head + 更多训练步数。
- **追赶链执行顺序（GPU 串行）**：① VLA-RL PPO（smoke 3 任务 120 iter 已启动）→ ② MT10 正式 PPO → ③ 覆盖度再升级（sequences-per-episode 8，重训 40k）→ ④ SMC-Attn / SAM 结构改进 → 每步闭环重测；未达 SOTA 如实入表。

### 11.15 工程推进（2026-08-06 15:15-15:40，MW 闭环后）
- **MW 闭环定稿**：16.3% [9.4, 24.1]（49×10，horizon 500）已回填论文 §5.3 + 对比表 + fig9_mw.png（SmolVLA 锚点改 0.45B 57.3）
- **差距归因入论文**：短任务近天花板（button 8/10、door-close 8/10、drawer-close 8/10、window-close 9/10）vs 长任务 0-3/10（OOD）+ prev/记忆误差累积
- **SMC 训练通道打通**：train.py 新增 `--attention-variant flat|smc`（3 处 config 构造透传；checkpoint config.__dict__ 自动持久化，下游 eval 全部免改）；`logs/mw_smc_chain.sh` 就绪
- **PPO 更新批量化（train_ppo_metaworld.py）**：ppo_update 从逐样本循环改为 minibatch 堆叠（encode_condition + flow_trajectory_log_prob 均 batch 化；memory=None 的首决策过渡保持逐样本路径）；数学等价（新测试 test_batched_condition_matches_per_sample / test_batched_ppo_update_mixed_memory 验证，pytest 52 全绿）；CPU 基准 64 样本 270s（GPU 预计 ~10-20s/update）
- **PPO 吞吐分析**：smoke ~60s/iter（rollout ~7s + 旧逐样本 update ~50s）；批量化后预期 iter ~15-20s；MT10 正式实验（10 envs × 64 macro）预计 ~1 min/iter → 1000 iters ~15h（过夜任务）
- **队列脚本全部就绪并 bash -n 验证**：logs/next_phase.sh（C1/C2 三件套 + cosine + L_m + C_OL）、logs/libero100_chain.sh（5000 样本链）、logs/vla_rl_mt10.sh（MT10 IL→RL）、logs/mw_smc_chain.sh
- **Grok 基线查证已入 §11.13**（SmolVLA 双行 / π0.5 OpenPI 转引 / π0 双第三方 / OpenVLA Appendix E / πRL RLinf 协议）
- **执行顺序（GPU 串行）**：smoke（~17:20 完）→ next_phase.sh（LIBERO 修复链，~2-3h）→ vla_rl_mt10.sh（过夜）→ libero100_chain.sh → CALVIN 尝试 → SMC/SAM 按结果决定

### 11.16 ⚠️ 闭环评估缺陷修复（2026-08-06 16:00，GPT 审查发现 + 数值验证）

**缺陷**：eval_metaworld.py 在首决策前（step 0-5）执行 `chunk = np.zeros((8,4))`（归一化零动作）。反归一化后 = `(aq99+aq01)/2 = [2.45, 2.27, -1.37, 0]`，环境 clip 后 `[1, 1, -1, 0]`——机械手在模型首次决策前被以最大速度推走 ~4.3cm（真零动作仅漂移 0.21cm）。精细抓取任务（peg/nut/sweep 等）直接进入 OOD 状态。**已数值验证**（metaworld_features_v2_full.pt 归一化参数）。

**影响**：49×10 闭环 16.3% 不能作为架构失效证据（可能被该缺陷显著低估）；7.1%（单 start）同样受影响。语言消融（开环）与开环指标不受影响（无闭环执行）。

**修复**（已提交 eval_metaworld.py + train_ppo_metaworld.py macro_rollout，同源缺陷）：
- 首帧渲染后立即用重复帧填充到满窗口（19 帧），使 step 0 即推理；
- 与训练分布一致：prepare 的 clip_frame_indices 用 max(0, d-offset*stride) 钳制，episode 首决策窗口本身是重复帧；
- 窗口取帧经模拟验证：step0=[0,0,0,0]、step6=[0,2,4,6]、step12=[6,8,10,12]——与训练窗口严格对齐。

**判决顺序（GPT 定稿，最小改动优先）**：
1. ✅ 修评估缺陷（已完成）
2. 同一 checkpoint（metaworld_va8_40k_full.pt）重跑小规模闭环 smoke（6 任务 × 10 trials，data/mw_subset_smoke6.pt 已建）
3. 若仍低 → 记忆每 4 次重置 / previous-action dropout 测试（不改架构）
4. 最后才考虑 SMC/SAM/LVK 或视觉池化改动

**论文口径**：16.3% 数字待重测后更新（tex/fig9 当前为旧值，重测后回填）。

### 11.17 修复扩散（2026-08-06 16:10）
- eval_calvin.py 同源缺陷已修（首决策前反归一化零动作 = 动作区间中点）；CALVIN 零样本评估待 GPU 队列尾部执行
- eval_libero_closedloop.py 无需修：robosuite 归一化动作接口，warm-up 用真零动作（无反归一化问题）
- 重测队列 logs/mw_eval_fix_retest.sh 就绪（smoke6 → full49，同一 checkpoint）
- 数字台账 logs/mw_numbers_ledger.md 建立（论文唯一数据源）

### 11.18 用户指令（2026-08-06 17:10）：闭环审计优先
**优先级调整（写进目标）**：① 优先做数据集的闭环审计（评估缺陷修复后的闭环重测：smoke3 RL → smoke6 → full49，同一 checkpoint 不重训）；② 闭环结果出来后交给 GPT/Codex 分析再定下一步（VLA-RL / cov8 / 记忆重置 / prev-dropout / SMC）；③ 其余 benchmark（LIBERO/LIBERO-100/CALVIN）一律延后到 MW 闭环审计与 SOTA 追赶完成之后。

### 11.19 闭环审计结果（2026-08-06 18:25）——评估缺陷假设证伪
- **full49 修复后重测（同 checkpoint 同种子）**：**13.9% [7.6%, 21.0%]**（68/490）vs 旧 16.3% [9.4%, 24.1%]——CI 重叠，修复无显著影响
- smoke3（RL smoke checkpoint，3 任务，修复后）：40.0%——RL 微调 120 iter 后 3 个简单任务 IL 基线（旧评估口径 8+2+0）/30 对比：RL checkpoint 明显更高（12/30 vs 10/30），初步 RL 有效信号
- smoke6（子集种子，6 任务）：65.0%——与 full49 不可逐任务比（种子重编号），仅作管线 sanity
- **结论**：评估缺陷排除；瓶颈在训练-部署契约/数据覆盖/容量（GPT 假设 4 项），不是评估协议
- **已执行诊断**：memory-reset-every=4 对照（推理侧，零训练，logs/mw_memreset4_smoke6.log，与 smoke6 同种子）
- **待 Codex 分析**（/tmp/codex-mw-verdict.md，后台）后定下一步实验顺序

### 11.20 记忆重置诊断结果（2026-08-06 18:32）——记忆递归假设排除
- memory-reset-every=4（每 4 决策重置递归记忆，与训练 T=4 深度对齐）：**58.3% [31.7%, 85.0%]**（35/60）vs 无重置基线 65.0% [38.3%, 90.0%]（同种子，smoke6 子集）
- **结论**：截断记忆到训练深度反而降低成功率 → 长程递归记忆有正贡献，记忆契约缺口不是闭环 13.9% 的瓶颈（GPT 假设 4b 排除）
- 已排除：评估协议缺陷（11.19）、记忆递归深度（本段）；剩余候选：prev 自激（需重训测）、数据覆盖（cov8）、训练量/容量（epochs/SMC/SAM）、VLA-RL
- 待 Codex 分析定序（/tmp/codex-mw-verdict.md）

### 11.21 Codex 分析定稿（2026-08-06 18:40，/tmp/codex-mw-verdict.md）+ 已执行项
**Codex 核心结论**：
1. **口径**：文献 Avg = 四难度组等权 macro4（非任务等权 micro）；必须同时报 micro/macro4/四组/逐任务。SmolVLA 0.45B 57.3% (macro4) ≈ 66.8% micro。
2. **已实测**：FIXED full49 micro=13.9% / **macro4=7.0%**（E 22.5%/M 3.6%/H 2.0%/VH 0.0%）；OLD micro=16.3% / macro4=8.0%。**macro4 与 SmolVLA 57.3% 的差距比 micro 更大**。
3. **lerobot 官方元数据核实**：total_tasks=49（2500 eps）——"MT50 第 50 任务"不存在，口径无缺口（论文注明 lerobot metaworld_mt50 官方 49 任务）。
4. **训练量嫌疑**：SmolVLA 100k×bs64 / Evo-1 10k+65k×bs16 vs 我们 40k×bs1（约 4 epochs）——容量不足不是唯一解释。
5. **推荐顺序**：契约诊断（16-task panel，已启动）→ W4-60k 续训/W8-40k 覆盖对照 → prev-dropout=0.2 → 表示/容量定向修复（39D oracle / specialist 探针）→ frontier PPO pilot（6-task gated）→ 全任务 PPO。
6. **RL 警示**：πRL 50.8→85.8 起点远强于我们；大量 0-success 任务无正奖励 → 先做 gated pilot 验证奖励覆盖，不做全量。
7. **prev 语义审计（已做）**：训练 prev = 上一原始步动作（norm_action[d-1]），闭环 last_norm = 上一原始步动作 ✓ 语义一致；首决策 prev=0 一致 ✓；仅剩"真值 vs 自激"差异（需 prev-dropout/RL）。
8. **统计**：修复前后同种子应做 paired 分析；CI 重叠不是"无差异"证明；bootstrap 按任务聚类。

**已执行**：16-task panel 启动（mem ∞/4 × prev self/zero，logs/mw_panel16.sh，~20:00 完成）。

### 11.22 用户 × GPT 容量讨论定稿（2026-08-06 19:30-19:40，粘贴整理）
**问题**：闭环 13.9% 是不是模型容量问题？Qwen 不是很大吗？
**GPT 结论（采纳）**：
1. **Qwen 只是"语言翻译器"不是控制器**：每任务只运行一次 → 13×2048 语言特征 → 释放；不看状态/图像/闭环误差，不收动作梯度。真正学习控制的是 **79.23M**（8 层 VA 71.5M + Flow 7.36M；论文"43.5M"是旧 4 层配置数字，**论文需更正为 8 层口径**）。
2. 语言不是瓶颈（wrong +108.5% 证明 Qwen 语义有效进入策略）。
3. 瓶颈候选排序：a) **V-JEPA 冻结 + 平均池化到 64 token 丢失精细位置/接触信息**（"眼睛"）；b) **Flow 仅 2 层且条件只在入口加一次**（"小脑"）；c) **训练不足 40k×bs1 ≈ 4 epochs**（"看起来像容量不足"）。
4. **最小判决实验（GPT 定稿）**：选一个困难任务让当前模型过拟合——训练误差压不下去 → Flow 2→4 层；训练能拟合但闭环仍失败 → 视觉表示/闭环分布漂移。扩容优先动作侧，不换更大 Qwen。
5. **结构演进方向（用户目标：利用 Qwen 又不被每步拖累）**：Qwen-Compiled Recurrent VA——指令+初始图像跑一次完整 Qwen → 4-8 program tokens P₀ → VA 内 P 与 V/A 双向更新（P 可被实时视觉修正）→ 事件触发刷新。最小验证路径：① oracle 阶段标签（reach/grasp/transport/release）探针——不涨则高层任务状态非瓶颈，停止该路线；② Qwen 生成 8 个 action-query offsets（零初始化）→ 对比静态 K/V；③ 有收益加递归 P tokens；④ bidir_va vs uni_a 对照。
6. **对照底线**：Qwen 版本必须显著超过 Task-ID/小 Encoder，且打乱 P 后性能下降；否则普通 Encoder 足够。
**执行中**：16-task panel ④ 收尾（~19:50）；随后按 GPT 最小判决跑困难任务过拟合探针。

### 11.23 16-task 契约 panel 定稿（2026-08-06 19:47）
| 条件 | SR (16×10) | vs 基线 |
|---|---|---|
| ① mem=∞, prev=self（基线） | **30.6%** [15.6, 47.5] | — |
| ② mem=4, prev=self | 27.5% [13.8, 42.5] | -3.1pp |
| ③ mem=∞, prev=zero | 6.9% [1.9, 13.1] | **-23.7pp** |
| ④ mem=4, prev=zero | 1.2% [0.0, 4.4] | -29.4pp |
**判决**：记忆重置无益（②<①）；prev 置零断崖下降（③④）→ prev 是必要输入（模型在闭环自洽使用自身输出），"prev 自激"不是主瓶颈；**契约缺口全部排除**。瓶颈候选收敛到：训练量/覆盖（40k×bs1≈4 epochs vs SmolVLA 100k×bs64）、视觉池化信息损失、Flow 容量。
**下一步（GPT 最小判决）**：困难任务（nut-peg，闭环 0/10）过拟合探针——train loss 压不下去 → Flow/优化容量；能压下去 → 视觉表示/泛化/闭环漂移。

### 11.24 方案 A 实现 + 训练启动（2026-08-06 19:51）
**Qwen-conditioned action queries（GPT 方案 A，第一版）**：
- model.py：`action_query_cond` config 开关；语言摘要（第 0 层投影 key 的 mask 加权均值 [B, hidden]）→ MLP（hidden→hidden→horizon×hidden，**zero-init 末层**）→ 每 horizon 步 query 偏移
- 语义升级：Qwen 从"被动被读的静态 K/V"→"决定动作查询找什么"的慢脑；VA 每步用视觉/状态修正查询
- 训练/推理路径统一（encode_condition 内部总是先 build cache）；checkpoint config 自动持久化，下游 eval 零改动
- 测试：test_action_query_cond_zero_init_equals_static（zero-init 与静态基线逐位等价 + 偏移生效）；pytest 53 全绿
- **训练已启动**（19:51，与基线同协议：同数据/seed/40k/bs1/8 层）：checkpoints/metaworld_va8_40k_aqc.pt → 开环 → 闭环 49×10（预计 ~22:20 全链完成）
- **判决**：闭环 >13.9% 且主要在 hard/very-hard 提升 → 方案 A 有效；持平 → 语言接口不是瓶颈，转训练量（W4-60k/W8-40k）

### 11.25 Codex 结构大改方案定稿（2026-08-06 20:20，/tmp/codex-mw-redesign.md）
**A. 根本原因（多因，非单一）**：
1. 训练量一级瓶颈：40k×bs1 ≈ 16 万 decision presentations vs SmolVLA 100k×bs64 = 640 万（~40× exposure、~8× epochs）
2. 动态接口一级瓶颈：Qwen 从未看图像/状态，语言无法在线落地（纯文本 K/V）
3. 视觉几何一级瓶颈（精细任务）：V-JEPA flat 池化到 64 token 丢毫米级位置/接触信息
4. 执行协议（6 步/决策 vs SmolVLA 每步重推理）混杂项未量化
5. 容量次一级（Flow 2 层、条件入口单次）
**B. P 结构（最终形态）**：K=6 program tokens（对象/目标/几何/阶段顺序/当前阶段/下一子目标，slot 先验无辅助 loss）；完整多模态 Qwen 一次前向（初始图像+指令+6 readout 问句 → 6×2048 → MLP → 6×512 P₀）；P 独立 stream 复用 V 分支参数；P_t 每步在 VA 内更新；最终 P 替代静态 L（诊断版保留 L）；参数量 80.57M（诊断版）/ 63.75M（纯 P）。
**C. 判决链（唯一执行顺序）**：A 跑完（不启动 W4/W8；A 有效门 Δmacro4≥+5pp 且 H+VH 升）→ execute 6/3/1 micro-panel（8 任务×5 trials，~45min，+8pp 则协议重要）→ oracle-stage 10k paired（evaluate_state 构造四阶段标签插入 P token；Go: Δmacro4≥+8pp 且 H+VH≥+10pp，乱序负对照）→ phase-spanning P-lite 10k（现有文本 hidden 初始化 P，跨阶段训练；carry vs reset vs shuffle）→ full-Qwen P 40k → 唯一一次正式 49×10。
**D. 失败预案矩阵**（P 强但绝对 <30 → 80-100k 训练量；oracle 强 P 弱 → 修 phase-spanning/TBPTT；oracle 弱 execute 强 → 按 cadence 重训；都弱 → 39D oracle/specialist；IL 覆盖后 → 6-task RL pilot）。**禁止一次全改**（P+W8+100k+execute=1+Flow4+RL）。
**E. 与 A 关系**：P 是 A 概念超集，首个 P 不叠 A；A 强则 P 输入改 mean(P)；A 只升 Easy 不证明阶段价值。
**当前动作**：等 A 链跑完（~22:20），记录 Δmacro4 与 Δ(H+VH)。

### 11.26 用户指示（2026-08-06 21:25）：e2e 解冻训练
**指示**：① V-JEPA 全量解冻训练；② Qwen 半解冻（部分层参与训练，非全冻非全解冻）。
**背景**：用户认为视觉表示是精细任务瓶颈（V-JEPA 从未适应任务域 + flat 池化丢毫米级信息）；Qwen 应部分适应任务域但防坍塌（B40k 教训）。
**已启动**：Codex e2e 设计讨论（/tmp/codex-e2e-design.md，后台）；AQC 闭环结果 ~22:30 出，先记录 AQC 再叠加 e2e 决策。
**待定**：解冻层数/LR、Qwen 半解冻防坍塌机制（cosine 复核门槛）、e2e 数据管线（多 start 窗口视频缓存 ~69GB）、子集快速验证。

### 11.27 AQC 闭环判决（2026-08-06 22:34）——弱阳性，未过 Go 门
- **AQC（Qwen-conditioned action queries）49×10 闭环：17.8% [11.0%, 25.3%]**（87/490）
- 对比：基线 13.9% [7.6, 21.0]；micro **+3.9pp**；macro4 7.0→**8.2%**（+1.2pp）；E 22.5→**30.0%**（+7.5pp）；M 3.6→2.7%（-0.9pp）；H 2.0→**0.0%**（-2.0pp）；VH 0→0
- 开环：chunk_mae 0.0850 vs 基线 0.0806（略降）
- **Codex 门判定**：Δmacro4 ≥+5pp 且 tail2 升 → **未过**（Δmacro4 仅 +1.2pp，hard 组归零）。结论："A 只提升 Easy——解决的是任务身份/静态路由，不证明阶段状态有效"（Codex 预设解读）
- **决策**：A 作为"静态语言路由消融"写入论文（+3.9pp micro，Easy +7.5pp，hard 无益）；**不作为 e2e 起点叠加**；主实验转向 e2e（用户指示：V-JEPA 全训 + Qwen 顶层解冻无 LoRA）
- e2e P0 修复完成：V-JEPA unfreeze_all（stem+blocks+norms）、Qwen unfreeze_last(freeze_final_norm=True)、--vision-unfreeze-all、eval_metaworld.py e2e 权重加载、measure_qwen_cosine mask 修复（pytest 53 绿）
