# MT-VJ 阶段 V（语言条件度量场）修复交接文档

日期：2026-08-10
状态：根因已锁定（探针实证），v4 修复配方已实现，tiny 过拟合门结果见 §5
后续主文档：`artifacts/mt_vj_design.md`（设计）、`artifacts/mt_vj_contract.md`（接口契约 v1）

---

## 1. 概要（TL;DR）

阶段 V 的 `LanguageMetricField`（2.17M 参数）第一版训练 20k 步后 train RMSE 钉死在 **50px**
（目标 5mm ≈ 4px），且与"预测每角色 GT 均值"基线持平——模型学到的是**固定位置先验**，不是逐样本定位。

经四轮探针实验 + 三个独立模型评审（Claude / Codex / 第三方），根因**不是**"V-JEPA 特征不含空间信息"
（一个 3KB 的闭式线性模板即可定位到 8-13px），而是三个叠加因素：

1. **读出形式**：全网格 softmax 期望读出 `p̂ = Σπ·(p+δ)` 在 V-JEPA 近乎全平的余弦面上（576 片中
   575 片相似度 > 0.5×max）≈ 均匀分布 → 预测钉死网格质心（40-80px）。argmax / 局部模式读出
   同一查询即 8-10px。
2. **目标函数**：CE 梯度 = Σ(π−y)·f ≈ mean(全图特征) − f_target，被静态背景/机械臂方向主导 →
   查询收敛到边缘分布；max-margin（hinge）目标梯度 = −f_target + f_best_other 逐样本相干，
   2000 步即达 8.5px。
3. **自由空间偏置捷径**：`spatial_bias` 梯度路径最短，~700 步先吃掉"每角色边缘分布"，之后
   饿死视觉通路（v1 的 50px 即此解）。

另有 2 个已实锤的数值 bug（见 §6）：σ=2px 高斯标签的 clamp 归一化伪影（交叉点相位 target 只和
0.45）；`_normalize_coords` 无法区分 [0,1] 与 [-1,1] 坐标。

**当前结论：修复配方 = L2 归一化 + 可学习温度 + 冻结 bias + hinge 目标 + 模式读出。**
tiny 过拟合门（64 固定样本）结果见 §5——过门后即可用同一配方开仿真流全量训练。

---

## 2. 背景

MT-VJ = Metric-Temporal V-JEPA for Action（`artifacts/mt_vj_design.md`）。阶段 V = 控制度量
视觉预训练（`train_metric_visual.py`）：冻结 V-JEPA 2.1（H5/H11 dense，1152 patch）+ 冻结
Qwen3.5 文本（整句掩码均值池 → Linear → 4 角色查询 q_r），只训 `LanguageMetricField`
（2.17M）。标签由仿真器免费生成（`prepare_metaworld_metric.py`，世界坐标 → pinhole 投影，
验证 <2px）。原训练：单任务 peg-insert-side-v3、单文本、20k 步、batch 4、lr 1e-3。

## 3. 失败时间线与证据

| 阶段 | 结果 |
|---|---|
| v1 训练（原始配置，`logs/train_metric_v.log`） | 第 700 步起 RMSE 钉死 50px；CE 4.2-6.5（随机 ln1152≈7.05）；vis_mean≈1.0 |
| 诊断 1（`scripts/diag_metric_field.py`） | 每角色 RMSE ≈ "GT 均值"基线（±5px）；预测跨样本 std << GT std；spatial_bias 学会大峰（0.9-4.2 nats）钉在每角色平均位置；heatmap 熵 2.0-2.7（确实峰化=自信地错）；角色查询无塌缩（cosine 0.01-0.45） |
| Oracle 读出（`scripts/diag_probe_oracle.py`） | GT 热图喂进读出管线 → 每角色 5-8px ≈ 量化下限（4.6px）——**读出/网格/标签侧无 bug** |
| 闭式模板探针 | 单模板（GT 位置特征均值）→ test RMSE **8-13px**（24 训练样本）——**线性信号存在** |
| 训练式探针（CE，随机/模板初始化 × 3 读出） | 全部收敛 49-58px——CE 目标下 SGD 找不到线性解 |
| 静态读出对照 | 同一查询：期望读出(t=10) 22-82px；argmax 8-10px；期望(t=100) 5.6-10.5px；**高相似 patch 数 575/576** |
| **Hinge 探针（max-margin）** | 400 步 hinge→0；2000 步 argmax RMSE **8.5px** ✅ |

## 4. 根因（按证据排序）

1. **余弦面近乎全平（决定性）**：L2 归一化后 576 个 patch 中 575-576 个的相似度 > 0.5×max。
   这是 V-JEPA 特征的固有形态（共享主成分主导）。全网格 softmax 期望读出对此灾难性敏感。
2. **CE 目标与常量查询的组合**：平坦面上 π≈均匀 → CE 梯度被 Σ_n π_n·f_n ≈ mean(全图) 主导 →
   查询被拉向固定背景方向 → 边缘分布解。模板初始化也会被 CE 200 步内破坏。
3. **期望读出**：即使查询正确，期望读出也被重尾拖向簇质心（同查询 argmax 8px vs 期望 45-82px）。
   注：`artifacts/c2irf_v2_vision_ablation.md` §3 早已诊断过同一问题（整体 soft-argmax 假中点），
   本次是它在真实特征上的完整复现。
4. **bias 捷径**（v1 的直接原因）：`spatial_bias` 零初始化 + 最短梯度路径 → 先收敛边缘分布。
5. **排除项**：V-JEPA 特征不可读（否，模板探针通过）；标签/投影错误（否，oracle 5-8px）；
   角色查询塌缩（否）；训练时长不足（否，2000 步 hinge 就够）；数据量不足（否，64 样本就够）。

## 5. 修复内容（已实现）

### 5.1 head（`va_compound/metric_visual_head.py`，全部向后兼容，默认行为逐字节不变）

- v2：`l2_norm`（query/d11 逐行 L2 → cosine）、`learnable_temp`（替代 1/√d，初始
  `temp_init`=10）、`freeze_bias`（冻结 spatial_bias）
- v3：`mode_readout`——片求和 heatmap 全局峰 + 局部 5×5 soft-argmax（+ 峰 patch 的 offset，
  取概率更高时间片）。这是 c2irf §3 的"模式读出"思想的实现
- 输出新增：`offset_full` [B,R,1152,2]（offset 直接监督）、`scores` [B,R,1152]（hinge 监督）

### 5.2 trainer（`train_metric_visual.py`）

新 flags：`--l2-norm` `--no-bias` `--temp-init` `--sigma-px`（默认仍 2.0，建议 ≥4）
`--loc-only`（跳过 rel/vis + relation encoder 训练）`--offset-supervision`（δ*=p*−p_center,
SmoothL1 于 GT patch）`--grad-accum` `--fixed-data`（tiny 集模式：特征一次性预计算，head-only）
`--mode-readout` `--hinge-loss` `--hinge-margin`（默认 0.1）。
`--verify` 路径的 `verify_labels` 是**死代码**（import 了 `prepare_metaworld_metric` 里不存在
的 `_env_pool/_role_points`），待修。

### 5.3 诊断脚本（`scripts/`）

`diag_metric_field.py`（per-role RMSE/查询退化/熵/bias 固定点）、`diag_probe_oracle.py`
（oracle 读出 + 模板探针）、`diag_trained_linear_probe.py`（训练式探针，含 `--init-template`
对照）。

### 5.4 测试

`tests/test_mt_vj.py`：relation 形状断言 4→6（拍板 2A 后测试未跟上）。全部 12 个测试通过。

### 5.5 验证结果

- 旧路径回归：pytest 全绿；CPU smoke（v2/v3 flags）通过
- tiny gate v1（CE+期望）：57px ❌；tiny gate v2（无 bias+L2+σ4）：58px ❌；
  tiny gate v3（+模式读出）：57px ❌
- **tiny gate v4（hinge+模式读出）**：train RMSE **11.98px @ 2000 步**（判据 <15px，
  step 800 即 14.3px 过门；hinge 损失 ~400 步归零，RMSE 仍在下行）——从 57px 提升 5 倍，
  模型学会真实逐样本定位。checkpoint：`checkpoints/metric_field_v4_gate.pt`

## 6. 已实锤的数值 bug（与主根因独立）

1. **σ=2px 高斯标签 clamp 伪影**（`train_metric_visual.gaussian_targets`）：关键点落在 4 个
   patch 交叉点相位时，4 格权重各 e^-16 ≈ 1.1e-7，和 4.5e-7 < `clamp_min(1e-6)` → 归一化后
   target 只和 **0.45**（实测 0.4501），CE 监督在该相位被系统性削弱。σ≥3px 消失。修复：默认
   σ 提升或去掉 clamp（改为 softmax 归一化）。
2. **`_normalize_coords` 坐标域歧义**（`metric_visual_head.py`）：`yx.abs().max() <= 1.01`
   对 [0,1] 输入同样成立 → 被错误平移到 [0.5,1]。当前管线安全（`_dense_coords` 恒为 [-1,1]），
   但契约"两种都接受"是假的。修复：按域显式区分或删掉 [0,1] 分支。

## 7. 后续路线（按序）

1. **仿真流全量训练**（v4 配方已过 tiny 门）：三任务 × 三文本，先 2-5k 步短跑，判据随机数据
   test RMSE < 10px（保 5mm≈4px 需 offset 监督 + 后续 MicroRefiner）
2. **语言条件化验证**（当前实验无法证明语言）：同图换指令 / text shuffle 配对测试；若失败，
   q_r 改 role token 对 Qwen token 序列 cross-attention（弃用均值池化）
3. **阶段 A 重接**：metric checkpoint 修复后重训 dense readout 策略（注意 `train.py` 的
   `_load_mtvj_metric_checkpoint` 按 ctor 签名过滤 config，新字段已记录）
4. **多峰/NULL**：c2irf §3 的 top-2 峰 + NULL 可见度模式（遮挡鲁棒）

**不做 DINOv2 正对照**（用户决策，2026-08-10）：该对照原用于区分"V-JEPA 特征不可读"vs
"读出/目标问题"；模板探针（8-13px）与 hinge 探针（8.5px）已直接证明 V-JEPA dense 特征
线性可读，对照的目的已经达成，不再需要。若后续三任务泛化失败，按 §8 处理（先查数据/实现，
再换读出架构），不回溯到骨干选择。

## 8. 关线决策树（评审一致，已按"不做 DINOv2"调整）

- hinge+模式读出仍 <15px 门失败 → 查实现 bug（连过拟合都不行 = 必有 bug）
- 单任务定位过、三任务泛化失败 → 先查数据覆盖/特征稳定性（数据×算法交叉）；仍失败 → 换读出
  架构（role slots cross-attention / 小解码器），再失败才考虑骨干
- 两者都失败 → 回去查标签/时序对齐/相机增广，不写表征负结果
- 定位成功但 text shuffle 无影响 → 语言主张不成立，换指令歧义配对任务

## 9. 注意事项 / 事故记录

- **`checkpoints/metric_field.pt`（v1，50px）已被 smoke 覆盖且不在 git**（checkpoints/ 在
  .gitignore）——原始失败 checkpoint 不可恢复，证据在 `logs/train_metric_v.log`。教训：
  诊断/实验 checkpoint 一律存新路径（如 `metric_field_v4_gate.pt`）。
- 阶段 A 训练（`train.py --dense-readout-mtvj ...`，PID 3174928，step ~7785）已于 2026-08-10
  按用户指示停止——它用的正是坏的 metric head（dense_evidence 仍有效，metric_tokens 是噪声）。
- GPU 已释放（15.1 GB）；重训 v4 时数据生成是 CPU 瓶颈（~7s/步，batch 4），全量 20k 步不可行，
  建议 tiny 门 → 短跑（2-5k 步）→ 按 §7 决策。
- `--l2-norm` 时温度初始 10 偏小（探针显示需 ~50-100 才能让期望读出工作；改用模式读出后温度
  不再关键）。
- 三个独立评审意见存档：Claude（网格对齐嫌疑→被 oracle 排除；σ/offset 监督；DINOv2 对照）、
  Codex（bias 分解消融；决策树；负结果措辞）、第三方（因子化损失；CE 梯度几何；四阶阶梯）。

## 10. 复现命令

```bash
# tiny 过拟合门（64 固定样本，特征预计算，~2-3 分钟 GPU）
python scripts/diag_probe_oracle.py --n-probe 32          # oracle + 模板探针
python train_metric_visual.py --fixed-data data/metric_tiny64.pt \
    --l2-norm --no-bias --loc-only --hinge-loss --mode-readout --offset-supervision \
    --steps 2000 --batch-size 8 --grad-accum 2 --save checkpoints/metric_field_v4_gate.pt \
    --device cuda
# 全量仿真流（门过后的下一阶段）
python train_metric_visual.py --tasks peg-insert-side-v3,assembly-v3,hand-insert-v3 \
    --l2-norm --no-bias --loc-only --hinge-loss --mode-readout --offset-supervision \
    --steps 5000 --batch-size 8 --grad-accum 4 --save checkpoints/metric_field_v4.pt --device cuda
```
