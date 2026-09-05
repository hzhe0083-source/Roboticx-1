# 双塔前端与逐层 Action Expert：本地实现报告

## 交付范围

旧代码基线已提交并推送：`task35-fullfix@4fa69ea`。
新代码隔离在 `feature/dual-tower-layerwise-expert`，配置版本为
`dual_tower_expert_v1`；未指定版本的旧配置仍走 `legacy`。
没有部署、启动训练、访问远程训练进程或修改远程 checkpoint。

## 实现结构

1. `vision/dual_tower.py` 让 DINO/Qwen 后六层真正交互。保留原生 DINO patch 准备和 Qwen causal mask、rotary、hybrid block forward。每对层同时读取融合前快照，双向 cross-attention 输出回到下一层主干；最终输出经过原生 norm。局部 hooks 在 finally 清理，拒绝 gradient checkpointing。融合输出矩阵零初始化，初始等价于独立主干；不是零门控叠加零输出的死梯度初始化。
2. `vision/dual_tower_batch.py` 按每个观测重新联合编码；语言输出为 `[B,T,L,D]`，不按任务复用旧语言缓存。VA 将机器人状态投影成独立 token，附带模态类型 embedding，追加到 VA 视觉流；不追加到 DINO patch anchor。
3. VA 最后三层 action-query 状态输出 `[B,3,H,D]`，逐层交给三个 Action Expert block。动作作 Q，对应 VA 状态作 K/V。每块按 self-attention、cross-attention、FFN 做一次各自的残差更新，并具有动作位置编码和时间条件。
4. Expert 输出连续 FM velocity。保持 $x_t=(1-t)\epsilon+ta$、目标 $a-\epsilon$ 和外部 Euler 积分；层深不是积分步。在一次 chunk 内只计算一次前端/VA 条件。H50/P15 保留独立 H6、H15、H50 expert 的前缀隔离。
5. VA/World 的 proposal 次数、memory recurrence、T8 窗口内状态传播及监督公式未改。新模式的 World 当前视觉 anchor 来自融合后前端（但不含状态 token），未来监督仍为独立在线 DINO stop-gradient 特征；这是需要实验验证的表征差异，不宣称两者完全同分布。

## 训练与恢复

LIBERO 使用 `train_libero.py`，也可通过保留的 `libero_train.py` 入口。
新版本支持 fresh 初始化和同版本 `--resume`；`--resume-weights` 仅用于旧 dense continuation，不能把旧权重/优化器直接映射到新架构。
阶段 1：Qwen 后六层、融合、VA/World、Expert 可训练，DINO 参数冻结，但不能对整个联合前端使用 no_grad。
阶段 2：另外解冻 DINO 后六层及原有 norm 范围。
融合和状态类型 embedding 使用新模块学习率组。去掉新模式中闲置的旧 flow-condition 线性投影。
新版本允许配置 `--stage1-steps`、`--epochs`、`--max-steps`；任务集合、数据布局、GPU 数量和批采样契约仍按既有 profiles 校验。旧版本 schedule 不放宽。

保存记录新架构、fresh optimizer、逐观测 World 语言和禁用旧视觉缓存等契约。旧 checkpoint 不需要新增 architecture_version 契约字段即可恢复。
评测 `eval_libero_closedloop.py` 从 checkpoint 识别架构，新版本必须提供 `--qwen`；每次决策重新联合编码，仍校验 Qwen/DINO 基座 hash 和已训练参数。

以下是命令模板，**本次没有执行训练**。替换占位路径，不要指向正在训练的输出文件：

```bash
python train_libero.py train \
  --architecture-version dual_tower_expert_v1 \
  --data /path/to/validated_t8_h50p15_payload.pt \
  --longtraj /path/to/longtraj \
  --dino /path/to/dinov2_vitl14_reg4.safetensors \
  --qwen /path/to/Qwen3.5-0.8B \
  --save /path/to/new_joint_run.pt \
  --gpus 2 --batch-size 8 --mixed-tasks 2 --anchor-fraction 0 \
  --epochs 4 --stage1-steps 800 --max-steps 802
```

该示例针对已校验的 LIBERO-Long task 3+4、9843-row 数据 profile；800 是示例阶段边界，不是已验证最佳超参数。取消 `--max-steps` 才执行完整 epoch 计划。恢复时保留数据、epoch、阶段和其它契约参数，加 `--resume /path/to/new_joint_run.pt`。40-task profile 使用 batch32/mixed4/anchor0.25。

## 验证结果

CPU 命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
/home/ryan/.venvs/pytorch-gpu/bin/python -m pytest \
  -q --tb=short -p no:cacheprovider tests
```

最新完整结果：**1102 passed, 9 failed, 8 skipped, 6 subtests passed**，76.64 秒。
9 个失败与基线相同：2 个 metric 任务计数契约、5 个 Qwen fake model 缺少 layers、1 个 peer trace fake 缺少 config、1 个 OSMesa/PyOpenGL 环境问题。未删除这些失败测试，也不宣称全绿。
完整日志：`/tmp/ora0-dual-final-tests.log`。

新增验证包含：
- 融合恒等初始化、双向梯度、padding、逐层实际消费与 hooks 清理。
- 真正 tiny timm ViT + transformers Qwen3.5 hybrid（linear/full attention）CPU forward/backward，零初始化时与两个原生主干逐位一致，无预训练下载。
- 三层条件分别有梯度、状态 token 与 WM memory、H6/H15 不读取后段噪声/条件。
- 每决策语言 T 维 rollout、融合 VA evidence 的 World loss、各决策语言与状态梯度。
- mock LIBERO 环境中的真实 policy/Expert 推理：两次决策只执行两次前端，即使每次 FM 积分三步；语言更新、memory 传播。
- 实际 checkpoint writer + strict model state + AdamW state 恢复，下一次 CPU 参数更新逐位一致；joint schedule 放宽而 legacy 保持严格。

## 尚未验证与保留限制

- 没有全尺寸预训练双塔的真实数据/GPU训练、显存/吞吐测量、多 GPU 同步实跑，或真实 LIBERO simulator 新 checkpoint 评测。mock rollout 与 tiny native 测试不能替代这些验证。
- 联合前端以同一时刻整个任务组和全部视图执行，`--encode-batch` 不切分联合观测；full-size T8 图可能增加显存。native 内部 API 存在版本依赖，当前已安装版本测试通过。
- MetaWorld 训练入口仍是 legacy，不提供新架构 CLI 开关。静态语言特征不能冒充在线双塔，不宣称 MetaWorld 新模式已接通。
- 保留原 PCGrad 所有权：World-private + shared-DINO 接收 World 梯度，Qwen/fusion 仍是 action-private。普通 backward 可达不等于实际 World 优化器更新会应用该梯度。既有 publisher 梯度分区、窗口覆盖问题不混入本次算法变更。
- 独立审查仅覆盖部分前端实现，未获得完整独立审查通过结论。主模型已读取最终生产 diff，审查并修正 Fast-contributor 产物和新增测试。
