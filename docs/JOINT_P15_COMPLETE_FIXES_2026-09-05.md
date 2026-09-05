# 新实验：完整 P15 监督与执行段梯度修正

本报告补充并取代上一份新架构报告中的数据启动说明。仅修改本地新实验；未部署、重建远程数据或启动训练。未使用失败轨迹、失败标签、DAgger 或其它失败数据利用方法。

## 再审查结论

闭环结果只有 6 条成败记录，没有失败录像，不能确认抽屉的失败阶段或唯一根因。实际 checkpoint 谱系为 v7 s5000 → dense T4 s1000 → T8 s4924，不能作为单变量 T4/T8 对照。

已验证的训练问题：
1. H50-first 窗口边界导致每条示范末尾 35 个动作从不出现在 P15 执行段标签内。它们可能出现在非执行的 H50 尾段，不能称为完全没有监督。
2. 新旧两套模型原先都阻断执行第 7–15 步对 VA 条件的直接梯度。前缀 forward 隔离并不需要这项 backward 截断。
3. 评测器虽识别 T8，却将 World offsets 写死为四个；合法 T8 数据被启动校验拒绝。这是评测启动错误，不是机器人动作失败原因。

已确认但未作为确定 bug 修改：任务干扰、World/private 梯度分工、在线 DINO teacher 更新、较长 memory 状态传播。这些涉及原研究目标，现有成败日志无法支持任意改写 World 损失、PCGrad 或增加新模块。

## 修正

- 新数据契约 `libero_4suite_h50p15_t8_dualview5_p15complete_joint_v9`，metadata `window_bound=complete_p15_masked_h50_v1`。
- 新模式总是按 `length - 1 - 105 - 15` 计算 T8 最后合法起点。每个决策保留完整真实 P15；不足 H50 的尾部重复末动作但 validity=false，World endpoint 仍真实有效。旧数据与默认 legacy 生成行为保持不变。
- 最后合法决策的第 15 个标签正好是示范最后动作。对于之前均可容纳完整 H50 的 100 条 hard2 示范，按边界公式会新增 3500 个窗口：预计 task3 6434、task4 6909，总计 13343；这是公式推算，尚未实际重建远程数据。训练以新生成 metadata 为准，不再写死旧 9843 行。
- 新 Expert 第 7–15 步直接使用 live VA 条件。H6/P15 的 forward 前缀隔离仍保留，第 16–50 步对 VA 条件仍按原默认 detach。legacy 梯度行为不改。
- checkpoint 记录并严格校验 `execution_gradient_contract=p15_live_h50_tail_detached_v1`。此前新架构 checkpoint 不得伪装成可严格恢复的修正版训练。新实验必须用重新准备的数据 fresh 启动，后续才可同契约 resume。
- 评测识别新数据契约、正确校验八个目标偏移、检查新梯度契约，使用独立 protocol 标签，不把 joint 结果记录成旧模式。

## 本地 CPU 验证

- 真实 HDF5 格式的合成 50 示范准备流程（仅 mock 官方任务注册和大图像存储）覆盖 prepare → padding/masks → donors → save/load → strict validator。
- 长度 160 的示范生成 40 起点/示范，最后起点 39、最后决策 144，对应 actions[145:160]，最后 35 个 H50 标签无效。
- 检查错误数据契约、无效 P15 拒绝；masked padding 无 loss 梯度；改变尾部噪声不影响 P15 输出。
- 检查第 7–15 步 loss 对三个 VA 条件层均有梯度、未来非执行段保持 detach，legacy 不变。
- 原生 tiny Qwen3.5/timm 测试补充 DINO 参数冻结时 fusion/Qwen 梯度仍然存在。
- 保存与恢复测试检查新契约并验证下一步 AdamW 参数更新逐位一致。

完整 CPU 回归：**1107 passed, 9 failed, 8 skipped, 6 subtests passed**，83.18 秒。9 项失败与之前相同（metric 计数、Qwen fake model、peer trace fake、OSMesa 环境），没有新增失败。日志 `/tmp/ora0-p15complete-tests.log`。本地为 HDF5 测试安装了 h5py；未改变远程依赖。

## 新实验命令模板（未执行）

必须使用新的输出路径，不能覆盖旧数据或正在使用的 checkpoint。以下是 hard2 示例：

```bash
python prepare_libero.py \
  --architecture-version dual_tower_expert_v1 --dense-windows \
  --hdf5-dir /path/to/official/datasets --suites libero_10 --local-task-ids 3,4 \
  --longtraj /path/to/validated/longtraj \
  --data /path/to/new_joint_p15complete.pt

python train_libero.py train \
  --architecture-version dual_tower_expert_v1 \
  --data /path/to/new_joint_p15complete.pt \
  --longtraj /path/to/validated/longtraj \
  --dino /path/to/dinov2_vitl14_reg4.safetensors \
  --qwen /path/to/Qwen3.5-0.8B --save /path/to/new_joint_p15complete_ckpt.pt \
  --gpus 2 --batch-size 8 --mixed-tasks 2 --anchor-fraction 0 \
  --epochs 4 --stage1-steps 800 --max-steps 802
```

800 是示例阶段边界，不是验证最佳值；`max-steps` 用于跨阶段 smoke，完整训练移除该项。新行数意味着每个 epoch 的更新数增加，不能直接把同 epoch 成绩解释成等算力对照。

本次更改修复已确认的训练/评测错误，不能据此宣称成功率提高。全尺寸 GPU 显存、真实数据重建、训练稳定性和闭环成功率仍未验证。WM 当前融合视觉与纯 DINO 未来监督之间的表征差异也仍需实验，不新增失败数据或伪造对照结果。
