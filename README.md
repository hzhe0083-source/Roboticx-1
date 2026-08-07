# VA Compound

最小实现只包含：Qwen3.5语言缓存、V-JEPA 2.1视觉接口、上一时刻目标视觉记忆、双向VA Attention和条件Flow Matching动作头。`bidir_va`是主结构，`uni_a`是禁止Memory/Action/Language写入Vision的等参数对照。

## 运行

复用现有GPU环境，不需要重新安装PyTorch：

```bash
/home/ryan/.venvs/pytorch-gpu/bin/python -m unittest discover -s tests -v
/home/ryan/.venvs/pytorch-gpu/bin/python train.py --mode bidir_va --steps 3 --batch-size 4 --sequence-length 4 --flow-steps 8
/home/ryan/.venvs/pytorch-gpu/bin/python train.py --mode uni_a --steps 3 --batch-size 4
```

PNPW是50条、18,464帧、30 FPS的单任务双臂示范。先使用环境相机生成冻结特征，再运行单任务Flow过拟合：

```bash
/home/ryan/.venvs/pytorch-gpu/bin/python prepare_pnpw_features.py --output data/pnpw_features.pt --batch-size 8  # 同时产出 flat 与 spatial 两套特征
/home/ryan/.venvs/pytorch-gpu/bin/python train.py --data data/pnpw_features.pt --single-task --steps 10000 --batch-size 8 --save checkpoints/pnpw_flow.pt
/home/ryan/.venvs/pytorch-gpu/bin/python train.py --data data/pnpw_features.pt --single-task --vision-pooling spatial --steps 10000 --batch-size 8 --save checkpoints/pnpw_flow_spatial.pt
/home/ryan/.venvs/pytorch-gpu/bin/python evaluate.py --checkpoint checkpoints/pnpw_flow_spatial.pt --data data/pnpw_features.pt  # 池化方式默认从 checkpoint contract 读取
```

转换器使用每3帧一个决策点、4个决策点的训练序列、4帧因果视觉窗口和8帧动作块；动作与状态按全数据1%/99%分位数归一化。PNPW只有`pick white cube into the basket`一个指令，因此该实验只验证任务过拟合与动作生成，不能验证语言切换创新。

准备实际模型约需5.8 GiB：

```bash
/home/ryan/.venvs/pytorch-gpu/bin/python prepare_models.py
```

模型固定为`Qwen/Qwen3.5-2B`与官方V-JEPA 2.1 ViT-B/16（384px，80M）。P0使用4帧窗口：encoder产生768维密集token，再固定池化为64个token；是否损伤细粒度动作信息必须在真实数据中验证。`--vision-pooling`选择特征变体：`flat`（1D自适应池化，历史A）或`spatial`（时间均值后2D网格池化，B）；`spatiotemporal`（C）保留时间轴但token数随帧数线性增长、不受64预算约束，尚未接入训练/评估管线。视觉输入接口为`[B,T,3,384,384]`，使用ImageNet均值方差预先归一化。

RTX 3080 Laptop上的单批次烟雾测试：V-JEPA 2.1约22.8ms，FP16 VA条件编码约3.06ms，8步Flow Head约5.12ms；不含视觉编码时VA+动作生成约8.19ms（122Hz），记忆仅0.25MiB。真实机器人频率仍以相机、数据搬运和控制器整链路为准。

训练时每个样本包含连续短序列，语言K/V只生成一次；VA逐时刻返回`VisualMemory`并在序列内反传。部署时使用无梯度语言缓存；每次新视觉只运行一次VA得到动作条件，随后仅运行轻量Flow Head。命令变化时将视觉记忆设为`None`。

```python
memory = None  # 新命令开始时清空
with torch.inference_mode():
    condition, memory = policy.encode_condition(
        vision, proprio, previous_action,
        language_cache=language_cache,
        visual_memory=memory,
        return_visual_memory=True,
    )
    actions = policy.sample_actions(condition, steps=8)
```

## 训练数据接口

`train.py --data features.pt`读取一个成对多指令tensor字典（维度由数据决定，脚本从数据推断config；PNPW实际为12维动作/状态）：

- `vision_tokens [N,T,64,768]`（flat池化）与可选`vision_tokens_spatial [N,T,64,768]`（时间均值+2D网格池化）
- `language_hidden [N,Nl,2048]`
- `proprio [N,T,12]`
- `previous_action [N,T,12]`
- `actions [N,T,H,12]`
- `pair_id [N]`与`instruction_id [N]`
- 可选`language_mask [N,Nl]`

`actions`必须先按每个动作维度归一化。默认要求`T>=4`。主Flow Matching训练独立采样高斯噪声$\epsilon$和时间$\tau$：

$$
a^\tau=(1-\tau)\epsilon+\tau a,\qquad u^*=a-\epsilon,
$$

$$
\mathcal L_{FM}=\|v_\theta(a^\tau,\tau\mid C)-u^*\|_2^2.
$$

同一`pair_id`的两个互斥指令在第0时刻拥有完全相同的视觉、状态和上一动作。辅助分支再令二者共享同一噪声$\epsilon_p$并固定$\tau=0$，所以$a_i^0=a_j^0=\epsilon_p$，Flow输入也完全相同：

$$
\mathcal L=\mathcal L_{FM}+\lambda_{pair}\operatorname{Huber}
\left[(v_\theta(\epsilon_p,0\mid C_i)-v_\theta(\epsilon_p,0\mid C_j)),(a_i-a_j)\right].
$$

因此配对分支只有语言不同，模型不能从带噪动作中偷看专家目标。多指令实验默认执行该契约；只有显式传入`--single-task`才会关闭配对损失，用于PNPW这类单任务过拟合。当前脚本训练预计算特征后的VA复合体和Flow Head；Qwen与V-JEPA保持冻结。
