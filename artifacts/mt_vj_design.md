# MT-VJ：Metric-Temporal V-JEPA for Action（GPT Pro 最终定稿，2026-08-10）

目标：视觉子系统在 MT50 上全面优于 Evo-1（空间精度/时序/语言定位/动作层视觉访问/闭环频率/显存）。
不做开放世界图文理解（那是 Evo-1 的通用 VLM 预训练优势）。

## 四层结构
```
四帧 RGB → 冻结 V-JEPA 2.1 → H5+H11 原生 2×24×24 dense evidence
  → 语言条件 Metric Field Head（tool/object/target/interface 连续位置 + patch 内 offset + 相对深度/尺度 + 可见度 + 时序位移）
  → Persistent Dense Action Readout（每层 action query 读完整 1152 patch + metric relation tokens + temporal innovation tokens）
  → VA + Flow/Direct Action Head
```

## 核心设计
1. **保留完整 H5/H11**：1152→288→25 聚合丢信息；改为投影到 192-256 维（D=W11·H11, G=W5·H5, T=Wτ·ΔtH11），0.42MiB/decision（vs 768D 1.69MiB，带宽降 4 倍）。
2. **语言定义度量场**：Qwen 一次性生成 4 角色查询 Q_L={q_tool,q_object,q_target,q_interface}，直接查询视觉网格 s_{r,n}=q_rᵀW_KD_n+b_r(t,y,x)；每 patch 预测连续偏移 δ_{r,n}=½tanh(f_offset(D,G,q_r))；位置 p̂_r=Σ softmax(s)(p_n+δ_n)——**patch 步距不是精度下限，连续 offset 读出是**。
3. **阶段 V：控制度量视觉预训练（关键新意）**：冻结 V-JEPA，只训 Metric Field Head（2-4M 参数）。仿真器免费真值（object/site/target/EEF 坐标→投影到图像）自动生成大量随机观测（随机 task/reset/物体位置/臂位/视角/颜色），损失 = CE(heatmap) + Huber(p̂,p*) + λg·Huber(ĝ,g*)，g* = [p_eef−p_obj, p_obj−p_target, axis, depth]。**视觉预训练不受 50 demos/task 限制**。训完冻结。
4. **阶段 A：策略训练不变**：L = L_FM + λ_pair·L_pair。action query 每层零初始化 residual：A^{l+1}=A_base^{l+1}+W_o^l·Attn(A^l, K_dense, V_dense)，K/V 含 [D,G,T,coord]；1152 永远只做 K/V，query 仅 8 个，无 1152² 自注意力，每层成本 8×1152。
5. **Micro-Refiner（0.5-1M 参数）**：近接触时原像素 64-96px ROI crop → depthwise CNN → δp^micro, δz^micro, c^contact；p_final = p_VJEPA + δp_micro。同批 simulator 标签训练，不进策略 loss。
6. **闭环（第一版不复杂化）**：监督过的关系状态 g_t=[p_tool−p_obj, p_obj−p_target, Δp, depth, contact] + ν_t=g_t−g_{t−1} → 两个 token z_g/z_ν 加入每层 action cross-attention。不声称物理 Jacobian。servo 支路（clip(K(g*−g),±δ)）仅在关系状态验证准确后加。

## 论文主张（vs Evo-1）
| Evo-1 | MT-VJ |
|---|---|
| 单帧 InternVL token | 四帧 V-JEPA 时空 token |
| 通用视觉语言对齐 | 控制专用角色与度量对齐 |
| 动作层读 fused tokens | 动作层读完整 dense + metric tokens |
| 448 中央裁剪 | 全局视野 + 局部连续坐标 |
| 无毫米级几何输出 | 显式关系状态 + patch 内 offset |

新意表述：语言角色度量场 + V-JEPA 时序 evidence + 持续 dense action readout + 原像素局部精修。

## 评估协议（匹配后对比）
四组：8D state+同 crop（公平击败 Evo-1）/ 4D EEF-only+RGB（视觉更强）/ state-only / blank+shuffled RGB（因果）。视觉硬指标：PCK@5mm>90%、relation RMSE<5mm、near-contact RMSE、遮挡鲁棒。

## 代码落点
- va_compound/backbones.py: forward_hierarchical_dense(video, out_layers=(5,11)) → {5:[B,1152,768], 11:[B,1152,768]}（官方 V-JEPA 原生支持 out_layers）
- va_compound/metric_visual_head.py: LanguageMetricField / PatchOffsetHead / RelationStateEncoder / MicroRefiner
- prepare_metaworld_metric.py: 仿真器自动生成（instruction, 4帧RGB, EEF/object/site/target 坐标, 可见度, 接触）
- train_metric_visual.py: 只训 Metric Field + Micro-Refiner（V-JEPA/Qwen 冻结，无 policy）
- va_compound/model.py: VACouplingLayer 加 dense_key/dense_value/metric_tokens
- train.py: 加载冻结 metric visual checkpoint，策略仍 L_FM+L_pair

## 与 E7 主线衔接
E7 baseline 必须完成（对照基准）；10 步序列第 4 步（ST288 vs H11 probe）验证信息量 → probe 阳性后实现阶段 V（metric 预训练，可并行开发）→ 阶段 A（dense readout + metric tokens）→ Micro-Refiner → 协议四组对比。
