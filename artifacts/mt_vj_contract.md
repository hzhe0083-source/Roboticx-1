# MT-VJ 接口契约 v1（2026-08-10）— 8 个并行实现 agent 的唯一互操作依据

仓库：/home/ryan/Documents/robot/ORA0（Python 3.12 + torch，venv: /home/ryan/.venvs/pytorch-gpu/bin/python）
完整设计：artifacts/mt_vj_design.md。本契约只规定接口，实现细节各自决定。

## 公共常量（所有模块统一）
- DENSE_TOKENS = 1152（2×24×24），H_DIM = 768（V-JEPA 特征维），D_PROJ = 192（投影维）
- ROLE_NAMES = ("tool", "object", "target", "interface")，N_ROLES = 4
- 角色查询来自语言：Qwen 缓存（language_hidden [B, L, 2048] + mask）经每个角色的 query 投影得到

## 1. va_compound/backbones.py 新增方法（VJEPA21Backbone 类内）
```python
def forward_hierarchical_dense(self, video: torch.Tensor,
                               out_layers: Sequence[int] = (5, 11),
                               ) -> dict[int, torch.Tensor]:
    """video: [B, W, 3, 384, 384]（W=4 帧窗，与 _encode 输入同构）
    → {5: [B, 1152, 768], 11: [B, 1152, 768]}（未池化全 patch，t→y→x 序）
    先读 VJEPA21Backbone._encode 实现；若官方 ViT 支持 out_layers 则复用，
    否则用 forward hook 或前向到目标层截断+后续层跳过。禁止改动既有行为。"""
```

## 2. 新文件 va_compound/metric_visual_head.py
```python
@dataclass
class MetricFieldOutput:
    p: torch.Tensor        # [B, N_ROLES, 2] 连续位置（图像坐标，归一化 0-1，y,x 序）
    visibility: torch.Tensor  # [B, N_ROLES] sigmoid
    offset: torch.Tensor   # [B, N_ROLES, 2] patch 内偏移（诊断）
    heatmap: torch.Tensor  # [B, N_ROLES, 24, 24] 分数图（诊断/可视化）
    relation: torch.Tensor # [B, 4] g_t = [p_eef−p_obj, p_obj−p_target, axis_alignment, depth]

class LanguageMetricField(nn.Module):
    """语言角色查询 → 全 patch 分数 + patch 内连续 offset → 连续位置/可见度/关系状态。
    分数 s_{r,n} = q_r^T W_K D_n + b_r(t,y,x)；位置 p̂_r = Σ softmax(s)(p_n + δ_n)，
    δ_n = ½tanh(f_offset(D_n, G_n, q_r))。D=W11·H11, G=W5·H5（投影到 D_PROJ）。"""
    def __init__(self, lang_dim=2048, h_dim=768, d_proj=192, n_roles=4)
    def forward(self, h5: Tensor, h11: Tensor, language_hidden: Tensor,
                language_mask: Tensor, coords: Tensor) -> MetricFieldOutput
    # coords: [1152, 3]（t,y,x 归一化 0-1 网格，来自 va_compound.live_vjepa._dense_coords() 或等价）

class RelationStateEncoder(nn.Module):
    """g_t + ν_t → 两个 d_model token（z_g, z_nu），加入每层 action cross-attention。"""
    def __init__(self, state_dim=4, d_model=512)
    def forward(self, g: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]

class MicroRefiner(nn.Module):
    """原像素 ROI 精修（0.5-1M 参数）：[B, 3, roi, roi] → [B, 4]（δp_y, δp_x, δz, contact）。"""
    def __init__(self, roi=96)
    def forward(self, roi_images: Tensor) -> Tensor
```
checkpoint 契约（train_metric_visual.py 保存）：`{"config": {...}, "metric_head": state_dict, "relation_encoder": state_dict, "contract": "mt_vj_metric_field_v1"}`

## 3. 新文件 prepare_metaworld_metric.py
```python
def make_metric_batch(task: str, rng: np.random.Generator, n: int,
                      frames_per_sample: int = 4) -> dict:
    """仿真器随机生成阶段 V 数据（无策略，任意观测）：
    {"frames": [n, 4, 384, 384, 3] uint8（当前帧+前 3 历史帧）,
     "language_text": [n] str, "keypoints": [n, 4, 2] float32（图像坐标 0-1, y,x）,
     "visibility": [n, 4] float32, "relation": [n, 4] float32, "contact": [n] float32}
    随机：task/reset/物体位置/臂位/视角/颜色。真值来源：env.data.body_xpos/site_xpos/
    mocap_pos + 相机投影（env.model.cam_*，先查 metaworld 现有投影工具或自己实现
    pinhole 投影：cam_pos/cam_fovy 推导，验证投影误差 <2px）。任务文本用
    scripts/build_longtraj_features.py 的 ENV_TO_TASK。"""
```

## 4. 新文件 train_metric_visual.py（阶段 V 预训练）
- argparse：--tasks 默认 "peg-insert-side-v3,assembly-v3,hand-insert-v3" --steps 20000 --batch-size 8 --lr 1e-3 --save checkpoints/metric_field.pt --device cuda
- 冻结 V-JEPA（prepare_pnpw_features.VJEPA21Backbone.from_pretrained(local_files_only=True) fp16）+ 冻结 Qwen（va_compound.backbones.QwenTextBackbone，语言缓存只算一次）
- loss = CE(heatmap, Gaussian 标签(σ=2px)) + Huber(p̂, p*) + λg·Huber(ĝ, g*) + BCE(visibility)，λg=1.0
- 每 1000 步打印 train RMSE（px）；保存 checkpoint（契约见 §2）

## 5. va_compound/model.py 扩展（VACompoundConfig 加 `dense_readout_mtvj: bool = False`）
- 仅 dense_readout_mtvj=True 时：VACouplingLayer（或等价 action transformer 层）每层增加：
  `z = CrossAttn(A, K_dense, V_dense); A_out = A_base + W_o·z`，W_o 零初始化（严格零，保证初始等价）
- dense K/V 输入约定（forward 额外参数，None 时行为与现在完全一致）：
  `dense_evidence: dict[int, Tensor] | None`（{5: [B,1152,768], 11: [B,1152,768]} 原维）
  `metric_tokens: Tensor | None`（[B, 2, d_model] 来自 RelationStateEncoder）
  K_dense = W_K·D + coord_emb；V_dense = W_V·[D, G, T, coord_emb]（T=ΔtH11）；coord_emb 查表或正弦
- 关键约束：1152 只做 K/V，query 仅 action tokens（≤48），绝不出现 1152×1152 自注意力
- 现有路径（dense_readout_mtvj=False）行为逐位不变

## 6. train.py 集成（新增参数，全部可选）
- `--metric-visual-checkpoint PATH`：加载并冻结 metric head + relation encoder（eval 模式 no_grad）
- `--dense-readout-mtvj`：开 model 的 dense 层
- 视觉特征：v1 用**在线编码**——batch 中 frames（需数据含原始帧；若 --data 是预计算特征文件则先支持 --live-vjepa 同款帧数据）→ forward_hierarchical_dense → metric head → dense_evidence + metric_tokens 注入
- 若实现中发现预计算路径更简单（复用现有 local_tokens 机制），允许选择，但必须与 eval 一致
- 训练 loss 不变（L_FM + L_pair）

## 7. eval_metaworld.py 集成
- --dense-readout-mtvj：与训练完全同构的在线编码 + dense 解码路径（帧窗契约 [d-6,d-4,d-2,d] 历史帧）
- --metric-visual-checkpoint 同参

## 8. tests/test_mt_vj.py（新增）
- forward_hierarchical_dense 输出形状/与 _encode 一致性（若 _encode 等价于 out_layers=(11,) 池化前）
- LanguageMetricField 前向形状、p 在 [0,1]
- MicroRefiner 前向形状
- dense_readout_mtvj=True 且 W_o 全零时输出与 False 逐元素一致（等价性测试，用随机小模型）
- metric checkpoint 保存/加载（weights_only=True 可加载）
- 现有测试必须全绿：python -m pytest tests/ -x -q

## 开发约定
- 每个 agent 只改自己模块的文件；model.py/train.py/eval_metaworld.py 由对应 agent 独占，不要互相编辑
- 所有新模块 import 用相对仓库根的绝对 import（如 `from va_compound.metric_visual_head import LanguageMetricField`）
- 不引入新第三方依赖（torch/numpy/PIL/metaworld 已有）
- 完成后运行自己模块的冒烟测试（形状/前向），报告改动文件清单 + 待集成点
