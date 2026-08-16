# Goal: WAM4VA（WAM for VA；实现模块曾名 WMRM）

**验收：** VA 输出唯一动作；WMRM 做世界预测 + belief/innovation + 分源调制；`wmrm=off` / 零门与 VA-only 比特级一致。

已实现：

1. 一次前向、一个 chunk。无 K 候选，无 `Δv` / action-shaped head。
2. `A ← A + q · (Σ π_j U_j + 分源 CA)`，`q` 与 source gates 零初始化。
3. 多层握手：`--wmrm-inject last|all|even`，belief/innovation 跨层传递。
4. Innovation：`e = evidence − predict(belief)`，同轮对 prev 线性去重。
5. 空间读出：learned queries CA 读 vision，不用纯 mean 作为唯一证据。
6. 时序分段：3 个 span world heads。
7. 分源 attention：belief / geometry / progress 三套 CA + 独立门。
8. `U = basis([A, evidence, proprio])`；mixer dropout 0.3；per-step π。
9. `world_goal` 前向封死；`wmrm_world_loss(z_hat, next metric_g)` 进 `action_total`。
10. `--wmrm` 与 `--wam-joint` 互斥。
11. Language 暂保持现状：VA 每层读文本 K/V；WAM4VA 可问整句 summary。**不做 subgoal。** 三段时钟只用于内部 pooling，mixer 只用受监督的 `z_hat`（一步未来）。
12. GPT Pro 审查后：π-KL 关 dropout；world loss 覆盖每个注入点；缺 `metric_g` fail-fast；与 direct/C² 互斥；innovation 只去正相关；CA 不再零输出初始化。
