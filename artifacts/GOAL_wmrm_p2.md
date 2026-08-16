# Goal: WAM4VA 强制中介（GPT Pro 剩余三座桥）

**一句话：** 训练后不允许 \(W_A=0\) 且 \(q=0\)；ẑ 必须靠动作解释 \(g_{t+1}\)，且必须改变 LN 后的 FM condition。

## 必须

1. \(L_{A\text{-dep}}=\mathrm{ReLU}(m_A-[L_W(A^{\mathrm{shuf}})-L_W(A)])\)：同 \(V,s,B,L\)，只打乱动作再算 ẑ。
2. \(L_{\mathrm{med}}=\mathrm{ReLU}(m_C-\|C_{\mathrm{FM}}(z)-C_{\mathrm{FM}}(z^{\mathrm{shuf}})\|_2)\)：测量点是 `action_norm` 之后，不是 π。
3. `ca_geo` 只吃受监督的 `z_hat`（可广播），`z_spans` 不进动作残差。
4. 每个注入点都算上述损失；缺 `metric_g` 仍 fail-fast。
5. \(q=0\Rightarrow A'=A\)、无候选、无 `Δv`、与 direct/C² 互斥：保持。

## 不做

subgoal / 阶段词典 / 解冻 Qwen / 多候选 / 旧 `wam_joint`。
