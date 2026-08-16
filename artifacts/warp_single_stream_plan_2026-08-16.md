# 单流双向 VA–WAM 计划清单（2026-08-16）

**一句话：** 不做候选动作。WAM 预测世界变化，VA 输出唯一动作；先前数学全部改写成这条结构上的约束、优化和验收门。

**架构合同（冻结，改合同先改本文）：**

```
VA 浅层 ⇄ WAM 浅层 → VA 中层 ⇄ WAM 中层 → VA 深层 ⇄ WAM 深层
                                                      │
                                                      ▼
                                              VA 单一动作 / FM
```

- 一次前向、一个动作 chunk。禁止 K 候选、树搜索、多次完整动作解码。
- WAM 没有 `Δa` / `Δv` / action head。最终动作只由 VA decoder/FM 发出。
- `wam=off` 必须与 VA-only 比特级一致。

---

## 0. 先前论证怎么用（不放弃）

| 旧论证 | 不再用来做什么 | 现在用来优化什么 |
|---|---|---|
| 函数类包含 / 零门控 | 证明完整双向无条件更优 | 写回/调制门零初始化，保证 `wam=off` 精确回退 |
| DPI / 条件互信息 | 堆更多层就更好 | 测 \(I(Y;H_{<L}\mid H_L)\) 的可检验代理：layerwise vs last-only |
| 信息瓶颈 | 通道越大越好 | 窄接口：innovation / 固定 M 个 world token，不倾倒全部 hidden |
| 多层重复噪声 | 再加一套 memory | 同轮 innovation + 分源 attention，禁止同一证据线性累加 |
| World 可被绕过 | 用 K 候选证书强绑 | 单流动作 residual 必须经 WAM 调制系数才能进入 VA |
| WAM 单点故障 | 静默 bypass | 显式 `wam=off` / 低置信保守门，不偷偷走第二条动作头 |
| 同参同 FLOPs | 比 WAM-on vs off | 比 last-only / repeat-last / layerwise / reverse-map / 强制调制 |
| Future leakage | 联合 noisy-future token | 未来只出现在损失右侧，前向图禁止真实未来 |

---

## 1. 目标结构：单流 World-Mediated Residual Modulation（WMRM）

每层只维护一条 VA 动作流 \(A_\ell\)。

1. VA 提出本层可执行修正基：
   \[
   U_\ell^{(1)},\ldots,U_\ell^{(r)}=E_\ell(A_\ell,o,s)
   \]
2. WAM 读当前动作假设，预测世界变化：
   \[
   \hat z_\ell=W_\ell(o,s,A_\ell,B_{\ell-1})
   \]
3. 与目标变化比较，只输出混合系数（可加置信门 \(q_\ell\)）：
   \[
   \pi_\ell=\mathrm{softmax}\,G_\ell(z^{\mathrm{goal}},\hat z_\ell),\quad
   A_{\ell+1}=A_\ell+q_\ell\sum_j\pi_{\ell j}U_\ell^{(j)}
   \]
4. 最终 \(a=H_{\mathrm{VA}}(A_L)\)。WAM 只许输出 \(\hat z,\pi,q\)，不许输出动作形状张量。

浅入深：浅层修几何/接触，深层修任务后果。纠错发生在每一层，而不是动作生成后再选一次。

---

## 2. 实施顺序（按停机条件推进）

### P0 — 合同与负对照（0.5 天，不训新权重）

1. 在设计文档写死：单流、无候选、无 `Δv`、VA 出动作、`wam=off` 恒等。
2. 列出当前代码违约点：`JointWorldActionFlow` 的 action residual、102-token 联合流、推理丢弃 world、noisy-future 进前向、12→8 重复读层。
3. 停机：合同未签字或仍计划挂 `v += αΔv` → 不写新模块。

### P1 — 因果探针，决定 WAM 是否该上（0.5–1 天）

4. 在目标长程任务上做：`Y ← state` vs `Y ← state+action` vs `Y ← state+shuffled-action`。
5. 目标用任务几何/阶段，不用整幅 DINO 1024D。
6. 通过门（预注册，数字用 held-out 方差校准后再填）：action-conditioned 优于 state-only；shuffle 吃掉大部分增益。
7. 停机：探针失败 → 先修数据/表示，不上双向 WAM。

### P2 — 单层 WMRM 骨架（2–3 天）

8. 实现：VA residual bases + WAM world head + 混合系数；无 action head。
9. 只在 VA 后段 1–2 个注入点先跑通（不是一上来 8 层全互写）。
10. `α/q` 零初始化；`wam=off` 比特级测试。
11. 未来目标只进 loss；前向禁止 `Y^*` 及其加噪版。
12. 停机：off 路径不等价，或 WAM 能直接吐动作形状 → 回滚。

### P3 — 浅入深双向（3–5 天，P2 过门后）

13. 按 stage 扩展：浅/中/深各一次握手，不是 12 层重复读 8 个快照。
14. VA→WAM 只发 innovation：\(e=m-\hat m(B)\)，同轮对已收方向做线性去重。不把整层 hidden 倾倒给 WAM。
15. 分源 attention：belief / geometry / progress 分开 CA，再门控融合。禁止 16+16+1 进同一个 softmax。
16. 层映射先固定 depth-aligned；另做 reverse-map 负对照。先不做可学习路由。
17. 停机：层数增加但 world 误差与动作误差不同向下降 → 缩回更少握手点。

### P4 — 反绕过与反回声（与 P3 同批测试）

18. 部署图去掉“未调制 residual 旁路”（训练可用课程：先均匀 \(\pi\) 再学 \(\pi\)）。
19. 干预：shuffle \(\hat z\)、把 \(\pi\) 换成训练集均值、反事实 \(z^{\mathrm{goal}}\)、交换动作假设。
20. 报告逐层 \(e_\ell^{\mathrm{world}}\) 与 provisional action error；不要求严格单调，但末层相对首层应下降。
21. 复制同一 innovation 2×/4×，置信度不得线性上涨。
22. 停机：均值 \(\pi\) 几乎不掉点 = WAM 未真正参与，不算成功。

### P5 — 故障回退（1 天）

23. 低 \(q\)：减小调制幅度，转入保守短程控制，不静默等于长期 WAM-off。
24. `wam=off` / NaN / stale：强制 \(\alpha=0\)，走冻结 VA baseline。
25. 任务成功率可以降，安全违反不得差于 VA-only。
26. 停机：WAM 坏时比 VA-only 更危险 → 不允许上线该注入点。

### P6 — 同参同 FLOPs 消融（2–3 天，同一 envelope）

全部条件实例化同一套 block，用 mask 改连通性；FLOPs 按真实 forward 计数（含握手次数，不含假参数）。

| ID | 条件 | 回答 |
|---|---|---|
| C0 | VA-only，算力补到同预算 | 是否只是多参数 |
| F | 只在最后阶段注入同样次数 | 逐层 vs 末层 |
| S | 重复读最终 WAM state | 多次注入本身 |
| A | depth-aligned 浅入深 | 层语义/深度对齐 |
| D | reverse-map | 对齐是否只是换了几个张量 |
| M | 强制经 \(\pi\) 调制（主方案） | world 是否必须参与动作形成 |
| B | 完整双向写回 VA hidden | 是否值得比 WMRM 更强耦合 |

27. 决策：`M` 必须先赢 `C0` 和 `F`；`A` 不赢 `S` 则不要讲“浅入深对齐”；`B` 不赢 `M` 则不上完整互写。
28. 停机：只在更大 FLOPs 下才赢 → 报告算力收益，不报告架构收益。

### P7 — 任务选择与评测

29. 不用 task35 当第一证明场（失败主因仍是接近/抓取，不是中程世界漂移）。
30. 选确需长程后果的任务（door-unlock / assembly 等），固定 paired seeds。
31. 同时报：成功率、world NMSE、shuffle/constant-π 敏感度、延迟 P50/P95、`wam=off` 安全。
32. 阈值用该任务 50-trial 方差预注册，不先写死百分点。

---

## 3. 明确不做

- 多候选动作、certificate 选动作、ToT/MCTS 动作树。
- 把旧 E7 `v += αΔv` 接到 DINO VA。
- noisy future latent 与动作 token 全互注意。
- 未过 P1 就训 60M 联合流。
- 用 attention 非零或 loss 下降代替因果干预。
- 把 VETO 方案 A（\(F_\psi\)、α=0）和本 action-path 实验混成一个合同。

Certificate / VETO 否决核若以后要做，放在 **WMRM 已经生成单一动作之后** 的审计层，不回头改成 K 候选生成。

---

## 4. 本周最小可交付

1. P0 合同一段话 + 当前代码违约清单。
2. P1 探针脚本与 go/no-go 数字。
3. P2 单注入点 WMRM：off 恒等 + 无动作头静态检查。
4. 三个干预：shuffle / constant-π / action-swap。

过这四项再加深双向层数。
