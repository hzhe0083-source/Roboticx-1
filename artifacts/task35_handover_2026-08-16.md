# task35 交接（2026-08-16 13:10 CST）

没有 `_step13000.pt`。下面的「13」是 **15k 赢家的 13 个失败 seed**（T0），不是 13k 训练步。

## 0. 一句话

DINOv2+MT-VJ+VA+FM H6 在 `peg-insert-side-v3` 上已经能插钉子。正式赢家是 **15k：37/50（74%）**。再训更差。完整 WAM **不要上**。下一步是拆这 13 局里的接近/抓取失败，不是加模块。

## 1. 工作树与分支

| 项 | 值 |
|---|---|
| 活跃工作树 | `/home/ryan/Documents/robot/ORA0-task35-fullfix` |
| 分支 | `task35-fullfix` @ `1a139ff`（写本文档时；推送后会再有一笔 handover commit） |
| 远程 | `https://github.com/hzhe0083-source/Xbot.git`（GitHub 提示已迁到 `Roboticx-1.git`） |
| 脏树，禁止改源码 | `/home/ryan/Documents/robot/ORA0` |
| `data` / `checkpoints` / `logs` | fullfix 里是指向 ORA0 的符号链接；**不要 commit** |
| Python | `/home/ryan/.venvs/pytorch-gpu/bin/python` |
| GPU | RTX 3080 Laptop 15.59 GiB；交接时无 compute 占用 |

不要在 ORA0 脏树上改 task35 源码。个人仓库可直接 push；密钥不得进提交。

## 2. 已完成（supported）

- 从 6k 档案 exact-resume 到 20k（中途 6832 被杀过一次，已从同一 6k SHA 接回）。
- 档案+SHA：1k, 2k, 3k, 6k, 9k, 12k, 15k, 18k, 20k。
- 50-seed 闭环只评了验收集 **12k / 15k / 18k / 20k**（用户叫停 3k/6k/9k）。
- 赢家选举绑 SHA；因果只跑赢家。
- 论文已写：`paper/ora0_paper.tex` §`sec:task35`，`paper/ora0_paper.md` §5.3.1（commit `d38b425`）。
- WAM go/no-go：`artifacts/task35_wam_go_nogo_2026-08-16.md`（`1a139ff`）。

### 闭环

| 步数 | 成功 | Wilson 95% CI | SHA |
|---|---:|---|---|
| 12k | 15/50（30%） | [19.1, 43.8] | `7da7e3db65c9118e…` |
| **15k** | **37/50（74%）** | **[60.4, 84.1]** | **`38885471fc09e15d…`** |
| 18k | 25/50（50%） | [36.6, 63.4] | `c917100070a9a613…` |
| 20k | 23/50（46%） | [33.0, 59.6] | `a42a687c05b0644d…` |

赢家文件：`checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000.pt`  
账本：`logs/task35_best_fm.json`

### 15k 因果（同 50 seed）

| 条件 | 成功 | Δ |
|---|---:|---:|
| none | 37/50 | — |
| dense-zero | 2/50 | −35 |
| temporal-reverse | 8/50 | −29 |
| geometry-zero | 29/50 | −8 |
| geometry-shuffle | 32/50 | −5 |
| roi-off | 39/50 | +2 |

成功靠 **dense MT-VJ + 4 帧时序**。运行时 ROI 不是瓶颈。

## 3. 契约（不要改）

- 任务：只有 `peg-insert-side-v3`（features `tasks[35]="Insert a peg sideways"`）。
- 角色：`[tool, pegGrasp, hole, pegHead]`，精度对 `(pegHead, hole)=(3,2)`。
- 主视觉：冻结 DINOv2 ViT-L/14-reg4；4 帧 `[d-6,d-4,d-2,d]`；**1024 token**，禁止退回 Pool16。
- Dense MT-VJ：只从 `[d-2,d]` 做加法 K/V。`dino_dense_metric=True` 时 **不要** 再开 V-JEPA dense 塔。
- 解码：只训 FM H6。`TASK35_ALLOW_DIRECT=1` 才能训 Direct。WAM 关，不挡验收。
- 评测：`--task-ids 35 --trials-per-task 50 --execute-steps 6 --horizon 500 --wam off --direct-head auto --flow-samples 1`。用 `--task-ids 35`，不要 `--max-tasks 1`。
- 语言：`language_hidden` 1807 行全相同；eval50 可走 `cached_task35_language`。
- 数据 SHA：payload `a27e4617da1c98cb326fbaefbb30183adf8761a3777dd83ceba7aa7845cdd9ec`；raw frames `d7699e9ef8e0ebfb…`；`block11.npy` `b8d3738d…`；`block23.npy` `ea740fb8…`；ROI `fca5ecd4…`。
- 验收里程碑：`(12000, 15000, 18000, 20000)`。1k–9k 只做机制，不能当选赢家。
- 训练时不要抢 GPU。进程用 `scripts/task35_proc.py`，不要 `pgrep -f`。

## 4. T0：13 个失败 seed（本次做完）

机器表：`artifacts/task35_15k_fail13.json`。  
源：`logs/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000_eval50.json`。

| seed | 桶 | min_d | best_step | 其他步数曾成功 |
|---:|---|---:|---:|---|
| 35002 | never-approach | 0.346 | 0 | 无（慢性） |
| 35004 | never-approach | 0.311 | 499 | 18k |
| 35009 | never-approach | 0.288 | 83 | 无（慢性） |
| 35014 | never-approach | 0.286 | 84 | 18k |
| 35028 | never-approach | 0.299 | 86 | 无（慢性） |
| 35033 | never-approach | 0.275 | 64 | 12k |
| 35007 | approach-no-grasp | 0.226 | 157 | 无（慢性） |
| 35027 | approach-no-grasp | 0.324 | 499 | 无（慢性） |
| 35044 | approach-no-grasp | 0.180 | 96 | 18k |
| 35046 | approach-no-grasp | 0.396 | 499 | 12k |
| 35021 | near-insert | 0.112 | 110 | 无（慢性） |
| 35036 | near-insert | 0.094 | 445 | 20k |
| 35039 | near-insert | 0.100 | 99 | 无（慢性） |

桶定义（与 go/no-go 一致）：

- never-approach：`grasp_r < 0.5` 且 `min_d > 0.2`
- approach-no-grasp：靠近或 grasp_r 高，但没抓住 / `min_d` 仍大
- near-insert：`min_d` 约 0.09–0.11

**结论（supported）**

- 10/13 是接近或抓取失败，不是插进去之后漂。
- 7/13 在 12k/15k/18k/20k **全部失败**（慢性）：`35002, 35007, 35009, 35021, 35027, 35028, 35039`。
- 6/13 在别的步数偶然成功过；再训不能稳定翻盘（18k/20k 宏平均更差）。
- 在这 13 个失败 seed 上，因果消融几乎救不回来（roi-off 只翻 4/13，且宏平均只 +2，当噪声）。

对照成功局：`min_d≈0.05–0.07`，`in_place=1`，`best_step` 中位数约 100。

T0 门已过：**≥8/13 从未真正到位** → 完整 WAM 是错工具。

## 5. 明确不要做

- 不要训 Direct。
- 不要 `--wam-joint` / 不要给 15k 挂 E7 WAM（H48 V-JEPA cache 对不上 H6 DINO）。
- 不要为了「再冲一把」从 20k 续训。
- 不要评 3k/6k/9k，除非用户改口。
- 不要改 ORA0 脏树。
- 不要用 slice / holdout / loss 选赢家。
- 不要把 74% 写成 MT50 分数。

## 6. 下一步（按顺序）

门槛不变：同一 50 seed **≥42/50**，且 Wilson 下界 **>74%**，才算赢过 15k。

1. **P0 数据，不是 WAM。** 针对慢性 7 seed + never-approach 6 seed，补接近/抓取 recovery 窗口（keep 15k 权重或只在新数据上短微调）。先测 `C−A`。
2. **P1 仅对 3 个 near-insert**（35021, 35036, 35039）：同一 15k 权重，`--execute-steps 2`。只有这 3 个翻了，才考虑很小的残差，不上 60M WAM。
3. **P2 可选回放。** 这 13 个 seed 的 RGB 还没从 env 里 dump。若要写进论文失败分析，用 15k ckpt 只滚这些 seed（会占 GPU）。
4. WAM 仍按 `artifacts/task35_wam_go_nogo_2026-08-16.md`：先过 `action[0:6] → Δ(pegHead−hole)` 探针，再改编 DINO-H6 契约。

## 7. 关键路径

```
工作树     /home/ryan/Documents/robot/ORA0-task35-fullfix
赢家 ckpt  checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000.pt
赢家 JSON  logs/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000_eval50.json
因果对比   logs/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000_causal_compare.json
选举账本   logs/task35_best_fm.json
状态账本   logs/task35_fm_status.md
失败表     artifacts/task35_15k_fail13.json
WAM 门     artifacts/task35_wam_go_nogo_2026-08-16.md
论文       paper/ora0_paper.tex  (§sec:task35)
数据       data/metaworld_longtraj_windows_h6_dino35_clean60_recovery30_v1.pt
cache      data/dino35_h6_clean60_recovery30_cache_v1
ROI        checkpoints/dino_metric_roi_task35_v2_native480_seed777_1k.pt
DINO 权重  ~/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/f3c408e77602bb412aa65fb03dfa0d5f95cb3832/model.safetensors
```

Cron 仍每 10 分钟跑 `scripts/monitor_task35_fm_train.py`。训练已结束，可留着当账本刷新，或删掉以免空转。

## 8. 复现赢家评测

```bash
cd /home/ryan/Documents/robot/ORA0-task35-fullfix
# GPU 必须空
scripts/run_task35_h6_eval50.sh \
  checkpoints/task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000.pt \
  task35_h6_dino_mtvj_fm_full15k_b6_sdpa_aux10b8_v1_step15000
```

不要再跑整套 suite：12k–20k 的 eval50 已齐，选举已写进 `logs/task35_best_fm.json`。
