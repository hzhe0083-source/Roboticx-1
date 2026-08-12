# E7 MT-VJ Joint 联合微调交接（当前执行快照）

更新时间：2026-08-12 02:15（Asia/Kuala_Lumpur）
项目：`/home/ryan/Documents/robot/ORA0`
Python：`/home/ryan/.venvs/pytorch-gpu/bin/python`
来源：`/home/ryan/Documents/Codex/2026-08-11/new-chat/outputs/E7_MTVJ_RELATION_JOINT_HANDOFF_2026-08-12.md`（Codex 版交接，306 行）

## 一句话

**50k 固定点 + joint 30k 联合续训（VA 5e-6 + relation 2e-5）= 有效 80k**，训练健康推进中。

## 当前训练（唯一主线）

- **Python PID：260533**（核对命令行后再 kill/resume）
- 日志：`logs/e7_mtvj_joint80k.log`
- checkpoint：`checkpoints/e7_mtvj_joint80k.pt`（`--save-every 1000` 覆盖写）
- 进度：**局部 step ~420 / 30000**（有效 48k+2k+420 ≈ 50.4k），步速 ~1420 step/h
- 预计完成：约 20-21 小时 → **8/12 深夜 ~22:00-23:00**

启动命令（resume 照抄）：
```bash
setsid nohup /home/ryan/.venvs/pytorch-gpu/bin/python -u train.py \
  --data data/metaworld_longtraj_windows_h48.pt \
  --dense-readout-mtvj \
  --metric-visual-checkpoint checkpoints/metric_field_v4.pt \
  --mtvj-train-relation --lr-mtvj-relation 2e-5 \
  --single-task --va-layers 8 \
  --lr 5e-6 --batch-size 16 --steps 30000 --seed 0 \
  --flow-cond adaln --flow-layers 6 --flow-steps 8 \
  --task-sampling weighted --prev-dropout 0.1 \
  --resume checkpoints/e7_mtvj_frozenrel50k.pt \
  --save checkpoints/e7_mtvj_joint80k.pt --save-every 1000 \
  > logs/e7_mtvj_joint80k.log 2>&1 < /dev/null &
```
**必须带 `--mtvj-train-relation` 和 `--lr-mtvj-relation 2e-5`**，否则架构不对。
启动日志必须出现：`冻结 V-JEPA` / `冻结 metric head + 可训练 relation encoder` / `trainable=10,240` / `relation action path 加入 optimizer ... lr=2e-05` / `relation encoder 从主 checkpoint 严格恢复`（均已确认 ✓）。

## 关键 checkpoint

| 文件 | 含义 | 状态 |
|---|---|---|
| `checkpoints/e7_mtvj_legacy48k.pt` | 修复前代码产物，dense-only 基线（无 relation） | 已固定（606MiB） |
| `checkpoints/e7_mtvj_step46k_pre_fix.pt` | 46k 备份 | 存在 |
| `checkpoints/e7_mtvj_frozenrel50k.pt` | **joint 训练起点（硬链接固定，有效 50k）** | 已固定（634,824,710 字节，独立 inode） |
| `checkpoints/e7_mtvj_joint80k.pt` | 本次输出（覆盖写） | 训练中，step1000 后首次落盘 |
| `checkpoints/e7_mtvj_contractfix_ft1k.pt` | 旧路线 FT1 残留（step443，已弃用） | 保留勿删 |

## 已完成的步骤（可信）

1. 旧 80k 进程（PID 3430476）停止，legacy48k 固定，旧 cron 看护删除。
2. 代码修复（H11→Pool16 统一、relation 保存/严格恢复、ν 首步置零、contract v2）落地并回归：`75 passed, 1 deselected`（含配对验收）。
3. 单线 32k（PID 165513）跑到局部 step 3010 后按新文档停止，50k 固定点完好。
4. Joint 30k（PID 260533）已启动，前 100 步 `rel_grad=` 非 0（~0.001-0.004），梯度路径正常。

## 待办（按文档顺序）

- [ ] 局部 step 1000：CPU 契约检查（对比 frozenrel50k，`metric_relation_joint_trained=True`、`metric_relation_lr==2e-5`、relation `max_abs_delta > 0`、全有限）
- [ ] 每 1000 步健康检查：`rg -n 'Traceback|CUDA out of memory|\bNaN\b|\bInf\b|No space left|Xid'`；立即停止条件同前
- [ ] 30k 跑满自然退出后：
  1. smoke6：`logs/e7_joint80k_smoke6.log`（6 任务×1，含 metric checkpoint）
  2. 50k 基线 49×10：`logs/e7_frozenrel50k_full49x10.log`
  3. 80k 候选 49×10：`logs/e7_joint80k_full49x10.log`
  4. `scripts/closedloop_ci.py` 两日志 → 汇总/CI
  5. `scripts/compare_closedloop_paired.py` 两日志 → **必须输出 `FINAL GATE: ACCEPT CANDIDATE`** 才算证明提升
- [ ] 接受条件：候选 success ≥ 60% **且** paired task-bootstrap 95% CI 下界 > 0；否则保留 50k 基线，如实记录"未证明提升"

## 关键行为模式（勿误判）

- 日志 flush 缓冲：静默 1-2 分钟批量写入正常；卡死判据 = mtime 停更 >8 分钟且进程/GPU 无活动
- 内存 avail 常在 2-7G 波动，此前最低 1-2G 安全度过；危险信号 = avail <3G **且** 日志停更
- 前几百步 loss 均值短暂变化（relation 适配）正常，不停机
- 磁盘 13G 可用：不要保存多个 checkpoint 副本

## 最终登记模板（完成后填写）

```text
base checkpoint: checkpoints/e7_mtvj_frozenrel50k.pt
joint steps: 30000
effective total: 80000
final checkpoint: checkpoints/e7_mtvj_joint80k.pt
relation first-20-step grad: normal
joint checkpoint delta check: pass / fail
training health: normal / abnormal
smoke6: x/6
baseline full49x10: x/490
full49x10: x/490
aggregate success:
macro success:
95% CI:
paired delta + 95% CI:
improvement confirmed: yes / no
success >=60%: yes / no
ROI/WAM added: no
remaining risks:
```
