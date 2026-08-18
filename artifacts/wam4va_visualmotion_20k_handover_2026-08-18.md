# WAM4VA visual-motion 20k 交接

更新时间：2026-08-18 14:49 (+08:00)

项目：/home/ryan/Documents/robot/ORA0

## 0. 一句话

当前唯一主线是 A / final。它已从 scratch 完成 50、300、1000 步诊断，正在
1000 -> 20000 精确续训；本文快照约为 global_step=1450。

第 8 节拦截现已由独立 watcher 接管：A 的 20k evaluator 出现后只 STOP 父
PID；GO 则 KILL 父进程跳过 B，NO-GO 则 CONT 让 B 从 scratch 启动。先前的
runner_readiness_audit 没有真正发出 SIGSTOP。

最新用户决策覆盖原来的“无条件 A+B”计划：

1. A 先跑满 20k，并完成固定 held-out。
2. A 的两个任务都通过最终门，停止，不跑 B。
3. A 未通过且问题仍指向动作约束不足，才让 B 从 scratch 开始。

原 runner 仍会无条件 A -> B，所以必须执行第 8 节的最终拦截。

## 1. A / B 含义

A、B 不是两个架构，也不是把两个任务拆开。两者都联合训练 assembly-v3 +
door-unlock-v3，数据、WAM/VA、DINO、Flow Matching、proposal handshake、
batch 和优化器完全相同，只改变 action-ranking 辅助损失的位置。

| 变体 | 唯一差异 | 当前用途 |
|---|---|---|
| A / final | action-ranking 只约束第 8 层最终输出 | 当前主实验 |
| B / cycle | action-ranking 约束全部 8 个 WAM<->VA stage | A 失败后的备用实验 |

如果都跑，每个模型各自从随机初始化训练 20,000 optimizer steps，总计算量约
40k。按最新决定，A work 就不跑 B。

## 2. 当前运行快照

PID 会变化。接手后先重新核对 /proc 和命令行，不要盲用旧值。

| 项 | 快照 |
|---|---|
| runner 父 bash | PID 3727416，PGID/SID 3727416，starttime 104112637，状态 Ss+ 未 STOP |
| A trainer | PID 3809244，`--world-action-rank-stage final --steps 19000` |
| 第 8 节拦截 | PID 4061643，独立 PGID/SID，未与训练组共享 |
| 拦截脚本 | scripts/intercept_wam4va_a20k_skip_b.sh |
| 拦截日志 | logs/intercept_wam4va_a20k_skip_b.log |
| 拦截状态 | diagnostics/intercept_wam4va_a20k_skip_b.state |
| 拦截锁 | /tmp/ora0_wam4va_a20k_intercept.lock |
| 全局锁 | /tmp/ora0_wam4va_visualmotion_train.lock |
| A 输出族 | mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k |
| B 输出族 | mw_hard2_wam4va_visualmotion_oraclestgapcycle_v14.research20k |
| Python | /home/ryan/.venvs/openvla/bin/python |
| batch | 3 |
| 步速 / ETA | 约 5.1 s/step；20000 约 2026-08-19 17:10 +08 |

A 当前日志：

logs/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.train_step1000_to_step20000.log

A step1000 固定 checkpoint：

checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.step1000.pt

A rolling/final checkpoint：

checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.pt

rolling 文件在第一个 1k 保存点，即 global_step=2000，才会首次出现。之后每
1000 步原子覆盖，避免保存 19 个约 1.6 GiB 的副本。

不要启动第二个 trainer。A 运行期间不要改 train.py、evaluator 或 runner。

## 3. 固定数据与契约

- 训练：data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_train_v1.pt
- held-out：data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_eval_v1.pt
- split：data/metaworld_longtraj_windows_h48_asm_doorunlock_visualmotion_split_v1.json
- 原始来源：data/metaworld_longtraj_windows_h48_asm_doorunlock_fitted.pt
- 来源 SHA：5933ee297b4f4fbdb5b9e0d249a92bbe8ecc2c302a331459b677515c377b8093
- 任务：assembly-v3（id 0）+ door-unlock-v3（id 16）联合、balanced sampling
- T=4/H48，cycle=6，8 层 WAM<->VA，DINO 16x16x1024
- World contract：visual_motion_oracle_stgap_v7
- held-out contract：wam4va_world_action_heldout_v1
- runner：scripts/run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh
- evaluator：scripts/eval_wam4va_world_action.py

## 4. 已完成的可信验证

- runner Bash syntax：通过。
- runner protocol tests：12 passed。
- World/evaluator 定向测试：67 passed。
- 固定 episode split、来源 SHA、manifest 绑定：preflight 通过。
- A 的 step50、step300、step1000 checkpoint、held-out、exact resume：通过。
- step1000 SHA 实算和 JSON 绑定一致。
- 已跨过旧 causalfix 在约 step299 的 segmentation fault 区间。
- 当前没有 NaN、OOM、Traceback、新 segfault；RSS/FD/CUDA 稳定。

旧 causalfix 只有日志，没有 checkpoint，不恢复、不覆盖：

- logs/mw_hard2_wam4va_causalfix.log
- logs/mw_hard2_wam4va_causalfix.nohup

工作树原本很脏。不要回退、清理或提交不属于本实验的用户改动。

## 5. A step1000 结果

报告：

diagnostics/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.gate_step1000.json

checkpoint SHA：

51ace9e7b78318cff15540c5867d2ed8f76d47733fe1cd26f24bc8da5be6a28c

step1000 是 NO-GO，但比 step300 有进展：

| 任务 | top10 gain | shuffle degradation | zero degradation | static/copy |
|---|---:|---:|---:|---:|
| assembly-v3 | 45.14% | 8.06%，CI 下界略低于 0 | 18.60% | 约 1.116x |
| door-unlock-v3 | 21.84% | -0.73% | 0.63% | 约 1.130x |

task-macro top10 gain 为 33.49%。两个任务的 pred_all <= copy_all 和 target
permutation 均通过。当前失败点是静态仍高于 1.05x，以及 door 尚未证明使用
action。这只是 step1000 诊断，不能当作 20k 结论。

## 6. 最终 GO 门

两个任务必须分别成立，不能靠 pooled 结果：

1. 每任务 pred_all <= copy_all。
2. task-macro relative_gain_top10 >= 10%。
3. 每任务 world_static <= 1.05 * copy_static。
4. 每任务 shuffle 和 zero 的 top10 MSE 都比 real 至少差 5%，95% CI 方向为正；
   其中至少一个达到 10%。
5. target permutation 后 prediction 逐位不变，只允许 loss 改变。

最终 A 应产生：

- checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.pt
- diagnostics/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.gate_step20000.json
- logs/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.gate_step20000.log
- logs/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.train_step1000_to_step20000.log

必须核对 JSON 的 gate.decision、gate.passed、两个 per_task 的 passed、
checkpoint.global_step=20000 和 checkpoint SHA。不要用训练中单个 batch 的
world_task 行作最终判断。

## 7. 日常监控

~~~bash
cd /home/ryan/Documents/robot/ORA0
tail -n 3 logs/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.train_step1000_to_step20000.log
ps -eo pid,ppid,pgid,sid,stat,etime,%cpu,%mem,cmd | rg 'run_mw_hard2|train.py|eval_wam4va_world_action.py' | rg -v 'rg '
rg -n 'Traceback|CUDA out of memory|segmentation fault|\bNaN\b|\bInf\b|No space left' logs/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.*
df -h /home/ryan/Documents/robot/ORA0
~~~

正常状态只有一个 train.py。交接前 RSS 约 12--14 GiB、FD 65、CUDA reserved
约 13--14 GiB。磁盘约剩 36 GiB；不要额外保留大量 checkpoint 副本。

## 8. 最终拦截 A -> B

runner 的 20k 分支无条件遍历 final 和 cycle。A 的最终 evaluator 启动后，
按下面顺序只暂停父 bash：

1. 先在 ps 中确认精确的 A step20000 evaluator：
   checkpoint 必须是 oraclestgapfinal_v14.research20k.pt，output 必须是
   oraclestgapfinal_v14.research20k.gate_step20000.json；同时确认对应 tee。
2. 重新核对父 PID 的 /proc/<pid>/cmdline 和 starttime。不要按预计时间提前停。
3. 对正父 PID 发 STOP。例如 PID 未变时是：
   kill -STOP 3727416
4. 禁止负 PID、pkill、Ctrl-Z。它们会作用整个 PGID，把 evaluator 也停掉。
5. 确认父状态是 T，而 evaluator/tee 仍为 R/S。父 bash 只是在等待
   python | tee；只停父 PID 不传播给子进程，flock 也仍保持。
6. 等 evaluator 和 tee 都结束。父被停止无法 reap，因此它们可能显示 Z，这是
   预期。不要在 JSON 文件刚出现时读取，因为 evaluator 是直接 open 后写入。
7. 核对 evaluator raw exit：0=GO、512=NO-GO（即退出码 2），tee raw exit=0；
   再用 jq 完整解析 JSON，核对 step、contract、checkpoint path 和实算 SHA，
   gate log 末尾也必须有 GO/NO-GO report written。

条件分支：

- GO：外部验证完成后，父仍为 STOP。只终止父正 PID以跳过 B。最确定的是
  kill -KILL <parent_pid>。此时 evaluator/tee 已结束，checkpoint/report/log
  已落盘。随后确认没有 cycle trainer，父消失，锁释放。若用 TERM，顺序必须
  TERM -> CONT，绝不能 CONT -> TERM。
- NO-GO：只给父正 PID发送 CONT。runner 会 reap、内置校验报告并启动 B。
  必须确认新 trainer 是 --world-action-rank-stage cycle、--steps 50、
  从 scratch 开始。
- evaluator 非 0/2、tee 非 0、JSON/sha 不一致：发 CONT，让 runner 自己走失败
  路径并释放锁；不要启动 B。

runner_readiness_audit 未发出 SIGSTOP。2026-08-18 14:49 起由独立会话
`scripts/intercept_wam4va_a20k_skip_b.sh` 执行本节。当前 watcher PID 4061643，
PGID/SID 4061643，不是训练组。父进程在 evaluator 出现前必须保持 S/R，不要
提前 STOP。ZCode cron `每20分钟检查A20k拦截watcher` 只负责把死掉的 watcher
拉起来。

## 9. A 崩溃时如何恢复

先保留日志和 traceback，不覆盖任何文件。检查 runner/trainer 已退出、GPU 空闲、
锁释放。找最新有效 checkpoint：

~~~bash
ls -lh checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k*
~~~

global_step < 2000 时只能从固定 step1000 checkpoint接回。global_step >= 2000
时优先使用 rolling checkpoint。runner 不会自动从中途 rolling checkpoint重启。

读取 global_step：

~~~bash
A_SRC=checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.pt
A_STEP=$(/home/ryan/.venvs/openvla/bin/python -B - "$A_SRC" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(payload["global_step"]))
PY
)
A_REMAIN=$((20000-A_STEP))
~~~

exact resume 的 --steps 是 A_REMAIN，不是 20000。source 与 save 要用不同路径；
先把最新 rolling checkpoint 固定成独立 source，避免恢复时覆盖输入：

~~~bash
cp --reflink=auto "$A_SRC" "checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.resume_source_step$A_STEP.pt"
~~~

恢复命令必须逐项复制 runner 第 488--520 行当前 A 的全部参数，仅作三处替换：

1. --steps 改成 A_REMAIN。
2. --resume-exact 指向 resume_source_step$A_STEP.pt。
3. --save 指向新的 recovered 输出，例如
   checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapfinal_v14.research20k.recovered.pt。

启动前用 runner 中的 verify_checkpoint 逻辑核对 contract、batch=3、
world_action_rank_stage=final 和 global_step；使用同一个 flock，不能与任何
trainer 并行。不要加载 world_10k.pt、旧 joint 或 causalfix 权重。

当前完整 trainer CLI 可在进程存活时取得：

~~~bash
tr '\0' ' ' < /proc/3809244/cmdline
~~~

## 10. B 只有在需要时启动

若 A 最终 NO-GO 且用户决定继续，优先 CONT 原 runner，不另开 shell。B 起点必须
满足：

- --world-action-rank-stage cycle
- --batch-size 3
- step0 -> 50 段没有 --resume-exact
- 输出为 checkpoints/mw_hard2_wam4va_visualmotion_oraclestgapcycle_v14.research20k.step50.pt

B 的产物全部使用 oraclestgapcycle_v14.research20k，绝不能覆盖 A 的
oraclestgapfinal_v14.research20k。

## 11. 完成审计

只有以下证据齐全才可宣布完成：

- A final checkpoint 的 global_step=20000，SHA 与最终 JSON 绑定。
- A final held-out 包含两个任务，protocol 正确，target permutation 均通过。
- 若 A GO：B 未启动，runner/锁清理，A 产物保留。
- 若 A NO-GO 且继续 B：B 从 scratch，唯一 trainer，A 产物未改变。
- 没把 step1000 NO-GO 当成 final，也没用单 batch 指标替代 held-out。
