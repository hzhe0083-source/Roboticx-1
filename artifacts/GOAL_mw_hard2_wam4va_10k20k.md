# GOAL：把这次双任务 WAM4VA 实验跑完

直到下面「完成」全部打勾，不要另开题目。

## 现在在跑什么

- 任务：`0 assembly-v3` + `16 door-unlock-v3`
- 数据：`data/metaworld_longtraj_windows_h48_asm_doorunlock_fitted.pt`（1474 窗；door-unlock 丢掉 823 条对不上 `_fixed` 帧的窗）
- 脚本：`bash scripts/run_mw_hard2_wam4va_10k20k.sh`
- 日志：`logs/mw_hard2_wam4va_10k20k.log`
- 进程应看到：`train.py ... --wmrm-only ... --steps 10000`，然后自动切联合 `--resume ... --steps 20000 --save-step-copies`

## 配方（不许改回去）

1. 第一阶段 10k：`--wmrm-only`，握手关，只训世界头。目标是**下一决策最后一帧** DINO `[B,1024,16,16]`，不要整图均值、不要 4 帧平均、不要 1024→32 通道平均。
2. 第二阶段 20k：resume 世界头，握手开，VA+FM+WAM 一起训。
3. DINO-main 必须走任务局部采样 + `decode_cache_tasks=2`。禁止再混 batch 把 JPEG 解爆、GPU 掉到 10%。
4. World 只留 `checkpoints/mw_hard2_wam4va_world_10k.pt`。
5. Joint 每 1k 留 `checkpoints/mw_hard2_wam4va_joint_s{k}.pt`，最新一份 `..._joint.pt`。

## 完成标准

- [ ] `world_10k.pt` 写出来，log 里 stage 1 到 10000
- [ ] `joint.pt` 和 `joint_s1000`…`joint_s20000` 写出来，log 最后有 `done`
- [ ] 联合段 `flow` 相对开训明显下降；`world` 不得再回到 ~0.001 抄图地板
- [ ] 跑完评测：`bash scripts/eval_mw_hard2_wam4va.sh checkpoints/mw_hard2_wam4va_joint.pt data/metaworld_longtraj_windows_h48_asm_doorunlock_fitted.pt`
- [ ] 把最终数字写进 `artifacts/EXPERIMENT_STATUS.md`（两任务 10 trial success）

## 崩了怎么处理

- 先看 log 尾和 `nvidia-smi`。不要同时再开一份 `train.py`。
- 若只是进程死了、还没有 `world_10k.pt`：可以重跑整个 `run_mw_hard2_wam4va_10k20k.sh`。
- 若已有 `world_10k.pt`、联合没跑完：只跑脚本的 stage 2（或 `--resume` 最近的 `joint_s*.pt` 若以后接上 exact resume）。
- GPU 又掉到长时间 10% 且刷「解码 assembly/door-unlock」：先查是不是又混任务了，再改，不要硬加大 batch。

## 不要做

- 不要改回均值世界目标
- 不要 `--wmrm-only` 开着跑联合
- 不要用对不上索引的原始 2297 窗硬训 door-unlock
- 不要为了「占满显存」把两个任务塞进同一个 batch
