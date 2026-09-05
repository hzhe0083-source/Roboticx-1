# Drawer H15 batch-four launch — 2026-09-05

User explicitly authorized stopping the prior run, retaining all assets, and starting new H15 training with global batch doubled from 2 to 4. GPU compute utilization near 90% is a requested target, not an achieved result.

## Running experiment

- Host: ksai.scnet.cn, SSH port 50493.
- Root: `/root/ora0_drawer_h15_b4_e23_20260905`.
- Formal launcher: `launch_b4.py`; launch record `train_launch.json`; parent PID 8543.
- Source commit: f5ddd31; architecture dual_tower_h15_v1.
- Data: `data/drawer_h15_v11.pt`, native shape (114,8,15,7), LIBERO-10 local task 3, 50 success demonstrations.
- Batch 4 globally, 2 ranks; encode-batch 20, seed 4042.
- 30 sampler updates/epoch, 23 epochs, 690 updates; stage1 60 updates; save every 30.
- Formal log: `logs/train.log`; checkpoint target: `drawer_e23.pt`.
- Starts fresh, not from the smoke checkpoint. Existing pretrained DINO and Qwen initialization retained.

## Verification and performance

Batch-four smoke `smoke_verified_b4` completed 3 optimizer updates, including stage2 after step1. Saved checkpoint identifies H15, contains two rank memory states, and all floating model tensors are finite.

Formal run verified alive and reached step10/690. Step10 action loss1.057594, World loss0.867359, gradient norm4.7416; no observed error in inspected log.

A 120-second formal-run sample at 2-second intervals (60 samples per GPU), following a 35-second delay, measured:

| GPU | Mean utilization | Median | p90 | Samples >=90% | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| 0 | 58.3% | 52% | 100% | 21.7% | 34979 |
| 1 | 57.5% | 48% | 100% | 23.3% | 34959 |

Source: `logs/train_gpu_initial.csv`. This is initial stage1 behavior, not a stage2 measurement. Sustained 90% is NOT achieved. Increasing batch and encoder chunk alone has not established saturation. Current episode rollout executes rows separately; this is a candidate bottleneck, not a profiled causal finding. No semantic batching refactor, synthetic GPU load, or further batch increase was silently introduced.

## Retained prior assets and launch corrections

Authorized old launcher7786 and workers7792/7793 stopped with SIGTERM and confirmed absent; GPU memory freed. All old files remain under `/root/ora0_drawer_joint_e23_20260905`, including the 8,225,598,343-byte checkpoint.

Initial new-directory deployment lacked configs; committed configs were added and data preparation succeeded. An initial generated launcher mistakenly retained batch2/encode5; its smoke process8244 was stopped after command inspection. Its log/files and initial launcher remain for traceability; it is NOT the formal run. The separately written `launch_b4.py` command was verified to use batch4/encode20 before the successful smoke and formal launch.
