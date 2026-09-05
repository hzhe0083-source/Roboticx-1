# Full-coverage H15 batch8 continuation

## Verified launch

- Remote root: `/root/ora0_drawer_allstarts_b8_20260906`.
- Formal parent PID: 10650; verified through update 40/7202.
- Log: `logs/train.log`; planned checkpoint: `drawer_continue27.pt` (epoch-boundary save).
- Training source revision: bceaccc. Original launcher recorded stale b29cf47; additive `train_verified_provenance.json` corrects provenance and includes the deployed gradient-microbatch file SHA256. No live code was changed.
- Global batch8, two GPUs, four windows per rank, gradient microbatch2, compatible rollout group limit2, joint observation chunk4.
- 27 exhaustive epochs, 7202 total optimizer updates. Each epoch covers all 11,684 legal starts across 50 demonstrations. H15/T8 and offset-specific chronological memory remain unchanged.

## Source and migration

Source is `/root/ora0_drawer_h15_b4_e23_20260905/drawer_e23.pt`, completed fixed-window update690 (23 epochs). Model, Qwen, DINO and AdamW are inherited; sampler and runtime streams restart at a clean boundary. No completed all-start epoch was available. The prior batch4 all-start process was stopped at approximately logged update210 before its first epoch checkpoint, so those unsaved updates are NOT retained and their data are replayed. All old assets remain.

## Validation

Full-graph batch8 ran out of GPU memory around44.5 GiB; the effective-batch8 microbatch path instead completed three smoke updates and saved `smoke_b8_accum2_fixed.pt`. Model/Qwen/DINO saved tensors were finite, optimizer contained980 state entries, Action/World sampler states matched, source step690 and execution configuration were verified.

Local targeted regression: 31 passed in7.97 seconds, covering NumPy frame slicing, weighted gradient accumulation, simulated distributed denominator with inactive microchunks, compatible rollout parity, joint chunks, execution flags, rebatch guards and all-start coverage. The simulated denominator test is not a substitute for a multiprocess numerical parity test. Remote smoke exercised real two-GPU training. Independent reviewer failed due to context exhaustion; no independent approval is claimed. No new full-suite run or closed-loop success evaluation was performed.

## Measurements

Formal run sampled every5 seconds for4 minutes. Excluding the initial12 samples (startup/warmup), mean utilization was73.97% /74.69%, with sampled peaks100%. Sampled peak device memory was39055 /39035 MiB (about38.1 GiB each). These are device samples, not allocator peak counters. Sustained90% was not achieved.

Observed update30 to40 took120.36 seconds (2-second polling resolution), or12.04 seconds/update. Reconstructed exact sampler batches31–40 contained402 active decision starts, giving3.34 unique training starts/second. Action and World streams both process these starts; throughput does not double-count them. This short segment does not establish an epoch-wide speedup against batch4. Straight-line remaining estimate at update40 is about24 hours, subject to window lengths and checkpoint I/O.

At update40: action0.164753, World visual0.069271, state0.011576. Logged microbatch diagnostics average microbatch scalar values and are not a fixed validation metric or necessarily the full-batch decision-weighted objective. Gradient scaling uses the complete task's real-decision denominator.
