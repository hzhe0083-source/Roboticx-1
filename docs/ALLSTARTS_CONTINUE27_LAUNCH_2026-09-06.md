# Authorized all-starts H15 continuation, 27 epochs

User explicitly requested continuing the completed 23-epoch H15 weights for27 additional full-coverage epochs. Source is `/root/ora0_drawer_h15_b4_e23_20260905/drawer_e23.pt`, step690; original files remain unchanged.

New root `/root/ora0_drawer_allstarts_continue27_20260906`; code commit fdb0f95; launcher `launch.py`; formal parent PID9822. Log `logs/train.log`; output checkpoint `drawer_continue27.pt`. Formal run starts from original step690 weights, not the separate smoke checkpoint.

Settings: unified H15, all_starts_random_tbptt8_v1, batch4 globally, two L20 GPUs, encode-batch20, stage1_steps0 (keep trained DINO tail unfrozen), seed4042, same learning rates and AdamW state retained. New sampler and Action/World replay banks start empty. Source lineage recorded as continue_h15_fixed_to_all_starts_v1 / continued_adamw_reset_streams_v1 / source_global_step690. Exact resume uses new contract; migration is an explicit --resume-weights route, not a forged exact resume.

Native compact data contains11,684 legal decisions from50 drawer success demos, shape(11684,1,15,7). Data SHA25668fffe9f5557f6fa9e7a8238e845882675753e7f9946c4b41c3bc42c55ab44bb. All legal starts are covered every epoch.

Epoch lengths:532,534,534,536,529,535,531,531,534,524,531,530,534,532,528,538,530,529,536,536,536,532,535,528,534,532,528. Total14,369 updates. Saves at real epoch boundaries and final requested step, not the legacy save-every30 interval.

Verification:15 CPU trainer/eval tests passed after transfer support. Separate dual-GPU smoke completed3 updates with all floating model tensors finite, preserved AdamW counters reaching693, new sampler cursor3 and new memory contract. Formal run verified alive at step1/14369: action0.204986, World0.125712, visual0.063853, state0.004030, grad2.9709. Instantaneous GPUs57%/59%, memory37021/37041MiB; no sustained90% utilization claim or reliable full-duration estimate. No closed-loop quality claim.
