# Drawer single-task joint training, 23 epochs

## Experiment

- LIBERO-10 local task 3: put the black bowl in the bottom cabinet drawer and close it.
- Joint architecture with native dual-tower exchanges, layerwise action experts,
  future joint/gripper delta supervision (weight 1), episode memory and T8 TBPTT.
- 50 success demonstrations, 804 valid P15 decisions, 114 nonoverlapping windows.
- Two L20 GPUs, global batch 2, one task, no anchor replay. 57 updates/epoch,
  23 epochs = 1311 updates. DINO frozen for 114 updates (2 epochs), tail six
  blocks unfrozen from update 115. Pretrained DINO/Qwen, fresh policy/optimizer.
- Seed 4042; checkpoint every 57 updates. No failure-data training.
- Single-task objectives retain explicit parameter ownership and distributed
  reductions, but bypass task-conflict projection with conflicts/comparisons 0.

## Deployment

New isolated directory: `/root/ora0_drawer_joint_e23_20260905`.
Existing remote training source/checkpoints were not modified. Existing success
HDF5 and longtraj frame assets are read-only inputs. New payload is
`data/drawer_episode_v10.pt`. Launcher is `launch.py`, run record `train_launch.json`.

Formal training launched with parent PID 7786:
- log: `/root/ora0_drawer_joint_e23_20260905/logs/train.log`
- checkpoint: `/root/ora0_drawer_joint_e23_20260905/drawer_e23.pt`

The formal run does not resume any smoke-test checkpoint.

## GPU acceptance fixes

The first real full-size mixed-precision runs exposed three integration errors:
1. Float32 modality type embeddings promoted visual tokens away from bf16 cached
   language. Type embeddings now match projected visual dtype before addition.
2. DINO unfreeze_last automatically enabled gradient checkpointing incompatible
   with scoped joint execution hooks. Joint trainer explicitly disables it after
   tail unfreezing, including resume; legacy behavior remains unchanged.
3. Cross-window memory reload cast visual memory to parameter float32 instead of
   active autocast dtype. Episode reload now uses the effective autocast dtype.

Failed smoke logs and their checkpoint files were preserved. `smoke4` completed
three updates across stage1->stage2, saved step 3, recorded two ranks of runtime
state and had finite model tensors. This is a smoke, not a success-rate result.

CPU verification: 1126 passed, 9 previously recorded failures, 8 skipped and 6
subtests passed in 85.02s. Log `/tmp/ora0-single-task-e23-tests-retry.log`.
An earlier full-suite attempt aborted during Python/torch import before tests;
retry completed as above. Independent reviewer failed from context exhaustion,
so it is not counted as approval. Main reviewed actual diff and tested fixes.
