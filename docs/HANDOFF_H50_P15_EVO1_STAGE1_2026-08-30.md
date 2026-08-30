# Handoff: MetaWorld H50/P15 Evo-1 Stage 1

Last updated: 2026-08-30 16:06 CST  
Local workspace: `/home/ryan/Documents/robot/ORA0`  
Remote workspace: `/root/private_data/ORA0_next`  
Remote host: `root@ksai.scnet.cn:50493`

## 1. Next action

Wait for the active 500-trial evaluation, then compare its merged JSON with s3224. Do not resume training before reading the four-tier and per-task results.

```bash
cd /root/private_data/ORA0_next
pgrep -af 'eval_metaworld.py.*paper10k_v1'
for f in logs/*paper10k_v1_seed4042_h50p15_t*.log; do
  printf '%s ' "$(basename "$f")"
  grep -c '^trial ' "$f"
done
```

Completion artifact:

```text
/root/private_data/ORA0_next/logs/mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1_mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1_seed4042_h50p15_merged.json
```

## 2. Current state

| Item | State |
|---|---|
| H50/P15 implementation | Complete and synced |
| Official Evo-1 data | 2500/2500 demonstrations converted |
| Two-GPU smoke | Passed |
| Stage 1 first epoch | Complete at global step 313 |
| Formal closed-loop evaluation | Running, 5 shards, 500 trials |

No training process is active. The GPUs are currently used only by evaluation.

## 3. Experiment contract

This is a narrow SFT experiment testing long action prediction without changing model capacity or adding RL objectives.

| Setting | Value |
|---|---|
| Start | immutable antiforget s3224 |
| Policy | predict H50, execute P15, then replan |
| Capacity | original VA8/World7; no expansion |
| Objective | FM/BC only; task action PCGrad enabled |
| World Model | context forward retained; loss off and weights frozen |
| Encoders | DINOv2 and cached Qwen language frozen |
| Batch | global 48, 2×L20, mixed4, anchor 25% |
| Optimizer | AdamW, policy LR `1e-5`, global clip 1 |

Evo-1 reports 10,000 Stage-1 and 65,000 Stage-2 optimizer steps. Its effective global batch is inconsistent across the paper, repository, DeepSpeed config, and released checkpoint. This run matches optimizer steps only; do not call it sample-exposure matched.

## 4. Checkpoints and logs

Base:

```text
/root/private_data/ORA0/checkpoints/mw_mt50_antiforget_mixed4_rawcache50_anchor25_pcgrad_lr1e5_from_s21762_e10_v5_s3224.pt
SHA256 cf734545c0e6ec9eb355b777126983d4f2dfd2be9003003501c61851eedbc7df
```

Immutable Stage-1 epoch-1 checkpoint:

```text
/root/ora0_ckpts/mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1_s313.pt
global_step 313
SHA256 29851a4b013d346b9dc9b2047beb2200236431c1e05d29aeafb77c57942b8cdb
size 2.6 GiB
```

Mutable main checkpoint and training log:

```text
/root/ora0_ckpts/mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1.pt
/root/private_data/ORA0_next/logs/mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1.log
```

Both s313 files are on the server overlay. Preserve one immutable or verified model-only copy before a server restart.

## 5. Data

Official source:

```text
MINT-SJTU/Evo1_MetaWorld_Dataset
/root/evo1_metaworld_dataset
2500 successful demonstrations = 50 tasks × 50 episodes
188,489 frames; corner2 RGB 480×480; 4D state; 4D action
```

Converted data:

```text
/root/evo1_metaworld_longtraj_v1  (8.0 GiB)
index:    evo1_mt50_online_index.json
manifest: evo1_mt50_raw_manifest.json
```

```text
index SHA256    0a2acb79d503a3f1faaa396b7a1b0bf59ce4b2c73fd8b1294caba4092be5a799
manifest SHA256 9e10ad21723ba80d9b7110aa476708150c26d74304f1ac0872f5b275f0e20d3e
```

Episode lengths are 14–364. The old sampler required at least 61 frames and would have lost 841/2500 demonstrations and all data for nine tasks. The new `repeat_last_mask_actions_v1` contract samples every real observation, pads tensors with the final observation, and masks every nonexistent action out of FM.

Official raw xyz actions exceed `[-1,1]` in 19.8%/28.1%/38.4% of frames. Clipping uses the exact s3224 executed-action contract and matches MetaWorld action-space execution. State clipping is about 1.0–2.4% per dimension.

## 6. Code changes

1. `va_compound/policy/model.py`: H50, protected prefixes `(6,15)`, extension Flow head for actions 15:50.
2. `train.py`: strict H15→H50 migration and single-stream `--va-only` PCGrad.
3. `va_compound/vision/longtraj_frames.py`: H50 labels and opt-in short-episode masking.
4. `eval_metaworld.py` and `va_compound/world/world_contract.py`: policy H50 with World/P15 and P15 execution.
5. `scripts/run_mw_mt50_h50_p15_stage1_v1.sh`: preflight, train, exact-resume, eval.

Data tooling/tests:

```text
scripts/convert_evo1_metaworld_dataset.py
tests/test_convert_evo1_metaworld_dataset.py
tests/test_h50_p15_stage1.py
tests/test_online_episode_sampling.py
```

Migration copies the first 15 action queries exactly, prevents the 35 new tokens from changing the old H6/H9 Flow paths, and clones the old tail Flow into the H35 extension. World remains a 15-step cycle.

## 7. Verification

Remote tests:

```text
15 passed: converter + H50 migration + online sampling
42 passed: task-locality + eval selection + H15/P15 protocol
```

Two-GPU smoke:

```text
global batch 48; peak CUDA allocation about 40.2 GiB/card
no OOM, NaN, NCCL error, or traceback
245 World tensors byte-identical to s3224 after one update
154 extension Flow tensors finite and updated
formal evaluator: prediction_horizon=50, execution_horizon=15
```

Training curve:

| Window | FM | first15 | tail35 | raw grad |
|---|---:|---:|---:|---:|
| first 50 | 0.3800 | 0.2152 | 0.5245 | 1.7900 |
| last 50 | 0.2481 | 0.1752 | 0.3135 | 0.6590 |
| all 313 | 0.2825 | 0.1867 | 0.3675 | 0.9937 |

Stable throughput was about 6 seconds/step. Training completed without numerical or process errors.

## 8. Evaluation gate

Formal protocol: 50 tasks × 10 trials, seeds 4042–4051, horizon 400, H50 prediction, P15 execution, 5 trial shards on 2 GPUs.

| Baseline | Task-macro | Evo-style four-tier |
|---|---:|---:|
| s3224 | 253/500 = 50.6% | 36.5% |

Decision:

1. If task-macro and four-tier both improve, exact-resume toward 10,000.
2. If macro is flat but Medium/Hard/VH improve, run to step 1209 and reevaluate.
3. If all tiers drop materially, stop; inspect action/camera parity and per-task regressions before DINO unfreezing.

A one-epoch score near baseline does not prove H50 is useless: the new H35 head still has FM 0.3135. A large closed-loop collapse is nevertheless a stop signal.

## 9. Exact resume after a positive gate

```bash
cd /root/private_data/ORA0_next
nohup env \
  ONLINE_INDEX=/root/evo1_metaworld_longtraj_v1/evo1_mt50_online_index.json \
  FRAMES_DIR=/root/evo1_metaworld_longtraj_v1 \
  CHECKPOINT_DIR=/root/ora0_ckpts \
  H50_STEPS=10000 SAVE_EVERY=1000 \
  RUN_ID=mw_mt50_h50_p15_stage1_evo1official_actiononly_mixed4_anchor25_pcgrad_lr1e5_from_s3224_paper10k_v1 \
  bash scripts/run_mw_mt50_h50_p15_stage1_v1.sh resume \
  > logs/mw_mt50_h50_p15_stage1_evo1official_resume_to10k.launcher.log 2>&1 &
```

Use exact resume, not weights-only resume. Do not enable World loss, dense reward, ordinal loss, model expansion, or DINO unfreezing in this Stage-1 run.

## 10. Deferred work and repository caution

After a positive H50 result, Stage 2 may unfreeze only the last DINO block plus final norm at about `1e-6`, with trained DINO suffix save/eval support. Do not begin with four/all blocks on 46 GiB L20 cards.

Dense reward/RLT, cross-trajectory ordinal value, WM/action gradient guarding, and VA16/World205M expansion are intentionally out of scope.

The worktrees contain unrelated changes and untracked experiments. No commit was created. Do not use blanket `git add -A`, reset, checkout, or cleanup; stage only the files in Section 6.
