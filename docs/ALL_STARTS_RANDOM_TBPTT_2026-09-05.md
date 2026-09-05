# Full legal-start coverage with randomized TBPTT windows

## Implemented locally

Use `--architecture-version dual_tower_h15_v1 --window-sampling all_starts_random_tbptt8_v1` for both `prepare_libero.py` and `train_libero.py`. Defaults preserve the old contiguous mode. New compact contract: `libero_joint_all_starts_t8_h15_state_delta_v12`.

Every legal decision start d in 0..L-16 appears exactly once per epoch. Within each original demonstration, d modulo15 identifies independent chronological replays. Each epoch randomizes replay order and first-window length (1..8 decisions); subsequent windows contain up to8 decisions. This changes window boundaries without shuffling time within a replay. It is exhaustive coverage, not replacement sampling.

Storage contains one T1 row per legal decision, native H15 action labels and frame references. Images are not duplicated into fifteen pre-materialized window datasets. The wrapper assembles and pads T8 windows at access time. Invalid slots and padded targets are masked.

Memory is owned by replay_id = original_episode_id*15+offset. Each offset starts with empty memory at its first observation; no claim is made to have encoded its skipped prefix. Memory carries chronologically across windows and detaches between updates. Replays do not share memory. New memory contract `offset_replay_tbptt8_v1` rejects old bank states. Evaluation still resets per environment trial.

## Schedules and compatibility

Epoch lengths may differ because window boundaries are randomized. The trainer computes deterministic epoch_lengths, defaults to freezing DINO for the first two full epochs, saves at real epoch boundaries and final requested step, and records/checks sampling, memory, data and epoch-length contracts on resume. Explicit stage1 overrides and max-steps smoke runs remain supported. Legacy and fixed H15 checkpoints remain separate; compact data with the old CLI sampling default is rejected.

Example flags to add to a separately configured run:

```bash
--architecture-version dual_tower_h15_v1 \
--window-sampling all_starts_random_tbptt8_v1 \
--batch-size 4 --mixed-tasks 1 --gpus 2 --anchor-fraction 0 --epochs 23
```

Prepare a new output data path; do not point at an existing fixed-window file or resume its checkpoint under the new contract. This document is not a remote launch authorization.

## Coverage measurement (synthetic, not remote drawer data)

For the checked 50-demo fixture with lengths160/175/190, seed4042, batch4 and two ranks:

- 7,985 legal starts per epoch, versus549 offset-zero starts.
- Epoch1: 1,677 real windows,1,684 slots,7 inactive slots (0.42%).
- 23 epoch lengths:421,418,420,416,416,412,410,412,415,418,415,417,415,413,412,419,410,409,418,415,421,417,415.
- Total9,554 updates; default stage1 ends after839 updates.
- Effective coverage is14.54x the offset-zero fixture. Reading five current/history images plus one World target entails47,910 frame references per stream per epoch before padding/cache reuse. Action and World streams remain separate; this is reference volume, not measured disk traffic or throughput.

The real remote data was not regenerated or inspected during this change. Do not apply these synthetic update counts to the running drawer experiment. Full-coverage epochs require substantially more work than old epochs.

## Verification

Main reviewed Gemini's actual data/trainer/evaluator diffs, corrected label-validation and test-fixture issues, and wrote the sampler/memory logic. Coverage tests merge both ranks and verify exact legal-start coverage, varying boundaries, chronological windows, offset isolation and deterministic sampler resume. Real LongTrajFramesDataset assembly is tested with synthetic frames. Tiny-model replay restore reproduces the next AdamW update exactly.

Final new-mode tests:11 passed (included in a12-test run with the isolated Qwen test). Full regression excluding the new evaluator test:1,143 passed,10 failed,8 skipped,6 subtests passed in102.24s. Nine failures match the recorded prior set; one additional Qwen final-norm test produced NaN in the full run and passed on isolated rerun. The new evaluator test passed separately. Do not report a wholly green full suite. Independent reviewer agent failed due to context exhaustion; no independent approval is claimed.

No GPU smoke test, throughput improvement, or closed-loop success improvement is established for this sampling change. No remote process, data, configuration or checkpoint was modified.
