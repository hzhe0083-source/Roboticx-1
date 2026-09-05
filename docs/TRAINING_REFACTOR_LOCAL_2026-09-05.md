# Local training refactor — 2026-09-05

## Scope and safety

Only local source was changed. Remote training processes, configuration, code and checkpoints were not modified. No commit or push was performed. No research assets, data, logs, checkpoints or historical launch scripts were deleted.

Source-of-truth pre-refactor backup (includes existing user edits): `/tmp/ora0-train-refactor-baseline-7i_x009z`. This directory is temporary, not a durable archive.

## Entrypoints and ownership

- `train_metaworld.py`: dedicated MetaWorld CLI; `train.py` is a command-only compatibility wrapper.
- `train_libero.py`: LIBERO training/preflight; `libero_train.py` is a command-only compatibility wrapper.
- `prepare_libero.py`: standalone LIBERO data preparation. Training command's prepare subcommand remains compatible.
- `va_compound/data`: sampling, feature datasets, LIBERO contracts.
- `va_compound/vision/encoding.py`: shared DINO encoding/cache. `metric_runtime.py` preserves independent metric diagnostic consumers.
- `va_compound/training`: argument validation, model setup, batch/prefetch, gradients, rollout, checkpoint and MetaWorld engine.
- `configs/libero/run_schedules.json`: actually loaded 40-task and hard-two-task schedules.

Production package/evaluation/diagnostic imports no longer depend on `train` or `libero_train`; tested with AST import inspection.

## Removed capabilities

Removed generic trainer execution paths for paired semantic/intervention/fork, C2, direct regression, future/EVSM, plan/scene teacher, compiled-task/semantic-adapter/E2E, generic MT-VJ/metric/local-slot/servo integration, SAM, non-peer World and retired migration setup. Corresponding exclusive training tests were removed. Independent model/evaluator tests remain, including restored metric evaluator and servo-gradient tests incorrectly removed in an intermediate worker edit.

No preexisting independent algorithm file was deleted merely because it had a historical name. Model configuration/state-dict compatibility classes and independent RLT/PPO/metric trainers remain. The only complete non-test file removed was this refactor's unused duplicate `configs/libero/long_hard2_t8_h50p15.json`, replaced by the consumed run schedules file.

Some neutral historical argument/checkpoint fields remain to preserve exact-resume identity. The config and engine are still sizable and retain neutral compatibility fields; this is not a claim that every historical symbol has disappeared. Old launch scripts for retired routes remain research assets, not supported training commands.

Measured against the saved working-tree baseline: original two trainers 15,048 lines; new entrypoints plus all extracted training/data/vision modules approximately 9,463 lines (5,585 fewer, about 37%). Subsequent small validation/test edits do not materially change that estimate.

## Verification

1. Extracted shared primitives were compared with original function/class ASTs before intentional SAM removal.
2. Real CPU small-model VA and World complete objective outputs, individual losses, gradients including None ownership, and one AdamW update matched the baseline bit-for-bit.
3. Joint, action-only, head-only and separate-predictor-LR optimizer parameter names/order and freezing flags matched the baseline.
4. Active H50/P15 checkpoint complete payload keys, scalar values and tensors matched the baseline.
5. Existing tests cover uninterrupted versus save/resume updates and uncommitted prefetched sample replay. Added tests cover objective isolation, retired CLI rejection, no reverse entrypoint imports and World loss refresh between calls.
6. All three new entrypoints pass `--help`. Standalone preparation parser provides all arguments read by preparation.
7. `git diff --check` passes.

Latest complete CPU suite: **1,070 passed, 9 failed, 8 skipped; 6 subtests passed**, 71.90 seconds. A subsequently added World-loss-refresh test passed separately. Log: `/tmp/ora0-train-refactor-baseline-7i_x009z/final-tests.log`.

All nine remaining failures were reproduced against pre-refactor source:
- Two tests expect 49 metric tasks although the baseline declares 50.
- Five tests use fake Qwen models without the `layers` attribute required by the baseline wrapper.
- One peer evaluation trace fake lacks `config` required by the baseline evaluator.
- One MuJoCo/OSMesa import fails with PyOpenGL `NoneType.glGetError` in this environment.

Baseline evidence: `unrelated-baseline.log` and `osmesa-baseline.log` in the backup directory. Those tests remain failing rather than being removed or weakened.

## Review findings and limitations

Main review caught and repaired a real extraction error: World-only objective fell through into FM reduction after removing the direct branch. Restored mutual exclusion and tested it with real-model baseline parity and explicit branch-isolation tests. Main review also restored independent evaluator/model tests mistakenly removed by a worker.

Two broad independent review attempts failed due tool context exhaustion. A bounded review of the objective repair completed without a blocking finding; its request to test stale World loss was addressed. This is not a successful independent audit of the entire new engine.

No GPU/distributed end-to-end run, real dataset training update, full DINO/Qwen training step, or remote deployment was performed. The local refactor must not be deployed over the live training tree on the strength of CPU tests alone.

The known publisher-gradient ownership discrepancy, legal-P15 window coverage issue, AdaLN residual design and alternative dual-backbone fusion were deliberately not changed in this behavior-preserving refactor.
