# Unified H15 local implementation — 2026-09-05

## Architecture

`dual_tower_h15_v1` constructs one H15 LayerwiseActionExpert with three internal blocks, each directly reading one of the final three VA conditions. These are gradient entry points, not gradient endpoints: the connected, unfrozen upstream action network remains trainable. No protected step-6 prefix or H50 extension expert exists in this version. Prediction, execution and World horizons are 15. FM integration remains external; all 15 actions use uniform loss weight 1.

Native data contract: `libero_joint_episode_t8_h15_state_delta_v11`. Stored action labels have shape N/T8/H15/A7, not H50 cropped by the trainer. Episode memory, T8 truncated backpropagation, future joint/gripper delta supervision and separate World parameter ownership remain. Shared DINO receives action and World gradients through the existing merge mechanism. Legacy and nested-expert versions retain their own loading paths.

## Validation

- Focused H15, episode memory, state supervision, distributed dummy-rank and single-task regressions: 31 passed (19.33 seconds).
- Full CPU suite: 1134 passed, 9 failed, 8 skipped, 6 subtests passed (94.47 seconds).
- The nine failures match the previously recorded set: two task-count expectations, five Qwen fake-model layer fixtures, one peer-trace fake config and one local OSMesa/OpenGL import failure.
- No real GPU smoke test, throughput measurement or closed-loop success evaluation has been performed for the new H15 version. CPU tests do not establish training success or GPU efficiency.

## Remote boundary and next-launch requirement

The existing isolated remote drawer 23-epoch experiment remains unchanged and still uses `dual_tower_expert_v1`. This change does not deploy H15, restart training, or modify any remote process, configuration, checkpoint or data.

User reports poor GPU core utilization in the existing run. For the next separately authorized new training launch, dual-GPU compute utilization and useful throughput are acceptance criteria, not merely memory occupancy. Measure per-device utilization over time, valid decisions/second, step latency, data waiting, row-serial episode execution and synchronization overhead before attributing the bottleneck. Tune batching and prefetch based on measurements without breaking episode memory, valid-mask, gradient ownership or sampling semantics. Do not silently change effective batch/training schedule solely to inflate utilization, and do not claim sustained saturation without measured evidence.
