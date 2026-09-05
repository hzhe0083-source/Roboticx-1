# Joint LIBERO: episode memory and future joint-state supervision

## Scope

The joint `dual_tower_expert_v1` experiment uses data contract
`libero_joint_episode_t8_p15_state_delta_v10`. Legacy training and its evaluation
reset intervals remain separate. This change does not deploy code, rebuild remote
data, start training, or consume failure trajectories.

## State supervision

LIBERO proprio is seven joint coordinates plus two gripper joint coordinates,
not an end-effector Euler pose. For a decision at index d, the target is
`2 * (raw_state[d+15] - raw_state[d]) / scale`, with scale equal to the training
state 99th minus 1st percentiles (a near-zero range uses 1). Unlike model-input
normalization, this target is not clipped. Bounded joint coordinates are not
wrapped as Euler angles. Labels and future images never enter policy conditions.

A small normalized MLP reads the pooled tokens of the same predicted World map
published to VA, with the map gradient enabled on the auxiliary path. It does not
receive a direct state/action bypass. Smooth L1 is averaged over state dimensions
and World stages, then valid decisions. Its configurable weight defaults to 1;
this is an initial experimental setting, not an empirically tuned optimum. The
head belongs to World-private optimizer parameters. Existing visual World losses
remain. A lower auxiliary loss alone does not prove removal of visual shortcuts.

## Episode and TBPTT semantics

- Decisions start at zero and advance by P15 while a complete real P15 label and
  future state are available. Consecutive nonoverlapping windows contain up to T8
  decisions. The final short window is retained, but storage padding is masked.
- Memory persists across windows of the same demonstration. All retained visual
  and World memory tensors are detached at a window boundary, preserving values
  but not an earlier computation graph. They were computed before the preceding
  optimizer update, as in truncated recurrent training.
- Action and logged-World objectives have separate memory banks. Stream identity,
  episode identity and expected next decision index must agree. Banks commit only
  after an optimizer update; episode ends remove the corresponding entry.
- The sampler groups tasks and episodes into distributed cohorts. Every real
  episode window occurs once per epoch. Exhausted slots are inactive placeholders:
  they do not update memory or contribute labels. Episodes are shuffled within
  equal-window-count buckets so epoch length is stable for strict resume.
- Per-row recurrent execution handles unequal lengths without inventing memory
  for new episodes. The frontend still encodes storage batches, including padded
  positions; avoiding that extra encoder work is not implemented here.
- This is no longer the old dense all-starts dataset. Epoch counts and update
  budgets differ substantially. A 50-epoch run is not an equal-compute comparison
  with the previous 50-epoch schedule. Anchor replay is disabled for joint mode.
- Complete P15 execution leaves a possible final residual of fewer than 15
  actions outside the executed-prefix labels. This is explicit, not a claim that
  all demonstration terminal actions are now covered by the fixed P15 cadence.

## Evaluation and recovery

Joint evaluation uses `memory_reset_every=0`: only a new environment episode
clears memory, not the eighth decision. The setting is validated against the
checkpoint. Each trial starts empty even if its task instruction is unchanged.

Strict joint checkpoints include sampler cursors and each rank's committed
Action/World memory as tensor-only records. They require matching world size,
content identity, supervision contract and loss weight. Old joint data and old
joint checkpoints must not be used as strict continuations of this experiment.

## Verification

Tests cover label indexing, invalid data rejection, episode continuity, short
windows, cross-window detach, independent stream reset, exact next optimizer
update after save/restore, target non-leakage, state-head/World gradients and a
two-process CPU/Gloo inactive-rank gradient reduction.

Final full CPU regression: **1119 passed, 9 failed, 8 skipped, 6 subtests passed**
in 85.26 seconds. The nine failures are the previously recorded task-count,
Qwen mock-model, peer trace mock and OpenGL environment failures; this is not an
all-green suite. Log: `/tmp/ora0-episode-state-final-tests.log`. After final
checkpoint assertions and removal of an unused diagnostic field, the targeted
suite passed another 17 tests. Independent reviewer execution failed from context
exhaustion and is not counted as approval; main-model review covered actual diffs.

Full-size joint backbone GPU memory, throughput, full trainer end-to-end resume
and simulator success remain unmeasured for this version. CPU next-update parity
uses the real tiny policy, optimizer and memory bank, not the full pretrained
trainer. Existing action-ranking terms are reduced within each real episode
window before valid-decision-weighted aggregation, unlike a single flattened
all-window donor reduction.
