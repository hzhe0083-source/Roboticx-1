# Step 6 — Proposed experiments

## Main: causal anatomy of bounded-state agents

- Freeze the existing VLA; attach a tiny MLP/GRU Sentry over recurrent state, state delta, and action statistics.
- At a detected event, compare `continue` and `replan` from an identical saved simulator state with common random numbers.
- Intervene independently on memory (`reset/freeze/donor swap`), verifier (`force/delay/suppress`), instruction (`blank/feasible swap`), and plan (`original/replanned/oracle upper bound`).
- Cross training data and algorithm: successful demonstrations versus success+failure/recovery data, each with random/periodic versus learned triggers.
- Measure paired replan treatment effect, recovery success, false-replan harm, trigger delay, oracle gaps, planner-call rate, end-to-end latency percentiles, and total persistent-state bytes.

## Backup: Memory Is Not Thinking

Factor temporal recurrence and within-step recurrent computation in a 2×2 design: carry/reset temporal state × one/four tied reasoning iterations. Compare with equal-parameter and equal-FLOP untied controls, then test donor-history state swaps on genuinely partially observable tasks.
