# VA communication diagram brief

## Objective

Explain the actual communication semantics between visual tokens (V) and latent action tokens (A) in the current implementation, then separate confirmed code facts from experiment hypotheses.

## Audience and use

- Robotics/ML engineers reviewing whether bidirectional VA attention is truly bidirectional.
- Used for design review, experiment planning, and debugging the action-conditioning path.

## Required facts

1. One `VACouplingLayer` concatenates `Q_V` and `Q_A`, and attends to `[V, M, A, L]` keys/values with one shared row-wise softmax.
2. In `bidir_va`, visual queries can read action keys/values (`A -> V`) and action queries can read visual keys/values (`V -> A`).
3. The two outputs are computed in parallel from pre-update states. Fresh `V^i` and `A^i` communicate again only in layer `i+1`.
4. In `uni_a`, visual queries are restricted to current visual keys; action queries still read all streams.
5. The A stream starts from learned horizon queries plus projected proprioception and previous action. It is a latent action-condition stream, not the current noisy/candidate action trajectory.
6. `C_t = LN(A^4)` is computed once. Flow Matching repeatedly combines fixed `C_t` with `a^tau` and time, and Euler sampling does not feed the candidate trajectory back into VA/Vision.

## Hypotheses to show explicitly as hypotheses

- One-layer delay may weaken early A-to-V grounding.
- The semantic gap between latent A tokens and physical candidate actions may make A-to-V less useful.
- Reusing a fixed condition during solver steps may miss useful closed-loop refinement.
- Shared-softmax token imbalance may dilute the eight A tokens relative to V/M/L tokens.
- With only the velocity objective, the A-to-V path may be ignored without a direct utilization signal.

## Visual constraints

- One 1900 x 1080 page, white background, publication-style vector diagram.
- Blue = vision, violet = action, amber = language/memory/context, green = generated output, red = risk/hypothesis.
- Section A: exact per-layer communication matrix.
- Section B: exact meaning and lifecycle of A.
- Section C: five bounded diagnostic experiments.
- Keep code facts and hypotheses visually distinct.

## Sources

- `va_compound/model.py`
- `README.md`
- `tests/test_model.py`

