# H15/P15 peer action trace diagnostic

- Source: `/home/ryan/Documents/robot/ORA0/exp_runs/2026-08-23-h15-p15-peer-action-trace/trace.json`
- SHA-256: `48db7670c8e87adccdab5d1402e267ddb2c022489119c8d4340c1aff6881d12a`
- Evidence: `reproduced` for trace distances/counts; `inferred` for geometry relation.

## Conclusion

**Mismatch exists but clip/geometry relation weak.** Readout-vs-flow disagreement persists after clipping, but n=3 does not show a clean seed-level causal relation to geometric stagnation/regression. Action-identity mismatch remains a candidate, not confirmed as the major cause. Representation blindness is not evaluated and is not confirmed.

## Key metrics

- Primary unit: seed, n=3. Executed tokens: seed 0=103 (success/early stop), seed 1=500, seed 2=500 (horizon truncation).
- Pooled H15 xyz L2 pre/post clip: 0.331555/0.31233; gripper absolute: 0.091983/0.091983. Pooled token values are descriptive, not independent inference.
- Tokens 0-5 xyz L2 pre/post: 0.370318/0.336027; xyz saturation disagreement: 210/448 (46.875%).
- Token states: aligned=21, hooked=528, both=1 of 1103. Continuous-down decisions: negative=45/75, fully -1 saturated=3/75.
- Large mismatch cutoff is post hoc Q75=0.382286; co-occurs with xy or |z_gap| stagnation/regression in 62/68 large-mismatch decisions. Per-seed metrics are in `summary.json`.

## Red flags

- n=3, one successful seed; no significance or mechanism claim.
- Repeated tokens/decisions are not independent; seed is primary.
- Exact pre-decision geometry is absent; after-token0 is an inferred proxy.
- Post hoc mismatch cutoff; no data×algorithm crossing.
