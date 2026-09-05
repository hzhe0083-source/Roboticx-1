# Step 7 — Verdict, Delta Statement, and Experimental Decision

- Timestamp: 2026-08-07

## Novelty verdict

**Level 3 — Medium overlap.** Prediction-error monitoring, recurrent memory, source gating, cached language conditioning, action-conditioned spatial attention, and adaptive chunking all have close recent precedents. A broad claim such as “use future prediction error to improve closed-loop VLA control” is not novel enough. The narrower EVSM claim remains defensible because its intervention target is persistent Task-Memory write semantics rather than action-suffix scheduling, and it needs no separately supervised verifier.

## Paper-safe contribution statement

> We introduce Evidence-Verified Speculative Memory, a transactional recurrent-state protocol in which action-conditioned task updates remain provisional and can become persistent only after agreement with next-cycle frozen visual evidence; disagreement rolls the policy back to its last evidence-committed state, without adding a verifier loss or rerunning the language model.

Avoid claiming that source gates, SAM, latent prediction, recurrent memory, language caching, or adaptive horizons are individually novel.

## Decision on the three internal ideas

1. **SMC-Attn:** worth fixing and using as infrastructure/ablation, not as the paper contribution. The current implementation is source-cardinality normalization rather than a learned source-wise gate, its source ordering is defective, and its sequential path bypasses it. A learned gate alone is S²-VLA-like.
2. **SAM:** conditional maybe. Generic spatial modulation is old and the current sequential A→V/T→A path already supplies action-conditioned reorganization. Try only the dual-aperture variant if a failure taxonomy shows contact/localization failures are at least 30% of failures.
3. **LVK:** do not prioritize. A stochastic latent variable normally needs a prior/KL or anti-collapse objective, violating the loss constraint; a deterministic latent kernel duplicates the existing Task Memory and recent recurrent-memory VLAs. A “language veto kernel” is better framed as the small CRVF conditioning ablation.

Ranking: repaired SMC infrastructure > targeted DA-SAM >>> LVK. The main mechanism should be EVSM, not any of these labels.

## Required baseline repair

1. Regenerate `previous_action` from the actually executed primitive (`decision-1`), not prior `chunk[-1]`; add an assertion against target `chunk[1]` equality.
2. Build K/V and masks from named source slices with one source order; include explicit state on every sample; apply the same logic in sequential attention.
3. Persist `task_hat` on the first recurrent cycle; set the sequential-layer count/config to the intended value; swap language hidden states and masks together in ablations.
4. Report true closed-loop environment success and end-to-end p50/p95 latency; label the current MAE-threshold statistic as a proxy, not “open-loop success.”

## Five-stage plan

1. **P0 / 0 steps:** repair and unit-test data causality, source masks, first Task write, config, ablation masks, and latency measurement; stop all mechanism comparisons if any causality assertion fails.
2. **P1 / 40k:** retrain the exact repaired baseline with the declared loss/config; kill the old baseline as evidence if three seeds are not reproducible within 5 pp.
3. **P2 / <=40k:** add EVSM only; at 10k kill a collapsed commit gate, and at 40k kill if gain is `<5 pp` overall and `<8 pp` on long/hard tasks.
4. **P3 / <=40k:** add exactly one failure-driven module—CRVF for late language drift or DA-SAM for contact failures; kill at `<2 pp` overall (DA-SAM also requires `+5 pp` contact-stage) or p95 `>100 ms`.
5. **P4 / <=40k:** apply standard PPO only to the winning action head with frozen features; kill if gain is `<3 pp`, language-ablation separation shrinks materially, or repaired IL success regresses.

## Final recommendation

Proceed with P0 immediately. Do not train SMC/SAM/LVK variants on the current feature file. If P1 confirms a real closed-loop gap, EVSM is the only proposed mechanism here with a sufficiently crisp causal story, a minimal implementation, and a defensible novelty delta for the main CCF-A experiment.
