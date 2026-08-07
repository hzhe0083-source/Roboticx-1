# Step 6 — Proposed Mechanisms and Novelty Comparison

- Timestamp: 2026-08-07
- Constraint: no new IL objective beyond the already declared flow/pair objective and the already-present future-latent predictor; frozen Qwen/V-JEPA; constant-size recurrent state; >=10 Hz.

## Recommended mechanism: Evidence-Verified Speculative Memory (EVSM)

### Mechanism

Treat action-conditioned Task-Memory writes as **speculation**, not evidence. At cycle `t`, the VA core may produce a fixed-size `T_spec,t` and a predicted next frozen visual latent `V_hat,t+1`, but persistent `T_commit` is unchanged. At cycle `t+1`, compare the real frozen V-JEPA latent with the stop-gradient prediction. A small deterministic confidence gate commits the preceding speculative state when reality agrees, or rolls back to `T_commit` and exposes an innovation token when reality disagrees. Only real visual/proprioceptive/executed-action inputs may update Evidence Memory.

This makes one causal rule explicit: **intent may propose state; only evidence may canonize it**. It uses the existing future-latent predictor rather than adding a verifier, label, or training loss.

### Minimal implementation surface

- `va_compound/model.py::VisualMemory`: add constant-size `task_commit`, `task_spec`, and `predicted_vision` fields; retain compatibility aliases during the experiment.
- `va_compound/model.py::encode_condition`: before the recurrent update, compute prediction innovation, commit/rollback the previous speculative state, update Evidence only from trusted inputs, and persist the current proposal as the next speculative state.
- `va_compound/model.py::TaskMemoryGate`: expose an explicit `merge(commit, spec, q)` operation. Use a fixed/calibrated monotone `q`; stop-gradient both the residual statistic and gate path so the flow objective cannot game its verifier.
- `va_compound/model.py::FutureLatentPredictor`: reuse the existing output; add no head or loss. Store its detached prediction in memory.
- `va_compound/eval_metaworld.py`: reset all three recurrent fields at episode start and log innovation/commit decisions for failure analysis.

### Kill criteria

- Pre-training: innovation cannot distinguish nominal from injected observation/action perturbations (`AUROC < 0.60`) — kill prediction-based commit and retain only the action-write firewall.
- By 10k steps: commit probability is below 0.05 or above 0.95 on more than 90% of cycles — kill or replace the calibration, not the whole model.
- At 40k: after the repaired baseline, less than `+5 pp` overall and less than `+8 pp` on long/hard tasks — kill EVSM as the paper mainline.

## Supporting mechanism A: Horizon-Provenance Source Routing (HPSR)

Build attention from named sources rather than positional concatenation. Each action-query horizon/head obtains gates over real Evidence, committed Task state, speculative Task state, current vision, cached language, proprioception, and innovation. Near-horizon queries favor real state; far-horizon queries may use committed task/language; innovation suppresses stale speculative state. Apply identical source accounting in shared and sequential layers.

- Code: factor named source slices in `VACouplingLayer.forward`, `_attend`, and `forward_sequential`; fix the existing K/V mask-order defect first.
- Novelty posture: generic learned source gates overlap S²-VLA. The defensible delta is horizon-conditioned routing over sources with explicit write provenance, inside EVSM; it is not a standalone headline.
- Kill: less than `+2 pp` over EVSM, source selection collapses to one source on >90% of queries, or p95 control latency exceeds 100 ms.

## Supporting mechanism B: Contract-Residual Vector Field (CRVF)

Compile a low-rank read-only contract code from the cached Qwen tokens once. Every control cycle, current Evidence/Task state queries those cached tokens and forms a bilinear residual `r = W_o[(W_s s_t) * (W_l l_t)]`. Inject `r` directly into action queries and the flow head's AdaLN blocks, so language remains a non-drifting constraint rather than surviving only through Task Memory.

- Code: extend `LanguageCache`, `build_language_cache`, `encode_condition`, and `FlowMatchingHead.forward`.
- Novelty posture: cached cross-attention, FiLM, and multiplicative conditioning are established primitives; the useful paper role is an efficient language-contract ablation, not the core novelty.
- Kill: less than `+2 pp` overall, no improvement in late-episode correct-vs-wrong-instruction separation, or p95 latency exceeds 100 ms.

## Supporting mechanism C: Dual-Aperture Spatial Attention Modulation (DA-SAM)

In only the action-conditioned visual reorganization pass, derive a coarse end-effector/action-proposal center and use two apertures: a local head for contact precision and a guaranteed global head for search/recovery. Prediction innovation widens the aperture after mismatch and narrows it near stable contact. This is worthwhile only after converting the current flat adaptive pooling to a genuine 2-D spatial grid.

- Code: `backbones.py::pool_spatial_tokens`, MetaWorld feature preparation/evaluation pooling, and `VACouplingLayer.forward_sequential/_attend`.
- Novelty posture: generic action-conditioned spatial modulation is strongly overlapped by AVA-VLA and AtVLA. The dual-aperture/innovation coupling is a modest composition, not a safe headline.
- Kill: contact-stage success improves by less than `+5 pp`, overall by less than `+2 pp`, or global-search/recovery tasks regress by more than `2 pp`.

## Closest-work comparison

| Prior work | Problem | Core mechanism | Key insight | Domain | Overlap level |
|---|---|---|---|---|---|
| VLA-Corrector | Match | External residual-dynamics monitor; truncate suffix and guide a replan | Prediction error detects stale execution | Match | Level 3 — medium |
| CheckVLA | Match | External world model/risk head; retain/rewrite chunk suffix | A chunk is a testable prediction | Match | Level 3 — medium |
| FFDC | Match | Separately supervised causal verifier; continue/replan | Future–reality consistency controls commitment | Match | Level 3 — medium |
| μVLA | Match | Recurrent memory with an action-to-memory attention prohibition | Prevent action-copy shortcuts in persistent state | Match | Level 3 — medium |
| S²-VLA | Match | GRU belief plus learned visual/intent/action source gates | Task phase should route modalities | Match | Level 3 — medium |
| AVA-VLA | Match | Previous action-state recurrence plus language FiLM/spatial weights | History selects current visual relevance | Match | Level 3 — medium |
| AtVLA | Partial | Uncertainty-triggered crop, high-resolution re-encode, rerun | Spend perception compute only when spatially uncertain | Match | Level 2–3 |

## Exact novelty delta

Unlike VLA-Corrector, CheckVLA, and FFDC, which use a separately trained verifier to change execution of an already committed action suffix, and unlike μVLA, which blocks action-to-memory writes, EVSM permits action-conditioned Task-Memory proposals but keeps them transactional until the next real observation validates or rolls them back inside the policy. The primitive ingredients are not individually new; the defensible contribution is the causal write protocol and its no-new-verifier-loss realization in a dual-memory VLA.

## Expected benefit after baseline repair

- EVSM target: `+5–10 pp` overall MetaWorld success and `+8–15 pp` on long/hard or perturbation-heavy subsets, primarily by preventing false phase advancement and enabling recovery after model–world mismatch.
- CRVF target: `+1–3 pp`, mainly late in episodes where a drifting Task Memory otherwise weakens the frozen language constraint.
- HPSR/DA-SAM should be treated as diagnostic add-ons; neither has a credible independent SOTA-size gain forecast under 40k steps.

These are experiment targets, not claims: the present 16–30% baseline is invalid for causal comparison because of the measured data leak and masking defects.
