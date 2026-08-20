# Step 5 — Full-Paper Deep Dive and Repository Audit

- Timestamp: 2026-08-07T06:04:02+08:00
- Retrieval: direct arXiv PDFs were downloaded and extracted with the `scoop-check` bundled script into `papers/*.txt`. The method, experiments, conclusion, and limitation passages were inspected; this is a targeted mechanism audit, not an end-to-end paper summary.

## Candidate 1 — VLA-Corrector

- **Problem framing (verified):** Action-chunked VLAs receive fresh observations during an open-loop horizon but do not use them until the next base-policy call; stale actions compound drift.
- **Core mechanism (verified):** After a VLA is trained, a separate latent dynamics corrector predicts the frozen visual-encoder residual induced by an executed action. Online cosine mismatch between expected and actual residuals feeds a median/MAD persistence trigger. A trigger discards the suffix and applies Online Gradient Guidance to the next flow replan.
- **Key insight (verified):** Residual latent dynamics suppress static scene content and expose whether local execution remains on track.
- **Application domain (verified):** MetaWorld, LIBERO, and an AgileX PiPER robot; π0.5, SmolVLA, and X-VLA backbones.
- **Venue:** arXiv preprint, 2026.
- **Assumptions & scope:** Needs a separately trained benchmark-specific corrector and its magnitude/cosine loss; fresh camera encoding remains available during the action horizon; recovery still depends on the frozen base policy and Online Gradient Guidance.
- **Closest-passage evidence:** Sections 3.1–3.4 define `L_corr`, expected/actual latent residual comparison, event-triggered truncation, and OGG. The conclusion explicitly describes monitoring latent visual dynamics and truncating stale actions.
- **Refined overlap:** problem = match; core = partial/different (external monitor and action-suffix intervention, not internal Task-Memory transaction); insight = partial (prediction error, but not write provenance); domain = match.

## Candidate 2 — CheckVLA

- **Problem framing (verified):** A committed action chunk predicts how observations should evolve, but policy confidence at dispatch cannot react to later deviations.
- **Core mechanism (verified):** A separately trained action-conditioned world model rolls predictions from each observation anchor. A causal risk head, calibrated on nominal success trajectories, triggers a latency-aware flow suffix rewrite. Risk exceedance determines old-suffix retention; an event-driven keyframe bank stores real-observation progress.
- **Key insight (verified):** An action chunk is both a control command and a testable prediction.
- **Application domain (verified):** RoboCasa365 long-horizon mobile manipulation.
- **Venue:** arXiv preprint, 2026.
- **Assumptions & scope:** Adds world-model Huber loss, risk-head classification, perturbation rollouts, calibration data, and an 88.4M monitor; evidence is simulator-only and thresholds require recalibration under distribution shift.
- **Closest-passage evidence:** Method Eqs. (1)–(10) cover rolling prediction, calibrated risk, hard-prefix suffix rewrite, risk-dependent retention, and the keyframe bank. The limitation text says the guarantee covers only unnecessary first intervention on exchangeable nominal successes.
- **Refined overlap:** problem = match; core = partial/different (external verifier, suffix repair, external keyframe memory); insight = partial; domain = match.

## Candidate 3 — When to Trust Imagination / FFDC

- **Problem framing (verified):** A world-action model predicts future video and action but executes a fixed number of actions without checking whether imagination matches reality.
- **Core mechanism (verified):** A separate causal-attention verifier consumes cached predicted past/future visual tokens, predicted future actions, the instruction, and each latest observation. A binary score decides continue versus replan. Training uses a verifier classification loss and corrupted/failure segments in addition to the WAM's action/video flow losses.
- **Key insight (verified):** Future–reality consistency is a better commitment signal than a globally fixed chunk length.
- **Application domain (verified):** RoboTwin and real-world manipulation with a Motus world-action model.
- **Venue:** arXiv preprint, 2026.
- **Assumptions & scope:** Requires a WAM that predicts future video, extra verifier labels/loss, and a fixed 0.5 threshold; it is execution scheduling, not recurrent-state integrity.
- **Closest-passage evidence:** Sections 3.1–3.3 define `L_WAM=L_act+L_vid`, the FFDC input `[L, predicted past, real now, predicted future, future action, CLS]`, and binary `L_ver`.
- **Refined overlap:** problem = match; core = partial/different; insight = partial; domain = match.

## Candidate 4 — S²-VLA

- **Problem framing (verified):** Fixed multimodal fusion does not adapt across approach, grasp, transport, and release, allowing early errors to propagate.
- **Core mechanism (verified):** A lightweight GRU belief state summarizes past action sequences and proprioception. Per-layer softmax gates fuse three parallel branches: low-level visual cross-attention, high-level intent cross-attention, and action-sequence self-attention. All parts are trained end-to-end with action MSE only.
- **Key insight (verified):** The relevant information source changes with task phase, and a recurrent belief should route attention accordingly.
- **Application domain (verified):** LIBERO and SimplerEnv long-horizon manipulation.
- **Venue:** IJCAI 2026 (verified from arXiv metadata).
- **Assumptions & scope:** Qwen3-VL and the fusion system are trained jointly; the paper does not impose evidence/task write permissions or distinguish proposed intent from realized state. The conclusion names extension to flow/diffusion as future work.
- **Closest-passage evidence:** Section 3.2 Eqs. (2), (7), and (8) define GRU belief and belief-guided three-branch gating; Eq. (10) is action-only training.
- **Refined overlap:** problem = match; core = partial (source gates and belief recurrence); insight = partial (phase adaptation rather than commit integrity); domain = match.

## Candidate 5 — AVA-VLA

- **Problem framing (verified):** History-agnostic visual processing is poor under partial observability.
- **Core mechanism (verified):** The preceding action-related hidden state is projected into a recurrent belief and initializes current action placeholders. Language-conditioned FiLM transforms visual features; cross-attention between current visual tokens and the recurrent state produces soft token weights applied to attention in all backbone layers.
- **Key insight (verified):** Historical context should decide which visual regions are relevant now.
- **Application domain (verified):** LIBERO, CALVIN, and Mobile ALOHA experiments.
- **Venue:** arXiv preprint, 2025.
- **Assumptions & scope:** Uses TBPTT and adds an L2 penalty on attention weights, violating the present no-new-loss rule; it modifies visual attention throughout a large VLA backbone.
- **Closest-passage evidence:** Sections 3.2–3.4, especially Eqs. (5)–(13), derive recurrent state, FiLM, visual weights, and the auxiliary penalty.
- **Refined overlap:** problem = match; core = partial (spatial modulation/recurrent belief); insight = different; domain = match.

## Candidate 6 — AtVLA

- **Problem framing (verified):** VLA visual encoders lose precise spatial/contact detail and exhibit attention artifacts.
- **Core mechanism (verified):** Register tokens absorb global embodied spatial information. Multiple flow samples estimate action uncertainty; uncertain predictions use action-to-image attention rollout to crop a region, re-encode it at high resolution, append tokens to a cached prefix, and rerun the action expert.
- **Key insight (verified):** Use action uncertainty to decide whether and where to reacquire spatial detail.
- **Application domain (verified):** LIBERO, SimplerEnv, and a single-view real-world benchmark.
- **Venue:** arXiv preprint, 2026.
- **Assumptions & scope:** Final training adds crop-coordinate and attention-grounding losses; uncertain steps pay another visual encoding/action generation; representative total compute is 1.4–1.6× the base policy.
- **Closest-passage evidence:** Method Eqs. (4)–(11) define multi-sample uncertainty, attention rollout/crop, re-encoding, and the final objective `L_pi0 + lambda_cp L_cp + lambda_ag L_ag`.
- **Refined overlap:** problem = different/partial; core = partial only for action-conditioned spatial reorganization; insight = different; domain = match.

## Candidate 7 — μVLA

- **Problem framing (verified):** Partial observability requires information no longer present in the current image.
- **Core mechanism (verified):** A bank of recurrent memory tokens is carried between environment steps inside backbone self-attention. Temporally ordered streams and TBPTT train it with action L1 only. Crucially, an attention-mask guard prevents memory/context tokens from reading action tokens, avoiding a self-referential action-copy shortcut. Deployment re-queries every environment step.
- **Key insight (verified):** Minimal recurrence helps, but action-to-memory writes can collapse into copying the current predicted chunk rather than storing environmental information.
- **Application domain (verified):** MIKASA-Robo partial-observability tasks and LIBERO.
- **Venue:** arXiv preprint, 2026.
- **Assumptions & scope:** Receding-horizon inference is materially more expensive; short TBPTT does not solve arbitrary long memory; transfer across unseen memory semantics remains weak.
- **Closest-passage evidence:** Section 4 defines the action-mask guard, episodic dataloader, TBPTT/EMA, and receding horizon; Appendix A limits the claim to compact online state estimation.
- **Refined overlap:** problem = match; core = partial (protected recurrent state); insight = partial (write provenance is central); domain = match. The exact delta is that μVLA forbids action writes, whereas the proposal allows them only into a provisional state and validates them before persistent commit.

## Repository-grounded audit before architecture experiments

1. **Exact future-action leak in `previous_action`.** `scripts/migrations/prepare_prev_fix_v3.py` assigns `actions[:, :-1, -1]`. With MetaWorld stride 6 and horizon 8, this is future primitive `d+7`, exactly the next decision target's `chunk[1]`; measured max absolute difference is 0. The deployment input is the actually executed `d-1`. Every sample-window `t=0` is also forcibly zero even for mid-episode windows.
2. **Shared-attention source order is inconsistent.** Actual K/V order is `[V,M,T,A,L,S]`, while key masking and SMC accounting assume non-language sources precede language. With padded language, the state token receives the final language-mask bit; in the MetaWorld data that bit is false for about 98% of samples, so the explicit state K/V is usually masked.
3. **First Task-Memory update is discarded.** On the first recurrent cycle, layers produce `task_hat`, but `encode_condition` writes it back only when `prev_task is not None`; the returned first-cycle task remains the pure language initialization.
4. **Implemented architecture differs from the prompt.** `sequential_coupling=2` makes layers 2/4/6/8 sequential (four layers, not two); the checkpoint uses `attention_variant=flat`, so SMC is off. The implemented SMC is `-log N_s` measure correction, not a learned source gate, and sequential `_attend` bypasses both SMC and QK normalization.
5. **Actual objective differs from the stated contract.** The MetaWorld VA2 train command uses `--single-task` (pair loss zero) and `--future-predict-weight 0.1`, so the checkpoint used `L_FM + 0.1 L_future`, not `L_FM + L_pair`.
6. **Language-ablation masks are not swapped with hidden states.** Wrong-instruction evaluation changes hidden tensors while retaining the original task mask, so different prompt lengths can truncate valid tokens or unmask padding; the reported +268%/+159% should be rerun with paired `(hidden, mask)` swaps.
7. **Latency benchmark is incomplete.** `scripts/benchmarks/bench_inference.py` builds VA conditions before timing and measures only repeated `sample_actions`; it excludes VA-core encoding and V-JEPA. The >=10 Hz claim needs end-to-end p50/p95 measurement under the actual visual update path.

These defects do not merely add noise: items 1–3 directly create the symptom “excellent teacher-forced/open-loop metrics, poor autonomous rollouts.” A new mechanism trained before fixing them would not yield interpretable evidence.
