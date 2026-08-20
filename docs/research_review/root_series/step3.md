# Step 3 — Abstract-Level Triage

- Timestamp: 2026-08-07T06:00:47+08:00
- Reference claim: execution-aligned speculative Task Memory whose action-conditioned write is committed or rolled back only after next-cycle frozen-latent verification; cached language participates in the commit/correction gate; no new IL loss.
- Score: number of plausibly matching axes among problem framing, core mechanism, key insight, and application domain.

## Structured papers

### 1
- **Title:** Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy
- **Date:** 2026-03
- **Problem framing:** Fixed diffusion-policy chunks react too slowly to changing dynamics.
- **Core mechanism:** Self-supervised dynamics features and a lightweight cross-attention correction encoder modify a base chunk online.
- **Key insight:** Fresh dynamics should correct a chunk before all planned actions execute.
- **Application domain:** Dynamic robot manipulation.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://arxiv.org/abs/2603.01953

### 2
- **Title:** RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning
- **Date:** 2024-09
- **Problem framing:** Imitation policies lack recovery behavior outside successful demonstration trajectories.
- **Core mechanism:** Language-guided recovery policy/data formulation.
- **Key insight:** Failure semantics and recovery examples improve return to the nominal trajectory.
- **Application domain:** Robot imitation learning.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://arxiv.org/abs/2409.14674

### 3
- **Title:** Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection
- **Date:** 2026
- **Problem framing:** Long-horizon manipulation requires planning and reflection after failure.
- **Core mechanism:** Adaptive high-level planning/reflection over skills.
- **Key insight:** Reconsidering plans from feedback improves multi-stage execution.
- **Application domain:** Long-horizon robot manipulation.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://www.semanticscholar.org/paper/da853cd33f24b4c02a8bf45b17e6fb544b0411ec

### 4
- **Title:** Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA
- **Date:** 2026
- **Problem framing:** Open-loop VLA plans need online verification.
- **Core mechanism:** Speculative planning with closed-loop verification.
- **Key insight:** Treat a proposed rollout as provisional until observations validate it.
- **Application domain:** VLA robot control.
- **Overlap score:** 3/4 (problem, insight, domain; memory transaction differs).
- **Source:** https://www.semanticscholar.org/paper/af49770d14c0e496f2555865969d34eda4fd8d8b

### 5
- **Title:** VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon
- **Date:** 2026-07
- **Problem framing:** Fixed action horizons create a closed-loop blind spot and compounding error.
- **Core mechanism:** A separately trained latent dynamics corrector compares predicted and actual visual residuals, truncates stale actions, then applies online gradient guidance.
- **Key insight:** Prediction–observation mismatch is an execution-time correction signal.
- **Application domain:** Action-chunked VLA manipulation on MetaWorld, LIBERO, and real hardware.
- **Overlap score:** 3/4 (problem, insight, domain; external monitor/replanning differs from Task-Memory commit).
- **Source:** https://arxiv.org/abs/2607.01804

### 6
- **Title:** Dynamic Execution Horizon Prediction for Chunk-based Robot Policies
- **Date:** 2026
- **Problem framing:** A fixed number of executed chunk actions is inefficient or brittle.
- **Core mechanism:** Predict an execution horizon from current policy/context signals.
- **Key insight:** Replanning cadence should depend on local state reliability.
- **Application domain:** Chunk-based robot control.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://www.semanticscholar.org/paper/e02afb9d9bf048be0c4cfdc87a534aade779e07a

### 7
- **Title:** RePLan: Robotic Replanning with Perception and Language Models
- **Date:** 2024-01
- **Problem framing:** Robots need to recover plans when the perceived world disagrees with plan assumptions.
- **Core mechanism:** Perception/language-model planning loop.
- **Key insight:** Replanning closes the loop between symbolic intent and observed state.
- **Application domain:** Robot task planning.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://arxiv.org/abs/2401.04157

### 8
- **Title:** Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection
- **Date:** 2026-03
- **Problem framing:** Detect execution deviation without a large separate monitor.
- **Core mechanism:** Probe/use internal VLA attention heads as deviation signals.
- **Key insight:** A trained policy may already encode useful failure evidence.
- **Application domain:** VLA execution monitoring.
- **Overlap score:** 2/4 (problem, domain; internal-signal idea is partial).
- **Source:** https://arxiv.org/abs/2603.13782

### 9
- **Title:** ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries
- **Date:** 2026-03
- **Problem framing:** Markov VLA policies forget short- and long-term context.
- **Core mechanism:** Frame-level and chunk-level recurrent query banks plus past-observation prediction.
- **Key insight:** Separate recurrence timescales improve memory-dependent behavior.
- **Application domain:** Simulated and real robot manipulation.
- **Overlap score:** 3/4 (problem, recurrent-memory mechanism, domain; no speculative commit semantics).
- **Source:** https://arxiv.org/abs/2603.12942

### 10
- **Title:** Towards Long-Horizon Vision-Language-Action System: Reasoning, Acting and Memory
- **Date:** 2025
- **Problem framing:** Long-horizon VLA needs explicit coordination of reasoning, action, and memory.
- **Core mechanism:** System-level integration of these components.
- **Key insight:** A reactive policy alone does not preserve task state over long episodes.
- **Application domain:** Long-horizon embodied manipulation.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://www.semanticscholar.org/paper/993b5ad471a0997dcd9e996248df8e5f845f397f

### 11
- **Title:** Explicit Language Memory for Long-Horizon Planning in Vision-Language-Action Models
- **Date:** 2026
- **Problem framing:** Language/task intent can be lost during long-horizon VLA execution.
- **Core mechanism:** An explicit language-memory channel supports planning.
- **Key insight:** Persistent intent should remain accessible after early observations.
- **Application domain:** Long-horizon VLA planning.
- **Overlap score:** 2/4 (cached contract idea, domain).
- **Source:** https://www.semanticscholar.org/paper/92599f272ba075cc6b6e5b9644a4b234bd1b8629

### 12
- **Title:** ReplanVLM: Replanning Robotic Tasks With Visual Language Models
- **Date:** 2024
- **Problem framing:** Visual feedback can invalidate robot plans.
- **Core mechanism:** A VLM diagnoses state and regenerates a plan.
- **Key insight:** Periodic visual replanning improves task robustness.
- **Application domain:** Robot planning/manipulation.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://doi.org/10.1109/LRA.2024.3471457

### 13
- **Title:** π0: A Vision-Language-Action Flow Model for General Robot Control
- **Date:** 2025
- **Problem framing:** General robot action generation from vision and language.
- **Core mechanism:** VLM-conditioned flow-matching action expert producing chunks.
- **Key insight:** Generative continuous-action modeling scales across embodiments.
- **Application domain:** General robot control.
- **Overlap score:** 1/4 (domain; it is also the flow-head ancestor).
- **Source:** https://doi.org/10.15607/RSS.2025.XXI.010

### 14
- **Title:** Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
- **Date:** 2025
- **Problem framing:** Adapt large VLA models efficiently while preserving success.
- **Core mechanism:** OpenVLA-OFT-style parallel action decoding and fine-tuning recipe.
- **Key insight:** Fine-tuning and decoding design determine latency/performance.
- **Application domain:** VLA manipulation.
- **Overlap score:** 1/4 (domain).
- **Source:** https://doi.org/10.15607/RSS.2025.XXI.017

### 15
- **Title:** CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations
- **Date:** 2025-05
- **Problem framing:** Learn continuous latent actions from unlabeled robot demonstrations.
- **Core mechanism:** Latent action representation learning.
- **Key insight:** Useful action abstractions can be learned without action labels.
- **Application domain:** Robot learning.
- **Overlap score:** 1/4 (domain only; “latent” is not memory).
- **Source:** https://arxiv.org/abs/2505.04999

### 16
- **Title:** DiLA: Disentangled Latent Action World Models
- **Date:** 2026-05
- **Problem framing:** Disentangle action-relevant latent dynamics in world models.
- **Core mechanism:** Latent action world-model factorization.
- **Key insight:** Separating latent factors improves controllable dynamics modeling.
- **Application domain:** Robot/world-model learning.
- **Overlap score:** 1/4 (domain; latent dynamics are tangential).
- **Source:** https://arxiv.org/abs/2605.15725

### 17
- **Title:** Light-WAM: Efficient World Action Models with State-Fusion Action Decoding
- **Date:** 2026
- **Problem framing:** Joint future/action models are costly for control.
- **Core mechanism:** Efficient state-fused world/action decoding.
- **Key insight:** Future prediction can be coupled to action generation efficiently.
- **Application domain:** World-action robot policy.
- **Overlap score:** 2/4 (future-prediction mechanism, domain).
- **Source:** https://www.semanticscholar.org/paper/f2d6863717cdae5ef1105f920e4d4cf81df4bf4d

### 18
- **Title:** S²-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation
- **Date:** 2026-06
- **Problem framing:** Static fusion amplifies early errors in long-horizon manipulation.
- **Core mechanism:** A GRU belief state dynamically gates visual, intent, and action-sequence attention branches using only action loss.
- **Key insight:** Task-phase belief should change modality/source emphasis over time.
- **Application domain:** LIBERO/SimplerEnv long-horizon VLA.
- **Overlap score:** 3/4 (problem, source-gating/recurrent-state mechanism, domain; no evidence-verified memory commit).
- **Source:** https://arxiv.org/abs/2606.27872

### 19
- **Title:** μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models
- **Date:** 2026-06
- **Problem framing:** Current observations omit information needed later.
- **Core mechanism:** Minimal recurrent memory tokens trained with TBPTT and action loss; an attention-mask guard prevents action-copy writes; inference is receding horizon.
- **Key insight:** Recurrence alone helps, but action-to-memory creates a degenerate self-referential shortcut.
- **Application domain:** MIKASA-Robo and LIBERO manipulation.
- **Overlap score:** 3/4 (problem, protected-memory mechanism, domain; it blocks action writes rather than validating provisional writes).
- **Source:** https://arxiv.org/abs/2606.12497

### 20
- **Title:** DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model
- **Date:** 2026-06
- **Problem framing:** Vision, language, and fast sensors should not share one clock.
- **Core mechanism:** Per-modality latent buffers refreshed at native rates and read by gated cross-attention.
- **Key insight:** Cache slow/static modalities and continuously read them from a fast action loop.
- **Application domain:** Contact-rich manipulation at 100 Hz.
- **Overlap score:** 2/4 (persistent cached contract/high-rate interface, domain).
- **Source:** https://arxiv.org/abs/2606.12105

### 21
- **Title:** AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
- **Date:** 2025-11
- **Problem framing:** Markov visual processing ignores history and partial observability.
- **Core mechanism:** The previous action-related hidden state becomes a recurrent belief; language-conditioned FiLM and belief cross-attention produce visual-token weights applied across backbone layers; an extra attention-weight penalty is trained.
- **Key insight:** Historical belief should modulate which current spatial tokens matter.
- **Application domain:** LIBERO, CALVIN, and real manipulation.
- **Overlap score:** 3/4 (problem, spatial/recurrent mechanism, domain; no transaction or future verification).
- **Source:** https://arxiv.org/abs/2511.18960

### 22
- **Title:** Look Where It Matters: Adaptive Visual Refinement for Vision-Language-Action Models
- **Date:** 2026-08
- **Problem framing:** VLA vision features lose fine spatial/contact information.
- **Core mechanism:** Visual register tokens plus action-disagreement-triggered attention rollout, cropping, high-resolution re-encoding, and rerunning the action expert; final training includes crop and attention-grounding losses.
- **Key insight:** Action uncertainty can decide when and where to reacquire fine visual detail.
- **Application domain:** LIBERO, SimplerEnv, and real-world manipulation.
- **Overlap score:** 2/4 (spatial-modulation family, domain).
- **Source:** https://arxiv.org/abs/2608.02197

### 23
- **Title:** CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation
- **Date:** 2026-07
- **Problem framing:** A committed chunk should be verified against observations arriving during execution.
- **Core mechanism:** A separately trained action-conditioned world model and calibrated risk head trigger latency-aware suffix rewriting; an event-driven bank preserves real-observation keyframes.
- **Key insight:** A chunk is both a command and a testable prediction of future observations.
- **Application domain:** RoboCasa365 long-horizon mobile manipulation.
- **Overlap score:** 3/4 (problem, insight, domain; external verifier/suffix rewrite differs from internal Task-Memory commit).
- **Source:** https://arxiv.org/abs/2607.26789

### 24
- **Title:** Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies
- **Date:** 2026-06
- **Problem framing:** Fixed flow-policy execution horizons ignore phase-dependent uncertainty.
- **Core mechanism:** Variance of clean-action estimates over late denoising steps selects a stable executable prefix at test time.
- **Key insight:** The flow trajectory itself reveals when future actions are unreliable.
- **Application domain:** LIBERO, RoboTwin, CALVIN, and real manipulation.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://arxiv.org/abs/2606.03847

### 25
- **Title:** Adaptive Action Chunking at Inference-time for Vision-Language-Action Models
- **Date:** 2026-04
- **Problem framing:** Fixed chunks trade responsiveness against temporal smoothness.
- **Core mechanism:** Action entropy chooses chunk size at inference.
- **Key insight:** Current prediction uncertainty should determine commitment depth.
- **Application domain:** Simulated/real VLA manipulation.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://arxiv.org/abs/2604.04161

### 26
- **Title:** When to Trust Imagination: Adaptive Action Execution for World Action Models
- **Date:** 2026-05
- **Problem framing:** World-action models blindly execute imagined futures with fixed horizons.
- **Core mechanism:** FFDC jointly attends predicted actions, predicted visual dynamics, current observations, and language, and classifies whether to continue or replan; it adds verifier supervision.
- **Key insight:** Future–reality consistency should govern execution commitment.
- **Application domain:** RoboTwin and real manipulation.
- **Overlap score:** 3/4 (problem, insight, domain; external binary verifier differs from Task-Memory transaction).
- **Source:** https://arxiv.org/abs/2605.06222

### 27
- **Title:** Leave No Observation Behind: Real-time Correction for VLA Action Chunks
- **Date:** 2025-09
- **Problem framing:** High-latency action chunks ignore observations received before the next base-policy call.
- **Core mechanism:** A lightweight per-control-step correction head uses the latest observation, base action, chunk index, and base features to add a residual.
- **Key insight:** A fast residual loop can recover reactivity without retraining the base VLA.
- **Application domain:** Kinetix and LIBERO Spatial.
- **Overlap score:** 2/4 (problem, domain).
- **Source:** https://arxiv.org/abs/2509.23224

### 28
- **Title:** Action Chunking with Transformers
- **Date:** 2023-04
- **Problem framing:** Fine-grained imitation suffers compounding error and multimodality.
- **Core mechanism:** CVAE Transformer predicts action chunks and temporally ensembles overlapping predictions.
- **Key insight:** Chunking reduces effective horizon while overlap smooths replanning boundaries.
- **Application domain:** Bimanual manipulation.
- **Overlap score:** 1/4 (action-chunk domain/mechanism ancestor).
- **Source:** model-recall; https://arxiv.org/abs/2304.13705

### 29
- **Title:** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **Date:** 2023-03
- **Problem framing:** Learn multimodal visuomotor control with stable closed-loop execution.
- **Core mechanism:** Conditional action diffusion with receding-horizon control.
- **Key insight:** Replan from new observations while predicting a coherent horizon.
- **Application domain:** Robot manipulation.
- **Overlap score:** 1/4 (domain/control ancestor).
- **Source:** model-recall; https://arxiv.org/abs/2303.04137

### 30
- **Title:** Flamingo: a Visual Language Model for Few-Shot Learning
- **Date:** 2022-04
- **Problem framing:** Add vision to a frozen language model efficiently.
- **Core mechanism:** Perceiver-resampled visual tokens enter frozen LM layers via gated cross-attention.
- **Key insight:** Cache/compress one modality and repeatedly inject it with learnable gates.
- **Application domain:** General multimodal learning, not robot control.
- **Overlap score:** 1/4 (gated cached-conditioning mechanism).
- **Source:** model-recall; https://arxiv.org/abs/2204.14198

### 31
- **Title:** Perceiver IO: A General Architecture for Structured Inputs & Outputs
- **Date:** 2021-07
- **Problem framing:** Process large structured multimodal inputs with bounded latent computation.
- **Core mechanism:** Fixed latent bottleneck with cross-attention in and query-based outputs.
- **Key insight:** Constant latent size decouples compute from input size.
- **Application domain:** General multimodal models.
- **Overlap score:** 1/4 (latent/resampler mechanism).
- **Source:** model-recall; https://arxiv.org/abs/2107.14795

### 32
- **Title:** Dreamer: Reinforcement Learning with Latent Dynamics Models
- **Date:** 2019-12
- **Problem framing:** Learn behavior from compact recurrent latent world dynamics.
- **Core mechanism:** Recurrent state-space model with deterministic/stochastic belief and imagined rollouts.
- **Key insight:** Prediction error and latent belief support planning under partial observability.
- **Application domain:** General model-based RL, not VLA imitation.
- **Overlap score:** 2/4 (latent dynamics mechanism and prediction-based insight).
- **Source:** model-recall; https://arxiv.org/abs/1912.01603

## Obvious false positives from the API search

All entries below score **0/4**: their problem framing, mechanism, insight, and domain differ from the proposed VLA memory transaction. The abstract/title was still inspected so the negative result is auditable.

| Title | Date | Problem/mechanism/domain summary | Source |
|---|---:|---|---|
| Compositional Context Fine-Tuning VLM for Complex Assembly Action Understanding | 2026 | Video action understanding, not robot closed-loop control | arXiv:2607.10797 |
| Towards Long-Horizon Vision-Language Navigation | 2024 | Navigation benchmark/method | arXiv:2412.09082 |
| SERF | 2026 | Spatiotemporal maps for mobile manipulation, no proposed transaction | arXiv:2606.12956 |
| EgoSteer | 2026 | Egocentric steerable dexterous system | Semantic Scholar search result |
| SimLingo | 2025 | Autonomous driving language-action alignment | DOI:10.1109/CVPR52734.2025.01120 |
| Bridging Retrospection and Prospection | unknown | High-level long-horizon framework; insufficient verified method detail | DOI:10.2139/ssrn.7201933 |
| Failure Report And Corrective Action System | 2024 | Reliability engineering, not learned robot policy | DOI:10.1109/RAMS51492.2024.10457668 |
| VLAG | 2025 | Graph planning, not latent policy memory | DOI:10.1115/DETC2025-169527 |
| A Lightweight Modular VLA Framework | 2026 | Generic modular control; no matching mechanism in returned metadata | DOI:10.1109/YAC71005.2026.11615600 |
| What Am I? | 2024 | Human perception of social robots | arXiv:2410.11085 |
| VLS | 2026 | VLM steering of pretrained policies | arXiv:2602.03973 |
| Trajectory-Level Redirection Attacks on VLA Models | 2026 | Adversarial attack | Semantic Scholar search result |
| Tool-Aligned VLA Models | 2026 | Tool-aligned long-horizon agents | Semantic Scholar search result |
| ExploreVLM | 2025 | Exploration task planning | DOI:10.1109/ROBIO66223.2025.11377225 |
| Grounding LLMs for Robot Task Planning Using Closed-Loop State Feedback | 2025 | Language-level task planning | DOI:10.1002/ADRR.202500072 |
| Closed Loop Sleep Motor Memory | 2024 | Neuroscience dataset | DOI:10.70883/QXBH2876 |
| Human-in-the-Loop Robot Action Replanning | 2025 | Human/LLM task replanning | DOI:10.1109/LRA.2025.3604702 |
| DELTA | 2025 | LLM task decomposition | DOI:10.1109/ICRA55743.2025.11127838 |
| Robots Can Multitask Too | 2024 | LLM/memory task action generation | DOI:10.1109/HUMANOIDS58906.2024.10769803 |
| Latent-Y | 2026 | Drug design | arXiv:2603.29727 |
| Latent-X | 2025 | Protein binder design | arXiv:2507.19375 |
| Drug-like antibodies with Latent-X2 | 2025 | Antibody design | arXiv:2512.20263 |
| CARE | 2026 | Generic continuous latent action representation | DOI:10.1109/ICASSP55912.2026.11460391 |
| Non-robot latent-memory and behavior-tree hits | 2024–2026 | Psychology, language models, games, or symbolic planning rather than VLA execution | Crossref/OpenAlex results |
