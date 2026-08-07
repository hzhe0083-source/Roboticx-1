# Step 2 — Search and Deduplicate

- Timestamp: 2026-08-07T06:00:47+08:00
- Window: 2024–2026 for API search; foundational works were added separately.
- Per-source cap: 5 papers per query.

## Queries

1. Original problem: `vision language action open loop accurate closed loop failure action chunk correction long horizon manipulation`
2. Broad domain: `robot vision language action closed loop replanning recurrent task memory`
3. Method signature (refined after decomposition): `speculative task memory commit rollback future latent prediction robot action policy`

An earlier method-signature attempt, `cached language contract predictive latent residual action correction recurrent memory robot`, was too lexical and returned mostly false positives; it is retained below as a negative-search result.

## Search-source status

- arXiv, Semantic Scholar, Crossref, and OpenAlex returned results.
- DBLP returned zero matches for these narrow phrases.
- OpenReview was unavailable because `openreview-py` is not installed.
- OpenAlex returned HTTP 504 for Query 1, then worked on later queries.
- These source failures are a coverage limitation; exact-title web searches and direct arXiv retrieval were used to augment the candidate set.

## Deduplicated relevant set

The API results, direct exact-title searches, and non-overlapping remembered foundations yielded the following relevant records:

1. Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy — 2026 — arXiv:2603.01953.
2. RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning — 2024 — arXiv:2409.14674.
3. Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection — 2026 — Semantic Scholar/arXiv.
4. Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA — 2026 — Semantic Scholar/arXiv.
5. VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon — 2026 — arXiv:2607.01804.
6. Dynamic Execution Horizon Prediction for Chunk-based Robot Policies — 2026 — Semantic Scholar.
7. RePLan: Robotic Replanning with Perception and Language Models — 2024 — arXiv:2401.04157.
8. Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection — 2026 — arXiv:2603.13782.
9. ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries — 2026 — arXiv:2603.12942.
10. Towards Long-Horizon Vision-Language-Action System: Reasoning, Acting and Memory — 2025 — ICCV.
11. Explicit Language Memory for Long-Horizon Planning in Vision-Language-Action Models — 2026 — Semantic Scholar.
12. ReplanVLM: Replanning Robotic Tasks With Visual Language Models — 2024 — IEEE RA-L.
13. π0: A Vision-Language-Action Flow Model for General Robot Control — 2025 — RSS.
14. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success — 2025 — RSS.
15. CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations — 2025 — arXiv:2505.04999.
16. DiLA: Disentangled Latent Action World Models — 2026 — arXiv:2605.15725.
17. Light-WAM: Efficient World Action Models with State-Fusion Action Decoding — 2026 — Semantic Scholar.
18. S²-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation — 2026 — arXiv:2606.27872 / IJCAI 2026.
19. μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models — 2026 — arXiv:2606.12497.
20. DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model — 2026 — arXiv:2606.12105.
21. AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention — 2025 — arXiv:2511.18960.
22. Look Where It Matters: Adaptive Visual Refinement for Vision-Language-Action Models (AtVLA) — 2026 — arXiv:2608.02197.
23. CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation — 2026 — arXiv:2607.26789.
24. Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies — 2026 — arXiv:2606.03847.
25. Adaptive Action Chunking at Inference-time for Vision-Language-Action Models — 2026 — arXiv:2604.04161 / CVPR 2026.
26. When to Trust Imagination: Adaptive Action Execution for World Action Models — 2026 — arXiv:2605.06222.
27. Leave No Observation Behind: Real-time Correction for VLA Action Chunks — 2025 — arXiv:2509.23224.

## Model-recall foundations (non-overlapping)

These were not returned by the 2024–2026 keyword search but are canonical mechanism ancestors; provenance is `model-recall`, with direct identifiers subsequently checked:

28. Action Chunking with Transformers (ACT) — 2023 — arXiv:2304.13705 — `Source: model-recall`.
29. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion — 2023 — arXiv:2303.04137 — `Source: model-recall`.
30. Flamingo: a Visual Language Model for Few-Shot Learning — 2022 — arXiv:2204.14198 — `Source: model-recall`.
31. Perceiver IO: A General Architecture for Structured Inputs & Outputs — 2021 — arXiv:2107.14795 — `Source: model-recall`.
32. Dreamer: Reinforcement Learning with Latent Dynamics Models — 2019 — arXiv:1912.01603 — `Source: model-recall`.

## Deduplicated obvious false positives retained for audit

The search also returned 24 title/abstract-level false positives: Compositional Context Fine-Tuning VLM for Assembly Action Understanding; Towards Long-Horizon Vision-Language Navigation; SERF; EgoSteer; SimLingo; Bridging Retrospection and Prospection; Failure Report and Corrective Action System; VLAG; A Lightweight Modular VLA Framework; What Am I?; VLS; Trajectory-Level Redirection Attacks; Tool-Aligned VLA Models; ExploreVLM; Grounding LLMs for Robot Task Planning; Closed Loop Sleep Motor Memory; Human-in-the-Loop Robot Action Replanning; DELTA; Robots Can Multitask Too; Latent-Y; Latent-X; Drug-like Antibodies with Latent-X2; CARE; and several non-robot latent-memory/behavior-tree results. They were deduplicated by normalized title and receive overlap score 0 in Step 3 rather than being silently omitted.

## Negative-search evidence

No returned paper used the specific mechanism signature “action proposal writes a provisional recurrent task state which is committed or rolled back only after next-step frozen-latent verification.” Searches did, however, find very close work on (a) prediction–observation verification for adaptive chunking, (b) belief-state source gating, (c) recurrent memory tokens, and (d) rollback/replanning outside the policy. Therefore absence of an exact phrase is not proof of novelty; the full mechanism comparison is required.
