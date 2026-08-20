# Step 4 — High-Potential Candidates

- Timestamp: 2026-08-07T06:00:47+08:00
- Selection rule: overlap >=2, direct core-mechanism overlap, ambiguity requiring full text, or a same-subfield paper from the last 24 months. Capped at seven.

1. **VLA-Corrector (arXiv:2607.01804)** — closest to using predicted-vs-observed frozen visual latent evolution as an online correction signal; selected to determine whether the proposed innovation token is already scooped.
2. **CheckVLA (arXiv:2607.26789)** — treats a chunk as a testable future prediction, uses action-conditioned verification, adaptive suffix retention, and real-observation memory; selected because it threatens both verification and commitment-decay claims.
3. **When to Trust Imagination / FFDC (arXiv:2605.06222)** — jointly verifies predicted vision, actions, current observation, and language before continuing a chunk; selected because it threatens “language-conditioned future–reality verification.”
4. **S²-VLA (arXiv:2606.27872)** — recurrent belief state generates three-source dynamic gates using only action loss; selected because it directly threatens SMC/source-wise gating and long-horizon-error claims.
5. **AVA-VLA (arXiv:2511.18960)** — recurrent belief and language-conditioned FiLM generate soft spatial visual-token weights; selected because it directly threatens SAM/spatial modulation.
6. **AtVLA (arXiv:2608.02197)** — action disagreement triggers action-attention localization and high-resolution re-encoding; selected as the newest and strongest action-conditioned spatial-attention neighbor.
7. **μVLA (arXiv:2606.12497)** — minimal recurrent memory tokens, action-only training, receding-horizon inference, and an explicit action-to-memory guard; selected because it directly threatens LVK/latent-memory claims while illuminating the proposed write-provenance delta.

Not deep-dived but retained as required baselines: A2C2 (per-step residual correction), DCDP (dynamic correction), DVAC/AAC (adaptive chunking), ReMem-VLA (dual-timescale recurrence), DAM-VLA (asynchronous cached modalities), ACT, and Diffusion Policy.
