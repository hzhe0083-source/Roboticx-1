# Step 3 — Retained closest work

| Work | Direct overlap | Residual gap |
|---|---|---|
| [HiMe](https://arxiv.org/abs/2607.03449) | Executor–Sentry–Planner hierarchy; event-triggered planning | Does not isolate stage-wise causal effects under a fixed-state budget |
| [Sentinel-VLA](https://arxiv.org/abs/2605.01191) | Lightweight status monitor and on-demand recovery | System result, not cloned-state causal attribution |
| [SAFE](https://arxiv.org/abs/2506.09937) | Small hidden-feature failure detector | Detection only; no planner/recovery attribution |
| [VLAs-as-Tools](https://arxiv.org/abs/2605.13119) | Large VLM dispatcher, local progress head, threshold-triggered replanning | Almost exact system pattern; lacks full causal chain decomposition |
| [UniIntervene](https://arxiv.org/abs/2606.12372) | Temporal failure trend, intervention, memory-based recovery | Heavier/unbounded memory; no bounded-state capacity study |
| [TapSampling](https://arxiv.org/abs/2605.25547) | Action-conditioned progress verifier | Candidate selection rather than causal analysis of replanning |
| [μVLA](https://arxiv.org/abs/2606.12497) | Recurrent tokens, TBPTT, no auxiliary loss, memory interventions | Stops before verifier/replan/recovery chain |
| [LIBERO-CF](https://arxiv.org/abs/2602.17659) | Counterfactual language instructions and visual shortcuts | Does not intervene at each agentic stage |
| [LIBERO-Para](https://arxiv.org/abs/2603.28301) | Language robustness and planning failures | Paraphrase robustness, not causal recovery utility |
| [Mechanistic Interpretability for Steering VLAs](https://proceedings.mlr.press/v305/haon25a.html) | Causal activation intervention | Representation steering rather than agentic control-loop attribution |
| [Event-Grounded SAEs](https://arxiv.org/abs/2605.17204) | Closed-loop feature intervention | Sparse features, not bounded-state plan–act–verify mechanisms |

Overall scoop level: **Level 2 — high overlap**. Three of four axes—problem, system pattern, and evaluation family—are already occupied. The viable delta is the paired, cloned-state causal protocol plus a cross-model mechanism claim.
