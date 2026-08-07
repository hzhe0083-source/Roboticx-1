# Step 1 — Decompose the Novelty

- Timestamp: 2026-08-07T05:49:11+08:00
- Research problem: A frozen-backbone VLA has low open-loop action-chunk error and strong language ablations, but fails under closed-loop execution because execution drift is not converted into an explicit within-policy correction signal, task progress must survive long rollouts, and a multi-step chunk is committed before new evidence is incorporated.
- Working novelty claim: **Execution-aligned predictive-residual coupling (EPRC)** — use the already trained next-latent predictor to carry a one-step prediction into recurrent state, turn the next observed-minus-predicted latent residual into a protected correction token, and let that token modulate only the action-correction/task paths and the amount of chunk commitment. The cached language contract is compiled once but participates multiplicatively in the correction gate every control cycle. No additional IL objective is introduced.

## Four atomic axes

- **Problem framing** — Inputs are a cached immutable language contract, current frozen visual tokens, proprioception, the truly executed previous primitive action, and constant-size recurrent memories. Output is an 8-step flow-matching action chunk evaluated by closed-loop MetaWorld success under a >=10 Hz decision budget. The target gap is low teacher-forced/open-loop error but poor autonomous recovery and long-horizon completion.
- **Core mechanism** — Store the previous cycle's predicted next visual latent; on the next observation compute a normalized prediction residual without writing action/language into protected evidence memory. Project the residual to an innovation token used in A->V/T->A correction and a small horizon-wise commitment gate. Generate that gate from the interaction of cached language-contract features and current evidence/task state, not by rerunning Qwen.
- **Key insight** — The missing variable is not more absolute context but a causal *innovation*: what actually happened relative to what the policy expected after its own executed action. Exposing this mismatch to the correction pass lets the same FM supervision learn recovery, while execution-aligned previous-action inputs and receding commitment remove the main train/deploy mismatch.
- **Application domain** — Lightweight recurrent VLA policies for long-horizon robot manipulation with frozen language/vision backbones, constant memory, action-chunk flow or diffusion heads, and strict real-time control.

## Scope note discovered during decomposition

The current MetaWorld VA2 artifact must first be treated as invalid evidence for the novelty claim: `data/metaworld_features_v3_prevfix.pt` sets `previous_action[t>0] = actions[t-1,-1]`. With decision stride 6 and horizon 8, that is future primitive `d+7`, exactly equal to the next decision's target `actions[t,1]`, rather than the last executed primitive `d+5 = actions[t-1,5]`. This creates an open-loop label leak and a train/deploy contract mismatch; structural gains are interpretable only after rebuilding this field.
