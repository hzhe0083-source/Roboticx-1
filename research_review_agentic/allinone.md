# Critical review: next step after the lightweight recurrent VLA

## Decision

Do not submit the next project as another `lightweight + constant-memory + event-triggered replan` architecture. That combination is already densely occupied. Use the architecture as a controlled instrument and make the paper about **when replanning causally helps, when it harms, and which internal stage is responsible**.

Main path: **Causal Anatomy of Bounded-State Plan–Act–Verify Agents**.

Backup: **Memory Is Not Thinking: temporal memory versus latent recurrent computation in VLAs**.

## Why the plain P0 story is insufficient

HiMe, Sentinel-VLA, SAFE, UniIntervene, TapSampling, Progress Critic/VLAC, and VLAs-as-Tools jointly cover lightweight monitors, temporal progress estimation, event-triggered planner calls, and recovery. The closest system, VLAs-as-Tools, already combines a VLM dispatcher, specialized VLA tools, a local progress head, and threshold-triggered replanning. Constant memory and latency are valuable constraints/Pareto axes, but not enough as the central scientific claim.

For unbounded tasks, a fixed-size state cannot retain an unbounded number of independent facts. Use the precise term **bounded-state agent** and state the task family and information assumptions. Measure total persistent state, planner cache, language cache, visual state, and recovery memory—not only the 0.25 MiB recurrent slot. Also separate action-head throughput from full perception-to-action latency.

## Why the causal route remains defensible

Blank/Swap is no longer unique by itself: LIBERO-CF studies counterfactual commands, LIBERO-Para studies paraphrases, μVLA applies recurrent-memory interventions, and mechanistic VLA papers intervene on internal features. The missing piece is a reusable, paired closed-loop protocol that branches an identical simulator state and estimates the treatment effect of replanning while independently perturbing instruction, memory, trigger, and plan.

The central claim should be narrower than “language dependence is caused by data structure.” Cross-benchmark differences do not identify that cause. A defensible claim would be: **within a matched task generator, instruction conditional information and state aliasing predict where the executor stores language-relevant information and where a verifier-triggered replan changes recovery probability**.

## First main experiment

Use all 10 LIBERO-Long tasks as a developer pilot after repairing official demo/environment initialization. Inject four prespecified online failures: target displacement, empty grasp/drop, fixture-state rollback, and wrong-object distractors. Save the exact state at the first failure prefix and run paired continuations with common random numbers.

Freeze the VLA and train a very small Sentry from recurrent state, state delta, and action statistics. Compare executor-only, periodic replan, rate-matched random triggers, learned triggers, and oracle triggers. Use oracle recovery only to decompose the trigger gap from the planner gap, never as the headline system.

Primary metrics are paired causal effect of replan versus continue, conditional recovery success, false-replan harm, trigger delay, planner calls, subgoal progress AUC, p50/p95/p99 full-stack latency, and total state bytes. Move the paper evaluation to RoboCasa365 Composite-Seen/Unseen, adding a prespecified BEHAVIOR subset or real tasks when feasible.

## First backup experiment

Run a 2×2 factorial: temporal state carry/reset × one/four tied latent iterations. Match parameters and FLOPs with widened one-pass and untied-depth controls. Evaluate genuine state-aliasing tasks plus fully observable long-horizon tasks; use memory donor swaps to test whether the stored state causally selects the target/action. This separates remembering across time from doing more computation within a frame.

## Six-to-twelve-month plan

- Weeks 1–6: repair closed-loop evaluation; run paired LIBERO-Long pilot and oracle decomposition.
- Weeks 7–12: go/no-go; add second executor/backbone only if the causal effect survives.
- Months 4–7: RoboCasa365 evaluation, held-out failure families, and full latency/state accounting.
- Months 7–10: real-robot or BEHAVIOR subset, protocol release, writing.
- Venue: CoRL 2027 is the natural target; NeurIPS 2027 requires multi-backbone generality and a reusable causal-evaluation contribution; RSS 2027 requires stronger physical-robot evidence.

## Bottom line

The architecture race is not the best use of the existing advantage. The valuable asset is not the Blank/Swap pair itself, but the habit of asking causal questions and building interventions. Turn that into the contribution: a stage-wise, closed-loop, paired-state causal audit of memory, verification, replanning, and recovery.
