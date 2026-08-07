# Step 1 — Candidate claim

## Proposed direction

Use the existing bounded recurrent visual state as an instrument for a plan–act–verify–recover agent, and study the causal chain

`instruction → memory state → verifier trigger → replan → recovery`.

## Falsifiable central claim

Event-triggered replanning is useful only for identifiable failure states; its causal benefit can be estimated by branching from the same simulator state and comparing `continue` versus `replan`, while separately intervening on instruction, memory, trigger, and plan.

## Initial novelty risk

Architecture novelty is weak because HiMe, Sentinel-VLA, SAFE, UniIntervene, Progress Critic/VLAC, TapSampling, and VLAs-as-Tools already cover lightweight monitors, progress estimation, event-triggered planning, or recovery. The defensible novelty must therefore be the intervention protocol and the mechanism-level findings, with bounded state as a controlled constraint rather than the headline contribution.
