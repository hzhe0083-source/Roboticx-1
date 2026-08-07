# Step 5 — Local evidence audit

The current local report does not yet support the strongest causal wording:

- LIBERO Blank/Swap effects are offline normalized chunk MSE over 360 samples/12 tasks, not closed-loop success.
- The effect is sampler-sensitive: the reported intervention gap changes substantially between 8 and 32 flow steps.
- LIBERO closed-loop evaluation was blocked by environment/dataset initialization mismatch in the recorded run.
- The report labels the old MetaWorld `+0.1%` result invalid/obsolete because of corrupted instruction IDs; it should not be used until independently rerun.
- Comparing LIBERO with MetaWorld changes dataset, tasks, observations, and action distributions together. It cannot alone identify dataset structure as the cause.

Required repair: manipulate instruction informativeness/ambiguity within a matched task generator, use feasible swapped commands, branch identical simulator states, and report absolute effects with confidence intervals alongside relative effects.
