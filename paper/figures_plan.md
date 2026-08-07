# Paper Figure Plan (ORA0 VA compound)

All figures must be generated from actual run logs / checkpoints; numbers
cross-checked against VA_COMPOUND_REPORT.md. Style: matplotlib, HarmonyOS
Sans SC, dpi 130 (existing scripts in artifacts/figures/).

## Fig 1 — Architecture (illustration, hand-drawn or diagram)
- Frozen Qwen cache (K_L/U_L) → VA composite (V/M/A streams) → flow head.
- Source: artifacts/va_compound_architecture/ + va_communication/ (drawio).
- Status: has drawio draft; needs final export.

## Fig 2 — PNPW depth & pooling ablation (Table 1 in paper)
- Data: §3.2/§3.4 (4/8 layers @10k/20k, flat/spatial, persistence baseline).
- Script: artifacts/figures/make_figures.py 图1 (already rendered).
- Status: DONE (benchmark_figures.png 图1).

## Fig 3 — Stream intervention audit (PNPW)
- Data: §3.3 (prev +3792%, proprio +291%, vision +155%, A→V +7.7%,
  M→A +2.9%, lang ~0%).
- Script: make_figures.py 图3.
- Status: DONE.

## Fig 4 — LIBERO language trio (core evidence)
- Data: §7.2 32-step: 1-scene blank +13751% / swap +1518%;
  3-scene blank +2381% / swap +607%.
- Script: make_figures.py 图4.
- Status: DONE.

## Fig 5 — LIBERO fit (A vs B40k vs C1 vs C2)
- Data: §8.6 + §11.9 + C2 verdict [TBD]: persistence 0.1486 / B10k 0.123 /
  A 0.0368 / B40k 0.0759 / C1 [TBD] / C2 [TBD] (32-step, v2 data for e2e rows).
- Script: make_figures.py 图5 (needs update after C2 eval).
- Status: PARTIAL.

## Fig 6 — Language sensitivity across datasets (key reversal)
- Data: §5.1 (PNPW ~0%) / §5.3 (MW wrong +182%) / §7.2 (LIBERO +2381%).
- Script: make_figures.py 图6 (needs MW wrong-instruction row added).
- Status: PARTIAL.

## Fig 7 — 2x2 control matrix (Qwen cosine + behavioral sensitivity)
- Data (2026-08-07 final, mask-corrected): cosine — original 0.8573 /
  random 0.0023 / B40k 0.9994 / C1 0.8573 / C2 0.9984; sensitivity — A
  +2381% / B40k +1.5% / C1 +33.7% / C2 +0.4%; clean chunk — 0.00254 /
  0.0759 / 0.08391 / 0.08390.
- Script: artifacts/figures/make_fig7_2x2.py — regenerated.
- Status: DONE (paper/fig7_2x2.png).

## Fig 8 — L_m same-scene dual-objective verdict
- Data: eval_libero_Lm.py output (5 pairs × D/O, bootstrap CI) for A and B40k.
- Script: artifacts/figures/make_fig_lm.py — READY (TBD slots).
- Status: script ready, numbers pending queue step 4/5.

## Fig 9 — MW open-loop + ablation + closed-loop
- Data: open-loop 0.0806; ablation wrong +108.5% / taskid +81.5%;
  closed-loop chain 7.1 → 16.3 → 17.8 (AQC); VA2 [TBD].
- Script: artifacts/figures/make_fig_mw.py — regenerated.
- Status: DONE except VA2 slot (paper/fig9_mw.png).

## Fig 10 — VLA-RL IL→RL curve
- Script: artifacts/figures/make_fig_rl.py — READY (TBD slots).
- Status: script ready, numbers pending VA2 + PPO run.

## Fig 11 — LIBERO-100 full-scale result
- Script: artifacts/figures/make_fig_libero100.py — READY (TBD slots).
- Status: script ready, numbers pending queue step 6/7.

## Tables
- Table 1: PNPW ablations (§5.1) — DONE.
- Table 2: literature comparison (§5.4) — DONE (footnotes per protocol).
- Table 3: efficiency (§5.8) — DONE.
- Table 4: C1/C2 vs A vs B40k 2x2 (cosine + trio + C_OL) — TODO (C2 eval).
- Table 5: L_m per pair (D, O, L_m, CI) — TODO.
- Table 6: VLA-RL IL vs RL (per task + macro) — TODO.
