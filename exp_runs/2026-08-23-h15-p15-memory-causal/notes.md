# H15/P15 assembly recurrent-memory causal audit

Date: 2026-08-23

## Question

Does the confirmed P2-training/P15-deployment recurrent-state mismatch explain the
assembly failures of the formal H15/P15 checkpoint?

## Fixed inputs

- Source checkpoint: `mw_hard2_l20_h15_p15_prefix_tail_from_s1752.to_equiv_s5037.pt`
- Source checkpoint SHA256: `98644bebb049f16c581ebc1808835793caecc2349b4719ec35d4808c035d6c92`
- Eval-only copy: optimizer/sampler/RNG entries removed; model/config/metric/contract entries unchanged.
- Eval-only SHA256: `b9d35b32b404ffdb003bb4bfed71c11cd492e0773c5c9d1db175c60c2ed315e1`
- DINO SHA256: `c893d72294d4c327e631ff92f428dbc14c4f93cb5581b6c5f9d89bb5d17def27`
- Task: assembly-v3, seeds 0-29, horizon 500, H15/P15, one Flow sample.
- Local environment: MetaWorld 3.0.0, MuJoCo 3.3.0, timm 1.0.27, PyTorch 2.10.0+cu128, EGL.
- The local baseline reproduced the remote seed 0/1/2 pattern exactly. Absolute results for later seeds are compared only within this local paired panel because remote PyTorch is 2.7.0+cu118.

## Arms

1. Baseline: persistent VA memory; World reset every four decisions; persistent map between resets.
2. Map reset: baseline plus `world_map` reset before every decision.
3. Full reset: all cross-decision `VisualMemory` reset before every decision.

## Results

| Arm | Seeds | Successes | Successful seeds |
|---|---:|---:|---|
| Baseline | 30 | 7 | 0, 4, 15, 25, 27, 28, 29 |
| Map reset | 10 | 3 | 0, 4, 8 |
| Full reset | 30 | 10 | 0, 4, 7, 8, 9, 15, 24, 25, 28, 29 |

For the paired 30-seed baseline versus full-reset comparison:

- Full reset rescued seeds 7, 8, 9, and 24.
- Full reset lost seed 27.
- Six seeds succeeded in both arms.
- Exact paired sign/McNemar test over the five discordant seeds: one-sided p=0.1875; two-sided p=0.375.
- The originally diagnosed failures, seeds 1 and 2, remained failures under map reset and full reset.

## Verdict

The P2/P15 recurrent-input mismatch is real in code and data, and persistent memory
can change outcomes. It is not established as the primary cause of assembly failure:
the 7/30 to 10/30 increase is inconclusive and the key failures were not rescued.
Do not rebuild the entire training contract around P15 memory solely on this evidence.
The next evaluation should use the newly trained assembly-center/all-stage-readout
checkpoint, then isolate tail/contact precision and phase recovery if failures remain.
