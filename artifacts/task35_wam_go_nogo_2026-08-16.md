# task35 × WAM：问题先行 Go / No-Go（2026-08-16）

Verdict: **现在不该上完整 E7 WAM。** 15k 基线已是 37/50；剩下 13 次失败里多数是没靠近/没抓住，不是“插入后世界不可预测”。WAM 的原动机（语言服从 / H48 V-JEPA 未来 latent）在当前单任务 H6+DINO 设定上不成立。

## 1. Nail

- Task: MetaWorld `peg-insert-side-v3`，固定 seeds 35000–35049，H6 FM VA。
- Observed failure: 15k 仍有 13/50 失败。
- Consequence: 74% 还没到可靠精确插入；再训到 18k/20k 更差。
- Constraint: 单卡 3080；语言行全相同；WAM 后置且验收关着。
- Metric: 同一 50 seed 成功率；先解释这 13 局，再加模块。
- Falsifiable core: **这 13 局是开环接近/抓取失败，还是执行 6 步后的几何漂移？**

## 2. Problem exists?

**Established (supported)**

- Winner 15k: 37/50 = 74% Wilson [60.4%, 84.1%]. Sources: `logs/task35_best_fm.json`, `logs/..._step15000_eval50.json`.
- Later training worse: 18k 25/50, 20k 23/50.
- Causal on the same seeds: dense-zero 2/50, temporal-reverse 8/50, geometry −8/−5, roi-off +2. Source: `logs/..._step15000_causal_compare.json`.

**15k fail breakdown (supported, same JSON)**

| bucket | n | seeds (examples) |
|---|---:|---|
| never-approach (`grasp_r<0.5`, `min_d>0.2`) | 6 | 35002, 35004, 35009, 35014, 35028, 35033 |
| approach-no-grasp | 4 | 35007, 35027, 35044, 35046 |
| near-insert miss (`min_d` 0.09–0.11) | 3 | 35021, 35036, 35039 |

Fail `min_d` mean 0.249 vs success 0.065. Only **3/13** look like “almost in the hole”.

**WAM-as-the-fix: unverified**

- Spec `docs/superpowers/specs/2026-08-13-e7-wam-design.md` is for E7 H48, V-JEPA H11 16 tokens, 48-step residual, language-obedience.
- Current policy is H6, DINOv2 1024 tokens, cached identical language (`language_hidden` all rows equal).
- `eval_metaworld._resolve_wam` requires `dense_readout_mtvj` (V-JEPA last-slice). DINO-main + `want_vjepa_dense_backbone=False` does not match that contract.

## 3. Solution families (not just WAM variants)

1. **Data / coverage (simplest).** Audit the 13 seeds in the env; add approach/grasp recovery windows for those inits; keep the 15k policy. Predicts: never-approach count falls.
2. **Stop / select, no new module.** 15k already beats 18k/20k. Do not train another 5k. Optional: only if user reverses the 3k/6k/9k skip, 9k is the one unevaluated neighbor; 12k was 30%, so 9k beating 74% is unlikely.
3. **Small residual / replan (algorithm, cheap).** Shorter execute, or a tiny residual on the last 2 of 6 steps, only on the 3 near-miss seeds. Predicts: those 3 flip, the 10 approach fails do not.
4. **Full WAM (~60M, H48 V-JEPA cache).** Predicts future Δlatent/geometry and adds `v += α Δv`. Only justified if family 3 shows mid-horizon drift *and* a G1 probe shows future geometry is action-dependent on *this* H6 DINO stack.

## 4. Minimum falsifiable test

Do **not** train WAM first.

**T0 (1–2 h, no new weights):** dump RGB + stage traces for the 13 fail seeds vs 3 success seeds. Go if ≥8/13 never reach the peg; then WAM is the wrong tool.

**T1 (only if ≥5/13 are near-insert drift):** freeze 15k, execute 2 instead of 6 on those seeds (same policy). Go to a residual corrector only if shortening the open-loop chunk raises success on that subset.

**T2 (WAM G1, only after T1 go):** linear probe `action[0:6] → Δ(pegHead−hole)` vs constant and shuffled-action baselines on task35 H6 windows. Go to any WAM-like module only if +6 geometry error improves ≥10% vs constant *and* shuffle kills ≥50% of that gain.

Threshold: on the same 50 seeds, need **≥42/50** and Wilson lower bound **>74%** (strictly above the 15k point estimate) before claiming a win. `TBD` until T0 says which bucket we are targeting.

## 5. Data × algorithm

| | keep 15k FM | add residual / WAM |
|---|---|---|
| current 1807 windows | **A = 37/50** (measured) | B = planned |
| more approach/recovery on the 13 inits | C = planned | D = planned |

Decision uses `C−A` (data) vs `B−A` (algorithm). Do not run D until C or B alone is measured.

## 6. Claims

- “Next step is combine with WAM” — **speculative**. Motivated by an older E7 plan, not by the 13-fail anatomy.
- “WAM will raise insertion because the policy needs a world model” — **unverified**. Dense+temporal already explain most of the 74%.
- “15k is the strongest reproducible FM VA on this acceptance set” — **supported**.
- “ROI is the next module” — **refuted** by roi-off 39/50.

## 7. Next actions

- P0: T0 fail-seed traces. Do not start `train --wam-joint` on the 15k ckpt.
- P1: if T0 is approach-heavy, collect targeted recovery; if drift-heavy, T1 execute-2 then a *small* residual, not 60M WAM.
- P2: if WAM is still desired later, first rewrite the cache/eval contract for DINO H6; the E7 V-JEPA H48 cache builder is the wrong artifact.
