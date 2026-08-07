# VA communication diagram defect log

## Cycle 1 — semantic and visual inventory

Screenshot: `cycle1.png`

| # | Priority | Finding | Disposition |
|---:|:---:|---|---|
| 1 | P0 | The legend says blue boxes are facts, but code facts also use violet and amber. | Fix legend wording. |
| 2 | P0 | The matrix arrow appears to leave the L cell, which can imply only L contributes. | Strengthen the K/U concatenation formula in the matrix title. |
| 3 | P0 | The most important missing experiment, direct A-intervention to V sensitivity, is not first. | Promote to P0 and first card. |
| 4 | P0 | “Code has a path” could be mistaken for proof that the trained model uses it. | Explicitly state path existence versus utilization. |
| 5 | P1 | Same-layer parallel update is easy to miss. | Increase callout text size and retain blue outline. |
| 6 | P1 | Fresh-state communication delay is not encoded in the layer formula itself. | Keep explicit next-layer callout; avoid inventing same-layer feedback arrow. |
| 7 | P1 | The A stream can be mistaken for the noisy Flow state. | Keep Section B title and initialization source explicit. |
| 8 | P1 | The candidate/noisy action does not enter VA, but the absence is only a sentence. | Retain red dashed no-feedback path and red code-fact callout. |
| 9 | P1 | Risk cards were not ordered by experiment priority. | Reorder as P0/P1/P2. |
| 10 | P1 | One-layer delay and path collapse were split in a way that hid the direct test. | Separate utilization test from delay test. |
| 11 | P1 | Shared-softmax token competition should be presented as a hypothesis, not a defect. | Keep it only in red hypothesis area. |
| 12 | P1 | Solver feedback absence is a fact; whether it hurts is a hypothesis. | Preserve fact in Section B and ablation in Section C. |
| 13 | P1 | Matrix title does not expose Q concatenation. | Add `Q=[Q_V;Q_A]`. |
| 14 | P1 | Matrix title does not expose K/U concatenation order. | Add `K/U=[V;M;A;L]`. |
| 15 | P2 | Role-mask text is small for a dense sentence. | Increase from 12 to 13 pt. |
| 16 | P2 | Timing callout text is small relative to its importance. | Increase from 12 to 13 pt. |
| 17 | P2 | No-feedback callout is visually subordinate to the pipeline. | Increase from 13 to 14 pt. |
| 18 | P2 | Risk-card body text leaves excessive unused space. | Increase card typography. |
| 19 | P2 | Footer is difficult to read at presentation scale. | Increase from 10 to 11 pt. |
| 20 | P2 | Subtitle is too small relative to the 1900 px canvas. | Increase from 13 to 14 pt. |
| 21 | P2 | Risk-card numbering implies sequence but not priority. | Replace numbers with P0/P1/P2. |
| 22 | P2 | Direct A→V ablation and gradient logging are separated. | Put ablation in P0; gradients with token competition. |
| 23 | P2 | “latent/physical gap” needs an executable comparison. | Keep projected `a^tau` injection ablation. |
| 24 | P2 | “No solver feedback” needs a matched baseline. | Specify fixed condition versus periodic re-encode. |
| 25 | P3 | Input labels mix Chinese and English. | Retain: English matches code identifiers; Chinese carries interpretation. |
| 26 | P3 | V and A use different fills but the same neutral outline in most boxes. | Retain to keep the palette coherent; emphasize cross paths with blue outline. |
| 27 | P3 | Section A is dense. | Retain because it is the evidence-bearing core; keep B/C simpler. |
| 28 | P3 | The four input projections have no extra crossing lines into every table cell. | Retain to avoid wire clutter; formula provides the mapping. |
| 29 | P3 | The output boxes sit close to the right edge. | Accepted; 30 px page margin remains. |
| 30 | P3 | The footer sits near the lower page edge. | Accepted; no clipping is visible. |
| 31 | P3 | The red feedback line crosses the pipeline only at its source-side return. | Accepted; it is deliberately dashed and labelled. |
| 32 | P3 | The matrix uses “U” rather than conventional “V” for attention values. | Retain because the implementation names the value projections `u_*`; avoid collision with visual V. |

Cycle-1 fixes applied: findings 1, 3–5, 9–10, 13–24. Findings 2 and 28 are resolved by the explicit concatenation formula rather than adding crossing wires.

## Cycle 2 — precision and ambiguity inventory

Screenshot: `cycle2.png`

| # | Priority | Finding | Disposition |
|---:|:---:|---|---|
| 1 | P0 | The legend still treats all red elements as hypotheses, but the no-feedback callout is a code fact. | Split legend semantics into data flow, missing loop, and hypothesis cards. |
| 2 | P0 | “No feedback” is broader than the actual missing link; the Flow Head does have iterative action-state feedback. | Rename line to `NO VA RE-ENCODE`. |
| 3 | P0 | “Candidate trajectory does not look at vision” can imply that vision never conditions the Flow Head. | Say it does not re-query VA/Vision. |
| 4 | P1 | M is drawn as always present, while the first time step passes `None`. | Add `t=0 absent`. |
| 5 | P1 | `Euler ×8` looks like a fixed architectural constant. | Add `(default)`. |
| 6 | P1 | “Old state” is inaccurate for the language cache and current memory input. | Rename to input state. |
| 7 | P1 | The A→V cell says “action writes vision,” which can be read as physical actions. | Qualify it as latent A condition. |
| 8 | P1 | The direct A→V test must hold V/M/L fixed to isolate causality. | Already explicit in P0 card; retain. |
| 9 | P2 | The subtitle is long but remains within its box. | Accepted after screenshot verification. |
| 10 | P2 | The concatenation formula is dense. | Accepted because it removes a more serious source-arrow ambiguity. |
| 11 | P2 | The mask paragraph is a single line. | Accepted; 13 pt text is readable and no overflow is visible. |
| 12 | P2 | Risk cards mix Chinese and English experiment terms. | Accepted to preserve one-to-one mapping with code/API names. |
| 13 | P2 | The dashed return path is close to the Section B header. | Accepted; it remains fully inside the pipeline region. |
| 14 | P3 | The Flow Head box does not repeat tensor shapes. | Accepted; adjacent condition/output boxes carry the shapes. |
| 15 | P3 | The matrix does not depict attention heads individually. | Accepted; eight heads do not alter the communication topology. |
| 16 | P3 | Previous visual memory is shown as a source for both query roles. | Verified against the implementation; retain. |
| 17 | P3 | The footer reports missing A→V sensitivity coverage in English. | Accepted because it matches the test concept precisely. |

Cycle-2 fixes applied: findings 1–7. Findings 8–17 were verified or intentionally retained.

## Cycle 3 — final refinement inventory

Screenshot: `cycle3.png`

| # | Priority | Zone | Finding | Disposition |
|---:|:---:|---|---|---|
| 1 | P1 | Text | The A→V subtitle wraps the final Chinese character onto a separate line. | Shorten to `latent A 条件 → ΔV`. |
| 2 | P1 | Typography | `NO VA RE-ENCODE` is the smallest semantic label in Section B. | Increase edge label from 10 to 11 pt. |
| 3 | P2 | Arrows | The row-output arrow visually begins at the L cell even though all four K/U sources participate. | Accept with explicit `K/U=[V;M;A;L]` formula; extra fan-in wires would reduce legibility. |
| 4 | P2 | Boxes | Equal-height risk cards leave different amounts of blank space. | Accept to preserve row alignment and comparison rhythm. |
| 5 | P2 | Spacing | Section A is denser than Sections B/C. | Accept because Section A carries the exact communication matrix. |
| 6 | P2 | Color | Vision and action cross-path cells share the same blue emphasis stroke. | Accept: the fill color still distinguishes source modality while the outline marks both critical paths. |
| 7 | P2 | Layout | No icons are used for V/A/M/L. | Accept: letter identities map directly to implementation variables and remain editable. |
| 8 | P2 | Text | The footer remains intentionally small. | Accept; it is readable at the 1900×1080 canvas screenshot and functions as provenance, not primary content. |
| 9 | P2 | Semantics | The matrix depicts `bidir_va`; `uni_a` is described textually rather than as a second matrix. | Accept to avoid duplicating the main panel; the role mask sentence is exact. |
| 10 | P2 | Style | English and Chinese are mixed in labels. | Accept because code identifiers and experiment names should remain searchable. |

Cycle-3 fixes applied: findings 1–2. All P0/P1 findings are resolved; findings 3–10 are acknowledged P2 tradeoffs.

## Fix verification across cycles

| Cycle | Claimed fixes | Old evidence | New evidence | Status |
|---:|---|---|---|---|
| 1 | Correct legend, expose Q/K/U concatenation, reorder experiments, enlarge dense text, prioritize direct A→V test. | `cycle1.png` | `cycle2.png` | ✅ FIXED — wording, hierarchy, and typography visibly changed; no new overlap. |
| 2 | Separate red fact/hypothesis semantics, qualify solver feedback, mark M absent at t=0, mark Euler steps as default. | `cycle2.png` | `cycle3.png` | ✅ FIXED — all five labels are visible and fit their boxes. |
| 3 | Remove the A→V orphan line-wrap and enlarge the no-reencode label. | `cycle3.png` | `final.png` | ✅ FIXED — subtitle is one line and the dashed-loop label is visibly larger. |

## Latest screenshot evidence

| Artifact | Capture type | Diagram coverage | Result |
|---|---|---:|---|
| `final.png` | Canvas-only crop | 100% of the 1900×1080 page | Valid for audit; no browser chrome, clipping, or hidden region. |

## Requirement and semantic audit

| Requirement | Latest screenshot evidence | Result |
|---|---|---|
| Show actual V→A and A→V paths. | Two emphasized matrix cells in Section A. | PASS |
| Show same-layer update semantics. | Timing callout states both outputs use pre-update states and re-communicate at layer i+1. | PASS |
| Distinguish bidirectional and action-only control. | Role-mask callout gives exact `bidir_va` and `uni_a` read permissions. | PASS |
| Explain what A actually represents. | Section B starts from learned queries plus robot state and names A as latent action-condition. | PASS |
| Show Flow Matching relation to VA. | Fixed condition C_t feeds the Flow Head; noisy action and time stay in the head. | PASS |
| Show the missing solver-to-VA loop. | Red dashed `NO VA RE-ENCODE` path and factual callout. | PASS |
| Separate facts from hypotheses. | Legend plus dedicated Section C with explicit hypothesis wording. | PASS |
| Give actionable diagnostics. | Five priority-ranked experiment cards. | PASS |

## Red-team audit — hostile reviewer pass

Evidence: `final.png`, canvas-only. All findings below are residual P2 tradeoffs; no P0/P1 blocker remains.

| # | Zone | Element/region | Residual finding | Severity | Fix or accept |
|---:|---|---|---|:---:|---|
| 1 | Text readability | `footer` | Provenance is readable but still the smallest text on the page. | P2 | Accept; it is secondary metadata. |
| 2 | Text readability | `noFeedbackLine` | The dashed-loop label is smaller than box labels. | P2 | Accept; it is now 11 pt and readable. |
| 3 | Text readability | matrix formulas | Superscripts/subscripts require slightly more attention than plain labels. | P2 | Accept; they preserve exact tensor-role notation. |
| 4 | Text readability | risk cards | Dense bilingual experiment lines are less glanceable than the top-level flow. | P2 | Accept; removing code terms would reduce executability. |
| 5 | Arrow hygiene | `e5`, `e7` | Output arrows start at the rightmost source cell, so the visual fan-in is implicit. | P2 | Accept; matrix title explicitly defines all K/U sources. |
| 6 | Arrow hygiene | projection-to-matrix gap | The four projection boxes do not have explicit wires to every matrix role. | P2 | Accept; direct wires would introduce crossings and duplicate the matrix. |
| 7 | Arrow hygiene | `noFeedbackLine` | A prohibition line is less conventional than a crossed arrow legend. | P2 | Accept; red dashed style, cross endpoint, label, and callout agree. |
| 8 | Box integrity | risk cards | Equal card heights create different unused bottom space. | P2 | Accept; equal geometry improves comparison. |
| 9 | Box integrity | `maskCallout` | A long single-line explanation occupies a wide box. | P2 | Accept; no clipping and the sentence is the exact control definition. |
| 10 | Spacing consistency | Section A | Left projections use wider horizontal gaps than matrix cells. | P2 | Accept; the gap separates preprocessing from attention semantics. |
| 11 | Spacing consistency | Section A vs B/C | Section A is visibly denser. | P2 | Accept; it is the evidence-bearing core. |
| 12 | Color/palette | red elements | Red has strong visual weight relative to the main blue path. | P2 | Accept; it flags missing connectivity and unverified risks. |
| 13 | Color/palette | cross-path outlines | V→A and A→V share one blue emphasis outline despite different fills. | P2 | Accept; outline means “critical path,” fill means modality. |
| 14 | Typography | whole page | Chinese prose and English identifiers share lines. | P2 | Accept; identifiers remain searchable and code-aligned. |
| 15 | Layout/composition | layer stack | Four VA layers are summarized as ×4 rather than drawn four times. | P2 | Accept; repetition would obscure the one-layer communication rule. |
| 16 | Layout/composition | temporal memory | The full t−1→t memory lifecycle is not expanded. | P2 | Accept; this artifact is scoped to VA communication. |
| 17 | Icons | whole page | No pictorial modality icons are used. | P2 | Accept; V/A/M/L symbols map more precisely to implementation variables. |
| 18 | Style coherence | overall | The figure is engineering-review dense rather than marketing-slide minimal. | P2 | Accept; density matches the requested actual architecture/debugging purpose. |

## Self-score

| Dimension | Score | Evidence and concrete deduction |
|---|---:|---|
| Text readability | 9/10 | All primary text is readable at 100%; deduct 1 because the footer and dashed-loop label remain 11 pt. |
| Arrow accuracy | 8/10 | Directions and missing loop are correct; deduct 2 because fan-in from four K/U columns is represented by the matrix formula rather than four explicit connectors. |
| Color coherence | 9/10 | Six coherent fills and three stroke colors; deduct 1 because red necessarily carries more visual weight than the core flow. |
| Layout consistency | 9/10 | Grid alignment, margins, and row rhythm are consistent; deduct 1 because Section A is substantially denser than B/C. |
| Style match to spec | 9/10 | Editable, publication-style, code-grounded architecture figure; deduct 1 because bilingual technical labels reduce stylistic minimalism. |
| **TOTAL** | **44/50** | **ALLOWED: total ≥40 and every dimension ≥6.** |

## Remaining gaps

| Gap | Severity | Reason retained | Next action if needed |
|---|:---:|---|---|
| K/U fan-in is tabular rather than individually wired. | P2 | Avoids eight crossing connectors. | Add a brace/aggregator only for a lecture version. |
| Full temporal memory lifecycle is summarized. | P2 | Outside the narrow VA-communication focus. | Pair this figure with the broader system architecture diagram. |
| Risks are hypotheses, not measured failures. | P2 | Current repository has no instrumentation results. | Implement the P0 A-intervention→V sensitivity test first. |

## Final validation

- Draw.io XML integrity: PASS — 1 page, 66 cells, 50 vertices, 14 edges, no raster/external images, no duplicate IDs.
- Strict visual-quality preflight: PASS — 0 FAIL, 0 WARN.
- Final canvas screenshot: PASS — 1900×1080 RGB PNG.
- Model semantic regression check: PASS — 15/15 `tests.test_model` unittest cases.
- `pytest` was unavailable in the configured GPU environment; the equivalent standard-library unittest suite completed successfully.
