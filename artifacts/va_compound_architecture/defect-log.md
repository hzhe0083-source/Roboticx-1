# VA Compound Architecture — Visual QA Log

## Pre-flight review

- Result before first preview: **0 FAIL, 4 WARN**.
- `va_policy` cavernous warning: valid signal; the box is visibly too tall and will be reduced in Cycle 1.
- Row-spacing warnings at `robot_state` and `query_streams`: cross-branch elements share a y-coordinate but are not a repeated grid; visually reviewed, with branch spacing retained intentionally.
- Column-spacing warning at `instruction`: the checker merges three different semantic sections; section headers make the differing vertical gaps intentional.

## Screenshot Review Cycle 1

Evidence: `pass-1-canvas.png` (1905×1065 canvas-only crop; diagram content &gt; 95%).

### P0 — Blockers

None. Static pre-flight found no collision or overflow FAIL, and the screenshot shows no box/arrow collision that changes semantics.

### P1 — Visible defects (all must be fixed)

| id | zone | element | defect / evidence | planned fix | status |
|---|---|---|---|---|---|
| C1-01 | 1 Text | `action_chunk` | The intended predicted-action symbol renders as the unrelated character “â”. | Replace with an unambiguous Chinese label plus indexed `a`. | OPEN |
| C1-02 | 1 Text | `subtitle` | 12 pt light-gray subtitle is weak at full-canvas scale. | Increase to 13 pt and darken one step. | OPEN |
| C1-03 | 1 Text | runtime edges | Most 9 pt connector labels require zoom and sit against arrow lines. | Raise connector labels to 10 pt and shorten long phrases. | OPEN |
| C1-04 | 1 Text | training edges | Long 9 pt labels such as “validate pair contract” are squeezed into 50 px gaps. | Shorten to compact semantic labels and use 10 pt. | OPEN |
| C1-05 | 1 Text | `training_note` | The source/boundary note is low-contrast and too near the page bottom. | Move upward, darken, and use 12 pt. | OPEN |
| C1-06 | 2 Arrows | `edge_action_policy`, `edge_memory_feedback` | Purple action input and orange temporal-memory input terminate within a few pixels at the policy’s lower-left edge. | Separate entry ports vertically after shrinking the policy box. | OPEN |
| C1-07 | 2 Arrows | two feedback edges | Orange memory and gray controller feedback labels occupy the same narrow band. | Increase lane separation and add opposite label offsets. | OPEN |
| C1-08 | 2 Arrows | `edge_mask_attention` | “allowed” is too small and nearly touches the attention border. | Rename to “mask”, increase label size, retain clean vertical route. | OPEN |
| C1-09 | 2 Arrows | layer output edges | `Vᶦ` and `Vᶦ,Aᶦ` labels are too small to distinguish the two branches quickly. | Raise to 10 pt and strengthen contrast. | OPEN |
| C1-10 | 2 Arrows | training edges | Labels visually touch the source/target borders, obscuring the arrow stems. | Use short labels that fit the gaps. | OPEN |
| C1-11 | 3 Boxes | `va_policy` | 450×290 policy box has excessive empty space around a nine-line core description. | Reduce to 450×220 and rebalance entry/exit ports. | OPEN |
| C1-12 | 3 Boxes | `role_mask` | The `uni_a` clause wraps awkwardly inside 460×45. | Widen to 560 px and simplify the sentence. | OPEN |
| C1-13 | 4 Spacing | feedback lanes | Orange and gray feedback lanes have only about 10 px vertical separation. | Move controller lane lower and memory lane higher. | OPEN |
| C1-14 | 4 Spacing | policy inputs | Three policy inputs are not evenly separated after automatic routing. | Use explicit entry ratios matching the three source rows. | OPEN |
| C1-15 | 4 Spacing | `training_note` | Bottom margin is visually smaller than the top/title margin. | Raise the note and leave a clear bottom margin. | OPEN |
| C1-16 | 5 Color | `edge_controller_state` | Thin light-gray feedback is hard to trace across the canvas. | Increase stroke to 2 px and use the established slate color. | OPEN |
| C1-17 | 6 Typography | query/KV projection boxes | The rendered layer detail omits the code’s pre-attention LayerNorm operations. | Add concise LayerNorm wording before the q/k/u projections. | OPEN |
| C1-18 | 6 Typography | `residual_ffn` | The FFN path does not state the separate pre-FFN norms used in code. | Add “pre-norm” to the V/A FFN description. | OPEN |
| C1-19 | 7 Layout/semantics | `language_cache` | Only the backbone hidden shape is shown; the per-layer projected cache shape is missing. | Add `[B,8,Nl,64]` for per-layer K/U. | OPEN |
| C1-20 | 7 Layout/semantics | `training_losses` | Pair loss does not say that the action difference is taken at shared time `t=0`. | Add the exact `t=0` constraint. | OPEN |
| C1-21 | 7 Layout/semantics | `paired_sampler` | It does not state the implementation’s exact two-rows-per-pair batch contract. | Add “each pair contributes exactly two rows”. | OPEN |

### P2 — Polish

| id | zone | element | defect / evidence | planned fix | status |
|---|---|---|---|---|---|
| C1-22 | 1 Text | `runtime_note` | Legend is small and visually detached from the feedback lanes. | Increase to 12 pt and align with the lane start. | OPEN |
| C1-23 | 1 Text | `controller` | “repo 外部” is too pale relative to surrounding text. | Darken the qualifier slightly. | OPEN |
| C1-24 | 2 Arrows | `edge_q_attention` | Q label sits directly on the elbow and is easy to miss. | Use a small label offset if the next render still looks crowded. | OPEN |
| C1-25 | 2 Arrows | `edge_kv_attention` | K/U label is visually close to the role-mask callout below. | Recheck after widening/repositioning the mask box. | OPEN |
| C1-26 | 3 Boxes | `training_features` | Four data families are dense in a single 300×100 box. | Keep because it is the exact training contract; increase line clarity if needed. | OPEN |
| C1-27 | 4 Spacing | Section B | The projection-to-attention gap is larger than adjacent compute-stage gaps. | Retain for the Q/K/U fan-in unless Cycle 2 shows excessive whitespace. | OPEN |
| C1-28 | 5 Color | policy core | Violet fill is slightly stronger than the smaller trainable boxes. | Unify the fill with other trainable components. | FIXED before Cycle 1 screenshot review was logged |
| C1-29 | 6 Typography | mixed labels | “Query streams / Key value streams” are English while nearby headings are Chinese. | Convert to concise bilingual labels for consistency. | OPEN |
| C1-30 | 6 Typography | `va_policy` | Core text uses a dense stack with weak line grouping. | Use the smaller box and clearer line ordering. | OPEN |
| C1-31 | 7 Layout | attention formula | The actual float32 score computation is not called out. | Add a compact “scores: fp32” note. | OPEN |
| C1-32 | 8 Icons | whole diagram | No icons are present; this is intentional under the engineering-schematic style contract, but modality scanning depends entirely on color/text. | Accept; do not add decorative icons. | ACCEPTED |
| C1-33 | 9 Style coherence | whole diagram | Edge-label scale is inconsistent with the otherwise readable engineering schematic. | The global 10 pt connector update should restore hierarchy. | OPEN |

### Five-dimension audit

- Requirement: all three required paths are present; cache shape and exact pair semantics need the fixes above.
- Semantic: arrow directions are correct; two policy-entry arrows need visual separation.
- Visual hygiene: no clipped boxes or crossings; small edge labels and feedback-lane crowding are the main issues.
- Style: palette is coherent after pre-flight consolidation; typography needs connector normalization.
- Regression: no prior screenshot exists; baseline established by `pass-1-canvas.png`.

## Fix Verification — Cycle 1

Compared `pass-1-canvas.png` → `pass-2-canvas.png`.

| defect id | verification | status |
|---|---|---|
| C1-01 | “â” is replaced by the readable indexed action label. | ✅ FIXED |
| C1-02 | Subtitle is darker and one point larger. | ✅ FIXED |
| C1-03 | Connector labels are 10 pt, but six short-gap runtime labels still crowd borders. | ⚠️ PARTIAL |
| C1-04 | Training labels are shorter, but the first two still exceed their 50 px gaps. | ⚠️ PARTIAL |
| C1-05 | Boundary/source note is darker, raised, and fully visible. | ✅ FIXED |
| C1-06 | Action and temporal-memory policy entries are now visibly separated. | ✅ FIXED |
| C1-07 | Lanes are separated, but the gray label moved into the Section B header. | 🔄 REGRESSION |
| C1-08 | Mask label is larger and clearly attached to the vertical mask arrow. | ✅ FIXED |
| C1-09 | Both layer-output branch labels are readable at full-canvas scale. | ✅ FIXED |
| C1-10 | Loss/save labels fit; the first two data-loader labels remain crowded. | ⚠️ PARTIAL |
| C1-11 | Policy box shrank by 70 px, though residual vertical whitespace remains. | ⚠️ PARTIAL |
| C1-12 | Role-mask sentence is now one clean line. | ✅ FIXED |
| C1-13 | Feedback lanes are separated by 18 px. | ✅ FIXED |
| C1-14 | Three policy inputs use distinct entry heights. | ✅ FIXED |
| C1-15 | Footer now has a visible bottom margin. | ✅ FIXED |
| C1-16 | Slate feedback line is now 2 px and traceable. | ✅ FIXED |
| C1-17 | Attention projections explicitly show LayerNorm. | ✅ FIXED |
| C1-18 | Residual box explicitly shows independent pre-norm FFNs. | ✅ FIXED |
| C1-19 | Per-layer cache shape `[B,8,Nl,64]` is visible. | ✅ FIXED |
| C1-20 | Pair loss now explicitly states `t=0`. | ✅ FIXED |
| C1-21 | Sampler now states exactly two rows per pair. | ✅ FIXED |

All Cycle 1 P1 items marked PARTIAL/REGRESSION are carried into Cycle 2 below; none is silently accepted.

## Screenshot Review Cycle 2

Evidence: `pass-2-canvas.png` (1905×1065 canvas-only crop).

### P0 — Blockers

| id | zone | element | defect / evidence | planned fix | status |
|---|---|---|---|---|---|
| C2-01 | 1 Text / 7 Layout | `edge_controller_state` | The full feedback label is partially hidden behind the Section B header; hidden text is a handoff blocker. | Move the label left and above its lane using an explicit x/y offset. | OPEN |

### P1 — Visible defects

| id | zone | element | defect / evidence | planned fix | status |
|---|---|---|---|---|---|
| C2-02 | 1 Text | six runtime gap labels | “text tokens”, tensor shapes, and “learned queries” are still wider than their short inter-box gaps. | Shorten to `tokens / hidden / frames / tokens / state / queries` and use 11 pt. | OPEN |
| C2-03 | 1 Text | first two training edges | “pair check” and “paired batch” still touch box borders. | Shorten to `check` and `pair`. | OPEN |
| C2-04 | 2 Arrows | `edge_policy_memory` | “保存每层 Vᶦ” is wider than the short policy→memory connector and competes with the arrowhead. | Replace with compact `Mₜ`. | OPEN |
| C2-05 | 3 Boxes / 4 Spacing | `role_mask`, `kv_projection` | The role-mask box begins exactly at the KV box’s right edge while their y-ranges overlap, creating a touching-corner tangent. | Move the mask box 30 px right and narrow it without re-wrapping. | OPEN |
| C2-06 | 3 Boxes | `va_policy` | The smaller policy box still contains visibly unused top/bottom space around five lines. | Reduce height to 190 px and retune entry/exit ratios. | OPEN |
| C2-07 | 2 Arrows | `edge_controller_state` | The gray label offset puts semantic text outside its own lane. | Consolidated with C2-01; anchor label over the left half of the gray lane. | OPEN |

### P2 — Polish

| id | zone | element | defect / evidence | decision | status |
|---|---|---|---|---|---|
| C2-08 | 1 Text | `edge_q_attention` | The Q label sits directly on the first elbow. | Offset only if it remains ambiguous after other geometry changes. | OPEN |
| C2-09 | 1 Text | `edge_kv_attention` | K/U label is close to the widened mask callout. | Recheck after mask movement. | OPEN |
| C2-10 | 4 Spacing | Section B fan-in | Projection→attention whitespace is larger than downstream stage gaps. | Keep because it separates two converging paths. | ACCEPTED |
| C2-11 | 5 Color | role mask vs memory | Both use amber, so mask/context and temporal memory share a semantic color. | Accept: both are non-trainable context/control state. | ACCEPTED |
| C2-12 | 6 Typography | Section B | English compute terms remain mixed with Chinese explanations. | Accept class/math names; preserve code vocabulary. | ACCEPTED |
| C2-13 | 6 Typography | subscripts/superscripts | Some tensor indices are smaller than nearby connector text. | Native mathematical hierarchy; accept if still readable. | ACCEPTED |
| C2-14 | 7 Layout | footer | One long source line carries many paths and is visually dense. | Keep as evidence; it is secondary and fully readable. | ACCEPTED |
| C2-15 | 8 Icons | whole diagram | No modality icons are used. | Intentional per style contract; no decorative assets added. | ACCEPTED |
| C2-16 | 9 Style coherence | short-gap arrows | Mixed label lengths make the left runtime grid feel less regular. | Compact all six labels. | OPEN |
| C2-17 | 9 Style coherence | policy core | Core visual weight is still slightly high versus the action head and memory boxes. | Reduce core height while preserving its dominant role. | OPEN |

### Five-dimension audit

- Requirement: exact cache and paired-training semantics are now present.
- Semantic: all arrow directions remain correct; no branch is missing.
- Visual hygiene: one hidden feedback label is the only blocker; tangent and short-label crowding are visible P1s.
- Style: color contract is stable; compact connector vocabulary will improve rhythm.
- Regression: C1-07 regressed and is promoted to C2-01/C2-07 rather than being marked fixed.

## Fix Verification — Cycle 2

Compared `pass-2-canvas.png` → `pass-3-canvas.png`; `pass-3-feedback-label-focus.png` confirms the feedback lanes at native resolution.

| defect id | verification | status |
|---|---|---|
| C2-01 | Full gray feedback label is visible above the left half of its lane and clear of Section B. | ✅ FIXED |
| C2-02 | Six runtime labels are now short, 11 pt, and fit between boxes. | ✅ FIXED |
| C2-03 | `check` and `pair` fit the training gaps. | ✅ FIXED |
| C2-04 | Policy→memory label is the compact `Mₜ`. | ✅ FIXED |
| C2-05 | A 30 px gap now separates role-mask and KV boxes. | ✅ FIXED |
| C2-06 | Policy core is 190 px tall and visually balanced around its five lines. | ✅ FIXED |
| C2-07 | Gray label is anchored left with a negative y-offset; it stays inside the runtime band. | ✅ FIXED |
| C2-08 | Q label remains readable on its elbow. | ✅ FIXED at acceptable polish level |
| C2-09 | K/U label remains clear after the mask box moved right. | ✅ FIXED at acceptable polish level |
| C2-16 | Compact connector vocabulary is regular across the runtime grid. | ✅ FIXED |
| C2-17 | Core still dominates semantically without appearing hollow. | ✅ FIXED |

No Cycle 2 P0/P1 defect remains unresolved in `pass-3-canvas.png`.

## Screenshot Review Cycle 3

Evidence: `pass-3-canvas.png` plus focused feedback crops. P0=0; only polish findings remain.

### P0 — Blockers

None.

### P1 — Visible defects

None.

### P2 — Polish

| id | zone | element | finding / evidence | action | status |
|---|---|---|---|---|---|
| C3-01 | 1 Text / 2 Arrows | `edge_cache_policy` | Full `K_Lᶦ,U_Lᶦ` notation is dense in the short cache→policy gap. | Shorten to `Lᶦ K/U`; tensor meaning remains inside the cache box. | OPEN |
| C3-02 | 1 Text / 2 Arrows | `edge_q_attention` | Q is readable but smaller than the now-normalized runtime labels. | Raise to 11 pt. | OPEN |
| C3-03 | 1 Text / 2 Arrows | `edge_kv_attention` | K/U is readable but smaller than the now-normalized runtime labels. | Raise to 11 pt. | OPEN |
| C3-04 | 3 Boxes | `role_mask` | The 520×45 callout is slightly heavy relative to the 240×105 attention box. | Reduce to 500×40 while retaining one line. | OPEN |
| C3-05 | 6 Typography / 7 Layout | `training_note` | Boundary and source provenance are one long dense footer line. | Split into left boundary note and right-aligned source note. | OPEN |
| C3-06 | 1 Text | `edge_losses_optimizer` | Standalone `∇` is compact but less obvious to non-ML readers. | Replace with the equally short `grad`. | OPEN |
| C3-07 | 4 Spacing | runtime lower band | Legend and gray feedback label both occupy the lower-left runtime band. | Accept: they are on separate baselines and the focused crop proves no overlap. | ACCEPTED |
| C3-08 | 5 Color | three amber concepts | Language cache, visual memory, and mask share amber. | Accept: all are non-trainable context/control state, and labels disambiguate them. | ACCEPTED |
| C3-09 | 8 Icons / 9 Style | whole diagram | Navigation is entirely typographic; there are no pictograms. | Accept under the declared editable engineering-schematic style. | ACCEPTED |

### Five-dimension audit

- Requirement: complete; no missing code-level component or training constraint.
- Semantic: every connector is traceable and points in the correct direction.
- Visual hygiene: no hidden text, crossings, or box overlaps; focused feedback crop confirms the tightest region.
- Style: consistent six-color semantic palette and 11–16 pt operational text hierarchy.
- Regression: Cycle 2 fixes are preserved; only the six targeted polish changes above will be made.

## Fix Verification — Cycle 3

Compared `pass-3-canvas.png` → `pass-4-canvas.png` → `pass-5-canvas.png`.

| defect id | verification | status |
|---|---|---|
| C3-01 | Cache connector now uses compact `Lᶦ K/U`. | ✅ FIXED |
| C3-02 | Q label is 11 pt and readable. | ✅ FIXED |
| C3-03 | K/U label is 11 pt and readable. | ✅ FIXED |
| C3-04 | First 500×40 attempt wrapped the final `K` in pass 4; pass 5 uses 520×40 at 11 pt and is one clean line. | ✅ FIXED after one regression repair |
| C3-05 | Boundary and provenance are separate aligned footer cells. | ✅ FIXED |
| C3-06 | Gradient edge says `grad`, which is compact and plain-language. | ✅ FIXED |
| C3-07 | Legend and gray feedback label remain on distinct baselines. | ✅ VERIFIED / ACCEPTED |
| C3-08 | Amber concepts remain explicitly labeled and semantically related. | ✅ VERIFIED / ACCEPTED |
| C3-09 | No icons were introduced; all objects remain editable primitives/text. | ✅ VERIFIED / ACCEPTED |

This completes three full screenshot→inventory→fix→verify cycles. No P0/P1 item remains.

## Red-Team Audit — Pass 5

Hostile review of `pass-5-canvas.png`, scanning all nine zones. All findings are residual P2 tradeoffs; none hides text, changes connector meaning, or violates the brief.

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| RT-01 | 1 Text | Subtitle is deliberately smaller than the title and needs normal-size viewing to read instantly. | P2 | Accept as secondary metadata. |
| RT-02 | 1 Text | Six left-grid connector labels remain smaller than body text even at 11 pt. | P2 | Accept; longer labels would collide in 40–50 px gaps. |
| RT-03 | 1 Text | Right-aligned source footer is the smallest provenance text on the page. | P2 | Accept; it is evidence, not process content. |
| RT-04 | 2 Arrows | Controller feedback spans almost the full page width. | P2 | Accept; it is the real closed-loop recurrence and is isolated in its own dashed lane. |
| RT-05 | 2 Arrows | Visual-memory recurrence also uses a long three-segment route. | P2 | Accept; orange styling and lane separation keep it traceable. |
| RT-06 | 2 Arrows | Q and K/U labels sit near orthogonal bends rather than in wide straight runs. | P2 | Accept; labels are readable and no alternate route is shorter without crowding. |
| RT-07 | 3 Boxes | The policy shows `×4` as a summarized core rather than four repeated mini-boxes. | P2 | Accept; Section B expands the exact repeated layer and avoids duplicating content. |
| RT-08 | 3 Boxes | Role-mask callout is unusually wide and shallow. | P2 | Accept; it keeps both comparison modes on one line. |
| RT-09 | 4 Spacing | Section B leaves more space before attention than after it. | P2 | Accept; the extra space hosts Q/K/U fan-in without crossings. |
| RT-10 | 4 Spacing | Runtime legend sits in the left half rather than centered under both feedback lanes. | P2 | Accept; centering would collide with the temporal-memory label. |
| RT-11 | 5 Color | Amber is reused for language cache, visual memory and role mask. | P2 | Accept; the color consistently means cached/context/control state. |
| RT-12 | 6 Typography | Chinese explanations mix with exact English class/function identifiers. | P2 | Accept; translating identifiers would reduce fidelity to code. |
| RT-13 | 7 Layout | The `features.pt` box is text-dense compared with later training stages. | P2 | Accept; it carries the real multi-tensor data contract. |
| RT-14 | 8 Icons | No pictograms distinguish text, video and robot-state inputs. | P2 | Accept under the no-decorative-icons style contract; color and labels are unambiguous. |
| RT-15 | 9 Style coherence | Stroke widths range from 1.5 to 2.5 px. | P2 | Accept as deliberate hierarchy: neutral/annotation &lt; components &lt; policy/attention core. |

Red-team result: 15 findings, 0 P0, 0 P1, 15 acknowledged P2.

## Self-Score — Pass 5

| dimension | score | screenshot-visible evidence |
|---|---:|---|
| Text readability | 9/10 | All labels are visible at 1905×1065; one point deducted because connector/provenance text is intentionally smaller than body text. |
| Arrow accuracy | 9/10 | All 29 arrows are directed and collision-free; one point deducted for the two necessarily long feedback routes. |
| Color coherence | 9/10 | Six semantic stroke colors are consistent; one point deducted because amber covers three related context concepts. |
| Layout consistency | 9/10 | Three bands, aligned grids and balanced outputs are stable; one point deducted for the asymmetric Q/K/U fan-in whitespace. |
| Style match to spec | 10/10 | Matches the declared compact editable engineering schematic with no raster/icon decoration. |
| **TOTAL** | **46/50 — ALLOWED** | Every dimension ≥ 6 and total ≥ 40. |

## Remaining Gaps

| gap | severity | reason | next action |
|---|---|---|---|
| Smaller connector/provenance typography | P2 | Prevents collisions at one-page density. | Split into multiple pages if presentation-scale typography is required. |
| Long runtime feedback routes | P2 | Represents real temporal/control recurrence. | Move runtime to its own page if a simpler executive view is desired. |
| No modality icons | P2 | Preserves editability and avoids decorative semantics. | Add only if the user supplies or requests an icon style. |
