# Defect Log

# VA–WAM figure defect log

Reference: `/tmp/codex-clipboard-3859853e-ac46-44b3-941f-9f771689ccac.png`

## Pass 0 - Initial Plan Review

| issue | reference evidence | planned fix |
|---|---|---|
| Preserve Transformer-level hierarchy | reference expands attention, feed-forward and residual modules | expand VA and ST predictor internals rather than using VA/WAM monoliths |
| Preserve arrow grammar | reference uses upward orthogonal arrows and local residual loops | use one direction per connector and independent residual lanes |
| Preserve palette/style | reference uses gray ground and muted pink/sand/blue/olive/lavender/sage boxes | sample exact hex values and use 3 px black rounded outlines |

## Screenshot Review

The three canvas-only review passes below record the complete inventory, patch, and verification chain.

## Cycle 1 — 2026-08-20

Canvas-only artifact: `review-cycle-1-canvas.png` (1200 × 1775).

| # | Zone | Priority | Finding | Resolution |
|---:|---|---|---|---|
| 1 | Stage merge | P0 | `Gated ΔV, ΔA → Gated Merge` approaches the merge from an unintuitive above/below route; arrowhead is hard to assign. | Open |
| 2 | World branch | P0 | `Predicted World Tokens → World State` and `→ Predicted Next DINO Map` leave from the same right-side port and overlap initially. | Open |
| 3 | WAM residuals | P0 | Three residual shortcuts occupy one continuous vertical trunk, so the three source–target pairs are not visually separable. | Open |
| 4 | VA residuals | P0 | Two residual shortcuts occupy one continuous vertical trunk and read as one long feedback path. | Open |
| 5 | Training branch | P0 | The training-only target edge renders almost solid at this scale, weakening the “not forward input” distinction. | Open |
| 6 | Stage semantics | P1 | `Gated Merge → Stage Commit` and `World State → Stage Commit` both enter the lower edge without labels; their roles are not distinguished. | Open |
| 7 | World branch | P1 | The map edge makes a long perimeter detour before reaching the map, making its source easy to lose. | Open |
| 8 | WAM state | P1 | The `Predicted World Tokens → World State` route passes near the map route and the WAM frame. | Open |
| 9 | WAM output | P1 | `Gated World Mixer → Gated ΔV, ΔA` is short and partially hidden by neighboring stage-level lines. | Open |
| 10 | VA output | P1 | `VA Proposal → Gated Merge` terminates near the box corner; the arrowhead competes with the container outline. | Open |
| 11 | Stage output | P1 | The `after ×8: A₈` label sits on the peer-stage top boundary, visually mixing the edge and container. | Open |
| 12 | WAM input | P1 | The ten-pixel gap from shared snapshot to evidence cross-attention compresses two arrowheads into one crowded joint. | Open |
| 13 | WAM input | P1 | Snapshot-to-evidence and evidence-to-belief share the same center axis with almost no separation. | Open |
| 14 | ST input | P1 | Belief update to first Pre-Norm is visually crossed by the first residual lane near the bottom of the ST frame. | Open |
| 15 | ST condition | P1 | Condition K/V edge has two arrow-like tips at the cross-attention boundary because the source and target borders are too close. | Open |
| 16 | ST output | P1 | Last residual add to predicted tokens is close to the ST frame title region and could be read as a frame connector. | Open |
| 17 | World loss | P1 | Map-to-loss arrow is clear, but the target-to-loss arrow aligns on the loss center and makes the loss look like a forward merge without a legend. | Open |
| 18 | Main output | P1 | The map branch sits adjacent to the action head chain without a visual separator or branch heading. | Open |
| 19 | VA main chain | P2 | Main-chain arrowheads vary in apparent size because of different gap lengths. | Open |
| 20 | WAM main chain | P2 | Main-chain arrows use the same weight as residual arrows; hierarchy is weak. | Open |
| 21 | Containers | P2 | Peer-stage top border is interrupted by the action output edge and label. | Open |
| 22 | Containers | P2 | WAM top border passes behind stage-level paths, reducing enclosure clarity. | Open |
| 23 | Containers | P2 | ST frame is close to the left residual trunk and creates an extra apparent parallel line. | Open |
| 24 | Input fan-in | P2 | The three embedding-to-snapshot arrows meet the snapshot at different visible angles due to auto-routing. | Open |
| 25 | Input fan-in | P2 | The language path bends before the snapshot while the visual path is straight; alignment feels accidental. | Open |
| 26 | Input fan-in | P2 | The state path has a longer horizontal segment than the other two, reducing symmetry. | Open |
| 27 | Labels | P2 | Repeated “residual” labels sit on the shortcut lanes and make the arrows feel busier than the reference. | Open |
| 28 | Labels | P2 | `training only / stop-grad` is detached from the dashed edge and slightly too far left. | Open |
| 29 | Labels | P2 | `Condition K/V` subtext is close to the arrow line and visually cramped. | Open |
| 30 | Style | P2 | Arrowheads are smaller than the reference at whole-figure scale. | Open |
| 31 | Style | P2 | Long world-branch orthogonal segments have more bends than the reference diagram. | Open |
| 32 | Style | P2 | Residual paths lack the compact U-shape used by the reference diagram. | Open |

Cycle-1 resolution note: items 1–5, 7–16, 19–20, 22–23, and 27–32 were addressed before cycle 2. Items 6, 17–18, 21, and 24–26 were rechecked in the second render.

## Cycle 2 — 2026-08-20

Canvas-only artifact: `review-cycle-2-canvas.png` (1200 × 1780).

| # | Zone | Priority | Finding | Resolution |
|---:|---|---|---|---|
| 1 | World semantics | P0 | The corrected map→tokens direction is now right, but map/state/loss outputs still form a dense right-side bus. | Fixed with explicit `World State Update` and separated lanes. |
| 2 | World state | P0 | Two arrowheads enter `World Stateᵢ` without an intermediate update operation, making belief/map persistence look like two unrelated writes. | Fixed with `World State Update`. |
| 3 | WAM input | P0 | Belief and DINO token boxes are ordered visually as one serial stream even though they are parallel branches. | Fixed by swapping them into distinct left/right branches. |
| 4 | ST residual 1 | P0 | First residual lane and token-stream input share the predictor’s right edge. | Fixed by moving the first residual to the left lane. |
| 5 | ST residual 2 | P0 | Second residual reuses the same left lane as its neighbors. | Fixed by alternating to the right lane. |
| 6 | ST residual 3 | P0 | Third residual is visually continuous with the second residual. | Fixed by alternating back to the left lane. |
| 7 | Edge labels | P1 | `belief + innovation` overlaps the Condition K/V box and reads like part of its contents. | Fixed by moving semantics into the state-update box. |
| 8 | Edge labels | P1 | `latent writeback` sits on the WAM title/border area. | Fixed by removing the redundant edge label. |
| 9 | Edge labels | P1 | `conditioning` sits far from the target and cannot be assigned quickly. | Fixed by removing it; the Condition K/V box carries the meaning. |
| 10 | Edge labels | P1 | `token stream` collides with the belief branch near the predictor input. | Fixed by removing it. |
| 11 | Loss branch | P1 | `supervised at all 8 stages` lies on the vertical loss lane. | Fixed by moving the statement into `World Loss`. |
| 12 | Stage title | P1 | Long peer-stage title is crossed by the action-head arrow. | Fixed by shortening the title. |
| 13 | State branch | P1 | Map-to-state and belief-to-state use nearly coincident vertical lanes. | Fixed with one state-update output edge. |
| 14 | Condition branch | P1 | Belief-to-condition route travels around the right perimeter although the belief box can be aligned with Condition K/V. | Fixed by moving the belief branch to the right column. |
| 15 | ST input | P1 | DINO clip input is on the right, forcing its main token path to bend across the predictor center. | Fixed by moving DINO input to the left branch. |
| 16 | World loss | P2 | The loss line initially intersects the new state-update box. | Fixed by routing below the state-update box, then upward on a dedicated lane. |
| 17 | WAM hierarchy | P2 | World map/tokens/mixer boxes are wider than the ST core and leave no state-update column. | Fixed by reducing them to the ST core width. |
| 18 | World-state label | P2 | `World Stateᵢ` does not identify itself as the recurrent state. | Fixed with a small `recurrent` subtitle. |

## Cycle 3 — 2026-08-20

Canvas-only evidence: `review-cycle-3-canvas.png`

| # | Zone | Priority | Finding | Verification in final canvas |
|---:|---|---|---|---|
| 1 | Auxiliary loss | P1 | The solid predicted-map→loss edge still looked like a forward-time data path. | Fixed: both loss inputs are dark-red dashed training-only edges. |
| 2 | Training label | P1 | The detached `training only / stop-grad` label was clipped to the word “grad”. | Fixed: text moved inside `Future DINO Target`. |
| 3 | Predictor title | P1 | The central output arrow crossed the long `ST World Predictor Block ×6` title. | Fixed: shortened to `ST Predictor ×6`. |
| 4 | State update | P1 | State-update subtext wrapped too tightly in a 115-pixel box. | Fixed: label shortened and box widened to 130 pixels. |
| 5 | Condition input | P1 | Condition K/V repeated “belief” and was too dense. | Fixed: concise three-line contents in a wider box. |
| 6 | WAM output | P1 | `Gated ΔV, ΔA` could be mistaken for executable action deltas. | Fixed: renamed `Gated Latent ΔV, ΔA`. |
| 7 | Stage title | P1 | The long stage title approached the action-head edge and the ellipsis rendered poorly. | Fixed: `Peer-Synchronous Stage ×8`. |
| 8 | State arrow | P0 | The last eight pixels of belief→State Update auto-routed horizontally, so its arrowhead pointed left instead of upward. | Fixed: final waypoint locked to the target bottom-center; final arrow is visibly upward. |
| 9 | Condition arrow | P0 | Belief→Condition K/V had the same final-segment auto-routing defect. | Fixed: final waypoint locked to the target bottom-center; final arrow is visibly upward. |
| 10 | World branch | P1 | The map-loss lane crossed the state-update box before the second routing pass. | Fixed: routed below State Update, then upward on a dedicated dashed lane. |

Cycle-3 verification: all P0/P1 items above are visibly fixed in `va_wam_main_v3.png`; no new overlap or wrong-direction regression was found.

## Screenshot Evidence

| pass | screenshot path | capture type | full canvas visible | crop/viewport notes |
|---|---|---|---|---|
| Cycle 1 | `review-cycle-1-canvas.png` | canvas-only | yes | 1200 × 1775 crop from the 1400 × 1900 editor capture |
| Cycle 2 | `review-cycle-2-canvas.png` | canvas-only | yes | 1200 × 1780 crop from the 1400 × 1950 editor capture |
| Cycle 3 | `review-cycle-3-canvas.png` | canvas-only | yes | 1200 × 1780 crop from the 1400 × 1950 editor capture |
| Final | `va_wam_main_v3.png` | canvas-only | yes | clean 1200 × 1780 deliverable with no browser/editor chrome |

## Requirement And Semantic Audit

| check | observed screenshot | expected from code/reference | actual | status |
|---|---|---|---|---|
| same snapshot | one shared box fans into VA, WAM evidence, and DINO clip paths | peer-sync VA and WAM read the same pre-stage snapshot | no VA→WAM serial edge exists | pass |
| VA internals | two expanded residual sublayers | Pre-Norm→shared MHA→add; Pre-Norm→FFN→add | exact two-sublayer chain | pass |
| predictor internals | three expanded residual sublayers | causal self-attn, conditional cross-attn, FFN | exact three-sublayer chain | pass |
| map direction | ST Predictor→Predicted DINO Map→World Tokens | map must precede token projection | arrows point upward in that order | pass |
| recurrent state | map and belief/innovation enter State Update | World State contains belief, innovation, map | one recurrent World State output | pass |
| latent writeback | mixer→Gated Latent ΔV,ΔA→Gated Merge | WAM modifies VA latent streams, not robot action directly | separate latent-delta box and merge | pass |
| unique action head | A₈ Layer Norm→Flow×6→Euler×8→H6×4 | WAM is not a second action head | only one sage output box | pass |
| loss isolation | map and future target enter World Loss on dashed edges | future target is training-only and stop-gradient | no dashed line enters runtime modules | pass |
| reference style | gray background, rounded black outlines, muted fills | match supplied Transformer visual family | exact sampled palette and orthogonal arrows | pass |

## Red-Team Visual Audit

Evidence: `va_wam_main_v3.png` (canvas-only, 1200 × 1780). All nine zones were rescanned after the three cycles.

| # | Zone | Residual finding | Severity | Disposition |
|---:|---|---|---|---|
| 1 | Text | The 11-pixel Condition K/V subtext is the smallest text in the figure. | P2 | Accepted; readable at 100%, needed to keep the side input compact. |
| 2 | Text | The predicted-map tensor-size subtitle is visually subordinate and requires normal-size viewing. | P2 | Accepted; it is metadata, not a flow label. |
| 3 | Text | Snapshot tuple text is smaller than the three input labels. | P2 | Accepted; hierarchy is intentional. |
| 4 | Text | `spatial projection / flatten` is close to the lower border of World Tokens. | P2 | Accepted; screenshot confirms no clipping. |
| 5 | Arrows | The dashed predicted-map→loss path is long. | P2 | Accepted; its perimeter lane prevents crossings and clearly marks training-only flow. |
| 6 | Arrows | Belief→State Update uses a long right-side lane. | P2 | Accepted; it is distinct from the dashed loss lane and ends with an upward arrow. |
| 7 | Arrows | Stage Commit→Layer Norm crosses the peer-stage boundary. | P2 | Accepted; this boundary crossing intentionally marks the single action-head exit. |
| 8 | Arrows | The three embedding fan-in paths are not equal in horizontal length. | P2 | Accepted; their three target ports are symmetric and arrowheads remain clear. |
| 9 | Boxes | Condition K/V is narrower than the predictor blocks. | P2 | Accepted; it is a side input, not a main-chain block. |
| 10 | Boxes | State Update is compact compared with World State. | P2 | Accepted; this preserves the map→update horizontal arrow and avoids a wider right column. |
| 11 | Spacing | WAM is denser than VA because it contains three residual sublayers plus state/loss branches. | P2 | Accepted; density follows the code rather than decorative symmetry. |
| 12 | Spacing | The top action chain has 15-pixel gaps while the stage-to-head gap is larger. | P2 | Accepted; the larger gap separates repeated backbone from the single emitter. |
| 13 | Color | Dark red `#6F5656` is one auxiliary color not present in the reference palette. | P2 | Accepted; it is reserved only for training-only dashed edges. |
| 14 | Typography | Secondary annotations use 11–12 px while the reference has fewer annotation levels. | P2 | Accepted; all primary module names remain 16–24 px. |
| 15 | Layout | The right edge is visually heavier because recurrent state and loss both originate from WAM. | P2 | Accepted; solid and dashed lanes separate runtime state from training supervision. |
| 16 | Icons | No icon marks stop-gradient; the distinction relies on pink fill, subtitle, and dashed edges. | P2 | Accepted; the supplied reference is predominantly box-and-arrow, so an extra icon would reduce stylistic fidelity. |
| 17 | Style | The final diagram is denser than the reference Transformer figure. | P2 | Accepted; finer hierarchy was explicitly requested and no module can be removed without hiding VA/WAM logic. |
| 18 | Text | `World Loss / all 8 stages` uses two hierarchy levels inside a small box. | P2 | Accepted; both lines are readable and keep supervision local. |
| 19 | Text | Unicode subscripts sit slightly lower than surrounding Latin letters in the browser font. | P2 | Accepted; the meaning is clearer than spelling every index inline. |
| 20 | Text | `H6 × 4` is project notation rather than a fully spelled-out tensor shape. | P2 | Accepted; it matches the active configuration and avoids an oversized output label. |
| 21 | Arrows | Main-chain and residual arrows share the same 3 px stroke weight. | P2 | Accepted; route topology and local U-shapes provide the hierarchy, matching the reference grammar. |
| 22 | Arrows | The map→State Update arrow has only a 25 px horizontal gap. | P2 | Accepted; the arrowhead remains fully visible and the short gap emphasizes direct projection. |
| 23 | Arrows | The Future Target→World Loss dashed edge has fewer dash cycles than the longer map-loss route. | P2 | Accepted; identical dash pattern is used and the short physical distance explains the count. |
| 24 | Boxes | Future DINO Target is narrower than the action-output box. | P2 | Accepted; it is an auxiliary target, not a peer output. |
| 25 | Boxes | The VA layer is shorter than the WAM stage. | P2 | Accepted; VA has two sublayers while WAM contains a three-sublayer predictor plus state logic. |
| 26 | Spacing | The State Update side column is 25 px from the predicted map but 40 px from the predictor frame. | P2 | Accepted; the closer gap belongs to its direct source. |
| 27 | Color | Pink is reused for both raw inputs and the future training target. | P2 | Accepted; both denote externally supplied data, while the dashed edge distinguishes training use. |
| 28 | Typography | Container titles are bold while repeated-count suffixes share the same weight. | P2 | Accepted; this matches the bold section-title treatment in the reference. |
| 29 | Layout | The auxiliary loss occupies the upper-right rather than aligning to the centered action head. | P2 | Accepted; spatial separation prevents it from being read as a second action emitter. |
| 30 | Style | Pre-Norm and Residual Add are separate boxes rather than one `Add & Norm` box as in the reference. | P2 | Accepted; this is the code-accurate pre-norm structure requested by the user. |

No red-team P0 or P1 remains.

## Preflight warning disposition

Final static check: 0 FAIL, 6 WARN.

- Four spacing warnings come from intentional alternating Transformer geometry (`Pre-Norm`/large operator/`Residual Add`) and the separated action head.
- One horizontal-spacing warning reflects VA/WAM width asymmetry required by the additional WAM state column.
- One edge-density warning is the expected vertical centerline through the three-layer ST Predictor; the canvas review confirms no arrow passes through text or a box.

## Self-score — pre-handoff

| Dimension | Score | Concrete deduction evidence |
|---|---:|---|
| Text readability | 8/10 | Four secondary annotations are 11–12 px; readable at 100% but smaller than the reference’s primary labels. |
| Arrow accuracy | 9/10 | Every runtime and training arrow has a verified source/target; one point deducted for the long, though separated, right-side auxiliary lanes. |
| Color coherence | 9/10 | Exact sampled reference palette is used; one point deducted for the necessary dark-red training-only connector color. |
| Layout consistency | 9/10 | Grid alignment and repeated-layer rhythm are consistent; one point deducted because WAM is inherently denser than VA. |
| Style match to reference/spec | 9/10 | Rounded black boxes, gray ground, muted fills and upward flow match; one point deducted for the extra annotation depth requested by the user. |
| **TOTAL** | **44/50** | **Allowed: all dimensions ≥ 6 and total ≥ 40.** |

## Self-Score Gate

| Dimension | Score |
|---|---:|
| Text readability | 8 |
| Arrow accuracy | 9 |
| Color coherence | 9 |
| Layout consistency | 9 |
| Style match to reference/spec | 9 |
| TOTAL | 44 |

## Remaining Gaps

- Only P2 tradeoffs listed in the red-team table remain; there are no known wrong-direction, missing-source, missing-target, overlap, or clipped-text defects.

## User-found semantic defect and repair — 2026-08-20

The user correctly identified that the earlier ImageGen raster and the editable Draw.io file had drifted into different architectures, and asked what the WM world state actually represents and how it is trained. This exposed a self-supervision failure: the previous figure used the generic label `World Stateᵢ` without separating direct map supervision from unlabeled recurrent latents.

Lesson recorded: a generative raster is never an independent architecture authority. For this project, code-grounded Draw.io XML is the sole semantic source; the clean PNG must be exported from the same XML. World-state figures must explicitly distinguish `world_map` direct supervision from `belief/innovation` indirect policy training.

### Revision cycle 4 — semantic expansion

Canvas evidence: `review-revision-4-canvas.png`; full editor evidence: `review-revision-4-full.png`.

| Zone | Priority | Finding | Repair |
|---|---|---|---|
| architecture consistency | P0 | Old ImageGen raster disagreed with the peer-synchronous code topology. | Deprecated the raster as a draft; retained Draw.io as the single source for both editable and raster output. |
| state meaning | P0 | `World Stateᵢ` hid three different tensors. | Expanded it to `Wᵢ={Bᵢ,Iᵢ,Zᵢ}` with active shapes `[8×512]`, `[8×512]`, and `[1024×16×16]`. |
| state training | P0 | Loss branch could imply all state fields had future labels. | Added an explicit note: `Z` direct WAM objective; `B/I` no state targets and learn through the gated path to `Lflow`. |
| stage recurrence | P1 | `×8` text did not show how state advances. | Added `Stage Commit Sᵢ → next pre-stage snapshot` recurrence. |
| predictor conditioning | P1 | Belief conditioning did not disclose stop-gradient. | Changed Condition K/V to `sg(Bᵢ)` and kept DINO clip as the token main path. |

### Revision cycle 5 — arrow cleanup

Canvas evidence: `review-revision-5-canvas.png`; focused evidence: `review-revision-5-wam-focus.png`.

| Zone | Priority | Finding | Repair |
|---|---|---|---|
| repeat loop | P1 | The first recurrence route crossed the stage-title band. | Moved the horizontal segment into the clear gap between Stage Commit and Gated Merge. |
| action output | P1 | `A₈ only` edge text crowded the action spine. | Removed the edge label and moved the restriction into `Layer Norm (A₈ only)`. |
| direct supervision | P1 | A long loss-edge label was clipped at the right canvas edge. | Removed the redundant label; dark-red dashed topology and objective text now carry the meaning. |
| VA residuals | P1 | Two VA residuals shared the same visual lane. | Moved the second VA residual to the opposite side. |

### Revision cycle 6 — focused WAM arrow audit

Canvas evidence: `review-revision-6-canvas.png`; focused evidence: `review-revision-6-wam-focus.png`.

| Zone | Priority | Finding | Repair / verification |
|---|---|---|---|
| conditional attention | P0 | Condition K/V input crossed the second ST residual lane and looked like a junction. | Put all three ST residuals on staggered left lanes; Condition K/V now enters Conditional Cross-Attention on an isolated right-to-left arrow. |
| recurrent packing | P1 | The old state box did not show which values are packed. | `Zᵢ` and `Bᵢ,Iᵢ` now have separate labeled inputs into `Pack Wᵢ`. |
| direct loss | P1 | Predicted-map loss path competed with the state-pack path. | Kept state packing solid and short; moved supervision to a dedicated far-right dark-red dashed lane. |
| target comparison | P2 | Future target and objective were too close for a visible arrowhead. | Increased their vertical separation before final export. |

Revision-6 focused verification: all module-to-module arrows have one visible target arrowhead; no residual line intersects the conditional-attention input; no loss edge enters belief or innovation; the only executable-action output remains `A₈ → Flow → Euler → H6×4`.

### Revision cycle 7–8 — 100% export and red-team pass

Final canvas evidence: `review-revision-8-canvas.png` (1200 × 1780). Focused evidence: `review-revision-8-wam-focus.png`.

The 100% export exposed one additional P1: the mixer-to-delta arrow occupied the WAM title band. Its source/target ports were moved rightward, so the arrow now rises in a dedicated lane between the title and recurrent-state column.

Fresh red-team scan of all nine zones:

| # | Zone | Residual finding | Severity | Disposition |
|---:|---|---|---|---|
| 1 | text | Objective equations use 11 px annotations. | P2 | Accepted; primary architecture labels remain 16–24 px and the formula is supplementary. |
| 2 | text | `Bᵢ,Iᵢ` edge label is intentionally compact. | P2 | Accepted; the expanded `Wᵢ` box repeats the semantics at normal reading size. |
| 3 | arrows | The predicted-map supervision route is long. | P2 | Accepted; the far-right lane prevents any runtime-path crossing. |
| 4 | arrows | The stage recurrence loop is also long. | P2 | Accepted; its black dashed style and left perimeter isolate it from red training edges. |
| 5 | arrows | Three ST residuals use staggered adjacent left lanes. | P2 | Accepted; each has its own arrowhead at the correct Residual Add and no condition-edge crossing remains. |
| 6 | arrows | `Wᵢ → Stage Commit` has one horizontal bend. | P2 | Accepted; it is the shortest non-overlapping route and ends upward at the commit box. |
| 7 | boxes | The WAM objective overlaps the peer-stage boundary. | P2 | Accepted; this deliberately marks the objective as training-side rather than a runtime module. |
| 8 | boxes | State-learning note has a dashed outline unlike operator boxes. | P2 | Accepted; it is explanatory metadata, not a forward operator. |
| 9 | spacing | WAM is denser than VA. | P2 | Accepted; WAM contains an extra residual sublayer, state pack, recurrence, and supervision. |
| 10 | spacing | World Tokens and predicted map have a five-pixel logical gap before scaling. | P2 | Accepted; rendered outlines and arrowhead remain fully separated at 100%. |
| 11 | color | Dark red is outside the sampled six-color fill palette. | P2 | Accepted; used only for training-only comparison edges and the explanatory note outline. |
| 12 | typography | Unicode subscripts vary slightly in baseline. | P2 | Accepted; tensor identity is clearer than verbose inline names. |
| 13 | layout | Auxiliary training boxes make the upper-right heavier. | P2 | Accepted; this keeps the unique action emitter centered and prevents a false second action head. |
| 14 | style | Pre-Norm and Residual Add remain separate rather than reference-style `Add & Norm`. | P2 | Accepted; the implementation is pre-norm, so separation is code-accurate. |
| 15 | style coherence | Recurrence uses black dashed while forward uses black solid. | P2 | Accepted; this is a deliberate third grammar distinct from red training supervision. |

No red-team P0/P1 remains in the 100% canvas or WAM-focused crop.

### Final self-score after user correction

| Dimension | Score | Evidence |
|---|---:|---|
| Text readability | 9/10 | All primary labels are readable at 100%; only formulas and tensor metadata use 11–12 px. |
| Arrow accuracy | 9/10 | Every forward, recurrence, and supervision edge has a verified source and target; two perimeter routes remain intentionally long. |
| Color coherence | 9/10 | Reference fills are preserved; dark red is reserved for training semantics. |
| Layout consistency | 9/10 | VA and WAM have aligned upward spines and local residual loops; WAM is necessarily denser. |
| Style match | 9/10 | Muted palette, rounded black boxes, orthogonal arrows, and Transformer-level hierarchy match the reference family. |
| **TOTAL** | **45/50** | **Allowed; every dimension ≥ 6 and no P0/P1 remains.** |
