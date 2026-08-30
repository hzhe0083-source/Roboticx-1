# VA Actual Implementation — Defect Log

## Screenshot Review Cycle 1
Evidence: `review_history/va-actual-cycle1-canvas.png` (1536×566 canvas export).

| id | zone | finding | severity | planned fix |
|---|---|---|---|---|
| C1-01 | color | Export background is transparent and renders black in the reviewer. | P0 | Set an explicit white graph background. |
| C1-02 | text | Main title has near-zero contrast on the black-rendered background. | P0 | Same root fix as C1-01. |
| C1-03 | text | Panel headings have near-zero contrast on the black-rendered background. | P0 | Same root fix as C1-01. |
| C1-04 | arrows | `K,V→Attention` crosses the visual residual rail without a bridge, implying a junction. | P0 | Add an arc jump at the crossing. |
| C1-05 | arrows | `A→K,V` crosses the visual residual rail without a bridge, implying a junction. | P0 | Add an arc jump at the crossing. |
| C1-06 | text | `Kᴸ,Vᴸ` is cramped against the panel boundary and long language bus. | P1 | Remove the redundant edge label. |
| C1-07 | text | Both `residual` edge labels sit directly on long rails and add noise. | P1 | Remove both labels; formulas already encode residuals. |
| C1-08 | text | `collect V₁…N` is squeezed between the central and output panels. | P1 | Remove it; VisualMemory already lists the layers. |
| C1-09 | text | `after layer N` is squeezed on the action-output connector. | P1 | Remove it; the LayerNorm box already names A_N. |
| C1-10 | text | `condition` is too close to the LayerNorm and Flow boxes. | P1 | Remove it; direction and formulas are sufficient. |
| C1-11 | text | `velocity` is too close to the Flow and Euler boxes. | P1 | Remove it; Euler formula names vθ. |
| C1-12 | boxes | V0, c_t, and A0 token boxes are visibly cramped. | P1 | Increase widths from 40 to 45 px and keep even gaps. |
| C1-13 | boxes | Euler block is narrow enough that the title/formula feel compressed. | P1 | Widen to 135 px. |
| C1-14 | boxes | Flow head formula is dense relative to its width. | P1 | Widen to 240 px and shift within the output panel. |
| C1-15 | semantics | `sg(Z)` is duplicated in the generic K/V source strip and the WM card. | P1 | Keep local sources in the strip; put the complete WM publication chain in the blue card. |
| C1-16 | text | Stream explanatory note repeats what the two lanes already show. | P1 | Delete it. |
| C1-17 | spacing | The title sits close to the top canvas boundary. | P2 | Retain unless the next export still feels crowded. |
| C1-18 | spacing | Initialization rows intentionally have unequal gaps around action-token construction. | P2 | Accept; the gap separates the two-step state→query construction. |
| C1-19 | spacing | Language→K/V uses a long vertical rail in the inter-panel gutter. | P2 | Accept if it stays clear after label removal. |
| C1-20 | arrows | V0→visual-stream uses two bends instead of a straight line. | P2 | Accept; it prevents overlap with the language rail. |
| C1-21 | arrows | A0→action-stream uses two bends instead of a straight line. | P2 | Accept for symmetry with V0. |
| C1-22 | boxes | VisualMemory is wider than its two text lines require. | P2 | Keep; it balances the output panel width. |
| C1-23 | layout | The output panel has a large gap between VisualMemory and the flow row. | P2 | Keep; it visually separates memory from action generation. |
| C1-24 | layout | Action output turns downward at Euler instead of staying fully horizontal. | P2 | Keep; the short turn avoids a wider canvas. |
| C1-25 | typography | Branch formulas use 13 pt while main blocks use 15 pt. | P2 | Keep for the two exact residual equations. |
| C1-26 | typography | Footnote is smaller than edge labels. | P2 | Keep as non-primary implementation scope. |
| C1-27 | color | Pink is used for both inputs and flow state. | P2 | Keep; both are externally supplied token/state values. |
| C1-28 | style | Dashed containers are taller than the central computational path. | P2 | Keep to align the three regions. |
| C1-29 | composition | Central panel is denser than initialization/output panels. | P2 | Keep; it is the requested implementation focus. |
| C1-30 | style | Main title says “actual implementation,” which is more explanatory than the reference. | P2 | Shorten to “internal structure.” |

Pre-flight warnings reviewed: the five spacing warnings compare unrelated shapes from different panels or intentional two-step input rows; none indicates a collision or clipping defect.

## User Structural Correction
- The user identified that the detailed Q/K/V + Attention + residual/FFN body is already the separate **VA Component** figure.
- Correction: the VA internal-structure figure now treats `VA Component × N` as an atomic repeated module and shows only token initialization, memory output, and flow-action generation.
- The previous Cycle 1/2 overview layout is superseded; its screenshots remain only as review evidence.

## Simplified Overview — Screenshot Review Cycle 1
Evidence: `review_history/va-overview-cycle1-canvas.png` (canvas-only export).

| id | zone | finding | severity | action |
|---|---|---|---|---|
| O1-01 | requirement | The note below the stack still enumerates Q/K/V, Attention, residuals, FFN, and WM injection. | P1 | Delete it completely. |
| O1-02 | text | `V0` edge label repeats the projection-box output. | P1 | Remove label. |
| O1-03 | text | `A0` edge label repeats the action-init formula. | P1 | Remove label. |
| O1-04 | text | `L` edge label repeats the language row. | P1 | Remove label. |
| O1-05 | text | `V1…N` label repeats the VisualMemory contents. | P1 | Remove label. |
| O1-06 | text | `A_N` edge label repeats the LayerNorm box. | P1 | Remove label. |
| O1-07 | semantics | Flow-head formula conditions on `A_N`, while code passes `LayerNorm(A_N)`. | P1 | Change formula to `LN(A_N)`. |
| O1-08 | text | Action-token formula wraps into four visually dense lines. | P1 | Shorten title and formula. |
| O1-09 | typography | `V/A coupling stack` uses the same 22 pt size as the block title. | P1 | Make it a 16 pt subtitle. |
| O1-10 | text | Main title is longer than necessary but remains readable. | P2 | Keep; it distinguishes this overview from the component figure. |
| O1-11 | text | Panel numbering is redundant with left-to-right flow. | P2 | Keep for presentation narration. |
| O1-12 | arrows | Vision→stack line is long but perfectly horizontal. | P2 | Accept. |
| O1-13 | arrows | Action→stack line is long but perfectly horizontal. | P2 | Accept. |
| O1-14 | arrows | Language→stack line is long but perfectly horizontal. | P2 | Accept. |
| O1-15 | arrows | Stack→VisualMemory uses two orthogonal bends. | P2 | Accept; it separates memory from the action path. |
| O1-16 | arrows | Stack→LayerNorm uses two orthogonal bends. | P2 | Accept; arrow direction is unambiguous. |
| O1-17 | arrows | Euler→Action chunk turns downward instead of continuing right. | P2 | Accept to avoid a wider canvas. |
| O1-18 | boxes | VA stack is visually dominant. | P2 | Accept; it is the subject of this overview. |
| O1-19 | boxes | VisualMemory box is wider than its two lines require. | P2 | Accept to balance the output region. |
| O1-20 | boxes | Language-projection box is 10 px taller than vision projection. | P2 | Accept for its extra cache line. |
| O1-21 | boxes | Action-token box is 20 px taller than vision projection. | P2 | Accept for the longer initialization formula. |
| O1-22 | spacing | Central stack is 10 px below the exact group center. | P2 | Accept; it aligns with all three input rows. |
| O1-23 | spacing | Memory and action paths leave a large vertical gap. | P2 | Accept; they are different output types. |
| O1-24 | spacing | Flow state sits above rather than inline with the action path. | P2 | Accept; it is an auxiliary flow input. |
| O1-25 | color | Inputs and flow state share pink. | P2 | Accept; both are externally supplied states. |
| O1-26 | color | VA stack uses attention orange despite being an atomic component. | P2 | Accept to link it visually to the component figure. |
| O1-27 | typography | Formula text uses 14–15 pt while headings use 18–22 pt. | P2 | Accept as deliberate hierarchy. |
| O1-28 | layout | No WM or previous-memory conditioning appears in this overview. | P2 | Intentional; those interfaces belong to the separate component figure. |
| O1-29 | icons | No icons are used. | P2 | Intentional; symbols and tensor names are sufficient. |
| O1-30 | style | Dashed group containers are taller than their densest content. | P2 | Accept for aligned three-panel framing. |

### Cycle 1 Fix Verification
| ids | result |
|---|---|
| O1-01 | FIXED — all component-detail enumeration removed from the overview. |
| O1-02…O1-06 | FIXED — five redundant tensor edge labels removed. |
| O1-07 | FIXED — Flow head now conditions on `LN(A_N)`. |
| O1-08 | FIXED — action-token title/formula shortened and remains readable. |
| O1-09 | FIXED — stack subtitle reduced from heading size to body size. |

## Simplified Overview — Screenshot Review Cycle 2
Evidence: `review_history/va-overview-cycle2-canvas.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| O2-01 | requirement | The block subtitle is no longer an internal formula, but it is still unnecessary explanatory text. | P1 | Remove subtitle entirely. |
| O2-02 | boxes | Atomic VA block is larger than needed after removing internal details. | P1 | Shrink from 300×220 to 280×180. |
| O2-03 | spacing | VisualMemory box can be 40 px narrower without wrapping. | P2 | Narrow and recenter. |
| O2-04 | spacing | Action chunk has only 10 px bottom margin. | P2 | Raise by 5 px. |
| O2-05 | text | Main title remains fully readable. | P2 | No change. |
| O2-06 | text | Action-token formula uses three lines but is readable. | P2 | No change. |
| O2-07 | text | Flow-head formula uses three lines and remains readable. | P2 | No change. |
| O2-08 | arrows | All three input arrows are now label-free and traceable. | P2 | No change. |
| O2-09 | arrows | Memory and action outputs are visually separated. | P2 | No change. |
| O2-10 | arrows | Flow-state arrow lands at the top-center of the Flow head. | P2 | No change. |
| O2-11 | arrows | Euler→Action arrow is short and unobstructed. | P2 | No change. |
| O2-12 | color | Palette matches the separate component diagram. | P2 | No change. |
| O2-13 | typography | Heading/body hierarchy is consistent after subtitle removal. | P2 | Verify next render. |
| O2-14 | layout | The central panel contains only one atomic module. | P2 | No change. |
| O2-15 | preflight | Two spacing warnings compare unrelated shapes in different panels. | P2 | Documented false positives; no geometry collision exists. |

### Cycle 2 Fix Verification
| ids | result |
|---|---|
| O2-01 | FIXED — subtitle removed; atomic block contains only `VA Component × N`. |
| O2-02 | FIXED — block reduced to 280×180 without text crowding. |
| O2-03 | FIXED — VisualMemory narrowed and recentered. |
| O2-04 | FIXED — Action chunk raised for a larger bottom margin. |

## Simplified Overview — Screenshot Review Cycle 3
Evidence: `review_history/va-overview-cycle3-canvas.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| O3-01 | arrows | VA→VisualMemory vertical bend nearly overlaps the output-panel dashed boundary. | P1 | Route both VA outputs through x=1060 in the inter-panel gutter. |
| O3-02 | arrows | VA→LayerNorm bend is not aligned with the memory-output bend. | P1 | Use the same x=1060 gutter lane. |
| O3-03 | boxes | Atomic VA block has generous padding around one label. | P2 | Accept; this is the requested abstraction boundary. |
| O3-04 | text | Flow-head formula remains readable at canvas scale. | P2 | No change. |
| O3-05 | arrows | Vision and language inputs make short final vertical jogs. | P2 | Accept; they land on distinct block ports. |
| O3-06 | spacing | VisualMemory and action paths remain clearly separated. | P2 | No change. |
| O3-07 | layout | Three panel widths are intentionally unequal. | P2 | Accept; the action head needs more horizontal room. |
| O3-08 | preflight | Remaining spacing WARNs compare shapes from separate regions. | P2 | Documented false positives. |

### Cycle 3 Fix Verification
| ids | result |
|---|---|
| O3-01 | FIXED — memory-output bend now uses x=1060, fully inside the inter-panel gutter. |
| O3-02 | FIXED — action-output bend uses the same gutter lane and no longer crowds the dashed border. |

## Screenshot Review Cycle 4 — Final Verification
Evidence: `review_history/va-overview-cycle4-canvas.png` and `va_actual_internal.png`.
- P0: 0.
- P1: 0.
- Component abstraction: PASS — only the atomic `VA Component × N` block remains.
- Text clipping/overlap: PASS.
- Arrow direction and landing: PASS.
- White-background export: PASS.

## Red-Team Audit
| id | zone | residual finding | severity | decision |
|---|---|---|---|---|
| RT-01 | text | Action-token formula is the densest label in the figure. | P2 | Accept; it is the only nontrivial initialization formula. |
| RT-02 | text | Superscripts in the language-cache label are small at thumbnail scale. | P2 | Accept; readable in the 1590 px export. |
| RT-03 | arrows | Vision and language inputs make a 19 px final jog. | P2 | Accept; distinct ports prevent false fan-in. |
| RT-04 | arrows | Euler output turns downward. | P2 | Accept; avoids widening the canvas. |
| RT-05 | boxes | Atomic VA block has deliberately generous padding. | P2 | Accept; reinforces the abstraction boundary. |
| RT-06 | spacing | Output panel is denser in its lower half. | P2 | Accept; upper memory path is intentionally separate. |
| RT-07 | color | No blue WM card appears in this overview. | P2 | Intentional; WM injection remains in the component figure. |
| RT-08 | typography | Formula body text is 14–15 pt versus 18–22 pt headings. | P2 | Accept as the chosen hierarchy. |
| RT-09 | layout | The three dashed regions use unequal widths. | P2 | Accept; output generation needs more width than initialization. |
| RT-10 | scope/style | Optional task/dense/dual-attention modes and VisualMemory world_state are not shown. | P2 | Intentional simplification; this is the core VA overview. |

## Self-Score
| dimension | score | evidence |
|---|---:|---|
| Text readability | 9/10 | All labels readable; action-token formula is slightly denser than peers. |
| Arrow accuracy | 10/10 | All 12 arrows have correct direction, target, and unobstructed routes. |
| Color coherence | 10/10 | Six Transformer-pastel roles plus consistent dark strokes. |
| Layout consistency | 9/10 | Grid is aligned; lower output path is intentionally denser. |
| Style match | 9/10 | Simple Transformer-like boxes and residual-free overview; horizontal format differs from the portrait reference. |
| **Total** | **47/50** | Allowed; no P0/P1 remains. |

## Symbolic Transformer Revision — Cycle 1
Evidence: `review_history/va-symbolic-cycle1-review.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| S1-01 | export | Action-chunk box is outside the captured viewport. | P0 | Raise output and render with SVG-height viewport. |
| S1-02 | export | Euler arrow ends at an apparently missing target. | P0 | Same fix as S1-01. |
| S1-03 | text | `repeated ×N` is prose-heavy. | P1 | Reduce to `×N`. |
| S1-04 | text | `h_t` edge label repeats the A0 equation. | P1 | Remove. |
| S1-05 | text | `Q_A` edge label repeats the A0 equation. | P1 | Remove. |
| S1-06 | text | `V0` edge label repeats S0. | P1 | Remove. |
| S1-07 | text | `A0` edge label repeats S0. | P1 | Remove. |
| S1-08 | text | `V1…VN` edge label repeats VisualMemory. | P1 | Remove. |
| S1-09 | text | `A_N` edge label repeats the output token. | P1 | Remove. |
| S1-10 | spacing | Flow head touches Euler at export scale. | P1 | Restore a 50 px gap. |
| S1-11 | arrows | VisualMemory target waypoint is off-center. | P1 | Center at the box top. |
| S1-12 | arrows | A_N target waypoint is off-center. | P1 | Center at the box top. |
| S1-13 | arrows | Language cache route approaches the S0 top edge. | P2 | Keep its own upper lane. |
| S1-14 | arrows | Visual-memory input uses a long side rail. | P2 | Accept; it shows per-layer reuse. |
| S1-15 | arrows | WM input uses a long blue side rail. | P2 | Accept; color distinguishes it. |
| S1-16 | layout | Context panel is narrower than initialization. | P2 | Accept; only two tensors live there. |
| S1-17 | layout | VA Component is the dominant block. | P2 | Accept; it is the intended abstraction. |
| S1-18 | layout | S0 sits directly on the component. | P2 | Accept; it reads as the component input port. |
| S1-19 | layout | S_N is separated from the component by a short gap. | P2 | Accept; it reads as the output port. |
| S1-20 | boxes | VisualMemory is wider than A_N. | P2 | Accept; it contains an entire layer stack. |
| S1-21 | boxes | Flow state is smaller than the flow head. | P2 | Accept; it is an auxiliary input. |
| S1-22 | color | DINO tokens and WM message share cyan. | P2 | Accept; both are visual/world representations. |
| S1-23 | color | V/A tokens share pink. | P2 | Accept; pink denotes VA token state. |
| S1-24 | color | Projections share lavender. | P2 | Accept; matches the Transformer palette role. |
| S1-25 | color | Action output uses pale yellow. | P2 | Accept; matches output/add semantics. |
| S1-26 | typography | Formula text is smaller than block titles. | P2 | Accept as hierarchy. |
| S1-27 | typography | Superscripts are small at thumbnail scale. | P2 | Accept at slide resolution. |
| S1-28 | semantics | No Q/K/V internals appear in this overview. | P2 | Intentional; Figure 2 owns those internals. |
| S1-29 | semantics | VisualMemory branches from the repeated block, not S_N. | P2 | Intentional; it collects every V_i. |
| S1-30 | style | No decorative icons are used. | P2 | Intentional symbolic style. |

## Symbolic Transformer Revision — Cycle 2
Evidence: `review_history/va-symbolic-cycle2-review.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| S2-01 | export | Full action chunk is now visible. | PASS | Keep. |
| S2-02 | text | All six redundant edge labels are gone. | PASS | Keep. |
| S2-03 | text | `×N` now matches Transformer-style repetition. | PASS | Keep. |
| S2-04 | spacing | Projection row triggers a false grouping warning. | P1 | Separate the language projection vertically. |
| S2-05 | spacing | Bottom VisualMemory aligns with the vision column and triggers a false vertical warning. | P1 | Shift the complete output row left. |
| S2-06 | spacing | Flow-state x aligns with language input and triggers a false warning. | P1 | Shift flow state left while preserving centering. |
| S2-07 | arrows | V0 and A0 reach distinct S0 ports. | PASS | Keep. |
| S2-08 | arrows | Language, memory, and WM reach three distinct component ports. | PASS | Keep. |
| S2-09 | arrows | Blue WM line never joins a black context line. | PASS | Keep. |
| S2-10 | semantics | A0 visibly equals learned queries plus projected robot history. | PASS | Keep. |
| S2-11 | semantics | Vision Projection visibly produces V0. | PASS | Keep. |
| S2-12 | semantics | S_N splits into memory and action-generation paths. | PASS | Keep. |
| S2-13 | layout | Bottom operator gaps are uniform. | PASS | Preserve after shifting. |
| S2-14 | color | Palette matches the accepted VA Component. | PASS | Keep. |
| S2-15 | scope | Figure contains no duplicated Q/K/V internals. | PASS | Keep. |

## Symbolic Transformer Revision — Cycle 3
Evidence: `review_history/va-symbolic-cycle4-review.png` (final source after the Cycle 3 geometry fixes).

| id | zone | finding | result |
|---|---|---|---|
| S3-01 | preflight | Arrow/box collision. | PASS — 0. |
| S3-02 | preflight | Spacing warning. | PASS — 0. |
| S3-03 | export | Clipped text or shape. | PASS — none. |
| S3-04 | arrows | Reversed or ambiguous arrow. | PASS — none. |
| S3-05 | semantics | VA Component internals duplicated. | PASS — no. |
| S3-06 | semantics | Robot history/action-query construction missing. | PASS — explicit. |
| S3-07 | color | Transformer role palette inconsistent. | PASS — consistent. |
| S3-08 | editability | Embedded raster or invalid XML. | PASS — none / valid. |

## Symbolic Revision — Final Red-Team
1. Inputs originate above their operators and outputs terminate below the model.
2. V0 comes only from DINO patches through Vision Projection.
3. A0 comes from learned action queries plus projected `[p_t ∥ u_{t−1}]`.
4. Language becomes a per-layer K/V cache.
5. Previous visual memory and WM message are distinct side inputs.
6. WM uses the blue path and never appears as a Q input in this overview.
7. `VA Component ×N` remains atomic; Figure 2 owns Q/K/V details.
8. VisualMemory collects `V1…VN`; it is not mislabeled as final V_N only.
9. A_N passes through LayerNorm and the flow head before Euler updates.
10. Flow state `(xτ,τ)` enters only the flow head.
11. All source/target IDs resolve; no raster image is embedded.
12. Real DrawIO render shows no clipping, overlap, false junction, or browser chrome.

## Symbolic Revision — Self-Score
| dimension | score |
|---|---:|
| Accuracy | 10/10 |
| Readability | 9/10 |
| Symbolic Transformer style | 9/10 |
| Color coherence | 10/10 |
| Editability | 10/10 |
| **Total** | **48/50** |

## User-Rejection Redesign — Cycle 1

Evidence: `review_history/va-redesign-cycle1.png`. The prior symbolic draft was rejected as visually uninviting, so this cycle starts a new composition rather than patching the old tower layout.

| id | finding | severity | disposition |
|---|---|---|---|
| R1-01 | Canvas is 1270 px tall and repeats the earlier long-slide problem. | P1 | Compress the vertical rhythm. |
| R1-02 | Top-to-projection gap is oversized. | P1 | Reduce from 39 px to 28 px. |
| R1-03 | Projection-to-token gap is oversized. | P1 | Reduce to a compact operator gap. |
| R1-04 | Token-to-`S0` merge consumes too much height. | P1 | Lift `S0`. |
| R1-05 | `S0`-to-component gap is too large. | P1 | Shorten the main spine. |
| R1-06 | Component height is visually cavernous for an atomic block. | P1 | Reduce to 88 px. |
| R1-07 | `N×` order is unlike Transformer notation. | P1 | Change to `×N`. |
| R1-08 | `F×` order is unlike Transformer notation. | P1 | Change to `×F`. |
| R1-09 | `K×` is buried in the Euler label. | P1 | Split it into `×K steps`. |
| R1-10 | VisualMemory uses token pink although it is a representation output. | P1 | Change to cyan. |
| R1-11 | VisualMemory branch has one unnecessary bend. | P1 | Shorten waypoints. |
| R1-12 | VisualMemory is too far below the repeated block. | P1 | Lift to the `A_N` row. |
| R1-13 | `S_N` is too far below the repeated block. | P1 | Lift to a 37 px gap. |
| R1-14 | `A_N` output is too far below `S_N`. | P1 | Tighten the output branch. |
| R1-15 | LayerNorm is too far below `A_N`. | P1 | Tighten to 31 px. |
| R1-16 | Flow head is too far below LayerNorm. | P1 | Tighten to 20 px. |
| R1-17 | Euler update is too far below the flow head. | P1 | Tighten while keeping arrowhead clearance. |
| R1-18 | Action chunk is too far below Euler. | P1 | Tighten to 30 px. |
| R1-19 | Right-side context cards are vertically loose. | P1 | Align tightly with the component ports. |
| R1-20 | Main action spine drifts horizontally at Euler. | P1 | Recenter the bottom operators. |
| R1-21 | Title-to-input whitespace is excessive. | P2 | Reduce 30 px. |
| R1-22 | Input cards are taller than their two-line content needs. | P2 | Reduce to 52 px. |
| R1-23 | Projection cards are taller than needed. | P2 | Reduce to 40 px. |
| R1-24 | Token cards are taller than needed. | P2 | Reduce to 46 px. |
| R1-25 | `S0` and `S_N` are taller than needed. | P2 | Reduce to 42 px. |
| R1-26 | Flow-state chip is slightly too low. | P2 | Recenter on the flow head. |
| R1-27 | `×N` sits too far from the block it modifies. | P2 | Move beside the component. |
| R1-28 | `×F` sits too far from the head it modifies. | P2 | Move beside the flow head. |
| R1-29 | Bottom output width is slightly heavy. | P2 | Reduce height while preserving width. |
| R1-30 | Overall composition reads as a tall workflow instead of a compact model. | P1 | Rebuild to 900×940. |

## User-Rejection Redesign — Cycle 2

Evidence: `review_history/va-redesign-cycle2.png`.

| id | check | result |
|---|---|---|
| R2-01 | Canvas reduced from 1007×1270 to 1007×1116. | PASS |
| R2-02 | Input, projection, token, component, and decoder gaps are visibly tighter. | PASS |
| R2-03 | `×N`, `×F`, and `×K steps` use one notation order. | PASS |
| R2-04 | VisualMemory is cyan and reads as an output representation. | PASS |
| R2-05 | VisualMemory branch no longer crosses text. | PASS |
| R2-06 | VA Component remains atomic; no Q/K/V internals are duplicated. | PASS |
| R2-07 | Local K,V and WM K,V reach distinct right ports. | PASS |
| R2-08 | Blue WM path never enters the action query construction. | PASS |
| R2-09 | Learned queries and projected robot state merge only at `A0`. | PASS |
| R2-10 | `V0` and `A0` merge only at `S0`. | PASS |
| R2-11 | VisualMemory and action decoding split cleanly after the repeated VA stack. | PASS |
| R2-12 | Main action spine is centered. | PASS |
| R2-13 | No clipping, text collision, or false junction appears. | PASS |
| R2-14 | Remaining label `Robot history` is less explicit than requested. | P1 — rename to `Robot state history`. |
| R2-15 | Final strict XML preflight has no structural finding. | PASS |

## User-Rejection Redesign — Cycle 3 Final

Evidence: `review_history/va-redesign-cycle3.png`.

| id | check | result |
|---|---|---|
| R3-01 | Robot-state source is explicit as `[p_t ∥ u_{t−1}]`. | PASS |
| R3-02 | Vision Projection visibly maps DINO patches to `V0`. | PASS |
| R3-03 | Action tokens visibly equal learned queries plus projected state history. | PASS |
| R3-04 | Repetition symbols are attached to their operators without extra containers. | PASS |
| R3-05 | Inputs enter from above and the action chunk exits below. | PASS |
| R3-06 | Real DrawIO export contains no browser chrome and fills the canvas. | PASS |
| R3-07 | Strict selected visual checks report 0 FAIL / 0 WARN; multi-column spacing was manually inspected because the generic variance rule false-groups independent rows. | PASS |
| R3-08 | Native DrawIO XML contains no raster or external image. | PASS |

## User-Rejection Redesign — Final Red-Team

1. No duplicated VA Component internals appear in the overview.
2. `Q_A` is learned and is not mislabeled as a physical action.
3. `u_{t−1}` appears only as part of robot-state history.
4. WM enters the VA stack only as a blue K,V context.
5. VisualMemory collects intermediate `V_i` outputs rather than action tokens.
6. The flow state `(x_τ,τ)` enters only the flow head.
7. Euler repetition is labeled as steps, not layers.
8. No edge crosses a node label or forms a false junction.
9. The canvas contains no outer workflow frame or decorative panel.
10. Official PNG and `.drawio` sources match the final reviewed render.

## User Feedback — Language Visibility

Evidence: user could not identify language in the VA overview.

| id | check | result |
|---|---|---|
| L-01 | `L` is an explicit `Language tokens L` input rather than an abbreviated combined card. | PASS |
| L-02 | Language and previous visual memory are distinct local K,V sources. | PASS |
| L-03 | The blue WM message remains a separate K,V source; it does not enter Q. | PASS |
