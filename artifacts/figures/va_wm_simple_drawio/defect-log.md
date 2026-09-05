# Defect Log

## Pass 0 — Initial Plan Review
| issue | source evidence | planned fix |
|---|---|---|
| Dense full-system content would conflict with the user's simplification request. | latest user feedback | restrict each figure to one principle |
| AI residual paths can be visually ambiguous. | VA AI sketch | encode exact source/target anchors in XML |
| Current `Z_i` must not feed current `VA_i`. | repository peer-sync semantics | send `Z_i` only to commit; blue return uses `Z_{i−1}` |

## Screenshot Evidence
| pass | screenshot path | capture type | full canvas visible | notes |
|---|---|---|---|---|
| Cycle 1 | `review_history/cycle-1-va-canvas.png` | canvas-only | yes | first editable VA render |
| Cycle 1 | `review_history/cycle-1-interaction-canvas.png` | canvas-only | yes | first editable interaction render |
| Cycle 2 | `review_history/cycle-2-va-canvas.png` | cropped canvas-only | yes | recaptured after removing DrawIO chrome |
| Cycle 2 | `review_history/cycle-2-interaction-canvas.png` | cropped canvas-only | yes | recaptured after removing DrawIO chrome |
| Cycle 3 | `review_history/cycle-3-va-canvas.png` | cropped canvas-only | yes | final-candidate VA render |
| Cycle 3 | `review_history/cycle-3-interaction-canvas.png` | cropped canvas-only | yes | final-candidate interaction render |
| Final verify | `va_internal_simple_landscape.png` | cropped canvas-only | yes | 1570×564 px; no UI; exact graph bounds |
| Final verify | `va_wm_interaction_simple_landscape.png` | cropped canvas-only | yes | 1550×745 px; blue entry aligned to K,V line |

## Screenshot Review
- Cycle 1 completed with 34 findings; all P0/P1 items were assigned a source-level fix before Cycle 2.

### Cycle 1 Inventory — 34 Findings
| id | priority | figure | finding | Cycle 2 action |
|---|---|---|---|---|
| C1-01 | P0 | VA | Five K/V chips appeared to float instead of fan into attention. | Replace with one connected K,V source group. |
| C1-02 | P0 | VA | Q existed only as prose, not as a visible path. | Label the input-to-attention edge `Q`. |
| C1-03 | P0 | VA | K,V existed only as prose, not as a visible path. | Label the source-to-attention edge `K,V`. |
| C1-04 | P0 | Interaction | Blue edge connected current `WM_i` directly to current `VA_i`. | Source it from a previous-snapshot WM message node. |
| C1-05 | P0 | Interaction | Blue route visually began near current `Z_i`, implying same-stage feedback. | Isolate `sg(Z_{i-1}) → Proj` in the old-state node. |
| C1-06 | P0 | Interaction | Orange edge connected current `VA_i` directly to current `WM_i`. | Source it from VA state stored in `S_{i-1}`. |
| C1-07 | P1 | VA | Input box contained excessive empty space. | Reduce height. |
| C1-08 | P1 | VA | Attention box contained excessive empty space. | Reduce height and remove repeated source prose. |
| C1-09 | P1 | VA | Output box contained excessive empty space. | Reduce height and match input dimensions. |
| C1-10 | P1 | VA | `V` could mean Vision or attention Value. | Use full source names; reserve edge `K,V` for attention roles. |
| C1-11 | P1 | VA | `M`, `L`, and `WM` were undefined abbreviations. | Spell out Visual Memory, Language, and projected WM message. |
| C1-12 | P1 | VA | WM projection detail was absent. | Name the source as `Projected WM_{i-1} message`. |
| C1-13 | P1 | VA | Source row, bus, and attention prose repeated the same information. | Collapse into one source group plus one edge. |
| C1-14 | P1 | VA | Residual routes were unlabeled. | Label both bypasses `residual`. |
| C1-15 | P1 | VA | `Pre-Norm` followed by `Add & Norm` mixed two conventions. | Use `Residual Add`; put Pre-Norm inside attention/FFN blocks. |
| C1-16 | P1 | VA | Source bus was not centered on attention. | Center the compact source group above attention. |
| C1-17 | P1 | VA | Main horizontal gaps were uneven. | Re-space modules on one baseline. |
| C1-18 | P1 | VA | Input and output widths did not match. | Set both to 170 px. |
| C1-19 | P1 | VA | WM blue emphasis disappeared at the attention input. | Keep projected WM source text blue. |
| C1-20 | P1 | VA | The figure did not name the attention computation. | Add the compact scaled dot-product equation. |
| C1-21 | P1 | Interaction | Current `Z_i` was not clearly restricted to commit. | Add labeled `Z_i → Commit S_i` edge. |
| C1-22 | P1 | Interaction | Current VA outputs were not labeled on commit edge. | Add `V_i, A_i` edge label. |
| C1-23 | P1 | Interaction | Orange formulas floated separately from their route. | Put formulas inside the orange source node. |
| C1-24 | P1 | Interaction | Blue formula floated separately from its route. | Put formula inside the blue source node. |
| C1-25 | P1 | Interaction | Delay pill was disconnected. | Fold `one-stage delay` into the blue source node. |
| C1-26 | P1 | Interaction | `Condition K,V` had no bound target. | Place it inside the orange node that points to WM. |
| C1-27 | P1 | Interaction | VA only said “include WM message.” | State `K,V ← previous WM message`. |
| C1-28 | P1 | Interaction | Commit did not state when its state becomes visible. | Add `visible to stage i+1`. |
| C1-29 | P1 | Interaction | VA and WM peer boxes were cavernous. | Reduce both to 440×345 px. |
| C1-30 | P1 | Interaction | Composition density was low. | Move compact message nodes into the central gap. |
| C1-31 | P2 | VA | Text glyph dividers looked like fake geometry. | Remove divider glyphs. |
| C1-32 | P2 | VA | Typography jumped between too many sizes. | Normalize body text to 18–22 px. |
| C1-33 | P2 | Interaction | Giant cards weakened hierarchy. | Reduce card size while retaining peer symmetry. |
| C1-34 | P2 | Interaction | Lower routes nearly merged with module borders. | Route output edges through a dedicated lower lane. |

### Cycle 2 Inventory — 18 Findings
| id | priority | figure | finding | Cycle 3 action |
|---|---|---|---|---|
| C2-01 | P0 | Both | Initial captures included DrawIO chrome and too much blank editor area. | Re-capture exact graph bounds; overwrite canvas evidence. |
| C2-02 | P1 | VA | Q/K/V were paths, but projection operators were still implicit. | Add `W_Q`, `W_K`, `W_V` labels without adding boxes. |
| C2-03 | P1 | VA | “Q attends to K,V” duplicated the equation. | Remove the prose line. |
| C2-04 | P1 | VA | Equation used `√d` instead of key dimension. | Use `√d_k`. |
| C2-05 | P1 | VA | Exported figure had no internal VA layer title. | Fold `VA Layer i` into the attention title. |
| C2-06 | P1 | VA | Right-side horizontal gaps compressed toward the output. | Re-space all compact blocks to 60 px gaps. |
| C2-07 | P1 | VA | FFN and output shared the same cyan fill. | Lighten output fill to a distinct cyan tint. |
| C2-08 | P1 | VA | Typography hierarchy was too flat in static preflight. | Widen body/title size ratio while retaining readability. |
| C2-09 | P1 | Interaction | Blue arrow entered VA near the former output line. | Move entry to the K,V area. |
| C2-10 | P1 | Interaction | VA text implied K,V equal only the WM message. | Change to “include projected previous WM message.” |
| C2-11 | P1 | Interaction | `Condition K,V` did not match any WM submodule. | Rename to `Predictor K,V`. |
| C2-12 | P1 | Interaction | Old-state source relation depended on vague “VA state” wording. | Use `Read … from S_{i−1}` on both message nodes. |
| C2-13 | P1 | Interaction | Snapshot edges visually shared the snapshot bottom border. | Add vertical drops before branching horizontally. |
| C2-14 | P1 | Interaction | VA/WM cards still had cavernous-box warnings. | Remove duplicate outputs, shrink cards, enlarge body text. |
| C2-15 | P2 | Interaction | `sg` was unexplained. | Replace with `stop-grad`. |
| C2-16 | P2 | Interaction | Outputs were duplicated inside modules and again on commit edges. | Keep them only on commit edges. |
| C2-17 | PASS | Interaction | No current-stage VA↔WM edge remained. | Preserve. |
| C2-18 | PASS | Interaction | Current `Z_i` had exactly one outgoing route, to Commit. | Preserve. |

### Cycle 1 P0/P1 Verification After Cycle 3
- All six Cycle 1 P0 blockers are closed.
- All Cycle 1 P1 items are closed or deliberately represented by concise labels rather than extra boxes.
- The two colored paths originate from old-snapshot message nodes; current module outputs only enter `Commit S_i`.

### Cycle 3 Inventory — 8 Findings
| id | priority | figure | finding | disposition |
|---|---|---|---|---|
| C3-01 | PASS | VA | `W_Q`, `W_K`, and `W_V` are visible and feed the correct Q/K/V paths. | verified |
| C3-02 | PASS | VA | Scaled dot-product equation is complete and uses `√d_k`. | verified |
| C3-03 | PASS | VA | Both residual sources, bypasses, and arrowheads terminate at the intended Add nodes. | verified |
| C3-04 | PASS | VA | Main node centers align and consecutive compact-module gaps are 60 px. | verified |
| C3-05 | PASS | Interaction | Both message cards explicitly read from `S_{i−1}`; no current-peer direct edge exists. | verified |
| C3-06 | PASS | Interaction | Current `Z_i` has only one outgoing edge, to `Commit S_i`. | verified |
| C3-07 | P1 | Interaction | Blue arrow landed between Q and K,V rows. | changed `entryY` from 0.70 to 0.78; final crop verifies K,V alignment |
| C3-08 | P2 | VA | `W_Q` and `Q` are stacked while K/V projections are inline. | accepted: 60 px gap needs compact stacking and remains unambiguous |

## Requirement And Semantic Audit
- Cycle 1 blocker: both colored interaction arrows encoded same-stage peer-to-peer exchange; this contradicted the shared-snapshot update rule.
- Cycle 1 blocker: VA attention did not expose Q and K,V as structural paths.

## Red-Team Visual Audit
- 21-item hostile re-scan completed across text, arrows, boxes, spacing, color, typography, layout, icons, and style coherence.
- P0: 0; P1: 0 after the `entryY=0.78` final correction.
- Confirmed: no arrow reversal, no current-stage VA↔WM shortcut, no `Z_i→VA_i`, no crossing, no clipped text, and no editor chrome in final crops.

| check | zone | result | evidence |
|---|---|---|---|
| RT-01 | text | PASS | VA input and output labels are complete and not clipped. |
| RT-02 | text | PASS | Interaction snapshot, message, peer, and commit labels are complete. |
| RT-03 | text | PASS | Scaled dot-product equation is readable at the final 1570 px export width. |
| RT-04 | text | PASS | `i−1`, `i`, superscript W, and `d_k` indices remain distinguishable. |
| RT-05 | arrows | PASS | VA main pipeline is strictly left-to-right. |
| RT-06 | arrows | PASS | `W_Q→Q` path points from VA inputs into Shared Attention. |
| RT-07 | arrows | PASS | K,V path points from the projection group down into attention. |
| RT-08 | arrows | PASS | First residual bypasses attention and terminates at the first Add. |
| RT-09 | arrows | PASS | Second residual bypasses FFN and terminates at the second Add. |
| RT-10 | arrows | PASS | Old snapshot points separately to current VA and current WM. |
| RT-11 | arrows | PASS | Orange old-VA message points only into current WM. |
| RT-12 | arrows | PASS | Blue delayed-WM message points only into the K,V row of current VA. |
| RT-13 | arrows | PASS | Current VA output edge reaches Commit and is labeled `V_i,A_i`. |
| RT-14 | arrows | PASS | Current WM output edge reaches Commit and is labeled `Z_i`. |
| RT-15 | boxes | PASS | VA boxes do not overlap and every arrow touches only intended borders. |
| RT-16 | boxes | PASS | Interaction cards do not overlap; central messages stay inside the gap. |
| RT-17 | spacing | PASS | VA compact modules have 60 px horizontal gaps and a common y-center. |
| RT-18 | spacing | PASS | Interaction snapshot, messages, and commit share the central axis. |
| RT-19 | color | PASS | Pink input, peach VA/attention, yellow-green state/Add, and cyan FFN/WM remain consistent. |
| RT-20 | color | PASS | Orange and blue provide redundant direction coding for the two old-state messages. |
| RT-21 | typography | PASS | Title/body hierarchy spans 16–28 pt without micro-text. |
| RT-22 | typography | PASS | Black text and dark blue/orange accents remain legible on pastel fills. |
| RT-23 | layout | PASS | Both deliverables are independent landscape figures, as requested. |
| RT-24 | layout | PASS | Top/current/bottom stage ordering is preserved in the interaction figure. |
| RT-25 | icons | PASS | No icons, decorative glyphs, logos, or embedded raster assets exist in DrawIO. |
| RT-26 | style | PASS | Rounded boxes, 2.5 px black strokes, and flat Transformer-like pastels are coherent. |
| RT-27 | style | PASS | No gradients, shadows, textures, or decorative callouts appear in DrawIO. |
| RT-28 | semantics | PASS | Time indices progress from `S_{i−1}` and old messages to current outputs and `S_i`. |
| RT-29 | semantics | PASS | No same-stage VA↔WM edge and no `Z_i→VA_i` edge exist. |
| RT-30 | export | PASS | Final crops are 1570×564 and 1550×745 with safe margins and no DrawIO chrome. |

## Self-Score
| dimension | concrete evidence / deduction | score |
|---|---|---:|
| Text readability | All labels are readable in the exact-bounds crops; one point deducted because the 16–17 pt context/formula lines are smaller than the 22–28 pt module text. | 9/10 |
| Arrow accuracy | Every source/target and arrowhead was checked in XML and screenshots; the delayed blue edge now lands on the K,V row. | 10/10 |
| Color coherence | Transformer-like pastel roles are consistent; one point deducted because the white K/V context card is intentionally neutral rather than modality-colored. | 9/10 |
| Layout consistency | Main VA centers and compact gaps are exact; one point deducted for the necessarily long first residual bypass. | 9/10 |
| Style match to specification | Clean horizontal split, flat fills, black strokes, no decoration; one point deducted because the requested horizontal redesign intentionally departs from the vertical reference composition. | 9/10 |
| **TOTAL** | Every dimension ≥6 and total ≥40; ALLOWED. | **46/50** |

## Remaining Gaps
- No blocking gaps. AI-generated variants are creative raster companions; DrawIO remains the editable semantic source of truth.

## Snapshot / Commit Clarification Loop (2026-08-27)

The interaction figure was reopened after the user noted that `Shared Snapshot` and `Commit` did not expose their real fields. The same critic reviewed every rendered revision until the blocking count reached zero.

| critic round | P0 | P1 | key correction |
|---|---:|---:|---|
| 1 | 6 | 14 | Expanded `S_{i−1}` / `S_i`; separated stored state, derived WM message, and fixed context. |
| 2 | 6 | 8 | Removed crossed/duplicated routes; added explicit VA input fork and WM proposal aggregation. |
| 3 | 2 | 3 | Added post-predictor map-conditioned belief so committed `B_i` has the correct source. |
| 4 | 1 | 2 | Corrected old-`Z` routing, kept `Z_i→proposal` inside WM, and separated current-state rails. |
| 5 | 0 | 1 | Exposed the old-`Z_{i−1}` leftward arrowhead at the Predictor boundary. |
| 6 | 0 | 0 | Final gate passed; no blocking semantic or visual defects remain. |

Final evidence:

- DrawIO render: `review_history/revision-snapshot-round6.png`
- Structural validator: `OK`
- Strict visual preflight: `FAIL 0 / WARN 0`
- Critic verdict: `P0=0 / P1=0 · 可交付`

## Simplification + Component Loop (2026-08-27)

The dense interaction figure was replaced by four small figures: overview, VA Q/K/V, WM update, and training/loss. The same critic rechecked the real rendered PNGs after every revision.

| cycle | P0 | P1 | key correction |
|---|---:|---:|---|
| 1 | 2 | 7 | Corrected committed-belief order, Pre-Norm residuals, innovation, flow loss, and WM objective. |
| 2 | 0 | 2 | Separated old/new `Z` ports and disambiguated the two independent training forwards. |
| 3 | 0 | 0 | Final rendered gate passed. |

Final evidence:

- `review_history/simplify-cycle3-overview.png`
- `review_history/simplify-cycle3-qkv.png`
- `review_history/simplify-cycle3-wm.png`
- `review_history/simplify-cycle3-training.png`
- Structural + strict visual validation: all four `OK`, `FAIL 0 / WARN 0`
- Red-team: 18 checks passed; no clipping, false junction, reversed arrow, or same-stage WM→VA feedback.
- Final score: **47/50** (`content 9 · logic 10 · readability 9 · style 9 · editability 10`).
