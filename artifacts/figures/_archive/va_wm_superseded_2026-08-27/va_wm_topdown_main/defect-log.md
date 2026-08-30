# Defect Log

## Preflight Review
- `validate_visual_quality.py`: 0 FAIL, 1 WARN before first preview.
- Accepted WARN `spacing-inconsistent-v` on the WM column: the larger gap reserves the two-line title band of the nested spatiotemporal predictor; the remaining sublayer gaps are intentionally compact. Screenshot inspection still decides whether the title band is sufficient.

## Screenshot Review Cycle 1
- Evidence: `review-cycle-1-canvas.png`, canvas-only crop, diagram content > 95%.
- Cross-check: user prompt, `README.md`, `policy/model.py`, `world/wmrm.py`, and the sampled Transformer style contract.

### P0 — Blockers
| id | zone | element | description | evidence | status |
|---|---|---|---|---|---|
| C1-P0-01 | semantics | `e_map_exchange` | Current `Z_i` is routed into the displayed `VA_i`, visually collapsing the required one-stage delay; it should become an incoming `W_{i-1}→VA_i` message and current `Z_i` should go only to the next-stage commit. | center exchange lane | OPEN |
| C1-P0-02 | arrows | `e_snapshot_wm` | Snapshot connector passes through the `World Memory stage WM_i` title. | upper-right tower | OPEN |
| C1-P0-03 | arrows/text | `e_snapshot_exchange` | Vertical snapshot connector crosses the exchange title and the `one-stage delay` subtitle. | exchange header | OPEN |
| C1-P0-04 | arrows | `e_map_exchange` + `e_va2wm_condition` | Blue WM→VA and black VA→WM condition routes share/cross the narrow gap, making direction ambiguous. | VA/WM gap around y≈670–760 | OPEN |

### P1 — Visible defects
| id | zone | element | description | evidence | status |
|---|---|---|---|---|---|
| C1-P1-01 | box integrity | `wm_map` | Predicted-map box protrudes below the WM tower border. | lower-right tower | OPEN |
| C1-P1-02 | text | `st_container` | The two-line predictor subtitle collides with the first self-attention box; `Pre-Norm...` is partly hidden. | ST predictor header | OPEN |
| C1-P1-03 | arrows | `e_stage_loop` | Recurrence route creates a second dashed vertical border at the far left and competes with the stage frame. | left edge | OPEN |
| C1-P1-04 | text | `e_stage_loop` | `next paired stage` label is isolated at the crop edge and is too small. | left edge | OPEN |
| C1-P1-05 | arrows | `e_commit_finalva` | Edge label sits directly on the outer dashed border. | below stage commit | OPEN |
| C1-P1-06 | arrows | `e_map_loss` | Training connector hugs the right stage border, making it look like part of the container. | far right | OPEN |
| C1-P1-07 | text | `e_map_loss` | `prediction` label is pushed to the extreme right and nearly clipped. | far right | OPEN |
| C1-P1-08 | arrows | `e_va2wm_evidence` | `K,V` label is squeezed into the VA/WM boundary. | right of VA→WM box | OPEN |
| C1-P1-09 | arrows | `e_va2wm_condition` | Second `K,V` label is squeezed against the ST predictor boundary and another connector. | center-right | OPEN |
| C1-P1-10 | arrows | `e_exchange_vaattn` | Blue edge label sits on the exchange border rather than on an open segment. | left of exchange lane | OPEN |
| C1-P1-11 | layout | `va_container` / `wm_container` | Tower bottoms are visually unequal because WM content extends farther down. | paired towers | OPEN |
| C1-P1-12 | typography | `wm_cross_attn` | Long K/V expression wraps awkwardly and splits the bracketed source list. | ST predictor middle | OPEN |
| C1-P1-13 | typography | `snapshot` | Detail line is too small relative to the main figure scale. | shared snapshot | OPEN |
| C1-P1-14 | typography | `stage_commit` | Explanatory line is too small and dense. | stage commit | OPEN |
| C1-P1-15 | style coherence | `kv_note` | Two-line negative explanation is more verbose than the requested minimalist style. | exchange lower band | OPEN |
| C1-P1-16 | layout | `stage_commit` | Commit box is only 20–25 px below dense tower outputs, producing a cramped convergence zone. | bottom of stage | OPEN |
| C1-P1-17 | arrow hygiene | `e_wm_commit` | The WM commit route takes a long horizontal detour and its `W_i` label lands on the stage convergence line. | lower-right to center | OPEN |
| C1-P1-18 | visual hierarchy | `wm_to_va` | Box heading says “next VA” while its arrow targets the displayed current VA; even apart from P0 semantics, the label hierarchy is confusing. | exchange center | OPEN |

### P2 — Polish
| id | zone | element | description | evidence | status |
|---|---|---|---|---|---|
| C1-P2-01 | spacing | input row | Top boxes have generous spacing, but the gap to the snapshot is much tighter than the gap to the stage. | top third | OPEN |
| C1-P2-02 | typography | tower titles | Subscript `i` sits close to the final letter in both tower headings. | tower headers | OPEN |
| C1-P2-03 | color | `wm_to_va` | Lavender fill is coherent but could use slightly stronger contrast against the white exchange lane. | exchange center | OPEN |
| C1-P2-04 | box integrity | `exchange_container` | Large empty lower area makes the exchange lane feel taller than its content. | center tower | OPEN |
| C1-P2-05 | spacing | VA stack | Attention-to-Add gap is slightly larger than FFN-to-Add gap. | left tower | OPEN |
| C1-P2-06 | spacing | WM stack | Evidence, belief, and predictor gaps are not on one visible rhythm. | right tower | OPEN |
| C1-P2-07 | typography | `world_loss` | `training only` sits very close to the lower border. | loss box | OPEN |
| C1-P2-08 | composition | output stack | Loss branch is farther right than the WM tower center and feels detached. | lower-right | OPEN |
| C1-P2-09 | style coherence | all arrows | Normal black arrows use both 2.0 and 2.5 px widths; the difference is barely meaningful. | full canvas | OPEN |
| C1-P2-10 | text | `delay_label` | Delay is stated both in the exchange container subtitle and a separate label. | exchange lane | OPEN |

Cycle 1 total: 32 findings (4 P0, 18 P1, 10 P2).

### Cycle 1 verification
- All four P0 semantic/route blockers were removed or rerouted.
- All eighteen P1 items were corrected before or during the Cycle 2 refinement; the predictor-header clearance was reopened below because one connector still crossed the simplified header band.
- P2 items were either tightened (exchange height, output alignment, loss spacing) or accepted where they encode hierarchy (slightly heavier structural arrows).

## Screenshot Review Cycle 2
- Evidence: `review-cycle-2-canvas.png`, canvas-only crop.
- Audit performed before the Cycle 2 refinement patch below.

### P0 — Blockers
| id | zone | element | description | fix | status |
|---|---|---|---|---|---|
| C2-P0-01 | semantics/arrows | `e_snapshot_exchange` | The arrowhead enters the `VA → WM` box from the right, visually contradicting the named left-to-right transfer. | Delete the redundant connector; `same S_i−1` already states provenance. | PATCHED—VERIFY C3 |
| C2-P0-02 | arrows/text | `e_wm_belief_self` | The belief-to-predictor line crosses the predictor title and `×6`. | Route into the self-attention block along the predictor's right margin. | PATCHED—VERIFY C3 |
| C2-P0-03 | arrows/text | `e_map_loss` | The dashed prediction path enters through `training only`. | Enter the upper-right side of the loss box. | PATCHED—VERIFY C3 |

### P1 — Visible defects
| id | zone | element | description | fix | status |
|---|---|---|---|---|---|
| C2-P1-01 | typography | `st_container` | The predictor subtitle remains dense and too close to the first sublayer. | Keep only the predictor title. | PATCHED—VERIFY C3 |
| C2-P1-02 | box integrity | `wm_map` | Predicted-map box has only about 8 px clearance to the WM border. | Move FFN/map upward and regularize the gap. | PATCHED—VERIFY C3 |
| C2-P1-03 | composition | `exchange_container` | The lower exchange lane is empty after the delay-path cleanup. | Reduce its height by 60 px. | PATCHED—VERIFY C3 |
| C2-P1-04 | route hygiene | `e_map_loss` | The loss route hugs the far-right crop edge. | Shorten the path and terminate at the right upper quadrant. | PATCHED—VERIFY C3 |
| C2-P1-05 | labeling | `e_target_loss` | The word `target` repeats the future-target box label. | Remove the edge label. | PATCHED—VERIFY C3 |
| C2-P1-06 | typography | `world_loss` | The subtitle sits too close to the lower border. | Increase box height by 5 px. | PATCHED—VERIFY C3 |
| C2-P1-07 | spacing | `wm_ffn` / `wm_map` | The final two WM blocks use a tighter rhythm than the preceding predictor blocks. | Shift and resize the final pair. | PATCHED—VERIFY C3 |
| C2-P1-08 | hierarchy | snapshot/exchange | A third snapshot branch competes with the two essential peer arrows. | Remove the redundant branch. | PATCHED—VERIFY C3 |

### P2 — Polish
| id | zone | element | description | disposition | status |
|---|---|---|---|---|---|
| C2-P2-01 | style | structural arrows | 2.5 px main-flow arrows remain slightly heavier than 2 px internal arrows. | Accepted hierarchy. | ACCEPTED |
| C2-P2-02 | style | `stage_container` | Dashed stage frame is visually stronger than the blue exchange frame. | Accepted grouping hierarchy. | ACCEPTED |
| C2-P2-03 | composition | lower stage | The commit band retains generous breathing room below the towers. | Accepted to separate recurrence from prediction. | ACCEPTED |
| C2-P2-04 | color | training branch | Muted maroon is the only non-Transformer accent. | Accepted to mark training-only supervision. | ACCEPTED |

Cycle 2 total: 15 findings (3 P0, 8 P1, 4 P2).

### Cycle 2 verification
- Cycle 3 screenshot confirms all three P0 and all eight P1 findings are fixed.
- The only preflight warning is the accepted WM vertical-spacing warning caused by the nested predictor title band.

## Screenshot Review Cycle 3
- Evidence: `review-cycle-3-canvas.png`, 1440 × 1250 canvas-only crop.
- No P0 or P1 defects remain.

### P2 — Final polish audit
| id | zone | observation | disposition |
|---|---|---|---|
| C3-P2-01 | stage title | The long paired-stage title is visually prominent. | Accepted: it replaces a separate legend. |
| C3-P2-02 | snapshot | The second line carries four state groups. | Accepted: it is the minimum needed to define the shared snapshot. |
| C3-P2-03 | VA residuals | Left and right residual loops are asymmetric. | Accepted: they map to different sublayers and remain unambiguous. |
| C3-P2-04 | exchange | Blue delayed K/V path is the only colored data arrow. | Accepted: it is the paper's focal interaction. |
| C3-P2-05 | WM predictor | The condition connector enters through the predictor's left boundary. | Accepted: no text or arrow is crossed. |
| C3-P2-06 | training branch | Loss supervision is visually detached from the action path. | Accepted: it prevents training-only loss from reading as an inference output. |
| C3-P2-07 | crop | The loss-route arrowhead is close to the right crop margin. | Accepted: the full arrowhead and route remain visible. |
| C3-P2-08 | caption | The figure has no embedded caption or global title. | Accepted: caption belongs in the manuscript. |

Cycle 3 total: 8 findings (0 P0, 0 P1, 8 P2).

## Red-team audit
| check | result |
|---|---|
| Inputs begin at the top and the physical action exits at the bottom. | PASS |
| No current `Z_i` is fed into the displayed current `VA_i`. | PASS |
| Delayed `WM_{i−1} → VA_i` path is blue and explicitly stop-gradient/projected. | PASS |
| VA attention shows both Q construction and full K/V source list. | PASS |
| WM evidence attention shows learned-Q and visual K/V. | PASS |
| WM belief update shows belief Q and innovation K/V. | PASS |
| WM predictor shows causal Q=K=V and conditional Q/K/V. | PASS |
| VA and WM both read the same pre-stage snapshot before commit. | PASS |
| Only the VA Flow head leads to the action chunk. | PASS |
| Training-only world loss cannot be mistaken for an inference edge. | PASS |

## Final score
| dimension | score / 10 |
|---|---:|
| semantic correctness | 10 |
| hierarchy/composition | 9 |
| typography/legibility | 9 |
| arrow clarity | 9 |
| style consistency | 9 |
| **total** | **46 / 50** |

## Final validation
- DrawIO integrity validator: **PASS** — 1 page, 70 cells, 35 vertices, 33 edges, no duplicate IDs, raster payloads, external images, placeholders, or embedded caption.
- Visual preflight: **PASS** — 0 FAIL, 1 accepted WARN for deliberately non-uniform spacing around the nested WM predictor title band.
- Browser render: **PASS** — canvas-only screenshot inspected at 1440 × 1250; the only console error is the preview server's irrelevant missing `favicon.ico`.
- AI variant inspected against the exact figure: top-down topology, VA/WM emphasis, delayed blue WM→VA K/V path, and sole VA action output are preserved.
