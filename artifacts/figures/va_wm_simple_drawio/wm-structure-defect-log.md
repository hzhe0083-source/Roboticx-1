# WM Structure Defect Log

The log is append-only after Screenshot Review Cycle 1.

## Pre-render semantic audit

- The committed `B_i` comes from the post-predictor map-conditioned update, not the pre-belief.
- `ν_i` and `Z_i` are explicitly written to the committed state.
- Candidate action is labeled as a readout from latent action tokens, not an executed action.
- `Z_{i−1}` is refinement context; it is not mixed with fixed context.

## Screenshot Review Cycle 1

Evidence: `review_history/wm-structure-cycle1.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| C1-01 | Text | Panel sentence repeats the title. | P1 | Reduce panel label to `WM_i`. |
| C1-02 | Text | Panel sentence is too long for a simple component figure. | P1 | Remove the prose summary. |
| C1-03 | Text | `atomic commit` is compact jargon. | P2 | Keep because the output tuple makes it concrete. |
| C1-04 | Text | `remove overlap` is secondary prose. | P2 | Keep as a muted implementation detail. |
| C1-05 | Text | `stage i+1` is plain text rather than a subscripted formula. | P2 | Accept for presentation readability. |
| C1-06 | Text | Predictor context uses compact `D·p·L` notation. | P2 | Inputs are defined in the surrounding talk. |
| C1-07 | Arrow | Old-state→Innovation has a tiny horizontal elbow. | P1 | Align exact source and target x coordinates. |
| C1-08 | Arrow | `Z_{i−1} refinement` sits directly on the dashed edge. | P1 | Move the meaning into the Predictor note and remove the edge label. |
| C1-09 | Arrow | Two old-state exits are close together. | P2 | Preserve because they encode distinct `B/ν` reads. |
| C1-10 | Arrow | Context→Predictor is the tallest input edge. | P2 | It remains straight and unobstructed. |
| C1-11 | Arrow | Three commit paths are longer than the main-chain edges. | P1 | Move Commit upward. |
| C1-12 | Arrow | The `B_i` commit path has a one-pixel lateral correction. | P1 | Align the target port with the source center. |
| C1-13 | Arrow | Edge labels use white backgrounds on a white canvas. | P2 | Retain to protect labels if routes shift. |
| C1-14 | Box | Evidence is 80 px high while adjacent blocks are 90 px. | P1 | Use a uniform 90 px row. |
| C1-15 | Box | Pre-belief is 80 px high while adjacent blocks are 90 px. | P1 | Use a uniform 90 px row. |
| C1-16 | Box | Predictor context is wider than the other inputs. | P2 | Required for the candidate-action expression. |
| C1-17 | Box | Commit bar is visually heavier than the transforms. | P1 | Reduce the empty band above it; retain state-boundary width. |
| C1-18 | Box | Map-conditioned title is close to the side padding. | P2 | Text remains fully readable. |
| C1-19 | Spacing | Main row→Commit gap is 125 px. | P1 | Reduce to 90 px. |
| C1-20 | Spacing | Bottom whitespace is larger than needed. | P1 | Reduce canvas and panel height. |
| C1-21 | Spacing | Top input gaps are unequal. | P2 | Their positions follow their consumers. |
| C1-22 | Spacing | Predictor context begins 5 px above the other inputs. | P2 | Its extra line requires the height offset. |
| C1-23 | Spacing | Panel label is close to the input row. | P2 | Shortening the label will resolve the visual competition. |
| C1-24 | Color | Old WM state and Predictor share blue-family semantics. | P2 | Correct: both are WM representations. |
| C1-25 | Color | Orange occurs only once. | P2 | Correct: only Evidence is attention-like. |
| C1-26 | Color | Commit and pre-belief share pale yellow. | P2 | Both are state boundaries, so reuse is semantic. |
| C1-27 | Typography | Title and long panel sentence are both bold. | P1 | Shorten the panel label. |
| C1-28 | Typography | Edge labels are smaller than body formulas. | P2 | Correct secondary hierarchy. |
| C1-29 | Typography | Gray notes use body-sized text. | P2 | Color provides the secondary level. |
| C1-30 | Layout | Commit bar dominates the lower third. | P1 | Compress the lower region. |
| C1-31 | Layout | The main flow remains clearly left-to-right. | P2 | Preserve. |
| C1-32 | Layout | Inputs correctly start above their consumers. | P2 | Preserve. |
| C1-33 | Icons | No visual or robot icon is used. | P2 | Intentional Transformer box grammar. |
| C1-34 | Icons | No legend is present. | P2 | Color meanings are local and do not need one. |
| C1-35 | Style | Mixed main-row heights weaken Transformer rhythm. | P1 | Standardize the row. |
| C1-36 | Style | Context has three text lines while other inputs have two. | P2 | Necessary to separate candidate action from fixed context. |

P0: 0. All P1 items are repaired in Cycle 2.

## Fix Verification — Cycle 1

- FIXED: C1-01, C1-02, C1-07, C1-08, C1-11, C1-12, C1-14, C1-15, C1-17, C1-19, C1-20, C1-27, C1-30, C1-35.
- No regression in input meaning, state fields, predictor conditions, or commit sources.

## Screenshot Review Cycle 2

Evidence: `review_history/wm-structure-cycle2.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| C2-01 | Text | `WM_i` alone is more cryptic than necessary. | P1 | Rename to `WM stage i`. |
| C2-02 | Text | `WM output / atomic commit` says the same role twice. | P1 | Keep only `Atomic commit`. |
| C2-03 | Text | Three edge labels repeat the output tuple. | P1 | Remove the labels and keep the three structural arrows. |
| C2-04 | Arrow | Innovation, Predictor, and final Belief now enter Commit with straight vertical paths. | P2 | Preserve. |
| C2-05 | Arrow | Dashed old-map edge is unlabeled. | P2 | Predictor note now identifies refinement with `Z_{i−1}`. |
| C2-06 | Arrow | Old-state fan-out uses three separate target ports. | P2 | Preserve. |
| C2-07 | Box | Map-conditioned title has the tightest horizontal padding. | P1 | Widen by 10 px. |
| C2-08 | Box | Predictor context is wider and taller than other inputs. | P2 | Required for three input groups; centers align. |
| C2-09 | Box | Commit remains wider than a transform box. | P2 | Its width spans the three committed producers. |
| C2-10 | Spacing | Main row has uniform 90 px heights and 40 px gaps. | P2 | Preserve. |
| C2-11 | Spacing | Context top differs by 5 px, but all input centers align. | P2 | Preserve center alignment. |
| C2-12 | Color | Predictor and old state share cyan. | P2 | Both are WM representations. |
| C2-13 | Typography | Muted notes remain readable at export scale. | P2 | Preserve. |
| C2-14 | Layout | Output is at the bottom without a redundant terminal arrow. | P2 | Preserve. |
| C2-15 | Icons | No pictograms or legend. | P2 | Intentional simple Transformer grammar. |
| C2-16 | Style | Title wording uses `internal structure`, unlike the shorter component labels. | P2 | Shorten to `WM component` for suite consistency. |

P0: 0. Cycle 3 removes the four remaining P1 items.

## Fix Verification — Cycle 2

- FIXED: C2-01, C2-02, C2-03, C2-07.
- No new overlap, clipping, false junction, or semantic regression.

## Screenshot Review Cycle 3 — Final Gate

Evidence: `review_history/wm-structure-cycle3.png`.

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| C3-01 | Text | Title and panel label both contain `WM`. | P2 | Needed when the panel is reused in a slide crop. |
| C3-02 | Arrow | Old map uses a dashed line without an edge label. | P2 | Predictor note names `Z_{i−1}` refinement. |
| C3-03 | Box | Context box remains the widest input. | P2 | It carries candidate action plus three fixed contexts. |
| C3-04 | Spacing | Commit bar spans three producers. | P2 | Width makes the atomic fan-in readable. |
| C3-05 | Color | Innovation and Predictor share cyan. | P2 | Both are WM representation transforms. |
| C3-06 | Typography | Nested subscripts are the smallest visible glyphs. | P2 | Native export remains readable. |
| C3-07 | Layout | No terminal arrow extends below Commit. | P2 | Commit itself is the requested bottom output. |
| C3-08 | Icons | No modality icons are used. | P2 | Matches the simple Transformer reference. |

- P0: 0.
- P1: 0.

## Red-Team Audit

| id | zone | hostile-review finding | severity | disposition |
|---|---|---|---|---|
| RT-01 | Text | `candidate û` requires one verbal definition in the presentation. | P2 | Keep exact method term; explain it as the proposed action chunk. |
| RT-02 | Text | `Ê(B)` abstracts the learned evidence prediction. | P2 | Appropriate component-level shorthand. |
| RT-03 | Arrow | Three old-state reads create the densest local region. | P2 | Ports are separated and routes do not touch. |
| RT-04 | Arrow | Dashed refinement path is longer than other input edges. | P2 | It uniquely denotes optional old-map refinement. |
| RT-05 | Box | Commit is much wider than one transform. | P2 | It collects three persistent outputs. |
| RT-06 | Spacing | Context is offset toward the predictor's right input port. | P2 | Prevents overlap with the old-map port. |
| RT-07 | Color | Green appears only on final belief. | P2 | It intentionally distinguishes post-predictor belief from pre-belief. |
| RT-08 | Typography | Gray notes are one hierarchy level below formulas only by color. | P2 | Readable and keeps box count low. |
| RT-09 | Layout | The active predictor's internal depth-6 blocks are not expanded. | P2 | This figure explains WM state logic, not predictor microarchitecture. |
| RT-10 | Style | No legend explains color roles. | P2 | Every role is written in its box; a legend would add noise. |

## Self-Score

| dimension | score | evidence |
|---|---:|---|
| Text readability | 10/10 | Every label is legible in the canvas-only export. |
| Arrow accuracy | 10/10 | All 12 connectors have clear source, target, and direction. |
| Color coherence | 10/10 | Six Transformer pastels plus one neutral stroke system. |
| Layout consistency | 9/10 | Commit is intentionally wider than transform boxes. |
| Style match | 9/10 | Context density is slightly higher than the approved VA figure. |
| **Total** | **48/50** | Allowed for handoff. |

## Final Validation

- Visual preflight: 0 FAIL / 0 WARN.
- Strict DrawIO validation: 0 errors / 0 warnings.
- Editable primitives only; no embedded raster or external image.

## User Feedback Repair — Symbolic Transformer Redesign

Found by user, missed by the previous final audit: the figure was still a labeled workflow, not a symbolic neural architecture.

| id | missed defect | severity | redesign rule |
|---|---|---|---|
| UF-S01 | One box per verbal step made the figure read like a business process. | P1 | Replace prose boxes with neural modules and operator nodes. |
| UF-S02 | Q/K/V existed only inside a formula. | P1 | Expose Q and K,V as separate attention ports. |
| UF-S03 | Visual tokens, belief slots, action chunks, and maps had no distinct glyphs. | P1 | Use semantically labeled token strips and patch grids. |
| UF-S04 | Innovation subtraction and overlap removal were written, not drawn. | P1 | Use `−` and `⊥ν` operator circles. |
| UF-S05 | Gated belief updates lacked Transformer-style residual bypasses. | P1 | Draw old-belief residual paths into Gated Add & Norm. |
| UF-S06 | The predictor looked identical to a generic process box. | P1 | Draw a repeated ST block with `×L` and patch-grid I/O. |
| UF-S07 | The wide Commit bar dominated the architecture. | P1 | Replace it with compact committed state symbols grouped as `W_i`. |

Reference check: original Transformer Figure 1 uses stacked modules, explicit residual routes, small Add/Norm primitives, symbolic embeddings/inputs, and a repetition mark rather than descriptive equations inside every module.

## Screenshot Review — Symbolic Cycle 1

Evidence: `review_history/wm-symbolic-cycle1-review.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| S1-01 | Text | `symbolic architecture` is useful during review but too meta for a slide figure. | P1 | Shorten the title to `World Memory (WM)`. |
| S1-02 | Text | `old innovation` and `new innovation` repeat what the indices already show. | P2 | Keep only the symbols in the final compact version. |
| S1-03 | Text | The prediction input says `robot · language`, while the variables already identify both. | P2 | Keep the tiny gloss only if space remains. |
| S1-04 | Text | `∥ condition` reads as a caption rather than an operator. | P1 | Use a compact concat symbol with a short `condition` label. |
| S1-05 | Arrow | The old-belief residual rises above the input row before descending. | P1 | Route it below the old-state symbols and down the panel gutter. |
| S1-06 | Arrow | The old-innovation path leaves the left tower before returning to `Proj⊥`. | P1 | Keep it inside the left gutter. |
| S1-07 | Arrow | The expected-evidence path makes a long horizontal shelf above the operators. | P1 | Stack subtraction and orthogonalization vertically. |
| S1-08 | Arrow | `Proj⊥ → ν_i` travels down-left, breaking the top-to-bottom reading order. | P1 | Place `ν_i` directly below `Proj⊥`. |
| S1-09 | Arrow | `ν_i → Gated Add & Norm` is horizontal although it is the main stream. | P1 | Put the gate directly below `ν_i`. |
| S1-10 | Arrow | Three long output rails run across the bottom. | P1 | Remove the rails; use a compact state definition. |
| S1-11 | Arrow | The map residual starts by moving upward, then loops below attention. | P1 | Use a conventional local U-shaped bypass. |
| S1-12 | Arrow | The `B̃_i → condition` line is the most visually dominant connector. | P1 | Thin it and keep the vertical segment in the center gutter. |
| S1-13 | Box | The left subtract and orthogonalization operators are side-by-side. | P1 | Stack them as sequential operators. |
| S1-14 | Box | `B̃_i` is repeated in a large strip and a boundary pill. | P2 | Keep the pill only as a fan-out port. |
| S1-15 | Box | The final right `Gated Add & Norm` is shifted far right. | P1 | Align it with `Map Cross-Attention`. |
| S1-16 | Box | The bottom `W_i` box is visually detached from both towers. | P1 | Replace it with a plain output-state line. |
| S1-17 | Box | The DINO map uses a textual matrix rather than literal tiny patch cells. | P2 | Accept: the single editable symbol is cleaner than decorative cells. |
| S1-18 | Box | The action chunk and token strips use consistent symbolic grammar. | P2 | Preserve. |
| S1-19 | Spacing | The left main stream alternates between centered and right-shifted nodes. | P1 | Recenter the full correction stack. |
| S1-20 | Spacing | The gap between evidence and subtraction is larger than later stage gaps. | P1 | Use a uniform vertical rhythm. |
| S1-21 | Spacing | The right tower has a large empty band around the predictor input merge. | P2 | Reduce after aligning the lower stack. |
| S1-22 | Spacing | The lower-right gate nearly touches the tower boundary. | P1 | Recenter it and keep a 25 px inner margin. |
| S1-23 | Color | Pastel roles match the Transformer reference. | P2 | Preserve. |
| S1-24 | Color | Blue residual/read-state paths are distinguishable from black main flow. | P2 | Preserve with slightly thinner strokes. |
| S1-25 | Color | Yellow is used for both evidence and state update families. | P2 | Keep distinct orange vs pale-yellow fills. |
| S1-26 | Typography | The main operator names are readable at slide scale. | P2 | Preserve 14–15 pt body hierarchy. |
| S1-27 | Typography | Edge labels `clip`, `cond`, and `refine` are smaller than necessary. | P2 | Keep only labels that disambiguate ports. |
| S1-28 | Typography | Nested subscript in `Proj⊥_{ν_{i−1}}` is dense. | P2 | Use `Proj⊥νᵢ₋₁` as the compact symbol. |
| S1-29 | Layout | Two towers clearly separate memory correction from prediction. | P2 | Preserve. |
| S1-30 | Layout | Inputs are at the top and state is defined at the bottom. | P2 | Preserve. |
| S1-31 | Icons | There are no decorative icons or modality pictograms. | P2 | Preserve the symbolic-only grammar. |
| S1-32 | Style | The right tower already resembles a Transformer stack; the left still resembles a flowchart. | P1 | Convert the left side to one vertical operator stack. |

P0: 0. Cycle 2 repairs every P1 while preserving the symbolic tensor grammar.

## Screenshot Review — Symbolic Cycle 2

Evidence: `review_history/wm-symbolic-cycle2-review.png`.

| id | zone | finding | severity | action |
|---|---|---|---|---|
| S2-01 | Text | The title is now slide-ready and no longer describes its own style. | P2 | Preserve. |
| S2-02 | Text | `residual` overlaps the nearby old-innovation edge label. | P1 | Remove both redundant edge labels. |
| S2-03 | Text | Bottom `Q` and `residual` labels compete beside the bridge. | P1 | Keep only `Q`; the bypass shape already denotes residual. |
| S2-04 | Text | `clip` and `cond` repeat the source/merge semantics. | P1 | Remove both labels. |
| S2-05 | Arrow | The left correction path is now strictly top-to-bottom. | P2 | Preserve. |
| S2-06 | Arrow | Old belief enters expected-evidence and the gated residual through distinct lanes. | P2 | Preserve. |
| S2-07 | Arrow | The old innovation enters `Proj⊥` from the right with a clear arrowhead. | P2 | Preserve, without the edge label. |
| S2-08 | Arrow | `B̃_i → condition` crosses the DINO clip line without a jump. | P1 | Add an arc jump to prevent a false junction. |
| S2-09 | Arrow | Q and K,V enter Map Cross-Attention at distinct ports. | P2 | Preserve. |
| S2-10 | Arrow | Map residual is local and no longer exits the panel. | P2 | Preserve. |
| S2-11 | Box | Both towers now use the same stacked-module grammar. | P2 | Preserve. |
| S2-12 | Box | `B̃_i` boundary pill cleanly exposes the shared tensor. | P2 | Preserve. |
| S2-13 | Spacing | Left operator gaps are uniform at approximately 18–23 px. | P2 | Preserve. |
| S2-14 | Spacing | Final right gate is centered under Map Cross-Attention. | P2 | Preserve. |
| S2-15 | Color | Pink inputs, orange attention, cyan representations, green map attention, and yellow normalization match Transformer roles. | P2 | Preserve across all three figures. |
| S2-16 | Typography | Main blocks are readable; only edge labels are visually noisy. | P1 | Delete the redundant labels, not the blocks. |
| S2-17 | Layout | The state definition is a light footer rather than a dominant output box. | P2 | Preserve. |
| S2-18 | Style | The figure now reads as a neural architecture rather than a workflow. | P2 | Final pass should only remove clutter. |

P0: 0. Cycle 3 removes S2-02, S2-03, S2-04, S2-08, and S2-16.

## Screenshot Review — Symbolic Cycle 3 Final Gate

Evidence: `review_history/wm-symbolic-cycle3-review.png`.

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| S3-01 | Text | Only Q, K,V, and refine remain as edge labels. | P2 | These three labels carry structural meaning. |
| S3-02 | Arrow | The blue condition path uses an arc at the DINO crossing. | P2 | Prevents a false junction. |
| S3-03 | Arrow | Old belief and old innovation use separate, unlabeled lanes. | P2 | Sources and arrowheads make both paths unambiguous. |
| S3-04 | Box | The left and right towers share the same vertical stack grammar. | P2 | Preserve. |
| S3-05 | Spacing | Both final gates remain inside their tower boundaries. | P2 | Preserve. |
| S3-06 | Color | Transformer pastel roles are consistent and each role is local. | P2 | Reuse identically in figures 1 and 3. |
| S3-07 | Layout | Output state is defined below both towers without extra rails. | P2 | Preserve. |
| S3-08 | Style | No workflow sentence or decorative icon remains. | P2 | Final symbolic target achieved. |

- P0: 0.
- P1: 0.

## Symbolic Red-Team Audit

| id | hostile-review check | result |
|---|---|---|
| SRT-01 | Could `B_i` be mistaken for pre-predictor belief? | PASS — pre-belief is `B̃_i`; `B_i` appears only after map cross-attention and gated norm. |
| SRT-02 | Could old `Z_{i−1}` be mistaken for the new map? | PASS — old map is dashed and only enters Predictor as `refine`. |
| SRT-03 | Could the candidate action be mistaken for an executed action? | PASS — it is explicitly `Readout(A_{i−1})` and shown as a condition token strip. |
| SRT-04 | Does evidence clearly read visual tokens through K,V? | PASS — `V_{i−1}` enters Cross-Attention at K,V. |
| SRT-05 | Is the evidence query source explicit? | PASS — `Q_E` is a separate symbolic query input. |
| SRT-06 | Is innovation subtraction represented as an operator? | PASS — `E_i−Ê` is an ellipse in the main stack. |
| SRT-07 | Does overlap removal use prior innovation? | PASS — old `ν_{i−1}` enters `Proj⊥ν_{i−1}`. |
| SRT-08 | Is the final belief conditioned on the live predicted map? | PASS — `Z_i→World Encoder→K,V→Map Cross-Attention→B_i`. |
| SRT-09 | Are residual and attention paths visually distinct? | PASS — blue bypasses and black main flow do not form false junctions. |
| SRT-10 | Can the figure be edited natively? | PASS — all content is DrawIO text, shape, and connector primitives. |

## Symbolic Self-Score

| dimension | score |
|---|---:|
| Text readability | 9/10 |
| Arrow accuracy | 10/10 |
| Color coherence | 10/10 |
| Layout consistency | 9/10 |
| Transformer style match | 9/10 |
| **Total** | **47/50** |

## User-Rejection Redesign — Cycle 1

Evidence: `review_history/wm-redesign-cycle1.png`. This is a clean-sheet replacement for the rejected framed-tower version.

| id | finding | severity | disposition |
|---|---|---|---|
| R1-01 | `B_i`, `Δ_i`, and `Z_i` writeback rails form a bottom perimeter frame. | P1 | Remove the rails. |
| R1-02 | `Z_i` writeback leaves the prediction spine, travels to the page edge, then returns. | P1 | Remove the writeback edge. |
| R1-03 | `B_i` writeback has a reversed-looking hook near Commit. | P1 | Use one direct downward arrow. |
| R1-04 | `Δ_i` writeback crosses beneath `B̃_i`. | P1 | List it in Commit without a long wire. |
| R1-05 | Old-belief residual uses a tall page-like rail. | P1 | Move `B_{i−1}` near its two consumers. |
| R1-06 | Old-belief source is too far from `Gated Add & Norm`. | P1 | Make the residual local. |
| R1-07 | `B̃_i` bridge is too far below Predictor. | P1 | Lift bridge and lower Predictor. |
| R1-08 | Predictor starts too high relative to the bridge. | P1 | Shift it down. |
| R1-09 | Right column has a large empty band between condition and Predictor. | P1 | Redistribute the spine. |
| R1-10 | `×L` sits on the blue `B̃_i` input line. | P1 | Move it above the line. |
| R1-11 | Innovation symbol `ν` resembles Latin `v` at slide scale. | P1 | Use defined `Δ` notation. |
| R1-12 | Old innovation is not verbally defined. | P1 | Define once, then keep the symbol compact. |
| R1-13 | `B_i` appears close to the crowded bottom rails. | P1 | Give it its own clean output row. |
| R1-14 | Commit is too far from the final belief. | P1 | Place Commit directly below `B_i`. |
| R1-15 | Commit field provenance is overdrawn by three separate wires. | P1 | Keep one direct commit arrow. |
| R1-16 | Left `B_{i−1}` fork and `Ê(B)` card are vertically loose. | P1 | Tighten their fork. |
| R1-17 | Map Cross-Attention residual and Q lines are too close. | P1 | Separate ports and short lanes. |
| R1-18 | `B̃_i` bridge-to-gate route is longer than needed. | P1 | Keep it local. |
| R1-19 | `B̃_i` bridge-to-attention route has an unnecessary bend. | P1 | Use a two-segment Q path. |
| R1-20 | `Z_i → World Encoder` gap is larger than neighboring gaps. | P2 | Normalize the right spine. |
| R1-21 | World Encoder-to-attention gap is larger than necessary. | P2 | Tighten while preserving K,V label. |
| R1-22 | Map-attention-to-gate gap is visually heavy. | P2 | Reduce. |
| R1-23 | Gate-to-`B_i` gap is visually heavy. | P2 | Reduce. |
| R1-24 | `B_i` box and Commit have unequal widths without reason. | P2 | Align their centers. |
| R1-25 | Left and right terminal states do not share a baseline. | P2 | Use Commit as the only terminal. |
| R1-26 | `ν_i` one-letter box is ambiguous. | P2 | Add `innovation`. |
| R1-27 | Old-innovation text box is too narrow. | P2 | Later reduce to the symbol only. |
| R1-28 | Right title is wider than the left title. | P2 | Keep text-only headings; no frame. |
| R1-29 | Figure is semantically correct but still reads partly as routed state plumbing. | P1 | Delete all nonessential state rails. |
| R1-30 | Final canvas needs a single obvious exit. | P1 | Use `Commit W_i` under the belief spine. |

## User-Rejection Redesign — Cycle 2

Evidence: `review_history/wm-redesign-cycle2.png`.

| id | check | result |
|---|---|---|
| R2-01 | Three long bottom writeback rails are gone. | PASS |
| R2-02 | Commit has one downward input from final `B_i`. | PASS |
| R2-03 | Commit still lists all persistent fields. | PASS |
| R2-04 | Predictor, `Z_i`, encoder, map attention, and final gate form one spine. | PASS |
| R2-05 | `B̃_i` reaches Predictor and map attention through one bridge token. | PASS |
| R2-06 | Q and K,V enter Map Cross-Attention at separate ports. | PASS |
| R2-07 | Old `Z_{i−1}` remains dashed and enters Predictor only. | PASS |
| R2-08 | Current `Z_i` enters the live World Encoder path without stop-grad. | PASS |
| R2-09 | Final `B_i` appears only after map-conditioned attention and gated normalization. | PASS |
| R2-10 | Old belief still sits lower than its conceptual input role. | P1 — move to the fork midpoint. |
| R2-11 | `ν` remains visually close to Latin `v`. | P1 — switch to `Δ`. |
| R2-12 | `×L` still touches the blue Predictor input. | P1 — lift the label. |
| R2-13 | Old-innovation card has too much prose for its width. | P1 — reduce to `Δ_{i−1}`. |
| R2-14 | No outer region frame or decorative card remains. | PASS |
| R2-15 | Real DrawIO render has no clipping or browser chrome. | PASS |

## User-Rejection Redesign — Cycle 3 and Final Polish

Evidence: `review_history/wm-redesign-cycle3.png`, `wm-redesign-cycle4.png`, and final `wm-redesign-cycle5.png`.

| id | check | result |
|---|---|---|
| R3-01 | Innovation uses the clearly defined `Δ` symbol throughout. | PASS |
| R3-02 | `×L` sits above Predictor rather than on an input edge. | PASS |
| R3-03 | `B_{i−1}` forks locally upward to `Ê(B)` and downward to the belief gate. | PASS |
| R3-04 | Old innovation is a compact `Δ_{i−1}` token with no clipped prose. | PASS |
| R3-05 | Commit is a single clean terminal and contains `[B_i │ Δ_i │ Z_i]`. | PASS |
| R3-06 | No edge crosses node text or creates a false junction. | PASS |
| R3-07 | Strict selected visual checks report 0 FAIL / 0 WARN; generic spacing variance was manually inspected because it false-groups the two independent spines. | PASS |
| R3-08 | XML preflight reports no warning, raster, external image, or invalid endpoint. | PASS |

## User-Rejection Redesign — Final Red-Team

1. Evidence attention receives `Q_E` as Q and `V_{i−1}` as K,V.
2. Expected evidence is derived from old belief, not current belief.
3. Innovation subtracts expected evidence and removes overlap with `Δ_{i−1}`.
4. The pre-predict belief is explicitly `B̃_i`, not the committed `B_i`.
5. Predictor condition includes DINO map, candidate action readout, proprioception, and language.
6. Old `Z_{i−1}` is dashed refinement context; new `Z_i` is a solid output.
7. `B̃_i` conditions Predictor and serves as Q/residual for the map-conditioned update.
8. `Z_i` reaches Map Cross-Attention through the live World Encoder path.
9. Committed `B_i` is downstream of map attention and gated normalization.
10. The final figure is fully editable DrawIO with no embedded raster.
