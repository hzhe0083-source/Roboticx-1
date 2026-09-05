# VA Layer Input Origin — Defect Log

## Screenshot Review Cycle 1

Evidence: `review_history/va-layer-input-cycle1.png`.

### P0 — Blockers

None.

### P1 — Visible defects

| id | zone | element | finding | planned fix |
|---|---|---|---|---|
| C1-01 | Text | `layer_output` | Output label is cramped and wraps more than peer boxes. | Widen output box. |
| C1-02 | Arrow | `e_v0_va1`, `e_a0_va1` | Two close vertical rails make the first-layer fan-in harder to scan. | Merge through one explicit initial-state box. |
| C1-03 | Box | `ellipsis` | Ellipsis is treated as a separate node between long component boxes. | Replace the three-node repetition with one previous-layers block. |
| C1-04 | Spacing | main chain | Large uneven gaps make the diagram unnecessarily wide. | Compact to a slide-friendly canvas. |
| C1-05 | Layout | overall | 2180×620 aspect ratio makes body text small on a 16:9 slide. | Reduce width to about 1880 px. |
| C1-06 | Semantics | first layer | The first-layer state `(V0,A0)` is only implied by two arrows. | Add `Initial VA state V0,A0`. |
| C1-07 | Semantics | `va_prev` | Showing only `VA1` and `VAi−1` can look like missing physical layers. | Use `Previous VA layers: VA1 → … → VAi−1`. |
| C1-08 | Typography | chain | The last output box is visibly smaller than the rest at equal font size. | Normalize box size and hierarchy. |
| C1-09 | Style | repetition | Ellipsis plus a no-arrow segment weakens the otherwise consistent arrow grammar. | Use ordinary arrowed transitions only. |
| C1-10 | Composition | right edge | The terminal output sits too close to the page edge. | Restore a 30–40 px margin. |

### P2 — Polish

| id | zone | element | finding |
|---|---|---|---|
| C1-11 | Text | subtitle | Subtitle could name the VA state explicitly. |
| C1-12 | Text | `vision_init` | Label is slightly longer than the action initializer label. |
| C1-13 | Text | `robot_state` | Three lines make the lower source visually heavier. |
| C1-14 | Text | `layer_input` | Gray explanatory line is useful but a little dense. |
| C1-15 | Arrow | `e_va1_more` | Missing arrowhead is inconsistent with data-flow edges. |
| C1-16 | Arrow | `e_more_prev` | Very short arrow after the ellipsis is visually weak. |
| C1-17 | Arrow | top fan-in | The bend into `VA Component1` is tall. |
| C1-18 | Arrow | bottom fan-in | The bend into `VA Component1` is tall. |
| C1-19 | Box | `v0` | Token box is much narrower than its projection box. |
| C1-20 | Box | `a0` | Token box is much narrower than its initializer box. |
| C1-21 | Box | `layer_input` | Height differs strongly from neighboring component boxes. |
| C1-22 | Spacing | left branch | Top and bottom source branches have slightly different y padding. |
| C1-23 | Spacing | `va1`→`va_prev` | Repetition section has more whitespace than other transitions. |
| C1-24 | Color | terminal output | Yellow-green appears only once; acceptable but visually isolated. |
| C1-25 | Color | state boxes | Four pink boxes are separated across a very wide canvas. |
| C1-26 | Typography | equations | Subscript sizes become small at slide scale. |
| C1-27 | Layout | title | Title is centered on canvas rather than the denser information region. |
| C1-28 | Layout | branches | Source branches occupy more vertical space than the main chain. |
| C1-29 | Icons | overall | No icons are used; this is intentional and keeps the figure editable. |
| C1-30 | Style coherence | overall | The palette matches existing figures, but density is lower than the VA component figure. |

Pre-flight WARN `spacing-inconsistent-h` was reviewed: it reflects the ellipsis/repetition construction and will disappear with the compact previous-layers block.

## Fix Verification — Cycle 1

| defects | result | evidence |
|---|---|---|
| C1-01, C1-08, C1-10 | FIXED | Output box widened and right margin restored in cycle 2. |
| C1-02, C1-06 | FIXED | Both initialization branches now terminate at the explicit `Initial VA state V0,A0` box. |
| C1-03, C1-07, C1-09 | FIXED | Ellipsis nodes replaced by one arrowed `Previous VA layers` block. |
| C1-04, C1-05 | FIXED | Canvas reduced from 2180×620 to 1880×600. |
| C1-11–C1-30 | FIXED or superseded | Compact structure removes the long repetition lanes; remaining palette/spacing choices are reviewed below. |

## Screenshot Review Cycle 2

Evidence: `review_history/va-layer-input-cycle2.png`.

### P0 — Blockers

None.

### P1 — Visible defects

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C2-01 | Semantics | `initial_state` | “VA state” can be confused with the separate `Robot state` source. | Rename to `Initial VA tokens`. |
| C2-02 | Semantics | `robot_state` | Learned queries appear inside the sensor/input box although they are model parameters. | Move them into action-token initialization. |
| C2-03 | Semantics | `previous_layers` | `VA1 … VAi−1` can be read as state tensors rather than component instances. | Write `Component1 … Componenti−1`. |
| C2-04 | Color | `layer_output` | Output tokens use yellow-green while all other V/A token states use pink. | Use the token-state pink fill. |
| C2-05 | Arrow | `e_vision_state`, `e_action_state` | `V0/A0` labels sit on vertical bends and add clutter; the target already names both tensors. | Remove the edge labels. |
| C2-06 | Text | subtitle | It does not explicitly state the special first-layer case. | Write “Layer 1 reads V0,A0; later layers read the previous output.” |

### P2 — Polish

| id | zone | element | finding |
|---|---|---|---|
| C2-07 | Text | `vision_init` | Long single-line label is close to the box sides. |
| C2-08 | Text | `action_init` | Long single-line label is close to the box sides. |
| C2-09 | Box | `previous_layers` | It is wider than the current component, intentionally encoding repetition. |
| C2-10 | Box | `layer_output` | It is smaller than the input state because it has fewer explanatory lines. |
| C2-11 | Spacing | title area | Vertical title-to-subtitle gap is slightly larger than subtitle-to-content rhythm. |
| C2-12 | Spacing | branch rails | The two fan-in rails are 10 px apart; acceptable after removing labels. |
| C2-13 | Typography | subscripts | Subscripts are small but remain readable at exported scale. |
| C2-14 | Layout | aspect ratio | Figure remains intentionally wide for the requested horizontal layout. |
| C2-15 | Icons | overall | No icons; intentional because they add no semantic information. |
| C2-16 | Style coherence | overall | Previous-layers box contains an inline arrow sequence while all other flow is connector-based; acceptable shorthand for repetition. |

## Fix Verification — Cycle 2

| defects | result | evidence |
|---|---|---|
| C2-01 | FIXED | `Initial VA tokens` now cannot be confused with the robot measurement box. |
| C2-02 | FIXED | Learned queries moved inside action-token initialization. |
| C2-03 | FIXED | Previous block explicitly says `Component`. |
| C2-04 | FIXED | Layer output now uses the same pink token-state fill. |
| C2-05 | FIXED | Bend labels removed; fan-in is visually clean. |
| C2-06 | FIXED | Subtitle now distinguishes layer 1 from later layers. |

## Screenshot Review Cycle 3

Evidence: `review_history/va-layer-input-cycle3.png`.

### P0 — Blockers

None.

### P1 — Visible defects

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C3-01 | Text | `previous_layers` | Component sequence wraps to three lines and weakens the horizontal rhythm. | Shorten to `Components 1 → ⋯ → i−1`. |
| C3-02 | Text | `action_init` | Explanation wraps to three lines while the vision encoder stays one line. | Use compact formula `queries + Proj(p,u)`. |

### P2 — Polish

| id | zone | element | finding |
|---|---|---|---|
| C3-03 | Text | subtitle | Long explanatory sentence is intentionally retained because it resolves the user’s confusion. |
| C3-04 | Box | `initial_state` | Box has more vertical space than its two-line content; it aligns with the three-line generic input box. |
| C3-05 | Box | `layer_output` | Smaller terminal box remains proportional to its shorter label. |
| C3-06 | Spacing | fan-in rails | Rails are 10 px apart but no longer carry labels or create a false junction. |
| C3-07 | Layout | overall | Wide aspect ratio is intentional for the requested landscape figure. |
| C3-08 | Style | repetition | Inline arrows inside the repeated-layers block are shorthand, not extra data connectors. |

## Fix Verification — Cycle 3

| defects | result | evidence |
|---|---|---|
| C3-01 | FIXED | Previous-layer sequence is now one compact line. |
| C3-02 | FIXED | Action initialization is now a two-line title/formula block. |

## Screenshot Review Cycle 4

Evidence: `review_history/va-layer-input-cycle4.png`.

- P0: 0
- P1: 0
- All labels are readable; all eight arrows have visible forward arrowheads; no connector crosses a box or text.
- The first-layer initialization and later-layer recurrence are both explicit.

## Red-Team Audit

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| RT-01 | Text | Subtitle is smaller than body text. | P2 | Intentional hierarchy. |
| RT-02 | Arrow | Fan-in rails are unlabeled. | P2 | Target explicitly lists `V0,A0`; labels would add clutter. |
| RT-03 | Box | Previous-layers block is wider than one component. | P2 | Width encodes repetition. |
| RT-04 | Spacing | Source branches consume more vertical space than the main chain. | P2 | Required to keep observation and robot state distinct. |
| RT-05 | Color | Pink is used for inputs, intermediate tokens, and output tokens. | P2 | Deliberate token-state color contract. |
| RT-06 | Typography | Mathematical subscripts are smaller at slide scale. | P2 | Export remains readable at native scale. |
| RT-07 | Layout | Aspect ratio is too wide for a single-column paper figure. | P2 | Requested landscape layout; intended for a slide. |
| RT-08 | Icons | No camera/robot icons. | P2 | Omitted because labels already communicate the modalities. |
| RT-09 | Style | Repetition uses inline arrows inside one block. | P2 | Compact shorthand avoids redundant component boxes. |
| RT-10 | Scope | Q/K/V internals are absent. | P2 | Intentional: they remain in the separate VA Component figure. |

## Self-Score

| dimension | score | evidence |
|---|---:|---|
| Text readability | 10/10 | No clipping or overflows in cycle 4. |
| Arrow accuracy | 10/10 | All connectors are forward and collision-free. |
| Color coherence | 10/10 | Three semantic fills plus one dark stroke system. |
| Layout consistency | 9/10 | Wide aspect ratio is optimized for slides, not a paper column. |
| Style match | 9/10 | Matches the existing Transformer pastel style; repetition uses one compact shorthand block. |
| **Total** | **48/50** | Allowed for handoff. |

Final compact validation: visual preflight 0 FAIL / 0 WARN; strict DrawIO validation 0 errors / 0 warnings; no embedded raster or external image.

## User Feedback Repair — Aesthetic Rebuild

User verdict: “非常丑.” This is correct and exposes a limitation of the previous checks: geometry was valid, but the composition still had excessive white space, a visually dominant merge rail, and mixed horizontal/vertical reading grammars.

| id | zone | failure | severity | rebuild rule |
|---|---|---|---|---|
| UF4-01 | Composition | Large empty center band made the figure look unfinished. | P1 | Use two bounded panels with medium density. |
| UF4-02 | Arrow | Long V0/A0 merge rails dominated the content. | P1 | Put `Initial VA tokens` inside the initialization panel and use short local fan-in. |
| UF4-03 | Layout | The eye had to switch from wide horizontal branches to a distant vertical stack. | P1 | Make the panel boundary explicit and use one cross-panel handoff. |
| UF4-04 | Text | Dimension annotations made ordinary boxes feel like debug output. | P1 | Keep only operation-defining formulas and `H slots`. |
| UF4-05 | Style | Equal-width recurrence boxes and large gaps felt like a form, not a Transformer figure. | P1 | Use compact 70–90 px blocks with 30–40 px rhythm. |

Rebuild target: `1320×700`; panel 1 initializes `V0,A0`, panel 2 applies the repeated VA stack. No icons, gradients, shadows, or decorative token bars.

## Screenshot Review Cycle 10 — Aesthetic Rebuild Draft

Evidence: `review_history/va-layer-input-aesthetic-cycle1.png`.

### P0

None.

### P1

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C10-01 | Style | panel outlines | Short gray dashes make the panels feel like annotations rather than Transformer modules. | Use an almost-solid light-gray outline while preserving editable container recognition. |
| C10-02 | Layout | panel bottoms | Both panels retain about 60–90 px of unused bottom space. | Reduce page and panel height; move the query box slightly upward. |
| C10-03 | Box | right stack | 280 px boxes are wider than their two-line labels require. | Reduce to 260 px and keep them centered. |

### P2

| id | zone | finding | disposition |
|---|---|---|---|
| C10-04 | Text | Main title and two panel titles create three heading levels. | Retain; hierarchy is clear. |
| C10-05 | Arrow | Initial-token handoff uses two orthogonal bends. | Required to enter the top stack without crossing a panel title. |
| C10-06 | Spacing | Visual and action rows are separated by a broad band. | Keep enough separation for the upward action path. |
| C10-07 | Color | DINO is the only blue-outlined box. | Retain external-encoder distinction. |
| C10-08 | Icons | No icons are present. | Matches the supplied Transformer reference. |
| C10-09 | Typography | `H slots` is explanatory English rather than a tensor shape. | More readable for a slide. |

## Fix Verification — Cycle 10

| defect | result | evidence |
|---|---|---|
| C10-01 | PARTIAL | `review_history/va-layer-input-aesthetic-cycle2.png` is visually solid, but the long-dash workaround leaves two tiny visible border gaps. |
| C10-02 | FIXED | Canvas height is 40 px shorter and unused bottom space is reduced. |
| C10-03 | FIXED | Right-stack boxes are 260 px and remain centered. |

## Screenshot Review Cycle 11 — Border and Rhythm

Evidence: `review_history/va-layer-input-aesthetic-cycle2.png`.

### P0

None.

### P1

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C11-01 | Box | panel outlines | The near-solid dash pattern leaves a visible gap in the left bottom border and right left border. | Make the dash length larger than either panel perimeter. |

### P2

| id | zone | finding | disposition |
|---|---|---|---|
| C11-02 | Arrow | Cross-panel handoff touches both panel boundaries. | Correct: it is the only relationship between panels. |
| C11-03 | Spacing | Visual row sits higher than the initial-token box. | The short bend expresses visual fan-in without diagonal arrows. |
| C11-04 | Typography | Panel numbers use `1 ·` / `2 ·` rather than `(a)` / `(b)`. | More natural for a teaching slide. |
| C11-05 | Layout | Initial-token box is the largest item in panel 1. | Deliberate focal point for the two fan-ins. |
| C11-06 | Color | Pink appears five times in the initialization panel. | It consistently denotes raw inputs or token states. |

## Fix Verification — Cycle 11

| defect | result | evidence |
|---|---|---|
| C11-01 | FIXED | `review_history/va-layer-input-aesthetic-cycle3.png` shows continuous panel outlines with no artificial gaps. |

## Screenshot Review Cycle 12 — Final Aesthetic Gate

Evidence: full canvas `review_history/va-layer-input-aesthetic-cycle3.png`; initialization crop `review_history/va-layer-input-aesthetic-focus.png`.

- P0: 0.
- P1: 0.
- Visual and action token construction are balanced within one panel.
- All fan-ins are short, local, and readable.
- Only one connector crosses the panel boundary.
- No text clipping, false junction, or decorative element remains.

### Residual P2

| id | zone | finding | disposition |
|---|---|---|---|
| C12-01 | Text | Labels remain English. | Matches the other method diagrams. |
| C12-02 | Arrow | Action construction reads partly upward. | Necessary to show two inputs entering one Add block; arrowheads are explicit. |
| C12-03 | Spacing | The initial-token box is offset below the visual row. | Creates a compact two-input merge without a long bus. |
| C12-04 | Box | Panel 1 is about twice the width of panel 2. | Reflects its greater number of implemented operations. |
| C12-05 | Style | Panel outlines are light gray rather than black. | Prevents the grouping frame from competing with component borders. |

## Red-Team Audit — Aesthetic Rebuild

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| RT5-01 | Text | `H slots` is conceptual rather than a full tensor shape. | P2 | Deliberate simplification. |
| RT5-02 | Text | Main title repeats words used in panel titles. | P2 | Useful when the figure is exported without surrounding prose. |
| RT5-03 | Arrow | Visual projection uses one short bend before the initial-token box. | P2 | Avoids a diagonal connector. |
| RT5-04 | Arrow | Query and state arrows meet the Add block from different axes. | P2 | This directly communicates the two operands. |
| RT5-05 | Box | Initial-token box is taller than the visual blocks. | P2 | It is a two-stream merge and focal point. |
| RT5-06 | Spacing | Visual and action paths use different vertical baselines. | P2 | Separates modalities and removes crossings. |
| RT5-07 | Color | Projection lavender appears twice while DINO blue appears once. | P2 | Semantic color reuse is correct. |
| RT5-08 | Typography | Panel headings are almost as bold as the main title. | P2 | Size difference maintains hierarchy. |
| RT5-09 | Layout | Cross-panel arrow enters the top stack box from the left rather than above. | P2 | Makes the panel handoff explicit. |
| RT5-10 | Icons | The figure has no pictograms. | P2 | Correct for the supplied Transformer box-and-arrow reference. |

## Self-Score — Aesthetic Rebuild

| dimension | score | evidence |
|---|---:|---|
| Text readability | 10/10 | All labels remain legible in full and focused exports. |
| Arrow accuracy | 10/10 | Thirteen arrows are directed, local, and collision-free. |
| Color coherence | 10/10 | Pastel role colors follow the extracted Transformer contract. |
| Layout consistency | 9/10 | Panel widths differ because initialization has more operations. |
| Style match | 9/10 | The grouped modules match the Transformer grammar; panel titles are a task-specific addition. |
| **Total** | **48/50** | Allowed for handoff. |

## Final Validation

- `validate_visual_quality.py`: 0 FAIL, 0 WARN.
- `validate_drawio.py --strict`: 0 errors, 0 warnings.
- Editable primitives only; no embedded raster or external image.

## User Feedback Repair — Vision Projection

User-found defect, missed by the previous audit: `DINO + Vision projection` was compressed into one box, so the reader could not see what tensor enters the projection, what operation is learned, or what changes.

| id | zone | element | finding | severity | required repair |
|---|---|---|---|---|---|
| UF-01 | Semantics | visual initialization | DINO feature extraction and the trainable linear adapter are conflated. | P1 | Split into `DINO patch tokens D_t` and `Linear adapter`. |
| UF-02 | Text | visual projection | No equation defines the output. | P1 | Add `V0 = Linear(D_t)`. |
| UF-03 | Semantics | dimensions | The figure does not show that token count is preserved while feature width changes. | P1 | Add `N×d_DINO → N×d_VA`. |
| UF-04 | Layout | initialization branches | Visual initialization becomes more detailed than action initialization. | P1 | Split action branch symmetrically into state projection and query addition. |

Correction target: `Observation → DINO patch tokens → Linear adapter → V0`; `Robot state → State projection → learned queries → A0`.

## Screenshot Review Cycle 5 — Vision Projection Split

Evidence: `review_history/va-layer-input-vision-cycle1.png`.

### P0

None.

### P1

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C5-01 | Semantics | `vision_projection` | `d_VA` is understandable but the implementation contract is `vision_dim → hidden_dim`. | Rename output width to `d_hidden`. |
| C5-02 | Text | `vision_projection` | “Linear adapter” does not explicitly show that its weights are learned. | Rename to `Learned Linear adapter`. |

### P2

| id | zone | element | finding | disposition |
|---|---|---|---|---|
| C5-03 | Text | `dino_tokens` | DINO implementation details are intentionally omitted. | Keep overview-level label. |
| C5-04 | Arrow | fan-in | Two clean rails feed `Initial VA tokens`. | Keep; no false junction. |
| C5-05 | Box | `vision_projection` | Three lines make this box denser than observation. | Required to explain operation and dimensions. |
| C5-06 | Color | `dino_tokens` | Blue stroke appears only on the DINO block. | Deliberately marks external visual features. |
| C5-07 | Layout | overall | Canvas grew from 1880 to 1940 px. | Acceptable for the requested landscape slide. |
| C5-08 | Scope | visual path | Optional frame embedding is omitted. | Keep out of the main principle figure. |

## Fix Verification — Cycle 5

Evidence: full canvas `review_history/va-layer-input-vision-cycle2.png`; focused visual-path crop `review_history/va-layer-input-vision-focus.png`.

| defects | result | evidence |
|---|---|---|
| UF-01 | FIXED | DINO encoding and learned projection are separate boxes with a directed edge. |
| UF-02 | FIXED | Projection box explicitly defines `V0 = Linear(Dt)`. |
| UF-03 | FIXED | Projection box shows `N×d_DINO → N×d_hidden`; `N` is unchanged. |
| UF-04 | FIXED | Action branch now mirrors the visual branch with state projection and learned action queries. |
| C5-01 | FIXED | Output width now uses implementation-aligned `d_hidden`. |
| C5-02 | FIXED | Box title now says `Learned Linear adapter`. |

## Red-Team Audit — After User Feedback Repair

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| RT2-01 | Text | Dimension line is smaller and gray. | P2 | Intentional explanatory hierarchy; readable in focus crop. |
| RT2-02 | Arrow | Visual and action fan-in rails are only 10 px apart. | P2 | They remain separate and have distinct target ports. |
| RT2-03 | Box | Projection box is 10 px taller than the DINO box. | P2 | Required by its additional dimension line. |
| RT2-04 | Spacing | Initialization takes more horizontal space than before. | P2 | Required to expose the missing operation. |
| RT2-05 | Color | DINO is the only blue-outlined box. | P2 | Marks external visual features consistently. |
| RT2-06 | Typography | Subscripts become small when the full figure is reduced. | P2 | Native export and slide-scale crop remain readable. |
| RT2-07 | Layout | Figure is unsuitable for a narrow paper column. | P2 | Intended as a horizontal presentation figure. |
| RT2-08 | Icons | No image/camera icon identifies observation. | P2 | Text label is clearer and fully editable. |
| RT2-09 | Scope | Optional temporal frame embedding is not drawn. | P2 | Deliberately omitted from this principle-level figure. |
| RT2-10 | Style | No trainable/frozen legend is present. | P2 | “Learned” is written directly on the only relevant adapter. |

No P0 or P1 finding remains.

## Updated Self-Score

| dimension | score | evidence |
|---|---:|---|
| Text readability | 10/10 | Full and focused exports show no clipping or overlap. |
| Arrow accuracy | 10/10 | All ten edges are forward and collision-free. |
| Color coherence | 10/10 | Pastel role colors remain consistent with the Transformer palette. |
| Layout consistency | 9/10 | Wide layout is slide-optimized rather than paper-column optimized. |
| Style match | 9/10 | Dimension annotation adds necessary density but remains within the established style. |
| **Total** | **48/50** | Allowed for handoff. |

Final validation: visual preflight 0 FAIL / 0 WARN; strict DrawIO validation 0 errors / 0 warnings; no embedded raster or external image.

## User Feedback Repair — Action Concatenation

User-found defect, missed by the preceding audits: the action branch showed the correct equations but did not visually distinguish the two operations.

| id | zone | element | finding | severity | required repair |
|---|---|---|---|---|---|
| UF2-01 | Semantics | robot/action input | `p_t` and `u_{t-1}` appear in one source box, hiding the feature-axis concat. | P1 | Draw them as two input boxes feeding `Concat`. |
| UF2-02 | Semantics | learned queries | Query addition is only written as a formula and can be mistaken for another concat. | P1 | Draw learned queries as a separate input to `Broadcast + Add`. |
| UF2-03 | Shapes | action tensors | The action-horizon axis `H` is not shown. | P1 | Label query and output shapes `H×d_hidden` and `B×H×d_hidden`. |
| UF2-04 | Arrow | action fan-in | The two distinct fan-ins are not structurally visible. | P1 | Use separate ports for Concat and Add. |

Correction target: `[p_t;u_{t-1}] --Linear→ h_t`; `A0 = Q_A + broadcast_H(h_t)`.

## Screenshot Review Cycle 6 — Explicit Action Construction

Evidence: full canvas `review_history/va-layer-input-action-cycle1.png`; focused action crop `review_history/va-layer-input-action-cycle1-focus.png`.

### P0

None.

### P1

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C6-01 | Spacing | left input column | The first preflight grouped the observation with the two action inputs and reported inconsistent vertical gaps. | Shift the observation 20 px right; semantic rows remain unchanged and preflight becomes clean. |

### P2

| id | zone | element | finding | disposition |
|---|---|---|---|---|
| C6-02 | Text | `Concat` | The feature dimension is stated in English rather than with a `dim=-1` code literal. | Keep concept-level wording for the slide. |
| C6-03 | Text | `Broadcast + Add` | Broadcasting is compactly written as `broadcast_H`. | Explain verbally that the same `h_t` is copied to all H slots. |
| C6-04 | Layout | action branch | Two stacked inputs make the action branch taller than the visual branch. | Required to expose the two distinct input tensors. |
| C6-05 | Layout | learned queries | Query box sits above the add box. | This makes the second fan-in explicit without crossing the main action path. |
| C6-06 | Typography | tensor shapes | Shape annotations use smaller gray text. | Intentional explanatory hierarchy. |
| C6-07 | Scope | previous action | The source of `u_{t−1}` is not expanded. | Keep out of this VA-initialization figure. |
| C6-08 | Scope | robot state | Proprioception dimensions are abstracted as `d_p`. | Concrete joint/gripper fields belong in accompanying text. |

## Fix Verification — Action Concatenation

| defects | result | evidence |
|---|---|---|
| UF2-01 | FIXED | `p_t` and `u_{t−1}` are separate pink inputs with two arrows into `Concat · feature dim`. |
| UF2-02 | FIXED | `Q_A` is a separate input and joins `h_t` only at `Broadcast + Add`. |
| UF2-03 | FIXED | Query and output shapes show `H×d_hidden` and `B×H×d_hidden`. |
| UF2-04 | FIXED | Concat and Add use separate, visible fan-in ports with no false junction. |
| C6-01 | FIXED | Final visual preflight reports 0 FAIL / 0 WARN. |

## Red-Team Audit — Action Construction

| id | check | result |
|---|---|---|
| RT3-01 | `p_t` and `u_{t−1}` are not added together. | PASS — both enter the Concat block. |
| RT3-02 | Concatenation is along the feature axis. | PASS — label and output shape are explicit. |
| RT3-03 | State projection occurs after concatenation. | PASS — directed arrow order is unambiguous. |
| RT3-04 | Learned queries are not concatenated with `h_t`. | PASS — they meet at Broadcast + Add. |
| RT3-05 | One projected state conditions all H query slots. | PASS — `broadcast_H(h_t)` is explicit. |
| RT3-06 | Batch and horizon axes are visible. | PASS — final shape is `B×H×d_hidden`. |
| RT3-07 | Visual and action branches remain separate until `V0,A0`. | PASS — distinct target ports and rails. |
| RT3-08 | All 13 arrows have valid source and target cells. | PASS — strict XML validation. |
| RT3-09 | No text clipping, overlap, or false junction is visible. | PASS — full and focused canvas review. |
| RT3-10 | Diagram stays editable. | PASS — 0 embedded raster and 0 external images. |

## Final Validation — Action Repair

- `validate_visual_quality.py`: 0 FAIL, 0 WARN.
- `validate_drawio.py --strict`: 0 errors, 0 warnings.
- Editable primitives only; no embedded raster or external image.

## User Feedback Repair — Compact Slide Layout

Found by user, missed by the preceding self-supervision: the `2160×650` canvas is technically readable at native size but becomes a thin strip when placed on a presentation slide.

| id | zone | element | finding | severity | required repair |
|---|---|---|---|---|---|
| UF3-01 | Layout | full canvas | Aspect ratio ≈3.32 is too wide for a normal slide. | P1 | Recompose near 16:9. |
| UF3-02 | Typography | full canvas | Slide scaling makes formulas and subscripts unnecessarily small. | P1 | Preserve native font sizes by reducing width rather than shrinking content. |
| UF3-03 | Flow | layer recurrence | A horizontal recurrence chain consumes most of the width. | P1 | Fold recurrence into a top-to-bottom column. |
| UF3-04 | Composition | initialization vs recurrence | Input construction and layer flow compete for one horizontal baseline. | P1 | Use two semantic columns: initialization left, recurrence right. |

Correction target: `1420×830`; left side constructs `V0,A0`, right side reads from top and emits `Vi,Ai` at the bottom.

## Screenshot Review Cycle 7 — Compact Layout Draft

Evidence: `review_history/va-layer-input-compact-cycle1.png`.

### P0

None.

### P1

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C7-01 | Text | `proprio` | “Current proprioception” sits too close to both side borders. | Widen both source boxes from 170 to 180 px. |
| C7-02 | Arrow | merge rails | The action-entry rail is only 4 px below the visual source rail near the initial-token box, creating a near-double line. | Move the action entry from 0.7 to 0.8 of the target height. |

### P2

| id | zone | element | finding | disposition |
|---|---|---|---|---|
| C7-03 | Box | right column | Equal 260 px widths leave extra space in the short `VA Component` label. | Keep equal widths to make recurrence readable as one column. |
| C7-04 | Spacing | center | White space separates visual and action branches. | Preserve modality separation. |
| C7-05 | Color | `dino_tokens` | Blue stroke is unique. | Retain the existing external-feature color contract. |
| C7-06 | Typography | shape lines | Tensor dimensions are smaller gray text. | Preserve hierarchy; readable at native export. |
| C7-07 | Layout | visual route | The V0 connector is intentionally longer than local action arrows. | It crosses no box and visually closes at the initial-token column. |
| C7-08 | Icons | all | No robot or camera icons. | Keep the Transformer-like box language simple. |
| C7-09 | Style | recurrence | Flow turns from horizontal initialization to vertical recurrence. | This fold is the requested compact-layout mechanism. |

## Fix Verification — Cycle 7

| defect | result | evidence |
|---|---|---|
| C7-01 | FIXED | `review_history/va-layer-input-compact-cycle2.png` shows comfortable side padding around “Current proprioception”. |
| C7-02 | FIXED | The two merge paths now have 16 px vertical separation and remain visually distinct. |

## Screenshot Review Cycle 8 — Compact Layout Tightening

Evidence: `review_history/va-layer-input-compact-cycle2.png`.

### P0

None.

### P1

| id | zone | element | finding | fix |
|---|---|---|---|---|
| C8-01 | Layout | recurrence column | The right column can move 50 px left without touching the action initializer; the remaining span still makes the composition wider than necessary. | Shift the complete recurrence column and its merge rails left, then reduce the canvas width. |

### P2

| id | zone | element | finding | disposition |
|---|---|---|---|---|
| C8-02 | Text | subtitle | Subtitle is a full sentence. | Keep because it resolves the first-layer/later-layer distinction. |
| C8-03 | Arrow | visual merge | Visual connector remains the longest edge. | Shorten with the column shift; do not add a noisy label. |
| C8-04 | Box | `initial_state` | Taller than later output box. | It receives two independently constructed token streams. |
| C8-05 | Spacing | right column | Uniform 40 px vertical gaps are intentionally generous. | Retain for slide readability. |
| C8-06 | Style | output direction | Output exits at the bottom rather than the right. | Retain; matches the requested compact top-to-bottom reading. |

## Fix Verification — Cycle 8

| defect | result | evidence |
|---|---|---|
| C8-01 | FIXED | `review_history/va-layer-input-compact-cycle3.png` shows the recurrence column 50 px closer and a canvas reduced to `1370×830`; no font was reduced. |

## Screenshot Review Cycle 9 — Final Compact Gate

Evidence: full canvas `review_history/va-layer-input-compact-cycle3.png`; recurrence crop `review_history/va-layer-input-compact-focus.png`.

- P0: 0.
- P1: 0.
- All labels remain readable at slide scale.
- Both initialization paths enter distinct ports; the visual and action rails do not touch.
- The right-side flow is strictly top-to-bottom and finishes at `V_i,A_i`.

### Residual P2

| id | zone | finding | disposition |
|---|---|---|---|
| C9-01 | Text | Labels remain English while discussion is Chinese. | Retain consistency with the other method figures. |
| C9-02 | Arrow | Visual initialization still owns the longest edge. | It is clean and expresses V0 fan-in without another box. |
| C9-03 | Spacing | Action initialization is denser than visual initialization. | Required by its two distinct operations. |
| C9-04 | Icons | No modality icons are present. | Intentional minimal Transformer style. |
| C9-05 | Layout | Canvas ratio ≈1.65 is slightly narrower than 16:9. | Better for readable placement inside a 16:9 slide. |

## Red-Team Audit — Compact Layout

| id | zone | residual finding | severity | disposition |
|---|---|---|---|---|
| RT4-01 | Text | Subtitle becomes the smallest text on the slide. | P2 | Deliberate hierarchy; still readable. |
| RT4-02 | Text | `broadcast_H` assumes the reader understands broadcasting. | P2 | Formula is paired with the operation title. |
| RT4-03 | Arrow | Visual connector has two bends. | P2 | Necessary to reach the upper input port without crossing the action rail. |
| RT4-04 | Arrow | Merge arrows are unlabeled. | P2 | Their source formulas already identify V0 and A0. |
| RT4-05 | Box | Right-column boxes use one width despite different text lengths. | P2 | Width consistency encodes a single recurrence chain. |
| RT4-06 | Spacing | Top visual row and bottom action row leave a broad center band. | P2 | Separates modalities and keeps arrows uncrossed. |
| RT4-07 | Color | Pink represents both source data and token states. | P2 | Matches the established token/input palette. |
| RT4-08 | Typography | Subscripts are smaller than accompanying symbols. | P2 | Standard mathematical hierarchy. |
| RT4-09 | Layout | The recurrence column is visually heavier than either single initialization branch. | P2 | It intentionally carries the temporal/layer narrative. |
| RT4-10 | Style | No dashed containers divide the two regions. | P2 | Avoids adding decorative boxes to a simple figure. |

## Self-Score — Compact Layout

| dimension | score | evidence |
|---|---:|---|
| Text readability | 10/10 | All text is readable in the full native export; no clipping. |
| Arrow accuracy | 10/10 | All 13 edges are directed, separated, and collision-free. |
| Color coherence | 10/10 | Existing Transformer pastel role colors are unchanged. |
| Layout consistency | 9/10 | Initialization branches have intentionally unequal density. |
| Style match | 9/10 | Compact folded flow differs from the original straight Transformer silhouette but matches its visual language. |
| **Total** | **48/50** | Allowed for handoff. |

Final aesthetic replacement: `va_layer_input_origin.drawio` validates with 0 FAIL / 0 WARN and 0 strict errors / 0 strict warnings; final PNG is `va_layer_input_origin.png`.
