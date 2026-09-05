# VA-WAM Main Figure — Diagram Brief

## User Goal

- Output: editable `.drawio` plus a clean PNG/SVG preview.
- Audience: robotics / machine-learning paper readers.
- Must communicate: three encoded inputs; one shared pre-stage snapshot; parallel VA and WAM computation; Transformer-like sublayers; gated merge; eight peer stages; one VA/flow action emitter; training-only world loss.
- Must not do: ambiguous arrows, serial VA→WAM flow, a WAM action head, candidate actions, future-target leakage, H48/7-D/V-JEPA labels, decorative shapes without semantics.

## Source Inventory

| id | source | type | role | priority | notes |
|---|---|---|---|---|---|
| S1 | `va_compound/model.py` | code | VA hierarchy and peer-stage structure | must | VA and WAM read the same pre-stage snapshot |
| S2 | `va_compound/wmrm.py` | code | WAM hierarchy and output semantics | must | WAM predicts world state and gated latent deltas; it is not an action head |
| S3 | `scripts/run_mw_hard2_wam4va_visualmotion_peer_sync_h6_v1.sh` | config | active dimensions | must | VA ×8, ST predictor ×6, H6×4, Euler ×8 |
| S4 | `codex-clipboard-3859853e-ac46-44b3-941f-9f771689ccac.png` | image | style and hierarchy reference | must | palette, rounded black strokes, nested blocks, portrait flow |
| S5 | `va_wam_main_v2.png` | image | content-density reference and defect source | should | retain useful hierarchy; replace every ambiguous connector |

## Requirement Traceability

| id | requirement | source evidence | level | planned encoding |
|---|---|---|---|---|
| R1 | VA and WAM are parallel peers | S1 | must | two towers fed independently from one pre-stage snapshot |
| R2 | VA uses shared V/A multi-head attention and FFNs | S1 | must | Pre-Norm → Shared MHA → Residual Add → Pre-Norm → Feed Forward → Residual Add |
| R3 | WAM updates evidence/belief and predicts a DINO latent map | S2 | must | Evidence CA → Belief/Innovation → ST predictor → World Tokens |
| R4 | ST block uses SA, condition CA, and FFN | S2 | must | three explicit Pre-Norm + sublayer + Residual groups, nested ×6 |
| R5 | Only gated ΔV/ΔA enter the VA merge | S1/S2 | must | separate `Gated ΔV, ΔA` and `World Stateᵢ` outputs |
| R6 | Final action comes only from VA/flow | S1/S2/S3 | must | Stage Commit → Layer Norm → Flow Transformer ×6 → Euler ×8 → H6×4 |
| R7 | Future DINO is training-only | S2 | must | dashed target→loss arrow; no target edge to forward path |
| R8 | every edge has a clear source and target | user feedback | must | orthogonal connectors, one arrowhead at target, no outer feedback loop |

## Semantic Model and Connector Contract

| id | source → target | meaning | cardinality / style |
|---|---|---|---|
| C01 | Instruction → Qwen Language Cache | language encoding | solid, upward |
| C02 | 4-frame Observation → DINO Visual Embedding | visual encoding | solid, upward |
| C03 | Robot State + Previous Action → State–Action Embedding | state/action encoding | solid, upward |
| C04 | each embedding → Pre-stage Snapshot Sᵢ₋₁ | snapshot assembly | three separate solid edges |
| C05 | Snapshot → VA Layer | parallel VA proposal input | solid, upward |
| C06 | Snapshot → WAM Stage | parallel WAM proposal input | solid, upward |
| C07 | VA sublayer input → matching Residual Add | local residual | two short left-side loops |
| C08 | WAM ST sublayer input → matching Residual Add | local residual | three short left-side loops |
| C09 | Condition K/V → Conditional Cross-Attention | WAM condition read | solid side edge |
| C10 | VA Proposal + Gated ΔV/ΔA → Gated Merge | latent visual/action commit | two-input fan-in |
| C11 | Gated Merge + World Stateᵢ → Stage Commit Sᵢ | stage state commit | two-input fan-in |
| C12 | Stage Commit → Layer Norm → Flow Transformer → Euler → Action Chunk | only physical action path | solid, upward |
| C13 | Predicted World Tokens → Predicted Next DINO Map → World Loss | world prediction supervision | solid |
| C14 | Future DINO Target → World Loss | stop-gradient training target | dashed |

No large return arrow is drawn. Repetition is encoded by the `Peer Stage × 8` container and the explicit `Sᵢ₋₁`/`Sᵢ` labels, matching standard Transformer stack notation without creating a second ambiguous path.

## Style Extraction: Transformer Reference

### 1. Palette

| role | hex | used on |
|---|---|---|
| background | `#DBDBDB` | full page |
| primary fill | `#A6C5D1` | attention, visual embeddings, world tokens |
| secondary fill | `#D7BE9E` | WAM attention/predictor modules |
| accent / residual | `#CFCFA5` | Pre-Norm, Residual Add, merge/commit |
| input fill | `#D6BEBE` | raw inputs and future target |
| flow fill | `#BABECB` | flow transformer and sampler |
| output fill | `#AFC3AF` | action chunk |
| border / arrow | `#000000` | boxes and connectors |
| text | `#111111` | all labels |
| muted | `#D0D0D1` | group fills / interior background |

Total distinct semantic fills: 8.

### 2. Typography

- Heading: Helvetica, 24 pt, bold.
- Tower/subheading: Helvetica, 21 pt, bold.
- Module text: Helvetica, 18 pt, regular.
- Small annotation: Helvetica, 14 pt, regular.
- No monospace labels.

### 3. Shape Language

- Corner radius: about 10 px (`rounded=1; arcSize=12`).
- Box stroke: 3 px.
- Arrow stroke: 3 px.
- Group/container stroke: 3 px, solid.
- Shadow: none.
- Group fill opacity: 18–28% equivalent via light gray fills.

### 4. Layout Rhythm

- Canvas: 1200×1760 portrait.
- Outer margin: 40 px.
- Major vertical gaps: 16–24 px.
- Same-row horizontal gaps: 24–40 px.
- Box padding: 10–14 px vertical, 16–20 px horizontal.
- Typical module: 260–360 px wide × 40–70 px high.
- Alignment grid: 10 px.

### 5. Arrow Grammar

- Arrowhead: filled classic, medium.
- Stroke: `#000000`, 3 px.
- Routing: vertical or orthogonal only.
- One arrowhead per connector, at the target only.
- Local residuals remain inside their own tower/sub-container.
- Training-only target edge is dashed; all inference/data edges are solid.

### 6. Icon Language

- No icons. The reference relies on labeled primitives and arrows.

### 7. Density and Composition

- Diagram type: nested two-tower Transformer architecture with fan-in/fan-out.
- Major regions: input encoding, peer stage, primary action output, world-loss branch.
- Density: dense but scan-friendly.
- Whitespace: moderate and regular.
- Legend/caption: none inside the figure.

## Semantic Justification

| element | visual form | represents | each unit corresponds to | justified? |
|---|---|---|---|---|
| nested VA stack | rounded module stack | one VA coupling layer | one real pre-norm/attention/FFN sublayer | yes |
| nested ST stack ×6 | rounded repeated stack | deep world predictor | one real predictor block | yes |
| three input boxes | colored rounded boxes | instruction, video window, robot state/action | one actual input modality | yes |
| pastel color categories | semantic fills | input/VA/WAM/residual/flow/output roles | one functional category | yes |
| residual loops | short orthogonal connectors | residual addition around one sublayer | one exact residual relation | yes |
| large token grids/icons | omitted | no extra information needed | n/a | deleted as decoration |

## Open Assumptions

| assumption | risk | verification |
|---|---|---|
| English labels are preferred for the paper figure | low | follows the supplied reference and previous drafts |
| `Flow Transformer ×6` is sufficient abstraction for the flow head | low | active runner config and user requested module hierarchy, not every flow sublayer |
| no explicit recurrence arrow is clearer than a large loop | low | `Sᵢ₋₁`, `Sᵢ`, and `×8` encode recurrence without ambiguous routing |
