# Synthesized Style Extraction

Sources: the user-provided Transformer figure for palette and the two approved simple AI sketches for density/layout only.

## 1. Palette
| role | hex | used on |
|---|---|---|
| background | `#FFFFFF` | canvas |
| input | `#FFDFDF` | VA visual/action inputs |
| attention / VA | `#FFE1B6` | attention and VA peer |
| normalization / shared state | `#F2F4BB` | Add & Norm, snapshot, commit |
| representation / FFN / WM | `#B7E9F9` | FFN, outputs, WM peer |
| delayed WM accent | `#4D78B8` | WM→VA arrow and label |
| same-stage VA accent | `#E87514` | VA→WM arrow and label |
| border / main arrows | `#111111` | boxes and structural arrows |
| body text | `#222222` | labels |

Total distinct colors: 9, with seven semantic fills/accents plus background and neutral stroke.

## 2. Typography
- Heading font: Helvetica, 24 pt, bold.
- Subheading font: Helvetica, 18–20 pt, bold.
- Body text: Helvetica, 16–18 pt.
- Small label: Helvetica, 14 pt.
- Code / mono font: none.

## 3. Shape Language
- Corner radius: 10–12 px.
- Box stroke: 2.5 px.
- Arrow stroke: 2.5 px; colored exchange arrows: 4 px.
- Dash pattern: `6 5` only for the delay pill if needed.
- Shadow: no.
- Fill opacity: 100%.

## 4. Layout Rhythm
- Outer margin: 35–45 px.
- Major-region gap: 60–80 px.
- Same-row gap: 60–80 px.
- Box padding: 14 px vertical, 18 px horizontal.
- Typical block: 120–520 px wide, 90–320 px high.
- Grid: 10 px.

## 5. Arrow Grammar
- Default arrow: classic filled arrowhead.
- Routing: orthogonal.
- Main arrow color: `#111111`.
- Relation colors: orange = same-stage VA→WM; blue = delayed WM→VA.
- Labels: separate text cells, 16–18 pt.

## 6. Icon Language
- Icons: none. Text boxes and connector semantics are sufficient.

## 7. Density & Composition
- Diagram type: sparse landscape architecture / interaction schematic.
- Major regions: 6 in VA; 4 in interaction.
- Whitespace: generous.
- Panel labels, legend, embedded caption: none.

## Semantic Justification
| element | visual form | method meaning | each unit corresponds to | justified? |
|---|---|---|---|---|
| V/M/A/L/WM strip | five labeled chips | actual attention K/V source groups | one chip = one source family | YES |
| orange connector | thick right arrow | same-snapshot VA evidence/action passed to WM | one arrow = grouped transfer | YES |
| blue connector | thick left arrow | previous WM map projected into next VA K/V | one arrow = delayed transfer | YES |
| residual bypasses | thin orthogonal lines | Transformer residual additions | one path = one residual sublayer | YES |

## Simplification Pass — Transformer Reference `codex-clipboard-f1cc…png`

### 1. Palette
| role | hex | used on |
|---|---|---|
| background | `#FFFFFF` | canvas |
| group background | `#F3F3F3` | module container |
| attention / VA | `#FFE2B2` | attention and VA blocks |
| representation / WM | `#B7E9FB` | WM and feature blocks |
| input / target | `#FFDFDF` | snapshot inputs and targets |
| linear / projection | `#D9DFF2` | Q/K/V projections |
| state / add | `#F1F4B9` | snapshot and commit |
| output / gate | `#C3E9CC` | output or optimizer step |
| border / arrows / text | `#262626` | all neutral structure |
| muted text | `#646464` | short notes only |

Total distinct colors: 10 including white and two neutrals; six semantic pastel fills.

### 2. Typography
- Heading: Helvetica/Arial, 18–20 pt, bold.
- Subheading: Helvetica/Arial, 15–16 pt, bold.
- Body: Helvetica/Arial, 13–15 pt.
- Edge label / caption: Helvetica/Arial, 11–12 pt.
- Monospace: none.

### 3. Shape Language
- Corner radius: subtle, 6–8 px.
- Box stroke: 2 px; arrow stroke: 2 px.
- Group containers: 2 px dark stroke, no shadow.
- Fill opacity: 100%; no gradient.

### 4. Layout Rhythm
- Outer margin: 30–40 px at manuscript scale.
- Major-region gap: 45–60 px.
- Component gap: 20–30 px.
- Box padding: 10–12 px vertical, 14–18 px horizontal.
- Typical component: 220–420 px wide × 55–85 px high.
- Grid: 10 px.

### 5. Arrow Grammar
- Classic filled arrowheads, medium size.
- Orthogonal or straight routing; loops only for true residual paths.
- Neutral arrows `#262626`; orange/blue only for VA→WM / WM→VA semantics.
- Labels are short tensor names, 11–12 pt.

### 6. Icon Language
- No icons. The reference communicates through boxes and arrows only.

### 7. Density & Composition
- Layered component stacks; medium density; moderate whitespace.
- Named panels, no legend, no embedded caption.
- One idea per box and at most two text lines per box.

### Semantic Justification
| element | visual form | method meaning | each unit corresponds to | justified? |
|---|---|---|---|---|
| snapshot / commit bars | pale state boxes | atomic peer state before/after one stage | one bar = one stage boundary | YES |
| Q and K/V projection boxes | lavender boxes | actual learned attention projections | one box = one projection family | YES |
| orange / blue connectors | colored arrows | VA→WM read and delayed WM→VA publication | one arrow = one causal message | YES |
| WM four-block stack | cyan boxes | evidence, belief, prediction, map fusion | one block = one implemented transform | YES |
| VA / WM loss boxes | peach / cyan boxes | actual training objectives | one box = one separately backpropagated objective | YES |
| decorative token bars or icons | omitted | no additional method entity | n/a | NO — omitted |

## Symbolic Transformer Architecture Repair

Source: the original *Attention Is All You Need* Figure 1 and the user's Transformer screenshot. This section supersedes the earlier flowchart grammar for the WM component only.

### 1. Palette

| role | hex | used on |
|---|---|---|
| canvas | `#FFFFFF` | background |
| input/token | `#FFDFDF` | visual and action tokens |
| attention | `#FFE2B2` | Cross-Attention |
| projection | `#D9DFF2` | readout / encoder |
| state/add | `#F1F4B9` | Gated Add & Norm |
| WM representation | `#B7E9FB` | belief and predictor |
| final state | `#C3E9CC` | post-map belief |
| neutral | `#262626` | borders, arrows, text |
| muted | `#646464` | Q/K/V and small notes |

Total distinct colors: 9 including canvas and two neutrals.

### 2. Typography

- Heading: Helvetica, 22 pt, bold.
- Tower title: Helvetica, 17 pt, bold.
- Module: Helvetica, 15–16 pt, bold.
- Token/operator label: Helvetica, 12–14 pt.
- Edge label: Helvetica, 11–12 pt.

### 3. Shape Language

- Module corner radius: 7–8 px.
- Module and arrow stroke: 2 px.
- Operator nodes: 42–46 px circles containing `−`, `⊥`, or `+`-style gate symbols.
- Token cells: 32–42 px × 26–30 px; every cell names a real token/slot.
- Residual paths: thin orthogonal loops; no giant explanatory cards.
- Shadow / gradient: none.

### 4. Layout Rhythm

- Canvas: about `1320×760`.
- Two vertical towers with a 70–90 px inter-tower gap.
- Module gaps: 28–40 px.
- Token-cell gaps: 2–4 px.
- Outer margin: 35–45 px.
- Inputs at the top; committed state at the bottom.

### 5. Arrow Grammar

- Classic filled arrowheads; orthogonal routing.
- `Q`, `K,V` labels sit beside attention ports.
- Dashed blue = prior-map refinement only.
- Residual old-belief paths bypass the transform and enter Gated Add & Norm.
- No prose on arrows.

### 6. Icon Language

- No pictograms.
- Symbol system is limited to token strips, spatial patch grids, operator circles, residual loops, and `×L` repetition marks.

### 7. Density & Composition

- Diagram type: paired Transformer-like towers.
- Left tower: evidence and belief correction.
- Right tower: spatiotemporal prediction and map-conditioned belief.
- Medium density, moderate whitespace, no legend, no caption.

### Semantic Justification

| element | visual form | method meaning | each unit corresponds to | justified? |
|---|---|---|---|---|
| `V_{i−1}` strip | four labeled cells | VA visual token sequence | one cell = one visual token / ellipsis endpoint | YES |
| `B_{i−1}` strip | three labeled cells | persistent belief slots | one cell = one belief slot / ellipsis endpoint | YES |
| `D_t` and `Z_i` grids | four labeled patch cells | current and predicted DINO spatial maps | one cell = one spatial patch / ellipsis endpoint | YES |
| `û^{1:H}` strip | four labeled cells | candidate executable action chunk | one cell = one horizon action / ellipsis endpoint | YES |
| `−` node | circle | evidence minus belief-predicted evidence | one node = the actual subtraction | YES |
| `⊥ν` node | circle | project current innovation away from previous innovation | one node = actual overlap removal | YES |
| residual bypass | thin orthogonal line | old belief survives gated update | one path = implemented residual/gate source | YES |
| `×L` mark | repetition label | repeated spatiotemporal predictor blocks | one mark = shared block depth shorthand | YES |

## User-Rejected Layout Repair — Strict Transformer Grammar

The user rejected the previous VA/WM figures because they still read as engineering flowcharts. This contract supersedes the earlier landscape-card composition.

### Palette

| role | hex | use |
|---|---|---|
| canvas | `#FFFFFF` | background |
| token/input | `#F8DDE1` | V/A/action inputs |
| attention/update | `#F6DDB3` | attention and repeated core |
| projection/norm | `#D8DCEA` | projections and normalization |
| representation/prediction | `#B9E3F0` | WM state and predictor |
| add/output | `#EEF1B2` | residual add and final action/state |
| soft output | `#C5E5CF` | committed belief/state |
| neutral | `#202020` | text, borders, main arrows |
| WM accent | `#4D78B8` | only the WM→VA message edge |

### Typography and Geometry

- Helvetica; title 22 pt, section 16 pt, module 14–15 pt, symbol 12–13 pt.
- Sharp/subtle corners (`arcSize=5–7`), 1.6–1.8 px strokes, no shadows.
- Typical operator box: 180–260 × 44–60 px; token cell: 40–64 × 30–36 px.
- Portrait or near-square composition; 28–36 px vertical rhythm; 16–24 px local gaps.
- No large enclosing region cards. Section names are text-only.

### Arrow Grammar

- One central top-to-bottom spine.
- Residuals stay within 30–45 px of their module stack.
- Side context enters through one short horizontal port.
- No prose labels on arrows; only `Q`, `K,V`, `×N`, `×L`, and tensor symbols.
- Any connector requiring a canvas-wide rail indicates a layout failure and must be redesigned.

### Semantic Justification

| element | visual form | meaning | justified? |
|---|---|---|---|
| V/A token cells | short labeled strips | actual visual/action token sequences | YES |
| DINO patch grid | 2×2 labeled cells | actual spatial feature map | YES |
| `VA Component ×N` | compact repeated block | actual policy layer stack | YES |
| evidence/predictor columns | two narrow spines | actual WM correction and prediction paths | YES |
| big dashed/rounded regions | omitted | carried no computation | NO — delete |
| explanatory cards and long buses | omitted | duplicated equations or captions | NO — delete |
