# Diagram Brief

## User Goal
- Output: editable DrawIO plus PNG/SVG paper figure.
- Audience: robotics / VLA paper readers.
- Must communicate: top-down input-to-output flow; peer-synchronous VA–WM coupling; the one-stage delay; the bottom-level transformer structure; explicit Q/K/V roles.
- Must not do: over-describe losses, training schedules, tensor bookkeeping, or reuse an old figure layout.

## Source Inventory
| id | source | type | role | priority | notes |
|---|---|---|---|---|---|
| S1 | user prompt | text | content + layout | must | inputs at top, outputs at bottom; VA/WM interaction is central |
| S2 | attached Transformer figure | image | palette + shape + arrow style | must | rounded pastel blocks, black orthogonal arrows, sparse labels |
| S3 | `README.md` | text | architecture contract | must | VA×8, WM×7, one-stage-delayed peer exchange, Flow-only action emitter |
| S4 | `va_compound/policy/model.py` | code | VA Q/K/V and stage ordering | must | VA queries are V/A; WM message is a K/V-only state source |
| S5 | `va_compound/world/wmrm.py` | code | WM internals and Q/K/V | must | evidence read, belief update, causal SA, conditional CA, future DINO map |
| S6 | repository figures | image | secondary style only | may | no geometry or content copied |

## Requirement Traceability
| id | requirement | source evidence | level | planned visual encoding |
|---|---|---|---|---|
| R1 | VA and WM interaction dominates | user | must | two equal towers plus a highlighted exchange lane |
| R2 | top-down flow | user | must | inputs at y=35; action chunk at the bottom |
| R3 | explicit Q/K/V | user | must | formulas inside VA attention, WM evidence read, WM self/cross-attention, and the WM→VA message |
| R4 | bottom-level structure | user | must | Pre-Norm → Attention → Add & Norm → FFN → Add & Norm blocks |
| R5 | simple, not verbose | user | must | only architecture-critical labels; one short training-only side branch |
| R6 | Transformer-like colors | user + S2 | must | sampled pastel pink, orange, cyan, yellow-green, lavender, green |
| R7 | one-stage delay | S3/S4 | must | WMᵢ publishes K/V to VAᵢ₊₁; both read the same Sᵢ₋₁ snapshot |
| R8 | only VA emits actions | S3 | must | only VA₈ connects to the Flow head and final action chunk |

## Semantic Model
| id | entity or relationship | direction / hierarchy / cardinality | visual encoding | uncertainty |
|---|---|---|---|---|
| M1 | encoded instruction, visual tokens, robot state | input → shared pre-stage snapshot | three pastel input branches | none |
| M2 | VA layer i | Sᵢ₋₁ → (Vᵢ,Aᵢ) | left transformer tower | none |
| M3 | WM stage i | Sᵢ₋₁ → Wᵢ={Bᵢ,Iᵢ,Zᵢ} | right world-memory tower | none |
| M4 | VA→WM | Vᵢ₋₁ supplies evidence K/V; executable Aᵢ₋₁ supplies conditional K/V | lower exchange arrow | none |
| M5 | WM→VA | sg(Zᵢ) → projection → Kᵂᵢ,Vᵂᵢ → VAᵢ₊₁ | upper exchange arrow | none |
| M6 | recurrent stage | (Vᵢ,Aᵢ,Wᵢ) → Sᵢ → next stage | compact dashed feedback loop | none |
| M7 | action generation | VA₈ → Flow Transformer×6 → Euler×8 → H6×4 | centered bottom stack | none |
| M8 | WM supervision | Zᵢ + sg(Zₜ₊₁) → L_WM | one dashed training-only branch | loss details intentionally omitted |

## Style Contract
| id | font | palette | stroke | icon style | density | reference |
|---|---|---|---|---|---|---|
| C1 | Helvetica/Arial, 12–24 pt | #FFFFFF #FFDFDF #FFE1B6 #B7E9F9 #F2F4BB #D9DFF0 #C6E8CD | #111111, 2–2.5 px | no icons | medium-sparse | attached Transformer figure |

## Layout Specification
- Canvas: 1600 × 1120 px, white background, 30 px outer margin, 10 px grid.
- Flow: raw inputs → encoders → shared snapshot → paired VA/WM stage → final VA → Flow → action chunk.
- Main stage: VA tower left, exchange lane center, WM tower right; same vertical baseline and equal visual weight.
- Arrows: black orthogonal data paths; blue WM→VA K/V path; muted dashed training-only or recurrence paths.
- No legend: every non-black connector is directly labeled.

## Open Assumptions
| assumption | risk | how verified |
|---|---|---|
| Use English labels for the English paper | low | terminology matches paper/code |
| Show WM×7 and a final VA₈ | low | verified in README and `encode_condition` loop |
| Collapse detailed WM losses to `L_WM` | low | user explicitly requested simplicity |
