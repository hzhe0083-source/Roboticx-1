# Diagram Brief — VA layer input origin

## User Goal
- Output: editable DrawIO source plus PNG preview.
- Audience: research supervisor / paper reader.
- Must communicate: `V_{i-1}, A_{i-1}` are the previous VA layer outputs; the first layer starts from encoded observation and robot state. The visual initialization must explicitly show `DINO patch tokens D_t → Linear adapter → V0`, including the dimension change. The action path must distinguish feature-axis concatenation `[p_t;u_{t-1}]` from broadcasting and adding the projected state to `H` learned queries. The complete figure must fit a 16:9 slide without shrinking into a long strip.
- Must not do: call the state a separate network or expand Q/K/V internals in this figure.

## Source Inventory
| source | role | use |
|---|---|---|
| `va_compound/policy/model.py` | content / structure | `V0`, `A0`, repeated VA layers |
| Existing VA figures | style | Transformer pastel palette and orthogonal arrows |
| User feedback | requirement | remove ambiguous “snapshot” wording |

## Semantic Model
| relationship | direction | visual encoding |
|---|---|---|
| observation → visual tokens | left to right | pink source → lavender encoder → `V0` |
| DINO tokens → VA tokens | left to right | blue DINO tokens → lavender learned linear adapter; token count unchanged |
| proprioception + previous action → state vector | fan-in | two pink inputs → explicit `Concat (feature dim)` |
| state vector + learned queries → action tokens | fan-in | lavender state projection plus learned-query input → explicit `Broadcast + Add` → `A0` |
| previous layer output → next layer input | left to right | dark orthogonal arrow |

## Compact Layout Contract
- Canvas: `1320×700`, slide-friendly without shrinking labels.
- Panel 1: compact visual/action token initialization; no long cross-canvas rails.
- Panel 2: one clean top-to-bottom VA recurrence stack.
- At most two semantic lines per normal box; tensor dimensions are removed unless they explain an operation.
- Two subtle dashed panel boundaries provide composition without decorative graphics.

## Style Contract
- Helvetica; title 22 pt, body 15–17 pt.
- Pastel pink `#FFDFDF`, lavender `#D9DFF2`, orange `#FFE2B2`, yellow-green `#F1F4B9`.
- Dark `#262626`, 2 px strokes; no icons or decoration.

## Open Assumptions
- The presentation uses English labels, matching the existing component figures.
