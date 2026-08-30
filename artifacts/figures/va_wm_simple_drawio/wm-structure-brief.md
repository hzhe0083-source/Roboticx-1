# WM Structure Diagram Brief

## User Goal

- Output: editable DrawIO plus PNG.
- Audience: advisor presentation.
- Must communicate: the internal WM architecture through Transformer-style symbols rather than explanatory process boxes.
- Must not do: repeat the VA–WM overview, show losses, use large prose cards, or decorate with meaningless tokens.

## Source Inventory

| source | role |
|---|---|
| `va_compound/world/wmrm.py` | structure and exact state-update semantics |
| `va_compound/policy/model.py` | candidate action and stage timing |
| approved VA figure and `style-extraction.md` | palette, typography, box and arrow grammar |

## Requirement Traceability

| requirement | visual encoding |
|---|---|
| Inputs start at the top | labeled token strips and patch grids |
| Q/K/V are explicit | port labels on both attention modules |
| WM internals are clear | two Transformer-like towers with operator circles and residual paths |
| Output leaves at the bottom | three committed state symbols grouped as `W_i` |
| Symbolic Transformer style | compact modules, token/grid symbols, `−`, `⊥`, residual loops, `×L` |

## Semantic Model

1. `Q_E` attends to `K,V=V_{i−1}` and emits evidence tokens `E_i`.
2. `E_i − Ê(B_{i−1})`, then projection orthogonal to `ν_{i−1}`, emits `ν_i`.
3. A gated residual update converts `B_{i−1},ν_i` into `B̃_i`.
4. The ST predictor reads current DINO patches, candidate actions, `p_t,L,B̃_i`, and optional old `Z_{i−1}` to produce a new patch grid `Z_i`.
5. `B̃_i` queries encoded `Z_i`; a gated residual update emits final `B_i`.
6. Bottom state symbols group `B_i,ν_i,Z_i` as `W_i`.

## Style Contract

- Canvas: approximately `1320×760`.
- Helvetica: title 22 pt, tower 17 pt, module 15–16 pt, token/operator 12–14 pt.
- Palette: Transformer pastels from `style-extraction.md`.
- Geometry: two towers, 8 px modules, 44 px operator circles, compact token cells and patch grids.
- Connectors: orthogonal, filled classic arrowheads, explicit Q/K/V ports, thin residual loops, dashed prior-map refinement.

## Open Assumptions

- The figure shows the active spatiotemporal-predictor path; legacy predictor variants are omitted.
- `Z_{i−1}` is shown only as refinement context for stages after the first.

## User-Rejected Layout Repair

- Remove both large tower containers and all canvas-height state rails.
- Keep two narrow symbolic spines: `Evidence update` and `World prediction`.
- Old belief/innovation connect locally to subtraction, orthogonal projection, and gated update.
- `B̃_i` crosses once into the predictor and once into map attention; both arrows are short.
- The only dashed edge is `Z_{i−1} → Predictor` refinement.
- Maximum visible labels: 18; no prose sentences, legends, or large explanatory cards.
