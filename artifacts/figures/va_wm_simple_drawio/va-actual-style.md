# Style Extraction — Transformer reference and current VA component

## 1. Palette
| role | hex | used on |
|---|---|---|
| background | `#FFFFFF` | canvas and panels |
| token / input | `#FFDFDF` | V/A inputs and outputs |
| projection | `#D9DFF2` | linear projections and K/V source strip |
| attention | `#FFE2B2` | shared attention |
| residual / output | `#F1F4B9` | residual and action output blocks |
| condition / flow | `#B7E9FB` | WM K/V and flow head |
| border / text | `#262626` | boxes, arrows, main text |
| muted | `#646464` | residual bypasses and notes |

Total distinct colors: 8.

## 2. Typography
- Heading: Helvetica, 22 pt, bold.
- Region heading: Helvetica, 18 pt, bold.
- Body: Helvetica, 15–16 pt.
- Small label: Helvetica, 12–13 pt.
- Code/mono: none.

## 3. Shape Language
- Corner radius: subtle, `arcSize=8`.
- Box and main-arrow stroke: 2 px.
- Residual stroke: 1.5 px gray.
- Region containers: 2 px dashed, `8 8`.
- Shadow: none.

## 4. Layout Rhythm
- Canvas: 2080 × 760 px.
- Outer margin: 30–40 px.
- Major-region gap: 30–40 px.
- Same-row box gap: 20–30 px.
- Box padding: about 10 px vertical and 14 px horizontal.
- Grid: 10 px.

## 5. Arrow Grammar
- Default arrow: filled classic arrowhead.
- Routing: orthogonal.
- Data arrows: dark `#262626`; WM conditioning: blue `#4D78B8`; residuals: gray `#646464`.
- Labels: only when they identify tensors (`Q`, `K,V`, `Y_V`, `Y_A`).

## 6. Icon Language
- No icons. Tensor names and formulas carry the meaning.

## 7. Density and Composition
- Diagram type: three-stage landscape pipeline.
- Major regions: 3.
- Density: medium.
- Whitespace: moderate.
- Panel labels: named regions, no A/B/C markers.
- Legend and caption: none.

## Semantic Justification
| element | visual form | meaning | justified? |
|---|---|---|---|
| V/A boxes | labeled rounded boxes | persistent visual/action token streams | yes |
| shared attention | orange rounded box | actual shared attention operation | yes |
| two post-attention lanes | separate rounded boxes | distinct `out_v/ffn_v` and `out_a/ffn_a` code paths | yes |
| dashed regions | containers | initialization, repeated VA body, and output head boundaries | yes |
| decorative token bars / icons | omitted | no additional information | yes |
