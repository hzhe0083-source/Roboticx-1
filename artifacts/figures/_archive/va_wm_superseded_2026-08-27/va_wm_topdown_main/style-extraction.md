# Style Extraction: Transformer Reference

## 1. Palette
| role | hex | used on |
|---|---|---|
| background | #FFFFFF | canvas and group interiors |
| primary fill | #B7E9F9 | feed-forward and world-token blocks |
| secondary fill | #FFE1B6 | attention and cross-attention blocks |
| accent / highlight | #F2F4BB | Add & Norm, recurrent snapshot |
| border stroke | #111111 | all main boxes |
| arrow stroke | #111111 | normal data flow |
| heading text | #111111 | titles |
| body text | #222222 | formulas and notes |
| muted / gray | #666666 | secondary annotation and dashed paths |
| special | #FFDFDF / #D9DFF0 / #C6E8CD | input / Flow / final action |

Total distinct colors: 8.

## 2. Typography
- Heading font: Helvetica / Arial, 22–24 pt, bold.
- Subheading font: Helvetica / Arial, 16–18 pt, bold.
- Body text font: Helvetica / Arial, 13–15 pt.
- Small label / caption font: Helvetica / Arial, 11–12 pt.
- Code / mono font: none.

## 3. Shape Language
- Corner radius: subtle rounded rectangles, approximately 10 px.
- Stroke width for boxes: 2.5 px.
- Stroke width for arrows: 2 px.
- Dash pattern for containers: 7 5.
- Shadow: no.
- Fill opacity for background regions: 100% white.

## 4. Layout Rhythm
- Outer margin: 30 px.
- Gap between major regions: 30 px vertical.
- Gap between same-row elements: 35–45 px horizontal.
- Padding inside boxes: 10 px vertical, 14 px horizontal.
- Typical box width: 250–430 px; height: 44–72 px.
- Grid alignment: 10 px grid.

## 5. Arrow Grammar
- Default arrow type: classic filled.
- Arrow color: #111111.
- Arrowhead size: medium.
- Routing style: orthogonal.
- Arrow labels: yes, only on VA↔WM exchange, 12 pt.
- Color coding: blue #4D78B8 only for the delayed WM→VA K/V publication; muted #7A5A5A dashed for training-only supervision.

## 6. Icon Language
- Icon style: none; labeled primitives only.
- Icon size: not applicable.
- Icon stroke width: not applicable.
- Icons color-coded: no.
- Source: none.

## 7. Density & Composition
- Diagram type: top-down pipeline with paired transformer towers and feedback.
- Major regions: input, shared snapshot, VA/WM coupled stage, output head.
- Content density: medium.
- Whitespace: moderate to generous.
- Panel labels: named regions, no A/B/C letters.
- Legend: no.
- Caption below diagram: no.

## Semantic Justification
| element | visual form | what it represents | each unit corresponds to | justified? |
|---|---|---|---|---|
| VA tower | vertical Transformer stack | one actual `VACouplingLayer` | one block = one implemented sublayer | YES |
| WM tower | vertical Transformer stack | evidence/belief update plus ST predictor | one block = one implemented operation | YES |
| exchange lane | two labeled orthogonal paths | same-snapshot VA→WM conditioning and delayed WM→VA K/V publication | one arrow = one code-level tensor path | YES |
| stage loop | dashed feedback connector | commit Sᵢ and repeat across seven paired stages | one loop = stage recurrence | YES |
| colored token bars / matrix grids | omitted | no extra discrete token category needs a decorative miniature | n/a | NO — omitted |
| training-only loss branch | small dashed side branch | future DINO supervision of Zᵢ | one branch = one training-only objective | YES |
