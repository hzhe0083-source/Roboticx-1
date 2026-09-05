# Diagram Brief

## User Goal
- Output: four concise editable DrawIO diagrams plus PNG previews and one AI-generated overview.
- Audience: readers of a robotics / world-model paper.
- Must communicate: (1) the peer-synchronous overview; (2) Q/K/V transport; (3) the WM update; (4) VA/WM losses and joint training.
- Must not do: reproduce the earlier dense full-system figure, embed raster references, or add decorative modules.

## Source Inventory
| id | source | type | role | priority | notes |
|---|---|---|---|---|---|
| S1 | user instructions in this task | text | content/layout | must | split figures, landscape, simple and clear |
| S2 | `va_internal_simple_landscape_ai.png` | image | layout sketch | should | six-block left-to-right VA flow |
| S3 | `va_wm_interaction_simple_landscape_ai.png` | image | layout sketch | should | two peers, two colored exchange directions |
| S4 | repository VA/WM implementation previously verified | code | semantic truth | must | shared pre-stage snapshot and one-stage delay |
| S5 | attached Transformer reference | image | palette/style only | should | pastel Transformer color grammar |

## Requirement Traceability
| id | requirement | evidence | level | planned encoding |
|---|---|---|---|---|
| R1 | main figure plus component figures | user | must | four `.drawio` source files |
| R2 | horizontal composition | user | must | 1600×700 and 1600×820 canvases |
| R3 | simple but logically complete | user | must | one idea per box; at most two text lines per box |
| R4 | VA Q/K/V sources | user + code | must | Q and K/V formulas inside attention block |
| R5 | same-stage VA→WM | code | must | orange rightward grouped arrow |
| R6 | delayed WM→VA | code | must | blue leftward arrow: `Z → sg → Proj → Kᵂ,Vᵂ` |
| R7 | no current-map-to-current-VA shortcut | code | must | `Z_i` routes only to commit |
| R8 | show training losses | user + code | must | compact VA-loss and WM-loss component figure |
| R9 | show gradient boundary | code | must | `stop-grad(Z)` and two backward / one optimizer step |

## Semantic Model
| id | entity / relation | direction | visual encoding | uncertainty |
|---|---|---|---|---|
| M1 | VA input tokens | left→right | pink input block | none |
| M2 | attention queries | VA tokens→attention Q | formula inside attention | none |
| M3 | attention memory | V/M/A/L/WM→attention K,V | grouped source strip | none |
| M4 | first residual | VA input→first Add & Norm | lower orthogonal bypass | none |
| M5 | second residual | first Add & Norm→second Add & Norm | upper orthogonal bypass | none |
| M6 | VA→WM | VA snapshot→WM evidence/condition | orange right arrow | none |
| M7 | WM→VA | previous WM map→next VA K/V | blue left arrow with delay label | none |
| M8 | stage commit | VA and WM outputs→`S_i` | two black converging arrows | none |

## Style Contract
| id | font | palette | stroke | icon style | density | source |
|---|---|---|---|---|---|---|
| ST1 | Helvetica / Arial | Transformer pastels | 2.5 px dark rounded | none | sparse | S2, S3, S5 |

## Open Assumptions
| assumption | risk | verification |
|---|---|---|
| English labels remain preferable for the manuscript figure. | low | matches existing architecture terminology |
| Component diagrams should be independent files rather than pages in one file. | low | easier independent manuscript placement |
