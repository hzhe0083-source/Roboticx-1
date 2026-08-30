# Visual Spec

## Source
- Reference images: `va_internal_simple_landscape_ai.png`, `va_wm_interaction_simple_landscape_ai.png`.
- Target DrawIO: `va_internal_simple_landscape.drawio`, `va_wm_interaction_simple_landscape.drawio`.
- Canvas: landscape, white background.
- Font policy: Helvetica with explicit sizes.

## Global Style
- Background: `#FFFFFF`.
- Primary font: Helvetica.
- Stroke: dark `#111111`, 2.5 px, rounded boxes.
- Arrow: orthogonal, filled classic head.
- Palette: Transformer pastels from `style-extraction.md`.

## Regions
| id | bbox x,y,w,h | role | notes |
|---|---|---|---|
| va_input | 40,230,170,270 | VA token input | two token families in one box |
| va_sources | 330,35,660,95 | K/V source strip | five labeled chips |
| va_attention | 290,180,520,340 | shared attention | Q/K/V principle |
| va_post | 900,285,490,120 | residual/FFN chain | three compact blocks |
| va_output | 1440,230,130,270 | VA token output | two token families |
| int_snapshot | 520,30,560,90 | common old state | feeds both peers |
| int_va | 45,190,520,470 | VA peer | minimal internals |
| int_wm | 1035,190,520,470 | WM peer | evidence→belief→predictor |
| int_commit | 570,725,460,75 | synchronized write | receives both outputs |

## Text Blocks
| id | bbox | text | font | alignment | priority |
|---|---|---|---|---|---|
| va_attention | 290,180,520,340 | attention title and two formulas | 18–25 pt | center/left | must |
| int_orange_label | 600,300,400,100 | two VA→WM K/V transfers | 17 pt | left | must |
| int_blue_label | 600,500,400,65 | delayed projection formula | 18 pt | center | must |

## Shapes
| id | bbox | type | fill | stroke | notes |
|---|---|---|---|---|---|
| VA input | 40,230,170,270 | rounded rect | pink | dark | compact |
| Attention | 290,180,520,340 | rounded rect | peach | dark | focal block |
| Add & Norm | two 120×110 boxes | rounded rect | yellow-green | dark | consistent pair |
| FFN | 120×110 | rounded rect | cyan | dark | centered between norms |
| VA / WM peers | 520×470 | rounded rect | peach / cyan | dark | equal weight |

## Connectors
| id | from | to | route | arrowhead | label | notes |
|---|---|---|---|---|---|---|
| va_main_* | input→attention→norm→ffn→norm→output | left-to-right | straight | classic | none | inference flow |
| va_res1 | input→norm1 | below blocks | classic | none | first residual |
| va_res2 | norm1→norm2 | above FFN | classic | none | second residual |
| int_snapshot_* | snapshot→VA / WM | split orthogonally | classic | none | same old state |
| int_va_wm | VA→WM | straight right | classic | orange label | same stage |
| int_wm_va | WM→VA | straight left | classic | blue formula | one-stage delay |
| int_commit_* | VA / WM→commit | bottom convergence | classic | none | atomic commit |

## Semantic Relations And Flow
| id | source | target | meaning | direction/cardinality | evidence |
|---|---|---|---|---|---|
| Q | V,A | VA attention | current query construction | fan-in | model implementation |
| KV | V,M,A,L,WM-old | VA attention | multimodal attention memory | fan-in | model implementation |
| VA-WM | pre-stage V,Aexec | WM | evidence and condition | grouped one-to-many | peer sync logic |
| WM-VA | previous Z | VA | stop-gradient projected K/V | delayed one-to-one | peer sync logic |
| Commit | VA_i and WM_i | S_i | synchronized stage update | fan-in | peer sync logic |

## Icons And Images
- None. The final DrawIO files contain only editable primitives and text.

