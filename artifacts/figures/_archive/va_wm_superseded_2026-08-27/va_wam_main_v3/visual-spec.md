# Visual Spec

## Source

- Reference image: `/tmp/codex-clipboard-3859853e-ac46-44b3-941f-9f771689ccac.png`
- Target drawio: `va_wam_main_v3.drawio`
- Canvas: 1200 × 1780 px, portrait
- Font policy: Helvetica-compatible sans serif; primary labels 16–24 px; annotations 11–13 px

## Global Style

- Background: `#DBDBDB`
- Primary font: Helvetica, black `#111111`
- Stroke style: rounded black outlines, `#000000`, 3 px, no shadows
- Arrow style: orthogonal, classic filled heads; solid black runtime flow; dashed `#6F5656` training-only flow
- Color palette: input pink `#D6BEBE`, operator sand `#D7BE9E`, transform blue `#A6C5D1`, norm/state olive `#CFCFA5`, flow lavender `#BABECB`, output sage `#AFC3AF`

## Regions

| id | bbox x,y,w,h | role | visual notes |
|---|---|---|---|
| action_head | 350,25,500,270 | single executable-action emitter | centered, upward chain |
| peer_stage | 40,330,1120,1170 | repeated peer-synchronous stage | one rounded outer frame |
| va_layer | 70,590,410,680 | detailed VA Transformer layer | two residual sublayers |
| wam_stage | 560,525,590,905 | detailed WAM stage | predictor, state update, mixer |
| st_predictor | 610,755,370,550 | repeated ST predictor block | three residual sublayers |
| training_loss | 910,195,240,235 | future target and weighted WAM objective | pink/sand boxes, dark-red dashed edges |
| inputs | 70,1550,1060,170 | three modality paths | aligned three-column fan-in |

## Text Blocks

| id | bbox x,y,w,h | text | font | alignment | priority |
|---|---|---|---|---|---|
| peer_title | 40,330,1120,40 | Peer-Synchronous Stage × 8 | Helvetica 24 bold | left/top | primary |
| va_title | 70,590,410,40 | VA Layer | Helvetica 22 bold | left/top | primary |
| wam_title | 560,525,590,40 | WAM Stage | Helvetica 22 bold | left/top | primary |
| st_title | 610,755,370,35 | ST Predictor × 6 | Helvetica 18 bold | left/top | secondary |
| action_output | 350,25,500,65 | H6 × 4 Action Chunk | Helvetica 24 bold | centered | primary |
| snapshot | 300,1445,600,55 | Shared Pre-stage Snapshot Sᵢ₋₁ | Helvetica 18 bold + 12 | centered | primary |

## Shapes

| id | bbox x,y,w,h | type | fill | stroke | notes |
|---|---|---|---|---|---|
| containers | multiple | rounded rectangles | none | `#000000` 3 px | solid-looking long-dash pattern for validator-safe containers |
| attention_ops | multiple | rounded rectangles | `#A6C5D1` or `#D7BE9E` | `#000000` 3 px | attention/cross-attention modules |
| norm_add | multiple | rounded rectangles | `#CFCFA5` | `#000000` 3 px | pre-norm, residual add, state boxes |
| modality_inputs | three | rounded rectangles | `#D6BEBE` | `#000000` 3 px | instruction, vision, robot state/action |
| flow_head | four | rounded rectangles | olive/lavender/sage | `#000000` 3 px | unique action emission chain |

## Connectors

| id | from | to | route | arrowheads | label | notes |
|---|---|---|---|---|---|---|
| snapshot_va | Shared Snapshot | VA Tokens | orthogonal left branch | target only | none | VA proposal reads pre-stage state |
| snapshot_wam_evidence | Shared Snapshot | Evidence Cross-Attention | orthogonal right branch | target only | none | parallel WAM evidence path |
| snapshot_wam_clip | Shared Snapshot | DINO Clip Tokens | orthogonal right branch | target only | none | parallel predictor token path |
| va_residuals | token/add outputs | corresponding Residual Add | two separate U lanes | target only | none | no shared residual bus |
| st_residuals | token/add outputs | corresponding Residual Add | three alternating U lanes | target only | none | no shared residual bus |
| map_runtime | Predicted DINO Map | World Tokens / State Update | short orthogonal branches | target only | none | runtime world representation |
| map_training | Predicted DINO Map / Future Target | World Loss | right perimeter | target only | none | dashed, training-only |
| stage_commit | Gated Merge / World State | Stage Commit | two separate upward routes | target only | none | latent and recurrent state remain distinct |
| action_emit | Stage Commit | H6 × 4 Action Chunk | vertical upward | target only | none | Layer Norm → Flow ×6 → Euler ×8 |

## Semantic Relations And Flow

| id | source | target | meaning | direction/cardinality | visual evidence |
|---|---|---|---|---|---|
| peer_read | Sᵢ₋₁ | VA and WAM | both proposals read the same snapshot | one-to-two | two arrows from the same snapshot box |
| va_update | VA Tokens | VA Proposal | shared MHA and FFN with local residuals | chain | left module stack |
| world_prediction | DINO clip + previous map | predicted next DINO latent map | ST prediction/refinement | chain | right module stack |
| world_projection | predicted map | world tokens | flatten/project spatial map | one-to-one | direct upward arrow |
| latent_writeback | gated mixer | latent ΔV, ΔA | WAM modifies VA latent streams only | one-to-one | upward arrow into latent-delta box |
| recurrent_state | map + belief/innovation | World Stateᵢ | persistent WAM state for later stages | fan-in then one output | State Update box |
| world_state_contract | belief, innovation, predicted map | Wᵢ | recurrent state is `{Bᵢ,Iᵢ,Zᵢ}`, not an RGB image or explicit physical state | three-field pack | expanded state box with exact active shapes |
| state_training | predicted map / latent belief and innovation | WAM objective / Flow objective | `Zᵢ` has direct future-DINO supervision; `Bᵢ/Iᵢ` have no state labels and learn through the gated policy path | direct plus indirect | red training branch plus state-learning note |
| stage_recurrence | Stage Commit Sᵢ | next pre-stage snapshot | committed `Vᵢ,Aᵢ,Wᵢ` become the next stage input for `i&lt;8` | loop | dedicated black dashed recurrence lane |
| action_output_relation | A₈ | action chunk | only Flow/Euler emits executable robot action | one-to-one | single top chain ending at `Layer Norm (A₈ only)` |
| stop_gradient | future DINO target | World Loss | training target, never forward input | dashed one-to-one | pink target + dashed dark-red line |

## Icons And Images

- No raster image or icon is embedded in the editable diagram.
- The supplied reference is used only for style extraction; all delivered shapes, text, and connectors are editable draw.io primitives.
- No semantic icon is required; modality and training distinctions are expressed by labels, palette, and line grammar.
