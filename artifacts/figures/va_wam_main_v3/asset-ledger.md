# Asset Ledger

## Exact Assets

| id | source | path | usage |
|---|---|---|---|
| style_reference | user-provided raster | `/tmp/codex-clipboard-3859853e-ac46-44b3-941f-9f771689ccac.png` | palette, rounded box, stroke, arrow and portrait-composition reference |
| code_semantics | local implementation | `/home/ryan/Documents/robot/ORA0/va_compound/model.py` | VA peer-stage and single action-head logic |
| wam_semantics | local implementation | `/home/ryan/Documents/robot/ORA0/va_compound/wmrm.py` | ST predictor, map→tokens, state update, loss semantics |

## Editable Primitive Icons

- None used. The reference’s visual language is box-and-arrow; every delivered element is a native editable draw.io rectangle, text cell, container, or connector.

## Approximations

| id | reference meaning | approximation | why |
|---|---|---|---|
| repeated_stack | explicit repeated Transformer blocks | one expanded block labeled ×8 or ×6 | preserves readability while keeping the detailed internal hierarchy |
| add_norm | reference combines Add & Norm | separate Residual Add and Pre-Norm boxes | matches the actual pre-norm implementation |
| training_distinction | reference has no training branch | pink target plus dark-red dashed connectors | prevents future targets from being mistaken for forward inputs |

## Missing Assets

- None. No external icon, font file, raster insert, or unavailable visual asset is needed for the final editable figure.

