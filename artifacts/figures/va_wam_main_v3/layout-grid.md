# Layout Grid

## Canvas

- width: 1200 px
- height: 1780 px
- scale assumption: 1 draw.io unit = 1 exported pixel at 100% preview
- margin: 40 px outer stage margin; 70 px bottom input margin

## Grid Lines

| name | x | y | purpose |
|---|---:|---:|---|
| canvas_center | 600 | 0 | central action-output axis |
| va_center | 275 | 0 | VA main-chain axis |
| wam_center | 800 | 0 | WAM/ST main-chain axis |
| side_state | 1040 | 0 | Condition and State Update side column |
| stage_top | 0 | 330 | peer-stage boundary |
| merge_row | 0 | 455 | VA/WAM proposal merge row |
| snapshot_row | 0 | 1445 | shared stage input |
| embedding_row | 0 | 1550 | three embedding boxes |
| raw_input_row | 0 | 1660 | raw modality inputs |

## Region Boxes

| id | x | y | w | h |
|---|---:|---:|---:|---:|
| peer_stage | 40 | 330 | 1120 | 1170 |
| va_layer | 70 | 590 | 410 | 680 |
| wam_stage | 560 | 525 | 590 | 905 |
| st_predictor | 610 | 755 | 370 | 550 |
| action_head | 350 | 25 | 500 | 270 |
| training_loss | 910 | 195 | 240 | 235 |
| shared_snapshot | 300 | 1445 | 600 | 55 |

## Repeated Components

| family | count | cell size | spacing | start x,y |
|---|---:|---|---|---|
| peer stage | 8 logical repeats | one expanded stage | represented by title, not duplicated | 40,330 |
| VA residual sublayer | 2 | 310 × 70 max | 25–28 px operator rhythm | 120,1015 |
| ST predictor residual sublayer | 3 | 300 × 55 max | 15–23 px local rhythm | 650,1190 |
| ST predictor block | 6 logical repeats | one expanded block | represented by title | 610,755 |
| flow transformer | 6 logical repeats | 400 × 55 | represented by title | 400,180 |
| Euler step | 8 logical repeats | 360 × 55 | represented by title | 420,110 |

## Drawing Order

1. canvas background
2. peer/VA/WAM/ST containers
3. operator and state shapes
4. orthogonal runtime connectors
5. dashed training connectors
6. labels and subtitles
7. exported canvas crop
