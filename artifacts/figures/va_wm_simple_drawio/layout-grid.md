# Layout Grid

## Canvas
- VA internal: 1600 × 700 px.
- VA–WM interaction: 1600 × 820 px.
- Scale: 1 diagram unit = 1 px.
- Margin: 35–45 px.

## Grid Lines
| name | x | y | purpose |
|---|---:|---:|---|
| va_mid | 800 | 350 | VA canvas center |
| va_pipeline | — | 350 | main left-to-right baseline |
| int_left_peer | 305 | 425 | VA center |
| int_right_peer | 1295 | 425 | WM center |
| int_exchange_top | — | 300 | orange path |
| int_exchange_bottom | — | 535 | blue path |

## Region Boxes
| id | x | y | w | h |
|---|---:|---:|---:|---:|
| va_input | 40 | 230 | 170 | 270 |
| va_sources | 330 | 35 | 660 | 95 |
| va_attention | 290 | 180 | 520 | 340 |
| va_norm1 | 900 | 295 | 120 | 110 |
| va_ffn | 1090 | 295 | 120 | 110 |
| va_norm2 | 1280 | 295 | 120 | 110 |
| va_output | 1440 | 230 | 130 | 270 |
| int_snapshot | 520 | 30 | 560 | 90 |
| int_va | 45 | 190 | 520 | 470 |
| int_wm | 1035 | 190 | 520 | 470 |
| int_commit | 570 | 725 | 460 | 75 |

## Repeated Components
| family | count | cell size | spacing | start |
|---|---:|---|---|---|
| K/V chips | 5 | 95×55 (WM 140×55) | 20 px | 350,55 |
| Add & Norm | 2 | 120×110 | 260 px | 900,295 |

## Drawing Order
1. peer and module backgrounds
2. source chips and labels
3. structural connectors
4. colored exchange connectors
5. text labels and delay marker

