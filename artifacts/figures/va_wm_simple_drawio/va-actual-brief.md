# Diagram Brief — VA Actual Implementation

## User Goal
- Output: one concise editable DrawIO figure showing the complete VA policy, while keeping the existing Q/K/V figure as a separate component diagram.
- Audience: a research-group presentation.
- Must communicate: token initialization, a repeated atomic `VA Component × N`, visual-memory output, and the flow-matching action head.
- Must not do: reopen Q/K/V, Attention, Residual, FFN, or WM-injection details that already belong to the separate VA Component figure.

## Source Inventory
| id | source | role | priority |
|---|---|---|---|
| S1 | `va_compound/policy/model.py` | content and connector semantics | must |
| S2 | current VA Q/K/V DrawIO | palette and component terminology | must |
| S3 | Transformer reference supplied earlier | style only | should |
| S4 | latest user request | simple landscape composition | must |

## Requirement Traceability
| id | requirement | planned encoding |
|---|---|---|
| R1 | Show what VA itself is | three regions: token initialization, coupling stack, action generation |
| R2 | Keep abstraction boundaries clean | one atomic `VA Component × N` block; its internals are shown only in the separate component diagram |
| R3 | Preserve the two outputs | the stack writes visual layers to VisualMemory and sends final A_N to the action head |
| R4 | Show final action generation | `LN(A_N)` conditions a flow head; Euler steps produce the action chunk |
| R5 | Keep it simple | one formula per transformation; optional task/dense branches omitted in a footnote |

## Semantic Model
| relation | direction | visual encoding |
|---|---|---|
| DINO tokens → V0 | left to right | pink input, lavender projection, pink token |
| proprio + previous action → A0 | left to right then down | state projection followed by learned-query addition |
| V0/A0/language → VA stack | left to right | three short labeled inputs into one atomic repeated block |
| VA Component × N | repeated transformation | one orange block; no internal cells or equations |
| A_N → action chunk | left to right | LayerNorm, flow head, Euler update |
| V1…VN → visual memory | left to right | separate upper output lane |

## Open Assumption
- The figure documents the core `peer_sync` VA path. Optional task tokens, dense readout, dual-attention, and other ablation switches remain outside this overview.
# Symbolic redesign — final three-figure suite

- Figure role: VA internal structure only.
- Global direction: raw inputs at the top, `V_i,A_i` at the bottom.
- Region 1: initialize visual tokens with `DINO → Vision Projection → V_0` and action tokens with `[p_t || u_{t−1}] → State Projection`, learned action queries, and `+ → A_0`.
- Region 2: one repeated VA layer with explicit Q projection, local K,V, WM K,V, K,V concat, Shared Attention, two residual Add & Norm operations, and FFN.
- Do not draw WM internals, losses, or peer synchronization here; those belong to figures 2 and 3.
- Reuse the Transformer palette contract from `style-extraction.md`.

## User-Rejected Layout Repair

- The previous landscape overview failed visually: large outer cards, a cavernous VA block, and long perimeter rails made it read like a workflow diagram.
- Replace it with a portrait Transformer spine: symbolic inputs at the top, a compact `VA Component ×N` in the middle, and action output at the bottom.
- Use real token cells for `V₀` and `A₀`; compress all per-layer side information into one small `C_i=[L,M^V,M^W]` port because Figure 2 already expands Q/K/V.
- Keep VisualMemory as one short side branch from the repeated stack.
- Maximum visible labels: 14; maximum large operator boxes: 6; no outer region containers.
