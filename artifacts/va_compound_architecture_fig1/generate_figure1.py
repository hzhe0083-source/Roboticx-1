#!/usr/bin/env python3
"""Generate the VA Compound Figure 1 as a dependency-free standalone SVG.

The SVG intentionally uses only text, rect, path, and polygon drawing primitives.
Rasterization is performed separately with CairoSVG so the vector source remains
self-contained and editable.
"""

from __future__ import annotations

from html import escape
from math import hypot
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "figure1_va_compound.svg"

W, H = 1402, 630
FONT = "Arial, Helvetica, sans-serif"

INK = "#182233"
MUTED = "#586579"
LIGHT_LINE = "#A9B2BF"
BLUE = "#3978B5"
BLUE_FILL = "#EAF3FC"
BLUE_FILL_2 = "#F2F7FD"
VIOLET = "#7456A6"
VIOLET_FILL = "#F1ECFA"
VIOLET_FILL_2 = "#FAF8FD"
GREEN = "#3E8A5B"
GREEN_FILL = "#EAF7EF"
GREEN_FILL_2 = "#F4FBF6"
AMBER = "#C48425"
AMBER_FILL = "#FFF3DB"
GRAY = "#7A8492"
GRAY_FILL = "#F3F5F7"
RED = "#B34A4A"
RED_FILL = "#FDECEC"
WHITE = "#FFFFFF"


elements: list[str] = []


def add(tag: str) -> None:
    elements.append(tag)


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = WHITE,
    stroke: str = INK,
    sw: float = 1.7,
    rx: float = 8,
    dash: str | None = None,
) -> None:
    attrs = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'width="{w:g}"',
        f'height="{h:g}"',
        f'rx="{rx:g}"',
        f'fill="{fill}"',
        f'stroke="{stroke}"',
        f'stroke-width="{sw:g}"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    add(f"  <rect {' '.join(attrs)}/>")


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: float = 13,
    weight: int | str = 400,
    fill: str = INK,
    anchor: str = "start",
    italic: bool = False,
    letter_spacing: float | None = None,
) -> None:
    attrs = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'font-family="{FONT}"',
        f'font-size="{size:g}"',
        f'font-weight="{weight}"',
        f'fill="{fill}"',
        f'text-anchor="{anchor}"',
    ]
    if italic:
        attrs.append('font-style="italic"')
    if letter_spacing is not None:
        attrs.append(f'letter-spacing="{letter_spacing:g}"')
    add(f"  <text {' '.join(attrs)}>{escape(value)}</text>")


def multiline(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: float = 13,
    line_gap: float | None = None,
    weight: int | str = 400,
    fill: str = INK,
    anchor: str = "start",
    first_weight: int | str | None = None,
) -> None:
    gap = line_gap if line_gap is not None else size + 3
    for index, line in enumerate(lines):
        text(
            x,
            y + index * gap,
            line,
            size=size,
            weight=first_weight if index == 0 and first_weight is not None else weight,
            fill=fill,
            anchor=anchor,
        )


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[str],
    *,
    fill: str,
    stroke: str,
    title_size: float = 14,
    body_size: float = 11.5,
    sw: float = 1.8,
    rx: float = 8,
    top_pad: float = 18,
) -> None:
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx)
    if not lines:
        return
    text(x + w / 2, y + top_pad, lines[0], size=title_size, weight=700, anchor="middle")
    if len(lines) > 1:
        body_y = y + top_pad + body_size + 4
        multiline(
            x + w / 2,
            body_y,
            lines[1:],
            size=body_size,
            line_gap=body_size + 3,
            fill=MUTED,
            anchor="middle",
        )


def badge(x: float, y: float, w: float, h: float, label: str, *, fill: str, stroke: str) -> None:
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.5, rx=h / 2)
    text(x + w / 2, y + h / 2 + 4.5, label, size=12, weight=700, fill=stroke, anchor="middle")


def arrowhead(x: float, y: float, dx: float, dy: float, color: str, size: float = 7.5) -> None:
    length = hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx, by = x - size * ux, y - size * uy
    half = size * 0.52
    points = [
        (x, y),
        (bx + half * px, by + half * py),
        (bx - half * px, by - half * py),
    ]
    pts = " ".join(f"{px_:g},{py_:g}" for px_, py_ in points)
    add(f'  <polygon points="{pts}" fill="{color}" stroke="none"/>')


def path_arrow(
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    sw: float = 1.8,
    dash: str | None = None,
    label: str | None = None,
    label_at: tuple[float, float] | None = None,
    label_size: float = 10.5,
    label_fill: str | None = None,
    label_anchor: str = "middle",
) -> None:
    d = "M " + " L ".join(f"{x:g} {y:g}" for x, y in points)
    attrs = [
        f'd="{d}"',
        'fill="none"',
        f'stroke="{color}"',
        f'stroke-width="{sw:g}"',
        'stroke-linecap="round"',
        'stroke-linejoin="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    add(f"  <path {' '.join(attrs)}/>")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    arrowhead(x1, y1, x1 - x0, y1 - y0, color)
    if label and label_at:
        text(
            label_at[0],
            label_at[1],
            label,
            size=label_size,
            weight=600,
            fill=label_fill or color,
            anchor=label_anchor,
        )


def straight_arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    *,
    label_dx: float = 0,
    label_dy: float = -5,
    color: str = INK,
    sw: float = 1.8,
    dash: str | None = None,
    label_size: float = 10.5,
) -> None:
    path_arrow(
        [(x1, y1), (x2, y2)],
        color=color,
        sw=sw,
        dash=dash,
        label=label,
        label_at=((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy),
        label_size=label_size,
    )


# Canvas and header.
rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
text(18, 28, "VA Compound: closed-loop vision–language–action policy", size=23, weight=700)
text(
    W - 18,
    27,
    "main reported configuration: flat pooling · 8 VA layers @20k",
    size=12.5,
    weight=600,
    fill=VIOLET,
    anchor="end",
)
text(18, 47, "RUNTIME / CLOSED LOOP", size=10.5, weight=700, fill=MUTED, letter_spacing=1.3)

# Language path.
box(
    18,
    57,
    130,
    57,
    ["Instruction text", "≤64 tokens"],
    fill=GRAY_FILL,
    stroke=GRAY,
    title_size=14,
    body_size=12,
)
box(
    170,
    49,
    218,
    73,
    ["Frozen Qwen3.5-2B", "language_model only", "no vision tower / LM head"],
    fill=BLUE_FILL,
    stroke=BLUE,
    title_size=15,
    body_size=11.5,
)
box(
    410,
    57,
    205,
    57,
    ["last_hidden_state", "[B, ≤64, 2048]"],
    fill=BLUE_FILL_2,
    stroke=BLUE,
    title_size=14,
    body_size=12,
)
box(
    637,
    45,
    278,
    81,
    [
        "Per-layer language cache",
        "each layer: LN → k_l / u_l",
        "K_l / U_l  [B, 8, ≤64, 64]",
        "compute once; reuse across steps",
    ],
    fill=AMBER_FILL,
    stroke=AMBER,
    title_size=14.5,
    body_size=11,
)
straight_arrow(148, 85, 170, 85, "tokenize", label_dy=-7, color=GRAY, label_size=9.5)
straight_arrow(388, 85, 410, 85, "encode once", label_dy=-7, color=BLUE, label_size=9.5)
straight_arrow(615, 85, 637, 85, "fan out", label_dy=-7, color=AMBER, label_size=9.5)

# Vision path.
box(
    18,
    166,
    128,
    65,
    ["Video window", "even F ≥ 2", "[B, F, 3, H, W]"],
    fill=GRAY_FILL,
    stroke=GRAY,
    title_size=14,
    body_size=11,
)
box(
    166,
    156,
    198,
    84,
    ["Frozen V-JEPA 2.1", "ViT-B/384 encoder", "per observation window"],
    fill=BLUE_FILL,
    stroke=BLUE,
    title_size=15,
    body_size=11.5,
)
box(
    384,
    163,
    188,
    70,
    ["Flat adaptive pooling", "1D avg-pool over [t,h,w]", "[B, ≤64, 768]"],
    fill=BLUE_FILL_2,
    stroke=BLUE,
    title_size=14,
    body_size=11,
)
box(
    592,
    166,
    147,
    65,
    ["Vision projection", "Linear 768 → 512", "[B, ≤64, 512]"],
    fill=VIOLET_FILL,
    stroke=VIOLET,
    title_size=13.5,
    body_size=11,
)
straight_arrow(146, 198, 166, 198, "encode / step", label_dy=-8, color=BLUE, label_size=9.5)
straight_arrow(364, 198, 384, 198, "[t,h,w] grid", label_dy=-8, color=BLUE, label_size=9.5)
straight_arrow(572, 198, 592, 198, "pool", label_dy=-8, color=BLUE, label_size=9.5)

# Robot-state path.
box(
    18,
    287,
    132,
    62,
    ["Proprioception", "[B, 14]"],
    fill=GRAY_FILL,
    stroke=GRAY,
    title_size=13.5,
    body_size=12,
)
box(
    170,
    287,
    132,
    62,
    ["Previous action", "[B, 7]"],
    fill=GRAY_FILL,
    stroke=GRAY,
    title_size=13.5,
    body_size=12,
)
box(
    326,
    293,
    108,
    50,
    ["Concat", "[B, 21]"],
    fill=GRAY_FILL,
    stroke=GRAY,
    title_size=13,
    body_size=11.5,
    top_pad=17,
)
box(
    458,
    287,
    146,
    62,
    ["State projection", "Linear 21 → 512", "[B, 512]"],
    fill=VIOLET_FILL,
    stroke=VIOLET,
    title_size=13.5,
    body_size=11,
)
box(
    626,
    278,
    113,
    80,
    ["Action tokens", "8 learned queries", "+ state embedding", "[B, 8, 512]"],
    fill=VIOLET_FILL,
    stroke=VIOLET,
    title_size=13.5,
    body_size=10.5,
)
straight_arrow(150, 318, 326, 310, "14-D", label_dy=-7, color=GRAY, label_size=10)
straight_arrow(302, 318, 326, 326, "7-D", label_dy=15, color=GRAY, label_size=10)
straight_arrow(434, 318, 458, 318, "project", label_dy=-8, color=VIOLET, label_size=9.5)
straight_arrow(604, 318, 626, 318, "broadcast + add", label_dy=-8, color=VIOLET, label_size=9.2)

# Main VA stack outer block.
rect(758, 136, 312, 365, fill=VIOLET_FILL, stroke=VIOLET, sw=2.2, rx=12)
text(775, 158, "TRAINABLE VA DECISION STACK", size=11, weight=700, fill=VIOLET, letter_spacing=0.7)
text(775, 179, "VACouplingLayer", size=18, weight=700)
text(775, 195, "one observation pass · layers 1...8", size=11.5, fill=MUTED)
badge(1017, 148, 39, 25, "×8", fill=WHITE, stroke=VIOLET)

# Input arrows to the stack.
straight_arrow(739, 198, 758, 198, "V_t", label_dy=-8, color=VIOLET, label_size=11)
straight_arrow(739, 318, 758, 318, "A_t", label_dy=-8, color=VIOLET, label_size=11)
path_arrow(
    [(915, 102), (1059, 102), (1059, 244), (1038, 244)],
    color=AMBER,
    sw=2,
    label="cached K_l / U_l → every layer",
    label_at=(984, 96),
    label_size=10.5,
)

# Single VACouplingLayer inset.
rect(775, 210, 278, 250, fill=VIOLET_FILL_2, stroke="#AA98C8", sw=1.4, rx=8)
text(786, 229, "single layer i: query streams + exact K/U fan-in", size=11.5, weight=700)
text(786, 246, "K/U layout", size=10.5, weight=700, fill=MUTED)
rect(850, 234, 42, 20, fill=BLUE_FILL, stroke=BLUE, sw=1.1, rx=3)
rect(892, 234, 45, 20, fill=AMBER_FILL, stroke=AMBER, sw=1.1, rx=3)
rect(937, 234, 40, 20, fill=VIOLET_FILL, stroke=VIOLET, sw=1.1, rx=3)
rect(977, 234, 61, 20, fill=AMBER_FILL, stroke=AMBER, sw=1.1, rx=3)
text(871, 248, "V", size=10.5, weight=700, fill=BLUE, anchor="middle")
text(914.5, 248, "M_{t-1}", size=9.2, weight=700, fill=AMBER, anchor="middle")
text(957, 248, "A", size=10.5, weight=700, fill=VIOLET, anchor="middle")
text(1007.5, 248, "L cache", size=10, weight=700, fill=AMBER, anchor="middle")

rect(786, 266, 103, 34, fill=BLUE_FILL_2, stroke=BLUE, sw=1.2, rx=5)
rect(937, 266, 101, 34, fill=VIOLET_FILL_2, stroke=VIOLET, sw=1.2, rx=5)
text(837.5, 287, "LN(V) → Q_v", size=11.5, weight=700, fill=BLUE, anchor="middle")
text(987.5, 287, "LN(A) → Q_a", size=11.5, weight=700, fill=VIOLET, anchor="middle")

rect(837, 315, 164, 45, fill=WHITE, stroke=VIOLET, sw=1.7, rx=7)
text(919, 333, "Joint multi-head attention", size=12.5, weight=700, anchor="middle")
text(919, 350, "8 heads × 64 · scale = 1/√64", size=10.5, fill=MUTED, anchor="middle")
straight_arrow(837, 300, 875, 315, "Q_v", label_dx=-4, label_dy=-6, color=BLUE, label_size=9.5)
straight_arrow(988, 300, 964, 315, "Q_a", label_dx=5, label_dy=-6, color=VIOLET, label_size=9.5)
path_arrow(
    [(944, 254), (944, 305), (933, 315)],
    color=AMBER,
    sw=1.6,
    label="K/U",
    label_at=(960, 303),
    label_size=9.5,
)

rect(774, 375, 136, 58, fill=BLUE_FILL_2, stroke=BLUE, sw=1.2, rx=5)
rect(914.5, 375, 136, 58, fill=VIOLET_FILL_2, stroke=VIOLET, sw=1.2, rx=5)
multiline(842, 391, ["out_v + residual", "LN → FFN_v", "512→2048→512 · GELU"], size=9.8, line_gap=15, weight=700, anchor="middle")
multiline(982.5, 391, ["out_a + residual", "LN → FFN_a", "512→2048→512 · GELU"], size=9.8, line_gap=15, weight=700, anchor="middle")
straight_arrow(883, 360, 842, 375, "split V", label_dx=-7, label_dy=-5, color=BLUE, label_size=9.5)
straight_arrow(958, 360, 982, 375, "split A", label_dx=8, label_dy=-5, color=VIOLET, label_size=9.5)

text(786, 463, "V_t^(i)  [B, ≤64, 512]", size=11.2, weight=700, fill=BLUE)
text(1038, 463, "A_t^(i)  [B, 8, 512]", size=11.2, weight=700, fill=VIOLET, anchor="end")
text(914, 486, "updated V/A feed the next VA layer", size=10.5, weight=600, fill=MUTED, anchor="middle")
straight_arrow(842, 433, 842, 451, "emit V", label_dx=-27, label_dy=0, color=BLUE, label_size=9.3)
straight_arrow(982, 433, 982, 451, "emit A", label_dx=27, label_dy=0, color=VIOLET, label_size=9.3)

# One-step, same-layer memory recurrence.
path_arrow(
    [(842, 451), (750, 451), (750, 247), (892, 247)],
    color=AMBER,
    sw=2.1,
    dash="7 5",
    label="M_{t-1}^(i) · same-layer feedback",
    label_at=(742, 368),
    label_size=10,
    label_anchor="end",
)

# Dedicated innovation callout for the temporal recurrence.
rect(18, 379, 410, 92, fill=AMBER_FILL, stroke=AMBER, sw=1.7, rx=8)
text(31, 398, "INNOVATION #1 · ONE-STEP LAYERWISE VISUAL MEMORY", size=11.5, weight=700, fill=AMBER)
text(31, 419, "For each VA layer i at step t:", size=11.5, weight=600)
text(31, 438, "M_{t-1}^(i) = visual output of the same layer at step t−1", size=11.5, weight=600)
text(31, 457, "Overwrite after every step; memory length is constant and never grows.", size=10.8, fill=MUTED)

# Role-mask note connected to attention.
rect(458, 379, 278, 92, fill=AMBER_FILL, stroke=AMBER, sw=1.6, rx=8)
text(471, 397, "ROLE MASK", size=11, weight=700, fill=AMBER, letter_spacing=0.7)
text(471, 416, "bidir_va: V/A queries → V, M, A, L", size=11.5, weight=600)
text(471, 434, "uni_a: V queries → V only; A → all", size=11.5, weight=600)
text(471, 452, "variant flag: per-head RMSNorm(Q,K)", size=10.5, fill=MUTED)
path_arrow(
    [(736, 414), (748, 414), (748, 339), (837, 339)],
    color=AMBER,
    sw=1.6,
    label="apply mask",
    label_at=(782, 334),
    label_size=9.5,
)

# Flow-matching head.
rect(1085, 136, 299, 365, fill=GREEN_FILL, stroke=GREEN, sw=2.2, rx=12)
text(1101, 158, "TRAINABLE / LIGHT-WEIGHT", size=11, weight=700, fill=GREEN, letter_spacing=0.7)
text(1101, 179, "FlowMatchingHead", size=18, weight=700)
text(1101, 195, "only this head repeats during sampling", size=11.5, fill=MUTED)

rect(1101, 207, 267, 42, fill=VIOLET_FILL_2, stroke=VIOLET, sw=1.3, rx=6)
text(1234.5, 224, "C_t = LayerNorm(A_t^8)", size=13, weight=700, fill=VIOLET, anchor="middle")
text(1234.5, 241, "[B, 8, 512]", size=11, fill=MUTED, anchor="middle")
path_arrow(
    [(1038, 452), (1077, 452), (1077, 228), (1101, 228)],
    color=VIOLET,
    sw=2,
    label="final action tokens → LN",
    label_at=(1072, 270),
    label_size=10,
    label_anchor="end",
)

rect(1101, 268, 126, 59, fill=GREEN_FILL_2, stroke=GREEN, sw=1.3, rx=6)
text(1164, 285, "Noisy actions a^tau", size=11.5, weight=700, anchor="middle")
text(1164, 302, "[B, 8, 7]", size=10.5, fill=MUTED, anchor="middle")
text(1164, 318, "Linear 7 → 512", size=10.5, fill=MUTED, anchor="middle")

rect(1240, 268, 128, 59, fill=GREEN_FILL_2, stroke=GREEN, sw=1.3, rx=6)
text(1304, 284, "Flow time tau", size=11.5, weight=700, anchor="middle")
text(1304, 300, "sin/cos: 256 + 256", size=10, fill=MUTED, anchor="middle")
text(1304, 316, "512→1024→512 · SiLU", size=9.8, fill=MUTED, anchor="middle")

rect(1165, 349, 139, 35, fill=WHITE, stroke=GREEN, sw=1.4, rx=6)
text(1234.5, 371, "sum: C_t + a^tau + time", size=11.5, weight=700, fill=GREEN, anchor="middle")
straight_arrow(1234.5, 249, 1234.5, 349, "condition", label_dx=-34, label_dy=-30, color=VIOLET, label_size=9.5)
path_arrow(
    [(1164, 327), (1164, 340), (1205, 349)],
    color=GREEN,
    sw=1.6,
    label="project",
    label_at=(1171, 346),
    label_size=9.2,
)
path_arrow(
    [(1304, 327), (1304, 340), (1265, 349)],
    color=GREEN,
    sw=1.6,
    label="embed",
    label_at=(1298, 346),
    label_size=9.2,
)

rect(1125, 399, 219, 43, fill=WHITE, stroke=GREEN, sw=1.4, rx=6)
text(1234.5, 417, "Transformer encoder ×2", size=12.5, weight=700, anchor="middle")
text(1234.5, 435, "d=512 · 8 heads · FFN 2048 · GELU · pre-norm", size=9.6, fill=MUTED, anchor="middle")
straight_arrow(1234.5, 384, 1234.5, 399, "denoise", label_dx=34, label_dy=-3, color=GREEN, label_size=9.3)

rect(1125, 450, 219, 41, fill=GREEN_FILL_2, stroke=GREEN, sw=1.4, rx=6)
text(1234.5, 467, "LayerNorm → velocity head 512→7", size=11, weight=700, anchor="middle")
text(1234.5, 484, "predicted velocity  [B, 8, 7]", size=10.2, fill=MUTED, anchor="middle")
straight_arrow(1234.5, 442, 1234.5, 450, "decode", label_dx=32, label_dy=2, color=GREEN, label_size=9.2)

# Sampling/output strip.
rect(1085, 506, 150, 36, fill=GREEN_FILL, stroke=GREEN, sw=1.5, rx=7)
text(1160, 521, "8-step Euler  tau: 0 → 1", size=11.2, weight=700, fill=GREEN, anchor="middle")
text(1160, 535, "C_t fixed; VA stack ran once", size=9.5, fill=MUTED, anchor="middle")
rect(1251, 506, 72, 36, fill=GREEN_FILL, stroke=GREEN, sw=1.5, rx=7)
text(1287, 521, "action chunk", size=10.2, weight=700, anchor="middle")
text(1287, 535, "[B, 8, 7]", size=10.2, fill=MUTED, anchor="middle")
rect(1339, 503, 49, 39, fill=GRAY_FILL, stroke=GRAY, sw=1.5, rx=7)
text(1363.5, 516, "robot ctrl.", size=8.8, weight=700, anchor="middle")
text(1363.5, 528, "external", size=8.5, fill=MUTED, anchor="middle")
text(1363.5, 539, "not in repo", size=8.2, fill=MUTED, anchor="middle")
path_arrow(
    [(1234.5, 491), (1234.5, 499), (1160, 499), (1160, 506)],
    color=GREEN,
    sw=1.7,
    label="v_theta each step",
    label_at=(1197, 497),
    label_size=9.2,
)
straight_arrow(1235, 524, 1251, 524, "integrate", label_dy=-7, color=GREEN, label_size=8.5)
straight_arrow(1323, 524, 1339, 524, "execute", label_dy=-7, color=GRAY, label_size=8.5)

# Bottom training band.
rect(10, 550, 1382, 70, fill=GRAY_FILL, stroke=LIGHT_LINE, sw=1.4, rx=9)
text(22, 566, "TRAINING BAND", size=10.5, weight=700, fill=MUTED, letter_spacing=1.0)
text(
    139, 566,
    "precomputed-feature path; at runtime the frozen backbones produce these features online",
    size=10.5,
    weight=600,
    fill=MUTED,
)
box(
    22,
    575,
    292,
    35,
    ["Precomputed-feature dataset", "vision tokens · language hidden · proprio · actions"],
    fill=WHITE,
    stroke=GRAY,
    title_size=11.5,
    body_size=9.6,
    top_pad=14,
    rx=5,
    sw=1.3,
)
box(
    351,
    575,
    239,
    35,
    ["Paired multi-instruction sampling", "same first state; distinct instruction/action"],
    fill=WHITE,
    stroke=GRAY,
    title_size=11.2,
    body_size=9.4,
    top_pad=14,
    rx=5,
    sw=1.3,
)
box(
    640,
    572,
    380,
    41,
    ["Flow MSE  +  λ · paired consistency", "Smooth-L1 on language-caused velocity difference at τ=0"],
    fill=RED_FILL,
    stroke=RED,
    title_size=12,
    body_size=9.8,
    top_pad=15,
    rx=5,
    sw=1.5,
)
box(
    1070,
    575,
    151,
    35,
    ["AdamW", "weight decay 1e-4"],
    fill=WHITE,
    stroke=GRAY,
    title_size=12,
    body_size=9.8,
    top_pad=14,
    rx=5,
    sw=1.3,
)
text(1238, 588, "short rollout T ≥ 4", size=10.5, weight=700, fill=MUTED)
text(1238, 605, "memory carries across t", size=10.2, fill=MUTED)
straight_arrow(314, 593, 351, 593, "pair batches", label_dy=-7, color=GRAY, label_size=9.3)
straight_arrow(590, 593, 640, 593, "FM + τ=0 pair", label_dy=-7, color=RED, label_size=9.1)
straight_arrow(1020, 593, 1070, 593, "backprop", label_dy=-7, color=RED, label_size=9.3)


svg = "\n".join(
    [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="178mm" height="80mm" viewBox="0 0 {W} {H}" '
            'shape-rendering="geometricPrecision" text-rendering="optimizeLegibility">'
        ),
        *elements,
        "</svg>",
        "",
    ]
)

SVG_PATH.write_text(svg, encoding="utf-8")
print(SVG_PATH)
