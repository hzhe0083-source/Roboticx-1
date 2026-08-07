#!/usr/bin/env python3
"""Generate the VA Compound contribution figure as an editable draw.io file.

Layout: 1900 x 1000 canvas.
  - A (left):   architecture carrier - constant-memory recursive visual coupling
  - B (right):  THE contribution - in-architecture causal evidence of language
                grounding (Blank/Swap interventions across three datasets)
  - C (bottom): two-level loss simplicity

Outputs:
  contribution_figure.drawio       - editable draw.io source
  preview.html                     - embed.diagrams.net viewer with the XML
                                     injected (rendered via Chrome headless)

Sources: model.py (VACouplingLayer / VisualMemory / LanguageCache),
         arch_review_positioning.md (evidence numbers, 2026-08-05),
         make_figures.py / make_fig7.py (same numbers, 32-step deployment setup).
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

W, H = 1900, 1000

# ---- palette (matches figure1_va_compound + va_compound_architecture) ----
INK = "#0F172A"
MUTED = "#475569"
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
GREEN_DARK = "#2E8B57"
AMBER = "#C48425"
AMBER_FILL = "#FFF3DB"
GRAY = "#64748B"
GRAY_FILL = "#F1F5F9"
RED = "#B34A4A"
RED_FILL = "#FDECEC"
WHITE = "#FFFFFF"

FONT = "Helvetica"

cells: list[str] = []
_edge_id = 0


def esc(value: str) -> str:
    # protect drawio newline entity from double-escaping
    v = value.replace("&#10;", "\x00NL\x00")
    v = (
        v.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return v.replace("\x00NL\x00", "&#10;")


def nl(lines: list[str]) -> str:
    return "&#10;".join(lines)


def vcell(
    cid: str,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = WHITE,
    stroke: str = INK,
    sw: float = 1.7,
    font_size: float = 12,
    font_color: str = INK,
    weight: int = 400,
    align: str = "center",
    rounding: float = 6,
    dashed: bool = False,
) -> None:
    style = (
        f"rounded=1;arcSize={rounding};whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
        f"fontFamily={FONT};fontSize={font_size};fontColor={font_color};"
        f"fontStyle={1 if weight >= 700 else 0};align={align};verticalAlign=middle;"
    )
    if dashed:
        style += "dashed=1;dashPattern=8 6;"
    cells.append(
        f'<mxCell id="{cid}" value="{esc(value)}" style="{style}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f"</mxCell>"
    )


def eedge(
    cid: str,
    src: str,
    tgt: str,
    *,
    color: str = INK,
    sw: float = 1.7,
    dashed: bool = False,
    label: str | None = None,
    label_size: float = 10,
    label_color: str | None = None,
    exit_xy: tuple[float, float] | None = None,
    entry_xy: tuple[float, float] | None = None,
    orthogonal: bool = True,
    points: list[tuple[float, float]] | None = None,
    label_offset: tuple[float, float] | None = None,
) -> None:
    global _edge_id
    if cid is None:
        cid = f"e{_edge_id}"
    _edge_id += 1
    style = (
        f"edgeStyle={'orthogonalEdgeStyle' if orthogonal else 'none'};"
        f"rounded=0;html=1;strokeColor={color};strokeWidth={sw};"
        f"fontFamily={FONT};fontSize={label_size};"
    )
    if label_color:
        style += f"fontColor={label_color};"
    if label:
        style += "labelBackgroundColor=#FFFFFF;"
    if dashed:
        style += "dashed=1;dashPattern=8 6;"
    if exit_xy:
        style += f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
    if entry_xy:
        style += f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
    geom = '<mxGeometry relative="1" as="geometry">'
    if points:
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points)
        geom += f'<Array as="points">{pts}</Array>'
    if label_offset:
        geom += f'<mxPoint x="{label_offset[0]}" y="{label_offset[1]}" as="offset"/>'
    geom += "</mxGeometry>"
    cells.append(
        f'<mxCell id="{cid}" value="{esc(label or "")}" style="{style}" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">{geom}</mxCell>'
    )


def header(cid: str, x: float, y: float, w: float, label: str, color: str, size: float = 15) -> None:
    vcell(
        cid,
        label,
        x,
        y,
        w,
        30,
        fill=WHITE,
        stroke=WHITE,
        sw=0,
        font_size=size,
        font_color=color,
        weight=700,
        align="left",
        rounding=0,
    )


def invis(cid: str, x: float, y: float, w: float, h: float) -> None:
    """Invisible label cell (for text inside a container, no own border)."""
    vcell(cid, "", x, y, w, h, fill=WHITE, stroke=WHITE, sw=0)


# =====================================================================
# Title
# =====================================================================
vcell(
    "title",
    "<b>Constant-Memory Recursive Visual Coupling for Language-Grounded Control</b>",
    40, 14, 1250, 36,
    fill=WHITE, stroke=WHITE, sw=0, font_size=23, align="left",
)
vcell(
    "subtitle",
    "VA Compound · 决策层 43.5M · 冻结骨干 · 32 步部署口径",
    1290, 20, 570, 26,
    fill=WHITE, stroke=WHITE, sw=0, font_size=12.5, font_color=MUTED,
    align="right",
)

# =====================================================================
# A · Architecture carrier (left)
# =====================================================================
header("a_header", 40, 62, 780, "A · 架构：恒定内存递归视觉耦合  constant-memory recursive visual coupling", VIOLET)

# inputs: language row
vcell("instr", nl(["指令文本", "≤64 tokens"]), 40, 120, 80, 58,
      fill=GRAY_FILL, stroke=GRAY, font_size=12.5)
vcell("qwen", nl(["Frozen Qwen3.5-2B", "language_model only", "no vision / LM head"]), 170, 112, 180, 74,
      fill=BLUE_FILL, stroke=BLUE, font_size=11.5)
vcell("l_cache", nl(["逐层语言缓存 L", "每层 LN → k_l / u_l", "K_l / U_l [B, 8, ≤64, 64]"]), 830, 66, 170, 54,
      fill=AMBER_FILL, stroke=AMBER, font_size=11)
eedge("a_instr", "instr", "qwen", color=GRAY, label="tokenize", label_size=9.5)
eedge("a_qwen", "qwen", "l_cache", color=AMBER, sw=1.9, orthogonal=False,
      label="encode once → 每层 fan-out", label_size=9.5)

# inputs: vision row
vcell("video", nl(["视频窗口", "F ≥ 2", "[B, F, 3, H, W]"]), 40, 205, 110, 58,
      fill=GRAY_FILL, stroke=GRAY, font_size=11.5)
vcell("vjepa", nl(["Frozen V-JEPA 2.1", "ViT-B/384", "per observation window"]), 170, 197, 180, 74,
      fill=BLUE_FILL, stroke=BLUE, font_size=11.5)
vcell("vtoks", nl(["V_t 视觉 token", "1D pool [t,h,w]", "[B, ≤64, 512]"]), 400, 205, 200, 58,
      fill=BLUE_FILL_2, stroke=BLUE, font_size=11.5)
eedge("a_video", "video", "vjepa", color=BLUE, sw=1.7)
eedge("a_pool", "vjepa", "vtoks", color=BLUE, label="[t,h,w]", label_size=9.5)

# inputs: state row
vcell("state", nl(["proprio [B,14]", "prev action [B,7]"]), 40, 290, 110, 58,
      fill=GRAY_FILL, stroke=GRAY, font_size=12)
vcell("atoks", nl(["A_t 动作 token", "8 learned queries", "+ state emb", "[B, 8, 512]"]), 215, 290, 150, 66,
      fill=VIOLET_FILL, stroke=VIOLET, font_size=11.5)
eedge("a_state", "state", "atoks", color=GRAY, label="concat+proj", label_size=9)

# ---- decision stack ----
vcell("stack", "", 620, 120, 380, 590, fill=VIOLET_FILL, stroke=VIOLET, sw=2.2, rounding=12)
vcell("stack_h1", "VACouplingLayer", 635, 132, 240, 26,
      fill=VIOLET_FILL, stroke=VIOLET, sw=0, font_size=18, weight=700, align="left")
vcell("stack_h2", "单层展开 · layers 1...8", 635, 158, 240, 20,
      fill=VIOLET_FILL, stroke=VIOLET, sw=0, font_size=11.5, font_color=MUTED, align="left")
vcell("x8", "×8", 900, 132, 46, 26, fill=WHITE, stroke=VIOLET, sw=1.5,
      font_size=13, weight=700, rounding=13)

vcell("q_v", "LN(V) → Q_v", 645, 185, 110, 40, fill=BLUE_FILL_2, stroke=BLUE, font_size=12)
vcell("q_a", "LN(A) → Q_a", 645, 265, 110, 40, fill=VIOLET_FILL_2, stroke=VIOLET, font_size=12)
vcell(
    "attn",
    nl(["联合注意力 · 8 heads × 64", "K/U = [V, M_{t-1}, A, L]", "role mask: bidir_va / uni_a", "scale = 1/√64 · fp32"]),
    645, 345, 270, 95,
    fill=WHITE, stroke=VIOLET, sw=1.7, font_size=11.5,
)
vcell("ffn_v", nl(["out_v + residual", "LN → FFN_v", "512 → 2048 → 512"]), 645, 475, 110, 60,
      fill=BLUE_FILL_2, stroke=BLUE, font_size=10.5)
vcell("ffn_a", nl(["out_a + residual", "LN → FFN_a", "512 → 2048 → 512"]), 775, 475, 110, 60,
      fill=VIOLET_FILL_2, stroke=VIOLET, font_size=10.5)
vcell("out_v", nl(["V_t^i 更新", "→ 下一层"]), 645, 560, 110, 40, fill=BLUE_FILL, stroke=BLUE, font_size=11)
vcell("out_a", nl(["A_t^i 更新", "→ 下一层"]), 775, 560, 110, 40, fill=VIOLET_FILL, stroke=VIOLET, font_size=11)

# stack internal edges
eedge("e_qv", "q_v", "attn", color=BLUE, sw=1.5, orthogonal=False)
eedge("e_qa", "q_a", "attn", color=VIOLET, sw=1.5, orthogonal=False)
eedge("e_split_v", "attn", "ffn_v", color=BLUE, label="V", label_size=10, sw=1.5,
      orthogonal=False, label_offset=(10, 0))
eedge("e_split_a", "attn", "ffn_a", color=VIOLET, label="A", label_size=10, sw=1.5,
      orthogonal=False, label_offset=(-10, 0))
eedge("e_emit_v", "ffn_v", "out_v", color=BLUE, sw=1.5, orthogonal=False)
eedge("e_emit_a", "ffn_a", "out_a", color=VIOLET, sw=1.5, orthogonal=False)

# external fan-in: V_t / A_t / L / M_{t-1}
# (V_t label removed: the q_v box "LN(V) → Q_v" already carries the semantics;
#  a label at the stack border would clip the container frame)
eedge("e_vt", "vtoks", "q_v", color=VIOLET, sw=1.9, orthogonal=False)
eedge("e_at", "atoks", "q_a", color=VIOLET, label="A_t", label_size=11, sw=1.9,
      label_offset=(0, -10))
eedge("e_l", "l_cache", "attn", color=AMBER, label="L K/U", label_size=10, sw=1.9,
      exit_xy=(0.5, 1), entry_xy=(1.0, 0))
# memory recurrence: out_v -> left -> attention (dashed amber), explicit waypoints
eedge("e_mem", "out_v", "attn", color=AMBER, sw=2.1, dashed=True,
      label="M_t ← V_t^i &#10;同层递推 · 内存恒定", label_size=10,
      exit_xy=(0.0, 0.5), entry_xy=(0.0, 0.3),
      points=[(600, 580), (600, 373)], label_offset=(-110, 0))

# ---- flow matching head ----
vcell("flow", "", 1010, 120, 200, 590, fill=GREEN_FILL, stroke=GREEN, sw=2.2, rounding=12)
vcell("flow_h1", "FlowMatchingHead", 1025, 128, 180, 26,
      fill=GREEN_FILL, stroke=GREEN, sw=0, font_size=18, weight=700, align="left")
vcell("flow_h2", "轻量 · 仅采样期重复", 1025, 154, 180, 18,
      fill=GREEN_FILL, stroke=GREEN, sw=0, font_size=11.5, font_color=MUTED, align="left")
vcell("c_t", nl(["C_t = LN(A_t^8)", "[B, 8, 512]"]), 1030, 178, 160, 44,
      fill=VIOLET_FILL_2, stroke=VIOLET, font_size=12)
vcell("noisy", nl(["噪声动作 a^τ", "+ flow time τ emb", "[B, 8, 7] → 512"]), 1030, 224, 160, 50,
      fill=GREEN_FILL_2, stroke=GREEN, font_size=10.5)
vcell("sum", "sum: C_t + a^τ + time", 1030, 300, 160, 40, fill=WHITE, stroke=GREEN, font_size=11.5)
vcell("enc", nl(["Transformer enc ×2", "d=512 · 8 heads · pre-norm"]), 1030, 370, 160, 50,
      fill=GREEN_FILL_2, stroke=GREEN, font_size=10.5)
vcell("head", nl(["velocity head", "LN → 512 → 7"]), 1030, 450, 160, 50,
      fill=GREEN_FILL_2, stroke=GREEN, font_size=10.5)
vcell("euler", nl(["8-step Euler", "τ : 0 → 1"]), 1030, 530, 160, 44, fill=GREEN_FILL, stroke=GREEN, font_size=11)
vcell("chunk", nl(["action chunk", "[B, 8, 7]"]), 1030, 600, 160, 44, fill=GREEN_FILL, stroke=GREEN_DARK, font_size=11.5)

eedge("e_flow_in", "out_a", "c_t", color=VIOLET, sw=2, label="A_t^8 → LN", label_size=10,
      exit_xy=(1.0, 0.5), entry_xy=(0.0, 0.5),
      points=[(1015, 580), (1015, 200)], label_offset=(-55, 0))
eedge("e_c", "c_t", "sum", color=VIOLET, sw=1.5, exit_xy=(1.0, 0.5), entry_xy=(1.0, 0.5),
      points=[(1205, 200), (1205, 300)])
eedge("e_noisy", "noisy", "sum", color=GREEN, sw=1.5, label="project + embed", label_size=9.5,
      label_offset=(14, 0))
eedge("e_denoise", "sum", "enc", color=GREEN, sw=1.5, orthogonal=False)
eedge("e_decode", "enc", "head", color=GREEN, sw=1.5, orthogonal=False)
eedge("e_vel", "head", "euler", color=GREEN, sw=1.5, label="v_θ 每步", label_size=9.5,
      label_offset=(14, 0))
eedge("e_int", "euler", "chunk", color=GREEN, sw=1.5, label="integrate", label_size=9.5,
      label_offset=(14, 0))

# =====================================================================
# B · Core contribution evidence (right) - visual protagonist
# =====================================================================
header("b_header", 1230, 62, 670, "B · 核心贡献：语言 grounding 的架构内因果证据", RED)
vcell(
    "b_note",
    "同一策略 · 同一 role mask · 同一干预（blank / swap 语言流）——仅数据指令结构不同",
    1230, 96, 670, 24,
    fill=WHITE, stroke=WHITE, sw=0, font_size=11, font_color=MUTED, align="left",
)

# card 1: PNPW
vcell("card1", "", 1230, 134, 670, 104, fill=GRAY_FILL, stroke=GRAY, sw=1.5, rounding=8)
vcell("c1_t", "PNPW 数据集", 1246, 148, 240, 24,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=15, weight=700, align="left")
vcell("c1_s", "常数指令 · 指令与动作一一对应", 1246, 174, 280, 20,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=10.5, font_color=MUTED, align="left")
vcell("c1_blank", "Blank  −0.1%", 1500, 152, 180, 24, fill=GRAY_FILL, stroke=GRAY, sw=0,
      font_size=15, weight=700, font_color=GREEN, align="left")
vcell("c1_swap", "Swap  −0.1%", 1500, 182, 180, 24, fill=GRAY_FILL, stroke=GRAY, sw=0,
      font_size=15, weight=700, font_color=GREEN, align="left")
vcell("c1_note", "语言冗余 → 屏蔽无影响", 1710, 165, 180, 20,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=10.5, font_color=MUTED, align="right")

# card 2: MetaWorld
vcell("card2", "", 1230, 254, 670, 104, fill=GRAY_FILL, stroke=GRAY, sw=1.5, rounding=8)
vcell("c2_t", "MetaWorld MT50", 1246, 268, 240, 24,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=15, weight=700, align="left")
vcell("c2_s", "视觉目标主导 · 49 任务", 1246, 294, 240, 20,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=10.5, font_color=MUTED, align="left")
vcell("c2_blank", "Blank  +0.1%", 1500, 272, 180, 24, fill=GRAY_FILL, stroke=GRAY, sw=0,
      font_size=15, weight=700, font_color=AMBER, align="left")
vcell("c2_swap", "Swap  +0.1%", 1500, 302, 180, 24, fill=GRAY_FILL, stroke=GRAY, sw=0,
      font_size=15, weight=700, font_color=AMBER, align="left")
vcell("c2_note", "语言冗余 → 屏蔽无影响", 1710, 285, 180, 20,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=10.5, font_color=MUTED, align="right")

# card 3: LIBERO - highlighted
vcell("card3", "", 1230, 374, 670, 130, fill=RED_FILL, stroke=RED, sw=2.4, rounding=8)
vcell("c3_t", "LIBERO · 3 场景 12 任务", 1246, 388, 300, 26,
      fill=RED_FILL, stroke=RED, sw=0, font_size=16, weight=700, font_color=RED, align="left")
vcell("c3_s", "同场景多指令 → 语言是唯一解", 1246, 416, 300, 22,
      fill=RED_FILL, stroke=RED, sw=0, font_size=11, font_color=MUTED, align="left")
vcell("c3_blank", "Blank  +2381%", 1500, 386, 220, 34, fill=RED_FILL, stroke=RED, sw=0,
      font_size=26, weight=700, font_color=RED, align="left")
vcell("c3_swap", "Swap  +607%", 1500, 426, 220, 28, fill=RED_FILL, stroke=RED, sw=0,
      font_size=20, weight=700, font_color=RED, align="left")
vcell("c3_note", "32 步部署口径 · 2026-08-05 实测", 1710, 460, 180, 20,
      fill=RED_FILL, stroke=RED, sw=0, font_size=10.5, font_color=MUTED, align="right")

# conclusion banner
vcell("banner", "", 1230, 528, 670, 112, fill=RED_FILL, stroke=RED, sw=2.2, rounding=10)
vcell("ban_main", "语言依赖是数据结构的函数，不是架构的", 1250, 540, 640, 30,
      fill=RED_FILL, stroke=RED, sw=0, font_size=18, weight=700, font_color=RED, align="left")
vcell("ban_s1", "因果证据来自架构内干预——而非注意力 mass 相关性（语言 attention 仅 0.2%，但 Blank 失效即崩）",
      1250, 576, 640, 22, fill=RED_FILL, stroke=RED, sw=0, font_size=11.5, align="left")
vcell("ban_s2", "7 篇 2026 竞品（ReMem / MemoryVLA / AVA / RB / CogVLA …）均无此类机制分析",
      1250, 602, 640, 22, fill=RED_FILL, stroke=RED, sw=0, font_size=11, font_color=MUTED, align="left")

# =====================================================================
# C · Two-level loss simplicity (bottom band)
# =====================================================================
vcell("c_band", "", 40, 880, 1820, 90, fill=GRAY_FILL, stroke=LIGHT_LINE, sw=1.4, rounding=9)
vcell("c_t1", "C · 两级损失简单性：two-level loss, no auxiliary objective", 60, 896, 780, 24,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=13.5, weight=700, align="left")
vcell("c_t2", "Flow MSE（Smooth-L1）+  λ · paired consistency（τ=0 处语言差异动作）", 60, 922, 900, 22,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=12, align="left")
vcell("c_t3", "对比 ReMem-VLA（图像重建 loss）· RB-VLA（world-model 目标）：无任何辅助损失", 60, 946, 900, 20,
      fill=GRAY_FILL, stroke=GRAY, sw=0, font_size=11, font_color=MUTED, align="left")

# =====================================================================
# Assemble
# =====================================================================
mxfile = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mxfile host="app.diagrams.net" agent="Codex" version="26.0.9" pages="1">\n'
    f'  <diagram id="va-contribution" name="VA Compound Core Contribution">\n'
    f'    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" '
    f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" '
    f'pageHeight="{H}" math="0" shadow="0">\n'
    f'      <root>\n'
    f'        <mxCell id="0"/>\n'
    f'        <mxCell id="1" parent="0"/>\n'
)
for c in cells:
    mxfile += f"        {c}\n"
mxfile += "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"

drawio_path = OUT_DIR / "contribution_figure.drawio"
drawio_path.write_text(mxfile, encoding="utf-8")

# ---- preview html with embedded XML ----
xml_embedded = mxfile.replace("\n", "\\n").replace("'", "\\'")
html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>VA Compound Core Contribution</title>
<style>
  html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #f7f7f7; }}
  iframe {{ border: 0; width: 100vw; height: 100vh; display: block; }}
</style>
</head>
<body>
<iframe id="drawio" src="https://embed.diagrams.net/?embed=1&proto=json&spin=1&ui=atlas&libraries=0&grid=0&pv=0&zoom=100zoom=1"></iframe>
<script>
const xml = '{xml_embedded}';
const diagramTitle = "VA Compound Core Contribution";
const DRAWIO_ORIGIN = 'https://embed.diagrams.net';
const iframe = document.getElementById('drawio');
let loaded = false;
function sendLoad() {{
  iframe.contentWindow.postMessage(JSON.stringify({{
    action: 'load',
    autosave: 0,
    modified: 0,
    title: diagramTitle,
    xml
  }}), DRAWIO_ORIGIN);
}}
window.addEventListener('message', (evt) => {{
  if (evt.source !== iframe.contentWindow || evt.origin !== DRAWIO_ORIGIN) return;
  let msg = evt.data;
  try {{ if (typeof msg === 'string') msg = JSON.parse(msg); }} catch (e) {{ return; }}
  if (!msg) return;
  if (!loaded && (msg.event === 'init' || msg.event === 'configure')) {{
    loaded = true;
    sendLoad();
  }}
}});
</script>
</body>
</html>
"""
(OUT_DIR / "preview.html").write_text(html, encoding="utf-8")

print("wrote", drawio_path)
print("wrote", OUT_DIR / "preview.html")
