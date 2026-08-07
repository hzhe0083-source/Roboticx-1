#!/usr/bin/env python3
"""Geometric preflight for the contribution figure (drawio XML -> issues)."""
import re

xml = open('contribution_figure.drawio', encoding='utf-8').read()

verts = {}
for m in re.finditer(
    r'<mxCell id="([^"]+)" value="([^"]*)" style="([^"]*)" vertex="1" parent="1">\s*<mxGeometry x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"',
    xml,
):
    cid, val, style, x, y, w, h = m.groups()
    verts[cid] = (float(x), float(y), float(w), float(h), val, style)

edges = len(re.findall(r'<mxCell id="[^"]+" value="[^"]*" style="[^"]*edgeStyle', xml))
W, H = 1900, 1000
issues = []

# 1) out-of-canvas
for cid, (x, y, w, h, v, s) in verts.items():
    if x < 0 or y < 0 or x + w > W or y + h > H:
        issues.append(f"OUT-OF-CANVAS {cid} ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")

# 2) real overlaps: ignore container-child pairs
containers = {
    'stack': ['stack_h1', 'stack_h2', 'x8', 'q_v', 'q_a', 'attn', 'ffn_v', 'ffn_a', 'out_v', 'out_a'],
    'flow': ['flow_h1', 'flow_h2', 'c_t', 'noisy', 'sum', 'enc', 'head', 'euler', 'chunk'],
    'card1': ['c1_t', 'c1_s', 'c1_blank', 'c1_swap', 'c1_note'],
    'card2': ['c2_t', 'c2_s', 'c2_blank', 'c2_swap', 'c2_note'],
    'card3': ['c3_t', 'c3_s', 'c3_blank', 'c3_swap', 'c3_note'],
    'banner': ['ban_main', 'ban_s1', 'ban_s2'],
    'c_band': ['c_t1', 'c_t2', 'c_t3'],
}
child_of = {}
for cont, kids in containers.items():
    for k in kids:
        child_of[k] = cont


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ox * oy


keys = list(verts)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        # skip if one is inside the other's container
        if child_of.get(a) == b or child_of.get(b) == a:
            continue
        if child_of.get(a) == child_of.get(b) and child_of.get(a) is not None:
            continue  # siblings inside same container are intentionally stacked
        bb_a = verts[a][:4]
        bb_b = verts[b][:4]
        ov = overlap(bb_a, bb_b)
        if ov > 400:
            issues.append(f"OVERLAP {a} x {b} = {int(ov)}px^2")

# 3) single-line text width estimate vs box (per longest line)
def line_width(line, size):
    w = 0
    for ch in line:
        if ord(ch) > 0x2E80 or ch in '—·≤√∈×':
            w += size
        else:
            w += size * 0.56
    return w

for cid, (x, y, w, h, v, s) in verts.items():
    if not v:
        continue
    m = re.search(r'fontSize=([\d.]+)', s)
    size = float(m.group(1)) if m else 12
    lines = v.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').split('&#10;')
    maxw = max(line_width(l, size) for l in lines)
    if maxw > w * 1.15:
        issues.append(f"TEXT-OVERFLOW {cid}: est {maxw:.0f}px vs box {w:.0f}px: {lines[0][:50]}")

print(f"vertices={len(verts)} edges={edges}")
if issues:
    print("ISSUES:")
    for i in issues:
        print(" -", i)
else:
    print("no geometric issues")
