"""Fig 9: MetaWorld MT50 — open-loop, language ablation, closed-loop.

Data: §8.2 (open loop chunk MAE vs persistence), §8.3 (wrong/taskid ablation),
§8.5 (closed loop). Closed-loop rows are parsed from the actual run logs so
the figure always matches the logs (audit-friendly).
Run: python artifacts/figures/make_fig_mw.py
"""
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")


def closed_loop_from_log(name, logfile):
    """Parse 'CLOSED-LOOP SUCCESS: X/Y = Z%' and 'macro ... [95% CI: a%, b%]'."""
    path = os.path.join(LOG, logfile)
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        text = f.read()
    m = re.search(r"CLOSED-LOOP SUCCESS:\s*([\d.]+)\s*/\s*([\d.]+)\s*=\s*([\d.]+)%", text)
    ci = re.search(r"95% CI:\s*([\d.]+)%,\s*([\d.]+)%", text)
    if not m:
        return None, None
    rate = float(m.group(3))
    bounds = tuple(float(x) for x in ci.groups()) if ci else None
    return rate, bounds


# ---- closed-loop rows (parsed from logs; None until the run exists) ----
_rows = [
    ("v5 direct 40k", "mw_v5_direct_closedloop.log"),
    ("C2 joint30k (pilot)", "mw_pilot_c2_joint30k_closedloop.log"),
    ("C2 full 40k (mainline)", "mw_v5_c2_full_closedloop.log"),
]
CLOSED_LOOP = {}
CLOSED_LOOP_CI = {}
for label, logfile in _rows:
    rate, bounds = closed_loop_from_log(label, logfile)
    CLOSED_LOOP[label] = rate
    CLOSED_LOOP_CI[label] = bounds

# v5 direct 开环/消融（chunk@all-seq，logs/mw_v5_direct_ablation.log）
OPEN_LOOP = {"model": 0.0251, "persistence": 0.1354}  # v5 direct 32-step chunk MAE
OPEN_LOOP_CI = (0.0229, 0.0273)
ABLATION = {"clean": 0.02504, "wrong": 0.25774, "task-id": 0.23823}  # chunk@all-seq
LIT = {"SmolVLA 0.45B": 57.3, "Evo-1": 80.6, "FabriVLA": 90.0}  # closed-loop literature anchors
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('MetaWorld MT50 (49 tasks, 40k steps, 32-step flow)', fontsize=13, fontweight='bold')

# (a) open loop
ax = axes[0]
keys = ['model', 'persistence']
vals = [OPEN_LOOP[k] for k in keys]
bars = ax.bar(keys, vals, color=['#55A868', '#C44E52'], width=0.5)
ax.errorbar([0], [OPEN_LOOP["model"]], yerr=[[OPEN_LOOP["model"]-OPEN_LOOP_CI[0]], [OPEN_LOOP_CI[1]-OPEN_LOOP["model"]]],
            fmt='none', ecolor='black', capsize=3)
ax.set_ylabel('chunk MAE (norm)')
ax.set_title('(a) Open-loop accuracy', fontsize=11)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.4f}', ha='center', fontsize=9)

# (b) ablation
ax = axes[1]
keys = list(ABLATION)
vals = list(ABLATION.values())
bars = ax.bar(keys, vals, color=['#55A868', '#C44E52', '#F0A500'], width=0.5)
ax.set_ylabel('chunk MAE (norm)')
ax.set_title('(b) Language ablation', fontsize=11)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.4f}', ha='center', fontsize=9)

# (c) closed loop vs literature
ax = axes[2]
keys = list(CLOSED_LOOP) + list(LIT)
vals = [CLOSED_LOOP[k] if CLOSED_LOOP[k] is not None else 0 for k in list(CLOSED_LOOP)] + list(LIT.values())
colors = (['#C44E52', '#55A868', '#F0A500', '#4C72B0'])[:len(CLOSED_LOOP)] + ['#AAAAAA']*len(LIT)
bars = ax.bar(range(len(keys)), vals, color=colors, width=0.55)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=15, fontsize=8)
ax.set_ylabel('closed-loop success %')
ax.set_title('(c) Closed-loop vs literature', fontsize=11)
for i, (k, v) in enumerate(zip(keys, vals)):
    if CLOSED_LOOP.get(k) is None and k in CLOSED_LOOP:
        ax.text(i, 2, 'TBD', ha='center', fontsize=9, color='#999999')
    else:
        ax.text(i, v+1.5, f'{v:.1f}', ha='center', fontsize=8)
    if k in CLOSED_LOOP_CI and CLOSED_LOOP_CI[k] is not None and CLOSED_LOOP_CI[k][0] is not None:
        lo, hi = CLOSED_LOOP_CI[k]
        ax.errorbar(i, v, yerr=[[v-lo], [hi-v]], fmt='none', ecolor='black', capsize=3)

fig.tight_layout()
out = '/home/ryan/Documents/robot/ORA0/paper/fig9_mw.png'
fig.savefig(out, dpi=130)
print('saved', out)
