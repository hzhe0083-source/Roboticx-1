"""Fig 7: 2x2 control — Qwen embedding cosine + behavioral language sensitivity.

Data sources (2026-08-06/07 final, mask-corrected last-token protocol, v2 12-task):
  cosine:  original 0.8573 / random 0.0023 / B40k 0.9994 / C1 0.8573 / C2 0.9984
  blank sensitivity (chunk MAE 32-step): A +2381% / B40k +1.5% / C1 +33.7% / C2 +0.4%
  clean chunk_mae: A 0.00254 / B40k 0.0759 / C1 0.08391 / C2 0.08390
Run: python artifacts/figures/make_fig7_2x2.py
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---- final numbers (logs/cosine_*.log, logs/eval_libero_e2e_*.log) ----
COSINE = {
    "original Qwen": 0.8573,
    "random feats": 0.0023,
    "A (frozen)": 0.8573,
    "B40k (LoRA)": 0.9994,
    "C1 (e2e frozen)": 0.8573,
    "C2 (LoRA only)": 0.9984,
}
BLANK_PCT = {
    "A (frozen)": 2381.0,
    "B40k (LoRA)": 1.5,
    "C1 (e2e frozen)": 33.7,
    "C2 (LoRA only)": 0.4,
}
CHUNK_MAE = {
    "A (frozen)": 0.00254,
    "B40k (LoRA)": 0.0759,
    "C1 (e2e frozen)": 0.08391,
    "C2 (LoRA only)": 0.08390,
}
# ------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('2x2 Control: Instruction Embedding Space vs Behavioral Language Sensitivity', fontsize=13, fontweight='bold')

# Panel 1: pairwise cosine
ax = axes[0]
keys = list(COSINE)
vals = [COSINE[k] if COSINE[k] is not None else 0 for k in keys]
colors = ['#4C72B0' if v < 0.9 else '#C44E52' if v > 0.95 else '#F0A500' for v in vals]
bars = ax.bar(range(len(keys)), vals, color=colors, width=0.55)
ax.axhline(0.8573, color='#888888', ls='--', lw=1)
ax.text(len(keys)-0.5, 0.867, 'original Qwen', color='#888888', fontsize=8, ha='right')
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(keys, rotation=15, fontsize=9)
ax.set_ylabel('pairwise cosine (12 instructions)')
ax.set_title('(a) Embedding collapse', fontsize=11)
for b, v, k in zip(bars, vals, keys):
    if COSINE[k] is not None:
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
    else:
        ax.text(b.get_x()+b.get_width()/2, 0.03, 'TBD', ha='center', fontsize=9, color='#999999')

# Panel 2: blank sensitivity (log scale)
ax = axes[1]
keys2 = list(BLANK_PCT)
vals2 = [BLANK_PCT[k] if BLANK_PCT[k] is not None else None for k in keys2]
colors2 = ['#C44E52' if v is not None and v > 100 else '#4C72B0' if v is not None else '#DDDDDD' for v in vals2]
xs2 = np.arange(len(keys2))
bars = ax.bar(xs2, [v if v is not None else 1.0 for v in vals2], color=colors2, width=0.55)
ax.set_yscale('symlog', linthresh=10)
ax.set_xticks(xs2)
ax.set_xticklabels(keys2, rotation=15, fontsize=9)
ax.set_ylabel('blank sensitivity (% chunk error change)')
ax.set_title('(b) Behavioral language grounding', fontsize=11)
for x, v in zip(xs2, vals2):
    if v is not None:
        ax.text(x, v*1.15 if v > 0 else 0.8, f'+{v:,.0f}%' if v >= 10 else f'+{v:.1f}%', ha='center', fontsize=9)
    else:
        ax.text(x, 2, 'TBD', ha='center', fontsize=9, color='#999999')

# Panel 3: clean chunk MAE
ax = axes[2]
keys3 = list(CHUNK_MAE)
vals3 = [CHUNK_MAE[k] if CHUNK_MAE[k] is not None else None for k in keys3]
xs3 = np.arange(len(keys3))
bars = ax.bar(xs3, [v if v is not None else 0 for v in vals3], color=['#55A868' if v is not None else '#DDDDDD' for v in vals3], width=0.55)
ax.set_xticks(xs3)
ax.set_xticklabels(keys3, rotation=15, fontsize=9)
ax.set_ylabel('clean chunk_mae (norm)')
ax.set_title('(c) Clean-task accuracy (no trade-off?)', fontsize=11)
for x, v in zip(xs3, vals3):
    if v is not None:
        ax.text(x, v+0.002, f'{v:.4f}', ha='center', fontsize=9)
    else:
        ax.text(x, 0.005, 'TBD', ha='center', fontsize=9, color='#999999')

fig.tight_layout()
out = '/home/ryan/Documents/robot/ORA0/paper/fig7_2x2.png'
fig.savefig(out, dpi=130)
print('saved', out)
