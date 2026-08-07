"""Fig 9: MetaWorld MT50 — open-loop, language ablation, closed-loop.

Data: §8.2 (open loop chunk MAE vs persistence), §8.3 (wrong/taskid ablation),
§8.5 (closed loop; multi-start retest numbers to replace the coverage-limited row).
Run: python artifacts/figures/make_fig_mw.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- EDIT HERE with final numbers ----
OPEN_LOOP = {"model": 0.0803, "persistence": 0.15935}  # chunk MAE, VA2 v4 开环（logs/mw_va2_v4_openloop.log）
OPEN_LOOP_CI = (0.0710, 0.0915)
ABLATION = {"clean": 0.07984, "wrong": 0.20018, "task-id": 0.17496}  # chunk@all-seq, VA2 v4 消融（logs/mw_va2_v4_ablation.log）
CLOSED_LOOP = {
    "old (0.33s coverage)": 7.1,  # 49x10, [2.7, 12.7]
    "multi-start rebuild": 16.3,  # logs/mw_full_closedloop.log, [9.4, 24.1]
    "AQC (lang queries)": 17.8,  # logs/mw_aqc_closedloop.log, [11.0, 25.3]
    "VA2 (this work)": None,  # TODO: logs/mw_va2_closedloop.log
}
CLOSED_LOOP_CI = {
    "old (0.33s coverage)": (2.7, 12.7),
    "multi-start rebuild": (9.4, 24.1),
    "AQC (lang queries)": (11.0, 25.3),
    "VA2 (this work)": None,
}
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
ax.text(0.5, 0.155, 'VA2 v4 待回填', ha='center', fontsize=9, color='#2E8B57')

# (b) ablation
ax = axes[1]
keys = list(ABLATION)
vals = list(ABLATION.values())
bars = ax.bar(keys, vals, color=['#55A868', '#C44E52', '#F0A500'], width=0.5)
ax.set_ylabel('chunk MAE (norm)')
ax.set_title('(b) Language ablation', fontsize=11)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.4f}', ha='center', fontsize=9)
ax.text(0.5, 0.17, 'VA2 v4 待回填', ha='center', fontsize=9, color='#C44E52')

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
