"""Fig 11: LIBERO-100 full-scale (100 tasks) — open loop + language trio.

Data: prepare_libero.py --scene ALL output + evaluate.py --perturb on
checkpoints/libero_100_va8_10k.pt.
Run: python artifacts/figures/make_fig_libero100.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- EDIT HERE with final numbers ----
N_TASKS = 100
CHUNK_MAE = None       # TODO: evaluate.py output
PERSISTENCE = None     # TODO
BLANK_PCT = None       # TODO
SWAP_PCT = None        # TODO
LIT = {"TurboVLA": 97.7, "Evo-1": 94.8, "SmolVLA": 89.0}  # LIBERO avg anchors (50/10 trials)
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(f'LIBERO-100 ({N_TASKS} tasks) — full-scale frozen-cache policy', fontsize=13, fontweight='bold')

# (a) open loop vs persistence + trio inset
ax = axes[0]
if CHUNK_MAE is not None:
    ax.bar(['model', 'persistence'], [CHUNK_MAE, PERSISTENCE], color=['#55A868', '#C44E52'], width=0.5)
    ax.set_ylabel('chunk MAE (norm)')
    ax.set_title(f'(a) Open loop (clean {CHUNK_MAE:.4f}, blank +{BLANK_PCT:.0f}%, swap +{SWAP_PCT:.0f}%)', fontsize=10)
else:
    ax.text(0.5, 0.5, 'TBD: queue step 7', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999999')
    ax.set_title('(a) Open loop', fontsize=10)

# (b) LIBERO avg vs literature anchors (protocol-aware)
ax = axes[1]
keys = list(LIT) + ['ours']
vals = list(LIT.values()) + [CHUNK_MAE]  # ours open-loop; not closed-loop comparable
bars = ax.bar(range(len(keys)), vals if all(v is not None for v in vals) else [0]*len(keys),
              color=['#4C72B0']*len(LIT) + ['#55A868'], width=0.5)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=15, fontsize=9)
ax.set_ylabel('LIBERO avg success % (literature: closed-loop)')
ax.set_title('(b) Protocol-aware comparison', fontsize=10)
ax.text(0.5, -0.22, 'lit. numbers are closed-loop (50/10 trials); ours is open-loop chunk MAE — not directly comparable',
        transform=ax.transAxes, ha='center', fontsize=8, color='#888888')

fig.tight_layout()
out = '/home/ryan/Documents/robot/ORA0/paper/fig11_libero100.png'
fig.savefig(out, dpi=130)
print('saved', out)
