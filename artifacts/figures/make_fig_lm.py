"""Fig 8: L_m same-scene dual-objective verdict (A vs B40k).

Data: eval_libero_Lm.py output — per-pair D, O, L_m with block-bootstrap CI.
Run: python artifacts/figures/make_fig_lm.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- EDIT HERE with final numbers (pair: (D, O, Lm_lo, Lm_hi)) ----
PAIRS = ["study\nback/front", "study\nleft/right", "kitchen\nback/front",
         "living\nsoup/butter", "living\nmilk/juice"]
A = {
    "D": [0.0]*5, "O": [0.0]*5, "Lm": [0.0]*5, "lo": [0.0]*5, "hi": [0.0]*5,
}
B40K = {
    "D": [0.0]*5, "O": [0.0]*5, "Lm": [0.0]*5, "lo": [0.0]*5, "hi": [0.0]*5,
}
# Filled 2026-08-07 from logs/Lm_libero_3scene_va8_20k.log + Lm_libero_e2e_B40k.log:
# all 5 pairs D=O=0.000, L_m=0.000 CI [0.000, 0.000] -> OOD-fragility verdict;
# obedience is resolved by the open-loop C_OL protocol (Sec. 5.6).
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle('L_m Same-Scene Dual-Objective Verdict (matched blocks, 95% CI)', fontsize=13, fontweight='bold')

for ax, (tag, data) in zip(axes, [('config A (frozen)', A), ('B40k (e2e LoRA)', B40K)]):
    x = np.arange(len(PAIRS))
    D = np.array([v if v is not None else np.nan for v in data["D"]])
    O = np.array([v if v is not None else np.nan for v in data["O"]])
    Lm = np.array([v if v is not None else np.nan for v in data["Lm"]])
    lo = np.array([v if v is not None else np.nan for v in data["lo"]])
    hi = np.array([v if v is not None else np.nan for v in data["hi"]])
    w = 0.28
    ax.bar(x-w, D, w, label='D (matched)', color='#55A868')
    ax.bar(x, O, w, label='O (swapped)', color='#C44E52')
    ax.bar(x+w, Lm, w, label='L_m = D-O', color='#4C72B0')
    if not np.isnan(lo).all():
        ax.errorbar(x+w, Lm, yerr=[Lm-lo, hi-Lm], fmt='none', ecolor='black', capsize=3)
    ax.axhline(0, color='#888888', ls='--', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(PAIRS, fontsize=8)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel('success rate / L_m')
    ax.set_title(tag, fontsize=11)
    ax.legend(fontsize=8)
    if (D == 0).all() and (O == 0).all():
        ax.text(0.5, 0.8, 'all 5 pairs D=O=0 → OOD-fragility verdict\n'
                          '(obedience resolved via C_OL, Sec. 5.6)',
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#999999')
    for xi, v in zip(x, Lm):
        if not np.isnan(v):
            ax.text(xi+w, v+0.03, f'{v:.2f}', ha='center', fontsize=8)

fig.tight_layout()
out = '/home/ryan/Documents/robot/ORA0/paper/fig8_lm.png'
fig.savefig(out, dpi=130)
print('saved', out)
