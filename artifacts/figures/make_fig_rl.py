"""Fig 10: VLA-RL — IL->RL success-rate curve on MetaWorld.

Data: train_ppo_metaworld.py log (per-iter ep_success / reward_sum) +
held-out closed-loop eval before/after (eval_metaworld.py protocol).
Run: python artifacts/figures/make_fig_rl.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- EDIT HERE: parse from /tmp/mw_ppo_smoke.log or main run log ----
STEPS = []        # PPO iteration
SUCCESS = []      # rolling success rate over envs
IL_BEFORE = None  # held-out closed-loop success before RL (e.g. 0.35)
RL_AFTER = None   # held-out closed-loop success after RL
# MT50 RL 文献锚点（Grok 2026-08-06 查证 πRL Table 6；SmolVLA 难度分桶 Avg；
# trials/task 原文未钉死→脚注；ReinFlow 无 MT50 数字，勿引）
LIT = {"π0 Flow-Noise (πRL)": 85.8, "π0 Flow-SDE (πRL)": 78.1, "π0.5 Flow-SDE": 70.7, "SmolVLA": 68.2}
LIT_FOOTNOTE = ("πRL MT50 difficulty-binned Avg; trials/task not pinned in paper — "
                "compare as magnitude reference only")
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.5))
if STEPS:
    ax.plot(STEPS, SUCCESS, '-o', color='#4C72B0', ms=3, label='PPO training success (rollout)')
    ax.set_xlabel('PPO iteration')
    ax.set_ylabel('rollout success rate')
else:
    ax.text(0.5, 0.5, 'TBD: parse train_ppo_metaworld.py log', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='#999999')

if IL_BEFORE is not None and RL_AFTER is not None:
    ax.plot([0, len(STEPS)-1], [IL_BEFORE*100, RL_AFTER*100], '--', color='#C44E52',
            label=f'held-out closed-loop: {IL_BEFORE:.0%} -> {RL_AFTER:.0%}')
ax.set_title('VLA-RL: PPO fine-tuning of the VA action head (sparse success)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
fig.tight_layout()
out = '/home/ryan/Documents/robot/ORA0/paper/fig10_rl.png'
fig.savefig(out, dpi=130)
print('saved', out)
