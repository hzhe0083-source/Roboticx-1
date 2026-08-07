#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig7: 双 benchmark 主表（LIBERO + MetaWorld 正式结果）

所有数字来自实际运行日志/报告（来源标注于行注释），VA2 数字产出后更新本文件重跑。
输出: artifacts/figures/fig7_benchmark_main_table.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/ryan/.local/share/fonts/HarmonyOS-Sans/HarmonyOS_Sans_SC_Light.ttf')
plt.rcParams['font.family'] = 'HarmonyOS Sans SC'
plt.rcParams['axes.unicode_minus'] = False

# ---- 数据（来源：paper/ora0_paper.md §5.2/§5.3/§5.6 + 对应日志；VA2 落地后替换）----
LIB = {
    'policy': 'VA 复合体（决策层 43.5M）',
    'action': '7D',
    'open_chunk': '0.00254',        # 3 场景 12 任务 clean，32 步口径（VA_COMPOUND_REPORT §7.2 重测记录）
    'baseline': '—',                # LIBERO 持久性基线未在论文报告
    'vs_base': '—',
    'open_success': '—',
    'blank': '+2381%',              # §7.2 语言流三件套（flow_steps=32，3 场景 12 任务）
    'swap': '+607%',                # 同上
    'taskid': '—',
    'col': '0.160',                 # C_OL 反事实位移比（logs/col_libero_3scene_va8_20k.log）
    'closed': 'D=O=0（OOD 脆弱，§5.6）',
}
MW = {
    'policy': 'VA 复合体（决策层 43.5M）',
    'action': '4D',
    'open_chunk': '0.0806 [0.0709, 0.0921]',  # logs/mw_full_openloop.log（多 start 重建）
    'baseline': '0.0930',                     # 同上
    'vs_base': '-13.4%',                      # 同上
    'open_success': '50.8% [45.3, 56.1]',     # 同上（持久性 88.5%）
    'blank': '+108.5%',                       # wrong 指令，chunk@all-seq（logs/mw_full_ablation.log）
    'swap': '+607%',                          # 占位不用——MW 用 task-id 消融
    'taskid': '+81.5%',                       # task-id token（logs/mw_full_ablation.log）
    'col': '—',
    'closed': '17.8% [11.0, 25.3]（链 7.1→16.3→17.8，49×10）',  # logs/mw_aqc_closedloop.log
}
LIT_MW = 'Evo-1 80.6%† / SmolVLA ~68%† / FabriVLA 90.0% / LA4VLA 87.5% / π0+ALAM 85.0% / Evo-Depth 84.4%'
LIT_LIB = 'TurboVLA 97.7%‡ / π0.5 ~97%‡ / Evo-1 94.8%† / SmolVLA ~89%† / π0 94.2%‡'

rows = [
    ('模型',            LIB['policy'],                     MW['policy']),
    ('动作维度',         LIB['action'],                     MW['action']),
    ('开环 chunk_mae_norm(2)', LIB['open_chunk'],          MW['open_chunk']),
    ('持久性基线 chunk_mae', LIB['baseline'],              MW['baseline']),
    ('vs 基线',         LIB['vs_base'],                     MW['vs_base']),
    ('开环首步 success', LIB['open_success'],               MW['open_success']),
    ('语言 Blank（误差变化）', LIB['blank'],               MW['blank']),
    ('语言 Swap / Task-id 消融',  LIB['swap'],             MW['taskid']),
    ('C_OL 反事实位移比', LIB['col'],                      MW['col']),
    ('闭环 success（宏平均）', LIB['closed'],              MW['closed']),
]
col_labels = ['指标', 'LIBERO（3 场景 12 任务）', 'MetaWorld MT50（49 任务）']
cell_text = [[r[0], r[1], r[2]] for r in rows]
cell_text += [['文献闭环 SOTA', LIT_LIB, LIT_MW]]

fig, ax = plt.subplots(figsize=(14, 6.8))
ax.axis('off')
table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.75)

# 样式：表头加深，本工作行浅绿，文献行浅蓝
header_idx = 0
for (r, c), cell in table.get_celld().items():
    if r == header_idx:
        cell.set_facecolor('#2F4858')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 0:
        cell.set_facecolor('#EDF1F5')
        cell.set_text_props(fontweight='bold')
    else:
        if r <= len(rows):
            cell.set_facecolor('#E8F4E4')
        else:
            cell.set_facecolor('#E3EDF7')

fig.suptitle('图7: 双 Benchmark 主表（VA 复合体，全部来自实际运行）', fontsize=14, fontweight='bold', y=0.97)
fig.text(0.06, 0.035,
         '(1) LIBERO 闭环二进制成功不可分辨（闭环 D=O=0，OOD 脆弱，§5.6）——语言服从性以开环三件套 + C_OL 为主证据。\n'
         '(2) 归一化动作全块误差，flow_steps=32 部署口径；口径详见 §4（宏平均、95% CI、固定种子）。\n'
         '(3) †10 trials/task（MT50）/LIBERO 10 trials；‡LIBERO 50 trials；文献数字均引自原文，未自行训练（§5.4）。\n'
         '(4) MW Blank 列实为 wrong-指令消融（+108.5%）；VA2 数字产出后替换本表。',
         fontsize=8, color='#444444')
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
out = '/home/ryan/Documents/robot/ORA0/artifacts/figures/fig7_benchmark_main_table.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print('saved', out)
