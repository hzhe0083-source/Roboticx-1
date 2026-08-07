import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
fm.fontManager.addfont('/home/ryan/.local/share/fonts/HarmonyOS-Sans/HarmonyOS_Sans_SC_Light.ttf')
plt.rcParams['font.family'] = 'HarmonyOS Sans SC'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(21, 12))
fig.suptitle('VA Compound 实测数据图表包（全部来自实际运行）', fontsize=15, fontweight='bold')

# 图1: 深度探针（PNPW 各配置 chunk vs 基线）
ax = axes[0, 0]
configs = ['4层@10k', '8层@10k', '8层@20k', '持久性基线']
chunks = [0.0497, 0.0518, 0.0345, 0.0552]
colors = ['#4C72B0', '#4C72B0', '#55A868', '#C44E52']
bars = ax.bar(configs, chunks, color=colors, width=0.6)
ax.set_ylabel('chunk_mae_norm')
ax.set_title('图1: PNPW 深度探针（越低越好）', fontsize=11)
for b, v in zip(bars, chunks):
    ax.text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
ax.axhline(0.0552, color='#C44E52', ls='--', lw=1)
ax.text(2.4, 0.0558, '基线 0.0552', color='#C44E52', fontsize=8)

# 图2: 四流注意力分配
ax = axes[0, 1]
labels = ['QV t0', 'QA t0', 'QV t>0', 'QA t>0']
v = [0.784, 0.816, 0.515, 0.658]
m = [0, 0, 0.304, 0.176]
a = [0.213, 0.184, 0.179, 0.166]
l = [0.003, 0.000, 0.002, 0.000]
x = np.arange(4)
ax.bar(x - 0.3, v, 0.2, label='V 视觉', color='#4C72B0')
ax.bar(x - 0.1, m, 0.2, label='M 记忆', color='#55A868')
ax.bar(x + 0.1, a, 0.2, label='A 动作', color='#F0A500')
ax.bar(x + 0.3, l, 0.2, label='L 语言', color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('注意力占比')
ax.set_title('图2: 四流注意力分配（PNPW 实测，报告 §3.3）', fontsize=11)
ax.legend(fontsize=8)

# 图3: 输入消融
ax = axes[0, 2]
abl = ['prev清零', 'proprio清零', '视觉清零', '阻断A→V', '阻断M→A', '阻断语言']
delta = [3792, 291, 155, 7.7, 2.9, 0.0]
colors3 = ['#C44E52' if d > 100 else '#F0A500' if d > 10 else '#4C72B0' for d in delta]
bars = ax.barh(abl, delta, color=colors3)
ax.set_xscale('log')
ax.set_xlabel('误差变化 %（对数轴）')
ax.set_title('图3: 输入消融（误差上升=该输入重要）', fontsize=11)
for b, d in zip(bars, delta):
    ax.text(b.get_width() * 1.05, b.get_y() + b.get_height()/2, f'+{d:.1f}%', va='center', fontsize=9)

# 图4: 语言流三件套（§7.2 新口径：--perturb、flow_steps=32，2026-08-05 重测）
ax = axes[1, 0]
x = np.arange(2)
blank = [13751, 2381]
swap = [1518, 607]
ax.bar(x - 0.15, blank, 0.3, label='Blank 空白指令', color='#C44E52')
ax.bar(x + 0.15, swap, 0.3, label='Swap 换指令', color='#F0A500')
ax.set_xticks(x); ax.set_xticklabels(['LIBERO 1场景×4任务', 'LIBERO 3场景×12任务'])
ax.set_ylabel('误差上升 %')
ax.set_title('图4: 语言流三件套（32步部署口径）', fontsize=11)
for i, (b, s) in enumerate(zip(blank, swap)):
    ax.text(i - 0.15, b + 200, f'+{b}%', ha='center', fontsize=9)
    ax.text(i + 0.15, s + 200, f'+{s}%', ha='center', fontsize=9)
ax.legend(fontsize=8)

# 图5: LIBERO 拟合度（B40k 终评后刷新；B40K_CHUNK 为 None 时画"待终评"占位柱）
ax = axes[1, 1]
configs5 = ['持久性基线', 'B 10k 中间', '冻结 8层@20k', 'B 40k 端到端']
chunks5 = [0.14856, 0.123, 0.0368, 0.07591]  # 持久性基线/冻结8层@20k 为阶段A实测（32步口径）；末位 B40k 终评（/tmp/pipeline_final_eval.log 2026-08-05）
colors5 = ['#C44E52', '#F0A500', '#55A868', '#2E8B57']
xs5 = np.arange(4)
bars = ax.bar(xs5, [v if v is not None else 0 for v in chunks5], color=colors5, width=0.55)
ax.set_xticks(xs5); ax.set_xticklabels(configs5)
ax.set_ylabel('chunk_mae_norm')
ax.set_title('图5: LIBERO 拟合度（B40k 终评后刷新）', fontsize=11)
for i, v in enumerate(chunks5):
    if v is None:
        ax.text(i, 0.006, '待终评', ha='center', fontsize=9, color='#888888')
    else:
        ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9)

# 图6: 各数据语言屏蔽对比（EvoStudio/PNPW = 推理 mask 阻断口径；LIBERO = §7.2 --perturb 32 步口径）
ax = axes[1, 2]
datasets = ['PNPW\n(常数指令)', 'EvoStudio\n(场景可分)', 'LIBERO 3场景\n(同场景多指令)']
deltas = [-0.1, -0.1, 2381]
bars = ax.bar(datasets, deltas, color=['#4C72B0', '#4C72B0', '#C44E52'], width=0.5)
ax.set_yscale('symlog', linthresh=1)
ax.set_ylabel('屏蔽语言误差变化 %（symlog）')
ax.set_title('图6: 语言屏蔽三数据对比（关键反转）', fontsize=11)
for b, d in zip(bars, deltas):
    ax.text(b.get_x() + b.get_width()/2, d + (50 if d > 0 else -20), f'{d:.1f}%', ha='center', fontsize=9)

fig.subplots_adjust(left=0.055, right=0.975, top=0.93, bottom=0.065, wspace=0.32, hspace=0.38)
plt.savefig('/home/ryan/Documents/robot/ORA0/benchmark_figures.png', dpi=130)
print('saved benchmark_figures.png')
