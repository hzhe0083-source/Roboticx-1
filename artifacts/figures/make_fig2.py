import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
fm.fontManager.addfont('/home/ryan/.local/share/fonts/HarmonyOS-Sans/HarmonyOS_Sans_SC_Light.ttf')
plt.rcParams['font.family'] = 'HarmonyOS Sans SC'
plt.rcParams['axes.unicode_minus'] = False

# LIBERO 实测
# ⚠️ 数据源状态（2026-08-05）：LIBERO 注意力占比（v/m/a/l 数组）来自旧会话测量，
# 报告未记录、无生成脚本 → 不可复现。论文正式版需在 GPU 空闲时用 libero checkpoint 重测，
# 或移除本面板。PNPW 面板（右侧）与报告 §3.3 一致，可复现。
labels = ['QV t0', 'QA t0', 'QV t>0', 'QA t>0']
v = [0.758, 0.780, 0.530, 0.590]
m = [0.0, 0.0, 0.345, 0.238]
a = [0.167, 0.156, 0.069, 0.117]
l = [0.076, 0.064, 0.056, 0.055]
x = np.arange(4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(x - 0.3, v, 0.2, label='V 视觉', color='#4C72B0')
ax1.bar(x - 0.1, m, 0.2, label='M 记忆', color='#55A868')
ax1.bar(x + 0.1, a, 0.2, label='A 动作', color='#F0A500')
ax1.bar(x + 0.3, l, 0.2, label='L 语言', color='#C44E52')
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel('注意力占比')
ax1.set_title('LIBERO 1场景（语言必要）— L=5.5~7.6%', fontsize=11)
ax1.legend(fontsize=8)
for i, lv in enumerate(l):
    ax1.text(i + 0.3, lv + 0.01, f'{lv:.3f}', ha='center', fontsize=8, color='#C44E52')

# PNPW 实测（codex 审计）
labels2 = ['QV t0', 'QA t0', 'QV t>0', 'QA t>0']
v2 = [0.784, 0.816, 0.515, 0.658]
m2 = [0.0, 0.0, 0.304, 0.176]
a2 = [0.213, 0.184, 0.179, 0.166]
l2 = [0.003, 0.000, 0.002, 0.000]
ax2.bar(x - 0.3, v2, 0.2, label='V 视觉', color='#4C72B0')
ax2.bar(x - 0.1, m2, 0.2, label='M 记忆', color='#55A868')
ax2.bar(x + 0.1, a2, 0.2, label='A 动作', color='#F0A500')
ax2.bar(x + 0.3, l2, 0.2, label='L 语言', color='#C44E52')
ax2.set_xticks(x); ax2.set_xticklabels(labels2)
ax2.set_ylabel('注意力占比')
ax2.set_title('PNPW 单任务（语言常数）— L≈0（模型确实不用语言）', fontsize=11)
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig('/home/ryan/Documents/robot/ORA0/attention_compare.png', dpi=130)
print('saved attention_compare.png')
