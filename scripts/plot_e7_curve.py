"""解析 logs/e7_train.log 并绘制 e7_mtvj 训练曲线。"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "logs/e7_train.log"
OUT = "artifacts/figures/e7_train_curve.png"

steps, losses, flows, grads = [], [], [], []
pat = re.compile(
    r"step=(\d+).*?loss=([\d.]+) flow=([\d.]+) .*?grad=([\d.]+)"
)
with open(LOG) as f:
    for line in f:
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            flows.append(float(m.group(3)))
            grads.append(float(m.group(4)))

steps = np.array(steps)
losses = np.array(losses)
flows = np.array(flows)
grads = np.array(grads)

# 移动平均
def movavg(x, w=200):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")

w = 200
ls_ma = movavg(losses, w)
gr_ma = movavg(grads, w)
step_ma = steps[: len(ls_ma)]

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                         gridspec_kw={"hspace": 0.12})

# 1. loss 全览（含早期尖峰，log 尺度）
ax = axes[0]
ax.semilogy(steps, losses, lw=0.4, alpha=0.35, color="#888", label="raw")
ax.semilogy(step_ma, ls_ma, lw=1.6, color="#c0392b", label=f"moving avg (w={w})")
ax.set_ylabel("flow loss (log)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_title("e7_mtvj training curves  (step %d / 80000, 46%%)" % steps[-1])

# 2. loss 平台期放大（线性，裁剪早期尖峰）
ax = axes[1]
ax.plot(steps, losses, lw=0.4, alpha=0.35, color="#888")
ax.plot(step_ma, ls_ma, lw=1.6, color="#c0392b")
ax.set_ylim(0, 0.5)
ax.set_ylabel("flow loss")
ax.grid(alpha=0.3)

# 3. grad norm
ax = axes[2]
ax.plot(steps, grads, lw=0.4, alpha=0.35, color="#888")
ax.plot(step_ma, gr_ma, lw=1.6, color="#2980b9")
ax.set_ylim(0, 2.5)
ax.set_ylabel("grad norm")
ax.set_xlabel("step")
ax.grid(alpha=0.3)

# 标注当前值
ax = axes[0]
ax.axvline(steps[-1], color="#27ae60", ls="--", lw=1, alpha=0.8)
ax.text(steps[-1], 0.05, f" now\n loss={losses[-1]:.3f}", fontsize=9,
        color="#27ae60", va="top")

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved -> {OUT}")
print(f"steps={len(steps)}  last_loss={losses[-1]:.4f}  "
      f"last_1000_mean={losses[-1000:].mean():.4f}  "
      f"grad_mean={grads[-1000:].mean():.3f}")
