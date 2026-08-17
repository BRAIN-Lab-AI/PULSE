"""Regenerate the progressive ablation improvement figure with the new final step."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

steps  = ["Baseline", "+ Augmentation", "+ Normalization",
          "+ Opt. Weight", "+ Lovász Loss",
          "+ Deep Sup\n+ TTA"]
values = [65.0, 76.0, 76.6, 80.3, 81.9, 88.6]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.set_facecolor("#f9f9f9")
fig.patch.set_facecolor("white")

x = np.arange(len(steps))
color_normal = "#4CAF50"
color_new    = "#1565C0"   # blue for the new step

colors = [color_normal] * (len(steps) - 1) + [color_new]

# filled area under curve
ax.fill_between(x, values, 60, alpha=0.18, color=color_normal, zorder=1)
ax.fill_between(x[-2:], values[-2:], 60, alpha=0.18, color=color_new, zorder=1)

# line
ax.plot(x[:-1], values[:-1], color=color_normal, linewidth=2.2, zorder=3)
ax.plot(x[-2:], values[-2:], color=color_new, linewidth=2.2, linestyle="--", zorder=3)

# dots
for i, (xi, yi) in enumerate(zip(x, values)):
    c = color_new if i == len(steps) - 1 else color_normal
    ax.scatter(xi, yi, color=c, s=62, zorder=5, edgecolors="white", linewidths=1.5)
    offset = 0.4 if i != len(steps) - 1 else 0.5
    ax.text(xi, yi + offset, f"{yi:.1f}%", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#1a1a1a" if i < len(steps) - 1 else color_new)

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=8.2)
ax.set_ylabel("Mean Dice (%)", fontsize=10)
ax.set_xlabel("Ablation Step", fontsize=10)
ax.set_ylim(60, 91)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend for new step, placed above the axes so it never overlaps the curve or labels
patch = mpatches.Patch(color=color_new, label="Full PULSE (fold 0 + TTA, this work)")
ax.legend(handles=[patch], fontsize=8.5, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout()
plt.savefig("paper_contents/FIGUREPROGRESSIVEIMPROVEMENT.png", dpi=200, bbox_inches="tight")
print("Saved FIGUREPROGRESSIVEIMPROVEMENT.png")
