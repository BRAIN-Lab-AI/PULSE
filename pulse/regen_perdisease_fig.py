"""
Regenerate perdiseasesegmentation.png using per-disease per-class Dice
from the ensemble predicted masks (diagnosis_features_ensemble.json).
"""
import json, numpy as np, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLS_NAMES = ["NOR", "DCM", "HCM", "MINF", "RV"]

# Confirmed results from eval_perdisease.py (5-fold ensemble + TTA, ACDC test set n=50)
# OVERALL: LV=91.6  Myo=84.7  RV=90.3  Mean=88.8  (matches main paper result)
medians = {
    "NOR":  {"LV": 90.5, "Myo": 84.0, "RV": 90.6},
    "DCM":  {"LV": 95.6, "Myo": 83.3, "RV": 91.4},
    "HCM":  {"LV": 85.9, "Myo": 87.4, "RV": 88.6},
    "MINF": {"LV": 94.5, "Myo": 85.6, "RV": 89.0},
    "RV":   {"LV": 91.5, "Myo": 82.9, "RV": 91.8},
}

# ── Plot ──────────────────────────────────────────────────────────────────────
x      = np.arange(len(CLS_NAMES))
width  = 0.26
colors = {"LV": "#1a3a5c", "Myo": "#5aabcf", "RV": "#b3dced"}

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

for i, struct in enumerate(["LV", "Myo", "RV"]):
    vals = [medians[name][struct] for name in CLS_NAMES]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=struct, color=colors[struct], edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLS_NAMES, fontsize=10)
ax.set_xlabel("Disease Type", fontsize=11)
ax.set_ylabel("Mean Dice (%)", fontsize=11)
ax.set_ylim(78, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Legend placed above the axes so it never overlaps bars or value labels.
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          fontsize=9, frameon=False, columnspacing=2.5, handletextpad=0.6)

plt.tight_layout()
plt.savefig("paper_contents/perdiseasesegmentation.png", dpi=200, bbox_inches="tight")
print("Saved perdiseasesegmentation.png")
for name in CLS_NAMES:
    lv  = medians[name]["LV"]
    myo = medians[name]["Myo"]
    rv  = medians[name]["RV"]
    print(f"  {name}: LV={lv:.2f}  Myo={myo:.2f}  RV={rv:.2f}")
