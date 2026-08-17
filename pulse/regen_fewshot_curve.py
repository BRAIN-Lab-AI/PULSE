"""
Regenerate CAMUS few-shot performance curve figure.
Run after eval_camus_fewshot.py completes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Confirmed results (5 seeds averaged, N=5/10/20) from eval_camus_fewshot.py
confirmed = {
    5:  {"Myo": 0.652, "LV": 0.721, "Mean": 0.687},
    10: {"Myo": 0.678, "LV": 0.733, "Mean": 0.706},
    20: {"Myo": 0.708, "LV": 0.756, "Mean": 0.732},
}

N_confirmed = sorted(confirmed.keys())
myo_vals  = [confirmed[n]["Myo"]  for n in N_confirmed]
lv_vals   = [confirmed[n]["LV"]   for n in N_confirmed]
mean_vals = [confirmed[n]["Mean"] for n in N_confirmed]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(N_confirmed, mean_vals, "o-", color="#0d2b5e", lw=2, ms=7, label="Mean Dice (2-class)")
ax.plot(N_confirmed, lv_vals,   "s--", color="#2563b8", lw=1.5, ms=6, label="LV Dice")
ax.plot(N_confirmed, myo_vals,  "^--", color="#6495d8", lw=1.5, ms=6, label="Myo Dice")

# Annotations for mean
for n, val in zip(N_confirmed, mean_vals):
    ax.annotate(f"{val:.3f}", (n, val), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8.5, color="#0d2b5e")

ax.set_xlabel("Number of labelled CAMUS subjects ($N$)", fontsize=10)
ax.set_ylabel("Dice Score", fontsize=10)
ax.set_title("CAMUS Few-Shot Transfer (2CH, 5-seed average)", fontsize=10)
ax.set_xlim(0, 25)
ax.set_ylim(0.55, 0.85)
ax.set_xticks(N_confirmed)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "fewshotzonecurve.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
