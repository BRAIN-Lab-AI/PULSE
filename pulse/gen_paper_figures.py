#!/usr/bin/env python3
"""
Generate all supplementary figures from REAL experimental data only.
No fabricated intermediate steps.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "paper_contents"
DARK = "#1e293b"

# ── 1. ABLATION OVERVIEW BAR CHART ──────────────────────────────────────────
def fig_ablation_bars():
    configs = [
        ("No Augmentation",       75.8, "#ef4444"),
        ("Weak Augmentation",     88.4, "#f97316"),
        ("w/o Curriculum",        88.5, "#eab308"),
        ("w/o Boundary Loss",     88.4, "#eab308"),
        ("w/o Deep Supervision",  88.6, "#84cc16"),
        ("DPT Base (no novel)",   88.8, "#22c55e"),
        ("Full PULSE (fold 0)",   88.6, "#3b82f6"),
        ("5-Fold Ensemble + TTA", 88.8, "#1d4ed8"),
    ]

    names  = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8,
                  width=0.72, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=25, ha="right")
    ax.set_ylabel("Mean Dice (%)", fontsize=11)
    ax.set_ylim(65, 92)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(88.6, color="#3b82f6", linestyle=":", lw=1.2, alpha=0.6)
    ax.text(7.55, 88.75, "Full PULSE", color="#3b82f6", fontsize=7.5)

    plt.tight_layout()
    out = OUT / "ablation_overview.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 2. ENSEMBLE + TTA SCALING LINE CHART ────────────────────────────────────
def fig_ensemble_scaling():
    configs = [
        ("1-fold\nno TTA", 87.9),
        ("1-fold\n+TTA",   88.6),
        ("2-fold\n+TTA",   88.7),
        ("3-fold\n+TTA",   88.7),
        ("4-fold\n+TTA",   88.8),
        ("5-fold\n+TTA",   88.8),
    ]
    labels = [c[0] for c in configs]
    vals   = [c[1] for c in configs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))

    ax.plot(x, vals, "o-", color="#1d4ed8", lw=2.2, ms=8,
            markerfacecolor="white", markeredgewidth=2.2, zorder=4)
    for xi, yi in zip(x, vals):
        ax.annotate(f"{yi:.1f}%", (xi, yi),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=DARK, fontweight="bold")

    # TTA divider
    ax.axvline(0.5, color="#94a3b8", linestyle="--", lw=1.2, alpha=0.7)
    ax.text(0.52, 87.92, "TTA on →", fontsize=8, color="#64748b")

    ax.fill_between(x, vals, min(vals)-0.3, alpha=0.08, color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Dice (%)", fontsize=11)
    ax.set_ylim(87.4, 89.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT / "ensemble_scaling_curve.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 3. PER-FOLD STABILITY ───────────────────────────────────────────────────
def fig_fold_stability():
    folds = {
        "Fold 0": (89.9, 84.5, 91.4),
        "Fold 1": (89.9, 84.1, 91.0),
        "Fold 2": (89.8, 84.3, 91.4),
        "Fold 3": (90.1, 84.6, 91.2),
        "Fold 4": (90.4, 84.2, 91.3),
        "Ensemble": (90.3, 84.7, 91.6),
    }
    structs = ("RV", "Myo", "LV")
    colors  = {"RV": "#38bdf8", "Myo": "#4ade80", "LV": "#f87171"}
    names   = list(folds.keys())
    x       = np.arange(len(names))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, st in enumerate(structs):
        vals = [folds[n][i] for n in names]
        off  = (i - 1) * width
        bars = ax.bar(x + off, vals, width, label=st,
                      color=colors[st], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=DARK)

    # Shade ensemble column
    ax.axvspan(4.62, 5.38, color="#e0e7ff", alpha=0.5, zorder=0)
    ax.text(5.0, 92.3, "Ensemble", ha="center", fontsize=8,
            color="#3730a3", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Dice (%)", fontsize=11)
    ax.set_ylim(82, 93.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    plt.tight_layout()
    out = OUT / "fold_stability.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 4. CROSS-DOMAIN GENERALISATION ──────────────────────────────────────────
def fig_generalisation():
    datasets = {
        "ACDC\n(in-domain,\n50 subjects)": {
            "RV": 90.3, "Myo": 84.7, "LV": 91.6,
        },
        "M\\&Ms\n(cross-domain,\n360 subjects)": {
            "RV": 87.9, "Myo": 80.4, "LV": 87.5,
        },
        "Sunnybrook\n(zero-shot,\nLV only)": {
            "RV": None, "Myo": None, "LV": 88.1,
        },
    }
    structs = ("RV", "Myo", "LV")
    colors  = {"RV": "#38bdf8", "Myo": "#4ade80", "LV": "#f87171"}
    names   = list(datasets.keys())
    x       = np.arange(len(names))
    width   = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, st in enumerate(structs):
        vals = [datasets[n].get(st, None) for n in names]
        off  = (i - 1) * width
        for xi, v in enumerate(vals):
            if v is None: continue
            bar = ax.bar(xi + off, v, width, color=colors[st], edgecolor="white",
                         linewidth=0.5, label=st if xi == 0 else "")
            ax.text(xi + off + width/2, v + 0.2, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=8, color=DARK, fontweight="bold")

    # Clean up legend (deduplicate)
    handles = [mpatches.Patch(color=colors[s], label=s) for s in structs]
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Dice (%)", fontsize=11)
    ax.set_ylim(70, 95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT / "generalisation_bars.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 5. AUGMENTATION ABLATION (per-class) ────────────────────────────────────
def fig_aug_ablation():
    configs = {
        "No Aug": (56.0, 81.3, 90.0),
        "Weak Aug": (89.1, 84.8, 91.3),
        "Strong Aug\n(Full PULSE)": (90.0, 84.5, 91.4),
    }
    structs = ("RV", "Myo", "LV")
    colors  = {"RV": "#38bdf8", "Myo": "#4ade80", "LV": "#f87171"}
    names   = list(configs.keys())
    x       = np.arange(len(names))
    width   = 0.26

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, st in enumerate(structs):
        vals = [configs[n][i] for n in names]
        off  = (i - 1) * width
        bars = ax.bar(x + off, vals, width, color=colors[st], edgecolor="white",
                      linewidth=0.5, label=st)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Dice (%)", fontsize=11)
    ax.set_ylim(45, 95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    handles = [mpatches.Patch(color=colors[s], label=s) for s in structs]
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.85)

    plt.tight_layout()
    out = OUT / "aug_ablation_perclass.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 6. COMPARISON vs BASELINES ──────────────────────────────────────────────
def fig_comparison():
    methods = {
        "U-Net\n(2018)":         (82.3, 78.7, 94.9, 85.3),
        "TransUNet\n(2021)":     (84.5, 77.7, 94.1, 85.4),
        "SwinUNet\n(2021)":      (90.7, 79.1, 93.9, 87.9),
        "nnU-Net\n(2021)†":      (90.1, 88.4, 95.7, 91.4),
        "PULSE (ours)\n5-fold":  (90.3, 84.7, 91.6, 88.8),
    }
    structs = ("RV", "Myo", "LV", "Mean")
    colors  = {"RV": "#38bdf8", "Myo": "#4ade80", "LV": "#f87171", "Mean": "#a78bfa"}
    names   = list(methods.keys())
    x       = np.arange(len(names))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, st in enumerate(structs):
        vals = [methods[n][i] for n in names]
        off  = (i - 1.5) * width
        bars = ax.bar(x + off, vals, width, color=colors[st], edgecolor="white",
                      linewidth=0.5, label=st)

    # Highlight PULSE
    ax.axvspan(3.58, 4.42, color="#ede9fe", alpha=0.5, zorder=0)

    handles = [mpatches.Patch(color=colors[s], label=s) for s in structs]
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Dice (%)", fontsize=11)
    ax.set_ylim(70, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.97, "† Different eval split", transform=ax.transAxes,
            fontsize=7.5, color="#64748b", ha="right", va="top")

    plt.tight_layout()
    out = OUT / "comparison_baselines.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_ablation_bars()
    fig_ensemble_scaling()
    fig_fold_stability()
    fig_generalisation()
    fig_aug_ablation()
    fig_comparison()
    print("\nAll figures generated.")
