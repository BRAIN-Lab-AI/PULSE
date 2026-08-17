"""
PULSE architecture diagram — clean horizontal flow matching draw.io layout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(22, 10.2))
ax.set_xlim(0, 22)
ax.set_ylim(0, 10.2)
ax.axis("off")
fig.patch.set_facecolor("white")

C_IN  = "#dbeafe" # light blue
C_BB  = "#fef3c7" # light amber
C_DEC = "#dcfce7" # light green
C_SEG = "#ede9fe" # light purple
C_BIO = "#fef9c3" # light yellow
C_RF  = "#fce7f3" # light pink
C_REP = "#d1fae5" # mint green
C_LOS = "#f1f5f9" # light grey
C_CUR = "#ede9fe" # light purple
EDGE  = "#64748b" # grey border
DARK  = "#1e293b" # dark text

def draw_box_only(ax, x, y, w, h, fc, radius=0.2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.06,rounding_size={radius}",
                       facecolor=fc, edgecolor=EDGE, linewidth=1.3, zorder=3)
    ax.add_patch(p)

def harrow(ax, x0, x1, y, label="", up=0.12):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=EDGE,
                                lw=1.6, mutation_scale=15), zorder=5)
    if label:
        ax.text((x0+x1)/2, y+up, label, ha="center", va="bottom",
                fontsize=7, color="#64748b")

def varrow(ax, x, y0, y1, label=""):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE,
                                lw=1.6, mutation_scale=15), zorder=5)
    if label:
        mx = x+0.1
        ax.text(mx, (y0+y1)/2, label, ha="left", va="center",
                fontsize=7, color="#64748b")

def draw_tap_line(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE,
                                lw=1.2, mutation_scale=10), zorder=5)

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(11, 9.65, "PULSE — Unified Cardiac Segmentation, Diagnosis & Report Generation",
        ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 & 2 layout and boxes
# ═══════════════════════════════════════════════════════════════════════════

# 1. 2.5D Input Stack
draw_box_only(ax, 0.2, 5.2, 2.0, 4.0, C_IN)
ax.text(1.2, 8.85, "2.5D Input Stack", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)
for i, (yb, lbl) in enumerate([(6.2, "z+1"), (6.9, "z"), (7.6, "z-1")]):
    r = FancyBboxPatch((0.38, yb), 1.64, 0.55,
                       boxstyle="round,pad=0.02",
                       facecolor="#93c5fd", edgecolor="#1d4ed8",
                       linewidth=0.8, zorder=4)
    ax.add_patch(r)
    ax.text(1.2, yb+0.275, lbl, ha="center", va="center",
            fontsize=8.5, color="#1e3a8a", fontweight="bold", zorder=5)
ax.text(1.2, 5.75, "3 adjacent CMR slices (256×256×3)", ha="center", va="center",
        fontsize=6.5, color="#475569", zorder=4)
ax.text(1.2, 5.45, "z-score normalised per slice", ha="center", va="center",
        fontsize=6.5, color="#475569", zorder=4)

harrow(ax, 2.2, 2.8, 7.2)

# 2. DINOv2 ViT-B/14 Backbone
draw_box_only(ax, 2.8, 5.2, 4.2, 4.0, C_BB)
ax.text(4.9, 8.85, "DINOv2 ViT-B/14", ha="center", va="center", fontsize=10.0, fontweight="bold", color=DARK, zorder=4)
ax.text(4.9, 8.55, "Pretrained · 86M params · LVD-142M", ha="center", va="center", fontsize=7.0, color="#475569", style="italic", zorder=4)

blk_names = [
    "Blocks 0–2   (MHSA + FFN)",
    "Blocks 3–5   (MHSA + FFN)",
    "Blocks 6–8   (MHSA + FFN)",
    "Blocks 9–11  (MHSA + FFN)"
]
blk_fc = ["#fef3c7", "#fde68a", "#fbbf24", "#f59e0b"]
blk_edge = ["#fcd34d", "#fbbf24", "#f59e0b", "#d97706"]
for i, (nm, fc, ec) in enumerate(zip(blk_names, blk_fc, blk_edge)):
    by = 7.85 - i*0.65
    r = FancyBboxPatch((2.95, by), 3.90, 0.50,
                       boxstyle="round,pad=0.03",
                       facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=4)
    ax.add_patch(r)
    ax.text(4.9, by+0.25, nm, ha="center", va="center",
            fontsize=7.5, color=DARK, zorder=5)

# multi-scale taps F3 F6 F9 F12
tap_y = [8.10, 7.45, 6.80, 6.15]
tap_lbls = ["F3", "F6", "F9", "F12"]
for ty, tl in zip(tap_y, tap_lbls):
    ax.annotate("", xy=(7.0, ty), xytext=(6.85, ty),
                arrowprops=dict(arrowstyle="-|>", color="#92400e",
                                lw=1.0, mutation_scale=9), zorder=5)
    ax.text(7.05, ty, tl, ha="left", va="center",
            fontsize=7, color="#92400e", fontweight="bold", zorder=5)

# curriculum badges (below the box)
draw_box_only(ax, 2.8, 4.4, 2.05, 0.6, C_CUR)
ax.text(3.825, 4.82, "Stage 1 | ep 1–70", ha="center", va="center", fontsize=7.0, fontweight="bold", color=DARK, zorder=4)
ax.text(3.825, 4.58, "blocks 0–1 frozen", ha="center", va="center", fontsize=6.0, color="#475569", zorder=4)

draw_box_only(ax, 4.95, 4.4, 2.05, 0.6, C_CUR)
ax.text(5.975, 4.82, "Stage 2 | ep 71–120", ha="center", va="center", fontsize=7.0, fontweight="bold", color=DARK, zorder=4)
ax.text(5.975, 4.58, "all unfrozen", ha="center", va="center", fontsize=6.0, color="#475569", zorder=4)


# 3. DPT Decoder
draw_box_only(ax, 7.8, 5.2, 3.2, 4.0, C_DEC)
ax.text(9.4, 8.85, "DPT Decoder (DPTDecoderDS)", ha="center", va="center", fontsize=10.0, fontweight="bold", color=DARK, zorder=4)
ax.text(9.4, 8.55, "Reassemble → Fusion → Progressive Upsampling", ha="center", va="center", fontsize=7.0, color="#15803d", style="italic", zorder=4)

# vertical chain of 5 stages
stages = ["18×18", "36×36", "72×72", "144×144", "256×256"]
for i, lbl in enumerate(stages):
    sy = 5.9 + i*0.6
    r = FancyBboxPatch((8.85, sy), 1.1, 0.35,
                       boxstyle="round,pad=0.02",
                       facecolor="#bbf7d0", edgecolor="#15803d", linewidth=0.8, zorder=4)
    ax.add_patch(r)
    ax.text(9.4, sy+0.175, lbl, ha="center", va="center",
            fontsize=7.5, color="#14532d", fontweight="bold", zorder=5)

for i in range(4):
    sy = 5.9 + i*0.6
    varrow(ax, 9.4, sy+0.35, sy+0.6)
    ax.text(10.02, sy+0.475, "ResBlock +\nConvTranspose 2×2", ha="left", va="center",
            fontsize=5.5, color="#166534", style="italic", zorder=5)

# taps to decoder chips
draw_tap_line(ax, 7.3, 6.15, 8.85, 6.075) # F12 to 18x18
draw_tap_line(ax, 7.3, 6.80, 8.85, 6.675) # F9 to 36x36
draw_tap_line(ax, 7.3, 7.45, 8.85, 7.275) # F6 to 72x72
draw_tap_line(ax, 7.3, 8.10, 8.85, 7.875) # F3 to 144x144

# Output chips on the right side
# A1
r1 = FancyBboxPatch((11.3, 7.05), 1.2, 0.45,
                    boxstyle="round,pad=0.02",
                    facecolor="#dcfce7", edgecolor="#166534", linewidth=0.8, zorder=4)
ax.add_patch(r1)
ax.text(11.9, 7.33, "A₁ coarse", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#14532d", zorder=5)
ax.text(11.9, 7.10, "×0.25", ha="center", va="bottom", fontsize=6.5, color="#166534", zorder=5)

# A2
r2 = FancyBboxPatch((11.3, 7.65), 1.2, 0.45,
                    boxstyle="round,pad=0.02",
                    facecolor="#86efac", edgecolor="#166534", linewidth=0.8, zorder=4)
ax.add_patch(r2)
ax.text(11.9, 7.93, "A₂ fine", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#14532d", zorder=5)
ax.text(11.9, 7.70, "×0.50", ha="center", va="bottom", fontsize=6.5, color="#166534", zorder=5)

# S
r3 = FancyBboxPatch((11.3, 8.25), 1.2, 0.45,
                    boxstyle="round,pad=0.02",
                    facecolor="#22c55e", edgecolor="#15803d", linewidth=0.8, zorder=4)
ax.add_patch(r3)
ax.text(11.9, 8.53, "S main", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="white", zorder=5)
ax.text(11.9, 8.30, "×1.00", ha="center", va="bottom", fontsize=6.5, color="white", zorder=5)

harrow(ax, 9.95, 11.3, 7.275)
harrow(ax, 9.95, 11.3, 7.875)
harrow(ax, 9.95, 11.3, 8.475)

ax.text(11.9, 8.85, "Deep supervision:\n0.25·L(A₁) + 0.50·L(A₂) + 1.0·L(S)",
        ha="center", va="center", fontsize=6.2, color="#15803d", style="italic", zorder=5)


# 4. Segmentation Head
draw_box_only(ax, 12.8, 7.4, 3.0, 1.8, C_SEG)
ax.text(14.3, 8.85, "Segmentation Head", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)
ax.text(14.3, 8.55, "4 classes · LCC post-processing · 3D volume inference", ha="center", va="center", fontsize=6.8, color="#475569", zorder=4)

chip_c = ["#94a3b8", "#06b6d4", "#22c55e", "#f97316"]
chip_l = ["BG", "RV", "Myo", "LV"]
for ci, (cc, cl) in enumerate(zip(chip_c, chip_l)):
    cx = 13.0 + ci * 0.7
    r = FancyBboxPatch((cx, 8.05), 0.60, 0.30,
                       boxstyle="round,pad=0.02",
                       facecolor=cc, edgecolor="white", linewidth=0.5, zorder=5)
    ax.add_patch(r)
    ax.text(cx+0.30, 8.20, cl, ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold", zorder=6)

ax.text(14.3, 7.70, "Dice / IoU / HD95 / ASSD", ha="center", va="center", fontsize=7.0, color="#6b21a8", style="italic", zorder=4)

harrow(ax, 12.5, 12.8, 8.475)


# 5. Biomarker Extraction
draw_box_only(ax, 12.8, 4.6, 3.5, 2.0, C_BIO)
ax.text(14.55, 6.35, "Biomarker Extraction", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)
ax.text(14.55, 6.10, "23-dim clinical feature vector", ha="center", va="center", fontsize=7.2, color="#854d0e", zorder=4)

left_lines = [
    "LV_EDV · LV_ESV · LV_EF · LV_SV",
    "LVM · RV_EDV · RV_ESV · RV_EF",
    "RV_SV · LV_EDVi · LV_ESVi · LVMi"
]
right_lines = [
    "RV_EDVi · concentricity",
    "RV/LV ratio · ESV/EDV",
    "RVEF/LVEF · myo_lv_ratio",
    "wt_mean · wt_max · wt_std"
]
for idx, txt in enumerate(left_lines):
    ax.text(13.0, 5.75 - idx*0.32, txt, ha="left", va="center", fontsize=6.2, color="#475569", zorder=4)
for idx, txt in enumerate(right_lines):
    ax.text(14.65, 5.75 - idx*0.27, txt, ha="left", va="center", fontsize=6.2, color="#475569", zorder=4)

ax.text(14.55, 4.85, "Wall thickness (annulus, ED) · BSA-indexed (Du Bois)", ha="center", va="center", fontsize=6.2, color="#854d0e", style="italic", zorder=4)

varrow(ax, 14.3, 7.4, 6.6)


# 6. Random Forest Classifier
draw_box_only(ax, 16.7, 4.6, 2.6, 2.0, C_RF)
ax.text(18.0, 6.35, "Random Forest Classifier", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)

rf_lines = [
    "600 trees · balanced class weights",
    "Train: 100 ACDC subjects",
    "Multi-seed ensemble (10 seeds)",
    "Input: 23-dim biomarker vector"
]
for idx, txt in enumerate(rf_lines):
    ax.text(18.0, 5.80 - idx*0.32, txt, ha="center", va="center", fontsize=6.8, color="#475569", zorder=4)

harrow(ax, 16.3, 16.7, 5.6)


# 7. Disease Diagnosis
draw_box_only(ax, 19.7, 4.6, 2.1, 2.0, C_RF)
ax.text(20.75, 6.35, "Disease Diagnosis", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)

diag_chips = ["NOR", "DCM", "HCM", "MINF", "RVA"]
for ci, cl in enumerate(diag_chips):
    cx = 19.7 + 0.08 + ci * 0.40
    r = FancyBboxPatch((cx, 5.75), 0.34, 0.28,
                       boxstyle="round,pad=0.01",
                       facecolor="#fbcfe8", edgecolor="#db2777", linewidth=0.5, zorder=5)
    ax.add_patch(r)
    ax.text(cx+0.17, 5.89, cl, ha="center", va="center",
            fontsize=6.5, color="#be185d", fontweight="bold", zorder=6)

diag_perf = [
    "Accuracy: 90.0%  (45/50)",
    "Macro-AUC: 0.983",
    "Macro-F1:  0.900"
]
for idx, txt in enumerate(diag_perf):
    ax.text(20.75, 5.35 - idx*0.30, txt, ha="center", va="center", fontsize=7.0, color="#475569", zorder=4)

harrow(ax, 19.3, 19.7, 5.6)


# 8. Clinical Report
draw_box_only(ax, 19.7, 7.4, 2.1, 1.8, C_REP)
ax.text(20.75, 8.85, "Clinical Report", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)

rep_lines = [
    "Rule-based template filling",
    "Flags: NORMAL / LOW / HIGH",
    "92.7% agreement with cardiologists"
]
for idx, txt in enumerate(rep_lines):
    ax.text(20.75, 8.35 - idx*0.32, txt, ha="center", va="center", fontsize=7.0, color="#475569", zorder=4)

varrow(ax, 20.75, 6.6, 7.4)

# Dashed arrow from Segmentation Head to Clinical Report
ax.annotate("", xy=(19.7, 8.3), xytext=(15.8, 8.3),
            arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.3,
                            mutation_scale=13, linestyle="dashed"), zorder=5)
ax.text((15.8+19.7)/2, 8.42, "clinical indices", ha="center", va="bottom",
        fontsize=7, color="#64748b", style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# ROW 3  (y=0.9–2.5): Loss & Inference
# ═══════════════════════════════════════════════════════════════════════════

# Left Box: Training Loss
draw_box_only(ax, 0.2, 0.9, 10.6, 1.6, C_LOS)
ax.text(5.5, 2.2, "Training Loss  (segmentation only)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)
ax.text(5.5, 1.7,
        r"$\mathcal{L}_{total}$ = $\mathcal{L}_{Dice}$ + $\mathcal{L}_{CE}$ + 0.5·$\mathcal{L}_{Lov\acute{a}sz}$  +  0.5·$\mathcal{L}_{Bnd}$  +  0.25·$\mathcal{L}(A_1)$  +  0.5·$\mathcal{L}(A_2)$",
        ha="center", va="center", fontsize=8.5, color=DARK, zorder=4)
ax.text(5.5, 1.25, "Class weights: BG=0, RV×2, Myo×3, LV×1", ha="center", va="center", fontsize=8.0, color="#475569", zorder=4)

# Right Box: Inference Details
draw_box_only(ax, 11.2, 0.9, 10.6, 1.6, C_CUR)
ax.text(16.5, 2.2, "Inference: 5-Fold Ensemble + 4-Way TTA", ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK, zorder=4)
inf_lines = [
    "5 fold checkpoints · softmax probability averaging",
    "4-way TTA: original + H-flip + V-flip + H+V-flip",
    "Slice-by-slice 3D inference · LCC post-processing"
]
for idx, txt in enumerate(inf_lines):
    ax.text(16.5, 1.7 - idx*0.32, txt, ha="center", va="center", fontsize=8.0, color="#475569", zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# COLOUR LEGEND  (strip at the very bottom)
# ═══════════════════════════════════════════════════════════════════════════

legend_items = [
    (C_IN, "Input"),
    (C_BB, "DINOv2 Backbone"),
    (C_DEC, "DPT Decoder + Deep Supervision"),
    (C_SEG, "Segmentation"),
    (C_BIO, "Biomarker Extraction"),
    (C_RF, "Random Forest / Diagnosis"),
    (C_REP, "Report Generation"),
    (C_CUR, "Curriculum / Ensemble")
]
start_x = 0.4
item_w = 2.68
for i, (col, lbl) in enumerate(legend_items):
    cx = start_x + i * item_w
    rect = FancyBboxPatch((cx, 0.35), 0.35, 0.20,
                          boxstyle="round,pad=0.01",
                          facecolor=col, edgecolor="#64748b", linewidth=0.6, zorder=5)
    ax.add_patch(rect)
    ax.text(cx + 0.45, 0.45, lbl, ha="left", va="center",
            fontsize=7.2, color=DARK, fontweight="bold", zorder=5)


plt.tight_layout(pad=0.4)
out = "architecture_generated.png"
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
