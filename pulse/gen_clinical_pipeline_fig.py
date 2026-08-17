#!/usr/bin/env python3
"""
Generate clinical_pipeline_fig.png — visual continuation of DL.png.
Matches style: light-gray grid background, green/blue rounded boxes,
dashed arrows, bold header labels, dimension sub-labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Palette matching DL.png ───────────────────────────────────────────────────
BG        = "#f2f2f2"   # canvas background (light gray)
GRID_C    = "#dddddd"   # grid lines
BOX_BLUE  = "#cde4f5"   # DPT-style light blue fill
BOX_GREEN = "#c8ecd0"   # post-processing / clinical green fill
BOX_GOLD  = "#fdf3cc"   # report / output gold fill
EDGE_BLUE = "#3a87c8"   # blue box border
EDGE_GRN  = "#2e8b57"   # green box border
EDGE_GOLD = "#c8a800"   # gold box border
HDR_BLUE  = "#2563a0"   # header text (blue)
HDR_GRN   = "#1a6b3a"   # header text (green)
HDR_GOLD  = "#7a5000"   # header text (gold)
DARK      = "#1a1a2e"   # primary text
DIM_C     = "#666680"   # dimension labels / secondary text
RED_FLAG  = "#c0392b"
GRN_FLAG  = "#1a6b3a"
ARROW_C   = "#333355"

W, H = 18, 10
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Grid background ───────────────────────────────────────────────────────────
for x in np.arange(0, W+0.5, 0.5):
    ax.plot([x, x], [0, H], color=GRID_C, lw=0.4, zorder=0)
for y in np.arange(0, H+0.5, 0.5):
    ax.plot([0, W], [y, y], color=GRID_C, lw=0.4, zorder=0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc=BOX_BLUE, ec=EDGE_BLUE, lw=1.6, r=0.2, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)

def hdr(x, y, w, h, text, fc=HDR_BLUE, fs=8.5, bold=True):
    ax.text(x+w/2, y+h-0.22, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=fc, zorder=3)

def txt(x, y, text, fs=7.5, color=DARK, ha="left", bold=False, va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, zorder=3)

def divider(x, y, w, color=EDGE_BLUE):
    ax.plot([x, x+w], [y, y], color=color, lw=0.9, zorder=3, ls="--", alpha=0.6)

def arr(x0, y0, x1, y1, lw=1.8, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                lw=lw, mutation_scale=13,
                                linestyle=ls),
                zorder=4)

def feat_pill(x, y, text, w=2.55, h=0.26, fc="#e8f3fc", ec=EDGE_BLUE):
    rbox(x, y, w, h, fc=fc, ec=ec, lw=0.8, r=0.08, zorder=3)
    ax.text(x+0.12, y+h/2, text, fontsize=6.8, va="center", color=DARK, zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK A — Segmentation output (bridge from DL.png)
# ═══════════════════════════════════════════════════════════════════════════════
ax0, ay0, aw, ah = 0.25, 5.2, 2.2, 4.3
rbox(ax0, ay0, aw, ah, fc=BOX_BLUE, ec=EDGE_BLUE)
hdr(ax0, ay0, aw, ah, "3-D Mask Output\n(ED + ES volumes)", fc=HDR_BLUE, fs=8.0)

# Colour-coded structure boxes
structs_col = [("#3b82f6","LV  (class 3)"),
               ("#93c5fd","Myo (class 2)"),
               ("#1d4ed8","RV  (class 1)")]
for i,(col,lbl) in enumerate(structs_col):
    sy = ay0 + ah - 1.15 - i*0.55
    rbox(ax0+0.18, sy, aw-0.36, 0.38, fc=col, ec=col, lw=0, r=0.08, zorder=3)
    ax.text(ax0+aw/2, sy+0.19, lbl, ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold", zorder=4)

divider(ax0+0.2, ay0+1.72, aw-0.4, EDGE_BLUE)
txt(ax0+aw/2, ay0+1.42, "Dice  RV 90.3 | Myo 84.7 | LV 91.6",
    fs=6.8, color=DIM_C, ha="center")
txt(ax0+aw/2, ay0+1.05, "Mean Dice: 88.8%  (5-fold + TTA)",
    fs=6.8, color=DIM_C, ha="center")
txt(ax0+aw/2, ay0+0.65, "Post-processing: LCC per class",
    fs=6.5, color=DIM_C, ha="center")
txt(ax0+aw/2, ay0+0.28, "Output: 4×D×H×W binary masks",
    fs=6.5, color=DIM_C, ha="center")

arr(ax0+aw, ay0+ah*0.5, 2.65, ay0+ah*0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK B — Clinical Biomarker Extractor
# ═══════════════════════════════════════════════════════════════════════════════
bx, by, bw, bh = 2.7, 0.4, 5.6, 9.2
rbox(bx, by, bw, bh, fc=BOX_GREEN, ec=EDGE_GRN, lw=2.0)
hdr(bx, by, bw, bh,
    "[ Clinical Biomarker Extractor  (23-dim feature vector) ]",
    fc=HDR_GRN, fs=9.0)

groups = [
    ("LV Volumes & Function",
     [("LV EDV (mL)", "LV ESV (mL)"),
      ("LV EF  (%)",  "LV SV  (mL)"),
      ("LV Mass (g)", None)]),
    ("Myocardium",
     [("Myo Volume ED (mL)", "Myo Volume ES (mL)")]),
    ("RV Volumes & Function",
     [("RV EDV (mL)", "RV ESV (mL)"),
      ("RV EF  (%)",  "RV SV  (mL)")]),
    ("Indexed Volumes  (÷ BSA)",
     [("LV EDVi", "LV ESVi"),
      ("LVMi",    "RV EDVi")]),
    ("Cross-Chamber Ratios",
     [("Concentricity",  "RV / LV EDV"),
      ("ESV / EDV",      "RVEF / LVEF"),
      ("Myo / LV ratio", None)]),
    ("Wall Thickness",
     [("wt mean (mm)", "wt max  (mm)"),
      ("wt std  (mm)", None)]),
]

pill_w = 2.55
gap = 0.12
cy = by + bh - 0.55

for grp_name, rows in groups:
    cy -= 0.30
    ax.text(bx+0.22, cy, grp_name,
            fontsize=7.8, color=HDR_GRN, fontweight="bold",
            va="top", zorder=3)
    cy -= 0.08
    for pair in rows:
        cy -= 0.31
        feat_pill(bx+0.22, cy, pair[0], w=pill_w)
        if pair[1]:
            feat_pill(bx+0.22+pill_w+gap, cy, pair[1], w=pill_w)
    cy -= 0.06

# Output vector banner
cy -= 0.18
rbox(bx+0.18, cy, bw-0.36, 0.38,
     fc=EDGE_GRN, ec=EDGE_GRN, lw=0, r=0.10, zorder=3)
ax.text(bx+bw/2, cy+0.19,
        "→  23-dim biomarker vector  →  Random Forest Classifier",
        ha="center", va="center", fontsize=8.2,
        color="white", fontweight="bold", zorder=4)

# Arrow B → C
arr(bx+bw, by+bh*0.72, 8.55, by+bh*0.72)

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK C — Random Forest Classifier
# ═══════════════════════════════════════════════════════════════════════════════
cx, cy2, cw, ch = 8.6, 5.2, 4.0, 4.4
rbox(cx, cy2, cw, ch, fc=BOX_BLUE, ec=EDGE_BLUE, lw=2.0)
hdr(cx, cy2, cw, ch, "[ Random Forest Classifier ]", fc=HDR_BLUE, fs=9.0)

rf_specs = [
    ("Estimators",     "600 decision trees"),
    ("Class weight",   "'balanced'"),
    ("Ensemble",       "10-seed averaged"),
    ("Validation",     "5-fold cross-validated"),
    ("Macro-AUC",      "0.982  (OvR)"),
    ("Balanced Acc.",  "90.0%  (50 test patients)"),
]
for i,(k,v) in enumerate(rf_specs):
    ry = cy2+ch-0.75-i*0.56
    rbox(cx+0.2, ry-0.09, cw-0.4, 0.38,
         fc="#e8f3fc", ec=EDGE_BLUE, lw=0.8, r=0.08, zorder=3)
    ax.text(cx+0.35, ry+0.10, k+":", fontsize=7.5,
            color=HDR_BLUE, fontweight="bold", va="center", zorder=4)
    ax.text(cx+cw-0.25, ry+0.10, v, fontsize=7.5,
            color=DARK, va="center", ha="right", zorder=4)

# Arrow C → D
arr(cx+cw, cy2+ch*0.50, 12.85, cy2+ch*0.50)

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK D — Disease Classification Output
# ═══════════════════════════════════════════════════════════════════════════════
dx, dy, dw, dh = 12.9, 5.2, 4.85, 4.4
rbox(dx, dy, dw, dh, fc=BOX_BLUE, ec=EDGE_BLUE, lw=2.0)
hdr(dx, dy, dw, dh, "Disease Classification Output", fc=HDR_BLUE, fs=9.0)

classes = [
    ("NOR","Normal",                1.00,"#0d2b5e"),
    ("DCM","Dilated CMP",           0.80,"#2563b8"),
    ("HCM","Hypertrophic CMP",      0.90,"#2563b8"),
    ("MINF","Myocardial Infarction",0.80,"#2563b8"),
    ("RV", "RV Dysfunction",        1.00,"#0d2b5e"),
]
bar0 = dx+1.0
barW = dw-1.25
for i,(code,name,acc,col) in enumerate(classes):
    ry = dy+dh-0.78-i*0.57
    ax.text(dx+0.15, ry+0.09, code, fontsize=8.2,
            color=DARK, fontweight="bold", va="center", zorder=4)
    # bar track
    rbox(bar0, ry-0.03, barW, 0.28, fc="#dce8f7", ec=EDGE_BLUE,
         lw=0.5, r=0.06, zorder=3)
    # filled bar
    rbox(bar0, ry-0.03, barW*acc, 0.28, fc=col, ec=col,
         lw=0, r=0.06, zorder=4)
    ax.text(bar0+0.08, ry+0.11, name,
            fontsize=7.0, va="center", color="white", zorder=5)
    ax.text(bar0+barW*acc+0.05, ry+0.11, f"{int(acc*100)}%",
            fontsize=7.5, va="center", color=DARK, zorder=5)

# Accuracy badge
rbox(dx+0.18, dy+0.10, dw-0.36, 0.38,
     fc=EDGE_BLUE, ec=EDGE_BLUE, lw=0, r=0.09, zorder=3)
ax.text(dx+dw/2, dy+0.29,
        "Overall Accuracy: 90.0%  |  Macro-AUC: 0.982",
        ha="center", va="center", fontsize=8.0,
        color="white", fontweight="bold", zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
# Arrow C down → E (report)
# ═══════════════════════════════════════════════════════════════════════════════
arr(cx+cw/2, cy2, cx+cw/2, 4.85, lw=1.8)

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK E — Structured Clinical Report
# ═══════════════════════════════════════════════════════════════════════════════
ex, ey, ew, eh = 8.6, 0.4, 9.15, 4.2
rbox(ex, ey, ew, eh, fc=BOX_GOLD, ec=EDGE_GOLD, lw=2.0)
hdr(ex, ey, ew, eh, "[ Structured Clinical Report ]", fc=HDR_GOLD, fs=9.0)

# Two columns inside: biomarker flags (left) + diagnosis text (right)
# --- Diagnosis header ---
rbox(ex+0.2, ey+eh-0.72, ew-0.4, 0.34,
     fc=EDGE_BLUE, ec=EDGE_BLUE, lw=0, r=0.09, zorder=3)
ax.text(ex+0.42, ey+eh-0.55, "DIAGNOSIS:", fontsize=8.0,
        color="white", fontweight="bold", va="center", zorder=4)
ax.text(ex+ew-0.25, ey+eh-0.55, "DCM — Dilated Cardiomyopathy",
        fontsize=8.0, color="#ffd080", fontweight="bold",
        va="center", ha="right", zorder=4)

# --- Biomarker rows in two columns ---
report = [
    ("LV EDV",     "185 mL", "HIGH ↑",   RED_FLAG),
    ("LV ESV",     "120 mL", "HIGH ↑",   RED_FLAG),
    ("LV EF",      "35 %",   "LOW ↓",    RED_FLAG),
    ("LV SV",      "65 mL",  "NORMAL",   GRN_FLAG),
    ("LV Mass",    "220 g",  "HIGH ↑",   RED_FLAG),
    ("RV EDV",     "145 mL", "HIGH ↑",   RED_FLAG),
    ("RV EF",      "42 %",   "LOW ↓",    RED_FLAG),
    ("Myo ED Vol", "98 mL",  "NORMAL",   GRN_FLAG),
    ("Concentr.",  "0.62",   "NORMAL",   GRN_FLAG),
    ("Wall Thick.", "7.2 mm","NORMAL",   GRN_FLAG),
]
half = len(report)//2
col_w = (ew-0.4)/2 - 0.1
for i, (k, v, flag, fc_flag) in enumerate(report):
    col = i // half
    row = i %  half
    rx = ex + 0.22 + col*(col_w+0.18)
    ry = ey + eh - 1.18 - row*0.60
    # row background
    rbox(rx, ry-0.09, col_w, 0.38, fc="#fefae8", ec=EDGE_GOLD, lw=0.6, r=0.07, zorder=3)
    ax.text(rx+0.10, ry+0.10, k, fontsize=7.2, color=DARK,
            fontweight="bold", va="center", zorder=4)
    ax.text(rx+col_w*0.44, ry+0.10, v, fontsize=7.2, color=DIM_C,
            va="center", zorder=4)
    ax.text(rx+col_w-0.08, ry+0.10, f"[{flag}]", fontsize=7.2,
            color=fc_flag, fontweight="bold", va="center", ha="right", zorder=4)

# Loss formula banner at bottom of report
ax.text(ex+ew/2, ey+0.20,
        "Rule-based flags: biomarker vs. published reference ranges  "
        "(NORMAL / LOW / HIGH)",
        ha="center", va="center", fontsize=7.2, color=HDR_GOLD,
        style="italic", zorder=4)

# ── Section label ─────────────────────────────────────────────────────────────
ax.text(W/2, H-0.22,
        "PULSE  —  Clinical Biomarker Extraction · Disease Diagnosis · Report Generation",
        ha="center", va="center", fontsize=10.5, fontweight="bold",
        color=DARK, zorder=5)

plt.tight_layout(pad=0.1)
out = "/SLURM/home/slurm_g202518690/student251/CVPR/paper_contents/clinical_pipeline_fig.png"
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
