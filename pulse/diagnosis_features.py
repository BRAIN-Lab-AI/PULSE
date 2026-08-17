#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULSE — Clinical-Biomarker Utility Functions
============================================
Pure utility module: ACDC patient I/O, biomarker computation, and segmentation
metrics.  Imported by ensemble_eval.py (the main evaluation pipeline).

NOT a standalone script — to extract features from trained models, run:
  python ensemble_eval.py --folds_dir folds --out diagnosis_features_ensemble.json
"""

import math, os, warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore")

# Data root — set ACDC_ROOT env var or pass --data_root to the calling script.
# Expected structure: <data_root>/training/patient001/ and <data_root>/testing/patient101/
ROOT       = Path(os.environ.get("ACDC_ROOT",
                  str(Path(__file__).resolve().parent / "ACDC" / "database")))
TRAIN_ROOT = ROOT / "training"
TEST_ROOT  = ROOT / "testing"

SEG_CLASSES = 4   # BG=0, RV=1, Myo=2, LV=3
CLS_MAP     = {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4}
CLS_NAMES   = ["NOR", "DCM", "HCM", "MINF", "RV"]

FEAT_KEYS = [
    "LV_EDV", "LV_ESV", "LV_EF", "LV_SV", "LVM", "Myo_ED", "Myo_ES",
    "RV_EDV", "RV_ESV", "RV_EF", "RV_SV",
    "LV_EDVi", "LV_ESVi", "LVMi", "RV_EDVi",
    "concentricity", "RV_LV_EDV", "ESV_EDV", "RVEF_LVEF", "myo_lv_ratio",
    "wt_mean", "wt_max", "wt_std",
]


# ── ACDC I/O ──────────────────────────────────────────────────────────────────
def parse_info_cfg(path: Path) -> Dict:
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            for key in ("ED", "ES", "Group", "NbFrame", "Height", "Weight"):
                if line.startswith(f"{key}:"):
                    val = line.split(":", 1)[1].strip()
                    if key in {"ED", "ES", "NbFrame"}:
                        info[key] = int(float(val))
                    elif key in {"Height", "Weight"}:
                        info[key] = float(val)
                    else:
                        info[key] = val
    return info


def list_patients(root: Path) -> Dict[str, Dict]:
    """Return dict of {pid: {pid, group, height, weight, phases: {ED, ES}}}."""
    out: Dict[str, Dict] = {}
    for pdir in sorted(root.glob("patient*/")):
        cfg = pdir / "Info.cfg"
        if not cfg.exists():
            continue
        info = parse_info_cfg(cfg)
        if "ED" not in info or "ES" not in info:
            continue
        rec = {"pid": pdir.name, "group": info.get("Group", "NOR"),
               "height": info.get("Height", float("nan")),
               "weight": info.get("Weight", float("nan")), "phases": {}}
        for phase in ("ED", "ES"):
            idx = info[phase]
            img_p = pdir / f"{pdir.name}_frame{idx:02d}.nii"
            gt_p  = pdir / f"{pdir.name}_frame{idx:02d}_gt.nii"
            if not img_p.exists(): img_p = img_p.with_suffix(".nii.gz")
            if not gt_p.exists():  gt_p  = gt_p.with_suffix(".nii.gz")
            if img_p.exists() and gt_p.exists():
                rec["phases"][phase] = {"img": img_p, "gt": gt_p}
        if len(rec["phases"]) == 2:
            out[pdir.name] = rec
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


# ── Biomarker feature engineering ─────────────────────────────────────────────
def bsa_dubois(height_cm: float, weight_kg: float) -> float:
    if not (height_cm > 0 and weight_kg > 0):
        return float("nan")
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)


def wall_thickness_stats(ed_mask: np.ndarray, dx: float, dy: float) -> Dict:
    """Annulus-based myocardial thickness per slice (mm).
    Outer radius (LV+Myo) minus inner radius (LV), in-plane.
    Captures HCM (thick walls) vs DCM (thin walls)."""
    px = math.sqrt(max(dx * dy, 1e-6))
    th = []
    for z in range(ed_mask.shape[-1]):
        m   = ed_mask[..., z]
        lv  = (m == 3).sum()
        myo = (m == 2).sum()
        if lv > 5 and myo > 5:
            r_in  = math.sqrt(lv  / math.pi)
            r_out = math.sqrt((lv + myo) / math.pi)
            th.append((r_out - r_in) * px)
    if not th:
        return {"wt_mean": 0.0, "wt_max": 0.0, "wt_std": 0.0}
    return {"wt_mean": float(np.mean(th)),
            "wt_max":  float(np.max(th)),
            "wt_std":  float(np.std(th))}


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing) -> np.ndarray:
    """Symmetric surface distances (mm) between binary masks a and b (3D)."""
    if a.sum() == 0 or b.sum() == 0:
        return np.array([np.nan])
    struct  = ndimage.generate_binary_structure(3, 1)
    a_surf  = a ^ ndimage.binary_erosion(a, struct)
    b_surf  = b ^ ndimage.binary_erosion(b, struct)
    dt_b    = ndimage.distance_transform_edt(~b_surf, sampling=spacing)
    dt_a    = ndimage.distance_transform_edt(~a_surf, sampling=spacing)
    return np.concatenate([dt_b[a_surf], dt_a[b_surf]])


def seg_metrics(pred3d: np.ndarray, gt3d: np.ndarray, spacing) -> Dict:
    """Per-class Dice, IoU, HD95, ASSD (mm). Classes: RV=1, Myo=2, LV=3."""
    out = {}
    for lab, name in [(1, "RV"), (2, "Myo"), (3, "LV")]:
        p     = (pred3d == lab)
        g     = (gt3d   == lab)
        inter = float((p & g).sum())
        psum  = float(p.sum())
        gsum  = float(g.sum())
        union = float((p | g).sum())
        dice  = (2 * inter / (psum + gsum)) if (psum + gsum) > 0 else float("nan")
        iou   = (inter / union)             if union          > 0 else float("nan")
        sd    = _surface_distances(p, g, spacing)
        hd95  = float(np.nanpercentile(sd, 95)) if not np.all(np.isnan(sd)) else float("nan")
        assd  = float(np.nanmean(sd))           if not np.all(np.isnan(sd)) else float("nan")
        out[name] = {"dice": dice, "iou": iou, "hd95": hd95, "assd": assd}
    return out


def biomarkers(ed_mask: np.ndarray, es_mask: np.ndarray,
               voxvol_mL: float, dx: float, dy: float, bsa: float) -> Dict:
    """Compute 23-dim clinical biomarker vector from 3D ED and ES masks."""
    def vol(m, lab): return float((m == lab).sum()) * voxvol_mL

    edv    = vol(ed_mask, 3);  esv    = vol(es_mask, 3)
    lvm    = vol(ed_mask, 2) * 1.05   # myocardial mass (g), density 1.05 g/mL
    rv_edv = vol(ed_mask, 1);  rv_esv = vol(es_mask, 1)
    myo_ed = vol(ed_mask, 2);  myo_es = vol(es_mask, 2)
    ef     = (edv - esv) / edv * 100    if edv    > 0 else 0.0
    rv_ef  = (rv_edv - rv_esv) / rv_edv * 100 if rv_edv > 0 else 0.0
    sv     = edv - esv
    rv_sv  = rv_edv - rv_esv

    def bi(v): return v / bsa if (bsa == bsa and bsa > 0) else float("nan")

    f = {
        "LV_EDV":       edv,
        "LV_ESV":       esv,
        "LV_EF":        ef,
        "LV_SV":        sv,
        "LVM":          lvm,
        "Myo_ED":       myo_ed,
        "Myo_ES":       myo_es,
        "RV_EDV":       rv_edv,
        "RV_ESV":       rv_esv,
        "RV_EF":        rv_ef,
        "RV_SV":        rv_sv,
        "LV_EDVi":      bi(edv),
        "LV_ESVi":      bi(esv),
        "LVMi":         bi(lvm),
        "RV_EDVi":      bi(rv_edv),
        "concentricity": lvm / edv       if edv    > 0 else 0.0,
        "RV_LV_EDV":    rv_edv / edv    if edv    > 0 else 0.0,
        "ESV_EDV":      esv / edv       if edv    > 0 else 0.0,
        "RVEF_LVEF":    rv_ef / ef      if ef     > 0 else 0.0,
        "myo_lv_ratio": myo_ed / edv    if edv    > 0 else 0.0,
    }
    f.update(wall_thickness_stats(ed_mask, dx, dy))
    return f
