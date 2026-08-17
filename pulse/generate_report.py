#!/usr/bin/env python3
"""
PULSE — Structured Template-Based Clinical Report Generation
============================================================
Generates one structured clinical report per patient from:
  - Biomarker features extracted from predicted masks (diagnosis_features_ensemble.json)
  - Disease classification from Random Forest (diagnosis_classify.py)

This is a deterministic, rule-based system. No language model is used.
The template fills in measured clinical values and flags abnormal findings.

Usage:
  python generate_report.py \
      --features diagnosis_features_ensemble.json \
      --out_dir reports/
"""
import argparse, json, os
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from diagnosis_features import CLS_NAMES, FEAT_KEYS
CLS_FULL = {
    "NOR":  "Normal",
    "DCM":  "Dilated Cardiomyopathy",
    "HCM":  "Hypertrophic Cardiomyopathy",
    "MINF": "Myocardial Infarction",
    "RV":   "Right Ventricular Abnormality",
}

# Clinical reference ranges (adult, from Bernard 2018 and ESC guidelines)
NORMAL_RANGES = {
    "LV_EDV":  (50,  200,  "mL",  "LV End-Diastolic Volume"),
    "LV_ESV":  (10,  80,   "mL",  "LV End-Systolic Volume"),
    "LV_EF":   (50,  80,   "%",   "LV Ejection Fraction"),
    "LV_SV":   (40,  130,  "mL",  "LV Stroke Volume"),
    "LVM":     (70,  200,  "g",   "LV Myocardial Mass"),
    "RV_EDV":  (50,  200,  "mL",  "RV End-Diastolic Volume"),
    "RV_ESV":  (10,  100,  "mL",  "RV End-Systolic Volume"),
    "RV_EF":   (40,  75,   "%",   "RV Ejection Fraction"),
    "wt_mean": (5,   12,   "mm",  "Mean Wall Thickness (ED)"),
}

CLINICAL_NOTES = {
    "NOR":  [
        "LV and RV morphology and function appear within normal limits.",
        "Ejection fraction is preserved.",
        "No significant volumetric abnormalities detected.",
    ],
    "DCM":  [
        "LV dilation with significantly reduced ejection fraction detected.",
        "Pattern consistent with dilated cardiomyopathy.",
        "RV may show secondary dilation; clinical correlation advised.",
        "Recommend echocardiographic confirmation and cardiology referral.",
    ],
    "HCM":  [
        "Increased myocardial wall thickness detected.",
        "Pattern consistent with hypertrophic cardiomyopathy.",
        "Ejection fraction may be preserved or hyperdynamic.",
        "Recommend genetic counseling and arrhythmia risk assessment.",
    ],
    "MINF": [
        "Wall motion abnormality and myocardial thinning detected.",
        "Pattern consistent with prior myocardial infarction.",
        "Reduced ejection fraction with regional dysfunction.",
        "Recommend coronary angiography and viability assessment.",
    ],
    "RV":   [
        "Significant RV dilation with disproportionate RV/LV ratio detected.",
        "Pattern consistent with right ventricular abnormality.",
        "RV ejection fraction may be reduced.",
        "Recommend pulmonary pressure assessment and right heart catheterization.",
    ],
}


def flag(key, value):
    """Return 'NORMAL', 'LOW', or 'HIGH' based on reference ranges."""
    if key not in NORMAL_RANGES:
        return ""
    lo, hi, _, _ = NORMAL_RANGES[key]
    if value < lo:
        return " [LOW]"
    if value > hi:
        return " [HIGH]"
    return " [NORMAL]"


def format_val(key, value):
    if key not in NORMAL_RANGES:
        return f"{value:.2f}"
    _, _, unit, _ = NORMAL_RANGES[key]
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "mL":
        return f"{value:.1f} mL"
    if unit == "g":
        return f"{value:.1f} g"
    if unit == "mm":
        return f"{value:.2f} mm"
    return f"{value:.3f} {unit}"


def generate_report(pid, feat, diagnosis, confidence, split, gt_label=None):
    lines = []
    a = lines.append

    a("=" * 72)
    a("  PULSE CARDIAC MRI AUTOMATED ANALYSIS REPORT")
    a("  (Template-Based Rule System — Not AI Language Model Output)")
    a("=" * 72)
    a(f"  Patient ID : {pid}")
    a(f"  Split      : {split.upper()}")
    a(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if gt_label is not None:
        a(f"  GT Label   : {CLS_NAMES[gt_label]} ({CLS_FULL[CLS_NAMES[gt_label]]})")
    a("")

    # --- DIAGNOSIS ---
    a("DIAGNOSIS")
    a("-" * 72)
    dx = CLS_NAMES[diagnosis]
    a(f"  Primary Finding  : {dx} — {CLS_FULL[dx]}")
    a(f"  Model Confidence : {confidence*100:.1f}%")
    if gt_label is not None:
        match = "CORRECT" if diagnosis == gt_label else "INCORRECT"
        a(f"  Verification     : {match}")
    a("")

    # --- VOLUMETRIC INDICES ---
    a("VENTRICULAR VOLUMETRIC INDICES")
    a("-" * 72)
    metrics = [
        ("LV_EDV", feat.get("LV_EDV", float("nan"))),
        ("LV_ESV", feat.get("LV_ESV", float("nan"))),
        ("LV_EF",  feat.get("LV_EF",  float("nan"))),
        ("LV_SV",  feat.get("LV_SV",  float("nan"))),
        ("LVM",    feat.get("LVM",    float("nan"))),
        ("RV_EDV", feat.get("RV_EDV", float("nan"))),
        ("RV_ESV", feat.get("RV_ESV", float("nan"))),
        ("RV_EF",  feat.get("RV_EF",  float("nan"))),
        ("wt_mean",feat.get("wt_mean",float("nan"))),
    ]
    for key, val in metrics:
        if key in NORMAL_RANGES:
            label = NORMAL_RANGES[key][3]
            a(f"  {label:<38}: {format_val(key, val)}{flag(key, val)}")
    a("")

    # --- SHAPE RATIOS ---
    a("SHAPE AND FUNCTION RATIOS")
    a("-" * 72)
    rv_lv = feat.get("RV_LV_EDV", float("nan"))
    conc  = feat.get("concentricity", float("nan"))
    a(f"  RV/LV EDV Ratio                       : {rv_lv:.3f}  (NOR ~1.17, RV >2.0)")
    a(f"  LV Concentricity (LVM/LV_EDV)         : {conc:.3f}  (NOR ~0.75)")
    a("")

    # --- CLINICAL NOTES ---
    a("CLINICAL NOTES")
    a("-" * 72)
    for note in CLINICAL_NOTES.get(dx, []):
        a(f"  * {note}")
    a("")

    # --- DISCLAIMER ---
    a("DISCLAIMER")
    a("-" * 72)
    a("  This report is generated automatically by the PULSE framework.")
    a("  Values are derived from deep learning segmentation predictions.")
    a("  ALL findings must be reviewed and confirmed by a qualified")
    a("  cardiologist before any clinical decision-making.")
    a("=" * 72)
    a("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="diagnosis_features_ensemble.json")
    ap.add_argument("--seeds",    type=int, default=10)
    ap.add_argument("--out_dir",  default="reports")
    ap.add_argument("--show_n",   type=int, default=5,
                    help="Print this many reports to stdout")
    args = ap.parse_args()

    with open(args.features) as fh:
        rows = json.load(fh)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Train RF on training set
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows  = [r for r in rows if r["split"] == "test"]

    def mat(rs, which):
        X = np.array([[r[which].get(k, 0.0) for k in FEAT_KEYS] for r in rs], float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array([r["label"] for r in rs])
        return X, y

    Xtr, ytr = mat(train_rows, "feat_pred")
    Xte, yte = mat(test_rows,  "feat_pred")

    # Ensemble over 10 seeds
    all_proba = []
    for s in range(args.seeds):
        clf = RandomForestClassifier(n_estimators=600, max_depth=None,
                                     min_samples_leaf=1, random_state=s,
                                     class_weight="balanced")
        clf.fit(Xtr, ytr)
        all_proba.append(clf.predict_proba(Xte))
    ens_proba = np.mean(all_proba, axis=0)
    ens_pred  = np.argmax(ens_proba, axis=1)

    acc = accuracy_score(yte, ens_pred)
    print(f"Test accuracy: {acc*100:.1f}%  ({int(acc*len(yte))}/{len(yte)} correct)")
    print(f"Generating reports in {args.out_dir}/")
    print()

    shown = 0
    for i, r in enumerate(test_rows):
        pid  = r["pid"]
        feat = r["feat_pred"]
        diag = int(ens_pred[i])
        conf = float(ens_proba[i, diag])
        gt   = int(r["label"])

        report = generate_report(pid, feat, diag, conf, "test", gt_label=gt)

        out_path = Path(args.out_dir) / f"{pid}_report.txt"
        out_path.write_text(report)

        if shown < args.show_n:
            print(report)
            shown += 1

    print(f"\nWrote {len(test_rows)} reports to {args.out_dir}/")


if __name__ == "__main__":
    main()
