#!/usr/bin/env python3
"""
PULSE — Structured Report Evaluation
=====================================
Evaluates the quality of PULSE structured reports on the ACDC test set (n=50).

Three evaluation axes:
  1. CLINICAL METRIC ACCURACY
     MAE (predicted mask features vs. GT mask features) for each biomarker.
     This quantifies how accurately the segmentation propagates to clinical values.

  2. ABNORMALITY FLAG ACCURACY
     For each metric with a reference range, compare the flag (NORMAL / LOW / HIGH)
     from predicted-mask features vs. GT-mask features.
     Reports flag agreement rate per metric and overall.

  3. DIAGNOSIS ACCURACY (in context of report)
     Patient-level classification accuracy, matching the report's diagnosis field
     to the ground-truth label.

Usage:
  python report_eval.py --features diagnosis_features_ensemble.json
"""
import argparse, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from diagnosis_features import CLS_NAMES, FEAT_KEYS

FEAT_UNITS = {
    "LV_EDV": "mL", "LV_ESV": "mL", "LV_EF": "%", "LV_SV": "mL",
    "LVM": "g", "Myo_ED": "mL", "Myo_ES": "mL",
    "RV_EDV": "mL", "RV_ESV": "mL", "RV_EF": "%", "RV_SV": "mL",
    "LV_EDVi": "mL/m2", "LV_ESVi": "mL/m2", "LVMi": "g/m2", "RV_EDVi": "mL/m2",
    "concentricity": "", "RV_LV_EDV": "", "ESV_EDV": "", "RVEF_LVEF": "", "myo_lv_ratio": "",
    "wt_mean": "mm", "wt_max": "mm", "wt_std": "mm",
}

# Metrics shown in the report and reportable in the paper
REPORT_METRICS = ["LV_EDV", "LV_ESV", "LV_EF", "LVM", "RV_EDV", "RV_ESV", "RV_EF", "wt_mean"]

NORMAL_RANGES = {
    "LV_EDV":  (50,  200),
    "LV_ESV":  (10,  80),
    "LV_EF":   (50,  80),
    "LV_SV":   (40,  130),
    "LVM":     (70,  200),
    "RV_EDV":  (50,  200),
    "RV_ESV":  (10,  100),
    "RV_EF":   (40,  75),
    "wt_mean": (5,   12),
}

def get_flag(key, val):
    if key not in NORMAL_RANGES or np.isnan(val):
        return "UNKNOWN"
    lo, hi = NORMAL_RANGES[key]
    if val < lo:   return "LOW"
    if val > hi:   return "HIGH"
    return "NORMAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="diagnosis_features_ensemble.json")
    ap.add_argument("--seeds",    type=int, default=10)
    args = ap.parse_args()

    with open(args.features) as fh:
        rows = json.load(fh)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows  = [r for r in rows if r["split"] == "test"]

    # ------------------------------------------------------------------ #
    # 1. CLINICAL METRIC ACCURACY                                          #
    # ------------------------------------------------------------------ #
    print("\n" + "="*64)
    print("CLINICAL METRIC ACCURACY  (pred mask vs GT mask, test n=50)")
    print("="*64)
    print(f"{'Metric':<18}{'MAE':>10}{'Std':>10}{'Max':>10}  Unit")
    print("-"*64)

    metric_maes = {}
    for key in FEAT_KEYS:
        pred_vals, gt_vals = [], []
        for r in test_rows:
            pv = r["feat_pred"].get(key, None)
            gv = r["feat_gt"].get(key, None)
            if pv is not None and gv is not None and not np.isnan(pv) and not np.isnan(gv):
                pred_vals.append(pv); gt_vals.append(gv)
        if not pred_vals:
            continue
        errs = np.abs(np.array(pred_vals) - np.array(gt_vals))
        mae = float(np.mean(errs)); std = float(np.std(errs)); mx = float(np.max(errs))
        metric_maes[key] = mae
        unit = FEAT_UNITS.get(key, "")
        mark = " *" if key in REPORT_METRICS else ""
        print(f"{key:<18}{mae:>10.3f}{std:>10.3f}{mx:>10.3f}  {unit}{mark}")

    print("\n  (* = metric shown in structured clinical report)")

    # ------------------------------------------------------------------ #
    # 2. ABNORMALITY FLAG ACCURACY                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "="*64)
    print("ABNORMALITY FLAG ACCURACY  (NORMAL/LOW/HIGH, test n=50)")
    print("="*64)
    print(f"{'Metric':<18}{'Flag Agree (%)':>16}  (predicted flag = GT flag?)")
    print("-"*64)

    total_flags, correct_flags = 0, 0
    for key in NORMAL_RANGES:
        agrees = []
        for r in test_rows:
            pv = r["feat_pred"].get(key, None)
            gv = r["feat_gt"].get(key, None)
            if pv is None or gv is None:
                continue
            pf = get_flag(key, pv); gf = get_flag(key, gv)
            agrees.append(1 if pf == gf else 0)
        if not agrees:
            continue
        rate = 100 * float(np.mean(agrees))
        total_flags   += len(agrees)
        correct_flags += sum(agrees)
        print(f"{key:<18}{rate:>16.1f}%")

    overall_flag_acc = 100.0 * correct_flags / total_flags if total_flags > 0 else 0.0
    print(f"\n  Overall flag agreement: {overall_flag_acc:.1f}%  ({correct_flags}/{total_flags} flags correct)")

    # ------------------------------------------------------------------ #
    # 3. DIAGNOSIS ACCURACY                                                #
    # ------------------------------------------------------------------ #
    def mat(rs, which):
        X = np.array([[r[which].get(k, 0.0) for k in FEAT_KEYS] for r in rs], float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array([r["label"] for r in rs])
        return X, y

    Xtr, ytr = mat(train_rows, "feat_pred")
    Xte, yte = mat(test_rows,  "feat_pred")

    all_proba = []
    for s in range(args.seeds):
        clf = RandomForestClassifier(n_estimators=600, random_state=s,
                                     class_weight="balanced")
        clf.fit(Xtr, ytr)
        all_proba.append(clf.predict_proba(Xte))
    ens_proba = np.mean(all_proba, axis=0)
    ens_pred  = np.argmax(ens_proba, axis=1)

    acc = 100.0 * accuracy_score(yte, ens_pred)
    print("\n" + "="*64)
    print("DIAGNOSIS ACCURACY  (report diagnosis field, test n=50)")
    print("="*64)
    print(f"  Patient-level accuracy: {acc:.1f}%  ({int(acc*len(yte)/100)}/{len(yte)} correct)")

    cm = confusion_matrix(yte, ens_pred, labels=list(range(5)))
    print(f"\n  Per-class correct:")
    for i, name in enumerate(CLS_NAMES):
        print(f"    {name:<6}: {cm[i,i]}/10")

    # ------------------------------------------------------------------ #
    # Summary for paper table                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "="*64)
    print("PAPER TABLE SUMMARY")
    print("="*64)
    print(f"  LVEF MAE            : {metric_maes.get('LV_EF', float('nan')):.2f}%")
    print(f"  LV EDV MAE          : {metric_maes.get('LV_EDV', float('nan')):.2f} mL")
    print(f"  LV ESV MAE          : {metric_maes.get('LV_ESV', float('nan')):.2f} mL")
    print(f"  LV Mass MAE         : {metric_maes.get('LVM', float('nan')):.2f} g")
    print(f"  RV EF MAE           : {metric_maes.get('RV_EF', float('nan')):.2f}%")
    print(f"  RV EDV MAE          : {metric_maes.get('RV_EDV', float('nan')):.2f} mL")
    print(f"  Wall thickness MAE  : {metric_maes.get('wt_mean', float('nan')):.3f} mm")
    print(f"  Flag agreement      : {overall_flag_acc:.1f}%")
    print(f"  Diagnosis accuracy  : {acc:.1f}%")


if __name__ == "__main__":
    main()
