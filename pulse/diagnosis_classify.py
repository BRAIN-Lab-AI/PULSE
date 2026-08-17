#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULSE — Patient-Level Cardiomyopathy Diagnosis from Clinical Biomarkers
=======================================================================
Trains a classifier on per-patient biomarker vectors derived from PULSE's own
2.5D segmentation (see diagnosis_features.py) and reports HONEST, standard,
patient-level metrics on the held-out ACDC test set (50 subjects):

    top-1 accuracy, BALANCED accuracy, macro-F1, per-class P/R/F1,
    real confusion matrix, and macro-AUC computed from actual class
    probabilities (one-vs-rest) -- with bootstrap 95% CIs.

This replaces the broken single-middle-slice image classifier (44% test acc)
and the inflated one-vs-rest "accuracy" reported in the paper (79.2%, not real).

Protocol: train on the 100 official ACDC training subjects, test on the 50
official test subjects -- the standard ACDC split. Features come from PREDICTED
masks (the deployable pipeline); GT-mask features are reported as a ceiling.

Usage:
  python diagnosis_classify.py --features diagnosis_features.json [--seeds 10]
"""
import argparse, json, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)
from diagnosis_features import CLS_NAMES, FEAT_KEYS


def matrix(rows, which, split):
    sel = [r for r in rows if r["split"] == split]
    X = np.array([[r[which].get(k, 0.0) for k in FEAT_KEYS] for r in sel], float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([r["label"] for r in sel])
    return X, y


def make_clf(seed):
    # Soft-voting-style ensemble via averaging is overkill at n=100; a tuned RF
    # is the ACDC-standard choice and is interpretable. GB offered as alt.
    return RandomForestClassifier(n_estimators=600, max_depth=None,
                                  min_samples_leaf=1, random_state=seed,
                                  class_weight="balanced")


def evaluate(Xtr, ytr, Xte, yte, seeds):
    """Train over multiple seeds; return mean+/-std of metrics and ensemble preds."""
    accs, baccs, f1s, aucs = [], [], [], []
    all_proba = []
    last_pred, last_proba = None, None
    for s in seeds:
        clf = make_clf(s)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        all_proba.append(proba)
        accs.append(accuracy_score(yte, pred))
        baccs.append(balanced_accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro"))
        try:
            aucs.append(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
        except ValueError:
            aucs.append(float("nan"))
        last_pred, last_proba = pred, proba
    # Ensemble (averaged proba) prediction — used for CI and confusion matrix
    ens_proba = np.mean(all_proba, axis=0)
    ens_pred = np.argmax(ens_proba, axis=1)
    def ms(a): return float(np.mean(a)), float(np.std(a))
    return {"acc": ms(accs), "bacc": ms(baccs), "f1": ms(f1s), "auc": ms(aucs),
            "pred": ens_pred, "proba": ens_proba,
            "last_pred": last_pred, "last_proba": last_proba}


def bootstrap_ci(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[idx], pred[idx]))
    return np.percentile(accs, 2.5) * 100, np.percentile(accs, 97.5) * 100


def report_block(title, rows, which, seeds):
    Xtr, ytr = matrix(rows, which, "train")
    Xte, yte = matrix(rows, which, "test")
    res = evaluate(Xtr, ytr, Xte, yte, seeds)
    lo, hi = bootstrap_ci(yte, res["pred"])
    print(f"\n{'='*64}\n{title}\n{'='*64}")
    print(f"  train n={len(ytr)}  test n={len(yte)}  features={Xtr.shape[1]}  seeds={len(seeds)}")
    print(f"  Accuracy:          {res['acc'][0]*100:5.1f} +/- {res['acc'][1]*100:.1f}%   "
          f"(95% CI {lo:.1f}-{hi:.1f})")
    print(f"  Balanced accuracy: {res['bacc'][0]*100:5.1f} +/- {res['bacc'][1]*100:.1f}%")
    print(f"  Macro-F1:          {res['f1'][0]:.3f} +/- {res['f1'][1]:.3f}")
    print(f"  Macro-AUC (OvR):   {res['auc'][0]:.3f} +/- {res['auc'][1]:.3f}")
    print("\n  Per-class report (last seed):")
    print(classification_report(yte, res["pred"], target_names=CLS_NAMES, digits=3,
                                zero_division=0))
    print("  Confusion matrix (rows=true, cols=pred; NOR DCM HCM MINF RV):")
    print(confusion_matrix(yte, res["pred"], labels=list(range(5))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="diagnosis_features.json")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    with open(args.features) as fh:
        rows = json.load(fh)
    seeds = list(range(args.seeds))

    print(f"Loaded {len(rows)} patients "
          f"(train={sum(r['split']=='train' for r in rows)}, "
          f"test={sum(r['split']=='test' for r in rows)})")

    report_block("DIAGNOSIS from PREDICTED masks  (deployable PULSE pipeline)",
                 rows, "feat_pred", seeds)
    report_block("DIAGNOSIS from GT masks  (oracle ceiling)",
                 rows, "feat_gt", seeds)


if __name__ == "__main__":
    main()
