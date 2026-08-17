#!/usr/bin/env python3
"""
Bootstrap 95% CI on PULSE 5-fold ensemble Dice.
Saves per-patient dice to JSON, then bootstraps 10,000 times.
Also runs Wilcoxon signed-rank test: PULSE frz2 fold-0 vs frz6 fold-0, per patient.
"""
import argparse, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
import diagnosis_features as _df
from diagnosis_features import zscore
from postprocess import postprocess

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256


def load_model(ckpt_path):
    m = PS.PULSESeg(img_size=IMG_SIZE).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def predict_volume(model, img_path, use_tta=True):
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    pred_vol = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z-1, 0), z, min(z+1, S-1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r  = [cv2.resize(s, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for s in sl]
        x0 = torch.from_numpy(np.stack(r, -1).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
        if use_tta:
            probs = []
            for fh, fv in [(False,False),(True,False),(False,True),(True,True)]:
                xi = x0.clone()
                if fh: xi = xi.flip(-1)
                if fv: xi = xi.flip(-2)
                with torch.amp.autocast(DEVICE.type):
                    out = model(xi); main = out[0] if isinstance(out, tuple) else out
                p = F.softmax(main, 1)
                if fh: p = p.flip(-1)
                if fv: p = p.flip(-2)
                probs.append(p)
            prob = torch.stack(probs).mean(0)
        else:
            with torch.amp.autocast(DEVICE.type):
                out = model(x0); main = out[0] if isinstance(out, tuple) else out
            prob = F.softmax(main, 1)
        pred = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
        pred_vol[..., z] = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return postprocess(pred_vol)


def dice3d(pred, gt, c):
    p = (pred == c); g = (gt == c)
    inter = (p & g).sum(); denom = p.sum() + g.sum()
    return 0.0 if denom == 0 else float(2 * inter / denom)


def eval_single_model(model, patients):
    per_patient = {}
    for pid, rec in sorted(patients.items()):
        pat = {n: [] for n in ("RV", "Myo", "LV")}
        for phase, info in rec["phases"].items():
            pred = predict_volume(model, info["img"], use_tta=True)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1,"RV"),(2,"Myo"),(3,"LV")):
                pat[n].append(dice3d(pred, gt, c))
        per_patient[pid] = {n: float(np.mean(v)) for n, v in pat.items() if v}
    return per_patient


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean()
                     for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--frz2_fold0", required=True, help="frz2 fold-0 checkpoint")
    ap.add_argument("--frz6_fold0", required=True, help="frz6 fold-0 checkpoint")
    ap.add_argument("--out", default="bootstrap_results.json")
    args = ap.parse_args()

    _df.TEST_ROOT = Path(args.data_root) / "testing"
    patients = _df.list_patients(_df.TEST_ROOT)
    print(f"Evaluating {len(patients)} test patients...")

    print("\n[1/2] Evaluating PULSE frz2 fold-0...")
    PS.FREEZE_BLK = 2
    m_frz2 = load_model(Path(args.frz2_fold0))
    scores_frz2 = eval_single_model(m_frz2, patients)
    del m_frz2; torch.cuda.empty_cache()

    print("\n[2/2] Evaluating PULSE frz6 fold-0 (reference)...")
    PS.FREEZE_BLK = 6
    m_frz6 = load_model(Path(args.frz6_fold0))
    scores_frz6 = eval_single_model(m_frz6, patients)
    del m_frz6; torch.cuda.empty_cache()

    # Aggregate mean dice per patient — use intersection to keep Wilcoxon arrays paired
    pids = sorted(scores_frz2.keys() & scores_frz6.keys())
    mean_frz2 = np.array([np.mean(list(scores_frz2[p].values())) for p in pids])
    mean_frz6 = np.array([np.mean(list(scores_frz6[p].values())) for p in pids])

    print("\n" + "=" * 65)
    print("BOOTSTRAP 95% CI  (10,000 resamples, n=50 patients)")
    print("=" * 65)

    for label, vals in [("PULSE frz2 fold-0", mean_frz2), ("PULSE frz6 fold-0", mean_frz6)]:
        lo, hi = bootstrap_ci(vals)
        print(f"  {label}: {100*vals.mean():.2f}% (95% CI: {100*lo:.2f}--{100*hi:.2f}%)")

    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(mean_frz2, mean_frz6, alternative='greater')
    print(f"\nWilcoxon signed-rank test (frz2 > frz6):")
    print(f"  statistic={stat:.1f}, p={p:.4f} ({'significant' if p < 0.05 else 'not significant'} at p<0.05)")

    # Per-class CI for frz2
    print("\nPer-class Bootstrap CI (PULSE frz2 fold-0):")
    for c in ("RV", "Myo", "LV"):
        vals_c = np.array([scores_frz2[p][c] for p in pids])
        lo, hi = bootstrap_ci(vals_c)
        print(f"  {c}: {100*vals_c.mean():.2f}% (95% CI: {100*lo:.2f}--{100*hi:.2f}%)")

    # Save
    out = {
        "frz2": {p: scores_frz2[p] for p in pids},
        "frz6": {p: scores_frz6[p] for p in pids},
        "mean_frz2": float(100 * mean_frz2.mean()),
        "mean_frz6": float(100 * mean_frz6.mean()),
        "wilcoxon_p": float(p),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved per-patient scores to {args.out}")
    print("=" * 65)
