#!/usr/bin/env python3
"""
PULSE Ensemble Size Ablation — evaluates ACDC test Dice with 1..5 fold models.
Shows how performance improves as more fold checkpoints are averaged.
All inference uses 4-way TTA (horizontal flip, vertical flip, both, neither).

Usage:
  python ablation_ensemble_size.py [--data_root /path/to/ACDC/database]

Output: Table for "Ensemble size and TTA contribution" ablation.
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
from diagnosis_features import list_patients, zscore, TEST_ROOT
from postprocess import postprocess

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256


def load_model(ckpt_path: Path) -> torch.nn.Module:
    m = PS.PULSESeg(img_size=IMG_SIZE).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    state = ck["model"] if "model" in ck else ck
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def predict_volume_ensemble(models, img_path: Path, use_tta: bool = True) -> np.ndarray:
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    pred_vol = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z - 1, 0), z, min(z + 1, S - 1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r  = [cv2.resize(s, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for s in sl]
        x0 = torch.from_numpy(np.stack(r, -1).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

        all_probs = []
        for m in models:
            if use_tta:
                for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
                    xi = x0.clone()
                    if fh: xi = xi.flip(-1)
                    if fv: xi = xi.flip(-2)
                    with torch.amp.autocast(DEVICE.type):
                        out = m(xi)
                        main = out[0] if isinstance(out, tuple) else out
                    p = F.softmax(main, 1)
                    if fh: p = p.flip(-1)
                    if fv: p = p.flip(-2)
                    all_probs.append(p)
            else:
                with torch.amp.autocast(DEVICE.type):
                    out = m(x0)
                    main = out[0] if isinstance(out, tuple) else out
                all_probs.append(F.softmax(main, 1))

        prob = torch.stack(all_probs).mean(0)
        pred_slice = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
        pred_vol[..., z] = cv2.resize(pred_slice, (W, H), interpolation=cv2.INTER_NEAREST)
    return postprocess(pred_vol)


def dice3d(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p = (pred == c); g = (gt == c)
    inter = (p & g).sum(); denom = p.sum() + g.sum()
    return 0.0 if denom == 0 else float(2 * inter / denom)


def eval_ensemble(models, patients: dict, use_tta: bool = True) -> dict:
    per_class = {n: [] for n in ("RV", "Myo", "LV")}
    for pid, rec in sorted(patients.items()):
        pat_dices = {n: [] for n in ("RV", "Myo", "LV")}
        for phase, info in rec["phases"].items():
            pred = predict_volume_ensemble(models, info["img"], use_tta=use_tta)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1, "RV"), (2, "Myo"), (3, "LV")):
                pat_dices[n].append(dice3d(pred, gt, c))
        for n in ("RV", "Myo", "LV"):
            if pat_dices[n]:
                per_class[n].append(float(np.mean(pat_dices[n])))
    means = {n: 100.0 * np.mean(v) for n, v in per_class.items() if v}
    means["Mean"] = float(np.mean(list(means.values())))
    return means


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--folds_dir", default=None)
    args = ap.parse_args()

    BASE      = Path(__file__).parent
    folds_dir = Path(args.folds_dir) if args.folds_dir else BASE / "folds"

    if args.data_root:
        import diagnosis_features as _df
        _df.TEST_ROOT = Path(args.data_root) / "testing"
        from diagnosis_features import TEST_ROOT

    # Load all 5 fold checkpoints in order
    fold_ckpts = [folds_dir / f"fold_{i}/best_model.pth" for i in range(5)]
    fold_models = []
    for ck in fold_ckpts:
        print(f"  loading {ck}")
        fold_models.append(load_model(ck))

    patients = list_patients(TEST_ROOT)
    print(f"\n  {len(patients)} test patients\n")

    print("=" * 70)
    print("PULSE Ensemble Size Ablation — ACDC test set (50 patients)")
    print("=" * 70)

    rows = []

    # Single model, no TTA (fold 0)
    print("  [1 fold, no TTA] evaluating...")
    r = eval_ensemble([fold_models[0]], patients, use_tta=False)
    rows.append(("1 fold, no TTA", r))
    print(f"  → RV={r['RV']:.1f}  Myo={r['Myo']:.1f}  LV={r['LV']:.1f}  Mean={r['Mean']:.1f}")

    # Single model, with TTA (fold 0)
    print("  [1 fold, +TTA] evaluating...")
    r = eval_ensemble([fold_models[0]], patients, use_tta=True)
    rows.append(("1 fold, +TTA", r))
    print(f"  → RV={r['RV']:.1f}  Myo={r['Myo']:.1f}  LV={r['LV']:.1f}  Mean={r['Mean']:.1f}")

    # 2-fold ensemble
    for n_folds in [2, 3, 4, 5]:
        label = f"{n_folds}-fold ensemble, +TTA"
        print(f"  [{label}] evaluating...")
        r = eval_ensemble(fold_models[:n_folds], patients, use_tta=True)
        rows.append((label, r))
        print(f"  → RV={r['RV']:.1f}  Myo={r['Myo']:.1f}  LV={r['LV']:.1f}  Mean={r['Mean']:.1f}")

    print()
    hdr = f"{'Configuration':<30} {'RV':>6} {'Myo':>6} {'LV':>6} {'Mean':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows:
        print(f"{label:<30} {r['RV']:>5.1f}% {r['Myo']:>5.1f}% {r['LV']:>5.1f}% {r['Mean']:>6.1f}%")
    print("=" * 70)
    print("\n→ Copy these into the ensemble-size ablation table in main.tex.")
