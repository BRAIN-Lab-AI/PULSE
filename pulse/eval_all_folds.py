#!/usr/bin/env python3
"""Evaluate each fold individually on the ACDC test set (3D patient-level Dice)."""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

import pulse_seg as PS
from postprocess import postprocess

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256


@torch.no_grad()
def predict_volume(model, img_path, use_tta=True):
    from diagnosis_features import zscore
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    pred_vol = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z-1, 0), z, min(z+1, S-1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r  = [cv2.resize(s, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
              for s in sl]
        x0 = torch.from_numpy(np.stack(r,-1).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE)
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
        ps = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
        pred_vol[..., z] = cv2.resize(ps, (W, H), interpolation=cv2.INTER_NEAREST)
    return postprocess(pred_vol)


def dice3d(pred, gt, c):
    p = (pred==c); g = (gt==c)
    inter = (p&g).sum(); denom = p.sum()+g.sum()
    return 0.0 if denom==0 else float(2*inter/denom)


def eval_fold(fold_ckpt, patients):
    m = PS.PULSESeg(img_size=IMG_SIZE).to(DEVICE)
    ck = torch.load(fold_ckpt, map_location=DEVICE)
    m.load_state_dict(ck["model"] if "model" in ck else ck); m.eval()
    per_cls = {n: [] for n in ("RV", "Myo", "LV")}
    for pid, rec in sorted(patients.items()):
        pat_d = {n: [] for n in ("RV", "Myo", "LV")}
        for phase, info in rec["phases"].items():
            pred = predict_volume(m, info["img"], use_tta=True)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1,"RV"),(2,"Myo"),(3,"LV")):
                pat_d[n].append(dice3d(pred, gt, c))
        for n in ("RV", "Myo", "LV"):
            if pat_d[n]:
                per_cls[n].append(float(np.mean(pat_d[n])))
    return {n: 100 * np.mean(v) if v else float("nan")
            for n, v in per_cls.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None,
                    help="Path to ACDC database/ dir. Overrides ACDC_ROOT env var.")
    ap.add_argument("--folds_dir", default=None,
                    help="Directory containing fold_0..fold_4 checkpoints.")
    args = ap.parse_args()

    import diagnosis_features as _df
    if args.data_root:
        _df.TEST_ROOT = Path(args.data_root) / "testing"
    from diagnosis_features import list_patients, TEST_ROOT

    patients = list_patients(TEST_ROOT)
    if not patients:
        raise RuntimeError(
            f"No test patients found at {TEST_ROOT}. "
            "Set --data_root or ACDC_ROOT env var."
        )

    BASE = Path(__file__).parent
    folds_dir = Path(args.folds_dir) if args.folds_dir else BASE / "folds"

    print("\n" + "="*65)
    print("Per-fold test evaluation (3D patient-level Dice, ACDC test set)")
    print(f"Test root: {TEST_ROOT}  |  {len(patients)} patients")
    print("="*65)
    print(f"{'Fold':<8} {'RV':>6} {'Myo':>6} {'LV':>6} {'Mean':>7}")
    print("-"*35)

    all_rv, all_myo, all_lv = [], [], []
    for f in range(5):
        ck = folds_dir / f"fold_{f}/best_model.pth"
        if not ck.exists():
            print(f"Fold {f}   MISSING"); continue
        r = eval_fold(ck, patients)
        mean = float(np.nanmean([r["RV"], r["Myo"], r["LV"]]))
        all_rv.append(r["RV"]); all_myo.append(r["Myo"]); all_lv.append(r["LV"])
        print(f"Fold {f}   {r['RV']:>5.1f}% {r['Myo']:>5.1f}% "
              f"{r['LV']:>5.1f}% {mean:>6.1f}%")

    print("-"*35)
    if all_rv:
        mean_all = float(np.nanmean([np.nanmean([rv,my,lv])
                                     for rv,my,lv in zip(all_rv,all_myo,all_lv)]))
        print(f"{'Mean':<8} {np.nanmean(all_rv):>5.1f}% "
              f"{np.nanmean(all_myo):>5.1f}% {np.nanmean(all_lv):>5.1f}% "
              f"{mean_all:>6.1f}%")
        print(f"Std       {np.nanstd(all_rv):>5.1f}  "
              f"{np.nanstd(all_myo):>5.1f}  {np.nanstd(all_lv):>5.1f}")
    print("="*65)
    print("\n→ Copy mean±std into the 5-fold CV table in main.tex.")


if __name__ == "__main__":
    main()
