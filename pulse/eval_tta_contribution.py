#!/usr/bin/env python3
"""
TTA contribution eval: compares PULSE fold-0 with TTA vs without TTA.
Uses same 3D patient-level Dice as all other evals.
"""
import argparse, sys, warnings
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


def load_model(ckpt_path: Path):
    m = PS.PULSESeg(img_size=IMG_SIZE).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    m.load_state_dict(state)
    m.eval()
    print(f"  loaded {ckpt_path.name}  best_dice={ck.get('best_dice', '?')}")
    return m


@torch.no_grad()
def predict_volume(model, img_path: Path, use_tta=True) -> np.ndarray:
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


def evaluate(label, ckpt_path, use_tta):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  [{label}] SKIPPED — checkpoint not found"); return None
    print(f"\n[{label}]")
    model = load_model(ckpt_path)
    patients = _df.list_patients(_df.TEST_ROOT)
    print(f"  {len(patients)} test patients, TTA={use_tta}")
    per_class = {n: [] for n in ("RV", "Myo", "LV")}
    for pid, rec in sorted(patients.items()):
        pat = {n: [] for n in ("RV", "Myo", "LV")}
        for phase, info in rec["phases"].items():
            pred = predict_volume(model, info["img"], use_tta=use_tta)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1,"RV"), (2,"Myo"), (3,"LV")):
                pat[n].append(dice3d(pred, gt, c))
        for n in ("RV", "Myo", "LV"):
            if pat[n]: per_class[n].append(float(np.mean(pat[n])))
    rv  = 100 * np.mean(per_class["RV"])
    myo = 100 * np.mean(per_class["Myo"])
    lv  = 100 * np.mean(per_class["LV"])
    mn  = (rv + myo + lv) / 3
    print(f"  RESULT: RV={rv:.1f}  Myo={myo:.1f}  LV={lv:.1f}  Mean={mn:.1f}")
    return {"RV": rv, "Myo": myo, "LV": lv, "Mean": mn}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--folds_dir", default=None)
    args = ap.parse_args()

    BASE = Path(__file__).parent
    fold0 = (Path(args.folds_dir) if args.folds_dir else BASE / "folds") / "fold_0/best_model.pth"
    _df.TEST_ROOT = Path(args.data_root) / "testing"

    print("=" * 65)
    print("TTA CONTRIBUTION — PULSE fold-0, 3D patient-level Dice")
    print("=" * 65)

    configs = [
        ("PULSE fold-0 + TTA",    fold0, True),
        ("PULSE fold-0 (no TTA)", fold0, False),
    ]

    results = {}
    for label, ckpt, tta in configs:
        r = evaluate(label, ckpt, tta)
        if r: results[label] = r

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    hdr = f"{'Configuration':<30} {'RV':>6} {'Myo':>6} {'LV':>6} {'Mean':>7}"
    print(hdr); print("-" * len(hdr))
    for label, r in results.items():
        print(f"{label:<30} {r['RV']:>5.1f}% {r['Myo']:>5.1f}% {r['LV']:>5.1f}% {r['Mean']:>6.1f}%")
    if len(results) == 2:
        vals = list(results.values())
        delta = vals[0]["Mean"] - vals[1]["Mean"]
        print(f"\nTTA gain: +{delta:.1f}%")
    print("=" * 65)
