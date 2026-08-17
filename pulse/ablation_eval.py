#!/usr/bin/env python3
"""
PULSE Ablation Evaluation — single-fold, 3D patient-level Dice on ACDC test set.
Mirrors the per-fold path in ensemble_eval.py so numbers are directly comparable
to the full-model fold-0 result (~87.0% mean Dice).

Also evaluates fold-0 WITHOUT TTA to measure TTA's contribution.

Usage (after ablation SLURM jobs finish):
  python ablation_eval.py [--data_root /path/to/ACDC/database]
"""
import argparse, os, sys, warnings
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
    bd = ck.get("best_dice", "?")
    print(f"  loaded {ckpt_path}  best_dice={bd:.4f}" if isinstance(bd, float)
          else f"  loaded {ckpt_path}  best_dice={bd}")
    return m


@torch.no_grad()
def predict_volume(model: torch.nn.Module, img_path: Path,
                   use_tta: bool = True) -> np.ndarray:
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    pred_vol = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z - 1, 0), z, min(z + 1, S - 1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r  = [cv2.resize(s, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
              for s in sl]
        x0 = (torch.from_numpy(np.stack(r, -1).transpose(2, 0, 1))
              .float().unsqueeze(0).to(DEVICE))
        if use_tta:
            probs = []
            for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
                xi = x0.clone()
                if fh: xi = xi.flip(-1)
                if fv: xi = xi.flip(-2)
                with torch.amp.autocast(DEVICE.type):
                    out = model(xi)
                    main = out[0] if isinstance(out, tuple) else out
                p = F.softmax(main, 1)
                if fh: p = p.flip(-1)
                if fv: p = p.flip(-2)
                probs.append(p)
            prob = torch.stack(probs).mean(0)
        else:
            with torch.amp.autocast(DEVICE.type):
                out = model(x0)
                main = out[0] if isinstance(out, tuple) else out
            prob = F.softmax(main, 1)

        pred_slice = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
        pred_vol[..., z] = cv2.resize(pred_slice, (W, H),
                                      interpolation=cv2.INTER_NEAREST)
    return postprocess(pred_vol)


def dice3d(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p = (pred == c); g = (gt == c)
    inter = (p & g).sum(); denom = p.sum() + g.sum()
    return 0.0 if denom == 0 else float(2 * inter / denom)


def eval_model(model: torch.nn.Module, patients: dict,
               use_tta: bool = True) -> dict:
    per_class = {n: [] for n in ("RV", "Myo", "LV")}
    for pid, rec in sorted(patients.items()):
        pat_dices = {n: [] for n in ("RV", "Myo", "LV")}
        for phase, info in rec["phases"].items():
            pred = predict_volume(model, info["img"], use_tta=use_tta)
            gt   = nib.load(str(info["gt"])).get_fdata().astype(np.int64)
            for c, n in ((1, "RV"), (2, "Myo"), (3, "LV")):
                pat_dices[n].append(dice3d(pred, gt, c))
        for n in ("RV", "Myo", "LV"):
            if pat_dices[n]:
                per_class[n].append(float(np.mean(pat_dices[n])))
    means = {n: 100.0 * np.mean(v) for n, v in per_class.items() if v}
    means["Mean"] = float(np.mean(list(means.values())))
    return means


def run(label: str, ckpt_path: Path, use_tta: bool = True) -> dict | None:
    if not ckpt_path.exists():
        print(f"  [{label}] checkpoint not found: {ckpt_path} → SKIPPED")
        return None
    model    = load_model(ckpt_path)
    patients = list_patients(TEST_ROOT)
    print(f"  [{label}] {len(patients)} test patients, TTA={use_tta} ...")
    result = eval_model(model, patients, use_tta=use_tta)
    print(f"  → RV={result['RV']:.1f}  Myo={result['Myo']:.1f}  "
          f"LV={result['LV']:.1f}  Mean={result['Mean']:.1f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None,
                    help="Path to ACDC database/ dir (contains training/ and testing/)."
                         " Overrides ACDC_ROOT env var.")
    ap.add_argument("--folds_dir", default=None,
                    help="Directory containing fold_0..fold_4 checkpoints (default: <script_dir>/folds)")
    ap.add_argument("--ablation_dir", default=None,
                    help="Directory containing no_ds/no_bnd/no_curriculum checkpoints "
                         "(default: <script_dir>/ablation_ckpts)")
    args = ap.parse_args()

    BASE = Path(__file__).parent
    folds_dir    = Path(args.folds_dir)    if args.folds_dir    else BASE / "folds"
    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else BASE / "ablation_ckpts"

    if args.data_root:
        import diagnosis_features as _df
        _df.TEST_ROOT = Path(args.data_root) / "testing"
        from diagnosis_features import TEST_ROOT  # re-import updated value

    CONFIGS = [
        # (label,                   checkpoint path,                                    TTA)
        ("Full model + TTA",       folds_dir / "fold_0/best_model.pth",                True ),
        ("Full model, no TTA",     folds_dir / "fold_0/best_model.pth",                False),
        ("No deep supervision",    ablation_dir / "no_ds/best_model.pth",              True ),
        ("No boundary loss",       ablation_dir / "no_bnd/best_model.pth",             True ),
        ("No 2-stage curriculum",  ablation_dir / "no_curriculum/best_model.pth",      True ),
    ]

    print("\n" + "=" * 65)
    print("PULSE ABLATION  —  fold 0, 3D patient-level Dice, ACDC test set")
    print("=" * 65)

    rows = []
    for label, ckpt, tta in CONFIGS:
        r = run(label, ckpt, use_tta=tta)
        rows.append((label, r))

    print("\n")
    hdr = f"{'Configuration':<30} {'RV':>6} {'Myo':>6} {'LV':>6} {'Mean':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows:
        if r:
            print(f"{label:<30} {r['RV']:>5.1f}% {r['Myo']:>5.1f}% "
                  f"{r['LV']:>5.1f}% {r['Mean']:>6.1f}%")
        else:
            print(f"{label:<30} {'---':>6} {'---':>6} {'---':>6} {'---':>7}")
    print("=" * 65)
    print("\n→ Copy these numbers into Table IX (new-component ablation) in main.tex.")
