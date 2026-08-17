"""
Sunnybrook zero-shot LV Dice evaluation with PULSE 5-fold ensemble + TTA.

Sunnybrook .mat label convention (1-based MATLAB indexing, NO label 0):
  label 1 = background  (~97-99% of image pixels, everything outside endo)
  label 2 = LV endocardial cavity (~1-3% of image, inside endo contour)
  (Myo ring is NOT separately annotated in the standard Sunnybrook release)

We stack all labeled slices per patient into a 3D volume, apply volume-wise
z-score, and use 2.5D context (adjacent spatial slices) for inference.

PULSE class 3 = LV. We compare pred==3 to gt==2 (LV endo cavity).
"""

import argparse, os, sys
import numpy as np
import scipy.io as sio
import pydicom
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as ps
from postprocess import postprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

SB_ROOT = Path(os.environ.get(
    "SB_ROOT",
    "/SLURM/home/slurm_g202518690/student251/SUNNYBROOK_EXTRACTED/Cardiac MRI"
))


def zscore_volume(vol: np.ndarray) -> np.ndarray:
    """Volume-wise z-score as in PULSE training."""
    m, s = vol.mean(), vol.std()
    return (vol - m) / (s + 1e-8)


def load_sb_patient_volume(img_dir: Path, lbl_dir: Path):
    """Load all slices for a patient, returning (vol, labels_vol).

    vol        : (H, W, D) float32 z-scored
    labels_vol : (H, W, D) int64,  values {1, 2} — 1=background, 2=LV endo cavity
                 (no label 0; Myo is NOT annotated in Sunnybrook)
    """
    dcm_files = sorted(img_dir.glob("*.dcm"),
                        key=lambda f: int(f.stem.split("rawdcm_")[-1]))
    slices, gt_slices = [], []
    for dcm_f in dcm_files:
        frame_id = dcm_f.stem.split("rawdcm_")[-1].zfill(4)
        mat_f    = lbl_dir / f"{lbl_dir.name}gtmask{frame_id}.mat"
        if not mat_f.exists():
            continue
        ds  = pydicom.dcmread(str(dcm_f))
        raw = ds.pixel_array.astype(np.float32)
        raw = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mat = sio.loadmat(str(mat_f))["labels"].astype(np.int64)
        mat = cv2.resize(mat.astype(np.float32), (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST).astype(np.int64)
        slices.append(raw)
        gt_slices.append(mat)

    if not slices:
        return None, None

    vol = np.stack(slices, axis=-1)          # (H, W, D)
    vol = zscore_volume(vol)
    gt  = np.stack(gt_slices, axis=-1)       # (H, W, D)
    return vol, gt


def load_model(ckpt_path: str) -> ps.PULSESeg:
    model = ps.PULSESeg().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_volume(models, vol: np.ndarray) -> np.ndarray:
    """Predict 3D volume (H, W, D) with 2.5D context, TTA, 5-fold ensemble."""
    H, W, D = vol.shape
    flips = [(False, False), (True, False), (False, True), (True, True)]

    pred_vol = np.zeros((H, W, D), dtype=np.int64)
    for z in range(D):
        z0 = max(z - 1, 0)
        z2 = min(z + 1, D - 1)
        x0 = np.stack([vol[..., z0], vol[..., z], vol[..., z2]], axis=0)  # (3, H, W)
        logits_sum = None
        for model in models:
            for fh, fv in flips:
                xi = x0.copy()
                if fh: xi = xi[:, :, ::-1].copy()
                if fv: xi = xi[:, ::-1, :].copy()
                t = torch.tensor(xi[None]).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast(DEVICE.type):
                    out    = model(t)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                if fh: probs = probs[:, :, ::-1].copy()
                if fv: probs = probs[:, ::-1, :].copy()
                logits_sum = probs if logits_sum is None else logits_sum + probs
        pred_vol[..., z] = logits_sum.argmax(0)

    return postprocess(pred_vol)


def lv_dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice of PULSE class 3 (LV) vs Sunnybrook label 2 (LV endo cavity).

    Sunnybrook .mat labels use 1-based indexing (no label 0):
      label 1 = background (everything outside endo, ~97-99% of image)
      label 2 = LV endocardial cavity (~1-3% of image, inside endo contour)
    """
    p = (pred == 3)
    g = (gt   == 2)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_root",   default=str(SB_ROOT))
    ap.add_argument("--folds_dir", default="folds")
    args = ap.parse_args()

    sb_root  = Path(args.sb_root)
    img_root = sb_root / "images"
    lbl_root = sb_root / "labels"

    patient_ids = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    print(f"Sunnybrook patients: {len(patient_ids)}")

    models = []
    for fold in range(5):
        ckpt = Path(args.folds_dir) / f"fold_{fold}" / "best_model.pth"
        if ckpt.exists():
            models.append(load_model(str(ckpt)))
            print(f"  Loaded fold {fold}")
    if not models:
        raise RuntimeError("No fold checkpoints found")

    all_dice = []
    pred_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for pid in patient_ids:
        vol, gt = load_sb_patient_volume(img_root / pid, lbl_root / pid)
        if vol is None:
            print(f"  Skipping {pid}: no matching label files")
            continue
        pred = predict_volume(models, vol)
        for c in range(4):
            pred_class_counts[c] += (pred == c).sum()
        d    = lv_dice_3d(pred, gt)
        all_dice.append(d)
        lv_pred_px = (pred == 3).sum()
        lv_gt_px   = (gt == 2).sum()
        print(f"  {pid}: LV Dice = {d:.3f}  (pred_LV={lv_pred_px}px, gt_LV={lv_gt_px}px)")

    total_pred = sum(pred_class_counts.values())
    print(f"\nPrediction class distribution (all patients):")
    for c, name in [(0, "BG"), (1, "RV"), (2, "Myo"), (3, "LV")]:
        print(f"  class {c} ({name}): {pred_class_counts[c]/total_pred:.1%}")

    lv_mean = float(np.mean(all_dice))
    lv_std  = float(np.std(all_dice))
    print(f"\n=== SUNNYBROOK ZERO-SHOT LV RESULTS (5-fold ensemble + TTA + postprocess) ===")
    print(f"Patients: {len(all_dice)}")
    print(f"  LV Dice: {lv_mean:.3f} ± {lv_std:.3f}  ({lv_mean:.1%})")


if __name__ == "__main__":
    main()
