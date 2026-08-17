"""
M&Ms-2 zero-shot evaluation with PULSE 5-fold ensemble + TTA.

M&Ms-2 SA (short-axis) label convention:
  0 = background
  1 = LV (endocardium)
  2 = Myo (myocardium)
  3 = RV

PULSE/ACDC label convention:
  0 = BG, 1 = RV, 2 = Myo, 3 = LV

GT remapping applied before Dice computation:
  MnM 1 (LV)  -> PULSE 3
  MnM 2 (Myo) -> PULSE 2
  MnM 3 (RV)  -> PULSE 1
"""

import argparse, os, sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as ps
from postprocess import postprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
MNM_ROOT = Path(os.environ.get(
    "MNM_ROOT", "/SLURM/home/slurm_g202518690/student251/MnMs_Extracted/MnM2/dataset"
))


def remap_mnm_gt(gt: np.ndarray) -> np.ndarray:
    """Remap M&Ms GT labels (LV=1,Myo=2,RV=3) to PULSE/ACDC convention (RV=1,Myo=2,LV=3)."""
    out = np.zeros_like(gt)
    out[gt == 1] = 3   # LV  -> class 3
    out[gt == 2] = 2   # Myo -> class 2
    out[gt == 3] = 1   # RV  -> class 1
    return out


def zscore(vol: np.ndarray) -> np.ndarray:
    m, s = vol.mean(), vol.std()
    return (vol - m) / (s + 1e-8)


def list_mnm_patients(root: Path, max_patients: int = None):
    """List (SA_ED_img, SA_ED_gt, SA_ES_img, SA_ES_gt) tuples per patient."""
    pats = sorted(root.iterdir())
    records = []
    for p in pats:
        pid = p.name
        ed_img = p / f"{pid}_SA_ED.nii.gz"
        ed_gt  = p / f"{pid}_SA_ED_gt.nii.gz"
        es_img = p / f"{pid}_SA_ES.nii.gz"
        es_gt  = p / f"{pid}_SA_ES_gt.nii.gz"
        if ed_img.exists() and ed_gt.exists() and es_img.exists() and es_gt.exists():
            records.append((ed_img, ed_gt, es_img, es_gt))
    if max_patients:
        records = records[:max_patients]
    return records


def load_model(ckpt_path: str) -> ps.PULSESeg:
    model = ps.PULSESeg().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_volume(models, vol: np.ndarray) -> np.ndarray:
    """Predict 3D volume (H, W, D) slice-by-slice with 2.5D context, TTA, and 5-fold ensemble."""
    H, W, D = vol.shape
    # Resize each slice to IMG_SIZE x IMG_SIZE if needed
    if H != IMG_SIZE or W != IMG_SIZE:
        vol_r = np.stack([
            cv2.resize(vol[..., z].astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            for z in range(D)
        ], axis=-1)
    else:
        vol_r = vol.copy().astype(np.float32)

    vol_n = zscore(vol_r)  # volume-wise z-score
    flips = [(False, False), (True, False), (False, True), (True, True)]
    pred_vol = np.zeros((D, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for z in range(D):
        z0 = max(z - 1, 0)
        z2 = min(z + 1, D - 1)
        x0 = np.stack([vol_n[..., z0], vol_n[..., z], vol_n[..., z2]], axis=0)  # (3, H, W)

        logits_sum = None
        for model in models:
            for fh, fv in flips:
                xi = x0.copy()
                if fh: xi = xi[:, :, ::-1].copy()
                if fv: xi = xi[:, ::-1, :].copy()
                t = torch.tensor(xi[None]).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast(DEVICE.type):
                    out = model(t)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                if fh: probs = probs[:, :, ::-1].copy()
                if fv: probs = probs[:, ::-1, :].copy()
                logits_sum = probs if logits_sum is None else logits_sum + probs

        pred_vol[z] = logits_sum.argmax(0)

    # pred_vol: (D, H, W) -> (H, W, D) for postprocess
    pred_hwz = pred_vol.transpose(1, 2, 0).astype(np.int64)
    pred_hwz = postprocess(pred_hwz)
    return pred_hwz  # (H, W, D)


def dice_per_class_3d(pred: np.ndarray, gt: np.ndarray):
    """3D Dice per class on remapped GT."""
    gt_r = remap_mnm_gt(gt.astype(np.int64))
    # Resize GT to match pred if needed
    if gt_r.shape != pred.shape:
        H, W, D = pred.shape
        gt_r2 = np.stack([
            cv2.resize(gt_r[..., z].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            for z in range(D)
        ], axis=-1)
        gt_r = gt_r2

    scores = {}
    for cls, name in [(1, "RV"), (2, "Myo"), (3, "LV")]:
        p = (pred == cls)
        g = (gt_r == cls)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        scores[name] = float(2 * inter / denom) if denom > 0 else 1.0
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnm_root", default=str(MNM_ROOT))
    ap.add_argument("--folds_dir", default="folds")
    ap.add_argument("--max_patients", type=int, default=None)
    args = ap.parse_args()

    mnm_root = Path(args.mnm_root)
    records  = list_mnm_patients(mnm_root, max_patients=args.max_patients)
    print(f"M&Ms SA patients found: {len(records)}")
    if not records:
        raise RuntimeError(f"No M&Ms SA data at {mnm_root}")

    # Load all 5 fold models
    models = []
    for fold in range(5):
        ckpt = Path(args.folds_dir) / f"fold_{fold}" / "best_model.pth"
        if ckpt.exists():
            models.append(load_model(str(ckpt)))
            print(f"  Loaded fold {fold}")

    if not models:
        raise RuntimeError("No fold checkpoints found")

    all_dice = []
    for i, (ed_img_p, ed_gt_p, es_img_p, es_gt_p) in enumerate(records):
        pat_dice = []
        for img_p, gt_p in [(ed_img_p, ed_gt_p), (es_img_p, es_gt_p)]:
            vol = nib.load(str(img_p)).get_fdata().astype(np.float32)  # (H, W, D)
            gt  = nib.load(str(gt_p)).get_fdata().astype(np.int64)
            pred = predict_volume(models, vol)
            d = dice_per_class_3d(pred, gt)
            pat_dice.append(d)
        avg = {k: np.mean([d[k] for d in pat_dice]) for k in pat_dice[0]}
        all_dice.append(avg)
        if (i + 1) % 20 == 0:
            rv  = np.mean([d["RV"]  for d in all_dice])
            myo = np.mean([d["Myo"] for d in all_dice])
            lv  = np.mean([d["LV"]  for d in all_dice])
            print(f"  [{i+1}/{len(records)}] RV={rv:.3f} Myo={myo:.3f} LV={lv:.3f}")

    rv_mean  = np.mean([d["RV"]  for d in all_dice])
    myo_mean = np.mean([d["Myo"] for d in all_dice])
    lv_mean  = np.mean([d["LV"]  for d in all_dice])
    mean_all = (rv_mean + myo_mean + lv_mean) / 3

    print("\n=== M&Ms SA ZERO-SHOT RESULTS (5-fold ensemble + TTA + postprocess) ===")
    print(f"Patients: {len(all_dice)}")
    print(f"  RV  Dice: {rv_mean:.1%}")
    print(f"  Myo Dice: {myo_mean:.1%}")
    print(f"  LV  Dice: {lv_mean:.1%}")
    print(f"  Mean:     {mean_all:.1%}")


if __name__ == "__main__":
    main()
