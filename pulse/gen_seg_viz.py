#!/usr/bin/env python3
"""
Generate qualitative segmentation visualizations using vanilla DINOv2 5-fold ensemble.
Produces DCMdeted.png, DCMdetes.png, HCMdeted.png, HCMdetes.png,
MINFdeted.png, MINFdetes.png, NORdeted.png, NORdetes.png,
RVdeted.png, RVdetes.png in paper_contents/.

Layout: horizontal strip of ~6 mid-cavity slices from a representative
patient, each panel showing input MRI with GT contour outline + filled
prediction overlay. Matches ~1900×460 px RGBA format of existing figures.
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
PS.BACKBONE = "facebook/dinov2-base"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BASE     = Path(__file__).parent
FOLDS    = BASE / "folds_vdino"
OUT      = BASE / "paper_contents"
DATA     = Path("data/ACDC/database/testing")

# Colors: RV=blue, Myo=green, LV=red  (RGB, 0-1)
COLORS = {1: (0.30, 0.55, 1.00),   # RV  — steel blue
          2: (0.20, 0.85, 0.35),   # Myo — green
          3: (1.00, 0.30, 0.25)}   # LV  — red

# Representative patient per class (first good patient in test set)
CLASS_PATIENTS = {
    "DCM":  "patient101",
    "HCM":  "patient104",
    "MINF": "patient103",
    "NOR":  "patient102",
    "RV":   "patient109",
}

def zscore(arr):
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-8)


def load_models():
    models = []
    for fold_dir in sorted(FOLDS.glob("fold_*")):
        ckpt = fold_dir / "best_model.pth"
        if not ckpt.exists():
            continue
        PS.FREEZE_BLK = 2
        m = PS.PULSESeg(img_size=IMG_SIZE, freeze_blocks=2).to(DEVICE)
        ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        state = ck["model"] if "model" in ck else ck
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        print(f"  Loaded {fold_dir.name}  best_dice={ck.get('best_dice','?')}")
    return models


@torch.no_grad()
def predict_slice(models, sl3ch):
    """sl3ch: (H,W,3) float32 zscore'd. Returns (H,W) uint8 prediction."""
    imgs = []
    for c in range(3):
        ch = sl3ch[:,:,c]
        imgs.append(cv2.resize(ch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR))
    x = torch.from_numpy(np.stack(imgs, 0)).float().unsqueeze(0).to(DEVICE)

    all_probs = []
    for model in models:
        probs_tta = []
        for fh, fv in [(False,False),(True,False),(False,True),(True,True)]:
            xi = x.clone()
            if fh: xi = xi.flip(-1)
            if fv: xi = xi.flip(-2)
            with torch.amp.autocast(DEVICE.type):
                out = model(xi)
                main = out[0] if isinstance(out, tuple) else out
            p = F.softmax(main, 1)
            if fh: p = p.flip(-1)
            if fv: p = p.flip(-2)
            probs_tta.append(p)
        all_probs.append(torch.stack(probs_tta).mean(0))
    prob = torch.stack(all_probs).mean(0)
    pred = prob[0].argmax(0).cpu().numpy().astype(np.uint8)
    H, W = sl3ch.shape[:2]
    return cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)


def overlay_on_image(img_gray, gt_mask, pred_mask):
    """
    Returns a (H, W, 3) uint8 RGB image:
    - Left half: input MRI + GT contour outlines (dashed-style, just 1px contour)
    - Right half: input MRI + filled translucent prediction overlay
    Both panels side-by-side in one image.
    """
    H, W = img_gray.shape
    # Normalise to 0-255
    img_norm = img_gray.copy().astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    img_u8 = (img_norm * 255).astype(np.uint8)
    img_rgb = np.stack([img_u8]*3, axis=-1)

    # Left panel: GT contours
    left = img_rgb.copy()
    for cls, color_rgb in COLORS.items():
        mask = (gt_mask == cls).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_bgr = (int(color_rgb[2]*255), int(color_rgb[1]*255), int(color_rgb[0]*255))
        cv2.drawContours(left, contours, -1, color_bgr, 2)

    # Right panel: filled pred overlay (alpha blend)
    right = img_rgb.copy().astype(np.float32)
    for cls, color_rgb in COLORS.items():
        mask = (pred_mask == cls)
        if mask.sum() == 0:
            continue
        for c, v in enumerate(color_rgb):
            right[:,:,c][mask] = right[:,:,c][mask] * 0.45 + v * 255 * 0.55
    right = right.clip(0, 255).astype(np.uint8)

    # Combine side by side with a 4px white separator
    sep = np.ones((H, 4, 3), dtype=np.uint8) * 255
    combined = np.concatenate([left, sep, right], axis=1)
    return combined


def select_slices(vol_shape, gt_vol, n=4):
    """Pick n most informative slices (most GT content) around mid-cavity."""
    S = vol_shape[-1]
    content = [(gt_vol[:,:,z] > 0).sum() for z in range(S)]
    # Sort by content descending, pick top-n spaced slices
    ranked = sorted(range(S), key=lambda z: content[z], reverse=True)
    # Take slices spread across z-range where there's content
    active = [z for z in range(S) if content[z] > 50]
    if not active:
        return list(range(S//2 - n//2, S//2 + n//2))
    lo, hi = min(active), max(active)
    step = max(1, (hi - lo) // (n - 1)) if n > 1 else 1
    selected = [lo + i*step for i in range(n)]
    selected = [min(s, S-1) for s in selected]
    return selected


def make_strip(vol, gt_vol, pred_vol, slices, target_h=460):
    """Build a horizontal strip from selected slices."""
    panels = []
    for z in slices:
        sl = vol[:,:,z].astype(np.float32)
        gt = gt_vol[:,:,z].astype(np.uint8)
        pr = pred_vol[:,:,z].astype(np.uint8)
        panel = overlay_on_image(sl, gt, pr)
        # Resize to target height
        h, w = panel.shape[:2]
        new_w = int(w * target_h / h)
        panel = cv2.resize(panel, (new_w, target_h))
        panels.append(panel)

    # Add small vertical gap between panels
    gap = np.ones((target_h, 8, 3), dtype=np.uint8) * 255
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, gap, p], axis=1)
    return strip


def process_patient(pat_id, phase_key, models):
    """
    phase_key: 'ED' or 'ES'
    Returns (strip_rgb, patient_info_str)
    """
    pat_dir = DATA / pat_id
    info = {}
    for line in (pat_dir / "Info.cfg").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()

    frame_no = int(info[phase_key])
    group = info["Group"]

    # Find nifti files (testing set uses .nii, not .nii.gz)
    img_path = pat_dir / f"{pat_id}_frame{frame_no:02d}.nii"
    gt_path  = pat_dir / f"{pat_id}_frame{frame_no:02d}_gt.nii"

    if not img_path.exists() or not gt_path.exists():
        print(f"  Missing files for {pat_id} frame {frame_no}")
        return None

    vol    = nib.load(str(img_path)).get_fdata().astype(np.float32)
    gt_vol = nib.load(str(gt_path)).get_fdata().astype(np.int64)

    S = vol.shape[-1]
    pred_vol = np.zeros_like(gt_vol, dtype=np.uint8)

    # Run ensemble on all slices
    for z in range(S):
        z0, z1, z2 = max(z-1,0), z, min(z+1,S-1)
        sl = [zscore(vol[:,:,zi]) for zi in (z0,z1,z2)]
        sl3 = np.stack(sl, axis=-1)
        pred_vol[:,:,z] = predict_slice(models, sl3)

    slices = select_slices(vol.shape, gt_vol, n=4)
    strip  = make_strip(vol, gt_vol, pred_vol, slices)
    return strip, group


def main():
    print("Loading 5-fold ensemble...")
    models = load_models()
    print(f"Loaded {len(models)} models\n")

    for group, pat_id in CLASS_PATIENTS.items():
        for phase in ("ED", "ES"):
            print(f"Processing {group} ({pat_id}) — {phase}...")
            result = process_patient(pat_id, phase, models)
            if result is None:
                continue
            strip, _ = result

            # Convert to RGBA (matching existing format)
            strip_rgba = cv2.cvtColor(strip, cv2.COLOR_RGB2RGBA)

            out_name = f"{group}det{'ed' if phase=='ED' else 'es'}.png"
            out_path = OUT / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(strip_rgba, cv2.COLOR_RGBA2BGRA))
            print(f"  Saved: {out_path}  ({strip.shape[1]}x{strip.shape[0]})")

    print("\nDone. All segmentation visualizations saved to paper_contents/")


if __name__ == "__main__":
    main()
