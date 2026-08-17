#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULSE — single-volume inference.
================================
Segment ONE short-axis cardiac MRI volume (NIfTI, .nii/.nii.gz) with the
trained PULSE model(s). Works on CPU or GPU automatically. If you point it at
a folder of `fold_*/best_model.pth` checkpoints it runs the 5-fold ensemble;
if you point it at a single `.pth` file it runs that one model.

Examples
--------
# GPU/CPU auto, 5-fold ensemble, save the predicted mask next to the input:
python pulse/infer.py --input sample.nii.gz --folds_dir checkpoints/folds_vdino

# single checkpoint, explicit output path, force CPU:
python pulse/infer.py --input sample.nii.gz \
       --weights checkpoints/folds_vdino/fold_0/best_model.pth \
       --output sample_pred.nii.gz --device cpu

Output: a NIfTI label map with the same shape/affine as the input, where
0=background, 1=RV, 2=Myocardium, 3=LV.
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F

import pulse_seg as PS
from postprocess import postprocess

LABELS = {0: "Background", 1: "RV", 2: "Myocardium", 3: "LV"}


def zscore(a: np.ndarray) -> np.ndarray:
    m, s = a.mean(), a.std()
    return (a - m) / (s + 1e-8)


def load_models(weights, folds_dir, img_size, device):
    models = []
    if weights:
        paths = [Path(weights)]
    else:
        paths = sorted(Path(folds_dir).glob("fold_*/best_model.pth"))
    if not paths:
        raise FileNotFoundError(
            "No checkpoints found. Pass --weights <file.pth> or --folds_dir <dir with fold_*/best_model.pth>. "
            "See checkpoints/README.md to download the weights.")
    for p in paths:
        m = PS.PULSESeg(img_size=img_size).to(device)
        ck = torch.load(str(p), map_location=device)
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        m.eval()
        models.append(m)
        print(f"  loaded {p}")
    return models


@torch.no_grad()
def predict(models, vol, device, img_size):
    """2.5D input + 4-way flip TTA, softmax-averaged over models."""
    H, W, S = vol.shape
    out = np.zeros((H, W, S), np.uint8)
    for z in range(S):
        z0, z1, z2 = max(z - 1, 0), z, min(z + 1, S - 1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r = [cv2.resize(s, (img_size, img_size), interpolation=cv2.INTER_LINEAR) for s in sl]
        x0 = torch.from_numpy(np.stack(r, -1).transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        probs = []
        for m in models:
            for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
                xi = x0.clone()
                if fh: xi = xi.flip(-1)
                if fv: xi = xi.flip(-2)
                o = m(xi)
                main = o[0] if isinstance(o, tuple) else o
                sp = F.softmax(main, 1)
                if fh: sp = sp.flip(-1)
                if fv: sp = sp.flip(-2)
                probs.append(sp)
        p = torch.stack(probs).mean(0).argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        out[..., z] = cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST)
    return postprocess(out)


def main():
    ap = argparse.ArgumentParser(description="PULSE single-volume cardiac MRI segmentation.")
    ap.add_argument("--input", required=True, help="Input short-axis MRI (.nii/.nii.gz).")
    ap.add_argument("--output", default=None, help="Output mask path (default: <input>_pred.nii.gz).")
    ap.add_argument("--folds_dir", default="checkpoints/folds_vdino",
                    help="Folder with fold_*/best_model.pth (5-fold ensemble).")
    ap.add_argument("--weights", default=None, help="Single checkpoint .pth (overrides --folds_dir).")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"Device: {device}")

    models = load_models(args.weights, args.folds_dir, args.img_size, device)
    nii = nib.load(args.input)
    vol = nii.get_fdata().astype(np.float32)
    if vol.ndim == 4:  # (H,W,S,T) -> take first frame
        vol = vol[..., 0]
    print(f"Input volume {args.input}  shape={vol.shape}")

    mask = predict(models, vol, device, args.img_size)

    out_path = args.output or str(Path(args.input).with_suffix("").with_suffix("")) + "_pred.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), nii.affine, nii.header), out_path)
    print(f"Saved segmentation -> {out_path}")
    for k, name in LABELS.items():
        print(f"  {name:11s}: {(mask == k).sum():>8d} voxels")


if __name__ == "__main__":
    main()
