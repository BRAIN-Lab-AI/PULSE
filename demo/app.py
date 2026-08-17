#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULSE cardiac MRI segmentation - Gradio demo.
Upload a short-axis cine cardiac MRI (.nii / .nii.gz) and PULSE segments the
right ventricle (RV), myocardium (Myo), and left ventricle (LV). Weights are
pulled from https://huggingface.co/hg-0403/PULSE. Paper: IEEE JBHI 2026.
Code: https://github.com/BRAIN-Lab-AI/PULSE
"""
import os
import sys

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import hf_hub_download

# On the Hugging Face Space, pulse_seg.py and postprocess.py sit next to this
# file. When running locally from the repo, they live in ../pulse.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pulse"))

import pulse_seg as PS
from postprocess import postprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
# Paper colour convention: RV = red, Myocardium = green, LV = blue.
COLORS = {1: (220, 40, 40), 2: (40, 200, 60), 3: (50, 90, 230)}
LABELS = {1: "RV", 2: "Myocardium", 3: "LV"}

_MODEL = None


def get_model():
    """Load a single fold once (fast on CPU). Weights come from the Hub."""
    global _MODEL
    if _MODEL is None:
        ckpt = hf_hub_download("hg-0403/PULSE", "folds_vdino/fold_0/best_model.pth")
        m = PS.PULSESeg(img_size=IMG_SIZE).to(DEVICE)
        sd = torch.load(ckpt, map_location=DEVICE)
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        m.eval()
        _MODEL = m
    return _MODEL


def zscore(a):
    return (a - a.mean()) / (a.std() + 1e-8)


def to_uint8(gray):
    g = zscore(gray)
    g = np.clip((g - g.min()) / (np.ptp(g) + 1e-8), 0, 1)
    return (np.stack([g] * 3, -1) * 255).astype(np.uint8)


@torch.no_grad()
def predict_volume(vol, tta=False):
    """2.5D single-model inference, optional 4-way flip TTA."""
    m = get_model()
    H, W, S = vol.shape
    out = np.zeros((H, W, S), np.uint8)
    flips = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
    for z in range(S):
        z0, z1, z2 = max(z - 1, 0), z, min(z + 1, S - 1)
        sl = [np.ascontiguousarray(zscore(vol[..., zi])) for zi in (z0, z1, z2)]
        r = [cv2.resize(s, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for s in sl]
        x0 = torch.from_numpy(np.stack(r, -1).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
        probs = []
        for fh, fv in flips:
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


def overlay(gray, mask):
    rgb = to_uint8(gray)
    for lab, c in COLORS.items():
        sel = mask == lab
        rgb[sel] = (0.45 * np.array(c) + 0.55 * rgb[sel]).astype(np.uint8)
    return rgb


def run(nii_file, tta):
    if nii_file is None:
        return None, None, "Please upload a short-axis cardiac MRI (.nii or .nii.gz)."
    nii = nib.load(nii_file)
    vol = nii.get_fdata().astype(np.float32)
    if vol.ndim == 4:          # (H, W, S, T) -> first frame
        vol = vol[..., 0]
    if vol.ndim != 3:
        return None, None, f"Expected a 3D volume, got shape {vol.shape}."

    mask = predict_volume(vol, tta)
    fg = (mask > 0).sum(axis=(0, 1))
    z = int(fg.argmax()) if fg.any() else vol.shape[2] // 2

    base = to_uint8(vol[..., z])
    ov = overlay(vol[..., z], mask[..., z])
    counts = "\n".join(f"  {LABELS[l]:12s}: {int((mask == l).sum()):>9,d} voxels" for l in (1, 2, 3))
    info = (f"Volume shape : {tuple(vol.shape)}\n"
            f"Shown slice  : {z} (largest segmented area)\n"
            f"TTA          : {'on (4-way flip)' if tta else 'off'}\n\n"
            f"Predicted volumes:\n{counts}")
    return base, ov, info


TITLE = "PULSE - Cardiac MRI Segmentation"
DESC = """
Segment the **right ventricle (red)**, **myocardium (green)**, and **left ventricle (blue)**
from a short-axis cine cardiac MRI. Upload a `.nii` / `.nii.gz` volume and press **Segment**.

This demo runs a single fold on CPU. The full 5-fold ensemble, diagnosis, and clinical-report
pipeline are in the [GitHub repository](https://github.com/BRAIN-Lab-AI/PULSE).
Model weights: [hg-0403/PULSE](https://huggingface.co/hg-0403/PULSE) - Paper: IEEE JBHI 2026.
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.File(label="Cardiac MRI (.nii / .nii.gz)", file_types=[".nii", ".gz"])
            tta = gr.Checkbox(label="Test-time augmentation (slower, slightly better)", value=False)
            btn = gr.Button("Segment", variant="primary")
            info = gr.Textbox(label="Summary", lines=10)
        with gr.Column(scale=2):
            with gr.Row():
                out_in = gr.Image(label="Input slice", height=360)
                out_ov = gr.Image(label="Segmentation (RV / Myo / LV)", height=360)
    btn.click(run, inputs=[inp, tta], outputs=[out_in, out_ov, info])
    gr.Markdown(
        "RV = red, Myocardium = green, LV = blue. "
        "For research and educational use only; not a medical device."
    )

if __name__ == "__main__":
    demo.launch()
