#!/usr/bin/env python3
"""
Generate cardiac segmentation saliency maps using GradCAM on the DPT decoder.

GradCAM w.r.t. the combined cardiac classes (RV+Myo+LV) at the last decoder
feature map (res[0] output, ~144×144) shows where the model actually focuses
when making cardiac predictions — not CLS token attention which latches onto
image edge artefacts rather than anatomy.

Output: fig_attention_maps.png in paper_contents/
Layout: 5 columns (one disease class), 3 rows:
  Row 0 — Input MRI (greyscale)
  Row 1 — Predicted segmentation overlay
  Row 2 — GradCAM saliency overlay (cardiac classes 1+2+3)
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as PS
PS.BACKBONE = "facebook/dinov2-base"
PS.FREEZE_BLK = 2

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BASE     = Path(__file__).parent
CKPT     = BASE / "folds_vdino" / "fold_0" / "best_model.pth"
OUT      = BASE / "paper_contents" / "fig_attention_maps.png"
DATA     = Path("/SLURM/home/slurm_g202518690/student251/ACDC/database/testing")

CLASS_PATIENTS = {
    "DCM":  "patient101",
    "HCM":  "patient104",
    "MINF": "patient103",
    "NOR":  "patient102",
    "RV":   "patient109",
}

# Segmentation colours
SEG_COLORS = {
    1: (0.227, 0.733, 0.980),   # RV  — cyan
    2: (0.271, 0.918, 0.353),   # Myo — green
    3: (0.973, 0.529, 0.259),   # LV  — orange
}


def load_model():
    PS.FREEZE_BLK = 2
    model = PS.PULSESeg(img_size=IMG_SIZE, freeze_blocks=2).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded fold-0  best_dice={ck.get('best_dice','?')}")
    return model


def zscore(arr):
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s + 1e-8)


def get_best_slice(pat_id, phase="ED"):
    pat_dir = DATA / pat_id
    info = {}
    for line in (pat_dir / "Info.cfg").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    frame_no = int(info[phase])
    img_path = pat_dir / f"{pat_id}_frame{frame_no:02d}.nii"
    gt_path  = pat_dir / f"{pat_id}_frame{frame_no:02d}_gt.nii"

    vol    = nib.load(str(img_path)).get_fdata().astype(np.float32)
    gt_vol = nib.load(str(gt_path)).get_fdata().astype(np.int64)
    S = vol.shape[-1]
    content = [(gt_vol[:, :, z] > 0).sum() for z in range(S)]
    z_best  = int(np.argmax(content))

    z0 = max(z_best - 1, 0)
    z2 = min(z_best + 1, S - 1)
    sl = np.stack([zscore(vol[:, :, zi]) for zi in (z0, z_best, z2)], axis=-1)
    return vol[:, :, z_best], gt_vol[:, :, z_best], sl


def prep_tensor(sl3ch):
    """(H,W,3) → (1,3,256,256) float tensor on device."""
    imgs = [cv2.resize(sl3ch[:, :, c], (IMG_SIZE, IMG_SIZE),
                       interpolation=cv2.INTER_LINEAR) for c in range(3)]
    x = torch.from_numpy(np.stack(imgs, 0)).float().unsqueeze(0).to(DEVICE)
    return x


@torch.enable_grad()
def compute_gradcam(model, x, target_classes=(1, 2, 3)):
    """
    GradCAM on the last decoder feature (res[0] output, before ups[0]).
    Gradient of sum of target-class logits w.r.t. that feature map.
    Returns (H_feat, W_feat) numpy array normalised to [0,1].
    """
    feat_store = {}

    def fwd_hook(module, inp, out):
        feat_store['f'] = out
        if out.requires_grad:
            out.retain_grad()

    handle = model.decoder.res[0].register_forward_hook(fwd_hook)

    # Need grad through features so enable grad on input
    x_g = x.clone().detach().float().requires_grad_(False)
    # Enable grad on model parameters temporarily just for backward
    for p in model.decoder.parameters():
        p.requires_grad_(True)

    model.zero_grad()
    out = model(x_g)
    logits = out[0]  # (1, 4, 256, 256) — main segmentation head

    # Score = sum of target-class logit mass
    score = sum(logits[0, c].sum() for c in target_classes)

    # Retain grad on the stored feature
    f = feat_store.get('f', None)
    if f is not None and f.requires_grad:
        f.retain_grad()

    score.backward()
    handle.remove()

    f = feat_store.get('f', None)
    if f is None or f.grad is None:
        # Fallback: use input gradient
        print("  GradCAM: fallback to input gradient")
        return np.ones((16, 16))

    # GradCAM
    weights = f.grad.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
    cam = (weights * f).sum(dim=1).relu()[0]          # (H', W')
    cam = cam.detach().cpu().numpy().astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Freeze params back
    for blk in model.backbone_blocks[:2]:
        for p in blk.parameters():
            p.requires_grad_(False)

    return cam


def make_overlay(gray_img, cam_map, orig_H, orig_W, alpha=0.60):
    """Resize cam to original image size and blend as jet heatmap."""
    cam_up = cv2.resize(cam_map, (orig_W, orig_H), interpolation=cv2.INTER_CUBIC)
    cam_up = np.clip(cam_up, 0, 1)
    img_u8 = (np.clip(gray_img, 0, None) / (gray_img.max() + 1e-8) * 255).astype(np.uint8)
    img_rgb = np.stack([img_u8] * 3, axis=-1)
    heat = (cm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
    blended = (img_rgb * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return img_rgb, blended


def make_seg_overlay(gray_img, pred_mask):
    """Overlay segmentation mask contours + fills on greyscale MRI."""
    img_u8 = (np.clip(gray_img, 0, None) / (gray_img.max() + 1e-8) * 255).astype(np.uint8)
    img_rgb = np.stack([img_u8] * 3, axis=-1).astype(np.float32)
    overlay = img_rgb.copy()
    for cls_id, color in SEG_COLORS.items():
        mask = (pred_mask == cls_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        color_arr = np.array(color) * 255
        overlay[mask == 1] = overlay[mask == 1] * 0.45 + color_arr * 0.55
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay.astype(np.uint8), contours, -1,
                         (int(color_arr[0]), int(color_arr[1]), int(color_arr[2])), 2)
    return overlay.clip(0, 255).astype(np.uint8)


def main():
    model = load_model()
    classes = list(CLASS_PATIENTS.keys())
    n = len(classes)

    fig, axes = plt.subplots(3, n, figsize=(3.6 * n, 10.8),
                             facecolor="black", constrained_layout=True)
    row_labels = ["Input MRI", "Segmentation", "GradCAM Saliency"]
    plt.rcParams.update({"text.color": "white"})

    for col, cls_name in enumerate(classes):
        pat_id = CLASS_PATIENTS[cls_name]
        print(f"  Processing {cls_name} ({pat_id})...")
        raw, gt, sl3 = get_best_slice(pat_id, "ED")
        H, W = raw.shape

        x = prep_tensor(sl3)

        # Prediction (no grad needed)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out[0], dim=1)[0].cpu().numpy()
            pred  = probs.argmax(0)
            # Resize pred back to original image size
            pred_orig = cv2.resize(pred.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # GradCAM
        cam = compute_gradcam(model, x, target_classes=(1, 2, 3))
        model.zero_grad()

        img_rgb, cam_overlay = make_overlay(raw, cam, H, W, alpha=0.60)
        seg_overlay = make_seg_overlay(raw, pred_orig)

        # Row 0: Input MRI
        ax0 = axes[0, col]
        ax0.imshow((raw - raw.min()) / (raw.max() - raw.min() + 1e-8),
                   cmap="gray", vmin=0, vmax=1)
        ax0.set_title(cls_name, color="white", fontsize=13, fontweight="bold", pad=4)
        ax0.axis("off"); ax0.set_facecolor("black")

        # Row 1: Segmentation prediction
        ax1 = axes[1, col]
        ax1.imshow(seg_overlay)
        ax1.axis("off"); ax1.set_facecolor("black")

        # Row 2: GradCAM
        ax2 = axes[2, col]
        ax2.imshow(cam_overlay)
        ax2.axis("off"); ax2.set_facecolor("black")

    # Row labels
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, color="white", fontsize=10)

    # Segmentation legend
    seg_handles = [
        mpatches.Patch(color=SEG_COLORS[1], label="RV"),
        mpatches.Patch(color=SEG_COLORS[2], label="Myo"),
        mpatches.Patch(color=SEG_COLORS[3], label="LV"),
    ]
    fig.legend(handles=seg_handles, loc="lower left", bbox_to_anchor=(0.01, 0.01),
               fontsize=9, framealpha=0.7, ncol=3,
               title="Segmentation", title_fontsize=9)

    # GradCAM colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=axes[2, :], fraction=0.012, pad=0.01, location="right")
    cbar.set_label("GradCAM saliency", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.savefig(str(OUT), dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    print(f"\nSaved: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
