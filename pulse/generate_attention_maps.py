#!/usr/bin/env python3
"""
Generate DINO-style self-attention maps from a PULSE-Seg fold checkpoint.
Produces a figure with 5 disease-class panels (one per ACDC class):
  col 1: CMR input slice (middle slice, ED phase)
  col 2: GT segmentation overlay
  col 3: PULSE prediction overlay
  col 4: Last-layer CLS attention map (averaged over heads)
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import pulse_seg as PS
from diagnosis_features import zscore, list_patients, TRAIN_ROOT, TEST_ROOT

warnings.filterwarnings("ignore")

# ── colour map for segmentation overlays ──────────────────────────────────────
SEG_COLORS = {1: (0.2, 0.6, 1.0), 2: (0.0, 0.9, 0.3), 3: (1.0, 0.2, 0.2)}  # RV blue, Myo green, LV red

ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "attn", ["#0d0221", "#4a0080", "#9b1dca", "#e85d04", "#f48c06", "#ffd166"], N=256)


def overlay_seg(img_gray, mask, alpha=0.55):
    """Blend segmentation colours onto a grayscale image."""
    rgb = np.stack([img_gray]*3, axis=-1).copy()
    for lab, col in SEG_COLORS.items():
        where = mask == lab
        for c, v in enumerate(col):
            rgb[..., c] = np.where(where, (1-alpha)*rgb[..., c] + alpha*v, rgb[..., c])
    return rgb


def extract_mid_slice(vol_path, mask_path=None, img_size=256):
    """Load volume, zscore, pick middle slice, return (img_norm, img_resized, mask_resized, z_idx)."""
    vol = nib.load(vol_path).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    z = S // 2
    img = zscore(vol[..., z])
    img_r = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_r = np.clip((img_r - img_r.min()) / (img_r.max() - img_r.min() + 1e-6), 0, 1)

    mask_r = None
    if mask_path is not None:
        m = nib.load(mask_path).get_fdata().astype(np.int64)[..., z]
        mask_r = cv2.resize(m.astype(np.uint8), (img_size, img_size),
                            interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return img, img_r, mask_r, z


def build_25d_tensor(vol_path, z, img_size, device):
    vol = nib.load(vol_path).get_fdata().astype(np.float32)
    H, W, S = vol.shape
    z0, z1, z2 = max(z-1, 0), z, min(z+1, S-1)
    sl = [cv2.resize(zscore(vol[..., zi]), (img_size, img_size), cv2.INTER_LINEAR) for zi in (z0, z1, z2)]
    x = torch.from_numpy(np.stack(sl, -1).transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return x


@torch.no_grad()
def get_attention_and_pred(model, x, device, img_size):
    """
    Extract DINO CLS self-attention from last ViT block.
    Returns (attention_map, pred_mask).
    """
    vit = model.vit  # AutoModel (ViT-B/14, patch_size=14)

    # RAD-DINOv2 uses patch_size=14; input must be divisible by 14.
    # Resize to 224x224 (224/14=16 exactly -> 16x16=256 patches).
    x_vit = F.interpolate(model.stem(x), size=(224, 224), mode='bilinear', align_corners=False)

    with torch.amp.autocast(device.type):
        vit_out = vit(pixel_values=x_vit, output_attentions=True)

    if vit_out.attentions is None:
        raise RuntimeError("Model did not return attentions; ensure output_attentions=True is supported.")

    # Last-layer attention: shape (B, num_heads, N+1, N+1) where N=256 for 224x224/patch14
    last_attn = vit_out.attentions[-1]   # (1, num_heads, N+1, N+1)
    # CLS row (index 0) over patch tokens (indices 1:)
    cls_attn = last_attn[0, :, 0, 1:]   # (num_heads, N)
    cls_attn = cls_attn.mean(0)          # (N,)
    N = cls_attn.shape[0]
    n = int(N ** 0.5)
    assert n * n == N, f"Patch count {N} is not a perfect square; got n={n}"
    attn_map = cls_attn.reshape(n, n).cpu().float().numpy()
    # Resize to img_size
    attn_map = cv2.resize(attn_map, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Also get full prediction via TTA
    probs = []
    with torch.amp.autocast(device.type):
        for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
            xi = x.clone()
            if fh: xi = xi.flip(-1)
            if fv: xi = xi.flip(-2)
            out = model(xi)
            main = out[0] if isinstance(out, tuple) else out
            sp = F.softmax(main, 1)
            if fh: sp = sp.flip(-1)
            if fv: sp = sp.flip(-2)
            probs.append(sp)
    pred = torch.stack(probs).mean(0).argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    return attn_map, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="folds/fold_4/best_model.pth")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--out", default="paper_contents/fig_attention_maps.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PS.PULSESeg(img_size=args.img_size).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"Loaded {args.ckpt}")

    # Collect one patient per disease class from test set
    all_pats = list_patients(TEST_ROOT)  # {pid: rec}
    class_order = ["NOR", "DCM", "HCM", "MINF", "RV"]
    reps = {}
    for pid, rec in sorted(all_pats.items()):
        g = rec["group"]
        if g in class_order and g not in reps:
            reps[g] = (pid, rec)
        if len(reps) == 5:
            break

    # Figure setup: 5 rows (classes) x 4 cols (img, gt, pred, attn)
    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    col_titles = ["CMR Input", "Ground Truth", "PULSE Prediction", "Self-Attention (CLS)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11, fontweight='bold', pad=6)

    for row_idx, grp in enumerate(class_order):
        pid, rec = reps[grp]
        ed_img_path = rec["phases"]["ED"]["img"]
        ed_gt_path  = rec["phases"]["ED"]["gt"]

        img_orig, img_r, gt_r, z = extract_mid_slice(ed_img_path, ed_gt_path, args.img_size)
        x = build_25d_tensor(ed_img_path, z, args.img_size, device)
        attn_map, pred = get_attention_and_pred(model, x, device, args.img_size)

        # col 0: raw CMR (greyscale)
        axes[row_idx, 0].imshow(img_r, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 0].set_ylabel(grp, fontsize=11, fontweight='bold', rotation=90, va='center', labelpad=4)

        # col 1: GT overlay
        gt_rgb = overlay_seg(img_r, gt_r)
        axes[row_idx, 1].imshow(gt_rgb, vmin=0, vmax=1)

        # col 2: Prediction overlay
        pred_rgb = overlay_seg(img_r, pred)
        axes[row_idx, 2].imshow(pred_rgb, vmin=0, vmax=1)

        # col 3: Attention map
        axes[row_idx, 3].imshow(img_r, cmap='gray', vmin=0, vmax=1, alpha=0.5)
        axes[row_idx, 3].imshow(attn_map, cmap=ATTENTION_CMAP, alpha=0.7, vmin=0, vmax=1)

        for c in range(4):
            axes[row_idx, c].axis('off')

    # Legend
    handles = [mpatches.Patch(color=SEG_COLORS[1], label='RV'),
               mpatches.Patch(color=SEG_COLORS[2], label='Myo'),
               mpatches.Patch(color=SEG_COLORS[3], label='LV')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.35, 0.01))

    plt.suptitle("PULSE Segmentation and RAD-DINOv2 Self-Attention Maps\n"
                 "Across ACDC Disease Classes (mid-slice ED, one representative per class)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
