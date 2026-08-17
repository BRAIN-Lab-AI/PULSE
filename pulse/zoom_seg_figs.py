#!/usr/bin/env python3
"""Crop-and-zoom the ACDC ED/ES segmentation montages so the colored
segmentations are large and clearly visible. Reuses the existing rendered
overlays (no model needed): detect the colored overlay in each montage,
pick the N slices with the most segmentation, crop a padded box around the
heart in each, and re-stitch into a clean strip."""
import sys
import numpy as np
from PIL import Image

N_KEEP = 3          # slices to keep per montage (was ~8)
PAD = 0.55          # fraction of box size added as context on each side
OUT_H = 480         # output panel height (px)
GAP = 10            # white gap between panels

def zoom_montage(path, out_path, n_keep=N_KEEP):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    H, W, _ = a.shape
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    sat = a.max(2) - a.min(2)            # grayscale MRI ~0; colored overlay high
    mask = sat > 28
    colcnt = mask.sum(0).astype(float)
    # find contiguous column runs that contain overlay (each ~ one slice's heart)
    on = colcnt > (0.01 * H)
    runs = []
    i = 0
    while i < W:
        if on[i]:
            j = i
            while j < W and on[j]:
                j += 1
            # merge tiny gaps: extend if next overlay within 25px
            k = j
            while k < min(W, j + 25) and not on[k]:
                k += 1
            if k < W and on[k]:
                j = k
                continue
            runs.append((i, j))
            i = j
        else:
            i += 1
    # score each run by overlay area, keep the strongest n_keep, then sort L->R
    scored = []
    for (x0, x1) in runs:
        area = mask[:, x0:x1].sum()
        if area > 200:
            scored.append((area, x0, x1))
    scored.sort(reverse=True)
    keep = sorted(scored[:n_keep], key=lambda t: t[1])
    if not keep:
        return False

    panels = []
    for _, x0, x1 in keep:
        sub = mask[:, x0:x1]
        ys, xs = np.where(sub)
        cy0, cy1 = ys.min(), ys.max()
        cx0, cx1 = xs.min() + x0, xs.max() + x0
        bw, bh = cx1 - cx0, cy1 - cy0
        side = int(max(bw, bh) * (1 + 2 * PAD))   # square crop with context
        ccx, ccy = (cx0 + cx1) // 2, (cy0 + cy1) // 2
        sx0 = max(0, ccx - side // 2); sx1 = min(W, sx0 + side); sx0 = max(0, sx1 - side)
        sy0 = max(0, ccy - side // 2); sy1 = min(H, sy0 + side); sy0 = max(0, sy1 - side)
        crop = im.crop((sx0, sy0, sx1, sy1))
        w, h = crop.size
        crop = crop.resize((int(w * OUT_H / h), OUT_H), Image.LANCZOS)
        panels.append(crop)

    total_w = sum(p.size[0] for p in panels) + GAP * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, OUT_H), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.size[0] + GAP
    canvas.save(out_path)
    return True

if __name__ == "__main__":
    args = sys.argv[1:]
    for p in args:
        ok = zoom_montage(p, p)
        im = Image.open(p)
        print(f"{'OK ' if ok else 'SKIP'} {p}  -> {im.size}")
