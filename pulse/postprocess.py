"""Post-processing for cardiac segmentation: keep largest connected component per class."""
import numpy as np
from scipy import ndimage


def postprocess(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component per foreground class (RV=1, Myo=2, LV=3)."""
    out = np.zeros_like(mask)
    for lab in (1, 2, 3):
        b = (mask == lab)
        if not b.any():
            continue
        labeled, n = ndimage.label(b)
        if n == 0:
            continue
        sizes = ndimage.sum(b, labeled, range(1, n + 1))
        biggest = int(np.argmax(sizes)) + 1
        out[labeled == biggest] = lab
    return out
