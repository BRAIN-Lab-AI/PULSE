#!/usr/bin/env python3
"""M&Ms zero-shot eval with vanilla DINOv2 backbone (facebook/dinov2-base).
Sets BACKBONE before importing eval_mnm so PULSESeg is built with the right weights.
Pass --folds_dir folds_vdino (or any dir with fold_0..fold_4/best_model.pth).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pulse_seg as ps
ps.BACKBONE = "facebook/dinov2-base"

import eval_mnm

if __name__ == "__main__":
    eval_mnm.main()
