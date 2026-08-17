#!/usr/bin/env python3
"""Vanilla DINOv2 ablation: no augmentation."""
import pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.FREEZE_BLK = 2
ps.STAGE1     = 70
ps._build_aug = lambda: None
if __name__ == "__main__": ps.main()
