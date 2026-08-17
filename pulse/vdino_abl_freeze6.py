#!/usr/bin/env python3
"""Vanilla DINOv2 ablation: 6-block Stage-1 freeze."""
import pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.FREEZE_BLK = 6
ps.STAGE1     = 70
if __name__ == "__main__": ps.main()
