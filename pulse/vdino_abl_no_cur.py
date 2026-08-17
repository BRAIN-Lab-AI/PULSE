#!/usr/bin/env python3
"""Vanilla DINOv2 ablation: no two-stage curriculum (0-block freeze)."""
import pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.FREEZE_BLK = 0
ps.STAGE1     = 9999
if __name__ == "__main__": ps.main()
