#!/usr/bin/env python3
"""Vanilla DINOv2 ablation: no boundary loss."""
import pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.FREEZE_BLK = 2
ps.STAGE1     = 70
ps.LAMBDA_BND = 0.0
if __name__ == "__main__": ps.main()
