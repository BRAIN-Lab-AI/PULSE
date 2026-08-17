#!/usr/bin/env python3
"""Vanilla DINOv2 ablation: DPT base (no novel components)."""
import pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.DS_WEIGHTS = [1.0, 0.0, 0.0]
ps.LAMBDA_BND = 0.0
ps.FREEZE_BLK = 0
if __name__ == "__main__": ps.main()
