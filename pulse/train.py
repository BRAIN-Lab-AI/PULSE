#!/usr/bin/env python3
"""Vanilla DINOv2 main training — 2-block Stage-1 freeze (the PULSE full model)."""
import pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.FREEZE_BLK = 2
ps.STAGE1     = 70
if __name__ == "__main__": ps.main()
