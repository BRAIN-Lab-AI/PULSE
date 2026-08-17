#!/usr/bin/env python3
"""
Ablation: NO TWO-STAGE CURRICULUM
Trains PULSE with FREEZE_BLK=0 — all backbone blocks train from epoch 1.
No stage-1 partial freeze, no stage-2 unfreeze step. Single-stage training
with the same total epochs (120) and base LR (3e-4) as Stage 1.
All other settings identical to the full model.
Run for fold 0 to compare against full model fold 0 (~87.0%).
"""
import pulse_seg as ps

# No frozen blocks at init → all layers train from epoch 1
ps.FREEZE_BLK = 0
# Set STAGE1 beyond total epochs so the unfreeze code never triggers
ps.STAGE1 = 9999

if __name__ == "__main__":
    ps.main()
