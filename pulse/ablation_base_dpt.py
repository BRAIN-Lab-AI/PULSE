#!/usr/bin/env python3
"""
Ablation: DPT BASE MODEL (no novel components)
Trains the DPT+RAD-DINOv2 model with the optimised training recipe
(Dice+CE+Lovász, strong augmentation, volume-wise normalisation)
but WITHOUT any of the three novel architectural contributions:
  - No deep supervision  (auxiliary heads disabled, DS_WEIGHTS=[1,0,0])
  - No boundary loss     (LAMBDA_BND=0)
  - No two-stage curriculum (FREEZE_BLK=0, backbone fully unfrozen from ep 1)

This gives the 'DPT base + recipe only' reference point that isolates
how much the novel components contribute on top of the DPT architecture.
Run for fold 0 to compare against full model fold 0 (~87.0%).
Output goes to ablation_ckpts/base_dpt/
"""
import pulse_seg as ps

ps.DS_WEIGHTS  = [1.0, 0.0, 0.0]   # no auxiliary supervision
ps.LAMBDA_BND  = 0.0                # no boundary loss
ps.FREEZE_BLK  = 0                  # no frozen blocks at init
ps.STAGE1      = 9999               # never trigger stage-2 unfreeze

if __name__ == "__main__":
    ps.main()
