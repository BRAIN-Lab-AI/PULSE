#!/usr/bin/env python3
"""
Ablation: FREEZE SCHEDULE — 4 BLOCKS in Stage 1 (on DPT architecture)
Trains full DPT+RAD-DINOv2 PULSE with all novel components active but
freezes the first 4 backbone blocks (out of 12) during Stage 1
(epochs 1-70), then unfreezes them in Stage 2 (epochs 71-120).

This isolates the effect of how deep the Stage-1 freeze extends:
  0 blk (no freeze)   → 86.5%  [abl_eval_14725.log, No-Curr row]
  2 blk               → TBD    [ablation_freeze_2blk.py]
  4 blk (this script) → TBD
  6 blk (full PULSE)  → 87.0%  [abl_eval_14725.log, Full PULSE row]
"""
import pulse_seg as ps

ps.FREEZE_BLK = 4   # freeze first 4 blocks in Stage 1
ps.STAGE1     = 70  # unfreeze at epoch 71 (same as full PULSE)

if __name__ == "__main__":
    ps.main()
