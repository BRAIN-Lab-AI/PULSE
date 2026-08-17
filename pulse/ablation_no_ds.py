#!/usr/bin/env python3
"""
Ablation: NO DEEP SUPERVISION
Trains PULSE with DS_WEIGHTS=[1.0, 0.0, 0.0] — main head only, aux heads
contribute zero loss. All other settings identical to the full model.
Run for fold 0 to compare against full model fold 0 (~87.0%).
"""
import pulse_seg as ps

# Disable deep supervision: set auxiliary head loss weights to 0
ps.DS_WEIGHTS = [1.0, 0.0, 0.0]

if __name__ == "__main__":
    ps.main()
