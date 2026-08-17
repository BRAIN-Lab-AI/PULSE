#!/usr/bin/env python3
"""
Ablation: NO BOUNDARY LOSS
Trains PULSE with LAMBDA_BND=0.0 — boundary loss term removed from seg_loss().
All other settings (deep supervision, curriculum, etc.) identical to full model.
Run for fold 0 to compare against full model fold 0 (~87.0%).
"""
import pulse_seg as ps

# Disable boundary loss
ps.LAMBDA_BND = 0.0

if __name__ == "__main__":
    ps.main()
