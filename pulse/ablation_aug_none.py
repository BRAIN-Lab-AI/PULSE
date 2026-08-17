#!/usr/bin/env python3
"""
Ablation: NO AUGMENTATION (on DPT architecture)
Trains the full DPT+RAD-DINOv2 model with ALL novel components active
(deep supervision, boundary loss, two-stage curriculum) but with the
augmentation pipeline completely disabled.

Compares against:
  - Full PULSE (strong aug): 87.0%  fold 0 [abl_eval_14725.log]
  - ablation_aug_weak.py    (weak aug): TBD
"""
import pulse_seg as ps

# Disable augmentation entirely — monkey-patch before any dataset is built
ps._build_aug = lambda: None

if __name__ == "__main__":
    ps.main()
