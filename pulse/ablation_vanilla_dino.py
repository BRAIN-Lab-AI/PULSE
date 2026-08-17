#!/usr/bin/env python3
"""
Ablation: vanilla DINOv2 (facebook/dinov2-base) vs RAD-DINOv2 (microsoft/rad-dino).
Swaps only the backbone pre-training weights; architecture and training are identical.
Saves to ablation_ckpts/vanilla_dino/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pulse_seg as ps

# Swap backbone to vanilla DINOv2 (same ViT-B/14 arch, ImageNet pre-training)
ps.BACKBONE = "facebook/dinov2-base"

# All other hyperparams identical to full PULSE (freeze_2blk is our new best config)
ps.FREEZE_BLK = 2

ps.main()
