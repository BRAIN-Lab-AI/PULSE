#!/usr/bin/env python3
"""Vanilla DINOv2 ablation: weak augmentation only."""
import cv2, albumentations as A, pulse_seg as ps
ps.BACKBONE   = "facebook/dinov2-base"
ps.FREEZE_BLK = 2
ps.STAGE1     = 70
def _weak_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-10, 10),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ])
ps._build_aug = _weak_aug
if __name__ == "__main__": ps.main()
