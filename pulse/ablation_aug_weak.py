#!/usr/bin/env python3
"""
Ablation: WEAK AUGMENTATION (on DPT architecture)
Trains the full DPT+RAD-DINOv2 model with ALL novel components active
but replaces the strong augmentation pipeline with a weak one:
  - flips + small rotation only
  - no elastic transform, no grid distortion, no noise, no blur

Strong (full PULSE) aug pipeline for reference:
  HorizontalFlip, VerticalFlip, RandomRotate90,
  Affine(translate=0.1, scale=0.85-1.15, rotate=-30..30, p=0.7),
  ElasticTransform(alpha=80, p=0.4), GridDistortion(p=0.3),
  RandomBrightnessContrast(0.25, p=0.5), GaussNoise(p=0.3), GaussianBlur(p=0.2)

Compares against:
  - Full PULSE (strong aug): 87.0%  fold 0 [abl_eval_14725.log]
  - ablation_aug_none.py   (no aug): TBD
"""
import cv2
import albumentations as A
import pulse_seg as ps


def _weak_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-10, 10),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ])


ps._build_aug = _weak_aug

if __name__ == "__main__":
    ps.main()
