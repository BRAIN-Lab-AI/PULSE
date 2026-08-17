# Training Guide

## The exact configuration used in the paper
- Backbone: DINOv2 ViT-B/14 (`facebook/dinov2-base`)
- Input: 2.5D (3 adjacent slices), 256x256
- Epochs: 120  |  Batch size: 16  |  Optimizer: AdamW  |  LR: 3e-4, weight decay 0.05
- Two-stage curriculum: epochs 1-70 freeze first 2 ViT blocks; epochs 71-120 unfreeze all (backbone LR 10-20x lower)
- Loss: class-weighted Dice + CE + 0.5*Lovasz + 0.5*Boundary, with deep supervision [1.0, 0.5, 0.25]
- 5-fold stratified cross-validation; 4-way TTA + 5-fold ensemble at inference
- Hardware used: 1x NVIDIA A100 40GB, ~20 min/fold

Full config is documented in `configs/default.yaml`.

## Train one fold
```bash
python pulse/train.py --fold 0 --nfolds 5 --epochs 120 --bs 16 \
       --ckpt_dir checkpoints/folds_vdino/fold_0 \
       --data_root /path/to/ACDC/database
```

## Train all 5 folds
```bash
for f in 0 1 2 3 4; do
  python pulse/train.py --fold $f --nfolds 5 --epochs 120 --bs 16 \
         --ckpt_dir checkpoints/folds_vdino/fold_$f \
         --data_root /path/to/ACDC/database
done
```

## Recommended settings by hardware (batch size is the main knob)

| GPU / machine | `--bs` | Notes |
|---|---|---|
| A100 / H100 (40-80 GB) | 16 | paper setting |
| RTX 3090 / 4090 (24 GB) | 8 | ~same accuracy |
| RTX 3060 / 2080 (8-12 GB) | 4 | add `--epochs 120`; slightly longer |
| Colab T4 (16 GB) | 6-8 | works; ~1-2 h/fold |
| CPU only / small laptop | — | training is impractical; use the released weights for inference |

Lowering the batch size does not change the method; it only trades memory for
speed. If you hit out-of-memory, halve `--bs`.

## What gets saved
Each fold writes `best_model.pth` (best validation Dice) and `history.json`
into its `--ckpt_dir`.
