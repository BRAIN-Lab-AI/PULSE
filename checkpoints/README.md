# Pretrained Weights

The trained PULSE checkpoints are hosted on the GitHub Release, not in the git
tree (each DINOv2 ViT-B/14 fold is ~358 MB). The final model is a 5-fold ensemble.

## Quick download (recommended)

```bash
bash checkpoints/download_weights.sh
```

This fetches the five folds from the [v1.0 release](https://github.com/BRAIN-Lab-AI/PULSE/releases/tag/v1.0)
and places them at:

```
checkpoints/folds_vdino/
├── fold_0/best_model.pth
├── fold_1/best_model.pth
├── fold_2/best_model.pth
├── fold_3/best_model.pth
└── fold_4/best_model.pth
```

## Manual download

Download `fold_0.pth` ... `fold_4.pth` from the
[Releases page](https://github.com/BRAIN-Lab-AI/PULSE/releases/tag/v1.0) and save
each as `checkpoints/folds_vdino/fold_{i}/best_model.pth`.

## Load in Python

```python
import torch, pulse_seg as ps
model = ps.PULSESeg()
ckpt = torch.load("checkpoints/folds_vdino/fold_0/best_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model"]); model.eval()
```
