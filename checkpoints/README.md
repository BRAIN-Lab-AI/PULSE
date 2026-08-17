# Pre-trained Weights

The trained PULSE checkpoints are **not stored in the git repository** (each
DINOv2 ViT-B/14 fold is ~180 MB). Download them into this folder.

## What you need

| Item | Folder layout | Size |
|---|---|---|
| 5 segmentation folds (main model) | `folds_vdino/fold_{0..4}/best_model.pth` | ~0.9 GB |
| Random Forest diagnosis (optional) | trained on the fly from biomarkers | — |

Final expected layout:

```
checkpoints/
└── folds_vdino/
    ├── fold_0/best_model.pth
    ├── fold_1/best_model.pth
    ├── fold_2/best_model.pth
    ├── fold_3/best_model.pth
    └── fold_4/best_model.pth
```

## Option A — Hugging Face Hub (recommended)

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="BRAIN-Lab-AI/PULSE-cardiac",
                  local_dir="checkpoints", allow_patterns=["folds_vdino/*"])
PY
```

## Option B — one-line download script

Edit the URLs at the top of `download_weights.sh`, then:

```bash
bash checkpoints/download_weights.sh
```

## Option C — manual

Download the release archive, unzip it here so that
`checkpoints/folds_vdino/fold_0/best_model.pth` exists.

> **Maintainers:** upload `folds_vdino/` to a GitHub Release, the Hugging Face
> Hub, or Google Drive/Zenodo, then replace `BRAIN-Lab-AI` / the URLs above.
