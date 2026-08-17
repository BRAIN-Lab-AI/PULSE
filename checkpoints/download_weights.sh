#!/usr/bin/env bash
# Download the PULSE pretrained 5-fold checkpoints from the GitHub Release
# and place them at checkpoints/folds_vdino/fold_{0..4}/best_model.pth
set -e
cd "$(dirname "$0")"
BASE="https://github.com/BRAIN-Lab-AI/PULSE/releases/download/v1.0"

for i in 0 1 2 3 4; do
  mkdir -p "folds_vdino/fold_$i"
  echo "Downloading fold_$i ..."
  curl -L -o "folds_vdino/fold_$i/best_model.pth" "$BASE/fold_$i.pth"
done
echo "Done. Verify: ls checkpoints/folds_vdino/fold_0/best_model.pth"
