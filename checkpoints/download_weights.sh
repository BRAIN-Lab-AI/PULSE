#!/usr/bin/env bash
# Download PULSE fold checkpoints. Replace the URLs below with your release links.
set -e
cd "$(dirname "$0")"

# --- EDIT THESE (GitHub Release / HF / Drive direct-download links) ----------
declare -A URLS=(
  ["folds_vdino/fold_0/best_model.pth"]="https://REPLACE_ME/fold_0.pth"
  ["folds_vdino/fold_1/best_model.pth"]="https://REPLACE_ME/fold_1.pth"
  ["folds_vdino/fold_2/best_model.pth"]="https://REPLACE_ME/fold_2.pth"
  ["folds_vdino/fold_3/best_model.pth"]="https://REPLACE_ME/fold_3.pth"
  ["folds_vdino/fold_4/best_model.pth"]="https://REPLACE_ME/fold_4.pth"
)
# ----------------------------------------------------------------------------

for dest in "${!URLS[@]}"; do
  url="${URLS[$dest]}"
  if [[ "$url" == *REPLACE_ME* ]]; then
    echo "!! Edit download_weights.sh and set the real URL for $dest"; continue
  fi
  mkdir -p "$(dirname "$dest")"
  echo "Downloading $dest ..."
  curl -L -o "$dest" "$url"
done
echo "Done. Verify: ls checkpoints/folds_vdino/fold_0/best_model.pth"
