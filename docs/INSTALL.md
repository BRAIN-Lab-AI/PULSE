# Installation Guide (works on any laptop / server)

PULSE runs on **Linux, macOS, and Windows**. A GPU makes training fast, but
**inference works fine on a CPU-only laptop** (just slower).

## 1. Get the code
```bash
git clone https://github.com/BRAIN-Lab-AI/PULSE.git
cd PULSE
```

## 2. Create an environment (pick ONE)

**Option A — conda (recommended, most portable):**
```bash
conda env create -f environment.yml
conda activate pulse
```

**Option B — plain pip + venv:**
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Install the right PyTorch for your machine
The `requirements.txt` installs a default PyTorch. For the best build:

- **NVIDIA GPU (CUDA):** follow the selector at https://pytorch.org/get-started/locally/
  e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- **CPU-only / Mac (Apple Silicon):** the default `pip install torch torchvision` is correct.
  On Apple Silicon, PULSE will use the `mps` backend automatically when available.

## 4. Verify the install
```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python pulse/train.py --help          # should print the training options
```

## 5. (Clusters only) cache the backbone for offline use
The DINOv2 ViT-B/14 backbone downloads automatically on first run. On an
offline cluster, cache it once on a login node:
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"
export TRANSFORMERS_OFFLINE=1   # then run training/inference offline
```
