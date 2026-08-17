#!/bin/bash
# Submit 4 new DPT-architecture ablation jobs across A100 / RTX3090 / A6000.
#
# Jobs created:
#   aug_none   — no augmentation          → RTX3090
#   aug_weak   — weak augmentation         → A6000
#   freeze_2   — 2-block Stage-1 freeze    → A6000
#   freeze_4   — 4-block Stage-1 freeze    → RTX3090
#
# All 4 train fold 0, 120 epochs, same hyperparams as full PULSE.
# Results land in:  CVPR/logs/abl_<name>_<jobid>.log
#                   CVPR/ablation_ckpts/<name>/best_model.pth

BASE=/SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
DATA=/SLURM/home/slurm_g202518690/student251/ACDC/database
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=${DATA}

mkdir -p ${BASE}/logs \
         ${BASE}/ablation_ckpts/aug_none \
         ${BASE}/ablation_ckpts/aug_weak \
         ${BASE}/ablation_ckpts/freeze_2blk \
         ${BASE}/ablation_ckpts/freeze_4blk

echo "=== Submitting augmentation + freeze-schedule ablation jobs ==="
echo ""

# ── 1. No Augmentation → RTX3090 ─────────────────────────────────────────────
J1=$(sbatch --parsable \
  --job-name=pulse_aug_none \
  --partition=RTX3090 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=06:00:00 \
  --output=${BASE}/logs/abl_aug_none_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_aug_none.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --data_root ${DATA} \
    --ckpt_dir ablation_ckpts/aug_none")
echo "  [1] No augmentation  → job $J1  (RTX3090)"

# ── 2. Weak Augmentation → A6000 ─────────────────────────────────────────────
J2=$(sbatch --parsable \
  --job-name=pulse_aug_weak \
  --partition=A6000 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=06:00:00 \
  --output=${BASE}/logs/abl_aug_weak_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_aug_weak.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --data_root ${DATA} \
    --ckpt_dir ablation_ckpts/aug_weak")
echo "  [2] Weak augmentation → job $J2  (A6000)"

# ── 3. Freeze 2 blocks → A6000 ───────────────────────────────────────────────
J3=$(sbatch --parsable \
  --job-name=pulse_frz2 \
  --partition=A6000 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=06:00:00 \
  --output=${BASE}/logs/abl_freeze_2blk_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_freeze_2blk.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --data_root ${DATA} \
    --ckpt_dir ablation_ckpts/freeze_2blk")
echo "  [3] Freeze 2 blocks   → job $J3  (A6000)"

# ── 4. Freeze 4 blocks → RTX3090 ─────────────────────────────────────────────
J4=$(sbatch --parsable \
  --job-name=pulse_frz4 \
  --partition=RTX3090 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=06:00:00 \
  --output=${BASE}/logs/abl_freeze_4blk_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_freeze_4blk.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --data_root ${DATA} \
    --ckpt_dir ablation_ckpts/freeze_4blk")
echo "  [4] Freeze 4 blocks   → job $J4  (RTX3090)"

echo ""
echo "Monitor all jobs:  squeue -u \$USER"
echo ""
echo "Log files:"
echo "  ${BASE}/logs/abl_aug_none_*.log"
echo "  ${BASE}/logs/abl_aug_weak_*.log"
echo "  ${BASE}/logs/abl_freeze_2blk_*.log"
echo "  ${BASE}/logs/abl_freeze_4blk_*.log"
echo ""
echo "When complete, copy the final TEST Dice lines from each log"
echo "into the ablation tables in main.tex."
echo ""
echo "Existing anchor points (no new training needed):"
echo "  No aug   → strong aug = Full PULSE = 87.0%  (RV 88.5, Myo 82.7, LV 89.7)"
echo "  0-block  → No-Curr ablation = 86.5%         (RV 88.5, Myo 81.3, LV 89.8)"
echo "  6-block  → Full PULSE       = 87.0%          (RV 88.5, Myo 82.7, LV 89.7)"
