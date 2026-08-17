#!/bin/bash
# Submit new DPT-architecture ablation jobs.
#   Job 1: DPT base model (no novel components) — ~3h training
#   Job 2: Ensemble size study                  — ~40min inference

BASE=/SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=/SLURM/home/slurm_g202518690/student251/ACDC/database

mkdir -p ${BASE}/logs ${BASE}/ablation_ckpts/base_dpt

echo "=== Submitting new DPT ablation jobs ==="

# ── 1. DPT base model (training, ~3h) ────────────────────────────────────────
JOB1=$(sbatch --parsable \
  --job-name=pulse_base_dpt \
  --partition=A100 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=04:30:00 \
  --output=${BASE}/logs/abl_base_dpt_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_base_dpt.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --ckpt_dir ablation_ckpts/base_dpt")
echo "  DPT base training → job $JOB1"

# ── 2. Ensemble size study (inference only, ~40min) ───────────────────────────
JOB2=$(sbatch --parsable \
  --job-name=pulse_ens_size \
  --partition=A100 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --output=${BASE}/logs/abl_ens_size_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_ensemble_size.py \
    --data_root ${ACDC_ROOT}")
echo "  Ensemble size study → job $JOB2"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results appear in: ${BASE}/logs/abl_base_dpt_*.log"
echo "                   ${BASE}/logs/abl_ens_size_*.log"
echo ""
echo "After both finish, run: python ablation_eval_extended.py to collect all results."
