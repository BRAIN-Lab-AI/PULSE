#!/bin/bash
# Submit all 3 ablation jobs to SLURM (A100 partition, fold 0 only, ~3h each).
# Results go into ablation_ckpts/{no_ds,no_bnd,no_curriculum}/
# After jobs finish run: python ablation_eval.py  to collect test Dice.

BASE=/SLURM/home/slurm_g202518690/student251/CVPR
PYTHON=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=/SLURM/home/slurm_g202518690/student251/ACDC/database

echo "=== Submitting ablation jobs ==="

# ── 1. No deep supervision ────────────────────────────────────────────────────
JOB1=$(sbatch --parsable \
  --job-name=pulse_no_ds \
  --partition=A100 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=04:00:00 \
  --output=${BASE}/logs/abl_no_ds_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_no_ds.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --ckpt_dir ablation_ckpts/no_ds")
echo "  no_ds    → job $JOB1"

# ── 2. No boundary loss ───────────────────────────────────────────────────────
JOB2=$(sbatch --parsable \
  --job-name=pulse_no_bnd \
  --partition=A100 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=04:00:00 \
  --output=${BASE}/logs/abl_no_bnd_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_no_bnd.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --ckpt_dir ablation_ckpts/no_bnd")
echo "  no_bnd   → job $JOB2"

# ── 3. No two-stage curriculum ────────────────────────────────────────────────
JOB3=$(sbatch --parsable \
  --job-name=pulse_no_curr \
  --partition=A100 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=40G \
  --time=04:00:00 \
  --output=${BASE}/logs/abl_no_curr_%j.log \
  --wrap="cd ${BASE} && ${PYTHON} ablation_no_curriculum.py \
    --fold 0 --nfolds 5 --epochs 120 --bs 16 \
    --ckpt_dir ablation_ckpts/no_curriculum")
echo "  no_curr  → job $JOB3"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "When done, run: cd ${BASE} && ${PYTHON} ablation_eval.py"
