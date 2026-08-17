#!/bin/bash
#SBATCH --job-name=pulse_sunny
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/sunny_%j.log
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/sunny_%j.err
#SBATCH --partition=A100,RTX3090
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=20G --time=01:00:00

PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export SB_ROOT="/SLURM/home/slurm_g202518690/student251/SUNNYBROOK_EXTRACTED/Cardiac MRI"
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs

$PY eval_sunnybrook.py \
    --sb_root "$SB_ROOT" \
    --folds_dir folds

echo "SUNNYBROOK ZERO-SHOT DONE"
