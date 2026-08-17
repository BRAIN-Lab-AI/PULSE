#!/bin/bash
#SBATCH --job-name=pulse_fold_eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/fold_eval_%j.log

PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=/SLURM/home/slurm_g202518690/student251/ACDC/database
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs

$PY eval_all_folds.py \
    --data_root /SLURM/home/slurm_g202518690/student251/ACDC/database \
    --folds_dir folds
