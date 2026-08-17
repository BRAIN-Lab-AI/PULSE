#!/bin/bash
#SBATCH --job-name=pulse_mnm
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/mnm_%j.log
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/mnm_%j.err
#SBATCH --partition=A100
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=02:00:00

PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export MNM_ROOT=/SLURM/home/slurm_g202518690/student251/MnMs_Extracted/MnM2/dataset
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs

$PY eval_mnm.py \
    --mnm_root $MNM_ROOT \
    --folds_dir folds

echo "MNM ZERO-SHOT DONE"
