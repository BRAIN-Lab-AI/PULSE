#!/bin/bash
#SBATCH --job-name=pulse_sunny2
#SBATCH --output=logs/sunny2_%j.out
#SBATCH --error=logs/sunny2_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=A100
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
source /SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/activate
cd /SLURM/home/slurm_g202518690/student251/CVPR
python eval_sunnybrook.py --folds_dir folds
