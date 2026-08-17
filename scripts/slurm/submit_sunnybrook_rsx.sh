#!/bin/bash
#SBATCH --job-name=pulse_sunny3
#SBATCH --output=logs/sunny3_%j.out
#SBATCH --error=logs/sunny3_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=RTX3090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
source /SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/activate
cd /SLURM/home/slurm_g202518690/student251/CVPR
python eval_sunnybrook.py --folds_dir folds
