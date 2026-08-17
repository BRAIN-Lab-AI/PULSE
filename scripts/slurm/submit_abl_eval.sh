#!/bin/bash
#SBATCH --job-name=pulse_abl_eval
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/abl_eval_%j.log

cd /SLURM/home/slurm_g202518690/student251/CVPR
export TRANSFORMERS_OFFLINE=1
export ACDC_ROOT=/SLURM/home/slurm_g202518690/student251/ACDC/database
mkdir -p logs
/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3 ablation_eval.py \
    --data_root /SLURM/home/slurm_g202518690/student251/ACDC/database
