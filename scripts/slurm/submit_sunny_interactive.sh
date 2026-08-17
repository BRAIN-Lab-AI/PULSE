#!/bin/bash
#SBATCH --job-name=sunny_interactive
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/sunny_interactive_%j.out
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/sunny_interactive_%j.err
#SBATCH --partition=interactive
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=01:30:00
PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export SB_ROOT=/SLURM/home/slurm_g202518690/student251/SUNNYBROOK_EXTRACTED/Cardiac\ MRI
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs
$PY eval_sunnybrook.py --folds_dir folds
echo "SUNNYBROOK DONE"
