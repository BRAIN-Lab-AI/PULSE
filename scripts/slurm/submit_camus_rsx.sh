#!/bin/bash
#SBATCH --job-name=pulse_camus3
#SBATCH --output=logs/camus3_%j.out
#SBATCH --error=logs/camus3_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=RTX3090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
source /SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/activate
cd /SLURM/home/slurm_g202518690/student251/CVPR
python eval_camus_fewshot.py --ckpt_dir folds --view 2CH
